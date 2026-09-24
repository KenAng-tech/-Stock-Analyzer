#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""超参调参 Worker — 子进程隔离 (2026-09-08)

背景: POST /api/dashboard/hyperparams 原在主进程 (Flask 请求线程) 内直跑
Optuna × LGBM/XGB 训练, 2026-09-08 实测触发 SIGSEGV (segfault 11) 杀死整个
API 服务 — 与 07-31 ml_training_worker SIGSEGV 同源 (C 扩展在主进程崩溃)。
修复: 照 ml_training_worker 模式把调参移入子进程, 主进程只做数据准备 +
结果 dict 合并。

协议: stdin 收 JSON {X, y, regimes, n_trials, embargo_periods},
     stdout 回 JSON {success, best_params, regime_params, current_regime,
                     ledger_run_id, message}
     (P0-3 2026-09-14: 另加 search_ledger 记账副作用, trials 走 SQLite 不走 stdout)
"""
import os
import sys
import json
import signal

# 2026-09-11: OpenMP 单线程防御 (链上 2400s 全杀根因 = libomp barrier 死锁,
# 详见 optimize_all_isolated 注释)。必须在任何 numpy/sklearn/lightgbm/xgboost
# import 之前设置 — 多 OMP runtime 混载时只有单线程能绕开 barrier。
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OMP_WAIT_POLICY', 'passive')

# 确保项目根目录在 sys.path (脚本位于 modules/ 下时 sys.path[0] 是 modules/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def _fail(msg):
    print(json.dumps({'success': False, 'message': msg}, ensure_ascii=False))
    sys.exit(1)


def _sigsegv_handler(signum, frame):
    _fail(f'C 扩展崩溃 (signal {signum})')


signal.signal(signal.SIGSEGV, _sigsegv_handler)
signal.signal(signal.SIGABRT, _sigsegv_handler)


# ── P0-3 (2026-09-14): 搜索全记账 — 每个 Optuna trial 完成后落 search_ledger ──
# 纪律: 只加记账副作用, 不改任何搜索/训练逻辑; import 失败或记账异常 →
# 静默跳过 (stderr 诊断), 调参链与 stdout JSON 协议不受影响。
# stdout 是结果 JSON 的单一通道, 本段所有诊断输出走 stderr。

def _make_run_id():
    """本次调参运行的账本 run_id (链接 worker trials 与链尾 run_artifact)"""
    from datetime import datetime
    return f"hp-{datetime.now():%Y%m%d-%H%M%S}-{os.getpid()}"


def _wrap_optimize(opt, model_name, run_id, record_trial):
    """包装单个 optimizer.optimize: 调用返回后把其新 study 的 COMPLETE trials 记账。

    scope 判定: optimize_all 结构 = 全局 study 先跑 (seq 0), 其后每调用一次
    即一个 per-regime study。grid 回退 (Optuna 缺失/study 异常) 不产生新
    study 对象 → 用 study 身份比对跳过, 防把上一次 study 的 trials 重复记账。
    """
    orig = opt.optimize
    state = {'seq': 0}

    def wrapper(X, y, *args, **kwargs):
        pre_study = getattr(opt, 'study', None)
        result = orig(X, y, *args, **kwargs)
        try:
            study = getattr(opt, 'study', None)
            seq = state['seq']
            state['seq'] += 1
            if study is None or study is pre_study:
                return result   # grid 回退/无新 study → 不记 (trial 语义只算 Optuna)
            scope = 'global' if seq == 0 else 'per-regime'
            for t in getattr(study, 'trials', []):
                try:
                    if t.state.name != 'COMPLETE' or t.value is None:
                        continue
                    record_trial(
                        source='hyperparam',
                        params=t.params or {},
                        metric_name='cv_ic',
                        metric_value=float(t.value),
                        run_id=run_id,
                        note=f'{model_name}/{scope} study#{seq} trial#{t.number}')
                except Exception:
                    continue    # 单 trial 记账失败不影响其余 trial 与主链
        except Exception as e:
            print(f'[ledger] trial 记账失败 (已忽略): {e}', file=sys.stderr)
        return result

    return wrapper


def _attach_trial_ledger(orch, run_id):
    """给三个 optimizer 实例挂记账包装。返回挂载数 (0 = 账本不可用, 链照常跑)"""
    try:
        from modules.search_ledger import record_trial
    except Exception as e:
        print(f'[ledger] search_ledger 不可用, 跳过记账: {e}', file=sys.stderr)
        return 0
    attached = 0
    for model_name in ('lgb', 'xgb', 'rf'):
        opt = getattr(orch, f'{model_name}_optimizer', None)
        if opt is None or not callable(getattr(opt, 'optimize', None)):
            continue
        opt.optimize = _wrap_optimize(opt, model_name, run_id, record_trial)
        attached += 1
    print(f'[ledger] trial 记账已挂载 ({attached} optimizers, run_id={run_id})',
          file=sys.stderr)
    return attached


def main():
    try:
        input_data = json.loads(sys.stdin.read())
    except Exception as e:
        _fail(f'输入解析失败: {e}')

    X = np.array(input_data['X'], dtype=np.float64)
    y = np.array(input_data['y'], dtype=np.float64)
    n = min(len(X), len(y))
    X, y = X[:n], y[:n]
    regimes = input_data.get('regimes')
    n_trials = int(input_data.get('n_trials', 12))   # 2026-09-11: 3→12 (PBO 功效 N≥10)
    embargo = int(input_data.get('embargo_periods', 0))

    try:
        from modules.hyperparam_optimizer import HyperParamOrchestrator

        orch = HyperParamOrchestrator(n_trials=n_trials, embargo_periods=embargo)
        # P0-3: 搜索全记账 (guarded, 失败不影响链); run_id 随结果透传给链尾工件
        ledger_run_id = _make_run_id()
        _attach_trial_ledger(orch, ledger_run_id)
        best = orch.optimize_all(X, y, regimes=regimes)
        print(json.dumps({
            'success': True,
            'best_params': best,
            'regime_params': orch._regime_params,
            'current_regime': regimes[-1] if regimes else None,
            'selection_diagnostics': orch.selection_summary(),   # #30 观察键
            'ledger_run_id': ledger_run_id,                      # P0-3 账本关联
        }, ensure_ascii=False, default=str))
    except Exception as e:
        _fail(f'{type(e).__name__}: {e}')


if __name__ == '__main__':
    main()
