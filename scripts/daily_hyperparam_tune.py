#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""每日超参调参脚本 — 供 launchd 02:00 调用 (2026-09-08 新增)

背景: 白天 POST /api/dashboard/hyperparams 与 API 服务线程 CPU 争用,
75 trials × 5 折 CV 在 3600s 预算内跑不完 (实测 >50 分钟)。凌晨空闲窗
(02:00, 距 23:00 SOTA 重训 3h / 距 05:30 晨间 prep 3.5h) 独立进程跑,
无 CPU 争用 + 失败只挂自身不影响 API 服务。

流程: 500 天 K 线 → 标签/特征 (时间对齐) → regime 分段
     → 特征分布漂移检测 (数据级, 对照 2026 模式: drift→retrain)
     → Optuna 调参 (子进程隔离, 同 POST 端点; 每 trial 记 data/search_ledger.db)
     → 写 data/hyperparam_snapshot.json
     → 链尾 run 工件 runs/hyperparam/ (P0-3, qlib trainer 工件纪律)

消费链: 22:00 ML 重训 (ml_predictor scheduler) custom_params 为空时
读快照 → 凌晨调参 → 当晚重训消费 = 每日闭环。GET /hyperparams 亦读
快照供前端展示。

运行方式:
    cd /Users/claw/stock_analyzer
    source venv/bin/activate
    python scripts/daily_hyperparam_tune.py

日志输出到: logs/hyperparam_tune.log
"""

import os
import sys
import logging
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 日志配置
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, 'hyperparam_tune.log')


def _rotate_oversized(path: str, max_bytes: int = 10 * 1024 * 1024, keep: int = 4) -> None:
    """启动时轮转: 超过 max_bytes 的日志滚动为 .1~.4 (同 daily_sota_train 模式)"""
    try:
        if not (os.path.exists(path) and os.path.getsize(path) > max_bytes):
            return
        oldest = f"{path}.{keep}"
        if os.path.exists(oldest):
            os.remove(oldest)
        for i in range(keep - 1, 0, -1):
            src, dst = f"{path}.{i}", f"{path}.{i + 1}"
            if os.path.exists(src):
                os.rename(src, dst)
        os.rename(path, f"{path}.1")
    except OSError as e:
        print(f"[WARN] 日志轮转失败 {path}: {e}")


_rotate_oversized(LOG_FILE)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger('daily_hyperparam_tune')

STOCK_CODE = 'sz300620'
DRIFT_RECENT_WINDOW = 60   # 特征漂移检测的"近期"窗口 (天)


def main():
    logger.info("=" * 60)
    logger.info(f"  每日超参调参 (02:00 窗) — {datetime.now().isoformat()}")
    logger.info("=" * 60)
    t0 = datetime.now()

    try:
        from modules.hyperparam_optimizer import (
            segment_regimes, optimize_all_isolated, save_param_snapshot,
        )
        from modules.drift_aware_pipeline import ConceptDriftDetector
        from modules.ml_predictor import MLPredictor
        from modules.kline_data_fetcher import KlineDataFetcher

        # 1. 数据准备 (独立进程, 无 API 服务 CPU 争用)
        fetcher = KlineDataFetcher()
        klines = fetcher.fetch_kline(STOCK_CODE, period='daily', count=500)
        if not klines or len(klines) < 320:
            raise ValueError(f"K线不足: {len(klines) if klines else 0} 条 (需 ≥320)")

        ml = MLPredictor()  # 独立进程新实例, 仅用数据准备方法
        labels = ml.create_labels(klines, horizon=5)
        features = ml.prepare_features_batch(klines, labels)
        if features is None or len(features) == 0 or len(labels) == 0:
            raise ValueError('特征/标签构建失败')

        n_common = min(len(features), len(labels))
        features = features[:n_common]
        labels = labels[:n_common]
        logger.info(f"  数据: {n_common} 样本 × {features.shape[1]} 特征, "
                    f"耗时 {(datetime.now()-t0).total_seconds():.0f}s")

        # 2. 特征分布漂移检测 (数据级 — 近期 60d vs 训练期基线, KS 检验)
        detector = ConceptDriftDetector()
        detector.update_baseline(features[:-DRIFT_RECENT_WINDOW])
        drift_hits = sum(1 for f in features[-DRIFT_RECENT_WINDOW:]
                         if detector.check_feature_drift(f))
        if drift_hits:
            logger.warning(f"  [Drift] 特征分布漂移 {drift_hits}/{DRIFT_RECENT_WINDOW} "
                           f"样本点触发 — 市场状态可能已迁移")

        # 3. regime 分段 + 调参 (子进程隔离, 预算 2400s)
        # 2026-09-08 预算重设计: n_trials 10→5 + cv_splits 5→3 + study timeout
        # 240/240/120→150/150/75 — 旧预算 (12 study × 10 trial × 5 折 = 375 次训练)
        # 三轮实测超 3600s 预算 (46 min 零 study 完成); 新预算总时长 ≤1500s
        # 2026-09-11 二次重设计: ① 2400s 全杀实锤 = libomp 多线程 barrier 死锁
        # (单线程全链实测仅 ~45s, 见 hyperparam_optimizer 根因注释) — 超时非预算超
        # ② n_trials 3→12 + cv_splits→8 (#30 统计功效 N≥10, PBO 多候选矩阵可算;
        # study timeout 240/240/120→150/150/75 早已失效为 240/240/120? 现 study timeout
        # 维持 240/240/120 + 子进程 2400s 仅作 deadlock/降级 guard, 预算余量 ~50x)
        regimes = segment_regimes(klines[:n_common])
        tuning = optimize_all_isolated(features, labels, regimes=regimes,
                                       n_trials=12, timeout=2400)
        if not tuning:
            raise RuntimeError('调参子进程无结果 (超时或崩溃, 详见本日志)')

        # 4. 快照落盘 (22:00 重训 + GET 展示消费)
        snapshot = {
            'generated_at': datetime.now().isoformat(),
            'stock_code': STOCK_CODE,
            'best_params': tuning.get('best_params'),
            'regime_params': tuning.get('regime_params'),
            'current_regime': tuning.get('current_regime'),
            'drift': {'feature_drift_hits': drift_hits,
                      'recent_window': DRIFT_RECENT_WINDOW},
            # #30 (2026-09-11): selection-bias 观察键 (deflated IC + suspect 计数);
            # 仅观测不门控 — 22:00 重训消费门控 = 后续决策 (先积累快照证据)
            'selection_diagnostics': tuning.get('selection_diagnostics'),
        }
        if not save_param_snapshot(snapshot):
            raise RuntimeError('快照写入失败')

        elapsed = (datetime.now() - t0).total_seconds()
        logger.info(f"  调参完成! 耗时 {elapsed:.0f}s, "
                    f"current_regime={tuning.get('current_regime')}, "
                    f"drift_hits={drift_hits}")
        # #30 观察链: selection-bias 报警 (只观测, 22:00 重训门控未接 = 先积累快照证据)
        sel = tuning.get('selection_diagnostics') or {}
        if sel.get('suspect_count'):
            logger.warning(f"  [Selection-Bias] 最优选择可信度低: "
                           f"{sel['suspect_count']}/{sel.get('evaluated')} study suspect, "
                           f"deflated_ic_min={sel.get('deflated_ic_min')} "
                           f"— best_params 疑似噪声放大 (观察模式)")
        # 多候选 PBO (CSCV) 观察行 (2026-09-11 #30 标准形, 同观察模式)
        pbo = sel.get('pbo') or {}
        if pbo.get('evaluated'):
            msg = (f"  [PBO] CSCV 多候选对比: {pbo['evaluated']}/{pbo['studies']} "
                   f"study 达统计功效, pbo_max={pbo.get('pbo_max')}")
            if pbo.get('suspect'):
                logger.warning(f"{msg} — {pbo['suspect']} 个 IS 最优 OOS 衰减 "
                               f"(选择偏疑似, 观察模式)")
            else:
                logger.info(f"{msg} — 多候选对比无衰减警报")

        # 5. P0-3 (2026-09-14): 链尾 run 工件 (qlib trainer 工件纪律) + 真实
        #    trial 数 (search_ledger, DSR/PBO 按真实 N 收缩)。只加记账, 不改
        #    任何搜索/训练逻辑: 工件异常 → warning, 不影响快照落盘与链退出码。
        try:
            from modules.run_artifact import start_run
            from modules.search_ledger import trial_count
            rc = start_run('hyperparam', conf={
                'stock_code': STOCK_CODE,
                'ledger_run_id': tuning.get('ledger_run_id'),
                'samples': int(n_common),
                'drift_recent_window': DRIFT_RECENT_WINDOW,
            })
            rc.save_metrics({
                'best_params': tuning.get('best_params'),
                'regime_params': tuning.get('regime_params'),
                'current_regime': tuning.get('current_regime'),
                'selection_diagnostics': tuning.get('selection_diagnostics'),
                'drift_feature_hits': drift_hits,
                'elapsed_sec': round(elapsed, 1),
                # 真实 trial 数 (worker 每 Optuna trial 记账, 近 24h 窗)
                'ledger_trials_24h': trial_count(source='hyperparam', since_days=1),
            })
            rc.finish('ok')
            logger.info(f"  [run-artifact] {rc.run_id} metrics+status=ok 已落盘")
        except Exception as e:
            logger.warning(f"  [run-artifact] 链尾工件失败 (不影响主链): {e}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"  调参失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        # P0-3: 失败也留工件 (run_artifact.check 复盘链可见 fail run)
        try:
            from modules.run_artifact import start_run
            rc = start_run('hyperparam', conf={'stock_code': STOCK_CODE})
            rc.finish('fail', error=f'{type(e).__name__}: {e}')
        except Exception:
            pass
        sys.exit(1)


if __name__ == '__main__':
    main()
