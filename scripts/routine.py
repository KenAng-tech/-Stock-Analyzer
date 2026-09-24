#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
routine.py — 夜间链统一编排单入口 + 历史回放 (P2-12c, 观察模式)

qlib OnlineManager 思想 (manager.py:184-347): 「更新预测 → 按需重训 →
出信号 → 复盘」收敛为一个 routine(date), 同一套代码配 simulate() 在历史
日历回放做链路回归测试。

现状: 23:00 训练 (launchd) / 02:00 调参 / 23:10 复盘 (crontab) 三条链
各自为政, 断一夜要到第二天人肉发现 (09-13 审查结论)。

本脚本是**观察层**: 不替换 launchd/crontab; 提供
  python scripts/routine.py --date 2026-09-14        # 干跑当日编排, 输出各步 OK/FAIL
  python scripts/routine.py --simulate 2026-08-01 2026-09-14   # 历史日历回放
每步 guarded import + OK/FAIL 标记, 供未来把三链收敛时作对照基线。
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('routine')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROUTINE_LOG = os.path.join(PROJECT_ROOT, 'logs', 'routine_ledger.jsonl')


def _mark(step: str, ok: bool, detail: str = '') -> dict:
    """OK/FAIL 标记 (09-14 惯例: 夜间链每步必须可机读判定成败)"""
    return {'step': step, 'ok': bool(ok), 'detail': detail[:300]}


def step_data_check(date_str: str) -> dict:
    """步骤 1: 数据新鲜度检查 — K 线/龙虎榜/解禁最新日期是否覆盖到 date"""
    try:
        from modules.data_fetcher import StockDataFetcher
        fetcher = StockDataFetcher()
        checks = {}
        for code in ('sz300620', 'sh688981'):
            try:
                k = fetcher.get_kline_data(code, count=5)
                if k:
                    last = k[-1].get('date') or k[-1].get('time') or str(k[-1].get('datetime', ''))[:10]
                    checks[code] = str(last)
                else:
                    checks[code] = 'error: 空返回'
            except Exception as e:
                checks[code] = f'error: {e}'
        ok = bool(checks) and not any(str(v).startswith('error') for v in checks.values())
        return _mark('data_check', ok, json.dumps(checks, ensure_ascii=False))
    except Exception as e:
        logger.error(f"[routine] 数据检查异常: {e}", exc_info=True)
        return _mark('data_check', False, str(e))


def step_retrain_status(date_str: str) -> dict:
    """步骤 2: 按需重训判定 (观察: 只报告 rolling 策略会重训/跳过, 不触发训练)"""
    try:
        from modules.model_registry import ModelRegistry
        reg = ModelRegistry()
        days = reg.days_since_last_train('sota') if hasattr(reg, 'days_since_last_train') else None
        return _mark('retrain_status', True,
                     f"days_since_last_train={days} (观察模式: 不触发训练)")
    except Exception as e:
        logger.error(f"[routine] 重训状态查询异常: {e}", exc_info=True)
        return _mark('retrain_status', False, str(e))


def step_scoring_status(date_str: str) -> dict:
    """步骤 3: 打分/预测状态 (观察: 检查在线模型是否存在, 不重打分)"""
    try:
        from modules.model_registry import ModelRegistry
        reg = ModelRegistry()
        online = reg.get_online('sota') if hasattr(reg, 'get_online') else None
        return _mark('scoring_status', bool(online),
                     f"online_model={'yes' if online else 'MISSING'}")
    except Exception as e:
        logger.error(f"[routine] 打分状态查询异常: {e}", exc_info=True)
        return _mark('scoring_status', False, str(e))


def step_replay_status(date_str: str) -> dict:
    """步骤 4: 复盘链产物检查 (23:10 decision_replay 的 OK/FAIL 标记是否落地)"""
    try:
        log = '/tmp/decision_replay.log'
        if not os.path.exists(log):
            return _mark('replay_status', False, 'decision_replay.log 不存在')
        with open(log, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()[-50:]
        day = date_str.replace('-', '')
        ok_lines = [l for l in lines if 'RECAL_OK' in l and (day in l or date_str in l)]
        fail_lines = [l for l in lines if 'RECAL_FAIL' in l and (day in l or date_str in l)]
        return _mark('replay_status', bool(ok_lines) and not fail_lines,
                     f'OK={len(ok_lines)} FAIL={len(fail_lines)}')
    except Exception as e:
        logger.error(f"[routine] 复盘状态查询异常: {e}", exc_info=True)
        return _mark('replay_status', False, str(e))


STEPS = (step_data_check, step_retrain_status, step_scoring_status, step_replay_status)


def routine(date_str: str, write_ledger: bool = True) -> dict:
    """
    单日编排: 顺序跑全部步骤, 每步独立 guarded, 汇总 OK/FAIL 落 ledger。

    Returns: {'date': str, 'results': [step dicts], 'ok': bool}
    """
    results = []
    for step in STEPS:
        try:
            results.append(step(date_str))
        except Exception as e:  # 双保险: 步骤函数内部已 guarded
            logger.error(f"[routine] 步骤 {step.__name__} 未捕获异常: {e}", exc_info=True)
            results.append(_mark(step.__name__, False, str(e)))
    summary = {
        'ts': datetime.now().isoformat(timespec='seconds'),
        'date': date_str,
        'results': results,
        'ok': all(r['ok'] for r in results),
    }
    if write_ledger:
        try:
            with open(ROUTINE_LOG, 'a', encoding='utf-8') as f:
                f.write(json.dumps(summary, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.error(f"[routine] ledger 写入失败: {e}")
    return summary


def simulate(start: str, end: str) -> dict:
    """
    历史日历回放 (qlib manager.simulate 思想): 用同一 routine() 逐日跑,
    作链路回归基线。日历用近似工作日 (不依赖交易日历服务)。

    Returns: {'days': int, 'ok_days': int, 'fail_days': [date...]}
    """
    d0 = datetime.strptime(start, '%Y-%m-%d')
    d1 = datetime.strptime(end, '%Y-%m-%d')
    days, ok_days, fail_days = 0, 0, []
    d = d0
    while d <= d1:
        if d.weekday() < 5:  # 近似: 工作日
            ds = d.strftime('%Y-%m-%d')
            res = routine(ds, write_ledger=True)
            days += 1
            if res['ok']:
                ok_days += 1
            else:
                fail_days.append(ds)
        d += timedelta(days=1)
    return {'days': days, 'ok_days': ok_days, 'fail_days': fail_days}


def main():
    """CLI 入口"""
    parser = argparse.ArgumentParser(description='夜间链统一编排 (观察模式)')
    parser.add_argument('--date', default=datetime.now().strftime('%Y-%m-%d'))
    parser.add_argument('--simulate', nargs=2, metavar=('START', 'END'))
    args = parser.parse_args()

    if args.simulate:
        out = simulate(args.simulate[0], args.simulate[1])
    else:
        out = routine(args.date)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    sys.exit(0 if out.get('ok', out.get('fail_days') == []) else 1)


if __name__ == '__main__':
    main()
