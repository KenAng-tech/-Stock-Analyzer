#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
可成交性判定共享小模块 (P1-10, 2026-09-14)

背景: qlib TopkDropoutStrategy 等组合层策略在换入/换出前必须过可成交性
(涨跌停锁死 / 停牌无量), 否则回测/观察语义与实盘脱节。此前各调用点
(exec_audit 等) 各写各的阈值, 本模块统一为纯函数, 无项目依赖, 可被任何
进程 (Flask / worker / 测试 / 离线链) 安全导入。

规则 (A 股板块化涨跌停):
- 科创板 688 / 创业板 300、301        → ±20%
- 北交所 8xx / 4xx / 920xxx           → ±30%
- ST 股 (仅主板默认板块)              → ±5%
  (注: 创业板/科创板/北交所的 ST 股沿用板块自身 ±20%/±30%, 故 ST 仅收紧默认板块)
- 其余 (沪深主板)                     → ±10%

涨跌停锁死判定带 TICK_TOL 容差: 涨停价按 0.01 元取整, 低价股实际封板
涨幅可比名义阈值低 ~0.3pp (如 3 元股 3.00→3.30 恰 10%, 3.01→3.31 仅
9.97%), 故 pct_change ≥ 阈值 - 0.005 即视为封板。
"""

from typing import Optional, Tuple

__all__ = ['limit_threshold_for', 'is_tradable']

# 封板容差 (涨幅与名义阈值的最大取整偏差, 见模块 docstring)
TICK_TOL = 0.005


def _digits(code: str) -> str:
    """剥离 sz/sh/bj 等字母前缀, 返回纯数字代码串"""
    return ''.join(ch for ch in str(code) if ch.isdigit())


def limit_threshold_for(code: str, name: Optional[str] = None) -> float:
    """
    板块化涨跌停阈值。

    Args:
        code: 股票代码 (支持 'sz300620' / '600519' / 'bj920xxx' 等带前缀形式)
        name: 股票名称 (含 'ST' 时主板阈值收紧至 0.05; 创业板/科创板/北交所
              的 ST 沿用板块自身阈值, 与交易所实际规则一致)

    Returns:
        涨跌停幅度 (小数, 如 0.10 = 10%)
    """
    d = _digits(code)
    # 北交所: 4xx / 8xx / 920xxx → ±30%
    if d.startswith('920') or d[:1] in ('4', '8'):
        return 0.30
    # 科创板 688 / 创业板 300、301 → ±20%
    if d.startswith('688') or d.startswith('300') or d.startswith('301'):
        return 0.20
    # 主板 ST → ±5% (仅默认板块收紧)
    if name and 'ST' in str(name).upper():
        return 0.05
    # 沪深主板 → ±10%
    return 0.10


def is_tradable(code: str, pct_change: float, direction: str,
                volume: Optional[float] = None,
                name: Optional[str] = None) -> Tuple[bool, str]:
    """
    判断当前时刻该标的是否可按方向成交 (观察/回测语义, 非撮合)。

    Args:
        pct_change: 当日涨跌幅 (小数, 0.10 = +10%)
        direction: 'buy' / 'sell' (其余值视为可成交)
        volume: 当日成交量 (可选; ≤0 视为停牌无量)
        name: 股票名称 (ST 判定用)

    Returns:
        (tradable, reason) — tradable=True 时 reason='ok'
    """
    try:
        pct = float(pct_change)
    except (TypeError, ValueError):
        return False, f'pct_change 非法: {pct_change!r}'

    if volume is not None:
        try:
            if float(volume) <= 0:
                return False, '停牌/无量 (volume≤0)'
        except (TypeError, ValueError):
            return False, f'volume 非法: {volume!r}'

    threshold = limit_threshold_for(code, name)
    d = str(direction).lower()
    if d in ('buy', 'long'):
        if pct >= threshold - TICK_TOL:
            return False, f'涨停封板 (涨幅 {pct:+.2%} ≥ 阈值 {threshold:.0%})'
    elif d in ('sell', 'short'):
        if pct <= -(threshold - TICK_TOL):
            return False, f'跌停封板 (跌幅 {pct:+.2%} ≤ -{threshold:.0%})'
    return True, 'ok'
