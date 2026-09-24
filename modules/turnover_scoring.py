#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
turnover_scoring.py — 换手率评分 (P2-12b, 2026-09-15)

换手率 → 0-100 评分 + 流动性分级。有历史序列时用分位数 (相对自身),
不足时退固定分档 (绝对经验值)。纯函数零依赖。
"""

import logging
from typing import List, Optional

logger = logging.getLogger('stock_analyzer.turnover_scoring')

# 绝对经验分档 (A股日换手率小数): <1% 劣, 1-3% 低, 3-10% 中, 10-30% 高, >30% 极高
_ABS_BANDS = ((0.01, 20.0), (0.03, 45.0), (0.10, 70.0), (0.30, 90.0))


def turnover_score(turnover_rate: float, history: Optional[List[float]] = None) -> float:
    """
    换手评分 0-100。

    Args:
        turnover_rate: 当日换手率 (小数, 0.05 = 5%)
        history: 自身历史换手率序列 (≥20 个样本才用分位数法)

    Returns:
        float 0-100 — 分位数法: 当日在自身历史的百分位×100;
        分档法: 绝对经验档位线性插值
    """
    try:
        r = float(turnover_rate)
    except (TypeError, ValueError):
        return 0.0
    if r < 0:
        return 0.0
    if history and len(history) >= 20:
        try:
            below = sum(1 for h in history if float(h) <= r)
            return round(100.0 * below / len(history), 1)
        except (TypeError, ValueError):
            pass
    # 分档法: 档内线性插值
    lo_r, prev_score = 0.0, 0.0
    for hi_r, score in _ABS_BANDS:
        if r <= hi_r:
            frac = 0.0 if hi_r in (0.0, float('inf')) or hi_r == lo_r else (r - lo_r) / (hi_r - lo_r)
            return round(prev_score + frac * (score - prev_score), 1)
        lo_r, prev_score = hi_r, score
    return 100.0


def liquidity_grade(score: float) -> str:
    """评分 → 流动性等级: ≥70 高 / ≥45 中 / ≥20 低 / <20 劣"""
    try:
        s = float(score)
    except (TypeError, ValueError):
        return '劣'
    if s >= 70:
        return '高'
    if s >= 45:
        return '中'
    if s >= 20:
        return '低'
    return '劣'
