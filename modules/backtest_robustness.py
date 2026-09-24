#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
backtest_robustness.py — 回测稳健性 (P2-11d, 2026-09-15)

初始持仓敏感性: 回测结论对初始持仓选择的依赖度量化。若换一组初始权重
结论就翻转 (离散度大), 则该结论不可外推 — 2026 诚实评估框架要求的
"结构性稳健"检查之一 (与五门审计 modules/honest_eval.py 互补)。

纯观察层: 只计算与报告, 不改任何回测行为。
"""

import logging
from typing import Callable, Dict, List, Optional, Sequence

logger = logging.getLogger('stock_analyzer.backtest_robustness')


def initial_position_sensitivity(
    equity_fn: Callable[[Sequence[float]], float],
    initial_weights_grid: List[Sequence[float]],
) -> Dict:
    """
    对一组初始持仓权重分别求终值净值, 量化结论敏感度。

    Args:
        equity_fn: callable(initial_weights) -> 终值净值 (float, 1.0=持平);
                   调用方负责跑回测, 本函数只做网格与统计
        initial_weights_grid: 初始权重组合列表, 如
            [[1.0], [0.5, 0.5], [0.2, 0.3, 0.5], ...]

    Returns:
        {'results': [{'weights': [...], 'nav': float|None, 'error': str|None}],
         'n_ok': int, 'n_error': int,
         'nav_min': float, 'nav_max': float, 'nav_mean': float,
         'dispersion': float|None,   # (max-min)/mean, 越大越不可信
         'verdict': 'stable'|'sensitive'|'insufficient',
         'note': str}
    """
    out: Dict = {'results': [], 'n_ok': 0, 'n_error': 0,
                 'nav_min': None, 'nav_max': None, 'nav_mean': None,
                 'dispersion': None, 'verdict': 'insufficient',
                 'note': '初始持仓敏感性 (P2-11d); 纯观察层不自动执行'}
    try:
        if not initial_weights_grid:
            out['note'] = '权重网格为空'
            return out
        navs = []
        for w in initial_weights_grid:
            entry = {'weights': list(w), 'nav': None, 'error': None}
            try:
                nav = equity_fn(w)
                entry['nav'] = float(nav)
                navs.append(float(nav))
                out['n_ok'] += 1
            except Exception as e:
                entry['error'] = str(e)[:200]
                out['n_error'] += 1
                logger.error(f"[BacktestRobustness] equity_fn 失败 w={w}: {e}")
            out['results'].append(entry)

        if navs:
            lo, hi = min(navs), max(navs)
            mean = sum(navs) / len(navs)
            out['nav_min'], out['nav_max'], out['nav_mean'] = lo, hi, mean
            if abs(mean) > 1e-12:
                out['dispersion'] = round((hi - lo) / abs(mean), 4)
            # 阈值经验值: <1% 稳定, 1-10% 中等敏感, >10% 高度敏感
            d = out['dispersion']
            if d is not None:
                out['verdict'] = 'stable' if d < 0.01 else ('sensitive' if d <= 0.10 else 'highly_sensitive')
        return out
    except Exception as e:
        logger.error(f"[BacktestRobustness] 失败: {e}", exc_info=True)
        out['note'] = f'error: {e}'
        return out
