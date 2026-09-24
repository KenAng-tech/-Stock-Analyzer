#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
角色信号相关 vs PnL 相关分析 (P1-7, 2026-09-14)

背景 (arXiv 2609.09588): 信号横截面 IC 相关 ≠ PnL 相关 — 两个角色信号
高度相关但实际持仓 PnL 可能几乎不相关 (触发时点/标的/仓位分布不同),
反之亦然。IC 加权 ensemble 用信号相关近似 PnL 相关存在理论缺陷; qlib
工业实践的对策是横截面 z-score 后等权。本模块量化两者差距 (gap),
为 consensus 双轨 A/B (zscore_equal_weight 旁路) 提供证据基础。

输入形状:
    role_decisions: {role: [(date, symbol, direction, confidence), ...]}
        date 'YYYY-MM-DD'; direction 'buy'/'sell'/'neutral' (含 strong_* 前缀);
    klines_by_symbol: {symbol: [{'date': ..., 'close': ...}, ...]} 按日期升序。

输出:
    analyze(...) -> (result, reason)
    - 成功: ({'pairs': [{'a','b','signal_corr','pnl_corr','gap',
                         'agreement_rate','conf_spearman','n_matched','n_days'}]}, None)
    - 数据不足: (None, '原因')

指标定义:
- signal_corr = 2 × 方向一致率 - 1 ∈ [-1,1] (同标的同日两角色方向同侧=一致;
  一致率 ∈ [0,1] 原值同时输出为 agreement_rate); conf_spearman 为同批匹配
  票 confidence 的 Spearman 秩相关 (并列取平均秩, 纯 numpy 无 scipy)。
- pnl_corr: 各角色按自己信号持仓 (position = 方向符号 × confidence, 决策日
  收盘建仓、次日收盘收益) 的日 PnL 序列, 对齐共同日期后取 Pearson。
- gap = |signal_corr - pnl_corr| — gap 大即「IC 代理失真」证据。
"""

from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from modules.logger import logger
except Exception:  # pragma: no cover - 项目外独立运行回退
    import logging
    logger = logging.getLogger('pnl_corr')

__all__ = ['analyze']

# 单对指标的最少样本数 (匹配票 / 共同交易日)
_MIN_MATCHED = 3
_MIN_DAYS = 3


def _side(direction: str) -> int:
    """方向 → 侧向符号: 多=+1, 空=-1, 中性/其他=0"""
    d = str(direction).lower()
    if d in ('buy', 'strong_buy', 'long'):
        return 1
    if d in ('sell', 'strong_sell', 'short'):
        return -1
    return 0


def _rankdata(x: np.ndarray) -> np.ndarray:
    """平均并列秩 (scipy.stats.rankdata 'average' 的 numpy 等价)"""
    order = np.argsort(x, kind='mergesort')
    ranks = np.empty(len(x), dtype=float)
    sx = x[order]
    i = 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def _pearson(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    """Pearson 相关; 任一侧零方差/样本不足 → None (诚实缺失, 不伪造)"""
    if len(a) < 2 or np.std(a) < 1e-15 or np.std(b) < 1e-15:
        return None
    r = float(np.corrcoef(a, b)[0, 1])
    if r != r:  # NaN
        return None
    return r


def _spearman(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    """Spearman 秩相关 (平均并列秩 + Pearson)"""
    if len(a) < 2:
        return None
    return _pearson(_rankdata(a), _rankdata(b))


def _next_day_returns(klines: List[Dict]) -> Dict[str, float]:
    """决策日 → 次日收盘收益 {decision_date: close[t+1]/close[t]-1}"""
    out: Dict[str, float] = {}
    for i in range(len(klines) - 1):
        try:
            d0 = str(klines[i].get('date', ''))[:10]
            d1 = str(klines[i + 1].get('date', ''))[:10]
            c0 = float(klines[i].get('close'))
            c1 = float(klines[i + 1].get('close'))
        except (TypeError, ValueError):
            continue
        if d0 and c0 > 0:
            out[d0] = c1 / c0 - 1.0
    return out


def analyze(role_decisions: Dict[str, List[Tuple]],
            klines_by_symbol: Dict[str, List[Dict]]
            ) -> Tuple[Optional[Dict], Optional[str]]:
    """
    角色两两「信号相关 vs PnL 相关」分析。

    Args:
        role_decisions: {role: [(date, symbol, direction, confidence), ...]}
        klines_by_symbol: {symbol: [{'date','close',...} 升序]}

    Returns:
        (result, reason) — 成功 ({'pairs': [...]}, None); 数据不足 (None, 原因)
    """
    roles = sorted(role_decisions or {})
    if len(roles) < 2:
        return None, f'角色数不足 2 (实际 {len(roles)})'
    if not klines_by_symbol:
        return None, '无 K 线数据'

    # 每标的的 决策日→次日收益 表 (一次构建, 全角色复用)
    ret_by_symbol = {s: _next_day_returns(kl) for s, kl in klines_by_symbol.items()
                     if kl}

    # 每角色: 决策表 {(date,symbol): (side, conf)} + 日 PnL 序列 {date: pnl}
    decision_map: Dict[str, Dict[Tuple[str, str], Tuple[int, float]]] = {}
    pnl_series: Dict[str, Dict[str, float]] = {}
    for role in roles:
        dmap: Dict[Tuple[str, str], Tuple[int, float]] = {}
        pnl: Dict[str, float] = {}
        for row in (role_decisions.get(role) or []):
            try:
                date, symbol, direction, conf = (str(row[0])[:10], str(row[1]),
                                                 row[2], float(row[3]))
            except (TypeError, ValueError, IndexError):
                logger.debug(f"[pnl_corr] {role} 跳过畸形记录: {row!r}")
                continue
            side = _side(direction)
            dmap[(date, symbol)] = (side, conf)
            if side == 0:
                continue  # 中性票空仓, 不贡献 PnL
            ret = ret_by_symbol.get(symbol, {}).get(date)
            if ret is None:
                continue  # 该决策日无次日 K 线 (未收盘/停牌), 诚实跳过
            pnl[date] = pnl.get(date, 0.0) + side * conf * ret
        decision_map[role] = dmap
        pnl_series[role] = pnl

    pairs: List[Dict] = []
    for a, b in combinations(roles, 2):
        # ── (a) 信号相关: 同标的同日匹配票 ──
        common_keys = sorted(set(decision_map[a]) & set(decision_map[b]))
        agreement_rate = conf_spearman = None
        if len(common_keys) >= _MIN_MATCHED:
            sides_a = np.array([decision_map[a][k][0] for k in common_keys], float)
            sides_b = np.array([decision_map[b][k][0] for k in common_keys], float)
            agree = float(np.mean(sides_a == sides_b))
            agreement_rate = agree
            conf_spearman = _spearman(
                np.array([decision_map[a][k][1] for k in common_keys]),
                np.array([decision_map[b][k][1] for k in common_keys]))
        signal_corr = None if agreement_rate is None else 2.0 * agreement_rate - 1.0

        # ── (b) PnL 相关: 共同交易日上的日收益序列 Pearson ──
        days = sorted(set(pnl_series[a]) & set(pnl_series[b]))
        pnl_corr = None
        if len(days) >= _MIN_DAYS:
            pa = np.array([pnl_series[a][d] for d in days])
            pb = np.array([pnl_series[b][d] for d in days])
            pnl_corr = _pearson(pa, pb)

        gap = (None if signal_corr is None or pnl_corr is None
               else abs(signal_corr - pnl_corr))
        if signal_corr is None and pnl_corr is None:
            continue  # 该对完全不可算 → 不输出伪行
        pairs.append({
            'a': a, 'b': b,
            'signal_corr': None if signal_corr is None else round(signal_corr, 4),
            'pnl_corr': None if pnl_corr is None else round(pnl_corr, 4),
            'gap': None if gap is None else round(gap, 4),
            'agreement_rate': None if agreement_rate is None else round(agreement_rate, 4),
            'conf_spearman': None if conf_spearman is None else round(conf_spearman, 4),
            'n_matched': len(common_keys),
            'n_days': len(days),
        })

    if not pairs:
        return None, ('数据不足: 无角色对具备 ≥'
                      f'{_MIN_MATCHED} 张匹配票或 ≥{_MIN_DAYS} 个共同交易日')
    return {'pairs': pairs}, None
