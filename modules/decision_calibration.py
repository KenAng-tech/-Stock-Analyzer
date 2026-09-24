#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""decision_calibration.py — 决策链 Brier/ECE 校准观察 (2026-09-11 决策链 B)

定位 (观察模式, 同 09-11 #30 selection-bias 形): 只观测不门控 —
consensus confidence 语义是 |score|+0.3 启发式 (投票强度, 非校准概率),
Brier/ECE 累积 "强度-结果一致性" 证据, 为后续是否上真校准 (Platt/regime 门控)
提供数据。n<20 诚实 skipped, 不装真值。

数学形:
  Brier = mean((p − y)²), p=consensus confidence ∈ (0.2, 1], y∈{0,1} (方向对不对)
  ECE   = Σ_b (n_b/N)·|acc_b − conf_b|, 4 bin 宽 0.2 ([0.2,0.4)…, p<0.2 不计)

链教训 (07-31/09-08 SIGSEGV, 09-11 四度重演): 纯函数零三方依赖,
unittest 可 in-process import 直跑安全。
"""

import math
from typing import Dict, List, Tuple


def brier_ece(records: List[Dict], min_n: int = 20,
              bin_n: int = 8) -> Dict:
    """Brier + ECE 校准观察 (纯函数, 无副作用).

    Args:
        records: [{'p': confidence, 'outcome': 0|1}] — 仅方向票入池;
                 p/outcome 不可解析或非有限 (NaN/Inf/越界) 先过滤再判
        min_n:   总样本功效门 — 不足诚实 skipped
        bin_n:   ECE 每 bin 最小样本 (不足该 bin 不统计, 不占权重)

    Returns:
        {'skipped': reason} 或
        {'n','brier','acc','ece','ece_bins','method':'brier_ece_v1'}
        (ece=None 表示无足量 bin — 只算 Brier 不装 ECE)
    """
    clean: List = []
    dropped = 0
    for r in records:
        try:
            p = float(r.get('p'))
            y = int(r.get('outcome'))
        except (TypeError, ValueError):
            dropped += 1
            continue
        if not (math.isfinite(p) and 0.0 <= p <= 1.0 and y in (0, 1)):
            dropped += 1
            continue
        clean.append((p, y))

    n = len(clean)
    if n < min_n:
        return {'skipped': f'n={n}<{min_n} (过滤后, 含丢弃 {dropped} 条非有限/坏形)'}

    brier = sum((p - y) ** 2 for p, y in clean) / n
    acc = sum(y for _, y in clean) / n

    # ECE: 只统计样本数 ≥ bin_n 的 bin (跨 bin 权重用有效样本和归一)
    valid = [(p, y) for p, y in clean if p >= 0.2]
    if len(valid) < min_n:
        return {'n': n, 'brier': round(brier, 4), 'acc': round(acc, 4),
                'ece': None, 'ece_bins': 0,
                'method': 'brier_ece_v1'}

    bins: Dict[int, List] = {}
    for p, y in valid:
        bins.setdefault(min(int((p - 0.2) / 0.2), 3), []).append((p, y))
    total = sum(len(v) for v in bins.values())
    ece, bins_used = 0.0, 0
    for vs in bins.values():
        if len(vs) < bin_n:
            continue
        conf_b = sum(p for p, _ in vs) / len(vs)
        acc_b = sum(y for _, y in vs) / len(vs)
        ece += (len(vs) / total) * abs(acc_b - conf_b)
        bins_used += 1

    out = {'n': n, 'brier': round(brier, 4), 'acc': round(acc, 4),
           'ece': round(ece, 4) if bins_used else None,
           'ece_bins': bins_used, 'method': 'brier_ece_v1'}
    return out


def role_gate(role_records: Dict[str, List[Tuple[float, int]]],
              threshold: float = 0.55, min_n: int = 20) -> Dict[str, Dict]:
    """角色级 Brier 校准门 (2026-09-12 整合 P0-a).

    2026 调研锚点: "profitability-evidence" (校准需利润证据) + TradingAgents-CN
    (辩论/角色分工生产化). 与 _recalibrate_weights 的 IC 闭环正交:
    IC = 方向命中率 (hit/(hit+miss)), Brier = confidence 校准质量 —
    高 conf 连错 (校准漂移, IC 不敏感) 在此被抓.

    Args:
        role_records: {role: [(p, y), …]} — p=票 confidence, y=方向对错 0|1
        threshold:    Brier 降权阈 (仅 gated=True, 降权由消费方执行)
        min_n:        角色级功效门 — 不足恒 gated=False (诚实, 不装真值)

    Returns:
        {role: {'n','brier','gated'}} — n<min_n 时 gated=False + brier=None
    """
    out: Dict[str, Dict] = {}
    for role, recs in (role_records or {}).items():
        clean = [(p, y) for p, y in recs
                 if isinstance(p, (int, float)) and math.isfinite(p)
                 and 0.0 <= p <= 1.0 and y in (0, 1)]
        if len(clean) < min_n:
            out[role] = {'n': len(clean), 'brier': None, 'gated': False}
            continue
        brier = sum((p - y) ** 2 for p, y in clean) / len(clean)
        out[role] = {'n': len(clean), 'brier': round(brier, 4),
                     'gated': brier > threshold}
    return out
