#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Brinson-Fachler 绩效归因模块

P2 优化 (2026-07-01):
  将投资组合收益分解为:
  1. 资产配置效应 (Allocation Effect) — 因超配/低配某行业带来的收益
  2. 选股效应 (Selection Effect) — 因在行业内选择优质个股带来的收益
  3. 交互效应 (Interaction Effect) — 配置和选股的交互作用

参考:
  - Brinson, G.P., Hood, L.R. and Beebower, D.L. (1986). "Determinants of Portfolio Performance"
  - Brinson, G.P., Singer, B.D. and Friend, G.L. (1991). "Equity Allocation and Portfolio Performance"

用法:
    analyzer = BrinsonAttribution()
    result = analyzer.analyze(
        portfolio_weights={'科技': 0.4, '消费': 0.3, '金融': 0.3},
        benchmark_weights={'科技': 0.3, '消费': 0.3, '金融': 0.4},
        portfolio_returns={'科技': 0.10, '消费': 0.05, '金融': 0.02},
        benchmark_returns={'科技': 0.08, '消费': 0.06, '金融': 0.03},
    )
"""

import numpy as np
from typing import Dict, Optional, List
from modules.logger import logger


class BrinsonAttribution:
    """Brinson-Fachler 绩效归因分析器"""

    def __init__(self):
        self._history: List[Dict] = []

    def analyze(self,
                portfolio_weights: Dict[str, float],
                benchmark_weights: Dict[str, float],
                portfolio_returns: Dict[str, float],
                benchmark_returns: Dict[str, float],
                period: str = 'daily') -> Dict:
        """
        执行 Brinson-Fachler 归因分析

        Args:
            portfolio_weights: 投资组合行业权重 {行业: 权重}
            benchmark_weights: 基准指数行业权重 {行业: 权重}
            portfolio_returns: 投资组合行业收益率 {行业: 收益率}
            benchmark_returns: 基准指数行业收益率 {行业: 收益率}
            period: 周期标记 ('daily'/'weekly'/'monthly')

        Returns:
            {
                'period': str,
                'portfolio_return': float,     # 组合总收益
                'benchmark_return': float,     # 基准总收益
                'excess_return': float,        # 超额收益
                'allocation_effect': float,    # 资产配置效应
                'selection_effect': float,     # 选股效应
                'interaction_effect': float,   # 交互效应
                'total_attribution': float,    # 总归因 (应 ≈ 超额收益)
                'sector_attribution': List[Dict],  # 各行业归因明细
                'conclusion': str,             # 自然语言结论
            }
        """
        all_sectors = set(list(portfolio_weights.keys()) + list(benchmark_weights.keys()))
        total_weight = sum(portfolio_weights.values())
        bench_total_weight = sum(benchmark_weights.values())

        # 计算组合和基准的总收益
        portfolio_return = sum(
            portfolio_weights.get(s, 0) * portfolio_returns.get(s, 0)
            for s in all_sectors
        )
        benchmark_return = sum(
            benchmark_weights.get(s, 0) * benchmark_returns.get(s, 0)
            for s in all_sectors
        )
        excess_return = portfolio_return - benchmark_return

        # 各行业归因
        sector_attributions = []
        total_allocation = 0.0
        total_selection = 0.0
        total_interaction = 0.0

        for sector in all_sectors:
            w_p = portfolio_weights.get(sector, 0)
            w_b = benchmark_weights.get(sector, 0)
            r_p = portfolio_returns.get(sector, 0)
            r_b = benchmark_returns.get(sector, 0)

            # Brinson-Fachler 分解:
            # Allocation = (w_p - w_b) * (r_b - R_b)
            # Selection  = w_p * (r_p - r_b)
            # Interaction = (w_p - w_b) * (r_p - r_b)
            # 其中 R_b = 基准总收益

            allocation = (w_p - w_b) * (r_b - benchmark_return)
            selection = w_p * (r_p - r_b)
            interaction = (w_p - w_b) * (r_p - r_b)

            total_allocation += allocation
            total_selection += selection
            total_interaction += interaction

            # 判断贡献方向
            alloc_direction = '超配' if allocation > 0 else '低配' if allocation < 0 else '中性'
            select_direction = '正向' if selection > 0 else '负向' if selection < 0 else '中性'

            sector_attributions.append({
                'sector': sector,
                'weight_diff': round(w_p - w_b, 4),
                'allocation_effect': round(allocation, 6),
                'selection_effect': round(selection, 6),
                'interaction_effect': round(interaction, 6),
                'total_effect': round(allocation + selection + interaction, 6),
                'allocation_direction': alloc_direction,
                'selection_direction': select_direction,
            })

        total_attribution = total_allocation + total_selection + total_interaction

        # 自然语言结论
        conclusion = self._generate_conclusion(
            excess_return, total_allocation, total_selection, total_interaction,
            sector_attributions
        )

        result = {
            'period': period,
            'portfolio_return': round(portfolio_return, 6),
            'benchmark_return': round(benchmark_return, 6),
            'excess_return': round(excess_return, 6),
            'allocation_effect': round(total_allocation, 6),
            'selection_effect': round(total_selection, 6),
            'interaction_effect': round(total_interaction, 6),
            'total_attribution': round(total_attribution, 6),
            'sector_attribution': sector_attributions,
            'conclusion': conclusion,
            'timestamp': __import__('datetime').datetime.now().isoformat(),
        }

        # 记录历史
        self._history.append(result)

        return result

    def _generate_conclusion(self, excess_return: float,
                             allocation: float, selection: float,
                             interaction: float,
                             sectors: List[Dict]) -> str:
        """生成自然语言结论"""
        parts = []

        if excess_return > 0.001:
            parts.append(f"组合跑赢基准 {excess_return:.2%}")
        elif excess_return < -0.001:
            parts.append(f"组合跑输基准 {abs(excess_return):.2%}")
        else:
            parts.append("组合与基准表现相当")

        # 找出最大贡献来源
        if abs(allocation) > abs(selection):
            direction = "正" if allocation > 0 else "负"
            parts.append(f"资产配置是主要{direction}贡献源 ({allocation:.2%})")
        else:
            direction = "正" if selection > 0 else "负"
            parts.append(f"选股能力是主要{direction}贡献源 ({selection:.2%})")

        # 找出最佳行业
        best_sector = max(sectors, key=lambda x: x['total_effect']) if sectors else None
        if best_sector and best_sector['total_effect'] > 0.001:
            parts.append(f"最佳行业: {best_sector['sector']} ({best_sector['total_effect']:.2%})")

        # 找出最差行业
        worst_sector = min(sectors, key=lambda x: x['total_effect']) if sectors else None
        if worst_sector and worst_sector['total_effect'] < -0.001:
            parts.append(f"拖累行业: {worst_sector['sector']} ({worst_sector['total_effect']:.2%})")

        return "。".join(parts) + "。"

    def get_history(self, n: int = 10) -> List[Dict]:
        """获取最近的归因历史"""
        return self._history[-n:] if self._history else []

    def cumulative_attribution(self) -> Dict:
        """累计归因分析"""
        if not self._history:
            return {'error': '无历史数据'}

        total_allocation = sum(h['allocation_effect'] for h in self._history)
        total_selection = sum(h['selection_effect'] for h in self._history)
        total_interaction = sum(h['interaction_effect'] for h in self._history)
        total_excess = sum(h['excess_return'] for h in self._history)

        return {
            'n_periods': len(self._history),
            'cumulative_excess_return': round(total_excess, 6),
            'cumulative_allocation': round(total_allocation, 6),
            'cumulative_selection': round(total_selection, 6),
            'cumulative_interaction': round(total_interaction, 6),
            'allocation_contribution_pct': round(
                total_allocation / abs(total_excess) * 100 if total_excess != 0 else 0, 1
            ),
            'selection_contribution_pct': round(
                total_selection / abs(total_excess) * 100 if total_excess != 0 else 0, 1
            ),
        }


# ── 全局单例 ────────────────────────────────────────────────

_brinson_instance: Optional[BrinsonAttribution] = None


def get_brinson_attribution() -> BrinsonAttribution:
    """获取 Brinson 归因分析器全局实例"""
    global _brinson_instance
    if _brinson_instance is None:
        _brinson_instance = BrinsonAttribution()
    return _brinson_instance


def reset_brinson_attribution():
    """重置全局实例"""
    global _brinson_instance
    _brinson_instance = None
