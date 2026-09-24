#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
冲击成本模型 (Impact Cost Model)

- Almgren 冲击模型: 冲击与市场深度成反比
- Square-root 模型: 冲击与交易量的平方根成正比
- 动态参数估计
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ImpactParameters:
    """冲击模型参数"""
    market_depth: float = 1e6          # 市场深度
    impact_coeff: float = 0.1          # 冲击系数
    sqrt_coeff: float = 0.01           # sqrt 模型系数
    half_life: float = 0.5             # 冲击半衰期 (小时)
    transient_impact: float = 0.001    # 瞬时效应
    permanent_impact: float = 0.002    # 永久效应


class ImpactCostModel:
    """
    冲击成本模型

    提供 Almgren 模型和 square-root 模型的冲击成本估计。
    参考: Almgren (2003) "Optimal Trading with Estimation Risk"
    """

    def __init__(self, params: Optional[ImpactParameters] = None):
        self.params = params or ImpactParameters()

    def almgren_impact(self, quantity: int, price: float,
                        market_depth: Optional[float] = None) -> float:
        """
        Almgren 冲击模型

        冲击成本 = quantity / (2 * depth) * impact_coeff

        Args:
            quantity: 交易数量
            price: 当前价格
            market_depth: 市场深度 (可选)

        Returns:
            冲击成本 (金额)
        """
        depth = market_depth or self.params.market_depth
        impact = (quantity * price) / (2 * depth) * self.params.impact_coeff
        return impact

    def sqrt_impact(self, quantity: int, price: float,
                    volume: int) -> float:
        """
        Square-root 冲击模型

        冲击 = coeff * sqrt(quantity / volume) * price

        Args:
            quantity: 交易数量
            price: 当前价格
            volume: 市场成交量

        Returns:
            冲击成本 (金额)
        """
        if volume <= 0:
            logger.warning("成交量为0，使用默认值")
            volume = 1e6
        impact_ratio = self.params.sqrt_coeff * np.sqrt(quantity / volume)
        return impact_ratio * price * quantity

    def total_impact(self, quantity: int, price: float,
                     volume: int,
                     model: str = 'sqrt') -> Dict:
        """
        计算总冲击成本

        Args:
            quantity: 交易数量
            price: 当前价格
            volume: 市场成交量
            model: 'algren' 或 'sqrt'

        Returns:
            冲击成本明细
        """
        if model == 'algren':
            transient = self.almgren_impact(quantity, price)
            permanent = transient * self.params.permanent_impact / self.params.transient_impact
        else:
            transient = self.sqrt_impact(quantity, price, volume)
            permanent = transient * 0.3  # 假设 30% 永久效应

        total = transient + permanent

        return {
            'transient_impact': round(transient, 4),
            'permanent_impact': round(permanent, 4),
            'total_impact': round(total, 4),
            'impact_ratio': round(total / (quantity * price) if quantity * price > 0 else 0, 6),
            'model': model,
        }

    def estimate_depth(self, price: float, volume: int,
                       turnover: float = 0.01) -> float:
        """
        估计市场深度

        depth = price * volume / turnover

        Args:
            price: 当前价格
            volume: 日成交量
            turnover: 日换手率

        Returns:
            市场深度估计
        """
        if turnover <= 0:
            turnover = 0.01
        depth = price * volume / turnover
        return depth

    def calibrate(self, historical_trades: List[Dict]) -> ImpactParameters:
        """
        基于历史成交数据校准模型参数

        Args:
            historical_trades: 历史成交列表 [{'quantity', 'price', 'benchmark'}]

        Returns:
            校准后的参数
        """
        if not historical_trades:
            logger.warning("无历史成交数据，使用默认参数")
            return self.params

        impacts = []
        for trade in historical_trades:
            benchmark = trade.get('benchmark', trade['price'])
            if benchmark <= 0:
                continue
            impact = abs(trade['price'] - benchmark) / benchmark
            qty = trade['quantity']
            impacts.append(impact / np.sqrt(qty) if qty > 0 else 0)

        if impacts:
            calibrated_coeff = np.median(impacts)
            self.params.sqrt_coeff = calibrated_coeff
            logger.info(f"[ImpactCostModel] 校准 sqrt_coeff = {calibrated_coeff:.6f}")

        return self.params
