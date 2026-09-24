#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
动态滑点估计模型 (Dynamic Slippage Model)

- 基于 ATR 的动态滑点
- 基于波动率的滑点调整
- 基于市场深度的滑点估计
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class SlippageParams:
    """滑点参数"""
    base_slippage: float = 0.001         # 基础滑点 (1bp)
    vol_multiplier: float = 2.0          # 波动率乘数
    depth_multiplier: float = 1.0        # 深度乘数
    min_slippage: float = 0.0001         # 最小滑点
    max_slippage: float = 0.01           # 最大滑点 (1%)
    atr_multiplier: float = 0.5          # ATR 乘数


class DynamicSlippageModel:
    """
    动态滑点估计模型

    根据市场条件动态调整滑点估计:
    slippage = base * (1 + vol_factor + depth_factor + atr_factor)

    参考: Bertsimas & Lo (1998) "Optimal Control of Execution Risk"
    """

    def __init__(self, params: Optional[SlippageParams] = None):
        self.params = params or SlippageParams()
        self._history: List[Dict] = []

    def estimate(self, price: float, volume: int,
                 volatility: Optional[float] = None,
                 market_depth: Optional[float] = None,
                 atr: Optional[float] = None) -> Dict:
        """
        估计动态滑点

        Args:
            price: 当前价格
            volume: 成交量
            volatility: 波动率 (可选)
            market_depth: 市场深度 (可选)
            atr: 平均真实波幅 (可选)

        Returns:
            滑点估计结果
        """
        if price <= 0:
            return {'slippage_ratio': 0, 'slippage_price': 0, 'model': 'dynamic'}

        slippage = self.params.base_slippage

        # 波动率调整
        if volatility is not None:
            vol_factor = self.params.vol_multiplier * volatility
            slippage *= (1 + vol_factor)

        # 市场深度调整
        if market_depth is not None and market_depth > 0:
            implied_depth = price * volume
            depth_ratio = implied_depth / market_depth
            depth_factor = self.params.depth_multiplier * min(depth_ratio, 1.0)
            slippage *= (1 + depth_factor)

        # ATR 调整
        if atr is not None and atr > 0 and price > 0:
            atr_ratio = atr / price
            atr_factor = self.params.atr_multiplier * atr_ratio
            slippage *= (1 + atr_factor)

        # 限制范围
        slippage = max(self.params.min_slippage, min(self.params.max_slippage, slippage))

        slippage_price = slippage * price

        result = {
            'slippage_ratio': round(slippage, 6),
            'slippage_price': round(slippage_price, 4),
            'price': price,
            'model': 'dynamic',
            'timestamp': datetime.now().isoformat(),
        }

        self._history.append(result)
        if len(self._history) > 1000:
            self._history = self._history[-500:]

        return result

    def estimate_buy(self, price: float, volume: int = 0,
                     volatility: Optional[float] = None,
                     atr: Optional[float] = None) -> float:
        """
        买入滑点价格

        Args:
            price: 当前价格
            volume: 成交量
            volatility: 波动率
            atr: ATR

        Returns:
            建议买入价格 (含滑点)
        """
        result = self.estimate(price, volume, volatility, atr=atr)
        return round(price * (1 + result['slippage_ratio']), 4)

    def estimate_sell(self, price: float, volume: int = 0,
                      volatility: Optional[float] = None,
                      atr: Optional[float] = None) -> float:
        """
        卖出滑点价格

        Args:
            price: 当前价格
            volume: 成交量
            volatility: 波动率
            atr: ATR

        Returns:
            建议卖出价格 (扣除滑点)
        """
        result = self.estimate(price, volume, volatility, atr=atr)
        return round(price * (1 - result['slippage_ratio']), 4)

    def get_statistics(self) -> Dict:
        """
        获取滑点统计信息

        Returns:
            滑点统计
        """
        if not self._history:
            return {
                'count': 0,
                'avg_slippage': self.params.base_slippage,
                'max_slippage': self.params.base_slippage,
                'min_slippage': self.params.base_slippage,
            }

        ratios = [h['slippage_ratio'] for h in self._history]
        return {
            'count': len(self._history),
            'avg_slippage': round(float(np.mean(ratios)), 6),
            'max_slippage': round(float(np.max(ratios)), 6),
            'min_slippage': round(float(np.min(ratios)), 6),
            'median_slippage': round(float(np.median(ratios)), 6),
            'p90_slippage': round(float(np.percentile(ratios, 90)), 6),
        }

    def update_params(self, new_params: Optional[SlippageParams] = None) -> None:
        """
        更新滑点参数

        Args:
            new_params: 新参数
        """
        if new_params:
            self.params = new_params
            logger.info(f"[DynamicSlippageModel] 参数已更新: base={self.params.base_slippage}")
