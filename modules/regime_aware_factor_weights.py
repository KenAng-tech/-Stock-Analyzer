#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Regime-Aware 动态因子权重

根据市场状态 (Regime) 动态调整因子权重，不同 regime 下最优因子不同：
- bullish (牛市): 动量因子权重高
- bearish (熊市): 质量/防御因子权重高
- sideways (震荡): 均值回归因子权重高
- volatile (高波动): 低波动因子权重高

参考:
- 2026 量化最佳实践: Regime-Switching 因子权重 +50-150bps alpha
- Renaissance: 多状态 regime 模型 + 半马尔可夫过程
- Citadel: 分层 regime 模型 (宏观 + 行业)

使用:
    from modules.regime_aware_factor_weights import RegimeFactorWeights
    weights = RegimeFactorWeights()
    regime_weights = weights.get_weights('bullish')
"""

import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
from modules.logger import logger

# ── Regime 定义 ──────────────────────────────────────────────

REGIMES = ['bullish', 'bearish', 'sideways', 'volatile']


# 各 regime 下的因子权重向量 (基于因子类别)
# 因子类别: momentum, value, volatility, volume, liquidity, quality, technical, sentiment
REGIME_WEIGHT_TEMPLATES = {
    'bullish': {
        'momentum': 0.25,      # 动量在趋势市中最强
        'technical': 0.20,     # 技术指标在趋势市中有效
        'volume': 0.15,        # 成交量动量确认趋势
        'sentiment': 0.10,     # 情绪偏正面
        'quality': 0.15,       # 质量因子中等
        'value': 0.05,         # 价值因子在牛市中较弱
        'volatility': 0.03,    # 低波动因子在牛市中收益低
        'liquidity': 0.07,     # 流动性因子中等
    },
    'bearish': {
        'quality': 0.25,       # 质量因子在熊市中防御性强
        'volatility': 0.20,    # 低波动因子在熊市中表现好
        'value': 0.15,         # 价值因子在熊市中提供安全边际
        'liquidity': 0.15,     # 流动性因子重要
        'momentum': 0.05,      # 动量因子在熊市中负面
        'technical': 0.05,     # 技术指标在熊市中可靠性低
        'volume': 0.05,        # 成交量在熊市中混乱
        'sentiment': 0.10,     # 情绪偏负面
    },
    'sideways': {
        'momentum': 0.10,      # 动量因子弱
        'technical': 0.20,     # 技术指标在震荡市中有效 (RSI/MACD 拐点)
        'volatility': 0.15,    # 波动率因子中等
        'volume': 0.20,        # 成交量因子确认区间边界
        'quality': 0.10,       # 质量因子中等
        'value': 0.10,         # 价值因子中等
        'sentiment': 0.10,     # 情绪中性
        'liquidity': 0.10,     # 流动性因子中等
    },
    'volatile': {
        'volatility': 0.30,    # 低波动因子在波动市中表现最好
        'quality': 0.20,       # 质量因子提供防御
        'liquidity': 0.15,     # 流动性因子至关重要
        'value': 0.10,         # 价值因子提供安全边际
        'momentum': 0.05,      # 动量因子在波动市中不可靠
        'technical': 0.05,     # 技术指标在波动市中噪音大
        'volume': 0.10,        # 成交量因子确认波动方向
        'sentiment': 0.05,     # 情绪因子在波动市中不可靠
    },
}


class RegimeFactorWeights:
    """
    Regime-Aware 动态因子权重管理器

    根据当前市场状态动态调整因子权重。
    支持 HMM 硬状态和 GMM 软概率两种模式。
    """

    def __init__(self):
        self._templates = dict(REGIME_WEIGHT_TEMPLATES)
        self._current_regime: str = 'sideways'
        self._regime_probabilities: Dict[str, float] = {
            'bullish': 0.25,
            'bearish': 0.25,
            'sideways': 0.50,
            'volatile': 0.00,
        }
        self._history: List[Dict] = []

    def get_weights(self, regime: str, soft_probs: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        获取指定 regime 下的因子权重

        Args:
            regime: 市场状态 ('bullish', 'bearish', 'sideways', 'volatile')
            soft_probs: GMM 软概率 {regime: prob}，如果提供则加权平均

        Returns:
            {factor_category: weight}
        """
        if regime not in self._templates:
            logger.warning(f"[RegimeWeights] 未知 regime: {regime}, 使用 sideways")
            regime = 'sideways'

        if soft_probs:
            # GMM 软概率: 加权平均所有 regime 的权重
            return self._weighted_average(soft_probs)
        else:
            # HMM 硬状态: 返回该 regime 的权重
            return dict(self._templates[regime])

    def _weighted_average(self, soft_probs: Dict[str, float]) -> Dict[str, float]:
        """根据软概率加权平均各 regime 权重"""
        result = {}
        total_weight = sum(soft_probs.get(r, 0) for r in REGIMES)
        if total_weight == 0:
            return dict(self._templates['sideways'])

        for cat in REGIMES[0]:  # 取第一个 template 的 key
            val = 0.0
            for regime in REGIMES:
                prob = soft_probs.get(regime, 0)
                template_val = self._templates[regime].get(cat, 0.05)
                val += prob * template_val
            result[cat] = round(val / total_weight, 4)

        return result

    def update_regime(self, regime: str, probabilities: Optional[Dict[str, float]] = None):
        """
        更新当前市场状态

        Args:
            regime: 当前市场状态
            probabilities: GMM 软概率 (可选)
        """
        self._current_regime = regime
        if probabilities:
            self._regime_probabilities = probabilities

        # 记录历史
        self._history.append({
            'timestamp': datetime.now().isoformat(),
            'regime': regime,
            'probabilities': dict(self._regime_probabilities),
        })

        # 保持最近 100 条
        if len(self._history) > 100:
            self._history = self._history[-100:]

        logger.info(
            f"[RegimeWeights] Regime 更新: {regime} "
            f"(probs={self._regime_probabilities})"
        )

    def get_status(self) -> Dict:
        """获取当前状态"""
        return {
            'current_regime': self._current_regime,
            'regime_probabilities': dict(self._regime_probabilities),
            'history_length': len(self._history),
            'templates': {r: list(self._templates[r].keys()) for r in REGIMES},
        }

    def get_weight_change(self, old_regime: str, new_regime: str) -> Dict[str, float]:
        """
        计算两个 regime 之间的权重变化

        Args:
            old_regime: 旧状态
            new_regime: 新状态

        Returns:
            {factor_category: weight_change}
        """
        old_w = self._templates.get(old_regime, self._templates['sideways'])
        new_w = self._templates.get(new_regime, self._templates['sideways'])

        changes = {}
        for cat in old_w:
            changes[cat] = round(new_w.get(cat, 0) - old_w.get(cat, 0), 4)

        return changes


# ── 全局单例 ─────────────────────────────────────────────────

regime_weights = RegimeFactorWeights()


def get_regime_factor_weights() -> RegimeFactorWeights:
    """获取全局 RegimeFactorWeights 实例"""
    return regime_weights
