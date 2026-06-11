"""
Factor Mining Module - 基于 AlphaCrafter 和 QuantaAlpha 的因子挖掘

SOTA Reference:
- AlphaCrafter (NJU, 2026-05): 连续自适应因子发现
- QuantaAlpha (SUFE, 2026-02): LLM 驱动的自进化因子挖掘
- FactorMiner (THU, 2026-02): 自进化 Agent 金融 Alpha 发现
"""

from .factor_mining import FactorMiningEngine, Factor, FactorEvaluator

__all__ = ['FactorMiningEngine', 'Factor', 'FactorEvaluator']
