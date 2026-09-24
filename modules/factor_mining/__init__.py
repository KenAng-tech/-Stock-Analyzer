"""
Factor Mining Module - 增强版因子挖掘

SOTA Reference:
- PySR (MilesCranmer, 2023): 符号回归因子挖掘
- AlphaCrafter (NJU, 2026): LLM 驱动因子发现
- QuantaAlpha (SUFE, 2026): 自进化因子挖掘
- Barra CNE6: 风格因子中性化
- Jacobsen (2023): IC Decay 分析
"""

from .factor_mining import FactorMiningEngine, Factor, FactorEvaluator
from .enhanced_factor_mining import (
    SymbolicRegressionEngine,
    SymbolicFactor,
    ICValidator,
    FactorMiningEngine as EnhancedFactorMiningEngine,
    FactorNeutralizer,
    ICDecayAnalyzer,
)

__all__ = [
    'FactorMiningEngine', 'Factor', 'FactorEvaluator',
    'SymbolicRegressionEngine', 'SymbolicFactor',
    'ICValidator', 'EnhancedFactorMiningEngine',
    'FactorNeutralizer', 'ICDecayAnalyzer',
]
