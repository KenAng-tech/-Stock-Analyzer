"""
factors/ - 因子计算子模块

包含所有因子计算、因子正交化、因子权重调度等。
"""

from .multi_factor_model_v2 import MultiFactorModelV2
from .alpha158_calculator import Alpha158Calculator
from .alpha360_calculator import Alpha360Calculator
from .barra_cne6_calculator import BarraCNE6Calculator
from .barra_risk_model import BarraStyleFactors, RiskDecomposer, RiskOptimizer
from .factor_ic_monitor import FactorICMonitor
from .factor_monitor import FactorICData, DecayWarning, FactorCorrelation
from .factor_orthogonalizer import FactorOrthogonalizer
from .factor_weight_scheduler import DynamicFactorWeightScheduler
from .factor_ensemble_optimizer import FactorEnsembleOptimizer
from .cross_market_factors import CrossMarketFactors
from .causal_factor_selector import CausalEffectResult, CausalReport, DoubleMLEstimator
from .enhanced_features import EnhancedFeatures
from .garch_volatility import GARCHVolatility
from .cross_asset_feature_engine import CrossAssetFeatureConfig, CrossAssetFeatureEngine

__all__ = [
    'MultiFactorModelV2',
    'Alpha158Calculator',
    'Alpha360Calculator',
    'BarraCNE6Calculator',
    'BarraStyleFactors', 'RiskDecomposer', 'RiskOptimizer',
    'FactorICMonitor',
    'FactorICData', 'DecayWarning', 'FactorCorrelation',
    'FactorOrthogonalizer',
    'DynamicFactorWeightScheduler',
    'FactorEnsembleOptimizer',
    'CrossMarketFactors',
    'CausalEffectResult', 'CausalReport', 'DoubleMLEstimator',
    'EnhancedFeatures',
    'GARCHVolatility',
    'CrossAssetFeatureConfig', 'CrossAssetFeatureEngine',
]
