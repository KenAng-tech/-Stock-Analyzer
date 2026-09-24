"""
Stock Analyzer Modules - Enhanced with Advanced Optimizations
"""

from .strategy_engine import StrategyEngine
from .kline_signal_analyzer import KlineSignalAnalyzer
from .atr_calculator import ATRCalculator, ADXCalculator
from .data_fetcher import StockDataFetcher
from .analysis_engine import AnalysisEngine
from .alert_engine import AlertEngine
from .report_generator import ReportGenerator
from .websocket_handler import WebSocketFundFlowHandler
from .fund_flow_optimizer import FundFlowOptimizer
from .heatmap_generator import HeatmapGenerator
from .portfolio_optimizer import PortfolioOptimizer
from .hmm_market_detector import MarketRegimeDetector
from .transaction_cost_model import TransactionCostModel

# 因子计算 (已重组到 factors/)
from .factors.factor_orthogonalizer import FactorOrthogonalizer
from .factors.alpha158_calculator import Alpha158Calculator, get_alpha158_calculator
from .factors.alpha360_calculator import Alpha360Calculator, get_alpha360_calculator
from .factors.barra_cne6_calculator import BarraCNE6Calculator, get_barra_cne6_calculator
from .factors.factor_ic_monitor import (
    FactorICMonitor, FactorSelection, ICDecay,
    ICQQPlot, FactorTurnover, LongShortAnalyzer, FactorGroupIC,
    factor_ic_monitor, factor_turnover, long_short_analyzer, factor_group_ic,
)

# 预测模型 (已重组到 models/)
try:
    from .models import (
        ChronosPredictor, DiffusionPredictor, DRLTradingAgent, FoundationModel,
        GNNPredictor, MambaClassifier, MoiraiPredictor,
        PatchTSTIntegrator, PatchTST, PatchTSTTrainer, RealTimeGNN,
        MaskedTimeSeriesModel,
        TimeLLM, TimeMoEPredictor, TimesFMPredictor, TimesNet,
        ConformalPredictor,
        TemporalStackingEnsemble, BayesianModelAverager, HybridEnsemble, EnsembleResult,
    )
except ImportError as e:
    import logging
    logging.warning(f"[Modules] 预测模型导入失败 (PyTorch 未安装?): {e}")

# 执行算法
from .execution import (
    VWAPExecutor, TWAPExecutor, ExecutionResult,
    ImpactCostModel, ImpactParameters,
    DynamicSlippageModel, SlippageParams,
)

# ML 预测器 (保留在根目录)
from .ml_predictor import MLPredictor, ml_predictor, ModelTrainingScheduler

__all__ = [
    'StrategyEngine',
    'KlineSignalAnalyzer',
    'ATRCalculator',
    'ADXCalculator',
    'StockDataFetcher',
    'AnalysisEngine',
    'AlertEngine',
    'ReportGenerator',
    'WebSocketFundFlowHandler',
    'FundFlowOptimizer',
    'HeatmapGenerator',
    'PortfolioOptimizer',
    'MarketRegimeDetector',
    'TransactionCostModel',
    'Alpha158Calculator',
    'get_alpha158_calculator',
    'Alpha360Calculator',
    'get_alpha360_calculator',
    'BarraCNE6Calculator',
    'get_barra_cne6_calculator',
    'FactorICMonitor',
    'FactorSelection',
    'ICDecay',
    'ICQQPlot',
    'FactorTurnover',
    'LongShortAnalyzer',
    'FactorGroupIC',
    'factor_ic_monitor',
    'factor_turnover',
    'long_short_analyzer',
    'factor_group_ic',
    'FactorOrthogonalizer',
    # 预测模型
    'ChronosPredictor', 'DiffusionPredictor', 'DRLTradingAgent', 'FoundationModel',
    'GNNPredictor', 'MambaClassifier', 'MoiraiPredictor',
    'PatchTSTIntegrator', 'PatchTST', 'PatchTSTTrainer', 'RealTimeGNN',
    'MaskedTimeSeriesModel',
    'TimeLLM', 'TimeMoEPredictor', 'TimesFMPredictor', 'TimesNet',
    'ConformalPredictor',
    'TemporalStackingEnsemble', 'BayesianModelAverager', 'HybridEnsemble', 'EnsembleResult',
    # 执行算法
    'VWAPExecutor', 'TWAPExecutor', 'ExecutionResult',
    'ImpactCostModel', 'ImpactParameters',
    'DynamicSlippageModel', 'SlippageParams',
    # ML
    'MLPredictor', 'ml_predictor', 'ModelTrainingScheduler',
]
