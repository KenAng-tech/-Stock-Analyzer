"""
Stock Analyzer - Main Flask Application
Web-based stock analysis system with real-time data and report generation

Enhanced with:
- Dynamic caching
- K-line data integration
- Dynamic Kelly position sizing
- Bayesian signal weight updating
- Structured logging
"""

import os
import sys
import json
import time
import threading
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
from flask_socketio import SocketIO
from flask_cors import CORS

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.data_fetcher import StockDataFetcher
from modules.analysis_engine import AnalysisEngine
from modules.report_generator import ReportGenerator
from modules.strategy_engine import StrategyEngine
from modules.kline_signal_analyzer import KlineSignalAnalyzer
from modules.atr_calculator import ATRCalculator, ADXCalculator

from modules.heatmap_generator import HeatmapGenerator
from modules.hmm_market_detector import MarketRegimeDetector as HMMRegimeDetector
from modules.factors.factor_orthogonalizer import FactorOrthogonalizer
from modules.transaction_cost_model import TransactionCostModel
from modules.alert_engine import AlertEngine
from modules.dynamic_cache import cache
from modules.logger import logger
from config import config

# Dashboard API Blueprint (P0-P3 量化模型仪表盘)
from modules.dashboard_api import bp as dashboard_bp

# Training API Blueprint (已在 modules/routes/__init__.py 中统一注册)

# WebSocket 事件处理器
from modules.routes.websocket_routes import init_socketio

# ML Predictor (for dashboard API)
from modules.ml_predictor import ml_predictor, model_training_scheduler
from modules.sota_training_scheduler import sota_scheduler

# P0 SOTA 优化: PatchTST + 概念漂移检测 + FinBERT
from modules.models.patchtst_integrator import get_patchtst, PatchTSTIntegrator
from modules.drift_monitor import get_drift_monitor, DriftMonitor
from modules.sentiment_engine import get_sentiment_engine, SentimentEngine

# P2 SOTA: TimesNet (ICML 2023) — 2026-09-20 断链修复: 原 import 写 modules.timesnet_predictor
# (顶层路径不存在) → 恒 None, sota_predict/sota_status 两活端点被饿 + health 永久 degraded
# (自检失明, 真故障会被假 degraded 掩盖)。真身 = modules.models.timesnet_predictor
# (09-17 迁移时 import 未跟进); sota 端点 hasattr+None 双守卫保留, 未训练态诚实空态。
timesnet_trainer = None
try:
    from modules.models.timesnet_predictor import TimesNetTrainer
    timesnet_trainer = TimesNetTrainer()
    logger.info("[App] TimesNet 加载成功 (models/ 真身)")
except Exception as e:
    logger.warning(f"[App] TimesNet 加载失败 (非断链): {e}")
    timesnet_trainer = None

# P2: Conformal Prediction (不确定性量化)
conformal_predictor = None
try:
    from modules.models.conformal_predictor import get_conformal_predictor
    conformal_predictor = get_conformal_predictor()
    logger.info("[App] ConformalPredictor 加载成功")
except Exception as e:
    logger.warning(f"[App] ConformalPredictor 加载失败: {e}")
    conformal_predictor = None

# Memory manager (lazy init)
memory_manager = None
try:
    from modules.memory_manager import get_memory_monitor
    memory_manager = get_memory_monitor()
    logger.debug(f"DIAGNOSTIC: memory_manager = {memory_manager}")
except Exception as e:
    logger.warning(f"[App] Memory manager 加载失败: {e}")

# P1 SOTA 优化: Diffusion + Mamba + Multi-Agent RL
try:
    from modules.models.diffusion_model import get_diffusion_predictor, DiffusionPredictor
except Exception as e:
    logger.warning(f"[App] Diffusion 模块加载失败: {e}")
    DiffusionPredictor = None

try:
    from modules.multi_agent_trading import get_multi_agent_coordinator, MultiAgentCoordinator
except Exception as e:
    logger.warning(f"[App] Multi-Agent 模块加载失败: {e}")
    MultiAgentCoordinator = None

try:
    from modules.models.hft_mamba import get_mamba_hft_predictor, MambaHFTPredictor
except Exception as e:
    logger.warning(f"[App] Mamba 模块加载失败: {e}")
    MambaHFTPredictor = None

# P2 SOTA 优化: 自监督预训练 + Qlib Alpha158
try:
    from modules.models.self_supervised import get_self_supervised_pretrainer, SelfSupervisedPretrainer
except Exception as e:
    logger.warning(f"[App] Self-Supervised 模块加载失败: {e}")
    SelfSupervisedPretrainer = None

try:
    from modules.factors.alpha158_calculator import get_alpha158_calculator, Alpha158Calculator
except Exception as e:
    logger.warning(f"[App] Alpha158 模块加载失败: {e}")
    Alpha158Calculator = None

# SOTA 优化: 动态因子权重 + GNN + CVaR/EVT + 概念漂移检测器 + 因子 IC 监控 + 跨市场因子
try:
    from modules.factor_weight_scheduler import get_factor_weight_scheduler, reset_factor_weight_scheduler
except Exception as e:
    logger.warning(f"[App] 因子权重调度器加载失败: {e}")

try:
    from modules.models.gnn_predictor import get_gnn_predictor, GNNPredictor
except Exception as e:
    logger.warning(f"[App] GNN 模块加载失败: {e}")
    GNNPredictor = None

try:
    from modules.cvar_evt_analyzer import get_cvar_analyzer
except Exception as e:
    logger.warning(f"[App] CVaR/EVT 模块加载失败: {e}")

try:
    from modules.concept_drift_detector import ConceptDriftDetector
    advanced_drift_detector = ConceptDriftDetector()
except Exception as e:
    logger.warning(f"[App] 高级概念漂移检测器加载失败: {e}")
    advanced_drift_detector = None

try:
    from modules.factors.factor_ic_monitor import FactorICMonitor, FactorSelection, ICDecay
except Exception as e:
    logger.warning(f"[App] 因子 IC 监控加载失败: {e}")

try:
    from modules.dynamic_ensemble import DynamicEnsemblePredictor, ModelWeightScheduler, MarketRegimeDetector
except Exception as e:
    logger.warning(f"[App] Dynamic Ensemble 模块加载失败: {e}")
    DynamicEnsemblePredictor = None

try:
    from modules.cross_market_factors import get_cross_market_factors, CrossMarketFactors
except Exception as e:
    logger.warning(f"[App] 跨市场因子加载失败: {e}")

# P2/P3: LLM Sentiment + TimesFM + Factor Weight Scheduler
try:
    from modules.llm_sentiment import get_llm_sentiment, LLMSentimentAnalyzer
except Exception as e:
    logger.warning(f"[App] LLM Sentiment 模块加载失败: {e}")
    LLMSentimentAnalyzer = None

try:
    from modules.timesfm_predictor import get_timesfm, TimesFMPredictor
except Exception as e:
    logger.warning(f"[App] TimesFM 模块加载失败: {e}")
    TimesFMPredictor = None

# P2/P3 SOTA 优化: PatchMamba + Conformal Prediction + Causal Discovery + Hierarchical RL + Adversarial Training + Foundation Model
try:
    from modules.patchmamba import get_patchmamba, PatchMamba
except Exception as e:
    logger.warning(f"[App] PatchMamba 加载失败: {e}")

try:
    from modules.causal_discovery import get_causal_discovery_engine, CausalDiscoveryEngine
except Exception as e:
    logger.warning(f"[App] Causal Discovery 加载失败: {e}")

try:
    from modules.drift_aware_pipeline import get_drift_aware_pipeline, reset_drift_aware_pipeline
except Exception as e:
    logger.warning(f"[App] Drift-Aware Pipeline 加载失败: {e}")

try:
    from modules.hierarchical_rl import get_hierarchical_rl_agent, HierarchicalRLAgent
except Exception as e:
    logger.warning(f"[App] Hierarchical RL 加载失败: {e}")

try:
    from modules.adversarial_training import get_adversarial_trainer, AdversarialTrainer, AdversarialConfig
except Exception as e:
    logger.warning(f"[App] Adversarial Training 加载失败: {e}")

try:
    from modules.foundation_model import get_foundation_model, FoundationModel
except Exception as e:
    logger.warning(f"[App] Foundation Model 加载失败: {e}")

# Initialize Flask app
app = Flask(__name__,
            template_folder='templates',
            static_folder='static')
app.config['TEMPLATES_AUTO_RELOAD'] = True

# CORS: 从配置读取允许的源，默认仅允许本地
_ALLOWED_ORIGINS = config.get('server.allowed_origins', ['http://127.0.0.1:5002', 'http://localhost:5002'])
CORS(app, origins=_ALLOWED_ORIGINS)

# API Key 认证 (可选): 通过 X-API-Key 请求头
_API_KEY = os.environ.get('STOCK_ANALYZER_API_KEY', config.get('server.api_key', ''))


def _require_api_key():
    """验证 API Key，未配置时跳过"""
    if not _API_KEY:
        return True
    provided = request.headers.get('X-API-Key', '')
    if provided == _API_KEY:
        return True
    logger.warning(f"[Auth] 无效 API Key: {provided[:8]}...")
    return False

# Initialize modules
data_fetcher = StockDataFetcher()
analysis_engine = AnalysisEngine()
report_generator = ReportGenerator()
strategy_engine = StrategyEngine()
kline_analyzer = KlineSignalAnalyzer()
atr_calculator = ATRCalculator()
# ── SocketIO 唯一实例 (threading 模式) ─────────────────────────
# 所有 WebSocket 事件处理器通过 init_socketio() 注册到此实例
# run_server.py 通过 from app import socketio 导入，不再创建新实例
socketio = SocketIO(
    app,
    cors_allowed_origins=_ALLOWED_ORIGINS,
    async_mode='threading',
    ping_timeout=60,
    ping_interval=25,
)
init_socketio(socketio)

# 注册核心依赖到单例模块 (消除循环导入)
from modules.dependencies import set_socketio, set_websocket_handler
set_socketio(socketio)
try:
    from modules.websocket_handler import get_websocket_handler
    set_websocket_handler(get_websocket_handler())
except Exception:
    pass
heatmap_generator = HeatmapGenerator()
alert_engine = AlertEngine()

# New optimization modules
hmm_detector = HMMRegimeDetector(n_states=3)
factor_orthogonalizer = FactorOrthogonalizer()
transaction_cost_model = TransactionCostModel()
adx_calculator = ADXCalculator()

# P0 SOTA 优化模块初始化
patchtst_integrator = get_patchtst()  # PatchTST 集成器
drift_monitor = get_drift_monitor()   # 概念漂移检测器
# 2026-09-08: 漂移→紧急重训链接通 — scheduler.should_retrain() 消费 drift_count
# (23:00/手动触发时 drift_count>3 → 立即重训+reset)。此前 set_drift_monitor 零调用
# 调用点 → scheduler 的漂移分支恒为死代码 (hasattr 恒 False)。feed 侧见
# analysis_engine._pending_drift_pair (跨时配对喂同一单例)
try:
    model_training_scheduler.set_drift_monitor(drift_monitor)
except Exception as _e:
    logger.error(f"漂移→重训链绑定失败: {_e}")
sentiment_engine = get_sentiment_engine()  # FinBERT 情感分析引擎

# RL Trader — 初始化并尝试加载已有模型
from modules.rl_trader_v2 import rl_trader_v2, TradingEnvV2, PPOAgentV2, SACAgentV2
from modules.portfolio_optimizer import PortfolioOptimizer
try:
    rl_trader_v2.load()
    logger.info("[RL Trader] 已加载已有模型")
except Exception as e:
    logger.warning(f"[RL Trader] 加载模型失败 (将使用随机策略): {e}")

# P1 SOTA 优化模块初始化
try:
    diffusion_predictor = get_diffusion_predictor()  # Diffusion 概率预测
except Exception as e:
    logger.warning(f"[App] Diffusion 初始化失败: {e}")
    diffusion_predictor = None

try:
    multi_agent_coordinator = get_multi_agent_coordinator()  # Multi-Agent RL
except Exception as e:
    logger.warning(f"[App] Multi-Agent 初始化失败: {e}")
    multi_agent_coordinator = None

try:
    mamba_hft = get_mamba_hft_predictor()  # Mamba 高频交易
except Exception as e:
    logger.warning(f"[App] Mamba 初始化失败: {e}")
    mamba_hft = None

# P2 SOTA 优化模块初始化 (长期优化)
try:
    self_supervised_pretrainer = get_self_supervised_pretrainer()  # 自监督预训练
except Exception as e:
    logger.warning(f"[App] Self-Supervised 初始化失败: {e}")
    self_supervised_pretrainer = None

try:
    alpha158_calculator = get_alpha158_calculator()  # Alpha158 因子计算
except Exception as e:
    logger.warning(f"[App] Alpha158 初始化失败: {e}")
    alpha158_calculator = None

# SOTA 优化模块: 动态因子权重 + GNN + CVaR/EVT + 概念漂移 + 跨市场因子
try:
    factor_weight_scheduler = get_factor_weight_scheduler()  # 动态因子权重调度器
    logger.info("[SOTA] 因子权重调度器已初始化")
except Exception as e:
    logger.warning(f"[App] 因子权重调度器初始化失败: {e}")
    factor_weight_scheduler = None

try:
    gnn_predictor = get_gnn_predictor()  # GNN 图神经网络预测器
    logger.info("[SOTA] GNN 预测器已初始化")
except Exception as e:
    logger.warning(f"[App] GNN 初始化失败: {e}")
    gnn_predictor = None

try:
    cvar_analyzer = get_cvar_analyzer()  # CVaR/EVT 风险分析器
    logger.info("[SOTA] CVaR/EVT 分析器已初始化")
except Exception as e:
    logger.warning(f"[App] CVaR/EVT 初始化失败: {e}")
    cvar_analyzer = None

try:
    factor_ic_monitor = FactorICMonitor()  # 因子 IC 监控
    logger.info("[SOTA] 因子 IC 监控已初始化")
except Exception as e:
    logger.warning(f"[App] 因子 IC 监控初始化失败: {e}")
    factor_ic_monitor = None

try:
    dynamic_ensemble = DynamicEnsemblePredictor()  # 动态集成预测器
    logger.info("[SOTA] 动态集成预测器已初始化")
except Exception as e:
    logger.warning(f"[App] 动态集成预测器初始化失败: {e}")
    dynamic_ensemble = None

try:
    cross_market_factors = CrossMarketFactors()  # 跨市场因子
    logger.info("[SOTA] 跨市场因子已初始化")
except Exception as e:
    logger.warning(f"[App] 跨市场因子初始化失败: {e}")
    cross_market_factors = None

# LLM Sentiment + TimesFM
try:
    llm_sentiment_analyzer = get_llm_sentiment()
except Exception as e:
    logger.warning(f"[App] LLM Sentiment 初始化失败: {e}")
    llm_sentiment_analyzer = None

try:
    timesfm_predictor = get_timesfm()
except Exception as e:
    logger.warning(f"[App] TimesFM 初始化失败: {e}")
    timesfm_predictor = None

# P2/P3 SOTA 优化模块初始化
try:
    patchmamba_model = get_patchmamba()  # PatchMamba 混合架构
    logger.info("[SOTA] PatchMamba 已初始化")
except Exception as e:
    logger.warning(f"[App] PatchMamba 初始化失败: {e}")
    patchmamba_model = None

try:
    causal_discovery_engine = get_causal_discovery_engine()  # 因果发现引擎
    logger.info("[SOTA] Causal Discovery Engine 已初始化")
except Exception as e:
    logger.warning(f"[App] Causal Discovery Engine 初始化失败: {e}")
    causal_discovery_engine = None

try:
    drift_aware_pipeline = get_drift_aware_pipeline()  # 概念漂移自动重训练 pipeline
    logger.info("[SOTA] Drift-Aware Pipeline 已初始化")
except Exception as e:
    logger.warning(f"[App] Drift-Aware Pipeline 初始化失败: {e}")
    drift_aware_pipeline = None

try:
    hierarchical_rl = get_hierarchical_rl_agent()  # Hierarchical RL 分层强化学习
    logger.info("[SOTA] Hierarchical RL Agent 已初始化")
except Exception as e:
    logger.warning(f"[App] Hierarchical RL 初始化失败: {e}")
    hierarchical_rl = None

try:
    adversarial_trainer = get_adversarial_trainer()  # Adversarial Training 对抗训练
    logger.info("[SOTA] Adversarial Training 已初始化")
except Exception as e:
    logger.warning(f"[App] Adversarial Training 初始化失败: {e}")
    adversarial_trainer = None

try:
    foundation_model = get_foundation_model(n_features=12)  # Foundation Model 统一架构 (n_features=12 与 ml_predictor 一致)
    logger.info("[SOTA] Foundation Model 已初始化 (n_features=12)")
except Exception as e:
    logger.warning(f"[App] Foundation Model 初始化失败: {e}")
    foundation_model = None

# P0-P3 新增模块初始化 (2026-08-07 深度优化)
# Causal Factor Selector
try:
    from modules.causal_factor_selector import CausalFactorSelector
    causal_factor_selector = CausalFactorSelector()
    logger.info("[SOTA] 因果因子选择器已初始化")
except Exception as e:
    logger.warning(f"[App] 因果因子选择器初始化失败: {e}")
    causal_factor_selector = None

# XAI Explainer
try:
    from modules.xai_explainer import XAIExplainer
    xai_explainer = XAIExplainer()
    logger.info("[SOTA] XAI 可解释性引擎已初始化")
except Exception as e:
    logger.warning(f"[App] XAI 引擎初始化失败: {e}")
    xai_explainer = None

# Time Series Ensemble (基础模型投票)
try:
    from modules.time_series_ensemble import TimeSeriesEnsemble
    ts_ensemble = TimeSeriesEnsemble()
    logger.info("[SOTA] 时序基础模型投票集成已初始化")
except Exception as e:
    logger.warning(f"[App] 时序集成初始化失败: {e}")
    ts_ensemble = None

# Chronos Predictor
try:
    from modules.models.chronos_predictor import ChronosPredictor
    chronos_predictor = ChronosPredictor()
    logger.info("[SOTA] Chronos 零样本预测器已初始化")
except Exception as e:
    logger.warning(f"[App] Chronos 预测器初始化失败: {e}")
    chronos_predictor = None

# Dynamic Graph GNN
try:
    from modules.dynamic_gnn import DynamicGraphBuilder, CrossMarketGNN
    dynamic_gnn_builder = DynamicGraphBuilder()
    dynamic_gnn = None  # 需要邻接矩阵才初始化
    logger.info("[SOTA] 动态图 GNN 构建器已初始化")
except Exception as e:
    logger.warning(f"[App] 动态图 GNN 初始化失败: {e}")
    dynamic_gnn_builder = None
    dynamic_gnn = None

# Drift-Aware Ensemble (P0)
try:
    from modules.drift_aware_ensemble import DriftAwareEnsemble
    drift_aware_ensemble = DriftAwareEnsemble()
    logger.info("[SOTA] 漂移感知集成已初始化")
except Exception as e:
    logger.warning(f"[App] 漂移感知集成初始化失败: {e}")
    drift_aware_ensemble = None

# Phase 2: 多智能体共识决策
try:
    # 先导入 unified_decision_engine
    from modules.unified_decision_engine import UnifiedDecisionEngine
    unified_decision_engine = UnifiedDecisionEngine()
    from modules import multi_agent_consensus as _consensus_mod
    _consensus_mod._consensus_engine = _consensus_mod.MultiAgentConsensus()
    _consensus_mod._consensus_engine.set_dependencies(unified_decision_engine, analysis_engine)
    logger.info("[Phase2] 多智能体共识引擎已初始化")
except Exception as e:
    logger.warning(f"[Phase2] 多智能体共识引擎初始化失败: {e}")
    _consensus_mod._consensus_engine = None

# Phase 2: SOTA 五阶段决策引擎 (2026-09-08 断链修复)
# /api/sota/decision 端点 hasattr(app_module, 'sota_engine') 恒 False —
# SOTAIntegrationEngine 全项目从未实例化 → 端点从未调用过真实决策流水线,
# 恒返回硬编码假 fallback (neutral/0.5)。实例化后: LLM 辩论(25s)→Factor→
# MultiModal→RL→Ensemble 真链 (8080 不可用时各阶段降级规则引擎, 非假数据)
try:
    from modules.sota_integration import SOTAIntegrationEngine
    sota_engine = SOTAIntegrationEngine()
    logger.info("[SOTA] SOTAIntegrationEngine 已初始化 (LLM 辩论/Factor/MultiModal/RL)")
except Exception as e:
    logger.warning(f"[SOTA] SOTAIntegrationEngine 初始化失败: {e}")
    sota_engine = None

# Phase 2: 自动重训练触发器
try:
    from modules.auto_retrain_trigger import AutoRetrainTrigger, get_auto_retrain_trigger
    auto_retrain_trigger = AutoRetrainTrigger()
    logger.info("[Phase2] 自动重训练触发器已初始化")
except Exception as e:
    logger.warning(f"[Phase2] 自动重训练触发器初始化失败: {e}")
    auto_retrain_trigger = None

# Phase 2: LLM 因子提取器
try:
    from modules.llm_factor_extractor import LLMFactorExtractor, get_llm_factor_extractor
    llm_factor_extractor = LLMFactorExtractor()
    logger.info("[Phase2] LLM 因子提取器已初始化")
except Exception as e:
    logger.warning(f"[Phase2] LLM 因子提取器初始化失败: {e}")
    llm_factor_extractor = None

# Phase 3: Safe RL 安全约束层
try:
    from modules import safe_rl_constraint as _safe_rl_mod
    _safe_rl_mod._safe_rl = _safe_rl_mod.SafeRLConstraintLayer()
    logger.info("[Phase3] Safe RL 安全约束层已初始化")
except Exception as e:
    logger.warning(f"[Phase3] Safe RL 安全约束层初始化失败: {e}")
    _safe_rl_mod._safe_rl = None

# Phase 3: 贝叶斯不确定性量化
try:
    from modules import bayesian_uncertainty as _bayesian_mod
    _bayesian_mod._bayesian = _bayesian_mod.BayesianUncertainty()
    logger.info("[Phase3] 贝叶斯不确定性量化已初始化")
except Exception as e:
    logger.warning(f"[Phase3] 贝叶斯不确定性量化初始化失败: {e}")
    _bayesian_mod._bayesian = None

# Phase 3: 在线学习 + EWC 防遗忘
try:
    from modules import online_learning_ewc as _ewc_mod
    _ewc_mod._online_ewc = _ewc_mod.OnlineLearningEWC()
    logger.info("[Phase3] 在线学习 EWC 防遗忘已初始化")
except Exception as e:
    logger.warning(f"[Phase3] 在线学习 EWC 防遗忘初始化失败: {e}")
    _ewc_mod._online_ewc = None

# Phase 4: MLOps 自动化流水线
try:
    from modules.mlops_pipeline import MLopsPipeline, get_mlops_pipeline
    mlops_pipeline = MLopsPipeline()
    logger.info("[Phase4] MLOps 自动化流水线已初始化")
except Exception as e:
    logger.warning(f"[Phase4] MLOps 流水线初始化失败: {e}")
    mlops_pipeline = None

# Phase 4: 实时图神经网络
# 2026-09-03 修: 路径错误 modules.realtime_gnn → modules.models.realtime_gnn
# (文件实际在 modules/models/ 下, 旧路径恒 ImportError → realtime_gnn 静默 None)
try:
    from modules.models.realtime_gnn import RealTimeGNN, get_realtime_gnn
    realtime_gnn = RealTimeGNN()
    logger.info("[Phase4] 实时图神经网络已初始化")
except Exception as e:
    logger.warning(f"[Phase4] 实时 GNN 初始化失败: {e}")
    realtime_gnn = None

# 初始化日志
logger.info("=" * 60)
logger.info("[App] SOTA 优化模块已加载:")
logger.info(f"  - PatchTST: {'已训练' if patchtst_integrator.is_trained() else '未训练'}")
logger.info(f"  - Drift Monitor: drift_count={drift_monitor._drift_count}, last_drift={drift_monitor._last_drift_time or 'N/A'}")
logger.info(f"  - Sentiment Engine: method={sentiment_engine._finbert._use_hf if sentiment_engine._finbert else 'dictionary'}")
logger.info("=" * 60)

# Register Dashboard API Blueprint
app.register_blueprint(dashboard_bp)

# Training API Blueprint 由 modules/routes/__init__.py 的 register_blueprints() 统一注册
from modules.routes import register_blueprints
register_blueprints(app)

# ── API Key 认证钩子 ──────────────────────────────────────
# 公开路由 (不需要认证)
_PUBLIC_ROUTES = {
    '/', '/webgui.html', '/dl_dashboard.html', '/sota_dashboard.html',
    '/api/health', '/api/stock/<stock_code>', '/api/stock/enhanced/<stock_code>',
    '/api/ths/realtime/', '/api/ths/klines/', '/api/ths/valuation/',
    '/api/ths/resolve', '/api/ths/status', '/api/ths/sync',
    '/api/ths/dragon-tiger', '/api/ths/limit-up-pool', '/api/ths/hot-stocks',
    '/static/<path:filename>',
}


def _is_public_route(rule):
    """检查路由是否公开"""
    if not rule:
        return False
    for public in _PUBLIC_ROUTES:
        if rule == public:
            return True
        # 通配符匹配: /api/stock/<stock_code>
        if '<' not in public and rule.startswith(public.rstrip('/<*>')):
            return True
    return False


@app.before_request
def _auth():
    """API Key 认证 (仅在配置了 API Key 时生效)"""
    if not _API_KEY:
        return None  # 未配置 API Key，跳过认证
    # 跳过静态文件和公开路由
    if request.path.startswith('/static/'):
        return None
    # 修复: 检查 request.path 而非 request.endpoint
    if _is_public_route(request.path):
        return None
    # 需要认证
    if not _require_api_key():
        return jsonify({'error': '未授权，请提供有效的 X-API-Key'}), 401


# ── 链式追踪 (2026-09-20 整合②, PanWatch 链尾可观测式) ────────────
# 每个 /api/* 请求 = 一个 trace: 取数→缓存→LLM→审计 全链日志同 trace,
# grep 一即串全链; 响应头 X-Trace-Id 让调用方/curl 直接对账日志。

@app.before_request
def _trace_bind():
    """请求级 trace 绑定 (非链式请求零注入 = 零污染)。"""
    from modules.log_context import clear_trace, gen_trace_id, set_trace
    if request.path.startswith('/api/'):
        clear_trace()
        target = '_'.join(request.path.strip('/').split('/')[1:3])
        set_trace(gen_trace_id('req', target))


@app.after_request
def _trace_header(response):
    """响应回显 X-Trace-Id = 链尾可观测 (响应 ↔ 日志一一对账)。"""
    from modules.log_context import get_trace
    tid = get_trace()
    if tid:
        response.headers['X-Trace-Id'] = tid
    return response


# ── 优雅关闭 ──────────────────────────────────────
import atexit
atexit.register(lambda: data_fetcher.close())


# Default stock configuration
DEFAULT_STOCK = {
    'code': 'sz300620',
    'name': '光库科技',
    'industry': '光通信',
    'cost_basis': 120
}

# ============================================================================
# 注意: 所有 API 路由已迁移至 modules/routes/ 下的 Blueprint 文件
# 包括: data_routes, backtest_routes, factor_routes, sota_*, quant_routes,
#       health_routes, cache_routes, alert_routes, config_routes,
#       extra_routes, static_routes, monitor_routes, regime_routes 等 44 个文件
# ============================================================================
