"""
SOTA Quantitative Model Integration Layer

整合所有 SOTA 量化模型到 Stock Analyzer:

架构:
┌─────────────────────────────────────────────────────────────────────────┐
│                    Stock Analyzer SOTA Integration                      │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │
│  │ LLM Multi-Agent │  │ Factor Mining   │  │ Multi-Modal     │         │
│  │ (TradingAgents) │  │ (AlphaCrafter)  │  │ (FCMR/MM-DREX)  │         │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘         │
│           │                    │                    │                    │
│           └────────────────────┼────────────────────┘                    │
│                                ▼                                        │
│                    ┌─────────────────────┐                              │
│                    │  Decision Engine    │                              │
│                    │  (Ensemble Layer)   │                              │
│                    └────────┬────────────┘                              │
│                             │                                           │
│                             ▼                                           │
│                    ┌─────────────────────┐                              │
│                    │  RL Execution       │                              │
│                    │  (Trading-R1)       │                              │
│                    └─────────────────────┘                              │
└─────────────────────────────────────────────────────────────────────────┘

SOTA References:
- TradingAgents v0.2.5 (Tauric Research, 2026-05) - 44,000+ Stars
- AlphaCrafter (NJU, 2026-05) - Full-Stack Multi-Agent Framework
- QuantAgent (SBU, 2025-09) - Price-Driven HFT
- Trading-R1 (UCLA, UW, Stanford, ICLR 2026) - RL Trading
- FCMR (Hanyang University, ACL 2025) - Cross-Modal Reasoning
- QuantaAlpha (SUFE, 2026-02) - LLM Factor Mining
"""

import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from modules.logger import logger
from modules.llm_agents.llm_client import LLMClient
from modules.llm_agents.agent_coordinator import AgentCoordinator, AgentDecision
from modules.factor_mining.factor_mining import FactorMiningEngine
from modules.multi_modal.cross_modal import MultiModalEngine


@dataclass
class SOTADecision:
    """SOTA 综合决策结果"""
    # LLM Multi-Agent 层
    llm_decision: Dict = field(default_factory=dict)
    
    # Factor Mining 层
    new_factors: List[Dict] = field(default_factory=list)
    factor_scores: Dict = field(default_factory=dict)
    
    # Multi-Modal 层
    cross_modal: Dict = field(default_factory=dict)
    
    # RL Execution 层
    rl_action: str = "hold"
    rl_confidence: float = 0.0
    rl_position_size: float = 0.0
    
    # Ensemble 层
    ensemble_score: float = 0.0
    ensemble_direction: str = "neutral"
    
    # 元数据
    timestamp: float = field(default_factory=time.time)
    execution_time_ms: float = 0.0


class SOTAIntegrationEngine:
    """
    SOTA 量化模型集成引擎
    
    整合所有 SOTA 模型形成统一的决策流水线:
    1. LLM Multi-Agent 决策 (TradingAgents)
    2. Factor Mining (AlphaCrafter)
    3. Multi-Modal Reasoning (FCMR)
    4. RL Execution (Trading-R1)
    5. Ensemble Aggregation
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # 初始化 LLM 客户端
        llm_config = {
            "omlx_url": self.config.get("omlx_url", "http://127.0.0.1:8080"),
            "omlx_model": self.config.get("omlx_model", "default"),
            "omlx_timeout": 10.0,  # 降低 OMLX 超时以加速降级
            "openai_key": self.config.get("openai_key", ""),
            "anthropic_key": self.config.get("anthropic_key", ""),
            "failure_threshold": 2,  # 降低熔断阈值
            "recovery_timeout": 10.0,
        }
        self.llm_client = LLMClient(llm_config)
        
        # 初始化各层模块
        self.agent_coordinator = AgentCoordinator(self.llm_client, self.config)
        self.factor_mining = FactorMiningEngine(self.llm_client)
        self.multi_modal = MultiModalEngine(self.llm_client)
        
        # 线程锁用于并发安全
        self._lock = threading.Lock()
        
        # 缓存最近决策
        self._recent_decisions = []
        self._max_cache_size = 100

        # 全局超时: 每个 LLM 阶段最多 15 秒
        self.stage_timeout = 15.0

    def _run_with_timeout(self, fn, args=(), kwargs=None, timeout=None):
        """在超时限制内运行函数，返回 (result, success)"""
        if kwargs is None:
            kwargs = {}
        timeout = timeout or self.stage_timeout

        result_container = [None]
        error_container = [None]

        def _wrap():
            try:
                result_container[0] = fn(*args, **kwargs)
            except Exception as e:
                error_container[0] = e

        thread = threading.Thread(target=_wrap)
        thread.daemon = True
        thread.start()
        thread.join(timeout=timeout)

        if thread.is_alive():
            logger.warning(f"[SOTAEngine] 阶段超时 ({timeout}s)，跳过")
            return None, False
        if error_container[0]:
            raise error_container[0]
        return result_container[0], True
    
    def make_decision(self, stock_data: Dict, portfolio_state: Dict = None) -> SOTADecision:
        """
        执行完整的 SOTA 决策流水线
        
        Args:
            stock_data: 股票数据
            portfolio_state: 投资组合状态
            
        Returns:
            SOTADecision: 综合决策结果
        """
        start_time = time.time()
        
        try:
            logger.info("[SOTAEngine] 开始 SOTA 决策流水线")
            
            # 阶段 1: LLM Multi-Agent 决策
            logger.info("[SOTAEngine] 阶段 1: LLM Multi-Agent 决策")
            agent_decision = self.agent_coordinator.make_decision(stock_data, portfolio_state)
            llm_decision = self.agent_coordinator.get_decision_json(agent_decision)
            
            # 阶段 2: Factor Mining (带超时)
            logger.info("[SOTAEngine] 阶段 2: Factor Mining")
            try:
                factors, ok = self._run_with_timeout(
                    self.factor_mining.mine_and_evaluate, (stock_data, {})
                )
                if ok and factors:
                    factor_scores = {f.name: {"ic": f.ic, "icir": f.icir, "efficacy": f.efficacy} for f in factors}
                else:
                    logger.warning("[SOTAEngine] Factor Mining 超时/失败，使用空因子")
                    factor_scores = {}
                    factors = []
            except Exception as e:
                logger.error(f"[SOTAEngine] Factor Mining 异常: {e}")
                factor_scores = {}
                factors = []

            # 阶段 3: Multi-Modal Reasoning (带超时)
            logger.info("[SOTAEngine] 阶段 3: Multi-Modal Reasoning")
            try:
                cross_modal, ok = self._run_with_timeout(self.multi_modal.analyze, (stock_data,))
                if not ok:
                    cross_modal = {"consistency_score": 0.5, "final_direction": "neutral"}
            except Exception as e:
                logger.error(f"[SOTAEngine] Multi-Modal 异常: {e}")
                cross_modal = {"consistency_score": 0.5, "final_direction": "neutral"}

            # 阶段 4: RL Execution (简化版 - 基于现有 RL Trader)
            logger.info("[SOTAEngine] 阶段 4: RL Execution")
            rl_action, rl_confidence, rl_position = self._execute_rl(stock_data)

            # 阶段 5: 市场状态检测 (用于动态权重)
            logger.info("[SOTAEngine] 阶段 5: 市场状态检测")
            market_regime = 'sideways'
            try:
                from modules.hmm_market_detector import MarketRegimeDetector
                regime_detector = MarketRegimeDetector()
                klines_data = stock_data.get('klines', stock_data.get('kline_data', None))
                if klines_data:
                    regime_result = regime_detector.detect_regime(klines_data)
                    market_regime = regime_result.get('regime', 'sideways')
            except Exception as e:
                logger.debug(f"[SOTAEngine] 市场状态检测失败: {e}")

            # 阶段 6: Ensemble Aggregation (动态权重)
            logger.info("[SOTAEngine] 阶段 6: Ensemble Aggregation (regime={})".format(market_regime))
            ensemble_score, ensemble_direction = self._ensemble_aggregate(
                llm_decision, factor_scores, cross_modal, rl_action,
                market_regime=market_regime
            )
            
            # 计算执行时间
            execution_time_ms = (time.time() - start_time) * 1000
            
            # 构建决策结果
            decision = SOTADecision(
                llm_decision=llm_decision,
                new_factors=[{"name": f.name, "efficacy": f.efficacy} for f in factors],
                factor_scores=factor_scores,
                cross_modal=cross_modal,
                rl_action=rl_action,
                rl_confidence=rl_confidence,
                rl_position_size=rl_position,
                ensemble_score=ensemble_score,
                ensemble_direction=ensemble_direction,
                execution_time_ms=execution_time_ms
            )
            
            # 缓存决策
            self._cache_decision(decision)
            
            logger.info(f"[SOTAEngine] 决策完成! 执行时间: {execution_time_ms:.1f}ms")
            return decision
            
        except Exception as e:
            logger.error(f"[SOTAEngine] 决策流水线错误: {e}")
            import traceback
            traceback.print_exc()
            
            return SOTADecision(
                llm_decision={"error": str(e)},
                ensemble_direction="neutral",
                ensemble_score=0.3,
                execution_time_ms=(time.time() - start_time) * 1000
            )
    
    def _execute_rl(self, stock_data: Dict) -> tuple:
        """
        RL 执行层 - 基于现有 RL Trader 的简化实现
        
        返回: (action, confidence, position_size)
        """
        try:
            # 使用现有的 RL Trader
            from modules.rl_trader_v2 import TradingEnvV2
            
            # 获取价格数据
            prices = stock_data.get("prices", [])
            if not prices:
                return "hold", 0.5, 0.0
            
            # 创建交易环境
            env = TradingEnvV2(
                prices=prices,
                features=[],
                initial_capital=1000000,
                transaction_cost=0.0015
            )
            
            # 模拟 RL 决策
            action = env.get_action(stock_data)
            
            return action, 0.6, 0.1  # 默认仓位 10%
            
        except Exception as e:
            logger.warning(f"[SOTAEngine] RL 执行失败: {e}")
            return "hold", 0.5, 0.0
    
    def _ensemble_aggregate(self, llm_decision: Dict,
                           factor_scores: Dict,
                           cross_modal: Dict,
                           rl_action: str,
                           market_regime: str = 'sideways') -> tuple:
        """
        动态集成聚合 — 根据 HMM 市场状态调整权重

        原理:
            - 牛市: RL 权重提升 (趋势跟踪效果好), 因子权重次之
            - 熊市: LLM 权重提升 (宏观判断更重要), RL 权重次之
            - 震荡市: 因子权重提升 (均值回归策略更有效)

        权重配置:
            regime     | LLM | Factor | MultiModal | RL
            ----------|-----|--------|------------|-----
            bullish   | 0.30| 0.25   | 0.20       | 0.25
            bearish   | 0.35| 0.20   | 0.15       | 0.30
            sideways  | 0.25| 0.35   | 0.25       | 0.15
        """
        # 不同 regime 的最优权重
        regime_weights = {
            'bullish':   {'llm': 0.30, 'factor': 0.25, 'multimodal': 0.20, 'rl': 0.25},
            'bearish':   {'llm': 0.35, 'factor': 0.20, 'multimodal': 0.15, 'rl': 0.30},
            'sideways':  {'llm': 0.25, 'factor': 0.35, 'multimodal': 0.25, 'rl': 0.15},
        }
        weights = regime_weights.get(market_regime, regime_weights['sideways'])

        # LLM 方向得分
        direction_scores = {
            "bullish": 0.7,
            "bearish": 0.3,
            "neutral": 0.5
        }
        llm_score = direction_scores.get(
            llm_decision.get("research_direction", "neutral"), 0.5
        )

        # 因子得分 (平均因子效能)
        avg_factor_efficacy = 0.0
        if factor_scores:
            avg_factor_efficacy = sum(
                f.get("efficacy", 0) for f in factor_scores.values()
            ) / len(factor_scores)

        # 跨模态得分
        cross_modal_score = cross_modal.get("consistency_score", 0.5)

        # RL 动作得分
        rl_scores = {"buy": 0.7, "sell": 0.3, "hold": 0.5}
        rl_score = rl_scores.get(rl_action, 0.5)

        # 加权聚合
        ensemble_score = (
            llm_score * weights['llm'] +
            avg_factor_efficacy * weights['factor'] +
            cross_modal_score * weights['multimodal'] +
            rl_score * weights['rl']
        )

        # 确定方向
        if ensemble_score > 0.6:
            direction = "bullish"
        elif ensemble_score < 0.4:
            direction = "bearish"
        else:
            direction = "neutral"

        return round(ensemble_score, 3), direction
    
    def _cache_decision(self, decision: SOTADecision):
        """缓存决策结果"""
        with self._lock:
            self._recent_decisions.append(decision)
            if len(self._recent_decisions) > self._max_cache_size:
                self._recent_decisions.pop(0)
    
    def get_recent_decisions(self, limit: int = 10) -> List[Dict]:
        """获取最近的决策"""
        with self._lock:
            return [
                {
                    "timestamp": d.timestamp,
                    "ensemble_direction": d.ensemble_direction,
                    "ensemble_score": d.ensemble_score,
                    "rl_action": d.rl_action,
                    "execution_time_ms": d.execution_time_ms
                }
                for d in self._recent_decisions[-limit:]
            ]
    
    def get_model_status(self) -> Dict:
        """获取模型状态"""
        return {
            "llm_client": "initialized",
            "agent_coordinator": "initialized",
            "factor_mining": "initialized",
            "multi_modal": "initialized",
            "rl_executor": "initialized",
            "uptime": time.time()
        }
