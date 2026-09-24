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

import os
import time
import threading
import numpy as np
from typing import Dict, List
from dataclasses import dataclass, field

from modules.logger import logger
from modules.llm_agents.llm_client import LLMClient
from modules.llm_agents.agent_coordinator import AgentCoordinator
from modules.factor_mining import EnhancedFactorMiningEngine
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
            "omlx_model": self.config.get("omlx_model", "GLM-4.7-Flash-MLX-8bit"),
            "omlx_timeout": 30.0,
            "openai_key": self.config.get("openai_key", ""),
            "anthropic_key": self.config.get("anthropic_key", ""),
            "failure_threshold": 3,  # 熔断阈值
            "recovery_timeout": 30.0,
        }
        self.llm_client = LLMClient(llm_config)
        
        # 初始化各层模块
        self.agent_coordinator = AgentCoordinator(self.llm_client, self.config)
        self.factor_mining = EnhancedFactorMiningEngine(self.llm_client)
        self.multi_modal = MultiModalEngine(self.llm_client)
        
        # 线程锁用于并发安全
        self._lock = threading.Lock()
        
        # 缓存最近决策
        self._recent_decisions = []
        self._max_cache_size = 100

        # 全局超时: 每个 LLM 阶段最多 60 秒 (2026-09-09: 10→25→45→60 — Factor Mining 链
        # SR 符号回归 ~20s + LLM 挖掘 17s + 传统评估, 串行 ~40-55s; Multi-Modal 单调用
        # ~17s + sem 排队。60s = 实测上限 + 余量)
        self.stage_timeout = 60.0

        # 缓存最近决策 (避免重复计算)
        self._decision_cache = {}
        self._cache_ttl = 60.0  # 缓存有效期 60 秒
        self._cache_lock = threading.Lock()

        # ===== Dynamic Weighting: 滚动 Sharpe 权重追踪 =====
        self._signal_history: Dict[str, List[float]] = {
            'llm': [],
            'factor': [],
            'multimodal': [],
            'rl': [],
        }
        self._max_history = 60  # 保留最近 60 次决策记录
        self._rolling_window = 20  # 滚动窗口大小
        self._shrinkage_factor = 0.1  # Ledoit-Wolf 收缩系数，防止过拟合

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
        # 检查缓存
        stock_code = stock_data.get('code', stock_data.get('stock_code', 'default'))
        cached_decision = self._get_cached_decision(stock_code)
        if cached_decision:
            return cached_decision

        start_time = time.time()

        try:
            logger.info("[SOTAEngine] 开始 SOTA 决策流水线")

            # 阶段 1: LLM Multi-Agent 决策 (2026-09-09: 25→80→125→190s 包裹 AgentCoordinator
            # GLOBAL_TIMEOUT=180 — 80s 时内部链在 81s 被外层截断 (差 1s 丢弃); 190 = 180 + 10
            # 余量, 内外层预算对齐。8080 实测每调用 ~17s (TTFT 15s + decode), 完整链 ~159s)
            logger.info("[SOTAEngine] 阶段 1: LLM Multi-Agent 决策")
            try:
                agent_decision, ok = self._run_with_timeout(
                    self.agent_coordinator.make_decision,
                    (stock_data, portfolio_state),
                    timeout=190.0
                )
                if ok and agent_decision:
                    llm_decision = self.agent_coordinator.get_decision_json(agent_decision)
                else:
                    logger.warning("[SOTAEngine] LLM Multi-Agent 超时/失败，使用降级决策")
                    llm_decision = {
                        "research_direction": "neutral",
                        "reasoning": "LLM unavailable, using rule-based fallback",
                        "confidence": 0.5,
                        "agents_consensus": False,
                    }
            except Exception as e:
                logger.error(f"[SOTAEngine] LLM Multi-Agent 异常: {e}")
                llm_decision = {
                    "research_direction": "neutral",
                    "reasoning": f"LLM error: {e}",
                    "confidence": 0.5,
                    "agents_consensus": False,
                }
            
            # 阶段 2: Factor Mining (带超时; 2026-09-09 修复: 此前未解包 (result, ok)
            # 元组 → isinstance(factors, list) 恒 False → 因子链即使按时跑完也恒被丢弃)
            logger.info("[SOTAEngine] 阶段 2: Factor Mining")
            try:
                factors, _ok = self._run_with_timeout(
                    self.factor_mining.mine_and_evaluate, (stock_data, {}, stock_code)
                )
                if factors and isinstance(factors, list) and len(factors) > 0:
                    factor_scores = {f.get('name', ''): {"ic": f.get('ic', 0), "icir": f.get('icir', 0), "efficacy": f.get('efficacy', 0)} for f in factors}
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
                # 2026-09-09: f 是 dict (mine_and_evaluate 返回 List[Dict], 非 Factor dataclass)
                new_factors=[{"name": f.get('name', ''), "efficacy": f.get('efficacy', 0)} for f in factors],
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
        RL 执行层 — 加载训练好的 PPO Agent 进行推理

        返回: (action, confidence, position_size)
        """
        try:
            from modules.rl_trader_v2 import PPOAgentV2, TradingEnvV2
            import numpy as np

            prices = stock_data.get("prices", [])
            if not prices or len(prices) < 20:
                return "hold", 0.5, 0.0

            prices_arr = np.array(prices, dtype=np.float64)

            # 1. 尝试加载训练好的 PPO Agent
            agent = None
            agent_path = os.path.join(
                os.path.dirname(__file__), 'rl_models', 'ppo_agent.pkl'
            )
            if os.path.exists(agent_path):
                try:
                    agent = PPOAgentV2.load(agent_path)
                    logger.debug("[SOTAEngine] 已加载训练好的 PPO Agent")
                except Exception as e:
                    logger.debug(f"[SOTAEngine] PPO Agent 加载失败: {e}，使用启发式")
                    agent = None

            if agent is not None:
                # 2. 使用训练好的 Agent 进行推理
                # 构造状态向量: 匹配训练时的 state_dim (features + cash + position + pnl + time)
                recent = prices_arr[-60:]  # 用更长窗口计算技术指标
                n_features = agent.state_dim - 5  # 减去 cash/position/pnl/time

                # 计算技术指标作为特征
                features = []
                # 1. 价格水平
                features.append(prices_arr[-1] / prices_arr[0])  # 相对价格
                # 2-4. 短中长期均线比率
                for period in [5, 10, 20]:
                    if len(prices_arr) >= period:
                        ma = np.mean(prices_arr[-period:])
                        features.append(prices_arr[-1] / max(ma, 1e-8))
                # 5-7. 动量
                for period in [3, 5, 10]:
                    if len(prices_arr) >= period:
                        mom = (prices_arr[-1] - prices_arr[-period]) / max(prices_arr[-period], 1e-8)
                        features.append(mom)
                # 8-9. 波动率
                for period in [5, 20]:
                    if len(prices_arr) >= period:
                        features.append(np.std(prices_arr[-period:]))
                # 10-12. RSI 近似 (涨跌幅占比)
                if len(prices_arr) >= 14:
                    gains = np.maximum(0, np.diff(prices_arr[-14:]))
                    losses = np.maximum(0, -np.diff(prices_arr[-14:]))
                    rs = np.sum(gains) / max(np.sum(losses), 1e-8)
                    features.append(rs / (1 + rs))  # 归一化 RSI
                    features.append(np.sum(gains) / max(np.sum(gains + losses), 1e-8))
                features.append(prices_arr[-1] - np.mean(prices_arr[-20:]))  # 偏离均值

                # 截断或填充到 n_features 维
                while len(features) < n_features:
                    features.append(0.0)
                features = features[:n_features]

                # 添加 cash(1), position(1), pnl(1), time(1)
                state = np.array(features + [0.5, 0.0, 0.0, 0.5], dtype=np.float64)[:agent.state_dim]
                # 确保正好是 state_dim 维
                if len(state) < agent.state_dim:
                    state = np.pad(state, (0, agent.state_dim - len(state)))
                state = state[:agent.state_dim]

                action_idx, log_prob, value = agent.select_action(state.squeeze(), explore=False)
                # PPO action: 0=hold, 1=buy, 2=sell
                action_map = {0: 'hold', 1: 'buy', 2: 'sell'}
                action = action_map.get(action_idx, 'hold')
                confidence = float(1.0 - np.exp(-abs(value)))  # 用 value 映射为置信度
                confidence = min(max(confidence, 0.3), 0.95)
                position_size = 0.3 if action != 'hold' else 0.0
                return action, confidence, position_size

            # 3. 无训练模型 → 价格趋势启发式
            momentum = (prices_arr[-1] - prices_arr[-5]) / max(prices_arr[-5], 1e-8)
            if momentum > 0.02:
                return "buy", min(abs(momentum) * 10, 0.8), 0.2
            elif momentum < -0.02:
                return "sell", min(abs(momentum) * 10, 0.8), 0.2
            return "hold", 0.5, 0.0

        except Exception as e:
            logger.warning(f"[SOTAEngine] RL 执行失败: {e}")
            return "hold", 0.5, 0.0
    
    def _ensemble_aggregate(self, llm_decision: Dict,
                           factor_scores: Dict,
                           cross_modal: Dict,
                           rl_action: str,
                           market_regime: str = 'sideways') -> tuple:
        """
        动态集成聚合 — Regime-based + Rolling Sharpe 双重加权

        架构:
            1. 计算各信号源的得分 (LLM/Factor/MultiModal/RL)
            2. 基础权重: 根据 HMM regime 选择先验权重
            3. 动态调整: 基于滚动 Sharpe 比率调整权重
            4. 收缩融合: Ledoit-Wolf 风格收缩，防止过拟合

        权重配置 (regime-based prior):
            regime     | LLM | Factor | MultiModal | RL
            ----------|-----|--------|------------|-----
            bullish   | 0.30| 0.25   | 0.20       | 0.25
            bearish   | 0.35| 0.20   | 0.15       | 0.30
            sideways  | 0.25| 0.35   | 0.25       | 0.15

        动态权重 (rolling Sharpe):
            - 每个信号源维护一个滚动窗口内的决策记录
            - 根据滚动 Sharpe 比率调整权重
            - 数据不足时回退到 regime-based 先验
        """
        # ===== 第 1 步: 计算各信号源得分 =====
        direction_scores = {
            "bullish": 0.7,
            "bearish": 0.3,
            "neutral": 0.5
        }
        llm_score = direction_scores.get(
            llm_decision.get("research_direction", "neutral"), 0.5
        )

        avg_factor_efficacy = 0.0
        if factor_scores:
            avg_factor_efficacy = sum(
                f.get("efficacy", 0) for f in factor_scores.values()
            ) / len(factor_scores)

        cross_modal_score = cross_modal.get("consistency_score", 0.5)

        rl_scores = {"buy": 0.7, "sell": 0.3, "hold": 0.5}
        rl_score = rl_scores.get(rl_action, 0.5)

        # ===== 第 2 步: 记录本次决策到历史 =====
        signal_scores = {
            'llm': llm_score,
            'factor': avg_factor_efficacy,
            'multimodal': cross_modal_score,
            'rl': rl_score,
        }

        with self._lock:
            for key, score in signal_scores.items():
                history = self._signal_history[key]
                history.append(score)
                if len(history) > self._max_history:
                    history.pop(0)

            history = {k: list(v) for k, v in self._signal_history.items()}

        # ===== 第 3 步: 计算 regime-based 先验权重 =====
        regime_weights = {
            'bullish':   {'llm': 0.30, 'factor': 0.25, 'multimodal': 0.20, 'rl': 0.25},
            'bearish':   {'llm': 0.35, 'factor': 0.20, 'multimodal': 0.15, 'rl': 0.30},
            'sideways':  {'llm': 0.25, 'factor': 0.35, 'multimodal': 0.25, 'rl': 0.15},
        }
        prior_weights = regime_weights.get(market_regime, regime_weights['sideways'])

        # ===== 第 4 步: 计算 rolling Sharpe 动态权重 =====
        dynamic_weights = self._compute_rolling_sharpe_weights(history)

        # ===== 第 5 步: 收缩融合 (prior + dynamic) =====
        # w_final = (1 - alpha) * prior + alpha * dynamic
        # alpha 随数据量增长: 10 条数据 alpha=0.2, 30 条以上 alpha=0.6
        n_data = min(len(history['llm']), len(history['factor']),
                     len(history['multimodal']), len(history['rl']))
        alpha = min(0.6, max(0.1, 0.1 * n_data))

        final_weights = {}
        for key in prior_weights:
            prior = prior_weights[key]
            dynamic = dynamic_weights.get(key, 0.25)
            final_weights[key] = (1 - alpha) * prior + alpha * dynamic

        # 归一化
        total = sum(final_weights.values())
        final_weights = {k: v / total for k, v in final_weights.items()}

        # ===== 第 6 步: 加权聚合 =====
        ensemble_score = (
            llm_score * final_weights['llm'] +
            avg_factor_efficacy * final_weights['factor'] +
            cross_modal_score * final_weights['multimodal'] +
            rl_score * final_weights['rl']
        )

        # 确定方向
        if ensemble_score > 0.6:
            direction = "bullish"
        elif ensemble_score < 0.4:
            direction = "bearish"
        else:
            direction = "neutral"

        return round(ensemble_score, 3), direction

    def _compute_rolling_sharpe_weights(self,
                                        history: Dict[str, List[float]]) -> Dict:
        """
        基于滚动窗口 Sharpe 比率计算动态权重

        原理:
            - 每个信号源的得分序列视为"策略收益"
            - 计算滚动窗口内的 Sharpe 比率 (mean / std)
            - 用 softmax 将 Sharpe 转换为权重

        Args:
            history: {signal_name: [score_1, score_2, ...]}

        Returns:
            {signal_name: weight}
        """
        sources = list(history.keys())
        sharpes = {}

        for source in sources:
            scores = history[source]
            n = len(scores)

            if n < 3:
                sharpes[source] = 0.0
                continue

            # 使用滚动窗口 (最近 window 条)
            window = min(self._rolling_window, n)
            window_scores = scores[-window:]

            mean_score = np.mean(window_scores)
            std_score = np.std(window_scores)

            # Sharpe = mean / std (risk-free rate = 0)
            if std_score > 1e-10:
                sharpes[source] = float(mean_score / std_score)
            else:
                sharpes[source] = 0.0

        # Softmax 转换 (加温度系数防止权重过于集中)
        temperatures = [1.5, 2.0, 2.5]  # 多温度取平均，更平滑
        weights = {s: 0.0 for s in sources}

        for temp in temperatures:
            sharps_arr = np.array([sharpes[s] for s in sources])
            # 数值稳定 softmax
            sharps_shifted = sharps_arr - np.max(sharps_arr)
            exp_vals = np.exp(sharps_shifted / temp)
            probs = exp_vals / np.sum(exp_vals)

            for i, s in enumerate(sources):
                weights[s] += probs[i]

        # 平均温度
        for s in sources:
            weights[s] /= len(temperatures)

        return weights

    def get_weight_history(self) -> Dict:
        """
        获取权重历史记录 (用于前端可视化)

        Returns:
            {
                'signal_history': {llm: [...], factor: [...], ...},
                'regime_weights': {...},
                'n_decisions': int,
            }
        """
        with self._lock:
            return {
                'signal_history': {k: v[-30:] for k, v in self._signal_history.items()},
                'n_decisions': {k: len(v) for k, v in self._signal_history.items()},
                'total_decisions': min(len(v) for v in self._signal_history.values())
                if self._signal_history else 0,
            }
    
    def _cache_decision(self, decision: SOTADecision):
        """缓存决策结果"""
        with self._lock:
            self._recent_decisions.append(decision)
            if len(self._recent_decisions) > self._max_cache_size:
                self._recent_decisions.pop(0)

    def _get_cached_decision(self, stock_code: str, max_age: float = 60.0) -> Optional[SOTADecision]:
        """
        获取缓存的决策结果

        Args:
            stock_code: 股票代码
            max_age: 最大缓存年龄 (秒)，默认 60 秒

        Returns:
            缓存的决策或 None
        """
        with self._lock:
            now = time.time()
            for d in reversed(self._recent_decisions):
                if getattr(d, 'stock_code', '') == stock_code and (now - d.timestamp) < max_age:
                    logger.debug(f"[SOTAEngine] 缓存命中: {stock_code} (年龄: {now - d.timestamp:.1f}s)")
                    return d
        return None
    
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
