#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Agent Coordinator — Agent 协调器 (2026-06-13 升级)

升级内容:
1. 跨模态推理 (Cross-Modal Reasoning): 文本 + 数值 + 视觉模态融合
2. 增强辩论机制: 引入反驳 (Counter-Argument) 轮次
3. 改进置信度评分: 考虑分歧度 (Disagreement Score)
4. 快速路径 (Fast Path): 高置信度时跳过部分 Agent

SOTA Reference:
- TradingAgents v0.2.5 (Tauric Research, 2026-05)
- AlphaCrafter (NJU, 2026-05)
- QuantAgent (SBU, 2025-09)
- Cross-Modal Reasoning (CMR) for Trading
"""

import json
import threading
import time
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from modules.logger import logger
from modules.llm_agents.llm_client import LLMClient, RuleEngine
from modules.llm_agents.analyst_team import AnalystTeam
from modules.llm_agents.research_team import ResearchTeam
from modules.llm_agents.trader_agent import TraderAgent
from modules.llm_agents.risk_manager import RiskManager
from modules.llm_agents.analyst_team import AnalystInsight
from modules.llm_agents.portfolio_manager import PortfolioManager


@dataclass
class AgentDecision:
    """Agent 协调决策结果"""
    # 分析师团队
    analyst_consensus: str
    analyst_confidence: float
    analyst_insights: List[Dict]
    
    # 研究团队
    research_direction: str
    research_recommendation: str
    research_confidence: float
    
    # 交易代理
    trade_action: str
    trade_quantity: float
    trade_confidence: float
    
    # 风险管理
    risk_level: str
    risk_adjustment: str
    
    # 投资组合
    portfolio_allocation: Dict[str, float]
    rebalance_needed: bool
    
    # 综合评分
    overall_score: float
    timestamp: float = field(default_factory=time.time)


class AgentCoordinator:
    """Agent 协调器 - 完整的决策流水线"""

    # 全局超时: 25→75→115→180 秒 (2026-09-09 三轮预算重设计:
    # ① 25→75: 热态 45-69s 恒截断, 辩论链从未完整跑完;
    # ② 75→115: max_tokens 2048→512 (生成时间减半);
    # ③ 115→180: 实测 8080 每调用固定开销 ~15s (TTFT, 5 模型驻留显存争用) +
    #    enable_thinking=false 后短输出 ~17s/调用。完整链 (第2+次): analyst 40
    #    (4 并行 sem2 两批) + 辩论 51 (3 串行) + 尾部 4×17 = 68 ≈ 159s, 180 含余量)
    GLOBAL_TIMEOUT = 180.0

    def _check_timeout(self, start_time: float) -> bool:
        """检查是否超时，超时则抛出异常终止执行"""
        elapsed = time.time() - start_time
        if elapsed > self.GLOBAL_TIMEOUT:
            raise TimeoutError(f"AgentCoordinator 超时 ({elapsed:.0f}s > {self.GLOBAL_TIMEOUT}s)")
        return elapsed

    def __init__(self, llm_client: LLMClient, config: Dict = None):
        self.llm = llm_client
        self.config = config or {}

        # 初始化所有 Agent
        self.analyst_team = AnalystTeam(llm_client, timeout=12.0)  # 分析师 12s 超时
        self.research_team = ResearchTeam(llm_client)
        self.trader_agent = TraderAgent(llm_client)
        self.risk_manager = RiskManager(llm_client, config)
        self.portfolio_manager = PortfolioManager(llm_client, config)

        # 快速规则引擎分析师 (用于 LLM 不可用时)
        self._rule_analyst = None
    
    def make_decision(self, stock_data: Dict, portfolio_state: Dict = None) -> AgentDecision:
        """
        执行完整的 Agent 决策流水线 (2026-06-13 升级)

        升级:
        1. 快速路径: 当分析师共识高度一致时，跳过研究辩论
        2. 跨模态推理: 融合文本 + 数值 + 视觉信息
        3. 增强辩论: 引入反驳轮次
        4. 分歧度检测: 当 Agent 间分歧大时降低置信度
        5. 全局超时: 60 秒超时后降级到规则引擎
        """
        logger.info("[AgentCoordinator] 开始决策流水线")
        start_time = time.time()

        try:
            # 阶段 1: 分析师团队并行分析
            logger.info("[AgentCoordinator] 阶段 1: 分析师团队分析")

            # 快速路径: 如果 LLM 响应慢 (>10s)，切换到规则引擎分析师
            use_rule_analysts = False
            if not self._rule_analyst:
                self._rule_analyst = RuleEngine()
                use_rule_analysts = True  # 首次调用也使用规则

            # 如果 LLM 上次调用超时 (>50s)，切换到规则引擎 (2026-09-09: 15→50 —
            # 8080 实测每调用 TTFT ~15s + decode, 单调用 ~17s; 15s 阈值使第 2+ 次
            # 调用恒降级规则引擎, LLM 分析师永不参与)
            if not use_rule_analysts and hasattr(self, '_last_llm_elapsed') and self._last_llm_elapsed > 50.0:
                logger.info(f"[AgentCoordinator] LLM 上次耗时 {self._last_llm_elapsed:.0f}s，切换到规则引擎")
                use_rule_analysts = True

            # 如果超过 50 秒，强制切换到规则引擎 (2026-09-09: 20→50 随 GLOBAL_TIMEOUT 缩放)
            if not use_rule_analysts:
                elapsed = self._check_timeout(start_time)
                if elapsed > 50.0:
                    logger.warning(f"[AgentCoordinator] 已耗时 {elapsed:.0f}s，强制切换到规则引擎")
                    use_rule_analysts = True

            if use_rule_analysts:
                logger.info("[AgentCoordinator] 使用规则引擎分析师 (LLM 超时保护)")
                analyst_insights = self._get_rule_insights(stock_data)
            else:
                # 使用线程包装分析师团队调用，强制 40 秒超时
                analyst_insights = [None]
                analyst_error = [None]
                analyst_done = threading.Event()

                def _run_analysts():
                    try:
                        analyst_insights[0] = self.analyst_team.analyze(stock_data)
                    except Exception as e:
                        analyst_error[0] = e
                    finally:
                        analyst_done.set()

                analyst_thread = threading.Thread(target=_run_analysts, daemon=True)
                analyst_thread.start()
                analyst_done.wait(timeout=40.0)  # 2026-09-09: 15→25→40 (4 分析师并行但
                                                # llm_client Semaphore(2) → 2 批 × ~17-20s)

                if analyst_thread.is_alive():
                    logger.warning("[AgentCoordinator] 分析师团队 40 秒超时，切换到规则引擎")
                    self._rule_analyst = RuleEngine()
                    use_rule_analysts = True
                    analyst_insights = self._get_rule_insights(stock_data)
                elif analyst_error[0]:
                    logger.warning(f"[AgentCoordinator] 分析师团队异常: {analyst_error[0]}，切换到规则引擎")
                    self._rule_analyst = RuleEngine()
                    use_rule_analysts = True
                    analyst_insights = self._get_rule_insights(stock_data)

            # 2026-09-10 断链修复: 线程包装器成功路径 analyst_insights 恒为 [[...]]
            # 双层嵌套 (初值 [None] + 线程内 [0]=List) → get_summary 内 i=list →
            # 'list' has no attribute 'direction' 恒炸 → 决策链 LLM 分析师分支从未
            # 真正跑通过 (每次降级到 neutral 0.3 占位)。解包后 get_summary 才吃得到真洞察。
            if isinstance(analyst_insights, list) and len(analyst_insights) == 1 \
                    and isinstance(analyst_insights[0], list):
                analyst_insights = analyst_insights[0]

            consensus = self.analyst_team.get_summary(analyst_insights)
            elapsed = time.time() - start_time
            # 记录 LLM 耗时，供下次调用判断是否降级到规则引擎
            self._last_llm_elapsed = elapsed
            logger.info(f"[AgentCoordinator] 共识: {consensus['consensus']}, 置信度: {consensus['avg_confidence']} (耗时: {elapsed:.0f}s)")

            # 全局超时检查
            if elapsed > self.GLOBAL_TIMEOUT * 0.5:
                logger.warning(f"[AgentCoordinator] LLM 过慢 ({elapsed:.0f}s)，跳过辩论阶段")
                return self._fallback_decision(consensus, stock_data, elapsed)

            # 快速路径: 高度一致时跳过研究辩论
            fast_path = (
                consensus['avg_confidence'] > 0.8 and
                consensus.get('agreement_ratio', 0) > 0.75
            )

            if fast_path:
                logger.info("[AgentCoordinator] 快速路径: 分析师高度一致，跳过研究辩论")
                research_conclusion = {
                    'final_direction': consensus['consensus'],
                    'recommendation': 'execute',
                    'confidence': consensus['avg_confidence'],
                    'reasoning': '分析师团队高度一致',
                }
            else:
                # 阶段 2: 研究团队辩论 (2026-09-09 重设计: 串行 4 轮 → bull/bear+合成 3 调用, 60s 硬上限)
                # - counter_bull/counter_bear 是死计算: 结果从未被 synthesize 消费，纯烧 42s×2 时延 → 删除
                # - bull→bear→synthesize 串行跑在 daemon 线程，主线程 join(60s) — 超时诚实降级
                #   分析师共识，不再拖死尾部 4 阶段 (旧设计 4 轮辩论 168s 恒超时 → 恒降级)
                logger.info("[AgentCoordinator] 阶段 2: 研究团队辩论 (Bull→Bear→Synthesize)")
                debate_start = time.time()
                debate_out = [None]
                debate_error = [None]

                def _run_debate():
                    try:
                        bull = self.research_team.bull_researcher.research(consensus, analyst_insights)
                        bear = self.research_team.bear_researcher.research(consensus, analyst_insights)
                        debate_out[0] = self.research_team.research_manager.synthesize(bull, bear)
                    except Exception as e:
                        debate_error[0] = e

                debate_thread = threading.Thread(target=_run_debate, daemon=True)
                debate_thread.start()
                debate_thread.join(timeout=60.0)  # 3×~17s + 排队余量; 饱和时熔断器快速失败

                if debate_out[0]:
                    research_conclusion = debate_out[0]
                    logger.info(
                        f"[AgentCoordinator] 研究结论: {research_conclusion['final_direction']}, "
                        f"推荐: {research_conclusion['recommendation']} "
                        f"(辩论耗时: {time.time() - debate_start:.0f}s)"
                    )
                else:
                    if debate_error[0]:
                        logger.warning(f"[AgentCoordinator] 辩论链异常: {debate_error[0]}，使用分析师共识")
                    else:
                        logger.warning("[AgentCoordinator] 辩论超 60s/无结果，使用分析师共识")
                    research_conclusion = {
                        'final_direction': consensus['consensus'],
                        'recommendation': 'execute',
                        'confidence': consensus['avg_confidence'],
                        'reasoning': '辩论超时，使用分析师共识',
                    }

            # 阶段 3: 交易代理决策
            logger.info("[AgentCoordinator] 阶段 3: 交易代理决策")
            self._check_timeout(start_time)  # LLM 调用前检查
            trade_decision = self.trader_agent.make_decision(research_conclusion, stock_data)
            logger.info(
                f"[AgentCoordinator] 交易决策: {trade_decision.action}, "
                f"数量: {trade_decision.quantity}%"
            )

            # 阶段 4: 风险管理
            logger.info("[AgentCoordinator] 阶段 4: 风险管理")
            self._check_timeout(start_time)  # LLM 调用前检查
            risk_assessment = self.risk_manager.assess_risk(
                {"action": trade_decision.action, "quantity": trade_decision.quantity,
                 "price_target": trade_decision.price_target,
                 "stop_loss": trade_decision.stop_loss, "take_profit": trade_decision.take_profit},
                stock_data
            )
            logger.info(f"[AgentCoordinator] 风险等级: {risk_assessment.overall_risk}")

            # 阶段 5: 投资组合管理
            logger.info("[AgentCoordinator] 阶段 5: 投资组合管理")
            self._check_timeout(start_time)  # LLM 调用前检查
            portfolio_decision = self.portfolio_manager.make_decision(
                [{"stock_name": stock_data.get("name", ""), "action": trade_decision.action,
                  "quantity": trade_decision.quantity}],
                portfolio_state or {}
            )
            logger.info(f"[AgentCoordinator] 组合分配: {portfolio_decision.allocation}")

            # ── 跨模态推理 ──
            cross_modal_score = self._cross_modal_reasoning(stock_data, consensus, research_conclusion)

            # ── 综合评分 (含分歧度检测) ──
            direction_score = {
                "bullish": 0.7, "bearish": 0.3, "neutral": 0.5
            }
            overall_score = (
                consensus["avg_confidence"] * 0.2 +
                research_conclusion.get("confidence", 0.5) * 0.2 +
                trade_decision.confidence * 0.2 +
                (0.7 if risk_assessment.overall_risk == "low" else 0.5) * 0.2 +
                direction_score.get(research_conclusion.get("final_direction", "neutral"), 0.5) * 0.2
            )

            # 跨模态一致性调整
            overall_score = overall_score * 0.85 + cross_modal_score * 0.15

            # 分歧度检测: 当分析师与研究团队意见不一致时降低置信度
            disagreement = abs(
                direction_score.get(consensus['consensus'], 0.5) -
                direction_score.get(research_conclusion.get('final_direction', 'neutral'), 0.5)
            )
            if disagreement > 0.3:
                overall_score *= 0.8  # 降低 20%
                logger.warning(
                    f"[AgentCoordinator] 高分歧检测: 分析师={consensus['consensus']}, "
                    f"研究={research_conclusion.get('final_direction')}"
                )

            # 构建决策结果
            decision = AgentDecision(
                analyst_consensus=consensus["consensus"],
                analyst_confidence=consensus["avg_confidence"],
                analyst_insights=consensus["insights"],
                research_direction=research_conclusion.get("final_direction", "neutral"),
                research_recommendation=research_conclusion.get("recommendation", "hold"),
                research_confidence=research_conclusion.get("confidence", 0.5),
                trade_action=trade_decision.action,
                trade_quantity=trade_decision.quantity,
                trade_confidence=trade_decision.confidence,
                risk_level=risk_assessment.overall_risk,
                risk_adjustment=risk_assessment.risk_adjustment,
                portfolio_allocation=portfolio_decision.allocation,
                rebalance_needed=portfolio_decision.rebalance_needed,
                overall_score=round(overall_score, 3),
            )

            total_elapsed = time.time() - start_time
            logger.info(f"[AgentCoordinator] 决策完成! 综合评分: {overall_score:.3f} (总耗时: {total_elapsed:.0f}s)")
            return decision

        except Exception as e:
            logger.error(f"[AgentCoordinator] 决策流水线错误: {e}")
            import traceback
            traceback.print_exc()
            return AgentDecision(
                analyst_consensus="neutral", analyst_confidence=0.3,
                analyst_insights=[], research_direction="neutral",
                research_recommendation="hold", research_confidence=0.3,
                trade_action="hold", trade_quantity=0, trade_confidence=0.3,
                risk_level="high", risk_adjustment="保持",
                portfolio_allocation={}, rebalance_needed=False,
                overall_score=0.3
            )

    def _fallback_decision(self, consensus: Dict, stock_data: Dict, elapsed: float) -> AgentDecision:
        """超时降级: 使用分析师共识 + 规则引擎"""
        from modules.llm_agents.llm_client import RuleEngine
        logger.warning(f"[AgentCoordinator] 超时降级 (已用 {elapsed:.0f}s)")

        # 使用规则引擎获取交易决策
        rule_result = RuleEngine.decide(stock_data)

        return AgentDecision(
            analyst_consensus=consensus.get('consensus', 'neutral'),
            analyst_confidence=consensus.get('avg_confidence', 0.5),
            analyst_insights=consensus.get('insights', []),
            research_direction=rule_result.get('direction', 'neutral'),
            research_recommendation='execute' if rule_result.get('direction') != 'neutral' else 'hold',
            research_confidence=rule_result.get('confidence', 0.5),
            trade_action=rule_result.get('direction', 'neutral'),
            trade_quantity=50.0 if rule_result.get('direction') != 'neutral' else 0,
            trade_confidence=rule_result.get('confidence', 0.5),
            risk_level='medium',
            risk_adjustment='保持',
            portfolio_allocation={},
            rebalance_needed=False,
            overall_score=consensus.get('avg_confidence', 0.5) * 0.6 + rule_result.get('confidence', 0.5) * 0.4,
        )
    
    def _get_rule_insights(self, stock_data: Dict) -> List[AnalystInsight]:
        """
        规则引擎分析师 — 基于技术指标生成分析师洞察

        替代 LLM 调用，用于 LLM 超时保护模式
        """
        rsi = stock_data.get('rsi_14', stock_data.get('rsi', 50))
        macd = stock_data.get('macd', 0)
        macd_signal = stock_data.get('macd_signal', 0)
        ma5 = stock_data.get('ma_5', stock_data.get('ma5', 0))
        ma20 = stock_data.get('ma_20', stock_data.get('ma20', 0))
        volume_ratio = stock_data.get('volume_ratio', 1.0)
        change_pct = stock_data.get('change_pct', 0)

        # 综合评分
        score = 0.0
        factors = []

        # RSI
        if rsi < 30:
            score += 0.3; factors.append("RSI超卖")
        elif rsi > 70:
            score -= 0.3; factors.append("RSI超买")

        # MACD
        if macd > macd_signal:
            score += 0.2; factors.append("MACD金叉")
        elif macd < macd_signal:
            score -= 0.2; factors.append("MACD死叉")

        # 均线
        if ma5 > 0 and ma20 > 0:
            if ma5 > ma20 * 1.02:
                score += 0.15; factors.append("MA5>MA20")
            elif ma5 < ma20 * 0.98:
                score -= 0.15; factors.append("MA5<MA20")

        # 成交量
        if volume_ratio > 1.5:
            score += 0.05 * (1 if change_pct > 0 else -1)
            factors.append(f"放量(量比{volume_ratio:.1f})")

        if score > 0.2:
            direction = "bullish"
        elif score < -0.2:
            direction = "bearish"
        else:
            direction = "neutral"

        confidence = min(0.7, abs(score) + 0.3)

        # 生成不同类型的分析师洞察
        insights = [
            AnalystInsight("technical", direction, confidence,
                          f"技术指标综合评分 {score:.3f}, 信号: {', '.join(factors)}"),
            AnalystInsight("sentiment", direction, confidence * 0.9,
                          f"市场情绪与价格方向一致 ({direction})"),
            AnalystInsight("fundamental", "neutral", 0.5,
                          "基本面数据不足，保持中性"),
            AnalystInsight("news", "neutral", 0.5,
                          "无重大新闻事件"),
        ]
        return insights

    def _cross_modal_reasoning(self, stock_data: Dict,
                                consensus: Dict,
                                research: Dict) -> float:
        """
        跨模态推理 — 融合文本 + 数值 + 视觉模态

        模态:
        1. 文本: 分析师共识 + 研究结论 (LLM 输出)
        2. 数值: 技术指标 (RSI, MACD, 布林带等)
        3. 视觉: K线形态 (通过技术指标推断)

        一致性得分:
            - 所有模态方向一致 → 高分 (0.8~1.0)
            - 部分一致 → 中分 (0.4~0.7)
            - 方向冲突 → 低分 (0.0~0.3)
        """
        # 文本模态方向
        text_direction = consensus.get('consensus', 'neutral')
        text_direction_score = {'bullish': 1.0, 'bearish': 0.0, 'neutral': 0.5}.get(text_direction, 0.5)

        # 数值模态方向 (基于技术指标)
        rsi = stock_data.get('rsi_14', 50)
        macd = stock_data.get('macd', 0)
        macd_signal = stock_data.get('macd_signal', 0)

        numeric_score = 0.5
        if rsi < 30:
            numeric_score += 0.3
        elif rsi > 70:
            numeric_score -= 0.3

        if macd > macd_signal:
            numeric_score += 0.2
        elif macd < macd_signal:
            numeric_score -= 0.2

        # 视觉模态方向 (基于 K 线形态推断)
        visual_score = numeric_score  # 简化: 与数值模态一致
        ma5 = stock_data.get('ma_5', 0)
        ma20 = stock_data.get('ma_20', 0)
        if ma5 > 0 and ma20 > 0:
            if ma5 > ma20 * 1.02:
                visual_score += 0.1
            elif ma5 < ma20 * 0.98:
                visual_score -= 0.1

        # 一致性计算
        modal_scores = [text_direction_score, numeric_score, visual_score]
        avg_score = np.mean(modal_scores)
        consistency = 1.0 - np.std(modal_scores)  # 标准差越小，一致性越高

        # 综合得分
        cross_modal_score = avg_score * 0.6 + consistency * 0.4

        logger.info(
            f"[CrossModal] 文本={text_direction_score:.2f}, "
            f"数值={numeric_score:.2f}, 视觉={visual_score:.2f}, "
            f"一致性={consistency:.2f}, 综合={cross_modal_score:.2f}"
        )

        return float(cross_modal_score)

    def get_decision_json(self, decision: AgentDecision) -> Dict:
        """将决策转换为 JSON 格式"""
        return {
            "timestamp": decision.timestamp,
            "analyst_consensus": decision.analyst_consensus,
            "analyst_confidence": decision.analyst_confidence,
            "analyst_insights": decision.analyst_insights,
            "research_direction": decision.research_direction,
            "research_recommendation": decision.research_recommendation,
            "research_confidence": decision.research_confidence,
            "trade_action": decision.trade_action,
            "trade_quantity": decision.trade_quantity,
            "trade_confidence": decision.trade_confidence,
            "risk_level": decision.risk_level,
            "risk_adjustment": decision.risk_adjustment,
            "portfolio_allocation": decision.portfolio_allocation,
            "rebalance_needed": decision.rebalance_needed,
            "overall_score": decision.overall_score
        }
