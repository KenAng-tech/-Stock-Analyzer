#!/usr/bin/env python3
"""
Agent Coordinator - Agent 协调器

整合所有 Agent 形成完整的决策流水线:

Market Data ──→ Analyst Team ──→ Research Team ──→ Trader ──→ Risk Manager ──→ Portfolio Manager

SOTA Reference:
- TradingAgents v0.2.5 (Tauric Research, 2026-05)
- AlphaCrafter (NJU, 2026-05)
- QuantAgent (SBU, 2025-09)
"""

import json
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from modules.logger import logger
from modules.llm_agents.llm_client import LLMClient
from modules.llm_agents.analyst_team import AnalystTeam
from modules.llm_agents.research_team import ResearchTeam
from modules.llm_agents.trader_agent import TraderAgent
from modules.llm_agents.risk_manager import RiskManager
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
    
    def __init__(self, llm_client: LLMClient, config: Dict = None):
        self.llm = llm_client
        self.config = config or {}
        
        # 初始化所有 Agent
        self.analyst_team = AnalystTeam(llm_client)
        self.research_team = ResearchTeam(llm_client)
        self.trader_agent = TraderAgent(llm_client)
        self.risk_manager = RiskManager(llm_client, config)
        self.portfolio_manager = PortfolioManager(llm_client, config)
    
    def make_decision(self, stock_data: Dict, portfolio_state: Dict = None) -> AgentDecision:
        """执行完整的 Agent 决策流水线"""
        logger.info("[AgentCoordinator] 开始决策流水线")
        
        try:
            # 阶段 1: 分析师团队并行分析
            logger.info("[AgentCoordinator] 阶段 1: 分析师团队分析")
            analyst_insights = self.analyst_team.analyze(stock_data)
            consensus = self.analyst_team.get_summary(analyst_insights)
            logger.info(f"[AgentCoordinator] 共识: {consensus['consensus']}, 置信度: {consensus['avg_confidence']}")
            
            # 阶段 2: 研究团队辩论
            logger.info("[AgentCoordinator] 阶段 2: 研究团队辩论")
            bull_debate = self.research_team.bull_researcher.research(consensus, analyst_insights)
            bear_debate = self.research_team.bear_researcher.research(consensus, analyst_insights)
            research_conclusion = self.research_team.research_manager.synthesize(bull_debate, bear_debate)
            logger.info(f"[AgentCoordinator] 研究结论: {research_conclusion['final_direction']}, 推荐: {research_conclusion['recommendation']}")
            
            # 阶段 3: 交易代理决策
            logger.info("[AgentCoordinator] 阶段 3: 交易代理决策")
            trade_decision = self.trader_agent.make_decision(research_conclusion, stock_data)
            logger.info(f"[AgentCoordinator] 交易决策: {trade_decision.action}, 数量: {trade_decision.quantity}%")
            
            # 阶段 4: 风险管理
            logger.info("[AgentCoordinator] 阶段 4: 风险管理")
            risk_assessment = self.risk_manager.assess_risk(
                {"action": trade_decision.action, "quantity": trade_decision.quantity,
                 "price_target": trade_decision.price_target,
                 "stop_loss": trade_decision.stop_loss, "take_profit": trade_decision.take_profit},
                stock_data
            )
            logger.info(f"[AgentCoordinator] 风险等级: {risk_assessment.overall_risk}")
            
            # 阶段 5: 投资组合管理
            logger.info("[AgentCoordinator] 阶段 5: 投资组合管理")
            portfolio_decision = self.portfolio_manager.make_decision(
                [{"stock_name": stock_data.get("name", ""), "action": trade_decision.action,
                  "quantity": trade_decision.quantity}],
                portfolio_state or {}
            )
            logger.info(f"[AgentCoordinator] 组合分配: {portfolio_decision.allocation}")
            
            # 综合评分计算
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
                overall_score=round(overall_score, 3)
            )
            
            logger.info(f"[AgentCoordinator] 决策完成! 综合评分: {overall_score:.3f}")
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
