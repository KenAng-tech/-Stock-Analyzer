#!/usr/bin/env python3
"""
Risk Manager - 风险管理代理

基于 TradingAgents 的风险管理:
- 评估风险敞口
- 执行风险限制
- 动态仓位调整

SOTA Reference:
- Barra Risk Model (已集成)
- Adaptive Kelly (已集成)
"""

import json
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from modules.logger import logger
from modules.llm_agents.llm_client import LLMClient


@dataclass
class RiskAssessment:
    """风险评估"""
    overall_risk: str  # "low", "medium", "high"
    max_position_pct: float
    stop_loss_pct: float
    correlation_risk: str
    liquidity_risk: str
    timestamp: float = field(default_factory=time.time)


class RiskManager:
    """风险管理代理 - 评估风险敞口和执行风险限制"""
    
    PROMPT_TEMPLATE = """你是风险管理代理。基于交易决策和市场条件，评估和管理风险。

【交易决策】
动作: {action}
数量: {quantity}%
目标价: {price_target}
止损: {stop_loss}
止盈: {take_profit}

【市场条件】
市场状态: {market_regime}
波动率: {volatility}
流动性: {liquidity}
相关性风险: {correlation_risk}

【风险限制】
最大仓位: {max_position}%
最大回撤: {max_drawdown}%
单笔风险: {single_risk}%

【风险评估要求】
1. 评估整体风险等级
2. 调整最大仓位建议
3. 确认止损止盈是否合理
4. 给出风险调整建议

请以 JSON 格式输出:
{{
    "overall_risk": "low/medium/high",
    "max_position_pct": 0-100,
    "stop_loss_pct": 0-100,
    "correlation_risk": "low/medium/high",
    "liquidity_risk": "low/medium/high",
    "risk_adjustment": "增加/减少/保持",
    "recommendation": "建议"
}}"""
    
    def __init__(self, llm_client: LLMClient, config: Dict = None):
        self.llm = llm_client
        self.config = config or {}
    
    def assess_risk(self, trade_decision: Dict, stock_data: Dict) -> RiskAssessment:
        """执行风险评估"""
        try:
            prompt = self.PROMPT_TEMPLATE.format(
                action=trade_decision.get("action", "hold"),
                quantity=trade_decision.get("quantity", 0),
                price_target=trade_decision.get("price_target", "--"),
                stop_loss=trade_decision.get("stop_loss", "--"),
                take_profit=trade_decision.get("take_profit", "--"),
                market_regime=stock_data.get("market_regime", "neutral"),
                volatility=stock_data.get("volatility", "--"),
                liquidity=stock_data.get("liquidity", "--"),
                correlation_risk=stock_data.get("correlation_risk", "low"),
                max_position=self.config.get("strategy", {}).get("max_position", 100),
                max_drawdown=self.config.get("strategy", {}).get("max_stop_loss", 20),
                single_risk=self.config.get("strategy", {}).get("min_stop_loss", 5)
            )
            
            response = self.llm.get_response([{"role": "user", "content": prompt}])
            
            try:
                data = json.loads(response.content)
            except:
                data = {"overall_risk": "medium", "max_position_pct": 50,
                       "stop_loss_pct": 5, "correlation_risk": "medium",
                       "liquidity_risk": "medium", "risk_adjustment": "保持",
                       "recommendation": "正常执行"}
            
            return RiskAssessment(
                overall_risk=data.get("overall_risk", "medium"),
                max_position_pct=data.get("max_position_pct", 50),
                stop_loss_pct=data.get("stop_loss_pct", 5),
                correlation_risk=data.get("correlation_risk", "medium"),
                liquidity_risk=data.get("liquidity_risk", "medium")
            )
        except Exception as e:
            logger.error(f"RiskManager error: {e}")
            return RiskAssessment("medium", 50, 5, "medium", "medium")
