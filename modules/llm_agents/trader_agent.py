#!/usr/bin/env python3
"""
Trader Agent - 交易代理

基于 TradingAgents 和 Trading-R1 的交易代理:
- 确定最优入场/出场点
- 基于辩论结果做出交易决策
- 支持多种交易模式

SOTA Reference:
- Trading-R1 (UCLA, UW, Stanford, ICLR 2026) - RL 驱动交易
- QuantAgent (SBU, 2025-09) - 价格驱动 HFT
"""

import json
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from modules.logger import logger
from modules.llm_agents.llm_client import LLMClient, LLMResponse


@dataclass
class TradeDecision:
    """交易决策"""
    action: str  # "buy", "sell", "hold", "reduce"
    quantity: float  # 数量或百分比
    price_target: float  # 目标价
    stop_loss: float  # 止损价
    take_profit: float  # 止盈价
    confidence: float  # 0-1
    reasoning: str
    timestamp: float = field(default_factory=time.time)


class TraderAgent:
    """交易代理 - 基于研究团队结论做出交易决策"""
    
    PROMPT_TEMPLATE = """你是专业交易代理。基于研究团队的综合结论，做出交易决策。

【研究团队结论】
最终方向: {final_direction}
推荐操作: {recommendation}
置信度: {confidence}
时间框架: {time_horizon}
风险等级: {risk_level}

【当前市场数据】
股票: {stock_name} ({stock_code})
当前价格: {current_price}
成本价: {cost_basis}
持仓盈亏: {profit_pct}%

【技术信号】
趋势: {trend}
支撑位: {support}
阻力位: {resistance}
RSI: {rsi}
MACD: {macd}

【交易决策要求】
1. 确定交易动作（买入/卖出/持有/减仓）
2. 计算最优仓位大小（基于 Kelly 公式和风险等级）
3. 设置止损止盈
4. 给出执行建议

请以 JSON 格式输出:
{{
    "action": "buy/sell/hold/reduce",
    "quantity_pct": 0-100,
    "price_target": "目标价",
    "stop_loss": "止损价",
    "take_profit": "止盈价",
    "confidence": 0.0-1.0,
    "reasoning": "交易决策分析",
    "execution_timing": "immediate/gradual/partial",
    "position_size": "small/medium/large"
}}"""
    
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
    
    def make_decision(self, research_conclusion: Dict, stock_data: Dict) -> TradeDecision:
        """做出交易决策"""
        try:
            prompt = self.PROMPT_TEMPLATE.format(
                final_direction=research_conclusion.get("final_direction", "neutral"),
                recommendation=research_conclusion.get("recommendation", "hold"),
                confidence=research_conclusion.get("confidence", 0.5),
                time_horizon=research_conclusion.get("time_horizon", "medium"),
                risk_level=research_conclusion.get("risk_level", "medium"),
                stock_name=stock_data.get("name", ""),
                stock_code=stock_data.get("code", ""),
                current_price=stock_data.get("current_price", "--"),
                cost_basis=stock_data.get("cost_basis", "--"),
                profit_pct=stock_data.get("profit_pct", "--"),
                trend=stock_data.get("trend", "neutral"),
                support=stock_data.get("support", "--"),
                resistance=stock_data.get("resistance", "--"),
                rsi=stock_data.get("rsi", "--"),
                macd=stock_data.get("macd", "--")
            )
            
            response = self.llm.get_response([{"role": "user", "content": prompt}])
            
            try:
                data = json.loads(response.content)
            except:
                data = {"action": "hold", "quantity_pct": 0, "price_target": "--",
                       "stop_loss": "--", "take_profit": "--", "confidence": 0.5,
                       "reasoning": "LLM 交易决策", "execution_timing": "gradual",
                       "position_size": "medium"}
            
            return TradeDecision(
                action=data.get("action", "hold"),
                quantity=data.get("quantity_pct", 0),
                price_target=float(data.get("price_target", 0)) if data.get("price_target") != "--" else 0,
                stop_loss=float(data.get("stop_loss", 0)) if data.get("stop_loss") != "--" else 0,
                take_profit=float(data.get("take_profit", 0)) if data.get("take_profit") != "--" else 0,
                confidence=data.get("confidence", 0.5),
                reasoning=data.get("reasoning", response.content)
            )
        except Exception as e:
            logger.error(f"TraderAgent error: {e}")
            return TradeDecision("hold", 0, 0, 0, 0, 0.3, f"决策出错: {e}")
