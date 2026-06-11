#!/usr/bin/env python3
"""
Portfolio Manager - 投资组合管理代理

基于 TradingAgents 的投资组合管理:
- 监督投资组合分配和平衡
- 多股票协同决策
- 组合风险优化

SOTA Reference:
- AlphaAgents (25-08): Multi-Agents for Equity Portfolio Constructions
- AlphaCrafter (26-05): Full-Stack Multi-Agent Framework
"""

import json
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from modules.logger import logger
from modules.llm_agents.llm_client import LLMClient


@dataclass
class PortfolioDecision:
    """投资组合决策"""
    allocation: Dict[str, float]  # 股票 → 权重
    total_exposure: float
    cash_reserve: float
    rebalance_needed: bool
    reasoning: str
    timestamp: float = field(default_factory=time.time)


class PortfolioManager:
    """投资组合管理代理 - 监督投资组合分配和平衡"""
    
    PROMPT_TEMPLATE = """你是投资组合管理代理。基于所有代理的决策，制定最终投资组合方案。

【当前投资组合】
总仓位: {total_position}%
现金储备: {cash_reserve}%
股票数量: {stock_count}

【各股票决策摘要】
{stock_decisions}

【组合约束】
最大单只股票: {max_single_stock}%
最小现金储备: {min_cash}%
行业集中度限制: {industry_limit}%
最大总敞口: {max_exposure}%

【投资组合决策要求】
1. 确定每只股票的最终权重
2. 评估是否需要再平衡
3. 计算最优现金储备
4. 给出组合风险调整建议

请以 JSON 格式输出:
{{
    "allocation": {{"股票1": 权重1, "股票2": 权重2}},
    "total_exposure": 0-100,
    "cash_reserve": 0-100,
    "rebalance_needed": true/false,
    "risk_adjustment": "增加/减少/保持",
    "reasoning": "投资组合决策分析"
}}"""
    
    def __init__(self, llm_client: LLMClient, config: Dict = None):
        self.llm = llm_client
        self.config = config or {}
    
    def make_decision(self, stock_decisions: List[Dict], portfolio_state: Dict) -> PortfolioDecision:
        """做出投资组合决策"""
        try:
            decisions_text = "\n".join([
                f"- {s.get('stock_name', '')}: {s.get('action', 'hold')} ({s.get('quantity', 0)}%)"
                for s in stock_decisions
            ])
            
            prompt = self.PROMPT_TEMPLATE.format(
                total_position=portfolio_state.get("total_position", 0),
                cash_reserve=portfolio_state.get("cash_reserve", 0),
                stock_count=portfolio_state.get("stock_count", 0),
                stock_decisions=decisions_text,
                max_single_stock=self.config.get("max_single_stock", 30),
                min_cash=self.config.get("min_cash", 10),
                industry_limit=self.config.get("industry_limit", 40),
                max_exposure=self.config.get("max_exposure", 100)
            )
            
            response = self.llm.get_response([{"role": "user", "content": prompt}])
            
            try:
                data = json.loads(response.content)
            except:
                data = {"allocation": {}, "total_exposure": 0, "cash_reserve": 0,
                       "rebalance_needed": False, "risk_adjustment": "保持",
                       "reasoning": "投资组合决策"}
            
            return PortfolioDecision(
                allocation=data.get("allocation", {}),
                total_exposure=data.get("total_exposure", 0),
                cash_reserve=data.get("cash_reserve", 0),
                rebalance_needed=data.get("rebalance_needed", False),
                reasoning=data.get("reasoning", response.content)
            )
        except Exception as e:
            logger.error(f"PortfolioManager error: {e}")
            return PortfolioDecision({}, 0, 0, False, f"决策出错: {e}")
