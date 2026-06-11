#!/usr/bin/env python3
"""
Research Team - 研究团队

基于 TradingAgents 框架的研究团队:
1. BullResearcher - 多头研究员（构建看涨论点）
2. BearResearcher - 空头研究员（构建看跌论点）
3. ResearchManager - 研究经理（综合辩论结果）

SOTA Reference:
- TradingAgents v0.2.5: Bull/Bear 辩论机制
- AlphaCrafter: 市场状态自适应
"""

import json
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from modules.logger import logger
from modules.llm_agents.llm_client import LLMClient, LLMResponse


@dataclass
class ResearchDebate:
    """研究辩论结果"""
    researcher_type: str  # "bull" or "bear"
    position: str  # "strong_bull", "moderate_bull", "cautious_bull", etc.
    confidence: float  # 0-1
    arguments: List[str]
    counter_arguments: List[str]
    key_catalysts: List[str]
    timestamp: float = field(default_factory=time.time)


class BullResearcher:
    """多头研究员 - 构建看涨论点"""
    
    PROMPT_TEMPLATE = """你是专业多头研究员。基于分析师团队的洞察，构建看涨论点。

【分析师团队共识】
{consensus}
平均置信度: {avg_confidence}

【各分析师观点】
{analyst_views}

【辩论要求】
1. 列出3-5个核心看涨论点
2. 预测潜在催化剂
3. 评估上行空间
4. 给出多头立场强度（strong/moderate/cautious）

请以 JSON 格式输出:
{{
    "position": "strong_bull/moderate_bull/cautious_bull",
    "confidence": 0.0-1.0,
    "arguments": ["论点1", "论点2", "论点3"],
    "catalysts": ["催化剂1", "催化剂2"],
    "upside_potential": "描述",
    "target_price": "目标价"
}}"""
    
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
    
    def research(self, consensus: Dict, analyst_insights: List) -> ResearchDebate:
        """执行多头研究"""
        try:
            analyst_views = "\n".join([
                f"- {i['type']}: {i['direction']} (置信度: {i['confidence']})"
                for i in consensus.get("insights", [])
            ])
            
            prompt = self.PROMPT_TEMPLATE.format(
                consensus=consensus.get("consensus", "neutral"),
                avg_confidence=consensus.get("avg_confidence", 0),
                analyst_views=analyst_views
            )
            
            response = self.llm.get_response([{"role": "user", "content": prompt}])
            
            try:
                data = json.loads(response.content)
            except:
                data = {"position": "moderate_bull", "confidence": 0.5, 
                       "arguments": ["LLM 分析"], "catalysts": []}
            
            return ResearchDebate(
                researcher_type="bull",
                position=data.get("position", "moderate_bull"),
                confidence=data.get("confidence", 0.5),
                arguments=data.get("arguments", []),
                counter_arguments=[],
                key_catalysts=data.get("catalysts", [])
            )
        except Exception as e:
            logger.error(f"BullResearcher error: {e}")
            return ResearchDebate("bull", "moderate_bull", 0.4, ["分析出错"], [], [])


class BearResearcher:
    """空头研究员 - 构建看跌论点"""
    
    PROMPT_TEMPLATE = """你是专业空头研究员。基于分析师团队的洞察，构建看跌论点。

【分析师团队共识】
{consensus}
平均置信度: {avg_confidence}

【各分析师观点】
{analyst_views}

【辩论要求】
1. 列出3-5个核心看跌论点
2. 识别潜在风险因素
3. 评估下行风险
4. 给出空头立场强度（strong_bear/moderate_bear/cautious_bear）

请以 JSON 格式输出:
{{
    "position": "strong_bear/moderate_bear/cautious_bear",
    "confidence": 0.0-1.0,
    "arguments": ["论点1", "论点2", "论点3"],
    "risks": ["风险1", "风险2"],
    "downside_risk": "描述",
    "support_level": "支撑位"
}}"""
    
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
    
    def research(self, consensus: Dict, analyst_insights: List) -> ResearchDebate:
        """执行空头研究"""
        try:
            analyst_views = "\n".join([
                f"- {i['type']}: {i['direction']} (置信度: {i['confidence']})"
                for i in consensus.get("insights", [])
            ])
            
            prompt = self.PROMPT_TEMPLATE.format(
                consensus=consensus.get("consensus", "neutral"),
                avg_confidence=consensus.get("avg_confidence", 0),
                analyst_views=analyst_views
            )
            
            response = self.llm.get_response([{"role": "user", "content": prompt}])
            
            try:
                data = json.loads(response.content)
            except:
                data = {"position": "moderate_bear", "confidence": 0.5,
                       "arguments": ["LLM 分析"], "risks": []}
            
            return ResearchDebate(
                researcher_type="bear",
                position=data.get("position", "moderate_bear"),
                confidence=data.get("confidence", 0.5),
                arguments=data.get("arguments", []),
                counter_arguments=[],
                key_catalysts=data.get("risks", [])
            )
        except Exception as e:
            logger.error(f"BearResearcher error: {e}")
            return ResearchDebate("bear", "moderate_bear", 0.4, ["分析出错"], [], [])


class ResearchManager:
    """研究经理 - 综合辩论结果"""
    
    PROMPT_TEMPLATE = """你是研究经理。综合多头和空头研究员的辩论，给出最终结论。

【多头研究】
立场: {bull_position}
置信度: {bull_confidence}
论点: {bull_arguments}
催化剂: {bull_catalysts}

【空头研究】
立场: {bear_position}
置信度: {bear_confidence}
论点: {bear_arguments}
风险: {bear_risks}

【综合评估要求】
1. 评估双方论点的强弱
2. 考虑市场状态（牛市/熊市/震荡）
3. 给出加权后的最终方向
4. 提供置信度区间

请以 JSON 格式输出:
{{
    "final_direction": "bullish/bearish/neutral",
    "confidence": 0.0-1.0,
    "bull_weight": 0.0-1.0,
    "bear_weight": 0.0-1.0,
    "reasoning": "综合辩论分析",
    "recommendation": "buy/hold/sell",
    "time_horizon": "short/medium/long",
    "risk_level": "low/medium/high"
}}"""
    
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
    
    def synthesize(self, bull_debate: ResearchDebate, bear_debate: ResearchDebate) -> Dict:
        """综合辩论结果"""
        try:
            prompt = self.PROMPT_TEMPLATE.format(
                bull_position=bull_debate.position,
                bull_confidence=bull_debate.confidence,
                bull_arguments=", ".join(bull_debate.arguments),
                bull_catalysts=", ".join(bull_debate.key_catalysts),
                bear_position=bear_debate.position,
                bear_confidence=bear_debate.confidence,
                bear_arguments=", ".join(bear_debate.arguments),
                bear_risks=", ".join(bear_debate.key_catalysts)
            )
            
            response = self.llm.get_response([{"role": "user", "content": prompt}])
            
            try:
                data = json.loads(response.content)
            except:
                data = {"final_direction": "neutral", "confidence": 0.5,
                       "bull_weight": 0.5, "bear_weight": 0.5,
                       "reasoning": "综合评估", "recommendation": "hold",
                       "time_horizon": "medium", "risk_level": "medium"}
            
            return data
        except Exception as e:
            logger.error(f"ResearchManager error: {e}")
            return {"final_direction": "neutral", "confidence": 0.3,
                   "bull_weight": 0.5, "bear_weight": 0.5,
                   "reasoning": f"综合评估出错: {e}", "recommendation": "hold",
                   "time_horizon": "medium", "risk_level": "medium"}


class ResearchTeam:
    """研究团队 - 整合多头和空头研究员"""
    
    def __init__(self, llm_client: LLMClient):
        self.bull_researcher = BullResearcher(llm_client)
        self.bear_researcher = BearResearcher(llm_client)
        self.research_manager = ResearchManager(llm_client)
    
    def debate(self, consensus: Dict, analyst_insights: List) -> Dict:
        """执行辩论"""
        bull_debate = self.bull_researcher.research(consensus, analyst_insights)
        bear_debate = self.bear_researcher.research(consensus, analyst_insights)
        return {"bull": bull_debate, "bear": bear_debate}
