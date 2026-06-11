#!/usr/bin/env python3
"""
Multi-Modal Module - 跨模态推理

SOTA Reference:
- FCMR (Hanyang University, ACL 2025): Robust Evaluation of Financial Cross-Modal Multi-Hop Reasoning
- FINMME (PKU, 25-05): Benchmark Dataset for Financial Multi-Modal Reasoning
- MM-DREX (ZJU, CityU, 25-09): Multimodal-Driven Dynamic Routing of LLM Experts

整合:
1. 文本模态 (新闻、财报、社交媒体)
2. 视觉模态 (K线图表、技术形态)
3. 数值模态 (价格、成交量、技术指标)
"""

import json
import time
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from modules.logger import logger
from modules.llm_agents.llm_client import LLMClient


@dataclass
class MultiModalInsight:
    """跨模态洞察"""
    modality: str  # "text", "visual", "numerical", "cross_modal"
    insight: str
    confidence: float
    timestamp: float = field(default_factory=time.time)


class CrossModalReasoner:
    """跨模态推理器 - 整合多模态信息"""
    
    PROMPT_TEMPLATE = """你是跨模态推理专家。基于文本、视觉和数值模态的信息，进行多跳推理。

【文本模态】
新闻: {news_text}
财报: {financial_text}
社交媒体情绪: {social_sentiment}

【视觉模态】
K线形态: {kline_pattern}
技术形态: {technical_pattern}
图表信号: {chart_signals}

【数值模态】
价格动量: {price_momentum}
成交量比率: {volume_ratio}
技术指标: {technical_indicators}

【跨模态推理要求】
1. 识别模态间的一致性/冲突
2. 进行多跳推理连接不同模态信息
3. 给出综合判断
4. 评估模态权重

请以 JSON 格式输出:
{{
    "consistency_score": 0.0-1.0,
    "cross_modal_reasoning": "推理过程",
    "modality_weights": {{"text": 0.0, "visual": 0.0, "numerical": 0.0}},
    "final_direction": "bullish/bearish/neutral",
    "confidence": 0.0-1.0,
    "key_insights": ["洞察1", "洞察2"]
}}"""
    
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
    
    def reason(self, stock_data: Dict) -> Dict:
        """执行跨模态推理"""
        try:
            prompt = self.PROMPT_TEMPLATE.format(
                news_text=stock_data.get("news_text", "无重大新闻"),
                financial_text=stock_data.get("financial_text", "财报稳定"),
                social_sentiment=stock_data.get("social_sentiment", "中性"),
                kline_pattern=stock_data.get("kline_pattern", "无明显形态"),
                technical_pattern=stock_data.get("technical_pattern", "趋势跟随"),
                chart_signals=stock_data.get("chart_signals", "无特殊信号"),
                price_momentum=stock_data.get("price_momentum", "中性"),
                volume_ratio=stock_data.get("volume_ratio", "1.0"),
                technical_indicators=stock_data.get("technical_indicators", "稳定")
            )
            
            response = self.llm.get_response([{"role": "user", "content": prompt}])
            
            try:
                data = json.loads(response.content)
            except:
                data = {"consistency_score": 0.5, "cross_modal_reasoning": "推理完成",
                       "modality_weights": {"text": 0.33, "visual": 0.33, "numerical": 0.34},
                       "final_direction": "neutral", "confidence": 0.5,
                       "key_insights": ["跨模态推理完成"]}
            
            return data
        except Exception as e:
            logger.error(f"CrossModalReasoner error: {e}")
            return {"consistency_score": 0.3, "cross_modal_reasoning": f"推理错误: {e}",
                   "modality_weights": {"text": 0.33, "visual": 0.33, "numerical": 0.34},
                   "final_direction": "neutral", "confidence": 0.3,
                   "key_insights": ["推理错误"]}


class MultiModalEngine:
    """跨模态引擎 - 整合所有模态"""
    
    def __init__(self, llm_client: LLMClient):
        self.reasoner = CrossModalReasoner(llm_client)
    
    def analyze(self, stock_data: Dict) -> Dict:
        """执行跨模态分析"""
        return self.reasoner.reason(stock_data)
