#!/usr/bin/env python3
"""
Analyst Team - 分析师团队

基于 TradingAgents 框架的四大分析师:
1. FundAnalyst - 基本面分析师
2. SentimentAnalyst - 情绪分析师  
3. TechnicalAnalyst - 技术分析师
4. NewsAnalyst - 新闻分析师

SOTA Reference:
- TradingAgents v0.2.5 (Tauric Research, 2026-05)
- AlphaCrafter (NJU, 2026-05) - 连续自适应因子发现
"""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from modules.logger import logger
from modules.llm_agents.llm_client import LLMClient, LLMResponse


@dataclass
class AnalystInsight:
    """分析师洞察"""
    analyst_type: str
    direction: str  # "bullish", "bearish", "neutral"
    confidence: float  # 0-1
    reasoning: str
    key_factors: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


class FundAnalyst:
    """基本面分析师 - 评估财务指标和内在价值"""
    
    PROMPT_TEMPLATE = """你是专业的股票基本面分析师。基于以下数据，分析股票的基本面状况。

股票: {stock_name} ({stock_code})
行业: {industry}

【基本面数据】
- 市盈率 (PE): {pe_ratio}
- 总市值: {market_cap}
- 流通市值: {circulating_cap}
- 营收增长: {revenue_growth}%
- 利润增长: {profit_growth}%
- 毛利率: {gross_margin}%

【分析要求】
1. 评估估值水平（高估/低估/合理）
2. 分析成长性（营收和利润趋势）
3. 判断盈利能力
4. 给出综合评分（0-100）

请以 JSON 格式输出:
{{
    "direction": "bullish/bearish/neutral",
    "confidence": 0.0-1.0,
    "reasoning": "详细分析",
    "key_factors": ["因素1", "因素2", "因素3"],
    "valuation_score": 0-100,
    "growth_score": 0-100,
    "profitability_score": 0-100,
    "composite_score": 0-100
}}"""
    
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
    
    def analyze(self, stock_data: Dict) -> AnalystInsight:
        """执行基本面分析"""
        try:
            prompt = self.PROMPT_TEMPLATE.format(
                stock_name=stock_data.get("name", ""),
                stock_code=stock_data.get("code", ""),
                industry=stock_data.get("industry", ""),
                pe_ratio=stock_data.get("pe_ratio", "--"),
                market_cap=stock_data.get("market_cap", "--"),
                circulating_cap=stock_data.get("circulating_cap", "--"),
                revenue_growth=stock_data.get("revenue_growth", "--"),
                profit_growth=stock_data.get("profit_growth", "--"),
                gross_margin=stock_data.get("gross_margin", "--")
            )
            
            response = self.llm.get_response([{"role": "user", "content": prompt}])
            
            # 解析 JSON 响应
            try:
                data = json.loads(response.content)
            except:
                data = {"direction": "neutral", "confidence": 0.5, "reasoning": response.content}
            
            return AnalystInsight(
                analyst_type="fundamental",
                direction=data.get("direction", "neutral"),
                confidence=data.get("confidence", 0.5),
                reasoning=data.get("reasoning", response.content),
                key_factors=data.get("key_factors", [])
            )
        except Exception as e:
            logger.error(f"FundAnalyst error: {e}")
            return AnalystInsight("fundamental", "neutral", 0.3, f"Analysis error: {e}")


class SentimentAnalyst:
    """情绪分析师 - 聚合新闻和社交媒体情绪"""
    
    PROMPT_TEMPLATE = """你是专业的情绪分析师。基于市场情绪指标，分析短期市场情绪。

股票: {stock_name} ({stock_code})

【市场情绪数据】
- 买/卖比率: {buy_sell_ratio}
- 外盘/内盘: {outer_inner_ratio}
- 资金流向: {fund_flow}
- 换手率: {turnover}%
- 振幅: {amplitude}%

【情绪分析要求】
1. 当前市场情绪（乐观/悲观/中性）
2. 主力资金动向
3. 散户情绪指标
4. 短期情绪趋势预测

请以 JSON 格式输出:
{{
    "direction": "bullish/bearish/neutral",
    "confidence": 0.0-1.0,
    "reasoning": "情绪分析",
    "key_factors": ["因素1", "因素2"],
    "sentiment_score": 0-100,
    "momentum": "increasing/decreasing/stable"
}}"""
    
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
    
    def analyze(self, stock_data: Dict) -> AnalystInsight:
        """执行情绪分析"""
        try:
            prompt = self.PROMPT_TEMPLATE.format(
                stock_name=stock_data.get("name", ""),
                stock_code=stock_data.get("code", ""),
                buy_sell_ratio=stock_data.get("buy_sell_ratio", "--"),
                outer_inner_ratio=stock_data.get("outer_inner_ratio", "--"),
                fund_flow=stock_data.get("fund_flow", "--"),
                turnover=stock_data.get("turnover", "--"),
                amplitude=stock_data.get("amplitude", "--")
            )
            
            response = self.llm.get_response([{"role": "user", "content": prompt}])
            
            try:
                data = json.loads(response.content)
            except:
                data = {"direction": "neutral", "confidence": 0.5, "reasoning": response.content}
            
            return AnalystInsight(
                analyst_type="sentiment",
                direction=data.get("direction", "neutral"),
                confidence=data.get("confidence", 0.5),
                reasoning=data.get("reasoning", response.content),
                key_factors=data.get("key_factors", [])
            )
        except Exception as e:
            logger.error(f"SentimentAnalyst error: {e}")
            return AnalystInsight("sentiment", "neutral", 0.3, f"Analysis error: {e}")


class TechnicalAnalyst:
    """技术分析师 - 使用技术指标检测交易模式"""
    
    PROMPT_TEMPLATE = """你是专业的技术分析师。基于技术指标分析价格走势。

股票: {stock_name} ({stock_code})

【技术指标】
- MA5: {ma5} | MA10: {ma10} | MA20: {ma20} | MA60: {ma60}
- RSI: {rsi}
- MACD: {macd}
- 布林带上轨: {boll_upper} | 下轨: {boll_lower}
- ATR: {atr}
- 成交量: {volume} | 均量: {avg_volume}

【技术分析要求】
1. 趋势判断（上升/下降/横盘）
2. 关键支撑位和阻力位
3. 买卖信号强度
4. 技术形态识别

请以 JSON 格式输出:
{{
    "direction": "bullish/bearish/neutral",
    "confidence": 0.0-1.0,
    "reasoning": "技术分析",
    "key_factors": ["因素1", "因素2"],
    "trend": "uptrend/downtrend/sideways",
    "support_level": "支撑位",
    "resistance_level": "阻力位",
    "signal_strength": "strong/medium/weak"
}}"""
    
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
    
    def analyze(self, stock_data: Dict) -> AnalystInsight:
        """执行技术分析"""
        try:
            prompt = self.PROMPT_TEMPLATE.format(
                stock_name=stock_data.get("name", ""),
                stock_code=stock_data.get("code", ""),
                ma5=stock_data.get("ma5", "--"),
                ma10=stock_data.get("ma10", "--"),
                ma20=stock_data.get("ma20", "--"),
                ma60=stock_data.get("ma60", "--"),
                rsi=stock_data.get("rsi", "--"),
                macd=stock_data.get("macd", "--"),
                boll_upper=stock_data.get("boll_upper", "--"),
                boll_lower=stock_data.get("boll_lower", "--"),
                atr=stock_data.get("atr", "--"),
                volume=stock_data.get("volume", "--"),
                avg_volume=stock_data.get("avg_volume", "--")
            )
            
            response = self.llm.get_response([{"role": "user", "content": prompt}])
            
            try:
                data = json.loads(response.content)
            except:
                data = {"direction": "neutral", "confidence": 0.5, "reasoning": response.content}
            
            return AnalystInsight(
                analyst_type="technical",
                direction=data.get("direction", "neutral"),
                confidence=data.get("confidence", 0.5),
                reasoning=data.get("reasoning", response.content),
                key_factors=data.get("key_factors", [])
            )
        except Exception as e:
            logger.error(f"TechnicalAnalyst error: {e}")
            return AnalystInsight("technical", "neutral", 0.3, f"Analysis error: {e}")


class NewsAnalyst:
    """新闻分析师 - 监控全球新闻和宏观经济指标"""
    
    PROMPT_TEMPLATE = """你是专业的新闻分析师。基于新闻和宏观环境影响分析。

股票: {stock_name} ({stock_code})
行业: {industry}

【新闻事件】
{news_events}

【宏观经济】
{macro_indicators}

【行业影响】
{industry_impact}

【分析要求】
1. 新闻对股价的正面/负面影响
2. 宏观经济环境评估
3. 行业政策变化影响
4. 潜在催化剂和风险

请以 JSON 格式输出:
{{
    "direction": "bullish/bearish/neutral",
    "confidence": 0.0-1.0,
    "reasoning": "新闻分析",
    "key_factors": ["因素1", "因素2"],
    "catalysts": ["正面催化剂"],
    "risks": ["风险因素"]
}}"""
    
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
    
    def analyze(self, stock_data: Dict) -> AnalystInsight:
        """执行新闻分析"""
        try:
            prompt = self.PROMPT_TEMPLATE.format(
                stock_name=stock_data.get("name", ""),
                stock_code=stock_data.get("code", ""),
                industry=stock_data.get("industry", ""),
                news_events=stock_data.get("news_events", "暂无重大新闻"),
                macro_indicators=stock_data.get("macro_indicators", "市场稳定"),
                industry_impact=stock_data.get("industry_impact", "无重大变化")
            )
            
            response = self.llm.get_response([{"role": "user", "content": prompt}])
            
            try:
                data = json.loads(response.content)
            except:
                data = {"direction": "neutral", "confidence": 0.5, "reasoning": response.content}
            
            return AnalystInsight(
                analyst_type="news",
                direction=data.get("direction", "neutral"),
                confidence=data.get("confidence", 0.5),
                reasoning=data.get("reasoning", response.content),
                key_factors=data.get("key_factors", [])
            )
        except Exception as e:
            logger.error(f"NewsAnalyst error: {e}")
            return AnalystInsight("news", "neutral", 0.3, f"Analysis error: {e}")


class AnalystTeam:
    """分析师团队 - 并行执行所有分析师"""

    def __init__(self, llm_client: LLMClient, timeout: float = 45.0):
        self.fund_analyst = FundAnalyst(llm_client)
        self.sentiment_analyst = SentimentAnalyst(llm_client)
        self.technical_analyst = TechnicalAnalyst(llm_client)
        self.news_analyst = NewsAnalyst(llm_client)
        self.timeout = timeout  # 全局超时

    def analyze(self, stock_data: Dict) -> List[AnalystInsight]:
        """并行执行所有分析师 (线程池)"""
        analysts = [
            (0, 'fundamental', self.fund_analyst),
            (1, 'sentiment', self.sentiment_analyst),
            (2, 'technical', self.technical_analyst),
            (3, 'news', self.news_analyst),
        ]

        insights = [None] * len(analysts)

        def _run(idx, name, analyst):
            try:
                insights[idx] = analyst.analyze(stock_data)
                logger.debug(f"[AnalystTeam] {name} 完成")
            except Exception as e:
                logger.error(f"[AnalystTeam] {name} 异常: {e}")
                insights[idx] = AnalystInsight(name, "neutral", 0.3, f"Error: {e}")

        try:
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(_run, idx, name, a) for idx, name, a in analysts]
                for f in futures:
                    f.result(timeout=self.timeout)  # 等待所有完成，带超时
        except FuturesTimeoutError:
            logger.warning("[AnalystTeam] 分析师并行超时，返回已完成的分析")
        except Exception as e:
            logger.error(f"[AnalystTeam] 并行执行异常: {e}")

        # 过滤 None
        result = [i for i in insights if i is not None]
        if not result:
            logger.error("[AnalystTeam] 所有分析师均失败，返回默认洞察")
            return [AnalystInsight("default", "neutral", 0.3, "All analysts failed")]
        return result
    
    def get_summary(self, insights: List[AnalystInsight]) -> Dict:
        """获取分析师团队总结"""
        if not insights:
            return {
                "consensus": "neutral", "bullish_count": 0, "bearish_count": 0,
                "neutral_count": 0, "avg_confidence": 0.3, "agreement_ratio": 0.0,
                "insights": [],
            }

        bullish_count = sum(1 for i in insights if i.direction == "bullish")
        bearish_count = sum(1 for i in insights if i.direction == "bearish")
        neutral_count = sum(1 for i in insights if i.direction == "neutral")
        avg_confidence = sum(i.confidence for i in insights) / len(insights)

        # 综合方向
        if bullish_count > bearish_count:
            consensus = "bullish"
        elif bearish_count > bullish_count:
            consensus = "bearish"
        else:
            consensus = "neutral"

        # 共识比例: 同意共识方向的分析师比例
        agree_count = {"bullish": bullish_count, "bearish": bearish_count, "neutral": neutral_count}[consensus]
        agreement_ratio = agree_count / len(insights)

        return {
            "consensus": consensus,
            "bullish_count": bullish_count,
            "bearish_count": bearish_count,
            "neutral_count": neutral_count,
            "avg_confidence": round(avg_confidence, 3),
            "agreement_ratio": round(agreement_ratio, 3),
            "insights": [
                {"type": i.analyst_type, "direction": i.direction,
                 "confidence": i.confidence, "reasoning": i.reasoning[:200]}
                for i in insights
            ]
        }
