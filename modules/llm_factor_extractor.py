#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
LLM 增强因子挖掘 (LLM-Augmented Alpha Discovery)

使用 LLM 分析新闻/财报/公告，提取非结构化 alpha 信号:
- 新闻事件提取: 并购、财报超预期、政策利好等
- 情感因子: 新闻情感 → 量化因子
- 事件驱动因子: 具体事件 → 量化评分
- 因果验证: 对 LLM 生成的因子进行因果检验

用法:
    extractor = LLMFactorExtractor()
    factors = extractor.extract_factors('sz300620', news_list)
"""

import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from modules.logger import logger


@dataclass
class LLMFactor:
    """LLM 生成的因子"""
    name: str
    value: float  # 归一化到 -1 ~ 1
    category: str  # sentiment/event/policy/management
    source: str  # news/guba/announcement
    confidence: float  # 0-1
    raw_signal: str  # LLM 原始输出
    timestamp: str = ''

    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'value': self.value,
            'category': self.category,
            'source': self.source,
            'confidence': self.confidence,
            'raw_signal': self.raw_signal,
            'timestamp': self.timestamp,
        }


class LLMFactorExtractor:
    """LLM 因子提取器"""

    # 事件类型定义
    EVENT_TYPES = {
        'earnings_beat': ('业绩超预期', 0.8),
        'earnings_miss': ('业绩不及预期', -0.8),
        'merger_acquisition': ('并购重组', 0.6),
        'policy_support': ('政策利好', 0.5),
        'policy_restrict': ('政策监管', -0.5),
        'insider_buy': ('高管增持', 0.4),
        'insider_sell': ('高管减持', -0.4),
        'dividend': ('分红派息', 0.3),
        'share_buyback': ('股份回购', 0.3),
        'lawsuit': ('诉讼风险', -0.6),
        'fine': ('监管处罚', -0.7),
        'new_contract': ('新订单', 0.3),
        'tech_breakthrough': ('技术突破', 0.5),
    }

    def __init__(self, llm_api_url: str = 'http://127.0.0.1:8080', api_key: str = ''):
        self.llm_api_url = llm_api_url
        self.api_key = api_key
        self._factors: Dict[str, List[LLMFactor]] = {}  # stock_code -> factors
        self._llm_client = None

    def _get_llm_client(self):
        """获取 LLM 客户端"""
        if self._llm_client is None:
            try:
                from openai import OpenAI
                self._llm_client = OpenAI(
                    base_url=self.llm_api_url,
                    api_key=self.api_key or 'no-key',
                )
            except ImportError:
                logger.warning("[LLMFactor] openai SDK 未安装，使用规则引擎")
                self._llm_client = 'rule_engine'
        return self._llm_client

    def extract_factors(self, stock_code: str, news_items: List[Dict]) -> List[LLMFactor]:
        """
        从新闻/公告中提取 LLM 因子

        Args:
            stock_code: 股票代码
            news_items: 新闻列表 [{title, content, source, date}, ...]

        Returns:
            LLMFactor 列表
        """
        factors = []

        for item in news_items:
            title = item.get('title', '')
            content = item.get('content', '') or item.get('summary', '')
            source = item.get('source', 'unknown')

            # 1. 规则引擎提取 (快速)
            rule_factors = self._rule_extract(title, content, source)
            factors.extend(rule_factors)

            # 2. LLM 提取 (如果可用)
            llm_client = self._get_llm_client()
            if llm_client == 'rule_engine':
                continue

            try:
                llm_factors = self._llm_extract(stock_code, title, content, source)
                factors.extend(llm_factors)
            except Exception as e:
                logger.warning(f"[LLMFactor] LLM 提取失败: {e}")

        # 合并同类因子
        factors = self._merge_factors(factors)
        self._factors[stock_code] = factors
        return factors

    def _rule_extract(self, title: str, content: str, source: str) -> List[LLMFactor]:
        """规则引擎提取因子"""
        factors = []
        text = title + ' ' + content
        text_lower = text.lower()

        for event_type, (label, score) in self.EVENT_TYPES.items():
            keywords = self._get_event_keywords(event_type)
            if any(kw in text_lower for kw in keywords):
                factor = LLMFactor(
                    name=f'event_{event_type}',
                    value=score,
                    category='event',
                    source=source,
                    confidence=0.7,
                    raw_signal=f"{label}: {text[:100]}",
                    timestamp=datetime.now().isoformat(),
                )
                factors.append(factor)

        # 情感因子
        sentiment_score = self._quick_sentiment(text)
        if abs(sentiment_score) > 0.2:
            factors.append(LLMFactor(
                name='sentiment_news',
                value=sentiment_score,
                category='sentiment',
                source=source,
                confidence=0.6,
                raw_signal=f"情感评分: {sentiment_score:.3f}",
                timestamp=datetime.now().isoformat(),
            ))

        return factors

    def _get_event_keywords(self, event_type: str) -> List[str]:
        """获取事件关键词"""
        keywords_map = {
            'earnings_beat': ['业绩超预期', '大幅预增', '净利润增长', '超预期'],
            'earnings_miss': ['业绩下滑', '亏损', '预亏', '不及预期'],
            'merger_acquisition': ['并购', '重组', '收购', '合并'],
            'policy_support': ['政策支持', '利好', '扶持', '补贴'],
            'policy_restrict': ['监管', '处罚', '限制', '反垄断'],
            'insider_buy': ['增持', '回购', '买入'],
            'insider_sell': ['减持', '套现', '退出'],
            'dividend': ['分红', '派息', '股息'],
            'share_buyback': ['回购', '股份回购'],
            'lawsuit': ['诉讼', '仲裁', '纠纷'],
            'fine': ['处罚', '罚款', '立案'],
            'new_contract': ['中标', '签约', '订单'],
            'tech_breakthrough': ['突破', '专利', '研发'],
        }
        return keywords_map.get(event_type, [])

    def _quick_sentiment(self, text: str) -> float:
        """快速情感评分 (基于关键词)"""
        positive_words = ['利好', '增长', '突破', '优秀', '领先', '创新', '盈利', '上涨']
        negative_words = ['利空', '下滑', '亏损', '风险', '处罚', '诉讼', '下跌', '暴跌']

        pos_count = sum(1 for w in positive_words if w in text)
        neg_count = sum(1 for w in negative_words if w in text)

        total = pos_count + neg_count
        if total == 0:
            return 0.0
        return (pos_count - neg_count) / total

    def _llm_extract(self, stock_code: str, title: str, content: str, source: str) -> List[LLMFactor]:
        """使用 LLM 提取因子"""
        prompt = f"""你是一个量化分析师。分析以下关于 {stock_code} 的新闻，提取可量化的 alpha 因子。

新闻标题: {title}
新闻内容: {content[:500]}

请返回 JSON 格式:
{{
    "factors": [
        {{"name": "因子名", "value": -1到1之间的数值, "category": "sentiment/event/policy", "confidence": 0到1之间的数值, "reason": "简要原因"}}
    ]
}}

只返回 JSON，不要其他内容。"""

        client = self._llm_client
        response = client.chat.completions.create(
            model='Qwen3.6-35B-A3B-Claude-4.6-Opus-abliterated-mlx-bf16',
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0,
            max_tokens=500,
        )

        text = response.choices[0].message.content
        try:
            data = json.loads(text)
            factors = []
            for item in data.get('factors', []):
                factors.append(LLMFactor(
                    name=item['name'],
                    value=float(item['value']),
                    category=item['category'],
                    source=source,
                    confidence=float(item.get('confidence', 0.5)),
                    raw_signal=item.get('reason', ''),
                    timestamp=datetime.now().isoformat(),
                ))
            return factors
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"[LLMFactor] LLM 响应解析失败: {e}")
            return []

    def _merge_factors(self, factors: List[LLMFactor]) -> List[LLMFactor]:
        """合并同类因子，取加权平均"""
        merged = {}
        for f in factors:
            key = f'{f.name}_{f.category}'
            if key in merged:
                existing = merged[key]
                # 加权平均
                total_weight = existing.confidence + f.confidence
                existing.value = (existing.value * existing.confidence + f.value * f.confidence) / total_weight
                existing.confidence = min(1.0, total_weight / 2)
            else:
                merged[key] = f
        return list(merged.values())

    def get_factors(self, stock_code: str) -> List[Dict]:
        """获取股票的 LLM 因子"""
        return [f.to_dict() for f in self._factors.get(stock_code, [])]

    def get_factor_summary(self, stock_code: str) -> Dict:
        """获取因子摘要"""
        factors = self._factors.get(stock_code, [])
        if not factors:
            return {'count': 0, 'factors': [], 'by_category': {}, 'avg_factor_value': 0, 'avg_confidence': 0}

        by_category = {}
        total_value = 0.0
        total_confidence = 0.0

        for f in factors:
            by_category[f.category] = by_category.get(f.category, 0) + 1
            total_value += f.value * f.confidence
            total_confidence += f.confidence

        avg_value = total_value / max(total_confidence, 1e-8)

        return {
            'count': len(factors),
            'by_category': by_category,
            'avg_factor_value': round(avg_value, 4),
            'avg_confidence': round(total_confidence / len(factors), 4),
            'factors': [f.to_dict() for f in factors],
        }


# 全局单例
_llm_extractor: Optional[LLMFactorExtractor] = None


def get_llm_factor_extractor() -> LLMFactorExtractor:
    """获取 LLM 因子提取器全局实例"""
    global _llm_extractor
    if _llm_extractor is None:
        _llm_extractor = LLMFactorExtractor()
    return _llm_extractor
