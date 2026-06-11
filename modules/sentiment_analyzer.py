#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Sentiment Analyzer — 情绪分析模块

从 akshare 获取新闻和股吧数据，基于情感词典进行简单情绪分析。
返回 -1 到 1 的情感分数。

情感词典来源:
  - 正面词汇: 上涨、突破、利好、增持、新高、强势、放量、突破、突破、看涨
  - 负面词汇: 下跌、破位、利空、减持、新低、弱势、缩量、破位、看空、暴跌
"""

import os
import re
import logging
from typing import Dict, Optional, List
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# ── 情感词典 ──────────────────────────────────────────────

POSITIVE_WORDS = [
    '上涨', '突破', '利好', '增持', '新高', '强势', '放量',
    '看涨', '反弹', '增长', '盈利', '超预期', '创新高',
    '突破阻力', '资金流入', '主力买入', '放量上涨', '均线多头',
    '金叉', '底背离', '反转', '景气', '复苏', '订单',
    '业绩预增', '分红', '回购', '增持', '推荐', '买入',
    '景气度提升', '需求旺盛', '产能扩张', '战略合作',
    '历史新高', '放量突破', '缩量回调', '企稳回升',
    '估值修复', '戴维斯双击', '业绩拐点',
]

NEGATIVE_WORDS = [
    '下跌', '破位', '利空', '减持', '新低', '弱势', '缩量',
    '看空', '暴跌', '跌停', '套牢', '出货', '主力流出',
    '主力卖出', '放量下跌', '均线空头', '死叉', '顶背离',
    '衰退', '亏损', '退市', 'ST', '处罚', '调查',
    '业绩预减', '业绩下滑', '营收下降', '利润下滑',
    '质押风险', '商誉减值', '应收账款', '现金流紧张',
    '破净资产', '估值过高', '泡沫', '回调', '下行风险',
]

# 否定词（否定后面的情绪词）
NEGATION_WORDS = ['不', '没', '非', '无', '未', '别', '勿', '毫不', '并非']

# 程度副词
INTENSIFIERS = {
    '非常': 1.5, '特别': 1.5, '极其': 1.8, '十分': 1.4,
    '较': 0.8, '比较': 0.8, '稍微': 0.6, '略': 0.5,
    '大幅': 1.6, '急剧': 1.7, '持续': 1.2, '明显': 1.3,
}


class SentimentAnalyzer:
    """基于情感词典的 A 股情绪分析器。"""

    def __init__(self):
        self.positive_words = set(POSITIVE_WORDS)
        self.negative_words = set(NEGATIVE_WORDS)
        self.negation_words = set(NEGATION_WORDS)
        self.intensifiers = INTENSIFIERS

    def analyze_text(self, text: str) -> Dict:
        """分析单段文本的情绪。

        Returns:
            {
                'score': -1.0 ~ 1.0,
                'positive_count': int,
                'negative_count': int,
                'positive_words': List[str],
                'negative_words': List[str],
                'dominant': 'positive' | 'negative' | 'neutral'
            }
        """
        if not text or not text.strip():
            return {
                'score': 0.0, 'positive_count': 0, 'negative_count': 0,
                'positive_words': [], 'negative_words': [], 'dominant': 'neutral'
            }

        # 分句
        sentences = re.split(r'[。！？；\n]', text)

        total_score = 0.0
        pos_count = 0
        neg_count = 0
        pos_words_found = []
        neg_words_found = []

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # 检查每个情感词
            for word in self.positive_words:
                if word in sentence:
                    # 检查否定词
                    pos_idx = sentence.index(word)
                    preceding = sentence[max(0, pos_idx - 3):pos_idx]
                    has_negation = any(neg in preceding for neg in self.negation_words)

                    # 检查程度副词
                    intensifier_score = 1.0
                    for intensifier, multiplier in self.intensifiers.items():
                        if intensifier in preceding:
                            intensifier_score = multiplier
                            break

                    weight = intensifier_score if not has_negation else -intensifier_score
                    total_score += weight
                    pos_count += 1
                    pos_words_found.append(word)

            for word in self.negative_words:
                if word in sentence:
                    pos_idx = sentence.index(word)
                    preceding = sentence[max(0, pos_idx - 3):pos_idx]
                    has_negation = any(neg in preceding for neg in self.negation_words)

                    intensifier_score = 1.0
                    for intensifier, multiplier in self.intensifiers.items():
                        if intensifier in preceding:
                            intensifier_score = multiplier
                            break

                    weight = -intensifier_score if not has_negation else intensifier_score
                    total_score += weight
                    neg_count += 1
                    neg_words_found.append(word)

        # 归一化到 -1 ~ 1
        total_mentions = pos_count + neg_count
        if total_mentions > 0:
            score = total_score / (total_mentions * 2.0)  # max possible = total_mentions * 2.0 / 2.0 = 1.0
            score = max(-1.0, min(1.0, score))
        else:
            score = 0.0

        if score > 0.1:
            dominant = 'positive'
        elif score < -0.1:
            dominant = 'negative'
        else:
            dominant = 'neutral'

        return {
            'score': round(score, 3),
            'positive_count': pos_count,
            'negative_count': neg_count,
            'positive_words': list(set(pos_words_found)),
            'negative_words': list(set(neg_words_found)),
            'dominant': dominant,
        }

    def get_sentiment_score(self, stock_code: str, stock_name: str = '') -> Dict:
        """获取股票的综合情绪分数。

        从多个来源获取数据并综合评分。

        Returns:
            {
                'score': -1.0 ~ 1.0,
                'news_sentiment': Dict,
                'social_sentiment': Dict,
                'technical_sentiment': Dict,
                'composite': float,
                'level': 'strong_bullish' | 'bullish' | 'neutral' | 'bearish' | 'strong_bearish'
            }
        """
        # 1. 新闻情绪（从 akshare 获取）
        news_sentiment = self._analyze_news_sentiment(stock_code, stock_name)

        # 2. 社交情绪（从 akshare 股吧获取）
        social_sentiment = self._analyze_social_sentiment(stock_name)

        # 3. 技术面情绪（从价格动量推断）
        technical_sentiment = self._analyze_technical_sentiment(stock_code)

        # 综合评分（加权平均）
        weights = {'news': 0.4, 'social': 0.3, 'technical': 0.3}
        scores = {
            'news': news_sentiment.get('score', 0),
            'social': social_sentiment.get('score', 0),
            'technical': technical_sentiment.get('score', 0),
        }
        composite = sum(weights[k] * scores[k] for k in weights)
        composite = max(-1.0, min(1.0, composite))

        # 情绪等级
        if composite > 0.4:
            level = 'strong_bullish'
        elif composite > 0.1:
            level = 'bullish'
        elif composite > -0.1:
            level = 'neutral'
        elif composite > -0.4:
            level = 'bearish'
        else:
            level = 'strong_bearish'

        return {
            'score': round(composite, 3),
            'news_sentiment': news_sentiment,
            'social_sentiment': social_sentiment,
            'technical_sentiment': technical_sentiment,
            'composite': round(composite, 3),
            'level': level,
        }

    def _analyze_news_sentiment(self, stock_code: str, stock_name: str) -> Dict:
        """分析新闻情绪。"""
        try:
            import akshare as ak
            # 获取个股新闻
            news_df = ak.stock_news_em(symbol=stock_code.replace('sz', '').replace('sh', ''))
            if news_df is not None and len(news_df) > 0:
                # 取最近 20 条新闻
                news_df = news_df.head(20)
                texts = news_df.get('内容', news_df.get('新闻内容', news_df.get('title', '')))
                if hasattr(texts, 'tolist'):
                    texts = texts.tolist()
                else:
                    texts = [str(t) for t in texts]

                sentiments = [self.analyze_text(str(t)) for t in texts[:10]]
                avg_score = sum(s['score'] for s in sentiments) / len(sentiments) if sentiments else 0

                pos_total = sum(s['positive_count'] for s in sentiments)
                neg_total = sum(s['negative_count'] for s in sentiments)

                return {
                    'score': round(avg_score, 3),
                    'sample_count': len(sentiments),
                    'positive_mentions': pos_total,
                    'negative_mentions': neg_total,
                    'dominant': 'positive' if avg_score > 0.1 else 'negative' if avg_score < -0.1 else 'neutral',
                }
        except Exception as e:
            logger.warning(f"News sentiment fetch failed for {stock_code}: {e}")

        return {'score': 0.0, 'sample_count': 0, 'positive_mentions': 0, 'negative_mentions': 0, 'dominant': 'neutral'}

    def _analyze_social_sentiment(self, stock_name: str) -> Dict:
        """分析股吧社交情绪。"""
        try:
            import akshare as ak
            # 获取股吧数据
            guba_df = ak.stock_gjxg_em(symbol=stock_name)
            if guba_df is not None and len(guba_df) > 0:
                # 取最近 10 条讨论
                guba_df = guba_df.head(10)
                # 尝试获取评论内容
                texts = []
                for col in ['content', '评论', 'content_text', 'message']:
                    if col in guba_df.columns:
                        texts = guba_df[col].tolist()
                        break
                if not texts:
                    # 如果没有评论列，使用标题
                    for col in ['title', '标题', 'content', 'content_text']:
                        if col in guba_df.columns:
                            texts = guba_df[col].tolist()
                            break

                if texts:
                    sentiments = [self.analyze_text(str(t)) for t in texts]
                    avg_score = sum(s['score'] for s in sentiments) / len(sentiments) if sentiments else 0
                    return {
                        'score': round(avg_score, 3),
                        'sample_count': len(sentiments),
                        'dominant': 'positive' if avg_score > 0.1 else 'negative' if avg_score < -0.1 else 'neutral',
                    }
        except Exception as e:
            logger.warning(f"Social sentiment fetch failed for {stock_name}: {e}")

        return {'score': 0.0, 'sample_count': 0, 'dominant': 'neutral'}

    def _analyze_technical_sentiment(self, stock_code: str) -> Dict:
        """基于技术指标推断情绪。"""
        try:
            from .real_data_loader import RealDataLoader
            loader = RealDataLoader()
            df = loader.load_klines(stock_code, lookback=60)

            if df.empty or len(df) < 20:
                return {'score': 0.0, 'reason': 'insufficient_data'}

            closes = df['close'].values.astype(float)
            volumes = df['volume'].values.astype(float)

            # 1. 短期动量（5 日）
            momentum_5d = (closes[-1] / closes[-5] - 1) if closes[-5] > 0 else 0

            # 2. 均线排列
            ma5 = np.mean(closes[-5:])
            ma20 = np.mean(closes[-20:]) if len(closes) >= 20 else closes[-1]
            ma60 = np.mean(closes[-60:]) if len(closes) >= 60 else closes[-1]

            # 3. RSI
            if len(closes) >= 15:
                deltas = np.diff(closes[-15:])
                gains = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
                losses = abs(np.mean(deltas[deltas < 0])) if np.any(deltas < 0) else 0.001
                rsi = 100 - (100 / (1 + gains / losses))
            else:
                rsi = 50

            # 4. 成交量趋势
            vol_recent = np.mean(volumes[-5:])
            vol_avg = np.mean(volumes[-20:]) if len(volumes) >= 20 else vol_recent
            vol_ratio = vol_recent / vol_avg if vol_avg > 0 else 1.0

            # 综合技术评分
            score = 0.0
            score += min(0.3, max(-0.3, momentum_5d * 5))  # 5 日动量
            score += 0.2 if ma5 > ma20 else -0.2  # 均线关系
            score += 0.15 if rsi < 30 else (-0.15 if rsi > 70 else 0)  # RSI 超卖/超买
            score += 0.15 if vol_ratio > 1.5 else -0.1 if vol_ratio < 0.5 else 0  # 成交量

            score = max(-1.0, min(1.0, score))

            reasons = []
            if momentum_5d > 0.03:
                reasons.append('短期动量强劲')
            elif momentum_5d < -0.03:
                reasons.append('短期动量弱势')
            if ma5 > ma20:
                reasons.append('站上20日均线')
            else:
                reasons.append('低于20日均线')

            return {
                'score': round(score, 3),
                'momentum_5d': round(momentum_5d, 4),
                'rsi': round(rsi, 1),
                'ma5': round(ma5, 2),
                'ma20': round(ma20, 2),
                'vol_ratio': round(vol_ratio, 2),
                'reasons': reasons,
            }
        except Exception as e:
            logger.warning(f"Technical sentiment fetch failed for {stock_code}: {e}")
            return {'score': 0.0, 'reason': 'fetch_error'}
