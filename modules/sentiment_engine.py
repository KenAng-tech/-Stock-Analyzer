#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
FinBERT 情感分析集成模块 — 替换词典法

功能:
1. 使用 FinBERT/HuggingFace 进行金融情感分析
2. 自动降级到词典法 (当模型不可用时)
3. 支持中文金融文本
4. 统一的情感分数接口

作者: 基于 yiyanghkust/finbert-tone
参考: modules/sentiment_finbert.py
"""

import os
import numpy as np
import threading

from modules.logger import logger


# ── 中文金融情感词典 (增强版) ────────────────────────────────

CHINESE_FINANCIAL_SENTIMENT = {
    'positive': {
        '上涨', '利好', '突破', '看好', '增长', '盈利', '超预期',
        '创新高', '放量', '金叉', '多头', '强势', '反弹', '反转',
        '业绩优良', '分红', '回购', '增持', '景气', '景气度高',
        '订单饱满', '产能扩张', '技术领先', '市场份额', '龙头',
        '受益', '复苏', '回暖', '增长确定', '护城河', '壁垒高',
        'bullish', 'positive', 'growth', 'profit', 'surge', 'rally',
        'breakout', 'upside', 'strong', 'beat', 'raise', 'upgrade',
        '涨停', '拉升', '底部', '爆发', '推荐', '买入', '优秀', '优质',
        '成长', '加速', '放量上涨', '量价齐升', '资金流入', '主力买入',
        '机构看好', '业绩预增', '技术突破', '国产替代', 'AI', '算力',
    },
    'negative': {
        '下跌', '利空', '破位', '看空', '下滑', '亏损', '低于预期',
        '创新低', '缩量', '死叉', '空头', '弱势', '回调', '下跌趋势',
        '业绩下滑', '减持', '衰退', '低迷', '订单减少', '产能过剩',
        '竞争激烈', '份额流失', '受损', '恶化', '降温', '增长放缓',
        '风险', '监管', '处罚', 'investigation', 'fraud', 'loss',
        'decline', 'crash', 'plunge', 'downgrade', 'sell', 'bearish',
        '跌停', '顶部', '见顶', '诉讼', '暴雷', '质押', '违约', '退市',
        'ST', '放量下跌', '量价齐跌', '资金流出', '主力卖出', '机构减持',
        '业绩预减', '竞争加剧', '贸易摩擦', '跌破', '走坏', '下行', '承压',
    },
    'degree': {
        '非常': 1.8, '极其': 1.8, '特别': 1.6, '十分': 1.5,
        '很': 1.5, '相当': 1.4, '比较': 1.2, '略': 0.8,
        '稍微': 0.7, '轻微': 0.6,
    },
    'negation': {
        '不', '没', '未', '非', '无', '并非', '未能',
        'not', 'no', 'never', 'neither', 'nor',
    },
}


# ── 词典情感分析器 (Fallback) ─────────────────────────────────

class DictionarySentimentAnalyzer:
    """
    词典情感分析器 — 基于规则的中文金融情感分析

    算法:
        1. 分词 + 贪心最长匹配
        2. 否定词处理 (不+好 = 不好)
        3. 程度副词加权 (非常+好 = 强正面)
        4. 综合情感分数 (-1 ~ +1)
    """

    def __init__(self):
        self.positive_words = CHINESE_FINANCIAL_SENTIMENT['positive']
        self.negative_words = CHINESE_FINANCIAL_SENTIMENT['negative']
        self.degree_words = CHINESE_FINANCIAL_SENTIMENT['degree']
        self.negation_words = CHINESE_FINANCIAL_SENTIMENT['negation']

    def analyze(self, text: str) -> Dict:
        """
        分析单条文本情感

        Args:
            text: 输入文本

        Returns:
            {
                'score': -1.0 ~ +1.0,
                'label': 'positive' | 'neutral' | 'negative',
                'confidence': 0.0 ~ 1.0,
                'scores': {'positive': float, 'neutral': float, 'negative': float},
                'positive_count': int,
                'negative_count': int,
            }
        """
        if not text:
            return self._empty_result()

        import re
        text = re.sub(r'[^\w一-鿿]', ' ', str(text)).strip()
        if not text:
            return self._empty_result()

        words_found = []
        pos_count = 0
        neg_count = 0
        used = [False] * len(text)

        # 贪心最长匹配
        i = 0
        while i < len(text):
            if used[i]:
                i += 1
                continue

            best_match = None
            best_length = 0

            for length in range(min(4, len(text) - i), 1, -1):
                ngram = text[i:i + length]
                if ngram in self.positive_words or ngram in self.negative_words:
                    best_match = ngram
                    best_length = length
                    break

            if best_match:
                # 检查前面的词
                degree = 1.0
                negation = False

                if i > 0:
                    prev_start = max(0, i - 3)
                    prev_text = text[prev_start:i]

                    for neg in self.negation_words:
                        if neg in prev_text:
                            negation = True
                            break

                    for deg_word, deg_val in self.degree_words.items():
                        if deg_word in prev_text:
                            degree = deg_val
                            break

                if best_match in self.positive_words:
                    pos_count += 1
                    if negation:
                        neg_count += 1
                    else:
                        pos_count += degree
                else:
                    neg_count += 1
                    if negation:
                        pos_count += 1
                    else:
                        neg_count += degree * 1.0

                words_found.append(best_match)
                for j in range(best_length):
                    if i + j < len(used):
                        used[i + j] = True

            i += 1

        # 计算最终分数
        total = pos_count + neg_count
        if total > 0:
            score = (pos_count - neg_count) / total
        else:
            score = 0.0

        # 归一化到 -1 ~ +1
        score = max(-1.0, min(1.0, score))

        # 标签
        if score > 0.1:
            label = 'positive'
        elif score < -0.1:
            label = 'negative'
        else:
            label = 'neutral'

        # 置信度
        confidence = min(1.0, abs(score) * 0.8 + 0.2)

        return {
            'score': float(score),
            'label': label,
            'confidence': float(confidence),
            'scores': {
                'positive': float(pos_count / max(total, 1)),
                'neutral': float(1 - abs(score) * 0.5),
                'negative': float(neg_count / max(total, 1)),
            },
            'positive_count': int(pos_count),
            'negative_count': int(neg_count),
            'words': words_found[:10],  # 最多 10 个词
        }

    def _empty_result(self) -> Dict:
        return {
            'score': 0.0,
            'label': 'neutral',
            'confidence': 0.0,
            'scores': {'positive': 0.0, 'neutral': 1.0, 'negative': 0.0},
            'positive_count': 0,
            'negative_count': 0,
            'words': [],
        }


# ── FinBERT 情感分析器 ──────────────────────────────────────

class FinBERTSentimentAnalyzer:
    """
    FinBERT 情感分析器 — 使用 HuggingFace 模型

    支持模型:
        - yiyanghkust/finbert-tone (英文金融情感)
        - hfl/chinese-roberta-wwm-ext (中文预训练模型)
        - 降级: DictionarySentimentAnalyzer
    """

    def __init__(self, model_name: str = 'yiyanghkust/finbert-tone'):
        """
        Args:
            model_name: HuggingFace 模型名称
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self._initialized = False
        self._use_hf = False

        self._init_model()

    def _init_model(self):
        """初始化模型"""
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification

            logger.info(f"[FinBERT] 加载模型: {self.model_name}")

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.eval()

            self._use_hf = True
            self._initialized = True
            logger.info("[FinBERT] 模型加载成功")
        except Exception as e:
            logger.warning(f"[FinBERT] 模型加载失败: {e}, 使用词典 fallback")
            self._use_hf = False

    def analyze(self, text: str) -> Dict:
        """
        分析文本情感

        Args:
            text: 输入文本

        Returns:
            {
                'score': -1.0 ~ +1.0,
                'label': 'positive' | 'neutral' | 'negative',
                'confidence': 0.0 ~ 1.0,
                'scores': {'positive': float, 'neutral': float, 'negative': float},
            }
        """
        if not self._initialized or not self._use_hf:
            return DictionarySentimentAnalyzer().analyze(text)

        try:
            import torch

            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                max_length=512,
                padding=True,
            )

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                confidence = probs.max().item()
                label_idx = probs.argmax().item()

            # FinBERT: 0=negative, 1=neutral, 2=positive
            label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
            label = label_map[label_idx]

            scores = {
                'negative': probs[0, 0].item(),
                'neutral': probs[0, 1].item(),
                'positive': probs[0, 2].item(),
            }

            # 转换为 -1 ~ +1
            score = scores['positive'] - scores['negative']

            return {
                'score': float(score),
                'label': label,
                'confidence': float(confidence),
                'scores': scores,
            }
        except Exception as e:
            logger.error(f"[FinBERT] 分析错误: {e}")
            return DictionarySentimentAnalyzer().analyze(text)


# ── 统一情感分析引擎 ────────────────────────────────────────

class SentimentEngine:
    """
    统一情感分析引擎

    自动选择最佳可用方法:
        1. FinBERT (如果可用)
        2. Dictionary Sentiment (fallback)
    """

    def __init__(self, use_finbert: bool = True):
        """
        Args:
            use_finbert: 是否优先使用 FinBERT (False 则强制词典法)
        """
        self.use_finbert = use_finbert
        self._finbert = None
        self._dictionary = DictionarySentimentAnalyzer()

        if use_finbert:
            self._init_finbert()

    def _init_finbert(self):
        """初始化 FinBERT"""
        try:
            self._finbert = FinBERTSentimentAnalyzer()
            logger.info("[SentimentEngine] FinBERT 已启用")
        except Exception as e:
            logger.warning(f"[SentimentEngine] FinBERT 初始化失败: {e}")
            self._finbert = None

    def analyze(self, text: str) -> Dict:
        """
        分析单条文本

        Returns:
            {
                'score': -1.0 ~ +1.0,
                'label': 'positive' | 'neutral' | 'negative',
                'confidence': 0.0 ~ 1.0,
                'scores': {'positive': float, 'neutral': float, 'negative': float},
                'method': 'finbert' | 'dictionary',
            }
        """
        if self._finbert is not None and self.use_finbert:
            result = self._finbert.analyze(text)
            result['method'] = 'finbert'
            return result
        else:
            result = self._dictionary.analyze(text)
            result['method'] = 'dictionary'
            return result

    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        """
        批量分析

        Returns:
            List[Dict] 每条文本的分析结果
        """
        return [self.analyze(text) for text in texts]

    def aggregate(self, texts: List[str], weights: Optional[List[float]] = None) -> Dict:
        """
        聚合多条文本的情感分析

        Args:
            texts: 文本列表
            weights: 可选权重 (默认等权重)

        Returns:
            {
                'aggregate_score': -1.0 ~ +1.0,
                'aggregate_label': 'positive' | 'neutral' | 'negative',
                'aggregate_confidence': 0.0 ~ 1.0,
                'text_count': int,
                'method': 'finbert' | 'dictionary',
            }
        """
        if not texts:
            return {
                'aggregate_score': 0.0,
                'aggregate_label': 'neutral',
                'aggregate_confidence': 0.0,
                'text_count': 0,
                'method': 'none',
            }

        results = self.analyze_batch(texts)

        if weights is None:
            weights = [1.0] * len(results)
        else:
            weights = weights[:len(results)]

        total_weight = sum(weights)
        weighted_score = sum(r['score'] * w for r, w in zip(results, weights)) / total_weight
        weighted_confidence = sum(r['confidence'] * w for r, w in zip(results, weights)) / total_weight

        # 聚合标签 (多数投票)
        labels = [r['label'] for r in results]
        label_counts = {'positive': labels.count('positive'), 'neutral': labels.count('neutral'), 'negative': labels.count('negative')}
        aggregate_label = max(label_counts, key=label_counts.get)

        return {
            'aggregate_score': float(weighted_score),
            'aggregate_label': aggregate_label,
            'aggregate_confidence': float(weighted_confidence),
            'text_count': len(texts),
            'method': results[0].get('method', 'unknown'),
            'label_distribution': label_counts,
        }


# ── 全局单例 ────────────────────────────────────────────────

_sentiment_engine_instance: Optional[SentimentEngine] = None
_sentiment_engine_lock = threading.Lock()


def get_sentiment_engine() -> SentimentEngine:
    """获取全局 SentimentEngine 实例 (线程安全)"""
    global _sentiment_engine_instance
    if _sentiment_engine_instance is None:
        with _sentiment_engine_lock:
            if _sentiment_engine_instance is None:
                _sentiment_engine_instance = SentimentEngine(use_finbert=True)
    return _sentiment_engine_instance


# ── 兼容性别名 ──────────────────────────────────────────────

sentiment_analyzer = SentimentEngine  # 兼容旧名称