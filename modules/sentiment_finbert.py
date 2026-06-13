#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
FinBERT 情感分析 — 替换词典法

使用 FinBERT (金融领域预训练 BERT) 进行中文情感分析。

支持两种模式:
    1. 微调版 FinBERT (推荐): 使用在中文金融数据上微调的模型
    2. 通用中文 BERT + 金融词典 fallback (当模型不可用时)

模型选择:
    - 首选: yiyanghkust/finbert-tone (英文金融情感，最成熟)
    - 中文: hfl/chinese-roberta-wwm-ext (通用中文，配合金融词典)
    - 降级: 词典法 (无依赖)

参考:
    - Yi, et al., "FinBERT: A Pre-trained Financial Language Model" (2020)
    - https://github.com/yiyanghkust/FinBERT
"""

import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import re

from modules.logger import logger

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logger.warning("[FinBERT] transformers 未安装，使用词典 fallback")


# ── 中文金融情感词典 ──────────────────────────────────────────

CHINESE_FINANCIAL_SENTIMENT = {
    'positive': {
        '上涨', '利好', '突破', '看好', '增长', '盈利', '超预期',
        '创新高', '放量', '金叉', '多头', '强势', '反弹', '反转',
        '业绩优良', '分红', '回购', '增持', '利好', '景气', '景气度高',
        '订单饱满', '产能扩张', '技术领先', '市场份额', '龙头',
        '受益', '复苏', '回暖', '增长确定', '护城河', '壁垒高',
        'bullish', 'positive', 'growth', 'profit', 'surge', 'rally',
        'breakout', 'upside', 'strong', 'beat', 'raise', 'upgrade',
    },
    'negative': {
        '下跌', '利空', '破位', '看空', '下滑', '亏损', '低于预期',
        '创新低', '缩量', '死叉', '空头', '弱势', '回调', '下跌趋势',
        '业绩下滑', '减持', '减持', '利空', '衰退', '低迷',
        '订单减少', '产能过剩', '竞争激烈', '份额流失', '挑战',
        '受损', '恶化', '降温', '增长放缓', '风险', '监管', '处罚',
        'investigation', 'fraud', 'loss', 'decline', 'crash', 'plunge',
        'downgrade', 'sell', 'bearish', 'warning', 'risk', 'debt',
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


class FinBERTSentiment:
    """
    FinBERT 情感分析器

    功能:
        - 使用预训练 FinBERT 模型进行情感分析
        - 支持多文本聚合 (新闻 + 股吧帖子)
        - 自动降级到词典法
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        use_hf_model: bool = True,
    ):
        """
        Args:
            model_name: 模型名称
                - None: 自动选择 (英文用 FinBERT，中文用中文 BERT)
                - 自定义模型路径
            use_hf_model: 是否使用 HuggingFace 模型 (False 则纯词典)
        """
        self.model_name = model_name or 'uer/roberta-base-finetuned-jd-binary-chinese'
        self.use_hf_model = use_hf_model and HAS_TRANSFORMERS
        self.tokenizer = None
        self.model = None
        self._initialized = False

        if self.use_hf_model:
            self._init_model()

    def _init_model(self):
        """初始化模型"""
        try:
            logger.info(f"[FinBERT] 加载模型: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.eval()
            self._initialized = True
            logger.info("[FinBERT] 模型加载成功")
        except Exception as e:
            logger.error(f"[FinBERT] 模型加载失败: {e}，使用词典 fallback")
            self._initialized = False

    def analyze(self, text: str) -> Dict:
        """
        分析单条文本情感

        Args:
            text: 输入文本 (新闻标题/摘要/股吧帖子)

        Returns:
            {
                'label': 'positive' | 'neutral' | 'negative',
                'confidence': 0.0 ~ 1.0,
                'scores': {'positive': 0.3, 'neutral': 0.4, 'negative': 0.3},
                'score': -1.0 ~ 1.0 (正=看多, 负=看空),
            }
        """
        if not self._initialized or not text:
            return self._dict_fallback(text or '')

        try:
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
                num_classes = probs.size(-1)
                confidence = probs.max().item()
                label_idx = probs.argmax().item()

            # 适配 2 类 (negative/positive) 和 3 类 (negative/neutral/positive)
            if num_classes == 2:
                label_map = {0: 'negative', 1: 'positive'}
                neg_score = float(probs[0, 0].item())
                pos_score = float(probs[0, 1].item())
                scores = {
                    'positive': pos_score,
                    'neutral': 1.0 - abs(pos_score - neg_score),
                    'negative': neg_score,
                }
            else:
                label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
                scores = {
                    'positive': float(probs[0, 2].item()),
                    'neutral': float(probs[0, 1].item()),
                    'negative': float(probs[0, 0].item()),
                }

            return {
                'label': label_map.get(label_idx, 'neutral'),
                'confidence': confidence,
                'scores': scores,
                'score': scores['positive'] - scores['negative'],
            }

        except Exception as e:
            logger.error(f"[FinBERT] 分析错误: {e}")
            return self._dict_fallback(text)

    def analyze_batch(self, texts: List[str]) -> Dict:
        """
        批量分析文本情感

        Args:
            texts: 文本列表

        Returns:
            {
                'labels': [...],
                'confidences': [...],
                'scores': [...],
                'aggregate': {'score': ..., 'label': ..., 'consensus': ...},
            }
        """
        if not texts:
            return {'labels': [], 'confidences': [], 'scores': [],
                    'aggregate': {'score': 0.0, 'label': 'neutral', 'consensus': 0.0}}

        results = [self.analyze(t) for t in texts]
        scores = [r['score'] for r in results]
        confidences = [r['confidence'] for r in results]

        # 聚合: 加权平均 (按 confidence 加权)
        total_conf = sum(confidences)
        if total_conf > 0:
            weighted_score = sum(s * c for s, c in zip(scores, confidences)) / total_conf
        else:
            weighted_score = np.mean(scores) if scores else 0.0

        # 共识度: 同意比例 (label 相同)
        labels = [r['label'] for r in results]
        from collections import Counter
        most_common_label, most_common_count = Counter(labels).most_common(1)[0]
        consensus = most_common_count / len(labels)

        # 聚合标签
        if weighted_score > 0.15:
            aggregate_label = 'positive'
        elif weighted_score < -0.15:
            aggregate_label = 'negative'
        else:
            aggregate_label = 'neutral'

        return {
            'labels': labels,
            'confidences': confidences,
            'scores': scores,
            'aggregate': {
                'score': float(weighted_score),
                'label': aggregate_label,
                'consensus': float(consensus),
            },
        }

    @staticmethod
    def _dict_fallback(text: str) -> Dict:
        """
        词典 fallback — 当模型不可用时使用

        使用中文金融情感词典 + 程度副词 + 否定词
        """
        if not text:
            return {
                'label': 'neutral', 'confidence': 0.5,
                'scores': {'positive': 0.33, 'neutral': 0.34, 'negative': 0.33},
                'score': 0.0,
            }

        # 匹配正面/负面词
        pos_count = 0
        neg_count = 0
        degree_multiplier = 1.0
        negation_active = False

        # 分词 (简单按字符/词组匹配)
        for word, weight in CHINESE_FINANCIAL_SENTIMENT['degree'].items():
            if word in text:
                degree_multiplier = max(degree_multiplier, weight)

        for word in CHINESE_FINANCIAL_SENTIMENT['negation']:
            if word in text:
                negation_active = True
                break

        for word in CHINESE_FINANCIAL_SENTIMENT['positive']:
            if word in text:
                pos_count += 1

        for word in CHINESE_FINANCIAL_SENTIMENT['negative']:
            if word in text:
                neg_count += 1

        # 应用程度和否定
        if negation_active:
            pos_count, neg_count = neg_count, pos_count

        pos_count *= degree_multiplier
        neg_count *= degree_multiplier

        total = pos_count + neg_count
        if total > 0:
            score = (pos_count - neg_count) / total
        else:
            score = 0.0

        if score > 0.2:
            label = 'positive'
        elif score < -0.2:
            label = 'negative'
        else:
            label = 'neutral'

        return {
            'label': label,
            'confidence': min(0.7, abs(score) + 0.2),  # 词典法置信度较低
            'scores': {
                'positive': max(0, score),
                'neutral': 1.0 - abs(score),
                'negative': max(0, -score),
            },
            'score': float(score),
        }


# ── 全局实例 ──────────────────────────────────────────────────

sentiment_analyzer = FinBERTSentiment()
