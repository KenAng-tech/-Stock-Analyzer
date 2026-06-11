#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
深度学习 NLP 情感分析 — FinBERT + LLM 集成

升级内容:
1. FinBERT — 金融领域 BERT 模型
2. LLM 集成 — 支持多种大语言模型 API
3. 集成学习 — 多模型投票/加权

架构:
    Input Text
        ↓
    ┌─────────────────────────────────────┐
    │  Text Preprocessing (Tokenizer)     │
    └─────────────────────────────────────┘
        ↓
    ┌──────────┐   ┌──────────┐   ┌──────────┐
    │ FinBERT  │   │ RoBERTa  │   │  LLM API │  ← 多模型并行
    └──────────┘   └──────────┘   └──────────┘
        ↓              ↓              ↓
    ┌─────────────────────────────────────┐
    │     Ensemble Aggregation            │
    └─────────────────────────────────────┘
        ↓
    Sentiment Score (-1 ~ +1) + Confidence

支持模型:
- shibing624/bert-base-chinese-finetune-financial (FinBERT)
- hfl/chinese-roberta-wwm-ext (中文 RoBERTa)
- API: ChatGLM, Qwen, Baichuan 等
"""

import re
import math
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import os
import pickle

from modules.logger import logger


# ── 依赖检测 ──────────────────────────────────────────────

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("[SentimentBERT] transformers 未安装")

try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


# ── 金融情感词典 (增强版) ─────────────────────────────────

FINANCIAL_POSITIVE = {
    # 价格相关
    '上涨', '拉升', '冲高', '跳空', '突破', '创新高', '涨停', '逼空',
    '反弹', '反转', '企稳', '回暖', '走强', '放量上涨', '量价齐升',
    # 利好消息
    '利好', '增持', '推荐', '买入', '目标价上调', '评级上调', '入围',
    '中标', '签约', '合作', '授权', '专利', '认证', '获奖',
    # 业绩相关
    '盈利', '增长', '预增', '超预期', '亮眼的', '创纪录', '历史新高',
    '翻倍', '暴增', '井喷', '爆发', '飙升', '大增', '跃升',
    # 资金流向
    '资金流入', '主力买入', '机构加仓', '北向资金', '融资买入',
    '大单买入', '筹码集中', '抢筹', '吸筹',
    # 技术形态
    '金叉', '多头排列', '底部放量', '突破压力', '主升浪', '戴维斯双击',
    '估值修复', '低估值', '安全边际',
    # 行业热点
    'AI', '人工智能', '算力', '数据中心', '5G', '6G', '新能源',
    '储能', '光伏', '芯片', '半导体', '国产替代', '自主可控',
    '自动驾驶', '机器人', '元宇宙', '区块链', '数字经济',
    # 财务指标
    '毛利率提升', '净利率改善', 'ROE 提升', '现金流改善', '负债降低',
    '营收增长', '利润增长', '订单饱满', '产能扩张',
}

FINANCIAL_NEGATIVE = {
    # 价格相关
    '下跌', '跳水', '破位', '杀跌', '跌停', '杀空', '回调', '走弱',
    '破位下行', '放量下跌', '量价齐跌', '阴跌', '暴跌', '重挫',
    # 利空消息
    '利空', '减持', '卖出', '调降', '评级下调', '立案', '调查',
    '处罚', '诉讼', '仲裁', '违约', '暴雷', '退市', 'ST', '*ST',
    # 业绩相关
    '亏损', '下滑', '预减', '不及预期', '断崖', '腰斩', '暴跌',
    '萎缩', '下降', '减少', '恶化', '跳水', '崩盘', '溃败',
    # 资金流向
    '资金流出', '主力卖出', '机构减持', '北向流出', '融资卖出',
    '大单卖出', '筹码松动', '出货', '派发',
    # 技术形态
    '死叉', '空头排列', '顶部放量', '跌破支撑', '断头铡刀',
    '估值泡沫', '高估值', '风险累积',
    # 财务指标
    '毛利率下降', '净利率下滑', 'ROE 下降', '现金流恶化', '负债增加',
    '营收下滑', '利润下滑', '订单减少', '产能过剩',
    # 其他风险
    '解禁', '质押', '平仓', '强平', '商誉减值', '应收账款',
    '存货积压', '客户流失', '竞争加剧', '贸易摩擦', '制裁',
}

FINANCIAL_NEUTRAL = {
    '震荡', '观望', '等待', '中性', '区间', '整理', '盘整', '横盘',
    '企稳', '修复', '分化', '轮动', '波动', '窄幅', '僵持',
}

NEGATION_WORDS = {'不', '没', '非', '无', '未', '勿', '别', '否', '缺乏', '不足', '未能', '难以', '拒绝'}

DEGREE_WORDS = {
    '极其': 2.0, '非常': 1.8, '特别': 1.8, '十分': 1.5, '很': 1.5,
    '较为': 0.8, '比较': 0.8, '略': 0.6, '稍微': 0.6, '轻微': 0.5,
    '大幅': 1.6, '显著': 1.5, '明显': 1.3, '小幅': 0.7, '温和': 0.8,
}

CONTRAST_WORDS = {'但是', '然而', '不过', '可是', '却', '反而', '尽管', '虽然'}


# ── 基础分词器 ──────────────────────────────────────────────

class ChineseTokenizer:
    """中文分词器 (支持 jieba 和 fallback)"""

    def __init__(self):
        self.use_jieba = JIEBA_AVAILABLE

    def tokenize(self, text: str) -> List[str]:
        """分词"""
        if not text:
            return []

        # 清理文本
        text = re.sub(r'[^\w\s一 - 鿿]', ' ', text).strip()
        if not text:
            return []

        if self.use_jieba:
            return list(jieba.cut(text))

        # Fallback: 简单分割
        return self._simple_tokenize(text)

    def _simple_tokenize(self, text: str) -> List[str]:
        """简单分词 (按字符和空格)"""
        result = []
        current = ''

        for char in text:
            if '一' <= char <= '鿿':  # 中文
                if current:
                    result.append(current)
                    current = ''
                result.append(char)
            elif char.isspace():
                if current:
                    result.append(current)
                    current = ''
            else:  # 英文/数字
                current += char

        if current:
            result.append(current)

        return [w for w in result if w.strip()]


# ── 词典法分析器 (增强版) ──────────────────────────────────

class DictionaryAnalyzer:
    """基于词典的情感分析器"""

    def __init__(self):
        self.tokenizer = ChineseTokenizer()

    def analyze(self, text: str) -> Dict:
        """分析单条文本"""
        if not text or len(text.strip()) < 2:
            return self._neutral_result()

        words = self.tokenizer.tokenize(text)
        if not words:
            return self._neutral_result()

        pos_score = 0.0
        neg_score = 0.0
        pos_count = 0
        neg_count = 0

        # 对比词标记 (对比词后的内容权重降低)
        contrast_mode = False

        for i, word in enumerate(words):
            # 检查对比词
            if word in CONTRAST_WORDS:
                contrast_mode = True
                continue

            # 检查程度副词
            degree = 1.0
            if i > 0 and words[i - 1] in DEGREE_WORDS:
                degree = DEGREE_WORDS[words[i - 1]]

            # 检查否定词 (前 3 个词内)
            negation = 1.0
            for j in range(max(0, i - 3), i):
                if words[j] in NEGATION_WORDS:
                    negation = -1.0
                    break

            # 匹配情感词
            if word in FINANCIAL_POSITIVE:
                contribution = degree * negation
                if contrast_mode:
                    contribution *= 0.5  # 对比词后减半

                pos_score += max(0, contribution)
                neg_score += min(0, contribution)
                pos_count += 1

            elif word in FINANCIAL_NEGATIVE:
                contribution = degree * negation
                if contrast_mode:
                    contribution *= 0.5

                neg_score += max(0, contribution)
                pos_score += min(0, contribution)
                neg_count += 1

        # 重置对比模式 (每遇到一个对比词只影响后续一段)
        contrast_mode = False

        # 计算最终分数
        total_abs = abs(pos_score) + abs(neg_score)
        if total_abs < 0.01:
            return self._neutral_result()

        normalized_pos = abs(pos_score) / total_abs
        normalized_neg = abs(neg_score) / total_abs

        score = normalized_pos - normalized_neg
        label = self._score_to_label(score)

        return {
            'positive': round(normalized_pos, 4),
            'negative': round(normalized_neg, 4),
            'neutral': round(1 - normalized_pos - normalized_neg, 4),
            'score': round(score, 4),
            'label': label,
            'n_positive_words': pos_count,
            'n_negative_words': neg_count,
        }

    def _neutral_result(self) -> Dict:
        return {
            'positive': 0.33,
            'negative': 0.33,
            'neutral': 0.34,
            'score': 0.0,
            'label': 'neutral',
            'n_positive_words': 0,
            'n_negative_words': 0,
        }

    @staticmethod
    def _score_to_label(score: float) -> str:
        if score > 0.15:
            return 'positive'
        elif score < -0.15:
            return 'negative'
        return 'neutral'


# ── FinBERT 分析器 ─────────────────────────────────────────

class FinBERTAnalyzer:
    """
    FinBERT 情感分析器

    使用金融领域预训练的 BERT 模型
    """

    def __init__(self, model_name: str = 'shibing624/bert-base-chinese-finetune-financial'):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = None

        if TRANSFORMERS_AVAILABLE:
            self._load_model()
        else:
            logger.warning("[FinBERT] transformers 未安装，无法加载模型")

    def _load_model(self):
        """加载 FinBERT 模型"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"[FinBERT] 模型已加载：{self.model_name}, device={self.device}")
        except Exception as e:
            logger.error(f"[FinBERT] 模型加载失败：{e}")
            self.model = None

    def analyze(self, text: str) -> Dict:
        """分析单条文本"""
        if self.model is None:
            # Fallback 到词典法
            return DictionaryAnalyzer().analyze(text)

        try:
            # 编码
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                max_length=512,
                padding=True
            )

            # 推理
            with torch.no_grad():
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)

            # 解析结果
            if len(probs[0]) == 3:
                # [negative, neutral, positive]
                neg_prob = probs[0][0].item()
                neu_prob = probs[0][1].item()
                pos_prob = probs[0][2].item()
            else:
                # [negative, positive]
                neg_prob = probs[0][0].item()
                pos_prob = probs[0][1].item()
                neu_prob = 1 - neg_prob - pos_prob

            score = pos_prob - neg_prob
            label = 'positive' if pos_prob > neg_prob and pos_prob > neu_prob else \
                    'negative' if neg_prob > pos_prob and neg_prob > neu_prob else 'neutral'

            return {
                'positive': round(pos_prob, 4),
                'negative': round(neg_prob, 4),
                'neutral': round(neu_prob, 4),
                'score': round(score, 4),
                'label': label,
                'confidence': round(max(pos_prob, neg_prob, neu_prob), 4),
                'model': 'finbert',
            }

        except Exception as e:
            logger.error(f"[FinBERT] 分析失败：{e}")
            return DictionaryAnalyzer().analyze(text)

    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        """批量分析"""
        if not texts:
            return []

        if self.model is None:
            return [DictionaryAnalyzer().analyze(t) for t in texts]

        try:
            # 批量编码
            inputs = self.tokenizer(
                texts,
                return_tensors='pt',
                truncation=True,
                max_length=512,
                padding=True
            )

            # 批量推理
            with torch.no_grad():
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)

            results = []
            for i, text in enumerate(texts):
                if len(probs[i]) == 3:
                    neg_prob = probs[i][0].item()
                    neu_prob = probs[i][1].item()
                    pos_prob = probs[i][2].item()
                else:
                    neg_prob = probs[i][0].item()
                    pos_prob = probs[i][1].item()
                    neu_prob = 1 - neg_prob - pos_prob

                score = pos_prob - neg_prob
                label = 'positive' if pos_prob > neg_prob and pos_prob > neu_prob else \
                        'negative' if neg_prob > pos_prob and neg_prob > neu_prob else 'neutral'

                results.append({
                    'positive': round(pos_prob, 4),
                    'negative': round(neg_prob, 4),
                    'neutral': round(neu_prob, 4),
                    'score': round(score, 4),
                    'label': label,
                    'confidence': round(max(pos_prob, neg_prob, neu_prob), 4),
                    'model': 'finbert',
                })

            return results

        except Exception as e:
            logger.error(f"[FinBERT] 批量分析失败：{e}")
            return [DictionaryAnalyzer().analyze(t) for t in texts]


# ── LLM 情感分析器 ─────────────────────────────────────────

class LLMAnalyzer:
    """
    LLM 情感分析器

    支持多种 LLM API:
    - OpenAI API 兼容接口
    - ChatGLM
    - Qwen
    - 本地模型
    """

    def __init__(self, api_url: str = None, api_key: str = None,
                 model_name: str = 'gpt-3.5-turbo', temperature: float = 0.1):
        self.api_url = api_url
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature

    def analyze(self, text: str) -> Dict:
        """
        使用 LLM 分析情感

        Prompt 设计:
        "请分析以下金融文本的情感倾向，只返回 positive/negative/neutral 和 0-1 之间的置信度分数。
         文本：{text}"
        """
        if not REQUESTS_AVAILABLE:
            return DictionaryAnalyzer().analyze(text)

        prompt = f"""请分析以下金融新闻/评论的情感倾向。
只返回 JSON 格式：{{"label": "positive/negative/neutral", "confidence": 0-1 之间的分数}}

文本：{text[:500]}"""  # 限制长度

        try:
            headers = {'Content-Type': 'application/json'}
            if self.api_key:
                headers['Authorization'] = f'Bearer {self.api_key}'

            payload = {
                'model': self.model_name,
                'messages': [{'role': 'user', 'content': prompt}],
                'temperature': self.temperature,
                'max_tokens': 100,
            }

            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                content = result.get('choices', [{}])[0].get('message', {}).get('content', '')

                # 解析 JSON 响应
                try:
                    # 提取 JSON 部分
                    json_match = re.search(r'\{[^}]+\}', content)
                    if json_match:
                        llm_result = json.loads(json_match.group())
                        label = llm_result.get('label', 'neutral')
                        confidence = llm_result.get('confidence', 0.5)

                        if label == 'positive':
                            score = confidence - (1 - confidence) * 0.3
                        elif label == 'negative':
                            score = -(confidence - (1 - confidence) * 0.3)
                        else:
                            score = 0

                        return {
                            'positive': confidence if label == 'positive' else (1 - confidence) / 2,
                            'negative': confidence if label == 'negative' else (1 - confidence) / 2,
                            'neutral': 1 - confidence,
                            'score': round(score, 4),
                            'label': label,
                            'confidence': round(confidence, 4),
                            'model': self.model_name,
                        }
                except json.JSONDecodeError:
                    pass

        except Exception as e:
            logger.error(f"[LLMAnalyzer] 分析失败：{e}")

        return DictionaryAnalyzer().analyze(text)


# ── 集成情感分析器 ─────────────────────────────────────────

class SentimentEnsembleV2:
    """
    情感分析集成 V2

    整合:
    1. FinBERT (金融 BERT)
    2. LLM (大语言模型)
    3. Dictionary (词典法)

    输出：加权平均 + 置信度
    """

    def __init__(self):
        self.analyzers = []
        self.weights = []

        # 默认添加词典法 (总是可用)
        self.add_analyzer('dictionary', DictionaryAnalyzer(), weight=1.0)

        # 尝试添加 FinBERT
        try:
            finbert = FinBERTAnalyzer()
            if finbert.model is not None:
                self.add_analyzer('finbert', finbert, weight=2.0)
        except Exception as e:
            logger.warning(f"[SentimentEnsembleV2] FinBERT 添加失败：{e}")

        # 可选：添加 LLM
        # self.add_analyzer('llm', LLMAnalyzer(...), weight=1.5)

    def add_analyzer(self, name: str, analyzer, weight: float = 1.0):
        """添加分析器"""
        self.analyzers.append((name, analyzer))
        self.weights.append(weight)

    def analyze(self, text: str) -> Dict:
        """集成分析单条文本"""
        if not text:
            return {
                'score': 0.0,
                'label': 'neutral',
                'confidence': 0.5,
                'n_models': 0,
            }

        results = []
        total_weight = 0

        for (name, analyzer), weight in zip(self.analyzers, self.weights):
            try:
                result = analyzer.analyze(text)
                results.append((result, weight, name))
                total_weight += weight
            except Exception as e:
                logger.debug(f"[SentimentEnsembleV2] {name} 分析失败：{e}")

        if not results:
            return {
                'score': 0.0,
                'label': 'neutral',
                'confidence': 0.5,
                'n_models': 0,
            }

        # 加权平均
        weighted_score = sum(r[0]['score'] * w for r, w, _ in results) / total_weight
        weighted_positive = sum(r[0]['positive'] * w for r, w, _ in results) / total_weight
        weighted_negative = sum(r[0]['negative'] * w for r, w, _ in results) / total_weight
        weighted_neutral = sum(r[0]['neutral'] * w for r, w, _ in results) / total_weight

        # 平均置信度 (仅考虑高置信度模型)
        confidences = [r[0].get('confidence', 0.5) for r, w, _ in results if w >= 1.0]
        avg_confidence = np.mean(confidences) if confidences else 0.5

        label = 'positive' if weighted_score > 0.15 else 'negative' if weighted_score < -0.15 else 'neutral'

        return {
            'score': round(weighted_score, 4),
            'label': label,
            'positive': round(weighted_positive, 4),
            'negative': round(weighted_negative, 4),
            'neutral': round(weighted_neutral, 4),
            'confidence': round(avg_confidence, 4),
            'n_models': len(results),
            'model_results': {name: {'score': r[0]['score'], 'label': r[0]['label']}
                             for r, _, name in results},
        }

    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        """批量分析"""
        return [self.analyze(t) for t in texts]

    def analyze_stock_news(self, news_list: List[Dict],
                           time_decay: bool = True) -> Dict:
        """
        分析股票新闻集合

        Args:
            news_list: 新闻列表，每项包含 title/content, days_ago, source
            time_decay: 是否应用时间衰减

        Returns:
            综合情感分析结果
        """
        if not news_list:
            return {
                'score': 0.0,
                'label': 'neutral',
                'n_news': 0,
            }

        weighted_scores = []
        total_weight = 0

        for news in news_list:
            text = news.get('title', news.get('content', ''))
            if not text:
                continue

            days_ago = news.get('days_ago', 0)
            source = news.get('source', 'news')

            # 基础权重
            weight = 1.0

            # 时间衰减
            if time_decay:
                weight *= math.exp(-0.1 * days_ago)

            # 来源权重
            source_weights = {
                'news': 1.0,      # 新闻
                'guba': 0.7,      # 股吧
                'forum': 0.5,     # 论坛
                'twitter': 0.6,   # Twitter
            }
            weight *= source_weights.get(source, 0.6)

            # 分析
            result = self.analyze(text)
            weighted_scores.append(result['score'] * weight)
            total_weight += weight

        if total_weight < 0.01:
            return {
                'score': 0.0,
                'label': 'neutral',
                'n_news': 0,
            }

        # 加权平均
        avg_score = sum(weighted_scores) / total_weight

        # 去重统计 (简化：按标题长度过滤)
        unique_count = sum(1 for n in news_list if len(n.get('title', '')) > 5)

        label = 'positive' if avg_score > 0.15 else 'negative' if avg_score < -0.15 else 'neutral'

        return {
            'score': round(avg_score, 4),
            'label': label,
            'n_news': unique_count,
            'weighted_positive': round(max(0, avg_score), 4),
            'weighted_negative': round(min(0, -avg_score), 4),
        }


# ── 情感分析 API 封装 ──────────────────────────────────────

class SentimentAPI:
    """
    情感分析 API 封装

    提供统一接口，支持:
    - 单文本分析
    - 批量分析
    - 股票新闻分析
    """

    def __init__(self):
        self.ensemble = SentimentEnsembleV2()
        self.cache = {}

    def analyze(self, text: str, use_cache: bool = True) -> Dict:
        """分析单条文本"""
        if use_cache:
            cache_key = hash(text)
            if cache_key in self.cache:
                return self.cache[cache_key]

        result = self.ensemble.analyze(text)

        if use_cache:
            self.cache[cache_key] = result

        return result

    def analyze_news(self, stock_code: str, news_list: List[Dict]) -> Dict:
        """分析股票新闻"""
        return self.ensemble.analyze_stock_news(news_list)


# 全局实例
dictionary_analyzer = DictionaryAnalyzer()
finbert_analyzer = FinBERTAnalyzer()
sentiment_ensemble = SentimentEnsembleV2()
sentiment_api = SentimentAPI()

# NumPy 导入 (用于均值计算)
try:
    import numpy as np
except ImportError:
    def np_mean(x):
        return sum(x) / len(x) if x else 0.0
    np = type('np', (), {'mean': np_mean})()
