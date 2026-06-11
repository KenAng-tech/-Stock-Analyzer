#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
FinBERT 情感分析模块 — Enhanced Sentiment Analysis

支持两种模式:
1. FinBERT: 使用 transformers 库的金融 BERT (如果已安装)
2. Enhanced Dict: 增强词典法 (无需外部依赖)

功能:
- 单文本情感分析
- 批量新闻分析
- 时间衰减加权
- 来源权重调整
- 否定词和程度副词处理
"""

import re
import math
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from modules.logger import logger

# 尝试导入 transformers
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# 尝试导入 jieba
try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False


# ═══════════════════════════════════════════════════
# 增强情感词典
# ═══════════════════════════════════════════════════

POSITIVE_WORDS = {
    '上涨', '利好', '突破', '看好', '增持', '强势', '新高', '反弹',
    '放量', '金叉', '多头', '盈利', '增长', '超预期', '走强',
    '拉升', '涨停', '反转', '景气', '爆发', '推荐', '买入',
    '优秀', '优质', '成长', '加速', '量价齐升', '资金流入',
    '主力买入', '机构看好', '业绩预增', '订单饱满', '产能扩张',
    '技术突破', '国产替代', 'AI', '算力', '数据中心', '5G',
    '新能源', '储能', '芯片', '半导体', '人工智能', '自动驾驶',
    '盈利超预期', '营收增长', '利润增长', '毛利率提升',
    '机构加仓', '北向资金', '融资买入', '筹码集中', '估值修复',
    '底部放量', '突破压力', '主升浪', '戴维斯双击',
}

NEGATIVE_WORDS = {
    '下跌', '利空', '破位', '看空', '卖出', '减持', '新低', '回调',
    '缩量', '死叉', '空头', '亏损', '下滑', '不及预期', '走弱',
    '跌停', '顶部', '见顶', '风险', '监管', '处罚', '诉讼', '暴雷',
    '质押', '违约', '退市', 'ST', '放量下跌', '量价齐跌', '资金流出',
    '主力卖出', '机构减持', '业绩预减', '产能过剩', '竞争加剧',
    '贸易摩擦', '跌破', '走坏', '下行', '承压', '解禁压力',
    '商誉减值', '应收账款', '存货积压', '现金流恶化',
}

NEUTRAL_WORDS = {
    '震荡', '观望', '等待', '中性', '区间', '整理', '盘整',
    '横盘', '企稳', '修复', '分化', '轮动',
}

NEGATION_WORDS = {'不', '没', '非', '无', '未', '勿', '别', '否', '缺乏', '不足', '未能', '难以'}

DEGREE_WORDS = {
    '极其': 2.0, '非常': 1.8, '很': 1.5, '十分': 1.5, '特别': 1.5,
    '较为': 0.8, '比较': 0.8, '略': 0.6, '稍微': 0.6, '轻微': 0.5,
}

CONTRAST_WORDS = {'但是', '然而', '不过', '可是', '却', '反而'}


class SentimentModelSelector:
    """情感模型选择器"""

    @staticmethod
    def detect_available() -> str:
        """检测可用模型"""
        if TRANSFORMERS_AVAILABLE:
            return 'finbert'
        return 'enhanced_dict'

    @staticmethod
    def load_model(strategy: str = None):
        """加载模型"""
        if strategy is None:
            strategy = SentimentModelSelector.detect_available()

        if strategy == 'finbert' and TRANSFORMERS_AVAILABLE:
            return FinBERTAnalyzer()
        return EnhancedDictAnalyzer()


class FinBERTAnalyzer:
    """FinBERT 情感分析 (需要 transformers)"""

    def __init__(self, model_name: str = 'shibing624/bert-base-chinese-finetune-financial'):
        self.model_name = model_name
        self._model = None
        self._tokenizer = None
        self._load_model()

    def _load_model(self):
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            logger.info(f"[FinBERT] 模型已加载: {self.model_name}")
        except Exception as e:
            logger.error(f"[FinBERT] 模型加载失败: {e}, 回退到词典法")
            self._model = None

    def analyze(self, text: str) -> Dict:
        """分析单条文本"""
        if self._model is None:
            return EnhancedDictAnalyzer().analyze(text)

        try:
            inputs = self._tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self._model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            scores = probs.tolist()[0]

            return {
                'positive': round(scores[-1] if len(scores) > 1 else 0.5, 4),
                'negative': round(scores[0] if len(scores) > 1 else 0.5, 4),
                'neutral': round(1 - scores[-1] - scores[0] if len(scores) > 1 else 0.0, 4),
                'label': 'positive' if scores[-1] > scores[0] else 'negative',
                'confidence': round(max(scores), 4),
            }
        except Exception as e:
            logger.error(f"[FinBERT] 分析失败: {e}")
            return EnhancedDictAnalyzer().analyze(text)

    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        """批量分析"""
        return [self.analyze(t) for t in texts]

    def analyze_stock_news(self, news_list: List[Dict]) -> Dict:
        """分析股票新闻集合"""
        if not news_list:
            return {'score': 0.0, 'label': 'neutral', 'n_news': 0}

        weighted_scores = []
        total_weight = 0

        for news in news_list:
            text = news.get('title', news.get('content', ''))
            days_ago = news.get('days_ago', 0)
            source = news.get('source', 'unknown')

            # 时间衰减
            time_weight = math.exp(-0.1 * days_ago)
            # 来源权重
            source_weight = {'新闻': 1.0, '股吧': 0.7, '论坛': 0.5}.get(source, 0.6)
            weight = time_weight * source_weight

            result = self.analyze(text)
            score = result['positive'] - result['negative']

            weighted_scores.append(score * weight)
            total_weight += weight

        if total_weight > 0:
            score = sum(weighted_scores) / total_weight
        else:
            score = 0.0

        label = 'positive' if score > 0.1 else 'negative' if score < -0.1 else 'neutral'

        return {
            'score': round(float(score), 4),
            'label': label,
            'n_news': len(news_list),
            'weighted_positive': round(max(0, score), 4),
            'weighted_negative': round(min(0, -score), 4),
        }


class EnhancedDictAnalyzer:
    """增强词典法情感分析 (无需外部依赖)"""

    def analyze(self, text: str) -> Dict:
        """分析单条文本"""
        if not text:
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0, 'label': 'neutral', 'score': 0.0}

        # 分词
        words = self._tokenize(text)

        positive_score = 0.0
        negative_score = 0.0
        n_pos = 0
        n_neg = 0

        i = 0
        while i < len(words):
            word = words[i]

            # 检查程度副词
            degree = 1.0
            if i > 0 and words[i-1] in DEGREE_WORDS:
                degree = DEGREE_WORDS[words[i-1]]

            # 检查否定词 (前3个字内)
            negation = 1.0
            for j in range(max(0, i-3), i):
                if words[j] in NEGATION_WORDS:
                    negation = -1.0
                    break

            # 检查对比词
            contrast = 1.0
            for j in range(max(0, i-5), i):
                if words[j] in CONTRAST_WORDS:
                    contrast = -0.5  # 对比词后内容权重减半并反向
                    break

            # 匹配情感词
            if word in POSITIVE_WORDS:
                positive_score += degree * negation * contrast
                n_pos += 1
            elif word in NEGATIVE_WORDS:
                negative_score += degree * negation * contrast
                n_neg += 1

            i += 1

        total = positive_score + negative_score
        if total == 0:
            return {'positive': 0.5, 'negative': 0.5, 'neutral': 0.0, 'label': 'neutral', 'score': 0.0}

        normalized_pos = positive_score / total
        normalized_neg = negative_score / total

        score = normalized_pos - normalized_neg
        label = 'positive' if score > 0.1 else 'negative' if score < -0.1 else 'neutral'

        return {
            'positive': round(normalized_pos, 4),
            'negative': round(normalized_neg, 4),
            'neutral': round(1 - normalized_pos - normalized_neg, 4),
            'label': label,
            'score': round(score, 4),
        }

    def _tokenize(self, text: str) -> List[str]:
        """分词"""
        if JIEBA_AVAILABLE:
            return list(jieba.cut(text))
        # 简单分割: 中文按字符, 英文按空格
        result = []
        current = ''
        for char in text:
            if '一' <= char <= '鿿':
                if current:
                    result.append(current)
                    current = ''
                result.append(char)
            else:
                current += char
        if current:
            result.append(current)
        return result

    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        """批量分析"""
        return [self.analyze(t) for t in texts]

    def analyze_stock_news(self, news_list: List[Dict]) -> Dict:
        """分析股票新闻集合"""
        if not news_list:
            return {'score': 0.0, 'label': 'neutral', 'n_news': 0}

        weighted_scores = []
        total_weight = 0

        for news in news_list:
            text = news.get('title', news.get('content', ''))
            days_ago = news.get('days_ago', 0)
            source = news.get('source', 'unknown')

            # 时间衰减
            time_weight = math.exp(-0.1 * days_ago)
            # 来源权重
            source_weight = {'新闻': 1.0, '股吧': 0.7, '论坛': 0.5}.get(source, 0.6)
            weight = time_weight * source_weight

            result = self.analyze(text)
            score = result['score']

            weighted_scores.append(score * weight)
            total_weight += weight

        if total_weight > 0:
            score = sum(weighted_scores) / total_weight
        else:
            score = 0.0

        # 去重: 相似内容只计一次 (简化: 按长度过滤)
        unique_news = [n for n in news_list if len(n.get('title', '')) > 5]

        label = 'positive' if score > 0.1 else 'negative' if score < -0.1 else 'neutral'

        return {
            'score': round(float(score), 4),
            'label': label,
            'n_news': len(unique_news),
            'weighted_positive': round(max(0, score), 4),
            'weighted_negative': round(min(0, -score), 4),
        }


class SentimentEnsemble:
    """情感分析集成"""

    def __init__(self):
        self._analyzers = []

    def add_analyzer(self, name: str, analyzer, weight: float = 1.0):
        """添加分析器"""
        self._analyzers.append((name, analyzer, weight))

    def analyze(self, text: str) -> Dict:
        """集成分析"""
        results = []
        total_weight = 0

        for name, analyzer, weight in self._analyzers:
            try:
                result = analyzer.analyze(text)
                results.append((result, weight))
                total_weight += weight
            except Exception as e:
                logger.error(f"[SentimentEnsemble] {name} 分析失败: {e}")

        if not results:
            return {'score': 0.0, 'label': 'neutral'}

        weighted_score = sum(r[0]['score'] * w for r, w in results) / total_weight
        label = 'positive' if weighted_score > 0.1 else 'negative' if weighted_score < -0.1 else 'neutral'

        return {
            'score': round(weighted_score, 4),
            'label': label,
            'n_analyzers': len(results),
        }

    def analyze_stock_news(self, news_list: List[Dict]) -> Dict:
        """集成新闻分析"""
        results = []
        total_weight = 0

        for name, analyzer, weight in self._analyzers:
            try:
                result = analyzer.analyze_stock_news(news_list)
                results.append((result, weight))
                total_weight += weight
            except Exception as e:
                logger.error(f"[SentimentEnsemble] {name} 新闻分析失败: {e}")

        if not results:
            return {'score': 0.0, 'label': 'neutral', 'n_news': 0}

        weighted_score = sum(r[0]['score'] * w for r, w in results) / total_weight
        label = 'positive' if weighted_score > 0.1 else 'negative' if weighted_score < -0.1 else 'neutral'

        return {
            'score': round(weighted_score, 4),
            'label': label,
            'n_news': news_list[0].get('n_news', len(news_list)) if news_list else 0,
        }


# 全局实例
sentiment_analyzer = EnhancedDictAnalyzer()
sentiment_ensemble = SentimentEnsemble()
sentiment_ensemble.add_analyzer('dict', EnhancedDictAnalyzer(), weight=1.0)
