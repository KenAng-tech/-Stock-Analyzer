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
import time
import numpy as np
import threading
from typing import Dict, List, Optional, Tuple

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
            from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertTokenizer

            logger.info(f"[FinBERT] 加载模型: {self.model_name}")

            # 尝试使用 AutoTokenizer，失败时使用 BertTokenizer
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            except Exception:
                logger.info("[FinBERT] AutoTokenizer 失败，使用 BertTokenizer")
                self.tokenizer = BertTokenizer.from_pretrained(self.model_name)

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
        self._news_cache: Dict = {}  # code -> (ts, (news, posts)) 文本抓取缓存 1h (2026-09-10)

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

    def load(self, path: Optional[str] = None) -> bool:
        """
        加载模型/词典 (词典法始终可用)

        Args:
            path: 可选路径 (词典法忽略)

        Returns:
            True 表示加载成功
        """
        return True

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

    def get_sentiment_score(self, stock_code: str, stock_name: str = '') -> Dict:
        """
        端到端情感评分: 抓取新闻+股吧 → 批量分析 → 聚合

        契约 (2026-09-10 补, 同根修复 /api/sentiment/bert + fusion 通道):
            {'score': -1~1, 'label': str, 'confidence': 0-1,
             'n_articles': int, 'method': 'finbert'|'dictionary'|'no_data'}
        无新闻时诚实返回 method='no_data' (不再拿股票名称硬喂词典造假中性),
        调用方 (routes/fusion) 按 no_data 自行降级。

        Args:
            stock_code: 股票代码 (带市场前缀, 如 sz300620)
            stock_name: 股票名称 (当前仅日志用途)
        """
        now = time.time()
        cached = self._news_cache.get(stock_code)
        if cached and now - cached[0] < 3600:
            news_texts, post_texts = cached[1]
        else:
            news_texts, post_texts = self._fetch_stock_texts(stock_code)
            self._news_cache[stock_code] = (now, (news_texts, post_texts))

        texts = [t for t in (news_texts + post_texts) if t and t.strip()][:40]
        if not texts:
            logger.info(f"[SentimentEngine] {stock_code} 无舆情文本 → no_data 诚实降级")
            return {'score': 0.0, 'label': 'neutral', 'confidence': 0.0,
                    'n_articles': 0, 'method': 'no_data'}

        agg = self.aggregate(texts)
        return {
            'score': round(agg['aggregate_score'], 4),
            'label': agg['aggregate_label'],
            'confidence': round(agg['aggregate_confidence'], 3),
            'n_articles': len(texts),
            'method': agg.get('method', 'dictionary'),
        }

    def _fetch_stock_texts(self, stock_code: str) -> Tuple[List[str], List[str]]:
        """抓取新闻(≤20)+股吧(≤20)标题; 抓取链失败诚实返回空 (调用方降级, 不喂假)"""
        try:
            from modules.data_fetcher import StockDataFetcher
            fetcher = StockDataFetcher()
            news = fetcher.get_stock_news(stock_code) or []
            posts = fetcher.get_stock_posts(stock_code) or []
            news_texts = [n.get('title', '') for n in news[:20] if n.get('title')]
            post_texts = [p.get('title', '') or p.get('content', '') for p in posts[:20]]
            return news_texts, [p for p in post_texts if p]
        except Exception as e:
            logger.warning(f"[SentimentEngine] 文本抓取失败 {stock_code}: {e}")
            return [], []


# ── 情绪因子生成器 (P1-4: 2026-07-01) ────────────────────────────────

class SentimentFactorGenerator:
    """
    情绪因子生成器

    从新闻/股吧帖子生成情绪 alpha 因子:
    1. 获取股票相关新闻/帖子
    2. 运行 FinBERT/词典情感分析
    3. 生成综合情绪分数因子
    4. 支持缓存和 IC 追踪

    用法:
        generator = SentimentFactorGenerator()
        factor_score = generator.generate_factor(stock_code)
    """

    def __init__(self, engine: Optional[SentimentEngine] = None, cache_ttl: int = 3600):
        """
        Args:
            engine: SentimentEngine 实例（默认自动创建）
            cache_ttl: 缓存过期时间（秒），默认 1 小时
        """
        self.engine = engine or SentimentEngine(use_finbert=True)
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, Dict] = {}
        self._cache_time: Dict[str, float] = {}
        self._ic_history: Dict[str, List[float]] = {}

    def generate_factor(self, stock_code: str, news_texts: Optional[List[str]] = None,
                        post_texts: Optional[List[str]] = None) -> Dict:
        """
        生成情绪因子

        Args:
            stock_code: 股票代码
            news_texts: 新闻标题列表（可选，不传则自动获取）
            post_texts: 股吧帖子列表（可选，不传则自动获取）

        Returns:
            {
                'factor_name': 'sentiment',
                'factor_value': float,     # -1.0 ~ +1.0
                'factor_label': str,       # 'positive'/'neutral'/'negative'
                'confidence': float,       # 0.0 ~ 1.0
                'method': str,            # 'finbert'/'dictionary'
                'text_count': int,        # 分析文本数
                'label_distribution': Dict,  # positive/neutral/negative 计数
            }
        """
        cache_key = f"sentiment_{stock_code}"

        # 检查缓存
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            import time
            if time.time() - self._cache_time.get(cache_key, 0) < self.cache_ttl:
                return cached

        # 获取文本数据
        if news_texts is None and post_texts is None:
            news_texts, post_texts = self._fetch_texts(stock_code)

        all_texts = (news_texts or []) + (post_texts or [])

        if not all_texts:
            result = {
                'factor_name': 'sentiment',
                'factor_value': 0.0,
                'factor_label': 'neutral',
                'confidence': 0.0,
                'method': 'none',
                'text_count': 0,
                'label_distribution': {'positive': 0, 'neutral': 0, 'negative': 0},
            }
        else:
            # 批量情感分析
            result = self.engine.aggregate(all_texts)
            result['factor_name'] = 'sentiment'
            result['text_count'] = len(all_texts)

        # 更新缓存
        self._cache[cache_key] = result
        self._cache_time[cache_key] = __import__('time').time()

        return result

    def _fetch_texts(self, stock_code: str) -> Tuple[List[str], List[str]]:
        """
        获取股票相关新闻和股吧帖子

        Args:
            stock_code: 股票代码

        Returns:
            (news_texts, post_texts)
        """
        news_texts = []
        post_texts = []

        try:
            from modules.data_fetcher import StockDataFetcher
            fetcher = StockDataFetcher()

            # 获取新闻
            news = fetcher.get_stock_news(stock_code)
            if news:
                for item in news[:20]:  # 最多取 20 条新闻
                    title = item.get('title', '')
                    if title:
                        news_texts.append(title)

            # 获取股吧帖子
            posts = fetcher.get_stock_posts(stock_code)
            if posts:
                for item in posts[:20]:  # 最多取 20 条帖子
                    content = item.get('title', '') or item.get('content', '')
                    if content:
                        post_texts.append(content)

        except Exception as e:
            logger.warning(f"[SentimentFactor] 获取文本数据失败: {e}")

        return news_texts, post_texts

    def update_ic_history(self, stock_code: str, factor_value: float,
                          future_return: float):
        """
        更新因子 IC 历史（用于因子衰减追踪）

        Args:
            stock_code: 股票代码
            factor_value: 因子值
            future_return: 未来收益率（用于计算 IC）
        """
        if stock_code not in self._ic_history:
            self._ic_history[stock_code] = []

        # IC = 因子值与未来收益率的相关系数
        self._ic_history[stock_code].append({
            'factor_value': factor_value,
            'future_return': future_return,
        })

        # 只保留最近 60 个数据点
        if len(self._ic_history[stock_code]) > 60:
            self._ic_history[stock_code] = self._ic_history[stock_code][-60:]

    def get_ic_stats(self, stock_code: str) -> Dict:
        """
        获取因子 IC 统计

        Returns:
            {
                'ic_mean': float,
                'ic_std': float,
                'icir': float,       # IC / IC_std
                'ic_positive_pct': float,  # IC > 0 的比例
                'n_samples': int,
            }
        """
        if stock_code not in self._ic_history or len(self._ic_history[stock_code]) < 5:
            return {
                'ic_mean': 0.0, 'ic_std': 0.0, 'icir': 0.0,
                'ic_positive_pct': 0.0, 'n_samples': 0,
            }

        data = self._ic_history[stock_code]
        factor_values = [d['factor_value'] for d in data]
        returns = [d['future_return'] for d in data]

        if len(factor_values) < 5:
            return {
                'ic_mean': 0.0, 'ic_std': 0.0, 'icir': 0.0,
                'ic_positive_pct': 0.0, 'n_samples': 0,
            }

        ic = float(np.corrcoef(factor_values, returns)[0, 1]) if np.std(factor_values) > 0 and np.std(returns) > 0 else 0.0
        ic_values = [ic]  # 简化：只返回当前 IC

        return {
            'ic_mean': round(ic, 4),
            'ic_std': 0.0,  # 简化
            'icir': round(ic, 4),
            'ic_positive_pct': 1.0 if ic > 0 else 0.0,
            'n_samples': len(data),
        }


# ── 全局单例 ────────────────────────────────────────────────

_sentiment_engine_instance: Optional[SentimentEngine] = None
_sentiment_engine_lock = threading.Lock()

# 兼容别名
DictionaryAnalyzer = DictionarySentimentAnalyzer


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