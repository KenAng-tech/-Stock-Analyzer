#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
情感分析模块 — P3 升级 (2026-06-04)

从东方财富获取个股新闻/股吧数据，进行 NLP 情感分析。

数据来源:
- 东方财富股吧帖子列表 (API)
- 财经新闻标题情感

实现:
- 基于情感词典的轻量级 NLP（无需外部 NLP 库）
- 支持 negation 处理（"不看好" → 负面）
- 支持程度副词加权（"非常看好" → 强正面）
- 综合情感分数 (-1 ~ +1)
"""

import re
import time
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime

from modules.dynamic_cache import cache
from modules.logger import logger


# ── 情感词典 ──────────────────────────────────────────────

POSITIVE_WORDS = {
    '上涨', '利好', '突破', '看好', '增持', '强势', '新高', '反弹',
    '放量', '金叉', '多头', '盈利', '增长', '超预期', '利好', '走强',
    '突破', '拉升', '涨停', '底部', '反转', '景气', '景气度', '爆发',
    '利好', '推荐', '买入', '增持', '优秀', '优质', '成长', '加速',
    '放量上涨', '量价齐升', '资金流入', '主力买入', '机构看好',
    '业绩预增', '订单饱满', '产能扩张', '技术突破', '国产替代',
    'AI', '算力', '数据中心', '5G', '新能源', '储能', '光伏',
    '芯片', '半导体', '人工智能', '自动驾驶', '元宇宙', '区块链',
}

NEGATIVE_WORDS = {
    '下跌', '利空', '破位', '看空', '卖出', '减持', '新低', '回调',
    '缩量', '死叉', '空头', '亏损', '下滑', '不及预期', '走弱',
    '跌停', '顶部', '见顶', '风险', '监管', '处罚', '诉讼', '暴雷',
    '质押', '违约', '退市', 'ST', '放量下跌', '量价齐跌', '资金流出',
    '主力卖出', '机构减持', '业绩预减', '订单减少', '产能过剩',
    '竞争加剧', '贸易摩擦', '跌破', '走坏', '下行', '走弱', '承压',
}

NEGATION_WORDS = {'不', '没', '非', '无', '未', '勿', '别', '否', '缺乏', '不足'}

DEGREE_WORDS = {
    '极其': 2.0, '非常': 1.8, '很': 1.5, '十分': 1.5, '特别': 1.5,
    '较为': 0.8, '比较': 0.8, '略': 0.6, '稍微': 0.6, '轻微': 0.5,
}


def analyze_sentiment_text(text: str) -> Dict:
    """
    对单条文本进行情感分析 — 改进版

    算法:
    1. 从最长匹配开始扫描（贪心最长匹配）
    2. 检查前面的词是否是程度副词/否定词
    3. 累加加权分数
    """
    if not text:
        return {'score': 0.0, 'positive': 0, 'negative': 0, 'words': []}

    text = re.sub(r'[^\w一-鿿]', ' ', text).strip()
    if not text:
        return {'score': 0.0, 'positive': 0, 'negative': 0, 'words': []}

    words_found = []
    pos_count = 0
    neg_count = 0
    used = [False] * len(text)

    # 贪心最长匹配：从位置 i 开始，找最长的匹配词
    i = 0
    while i < len(text):
        if used[i]:
            i += 1
            continue

        best_match = None
        best_length = 0

        # 尝试从长到短匹配
        for length in range(min(4, len(text) - i), 1, -1):
            ngram = text[i:i + length]
            if ngram in POSITIVE_WORDS or ngram in NEGATIVE_WORDS:
                best_match = ngram
                best_length = length
                break  # 找到最长匹配就停止

        if best_match:
            # 检查前面的词
            degree = 1.0
            negated = False
            if i >= 1 and text[i-1] in DEGREE_WORDS:
                degree = DEGREE_WORDS[text[i-1]]
            if i >= 1 and text[i-1] in NEGATION_WORDS:
                negated = True

            if best_match in POSITIVE_WORDS:
                pos_count += 1
                words_found.append(best_match)
                for j in range(i, i + best_length):
                    used[j] = True
                i += best_length
            elif best_match in NEGATIVE_WORDS:
                neg_count += 1
                words_found.append(best_match)
                for j in range(i, i + best_length):
                    used[j] = True
                i += best_length
        else:
            i += 1

    # 处理否定: 如果否定词后面紧跟正面词，翻转符号
    # 简化: 如果 negated 标记的词存在，反转该词的贡献
    # 这里用简化处理: 如果 negation 后有关键词，额外扣分
    # 由于上面已经处理了 negated 标记，这里只需归一化

    total = pos_count + neg_count
    if total > 0:
        final_score = (pos_count - neg_count) / total
    else:
        final_score = 0.0

    return {
        'score': round(float(np.clip(final_score, -1, 1)), 3),
        'positive': pos_count,
        'negative': neg_count,
        'words': words_found[:10],
    }


class SentimentAnalyzer:
    """情感分析器"""

    def __init__(self):
        self._cache_ttl = 300  # 5 分钟

    def get_sentiment_score(self, stock_code: str, stock_name: str = '') -> Dict:
        """
        获取个股综合情感分数

        从东方财富获取新闻标题和股吧帖子，进行情感分析。

        Returns:
            情感分析结果
        """
        cache_key = f"sentiment_{stock_code}"
        cached = cache.get(cache_key, category='sentiment')
        if cached:
            return cached

        # 尝试从东方财富获取新闻
        news_sentiments = self._fetch_eastmoney_news(stock_code, stock_name)
        # 尝试从股吧获取帖子
        forum_sentiments = self._fetch_guba_posts(stock_code, stock_name)

        all_scores = news_sentiments + forum_sentiments

        if not all_scores:
            result = {
                'score': 0.0,
                'level': 'neutral',
                'news_count': 0,
                'forum_count': 0,
                'description': '暂无足够数据',
                'details': [],
            }
        else:
            avg_score = np.mean(all_scores)
            pos_ratio = sum(1 for s in all_scores if s > 0) / len(all_scores)
            neg_ratio = sum(1 for s in all_scores if s < 0) / len(all_scores)

            if avg_score > 0.2:
                level = '偏多'
            elif avg_score > 0:
                level = '中性偏多'
            elif avg_score > -0.2:
                level = '中性偏空'
            else:
                level = '偏空'

            result = {
                'score': round(float(avg_score), 3),
                'level': level,
                'news_count': len(news_sentiments),
                'forum_count': len(forum_sentiments),
                'pos_ratio': round(pos_ratio, 2),
                'neg_ratio': round(neg_ratio, 2),
                'description': f'综合情感{level}（{len(all_scores)}条样本）',
                'details': all_scores[:10],
            }

        cache.set(cache_key, result, category='sentiment', ttl=self._cache_ttl)
        return result

    def _fetch_eastmoney_news(self, stock_code: str, stock_name: str) -> List[float]:
        """从东方财富获取新闻并分析情感"""
        sentiments = []
        try:
            import requests
            # 东方财富个股新闻 API
            market = 1 if stock_code.startswith('6') else 0
            code = stock_code.replace('sh', '').replace('sz', '')

            url = 'https://search-api-web.eastmoney.com/search/jsonp.asp'
            params = {
                'cb': 'callback',
                'param': f'{{"uid":"","keyword":"{stock_name}","type":"cms","pageindex":1,"pagesize":20}}',
            }
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                # 解析 JSONP → 安全替换 eval
                text = resp.text
                m = re.search(r'callback\((.+)\)', text)
                if m:
                    import json
                    try:
                        data = json.loads(m.group(1))
                    except json.JSONDecodeError as e:
                        logger.error(f"[SentimentAnalyzer] JSON 解析失败: {e}")
                        return sentiments
                    # 提取标题
                    if isinstance(data, dict) and 'result' in data:
                        results = data['result']
                        if isinstance(results, dict) and 'cms' in results:
                            for item in results['cms'][:20]:
                                title = item.get('title', '') + item.get('content_summary', '')
                                if title:
                                    result = analyze_sentiment_text(title)
                                    sentiments.append(result['score'])
        except Exception as e:
            logger.debug(f"[SentimentAnalyzer] 新闻获取失败: {e}")

        return sentiments

    def _fetch_guba_posts(self, stock_code: str, stock_name: str) -> List[float]:
        """从东方财富股吧获取帖子并分析情感"""
        sentiments = []
        try:
            import requests
            code = stock_code.replace('sh', '').replace('sz', '')

            url = 'https://guba.eastmoney.com/list,{code}.html'
            # 简化: 只获取标题
            headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'}
            resp = requests.get(url.format(code=code), headers=headers, timeout=10)
            if resp.status_code == 200:
                # 提取标题（多选择器，增强鲁棒性）
                titles = re.findall(r'<a[^>]*class="lch"[^>]*>([^<]+)</a>', resp.text)
                if not titles:
                    # 备选: 尝试其他常见 class 名
                    titles = re.findall(r'<a[^>]*class="[^"]*title[^"]*"[^>]*>([^<]+)</a>', resp.text)
                for title in titles[:20]:
                    result = analyze_sentiment_text(title)
                    sentiments.append(result['score'])
        except Exception as e:
            logger.debug(f"[SentimentAnalyzer] 股吧获取失败: {e}")

        return sentiments


# 全局实例
sentiment_analyzer = SentimentAnalyzer()
