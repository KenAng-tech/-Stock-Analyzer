#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
LLM 情感分析模块 — 基于 OMLX/OMLX Proxy 的大语言模型情感分析

功能:
1. 调用 LLM (如 MiniMax-M2.7, GPT-4o, Claude) 分析股票新闻/帖子
2. 支持结构化 JSON 输出
3. 自动降级到词典法 (当 LLM 不可用时)
4. 支持批量分析
5. 与 SentimentEngine 集成

参考:
- 2026 年 LLM 在金融情感分析中的最新应用
- FinGPT, FinLLM 等金融大模型
"""

import os
import json
import math
import time
import threading
import urllib.request
import urllib.error
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from modules.logger import logger


# ── LLM 配置 ──────────────────────────────────────────────

@dataclass
class LLMConfig:
    """LLM 配置"""
    # OMLX 配置 — 直接从 settings.json 读取正确 API Key
    _OMLX_SETTINGS_PATH = os.path.expanduser("~/.omlx/settings.json")

    @staticmethod
    def _get_omlx_api_key() -> str:
        """从 OMLX settings.json 读取 API Key"""
        try:
            with open(LLMConfig._OMLX_SETTINGS_PATH, 'r') as f:
                import json
                settings = json.load(f)
            key = settings.get('auth', {}).get('api_key', '')
            if key:
                return key
        except Exception as e:
            logger.debug(f"[LLM] 读取 OMLX settings 失败: {e}")
        # Fallback: 环境变量 (不设置则返回 None)
        return os.environ.get("OMLX_API_KEY")

    # OMLX 配置 (直连 8080，proxy 8082 可能 502)
    omlx_url: str = os.environ.get("OMLX_URL", "http://127.0.0.1:8080")
    omlx_model: str = os.environ.get("OMLX_MODEL", "GLM-4.7-Flash-MLX-8bit")
    omlx_api_key: str = _get_omlx_api_key.__func__()
    omlx_timeout: float = 15.0

    # OpenAI 兼容接口 (OMLX 支持 OpenAI 格式)
    openai_base: str = "http://127.0.0.1:8080/v1"
    openai_model: str = os.environ.get("OMLX_MODEL", "GLM-4.7-Flash-MLX-8bit")
    openai_api_key: str = omlx_api_key
    openai_timeout: float = 15.0

    # 提示词
    sentiment_prompt: str = """你是一个专业的金融情感分析师。请分析以下关于某只股票的内容，
判断其情感倾向（正面/中性/负面），并给出置信度 (0-1) 和理由。
请直接返回 JSON 格式：
{{
  "direction": "positive" | "neutral" | "negative",
  "score": -1.0 ~ 1.0,
  "confidence": 0.0 ~ 1.0,
  "reason": "简要理由",
  "keywords": ["关键词1", "关键词2"]
}}"""


# ── 提示词模板 ──────────────────────────────────────────────

NEWS_SENTIMENT_PROMPT = """请分析以下新闻标题的情感倾向，判断其对股票价格的影响。

{news}

请返回 JSON:
{{
  "direction": "positive" | "neutral" | "negative",
  "score": -1.0 ~ 1.0,
  "confidence": 0.0 ~ 1.0,
  "reason": "简要理由"
}}"""

POST_SENTIMENT_PROMPT = """请分析以下股吧帖子内容的情感倾向，
判断投资者对该股票的情绪。

{posts}

请返回 JSON:
{{
  "direction": "positive" | "neutral" | "negative",
  "score": -1.0 ~ 1.0,
  "confidence": 0.0 ~ 1.0,
  "reason": "简要理由"
}}"""

# ── LLM 客户端 ──────────────────────────────────────────────

class LLMClient:
    """LLM 客户端，支持 OpenAI 兼容接口"""

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self._initialized = False
        self._init()

    def _init(self):
        """初始化 LLM 客户端"""
        try:
            # 测试 OMLX 连接
            url = f"{self.config.omlx_url}/v1/models"
            req = urllib.request.Request(url)
            req.add_header("Authorization", f"Bearer {self.config.omlx_api_key}")
            req.add_header("Content-Type", "application/json")
            resp = urllib.request.urlopen(req, timeout=self.config.omlx_timeout)
            if resp.status == 200:
                self._initialized = True
                logger.info(f"[LLM] OMLX 连接成功: {self.config.omlx_url}")
        except Exception as e:
            logger.warning(f"[LLM] OMLX 连接失败: {e}, 尝试 OpenAI 兼容接口")
            try:
                url = f"{self.config.openai_base}/models"
                req = urllib.request.Request(url)
                req.add_header("Authorization", f"Bearer {self.config.openai_api_key}")
                resp = urllib.request.urlopen(req, timeout=self.config.openai_timeout)
                if resp.status == 200:
                    self._initialized = True
                    logger.info(f"[LLM] OpenAI 兼容接口连接成功: {self.config.openai_base}")
            except Exception as e2:
                logger.warning(f"[LLM] OpenAI 兼容接口也失败: {e2}")
                self._initialized = False

    def _call_api(self, messages: List[Dict], model: Optional[str] = None) -> Optional[str]:
        """调用 LLM API"""
        model = model or self.config.omlx_model
        url = f"{self.config.omlx_url}/v1/chat/completions"
        payload = json.dumps({
            "model": model,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 512,
            "response_format": {"type": "json_object"},
        }).encode()

        req = urllib.request.Request(url, data=payload, method="POST")
        req.add_header("Authorization", f"Bearer {self.config.omlx_api_key}")
        req.add_header("Content-Type", "application/json")

        try:
            resp = urllib.request.urlopen(req, timeout=self.config.omlx_timeout)
            result = json.loads(resp.read().decode())
            content = result["choices"][0]["message"]["content"]
            return content
        except Exception as e:
            logger.debug(f"[LLM] API 调用失败: {e}")
            return None

    def analyze(self, text: str, prompt: Optional[str] = None) -> Optional[Dict]:
        """
        分析文本情感

        Args:
            text: 输入文本
            prompt: 可选提示词模板

        Returns:
            {
                'direction': 'positive' | 'neutral' | 'negative',
                'score': -1.0 ~ 1.0,
                'confidence': 0.0 ~ 1.0,
                'reason': str,
                'keywords': List[str],
            }
        """
        if not self._initialized:
            return None

        prompt_text = prompt or self.config.sentiment_prompt
        messages = [
            {"role": "system", "content": prompt_text},
            {"role": "user", "content": text},
        ]

        result = self._call_api(messages)
        if result:
            try:
                # 解析 JSON (处理可能的 markdown 代码块)
                content = result.strip()
                if content.startswith("```"):
                    lines = content.split("\n")
                    content = "\n".join(lines[1:-1])
                parsed = json.loads(content)
                # LLM 可能返回 list 而非 dict，做类型检查
                if isinstance(parsed, list):
                    parsed = parsed[0] if parsed else {}
                if not isinstance(parsed, dict):
                    logger.debug(f"[LLM] 返回格式非 dict: {type(parsed)}")
                    return None
                # ── tool contract (2026-09-11 决策链 B): 契约三键 fail-closed ──
                # 缺 direction/score/confidence 或非有限/越界 → None → analyze_news
                # 标 *_fallback (诚实); 旧版 get(默认值) 会把漂移形 (缺键/字符串
                # score/NaN conf) 伪装成 source=llm_news 的真中性链 → 假信号进共识
                d = parsed.get("direction")
                if d not in ("positive", "negative", "neutral"):
                    logger.debug(f"[LLM] 契约校验失败: direction={d!r}")
                    return None
                s, c = parsed.get("score"), parsed.get("confidence")
                try:
                    s, c = float(s), float(c)
                    if not (math.isfinite(s) and math.isfinite(c)):
                        return None
                except (TypeError, ValueError):
                    return None
                if not (-1.0 <= s <= 1.0 and 0.0 <= c <= 1.0):
                    return None
                return {
                    "direction": d,
                    "score": s,
                    "confidence": c,
                    "reason": str(parsed.get("reason", "")),
                    "keywords": parsed.get("keywords", []),
                }
            except (json.JSONDecodeError, KeyError, IndexError, ValueError) as e:
                logger.debug(f"[LLM] JSON 解析失败: {e}")
                return None
        return None


# ── LLM 情感分析器 ──────────────────────────────────────────────

class LLMSentimentAnalyzer:
    """
    LLM 情感分析器

    使用 LLM (如 GPT-4o, MiniMax-M2.7) 分析股票新闻/帖子的情感。
    自动降级到词典法。
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self.client = LLMClient(self.config)
        self._cache: Dict[str, Dict] = {}
        self._cache_time: Dict[str, float] = {}
        self._cache_ttl = 3600  # 1 小时缓存

    def analyze_news(self, stock_code: str, news_titles: List[str]) -> Dict:
        """
        分析新闻标题情感

        Args:
            stock_code: 股票代码
            news_titles: 新闻标题列表

        Returns:
            情感分析结果
        """
        if not news_titles:
            return self._empty_result()

        cache_key = f"llm_news_{stock_code}"
        # 检查缓存
        if cache_key in self._cache:
            if time.time() - self._cache_time.get(cache_key, 0) < self._cache_ttl:
                return self._cache[cache_key]

        # 拼接文本
        text = "\n".join(f"- {t}" for t in news_titles[:20])
        prompt = NEWS_SENTIMENT_PROMPT.format(news=text)

        result = self.client.analyze(text, prompt)
        if result:
            result["stock_code"] = stock_code
            result["source"] = "llm_news"
            result["text_count"] = len(news_titles)
        else:
            result = self._empty_result()
            result["source"] = "llm_news_fallback"

        self._cache[cache_key] = result
        self._cache_time[cache_key] = time.time()
        return result

    def analyze_posts(self, stock_code: str, post_contents: List[str]) -> Dict:
        """
        分析股吧帖子情感

        Args:
            stock_code: 股票代码
            post_contents: 帖子内容列表

        Returns:
            情感分析结果
        """
        if not post_contents:
            return self._empty_result()

        cache_key = f"llm_posts_{stock_code}"
        if cache_key in self._cache:
            if time.time() - self._cache_time.get(cache_key, 0) < self._cache_ttl:
                return self._cache[cache_key]

        text = "\n".join(f"- {c}" for c in post_contents[:20])
        prompt = POST_SENTIMENT_PROMPT.format(posts=text)

        result = self.client.analyze(text, prompt)
        if result:
            result["stock_code"] = stock_code
            result["source"] = "llm_posts"
            result["text_count"] = len(post_contents)
        else:
            result = self._empty_result()
            result["source"] = "llm_posts_fallback"

        self._cache[cache_key] = result
        self._cache_time[cache_key] = time.time()
        return result

    def analyze_combined(self, stock_code: str,
                         news_titles: List[str],
                         post_contents: List[str]) -> Dict:
        """
        综合分析新闻 + 帖子

        Args:
            stock_code: 股票代码
            news_titles: 新闻标题
            post_contents: 帖子内容

        Returns:
            综合情感分析结果
        """
        news_result = self.analyze_news(stock_code, news_titles)
        posts_result = self.analyze_posts(stock_code, post_contents)

        # 加权平均
        total_weight = 2.0 + 1.0  # 新闻权重 2x
        combined_score = (
            news_result["score"] * 2.0 + posts_result["score"] * 1.0
        ) / total_weight
        combined_confidence = (
            news_result["confidence"] * 2.0 + posts_result["confidence"] * 1.0
        ) / total_weight

        # 多数投票决定方向
        directions = [news_result["direction"]] * 2 + [posts_result["direction"]]
        from collections import Counter
        combined_direction = Counter(directions).most_common(1)[0][0]

        return {
            "stock_code": stock_code,
            "direction": combined_direction,
            "score": round(combined_score, 4),
            "confidence": round(combined_confidence, 4),
            "news": news_result,
            "posts": posts_result,
            "source": "llm_combined",
            "text_count": news_result.get("text_count", 0) + posts_result.get("text_count", 0),
        }

    def analyze_with_fallback(self, stock_code: str,
                              news_titles: List[str],
                              post_contents: List[str]) -> Dict:
        """
        综合分析 + 超时保护 (避免 LLM 服务挂起)

        使用 threading + timeout 机制，如果 LLM 调用超时则回退到词典法
        """
        import threading
        import time

        result_holder = {'result': None, 'error': None}

        def _analyze():
            try:
                result_holder['result'] = self.analyze_combined(stock_code, news_titles, post_contents)
            except Exception as e:
                result_holder['error'] = e

        thread = threading.Thread(target=_analyze, daemon=True)
        thread.start()
        thread.join(timeout=5.0)  # 5 秒超时

        if result_holder['result']:
            return result_holder['result']

        # 超时或出错 → 回退到词典法
        logger.warning(f"[LLM Sentiment] LLM 调用超时/失败 ({result_holder['error']})，使用词典回退")
        return self._dict_fallback(stock_code, news_titles, post_contents)

    def _dict_fallback(self, stock_code: str,
                       news_titles: List[str],
                       post_contents: List[str]) -> Dict:
        """词典法情感分析 (LLM 不可用时的回退)"""
        from modules.sentiment_engine import DictionarySentimentAnalyzer

        # 决策链 B 后续 (2026-09-11): 断链修复 — 原 analyzer.analyze_batch(...)
        # 方法不存在 (DictionarySentimentAnalyzer 只有单条 analyze→Dict, 09-10
        # "analyze 方法不存在" 断链同族) + score.score 假想对象形 + bullish/
        # bearish 词表与主链 positive/negative/neutral 漂移 (前端 3-色映射会 miss)。
        # 此前 n:0 空输入直接绕过本函数 = 三层死形从未运行过 (B 修复后 news 入
        # 手才暴露)。本版: 逐条 analyze + 均值聚合 + 词表对齐。
        analyzer = DictionarySentimentAnalyzer()
        texts = [t for t in (news_titles + post_contents) if t]
        if not texts:
            return self._empty_result()
        scores = [float(analyzer.analyze(t).get('score', 0.0)) for t in texts]
        avg = sum(scores) / len(scores)
        direction = 'positive' if avg > 0.1 else ('negative' if avg < -0.1 else 'neutral')
        return {
            'stock_code': stock_code,
            'direction': direction,
            'score': round(avg, 4),
            'confidence': round(min(abs(avg) + 0.3, 1.0), 4),
            'source': 'dictionary_fallback',
            'reason': f'LLM 服务不可用，词典法 {len(texts)} 条均分',
        }

    def _empty_result(self) -> Dict:
        return {
            "direction": "neutral",
            "score": 0.0,
            "confidence": 0.0,
            "reason": "无数据",
            "keywords": [],
            "source": "empty",
            "text_count": 0,
        }


# ── 全局单例 ──────────────────────────────────────────────

_llm_sentiment_instance: Optional[LLMSentimentAnalyzer] = None
_llm_sentiment_lock = threading.Lock()


def get_llm_sentiment(config: Optional[LLMConfig] = None) -> LLMSentimentAnalyzer:
    """获取全局 LLMSentimentAnalyzer 实例 (线程安全)"""
    global _llm_sentiment_instance
    if _llm_sentiment_instance is None:
        with _llm_sentiment_lock:
            if _llm_sentiment_instance is None:
                _llm_sentiment_instance = LLMSentimentAnalyzer(config)
    return _llm_sentiment_instance


# ── 兼容性别名 ──────────────────────────────────────────────

llm_sentiment_analyzer = LLMSentimentAnalyzer  # 兼容旧名称
