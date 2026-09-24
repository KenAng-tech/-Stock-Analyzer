#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""LLM 投资洞察引擎 (2026-09-12 整合 P0-b, 2026 调研「RAG 基本面→洞察」式)

TradingAgents/RAG 论文 (2026) 共识: LLM 单独投票不稳, 但**喂结构化证据
(新闻/龙虎榜/财务) + 强制输出 JSON 契约 + valid 诚实标记** 的「洞察叙事层」
可审计、可积累。本引擎 = 把 09-11 修好的两条真链 (东财新闻 Scrapling 旁路
+ 龙虎榜 datacenter 真链) 喂进 8080 LLM, 产出**人可读、可审计**的洞察块:

  {direction, conviction, evidence[≤5], invalidation, valid, source, ts}

诚实边界 (09-11 决策链 B 纪律延续):
- **空输入 abstain** (无新闻且无龙虎榜 → 不打 LLM, 不空转);
- LLM 降级/fallback/解析失败 → valid:false + abstain 标记 (永不伪装真链 —
  旧版 get(默认值) 把漂移形伪装成真信号的教训);
- **独立观察链**: 不进 6-role 共识投票 (投票链已被 Brier 观察 + IC 闭环
  占据, 加第 7 票会稀释共识分母 — 7 票整合推迟到 Brier 证据成熟);
- 线程取数 5s 硬超时 (同 analyze_with_fallback 45s 预算先例)。

链位置: 前端「AI 洞察」面板 (/api/insight) 独立消费; llm_sentiment 的
情感票 (0-1 连续) 仍是共识唯一舆情入口, 两者正交。

运行依赖: llm_router (可选降级), stdlib 其余 — 可 unittest in-process mock。
"""

import json
import threading
import time
from typing import Dict, List, Optional

from modules.logger import logger

_INSIGHT_PROMPT = """你是 A 股投研分析师。基于以下结构化证据, 对股票 {code} \
给出**可证伪**的短判断 (证据只说「发生了什么」, 你要回答「所以呢」):

【近期新闻】{news}

【龙虎榜/资金异动】{board}

【价格位置】{price}

要求:
1. direction: bullish|bearish|neutral (证据不足以定方向时诚实 neutral)
2. conviction: 0.30~0.70 (证据弱就给低值 — 高于 0.7 仅在多源同向共振时)
3. evidence: ≤5 条, 每条 ≤40 字, 只允许引用上面出现的事实 (禁止编造数字)
4. invalidation: 什么可观测事件会推翻你的判断 (可证伪化要求)

只输出 JSON, 格式 {{"direction":"neutral","conviction":0.45,"evidence":["…"],\
"invalidation":"…"}}
注意: 龙虎榜高净买历史上常伴随高位放量派发 (09-12 实测高净买 D20 收益可\
为负), 不要因「大资金买入」直接推导看涨。"""


class InsightEngine:
    """新闻+龙虎榜 → LLM 投资洞察 (单例可复用, 线程安全)"""

    _TTL = 3600.0                      # 同票洞察缓存 1h (同 llm_sentiment 形)

    def __init__(self):
        self._cache: Dict[str, tuple] = {}          # code -> (ts, result)
        self._lock = threading.Lock()

    # ── 组装 ──────────────────────────────────────────────────────

    def get_insight(self, stock_code: str,
                    news: Optional[List[Dict]] = None,
                    posts: Optional[List[Dict]] = None,
                    dragon_tiger: Optional[List[Dict]] = None,
                    price_note: str = '') -> Dict:
        """一次洞察 (cache-first; 空输入 abstain — 不空转 LLM)"""
        now = time.time()
        with self._lock:
            hit = self._cache.get(stock_code)
            if hit and (now - hit[0]) < self._TTL:
                return hit[1]

        news = news or []
        dragon_tiger = dragon_tiger or []
        if not news and not dragon_tiger:
            out = self._abstain('empty', '无新闻/无龙虎榜证据 — 不空转 LLM')
            return out

        news_txt = '\n'.join(
            f"- {n.get('title', '') if isinstance(n, dict) else n}"
            f" ({n.get('date', '')[:10] if isinstance(n, dict) else ''})"
            for n in news[:15]) or '(无)'
        board_txt = '\n'.join(
            f"- 上榜 {b.get('date', '')}: 净买 {float(b.get('net_buy') or 0)/1e4:+.0f}万,"
            f" 换手 {b.get('turnover_pct', '?')}%,"
            f" D20收益 {b.get('d20_pct', '无数据')}"
            for b in dragon_tiger[:3]) or '(无)'

        prompt = _INSIGHT_PROMPT.format(
            code=stock_code, news=news_txt, board=board_txt,
            price=price_note or '(无价格快照)')

        try:
            from modules.llm_router import llm_router as router
            resp = router.route(prompt, timeout=60.0)  # 2026-09-15: 25→60, 8080 实测 17s+ 辩论/洞察常超时
        except Exception as e:
            logger.warning(f"[Insight] LLM 路由不可用: {e}")
            resp = {'success': False, 'fallback': True}

        out = self._parse(resp, stock_code) if (
            resp.get('success') and not resp.get('fallback')
            and resp.get('provider') != 'rule_engine') else \
            self._abstain('llm_fallback',
                          f"LLM 降级 (provider={resp.get('provider', '?')}),"
                          f" 不伪造洞察")

        if out.get('valid'):
            with self._lock:
                self._cache[stock_code] = (time.time(), out)
        return out

    # ── 解析 (fail-closed, 同决策链 B 契约) ─────────────────────────

    def _parse(self, resp: Dict, stock_code: str) -> Dict:
        """LLM 返回 → 洞察块; 缺键/越界 → abstain (假洞察比无洞察危险)"""
        content = (resp.get('content') or '').strip()
        if content.startswith('```'):
            content = content.strip('`').lstrip('json').strip()
        try:
            start, end = content.find('{'), content.rfind('}')
            if start < 0 or end <= start:
                raise ValueError('no json braces')
            data = json.loads(content[start:end + 1])
        except Exception as e:
            logger.debug(f"[Insight] 解析失败 (跳过): {e}")
            return self._abstain('parse_failed', f'LLM 返回不可解析: {str(e)[:40]}')

        if not isinstance(data, dict):
            return self._abstain('parse_failed', '非 dict 形')
        d = data.get('direction')
        if d not in ('bullish', 'bearish', 'neutral'):
            return self._abstain('contract_drift', f'direction={d!r}')
        try:
            cv = float(data.get('conviction'))
            if not (0.0 <= cv <= 1.0) or cv != cv:
                raise ValueError('conviction out of range/NaN')
        except (TypeError, ValueError) as e:
            return self._abstain('contract_drift', f'conviction={data.get("conviction")!r}')

        return {
            'valid': True,
            'stock_code': stock_code,
            'direction': d,
            'conviction': round(cv, 3),
            'evidence': [str(e)[:60] for e in (data.get('evidence') or [])[:5]],
            'invalidation': str(data.get('invalidation', ''))[:120],
            'source': 'llm_insight',
            'provider': resp.get('provider', ''),
            'ts': time.strftime('%Y-%m-%dT%H:%M:%S'),
        }

    def _abstain(self, source: str, reason: str) -> Dict:
        """诚实空态 (前端消费: valid=false → 灰色「无洞察」而非假判断)"""
        return {'valid': False, 'abstain': True, 'source': source,
                'reason': reason, 'direction': 'neutral', 'conviction': 0.0,
                'ts': time.strftime('%Y-%m-%dT%H:%M:%S')}


_insight_engine: Optional[InsightEngine] = None
_insight_lock = threading.Lock()


def get_insight_engine() -> InsightEngine:
    """全局单例 (同 get_sentiment_engine 惯例)"""
    global _insight_engine
    if _insight_engine is None:
        with _insight_lock:
            if _insight_engine is None:
                _insight_engine = InsightEngine()
    return _insight_engine
