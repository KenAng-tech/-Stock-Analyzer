#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""龙虎榜数据抓取 (2026-09-12 整合 P1, TradingAgents-CN 式 A 股特异数据)

2026 调研锚点: 多智能体 LLM 交易 (TradingAgents-CN 龙虎榜/游资/解禁为 A 股
辩论证据源) + 执行假设审计 (2026-06/07 两论文)。龙虎榜 = A 股「决策→成交」
审计层最值钱的公开数据: 席位明细 (谁在买) + 期后收益 (D1/D20 验证决策对错)。

接口 (今日实测真链, HTTP 200 非 JSONP, success:true):
  东财数据中心 RPT_DAILYBILLBOARD_DETAILSNEW — 单票上榜史 + 全市场当日榜双形态。

链形态 (同 09-11 链尸教训): 失败/空 → 恒 [] 诚实空态, 绝不伪造;
ttl 6h 缓存 (23:00 decision_replay 链预热带); 主链 B 新闻链已修, 此层只补
「资金面情报」维度 (新闻链补不了的资金异动)。

反例教训 (09-12 实测 300620 龙虎榜): 2026-06-16 亿级净买 D20 收益 **-26.7%**
— 上榜/净买≠好收益, 本模块只做「数据 + 证据」, 不构成买入理由。

运行依赖: 仅 stdlib (urllib/json/threading/time), 零三方。
"""

import json
import threading
import time
import urllib.request
from typing import Dict, List, Optional
from urllib.parse import urlparse

from modules.logger import logger
from utils.outbound_guard import host_throttle

_DATA_URL = ("https://datacenter-web.eastmoney.com/api/data/v1/get?"
             "reportName=RPT_DAILYBILLBOARD_DETAILSNEW&columns=ALL")

# 模块级缓存 (23:00 决策复盘链预热 + 实时读走缓存, 不打穿数据源)
_cache: Dict[str, tuple] = {}          # key -> (ts, data)
_cache_lock = threading.Lock()
_TTL = 6 * 3600                        # 6h


def _strip_code(stock_code: str) -> str:
    """sz300620/SH688981 → 300620/688981 (东财 filter 用 6 位裸码)"""
    code = (stock_code or '').strip().upper()
    if code[:2] in ('SZ', 'SH', 'BJ'):
        return code[2:]
    return code


def _fetch(url: str, timeout: float = 10.0) -> List[Dict]:
    """GET + 剥壳 — 任何失败 (网络/坏 JSON/坏形) → [], 不抛 (链尸形态)"""
    try:
        # P0-3 (2026-09-20): 同 host 节流 (intel 4-worker burst 串行化, 防数据中心限频)
        host_throttle.acquire(host_key=urlparse(url).netloc)
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0',
            'Referer': 'https://data.eastmoney.com/',
        })
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode('utf-8', errors='replace')
        if not raw.strip():
            return []
        # 防未来变回 JSONP 形 (search-api-web 主链 09-11 同款坑): 先剥壳
        if raw.lstrip().startswith(('jQuery', 'callback', '_')) and '(' in raw:
            raw = raw[raw.index('(') + 1:raw.rindex(')')]
        data = json.loads(raw)
        rows = ((data.get('result') or {}).get('data')) or []
        if not isinstance(rows, list):
            return []
        return rows
    except Exception as e:
        logger.debug(f"[DragonTiger] 取数失败: {type(e).__name__}: {str(e)[:120]}")
        return []


def _cache_get(key: str) -> Optional[List[Dict]]:
    with _cache_lock:
        hit = _cache.get(key)
    if hit and (time.time() - hit[0]) < _TTL:
        return hit[1]
    return None


def _cache_put(key: str, data: List[Dict]) -> None:
    with _cache_lock:
        _cache[key] = (time.time(), data)
        if len(_cache) > 64:           # 防常驻膨胀 (同日志轮转思路)
            for k in sorted(_cache, key=lambda k: _cache[k][0])[:16]:
                _cache.pop(k, None)


def _safe_seats(val) -> List:
    """席位串容错解析 (09-11 教训: 解析失败必记日志, 不静默吞整行)"""
    if not isinstance(val, str) or not val.strip():
        return []
    try:
        v = json.loads(val)
        return v if isinstance(v, list) else []
    except Exception as e:
        logger.debug(f"[DragonTiger] 席位解析跳过: {e}")
        return []


def to_display(raw: List[Dict]) -> List[Dict]:
    """API 原形 → 决策链/前端消费形 (白名单键, 缺键安全)"""
    out: List[Dict] = []
    for r in (raw or []):
        try:
            seats_buy = _safe_seats(r.get('BUY_SEAT'))
            seats_sell = _safe_seats(r.get('SELL_SEAT'))
            out.append({
                'date': str(r.get('TRADE_DATE', ''))[:10],
                'code': r.get('SECURITY_CODE', ''),
                'name': r.get('SECURITY_NAME_ABBR', ''),
                'close': r.get('CLOSE_PRICE'),
                'change_pct': r.get('CHANGE_RATE'),
                'net_buy': r.get('BILLBOARD_NET_AMT'),          # 元
                'buy_amt': r.get('BILLBOARD_BUY_AMT'),
                'sell_amt': r.get('BILLBOARD_SELL_AMT'),
                'deal_amt': r.get('BILLBOARD_DEAL_AMT'),
                'turnover_pct': r.get('TURNOVERRATE'),
                'explanation': r.get('EXPLANATION') or r.get('EXPLAIN') or '',
                'seats_buy': [s.get('SECURITY_NAME_ABBR') or s.get('NAME', '')
                              if isinstance(s, dict) else str(s)
                              for s in seats_buy][:3],
                'seats_sell': [s.get('SECURITY_NAME_ABBR') or s.get('NAME', '')
                               if isinstance(s, dict) else str(s)
                               for s in seats_sell][:3],
                'd1_pct': r.get('D1_CLOSE_ADJCHRATE'),           # 期后收益
                'd20_pct': r.get('D20_CLOSE_ADJCHRATE'),          # (Brier 反哺锚)
            })
        except Exception as e:
            logger.debug(f"[DragonTiger] 行解析跳过: {e}")
    return out


def fetch_dragon_tiger_history(stock_code: str, limit: int = 5,
                               use_cache: bool = True) -> List[Dict]:
    """单票上榜史 (最近 limit 条, 日期倒序) — 决策审计/情报雷达输入"""
    code = _strip_code(stock_code)
    if not code or len(code) < 6:
        return []
    key = f"dt_h_{code}_{limit}"
    if use_cache:
        hit = _cache_get(key)
        if hit is not None:
            return hit
    url = (f"{_DATA_URL}&pageSize={limit}"
           f"&filter=(SECURITY_CODE%3D%22{code}%22)"
           "&sortColumns=TRADE_DATE&sortTypes=-1")
    rows = to_display(_fetch(url))
    if rows:
        _cache_put(key, rows)
    return rows


def fetch_dragon_tiger_market(limit: int = 30,
                              use_cache: bool = True) -> List[Dict]:
    """全市场当日龙虎榜 (limit 条, 日期倒序) — 情报雷达/异动预警输入"""
    key = f"dt_m_{limit}"
    if use_cache:
        hit = _cache_get(key)
        if hit is not None:
            return hit
    url = (f"{_DATA_URL}&pageSize={limit}&pageNumber=1"
           "&sortColumns=TRADE_DATE%2CSECURITY_CODE&sortTypes=-1%2C1")
    rows = to_display(_fetch(url))
    if rows:
        _cache_put(key, rows)
    return rows


if __name__ == '__main__':
    # 独立冒烟 (非链主路径): python -m modules.dragon_tiger_fetcher
    h = fetch_dragon_tiger_history('sz300620', limit=2)
    m = fetch_dragon_tiger_market(limit=3)
    print(f"history={len(h)} market={len(m)}")
    for r in (h or [])[:2]:
        print('H:', json.dumps(r, ensure_ascii=False)[:220])
    for r in (m or [])[:2]:
        print('M:', json.dumps(r, ensure_ascii=False)[:220])
