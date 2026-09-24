#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""情报雷达聚合引擎 (2026-09-12 整合 P1, TradingAgents-CN 式 A 股特异数据)

三源聚合 → /api/intel/<code> 端点 + 前端「情报雷达」面板:
  1. 龙虎榜   dragon_tiger_fetcher (datacenter 真链实测 200, 席位/D20 收益)
  2. 情感     SentimentEngine.get_sentiment_score (09-17/09-11 两条真链复用:
              新闻主链+Scrapling 旁路 → FinBERT/词典聚合)
  3. 解禁     akshare stock_restricted_release_detail_em (09-12 实测: 508 行
              12 列真数据, 日期区间形; queue_sina 两票皆空 shape — 弃)

设计纪律 (09-10/09-11 链尸教训):
- 每源独立取数独立降级 (一源坏 ≠ 整卡坏); 全空诚实「无情报」非假数据;
- detail_em 全市场日期区间取数 (23:00 decision_replay 预热, 端点 6h 缓存优先);
- 300620 反例锚: 高净买 D20 -26% — 本层呈现证据, 不构成任何方向的票。

依赖: dragon_tiger_fetcher / sentiment_engine / data_fetcher / akshare (可选)。
"""

import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from modules.logger import logger

_unlock_cache: Dict = {}          # 'window' -> (ts, rows, note)
_unlock_lock = threading.Lock()


def fetch_unlock_calendar(days_before: int = 60, days_after: int = 30) -> List[Dict]:
    """全市场解禁日历 (东财 detail_em, 09-12 实测真链形).

    Returns: rows [{code, name, unlock_date, type, qty, mv, prev_close,
                   pre20_pct, post20_pct}] — 失败/空 → [] (诚实降级)
    """
    now = time.time()
    with _unlock_lock:
        hit = _unlock_cache.get('window')
        if hit and (now - hit[0]) < 6 * 3600:
            return hit[1]
    try:
        import akshare as ak
        start = (datetime.now() - timedelta(days=days_before)).strftime('%Y%m%d')
        end = (datetime.now() + timedelta(days=days_after)).strftime('%Y%m%d')
        df = ak.stock_restricted_release_detail_em(start_date=start,
                                                    end_date=end)
        rows: List[Dict] = []
        if df is not None and len(df):
            for _, r in df.iterrows():
                try:
                    rows.append({
                        'code': str(r.get('股票代码', '')),
                        'name': r.get('股票简称', ''),
                        'unlock_date': str(r.get('解禁时间', ''))[:10],
                        'type': str(r.get('限售股类型', '')),
                        'qty': float(r.get('实际解禁数量') or 0),
                        'mv': float(r.get('实际解禁市值') or 0),
                        'prev_close': r.get('解禁前一交易日收盘价'),
                        'pre20_pct': r.get('解禁前20日涨跌幅'),
                        'post20_pct': r.get('解禁后20日涨跌幅'),
                    })
                except Exception:
                    continue
        with _unlock_lock:
            _unlock_cache['window'] = (time.time(), rows)
        logger.info(f"[Intel] 解禁日历刷新: {len(rows)} 条")
        return rows
    except Exception as e:
        logger.warning(f"[Intel] 解禁日历失败 (跳过): {type(e).__name__}: "
                       f"{str(e)[:100]}")
        return []


def intel_summary(stock_code: str) -> Dict:
    """单票情报摘要 (决策链「情报雷达」主入口, 三源独立降级).

    端点/前端/23:00 预热共用; 全程 try 包裹 — 任何源坏只影响该卡。
    """
    code6 = (stock_code or '')[-6:]
    out: Dict = {'stock_code': stock_code, 'ts': time.strftime('%Y-%m-%dT%H:%M:%S')}

    # 1. 龙虎榜 (近 5 次上榜, datacenter 真链)
    try:
        from modules.dragon_tiger_fetcher import fetch_dragon_tiger_history
        rows = fetch_dragon_tiger_history(stock_code, limit=5) or []
        out['dragon_tiger'] = {
            'on_board': bool(rows),
            'count': len(rows),
            'items': rows[:5],
            'latest': rows[0] if rows else None,
        }
    except Exception as e:
        logger.debug(f"[Intel] 龙虎榜源降级: {e}")
        out['dragon_tiger'] = {'on_board': False, 'count': 0, 'items': [],
                               'latest': None, 'degraded': True}

    # 2. 新闻情感 (复用 09-17 修复真链: 新闻主链+旁路+FinBERT/词典聚合)
    try:
        from modules.sentiment_engine import get_sentiment_engine
        eng = get_sentiment_engine()
        senti = eng.get_sentiment_score(stock_code) or {}
        out['sentiment'] = {
            'score': senti.get('score'),
            'label': senti.get('label'),
            'n_articles': senti.get('n_articles', 0),
            'method': senti.get('method'),     # finbert/dictionary/no_data
            'degraded': senti.get('method') == 'no_data',
        }
        # 新闻明细喂 AI 洞察链 (P0-b) — 同一次取数, 不二次打新闻源
        try:
            from modules.data_fetcher import StockDataFetcher
            f = StockDataFetcher()
            news = f.get_stock_news(stock_code) or []
            posts = f.get_stock_posts(stock_code) or []
            # ── 最新日期纪律 (09-13 核查修) ──
            # ① 日期键双形: 主链='date' / 旁路 (eastmoney_news_fetcher)='time' —
            #   只读 'date' 会丢光旁路日期 (09-13 实测 n=10 feed 日期全空,
            #   洞察链无从判新鲜) → 两键都取;
            # ② 日期 >7d 的旧新闻不入 feed (洞察 prompt「近期新闻」名实相符;
            #   无日期键的帖子/旧主链按新鲜处理 — 空排除=空转喂料)。
            def _fresh(dstr):
                try:
                    d = str(dstr or '')[:10].replace('/', '-')
                    if not d:
                        return True
                    return (datetime.now() - datetime.strptime(
                        d, '%Y-%m-%d')).days <= 7
                except ValueError:
                    return True
            texts = ([{'title': n.get('title', ''),
                       'date': (n.get('date') or n.get('time', ''))[:10]}
                      for n in news if isinstance(n, dict)
                      and _fresh(n.get('date') or n.get('time'))][:15]
                     + [{'title': (p.get('title') or p.get('content', ''))[:120],
                         'date': ''} for p in posts if isinstance(p, dict)][:5])
            out['news'] = {'titles': [t['title'] for t in texts][:20],
                           'count': len(news)}
            out['_insight_feed'] = {'news': texts}   # P0-b 洞察链输入 (非展示)
        except Exception as e:
            logger.debug(f"[Intel] 新闻明细降级: {e}")
            out['news'] = {'titles': [], 'count': 0, 'degraded': True}
            out['_insight_feed'] = {'news': []}
    except Exception as e:
        logger.debug(f"[Intel] 情感源降级: {e}")
        out['sentiment'] = {'score': None, 'label': None, 'degraded': True}

    # 3. 解禁 (detail_em 真链过滤本票; 无记录 = 诚实 None 非造假 0)
    try:
        cal = fetch_unlock_calendar()
        mine = [r for r in cal if r.get('code') == code6]
        future = [r for r in mine if r.get('unlock_date', '') >=
                  time.strftime('%Y-%m-%d')]
        past = sorted((r for r in mine if r.get('unlock_date', '') <
                       time.strftime('%Y-%m-%d')),
                      key=lambda r: r.get('unlock_date', ''), reverse=True)
        out['unlock'] = {
            'next': future[0] if future else None,
            'recent_past': past[:2],
            'calendar_count': len(cal),        # 全市场日历条数 (真链健康度)
        }
    except Exception as e:
        logger.debug(f"[Intel] 解禁源降级: {e}")
        out['unlock'] = {'next': None, 'recent_past': [], 'degraded': True}

    return out
