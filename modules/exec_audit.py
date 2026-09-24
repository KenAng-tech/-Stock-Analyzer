#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""可成交性审计 (2026-09-12 整合 P0-c, 2026-06/07 两论文「执行假设审计」式)

定位 (诚实边界, 同 09-11 观察链形态): **审计不投票** — 只回答一个问题:
「假如 consensus 此刻给出 buy/sell, 这个决策能按假设执行吗?」
执行假设 (流动性/滑点/触板) 在 2026 被独立研究层证实为 LLM 交易回测
失真最大头 (Can LLMs Execute Parent Orders? / Beyond Agent Architecture)。

审计三元组 (输入全可选, 缺则诚实 None — 数据缺不猜):
  1. 触板: |涨跌幅| ≥ 板限-0.5pp → 触板/贴板 (创业板/科创 20cm, 主板 10)
  2. 流动性: 5日均成交额 <1亿 → 收缩预警 (volume 股数×close 实测形)
  3. 龙虎榜: 近 2 日上榜 (换手异动) + D20 期后收益 (Brier 反哺锚)

成本常数: 往返 ≈26bp (佣金双边 3+3 + 印花税单边 10 + 滑点 10),
回测 25bp 同源; 方向票期望优势需先扣它 (backtest.md 规则的决策时呈现)。

⚠ 300620 实测反例 (09-12): 2026-06 两笔亿级龙虎榜净买, D20 -16%/-26.7%
— 上榜≠好收益。本层只呈现证据, 不构成任何方向的票 (防「数据=结论」污染)。

运行依赖: stdlib only (链尸教训: 纯函数可 unittest in-process 直跑)。
"""

from datetime import datetime
from typing import Dict, List, Optional

from modules.logger import logger

# 板块涨跌幅限制 (无 ST 源, 诚实不判 ST — 缺数据不猜)
_LIMIT_BOARDS = {
    'sz300': 20.0, 'sz301': 20.0,      # 创业板
    'sh688': 20.0, 'sh689': 20.0,      # 科创板
}
_DEFAULT_LIMIT = 10.0                   # 主板

# 往返成本常数 (bp): 佣金 3+3, 印花税 10 (卖出单边), 滑点 10
_ROUND_TRIP_BP = 26.0
_LIQUIDITY_MIN_YI = 1.0                # 5日均成交额 <1亿 → 流动性预警


def _board_limit(stock_code: str) -> float:
    code = (stock_code or '').lower()
    return _LIMIT_BOARDS.get(code[:5], _DEFAULT_LIMIT)


def audit_execution(stock_code: str, decision: str,
                    confidence: Optional[float] = None,
                    snapshot: Optional[Dict] = None,
                    klines: Optional[List[Dict]] = None,
                    dragon_tiger: Optional[List[Dict]] = None) -> Dict:
    """可成交性审计 (观察层, 不改变任何投票/共识).

    Args:
        stock_code:  股票代码 (sz300620 形)
        decision:    共识方向 (buy/sell/neutral/… — 仅作 hints 语境参考)
        confidence:  共识置信度 (透传呈现)
        snapshot:    实时行情 dict (含 close/change_pct/prev_close, 可缺)
        klines:      日K 列表 (close/volume 键, 可缺)
        dragon_tiger: 龙虎榜 to_display 行 (可缺)

    Returns:
        exec_audit dict (面板/端点消费形); 单维缺数据 → 该维 None/降级,
        整链绝不抛 (供 worker 在 45s 预算内最坏跳过审计, 不拖决策)。
    """
    snap = snapshot or {}
    kl = klines or []
    dt = dragon_tiger or []
    hints: List[str] = []
    buy_side = decision in ('buy', 'strong_buy')
    sell_side = decision in ('sell', 'strong_sell')

    # ── 1. 触板/贴板 (涨跌停可成交性) ──
    limit_block: Dict = {'status': None, 'threshold_pct': _board_limit(stock_code),
                        'last_pct': None, 'source': None}
    last_pct = None
    try:
        if isinstance(snap.get('change_pct'), (int, float)):
            last_pct, limit_block['source'] = float(snap['change_pct']), 'snapshot'
        elif len(kl) >= 2:
            c0, c1 = kl[-2].get('close'), kl[-1].get('close')
            if c0 and c1:
                last_pct = (float(c1) / float(c0) - 1) * 100
                limit_block['source'] = 'kline'
    except Exception:
        last_pct = None
    if last_pct is not None:
        limit_block['last_pct'] = round(last_pct, 2)
        lim = limit_block['threshold_pct']
        if last_pct >= lim - 0.5:
            limit_block['status'] = 'limit_up'
            if buy_side:
                hints.append(f'已涨停/贴板 (+{last_pct:.2f}%, 板限 {lim}%) '
                             f'— buy 假设难以按假设价成交, 追价滑点不可估')
        elif last_pct <= -lim + 0.5:
            limit_block['status'] = 'limit_down'
            if sell_side:
                hints.append(f'已跌停/贴板 ({last_pct:.2f}%, 板限 {lim}%) '
                             f'— sell 假设可能排队未成交, 止损假设失真')

    # ── 2. 流动性 (5日均成交额, volume=股数 × close=元 实测形) ──
    liquidity: Dict = {'ok': None, 'amount_yi_avg5': None, 'note': None}
    try:
        recent = kl[-5:] if len(kl) >= 5 else kl
        amts = [float(k['volume']) * float(k['close'])
                for k in recent if k.get('volume') and k.get('close')]
        if amts:
            avg_yi = sum(amts) / len(amts) / 1e8
            liquidity['amount_yi_avg5'] = round(avg_yi, 2)
            if avg_yi < _LIQUIDITY_MIN_YI:
                liquidity['ok'] = False
                liquidity['note'] = (f'5日均成交额 {avg_yi:.2f}亿 < '
                                     f'{_LIQUIDITY_MIN_YI}亿 — 滑点可能吃掉'
                                     f'目标收益 (往返成本已≈{_ROUND_TRIP_BP}bp)')
                if buy_side or sell_side:
                    hints.append(liquidity['note'])
            else:
                liquidity['ok'] = True
    except Exception as e:
        logger.debug(f"[ExecAudit] 流动性跳过: {e}")

    # ── 3. 龙虎榜 (资金异动证据, 反例锚) ──
    board: Dict = {'on_board': False}
    try:
        if dt:
            top = dt[0]
            board = {
                'on_board': True,
                'date': top.get('date'),
                'net_buy': top.get('net_buy'),
                'turnover_pct': top.get('turnover_pct'),
                'explanation': str(top.get('explanation', ''))[:80],
                'd20_pct': top.get('d20_pct'),
            }
            # hint 真形 (09-13 核查修): dt[0] = 最近一次上榜, 可能是数月前
            # (09-13 实测 300620 拿 08-04 数据喊「近 2 日」= 事实错标)。
            # 仅近 2 个交易日内上榜才出 hint, hint 直呈真实上榜日期
            _recent = False
            try:
                _recent = (datetime.now() - datetime.strptime(
                    str(top.get('date', ''))[:10], '%Y-%m-%d')).days <= 4
            except ValueError:
                pass
            if _recent and isinstance(top.get('turnover_pct'), (int, float)) \
                    and top['turnover_pct'] > 15:
                hints.append(f"龙虎榜 {str(top.get('date', ''))[:10]} "
                             f"(换手 {top['turnover_pct']}%) — 情绪驱动票, "
                             f"趋势/均值回归假设需重估")
    except Exception as e:
        logger.debug(f"[ExecAudit] 龙虎榜跳过: {e}")

    return {
        'version': 'exec_audit_v1',
        'ts': datetime.now().isoformat(),
        'decision_context': {'decision': decision,
                             'confidence': confidence},
        'limit': limit_block,
        'liquidity': liquidity,
        'cost': {'round_trip_bp': _ROUND_TRIP_BP,
                 'note': f'往返成本≈{_ROUND_TRIP_BP}bp (佣金双边6+印花税单边10'
                         f'+滑点10); 方向票期望优势须先扣成本'},
        'board': board,
        'block_hints': hints,
        'note': '审计=执行假设观察层, 不构成任何方向的投票或降权 (2026 调研'
                '「执行假设审计」形, 300620 龙虎榜反例锚)',
    }


def audit_for_stock(stock_code: str, decision: str,
                    confidence: Optional[float] = None,
                    timeout: float = 5.0) -> Dict:
    """取数封装 (端点/worker 共用): 日K + 快照 + 龙虎榜 → audit_execution.

    取数全缓存优先 (K线走 data_fetcher 缓存链, 龙虎榜只读 6h 缓存不打穿
    源), 三源**并行**取数 + deadline join = timeout (09-13 核查修: 原单线程
    串行三源冷启动 ~3-6s > 5-8s 超时 → 审计全空转 = 无最新数据分析;
    并行后单源并发 max≈4s, timeout 才真正有效) — 最坏跳过审计,
    绝不拖决策/端点 (同 llm_sentiment.analyze_with_fallback 45s 预算先例)。
    """
    import threading
    import time

    box: Dict = {}
    lk = threading.Lock()

    def _job(key, fn):
        try:
            v = fn()
            with lk:
                box[key] = v
        except Exception as e:
            with lk:
                box['err'] = (box.get('err') or '') + \
                             f' {key}: {type(e).__name__}'

    def _gather():
        try:
            from modules.data_fetcher import StockDataFetcher
            from modules.dragon_tiger_fetcher import fetch_dragon_tiger_history
            f = StockDataFetcher()
            # ── 三源并行 (09-10 并行先例, 09-13 修复重演) ──
            ths = [threading.Thread(target=_job, args=j, daemon=True)
                   for j in (
                       ('kl', lambda: f.get_kline_data(
                           stock_code, 'daily', 5) or []),
                       ('snap', lambda: f.get_stock_info(stock_code) or {}),
                       ('dt', lambda: fetch_dragon_tiger_history(
                           stock_code, limit=2, use_cache=True) or []))]
            t0 = time.time()
            for th in ths:
                th.start()
            for th in ths:
                th.join(timeout=max(0.05, timeout - (time.time() - t0)))
            if any(th.is_alive() for th in ths):
                logger.debug(f"[ExecAudit] {stock_code} 审计取数超时 "
                             f"(>{timeout}s), 缺源诚实降级")
        except Exception as e:
            box['err'] = f' 聚合: {type(e).__name__}'
            logger.debug(f"[ExecAudit] {stock_code} gather 降级: {e}")

    t = threading.Thread(target=_gather, daemon=True)
    t.start()
    t.join(timeout=timeout + 0.5)      # 外圈余量 = 内圈 deadline (09-13)
    try:
        out = audit_execution(stock_code, decision, confidence,
                              snapshot=box.get('snap') or {},
                              klines=box.get('kl') or [],
                              dragon_tiger=box.get('dt') or [])
        if box.get('err'):
            out['note'] = f"部分数据缺失 ({box['err'][:60]}), 审计降级"
        return out
    except Exception as e:
        logger.error(f"[ExecAudit] 审计失败 (跳过): {e}")
        return {'version': 'exec_audit_v1', 'skipped': True,
                'reason': str(e)[:120]}
