#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
regulatory_tags.py — 龙虎榜四类异常代理标签 (P2-12a, 2026-09-15)

A股 2026 监管深化细化高频监控 (吴清: 四类异常行为)。本模块基于**公开榜单
字段**给出四类异常的代理标签 — 诚实定位: 这是可得信号的量级代理, 不是监管
认定, 不可作合规结论依据。

四类: 虚假申报 / 拉抬打压 / 涨跌幅偏离 / 严重异常波动。
纯函数, 输入榜单记录 dict, 输出 [{tag, hit, evidence}]。
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger('stock_analyzer.regulatory_tags')

try:
    from modules.tradability import limit_threshold_for
except ImportError:
    def limit_threshold_for(code: str, name: Optional[str] = None) -> float:
        return 0.10


def _f(x, default=0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def tag_dragon_tiger(record: Dict) -> List[Dict]:
    """
    对一条龙虎榜记录打四类异常代理标签。

    Args:
        record: 榜单记录, 可用字段 (缺失即跳过对应标签):
            code, name, pct_change (小数), turnover (换手率小数),
            high/low/prev_close (算振幅), net_buy, amount (成交额),
            consecutive_days (连续上榜天数), reason (上榜原因文本)

    Returns:
        [{tag, hit: bool, evidence: str}] — 四类各一条, 未命中也列出 (可审计)
    """
    out = []
    try:
        code = str(record.get('code', ''))
        name = record.get('name')
        pct = _f(record.get('pct_change'))
        threshold = limit_threshold_for(code, name)

        # 1. 涨跌幅偏离: 涨幅超板块限制八成以上 (偏离基准/指数的代理)
        dev_hit = abs(pct) >= threshold * 0.8
        out.append({'tag': '涨跌幅偏离', 'hit': bool(dev_hit),
                    'evidence': f'涨跌 {pct:+.2%} vs 板块幅度 ±{threshold:.0%}'})

        # 2. 拉抬/打压: 净买占比极端 且 同向价格大幅移动
        net_buy, amount = _f(record.get('net_buy')), _f(record.get('amount'))
        ratio = (net_buy / amount) if amount > 0 else 0.0
        pump = ratio >= 0.3 and pct >= threshold * 0.8
        dump = ratio <= -0.3 and pct <= -threshold * 0.8
        out.append({'tag': '拉抬打压', 'hit': bool(pump or dump),
                    'evidence': (f'净买占比 {ratio:+.1%}, 涨跌 {pct:+.2%}'
                                 if (pump or dump) else f'净买占比 {ratio:+.1%} 未达极端')})

        # 3. 严重异常波动: 连续上榜 ≥3 天 或 涨幅超 1.5 倍板块幅度
        consec = int(_f(record.get('consecutive_days')))
        severe = consec >= 3 or abs(pct) >= threshold * 1.5
        out.append({'tag': '严重异常波动', 'hit': bool(severe),
                    'evidence': f'连续上榜 {consec} 天, 涨跌 {pct:+.2%}'})

        # 4. 虚假申报代理: 换手极高 + 振幅极大 (频繁申报拉抬的量级特征)
        turnover = _f(record.get('turnover'))
        amp = None
        try:
            prev = _f(record.get('prev_close'))
            if prev > 0:
                amp = (_f(record.get('high')) - _f(record.get('low'))) / prev
        except Exception:
            amp = None
        fake = turnover >= 0.30 and (amp is not None and amp >= 0.15)
        out.append({'tag': '虚假申报(代理)', 'hit': bool(fake),
                    'evidence': f'换手 {turnover:.1%}, 振幅 {amp:.1%}' if amp is not None
                    else f'换手 {turnover:.1%}, 振幅未知'})
    except Exception as e:
        logger.error(f"[RegTags] 打标失败: {e}", exc_info=True)
        out.append({'tag': 'error', 'hit': False, 'evidence': str(e)})
    return out
