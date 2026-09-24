# -*- coding: utf-8 -*-
"""
portfolio_store.py — 真实组合持仓读取层 (2026-09-09)

数据文件: ~/.stock_analyzer_portfolio.json (个人配置, 不进 git;
与 ~/.stock_analyzer_llm_config.db 同位置同风格)

提供:
    get_total_value()  — 组合总金额 (含现金), 无文件 → 0.0
    get_cash()         — 现金余额
    get_holding(code)  — 该股真实持仓 {cost, qty, name}, 无 → None
    get_snapshot()     — 全持仓快照 (含现价/市值/浮亏/权重, 供组合层 API 消费)

线程安全: mtime 缓存 + 锁 (quant/cvar 多端点并发读)。
"""

import json
import os
import threading
from datetime import datetime

from modules.logger import logger

_PORTFOLIO_PATH = os.path.expanduser("~/.stock_analyzer_portfolio.json")


def _bare(code: str) -> str:
    """代码归一化: sz300620 / 688981.SH / sh688981 / 688981 → 6 位裸码"""
    c = (code or '').strip().lower().replace('.', '').replace('_', '')
    for p in ('sh', 'sz', 'bj'):
        # 前缀 (sz300620/sh688981) 和后缀 (688981.SH/300620.sz) 两种格式
        if c.startswith(p) and len(c) > len(p):
            return c[len(p):]
        if c.endswith(p) and len(c) > len(p):
            return c[:-len(p)]
    return c


class PortfolioStore:
    """真实持仓读取层 (文件只读, mtime 缓存)"""

    def __init__(self):
        self._lock = threading.Lock()
        self._data = None
        self._mtime = 0.0

    def _load(self):
        """mtime 变化才重读 JSON; 读取失败返回旧缓存 (不吞错, 记日志)"""
        try:
            mtime = os.path.getmtime(_PORTFOLIO_PATH)
            with self._lock:
                if self._data is not None and mtime == self._mtime:
                    return self._data
                with open(_PORTFOLIO_PATH, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self._data = data
                self._mtime = mtime
                return data
        except FileNotFoundError:
            return None
        except Exception as e:
            logger.error(f"[PortfolioStore] 持仓文件读取失败: {e}")
            return self._data

    def get_total_value(self) -> float:
        """组合总金额 (含现金); 无配置 → 0.0"""
        data = self._load()
        if not data:
            return 0.0
        return float(data.get('total_value', 0) or 0)

    def get_cash(self) -> float:
        data = self._load()
        if not data:
            return 0.0
        return float(data.get('cash', 0) or 0)

    def get_holding(self, code: str):
        """按代码查真实持仓 (归一化匹配); 无 → None"""
        data = self._load()
        if not data:
            return None
        bare = _bare(code)
        for p in data.get('positions', []):
            if _bare(p.get('code', '')) == bare:
                return dict(p)
        return None

    def get_snapshot(self, price_lookup=None):
        """
        真实持仓快照 (替代 mock 组合).

        Args:
            price_lookup: code → 现价 的 callable (由调用方注入, 本层不耦合取数)

        Returns:
            {total_value, cash, positions: [...]} 或 None (无持仓)
            每行字段兼容 /api/quant/positions 消费端:
            symbol/name/price/unrealized_pnl/side/position_pct +
            扩展: cost/qty/value/weight_pct/pl_pct/pl_usd/timestamp
        """
        data = self._load()
        if not data:
            return None
        total = float(data.get('total_value', 0) or 1_000_000.0)
        ts = datetime.now().isoformat()
        rows = []
        for p in data.get('positions', []):
            code = str(p.get('code', ''))
            cost = float(p.get('cost', 0) or 0)
            qty = int(p.get('qty', 0) or 0)
            if cost <= 0 or qty <= 0 or not code:
                continue
            price = None
            if price_lookup:
                try:
                    price = price_lookup(code)
                except Exception as e:
                    logger.error(f"[PortfolioStore] 取现价失败 {code}: {e}")
            if not price or price <= 0:
                price = cost  # 现价不可得 → 用成本近似 (降级显示, 不断链)
            value = round(price * qty, 2)
            pl_usd = round((price - cost) * qty, 2)
            weight_pct = round(value / total * 100, 2) if total > 0 else 0.0
            rows.append({
                'symbol': code.upper(),
                'name': p.get('name', ''),
                'price': round(price, 3),
                'cost': cost,
                'qty': qty,
                'value': value,
                'weight_pct': weight_pct,
                'position_pct': weight_pct,          # 兼容旧 mock 字段
                'pl_usd': pl_usd,
                'pl_pct': round((price - cost) / cost * 100, 2),
                'unrealized_pnl': pl_usd,            # 兼容旧 mock 字段 (USD 位置: 金额)
                'side': 'LONG',
                'kline_score': None,
                'trend': '—',
                'timestamp': ts,
            })
        return {
            'total_value': total,
            'cash': float(data.get('cash', 0) or 0),
            'positions': rows,
            'timestamp': ts,
        }


portfolio_store = PortfolioStore()
