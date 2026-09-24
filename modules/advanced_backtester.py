#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
DEPRECATED — 已弃用，请使用 event_backtester.py + walkforward_backtester.py

此文件仅为向后兼容保留。
所有新的代码应使用:
  - 批量回测: from modules.walkforward_backtester import WalkForwardBacktester
  - 事件驱动: from modules.event_backtester import EventDrivenBacktester

保留时间: 2026-07-01 之后 3 个月
"""

import warnings

warnings.warn(
    "[DEPRECATED] modules.advanced_backtester 已弃用，请使用 modules.event_backtester 或 modules.walkforward_backtester",
    DeprecationWarning,
    stacklevel=2,
)

from modules.event_backtester import EventDrivenBacktester
from modules.walkforward_backtester import WalkForwardBacktester, TransactionCostModel

# ── 向后兼容别名 ─────────────────────────────────────────────
# dashboard_api.py 使用的简化接口


class BacktestEngine:
    """BacktestEngine 简化版 (兼容 dashboard_api.py)"""

    def __init__(self, initial_capital: float = 1_000_000,
                 cost_model=None):
        self.initial_capital = initial_capital
        self.cost_model = cost_model or TransactionCostModel()

    def calculate_buy(self, price: float, shares: int) -> Dict:
        """计算买入成本"""
        trade_value = price * shares
        commission = max(trade_value * self.cost_model.commission_rate, 5)
        slippage = trade_value * self.cost_model.slippage_bps / 10000
        total = trade_value + commission + slippage
        return {'total': round(total, 2), 'total_cost': round(total, 2)}

    def calculate_sell(self, price: float, shares: int) -> Dict:
        """计算卖出收入"""
        trade_value = price * shares
        commission = max(trade_value * self.cost_model.commission_rate, 5)
        stamp_tax = trade_value * self.cost_model.stamp_tax
        slippage = trade_value * self.cost_model.slippage_bps / 10000
        total = trade_value - commission - stamp_tax - slippage
        return {'total': round(total, 2), 'total_cost': round(commission + stamp_tax + slippage, 2)}


class BacktestResult:
    """BacktestResult 简化版 (兼容 dashboard_api.py)"""

    def __init__(self):
        self.trades: list = []
        self.equity_curve: list = []
        self.positions: dict = {}

    def calculate_metrics(self) -> Dict:
        if not self.equity_curve:
            return {}
        equities = [e[1] for e in self.equity_curve]
        n_days = len(equities) - 1
        total_return = (equities[-1] - equities[0]) / equities[0]
        daily_returns = [(equities[i] - equities[i - 1]) / equities[i - 1]
                         for i in range(1, len(equities))]
        annual_return = (1 + total_return) ** (252 / max(n_days, 1)) - 1
        annual_vol = (sum(r ** 2 for r in daily_returns) / len(daily_returns)) ** 0.5 * (252 ** 0.5) if daily_returns else 0
        sharpe = (annual_return - 0.02) / annual_vol if annual_vol > 0 else 0
        max_dd = 0
        peak = equities[0]
        for eq in equities:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak
            if dd > max_dd:
                max_dd = dd
        return {
            'total_return': round(total_return, 4),
            'annual_return': round(annual_return, 4),
            'annual_volatility': round(annual_vol, 4),
            'sharpe_ratio': round(sharpe, 3),
            'max_drawdown': round(max_dd, 4),
            'n_days': n_days,
            'final_equity': round(equities[-1], 2),
        }


__all__ = ['EventDrivenBacktester', 'WalkForwardBacktester',
           'BacktestEngine', 'BacktestResult', 'TransactionCostModel']
