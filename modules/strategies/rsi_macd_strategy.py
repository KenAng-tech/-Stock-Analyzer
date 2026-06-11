#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
示例策略 — P3 用于 EventDrivenBacktester

RSI + MACD + 均线 多因子策略
"""

import numpy as np
from typing import Dict, List
from modules.event_backtester import (
    StrategyBase, Order, OrderType, Side, Position
)


class RSIMACDStrategy(StrategyBase):
    """
    RSI + MACD + 均线策略

    入场:
    - 价格 > MA20 (趋势过滤)
    - RSI < 35 (超卖)
    - MACD 柱状图 > 0 (动能转正)

    出场:
    - RSI > 65 (超买)
    - MACD 柱状图 < 0 (动能转负)
    - 止损: 跌破入场价 5%
    - 止盈: 盈利 15%
    """

    def __init__(self, rsi_period: int = 14,
                 buy_rsi: float = 35,
                 sell_rsi: float = 65,
                 stop_loss_pct: float = 0.05,
                 take_profit_pct: float = 0.15,
                 position_pct: float = 0.8):
        self.rsi_period = rsi_period
        self.buy_rsi = buy_rsi
        self.sell_rsi = sell_rsi
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.position_pct = position_pct
        self.entry_price = None

    def _calculate_rsi(self, closes: List[float], period: int = 14) -> float:
        if len(closes) < period + 1:
            return 50.0
        deltas = np.diff(closes[-(period + 1):])
        gains = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
        losses = abs(np.mean(deltas[deltas < 0])) if np.any(deltas < 0) else 0.001
        rs = gains / losses
        return 100 - (100 / (1 + rs))

    def _calculate_macd_histogram(self, closes: List[float]) -> float:
        if len(closes) < 35:
            return 0.0

        def ema(data, period):
            if len(data) < period:
                return float(np.mean(data))
            m = 2.0 / (period + 1)
            r = float(data[0])
            for p in data[1:]:
                r = (p - r) * m + r
            return r

        ema12 = ema(closes, 12)
        ema26 = ema(closes, 26)
        macd_line = ema12 - ema26

        # 计算 signal line
        macd_values = []
        for i in range(26, len(closes)):
            e12 = ema(closes[:i+1], 12)
            e26 = ema(closes[:i+1], 26)
            macd_values.append(e12 - e26)

        if len(macd_values) >= 9:
            signal = ema(macd_values[-9:], 9)
            return macd_values[-1] - signal
        return macd_line * 0.1

    def on_bar(self, bar: Dict, positions: Dict[str, Position],
               equity: float) -> List[Order]:
        orders = []
        code = bar.get('stock_code', 'unknown')
        close = bar.get('close', 0)
        pos = positions.get(code)

        # 获取历史收盘价
        closes = [float(bar.get('close', 0))]
        # 需要从完整数据获取，这里简化

        # 计算指标
        rsi = bar.get('rsi', 50)
        macd_hist = bar.get('macd_histogram', 0)
        ma20 = bar.get('ma20', close)

        if pos is None:
            pos = Position(code)
            positions[code] = pos

        # ── 入场逻辑 ──
        if pos.quantity == 0 and self.entry_price is None:
            if close > ma20 and rsi < self.buy_rsi and macd_hist > 0:
                # 买入: 用指定比例资金
                buy_value = equity * self.position_pct
                shares = int(buy_value / close / 100) * 100
                if shares >= 100:
                    orders.append(Order(
                        stock_code=code,
                        side=Side.BUY,
                        order_type=OrderType.MARKET,
                        quantity=shares,
                        price=close,
                        strategy='RSI_MACD',
                        timestamp=bar.get('date', ''),
                    ))
                    self.entry_price = close

        # ── 出场逻辑 ──
        elif pos.quantity > 0 and self.entry_price is not None:
            pnl_pct = (close - self.entry_price) / self.entry_price

            # 止损
            if pnl_pct < -self.stop_loss_pct:
                orders.append(Order(
                    stock_code=code,
                    side=Side.SELL,
                    order_type=OrderType.MARKET,
                    quantity=pos.quantity,
                    price=close,
                    strategy='RSI_MACD_STOP_LOSS',
                    timestamp=bar.get('date', ''),
                ))
                self.entry_price = None

            # 止盈
            elif pnl_pct > self.take_profit_pct:
                orders.append(Order(
                    stock_code=code,
                    side=Side.SELL,
                    order_type=OrderType.MARKET,
                    quantity=pos.quantity,
                    price=close,
                    strategy='RSI_MACD_TAKE_PROFIT',
                    timestamp=bar.get('date', ''),
                ))
                self.entry_price = None

            # RSI 超买卖出
            elif rsi > self.sell_rsi and macd_hist < 0:
                orders.append(Order(
                    stock_code=code,
                    side=Side.SELL,
                    order_type=OrderType.MARKET,
                    quantity=pos.quantity,
                    price=close,
                    strategy='RSI_MACD',
                    timestamp=bar.get('date', ''),
                ))
                self.entry_price = None

        return orders

    def on_timer(self, date: str, positions: Dict[str, Position]):
        pass
