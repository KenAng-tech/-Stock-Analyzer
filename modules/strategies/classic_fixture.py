#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
经典策略 fixture 库 (2026-09-11 P2)

来源: vnpy_ctastrategy 示例策略思想翻译 (DualThrust/Donchian/ATR_RSI/KingKeltner),
MIT 思想级借鉴 — 零 vnpy/talib import, 纯 python/numpy, 无 bars 注入依赖
(rolling 窗口工厂内部自持, 与 services 链 enhanced_strategy 的预注入形互补)。

形 = WalkForwardBacktester.strategy_func:
    func(bar: Dict, position: int, capital: float) -> 'buy'|'sell'|'hold'
每次 run 用独立工厂实例 (rolling 状态独立, 不跨 run 污染)。

用途:
  ① tests 回归基准链 — 替换 test_backtester 的 lambda: hold 死 mock
     (恒 hold 只 smoke 引擎不烟策略链, 09-08/09-10 教训)
  ② 喂 statistical_overfit_check (DSR + PBO/CSCV + t 检验) 真激活:
     恒 skipped 根因 = low_activity (非零收益占比 <10%), 活跃 fixture 喂链
     后三块统计诊断才首次获得真实输入 (PBO 需活跃策略才非假信号)
  ③ Optuna 调参 / 多候选对比的候选池 (23:00 调度链预留, 未接)

A 股语义: 无做空 — 空仓→'buy' 决策, 持仓→'sell' 退出 (无 short/cover)。

运行: python -m unittest tests.test_backtester
"""

from collections import deque
import numpy as np


def make_dual_thrust(n: int = 4, k: float = 0.5):
    """DualThrust 区间突破 (vnpy DualThrustStrategy 项目化)

    轨: range = max(窗口最高-昨收, 昨收-窗口最低); 上轨=open+k*range 买,
    下轨=open-k*range 空 → A 股化: 下轨仅作持仓退出信号。
    """
    hist = deque(maxlen=n)

    def strategy(bar, position, capital):
        out = 'hold'
        if len(hist) >= n:
            hh = max(h for h, l in hist)
            ll = min(l for h, l in hist)
            lc = hist[-1][1]
            rng = max(hh - lc, lc - ll)
            o, c = bar['open'], bar['close']
            if position == 0 and c > o + k * rng:
                out = 'buy'
            elif position > 0 and c < o - k * rng:
                out = 'sell'
        hist.append((bar['high'], bar['low']))
        return out

    return strategy


def make_donchian(entry: int = 20, exit_: int = 10):
    """唐奇安通道趋势跟随 (vnpy turtle_signal_strategy 项目化, 标 SSA/SSB 语义)

    买: 收盘创 entry 日新高 (突破); 卖: 持仓创 exit_ 日新低 (回落离场)。
    """
    highs = deque(maxlen=entry + 1)
    lows = deque(maxlen=exit_ + 1)

    def strategy(bar, position, capital):
        h, l, c = bar['high'], bar['low'], bar['close']
        out = 'hold'
        if len(highs) >= entry and len(lows) >= exit_:
            if position == 0 and c > max(list(highs)[:entry]):
                out = 'buy'
            elif position > 0 and c < min(list(lows)[:exit_]):
                out = 'sell'
        highs.append(h)
        lows.append(l)
        return out

    return strategy


def make_atr_rsi(period: int = 14, buy_level: float = 30.0, sell_level: float = 70.0):
    """RSI 均值回归 (vnpy atr_rsi_strategy 项目化, 无做空化)

    买: RSI < buy_level (超卖); 卖: RSI > sell_level (超买)。
    RSI 用 Wilder 简化 (简单均值版, 同 test_backtester._generate_klines 链形)。
    """
    closes = deque(maxlen=period + 1)

    def strategy(bar, position, capital):
        c = bar['close']
        out = 'hold'
        if len(closes) >= period + 1:
            arr = np.array(closes)
            deltas = np.diff(arr)
            gains = float(np.mean(deltas[deltas > 0])) if np.any(deltas > 0) else 0.0
            losses = float(abs(np.mean(deltas[deltas < 0]))) if np.any(deltas < 0) else 1e-9
            rsi = 100.0 - 100.0 / (1.0 + gains / losses)
            if position == 0 and rsi < buy_level:
                out = 'buy'
            elif position > 0 and rsi > sell_level:
                out = 'sell'
        closes.append(c)
        return out

    return strategy


def make_keltner(period: int = 20, mult: float = 2.0):
    """Keltner 通道突破 (vnpy king_keltner_strategy 项目化, A 股无做空)

    买: 持仓外 close > EMA20 + mult*ATR20 (上轨突破); 卖: 持仓 close < EMA20 (中轨离场)。
    ATR 用简化 TR (high-low / |high-pc| / |low-pc| 均值, 20 窗)。
    """
    window = deque(maxlen=period + 1)

    def strategy(bar, position, capital):
        out = 'hold'
        if len(window) >= period:
            hist = list(window)
            closes = [b[2] for b in hist]
            ema20 = float(np.mean(closes[-period:]))   # 项目简化: 均值近似 EMA 中段
            trs, prev = [], closes[0]
            for b in hist[-period:]:
                trs.append(max(b[0] - b[1], abs(b[0] - prev), abs(b[1] - prev)))
                prev = b[2]
            atr = float(np.mean(trs)) if trs else 0.0
            if position == 0 and bar['close'] > ema20 + mult * atr:
                out = 'buy'
            elif position > 0 and bar['close'] < ema20:
                out = 'sell'
        window.append((bar['high'], bar['low'], bar['close']))
        return out

    return strategy


ALL_FIXTURES = {
    'dual_thrust': lambda: make_dual_thrust(4, 0.5),
    'donchian': lambda: make_donchian(20, 10),
    'atr_rsi': lambda: make_atr_rsi(14, 30.0, 70.0),
    'keltner': lambda: make_keltner(20, 2.0),
}
