#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
技术指标工具函数 — 统一 EMA/MACD/RSI/布林带实现

所有模块应从此处导入，避免重复实现。
"""

import numpy as np
from typing import Tuple


def ema(data: np.ndarray, period: int) -> float:
    """计算 EMA 最后一个值

    Args:
        data: 价格数组
        period: EMA 周期

    Returns:
        最后一个 EMA 值
    """
    if len(data) < period:
        return float(np.mean(data)) if len(data) > 0 else 0.0
    multiplier = 2.0 / (period + 1)
    result = float(data[0])
    for price in data[1:]:
        result = (price - result) * multiplier + result
    return result


def ema_array(data: np.ndarray, period: int) -> np.ndarray:
    """计算完整 EMA 数组

    Args:
        data: 价格数组
        period: EMA 周期

    Returns:
        完整 EMA 数组
    """
    if len(data) < period:
        return data.astype(float)
    multiplier = 2.0 / (period + 1)
    result = np.zeros(len(data))
    result[0] = float(data[0])
    for i in range(1, len(data)):
        result[i] = (data[i] - result[i - 1]) * multiplier + result[i - 1]
    return result


def rsi(data: np.ndarray, period: int = 14) -> float:
    """计算 RSI

    Args:
        data: 价格数组
        period: RSI 周期

    Returns:
        RSI 值 (0-100)
    """
    if len(data) < period + 1:
        return 50.0
    delta = np.diff(data)
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    if avg_loss < 1e-10:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def rsi_array(data: np.ndarray, period: int = 14) -> np.ndarray:
    """计算完整 RSI 数组

    Args:
        data: 价格数组
        period: RSI 周期

    Returns:
        RSI 数组
    """
    delta = np.diff(data)
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)

    avg_gain = np.zeros(len(data))
    avg_loss = np.zeros(len(data))
    avg_gain[:period] = np.mean(gains[:period])
    avg_loss[:period] = np.mean(losses[:period])

    for i in range(period, len(gains)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i]) / period

    rs = np.where(avg_loss < 1e-10, 1e10, avg_gain / avg_loss)
    return 100.0 - (100.0 / (1.0 + rs))


def macd_histogram(closes: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> float:
    """计算 MACD 柱状图 (MACD - Signal)

    Args:
        closes: 收盘价数组
        fast: 快速 EMA 周期
        slow: 慢速 EMA 周期
        signal: 信号线 EMA 周期

    Returns:
        MACD 柱状图最后一个值
    """
    if len(closes) < slow:
        return 0.0

    ema_fast = ema_array(closes, fast)
    ema_slow = ema_array(closes, slow)
    macd_line = ema_fast - ema_slow

    if len(macd_line) < signal:
        return float(macd_line[-1]) if len(macd_line) > 0 else 0.0

    signal_line = ema(macd_line[-signal * 2:], signal)  # 用最近 2 倍数据更稳定
    macd_value = macd_line[-1]
    return float(macd_value - signal_line)


def bollinger_bands(
    data: np.ndarray, period: int = 20, num_std: float = 2.0
) -> Tuple[float, float, float]:
    """计算布林带上中下轨

    Args:
        data: 价格数组
        period: 均线周期
        num_std: 标准差倍数

    Returns:
        (upper, middle, lower)
    """
    if len(data) < period:
        mean = float(np.mean(data)) if len(data) > 0 else 0.0
        return (mean, mean, mean)
    window = data[-period:]
    middle = float(np.mean(window))
    std = float(np.std(window))
    return (middle + num_std * std, middle, middle - num_std * std)


def atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
    """计算 ATR (Average True Range)

    Args:
        highs: 最高价数组
        lows: 最低价数组
        closes: 收盘价数组
        period: ATR 周期

    Returns:
        ATR 值
    """
    if len(highs) < 2:
        return 0.0
    trs = []
    for i in range(1, len(highs)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)
    if len(trs) < period:
        return float(np.mean(trs)) if trs else 0.0
    return float(np.mean(trs[-period:]))
