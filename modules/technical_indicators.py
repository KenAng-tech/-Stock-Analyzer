#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
技术指标计算模块 — 真实 RSI / MACD / KDJ / 布林带 / ATR 计算

P0 修复 (2026-07-01):
  替代 analysis_engine.py 中使用 change_pct 估算虚假技术指标的做法。
  所有指标基于真实 K 线数据计算。

包含:
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - KDJ (Stochastic Oscillator)
  - Bollinger Bands
  - ATR (Average True Range)
  - EMA / SMA 均线
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────
# 辅助函数: 均线计算
# ─────────────────────────────────────────────

def sma(values: np.ndarray, period: int) -> np.ndarray:
    """简单移动平均线 (SMA)"""
    if len(values) < period:
        return np.full_like(values, np.nan)
    result = np.full(len(values), np.nan)
    for i in range(period - 1, len(values)):
        result[i] = np.mean(values[i - period + 1: i + 1])
    return result


def ema(values: np.ndarray, period: int) -> np.ndarray:
    """指数移动平均线 (EMA)"""
    if len(values) == 0:
        return np.array([])
    result = np.full(len(values), np.nan)
    multiplier = 2.0 / (period + 1)
    # 第一个值用 SMA 初始化
    if len(values) >= period:
        result[period - 1] = np.mean(values[:period])
        # 后续用 EMA 公式
        for i in range(period, len(values)):
            result[i] = (values[i] - result[i - 1]) * multiplier + result[i - 1]
    else:
        # 数据不足 period，用首个值初始化
        result[0] = values[0]
        for i in range(1, len(values)):
            result[i] = (values[i] - result[i - 1]) * multiplier + result[i - 1]
    return result


# ─────────────────────────────────────────────
# RSI (Relative Strength Index)
# ─────────────────────────────────────────────

def calculate_rsi(closes: np.ndarray, period: int = 14) -> float:
    """
    计算真实 RSI 值

    Args:
        closes: 收盘价序列 (至少 period+1 个数据点)
        period: 计算周期，默认 14

    Returns:
        RSI 值 (0-100)，数据不足返回 50.0
    """
    if len(closes) < period + 1:
        return 50.0

    deltas = np.diff(closes[-period - 1:])
    gains = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0.0
    losses = abs(np.mean(deltas[deltas < 0])) if np.any(deltas < 0) else 0.001

    rs = gains / losses
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return float(np.clip(rsi, 0, 100))


def rsi_signal(rsi: float) -> str:
    """根据 RSI 值生成信号"""
    if rsi < 20:
        return '严重超卖'
    elif rsi < 30:
        return '超卖区'
    elif rsi > 80:
        return '严重超买'
    elif rsi > 70:
        return '超买区'
    elif rsi > 55:
        return '偏多'
    elif rsi > 45:
        return '中性'
    else:
        return '偏空'


# ─────────────────────────────────────────────
# MACD (Moving Average Convergence Divergence)
# ─────────────────────────────────────────────

def calculate_macd(closes: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
    """
    计算真实 MACD 指标

    Args:
        closes: 收盘价序列
        fast: 快线周期，默认 12
        slow: 慢线周期，默认 26
        signal: 信号线周期，默认 9

    Returns:
        {
            'macd': float,          # DIF 线 (快线 - 慢线)
            'signal': float,        # DEA 线 (信号线)
            'histogram': float,     # MACD 柱 (DIF - DEA) * 2
            'signal_type': str,     # '金叉' / '死叉' / '延续多' / '延续空'
            'divergence': str,     # '底背离' / '顶背离' / '无'
        }
    """
    if len(closes) < slow + signal + 10:
        return {'macd': 0.0, 'signal': 0.0, 'histogram': 0.0, 'signal_type': '数据不足', 'divergence': '无'}

    ema_fast = ema(closes, fast)
    ema_slow = ema(closes, slow)
    dif = ema_fast - ema_slow  # DIF 线
    dea = ema(dif, signal)     # DEA 线
    histogram = (dif - dea) * 2  # MACD 柱

    # 判断金叉/死叉
    if len(dif) >= 2 and len(dea) >= 2:
        if dif[-1] > dea[-1] and dif[-2] <= dea[-2]:
            signal_type = '金叉'
        elif dif[-1] < dea[-1] and dif[-2] >= dea[-2]:
            signal_type = '死叉'
        elif dif[-1] > dea[-1]:
            signal_type = '延续多'
        else:
            signal_type = '延续空'
    else:
        signal_type = '数据不足'

    # 简单的背离检测
    divergence = '无'
    if len(closes) >= 60:
        price_low_20 = np.min(closes[-20:])
        price_high_20 = np.max(closes[-20:])
        dif_low_recent = np.min(dif[-20:]) if len(dif) >= 20 else 0
        dif_high_recent = np.max(dif[-20:]) if len(dif) >= 20 else 0

        if len(closes) >= 60:
            price_low_prev = np.min(closes[-60:-20])
            price_high_prev = np.max(closes[-60:-20])
            dif_low_prev = np.min(dif[-60:-20]) if len(dif) >= 60 else 0
            dif_high_prev = np.max(dif[-60:-20]) if len(dif) >= 60 else 0
            if price_low_20 < price_low_prev and dif_low_recent > dif_low_prev:
                divergence = '底背离'
            elif price_high_20 > price_high_prev and dif_high_recent < dif_high_prev:
                divergence = '顶背离'

    return {
        'macd': round(float(dif[-1]), 4),
        'signal': round(float(dea[-1]), 4),
        'histogram': round(float(histogram[-1]), 4),
        'signal_type': signal_type,
        'divergence': divergence,
    }


# ─────────────────────────────────────────────
# KDJ (Stochastic Oscillator)
# ─────────────────────────────────────────────

def calculate_kdj(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                  n: int = 9, m1: int = 3, m2: int = 3) -> Dict:
    """
    计算真实 KDJ 指标

    Args:
        highs: 最高价序列
        lows: 最低价序列
        closes: 收盘价序列
        n: RSV 计算周期，默认 9
        m1: K 值平滑因子，默认 3
        m2: D 值平滑因子，默认 3

    Returns:
        {
            'k': float,     # K 值
            'd': float,     # D 值
            'j': float,     # J 值
            'signal': str,  # '超卖' / '超买' / '金叉' / '死叉' / '中性'
        }
    """
    if len(closes) < n + 1:
        return {'k': 50.0, 'd': 50.0, 'j': 50.0, 'signal': '数据不足'}

    # 计算 RSV = (Close - Low_n) / (High_n - Low_n) * 100
    lowest_n = np.min(lows[-n:])
    highest_n = np.max(highs[-n:])

    if highest_n == lowest_n:
        rsv = 50.0
    else:
        rsv = (closes[-1] - lowest_n) / (highest_n - lowest_n) * 100

    # 初始化 K, D (使用 SMA)
    k = rsv
    d = rsv

    # 用滚动窗口计算更精确的 K, D
    rsv_series = []
    for i in range(n, len(closes)):
        low_n = np.min(lows[i - n: i + 1])
        high_n = np.max(highs[i - n: i + 1])
        if high_n == low_n:
            rsv_series.append(50.0)
        else:
            rsv_series.append((closes[i] - low_n) / (high_n - low_n) * 100)

    if len(rsv_series) >= m1:
        k = rsv_series[-1]
        for i in range(len(rsv_series) - 2, max(len(rsv_series) - m1 - 1, -1), -1):
            k = (m1 - 1) * k / m1 + rsv_series[i] / m1

    if len(rsv_series) >= m2:
        d = k
        for i in range(len(rsv_series) - 2, max(len(rsv_series) - m2 - 1, -1), -1):
            d = (m2 - 1) * d / m2 + rsv_series[i] / m2

    j = 3 * k - 2 * d

    # 信号判断
    if k < 20 and d < 20:
        signal = '超卖'
    elif k > 80 and d > 80:
        signal = '超买'
    elif len(rsv_series) >= 2:
        prev_k = (m1 - 1) * ((m1 - 1) * rsv_series[-2] / m1 + rsv_series[-2] / m1) / m1 + rsv_series[-2] / m1 if len(rsv_series) >= 2 else 50
        if k > d and (prev_k if len(rsv_series) >= 2 else 50) <= d:
            signal = 'KDJ 金叉'
        elif k < d and (prev_k if len(rsv_series) >= 2 else 50) >= d:
            signal = 'KDJ 死叉'
        else:
            signal = '中性'
    else:
        signal = '中性'

    return {
        'k': round(float(k), 2),
        'd': round(float(d), 2),
        'j': round(float(j), 2),
        'signal': signal,
    }


# ─────────────────────────────────────────────
# Bollinger Bands (布林带)
# ─────────────────────────────────────────────

def calculate_bollinger(closes: np.ndarray, period: int = 20, std_dev_mult: float = 2.0) -> Dict:
    """
    计算布林带

    Args:
        closes: 收盘价序列
        period: 中轨周期，默认 20
        std_dev_mult: 标准差倍数，默认 2.0

    Returns:
        {
            'upper': float,   # 上轨
            'middle': float,  # 中轨 (SMA)
            'lower': float,   # 下轨
            'width': float,   # 带宽 (upper - lower) / middle
            'position': str,  # 位置: 接近上轨/中轨附近/接近下轨
        }
    """
    if len(closes) < period:
        return {'upper': 0, 'middle': 0, 'lower': 0, 'width': 0, 'position': '数据不足'}

    middle = np.mean(closes[-period:])
    std = np.std(closes[-period:])
    upper = middle + std_dev_mult * std
    lower = middle - std_dev_mult * std
    width = (upper - lower) / middle if middle > 0 else 0

    current = closes[-1]
    if upper > lower and upper - lower > 0:
        pct = (current - lower) / (upper - lower)
        if pct > 0.8:
            position = '接近上轨'
        elif pct > 0.6:
            position = '偏上'
        elif pct > 0.4:
            position = '中轨附近'
        elif pct > 0.2:
            position = '偏下'
        else:
            position = '接近下轨'
    else:
        position = '数据不足'

    return {
        'upper': round(float(upper), 2),
        'middle': round(float(middle), 2),
        'lower': round(float(lower), 2),
        'width': round(float(width), 4),
        'position': position,
    }


# ─────────────────────────────────────────────
# ATR (Average True Range)
# ─────────────────────────────────────────────

def calculate_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                  period: int = 14) -> float:
    """
    计算 ATR (Average True Range)

    Args:
        highs: 最高价序列
        lows: 最低价序列
        closes: 收盘价序列
        period: 计算周期，默认 14

    Returns:
        ATR 值，数据不足返回 0
    """
    if len(closes) < period + 1:
        return 0.0

    # 计算 True Range
    trs = []
    for i in range(1, len(closes)):
        high_low = highs[i] - lows[i]
        high_close = abs(highs[i] - closes[i - 1])
        low_close = abs(lows[i] - closes[i - 1])
        tr = max(high_low, high_close, low_close)
        trs.append(tr)

    if len(trs) < period:
        return float(np.mean(trs)) if trs else 0.0

    # 使用 EMA 计算 ATR
    atr = np.mean(trs[-period:])
    return float(atr)


# ─────────────────────────────────────────────
# 综合技术指标分析
# ─────────────────────────────────────────────

def comprehensive_technical_analysis(klines: List[Dict]) -> Dict:
    """
    综合技术分析 — 从 K 线数据计算所有技术指标

    Args:
        klines: K 线数据列表，每个元素包含 date/open/high/low/close/volume

    Returns:
        包含所有技术指标的综合分析结果
    """
    if not klines or len(klines) < 15:
        return {'error': 'K 线数据不足 (至少 15 根)'}

    closes = np.array([float(k['close']) for k in klines if k.get('close', 0) > 0])
    highs = np.array([float(k['high']) for k in klines if k.get('high', 0) > 0])
    lows = np.array([float(k['low']) for k in klines if k.get('low', 0) > 0])
    volumes = np.array([float(k['volume']) for k in klines if k.get('volume', 0) > 0])

    if len(closes) < 15:
        return {'error': '有效 K 线数据不足'}

    # 计算均线
    ma5 = float(np.mean(closes[-5:])) if len(closes) >= 5 else closes[-1]
    ma10 = float(np.mean(closes[-10:])) if len(closes) >= 10 else closes[-1]
    ma20 = float(np.mean(closes[-20:])) if len(closes) >= 20 else closes[-1]
    ma60 = float(np.mean(closes[-60:])) if len(closes) >= 60 else closes[-1]
    ma120 = float(np.mean(closes[-120:])) if len(closes) >= 120 else closes[-1]

    # 计算所有技术指标
    rsi = calculate_rsi(closes, 14)
    macd = calculate_macd(closes)
    kdj = calculate_kdj(highs, lows, closes)
    boll = calculate_bollinger(closes, 20, 2.0)
    atr = calculate_atr(highs, lows, closes, 14)

    # 成交量分析
    vol_ratio = volumes[-1] / np.mean(volumes[-20:]) if len(volumes) >= 20 and np.mean(volumes[-20:]) > 0 else 1.0
    vol_signal = '放量' if vol_ratio > 1.5 else '缩量' if vol_ratio < 0.7 else '平量'

    # 综合判断
    signals = []
    if rsi < 30:
        signals.append(('RSI', '超卖信号'))
    elif rsi > 70:
        signals.append(('RSI', '超买信号'))

    if macd['signal_type'] == '金叉':
        signals.append(('MACD', '金叉买入'))
    elif macd['signal_type'] == '死叉':
        signals.append(('MACD', '死叉卖出'))

    if kdj['signal'] == '超卖':
        signals.append(('KDJ', '超卖'))
    elif kdj['signal'] == '超买':
        signals.append(('KDJ', '超买'))

    if macd.get('divergence') == '底背离':
        signals.append(('MACD', '底背离'))
    elif macd.get('divergence') == '顶背离':
        signals.append(('MACD', '顶背离'))

    # 综合趋势判断
    bullish_count = sum(1 for s in signals if any(kw in s[1] for kw in ['超卖', '金叉', '底背离', '买入']))
    bearish_count = sum(1 for s in signals if any(kw in s[1] for kw in ['超买', '死叉', '顶背离', '卖出']))

    if bullish_count >= 2:
        overall_trend = '偏多'
    elif bearish_count >= 2:
        overall_trend = '偏空'
    else:
        overall_trend = '中性'

    return {
        'rsi': {
            'value': round(rsi, 2),
            'signal': rsi_signal(rsi),
            'period': 14,
        },
        'macd': macd,
        'kdj': kdj,
        'bollinger': boll,
        'atr': round(atr, 4),
        'moving_averages': {
            'ma5': round(ma5, 2),
            'ma10': round(ma10, 2),
            'ma20': round(ma20, 2),
            'ma60': round(ma60, 2),
            'ma120': round(ma120, 2),
        },
        'volume': {
            'ratio': round(vol_ratio, 2),
            'signal': vol_signal,
        },
        'summary': {
            'overall_trend': overall_trend,
            'bullish_signals': len(signals) - bearish_count,
            'bearish_signals': bearish_count,
            'all_signals': [{'indicator': s[0], 'signal': s[1]} for s in signals],
        },
    }
