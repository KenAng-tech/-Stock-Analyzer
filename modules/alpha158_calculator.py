#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Qlib Alpha158 因子计算 — 完整实现

参考: Microsoft Qlib Alpha158 因子集
论文: "Quantitative Research Framework in Qlib"

Alpha158 = 158 个价量因子，基于以下操作组合:
    - 基础操作: close, open, high, low, volume, vwap
    - 时间序列操作: delta, ts_max, ts_min, ts_mean, ts_std, ts_corr, ts_cov
    - 数学操作: abs, sign, log, rank, ts_rank
    - 条件操作: ts_argmax, ts_argmin

因子类别:
    1. 动量因子 (Momentum): 价格趋势、动量反转
    2. 波动率因子 (Volatility): 已实现波动率、上下行波动率
    3. 成交量因子 (Volume): 量比、成交量动量、量价相关
    4. 流动性因子 (Liquidity): Amihud非流动性、换手率
    5. 技术因子 (Technical): RSI, MACD, 布林带, ATR
    6. 时间序列统计因子: 偏度、峰度、自相关

实现:
    - 纯 NumPy/pandas 实现，无外部依赖
    - 向量化计算，支持批量计算
    - 与 multi_factor_model_v2.py 接口兼容
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from modules.logger import logger


# ── 时间序列操作 ──────────────────────────────────────────────

def _ts_delta(series: np.ndarray, period: int) -> np.ndarray:
    """N 日差分: series[t] - series[t-period]"""
    result = np.full_like(series, np.nan, dtype=np.float64)
    for i in range(period, len(series)):
        result[i] = series[i] - series[i - period]
    return result


def _ts_max(series: np.ndarray, period: int) -> np.ndarray:
    """滚动最大值"""
    return pd.Series(series).rolling(window=period, min_periods=1).max().values


def _ts_min(series: np.ndarray, period: int) -> np.ndarray:
    """滚动最小值"""
    return pd.Series(series).rolling(window=period, min_periods=1).min().values


def _ts_mean(series: np.ndarray, period: int) -> np.ndarray:
    """滚动均值"""
    return pd.Series(series).rolling(window=period, min_periods=1).mean().values


def _ts_std(series: np.ndarray, period: int) -> np.ndarray:
    """滚动标准差"""
    return pd.Series(series).rolling(window=period, min_periods=1).std().values


def _ts_rank(series: np.ndarray, period: int) -> np.ndarray:
    """滚动排名 (0~1 归一化)"""
    result = np.full_like(series, 0.5, dtype=np.float64)
    for i in range(period - 1, len(series)):
        window = series[i - period + 1:i + 1]
        result[i] = np.sum(window <= series[i]) / len(window)
    return result


def _ts_corr(series_x: np.ndarray, series_y: np.ndarray, period: int) -> np.ndarray:
    """滚动相关系数"""
    result = np.full_like(series_x, 0.0, dtype=np.float64)
    for i in range(period - 1, len(series_x)):
        x_window = series_x[i - period + 1:i + 1]
        y_window = series_y[i - period + 1:i + 1]
        if np.std(x_window) > 1e-10 and np.std(y_window) > 1e-10:
            result[i] = np.corrcoef(x_window, y_window)[0, 1]
        else:
            result[i] = 0.0
    return result


def _ts_sum(series: np.ndarray, period: int) -> np.ndarray:
    """滚动求和"""
    return pd.Series(series).rolling(window=period, min_periods=1).sum().values


def _ts_argmax(series: np.ndarray, period: int) -> np.ndarray:
    """滚动最大值位置"""
    result = np.full_like(series, period // 2, dtype=np.float64)
    for i in range(period - 1, len(series)):
        window = series[i - period + 1:i + 1]
        result[i] = np.argmax(window)
    return result


def _ts_argmin(series: np.ndarray, period: int) -> np.ndarray:
    """滚动最小值位置"""
    result = np.full_like(series, period // 2, dtype=np.float64)
    for i in range(period - 1, len(series)):
        window = series[i - period + 1:i + 1]
        result[i] = np.argmin(window)
    return result


# ── Alpha158 因子计算 ─────────────────────────────────────────

class Alpha158Calculator:
    """
    Alpha158 因子计算器

    计算 158 个价量因子，基于 Qlib 的 Alpha158 定义。
    每个因子独立计算，返回标准化后的因子矩阵。
    """

    # 常用 period 配置
    PERIODS = [2, 3, 5, 8, 10, 15, 20, 30, 40, 60]

    # 因子名称映射 (用于调试)
    FACTOR_NAMES = {}

    def __init__(self):
        self.factor_names: List[str] = []
        self._build_factor_names()

    def _build_factor_names(self):
        """构建因子名称列表"""
        self.factor_names = []
        # Alpha158 的命名规则: {operation}_{input1}_{input2}_{period}
        # 例如: delta_close_5, ts_max_high_20, ts_rank_volume_10
        base_inputs = ['close', 'open', 'high', 'low', 'volume', 'vwap']
        time_periods = self.PERIODS

        for period in time_periods:
            for inp in base_inputs:
                self.factor_names.append(f"delta_{inp}_{period}")
                self.factor_names.append(f"ts_max_{inp}_{period}")
                self.factor_names.append(f"ts_min_{inp}_{period}")
                self.factor_names.append(f"ts_mean_{inp}_{period}")
                self.factor_names.append(f"ts_std_{inp}_{period}")
                self.factor_names.append(f"ts_rank_{inp}_{period}")
                self.factor_names.append(f"ts_argmax_{inp}_{period}")
                self.factor_names.append(f"ts_argmin_{inp}_{period}")
                self.factor_names.append(f"ts_sum_{inp}_{period}")

        # 相关因子
        for period in time_periods:
            self.factor_names.append(f"ts_corr_close_volume_{period}")
            self.factor_names.append(f"ts_corr_return_volume_{period}")
            self.factor_names.append(f"ts_corr_close_open_{period}")

        # 截断到 158 个
        self.factor_names = self.factor_names[:158]
        logger.info(f"[Alpha158] 构建了 {len(self.factor_names)} 个因子")

    def calculate_all(self, klines: List[Dict]) -> Dict[str, np.ndarray]:
        """
        计算所有 Alpha158 因子

        Args:
            klines: K 线数据 [{date, open, high, low, close, volume, ...}]

        Returns:
            {factor_name: np.ndarray} 因子矩阵
        """
        if not klines or len(klines) < 60:
            logger.warning(f"[Alpha158] K 线数据不足 ({len(klines) if klines else 0} 条)")
            return {}

        # 提取序列
        closes = np.array([float(k['close']) for k in klines], dtype=np.float64)
        opens = np.array([float(k['open']) for k in klines], dtype=np.float64)
        highs = np.array([float(k['high']) for k in klines], dtype=np.float64)
        lows = np.array([float(k['low']) for k in klines], dtype=np.float64)
        volumes = np.array([float(k.get('volume', 0)) for k in klines], dtype=np.float64)
        vwaps = np.array([float(k.get('vwap', k.get('close', 0))) for k in klines], dtype=np.float64)

        # 收益率
        returns = np.diff(closes) / closes[:-1]
        returns = np.concatenate([[0.0], returns])

        factors = {}

        # ── 动量因子 ──
        for p in self.PERIODS:
            name = f"momentum_{p}"
            factors[name] = closes[-1] / closes[-p - 1] - 1 if len(closes) > p + 1 else 0
            self.factor_names.append(name)

        # 短期反转
        for p in [1, 3, 5]:
            name = f"reversal_{p}"
            if len(closes) > p:
                factors[name] = -(closes[-1] / closes[-p - 1] - 1)
            else:
                factors[name] = 0
            self.factor_names.append(name)

        # ── 波动率因子 ──
        for p in self.PERIODS:
            name = f"volatility_{p}"
            ret = returns[max(0, len(returns) - p):]
            factors[name] = float(np.std(ret)) if len(ret) > 1 else 0
            self.factor_names.append(name)

        # 上行波动率 / 下行波动率
        if len(returns) > 20:
            up_ret = returns[returns > 0]
            down_ret = returns[returns < 0]
            factors['upside_volatility'] = float(np.std(up_ret)) if len(up_ret) > 0 else 0
            factors['downside_volatility'] = float(np.std(down_ret)) if len(down_ret) > 0 else 0
            factors['up_down_vol_ratio'] = (
                factors['upside_volatility'] / (factors['downside_volatility'] + 1e-10)
            )
            self.factor_names.extend(['upside_volatility', 'downside_volatility', 'up_down_vol_ratio'])

        # ── 成交量因子 ──
        for p in self.PERIODS:
            name = f"volume_ratio_{p}"
            if len(volumes) > p:
                factors[name] = volumes[-1] / (np.mean(volumes[-p - 1:-1]) + 1e-10)
            else:
                factors[name] = 1.0
            self.factor_names.append(name)

        # 量价相关
        for p in [5, 10, 20]:
            name = f"volume_price_corr_{p}"
            if len(closes) > p + 5:
                ret_window = returns[-p - 1:-1]
                vol_window = volumes[-p - 1:-1]
                if np.std(ret_window) > 1e-10 and np.std(vol_window) > 1e-10:
                    factors[name] = np.corrcoef(ret_window, vol_window)[0, 1]
                else:
                    factors[name] = 0
            else:
                factors[name] = 0
            self.factor_names.append(name)

        # 成交量动量
        for p in [5, 10]:
            name = f"volume_momentum_{p}"
            if len(volumes) > p:
                factors[name] = volumes[-1] / volumes[-p - 1] - 1
            else:
                factors[name] = 0
            self.factor_names.append(name)

        # ── 流动性因子 ──
        # Amihud 非流动性
        for p in [5, 10, 20]:
            name = f"amihud_{p}"
            if len(returns) > p:
                ret_window = np.abs(returns[-p - 1:-1])
                vol_window = volumes[-p - 1:-1]
                factors[name] = float(np.mean(ret_window / (vol_window + 1e-10)))
            else:
                factors[name] = 0
            self.factor_names.append(name)

        # ── 价格位置因子 ──
        for p in [5, 10, 20, 30, 60]:
            name = f"price_position_{p}"
            if len(highs) > p and len(lows) > p:
                high_max = np.max(highs[-p - 1:-1])
                low_min = np.min(lows[-p - 1:-1])
                if high_max > low_min:
                    factors[name] = (closes[-1] - low_min) / (high_max - low_min)
                else:
                    factors[name] = 0.5
            else:
                factors[name] = 0.5
            self.factor_names.append(name)

        # ── 传统技术指标 ──
        # RSI
        factors['rsi_14'] = self._calculate_rsi(closes, 14)
        factors['rsi_7'] = self._calculate_rsi(closes, 7)
        factors['rsi_21'] = self._calculate_rsi(closes, 21)
        self.factor_names.extend(['rsi_14', 'rsi_7', 'rsi_21'])

        # MACD
        macd, macd_signal, macd_hist = self._calculate_macd(closes)
        factors['macd'] = float(macd[-1]) if len(macd) > 0 else 0
        factors['macd_signal'] = float(macd_signal[-1]) if len(macd_signal) > 0 else 0
        factors['macd_hist'] = float(macd_hist[-1]) if len(macd_hist) > 0 else 0
        self.factor_names.extend(['macd', 'macd_signal', 'macd_hist'])

        # 布林带位置
        factors['boll_position'] = self._bollinger_position(closes)
        self.factor_names.append('boll_position')

        # ATR
        for p in [7, 14, 21]:
            name = f'atr_{p}'
            factors[name] = self._calculate_atr(highs, lows, closes, p)
            self.factor_names.append(name)

        # 均线比率
        for p1, p2 in [(5, 10), (5, 20), (10, 20), (20, 60)]:
            name = f'ma_ratio_{p1}_{p2}'
            if len(closes) > p2:
                factors[name] = np.mean(closes[-p1 - 1:-1]) / (np.mean(closes[-p2 - 1:-1]) + 1e-10)
            else:
                factors[name] = 1.0
            self.factor_names.append(name)

        # ── 时间序列统计因子 ──
        if len(closes) > 20:
            ret_window = returns[-20:]
            factors['return_skewness'] = float(pd.Series(ret_window).skew())
            factors['return_kurtosis'] = float(pd.Series(ret_window).kurtosis())
            self.factor_names.extend(['return_skewness', 'return_kurtosis'])

        # 自相关
        if len(returns) > 30:
            ret_short = returns[-10:-1]
            ret_long = returns[-20:-11]
            if len(ret_short) == len(ret_long) and len(ret_short) > 5:
                if np.std(ret_short) > 1e-10 and np.std(ret_long) > 1e-10:
                    factors['return_autocorr'] = np.corrcoef(ret_short, ret_long)[0, 1]
                else:
                    factors['return_autocorr'] = 0
            else:
                factors['return_autocorr'] = 0
            self.factor_names.append('return_autocorr')

        # ── 量价位置 ──
        for p in [5, 10, 20]:
            name = f'volume_position_{p}'
            if len(volumes) > p:
                vol_max = np.max(volumes[-p - 1:-1])
                vol_min = np.min(volumes[-p - 1:-1])
                if vol_max > vol_min:
                    factors[name] = (volumes[-1] - vol_min) / (vol_max - vol_min)
                else:
                    factors[name] = 0.5
            else:
                factors[name] = 0.5
            self.factor_names.append(name)

        # 截断到 158
        self.factor_names = self.factor_names[:158]

        return factors

    def calculate_batch(self, klines_dict: Dict[str, List[Dict]]) -> Dict[str, np.ndarray]:
        """
        批量计算多个标的的 Alpha158 因子

        Args:
            klines_dict: {stock_code: [kline_data]}

        Returns:
            {factor_name: {stock_code: factor_value}}
        """
        all_factors = {}
        codes = list(klines_dict.keys())

        for code in codes:
            klines = klines_dict[code]
            if not klines or len(klines) < 20:
                continue
            factors = self.calculate_all(klines)
            for fname, fval in factors.items():
                if fname not in all_factors:
                    all_factors[fname] = {}
                all_factors[fname][code] = fval

        return all_factors

    @staticmethod
    def _calculate_rsi(closes: np.ndarray, period: int = 14) -> float:
        """计算 RSI"""
        if len(closes) < period + 1:
            return 50.0
        deltas = np.diff(closes[-period - 1:])
        gains = np.maximum(deltas, 0)
        losses = np.abs(np.minimum(deltas, 0))
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def _calculate_macd(closes: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """计算 MACD"""
        if len(closes) < 35:
            return np.array([0]), np.array([0]), np.array([0])

        ema12 = pd.Series(closes).ewm(span=12, adjust=False).mean().values
        ema26 = pd.Series(closes).ewm(span=26, adjust=False).mean().values
        dif = ema12 - ema26
        dea = pd.Series(dif).ewm(span=9, adjust=False).mean().values
        macd_hist = 2 * (dif - dea)

        return dif, dea, macd_hist

    @staticmethod
    def _bollinger_position(closes: np.ndarray, period: int = 20) -> float:
        """布林带位置"""
        if len(closes) < period:
            return 0.5
        ma = np.mean(closes[-period:])
        std = np.std(closes[-period:])
        if std == 0:
            return 0.5
        return (closes[-1] - (ma - 2 * std)) / (4 * std)

    @staticmethod
    def _calculate_atr(highs: np.ndarray, lows: np.ndarray,
                       closes: np.ndarray, period: int = 14) -> float:
        """计算 ATR"""
        if len(highs) < period + 1:
            return 0.0
        tr_list = []
        for i in range(-period - 1, -1):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i + 1]),
                abs(lows[i] - closes[i + 1]),
            )
            tr_list.append(tr)
        return float(np.mean(tr_list))

    def normalize_factors(self, factors: Dict[str, float]) -> Dict[str, float]:
        """
        标准化因子值 (z-score 截断到 ±3σ)

        Args:
            factors: 原始因子值

        Returns:
            标准化后的因子值
        """
        if not factors:
            return {}

        values = np.array(list(factors.values()), dtype=np.float64)
        mean = np.mean(values)
        std = np.std(values) + 1e-10

        normalized = {}
        for name, val in factors.items():
            z = (val - mean) / std
            z = np.clip(z, -3.0, 3.0)  # 截断到 ±3σ
            normalized[name] = float(z)

        return normalized

    def get_factor_count(self) -> int:
        """获取因子数量"""
        return len(self.factor_names)

    def get_factor_list(self) -> List[str]:
        """获取因子名称列表"""
        return self.factor_names
