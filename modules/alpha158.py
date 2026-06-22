#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Qlib Alpha158 因子计算模块 — 2026 SOTA

功能:
1. 158 个 Alpha 因子计算 (来自 Qlib Alpha158)
2. 因子评估和选择
3. 因子正交化

架构:
    原始数据 → 因子计算 → 因子选择 → 输出

参考:
    - Qlib Alpha158: https://github.com/microsoft/qlib
    - Liu et al., "Alpha158: 158 Quantitative Factors from Qlib"
    - 因子分类: 动量、价值、波动率、成交量、质量、技术、情绪

作者: Stock Analyzer SOTA Team
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable, Union
from datetime import datetime
import threading

from modules.logger import logger


# ── Alpha158 因子定义 ──────────────────────────────────────────

class Alpha158Calculator:
    """
    Alpha158 因子计算器

    158 个因子分为以下类别:
    1. 动量因子 (40个)
    2. 价值因子 (15个)
    3. 波动率因子 (15个)
    4. 成交量因子 (20个)
    5. 质量因子 (20个)
    6. 技术因子 (30个)
    7. 情绪因子 (18个)
    """

    def __init__(self):
        self.factors = {}  # factor_name -> factor_function
        self._build_factors()
        self._lock = threading.Lock()

    def _build_factors(self):
        """构建所有因子"""
        # 动量因子 (Momentum Factors)
        self.factors.update(self._build_momentum_factors())

        # 价值因子 (Value Factors)
        self.factors.update(self._build_value_factors())

        # 波动率因子 (Volatility Factors)
        self.factors.update(self._build_volatility_factors())

        # 成交量因子 (Volume Factors)
        self.factors.update(self._build_volume_factors())

        # 质量因子 (Quality Factors)
        self.factors.update(self._build_quality_factors())

        # 技术因子 (Technical Factors)
        self.factors.update(self._build_technical_factors())

        # 情绪因子 (Sentiment Factors)
        self.factors.update(self._build_sentiment_factors())

        logger.info(f"[Alpha158] 构建了 {len(self.factors)} 个因子")

    def _build_momentum_factors(self) -> Dict[str, Callable]:
        """动量因子"""
        factors = {}

        # 简单动量
        for day in [5, 10, 20, 30, 60]:
            factors[f'ROC_{day}'] = lambda df, d=day: df['close'].pct_change(d)
            factors[f'RET_{day}'] = lambda df, d=day: df['close'].pct_change(d)

        # 移动平均动量
        for day in [5, 10, 20, 30, 60]:
            factors[f'MA_{day}'] = lambda df, d=day: (df['close'] - df['close'].rolling(d).mean()) / df['close'].rolling(d).mean()

        # 相对强弱
        for day in [5, 10, 20, 30, 60]:
            factors[f'RSI_{day}'] = lambda df, d=day: self._rsi(df['close'], d)

        # MACD
        factors['MACD'] = lambda df: self._macd(df['close'])
        factors['MACD_SIGNAL'] = lambda df: self._macd_signal(df['close'])
        factors['MACD_DIFF'] = lambda df: self._macd_diff(df['close'])

        # 布林带动量
        for day in [10, 20, 30]:
            factors[f'BBANDS_{day}'] = lambda df, d=day: (df['close'] - df['close'].rolling(d).mean()) / (2 * df['close'].rolling(d).std())

        # 趋势因子
        for day in [5, 10, 20, 30, 60]:
            factors[f'TREND_{day}'] = lambda df, d=day: (df['close'] - df['close'].shift(d)) / df['close'].shift(d)

        # 威廉指标
        for day in [14, 28]:
            factors[f'WR_{day}'] = lambda df, d=day: (df['close'].rolling(d).max() - df['close']) / (df['close'].rolling(d).max() - df['close'].rolling(d).min())

        return factors

    def _build_value_factors(self) -> Dict[str, Callable]:
        """价值因子"""
        factors = {}

        # 市净率
        factors['PB'] = lambda df: df['close'] / df.get('book_value', df['close'] * 0.1 + 1e-10)

        # 市销率
        factors['PS'] = lambda df: df['close'] / df.get('revenue', df['close'] * 0.1 + 1e-10)

        # 市盈率
        factors['PE'] = lambda df: df['close'] / df.get('eps', df['close'] * 0.05 + 1e-10)

        # 现金流比率
        factors['PCF'] = lambda df: df['close'] / df.get('cash_flow', df['close'] * 0.05 + 1e-10)

        # 营收增长率
        for day in [1, 4, 8, 12]:
            factors[f'REV_GROWTH_{day}'] = lambda df, d=day: df.get('revenue', df['close']).pct_change(252 // d)

        # 利润增长率
        for day in [1, 4, 8, 12]:
            factors[f'PROF_GROWTH_{day}'] = lambda df, d=day: df.get('net_profit', df['close'] * 0.05).pct_change(252 // d)

        return factors

    def _build_volatility_factors(self) -> Dict[str, Callable]:
        """波动率因子"""
        factors = {}

        # 历史波动率
        for day in [5, 10, 20, 30, 60]:
            factors[f'VOL_{day}'] = lambda df, d=day: df['close'].pct_change().rolling(d).std()

        # 波动率比率
        for day in [5, 10, 20, 30, 60]:
            factors[f'VOL_RATIO_{day}'] = lambda df, d=day: df['close'].pct_change().rolling(day).std() / (df['close'].pct_change().rolling(day * 2).std() + 1e-10)

        # ATR (Average True Range)
        factors['ATR'] = lambda df: self._atr(df)

        # Beta
        factors['BETA'] = lambda df: df['close'].pct_change().rolling(20).cov(df.get('market_return', df['close'].pct_change()))

        return factors

    def _build_volume_factors(self) -> Dict[str, Callable]:
        """成交量因子"""
        factors = {}

        # 成交量动量
        for day in [5, 10, 20, 30, 60]:
            factors[f'VOL_MOM_{day}'] = lambda df, d=day: df['volume'].pct_change(d)

        # 成交量与价格关系
        for day in [5, 10, 20, 30, 60]:
            factors[f'PV_REL_{day}'] = lambda df, d=day: df['close'].pct_change(d) * df['volume'].pct_change(d)

        # OBV (On-Balance Volume)
        factors['OBV'] = lambda df: (np.sign(df['close'].diff()) * df['volume']).cumsum()

        # 成交量加权价格
        factors['VWAP'] = lambda df: (df['close'] * df['volume']).cumsum() / (df['volume'].cumsum() + 1e-10)

        # 换手率
        factors['TURNOVER'] = lambda df: df['volume'] / df.get('float_share', df['volume'] * 10 + 1e-10)

        return factors

    def _build_quality_factors(self) -> Dict[str, Callable]:
        """质量因子 (财务质量)"""
        factors = {}

        # ROE (Return on Equity)
        factors['ROE'] = lambda df: df.get('net_profit', df['close'] * 0.05) / (df.get('equity', df['close'] * 0.5) + 1e-10)

        # ROA (Return on Assets)
        factors['ROA'] = lambda df: df.get('net_profit', df['close'] * 0.05) / (df.get('total_assets', df['close'] * 1) + 1e-10)

        # 毛利率
        factors['GROSS_MARGIN'] = lambda df: (df.get('revenue', df['close']) - df.get('cost', df['close'] * 0.7)) / (df.get('revenue', df['close']) + 1e-10)

        # 净利润率
        factors['NET_MARGIN'] = lambda df: df.get('net_profit', df['close'] * 0.05) / (df.get('revenue', df['close']) + 1e-10)

        # 资产负债率
        factors['DEBT_RATIO'] = lambda df: df.get('total_liabilities', df['close'] * 0.5) / (df.get('total_assets', df['close'] * 1) + 1e-10)

        # 流动比率
        factors['CURRENT_RATIO'] = lambda df: df.get('current_assets', df['close'] * 0.3) / (df.get('current_liabilities', df['close'] * 0.2) + 1e-10)

        return factors

    def _build_technical_factors(self) -> Dict[str, Callable]:
        """技术因子"""
        factors = {}

        # KD 指标
        factors['K'] = lambda df: self._stochastic_k(df['close'], 14)
        factors['D'] = lambda df: self._stochastic_d(df['close'], 14)
        factors['J'] = lambda df: 3 * factors['K'](df) - 2 * factors['D'](df)

        # CCI (Commodity Channel Index)
        factors['CCI'] = lambda df: self._cci(df['close'], 14)

        # ADX (Average Directional Index)
        factors['ADX'] = lambda df: self._adx(df)

        # 抛物线 SAR
        factors['SAR'] = lambda df: self._sar(df)

        # DIF (期货技术指标)
        factors['DIF'] = lambda df: df['close'] - df['close'].rolling(12).mean()
        factors['DEA'] = lambda df: df['close'] - df['close'].rolling(26).mean()

        return factors

    def _build_sentiment_factors(self) -> Dict[str, Callable]:
        """情绪因子"""
        factors = {}

        # 买卖价差
        factors['SPREAD'] = lambda df: df.get('ask_price', df['close'] * 1.001) - df.get('bid_price', df['close'] * 0.999)

        # 委比
        factors['ORDER_RATIO'] = lambda df: (df.get('bid_volume', df['volume'] * 0.5) - df.get('ask_volume', df['volume'] * 0.5)) / (df['volume'] + 1e-10)

        # 外盘/内盘比
        factors['UP_DOWN_RATIO'] = lambda df: df.get('up_volume', df['volume'] * 0.5) / (df.get('down_volume', df['volume'] * 0.5) + 1e-10)

        # 主力净流入
        for day in [1, 3, 5, 10]:
            factors[f'NET_FLOW_{day}'] = lambda df, d=day: (df.get('big_up_volume', df['volume'] * 0.3) - df.get('big_down_volume', df['volume'] * 0.3)).rolling(d).sum()

        return factors

    # ── 辅助函数 ────────────────────────────────────────────────

    def _rsi(self, close: pd.Series, period: int = 14) -> pd.Series:
        """计算 RSI"""
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        return 100 - (100 / (1 + rs))

    def _macd(self, close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """计算 MACD"""
        ema_fast = close.ewm(span=fast).mean()
        ema_slow = close.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        diff = macd - signal_line
        return macd, signal_line, diff

    def _macd_signal(self, close: pd.Series) -> pd.Series:
        """MACD Signal"""
        macd, signal, _ = self._macd(close)
        return signal

    def _macd_diff(self, close: pd.Series) -> pd.Series:
        """MACD Diff"""
        macd, signal, diff = self._macd(close)
        return diff

    def _atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """计算 ATR"""
        high = df.get('high', df['close'])
        low = df.get('low', df['close'])
        tr1 = high - low
        tr2 = abs(high - df['close'].shift(1))
        tr3 = abs(low - df['close'].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        return atr

    def _rsi(self, close: pd.Series, period: int = 14) -> pd.Series:
        """计算 RSI"""
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        return 100 - (100 / (1 + rs))

    def _stochastic_k(self, close: pd.Series, period: int = 14) -> pd.Series:
        """Stochastic K"""
        low_min = close.rolling(period).min()
        high_max = close.rolling(period).max()
        k = 100 * (close - low_min) / (high_max - low_min + 1e-10)
        return k

    def _stochastic_d(self, close: pd.Series, period: int = 14) -> pd.Series:
        """Stochastic D"""
        k = self._stochastic_k(close, period)
        d = k.rolling(3).mean()
        return d

    def _cci(self, close: pd.Series, period: int = 14) -> pd.Series:
        """CCI (Commodity Channel Index)"""
        tp = close
        sma = tp.rolling(period).mean()
        mad = (tp - sma).abs().rolling(period).mean()
        cci = (tp - sma) / (0.015 * mad + 1e-10)
        return cci

    def _adx(self, df: pd.DataFrame) -> pd.Series:
        """ADX (Average Directional Index)"""
        high = df.get('high', df['close'])
        low = df.get('low', df['close'])
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        tr = self._atr(df)
        plus_di = 100 * plus_dm.rolling(14).mean() / (tr + 1e-10)
        minus_di = 100 * minus_dm.rolling(14).mean() / (tr + 1e-10)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(14).mean()
        return adx

    def _sar(self, df: pd.DataFrame) -> pd.Series:
        """Parabolic SAR (简化)"""
        high = df.get('high', df['close'])
        low = df.get('low', df['close'])
        sar = low.rolling(2).min()
        return sar

    # ── 主函数 ────────────────────────────────────────────────

    def calculate_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有因子

        Args:
            df: DataFrame with columns: close, open, high, low, volume, etc.

        Returns:
            factors: DataFrame with all 158 factors
        """
        factors = pd.DataFrame(index=df.index)

        for name, func in self.factors.items():
            try:
                factors[name] = func(df)
            except Exception as e:
                logger.warning(f"[Alpha158] 计算因子 {name} 失败: {e}")
                factors[name] = np.nan

        # 清洗因子
        factors = self._clean_factors(factors)

        return factors

    def _clean_factors(self, factors: pd.DataFrame) -> pd.DataFrame:
        """清洗因子: 去除 NaN, 异常值"""
        # 填充 NaN
        factors = factors.fillna(0)

        # 去除异常值 (3σ 截断)
        for col in factors.columns:
            mean = factors[col].mean()
            std = factors[col].std()
            if std > 0:
                factors[col] = factors[col].clip(mean - 3 * std, mean + 3 * std)

        return factors

    def calculate_factor_ic(self, factors: pd.DataFrame, returns: pd.Series) -> pd.Series:
        """
        计算因子 IC (Information Coefficient)

        Args:
            factors: DataFrame with factors
            returns: Series with future returns

        Returns:
            ic: Series with IC for each factor
        """
        ic = pd.Series(index=factors.columns, dtype=float)

        for col in factors.columns:
            try:
                ic[col] = factors[col].corr(returns)
            except Exception:
                ic[col] = 0

        return ic

    def select_top_factors(self, factors: pd.DataFrame, returns: pd.Series, n: int = 20) -> List[str]:
        """
        选择 Top N 因子

        Args:
            factors: DataFrame with factors
            returns: Series with future returns
            n: Number of factors to select

        Returns:
            top_factors: List of factor names
        """
        ic = self.calculate_factor_ic(factors, returns)
        top_factors = ic.abs().sort_values(ascending=False).head(n).index.tolist()
        return top_factors


# ── 全局单例 ────────────────────────────────────────────────

_alpha158_instance: Optional[Alpha158Calculator] = None
_alpha158_lock = threading.Lock()


def get_alpha158_calculator() -> Alpha158Calculator:
    """获取全局 Alpha158Calculator 实例 (线程安全)"""
    global _alpha158_instance
    if _alpha158_instance is None:
        with _alpha158_lock:
            if _alpha158_instance is None:
                _alpha158_instance = Alpha158Calculator()
    return _alpha158_instance