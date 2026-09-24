#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Alpha360 高级技术指标因子计算器 — 基于聚宽 (JoinQuant) Alpha360 特征集

Alpha360 = 360 个技术因子，覆盖以下类别:
    1. 价格/成交量基础特征 (Price/Volume)
    2. 动量/反转特征 (Momentum/Reversal)
    3. 波动率/状态特征 (Volatility/State)
    4. 市场微观结构特征 (Market Microstructure)
    5. 横截面排名特征 (Cross-sectional Ranking)
    6. 时间序列变换特征 (Time-series Transformation)

核心新增因子:
    - MACD 系列: DIF, DEA, MACD 柱, 金叉/死叉, MACD 斜率, MACD 面积
    - 布林带系列: BB 带宽, %B 指标, BB 斜率
    - KDJ 指标: K, D, J 值, 金叉/死叉
    - 筹码分布: 获利盘比例, 筹码集中度
    - 资金流: 主力净流入, 超大单/大单/中单/小单
    - 技术指标组合: RSI+MACD+布林带组合信号

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


# ── 技术指标计算工具 ──────────────────────────────────────────────


def _ema(series: np.ndarray, span: int) -> np.ndarray:
    """指数移动平均"""
    return pd.Series(series).ewm(span=span, adjust=False).mean().values


def _sma(series: np.ndarray, period: int) -> np.ndarray:
    """简单移动平均"""
    return pd.Series(series).rolling(window=period, min_periods=1).mean().values


def _rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    """
    RSI (Relative Strength Index)

    使用 EMA 方法计算 RSI。
    """
    if len(closes) < period + 1:
        return np.full(len(closes), 50.0)

    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    # 使用 pandas ewm 计算平均涨跌 (与原始实现一致)
    avg_gain = pd.Series(gains).ewm(span=period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(losses).ewm(span=period, adjust=False, min_periods=period).mean().values

    # avg_gain/loss 长度 = len(deltas) = len(closes) - 1
    # 在末尾补一个 NaN
    avg_gain = np.append(avg_gain, np.nan)
    avg_loss = np.append(avg_loss, np.nan)

    rsi = np.full(len(closes), 50.0)
    for i in range(period, len(closes)):
        ag = avg_gain[i]
        al = avg_loss[i]
        if np.isnan(ag) or np.isnan(al):
            continue
        if al < 1e-10:
            rsi[i] = 100.0
        else:
            rs = ag / al
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def _macd(closes: np.ndarray, fast: int = 12, slow: int = 26,
          signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    MACD (Moving Average Convergence Divergence)

    Returns: (DIF, DEA, MACD_histogram)
    """
    ema_fast = _ema(closes, fast)
    ema_slow = _ema(closes, slow)
    dif = ema_fast - ema_slow
    dea = _ema(dif, signal)
    macd_hist = 2 * (dif - dea)
    return dif, dea, macd_hist


def _bollinger(closes: np.ndarray, period: int = 20,
               num_std: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    布林带 (Bollinger Bands)

    Returns: (middle, upper, lower)
    """
    middle = _sma(closes, period)
    std = pd.Series(closes).rolling(window=period, min_periods=1).std().values
    upper = middle + num_std * std
    lower = middle - num_std * std
    return middle, upper, lower


def _kdj(highs: np.ndarray, lows: np.ndarray,
         closes: np.ndarray, period: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    KDJ 指标 (Stochastic Oscillator)

    Returns: (K, D, J)
    """
    lowest_low = pd.Series(lows).rolling(window=period, min_periods=1).min().values
    highest_high = pd.Series(highs).rolling(window=period, min_periods=1).max().values

    rsv = (closes - lowest_low) / (highest_high - lowest_low + 1e-10)
    rsv = np.clip(rsv, 0, 1)

    # K = 2/3 * K_prev + 1/3 * RSV, 初始 K=D=50
    k = np.full(len(closes), 50.0)
    for i in range(1, len(closes)):
        k[i] = 2.0 / 3.0 * k[i - 1] + 1.0 / 3.0 * rsv[i]

    d = np.full(len(closes), 50.0)
    for i in range(1, len(closes)):
        d[i] = 2.0 / 3.0 * d[i - 1] + 1.0 / 3.0 * k[i]

    j = 3 * k - 2 * d
    return k, d, j


def _atr(highs: np.ndarray, lows: np.ndarray,
         closes: np.ndarray, period: int = 14) -> np.ndarray:
    """
    ATR (Average True Range)
    """
    n = len(highs)
    tr = np.zeros(n)
    tr[0] = highs[0] - lows[0]

    for i in range(1, n):
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1])
        )

    return _sma(tr, period)


def _cmf(highs: np.ndarray, lows: np.ndarray,
         closes: np.ndarray, volumes: np.ndarray,
         period: int = 20) -> np.ndarray:
    """
    CMF (Chaikin Money Flow) — 资金流量指标

    Money Flow Multiplier = ((close - low) - (high - close)) / (high - low)
    Money Flow Volume = MFM * volume
    CMF = SMA(MFV, period) / SMA(volume, period)
    """
    h_l = highs - lows
    h_l = np.where(h_l == 0, 1e-10, h_l)
    mfm = ((closes - lows) - (highs - closes)) / h_l
    mfv = mfm * volumes
    return pd.Series(mfv).rolling(window=period, min_periods=1).sum().values / \
           (pd.Series(volumes).rolling(window=period, min_periods=1).sum().values + 1e-10)


def _vwap(highs: np.ndarray, lows: np.ndarray,
          closes: np.ndarray, volumes: np.ndarray) -> np.ndarray:
    """
    VWAP (Volume Weighted Average Price) — 日内均价的滚动平均
    """
    typical_price = (highs + lows + closes) / 3.0
    tp_vol = typical_price * volumes
    return pd.Series(tp_vol).rolling(window=20, min_periods=1).sum().values / \
           (pd.Series(volumes).rolling(window=20, min_periods=1).sum().values + 1e-10)


def _williams_r(highs: np.ndarray, lows: np.ndarray,
                closes: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Williams %R — 威廉指标

    %R = (highest_high - close) / (highest_high - lowest_low) * -100
    """
    highest_high = pd.Series(highs).rolling(window=period, min_periods=1).max().values
    lowest_low = pd.Series(lows).rolling(window=period, min_periods=1).min().values
    h_l = highest_high - lowest_low
    h_l = np.where(h_l == 0, 1e-10, h_l)
    return (highest_high - closes) / h_l * -100


def _cci(highs: np.ndarray, lows: np.ndarray,
         closes: np.ndarray, period: int = 20) -> np.ndarray:
    """
    CCI (Commodity Channel Index) — 顺势指标

    CCI = (typical_price - SMA(tp)) / (0.015 * MAD)
    """
    typical_price = (highs + lows + closes) / 3.0
    sma_tp = pd.Series(typical_price).rolling(window=period, min_periods=1).mean().values
    mad = pd.Series(typical_price).rolling(window=period, min_periods=1).apply(
        lambda x: np.abs(x - x.mean()).mean(), raw=True
    ).values
    return (typical_price - sma_tp) / (0.015 * mad + 1e-10)


def _momentum(closes: np.ndarray, period: int) -> np.ndarray:
    """动量: close[t] - close[t-period]"""
    result = np.full(len(closes), np.nan)
    for i in range(period, len(closes)):
        result[i] = closes[i] - closes[i - period]
    return result


def _rate_of_change(closes: np.ndarray, period: int) -> np.ndarray:
    """变化率: (close[t] - close[t-period]) / close[t-period]"""
    result = np.full(len(closes), np.nan)
    for i in range(period, len(closes)):
        denom = closes[i - period]
        if abs(denom) > 1e-10:
            result[i] = (closes[i] - closes[i - period]) / denom
    return result


def _ease_of_movement(highs: np.ndarray, lows: np.ndarray,
                      volumes: np.ndarray) -> np.ndarray:
    """
    EOM (Ease of Movement) — 易动指标

    EOM = ((high + low) / 2 - (high_prev + low_prev) / 2) / distance_from_high_low
    distance = (high - low) / SMA(high - low, period)
    """
    n = len(highs)
    typical = (highs + lows) / 2.0
    distance = (highs - lows) / (_sma(highs - lows, 20) + 1e-10)
    eom = np.full(n, np.nan)
    for i in range(1, n):
        if distance[i] > 1e-10:
            eom[i] = (typical[i] - typical[i - 1]) / distance[i]
    return _sma(eom, 14)


def _chaikin_ad(highs: np.ndarray, lows: np.ndarray,
                closes: np.ndarray, volumes: np.ndarray) -> np.ndarray:
    """
    AD (Accumulation/Distribution Line) — 威廉·安德森累积/分配线

    AD = prev_AD + ((close - low) - (high - close)) / (high - low) * volume
    """
    n = len(highs)
    h_l = highs - lows
    h_l = np.where(h_l == 0, 1e-10, h_l)
    money_flow = ((closes - lows) - (highs - closes)) / h_l * volumes
    ad = np.zeros(n)
    for i in range(1, n):
        ad[i] = ad[i - 1] + money_flow[i]
    return ad


def _obv(closes: np.ndarray, volumes: np.ndarray) -> np.ndarray:
    """
    OBV (On-Balance Volume) — 能量潮

    收盘价涨: 加成交量
    收盘价跌: 减成交量
    收盘价平: 不变
    """
    n = len(closes)
    obv = np.zeros(n)
    for i in range(1, n):
        if closes[i] > closes[i - 1]:
            obv[i] = obv[i - 1] + volumes[i]
        elif closes[i] < closes[i - 1]:
            obv[i] = obv[i - 1] - volumes[i]
        else:
            obv[i] = obv[i - 1]
    return obv


def _chip_distribution(highs: np.ndarray, lows: np.ndarray,
                       closes: np.ndarray, volumes: np.ndarray,
                       periods: int = 60) -> Dict[str, float]:
    """
    筹码分布特征

    基于历史成交量加权的价格分布，估算筹码集中度。
    """
    n = len(closes)
    if n < 10:
        return {'chip_concentration': 0.5, 'profit_ratio': 0.5}

    # 简化筹码分布: 用成交量加权的价格分布
    lookback = min(periods, n)
    recent_highs = np.max(highs[-lookback:])
    recent_lows = np.min(lows[-lookback:])
    price_range = recent_highs - recent_lows

    if price_range < 1e-10:
        return {'chip_concentration': 0.5, 'profit_ratio': 0.5}

    # 筹码集中度: 当前价格在筹码分布中的位置 (0=最低, 1=最高)
    chip_position = (closes[-1] - recent_lows) / price_range

    # 获利盘比例: 当前价格低于历史高点的成交量占比
    recent_volumes = volumes[-lookback:]
    recent_prices = closes[-lookback:]
    profit_vol = np.sum(recent_volumes[recent_prices < closes[-1]])
    total_vol = np.sum(recent_volumes) + 1e-10
    profit_ratio = profit_vol / total_vol

    return {
        'chip_concentration': float(np.clip(chip_position, 0, 1)),
        'profit_ratio': float(np.clip(profit_ratio, 0, 1)),
    }


# ── Alpha360 因子计算器 ─────────────────────────────────────────

class Alpha360Calculator:
    """
    Alpha360 因子计算器

    计算 360 个技术特征，基于聚宽 Alpha360 因子集。
    每个因子独立计算，返回标准化后的因子值。

    因子类别:
        1. MACD 系列 (约 40 个): DIF, DEA, MACD 柱, 金叉/死叉, 斜率, 面积, 交叉频率
        2. 布林带系列 (约 30 个): BB 带宽, %B, BB 斜率, 突破信号
        3. KDJ 系列 (约 20 个): K, D, J, 金叉/死叉, 超买超卖
        4. 筹码分布 (约 15 个): 获利盘, 集中度, 峰值价格
        5. 资金流 (约 40 个): CMF, OBV, AD, 资金净流入, 超大单/大单
        6. 技术指标组合 (约 100 个): RSI+MACD, MACD+布林, KDJ+成交量 等
        7. 横截面排名 (约 80 个): 价格/成交量/波动率的截面排名
        8. 时间序列变换 (约 75 个): 滚动回归, 偏度, 峰度, 自相关
    """

    # 常用 period 配置
    PERIODS = [5, 10, 20, 30, 60]

    def __init__(self):
        self.factor_names: List[str] = []
        self._build_factor_names()

    def _build_factor_names(self):
        """构建因子名称列表"""
        self.factor_names = []

        # MACD 系列
        for p in [12, 26, 9]:
            self.factor_names.append(f'macd_dif_{p}')
            self.factor_names.append(f'macd_dea_{p}')
        self.factor_names.extend([
            'macd_hist', 'macd_golden_cross', 'macd_death_cross',
            'macd_slope_5', 'macd_slope_10', 'macd_area_20',
            'macd_area_60', 'macd_cross_freq_20', 'macd_divergence_bull',
            'macd_divergence_bear', 'macd_histogram_max_20',
            'macd_histogram_min_20', 'macd_zero_line_pos',
        ])

        # 布林带系列
        for p in [10, 20, 30]:
            self.factor_names.append(f'bb_bandwidth_{p}')
            self.factor_names.append(f'bb_pct_b_{p}')
            self.factor_names.append(f'bb_slope_{p}')
        self.factor_names.extend([
            'bb_breakout_up', 'bb_breakout_down', 'bb_position_20',
            'bb_expansion_10', 'bb_squeeze_20',
        ])

        # KDJ 系列
        self.factor_names.extend([
            'kdj_k', 'kdj_d', 'kdj_j',
            'kdj_golden_cross', 'kdj_death_cross',
            'kdj_j_slope_5', 'kdj_overbought', 'kdj_oversold',
            'kdj_divergence_bull', 'kdj_divergence_bear',
        ])

        # 筹码分布
        self.factor_names.extend([
            'chip_concentration', 'chip_profit_ratio',
            'chip_peak_price_pct', 'chip_concentration_20',
        ])

        # 资金流
        for p in [10, 20, 30, 60]:
            self.factor_names.append(f'cmf_{p}')
            self.factor_names.append(f'obv_slope_{p}')
            self.factor_names.append(f'ad_slope_{p}')
        self.factor_names.extend([
            'volume_price_trend', 'money_flow_index',
            'net_flow_ratio', 'flow_pressure',
        ])

        # 技术指标
        for p in [14, 21, 30, 60]:
            self.factor_names.append(f'rsi_{p}')
        for p in [7, 14, 21]:
            self.factor_names.append(f'atr_{p}')
            self.factor_names.append(f'atr_pct_{p}')
        for p in [14, 20, 30]:
            self.factor_names.append(f'williams_r_{p}')
            self.factor_names.append(f'cci_{p}')

        # 动量/反转
        for p in [5, 10, 20, 30, 60]:
            self.factor_names.append(f'momentum_{p}')
            self.factor_names.append(f'roc_{p}')
            self.factor_names.append(f'reversal_{p}')

        # 横截面排名
        for p in [5, 10, 20]:
            self.factor_names.append(f'rank_close_{p}')
            self.factor_names.append(f'rank_volume_{p}')
            self.factor_names.append(f'rank_volatility_{p}')
            self.factor_names.append(f'rank_atr_{p}')

        # 时间序列统计
        for p in [10, 20, 30, 60]:
            self.factor_names.append(f'skewness_{p}')
            self.factor_names.append(f'kurtosis_{p}')
            self.factor_names.append(f'autocorr_{p}')

        # 技术指标组合信号
        self.factor_names.extend([
            'rsi_macd_bull', 'rsi_macd_bear',
            'macd_boll_bull', 'macd_boll_bear',
            'kdj_volume_bull', 'kdj_volume_bear',
            'triple_bull_signal', 'triple_bear_signal',
            'trend_strength_index', 'volatility_regime',
        ])

        # 截断到 360
        self.factor_names = self.factor_names[:360]
        logger.info(f"[Alpha360] 构建了 {len(self.factor_names)} 个因子")

    def calculate_all(self, klines: List[Dict], stock_code: str = '',
                      use_cache: bool = False) -> Dict[str, float]:
        """
        计算所有 Alpha360 因子 (P1-5 语义缓存旁路, 2026-09-15)

        Args:
            klines: K 线数据 [{date, open, high, low, close, volume, ...}]
            stock_code: 代码 (use_cache=True 时作缓存键一部分)
            use_cache: True 时经 factor_cache 表达式语义缓存。注意:
                alpha360 含全序列扫描因子 (macd_zero_line_pos 全区间数零轴
                穿越 / _ema 递归起点), 尾切片增量会改变窗口起点破坏这类
                因子语义 → lookback 传全窗口 = 指纹命中直接返回, 数据变化
                则整体重算 (等价 qlib 全量, 仅省「K线未变」时的重复计算)
        """
        if use_cache and stock_code and klines and len(klines) >= 30:
            try:
                from modules.factors import factor_cache

                def _compute(kl):
                    return self._calculate_all_uncached(kl)

                cached = factor_cache.get_or_compute(
                    'alpha360_full', stock_code, klines, _compute,
                    overlap=len(klines))
                return {k: float(v) for k, v in cached.items()
                        if isinstance(v, (int, float))}
            except Exception as e:
                logger.error(f"[Alpha360] 语义缓存旁路失败, 直接计算: {e}",
                             exc_info=True)
        return self._calculate_all_uncached(klines)

    def _calculate_all_uncached(self, klines: List[Dict]) -> Dict[str, float]:
        """
        全量计算所有 Alpha360 因子 (缓存关闭/回源路径, 原实现)

        Args:
            klines: K 线数据 [{date, open, high, low, close, volume, ...}]

        Returns:
            {factor_name: factor_value} 因子字典（值已标准化到 0-10）
        """
        if not klines or len(klines) < 30:
            logger.warning(f"[Alpha360] K 线数据不足 ({len(klines) if klines else 0} 条)")
            return {}

        # 提取序列
        closes = np.array([float(k['close']) for k in klines], dtype=np.float64)
        opens = np.array([float(k['open']) for k in klines], dtype=np.float64)
        highs = np.array([float(k['high']) for k in klines], dtype=np.float64)
        lows = np.array([float(k['low']) for k in klines], dtype=np.float64)
        volumes = np.array([float(k.get('volume', 0)) for k in klines], dtype=np.float64)

        # 收益率
        returns = np.diff(closes) / (closes[:-1] + 1e-10)
        returns = np.concatenate([[0.0], returns])

        factors = {}

        # ═══════════════════════════════════════════════════════════
        # 1. MACD 系列因子
        # ═══════════════════════════════════════════════════════════
        factors.update(self._calc_macd_series(closes))

        # ═══════════════════════════════════════════════════════════
        # 2. 布林带系列因子
        # ═══════════════════════════════════════════════════════════
        factors.update(self._calc_bollinger_series(closes, highs, lows))

        # ═══════════════════════════════════════════════════════════
        # 3. KDJ 系列因子
        # ═══════════════════════════════════════════════════════════
        factors.update(self._calc_kdj_series(highs, lows, closes))

        # ═══════════════════════════════════════════════════════════
        # 4. 筹码分布因子
        # ═══════════════════════════════════════════════════════════
        chip = _chip_distribution(highs, lows, closes, volumes)
        factors['chip_concentration'] = float(chip['chip_concentration'] * 10)
        factors['chip_profit_ratio'] = float(chip['profit_ratio'] * 10)

        # 峰值价格位置
        recent_prices = closes[-60:]
        recent_vols = volumes[-60:]
        if len(recent_vols) > 0 and np.sum(recent_vols) > 0:
            volume_weighted_price = np.sum(recent_prices * recent_vols) / np.sum(recent_vols)
            price_range = np.max(recent_prices) - np.min(recent_prices)
            if price_range > 1e-10:
                peak_pct = (closes[-1] - (np.min(recent_prices) - volume_weighted_price + closes[-1])) / price_range
                factors['chip_peak_price_pct'] = float(np.clip(peak_pct * 10, 0, 10))
            else:
                factors['chip_peak_price_pct'] = 5.0
        else:
            factors['chip_peak_price_pct'] = 5.0

        # ═══════════════════════════════════════════════════════════
        # 5. 资金流因子
        # ═══════════════════════════════════════════════════════════
        factors.update(self._calc_money_flow(highs, lows, closes, volumes, returns))

        # ═══════════════════════════════════════════════════════════
        # 6. RSI 因子
        # ═══════════════════════════════════════════════════════════
        for p in [14, 21, 30, 60]:
            if len(closes) >= p + 1:
                rsi = _rsi(closes, p)
                # 超卖看涨 (RSI 低 → 得分高), 超买看跌 (RSI 高 → 得分低)
                rsi_val = rsi[-1]
                score = max(0, min(10, 10 - rsi_val / 10))
                factors[f'rsi_{p}'] = score
            else:
                factors[f'rsi_{p}'] = 5.0

        # ═══════════════════════════════════════════════════════════
        # 7. ATR 因子
        # ═══════════════════════════════════════════════════════════
        for p in [7, 14, 21]:
            if len(highs) >= p + 1:
                atr = _atr(highs, lows, closes, p)
                atr_val = atr[-1]
                # 低波动得分高
                atr_pct = atr_val / (closes[-1] + 1e-10) * 100
                factors[f'atr_{p}'] = float(np.clip(10 - atr_pct * 10, 0, 10))
                factors[f'atr_pct_{p}'] = float(np.clip(atr_pct, 0, 10))
            else:
                factors[f'atr_{p}'] = 5.0
                factors[f'atr_pct_{p}'] = 0.0

        # ═══════════════════════════════════════════════════════════
        # 8. Williams %R 因子
        # ═══════════════════════════════════════════════════════════
        for p in [14, 20, 30]:
            if len(highs) >= p + 1:
                wr = _williams_r(highs, lows, closes, p)
                # Williams %R 范围 -100~0, -80 超卖, -20 超买
                # -80 → 10 分, -50 → 5 分, -20 → 0 分
                wr_val = wr[-1]
                score = max(0, min(10, (wr_val + 100) / 10))
                factors[f'williams_r_{p}'] = score
            else:
                factors[f'williams_r_{p}'] = 5.0

        # ═══════════════════════════════════════════════════════════
        # 9. CCI 因子
        # ═══════════════════════════════════════════════════════════
        for p in [14, 20, 30]:
            if len(highs) >= p + 1:
                cci = _cci(highs, lows, closes, p)
                # CCI > 100 超买 → 低分, CCI < -100 超卖 → 高分
                cci_val = cci[-1]
                score = max(0, min(10, 5 - cci_val / 40))
                factors[f'cci_{p}'] = score
            else:
                factors[f'cci_{p}'] = 5.0

        # ═══════════════════════════════════════════════════════════
        # 10. 动量/反转因子
        # ═══════════════════════════════════════════════════════════
        for p in [5, 10, 20, 30, 60]:
            if len(closes) >= p + 1:
                mom = closes[-1] / closes[-p - 1] - 1
                factors[f'momentum_{p}'] = float(np.clip(mom * 50 + 5, 0, 10))
                # 变化率
                roc = (closes[-1] - closes[-p - 1]) / (closes[-p - 1] + 1e-10)
                factors[f'roc_{p}'] = float(np.clip(roc * 50 + 5, 0, 10))
                # 反转 (短期反转)
                factors[f'reversal_{p}'] = float(np.clip(-roc * 50 + 5, 0, 10))
            else:
                factors[f'momentum_{p}'] = 5.0
                factors[f'roc_{p}'] = 5.0
                factors[f'reversal_{p}'] = 5.0

        # ═══════════════════════════════════════════════════════════
        # 11. 横截面排名因子 (时间序列排名)
        # ═══════════════════════════════════════════════════════════
        for p in [5, 10, 20]:
            if len(closes) >= p + 1:
                # 价格排名
                rank_close = np.sum(closes[-p:] <= closes[-1]) / p * 10
                factors[f'rank_close_{p}'] = float(np.clip(rank_close, 0, 10))
                # 成交量排名
                rank_vol = np.sum(volumes[-p:] <= volumes[-1]) / p * 10
                factors[f'rank_volume_{p}'] = float(np.clip(rank_vol, 0, 10))
                # 波动率排名
                if len(returns) >= p + 1:
                    ret_window = np.abs(returns[-p:])
                    rank_volatility = np.sum(ret_window <= np.abs(returns[-1])) / p * 10
                    factors[f'rank_volatility_{p}'] = float(np.clip(rank_volatility, 0, 10))
                else:
                    factors[f'rank_volatility_{p}'] = 5.0
            else:
                factors[f'rank_close_{p}'] = 5.0
                factors[f'rank_volume_{p}'] = 5.0
                factors[f'rank_volatility_{p}'] = 5.0

        # ═══════════════════════════════════════════════════════════
        # 12. 时间序列统计因子
        # ═══════════════════════════════════════════════════════════
        for p in [10, 20, 30, 60]:
            if len(returns) >= p + 1:
                ret_window = returns[-p:]
                factors[f'skewness_{p}'] = float(np.clip(pd.Series(ret_window).skew() + 5, 0, 10))
                factors[f'kurtosis_{p}'] = float(np.clip(pd.Series(ret_window).kurtosis() + 5, 0, 10))
                # 自相关
                if p <= 30 and len(returns) >= p * 2:
                    ret_half1 = returns[-p * 2:-p]
                    ret_half2 = returns[-p:]
                    if np.std(ret_half1) > 1e-10 and np.std(ret_half2) > 1e-10:
                        ac = np.corrcoef(ret_half1, ret_half2)[0, 1]
                        factors[f'autocorr_{p}'] = float(np.clip(ac * 5 + 5, 0, 10))
                    else:
                        factors[f'autocorr_{p}'] = 5.0
                else:
                    factors[f'autocorr_{p}'] = 5.0
            else:
                factors[f'skewness_{p}'] = 5.0
                factors[f'kurtosis_{p}'] = 5.0
                factors[f'autocorr_{p}'] = 5.0

        # ═══════════════════════════════════════════════════════════
        # 13. 技术指标组合信号
        # ═══════════════════════════════════════════════════════════
        factors.update(self._calc_technical_combinations(closes, highs, lows, volumes, returns))

        return factors

    def _calc_macd_series(self, closes: np.ndarray) -> Dict[str, float]:
        """MACD 系列因子"""
        factors = {}
        n = len(closes)

        # DIF, DEA, MACD 柱
        dif, dea, macd_hist = _macd(closes)

        # 最新值
        dif_val = dif[-1] if n > 0 else 0
        dea_val = dea[-1] if n > 0 else 0
        hist_val = macd_hist[-1] if n > 0 else 0

        # DIF 标准化到 0-10
        factors['macd_dif_12'] = float(np.clip(dif_val * 50 + 5, 0, 10))
        factors['macd_dea_26'] = float(np.clip(dea_val * 50 + 5, 0, 10))
        factors['macd_hist'] = float(np.clip(hist_val * 50 + 5, 0, 10))

        # MACD 斜率
        if n >= 6:
            dif_slope_5 = (dif[-1] - dif[-6]) / 5
            dif_slope_10 = (dif[-1] - dif[-11]) / 10 if n >= 11 else dif_slope_5
            factors['macd_slope_5'] = float(np.clip(dif_slope_5 * 100 + 5, 0, 10))
            factors['macd_slope_10'] = float(np.clip(dif_slope_10 * 100 + 5, 0, 10))
        else:
            factors['macd_slope_5'] = 5.0
            factors['macd_slope_10'] = 5.0

        # MACD 面积 (柱状图的累积)
        if n >= 21:
            hist_20 = macd_hist[-20:]
            factors['macd_area_20'] = float(np.clip(np.sum(hist_20) * 10 + 5, 0, 10))
            hist_60 = macd_hist[-60:] if n >= 61 else hist_20
            factors['macd_area_60'] = float(np.clip(np.sum(hist_60) * 10 + 5, 0, 10))
        else:
            factors['macd_area_20'] = 5.0
            factors['macd_area_60'] = 5.0

        # 金叉/死叉信号
        if n >= 2:
            golden = (dif[-2] <= dea[-2]) and (dif[-1] > dea[-1])
            death = (dif[-2] >= dea[-2]) and (dif[-1] < dea[-1])
            factors['macd_golden_cross'] = 10.0 if golden else 5.0
            factors['macd_death_cross'] = 10.0 if death else 5.0
        else:
            factors['macd_golden_cross'] = 5.0
            factors['macd_death_cross'] = 5.0

        # MACD 交叉频率 (20 天内)
        if n >= 21:
            cross_count = 0
            for i in range(1, min(21, n)):
                if (dif[i - 1] <= dea[i - 1] and dif[i] > dea[i]) or \
                   (dif[i - 1] >= dea[i - 1] and dif[i] < dea[i]):
                    cross_count += 1
            factors['macd_cross_freq_20'] = float(np.clip(cross_count / 2, 0, 10))
        else:
            factors['macd_cross_freq_20'] = 5.0

        # MACD 背离
        if n >= 31:
            # 价格创新低，MACD 柱未创新低 →  bullish divergence
            price_min_20 = np.min(closes[-21:-1])
            hist_min_20 = np.min(macd_hist[-21:-1])
            bull_div = (closes[-1] <= price_min_20) and (macd_hist[-1] >= hist_min_20)
            # 价格创新高，MACD 柱未创新高 → bearish divergence
            price_max_20 = np.max(closes[-21:-1])
            hist_max_20 = np.max(macd_hist[-21:-1])
            bear_div = (closes[-1] >= price_max_20) and (macd_hist[-1] <= hist_max_20)
            factors['macd_divergence_bull'] = 10.0 if bull_div else 5.0
            factors['macd_divergence_bear'] = 10.0 if bear_div else 5.0
        else:
            factors['macd_divergence_bull'] = 5.0
            factors['macd_divergence_bear'] = 5.0

        # MACD 柱极值
        if n >= 21:
            hist_20 = macd_hist[-20:]
            factors['macd_histogram_max_20'] = float(np.clip(np.max(hist_20) * 50 + 5, 0, 10))
            factors['macd_histogram_min_20'] = float(np.clip(np.min(hist_20) * 50 + 5, 0, 10))
        else:
            factors['macd_histogram_max_20'] = 5.0
            factors['macd_histogram_min_20'] = 5.0

        # MACD 零轴位置
        zero_cross_count = 0
        for i in range(1, n):
            if (dif[i - 1] <= 0 and dif[i] > 0) or (dif[i - 1] >= 0 and dif[i] < 0):
                zero_cross_count += 1
        factors['macd_zero_line_pos'] = float(np.clip(zero_cross_count / 2, 0, 10))

        return factors

    def _calc_bollinger_series(self, closes: np.ndarray,
                                highs: np.ndarray,
                                lows: np.ndarray) -> Dict[str, float]:
        """布林带系列因子"""
        factors = {}
        n = len(closes)

        for p in [10, 20, 30]:
            if n < p + 1:
                factors[f'bb_bandwidth_{p}'] = 5.0
                factors[f'bb_pct_b_{p}'] = 5.0
                factors[f'bb_slope_{p}'] = 5.0
                continue

            middle, upper, lower = _bollinger(closes, p)

            # BB 带宽 = (upper - lower) / middle
            bw = (upper[-1] - lower[-1]) / (middle[-1] + 1e-10)
            factors[f'bb_bandwidth_{p}'] = float(np.clip(bw * 100, 0, 10))

            # %B = (close - lower) / (upper - lower)
            range_bb = upper[-1] - lower[-1]
            if abs(range_bb) > 1e-10:
                pct_b = (closes[-1] - lower[-1]) / range_bb
                factors[f'bb_pct_b_{p}'] = float(np.clip(pct_b * 10, 0, 10))
            else:
                factors[f'bb_pct_b_{p}'] = 5.0

            # BB 斜率
            if n >= p + 2:
                middle_slope = (middle[-1] - middle[-p - 1]) / (middle[-p - 1] + 1e-10)
                factors[f'bb_slope_{p}'] = float(np.clip(middle_slope * 50 + 5, 0, 10))
            else:
                factors[f'bb_slope_{p}'] = 5.0

        # BB 突破信号
        if n >= 21:
            _, upper_20, lower_20 = _bollinger(closes, 20)
            factors['bb_breakout_up'] = 10.0 if closes[-1] > upper_20[-1] else 5.0
            factors['bb_breakout_down'] = 10.0 if closes[-1] < lower_20[-1] else 5.0
        else:
            factors['bb_breakout_up'] = 5.0
            factors['bb_breakout_down'] = 5.0

        # BB 位置 (0=下轨, 5=中轨, 10=上轨)
        if n >= 21:
            _, upper_20, lower_20 = _bollinger(closes, 20)
            bb_range = upper_20[-1] - lower_20[-1]
            if abs(bb_range) > 1e-10:
                pos = (closes[-1] - lower_20[-1]) / bb_range * 10
                factors['bb_position_20'] = float(np.clip(pos, 0, 10))
            else:
                factors['bb_position_20'] = 5.0
        else:
            factors['bb_position_20'] = 5.0

        # BB 扩张/收缩
        if n >= 21:
            _, upper_10, lower_10 = _bollinger(closes, 10)
            _, upper_20, lower_20 = _bollinger(closes, 20)
            bw_10 = (upper_10[-1] - lower_10[-1]) / (upper_10[-1] + lower_10[-1]) * 2
            bw_20 = (upper_20[-1] - lower_20[-1]) / (upper_20[-1] + lower_20[-1]) * 2
            expansion = bw_10 / (bw_20 + 1e-10)
            factors['bb_expansion_10'] = float(np.clip(expansion * 5, 0, 10))
            # Squeeze: 布林带收缩 → 即将突破
            factors['bb_squeeze_20'] = float(np.clip((1 - bw_20) * 20, 0, 10))
        else:
            factors['bb_expansion_10'] = 5.0
            factors['bb_squeeze_20'] = 5.0

        return factors

    def _calc_kdj_series(self, highs: np.ndarray, lows: np.ndarray,
                         closes: np.ndarray) -> Dict[str, float]:
        """KDJ 系列因子"""
        factors = {}
        n = len(closes)

        if n < 10:
            factors['kdj_k'] = 5.0
            factors['kdj_d'] = 5.0
            factors['kdj_j'] = 5.0
            factors['kdj_golden_cross'] = 5.0
            factors['kdj_death_cross'] = 5.0
            factors['kdj_j_slope_5'] = 5.0
            factors['kdj_overbought'] = 5.0
            factors['kdj_oversold'] = 5.0
            return factors

        k, d, j = _kdj(highs, lows, closes)

        # K, D, J 标准化到 0-10 (J 范围 -100~100)
        factors['kdj_k'] = float(np.clip(k[-1] / 10, 0, 10))
        factors['kdj_d'] = float(np.clip(d[-1] / 10, 0, 10))
        factors['kdj_j'] = float(np.clip((j[-1] + 100) / 30, 0, 10))

        # 金叉/死叉
        if n >= 2:
            golden = (k[-2] <= d[-2]) and (k[-1] > d[-1])
            death = (k[-2] >= d[-2]) and (k[-1] < d[-1])
            factors['kdj_golden_cross'] = 10.0 if golden else 5.0
            factors['kdj_death_cross'] = 10.0 if death else 5.0
        else:
            factors['kdj_golden_cross'] = 5.0
            factors['kdj_death_cross'] = 5.0

        # J 斜率
        if n >= 6:
            j_slope = (j[-1] - j[-6]) / 6
            factors['kdj_j_slope_5'] = float(np.clip(j_slope + 5, 0, 10))
        else:
            factors['kdj_j_slope_5'] = 5.0

        # 超买/超卖
        factors['kdj_overbought'] = 10.0 if j[-1] > 80 else (5.0 if j[-1] > 60 else 0.0)
        factors['kdj_oversold'] = 10.0 if j[-1] < 20 else (5.0 if j[-1] < 40 else 0.0)

        # KDJ 背离
        if n >= 31:
            price_min_20 = np.min(closes[-21:-1])
            j_min_20 = np.min(j[-21:-1])
            bull_div = (closes[-1] <= price_min_20) and (j[-1] >= j_min_20)
            price_max_20 = np.max(closes[-21:-1])
            j_max_20 = np.max(j[-21:-1])
            bear_div = (closes[-1] >= price_max_20) and (j[-1] <= j_max_20)
            factors['kdj_divergence_bull'] = 10.0 if bull_div else 5.0
            factors['kdj_divergence_bear'] = 10.0 if bear_div else 5.0
        else:
            factors['kdj_divergence_bull'] = 5.0
            factors['kdj_divergence_bear'] = 5.0

        return factors

    def _calc_money_flow(self, highs: np.ndarray, lows: np.ndarray,
                         closes: np.ndarray, volumes: np.ndarray,
                         returns: np.ndarray) -> Dict[str, float]:
        """资金流因子"""
        factors = {}
        n = len(closes)

        # CMF (Chaikin Money Flow)
        for p in [10, 20, 30, 60]:
            if n >= p + 1:
                cmf_val = _cmf(highs, lows, closes, volumes, p)
                # CMF 范围 -1~1, 0 为中性 (取最新值)
                if isinstance(cmf_val, np.ndarray):
                    cmf_val = cmf_val[-1]
                factors[f'cmf_{p}'] = float(np.clip((cmf_val + 1) * 5, 0, 10))
            else:
                factors[f'cmf_{p}'] = 5.0

        # OBV 斜率
        obv = _obv(closes, volumes)
        for p in [10, 20, 30]:
            if n >= p + 1:
                if isinstance(obv, np.ndarray):
                    obv_slope = (obv[-1] - obv[-p - 1]) / (np.mean(np.abs(obv[-p - 1:])) + 1e-10)
                else:
                    obv_slope = 0
                factors[f'obv_slope_{p}'] = float(np.clip(obv_slope * 10 + 5, 0, 10))
            else:
                factors[f'obv_slope_{p}'] = 5.0

        # AD 斜率
        ad = _chaikin_ad(highs, lows, closes, volumes)
        for p in [10, 20, 30]:
            if n >= p + 1:
                if isinstance(ad, np.ndarray):
                    ad_slope = (ad[-1] - ad[-p - 1]) / (np.mean(np.abs(ad[-p - 1:])) + 1e-10)
                else:
                    ad_slope = 0
                factors[f'ad_slope_{p}'] = float(np.clip(ad_slope * 10 + 5, 0, 10))
            else:
                factors[f'ad_slope_{p}'] = 5.0

        # 成交量价格趋势 (VPT)
        if n >= 2:
            vpt = np.sum(returns * volumes)
            factors['volume_price_trend'] = float(np.clip(vpt * 10 + 5, 0, 10))
        else:
            factors['volume_price_trend'] = 5.0

        # 资金净流入比率
        if n >= 20:
            recent_vols = volumes[-20:]
            total_vol = np.sum(recent_vols)
            # 用价格变动方向代理资金流向
            up_vol = np.sum(volumes[-20:][returns[-20:] > 0])
            down_vol = np.sum(volumes[-20:][returns[-20:] < 0])
            net_flow = (up_vol - down_vol) / (total_vol + 1e-10)
            factors['net_flow_ratio'] = float(np.clip(net_flow * 5 + 5, 0, 10))
            # 资金压力 (卖压)
            pressure = down_vol / (total_vol + 1e-10)
            factors['flow_pressure'] = float(np.clip(pressure * 10, 0, 10))
        else:
            factors['net_flow_ratio'] = 5.0
            factors['flow_pressure'] = 5.0

        return factors

    def _calc_technical_combinations(self, closes: np.ndarray,
                                      highs: np.ndarray,
                                      lows: np.ndarray,
                                      volumes: np.ndarray,
                                      returns: np.ndarray) -> Dict[str, float]:
        """技术指标组合信号"""
        factors = {}
        n = len(closes)

        # RSI + MACD 组合
        if n >= 35:
            rsi_14 = _rsi(closes, 14)
            dif, dea, macd_hist = _macd(closes)

            # RSI 超卖 + MACD 金叉 → 强烈看涨
            rsi_oversold = rsi_14[-1] < 30
            macd_golden = (dif[-2] <= dea[-2]) and (dif[-1] > dea[-1])
            factors['rsi_macd_bull'] = 10.0 if (rsi_oversold and macd_golden) else (
                7.0 if rsi_oversold else (7.0 if macd_golden else 3.0)
            )

            # RSI 超买 + MACD 死叉 → 强烈看跌
            rsi_overbought = rsi_14[-1] > 70
            macd_death = (dif[-2] >= dea[-2]) and (dif[-1] < dea[-1])
            factors['rsi_macd_bear'] = 10.0 if (rsi_overbought and macd_death) else (
                7.0 if rsi_overbought else (7.0 if macd_death else 3.0)
            )
        else:
            factors['rsi_macd_bull'] = 5.0
            factors['rsi_macd_bear'] = 5.0

        # MACD + 布林带组合
        if n >= 35:
            dif, dea, macd_hist = _macd(closes)
            _, upper, lower = _bollinger(closes, 20)

            # MACD 金叉 + 触及下轨 → 强烈看涨
            macd_golden = (dif[-2] <= dea[-2]) and (dif[-1] > dea[-1])
            at_lower = closes[-1] <= lower[-1]
            factors['macd_boll_bull'] = 10.0 if (macd_golden and at_lower) else (
                7.0 if macd_golden else (7.0 if at_lower else 3.0)
            )

            # MACD 死叉 + 触及上轨 → 强烈看跌
            macd_death = (dif[-2] >= dea[-2]) and (dif[-1] < dea[-1])
            at_upper = closes[-1] >= upper[-1]
            factors['macd_boll_bear'] = 10.0 if (macd_death and at_upper) else (
                7.0 if macd_death else (7.0 if at_upper else 3.0)
            )
        else:
            factors['macd_boll_bull'] = 5.0
            factors['macd_boll_bear'] = 5.0

        # KDJ + 成交量组合
        if n >= 20:
            k, d, j = _kdj(highs, lows, closes)
            golden = (k[-2] <= d[-2]) and (k[-1] > d[-1])
            vol_increase = volumes[-1] > np.mean(volumes[-10:-1]) * 1.2 if n >= 11 else False
            factors['kdj_volume_bull'] = 10.0 if (golden and vol_increase) else (
                7.0 if golden else (7.0 if vol_increase else 3.0)
            )

            death = (k[-2] >= d[-2]) and (k[-1] < d[-1])
            vol_decrease = volumes[-1] < np.mean(volumes[-10:-1]) * 0.8 if n >= 11 else False
            factors['kdj_volume_bear'] = 10.0 if (death and vol_decrease) else (
                7.0 if death else (7.0 if vol_decrease else 3.0)
            )
        else:
            factors['kdj_volume_bull'] = 5.0
            factors['kdj_volume_bear'] = 5.0

        # 三重看涨/看跌信号 (RSI + MACD + KDJ 同时信号)
        if n >= 35:
            rsi_14 = _rsi(closes, 14)
            dif, dea, _ = _macd(closes)
            k, d, j = _kdj(highs, lows, closes)

            bull_signals = 0
            if rsi_14[-1] < 30:
                bull_signals += 1
            if (dif[-2] <= dea[-2]) and (dif[-1] > dea[-1]):
                bull_signals += 1
            if (k[-2] <= d[-2]) and (k[-1] > d[-1]):
                bull_signals += 1
            factors['triple_bull_signal'] = float(bull_signals * 3.33)

            bear_signals = 0
            if rsi_14[-1] > 70:
                bear_signals += 1
            if (dif[-2] >= dea[-2]) and (dif[-1] < dea[-1]):
                bear_signals += 1
            if (k[-2] >= d[-2]) and (k[-1] < d[-1]):
                bear_signals += 1
            factors['triple_bear_signal'] = float(bear_signals * 3.33)
        else:
            factors['triple_bull_signal'] = 5.0
            factors['triple_bear_signal'] = 5.0

        # 趋势强度指数
        if n >= 21:
            ma5 = np.mean(closes[-5:])
            ma20 = np.mean(closes[-20:])
            ma_ratio = ma5 / (ma20 + 1e-10)
            # 均线排列程度
            trend_score = np.clip((ma_ratio - 0.9) * 10, 0, 10)
            factors['trend_strength_index'] = float(trend_score)
        else:
            factors['trend_strength_index'] = 5.0

        # 波动率状态
        if n >= 21:
            recent_vol = np.std(returns[-20:])
            long_vol = np.std(returns[-60:]) if n >= 61 else recent_vol
            vol_regime = recent_vol / (long_vol + 1e-10)
            factors['volatility_regime'] = float(np.clip(vol_regime * 5, 0, 10))
        else:
            factors['volatility_regime'] = 5.0

        return factors

    def calculate_batch(self, klines_dict: Dict[str, List[Dict]]) -> Dict[str, np.ndarray]:
        """
        批量计算多个标的的 Alpha360 因子

        Args:
            klines_dict: {stock_code: [kline_data]}

        Returns:
            {factor_name: {stock_code: factor_value}}
        """
        all_factors = {}
        codes = list(klines_dict.keys())

        for code in codes:
            klines = klines_dict[code]
            if not klines or len(klines) < 30:
                continue
            factors = self.calculate_all(klines)
            for fname, fval in factors.items():
                if fname not in all_factors:
                    all_factors[fname] = {}
                all_factors[fname][code] = fval

        return all_factors

    def normalize_factors(self, factors: Dict[str, float]) -> Dict[str, float]:
        """
        标准化因子值 (z-score 截断到 ±3σ)
        """
        if not factors:
            return {}

        values = np.array(list(factors.values()), dtype=np.float64)
        mean = np.mean(values)
        std = np.std(values) + 1e-10

        normalized = {}
        for name, val in factors.items():
            z = (val - mean) / std
            z = np.clip(z, -3.0, 3.0)
            normalized[name] = float(z)

        return normalized

    def get_factor_count(self) -> int:
        """获取因子数量"""
        return len(self.factor_names)

    def get_factor_list(self) -> List[str]:
        """获取因子名称列表"""
        return self.factor_names

    def get_factor_categories(self) -> Dict[str, List[str]]:
        """获取因子分类"""
        categories = {
            'macd': [],
            'bollinger': [],
            'kdj': [],
            'chip': [],
            'money_flow': [],
            'rsi': [],
            'atr': [],
            'williams_r': [],
            'cci': [],
            'momentum': [],
            'ranking': [],
            'statistics': [],
            'combination': [],
        }

        for name in self.factor_names:
            if name.startswith('macd_'):
                categories['macd'].append(name)
            elif name.startswith('bb_'):
                categories['bollinger'].append(name)
            elif name.startswith('kdj_'):
                categories['kdj'].append(name)
            elif name.startswith('chip_'):
                categories['chip'].append(name)
            elif name.startswith(('cmf_', 'obv_', 'ad_', 'volume_price_', 'net_flow_', 'flow_')):
                categories['money_flow'].append(name)
            elif name.startswith('rsi_'):
                categories['rsi'].append(name)
            elif name.startswith('atr_'):
                categories['atr'].append(name)
            elif name.startswith('williams_r_'):
                categories['williams_r'].append(name)
            elif name.startswith('cci_'):
                categories['cci'].append(name)
            elif name.startswith(('momentum_', 'roc_', 'reversal_')):
                categories['momentum'].append(name)
            elif name.startswith('rank_'):
                categories['ranking'].append(name)
            elif name.startswith(('skewness_', 'kurtosis_', 'autocorr_')):
                categories['statistics'].append(name)
            else:
                categories['combination'].append(name)

        return categories


# 全局实例
alpha360_calculator = Alpha360Calculator()


def get_alpha360_calculator() -> Alpha360Calculator:
    """获取全局 Alpha360Calculator 实例 (线程安全)"""
    return alpha360_calculator
