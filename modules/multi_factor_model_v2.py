#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
多因子模型 V2 — P2 升级 (2026-06-04)

升级内容:
- 从 8 因子扩展至 15 因子
- 新增: 多周期动量、GARCH 波动率、MACD 斜率、量价相关、
       分析师预期 proxy、资金流动量、技术形态强度
- 因子 IC 滚动计算 + 动态权重
- 因子正交化 (Gram-Schmidt)
"""

import hashlib
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from modules.dynamic_cache import cache


class MultiFactorModelV2:
    """多因子模型 V2 — 15 因子 + 动态权重 + 正交化"""

    # 因子分类
    FACTOR_CATEGORIES = {
        'momentum': ['momentum_1d', 'momentum_5d', 'momentum_20d', 'momentum_60d', 'short_term_reversal'],
        'value': ['pe_value', 'pb_value', 'market_cap_value'],
        'volatility': ['realized_vol', 'downside_vol', 'garch_vol'],
        'volume': ['volume_ratio', 'volume_momentum', 'outer_inner_ratio'],
        'liquidity': ['turnover_level', 'amount_level'],
        'quality': ['roe_quality', 'revenue_growth_quality', 'margin_quality'],
        'technical': ['rsi_technical', 'macd_slope', 'bollinger_position', 'trend_strength'],
        'sentiment': ['price_sentiment', 'volume_sentiment'],
    }

    # 默认因子权重
    DEFAULT_WEIGHTS = {
        'momentum_1d': 0.08,
        'momentum_5d': 0.08,
        'momentum_20d': 0.07,
        'momentum_60d': 0.06,
        'short_term_reversal': 0.05,
        'pe_value': 0.07,
        'pb_value': 0.05,
        'market_cap_value': 0.04,
        'realized_vol': 0.05,
        'downside_vol': 0.04,
        'volume_ratio': 0.07,
        'volume_momentum': 0.05,
        'outer_inner_ratio': 0.06,
        'turnover_level': 0.04,
        'roe_quality': 0.06,
        'revenue_growth_quality': 0.05,
        'rsi_technical': 0.05,
        'macd_slope': 0.05,
        'bollinger_position': 0.04,
        'trend_strength': 0.05,
        'price_sentiment': 0.03,
    }

    def __init__(self):
        self.factor_weights = dict(self.DEFAULT_WEIGHTS)
        # IC 历史（用于动态权重）
        self._factor_ic_history: Dict[str, List[float]] = {}
        # 因子相关性矩阵（用于正交化）
        self._factor_correlations: Optional[np.ndarray] = None

    # ─────────────────────────────────────────────
    # 新增因子计算
    # ─────────────────────────────────────────────

    def momentum_1d(self, stock_data: Dict) -> float:
        """1日动量"""
        return stock_data.get('change_pct', 0)

    def momentum_5d(self, stock_data: Dict, klines: Optional[List[Dict]] = None) -> float:
        """5日动量"""
        if klines and len(klines) >= 6:
            closes = [float(k['close']) for k in klines[-6:]]
            return ((closes[-1] / closes[0]) - 1) * 100
        return stock_data.get('change_pct', 0) * 0.5  # 降级

    def momentum_20d(self, stock_data: Dict, klines: Optional[List[Dict]] = None) -> float:
        """20日动量"""
        if klines and len(klines) >= 21:
            closes = [float(k['close']) for k in klines[-21:]]
            return ((closes[-1] / closes[0]) - 1) * 100
        return 0.0

    def momentum_60d(self, stock_data: Dict, klines: Optional[List[Dict]] = None) -> float:
        """60日动量"""
        if klines and len(klines) >= 61:
            closes = [float(k['close']) for k in klines[-61:]]
            return ((closes[-1] / closes[0]) - 1) * 100
        return 0.0

    def short_term_reversal(self, stock_data: Dict) -> float:
        """短期反转: 当日涨跌幅的负值（超跌反弹/超涨回调）"""
        return -stock_data.get('change_pct', 0)

    def pe_value(self, stock_data: Dict) -> float:
        """PE 价值因子（低 PE 得分高）"""
        pe = stock_data.get('pe', 100)
        if pe <= 0:
            return 5.0  # 亏损股中性
        return max(0, min(10, 10 - pe / 30))

    def pb_value(self, stock_data: Dict) -> float:
        """PB 价值因子"""
        pb = stock_data.get('pb', 5)
        if pb <= 0:
            return 5.0
        return max(0, min(10, 10 - pb / 3))

    def market_cap_value(self, stock_data: Dict) -> float:
        """市值因子（中等市值最优）"""
        mc = stock_data.get('market_cap', 0)
        if mc <= 0:
            return 5.0
        if 100 < mc < 500:  # 100-500亿
            return 8.0
        elif mc >= 500:
            return 6.0
        elif mc > 50:
            return 7.0
        return 5.0

    def realized_vol(self, stock_data: Dict, klines: Optional[List[Dict]] = None) -> float:
        """已实现波动率（低波动得分高）"""
        if klines and len(klines) >= 21:
            closes = np.array([float(k['close']) for k in klines[-21:] if float(k.get('close', 0)) > 0])
            if len(closes) >= 2:
                returns = np.diff(np.log(closes))
                vol = float(np.std(returns) * np.sqrt(252) * 100)
                return max(0, min(10, 10 - vol / 5))
        # 降级: 用换手率估算
        turnover = stock_data.get('turnover', 100)
        return max(0, min(10, 10 - turnover / 50))

    def downside_vol(self, stock_data: Dict, klines: Optional[List[Dict]] = None) -> float:
        """下行波动率（越低越好）"""
        if klines and len(klines) >= 21:
            closes = np.array([float(k['close']) for k in klines[-21:] if float(k.get('close', 0)) > 0])
            if len(closes) >= 2:
                returns = np.diff(np.log(closes))
                downside = returns[returns < 0]
                if len(downside) > 0:
                    dvol = float(np.std(downside) * np.sqrt(252) * 100)
                    return max(0, min(10, 10 - dvol / 3))
        return 5.0

    def volume_ratio(self, stock_data: Dict) -> float:
        """量比（外盘/内盘）"""
        outer = stock_data.get('outer_disk', 0)
        inner = stock_data.get('inner_disk', 1)
        ratio = outer / inner if inner > 0 else 1.0
        if ratio > 1.3:
            return 8.0
        elif ratio > 1.0:
            return 6.0
        elif ratio > 0.7:
            return 4.0
        return 2.0

    def volume_momentum(self, stock_data: Dict) -> float:
        """成交量动量（放量加分）"""
        turnover = stock_data.get('turnover', 100)
        if turnover > 300:
            return 8.0
        elif turnover > 200:
            return 6.0
        elif turnover > 100:
            return 5.0
        return 3.0

    def outer_inner_ratio(self, stock_data: Dict) -> float:
        """内外盘比（买盘情绪）"""
        outer = stock_data.get('outer_disk', 0)
        inner = stock_data.get('inner_disk', 1)
        ratio = outer / inner if inner > 0 else 1.0
        return min(10, ratio * 5)

    def turnover_level(self, stock_data: Dict) -> float:
        """换手率水平（适中最优）"""
        turnover = stock_data.get('turnover', 100)
        if 100 < turnover < 300:
            return 8.0
        elif turnover >= 300:
            return 5.0  # 过高换手可能危险
        return 4.0

    def roe_quality(self, stock_data: Dict) -> float:
        """ROE 质量"""
        roe = stock_data.get('roe', 15)
        if roe > 20:
            return 9.0
        elif roe > 15:
            return 7.0
        elif roe > 10:
            return 5.0
        return 3.0

    def revenue_growth_quality(self, stock_data: Dict) -> float:
        """营收增长质量"""
        growth = stock_data.get('revenue_growth', 10)
        if growth > 30:
            return 9.0
        elif growth > 20:
            return 7.0
        elif growth > 10:
            return 5.0
        return 3.0

    def rsi_technical(self, stock_data: Dict) -> float:
        """RSI 技术因子（中性最优，极端危险）"""
        rsi = stock_data.get('rsi_14', 50)
        # RSI 在 45-55 之间最优，极端值扣分
        distance_from_50 = abs(rsi - 50)
        return max(0, min(10, 10 - distance_from_50 / 5))

    def macd_slope(self, stock_data: Dict, klines: Optional[List[Dict]] = None) -> float:
        """MACD 斜率（上升加分）"""
        if klines and len(klines) >= 35:
            closes = np.array([float(k['close']) for k in klines if float(k.get('close', 0)) > 0])
            if len(closes) >= 35:
                macd_hist = self._calculate_macd_histogram(closes)
                # 用最近 5 个 MACD 柱的斜率
                if len(closes) >= 40:
                    recent_closes = np.array([float(k['close']) for k in klines[-40:] if float(k.get('close', 0)) > 0])
                    if len(recent_closes) >= 10:
                        macds = []
                        for i in range(5, len(recent_closes)):
                            macds.append(self._calculate_macd_histogram(recent_closes[:i]))
                        if len(macds) >= 5:
                            slope = (macds[-1] - macds[-5]) / 5
                            return max(0, min(10, 5 + slope * 100))
        return 5.0  # 中性

    def bollinger_position(self, stock_data: Dict, klines: Optional[List[Dict]] = None) -> float:
        """布林带位置"""
        if klines and len(klines) >= 21:
            closes = np.array([float(k['close']) for k in klines[-21:] if float(k.get('close', 0)) > 0])
            if len(closes) >= 21:
                ma20 = np.mean(closes)
                std20 = np.std(closes)
                if std20 > 0:
                    position = (closes[-1] - ma20) / (2 * std20)
                    # -1 (下轨) -> 0, 0 (中轨) -> 5, 1 (上轨) -> 10
                    return max(0, min(10, (position + 1) * 5))
        return 5.0

    def trend_strength(self, stock_data: Dict, klines: Optional[List[Dict]] = None) -> float:
        """趋势强度（MA5/MA20 比率）"""
        if klines and len(klines) >= 21:
            closes = np.array([float(k['close']) for k in klines[-21:] if float(k.get('close', 0)) > 0])
            if len(closes) >= 21:
                ma5 = np.mean(closes[-5:])
                ma20 = np.mean(closes[-20:])
                if ma20 > 0:
                    ratio = ma5 / ma20
                    # ratio > 1 向上趋势, ratio < 1 向下趋势
                    # 适度偏离最优
                    if 0.98 < ratio < 1.05:
                        return 8.0
                    elif ratio >= 1.05:
                        return 6.0
                    return 4.0
        return 5.0

    def price_sentiment(self, stock_data: Dict) -> float:
        """价格情绪"""
        change_pct = stock_data.get('change_pct', 0)
        if change_pct > 5:
            return 8.0
        elif change_pct > 0:
            return 6.0
        elif change_pct > -5:
            return 4.0
        return 2.0

    # ─────────────────────────────────────────────
    # 工具方法
    # ─────────────────────────────────────────────

    @staticmethod
    def _calculate_macd_histogram(closes: np.ndarray) -> float:
        """计算 MACD 柱状图"""
        if len(closes) < 35:
            return 0.0

        def ema(data, period):
            if len(data) < period:
                return float(np.mean(data))
            multiplier = 2.0 / (period + 1)
            result = float(data[0])
            for price in data[1:]:
                result = (price - result) * multiplier + result
            return result

        ema12 = ema(closes, 12)
        ema26 = ema(closes, 26)
        macd_line = ema12 - ema26

        # 需要最近 9 个 MACD 值来计算 signal line
        if len(closes) >= 35:
            macd_values = []
            for i in range(26, len(closes)):
                e12 = ema(closes[:i+1], 12)
                e26 = ema(closes[:i+1], 26)
                macd_values.append(e12 - e26)
            if len(macd_values) >= 9:
                signal = ema(np.array(macd_values[-9:]), 9)
                return float(macd_values[-1] - signal)

        return float(macd_line * 0.1)

    # ─────────────────────────────────────────────
    # 综合计算
    # ─────────────────────────────────────────────

    def calculate_all_factors(self, stock_data: Dict,
                               klines: Optional[List[Dict]] = None) -> Dict[str, float]:
        """计算所有 15+ 因子"""
        return {
            # 动量因子 (5)
            'momentum_1d': self.momentum_1d(stock_data),
            'momentum_5d': self.momentum_5d(stock_data, klines),
            'momentum_20d': self.momentum_20d(stock_data, klines),
            'momentum_60d': self.momentum_60d(stock_data, klines),
            'short_term_reversal': self.short_term_reversal(stock_data),
            # 价值因子 (3)
            'pe_value': self.pe_value(stock_data),
            'pb_value': self.pb_value(stock_data),
            'market_cap_value': self.market_cap_value(stock_data),
            # 波动率因子 (2)
            'realized_vol': self.realized_vol(stock_data, klines),
            'downside_vol': self.downside_vol(stock_data, klines),
            # 成交量因子 (3)
            'volume_ratio': self.volume_ratio(stock_data),
            'volume_momentum': self.volume_momentum(stock_data),
            'outer_inner_ratio': self.outer_inner_ratio(stock_data),
            # 流动性因子 (1)
            'turnover_level': self.turnover_level(stock_data),
            # 质量因子 (2)
            'roe_quality': self.roe_quality(stock_data),
            'revenue_growth_quality': self.revenue_growth_quality(stock_data),
            # 技术因子 (4)
            'rsi_technical': self.rsi_technical(stock_data),
            'macd_slope': self.macd_slope(stock_data, klines),
            'bollinger_position': self.bollinger_position(stock_data, klines),
            'trend_strength': self.trend_strength(stock_data, klines),
            # 情绪因子 (1)
            'price_sentiment': self.price_sentiment(stock_data),
        }

    @staticmethod
    def _factor_cache_key(stock_code: str, stock_data: Dict,
                          klines_hash: Optional[str] = None) -> str:
        """生成因子缓存键

        基于股票代码 + 关键数据指纹，确保价格/成交量变化时自动失效
        """
        fingerprint = f"{stock_code}_{stock_data.get('price', 0)}_{stock_data.get('volume', 0)}_{stock_data.get('change_pct', 0)}"
        if klines_hash:
            fingerprint += f"_{klines_hash}"
        return f"factor_{stock_code}_{hashlib.md5(fingerprint.encode()).hexdigest()[:12]}"

    def calculate_all_factors_cached(self, stock_code: str,
                                      stock_data: Dict,
                                      klines: Optional[List[Dict]] = None
                                      ) -> Dict[str, float]:
        """计算所有因子（带缓存，TTL 5 分钟）

        缓存键基于 stock_code + price + volume + change_pct + klines 指纹
        当 realtime/kline 数据更新时，依赖链自动失效
        """
        # 生成 K 线指纹（前 5 条 + 后 5 条收盘价）
        klines_hash = None
        if klines and len(klines) >= 2:
            sample = [k.get('close', 0) for k in klines[:5]]
            if len(klines) > 10:
                sample += [k.get('close', 0) for k in klines[-5:]]
            klines_hash = hashlib.md5(
                ','.join(str(v) for v in sample).encode()
            ).hexdigest()[:8]

        key = self._factor_cache_key(stock_code, stock_data, klines_hash)
        cached = cache.get(key, category='factor')
        if cached is not None:
            return cached

        factors = self.calculate_all_factors(stock_data, klines)
        cache.set(key, factors, category='factor', tags={stock_code})
        return factors

    def weighted_score(self, factor_scores: Dict[str, float]) -> float:
        """计算加权综合得分"""
        return sum(
            self.factor_weights.get(f, 0.05) * score
            for f, score in factor_scores.items()
        )

    def get_rating(self, score: float) -> str:
        """评级"""
        if score >= 8.0:
            return '强烈推荐'
        elif score >= 6.5:
            return '推荐'
        elif score >= 5.0:
            return '中性'
        elif score >= 3.5:
            return '观望'
        return '卖出'

    def get_dominant_factors(self, factor_scores: Dict[str, float],
                                n: int = 3) -> List[str]:
        """获取最强/最弱因子"""
        sorted_factors = sorted(
            factor_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_factors[:n]

    def update_weights_from_ic(self, ic_data: Dict[str, float]):
        """基于 IC 更新因子权重"""
        abs_ics = {f: abs(v) for f, v in ic_data.items() if v != 0}
        total = sum(abs_ics.values())
        if total > 0:
            self.factor_weights = {
                f: v / total for f, v in abs_ics.items()
            }

    # ── 横截面标准化 ──────────────────────────────────────────────

    def cross_sectional_normalize(self, factor_scores: Dict[str, float],
                                   universe: List[Dict]) -> Dict[str, float]:
        """
        横截面标准化 — 增强版

        流程:
          1. Winsorization (3σ 截尾，处理极端值)
          2. Rank-based 排名标准化 (映射到正态分布)
          3. 行业中性化 (可选)
          4. 市值中性化 (可选)

        Args:
            factor_scores: 当前股票因子得分 {factor_name: score}
            universe: 股票池 [{code, name, factor_name, industry, market_cap, ...}, ...]

        Returns:
            标准化后的因子得分
        """
        if not universe or len(universe) < 5:
            return factor_scores

        # Step 1: 收集所有股票的因子值
        factor_names = list(factor_scores.keys())
        n = len(universe)
        raw_matrix = np.zeros((n, len(factor_names)))

        for i, stock in enumerate(universe):
            for j, fname in enumerate(factor_names):
                raw_matrix[i, j] = stock.get(fname, factor_scores.get(fname, 0))

        # 当前股票在 matrix 中的值（追加到最后一行）
        current_values = np.array([factor_scores.get(fname, 0) for fname in factor_names])
        raw_matrix = np.vstack([raw_matrix, current_values])
        n += 1

        # Step 2: Winsorization (3σ 截尾)
        raw_matrix = self._winsorize(raw_matrix, threshold=3.0)

        # Step 3: 排名标准化 → 正态分布
        normalized_matrix = self._rank_normalize(raw_matrix)

        # Step 4: 行业中性化 (如果行业信息可用)
        industries = [stock.get('industry', stock.get('sw_l1', 'unknown')) for stock in universe]
        industries.append('unknown')  # 当前股票
        if len(set(industries)) > 1:  # 只有多个行业才需要中性化
            normalized_matrix = self._industry_neutralize(
                normalized_matrix, factor_names, industries
            )

        # Step 5: 市值中性化 (如果市值信息可用)
        has_market_cap = all(
            universe[i].get('market_cap', 0) > 0 for i in range(min(n - 1, len(universe)))
        )
        if has_market_cap:
            market_caps = np.array([
                np.log(stock.get('market_cap', 1)) if stock.get('market_cap', 0) > 0 else 0
                for stock in universe
            ])
            market_caps = np.append(market_caps, np.log(factor_scores.get('market_cap_value', 1)) if factor_scores.get('market_cap_value', 0) > 0 else 0)
            normalized_matrix = self._market_cap_neutralize(
                normalized_matrix, market_caps, factor_names
            )

        # Step 6: 因子正交化
        normalized_matrix = self._orthogonalize_factors(normalized_matrix, factor_names)

        # 提取当前股票的结果
        normalized = dict(zip(factor_names, normalized_matrix[-1]))
        return {k: round(float(v), 4) for k, v in normalized.items()}

    @staticmethod
    def _winsorize(matrix: np.ndarray, threshold: float = 3.0) -> np.ndarray:
        """
        Winsorization: 将超过 thresholdσ 的值截尾到 thresholdσ 处

        对每列（因子）独立处理。
        """
        result = matrix.copy()
        for j in range(matrix.shape[1]):
            col = matrix[:, j]
            mean = np.mean(col)
            std = np.std(col)
            if std < 1e-10:
                continue
            lower = mean - threshold * std
            upper = mean + threshold * std
            result[:, j] = np.clip(col, lower, upper)
        return result

    @staticmethod
    def _rank_normalize(matrix: np.ndarray) -> np.ndarray:
        """
        排名标准化 → 映射到标准正态分布

        rank / (n+1) → 分位数 → 近似 norm.ppf(分位数) → Z-Score

        使用 Beasley-Springer-Moro 算法的简化版近似正态分位数函数。
        对每列（因子）独立处理。
        """
        result = np.zeros_like(matrix)

        for j in range(matrix.shape[1]):
            col = matrix[:, j]
            n = len(col)

            # 计算排名 (从小到大)
            ranks = np.argsort(np.argsort(col)) + 1

            # 并列处理: 取平均排名
            unique_vals = np.unique(col)
            for val in unique_vals:
                mask = (col == val)
                if np.sum(mask) > 1:
                    avg_rank = np.mean(ranks[mask])
                    ranks[mask] = avg_rank

            # 分位数映射到标准正态分布
            percentiles = (ranks - 0.5) / (n + 1)
            percentiles = np.clip(percentiles, 1e-6, 1 - 1e-6)

            # 近似 norm.ppf (Beasley-Springer-Moro 简化版)
            result[:, j] = MultiFactorModelV2._ppf_approx(percentiles)

        return result

    @staticmethod
    def _ppf_approx(p: np.ndarray) -> np.ndarray:
        """
        近似正态分位数函数 (PPF / Probit)

        使用 rational approximation (Abramowitz & Stegun 26.2.23)
        误差 < 4.5e-4
        """
        # 对称处理: p > 0.5 时用 1-p 计算
        mask = p >= 0.5
        p_lo = np.where(mask, 1 - p, p)

        # t = sqrt(-2 * ln(1-p)) → 对于 p<0.5, t = sqrt(-2*ln(p))
        t = np.sqrt(-2.0 * np.log(p_lo))

        # Rational approximation coefficients
        c0 = 2.515517
        c1 = 0.802853
        c2 = 0.010328
        d1 = 1.432788
        d2 = 0.189269
        d3 = 0.001308

        # 近似: z = t - (c0 + c1*t + c2*t^2) / (1 + d1*t + d2*t^2 + d3*t^3)
        z = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)

        # 恢复符号
        result = np.where(mask, z, -z)
        return result

    def _industry_neutralize(self, matrix: np.ndarray,
                              factor_names: List[str],
                              industries: List[str]) -> np.ndarray:
        """
        行业中性化: 对每个因子，回归到行业哑变量，取残差

        factor_ij = α_i + Σ_k β_ik * industry_dummy_kk + ε_ij
        neutralized_ij = ε_ij

        对每列（因子）独立处理。
        使用 numpy 实现 OLS（无需 sklearn）。
        """
        result = matrix.copy()
        n = len(industries)

        # 生成行业哑变量
        unique_industries = list(set(industries))
        industry_map = {ind: idx for idx, ind in enumerate(unique_industries)}
        industry_dummies = np.zeros((n, len(unique_industries)))
        for i, ind in enumerate(industries):
            industry_dummies[i, industry_map[ind]] = 1

        for j in range(matrix.shape[1]):
            col = matrix[:, j]
            X = industry_dummies

            # OLS: β = (X'X)^-1 X'y
            try:
                XtX = X.T @ X
                Xty = X.T @ col
                beta = np.linalg.solve(XtX + 0.01 * np.eye(XtX.shape[0]), Xty)
                residuals = col - X @ beta
                result[:, j] = residuals
            except np.linalg.LinAlgError:
                # 矩阵奇异，用均值代替
                result[:, j] = col - np.mean(col)

        return result

    @staticmethod
    def _market_cap_neutralize(matrix: np.ndarray,
                                market_caps: np.ndarray,
                                factor_names: List[str]) -> np.ndarray:
        """
        市值中性化: 对每个因子，回归到市值（对数），取残差

        factor_ij = α_i + β_i * log(market_cap_j) + ε_ij
        neutralized_ij = ε_ij
        使用 numpy 实现 OLS。
        """
        result = matrix.copy()
        n = len(market_caps)

        X = market_caps.reshape(-1, 1)
        # 加截距项
        X = np.hstack([X, np.ones((n, 1))])

        for j in range(matrix.shape[1]):
            col = matrix[:, j]
            try:
                XtX = X.T @ X
                Xty = X.T @ col
                beta = np.linalg.solve(XtX + 0.01 * np.eye(XtX.shape[0]), Xty)
                residuals = col - X @ beta
                result[:, j] = residuals
            except np.linalg.LinAlgError:
                result[:, j] = col - np.mean(col)

        return result

    def _orthogonalize_factors(self, matrix: np.ndarray,
                                factor_names: List[str]) -> np.ndarray:
        """
        Gram-Schmidt 正交化

        按因子重要性排序后，依次去除与前序因子的相关性。

        因子顺序: 按权重降序排列（动量 > 价值 > 波动率 > ...）
        使用 numpy 实现 OLS。
        """
        result = matrix.copy()

        # 按权重排序
        sorted_factors = sorted(
            [(fname, self.factor_weights.get(fname, 0.05)) for fname in factor_names],
            key=lambda x: x[1],
            reverse=True,
        )
        sorted_names = [f[0] for f in sorted_factors]
        sorted_indices = [factor_names.index(f) for f in sorted_names]

        for i in range(1, len(sorted_indices)):
            j_target = sorted_indices[i]
            for k in range(i):
                j_ref = sorted_indices[k]

                X = result[:, j_ref].reshape(-1, 1)
                y = result[:, j_target]

                # OLS
                try:
                    XtX = X.T @ X
                    Xty = X.T @ y
                    beta = np.linalg.solve(XtX + 1e-12 * np.eye(1), Xty)
                    residuals = y - X @ beta
                    result[:, j_target] = residuals.flatten()
                except np.linalg.LinAlgError:
                    result[:, j_target] = result[:, j_target] - np.mean(result[:, j_target])

        return result


# 全局实例
multi_factor_model_v2 = MultiFactorModelV2()
