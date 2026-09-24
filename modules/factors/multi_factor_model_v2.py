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
import warnings
warnings.filterwarnings('ignore')

from modules.dynamic_cache import cache
from modules.logger import logger
from modules.factors.alpha158_calculator import alpha158_calculator
from modules.factors.alpha360_calculator import alpha360_calculator
from modules.factors.barra_cne6_calculator import barra_cne6_calculator


class MultiFactorModelV2:
    """多因子模型 V2 — 15 因子 + 动态权重 + 正交化"""

    # 因子分类 (扩展: 基础因子 + Alpha158 + Alpha360 + Barra CNE6)
    FACTOR_CATEGORIES = {
        'momentum': ['momentum_1d', 'momentum_5d', 'momentum_20d', 'momentum_60d', 'short_term_reversal'],
        'value': ['pe_value', 'pb_value', 'market_cap_value'],
        'volatility': ['realized_vol', 'downside_vol', 'garch_vol'],
        'volume': ['volume_ratio', 'volume_momentum', 'outer_inner_ratio'],
        'liquidity': ['turnover_level', 'amount_level'],
        'quality': ['roe_quality', 'revenue_growth_quality', 'margin_quality'],
        'technical': ['rsi_technical', 'macd_slope', 'bollinger_position', 'trend_strength'],
        'sentiment': ['price_sentiment', 'volume_sentiment'],
        'fund_flow': ['northbound_flow'],
        'leverage': ['margin_balance_ratio'],
        'analyst': ['analyst_rating_change'],
        'macro': ['macro_liquidity_proxy'],
        # Alpha158 价量因子 (示例: 取前 20 个)
        'alpha158': ['momentum_5', 'momentum_10', 'momentum_20', 'momentum_60',
                     'reversal_1', 'reversal_3', 'reversal_5',
                     'rsi_14', 'rsi_7', 'rsi_21',
                     'macd', 'macd_signal', 'macd_hist',
                     'boll_position', 'atr_14',
                     'volume_ratio_5', 'volume_ratio_10', 'volume_ratio_20',
                     'price_position_20', 'price_position_60'],
        # Alpha360 技术因子
        'alpha360': ['macd_hist', 'macd_golden_cross', 'macd_death_cross',
                     'bb_pct_b_20', 'bb_bandwidth_20', 'bb_position_20',
                     'kdj_k', 'kdj_d', 'kdj_j',
                     'kdj_golden_cross', 'kdj_death_cross',
                     'chip_concentration', 'chip_profit_ratio',
                     'cmf_20', 'obv_slope_10', 'net_flow_ratio',
                     'rsi_macd_bull', 'rsi_macd_bear', 'triple_bull_signal',
                     'triple_bear_signal', 'trend_strength_index',
                     'momentum_10', 'momentum_20', 'momentum_30', 'momentum_60',
                     'williams_r_14', 'cci_20', 'atr_14', 'atr_pct_14'],
        # Barra CNE6 风格因子
        'barra_size': ['size'],
        'barra_value': ['btoe', 'btope', 'bvopen'],
        'barra_growth': ['egr', 'dgr', 'ogr'],
        'barra_profitability': ['roe', 'roa', 'gross_margin'],
        'barra_momentum': ['momentum_12_1', 'short_term_reversal', 'long_term_reversal'],
        'barra_risk': ['beta', 'idvol', 'volatility'],
        'barra_liquidity': ['turnover', 'illiquidity', 'nlv'],
        'barra_leverage': ['de', 'dlr'],
        'barra_dividend': ['dividend_yield'],
        'barra_quality': ['profit_growth', 'asset_turnover', 'working_capital'],
        'barra_technical': ['price_ma_ratio', 'rsi_momentum'],
    }

    # 默认因子权重 (扩展: 基础因子 + Alpha360 + Barra CNE6)
    DEFAULT_WEIGHTS = {
        # 基础动量因子
        'momentum_1d': 0.06,
        'momentum_5d': 0.06,
        'momentum_20d': 0.05,
        'momentum_60d': 0.04,
        'short_term_reversal': 0.04,
        # 基础价值因子
        'pe_value': 0.06,
        'pb_value': 0.04,
        'market_cap_value': 0.03,
        # 基础波动率因子
        'realized_vol': 0.04,
        'downside_vol': 0.03,
        'garch_vol': 0.03,
        # 基础成交量因子
        'volume_ratio': 0.05,
        'volume_momentum': 0.04,
        'outer_inner_ratio': 0.04,
        # 基础流动性因子
        'turnover_level': 0.03,
        # 基础质量因子
        'roe_quality': 0.05,
        'revenue_growth_quality': 0.04,
        # 基础技术因子
        'rsi_technical': 0.04,
        'macd_slope': 0.04,
        'bollinger_position': 0.03,
        'trend_strength': 0.04,
        # 基础情绪因子
        'price_sentiment': 0.02,
        # 基础资金面因子
        'northbound_flow': 0.03,
        # 基础杠杆因子
        'margin_balance_ratio': 0.02,
        # 基础分析师因子
        'analyst_rating_change': 0.03,
        # 基础宏观因子
        'macro_liquidity_proxy': 0.02,
        # Alpha360 技术因子 (共享动量/波动率权重)
        'macd_hist': 0.04,
        'macd_golden_cross': 0.03,
        'macd_death_cross': 0.03,
        'bb_pct_b_20': 0.03,
        'bb_bandwidth_20': 0.02,
        'bb_position_20': 0.03,
        'kdj_k': 0.03,
        'kdj_j': 0.03,
        'kdj_golden_cross': 0.02,
        'kdj_death_cross': 0.02,
        'chip_concentration': 0.02,
        'chip_profit_ratio': 0.02,
        'cmf_20': 0.03,
        'obv_slope_10': 0.02,
        'net_flow_ratio': 0.03,
        'rsi_macd_bull': 0.03,
        'rsi_macd_bear': 0.03,
        'triple_bull_signal': 0.04,
        'triple_bear_signal': 0.04,
        'trend_strength_index': 0.03,
        'momentum_10': 0.04,
        'momentum_20': 0.04,
        'momentum_30': 0.03,
        'momentum_60': 0.03,
        'williams_r_14': 0.02,
        'cci_20': 0.02,
        'atr_14': 0.02,
        'atr_pct_14': 0.02,
        # Barra CNE6 风格因子
        'size': 0.04,
        'btoe': 0.04,
        'btope': 0.03,
        'bvopen': 0.03,
        'roe': 0.05,
        'roa': 0.03,
        'gross_margin': 0.03,
        'momentum_12_1': 0.05,
        'long_term_reversal': 0.03,
        'beta': 0.04,
        'idvol': 0.03,
        'volatility': 0.03,
        'turnover': 0.04,
        'illiquidity': 0.03,
        'nlv': 0.03,
        'de': 0.03,
        'dlr': 0.02,
        'dividend_yield': 0.02,
        'egr': 0.04,
        'dgr': 0.03,
        'ogr': 0.03,
        'profit_growth': 0.04,
        'asset_turnover': 0.03,
        'working_capital': 0.02,
        'price_ma_ratio': 0.03,
        'rsi_momentum': 0.02,
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

    def garch_vol(self, stock_data: Dict, klines: Optional[List[Dict]] = None) -> float:
        """GARCH(1,1) 条件波动率（越低越好，预测未来波动）"""
        if klines and len(klines) >= 60:
            closes = np.array([float(k['close']) for k in klines[-60:] if float(k.get('close', 0)) > 0])
            if len(closes) >= 30:
                try:
                    from modules.factors.garch_volatility import GARCHVolatility
                    returns = np.diff(np.log(closes)).tolist()
                    garch = GARCHVolatility(omega=1e-6, alpha=0.05, beta=0.94)
                    forecast = garch.forecast_volatility(returns, steps_ahead=1)
                    # 年化波动率转为百分制（低波动得分高）
                    vol_pct = forecast * 100
                    return max(0, min(10, 10 - vol_pct / 10))
                except Exception:
                    pass
        # 降级: 用 realized_vol 近似
        return self.realized_vol(stock_data, klines)

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
        """RSI 技术因子 — 超卖看涨(高分), 超买看跌(低分)

        RSI < 30: 超卖 → 强烈看涨 → 10分
        RSI 30-40: 偏弱 → 6分
        RSI 40-60: 中性 → 5分
        RSI 60-70: 偏强 → 3分
        RSI > 70: 超买 → 强烈看跌 → 0分
        """
        rsi = stock_data.get('rsi_14', 50)
        if rsi < 20:    return 10.0  # 深度超卖 → 强烈看涨
        elif rsi < 30:  return 8.0   # 超卖 → 看涨
        elif rsi < 40:  return 6.0   # 偏弱
        elif rsi < 60:  return 5.0   # 中性
        elif rsi < 70:  return 3.0   # 偏强
        elif rsi < 80:  return 2.0   # 超买 → 看跌
        else:           return 0.0   # 深度超买 → 强烈看跌

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
    # 新增因子: 资金面 / 杠杆 / 分析师 / 宏观
    # ─────────────────────────────────────────────

    def northbound_flow(self, stock_data: Dict) -> float:
        """北向资金净买入因子 (越高越好)

        通过 AKShare 获取北向资金当日净买入额，标准化为 0-10 分。
        降级: 用换手率变化代理资金活跃度。
        """
        # 优先使用 stock_data 中的北向数据
        north_net = stock_data.get('northbound_net_flow', 0)  # 亿元
        if north_net != 0:
            #  sigmoid 映射: 0 → 5, +5亿 → ~8.8, -5亿 → ~1.2
            return float(10 / (1 + np.exp(-north_net / 2)))

        # 降级: 用换手率变化代理
        turnover = stock_data.get('turnover', 5)
        prev_turnover = stock_data.get('prev_turnover', turnover)
        if prev_turnover > 0:
            turnover_change = (turnover - prev_turnover) / prev_turnover
            return max(0, min(10, 5 + turnover_change * 50))
        return 5.0

    def margin_balance_ratio(self, stock_data: Dict) -> float:
        """融资融券余额占比因子 (杠杆情绪，过高看跌)

        融资余额/流通市值 比值过高说明杠杆过热，反向指标。
        降级: 用换手率代理。
        """
        margin_balance = stock_data.get('margin_balance', 0)  # 融资余额
        float_cap = stock_data.get('float_market_cap', 1)  # 流通市值
        if float_cap > 0 and margin_balance > 0:
            ratio = margin_balance / float_cap
            # ratio 通常 0.01-0.05, 越高越过热 → 得分越低
            score = max(0, min(10, 10 - ratio * 500))
            return score
        # 降级: 无数据用中性分
        return 5.0

    def analyst_rating_change(self, stock_data: Dict) -> float:
        """分析师评级变化因子 (上调看涨，下调看跌)

        使用分析师评级均值变化 (1=强烈推荐 → 5=卖出)。
        降级: 用营收增速代理基本面预期。
        """
        rating = stock_data.get('analyst_rating', 3)  # 1-5, 1=强烈推荐
        rating_change = stock_data.get('analyst_rating_change', 0)  # 较上期变化 (+1=上调)

        if rating_change != 0:
            # 评级上调 → 高分
            return max(0, min(10, 5 + rating_change * 3))

        # 降级: 用营收增速
        revenue_growth = stock_data.get('revenue_yoy', 0)
        if revenue_growth > 30:
            return 8.0
        elif revenue_growth > 10:
            return 6.0
        elif revenue_growth > 0:
            return 4.0
        return 2.0

    def macro_liquidity_proxy(self, stock_data: Dict) -> float:
        """宏观流动性代理因子

        使用市场整体成交额/换手率作为流动性 proxy。
        流动性充裕 → 高分。
        """
        market_amount = stock_data.get('market_total_amount', 0)  # 全市场成交额 亿元
        prev_amount = stock_data.get('prev_market_amount', 0)

        if market_amount > 0 and prev_amount > 0:
            # 成交额同比变化
            amount_change = (market_amount - prev_amount) / prev_amount
            # 成交额增加 → 流动性改善
            return max(0, min(10, 5 + amount_change * 20))

        # 降级: 用个股换手率
        turnover = stock_data.get('turnover', 5)
        return max(0, min(10, turnover / 2))

    # ─────────────────────────────────────────────
    # 工具方法
    # ─────────────────────────────────────────────

    @staticmethod
    def _calculate_macd_histogram(closes: np.ndarray) -> float:
        """计算 MACD 柱状图（委托给共享工具）"""
        from modules.technical import macd_histogram
        return macd_histogram(closes)

    # ─────────────────────────────────────────────
    # Alpha360 因子计算
    # ─────────────────────────────────────────────

    def calculate_alpha360_factors(self, klines: List[Dict],
                                   stock_code: str = '') -> Dict[str, float]:
        """
        计算 Alpha360 技术因子

        调用 alpha360_calculator 模块，返回标准化后的因子值。
        stock_code 传入时走 factor_cache 语义缓存旁路 (2026-09-15,
        killswitch=FACTOR_SEMANTIC_CACHE=0); 无代码时自动退回直算。
        """
        if not klines or len(klines) < 30:
            logger.debug(f"[MultiFactorModelV2] K 线数据不足 ({len(klines) if klines else 0} 条)，跳过 Alpha360")
            return {}

        try:
            factors = alpha360_calculator.calculate_all(
                klines, stock_code=stock_code, use_cache=True)
            # 将因子值从 0-10 范围映射到 -1~1 (与 IC 计算兼容)
            normalized = {k: (v - 5) / 5 for k, v in factors.items()}
            logger.debug(f"[MultiFactorModelV2] Alpha360 计算完成: {len(factors)} 个因子")
            return normalized
        except Exception as e:
            logger.error(f"[MultiFactorModelV2] Alpha360 计算失败: {e}")
            return {}

    # ─────────────────────────────────────────────
    # Barra CNE6 风格因子计算
    # ─────────────────────────────────────────────

    def calculate_barra_factors(self, stock_data: Dict,
                                 klines: Optional[List[Dict]] = None,
                                 fundamentals: Optional[Dict] = None) -> Dict[str, float]:
        """
        计算 Barra CNE6 风格因子

        调用 barra_cne6_calculator 模块，返回标准化后的因子值。
        """
        try:
            factors = barra_cne6_calculator.calculate_all(stock_data, klines, fundamentals)
            logger.debug(f"[MultiFactorModelV2] Barra CNE6 计算完成: {len(factors)} 个因子")
            return factors
        except Exception as e:
            logger.error(f"[MultiFactorModelV2] Barra CNE6 计算失败: {e}")
            return {}

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
            # 波动率因子 (3)
            'realized_vol': self.realized_vol(stock_data, klines),
            'downside_vol': self.downside_vol(stock_data, klines),
            'garch_vol': self.garch_vol(stock_data, klines),
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
            # 资金面因子 (1)
            'northbound_flow': self.northbound_flow(stock_data),
            # 杠杆因子 (1)
            'margin_balance_ratio': self.margin_balance_ratio(stock_data),
            # 分析师因子 (1)
            'analyst_rating_change': self.analyst_rating_change(stock_data),
            # 宏观因子 (1)
            'macro_liquidity_proxy': self.macro_liquidity_proxy(stock_data),
        }

    def calculate_all_factors_with_alpha158(self, stock_data: Dict,
                                               klines: Optional[List[Dict]] = None) -> Dict[str, float]:
        """
        计算所有因子（包含 Alpha158 扩展因子）

        扩展内容:
        - 在原有 21 个因子基础上，增加 Alpha158 的 69 个因子
        - 总计约 90 个因子

        Returns:
            {factor_name: factor_value} 因子字典
        """
        # 获取原有因子
        base_factors = self.calculate_all_factors(stock_data, klines)

        # 获取 Alpha158 因子
        if klines and len(klines) >= 60:
            try:
                alpha_factors = alpha158_calculator.calculate_all(klines)
                # 归一化 alpha 因子到 0-10 范围（与原有因子一致）
                for name, value in alpha_factors.items():
                    if isinstance(value, (int, float)) and not np.isnan(value) and not np.isinf(value):
                        # 压缩到 0-10 范围
                        normalized = float(np.clip(value * 5 + 5, 0, 10))
                        base_factors[f'alpha158_{name}'] = normalized
                    else:
                        base_factors[f'alpha158_{name}'] = 5.0  # 中性值
            except Exception as e:
                logger.debug(f"[MultiFactorModelV2] Alpha158 计算失败: {e}")
        else:
            logger.debug(f"[MultiFactorModelV2] K 线数据不足，无法计算 Alpha158")

        return base_factors

    def calculate_all_factors_comprehensive(self, stock_code: str,
                                           stock_data: Dict,
                                           klines: Optional[List[Dict]] = None,
                                           fundamentals: Optional[Dict] = None
                                           ) -> Dict[str, float]:
        """
        综合因子计算 — 基础因子 + Alpha158 + Alpha360 + Barra CNE6

        这是最全面的因子计算方法，整合了:
        1. 基础 21 因子 (动量/价值/波动率/成交量/质量/技术/情绪/资金面/杠杆/分析师/宏观)
        2. Alpha158 价量因子 (~90 个)
        3. Alpha360 技术因子 (~40 个)
        4. Barra CNE6 风格因子 (~27 个)

        总计: ~180 个因子

        Args:
            stock_code: 股票代码
            stock_data: 股票数据字典
            klines: K 线数据
            fundamentals: 基本面数据

        Returns:
            {factor_name: factor_value} 因子字典
        """
        result = {}

        # 1. 基础 21 因子
        try:
            base_factors = self.calculate_all_factors(stock_data, klines)
            result.update(base_factors)
            logger.debug(f"[MultiFactorModelV2] 基础因子: {len(base_factors)} 个")
        except Exception as e:
            logger.error(f"[MultiFactorModelV2] 基础因子计算失败: {e}")

        # 2. Alpha158 价量因子
        if klines and len(klines) >= 60:
            try:
                alpha_factors = alpha158_calculator.calculate_all(
                    klines, stock_code=stock_code, use_cache=True)
                for name, value in alpha_factors.items():
                    if isinstance(value, (int, float)) and not np.isnan(value) and not np.isinf(value):
                        normalized = float(np.clip(value * 5 + 5, 0, 10))
                        result[f'alpha158_{name}'] = normalized
                    else:
                        result[f'alpha158_{name}'] = 5.0
                logger.debug(f"[MultiFactorModelV2] Alpha158 因子: {len(alpha_factors)} 个")
            except Exception as e:
                logger.error(f"[MultiFactorModelV2] Alpha158 计算失败: {e}")

        # 3. Alpha360 技术因子
        if klines and len(klines) >= 30:
            try:
                alpha360_factors = self.calculate_alpha360_factors(klines, stock_code)
                result.update(alpha360_factors)
                logger.debug(f"[MultiFactorModelV2] Alpha360 因子: {len(alpha360_factors)} 个")
            except Exception as e:
                logger.error(f"[MultiFactorModelV2] Alpha360 计算失败: {e}")

        # 4. Barra CNE6 风格因子
        try:
            barra_factors = self.calculate_barra_factors(stock_data, klines, fundamentals)
            result.update(barra_factors)
            logger.debug(f"[MultiFactorModelV2] Barra CNE6 因子: {len(barra_factors)} 个")
        except Exception as e:
            logger.error(f"[MultiFactorModelV2] Barra CNE6 计算失败: {e}")

        logger.info(f"[MultiFactorModelV2] 综合因子总计: {len(result)} 个")
        return result

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

    def update_weights_rolling_ic(
        self,
        factor_history: Dict[str, List[float]],
        returns: List[float],
        window: int = 60,
        min_icir: float = 0.05,
        decay: float = 0.95,
    ):
        """
        基于滚动 IC/ICIR 动态更新因子权重

        原理:
            - 计算每个因子在最近 window 天内的 IC (Information Coefficient)
            - IC = 因子值与未来收益率的 Rank Spearman 相关系数
            - ICIR = mean(IC) / std(IC)，衡量稳定性
            - 权重 ∝ max(ICIR, min_icir)，确保每个因子至少有一定权重
            - 使用指数衰减，近期 IC 权重更高

        Args:
            factor_history: {factor_name: [factor_value_t1, factor_value_t2, ...]}
            returns: 未来收益率序列 [r_t1, r_t2, ...]
            window: 滚动窗口大小 (默认 60 天)
            min_icir: 最小 ICIR 阈值 (默认 0.05)
            decay: 指数衰减因子 (默认 0.95，越近的数据权重越高)
        """
        new_weights = {}
        for factor_name, factor_values in factor_history.items():
            if len(factor_values) < window:
                continue

            # 取最近 window 个值
            fv = np.array(factor_values[-window:])
            ret = np.array(returns[-window:])

            # 跳过全为常量的因子
            if np.std(fv) < 1e-10 or np.std(ret) < 1e-10:
                continue

            # Rank Spearman 相关系数
            from scipy.stats import rankdata
            fv_rank = rankdata(fv)
            ret_rank = rankdata(ret)
            ic = np.corrcoef(fv_rank, ret_rank)[0, 1]

            if np.isnan(ic):
                continue

            # 指数衰减 IC (近期权重更高)
            decay_weights = decay ** np.arange(window)
            decay_weights = decay_weights / decay_weights.sum()
            decay_ic = np.sum(decay_weights * fv_rank * ret_rank)
            decay_ic = decay_ic / (np.std(fv_rank) * np.std(ret_rank) + 1e-10)

            # ICIR (结合原始 IC 和衰减 IC)
            icir = 0.6 * abs(ic) + 0.4 * abs(decay_ic)
            if icir >= min_icir:
                new_weights[factor_name] = icir

        total = sum(new_weights.values())
        if total > 0:
            old_weights = dict(self.factor_weights)
            self.factor_weights = {k: v / total for k, v in new_weights.items()}
            logger.info(
                f"[FactorModel] 因子权重已基于滚动 IC/ICIR 更新 "
                f"(window={window}, decay={decay}, 有效因子={len(new_weights)}/{len(factor_history)})"
            )
            logger.debug(f"[FactorModel] 权重变化: {old_weights} → {self.factor_weights}")
        else:
            logger.warning("[FactorModel] 无因子满足 ICIR 阈值，权重未更新")

    def get_factor_efficacy_report(self, window: int = 60) -> Dict[str, float]:
        """获取因子效能报告 (IC/ICIR)"""
        report = {}
        for factor_name, factor_values in self._factor_ic_history.items():
            if len(factor_values) < window:
                continue
            vals = np.array(factor_values[-window:])
            ic = float(np.mean(vals))
            icir = ic / (np.std(vals) + 1e-10)
            report[factor_name] = round(icir, 4)
        return report

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

    def detect_high_correlation(self, matrix: np.ndarray,
                                 factor_names: List[str],
                                 threshold: float = 0.7) -> List[Dict]:
        """
        自动检测高相关性因子对

        Args:
            matrix: 标准化后的因子矩阵 (n_dates, n_factors)
            factor_names: 因子名称列表
            threshold: 相关系数阈值，默认 0.7

        Returns:
            高相关因子对列表 [{'factor_a': ..., 'factor_b': ..., 'correlation': ...}]
        """
        if matrix.shape[1] < 2:
            return []

        corr_matrix = np.corrcoef(matrix.T)
        n = len(factor_names)
        pairs = []

        for i in range(n):
            for j in range(i + 1, n):
                corr = abs(corr_matrix[i, j])
                if corr > threshold:
                    pairs.append({
                        'factor_a': factor_names[i],
                        'factor_b': factor_names[j],
                        'correlation': round(float(corr), 4),
                        'action': '降权或正交化',
                    })

        # 按相关性降序排序
        pairs.sort(key=lambda x: x['correlation'], reverse=True)
        return pairs

    def orthogonalize_with_pca(self, matrix: np.ndarray,
                                factor_names: List[str],
                                n_components: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """
        PCA 正交化

        将因子投影到主成分空间，消除共线性。

        Args:
            matrix: 标准化后的因子矩阵 (n_dates, n_factors)
            factor_names: 因子名称列表
            n_components: 保留的主成分数，None 表示保留全部

        Returns:
            (orthogonal_matrix, pca_info)
        """
        if matrix.shape[1] < 2:
            return matrix, {'method': 'pca', 'n_components': matrix.shape[1]}

        # 中心化
        mean = np.mean(matrix, axis=0)
        centered = matrix - mean

        # SVD
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)

        # 确定主成分数
        if n_components is None:
            # 保留累计方差贡献率 >= 95% 的主成分
            variance_explained = (S ** 2) / np.sum(S ** 2)
            cumulative = np.cumsum(variance_explained)
            n_components = int(np.searchsorted(cumulative, 0.95) + 1)
            n_components = max(1, min(n_components, matrix.shape[1]))

        # 投影到主成分
        orthogonal = U[:, :n_components] * S[:n_components]

        # 计算方差贡献率
        variance_explained = (S ** 2) / np.sum(S ** 2)

        pca_info = {
            'method': 'pca',
            'n_original': matrix.shape[1],
            'n_components': n_components,
            'variance_explained': [round(float(v), 4) for v in variance_explained[:n_components]],
            'cumulative_variance': [round(float(c), 4) for c in np.cumsum(variance_explained[:n_components])],
        }

        return orthogonal, pca_info

    def auto_weight_adjustment(self, factor_names: List[str],
                                ic_series: Dict[str, List[float]],
                                correlation_pairs: List[Dict]) -> Dict[str, float]:
        """
        基于 IC 和相关性的自动权重调整

        Args:
            factor_names: 因子名称列表
            ic_series: {factor_name: [ic_values]}
            correlation_pairs: 高相关因子对列表

        Returns:
            调整后的权重 {factor_name: weight}
        """
        adjusted_weights = dict(self.factor_weights)

        # 1. 基于 IC 调整权重
        for factor_name, series in ic_series.items():
            if len(series) < 12:
                continue
            arr = np.array(series)
            mean_ic = np.mean(arr)
            icir = mean_ic / np.std(arr) if np.std(arr) > 1e-10 else 0

            # ICIR 低则降权
            if icir < 0.3:
                adjusted_weights[factor_name] = max(0.01, adjusted_weights.get(factor_name, 0.05) * 0.5)
            elif icir > 1.0:
                adjusted_weights[factor_name] = min(2.0, adjusted_weights.get(factor_name, 0.05) * 1.5)

        # 2. 高相关因子对降权
        for pair in correlation_pairs:
            fa, fb = pair['factor_a'], pair['factor_b']
            if fa in adjusted_weights and fb in adjusted_weights:
                # 两个都降权
                adjusted_weights[fa] = max(0.01, adjusted_weights[fa] * 0.7)
                adjusted_weights[fb] = max(0.01, adjusted_weights[fb] * 0.7)

        # 3. 归一化权重
        total = sum(adjusted_weights.values())
        if total > 0:
            adjusted_weights = {k: v / total for k, v in adjusted_weights.items()}

        return adjusted_weights


# 全局实例
multi_factor_model_v2 = MultiFactorModelV2()
