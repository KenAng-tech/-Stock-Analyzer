#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
跨市场因子模块 — Cross-Market Factors

功能:
1. A 股/港股/美股跨市场因子
2. 汇率因子 (USD/CNY)
3. 利率因子 (国债收益率)
4. 大宗商品因子 (原油、铜、黄金)
5. 全球市场联动因子 (VIX、标普 500、纳斯达克)
6. 北向资金因子
7. 行业轮动因子

参考:
- Asness, C.J., Liew, J.M., and Stewart, L., "Simultaneous Measurement of Risk Premia" (1997)
- Ang, A., "Asset Pricing" (2008)
- 北向资金: 陆股通/深股通
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from modules.logger import logger


class CrossMarketFactors:
    """跨市场因子计算器"""

    def __init__(self):
        # 市场状态缓存
        self._market_regimes: Dict[str, str] = {}
        self._last_update: Dict[str, datetime] = {}

    def calculate_all(self, stock_data: Dict,
                      klines: Optional[List[Dict]] = None) -> Dict[str, float]:
        """计算所有跨市场因子"""
        return {
            # A 股/港股/美股联动
            'a_hk_correlation': self._a_hk_correlation(stock_data),
            'a_us_correlation': self._a_us_correlation(stock_data),
            'a_hk_us_correlation': self._a_hk_us_correlation(stock_data),

            # 汇率因子
            'usd_cny_rate': self._usd_cny_rate(stock_data),
            'usd_cny_change': self._usd_cny_change(stock_data),

            # 利率因子
            'treasury_yield': self._treasury_yield(stock_data),
            'yield_spread': self._yield_spread(stock_data),

            # 大宗商品因子
            'commodity_trend': self._commodity_trend(stock_data),
            'oil_price_trend': self._oil_price_trend(stock_data),
            'gold_price_trend': self._gold_price_trend(stock_data),

            # 全球市场联动
            'vix_level': self._vix_level(stock_data),
            'sp500_trend': self._sp500_trend(stock_data),
            'nasdaq_trend': self._nasdaq_trend(stock_data),

            # 北向资金
            'northbound_flow': self._northbound_flow(stock_data),
            'northbound_trend': self._northbound_trend(stock_data),

            # 行业轮动
            'sector_momentum': self._sector_momentum(stock_data),
            'sector_rotation_signal': self._sector_rotation_signal(stock_data),

            # 宏观流动性
            'macro_liquidity': self._macro_liquidity(stock_data),
            'm2_growth': self._m2_growth(stock_data),

            # 市场情绪
            'market_sentiment': self._market_sentiment(stock_data),
            'put_call_ratio': self._put_call_ratio(stock_data),
        }

    def _a_hk_correlation(self, stock_data: Dict) -> float:
        """A 股/港股相关性"""
        # 使用 HKEX 指数作为代理
        hk_correlation = stock_data.get('hk_correlation', 0.5)
        # 相关性越高，A 股受港股影响越大
        return float(np.clip(hk_correlation, 0, 1))

    def _a_us_correlation(self, stock_data: Dict) -> float:
        """A 股/美股相关性"""
        us_correlation = stock_data.get('us_correlation', 0.3)
        return float(np.clip(us_correlation, 0, 1))

    def _a_hk_us_correlation(self, stock_data: Dict) -> float:
        """A/HK/US 三者相关性"""
        a_hk = stock_data.get('hk_correlation', 0.5)
        a_us = stock_data.get('us_correlation', 0.3)
        return float((a_hk + a_us) / 2)

    def _usd_cny_rate(self, stock_data: Dict) -> float:
        """USD/CNY 汇率 (逆序: 人民币升值得分高)"""
        rate = stock_data.get('usd_cny_rate', 7.25)
        # 7.25 为基准，越高越弱
        return float(np.clip((7.5 - rate) * 10, 0, 10))

    def _usd_cny_change(self, stock_data: Dict) -> float:
        """USD/CNY 变化率"""
        change = stock_data.get('usd_cny_change', 0)
        # 人民币升值 (USD/CNY 下降) 对 A 股通常是利好
        return float(np.clip(-change * 100, -10, 10))

    def _treasury_yield(self, stock_data: Dict) -> float:
        """10 年期国债收益率"""
        yield_ = stock_data.get('treasury_yield_10y', 2.5)
        # 收益率适中最好 (太低表示经济弱，太高表示收紧)
        return float(np.clip(10 - abs(yield_ - 2.5) * 5, 0, 10))

    def _yield_spread(self, stock_data: Dict) -> float:
        """利差 (10y - 1y 国债收益率)"""
        spread = stock_data.get('yield_spread', 0.5)
        # 正利差表示经济预期乐观
        return float(np.clip(spread * 20, 0, 10))

    def _commodity_trend(self, stock_data: Dict) -> float:
        """大宗商品趋势"""
        trend = stock_data.get('commodity_trend', 0)
        return float(np.clip(trend * 5, -10, 10))

    def _oil_price_trend(self, stock_data: Dict) -> float:
        """原油价格趋势"""
        trend = stock_data.get('oil_price_trend', 0)
        # 原油上涨对制造业不利，对能源业有利
        return float(np.clip(trend * 5, -10, 10))

    def _gold_price_trend(self, stock_data: Dict) -> float:
        """黄金价格趋势"""
        trend = stock_data.get('gold_price_trend', 0)
        # 黄金上涨通常表示避险情绪
        return float(np.clip(-trend * 5, -10, 10))

    def _vix_level(self, stock_data: Dict) -> float:
        """VIX 恐慌指数"""
        vix = stock_data.get('vix', 20)
        # VIX < 15: 低恐慌 → 高分
        # VIX > 30: 高恐慌 → 低分
        return float(np.clip(10 - (vix - 15) / 3, 0, 10))

    def _sp500_trend(self, stock_data: Dict) -> float:
        """标普 500 趋势"""
        trend = stock_data.get('sp500_trend', 0)
        return float(np.clip(trend * 5, -10, 10))

    def _nasdaq_trend(self, stock_data: Dict) -> float:
        """纳斯达克趋势"""
        trend = stock_data.get('nasdaq_trend', 0)
        return float(np.clip(trend * 5, -10, 10))

    def _northbound_flow(self, stock_data: Dict) -> float:
        """北向资金净买入"""
        flow = stock_data.get('northbound_net_flow', 0)
        # 净流入为正 → 高分
        return float(np.clip(flow * 2, 0, 10))

    def _northbound_trend(self, stock_data: Dict) -> float:
        """北向资金趋势 (5 日移动平均)"""
        trend = stock_data.get('northbound_trend', 0)
        return float(np.clip(trend * 2, -10, 10))

    def _sector_momentum(self, stock_data: Dict) -> float:
        """行业动量"""
        momentum = stock_data.get('sector_momentum', 0)
        return float(np.clip(momentum * 5, -10, 10))

    def _sector_rotation_signal(self, stock_data: Dict) -> float:
        """行业轮动信号"""
        signal = stock_data.get('sector_rotation_signal', 0)
        return float(np.clip(signal * 5, -10, 10))

    def _macro_liquidity(self, stock_data: Dict) -> float:
        """宏观流动性"""
        liquidity = stock_data.get('macro_liquidity', 0)
        return float(np.clip(liquidity * 5, -10, 10))

    def _m2_growth(self, stock_data: Dict) -> float:
        """M2 同比增速"""
        growth = stock_data.get('m2_growth', 10)
        # M2 增速越高，流动性越充裕
        return float(np.clip(growth, 0, 10))

    def _market_sentiment(self, stock_data: Dict) -> float:
        """市场情绪"""
        sentiment = stock_data.get('market_sentiment', 0)
        return float(np.clip(sentiment, 0, 10))

    def _put_call_ratio(self, stock_data: Dict) -> float:
        """Put/Call 比率"""
        ratio = stock_data.get('put_call_ratio', 1.0)
        # 比率低表示看涨情绪高
        return float(np.clip((2 - ratio) * 5, 0, 10))


# 全局实例
cross_market_factors = CrossMarketFactors()


def get_cross_market_factors(stock_data: Dict,
                              klines: Optional[List[Dict]] = None) -> Dict[str, float]:
    """获取跨市场因子 (便捷函数)"""
    return cross_market_factors.calculate_all(stock_data, klines)
