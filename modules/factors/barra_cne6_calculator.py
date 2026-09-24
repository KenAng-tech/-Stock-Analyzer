#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Barra CNE6 风格因子计算器 — 基于 MSCI Barra CNE5/CNE6 风险模型

Barra 多因子模型是业界标准，CNE6 是中国 A 股版本。

风格因子 (18 个):
    1. 规模因子 (Size, S): 市值对数
    2. 回报因子 (Return, R): 累计超额回报
    3. 流动性因子 (Liquidity, L): 换手率
    4. 账面市值比因子 (Book-to-Price, B): 账面市值比
    5. 杠杆因子 (Leverage, E): 资产负债率
    6. 盈利因子 (Profitability, Q): ROE
    7. 经营盈利因子 (Operating Profitability, G): ROA
    8. 短期反转因子 (Short-term Reversal, IRL): 过去 5 日收益率
    9. 特质波动率因子 (Idiosyncratic Volatility, IIV): 特质波动率
    10. 动量因子 (Momentum, IIM): 过去 12 月收益率 (跳过最近 1 月)
    11. 非正常成交量因子 (Non-normal Volume, INV): 日均成交量
    12. 股利因子 (Dividend Yield, D): 股息率
    13. 长期反转因子 (Long-term Reversal, ILR): 过去 60 日收益率
    14. 收益率贝塔因子 (Beta, BETA): 市场贝塔
    15. 成长因子 (Growth, IGR): 营收增长率
    16. 资本稀缺因子 (Capital Scarcity, ICS): 资本支出/总资产
    17. 流动性使用因子 (Liquidity Usage, ILU): 成交量标准差
    18. 价值因子 (Value, IVA): 多种价值比率的组合

行业因子 (20-30 个):
    基于申万一级行业分类，生成行业哑变量因子

实现:
    - 纯 NumPy/pandas 实现，无外部依赖
    - 与 multi_factor_model_v2.py 接口兼容
    - 支持横截面标准化 (Z-Score)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from modules.logger import logger


# ── Barra CNE6 风格因子计算器 ─────────────────────────────────────

class BarraCNE6Calculator:
    """
    Barra CNE6 风格因子计算器

    计算 18 个风格因子 + 行业因子。

    因子值标准化:
        - 所有因子值经过横截面 Z-Score 标准化
        - 标准化后均值为 0，标准差为 1
        - 极端值截断到 ±5σ

    与 multi_factor_model_v2.py 的集成:
        - 风格因子可直接作为因子输入
        - 行业因子可作为中性化基准
        - 因子权重基于 Barra 风险模型
    """

    # 风格因子分类
    FACTOR_GROUPS = {
        'size': ['size'],
        'value': ['btoe', 'btope', 'bvopen'],
        'growth': ['egr', 'dgr', 'ogr'],
        'profitability': ['roe', 'roa', 'gross_margin'],
        'momentum': ['momentum_12_1', 'short_term_reversal', 'long_term_reversal'],
        'risk': ['beta', 'idvol', 'volatility'],
        'liquidity': ['turnover', 'illiquidity', 'nlv'],
        'leverage': ['de', 'dlr'],
        'dividend': ['dividend_yield'],
        'quality': ['profit_growth', 'asset_turnover', 'working_capital'],
        'technical': ['price_ma_ratio', 'rsi_momentum'],
    }

    # 默认因子权重 (基于 Barra 风险模型)
    DEFAULT_WEIGHTS = {
        'size': 0.08,
        'btoe': 0.06,
        'btope': 0.05,
        'bvopen': 0.05,
        'roe': 0.07,
        'roa': 0.05,
        'gross_margin': 0.04,
        'momentum_12_1': 0.08,
        'short_term_reversal': 0.05,
        'long_term_reversal': 0.04,
        'beta': 0.06,
        'idvol': 0.05,
        'volatility': 0.04,
        'turnover': 0.06,
        'illiquidity': 0.04,
        'nlv': 0.05,
        'de': 0.05,
        'dlr': 0.04,
        'dividend_yield': 0.03,
        'egr': 0.05,
        'dgr': 0.04,
        'ogr': 0.04,
        'profit_growth': 0.05,
        'asset_turnover': 0.04,
        'working_capital': 0.03,
        'price_ma_ratio': 0.05,
        'rsi_momentum': 0.04,
    }

    # 申万一级行业列表 (简化版)
    SW_INDUSTRIES = [
        '农林牧渔', '食品饮料', '医药生物', '电子', '计算机',
        '通信', '传媒', '家用电器', '汽车', '房地产',
        '建筑材料', '建筑装饰', '电力设备', '机械设备', '化工',
        '纺织服饰', '轻工制造', '美容护理', '钢铁', '有色金属',
        '银行', '非银金融', '公用事业', '交通运输', '商贸零售',
        '社会服务', '传媒', '综合',
    ]

    def __init__(self):
        self.factor_names: List[str] = []
        self._build_factor_names()

    def _build_factor_names(self):
        """构建因子名称列表"""
        self.factor_names = []

        # 规模因子
        self.factor_names.append('size')

        # 价值因子
        self.factor_names.extend(['btoe', 'btope', 'bvopen'])

        # 成长因子
        self.factor_names.extend(['egr', 'dgr', 'ogr'])

        # 盈利因子
        self.factor_names.extend(['roe', 'roa', 'gross_margin'])

        # 动量因子
        self.factor_names.extend(['momentum_12_1', 'short_term_reversal', 'long_term_reversal'])

        # 风险因子
        self.factor_names.extend(['beta', 'idvol', 'volatility'])

        # 流动性因子
        self.factor_names.extend(['turnover', 'illiquidity', 'nlv'])

        # 杠杆因子
        self.factor_names.extend(['de', 'dlr'])

        # 股息因子
        self.factor_names.append('dividend_yield')

        # 质量因子
        self.factor_names.extend(['profit_growth', 'asset_turnover', 'working_capital'])

        # 技术因子
        self.factor_names.extend(['price_ma_ratio', 'rsi_momentum'])

        logger.info(f"[BarraCNE6] 构建了 {len(self.factor_names)} 个风格因子")

    def calculate_all(self, stock_data: Dict,
                      klines: Optional[List[Dict]] = None,
                      fundamentals: Optional[Dict] = None) -> Dict[str, float]:
        """
        计算所有 Barra CNE6 风格因子

        Args:
            stock_data: 股票数据字典 (价格、成交量、换手率等)
            klines: K 线数据 (用于动量、波动率等时间序列因子)
            fundamentals: 基本面数据 (ROE、营收增长等)

        Returns:
            {factor_name: z_score} 标准化后的因子值
        """
        raw_factors = {}

        # 获取基本面数据 (优先使用 fundamentals 参数，降级使用 stock_data)
        if fundamentals is None:
            fundamentals = stock_data

        # ═══════════════════════════════════════════════════════════
        # 1. 规模因子 (Size)
        # ═══════════════════════════════════════════════════════════
        raw_factors['size'] = self._calc_size(stock_data)

        # ═══════════════════════════════════════════════════════════
        # 2. 价值因子 (Value)
        # ═══════════════════════════════════════════════════════════
        raw_factors.update(self._calc_value_factors(stock_data))

        # ═══════════════════════════════════════════════════════════
        # 3. 成长因子 (Growth)
        # ═══════════════════════════════════════════════════════════
        raw_factors.update(self._calc_growth_factors(fundamentals))

        # ═══════════════════════════════════════════════════════════
        # 4. 盈利因子 (Profitability)
        # ═══════════════════════════════════════════════════════════
        raw_factors.update(self._calc_profitability_factors(fundamentals, stock_data))

        # ═══════════════════════════════════════════════════════════
        # 5. 动量因子 (Momentum)
        # ═══════════════════════════════════════════════════════════
        raw_factors.update(self._calc_momentum_factors(klines))

        # ═══════════════════════════════════════════════════════════
        # 6. 风险因子 (Risk)
        # ═══════════════════════════════════════════════════════════
        raw_factors.update(self._calc_risk_factors(stock_data, klines))

        # ═══════════════════════════════════════════════════════════
        # 7. 流动性因子 (Liquidity)
        # ═══════════════════════════════════════════════════════════
        raw_factors.update(self._calc_liquidity_factors(stock_data, klines))

        # ═══════════════════════════════════════════════════════════
        # 8. 杠杆因子 (Leverage)
        # ═══════════════════════════════════════════════════════════
        raw_factors.update(self._calc_leverage_factors(fundamentals, stock_data))

        # ═══════════════════════════════════════════════════════════
        # 9. 股息因子 (Dividend)
        # ═══════════════════════════════════════════════════════════
        raw_factors['dividend_yield'] = self._calc_dividend_yield(stock_data, fundamentals)

        # ═══════════════════════════════════════════════════════════
        # 10. 质量因子 (Quality)
        # ═══════════════════════════════════════════════════════════
        raw_factors.update(self._calc_quality_factors(fundamentals, stock_data))

        # ═══════════════════════════════════════════════════════════
        # 11. 技术因子 (Technical)
        # ═══════════════════════════════════════════════════════════
        raw_factors.update(self._calc_technical_factors(stock_data, klines))

        # 标准化 (Z-Score) — 单只股票时使用默认均值/标准差
        normalized = {}
        for name, value in raw_factors.items():
            if name in self._get_default_params():
                mean, std = self._get_default_params()[name]
                z = (value - mean) / (std + 1e-10)
                normalized[name] = float(np.clip(z, -5.0, 5.0))
            else:
                normalized[name] = 0.0

        return normalized

    def _get_default_params(self) -> Dict[str, Tuple[float, float]]:
        """获取各因子的默认均值和标准差 (用于单只股票标准化)"""
        return {
            'size': (8.0, 2.0),           # log(market_cap), 通常 7-10
            'btoe': (0.3, 0.2),           # book/value
            'btope': (0.15, 0.1),         # book/earnings proxy
            'bvopen': (0.5, 0.3),         # book/open
            'roe': (0.12, 0.08),          # ROE
            'roa': (0.05, 0.04),          # ROA
            'gross_margin': (0.35, 0.15), # 毛利率
            'momentum_12_1': (0.1, 0.2),  # 12 个月动量 (跳过 1 月)
            'short_term_reversal': (-0.02, 0.1),  # 5 日反转
            'long_term_reversal': (0.05, 0.25),   # 60 日反转
            'beta': (1.0, 0.3),           # 市场贝塔
            'idvol': (0.03, 0.02),        # 特质波动率
            'volatility': (0.03, 0.02),   # 已实现波动率
            'turnover': (0.03, 0.02),     # 日均换手率
            'illiquidity': (0.01, 0.01),  # Amihud 非流动性
            'nlv': (0.02, 0.015),         # 非正常成交量
            'de': (0.4, 0.2),             # 资产负债率
            'dlr': (0.05, 0.03),          # 债务比率
            'dividend_yield': (0.02, 0.015),  # 股息率
            'egr': (0.15, 0.15),          # 营收增长率
            'dgr': (0.10, 0.10),          # 净利润增长率
            'ogr': (0.12, 0.12),          # 营业利润增长率
            'profit_growth': (0.20, 0.15),  # 利润增长率
            'asset_turnover': (0.6, 0.3), # 资产周转率
            'working_capital': (0.15, 0.1),  # 营运资金/总资产
            'price_ma_ratio': (1.02, 0.05),  # 价格/MA 比率
            'rsi_momentum': (0.0, 1.0),   # RSI 动量
        }

    # ─────────────────────────────────────────────
    # 规模因子
    # ─────────────────────────────────────────────

    def _calc_size(self, stock_data: Dict) -> float:
        """
        规模因子: 市值对数

        Barra 定义: S = log(market_cap)
        规模越大，得分越高
        """
        market_cap = stock_data.get('market_cap', 0)  # 亿元
        if market_cap > 0:
            return float(np.log(market_cap + 1))
        # 降级: 用流通市值
        float_cap = stock_data.get('float_market_cap', 0)
        if float_cap > 0:
            return float(np.log(float_cap + 1))
        return 8.0  # 默认中等市值

    # ─────────────────────────────────────────────
    # 价值因子
    # ─────────────────────────────────────────────

    def _calc_value_factors(self, stock_data: Dict) -> Dict[str, float]:
        """
        价值因子:

        btoe: Book-to-Earnings = 账面价值 / 净利润
        btope: Book-to-Price-Earnings = PE 的倒数 proxy
        bvopen: Book-to-Open = 账面价值 / 开盘价
        """
        factors = {}

        # B/EP (账面市值比 proxy)
        pe = stock_data.get('pe', 0)
        pb = stock_data.get('pb', 1)
        if pe and pe > 0:
            factors['btoe'] = 1.0 / pe  # E/P
        else:
            factors['btoe'] = 0.05  # 默认低价值

        # B/P (账面市值比)
        if pb and pb > 0:
            factors['btope'] = 1.0 / pb
        else:
            factors['btope'] = 0.3

        # B/O (账面价值/开盘价)
        price = stock_data.get('price', stock_data.get('last_close', 0))
        book_value = stock_data.get('total_equity', price * stock_data.get('total_share', 1))
        if price > 0:
            factors['bvopen'] = book_value / (price * stock_data.get('total_share', 1) + 1e-10)
        else:
            factors['bvopen'] = 0.5

        return factors

    # ─────────────────────────────────────────────
    # 成长因子
    # ─────────────────────────────────────────────

    def _calc_growth_factors(self, fundamentals: Dict) -> Dict[str, float]:
        """
        成长因子:

        EGR: Earnings Growth Rate = 净利润同比增长率
        DGR: Debt Growth Rate = 营收同比增长率
        OGR: Operating Profit Growth Rate = 营业利润同比增长率
        """
        factors = {}

        # 净利润增长率
        net_profit_growth = fundamentals.get('net_profit_growth',
                                              fundamentals.get('profit_yoy', 0))
        factors['egr'] = float(np.clip(net_profit_growth, -0.5, 1.0))

        # 营收增长率
        revenue_growth = fundamentals.get('revenue_growth',
                                           fundamentals.get('revenue_yoy', 0))
        factors['dgr'] = float(np.clip(revenue_growth, -0.5, 1.0))

        # 营业利润增长率
        operating_profit_growth = fundamentals.get('operating_profit_growth',
                                                    fundamentals.get('gross_profit_yoy', 0))
        factors['ogr'] = float(np.clip(operating_profit_growth, -0.5, 1.0))

        return factors

    # ─────────────────────────────────────────────
    # 盈利因子
    # ─────────────────────────────────────────────

    def _calc_profitability_factors(self, fundamentals: Dict,
                                     stock_data: Dict) -> Dict[str, float]:
        """
        盈利因子:

        ROE: Return on Equity = 净资产收益率
        ROA: Return on Assets = 总资产收益率
        Gross Margin: 毛利率
        """
        factors = {}

        # ROE
        roe = fundamentals.get('roe', stock_data.get('roe', 0.1))
        factors['roe'] = float(np.clip(roe / 100, 0, 1))  # 转为小数

        # ROA
        roa = fundamentals.get('roa', stock_data.get('roa', 0.05))
        factors['roa'] = float(np.clip(roa / 100, 0, 1))

        # 毛利率
        gross_margin = fundamentals.get('gross_margin',
                                         fundamentals.get('gross_profit_margin', 0.35))
        factors['gross_margin'] = float(np.clip(gross_margin / 100, 0, 1))

        return factors

    # ─────────────────────────────────────────────
    # 动量因子
    # ─────────────────────────────────────────────

    def _calc_momentum_factors(self, klines: Optional[List[Dict]]) -> Dict[str, float]:
        """
        动量因子:

        Momentum (12, 1): 过去 12 个月收益率，跳过最近 1 个月
        Short-term Reversal: 过去 5 日收益率的负值
        Long-term Reversal: 过去 60 日收益率的负值
        """
        factors = {}

        if not klines or len(klines) < 6:
            factors['momentum_12_1'] = 0.0
            factors['short_term_reversal'] = 0.0
            factors['long_term_reversal'] = 0.0
            return factors

        closes = np.array([float(k['close']) for k in klines], dtype=np.float64)
        n = len(closes)

        # Momentum 12-1: 需要 ~252 个交易日数据
        # 降级: 用可用数据
        if n >= 60:
            # 用 60 日收益率 proxy 12-1 动量
            mom = closes[-1] / closes[-61] - 1 if n >= 61 else 0
            factors['momentum_12_1'] = float(np.clip(mom, -1, 1))
        elif n >= 6:
            # 短期 proxy
            mom = closes[-1] / closes[-6] - 1
            factors['momentum_12_1'] = float(np.clip(mom * 12, -1, 1))
        else:
            factors['momentum_12_1'] = 0.0

        # Short-term Reversal: 5 日
        if n >= 6:
            ret_5d = closes[-1] / closes[-6] - 1
            factors['short_term_reversal'] = float(-ret_5d)
        else:
            factors['short_term_reversal'] = 0.0

        # Long-term Reversal: 60 日
        if n >= 61:
            ret_60d = closes[-1] / closes[-61] - 1
            factors['long_term_reversal'] = float(-ret_60d)
        elif n >= 21:
            ret_20d = closes[-1] / closes[-21] - 1
            factors['long_term_reversal'] = float(-ret_20d * 3)  # 年化 proxy
        else:
            factors['long_term_reversal'] = 0.0

        return factors

    # ─────────────────────────────────────────────
    # 风险因子
    # ─────────────────────────────────────────────

    def _calc_risk_factors(self, stock_data: Dict,
                            klines: Optional[List[Dict]]) -> Dict[str, float]:
        """
        风险因子:

        Beta: 市场贝塔
        Idiosyncratic Volatility: 特质波动率
        Volatility: 已实现波动率
        """
        factors = {}

        # Beta
        beta = stock_data.get('beta', 1.0)
        factors['beta'] = float(np.clip(beta, 0.3, 2.5))

        # 已实现波动率
        if klines and len(klines) >= 21:
            closes = np.array([float(k['close']) for k in klines[-21:] if float(k.get('close', 0)) > 0])
            if len(closes) >= 2:
                returns = np.diff(np.log(closes))
                realized_vol = float(np.std(returns) * np.sqrt(252))
                factors['volatility'] = float(np.clip(realized_vol, 0, 1))
            else:
                factors['volatility'] = 0.03
        else:
            # 降级: 用换手率 proxy
            turnover = stock_data.get('turnover', 3)
            factors['volatility'] = float(np.clip(turnover / 100, 0, 1))

        # 特质波动率 (用已实现波动率 - 市场波动率 proxy)
        market_vol = 0.15  # 市场年化波动率 ~15%
        if klines and len(klines) >= 21:
            closes = np.array([float(k['close']) for k in klines[-21:] if float(k.get('close', 0)) > 0])
            if len(closes) >= 2:
                returns = np.diff(np.log(closes))
                realized_vol = np.std(returns) * np.sqrt(252)
                idio_vol = max(0, realized_vol ** 2 - market_vol ** 2) ** 0.5
                factors['idvol'] = float(np.clip(idio_vol, 0, 1))
            else:
                factors['idvol'] = 0.02
        else:
            factors['idvol'] = 0.02

        return factors

    # ─────────────────────────────────────────────
    # 流动性因子
    # ─────────────────────────────────────────────

    def _calc_liquidity_factors(self, stock_data: Dict,
                                 klines: Optional[List[Dict]]) -> Dict[str, float]:
        """
        流动性因子:

        Turnover: 日均换手率
        Illiquidity: Amihud 非流动性指标
        NLV: 非正常成交量 (Non-normal Volume)
        """
        factors = {}

        # 换手率
        turnover = stock_data.get('turnover', 3) / 100  # 转为小数
        factors['turnover'] = float(np.clip(turnover, 0, 1))

        # Amihud 非流动性: |return| / volume
        if klines and len(klines) >= 21:
            closes = np.array([float(k['close']) for k in klines[-21:] if float(k.get('close', 0)) > 0])
            volumes = np.array([float(k.get('volume', 0)) for k in klines[-21:] if float(k.get('volume', 0)) > 0])
            if len(closes) >= 2 and len(volumes) >= 2:
                returns = np.abs(np.diff(np.log(closes)))
                vol = volumes[1:]  # 对齐
                amihud = np.mean(returns / (vol + 1e-10))
                factors['illiquidity'] = float(np.clip(amihud * 1000, 0, 1))
            else:
                factors['illiquidity'] = 0.01
        else:
            factors['illiquidity'] = float(np.clip(turnover * 0.5, 0, 1))

        # 非正常成交量: 当日成交量 / 平均成交量
        if klines and len(klines) >= 11:
            recent_vol = klines[-1].get('volume', 0)
            avg_vol = np.mean([k.get('volume', 0) for k in klines[-11:-1]])
            if avg_vol > 0:
                nlv = recent_vol / avg_vol
                factors['nlv'] = float(np.clip((nlv - 1) / 2, -1, 1))
            else:
                factors['nlv'] = 0.0
        else:
            factors['nlv'] = 0.0

        return factors

    # ─────────────────────────────────────────────
    # 杠杆因子
    # ─────────────────────────────────────────────

    def _calc_leverage_factors(self, fundamentals: Dict,
                                stock_data: Dict) -> Dict[str, float]:
        """
        杠杆因子:

        DE: Debt-to-Equity = 资产负债率
        DLR: Debt-to-Lending Ratio = 短期债务/总债务
        """
        factors = {}

        # 资产负债率
        de = fundamentals.get('de_ratio', stock_data.get('debt_to_asset', 0.4))
        factors['de'] = float(np.clip(de / 100, 0, 1))

        # 短期债务比率
        short_debt = fundamentals.get('short_term_debt', 0)
        total_debt = fundamentals.get('total_debt', 0)
        if total_debt > 0:
            factors['dlr'] = float(np.clip(short_debt / total_debt, 0, 1))
        else:
            factors['dlr'] = 0.05  # 默认低短期债务

        return factors

    # ─────────────────────────────────────────────
    # 股息因子
    # ─────────────────────────────────────────────

    def _calc_dividend_yield(self, stock_data: Dict,
                              fundamentals: Dict) -> float:
        """
        股息率: Dividend Yield = 每股股息 / 股价

        降级: 用净利润率 proxy
        """
        dividend_per_share = fundamentals.get('dividend_per_share', 0)
        price = stock_data.get('price', stock_data.get('last_close', 0))

        if price > 0 and dividend_per_share > 0:
            return float(np.clip(dividend_per_share / price, 0, 0.1))

        # 降级: 用净利润率 proxy
        net_margin = fundamentals.get('net_margin', stock_data.get('net_profit_margin', 0.1))
        # 假设派息率 30%
        return float(np.clip(net_margin * 0.3, 0, 0.08))

    # ─────────────────────────────────────────────
    # 质量因子
    # ─────────────────────────────────────────────

    def _calc_quality_factors(self, fundamentals: Dict,
                               stock_data: Dict) -> Dict[str, float]:
        """
        质量因子:

        Profit Growth: 利润增长率
        Asset Turnover: 资产周转率
        Working Capital: 营运资金/总资产
        """
        factors = {}

        # 利润增长率
        profit_growth = fundamentals.get('net_profit_growth',
                                          fundamentals.get('profit_yoy', 0))
        factors['profit_growth'] = float(np.clip(profit_growth / 100, -0.5, 1.0))

        # 资产周转率
        revenue = fundamentals.get('revenue', 0)
        total_assets = fundamentals.get('total_assets', 1)
        if total_assets > 0:
            factors['asset_turnover'] = float(np.clip(revenue / total_assets, 0, 3))
        else:
            factors['asset_turnover'] = 0.6

        # 营运资金/总资产
        current_assets = fundamentals.get('current_assets', 0)
        current_liabilities = fundamentals.get('current_liabilities', 0)
        if total_assets > 0:
            working_capital = (current_assets - current_liabilities) / total_assets
            factors['working_capital'] = float(np.clip(working_capital, -1, 1))
        else:
            factors['working_capital'] = 0.15

        return factors

    # ─────────────────────────────────────────────
    # 技术因子
    # ─────────────────────────────────────────────

    def _calc_technical_factors(self, stock_data: Dict,
                                 klines: Optional[List[Dict]]) -> Dict[str, float]:
        """
        技术因子:

        Price/MA Ratio: 价格/均线比率
        RSI Momentum: RSI 变化率
        """
        factors = {}

        # 价格/MA5 比率
        price = stock_data.get('price', stock_data.get('last_close', 0))
        if klines and len(klines) >= 6:
            closes = [float(k['close']) for k in klines[-6:-1]]
            ma5 = np.mean(closes)
            if ma5 > 0:
                factors['price_ma_ratio'] = float(np.clip(price / ma5 - 0.9, -0.5, 0.5))
            else:
                factors['price_ma_ratio'] = 0.02
        else:
            factors['price_ma_ratio'] = 0.02

        # RSI 动量 (RSI 变化)
        rsi = stock_data.get('rsi_14', 50)
        prev_rsi = stock_data.get('prev_rsi_14', rsi)
        factors['rsi_momentum'] = float(np.clip((rsi - prev_rsi) / 20, -1, 1))

        return factors

    # ─────────────────────────────────────────────
    # 横截面标准化
    # ─────────────────────────────────────────────

    def cross_sectional_normalize(self, factor_dict: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """
        横截面标准化 — 对多只股票的因子值进行 Z-Score 标准化

        Args:
            factor_dict: {stock_code: {factor_name: raw_value}}

        Returns:
            {stock_code: {factor_name: z_score}}
        """
        if not factor_dict or len(factor_dict) < 2:
            return factor_dict

        # 收集所有因子名
        all_factors = set()
        for stock_factors in factor_dict.values():
            all_factors.update(stock_factors.keys())

        # 对每个因子做横截面标准化
        normalized = {}
        for factor_name in all_factors:
            values = []
            codes = []
            for code, factors in factor_dict.items():
                if factor_name in factors:
                    values.append(factors[factor_name])
                    codes.append(code)

            if len(values) < 2:
                continue

            values = np.array(values, dtype=np.float64)
            mean = np.mean(values)
            std = np.std(values)

            if std < 1e-10:
                continue

            z_scores = (values - mean) / std
            z_scores = np.clip(z_scores, -5.0, 5.0)

            for i, code in enumerate(codes):
                if code not in normalized:
                    normalized[code] = {}
                normalized[code][factor_name] = float(z_scores[i])

        # 合并未标准化的因子
        for code, factors in factor_dict.items():
            if code not in normalized:
                normalized[code] = {}
            for fname, fval in factors.items():
                if fname not in normalized[code]:
                    normalized[code][fname] = fval

        return normalized

    # ─────────────────────────────────────────────
    # 行业因子
    # ─────────────────────────────────────────────

    def get_industry_dummies(self, industry: str) -> Dict[str, float]:
        """
        生成行业哑变量

        Args:
            industry: 申万一级行业名称

        Returns:
            {industry_name: 0 or 1}
        """
        dummies = {}
        for ind in self.SW_INDUSTRIES:
            dummies[f'industry_{ind}'] = 1.0 if ind == industry else 0.0
        return dummies

    def get_industry_factor_names(self) -> List[str]:
        """获取行业因子名称列表"""
        return [f'industry_{ind}' for ind in self.SW_INDUSTRIES]

    # ─────────────────────────────────────────────
    # 工具方法
    # ─────────────────────────────────────────────

    def get_factor_count(self) -> int:
        """获取因子数量"""
        return len(self.factor_names)

    def get_factor_list(self) -> List[str]:
        """获取因子名称列表"""
        return self.factor_names

    def get_factor_categories(self) -> Dict[str, List[str]]:
        """获取因子分类"""
        return dict(self.FACTOR_GROUPS)

    def get_weighted_score(self, factors: Dict[str, float]) -> float:
        """计算加权得分"""
        weights = self.DEFAULT_WEIGHTS
        return sum(weights.get(f, 0.05) * v for f, v in factors.items())

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


# 全局实例
barra_cne6_calculator = BarraCNE6Calculator()


def get_barra_cne6_calculator() -> BarraCNE6Calculator:
    """获取全局 BarraCNE6Calculator 实例 (线程安全)"""
    return barra_cne6_calculator
