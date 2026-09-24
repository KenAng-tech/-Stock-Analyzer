#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
跨市场集成模块

P3 SOTA 集成 (2026-07-01):
  整合 A 股 (沪深)、港股 (HK)、美股 (US) 三大市场的数据和分析。

功能:
  1. 跨市场因子计算 (AH 溢价、港股通、美股映射)
  2. 跨市场套利信号 (A+H 价差、ETF 溢价)
  3. 全球市场联动分析 (纳斯达克→A 股、汇率影响)
  4. 统一预测接口 (跨市场融合预测)

市场映射:
  - A 股: sh688xxx (科创板), sh601xxx (上证), sz000xxx (深证), sz300xxx (创业板)
  - 港股: hk00700 (腾讯), hk09988 (阿里)
  - 美股: aapl (苹果), nvda (英伟达), msft (微软)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from modules.logger import logger


@dataclass
class MarketData:
    """统一市场数据"""
    market: str            # 'A', 'HK', 'US'
    symbol: str            # 股票代码
    name: str              # 股票名称
    price: float           # 当前价格
    prev_close: float      # 前一日收盘价
    open: float            # 开盘价
    high: float            # 最高价
    low: float             # 最低价
    volume: float          # 成交量
    currency: str          # 货币 (CNY/HKD/USD)
    exchange_rate: float   # 汇率 (相对于 CNY)
    market_cap: float      # 市值
    pe_ratio: float        # 市盈率
    sector: str            # 行业
    date: str              # 日期


@dataclass
class CrossMarketSignal:
    """跨市场信号"""
    signal_type: str       # 'arbitrage', 'correlation', 'spillover', 'convergence', 'momentum', 'rotation', 'hedge'
    source_market: str
    target_market: str
    source_symbol: str
    target_symbol: str
    direction: str         # 'buy'/'sell'/'hold'
    confidence: float
    expected_return: float
    reason: str
    timestamp: str = ''


@dataclass
class CrossMarketMomentum:
    """跨市场动量信号"""
    signal_type: str = 'momentum'
    market: str = ''
    symbol: str = ''
    direction: str = 'buy'
    strength: float = 0.0
    momentum_score: float = 0.0
    reason: str = ''
    timestamp: str = ''


@dataclass
class SectorRotation:
    """板块轮动信号"""
    sector: str = ''
    source_market: str = 'A'
    target_market: str = 'HK'
    direction: str = 'rotate'
    confidence: float = 0.5
    reason: str = ''
    timestamp: str = ''


@dataclass
class HedgingSignal:
    """对冲信号"""
    signal_type: str = 'hedge'
    long_symbol: str = ''
    short_symbol: str = ''
    long_market: str = 'A'
    short_market: str = 'HK'
    hedge_ratio: float = 1.0
    confidence: float = 0.5
    reason: str = ''
    timestamp: str = ''


class CrossMarketAnalyzer:
    """跨市场综合分析器

    分析维度:
    1. AH 溢价: A 股 vs 港股同公司的价差
    2. 港股通: 南向资金流向对港股的影响
    3. 美股映射: 中概股与 A 股关联
    4. 全球联动: 纳斯达克/标普对 A 股开盘影响
    5. 汇率传导: USD/CNY 对出口/进口企业影响
    """

    # AH 溢价指数基准 (100 = 无溢价)
    AH_PREMIUM_BENCHMARK = 100.0

    # 行业映射表 (A 股 ↔ 美股)
    INDUSTRY_MAP = {
        '半导体': ['sh688981', 'nvda', 'tsmc'],       # 半导体
        '互联网': ['sz300750', 'googl', 'meta'],       # 互联网
        '新能源': ['sz300750', 'tsla', 'enph'],        # 新能源
        '医药': ['sh600196', 'jnj', 'pfe'],            # 医药
        '消费': ['sz000858', 'ko', 'pep'],             # 消费
        '金融': ['sh601318', 'jpm', 'gs'],             # 金融
    }

    def __init__(self):
        self._market_data: Dict[str, MarketData] = {}
        self._historical_prices: Dict[str, List[float]] = {}
        self._cross_correlations: Dict[str, Dict[str, float]] = {}
        logger.info("[CrossMarket] 跨市场分析器初始化")

    def register_market_data(self, data: MarketData):
        """注册市场数据"""
        key = f"{data.market}_{data.symbol}"
        self._market_data[key] = data

        if key not in self._historical_prices:
            self._historical_prices[key] = []
        self._historical_prices[key].append(data.price)

        # 保持最近 252 个交易日
        if len(self._historical_prices[key]) > 252:
            self._historical_prices[key] = self._historical_prices[key][-252:]

    def calculate_cross_correlations(self) -> Dict[str, Dict[str, float]]:
        """计算跨市场收益率相关性"""
        correlations = {}
        keys = list(self._historical_prices.keys())

        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                prices_i = np.array(self._historical_prices[keys[i]])
                prices_j = np.array(self._historical_prices[keys[j]])

                # 取共同长度
                min_len = min(len(prices_i), len(prices_j))
                if min_len < 20:
                    continue

                returns_i = np.diff(prices_i[-min_len:]) / prices_i[-min_len:-1]
                returns_j = np.diff(prices_j[-min_len:]) / prices_j[-min_len:-1]

                # 皮尔逊相关系数
                if np.std(returns_i) > 0 and np.std(returns_j) > 0:
                    corr = np.corrcoef(returns_i, returns_j)[0, 1]
                    correlations[f"{keys[i]}↔{keys[j]}"] = float(np.clip(corr, -1, 1))

        self._cross_correlations = correlations
        logger.info(f"[CrossMarket] 计算了 {len(correlations)} 对跨市场相关性")
        return correlations

    def calculate_ah_premium(self, a_stock: str, h_stock: str) -> Dict:
        """
        计算 AH 溢价

        Args:
            a_stock: A 股代码 (如 sh601318)
            h_stock: 港股代码 (如 hk01318)

        Returns:
            {
                'a_price': float,
                'h_price_cny': float,
                'premium_pct': float,  # 正数表示 A 股更贵
                'signal': str,         # 'overvalued_A' / 'fair' / 'overvalued_H'
            }
        """
        a_key = f"A_{a_stock}"
        h_key = f"HK_{h_stock}"

        a_data = self._market_data.get(a_key)
        h_data = self._market_data.get(h_key)

        if not a_data or not h_data:
            return {
                'a_price': 0, 'h_price_cny': 0,
                'premium_pct': 0, 'signal': 'no_data',
            }

        # 港股价格转换为 CNY
        h_price_cny = h_data.price * h_data.exchange_rate
        premium = (a_data.price - h_price_cny) / h_price_cny * 100

        if premium > 15:
            signal = 'A股高估'
        elif premium < -15:
            signal = 'H股高估'
        else:
            signal = '溢价合理'

        return {
            'a_price': round(a_data.price, 2),
            'h_price': round(h_data.price, 2),
            'h_price_cny': round(h_price_cny, 2),
            'exchange_rate': h_data.exchange_rate,
            'premium_pct': round(premium, 2),
            'signal': signal,
        }

    def detect_arbitrage_opportunities(self) -> List[CrossMarketSignal]:
        """检测跨市场套利机会"""
        signals = []
        now = datetime.now().isoformat()

        # 1. AH 溢价套利
        for sector, symbols in self.INDUSTRY_MAP.items():
            a_symbols = [s for s in symbols if s.startswith(('sh', 'sz'))]
            h_symbols = [s for s in symbols if s.startswith('hk')]

            for a_sym in a_symbols:
                for h_sym in h_symbols:
                    premium_info = self.calculate_ah_premium(a_sym, h_sym)
                    if premium_info['premium_pct'] > 20:
                        signals.append(CrossMarketSignal(
                            signal_type='arbitrage',
                            source_market='A', target_market='HK',
                            source_symbol=a_sym, target_symbol=h_sym,
                            direction='sell_A_buy_H',
                            confidence=min(abs(premium_info['premium_pct']) / 100, 1.0),
                            expected_return=premium_info['premium_pct'] / 100,
                            reason=f"AH 溢价 {premium_info['premium_pct']:.1f}%, "
                                   f"{sector}板块",
                            timestamp=now,
                        ))

        # 2. 全球联动信号
        us_correlations = {
            k: v for k, v in self._cross_correlations.items()
            if 'US_' in k
        }
        for pair, corr in us_correlations.items():
            if corr > 0.7:  # 高相关
                signals.append(CrossMarketSignal(
                    signal_type='spillover',
                    source_market='US', target_market='A',
                    source_symbol=pair.split('↔')[0].replace('US_', ''),
                    target_symbol=pair.split('↔')[1].replace('A_', ''),
                    direction='follow_us' if corr > 0 else 'reverse_us',
                    confidence=abs(corr),
                    expected_return=corr * 0.02,
                    reason=f"美股联动 (r={corr:.2f})",
                    timestamp=now,
                ))

        return signals

    def analyze_market_sentiment(self, market: str = 'A') -> Dict:
        """
        分析市场情绪

        Args:
            market: 'A'/'HK'/'US'

        Returns:
            {
                'sentiment': 'bullish'/'bearish'/'neutral',
                'score': float,  # -1 ~ 1
                'breadth': float,  # 上涨家数/总家数
                'volume_trend': 'increasing'/'decreasing'/'stable',
                'vix_level': float,  # 恐慌指数代理
            }
        """
        market_keys = [k for k in self._market_data if k.startswith(f"{market}_")]

        if not market_keys:
            return {
                'sentiment': 'neutral', 'score': 0.0,
                'breadth': 0.5, 'volume_trend': 'stable',
                'vix_level': 20.0,
            }

        # 简化的情绪计算
        prices = []
        volumes = []
        for key in market_keys:
            data = self._market_data[key]
            prices.append(data.price / data.prev_close - 1)
            volumes.append(data.volume)

        # 上涨比例
        breadth = sum(1 for p in prices if p > 0) / max(len(prices), 1)

        # 情绪分数
        avg_return = np.mean(prices)
        score = np.clip(avg_return * 10 + (breadth - 0.5) * 2, -1, 1)

        # 成交量趋势
        if len(volumes) >= 5:
            recent_vol = np.mean(volumes[-3:])
            old_vol = np.mean(volumes[:3])
            volume_trend = 'increasing' if recent_vol > old_vol * 1.1 else \
                          'decreasing' if recent_vol < old_vol * 0.9 else 'stable'
        else:
            volume_trend = 'stable'

        # VIX 代理 (用波动率)
        vix = abs(np.std(prices)) * 100 * 10 if prices else 20

        return {
            'sentiment': 'bullish' if score > 0.2 else 'bearish' if score < -0.2 else 'neutral',
            'score': round(float(score), 3),
            'breadth': round(float(breadth), 3),
            'volume_trend': volume_trend,
            'vix_level': round(float(vix), 1),
            'n_stocks': len(market_keys),
        }

    def unified_prediction(self,
                           a_signals: List[Dict],
                           hk_signals: List[Dict],
                           us_signals: List[Dict],
                           correlations: Optional[Dict] = None) -> Dict:
        """
        跨市场融合预测

        使用相关性加权融合各市场信号:
        - 高相关市场信号互相增强
        - 低相关市场信号提供多样化信息

        Args:
            a_signals: A 股信号列表
            hk_signals: 港股信号列表
            us_signals: 美股信号列表
            correlations: 跨市场相关性矩阵

        Returns:
            融合预测结果
        """
        all_signals = (
            [(s, 'A', 1.0) for s in a_signals] +
            [(s, 'HK', 0.9) for s in hk_signals] +  # 港股通有时差
            [(s, 'US', 0.8) for s in us_signals]     # 美股影响有延迟
        )

        if not all_signals:
            return {
                'direction': 'neutral', 'confidence': 0.0,
                'market_breakdown': {'A': 'neutral', 'HK': 'neutral', 'US': 'neutral'},
                'fused_score': 0.0,
                'n_signals': 0,
            }

        # 计算加权方向分数
        buy_score = 0.0
        sell_score = 0.0
        n_signals = 0

        for signal, market, weight in all_signals:
            direction = signal.get('direction', signal.get('signal', 'hold'))
            confidence = signal.get('confidence', signal.get('weight', 0.5))

            if direction in ('buy', 'BUY', 'buy_A_buy_H', 'follow_us'):
                buy_score += weight * confidence
            elif direction in ('sell', 'SELL', 'sell_A_buy_H', 'reverse_us'):
                sell_score += weight * confidence
            n_signals += 1

        # 相关性加权 (简化)
        if correlations:
            for key, corr in correlations.items():
                if abs(corr) > 0.5:
                    buy_score *= (1 + abs(corr) * 0.1)
                    sell_score *= (1 + abs(corr) * 0.1)

        # 融合方向
        total = buy_score + sell_score + 1e-8
        fused_score = (buy_score - sell_score) / total

        if fused_score > 0.15:
            direction = 'bullish'
        elif fused_score < -0.15:
            direction = 'bearish'
        else:
            direction = 'neutral'

        # 各市场独立信号
        market_breakdown = {}
        for market, signals_list in [('A', a_signals), ('HK', hk_signals), ('US', us_signals)]:
            buys = sum(1 for s in signals_list if s.get('direction', s.get('signal', '').startswith('buy') or s.get('direction', s.get('signal', '')).startswith('follow')))
            total_m = max(len(signals_list), 1)
            if buys / total_m > 0.6:
                market_breakdown[market] = 'bullish'
            elif buys / total_m < 0.4:
                market_breakdown[market] = 'bearish'
            else:
                market_breakdown[market] = 'neutral'

        return {
            'direction': direction,
            'confidence': round(abs(fused_score), 3),
            'fused_score': round(float(fused_score), 4),
            'market_breakdown': market_breakdown,
            'buy_score': round(buy_score, 3),
            'sell_score': round(sell_score, 3),
            'n_signals': n_signals,
            'correlations_used': len(correlations or {}),
        }

    def detect_cross_market_momentum(self) -> List[CrossMarketMomentum]:
        """
        检测跨市场动量信号

        基于各市场近期收益率计算动量:
        - 高动量 → 买入
        - 低动量 → 卖出
        - 跨市场动量差异 → 动量轮动
        """
        signals = []
        now = datetime.now().isoformat()

        for key, prices in self._historical_prices.items():
            if len(prices) < 10:
                continue

            prices_arr = np.array(prices[-20:])
            short_mom = (prices_arr[-1] / prices_arr[-5] - 1) * 100
            long_mom = (prices_arr[-1] / prices_arr[0] - 1) * 100
            avg_mom = (short_mom + long_mom) / 2

            market = key.split('_')[0]
            symbol = key.split('_', 1)[1]

            if avg_mom > 5:
                direction = 'buy'
                strength = min(abs(avg_mom) / 20, 1.0)
            elif avg_mom < -5:
                direction = 'sell'
                strength = min(abs(avg_mom) / 20, 1.0)
            else:
                direction = 'hold'
                strength = abs(avg_mom) / 20

            signals.append(CrossMarketMomentum(
                signal_type='momentum',
                market=market,
                symbol=symbol,
                direction=direction,
                strength=round(strength, 3),
                momentum_score=round(avg_mom, 2),
                reason=f"短期动量 {short_mom:.1f}%, 长期动量 {long_mom:.1f}%",
                timestamp=now,
            ))

        return signals

    def detect_sector_rotation(self) -> List[SectorRotation]:
        """
        检测板块轮动信号

        基于行业映射表和各市场表现:
        - 某行业在 A 股表现好但在港股表现差 → 轮动到港股
        - 美股映射行业联动
        """
        rotations = []
        now = datetime.now().isoformat()

        for sector, symbols in self.INDUSTRY_MAP.items():
            a_symbols = [s for s in symbols if s.startswith(('sh', 'sz'))]
            h_symbols = [s for s in symbols if s.startswith('hk')]
            us_symbols = [s for s in symbols if s.startswith(('nvda', 'tsla', 'googl', 'meta', 'jnj', 'ko', 'jpm'))]

            # 计算各市场该板块的表现
            a_returns = []
            for sym in a_symbols:
                key = f"A_{sym}"
                if key in self._historical_prices:
                    prices = np.array(self._historical_prices[key][-20:])
                    if len(prices) > 5:
                        a_returns.append((prices[-1] / prices[0] - 1) * 100)

            h_returns = []
            for sym in h_symbols:
                key = f"HK_{sym}"
                if key in self._historical_prices:
                    prices = np.array(self._historical_prices[key][-20:])
                    if len(prices) > 5:
                        h_returns.append((prices[-1] / prices[0] - 1) * 100)

            us_returns = []
            for sym in us_symbols:
                key = f"US_{sym}"
                if key in self._historical_prices:
                    prices = np.array(self._historical_prices[key][-20:])
                    if len(prices) > 5:
                        us_returns.append((prices[-1] / prices[0] - 1) * 100)

            avg_a = np.mean(a_returns) if a_returns else 0
            avg_h = np.mean(h_returns) if h_returns else 0
            avg_us = np.mean(us_returns) if us_returns else 0

            # 检测轮动机会
            if abs(avg_a - avg_h) > 10:
                if avg_a > avg_h:
                    direction = 'rotate_A_to_HK'
                    reason = f"{sector}: A股表现优于港股 (A={avg_a:.1f}% vs HK={avg_h:.1f}%)"
                else:
                    direction = 'rotate_HK_to_A'
                    reason = f"{sector}: 港股表现优于A股 (HK={avg_h:.1f}% vs A={avg_a:.1f}%)"
                rotations.append(SectorRotation(
                    sector=sector,
                    source_market='A' if avg_a > avg_h else 'HK',
                    target_market='HK' if avg_a > avg_h else 'A',
                    direction=direction,
                    confidence=round(min(abs(avg_a - avg_h) / 20, 1.0), 3),
                    reason=reason,
                    timestamp=now,
                ))

        return rotations

    def detect_hedging_opportunities(self) -> List[HedgingSignal]:
        """
        检测对冲机会

        基于 AH 溢价和跨市场相关性:
        - 高溢价 + 高相关 → 对冲机会
        - 低溢价 + 低相关 → 套利机会
        """
        signals = []
        now = datetime.now().isoformat()

        for sector, symbols in self.INDUSTRY_MAP.items():
            a_symbols = [s for s in symbols if s.startswith(('sh', 'sz'))]
            h_symbols = [s for s in symbols if s.startswith('hk')]

            for a_sym in a_symbols:
                for h_sym in h_symbols:
                    premium_info = self.calculate_ah_premium(a_sym, h_sym)
                    if premium_info['premium_pct'] == 0:
                        continue

                    # 检查相关性
                    pair_key = f"A_{a_sym}↔HK_{h_sym}"
                    corr = self._cross_correlations.get(pair_key, 0.5)

                    # 高溢价 + 高相关 → 对冲
                    if premium_info['premium_pct'] > 15 and corr > 0.5:
                        signals.append(HedgingSignal(
                            long_symbol=h_sym,
                            short_symbol=a_sym,
                            long_market='HK',
                            short_market='A',
                            hedge_ratio=abs(premium_info['premium_pct']) / 100,
                            confidence=round(min(abs(corr) * abs(premium_info['premium_pct']) / 100, 1.0), 3),
                            reason=f"{sector} AH 溢价 {premium_info['premium_pct']:.1f}%, 相关性 {corr:.2f}",
                            timestamp=now,
                        ))
                    # 低溢价 + 低相关 → 套利
                    elif premium_info['premium_pct'] < -10 and corr < 0.5:
                        signals.append(HedgingSignal(
                            long_symbol=a_sym,
                            short_symbol=h_sym,
                            long_market='A',
                            short_market='HK',
                            hedge_ratio=abs(premium_info['premium_pct']) / 100,
                            confidence=round(min(abs(premium_info['premium_pct']) / 20, 1.0), 3),
                            reason=f"{sector} H 股高估, 低相关套利",
                            timestamp=now,
                        ))

        return signals

    def update_correlations_realtime(self, new_prices: Dict[str, float]):
        """
        实时更新跨市场相关性

        Args:
            new_prices: {key: price}
        """
        for key, price in new_prices.items():
            if key in self._historical_prices:
                self._historical_prices[key].append(price)
                if len(self._historical_prices[key]) > 252:
                    self._historical_prices[key] = self._historical_prices[key][-252:]

        # 重新计算相关性
        self.calculate_cross_correlations()
        logger.info(f"[CrossMarket] 实时更新相关性, 新增 {len(new_prices)} 个价格")

    def get_summary(self) -> Dict:
        """获取跨市场分析摘要"""
        registered = len(self._market_data)
        correlations = len(self._cross_correlations)

        # 各市场统计
        market_counts = {}
        for key in self._market_data:
            m = key.split('_')[0]
            market_counts[m] = market_counts.get(m, 0) + 1

        # 额外信号
        momentum_signals = self.detect_cross_market_momentum()
        rotation_signals = self.detect_sector_rotation()
        hedge_signals = self.detect_hedging_opportunities()

        return {
            'total_stocks': registered,
            'markets': market_counts,
            'correlations_computed': correlations,
            'n_historical_days': max(
                (len(v) for v in self._historical_prices.values()), default=0
            ),
            'arbitrage_opportunities': len(self.detect_arbitrage_opportunities()),
            'momentum_signals': len(momentum_signals),
            'rotation_signals': len(rotation_signals),
            'hedge_signals': len(hedge_signals),
            'top_correlations': dict(sorted(
                self._cross_correlations.items(), key=lambda x: abs(x[1]), reverse=True
            )[:5]),
        }


# 全局单例
_analyzer: Optional[CrossMarketAnalyzer] = None
_populated: bool = False

# 导出
__all__ = [
    'CrossMarketAnalyzer',
    'MarketData',
    'CrossMarketSignal',
    'CrossMarketMomentum',
    'SectorRotation',
    'HedgingSignal',
    'populate_cross_market_analyzer',
    'get_cross_market_analyzer',
    'reset_cross_market_analyzer',
]


def _make_market_data(market: str, symbol: str, name: str, price: float,
                      prev_close: float, open_price: float, high: float, low: float,
                      volume: float, currency: str, exchange_rate: float,
                      market_cap: float, pe_ratio: float, date: str,
                      sector: str = '') -> MarketData:
    """将字典数据转换为 MarketData 对象"""
    return MarketData(
        market=market, symbol=symbol, name=name,
        price=price, prev_close=prev_close, open=open_price,
        high=high, low=low, volume=volume,
        currency=currency, exchange_rate=exchange_rate,
        market_cap=market_cap, pe_ratio=pe_ratio,
        sector=sector, date=date,
    )


def populate_cross_market_analyzer(stock_data: Optional[Dict] = None,
                                   daily_klines: Optional[List] = None) -> None:
    """
    将实际股票数据注册到跨市场分析器。

    当 _analyzer 为空或 _populated 为 False 时自动调用。
    """
    global _analyzer, _populated
    if _analyzer is None:
        _analyzer = CrossMarketAnalyzer()

    if _populated:
        return  # 已填充过则跳过

    analyzer = _analyzer

    # 如果没有传入 stock_data，尝试从 app 中获取
    if stock_data is None:
        try:
            from modules.dependencies import get_data_fetcher
            stock_data = get_data_fetcher().get_stock_info(None)
        except Exception as e:
            logger.warning(f"[CrossMarket] 获取 stock_data 失败: {e}")

    # 1. 注册 A 股数据
    today = datetime.now().strftime('%Y-%m-%d')
    if stock_data:
        code = stock_data.get('code', 'sz300620')
        price = stock_data.get('price', 0)
        prev_close = stock_data.get('prev_close', price)
        open_price = stock_data.get('open', price)
        high = stock_data.get('high', price)
        low = stock_data.get('low', price)
        volume = stock_data.get('volume', 0)
        market_cap = stock_data.get('market_cap', 0)
        pe_ratio = stock_data.get('pe', 0)
        name = stock_data.get('name', '未知')
        if len(stock_data.get('timestamp', '')) >= 10:
            today = stock_data['timestamp'][:10]

        a_data = _make_market_data(
            market='A', symbol=code, name=name,
            price=price, prev_close=prev_close, open_price=open_price,
            high=high, low=low, volume=volume,
            currency='CNY', exchange_rate=1.0,
            market_cap=market_cap, pe_ratio=pe_ratio, date=today,
            sector=stock_data.get('sector', '光电通信'),
        )
        analyzer.register_market_data(a_data)
        logger.info(f"[CrossMarket] 注册 A 股: {name} ({code}) @ {price}")

    # 2. 注册港股数据 (腾讯 hk00700)
    hk_price = 420.0  # 模拟港股价格
    hk_data = _make_market_data(
        market='HK', symbol='hk00700', name='腾讯控股',
        price=hk_price, prev_close=hk_price * 0.98,
        open_price=hk_price * 0.99, high=hk_price * 1.02,
        low=hk_price * 0.97, volume=25000000,
        currency='HKD', exchange_rate=0.92,
        market_cap=4000.0, pe_ratio=28.5, date=today,
        sector='互联网',
    )
    analyzer.register_market_data(hk_data)
    logger.info(f"[CrossMarket] 注册港股: 腾讯控股 (hk00700) @ {hk_price}")

    # 3. 注册美股数据 (纳指相关)
    us_data = _make_market_data(
        market='US', symbol='nvda', name='英伟达',
        price=140.0, prev_close=138.0,
        open_price=139.0, high=142.0, low=138.5,
        volume=45000000,
        currency='USD', exchange_rate=7.25,
        market_cap=3500.0, pe_ratio=55.0, date=today,
        sector='半导体',
    )
    analyzer.register_market_data(us_data)
    logger.info(f"[CrossMarket] 注册美股: 英伟达 (nvda) @ {us_data.price}")

    # 4. 如果有 K 线数据，注册历史价格
    if daily_klines and len(daily_klines) > 0:
        prices = [k.get('close', 0) for k in daily_klines[-30:]]  # 最近 30 天
        key = f"A_{stock_data.get('code', 'sz300620') if stock_data else 'sz300620'}"
        analyzer._historical_prices[key] = prices
        logger.info(f"[CrossMarket] 注册 {len(prices)} 天历史价格")
    else:
        # 无 K 线时也为 A 股生成合成历史价格
        key = f"A_{stock_data.get('code', 'sz300620') if stock_data else 'sz300620'}"
        if len(analyzer._historical_prices.get(key, [])) < 20:
            base_price = stock_data.get('price', 100) if stock_data else 100
            prices = [base_price]
            for _ in range(29):
                ret = 0.0005 + 0.015 * np.random.randn()
                prices.append(prices[-1] / (1 + ret))
            prices.reverse()
            analyzer._historical_prices[key] = prices

    # 4b. 为港股和美股生成合成历史价格（用于相关性/动量计算）
    np.random.seed(42)
    for market_key in ['HK_hk00700', 'US_nvda']:
        if market_key not in analyzer._historical_prices or len(analyzer._historical_prices[market_key]) < 2:
            if market_key == 'HK_hk00700':
                base_price = hk_price
            else:
                base_price = us_data.price
            prices = [base_price]
            for _ in range(29):
                ret = 0.0005 + 0.015 * np.random.randn()
                prices.append(prices[-1] / (1 + ret))
            prices.reverse()
            analyzer._historical_prices[market_key] = prices
            logger.info(f"[CrossMarket] 为 {market_key} 生成 {len(prices)} 天合成历史价格")

    # 5. 计算跨市场相关性
    analyzer.calculate_cross_correlations()

    _populated = True
    logger.info("[CrossMarket] 跨市场分析器已填充数据")


def get_cross_market_analyzer() -> CrossMarketAnalyzer:
    """获取跨市场分析师全局实例"""
    global _analyzer, _populated
    if _analyzer is None:
        _analyzer = CrossMarketAnalyzer()
        _populated = False
        populate_cross_market_analyzer()
    elif not _populated:
        # analyzer 已存在但未填充，直接填充
        populate_cross_market_analyzer()
    return _analyzer


def reset_cross_market_analyzer():
    """重置全局实例"""
    global _analyzer, _populated
    _analyzer = None
    _populated = False
