#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
SOTA 模块单元测试: 跨市场分析

P3 SOTA 测试 (2026-07-01) — 2026-09-20 清尸:
  moirai_predictor (被 PatchTST 池替换) 与 drl_agent (07 实验链下线) 均已删除,
  原文件 import 死模块 → 整文件 collection error (跨市场分析活链测试被陪葬)。
  死链测试块随模块一并摘除; 跨市场契约锁保留在真身 modules/cross_market。

运行: python -m unittest tests.test_sota_modules
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.cross_market import CrossMarketAnalyzer, MarketData


class TestCrossMarketAnalyzer(unittest.TestCase):
    """测试跨市场分析器"""

    def setUp(self):
        self.analyzer = CrossMarketAnalyzer()

        # 注册测试数据
        self.analyzer.register_market_data(MarketData(
            market='A', symbol='sh601318', name='中国平安',
            price=45.0, prev_close=44.5, open=44.8, high=45.5, low=44.3,
            volume=1000000, currency='CNY', exchange_rate=1.0,
            market_cap=920000000000, pe_ratio=8.5, sector='金融', date='2026-07-01',
        ))
        self.analyzer.register_market_data(MarketData(
            market='HK', symbol='01318', name='中国平安(港股)',
            price=38.0, prev_close=37.5, open=37.8, high=38.5, low=37.6,
            volume=500000, currency='HKD', exchange_rate=0.92,
            market_cap=780000000000, pe_ratio=7.8, sector='金融', date='2026-07-01',
        ))

    def test_register_market_data(self):
        """测试注册市场数据"""
        self.assertEqual(len(self.analyzer._market_data), 2)
        self.assertEqual(len(self.analyzer._historical_prices), 2)

    def test_ah_premium(self):
        """测试 AH 溢价计算"""
        result = self.analyzer.calculate_ah_premium('sh601318', '01318')
        self.assertIn('a_price', result)
        self.assertIn('h_price_cny', result)
        self.assertIn('premium_pct', result)
        self.assertIn('signal', result)
        self.assertEqual(result['a_price'], 45.0)
        # H股 CNY = 38 * 0.92 = 34.96
        self.assertAlmostEqual(result['h_price_cny'], 34.96, places=1)

    def test_market_sentiment(self):
        """测试市场情绪分析"""
        result = self.analyzer.analyze_market_sentiment('A')
        self.assertIn('sentiment', result)
        self.assertIn('score', result)
        self.assertIn('breadth', result)
        self.assertIn('volume_trend', result)
        self.assertIn('vix_level', result)
        self.assertGreater(result['n_stocks'], 0)

    def test_cross_correlations(self):
        """测试跨市场相关性计算"""
        # 添加更多数据点以计算相关性
        for i in range(30):
            for key in list(self.analyzer._market_data.keys()):
                data = self.analyzer._market_data[key]
                new_price = data.price * (1 + np.random.randn() * 0.01)
                self.analyzer.register_market_data(MarketData(
                    market=data.market, symbol=data.symbol, name=data.name,
                    price=new_price, prev_close=data.price, open=data.price,
                    high=new_price * 1.01, low=new_price * 0.99,
                    volume=data.volume, currency=data.currency,
                    exchange_rate=data.exchange_rate,
                    market_cap=data.market_cap, pe_ratio=data.pe_ratio,
                    sector=data.sector, date='2026-07-01',
                ))

        correlations = self.analyzer.calculate_cross_correlations()
        self.assertIn('A_sh601318↔HK_01318', correlations)
        self.assertGreaterEqual(correlations['A_sh601318↔HK_01318'], -1.0)
        self.assertLessEqual(correlations['A_sh601318↔HK_01318'], 1.0)

    def test_unified_prediction(self):
        """测试跨市场融合预测"""
        a_signals = [{'direction': 'buy', 'confidence': 0.7}]
        hk_signals = [{'direction': 'buy', 'confidence': 0.6}]
        us_signals = [{'direction': 'hold', 'confidence': 0.5}]

        result = self.analyzer.unified_prediction(a_signals, hk_signals, us_signals)
        self.assertIn('direction', result)
        self.assertIn('confidence', result)
        self.assertIn('fused_score', result)
        self.assertIn('market_breakdown', result)
        self.assertIn('n_signals', result)
        self.assertEqual(result['n_signals'], 3)

    def test_unified_prediction_empty(self):
        """测试空信号融合"""
        result = self.analyzer.unified_prediction([], [], [])
        self.assertEqual(result['direction'], 'neutral')
        self.assertEqual(result['confidence'], 0.0)
        self.assertEqual(result['n_signals'], 0)

    def test_get_summary(self):
        """测试摘要"""
        summary = self.analyzer.get_summary()
        self.assertEqual(summary['total_stocks'], 2)
        self.assertIn('A', summary['markets'])
        self.assertIn('HK', summary['markets'])
        self.assertGreaterEqual(summary['n_historical_days'], 1)


if __name__ == '__main__':
    unittest.main()
