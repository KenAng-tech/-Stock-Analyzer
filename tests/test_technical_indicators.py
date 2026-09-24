#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
技术指标模块单元测试

P3 测试 (2026-07-01):
  验证 RSI/MACD/KDJ/布林带 等指标计算的准确性。
"""

import sys
import os
import unittest
import numpy as np

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.technical_indicators import (
    calculate_rsi, rsi_signal,
    calculate_macd,
    calculate_kdj,
    calculate_bollinger,
    calculate_atr,
    comprehensive_technical_analysis,
    sma, ema,
)


class TestSMaEMA(unittest.TestCase):
    """测试 SMA 和 EMA 计算"""

    def test_sma_simple(self):
        """测试简单移动平均线"""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = sma(values, 3)
        # 最后 3 个值的 SMA: (1+2+3)/3=2, (2+3+4)/3=3, (3+4+5)/3=4
        self.assertEqual(result[4], 4.0)
        self.assertEqual(result[3], 3.0)
        self.assertEqual(result[2], 2.0)  # SMA(3) 在索引 2 处 = (1+2+3)/3 = 2

    def test_ema_simple(self):
        """测试指数移动平均线"""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = ema(values, 3)
        self.assertFalse(np.isnan(result[2]))  # 第 3 个值有 SMA 初始化


class TestRSI(unittest.TestCase):
    """测试 RSI 计算"""

    def test_rsi_rising_market(self):
        """上升趋势中 RSI 应较高"""
        closes = np.array([100 + i * 0.5 for i in range(30)])  # 稳定上涨
        rsi = calculate_rsi(closes, 14)
        self.assertGreater(rsi, 60)  # 上升趋势 RSI > 60

    def test_rsi_falling_market(self):
        """下降趋势中 RSI 应较低"""
        closes = np.array([100 - i * 0.5 for i in range(30)])  # 稳定下跌
        rsi = calculate_rsi(closes, 14)
        self.assertLess(rsi, 40)  # 下降趋势 RSI < 40

    def test_rsi_oscillating(self):
        """震荡市场中 RSI 应接近 50"""
        np.random.seed(42)
        closes = np.array([100 + sum(np.random.randn(3) * 0.5) for _ in range(30)])
        rsi = calculate_rsi(closes, 14)
        self.assertGreater(rsi, 30)
        self.assertLess(rsi, 70)

    def test_rsi_insufficient_data(self):
        """数据不足时返回 50"""
        closes = np.array([100.0, 101.0])
        rsi = calculate_rsi(closes, 14)
        self.assertEqual(rsi, 50.0)

    def test_rsi_signal(self):
        """测试 RSI 信号生成"""
        self.assertEqual(rsi_signal(15), '严重超卖')
        self.assertEqual(rsi_signal(25), '超卖区')
        self.assertEqual(rsi_signal(50), '中性')
        self.assertEqual(rsi_signal(75), '超买区')
        self.assertEqual(rsi_signal(85), '严重超买')


class TestMACD(unittest.TestCase):
    """测试 MACD 计算"""

    def test_macd_trend(self):
        """趋势数据中 MACD 应有明确方向"""
        closes = np.array([100 + i * 0.3 for i in range(50)])
        macd = calculate_macd(closes)
        self.assertIn(macd['signal_type'], ['金叉', '死叉', '延续多', '延续空', '数据不足'])
        self.assertIn(macd['divergence'], ['底背离', '顶背离', '无'])

    def test_macd_insufficient_data(self):
        """数据不足时返回默认值"""
        closes = np.array([100.0, 101.0])
        macd = calculate_macd(closes)
        self.assertEqual(macd['signal_type'], '数据不足')


class TestKDJ(unittest.TestCase):
    """测试 KDJ 计算"""

    def test_kdj_basic(self):
        """基本 KDJ 计算"""
        highs = np.array([10.5, 11.0, 10.8, 11.2, 11.5, 11.3, 11.8, 12.0, 11.9, 12.2] * 3)
        lows = np.array([10.0, 10.5, 10.3, 10.7, 11.0, 10.8, 11.3, 11.5, 11.4, 11.7] * 3)
        closes = np.array([10.3, 10.8, 10.5, 11.0, 11.3, 11.1, 11.6, 11.8, 11.7, 12.0] * 3)
        kdj = calculate_kdj(highs, lows, closes)
        self.assertIn('k', kdj)
        self.assertIn('d', kdj)
        self.assertIn('j', kdj)
        self.assertIn('signal', kdj)


class TestBollinger(unittest.TestCase):
    """测试布林带计算"""

    def test_bollinger_basic(self):
        """基本布林带计算"""
        np.random.seed(42)
        closes = np.array([100 + sum(np.random.randn(3) * 0.3) for _ in range(30)])
        boll = calculate_bollinger(closes, 20, 2.0)
        self.assertGreater(boll['upper'], boll['middle'])
        self.assertLess(boll['lower'], boll['middle'])
        self.assertIn(boll['position'], ['接近上轨', '偏上', '中轨附近', '偏下', '接近下轨', '数据不足'])


class TestATR(unittest.TestCase):
    """测试 ATR 计算"""

    def test_atr_basic(self):
        """基本 ATR 计算"""
        highs = np.array([10.5, 11.0, 10.8, 11.2, 11.5] * 5)
        lows = np.array([10.0, 10.5, 10.3, 10.7, 11.0] * 5)
        closes = np.array([10.3, 10.8, 10.5, 11.0, 11.3] * 5)
        atr = calculate_atr(highs, lows, closes, 14)
        self.assertGreaterEqual(atr, 0)


class TestComprehensiveTechnicalAnalysis(unittest.TestCase):
    """测试综合技术分析"""

    def test_comprehensive_basic(self):
        """基本综合技术分析"""
        np.random.seed(42)
        klines = []
        price = 100.0
        for i in range(30):
            price *= (1 + np.random.randn() * 0.02)
            klines.append({
                'date': f'2026-01-{i+1:02d}',
                'open': price * (1 + np.random.randn() * 0.01),
                'high': price * (1 + abs(np.random.randn()) * 0.02),
                'low': price * (1 - abs(np.random.randn()) * 0.02),
                'close': price,
                'volume': int(np.random.randint(100000, 1000000)),
            })

        result = comprehensive_technical_analysis(klines)
        self.assertNotIn('error', result)
        self.assertIn('rsi', result)
        self.assertIn('macd', result)
        self.assertIn('kdj', result)
        self.assertIn('bollinger', result)
        self.assertIn('summary', result)

    def test_comprehensive_insufficient_data(self):
        """数据不足时返回错误"""
        result = comprehensive_technical_analysis([])
        self.assertIn('error', result)

        result = comprehensive_technical_analysis([{'close': 100.0}] * 10)
        self.assertIn('error', result)


if __name__ == '__main__':
    unittest.main()
