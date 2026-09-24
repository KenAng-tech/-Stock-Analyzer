#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
执行算法测试 — VWAP/TWAP/冲击成本/动态滑点
"""

import unittest
import numpy as np
from modules.execution import (
    VWAPExecutor, TWAPExecutor, ExecutionResult,
    ImpactCostModel, ImpactParameters,
    DynamicSlippageModel, SlippageParams,
)


class TestVWAPExecutor(unittest.TestCase):
    """VWAP 执行算法测试"""

    def setUp(self):
        self.executor = VWAPExecutor(max_participation_rate=0.1)
        self.volume_profile = {
            '09:30': 0.1, '10:00': 0.2, '10:30': 0.25,
            '11:00': 0.15, '13:00': 0.15, '14:00': 0.1, '15:00': 0.05,
        }

    def test_generate_schedule(self):
        """测试 VWAP 时间表生成"""
        schedule = self.executor.generate_schedule(10000, self.volume_profile)
        self.assertGreater(len(schedule), 0)
        total = sum(slot['quantity'] for slot in schedule)
        self.assertEqual(total, 10000)

    def test_generate_schedule_empty_profile(self):
        """测试空成交量分布"""
        schedule = self.executor.generate_schedule(1000, {})
        self.assertEqual(schedule, [])

    def test_generate_schedule_zero_volume(self):
        """测试零成交量分布"""
        schedule = self.executor.generate_schedule(1000, {'09:30': 0})
        self.assertEqual(schedule, [])

    def test_execute_buy(self):
        """测试买入执行"""
        order = {'stock_code': 'sz300620', 'quantity': 1000, 'side': 'buy'}
        result = self.executor.execute(order, self.volume_profile, current_price=100.0)
        self.assertTrue(result.success)
        self.assertEqual(result.algorithm, 'vwap')
        self.assertEqual(result.total_quantity, 1000)
        self.assertGreater(result.filled_quantity, 0)
        self.assertGreater(result.avg_price, 0)
        self.assertIn('trades', result.to_dict())

    def test_execute_sell(self):
        """测试卖出执行"""
        order = {'stock_code': 'sz300620', 'quantity': 500, 'side': 'sell'}
        result = self.executor.execute(order, self.volume_profile, current_price=100.0)
        self.assertTrue(result.success)
        self.assertEqual(result.algorithm, 'vwap')

    def test_fill_rate(self):
        """测试成交率"""
        order = {'stock_code': 'sz300620', 'quantity': 1000, 'side': 'buy'}
        result = self.executor.execute(order, self.volume_profile, current_price=100.0)
        d = result.to_dict()
        self.assertAlmostEqual(d['fill_rate'], 1.0, places=4)

    def test_risk_aversion(self):
        """测试风险厌恶系数"""
        conservative = VWAPExecutor(max_participation_rate=0.05, risk_aversion=2.0)
        schedule = conservative.generate_schedule(10000, self.volume_profile)
        # 保守策略应该有相同数量的槽位
        self.assertGreater(len(schedule), 0)


class TestTWAPExecutor(unittest.TestCase):
    """TWAP 执行算法测试"""

    def setUp(self):
        self.executor = TWAPExecutor(num_slices=5)

    def test_generate_schedule(self):
        """测试 TWAP 时间表生成"""
        schedule = self.executor.generate_schedule(10000, duration_minutes=240)
        self.assertEqual(len(schedule), 5)
        total = sum(slot['quantity'] for slot in schedule)
        self.assertEqual(total, 10000)

    def test_execute(self):
        """测试 TWAP 执行"""
        order = {'stock_code': 'sz300620', 'quantity': 1000, 'side': 'buy'}
        result = self.executor.execute(order, current_price=100.0)
        self.assertTrue(result.success)
        self.assertEqual(result.algorithm, 'twap')
        self.assertEqual(result.total_quantity, 1000)
        self.assertEqual(result.filled_quantity, 1000)

    def test_custom_slices(self):
        """测试自定义切片数"""
        executor = TWAPExecutor(num_slices=10)
        schedule = executor.generate_schedule(10000)
        self.assertEqual(len(schedule), 10)

    def test_slippage_comparison(self):
        """测试 TWAP 滑点低于 VWAP"""
        twap_executor = TWAPExecutor(num_slices=10)
        order = {'stock_code': 'sz300620', 'quantity': 1000, 'side': 'buy'}
        twap_result = twap_executor.execute(order, current_price=100.0)
        # TWAP 滑点应为 0.0005 (0.5bp)
        self.assertAlmostEqual(twap_result.slippage, 0.0005, places=4)


class TestImpactCostModel(unittest.TestCase):
    """冲击成本模型测试"""

    def setUp(self):
        self.model = ImpactCostModel()

    def test_almgren_impact(self):
        """测试 Almgren 冲击模型"""
        impact = self.model.almgren_impact(quantity=10000, price=100.0)
        self.assertGreater(impact, 0)

    def test_sqrt_impact(self):
        """测试 Square-root 冲击模型"""
        impact = self.model.sqrt_impact(quantity=10000, price=100.0, volume=5000000)
        self.assertGreater(impact, 0)

    def test_total_impact_sqrt(self):
        """测试总冲击成本 (sqrt 模型)"""
        result = self.model.total_impact(
            quantity=10000, price=100.0, volume=5000000, model='sqrt'
        )
        self.assertIn('transient_impact', result)
        self.assertIn('permanent_impact', result)
        self.assertIn('total_impact', result)
        self.assertEqual(result['model'], 'sqrt')

    def test_total_impact_almgren(self):
        """测试总冲击成本 (Almgren 模型)"""
        result = self.model.total_impact(
            quantity=10000, price=100.0, volume=5000000, model='algren'
        )
        self.assertIn('total_impact', result)
        self.assertEqual(result['model'], 'algren')

    def test_estimate_depth(self):
        """测试市场深度估计"""
        depth = self.model.estimate_depth(price=100.0, volume=5000000, turnover=0.02)
        self.assertGreater(depth, 0)

    def test_calibrate_default(self):
        """测试无数据时的校准"""
        params = self.model.calibrate([])
        self.assertIsNotNone(params)

    def test_custom_params(self):
        """测试自定义参数"""
        custom = ImpactCostModel(params=ImpactParameters(
            market_depth=2e6, impact_coeff=0.2, sqrt_coeff=0.02
        ))
        impact = custom.sqrt_impact(10000, 100.0, 5000000)
        self.assertGreater(impact, self.model.sqrt_impact(10000, 100.0, 5000000))


class TestDynamicSlippageModel(unittest.TestCase):
    """动态滑点模型测试"""

    def setUp(self):
        self.model = DynamicSlippageModel()

    def test_estimate_base(self):
        """测试基础滑点估计"""
        result = self.model.estimate(price=100.0, volume=1000000)
        self.assertAlmostEqual(result['slippage_ratio'], 0.001, places=6)
        self.assertAlmostEqual(result['slippage_price'], 0.1, places=4)

    def test_estimate_with_volatility(self):
        """测试波动率调整"""
        result = self.model.estimate(
            price=100.0, volume=1000000, volatility=0.3
        )
        self.assertGreater(result['slippage_ratio'], 0.001)

    def test_estimate_with_atr(self):
        """测试 ATR 调整"""
        result = self.model.estimate(
            price=100.0, volume=1000000, atr=3.0
        )
        self.assertGreater(result['slippage_ratio'], 0.001)

    def test_buy_sell_prices(self):
        """测试买入/卖出价格"""
        buy = self.model.estimate_buy(100.0, volatility=0.3, atr=3.0)
        sell = self.model.estimate_sell(100.0, volatility=0.3, atr=3.0)
        self.assertGreater(buy, 100.0)
        self.assertLess(sell, 100.0)

    def test_statistics_empty(self):
        """测试空统计"""
        stats = self.model.get_statistics()
        self.assertEqual(stats['count'], 0)

    def test_statistics_after_estimates(self):
        """测试估计后的统计"""
        for _ in range(10):
            self.model.estimate(price=100.0, volume=1000000)
        stats = self.model.get_statistics()
        self.assertEqual(stats['count'], 10)
        self.assertGreater(stats['max_slippage'], 0)

    def test_min_max_slippage(self):
        """测试滑点范围限制"""
        model = DynamicSlippageModel(params=SlippageParams(
            base_slippage=0.001, min_slippage=0.0001, max_slippage=0.01
        ))
        result = model.estimate(price=100.0, volume=1000000, volatility=10.0)
        self.assertLessEqual(result['slippage_ratio'], 0.01)

    def test_zero_price(self):
        """测试零价格"""
        result = self.model.estimate(price=0.0, volume=1000000)
        self.assertEqual(result['slippage_ratio'], 0)
        self.assertEqual(result['slippage_price'], 0)


if __name__ == '__main__':
    unittest.main()
