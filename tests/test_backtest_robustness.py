#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""P0-2 回测真实性 + P2-11d 初始持仓敏感性 单元测试 — 零网络"""

import unittest

from modules.event_backtester import (
    ExecutionEngine, Order, OrderStatus, OrderType, Side)
from modules.backtest_robustness import initial_position_sensitivity


def _order(code='sz300620', side=Side.BUY, qty=100):
    return Order(stock_code=code, side=side, order_type=OrderType.MARKET,
                 quantity=qty, price=10.0)


def _bar(open_=10.0, high=10.5, low=9.5, close=10.0, volume=1e6, prev_close=None):
    b = {'open': open_, 'high': high, 'low': low, 'close': close,
         'volume': volume, 'date': '2026-09-14'}
    if prev_close is not None:
        b['prev_close'] = prev_close
    return b


class TradabilityTest(unittest.TestCase):
    def test_observe_limit_up_buy_still_fills(self):
        eng = ExecutionEngine(tradability_mode='observe')
        # 创业板 20% 涨停: close/prev = 12/10
        o = eng.execute_order(_order(), _bar(open_=11.5, high=12.0, low=11.0,
                                             close=12.0, prev_close=10.0))
        self.assertEqual(o.status, OrderStatus.FILLED)  # observe 不拦截
        self.assertEqual(eng.reject_stats['observe_would_reject'], 1)

    def test_enforce_limit_up_buy_rejected(self):
        eng = ExecutionEngine(tradability_mode='enforce')
        o = eng.execute_order(_order(), _bar(open_=11.5, high=12.0, low=11.0,
                                             close=12.0, prev_close=10.0))
        self.assertEqual(o.status, OrderStatus.REJECTED)
        self.assertIn('rejected:', o.reason)
        self.assertEqual(eng.reject_stats['enforce_rejected'], 1)

    def test_enforce_limit_down_sell_rejected(self):
        eng = ExecutionEngine(tradability_mode='enforce')
        o = eng.execute_order(_order(side=Side.SELL),
                              _bar(open_=8.5, high=8.8, low=8.0, close=8.0, prev_close=10.0))
        self.assertEqual(o.status, OrderStatus.REJECTED)

    def test_enforce_limit_up_sell_allowed(self):
        eng = ExecutionEngine(tradability_mode='enforce')
        o = eng.execute_order(_order(side=Side.SELL),
                              _bar(open_=11.5, high=12.0, low=11.0, close=12.0, prev_close=10.0))
        self.assertEqual(o.status, OrderStatus.FILLED)  # 涨停可卖

    def test_normal_bar_fills_and_no_reject(self):
        eng = ExecutionEngine(tradability_mode='enforce')
        o = eng.execute_order(_order(), _bar(close=10.1, prev_close=10.0))
        self.assertEqual(o.status, OrderStatus.FILLED)
        self.assertEqual(sum(eng.reject_stats.values()), 0)

    def test_no_prev_close_skipped_then_remembered(self):
        eng = ExecutionEngine(tradability_mode='enforce')
        o1 = eng.execute_order(_order(), _bar(close=10.1))  # 无 prev → 跳过
        self.assertEqual(o1.status, OrderStatus.FILLED)
        self.assertEqual(eng.reject_stats['skipped_no_price'], 1)
        self.assertAlmostEqual(eng._last_close['sz300620'], 10.1)
        # 第二根: 12/10.1 = +18.8% < 19.5% 容忍线, 仍可买
        o2 = eng.execute_order(_order(), _bar(open_=11.5, high=12.0, low=11.0, close=12.0))
        self.assertEqual(o2.status, OrderStatus.FILLED)

    def test_suspended_zero_volume_rejected(self):
        eng = ExecutionEngine(tradability_mode='enforce')
        o = eng.execute_order(_order(), _bar(close=10.0, volume=0, prev_close=10.0))
        self.assertEqual(o.status, OrderStatus.REJECTED)

    def test_default_mode_is_observe(self):
        eng = ExecutionEngine()
        self.assertEqual(eng.tradability_mode, 'observe')


class ImpactModelTest(unittest.TestCase):
    def test_quadratic_impact_calibration(self):
        eng = ExecutionEngine(impact_model='quadratic')
        # 参与率 10% → 0.1²×0.5 = 0.5%
        c = eng._impact_cost(_order(qty=100), {'volume': 1000})
        self.assertAlmostEqual(float(c), 0.005, places=6)

    def test_linear_default_unchanged(self):
        eng = ExecutionEngine()
        c = eng._impact_cost(_order(qty=100), {'volume': 1000})
        self.assertAlmostEqual(float(c), 0.05, places=6)  # 0.1×0.5 原值

    def test_quadratic_marginal_far_below_linear(self):
        q = ExecutionEngine(impact_model='quadratic')._impact_cost(_order(qty=10), {'volume': 1000})
        l = ExecutionEngine()._impact_cost(_order(qty=10), {'volume': 1000})
        self.assertLess(float(q), float(l))  # 小单二次冲击远小于线性


class InitialPositionSensitivityTest(unittest.TestCase):
    def test_dispersion_hand_computed(self):
        navs = {1.0: 1.0, 2: 1.02, 3: 0.98}
        res = initial_position_sensitivity(
            lambda w: navs[w[0]], [[1.0], [2], [3]])
        self.assertEqual(res['n_ok'], 3)
        self.assertAlmostEqual(res['nav_min'], 0.98)
        self.assertAlmostEqual(res['nav_max'], 1.02)
        self.assertAlmostEqual(res['dispersion'], 0.04)  # (1.02-0.98)/1.0
        self.assertEqual(res['verdict'], 'sensitive')

    def test_stable(self):
        res = initial_position_sensitivity(lambda w: 1.0, [[0.5], [0.5, 0.5]])
        self.assertEqual(res['dispersion'], 0.0)
        self.assertEqual(res['verdict'], 'stable')

    def test_highly_sensitive(self):
        res = initial_position_sensitivity(lambda w: w[0], [[1.0], [1.5]])
        self.assertEqual(res['verdict'], 'highly_sensitive')

    def test_error_isolation(self):
        def fn(w):
            if w[0] < 0:
                raise ValueError('neg')
            return 1.0

        res = initial_position_sensitivity(fn, [[1.0], [-1.0], [2.0]])
        self.assertEqual(res['n_ok'], 2)
        self.assertEqual(res['n_error'], 1)
        self.assertIsNotNone(res['results'][1]['error'])

    def test_empty_grid(self):
        res = initial_position_sensitivity(lambda w: 1.0, [])
        self.assertEqual(res['verdict'], 'insufficient')


if __name__ == '__main__':
    unittest.main()
