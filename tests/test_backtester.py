#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
回测引擎单元测试

P3 测试 (2026-07-01):
  验证回测引擎的正确性: 涨跌停检查、成本计算、止损逻辑。
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.walkforward_backtester import (
    WalkForwardBacktester,
    TransactionCostModel,
)
from modules.event_backtester import (
    EventDrivenBacktester,
    Side,
    OrderType,
    ExecutionEngine,
)


class TestTransactionCostModel(unittest.TestCase):
    """测试交易成本模型"""

    def test_round_trip_cost(self):
        """测试往返交易成本计算"""
        model = TransactionCostModel(commission_rate=0.0003, stamp_tax=0.001, slippage_bps=2)
        cost = model.calculate_round_trip_cost(100000)  # 10 万元

        # 买入佣金: max(100000 * 0.0003, 5) = 30
        # 卖出佣金: 30
        # 印花税: 100000 * 0.001 = 100
        # 滑点: 100000 * 2 / 10000 = 20
        # 总计: 30 + 30 + 100 + 20 = 180
        self.assertAlmostEqual(cost['total_cost'], 180.0, places=1)
        self.assertAlmostEqual(cost['total_cost_pct'], 0.0018, places=4)

    def test_min_commission(self):
        """测试最低佣金 5 元"""
        model = TransactionCostModel(commission_rate=0.0003, stamp_tax=0.001, slippage_bps=2)
        cost = model.calculate_round_trip_cost(1000)  # 1000 元

        # 买入佣金: max(1000 * 0.0003, 5) = 5
        # 卖出佣金: 5
        # 印花税: 1000 * 0.001 = 1
        # 滑点: 1000 * 2 / 10000 = 0.2
        # 总计: 5 + 5 + 1 + 0.2 = 11.2
        self.assertAlmostEqual(cost['total_cost'], 11.2, places=1)


class TestLimitUpDown(unittest.TestCase):
    """测试涨跌停检查"""

    def test_limit_up_main_board(self):
        """主板涨停 10%"""
        self.assertTrue(WalkForwardBacktester.is_limit_up(11.0, 10.0))
        self.assertFalse(WalkForwardBacktester.is_limit_up(10.9, 10.0))

    def test_limit_up_gem(self):
        """创业板/科创板涨停 20%"""
        self.assertTrue(WalkForwardBacktester.is_limit_up(12.0, 10.0, is_gem=True))
        self.assertFalse(WalkForwardBacktester.is_limit_up(11.9, 10.0, is_gem=True))

    def test_limit_down_main_board(self):
        """主板跌停 10%"""
        self.assertTrue(WalkForwardBacktester.is_limit_down(9.0, 10.0))
        self.assertFalse(WalkForwardBacktester.is_limit_down(9.1, 10.0))

    def test_st_stock(self):
        """ST 股票涨跌停 5%"""
        self.assertTrue(WalkForwardBacktester.is_limit_up(10.5, 10.0, is_st=True))
        self.assertTrue(WalkForwardBacktester.is_limit_down(9.5, 10.0, is_st=True))


class TestWalkForwardBacktester(unittest.TestCase):
    """测试 Walk-Forward 回测"""

    def _generate_klines(self, n: int = 300, start_price: float = 100.0) -> list:
        """生成模拟 K 线数据"""
        np.random.seed(42)
        klines = []
        price = start_price
        for i in range(n):
            ret = np.random.randn() * 0.02
            open_price = price
            close_price = price * (1 + ret)
            high_price = max(open_price, close_price) * (1 + abs(np.random.randn()) * 0.01)
            low_price = min(open_price, close_price) * (1 - abs(np.random.randn()) * 0.01)
            klines.append({
                'date': f'2025-01-{(i % 28) + 1:02d}',
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'volume': int(np.random.randint(100000, 1000000)),
                'prev_close': round(open_price, 2),
                'atr': round(start_price * 0.03, 2),
            })
            price = close_price
        return klines

    def test_walk_forward_with_callback(self):
        """测试带 retrain_callback 的 Walk-Forward"""
        def dummy_callback(train_data, params):
            return {'train_size': len(train_data)}

        bt = WalkForwardBacktester(
            strategy_func=lambda bar, pos, cap: 'hold',
            initial_capital=1000000,
        )

        klines = self._generate_klines(300)
        result = bt.run_walk_forward(klines, train_period=60, test_period=20, n_windows=5,
                                      retrain_callback=dummy_callback)

        self.assertEqual(result['n_windows'], 5)
        self.assertEqual(result['method'], 'walk_forward')
        self.assertIn('summary', result)

    def test_walk_forward_without_callback(self):
        """测试不带 callback 的 Walk-Forward (退化为普通回测)"""
        bt = WalkForwardBacktester(
            strategy_func=lambda bar, pos, cap: 'hold',
            initial_capital=1000000,
        )

        klines = self._generate_klines(300)
        result = bt.run_walk_forward(klines, train_period=60, test_period=20, n_windows=3)

        self.assertEqual(result['n_windows'], 3)


class TestStatisticalOverfitCheck(unittest.TestCase):
    """统计过拟合诊断测试 (P0, 2026-09-10) — 锁定链尾真实链行为"""

    def _bt(self):
        return WalkForwardBacktester(lambda bar, pos, cap: 'hold', initial_capital=1000000)

    def test_quasi_flat_chain_skipped(self):
        """低活跃链 (r1 实测形态: 仅 w1 有 4/22 非零) → 三块全 skipped, 无 PBO 假信号"""
        import random
        random.seed(7)
        w1_dr = [0.0] * 22
        for i in (3, 8, 14, 19):
            w1_dr[i] = random.uniform(-0.01, 0.01)
        windows = [{'window': 1, 'daily_returns': w1_dr, 'sharpe_ratio': -0.5,
                    'total_return': -0.001, 'win_rate': 0.5}]
        windows += [{'window': w, 'daily_returns': [0.0] * 22, 'sharpe_ratio': 0.0,
                     'total_return': 0.0, 'win_rate': 0.5} for w in range(2, 6)]
        out = self._bt().statistical_overfit_check(windows)
        self.assertIn('skipped', out)
        self.assertTrue(out['skipped'].startswith('low_activity'))
        self.assertNotIn('dsr', out)
        self.assertNotIn('pbo_cscv', out)
        self.assertEqual(out['t_test'], {})

    def test_active_chain_full_diagnostics(self):
        """活跃链 → DSR + PBO-CSCV + t 检验三块全输出 (win_rate 恒等 → NaN 防护跳过)"""
        rng = np.random.RandomState(7)
        windows = [{'window': w,
                    'daily_returns': list(rng.randn(22) * 0.01),
                    'sharpe_ratio': float(rng.randn()),
                    'total_return': float(rng.randn() * 0.05),
                    'win_rate': 0.55} for w in range(1, 6)]
        out = self._bt().statistical_overfit_check(windows)
        self.assertIn('dsr', out)
        self.assertIn('pbo_cscv', out)
        self.assertEqual(out['pbo_cscv']['n_combinations'], 252)
        self.assertIn('sharpe_ratio', out['t_test'])
        self.assertNotIn('win_rate', out['t_test'])  # 常量指标不做 t 检验 (NaN 防护)


class TestClassicStrategyFixtures(unittest.TestCase):
    """经典策略 fixture 喂链 (2026-09-11 P2) — walk_forward+DSR/PBO 真链锁定

    断链真相 (本次 probe 实锤, 锁成回归):
      ① run_backtest :145 len<30 → {'error':'数据不足'} 被吞 → test_period=20 历史用例
         全窗 error + n_windows==5 假绿 (200 tests 绿 ≠ 验链, 07-31/09-10 教训三度重演)
      ② statistical_overfit_check 在恒 hold/skip 链上恒 skipped = DSR/PBO 从未真跑
    修复形: test_period=42 (服务链 backtest_routes 同形) + 活跃 fixture 喂链
    → tx>0 + DSR/PBO/CSCV/t 真激活 (实测 26-41/110 活跃 ≫10% skipped 门)。
    """

    def _generate_klines(self, n: int = 300, seed: int = 42) -> list:
        """随机游走 OHLC (vol 2%/日, 内嵌日内振幅 = 保证突破类策略有信号)"""
        rng = np.random.RandomState(seed)
        klines, price = [], 100.0
        for i in range(n):
            ret = rng.randn() * 0.02
            o, c = price, price * (1 + ret)
            hi = max(o, c) * (1 + abs(rng.randn()) * 0.01)
            lo = min(o, c) * (1 - abs(rng.randn()) * 0.01)
            klines.append({
                'date': f'2025-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}',
                'open': o, 'high': hi, 'low': lo, 'close': c,
                'volume': int(rng.randint(100000, 1000000)),
                'prev_close': o, 'atr': price * 0.03,
            })
            price = c
        return klines

    FIXTURES = ['dual_thrust', 'donchian', 'atr_rsi', 'keltner']

    def test_fixture_signal_density_non_hold(self):
        """四 fixture 原始信号密度 ≥1 buy/300 bars (绕开引擎直测链; 恒 hold mock 的反面锁)"""
        from modules.strategies.classic_fixture import ALL_FIXTURES
        klines = self._generate_klines(300)
        for name in self.FIXTURES:
            f = ALL_FIXTURES[name]()
            acts = [f(b, 0, 1e6) for b in klines]
            self.assertGreaterEqual(acts.count('buy'), 1, f'{name} 恒 hold = 死 mock')

    def test_walk_forward_fixture_active_diagnostics(self):
        """walk_forward(60/42/5w)+fixture → tx>0 + DSR/PBO/CSCV 真激活 (非 skipped)"""
        from modules.strategies.classic_fixture import ALL_FIXTURES
        klines = self._generate_klines(270)  # 60+42×5
        for name in self.FIXTURES:
            bt = WalkForwardBacktester(ALL_FIXTURES[name](), initial_capital=1_000_000)
            r = bt.run_walk_forward(klines, train_period=60, test_period=42, n_windows=5)
            self.assertEqual(r['n_windows'], 5, f'{name} 窗口未跑')
            tx = [t for w in r['windows'] for t in w.get('transactions', [])]
            self.assertGreater(len(tx), 0, f'{name}: 全窗零成交 = 断链')
            self.assertNotIn('error', r['windows'][0], f'{name}: 窗内 error 未暴露')
            so = bt.statistical_overfit_check(r['windows'])
            self.assertNotIn('skipped', so, f'{name}: 统计诊断仍 skipped = 喂链失败')
            self.assertIn('dsr', so, f'{name}: DSR 未激活')
            self.assertIn('pbo_cscv', so, f'{name}: PBO-CSCV 未激活')

    def test_short_window_error_guard(self):
        """:145 门 (len<30→数据不足) 锁定 — test20 时代假绿的反证链 (改 42 后此形=42 已验)"""
        klines = self._generate_klines(25)
        bt = WalkForwardBacktester(lambda bar, pos, cap: 'hold', initial_capital=1_000_000)
        r = bt.run_backtest(klines[:25])
        self.assertEqual(r.get('error'), '数据不足')


class TestExecutionEngine(unittest.TestCase):
    """测试执行引擎"""

    def test_market_order(self):
        """测试市价单执行"""
        engine = ExecutionEngine(commission_rate=0.0003, stamp_tax=0.001, slippage_bps=2)
        order = type('Order', (), {
            'stock_code': 'sz300620', 'side': Side.BUY,
            'order_type': OrderType.MARKET, 'quantity': 100,
            'price': 0, 'strategy': 'test', 'stop_price': 0,
            'limit_price': 0, 'timestamp': '',
        })()

        market_data = {'open': 100.0, 'high': 101.0, 'low': 99.0, 'close': 100.5, 'date': '2025-01-01'}
        result = engine.execute_order(order, market_data)
        self.assertIsNotNone(result)
        self.assertEqual(result.status.value, 'filled')
        self.assertGreater(result.avg_fill_price, 0)


if __name__ == '__main__':
    unittest.main()
