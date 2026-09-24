#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
P1 测试 (2026-09-10): TradingEnvV2 a_share 执行保真模型 (默认关)

锁链 (09-10 P0 同风格 — 真实链形态锁进测试, 不盲改):
  1. 默认 simple = 全链尾 (200 步) — 保证无 a_share 调用方不再断
  2. 涨停拒买 / 跌停拒卖 (simple 对照 = 照常成交, 证明保真只作用于 a_share)

运行: python -m unittest tests.test_rl_env
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.rl_trader_v2 import TradingEnvV2


def _mk_env(prices, execution_model='simple', limit_pct=0.10, n_feat=6):
    """构造 env — execution_model 默认 'simple' (不传参即默认路径)"""
    feat = np.random.RandomState(4).randn(len(prices), n_feat)
    return TradingEnvV2(np.array(prices, dtype=float), feat,
                        execution_model=execution_model, limit_pct=limit_pct)


class TestTradingEnvV2ExecutionModel(unittest.TestCase):
    """TradingEnvV2 execution_model 参数 (默认关) + a_share 拒单语义"""

    def test_default_simple_full_episode(self):
        """默认 (不传 execution_model) = simple 旧路径, 全链尾 200 步跑通"""
        prices = 100 + np.random.RandomState(3).randn(260).cumsum() * 0.5
        env = _mk_env(prices)
        obs = env.reset(seed=42)
        self.assertEqual(obs.shape, (11,))  # n_features(6)+5 维 (obs 维度不变 = 4 推理消费方兼容)
        rng = np.random.RandomState(42)
        steps, done = 0, False
        while not done and steps < 200:
            obs, r, done, info = env.step(int(rng.choice([0, 1, 2])))
            steps += 1
        self.assertGreater(steps, 5)
        m = env.get_episode_metrics()
        self.assertTrue(np.isfinite(m['total_return']))
        self.assertTrue(np.isfinite(m['sharpe_ratio']))

    def test_a_share_full_episode_smoke(self):
        """a_share 全链尾: 200 步不崩 (拒单/成本分支被踩到即可)"""
        prices = 100 + np.random.RandomState(3).randn(260).cumsum() * 0.5
        env = _mk_env(prices, execution_model='a_share')
        env.reset(seed=42)
        rng = np.random.RandomState(42)
        steps, done = 0, False
        while not done and steps < 200:
            obs, r, done, info = env.step(int(rng.choice([0, 1, 2])))
            steps += 1
        self.assertGreater(steps, 5)

    def test_limit_up_rejects_buy(self):
        """a_share: 涨停日 (close +21%) 买 → 拒单 (无成交), 非涨跌停日照常"""
        prices = [100.0] * 30
        prices[10] = 121.0  # step10: +21% → limit_up (主板 10% / 创业 20% 双触发)
        env = _mk_env(prices, execution_model='a_share', limit_pct=0.10)
        env.reset(seed=7)
        log = {}
        for i in range(1, 26):
            _, _, done, info = env.step(1)
            log[i] = info.get('rejected')
            if done:
                break
        self.assertEqual(log.get(10), 'limit_up')
        self.assertNotIn(10, [t['step'] for t in env.trades])  # 拒单日无成交

    def test_limit_down_rejects_sell(self):
        """a_share: 跌停日卖 → 拒单持仓保留; 次日卖 → 成交"""
        prices = [100.0, 100.0, 79.0, 100.0, 100.0, 100.0]
        env = _mk_env(prices, execution_model='a_share', limit_pct=0.10)
        env.reset(seed=7)
        _, _, _, i1 = env.step(1)   # step1: pct=0 正常买入
        self.assertIsNone(i1['rejected'])
        self.assertEqual(len(env.trades), 1)
        _, _, _, i2 = env.step(2)   # step2: -21% 跌停 → 卖拒单
        self.assertEqual(i2['rejected'], 'limit_down')
        self.assertEqual(len(env.trades), 1)  # 拒单 = 无成交
        _, _, _, i3 = env.step(2)   # step3: +26.6% 反弹日卖 → 允许 (非跌停止)
        self.assertIsNone(i3['rejected'])
        self.assertEqual(len(env.trades), 2)

    def test_simple_control_no_rejection(self):
        """对照: simple 永不产生 rejected (默认关语义锁定, 行为与 2026-07 逐字节一致)"""
        prices = [100.0] * 30
        prices[10] = 121.0
        env = _mk_env(prices, execution_model='simple')
        env.reset(seed=7)
        for i in range(1, 26):
            _, _, done, info = env.step(1)
            self.assertIsNone(info.get('rejected'))
            if done:
                break


class TestRLTraderV2TrainChain(unittest.TestCase):
    """RLTraderV2.train adapter (2026-09-10 死链修复) — POST /api/rl/train 端点链冒烟"""

    def _mk_data(self, n=60, f=12, seed=5):
        rng = np.random.RandomState(seed)
        X = rng.randn(n, f) * 0.1
        closes = 100 + rng.randn(n).cumsum() * 0.3
        return X, closes

    def test_train_chain_smoke_singleton_shape(self):
        """真实 singleton 形: state_dim=20, X 12D → pad 15, 训练链跑通 (endpoint .train 同形)"""
        from modules.rl_trader_v2 import RLTraderV2
        X, closes = self._mk_data()
        trader = RLTraderV2(sac_as_default=True)
        result = trader.train(X, None, closes=closes, n_episodes=2, timeout=120.0)
        self.assertEqual(result.get('agent'), 'ppo')
        self.assertEqual(result.get('episodes'), 2)
        self.assertIn('mean_reward', result)
        self.assertIn('elapsed_s', result)
        self.assertTrue(np.isfinite(result['mean_reward']))

    def test_train_pkl_shape_chain(self):
        """服务 load 链同形: ppo_agent 替换为 17D (07-15 pkl 形) → X(12D) 原生直通 (09-11 链尾实证形)"""
        from modules.rl_trader_v2 import RLTraderV2, PPOAgentV2
        X, closes = self._mk_data()
        trader = RLTraderV2(sac_as_default=True)
        trader.ppo_agent = PPOAgentV2(17, 3)   # 模拟 service 启动 rl_trader_v2.load() 后的实例
        result = trader.train(X, None, closes=closes, n_episodes=2, timeout=120.0)
        self.assertEqual(result.get('agent'), 'ppo')
        self.assertEqual(result.get('episodes'), 2)
        self.assertTrue(np.isfinite(result['mean_reward']))

    def test_train_guard_breaks_at_deadline(self):
        """墙钟 budget guard: 超时 → break (防 2400s 式单点阻塞) + 空 replay guard (无 NaN)"""
        import time as _t
        from modules.rl_trader_v2 import RLTraderV2, TradingEnvV2
        prices = np.linspace(100, 101, 120)
        feat = np.random.RandomState(1).randn(120, 15)
        trader = RLTraderV2(sac_as_default=True)
        env = TradingEnvV2(prices, feat)
        result = trader.train_ppo(env, n_episodes=3, timeout=_t.perf_counter() - 1.0)
        self.assertEqual(result.get('episodes'), 0)
        self.assertTrue(np.isfinite(result.get('mean_reward', 0.0)))
        # adapter 层: 全预算耗尽同样不崩 (max(0.05) guard)
        result2 = trader.train(np.random.RandomState(2).randn(60, 12), None, n_episodes=2, timeout=240.0)
        self.assertIn('agent', result2)


if __name__ == '__main__':
    unittest.main()
