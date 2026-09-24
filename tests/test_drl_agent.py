#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
DRLTradingAgent train_step 梯度修复验证 (2026-09-04)

背景: 23:00 自动训练链在 DRL 步崩溃 (ValueError: non-broadcastable
output operand with shape (64,1) doesn't match the broadcast shape (64,64))。
根因是 train_step 中 4 处梯度更新 shape 错误 + 策略更新不含动作信息。

测试策略 (合成+噪声双测验):
1. test_train_no_crash_2d — 2D 12 特征输入 (真实链路径: 480 样本 12 特征)
2. test_train_no_crash_3d — 3D 序列输入 (reshape 路径)
3. test_stability_high_lr — lr=0.01 下发散守卫 (修复前 15 epochs 权重爆炸到 2.7e15)
4. test_noise_no_nan — 纯噪声标签: 无 NaN/Inf, 权重保持有限
5. test_decide_after_train — 训练后 decide() 推理路径可用

注: 本测试验证的是修复的稳定性声明 (不崩溃/不发散/推理可用),
不验证"学会交易" — 随机采样 (s,s',r) 三元组 + γ=0.99 bootstrap 的设计
下 value net 无法收敛 (2026-09-04 实测 corr(V,r)=-0.06), 重设计另议。
"""

import unittest
import numpy as np

from modules.models.drl_agent import DRLTradingAgent, PortfolioState


def _make_env_data(n_samples: int = 200, n_features: int = 12,
                   signal: bool = True, seed: int = 7):
    """构造 env_data — signal=True 时标签与特征相关 (可学习), False 时纯噪声"""
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(n_samples, n_features))
    if signal:
        # 标签 = 第 0 特征 + 噪声 → 存在可学习结构
        labels = np.tanh(features[:, 0] * 2.0 + rng.normal(scale=0.1, size=n_samples))
    else:
        labels = rng.normal(size=n_samples)
    return {
        'features': features,
        'labels': labels,
        'seq_len': 20,
        'n_features': n_features,
    }


class TestDRLTrainStep(unittest.TestCase):
    """train_step 梯度 shape 修复验证"""

    def test_train_no_crash_2d(self):
        """2D 12 特征输入 (真实 23:00 链路径) — 修复前在此崩溃"""
        agent = DRLTradingAgent()
        result = agent.train(_make_env_data(signal=True), epochs=5, batch_size=32)
        self.assertTrue(agent._trained)
        self.assertTrue(np.isfinite(result['value_loss']))
        self.assertTrue(np.isfinite(result['policy_loss']))

    def test_train_no_crash_3d(self):
        """3D 序列输入 → reshape 到 STATE_DIM=24 路径"""
        agent = DRLTradingAgent()
        rng = np.random.default_rng(11)
        seq = rng.normal(size=(150, 20, 12))
        labels = rng.normal(size=150)
        result = agent.train(
            {'features': seq, 'labels': labels, 'seq_len': 20, 'n_features': 12},
            epochs=3, batch_size=16,
        )
        self.assertTrue(agent._trained)
        self.assertTrue(np.isfinite(result['value_loss']))

    def test_stability_high_lr(self):
        """发散守卫: lr=0.01 (10x 默认) 下权重必须保持有限。

        修复前该配置下 value_w2 在 ~15 epochs 内爆炸至 2.7e15 → NaN。
        根因: bootstrap 目标 (r + γV(s'), γ=0.99) 无 TD 裁剪时 V 过估
        → td 增大 → 权重指数发散。TD clip ±1 后必须全程有限。
        """
        agent = DRLTradingAgent(learning_rate=0.01)
        env = _make_env_data(n_samples=300, signal=True, seed=42)
        agent.train(env, epochs=50, batch_size=32)
        for name in ('policy_w1', 'policy_w2', 'value_w1', 'value_w2'):
            w = getattr(agent, name)
            self.assertTrue(np.all(np.isfinite(w)),
                            f"{name} 在 lr=0.01 下发散 (含 NaN/Inf) — TD clip 守卫失效")

    def test_signal_learning(self):
        """可学习性 (2026-09-08 轨迹化重设计核心声明):

        时序相关信号数据 (AR(1) 特征 + 信号生成标签) 上训练后,
        确定性策略 (argmax logits) 的动作方向与标签方向一致率
        必须从 ~0.5 随机基线显著提升。
        重设计前实测: 随机散点采样 + 动作无关 reward 下 corr(V,r)=-0.06,
        策略精度无提升 (装饰性训练); 重设计后实测 0.571 → 0.683。
        """
        rng = np.random.default_rng(42)
        n, F = 480, 12
        f = rng.normal(size=(n, F))
        f0 = np.zeros(n)
        f0[0] = rng.normal()
        for t in range(1, n):
            f0[t] = 0.7 * f0[t - 1] + rng.normal(scale=0.5)
        f[:, 0] = f0
        labels = np.where(f0 > 0.5, 2, np.where(f0 < -0.5, 0, 1))

        def policy_accuracy(agent):
            """确定性策略动作方向与标签方向的一致率 (跳过中性样本)"""
            correct = total = 0
            for t in range(100, 200):
                if labels[t] == 1:
                    continue
                s = np.pad(f[t], (0, 12))
                h = np.maximum(s @ agent.policy_w1 + agent.policy_b1, 0)
                logits = h @ agent.policy_w2 + agent.policy_b2
                idx = int(np.argmax(logits))  # 0=卖 1=买
                want = 1 if labels[t] == 2 else 0
                correct += (idx == want)
                total += 1
            return correct / total

        before = DRLTradingAgent()
        acc_before = policy_accuracy(before)

        trained = DRLTradingAgent()
        env = {'features': f, 'labels': labels, 'seq_len': 20, 'n_features': 12}
        trained.train(env, epochs=50, batch_size=32)
        acc_after = policy_accuracy(trained)

        self.assertGreater(acc_after, 0.55,
                           f"训练后策略精度 {acc_after:.3f} 未显著超过随机基线 0.5")
        self.assertGreater(acc_after, acc_before,
                           f"训练后精度 {acc_after:.3f} 未超过训练前 {acc_before:.3f}")

    def test_noise_no_nan(self):
        """纯噪声标签: 训练不崩溃且权重无 NaN/Inf"""
        agent = DRLTradingAgent()
        agent.train(_make_env_data(signal=False), epochs=5, batch_size=32)
        for name in ('policy_w1', 'policy_w2', 'value_w1', 'value_w2'):
            w = getattr(agent, name)
            self.assertTrue(np.all(np.isfinite(w)), f"{name} 含 NaN/Inf")

    def test_decide_after_train(self):
        """训练后 decide() 推理路径可用"""
        agent = DRLTradingAgent()
        agent.train(_make_env_data(signal=True), epochs=3, batch_size=16)
        rng = np.random.default_rng(3)
        klines = [{'close': float(100 + rng.normal()), 'high': float(101 + rng.normal()),
                   'low': float(99 + rng.normal()), 'volume': 1e6} for _ in range(60)]

        decision = agent.decide(klines, PortfolioState())
        self.assertIsNotNone(decision)
        self.assertIn(decision.action, ('buy', 'sell', 'hold', 'close'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
