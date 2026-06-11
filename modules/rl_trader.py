#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
强化学习交易代理 — RL Trader

简化版 PPO (Proximal Policy Optimization) 交易代理, 纯 numpy 实现。

架构:
- TradingEnv: Gym 风格交易环境
- PPOAgent: 策略梯度代理
- RLTrader: 便捷接口
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from modules.logger import logger


class TradingEnv:
    """
    Gym 风格交易环境

    State: 特征值 + 持仓 + 现金 + 时间编码
    Action: 连续空间 [-1, 1] → 调仓幅度
      -1 = 全仓卖出, 0 = 不变, 1 = 满仓买入
    Reward: 风险调整收益 - 交易成本
    """

    def __init__(self, prices: np.ndarray, features: np.ndarray,
                 initial_capital: float = 1000000,
                 transaction_cost: float = 0.002):
        self.prices = prices  # (T,) 价格序列
        self.features = features  # (T, n_features) 特征
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost

        self.n_steps = len(prices)
        self.n_features = features.shape[1] if len(features.shape) > 1 else 1

        # 状态空间
        # [feature_dim, cash_ratio, position_ratio, time_ratio]
        self.state_dim = self.n_features + 3

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """重置环境"""
        if seed is not None:
            np.random.seed(seed)

        self.cash = self.initial_capital
        self.position = 0  # 持仓比例 [0, 1]
        self._step = 0
        self.total_value = self.initial_capital
        self.values = [self.initial_capital]
        self.trades = []

        return self._get_observation()

    def step(self, action: float) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        执行一步

        Args:
            action: 连续动作 [-1, 1]

        Returns:
            (observation, reward, done, info)
        """
        self._step += 1
        done = self._step >= self.n_steps - 1

        # 执行调仓
        old_position = self.position
        target_position = np.clip(self.position + action * 0.1, 0, 1)

        # 交易成本
        trade_size = abs(target_position - old_position)
        cost = trade_size * self.total_value * self.transaction_cost
        self.total_value -= cost

        # 更新持仓
        self.position = target_position
        self.cash = self.total_value * (1 - self.position)

        # 计算收益
        if self._step < self.n_steps:
            price_change = (self.prices[self._step] / self.prices[self._step - 1] - 1) if self._step > 0 else 0
            portfolio_return = self.position * price_change
            self.total_value *= (1 + portfolio_return)
            self.values.append(self.total_value)
        else:
            portfolio_return = 0

        # 奖励: 风险调整收益 - 交易成本
        reward = portfolio_return
        if abs(action) > 0.01:
            reward -= cost / self.initial_capital * 10  # 交易成本惩罚

        # 夏普比率奖励 (累积)
        if len(self.values) > 10:
            vals = np.array(self.values[-10:])
            recent_returns = np.diff(vals) / (vals[:-1] + 1e-10)
            if np.std(recent_returns) > 0:
                sharpe_bonus = np.mean(recent_returns) / np.std(recent_returns) * 0.01
                reward += sharpe_bonus

        info = {
            'total_value': self.total_value,
            'position': self.position,
            'cash': self.cash,
            'return': portfolio_return,
            'reward': reward,
        }

        return self._get_observation(), reward, done, info

    def _get_observation(self) -> np.ndarray:
        """获取当前状态"""
        obs = np.zeros(self.state_dim)

        # 特征值
        if self._step < self.n_steps:
            feat = self.features[self._step]
            if len(feat.shape) == 0:
                feat = np.array([feat])
            obs[:self.n_features] = feat / (np.abs(feat).max() + 1e-10)

        # 现金比例
        obs[self.n_features] = self.cash / (self.total_value + 1e-10)
        # 持仓比例
        obs[self.n_features + 1] = self.position
        # 时间编码
        obs[self.n_features + 2] = self._step / self.n_steps

        return obs


class PPOAgent:
    """
    简化版 PPO 代理

    使用两个 MLP (策略网络 + 价值网络)
    纯 numpy 实现前向传播和梯度更新
    """

    def __init__(self, state_dim: int, action_dim: int,
                 lr: float = 0.001, gamma: float = 0.99,
                 epsilon_clip: float = 0.2, gae_lambda: float = 0.95):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon_clip = epsilon_clip
        self.gae_lambda = gae_lambda

        # 经验回放
        self._observations = []
        self._actions = []
        self._rewards = []
        self._dones = []
        self._log_probs = []
        self._values = []

        # Adam 优化器状态 (policy + value 各 4 个参数)
        self._adam_state = {}

        # 网络参数 (简化: 单隐藏层)
        self._init_networks()

    def _init_networks(self):
        """初始化网络参数"""
        np.random.seed(42)
        # 策略网络: state → action_mean, action_log_std
        self.policy_w1 = np.random.randn(self.state_dim, 64) * 0.1
        self.policy_b1 = np.zeros(64)
        self.policy_w2 = np.random.randn(64, 1) * 0.01
        self.policy_b2 = np.zeros(1)

        # 价值网络: state → value
        self.value_w1 = np.random.randn(self.state_dim, 64) * 0.1
        self.value_b1 = np.zeros(64)
        self.value_w2 = np.random.randn(64, 1) * 0.01
        self.value_b2 = np.zeros(1)

    def select_action(self, observation: np.ndarray,
                      explore: float = 0.1) -> Tuple[float, float, float]:
        """
        选择动作 (epsilon-greedy + 策略网络)

        Returns:
            (action, log_prob, value)
        """
        if np.random.random() < explore:
            action = np.random.uniform(-1, 1)
            return action, 0.0, 0.0

        obs = observation.reshape(1, -1)

        # 策略网络前向
        h = self._relu(obs @ self.policy_w1 + self.policy_b1)
        action_mean = self._sigmoid(h @ self.policy_w2 + self.policy_b2) * 2 - 1  # [-1, 1]

        # 高斯探索
        log_std = np.array([0.0])
        std = np.exp(log_std)
        action = float(action_mean[0, 0] + np.random.randn() * std[0])
        action = np.clip(action, -1, 1)

        # Log probability (高斯)
        log_prob = -0.5 * ((action - action_mean[0, 0]) / std[0]) ** 2 - np.log(std[0] * np.sqrt(2 * np.pi))

        # 价值网络前向
        vh = self._relu(obs @ self.value_w1 + self.value_b1)
        value = float(self._sigmoid(vh @ self.value_w2 + self.value_b2)[0, 0] * 10 - 5)

        return float(action), float(log_prob), value

    def store_transition(self, obs, action, reward, next_obs, done):
        """存储经验"""
        self._observations.append(obs)
        self._actions.append(action)
        self._rewards.append(reward)
        self._dones.append(done)

    def train(self, batch_size: int = 64, epochs: int = 10):
        """训练策略和价值网络 (PPO + GAE)

        使用 numpy 实现完整的 PPO 更新:
        1. GAE 优势估计
        2. 裁剪代理目标 (clipped surrogate objective)
        3. MSE 价值损失
        4. Adam 优化器
        """
        n = len(self._rewards)
        if n < 2:
            return

        obs = np.array(self._observations)  # (n, state_dim)
        actions = np.array(self._actions)   # (n,)
        rewards = np.array(self._rewards)   # (n,)
        dones = np.array(self._dones)       # (n,)

        # ========== 1. 计算 GAE 优势 ==========
        # 价值网络当前预测
        v_pred = self._value_forward(obs).flatten()

        advantages = np.zeros(n)
        gae = 0.0
        for t in reversed(range(n)):
            if t == n - 1:
                delta = rewards[t] - v_pred[t]
            else:
                next_v = v_pred[t + 1] if not dones[t] else 0.0
                delta = rewards[t] + self.gamma * next_v - v_pred[t]
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages[t] = gae

        # 目标回报 = 优势 + 价值 (bootstrap)
        returns = advantages + v_pred
        # 标准化优势
        adv_mean = np.mean(advantages)
        adv_std = np.std(advantages) + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        # ========== 2. PPO 更新循环 ==========
        for epoch in range(epochs):
            indices = np.random.permutation(n)

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                idx = indices[start:end]
                bs = len(idx)

                batch_obs = obs[idx]       # (bs, state_dim)
                batch_adv = advantages[idx]  # (bs,)
                batch_ret = returns[idx]    # (bs,)
                batch_act = actions[idx]    # (bs,)

                # ---- 策略网络前向 ----
                h = self._relu(batch_obs @ self.policy_w1 + self.policy_b1)  # (bs, 64)
                action_mean = self._sigmoid(h @ self.policy_w2 + self.policy_b2) * 2 - 1  # (bs, 1)
                action_mean = action_mean.flatten()  # (bs,)
                std = 1.0

                # 当前 log probability (高斯)
                log_prob = -0.5 * ((batch_act - action_mean) / std) ** 2 - np.log(std * np.sqrt(2 * np.pi))

                # ---- 价值网络前向 ----
                vh = self._relu(batch_obs @ self.value_w1 + self.value_b1)
                v_pred_batch = self._sigmoid(vh @ self.value_w2 + self.value_b2).flatten()

                # ---- 策略损失 (clipped surrogate) ----
                ratio = np.exp(log_prob)
                surr1 = ratio * batch_adv
                surr2 = np.clip(ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * batch_adv
                policy_loss = -np.mean(np.minimum(surr1, surr2))

                # ---- 价值损失 (MSE) ----
                value_loss = np.mean((v_pred_batch - batch_ret) ** 2)

                # ---- 策略网络梯度 ----
                # d(policy_loss)/d(action_mean) via chain rule
                # min(ratio*a, clip(ratio)*a) 的梯度
                ratio_a = ratio * batch_adv
                clipped_ratio_a = np.clip(ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * batch_adv
                which = ratio_a < clipped_ratio_a  # 被 clipped 的位置

                # d(ratio)/d(action_mean) = ratio * (action_mean - act) / std^2
                d_mean_unclipped = ratio * (action_mean - batch_act) / (std ** 2)
                d_mean_clipped = ratio * (action_mean - batch_act) / (std ** 2)

                d_mean = np.where(which, d_mean_unclipped, d_mean_clipped)  # (bs,)

                # 通过 sigmoid + 全连接层的梯度
                sig_out = self._sigmoid(h @ self.policy_w2 + self.policy_b2)  # (bs, 1)
                d_action_mean = -d_mean.reshape(-1, 1) * 2 * sig_out * (1 - sig_out)  # (bs, 1)

                grad_w2 = h.T @ d_action_mean / bs  # (64, 1)
                grad_b2 = np.mean(d_action_mean, axis=0)  # (1,)

                d_h = d_action_mean @ self.policy_w2.T  # (bs, 64)
                d_pre = d_h * (h > 0).astype(float)  # ReLU 梯度
                grad_w1 = batch_obs.T @ d_pre / bs  # (8, 64)
                grad_b1 = np.mean(d_pre, axis=0)  # (64,)

                # ---- 价值网络梯度 ----
                v_err = 2 * (v_pred_batch - batch_ret) / bs  # (bs,)
                d_v = v_err.reshape(-1, 1) * self._sigmoid(vh @ self.value_w2 + self.value_b2) * (1 - self._sigmoid(vh @ self.value_w2 + self.value_b2))  # (bs, 1)

                grad_vw2 = vh.T @ d_v / bs  # (64, 1)
                grad_vb2 = np.mean(d_v, axis=0)  # (1,)

                d_vh = d_v @ self.value_w2.T  # (bs, 64)
                d_vpre = d_vh * (vh > 0).astype(float)  # ReLU 梯度
                grad_vw1 = batch_obs.T @ d_vpre / bs  # (8, 64)
                grad_vb1 = np.mean(d_vpre, axis=0)  # (64,)

                # ---- Adam 更新 ----
                self._adam_update('pw1', self.policy_w1, grad_w1)
                self._adam_update('pb1', self.policy_b1, grad_b1)
                self._adam_update('pw2', self.policy_w2, grad_w2)
                self._adam_update('pb2', self.policy_b2, grad_b2)
                self._adam_update('vw1', self.value_w1, grad_vw1)
                self._adam_update('vb1', self.value_b1, grad_vb1)
                self._adam_update('vw2', self.value_w2, grad_vw2)
                self._adam_update('vb2', self.value_b2, grad_vb2)

        # 清空经验
        self._observations.clear()
        self._actions.clear()
        self._rewards.clear()
        self._dones.clear()

    @staticmethod
    def _relu(x):
        return np.maximum(0, x)

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def _value_forward(self, obs: np.ndarray) -> np.ndarray:
        """价值网络前向 (批量)"""
        h = self._relu(obs @ self.value_w1 + self.value_b1)
        return self._sigmoid(h @ self.value_w2 + self.value_b2) * 10 - 5

    def _adam_update(self, key: str, param: np.ndarray, grad: np.ndarray):
        """Adam 优化器单步更新"""
        if key not in self._adam_state:
            self._adam_state[key] = {
                'm': np.zeros_like(param),
                'v': np.zeros_like(param),
                't': 0,
            }
        state = self._adam_state[key]
        state['t'] += 1
        t = state['t']

        # 一阶 / 二阶动量估计
        state['m'] = state['m'] * 0.9 + grad * 0.1
        state['v'] = state['v'] * 0.999 + grad ** 2 * 0.001

        # 偏差修正
        m_hat = state['m'] / (1 - 0.9 ** t)
        v_hat = state['v'] / (1 - 0.999 ** t)

        # 更新参数
        param -= self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)

    def save(self, path: str):
        """保存模型"""
        data = {
            'policy_w1': self.policy_w1, 'policy_b1': self.policy_b1,
            'policy_w2': self.policy_w2, 'policy_b2': self.policy_b2,
            'value_w1': self.value_w1, 'value_b1': self.value_b1,
            'value_w2': self.value_w2, 'value_b2': self.value_b2,
        }
        np.savez(path, **data)
        logger.info(f"[PPOAgent] 模型已保存: {path}")

    def load(self, path: str) -> bool:
        """加载模型"""
        try:
            data = np.load(path + '.npz')
            self.policy_w1 = data['policy_w1']
            self.policy_b1 = data['policy_b1']
            self.policy_w2 = data['policy_w2']
            self.policy_b2 = data['policy_b2']
            self.value_w1 = data['value_w1']
            self.value_b1 = data['value_b1']
            self.value_w2 = data['value_w2']
            self.value_b2 = data['value_b2']
            logger.info(f"[PPOAgent] 模型已加载: {path}")
            return True
        except Exception as e:
            logger.error(f"[PPOAgent] 加载失败: {e}")
            return False


class RLTrader:
    """RL 交易便捷接口"""

    def __init__(self, state_dim: int = 20, action_dim: int = 1):
        self.agent = PPOAgent(state_dim, action_dim)
        self._trained = False

    def train(self, prices: np.ndarray, features: np.ndarray,
              days: int = 120, episodes: int = 50) -> Dict:
        """训练 RL 代理"""
        logger.info(f"[RLTrader] 开始训练, {days} 天, {episodes} 轮")

        env = TradingEnv(prices[:days], features[:days])

        for ep in range(episodes):
            obs = env.reset()
            total_reward = 0
            explore = max(0.1, 1.0 - ep * 0.02)  # 逐步降低探索率

            while True:
                action, log_prob, value = self.agent.select_action(obs, explore)
                next_obs, reward, done, info = env.step(action)
                self.agent.store_transition(obs, action, reward, next_obs, done)
                total_reward += reward
                obs = next_obs
                if done:
                    break

            # 训练网络
            self.agent.train()

            if ep % 10 == 0:
                logger.info(f"[RLTrader] Episode {ep}/{episodes}, Reward: {total_reward:.4f}")

        self._trained = True
        return {
            'trained': True,
            'total_episodes': episodes,
            'final_reward': total_reward,
        }

    def trade(self, stock_code: str, capital: float = 100000) -> Dict:
        """实盘交易接口 (简化)"""
        if not self._trained:
            return {'error': '模型未训练'}
        return {
            'stock': stock_code,
            'capital': capital,
            'status': 'ready',
            'message': 'RL 代理已就绪',
        }

    def get_report(self) -> Dict:
        """交易报告"""
        return {
            'trained': self._trained,
            'state_dim': self.agent.state_dim,
            'action_dim': self.agent.action_dim,
        }


# 全局实例
rl_trader = RLTrader()
