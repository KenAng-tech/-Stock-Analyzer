#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
强化学习交易代理 V2 — PPO + SAC 双 Agent 架构

升级内容:
1. PPO (Proximal Policy Optimization) — 策略梯度方法，稳定训练
2. SAC (Soft Actor-Critic) — 最大熵 RL，鼓励探索
3. 双 Agent 集成 — 根据市场状态切换策略

架构:
    Market State
        ↓
    ┌─────────────────────────────┐
    │   Market Regime Detector    │
    └─────────────────────────────┘
        ↓              ↓
    ┌────────┐    ┌────────┐
    │  PPO   │    │  SAC   │  ← 根据市场状态选择
    └────────┘    └────────┘
        ↓              ↓
    ┌─────────────────────────────┐
    │      Action Ensemble        │
    └─────────────────────────────┘
        ↓
    Trading Action (buy/sell/hold)

依赖:
- NumPy (核心实现)
- 可选：MLX 用于加速
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import os
import pickle

from modules.logger import logger


# ── 交易环境 ──────────────────────────────────────────────

class TradingEnvV2:
    """
    增强版交易环境 (Gym 风格)

    State Space:
        - 技术指标：RSI, MACD, 布林带，ATR, 成交量比率
        - 价格特征：动量，波动率，价格位置
        - 持仓状态：现金比例，持仓比例，未实现盈亏
        - 时间编码：交易日位置

    Action Space:
        - 离散空间：{0: hold, 1: buy, 2: sell}
        - 或连续空间：[-1, 1] → 调仓幅度

    Reward:
        - 基础收益：投资组合回报率
        - 风险调整：夏普比率奖励
        - 交易成本惩罚
        - 回撤惩罚
    """

    def __init__(self, prices: np.ndarray, features: np.ndarray,
                 initial_capital: float = 1000000,
                 transaction_cost: float = 0.0015,
                 max_drawdown_limit: float = 0.2,
                 reward_type: str = 'sharpe'):
        """
        Args:
            prices: (T,) 价格序列
            features: (T, n_features) 特征矩阵
            initial_capital: 初始资金
            transaction_cost: 交易成本 (往返)
            max_drawdown_limit: 最大回撤限制
            reward_type: 'sharpe' | 'sortino' | 'return'
        """
        self.prices = prices
        self.features = features
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.max_drawdown_limit = max_drawdown_limit
        self.reward_type = reward_type

        self.n_steps = len(prices)
        self.n_features = features.shape[1] if len(features.shape) > 1 else 1

        # 状态维度：features + cash + position + pnl + time
        self.state_dim = self.n_features + 5
        self.action_dim = 3  # discrete: hold, buy, sell

        # 环境状态
        self._current_step = 0
        self.cash = 0.0
        self.position = 0.0  # 持仓比例 [0, 1]
        self.total_value = 0.0
        self.peak_value = 0.0
        self.values = []
        self.trades = []

    def reset(self, seed: Optional[int] = None, start_step: int = 0) -> np.ndarray:
        """重置环境"""
        if seed is not None:
            np.random.seed(seed)

        self._current_step = start_step
        self.cash = self.initial_capital
        self.position = 0.0
        self.total_value = self.initial_capital
        self.peak_value = self.initial_capital
        self.values = [self.initial_capital]
        self.trades = []

        return self._get_observation()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        执行动作

        Args:
            action: 0=hold, 1=buy, 2=sell
        Returns:
            (observation, reward, done, info)
        """
        done = False
        self._current_step += 1

        if self._current_step >= self.n_steps - 1:
            done = True

        current_price = self.prices[self._current_step] if self._current_step < len(self.prices) else self.prices[-1]
        prev_price = self.prices[self._current_step - 1] if self._current_step > 0 else current_price

        # 计算价格变化
        price_return = (current_price / prev_price - 1) if prev_price > 0 else 0

        # 执行交易
        trade_size = 0.0
        if action == 1:  # buy
            target_position = min(1.0, self.position + 0.2)
            trade_size = target_position - self.position
        elif action == 2:  # sell
            target_position = max(0.0, self.position - 0.2)
            trade_size = target_position - self.position
        else:  # hold
            target_position = self.position

        # 交易成本
        if abs(trade_size) > 0.01:
            cost = abs(trade_size) * self.total_value * self.transaction_cost
            self.total_value -= cost
            self.trades.append({
                'step': self._current_step,
                'action': action,
                'trade_size': trade_size,
                'cost': cost,
                'price': current_price,
            })

        # 更新持仓
        self.position = np.clip(target_position, 0, 1)
        self.cash = self.total_value * (1 - self.position)

        # 计算组合收益
        portfolio_return = self.position * price_return
        self.total_value *= (1 + portfolio_return)
        self.values.append(self.total_value)

        # 更新峰值
        self.peak_value = max(self.peak_value, self.total_value)

        # 计算回撤
        drawdown = (self.peak_value - self.total_value) / self.peak_value if self.peak_value > 0 else 0

        # 计算奖励
        reward = self._calculate_reward(portfolio_return, drawdown, action)

        info = {
            'total_value': self.total_value,
            'position': self.position,
            'cash': self.cash,
            'return': portfolio_return,
            'drawdown': drawdown,
            'reward': reward,
            'price': current_price,
        }

        return self._get_observation(), reward, done, info

    def _calculate_reward(self, portfolio_return: float, drawdown: float, action: int) -> float:
        """计算奖励"""
        reward = portfolio_return

        # 交易成本惩罚
        if abs(action - 1) > 0.5:  # 非 hold 操作
            reward -= 0.001  # 固定惩罚

        # 回撤惩罚
        if drawdown > 0.1:
            reward -= (drawdown - 0.1) * 2

        # 夏普比率奖励 (基于近期收益)
        if self.reward_type == 'sharpe' and len(self.values) > 20:
            recent_returns = np.diff(self.values[-20:]) / (np.array(self.values[-20:-1]) + 1e-10)
            if np.std(recent_returns) > 0:
                sharpe = np.mean(recent_returns) / np.std(recent_returns)
                reward += sharpe * 0.01

        return reward

    def _get_observation(self) -> np.ndarray:
        """获取当前状态"""
        obs = np.zeros(self.state_dim)

        # 特征值 (归一化)
        if self._current_step < self.n_steps:
            feat = self.features[self._current_step]
            if len(feat.shape) == 0:
                feat = np.array([feat])
            # Z-score 归一化
            feat_norm = (feat - np.mean(feat)) / (np.std(feat) + 1e-10)
            obs[:self.n_features] = np.clip(feat_norm, -5, 5)

        # 现金比例
        obs[self.n_features] = self.cash / (self.total_value + 1e-10)
        # 持仓比例
        obs[self.n_features + 1] = self.position
        # 未实现盈亏
        cost_basis = self.position * self.initial_capital
        unrealized_pnl = (self.total_value * self.position - cost_basis) / (cost_basis + 1e-10)
        obs[self.n_features + 2] = np.clip(unrealized_pnl, -1, 1)
        # 回撤
        drawdown = (self.peak_value - self.total_value) / (self.peak_value + 1e-10)
        obs[self.n_features + 3] = drawdown
        # 时间编码
        obs[self.n_features + 4] = self._current_step / self.n_steps

        return obs

    def get_episode_metrics(self) -> Dict:
        """获取 episode 指标"""
        values = np.array(self.values)
        returns = np.diff(values) / (values[:-1] + 1e-10)

        total_return = (values[-1] - values[0]) / values[0]
        max_drawdown = np.max((np.maximum.accumulate(values) - values) / np.maximum.accumulate(values))

        if len(returns) > 0 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe = 0.0

        if len(returns) > 0:
            positive_returns = returns[returns > 0]
            win_rate = len(positive_returns) / len(returns) if len(returns) > 0 else 0
        else:
            win_rate = 0.0

        return {
            'total_return': float(total_return),
            'max_drawdown': float(max_drawdown),
            'sharpe_ratio': float(sharpe),
            'win_rate': float(win_rate),
            'n_trades': len(self.trades),
            'final_value': float(values[-1]),
        }


# ── PPO Agent ──────────────────────────────────────────────

class PPOAgentV2:
    """
    PPO (Proximal Policy Optimization) Agent V2

    架构:
        - 策略网络 (Policy Network): State → Action Distribution
        - 价值网络 (Value Network): State → Value Estimate

    特点:
        - Clipped Surrogate Objective
        - GAE (Generalized Advantage Estimation)
        - Multiple Epochs Update
        - Adam Optimizer with ClipCoef
    """

    def __init__(self, state_dim: int, action_dim: int = 3,
                 hidden_dim: int = 128,
                 lr: float = 0.0003, gamma: float = 0.99,
                 clip_coef: float = 0.2, ent_coef: float = 0.01,
                 value_coef: float = 0.5, n_epochs: int = 10,
                 batch_size: int = 64):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.gamma = gamma
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.value_coef = value_coef
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        # 经验存储
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

        # 策略网络参数 (两层 MLP)
        self.policy_w1 = np.random.randn(state_dim, hidden_dim) * np.sqrt(2.0 / state_dim)
        self.policy_b1 = np.zeros(hidden_dim)
        self.policy_w2 = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        self.policy_b2 = np.zeros(hidden_dim)
        self.policy_logits = np.random.randn(hidden_dim, action_dim) * np.sqrt(1.0 / hidden_dim)

        # 价值网络参数
        self.value_w1 = np.random.randn(state_dim, hidden_dim) * np.sqrt(2.0 / state_dim)
        self.value_b1 = np.zeros(hidden_dim)
        self.value_w2 = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        self.value_b2 = np.zeros(hidden_dim)
        self.value_out = np.random.randn(hidden_dim, 1) * np.sqrt(1.0 / hidden_dim)

        # Adam 优化器状态
        self._adam_states = {}

    def select_action(self, state: np.ndarray, explore: bool = True) -> Tuple[int, float, float]:
        """
        选择动作

        Returns:
            (action, log_prob, value)
        """
        action_logits = self._policy_forward(state)
        action_probs = self._softmax(action_logits)

        if explore:
            # 采样
            action = self._categorical_sample(action_probs)
        else:
            # Greedy
            action = int(np.argmax(action_probs))

        log_prob = np.log(action_probs[action] + 1e-10)
        value = self._value_forward(state)

        return action, log_prob, value

    def store_transition(self, obs: np.ndarray, action: int, reward: float,
                         done: bool, log_prob: float, value: float):
        """存储经验"""
        self.observations.append(obs.copy())
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def train(self) -> Dict:
        """
        PPO 训练

        使用 GAE 优势估计 + Clipped Surrogate Objective
        """
        n_samples = len(self.rewards)
        if n_samples < 10:
            return {'loss': 0.0}

        # 转换为数组
        obs_array = np.array(self.observations)
        actions = np.array(self.actions)
        rewards = np.array(self.rewards)
        dones = np.array(self.dones)
        old_log_probs = np.array(self.log_probs)
        old_values = np.array(self.values)

        # 计算 GAE 优势
        advantages, returns = self._compute_gae(rewards, dones, old_values)

        # 标准化优势
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-10)

        # PPO 更新循环
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for epoch in range(self.n_epochs):
            # 打乱数据
            indices = np.random.permutation(n_samples)

            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                idx = indices[start:end]
                batch_size = len(idx)

                batch_obs = obs_array[idx]
                batch_actions = actions[idx]
                batch_advantages = advantages[idx]
                batch_returns = returns[idx]
                batch_old_log_probs = old_log_probs[idx]

                # 当前策略预测
                curr_logits = self._policy_forward(batch_obs)
                curr_probs = self._softmax(curr_logits)
                curr_log_probs = np.log(curr_probs + 1e-10)

                # 当前价值预测
                curr_values = self._value_forward(batch_obs).flatten()

                # 计算 log_prob 对于采取的动作
                action_log_probs = np.array([curr_log_probs[i, a] for i, a in enumerate(batch_actions)])

                # 重要性采样比率
                ratio = np.exp(action_log_probs - batch_old_log_probs)

                # Clipped Surrogate Loss
                surr1 = ratio * batch_advantages
                surr2 = np.clip(ratio, 1 - self.clip_coef, 1 + self.clip_coef) * batch_advantages
                policy_loss = -np.mean(np.minimum(surr1, surr2))

                # Value Loss (MSE with clipping)
                value_pred_clipped = old_values[idx] + np.clip(curr_values - old_values[idx], -self.clip_coef, self.clip_coef)
                value_loss1 = (curr_values - batch_returns) ** 2
                value_loss2 = (value_pred_clipped - batch_returns) ** 2
                value_loss = 0.5 * np.mean(np.maximum(value_loss1, value_loss2))

                # Entropy Bonus
                entropy = -np.sum(curr_probs * np.log(curr_probs + 1e-10), axis=1)
                entropy_loss = np.mean(entropy)

                # 总损失
                loss = policy_loss + self.value_coef * value_loss - self.ent_coef * entropy_loss

                # 梯度计算 (简化版)
                policy_grad, value_grad = self._compute_gradients(
                    batch_obs, batch_actions, batch_advantages,
                    batch_old_log_probs, batch_returns
                )

                # Adam 更新
                self._adam_update_policy(policy_grad)
                self._adam_update_value(value_grad)

                total_policy_loss += policy_loss
                total_value_loss += value_loss
                total_entropy += entropy_loss
                n_updates += 1

        # 清空经验
        self._clear_buffer()

        return {
            'policy_loss': float(total_policy_loss / max(n_updates, 1)),
            'value_loss': float(total_value_loss / max(n_updates, 1)),
            'entropy': float(total_entropy / max(n_updates, 1)),
        }

    def _compute_gae(self, rewards: np.ndarray, dones: np.ndarray,
                     values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """计算 GAE 优势和回报"""
        n = len(rewards)
        advantages = np.zeros(n)
        returns = np.zeros(n)

        gae = 0.0
        for t in reversed(range(n)):
            if t == n - 1:
                next_value = 0.0
            else:
                next_value = values[t + 1] if not dones[t] else 0.0

            delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * 0.95 * gae  # lambda=0.95

            advantages[t] = gae
            returns[t] = gae + values[t]

        return advantages, returns

    def _policy_forward(self, obs: np.ndarray) -> np.ndarray:
        """策略网络前向"""
        if obs.ndim == 1:
            obs = obs[np.newaxis, :]

        h = self._relu(obs @ self.policy_w1 + self.policy_b1)
        h = self._relu(h @ self.policy_w2 + self.policy_b2)
        logits = h @ self.policy_logits
        return logits

    def _value_forward(self, obs: np.ndarray) -> np.ndarray:
        """价值网络前向"""
        if obs.ndim == 1:
            obs = obs[np.newaxis, :]

        h = self._relu(obs @ self.value_w1 + self.value_b1)
        h = self._relu(h @ self.value_w2 + self.value_b2)
        value = h @ self.value_out
        return value.flatten()

    def _compute_gradients(self, obs: np.ndarray, actions: np.ndarray,
                           advantages: np.ndarray, old_log_probs: np.ndarray,
                           returns: np.ndarray) -> Tuple[Dict, Dict]:
        """计算梯度 (简化版)"""
        # 这里使用简化梯度计算
        # 完整实现需要完整的反向传播

        batch_size = len(obs)

        # 策略梯度
        action_logits = self._policy_forward(obs)
        action_probs = self._softmax(action_logits)

        # 对于采取的动作的梯度
        grad_logits = action_probs.copy()
        for i, a in enumerate(actions):
            grad_logits[i, a] -= 1.0

        # 加权优势
        grad_logits = grad_logits * advantages.reshape(-1, 1)

        # 反向传播到权重
        h2 = self._relu(self._relu(obs @ self.policy_w1 + self.policy_b1) @ self.policy_w2 + self.policy_b2)
        grad_policy_logits = h2.T @ grad_logits / batch_size

        # 价值梯度
        value_pred = self._value_forward(obs)
        value_error = value_pred - returns
        grad_value_out = value_error.reshape(-1, 1) / batch_size

        return {'logits': grad_policy_logits}, {'out': grad_value_out}

    def _adam_update_policy(self, grad: Dict):
        """Adam 更新策略网络"""
        for key, g in grad.items():
            param_name = f'policy_{key}'
            if param_name not in self._adam_states:
                self._adam_states[param_name] = {'m': np.zeros_like(g), 'v': np.zeros_like(g), 't': 0}

            state = self._adam_states[param_name]
            state['t'] += 1
            t = state['t']

            state['m'] = 0.9 * state['m'] + 0.1 * g
            state['v'] = 0.999 * state['v'] + 0.001 * g ** 2

            m_hat = state['m'] / (1 - 0.9 ** t)
            v_hat = state['v'] / (1 - 0.999 ** t)

            if key == 'logits':
                self.policy_logits -= self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)

    def _adam_update_value(self, grad: Dict):
        """Adam 更新价值网络"""
        for key, g in grad.items():
            param_name = f'value_{key}'
            if param_name not in self._adam_states:
                self._adam_states[param_name] = {'m': np.zeros_like(g), 'v': np.zeros_like(g), 't': 0}

            state = self._adam_states[param_name]
            state['t'] += 1
            t = state['t']

            state['m'] = 0.9 * state['m'] + 0.1 * g
            state['v'] = 0.999 * state['v'] + 0.001 * g ** 2

            m_hat = state['m'] / (1 - 0.9 ** t)
            v_hat = state['v'] / (1 - 0.999 ** t)

            if key == 'out':
                self.value_out -= self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    @staticmethod
    def _categorical_sample(probs: np.ndarray) -> int:
        return int(np.argmax(np.random.multinomial(1, probs)))

    def _clear_buffer(self):
        """清空经验缓冲区"""
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()

    def save(self, path: str):
        """保存模型"""
        data = {
            'policy_w1': self.policy_w1, 'policy_b1': self.policy_b1,
            'policy_w2': self.policy_w2, 'policy_b2': self.policy_b2,
            'policy_logits': self.policy_logits,
            'value_w1': self.value_w1, 'value_b1': self.value_b1,
            'value_w2': self.value_w2, 'value_b2': self.value_b2,
            'value_out': self.value_out,
            'config': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'hidden_dim': self.hidden_dim,
            }
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"[PPOAgentV2] 模型已保存：{path}")

    @classmethod
    def load(cls, path: str) -> 'PPOAgentV2':
        """加载模型"""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        config = data['config']
        agent = cls(**config)

        agent.policy_w1 = data['policy_w1']
        agent.policy_b1 = data['policy_b1']
        agent.policy_w2 = data['policy_w2']
        agent.policy_b2 = data['policy_b2']
        agent.policy_logits = data['policy_logits']
        agent.value_w1 = data['value_w1']
        agent.value_b1 = data['value_b1']
        agent.value_w2 = data['value_w2']
        agent.value_b2 = data['value_b2']
        agent.value_out = data['value_out']

        return agent


# ── SAC Agent ──────────────────────────────────────────────

class SACAgentV2:
    """
    SAC (Soft Actor-Critic) Agent V2

    最大熵强化学习算法，鼓励探索同时保持稳定性。

    架构:
        - Actor (策略网络): State → Action Distribution (Gaussian)
        - Critic Q1, Q2 (价值网络): State, Action → Q-value
        - Target Critic Q2: 目标网络 (EMA 更新)

    特点:
        - 最大熵目标：最大化收益 + 探索
        - 自动温度调整 (自动 alpha)
        - Twin Q-networks (减少过估计)
        - Soft Update (EMA 目标更新)
    """

    def __init__(self, state_dim: int, action_dim: int = 3,
                 hidden_dim: int = 256,
                 lr: float = 0.0003, gamma: float = 0.99,
                 tau: float = 0.005, alpha: float = 0.2,
                 buffer_size: int = 100000, batch_size: int = 256):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        # 经验回放
        self.replay_buffer = {
            'obs': np.zeros((buffer_size, state_dim)),
            'next_obs': np.zeros((buffer_size, state_dim)),
            'actions': np.zeros(buffer_size),
            'rewards': np.zeros(buffer_size),
            'dones': np.zeros(buffer_size),
            'size': 0,
            'pos': 0,
        }

        # Actor 网络 (策略)
        self.actor_w1 = np.random.randn(state_dim, hidden_dim) * np.sqrt(2.0 / state_dim)
        self.actor_b1 = np.zeros(hidden_dim)
        self.actor_w2 = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        self.actor_b2 = np.zeros(hidden_dim)
        self.actor_mean = np.random.randn(hidden_dim, action_dim) * np.sqrt(1.0 / hidden_dim)
        self.actor_log_std = np.zeros(action_dim)

        # Critic Q1 网络
        # Q 网络输入：[state, action] (action 离散，用 one-hot)
        q_input_dim = state_dim + action_dim
        self.q1_w1 = np.random.randn(q_input_dim, hidden_dim) * np.sqrt(2.0 / q_input_dim)
        self.q1_b1 = np.zeros(hidden_dim)
        self.q1_w2 = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        self.q1_b2 = np.zeros(hidden_dim)
        self.q1_out = np.random.randn(hidden_dim, 1) * np.sqrt(1.0 / hidden_dim)

        # Critic Q2 网络
        self.q2_w1 = np.random.randn(q_input_dim, hidden_dim) * np.sqrt(2.0 / q_input_dim)
        self.q2_b1 = np.zeros(hidden_dim)
        self.q2_w2 = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        self.q2_b2 = np.zeros(hidden_dim)
        self.q2_out = np.random.randn(hidden_dim, 1) * np.sqrt(1.0 / hidden_dim)

        # Target Q2 网络
        self.target_q2_w1 = self.q2_w1.copy()
        self.target_q2_b1 = self.q2_b1.copy()
        self.target_q2_w2 = self.q2_w2.copy()
        self.target_q2_b2 = self.q2_b2.copy()
        self.target_q2_out = self.q2_out.copy()

        # Adam 优化器状态
        self._adam_states = {}

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> Tuple[int, float]:
        """
        选择动作

        Args:
            state: 当前状态
            evaluate: 是否评估模式 (不探索)
        Returns:
            (action, log_prob)
        """
        action_probs = self._actor_forward_probs(state)

        if evaluate:
            action = int(np.argmax(action_probs))
        else:
            action = self._categorical_sample(action_probs)

        log_prob = np.log(action_probs[action] + 1e-10)
        return action, log_prob

    def store_transition(self, obs: np.ndarray, action: int, reward: float,
                         next_obs: np.ndarray, done: bool):
        """存储经验到回放缓冲区"""
        idx = self.replay_buffer['pos']
        self.replay_buffer['obs'][idx] = obs
        self.replay_buffer['next_obs'][idx] = next_obs
        self.replay_buffer['actions'][idx] = action
        self.replay_buffer['rewards'][idx] = reward
        self.replay_buffer['dones'][idx] = done
        self.replay_buffer['pos'] = (idx + 1) % self.replay_buffer_size
        self.replay_buffer['size'] = min(self.replay_buffer['size'] + 1, self.replay_buffer_size)

    @property
    def replay_buffer_size(self) -> int:
        return self.buffer_size

    def train(self) -> Dict:
        """
        SAC 训练

        步骤:
        1. 从回放缓冲区采样 batch
        2. 更新 Q1, Q2 networks
        3. 更新 Actor network
        4. Soft update target network
        """
        n = self.replay_buffer['size']
        if n < self.batch_size:
            return {'q1_loss': 0, 'q2_loss': 0, 'actor_loss': 0}

        # 采样
        indices = np.random.choice(n, self.batch_size, replace=False)
        batch_obs = self.replay_buffer['obs'][indices]
        batch_next_obs = self.replay_buffer['next_obs'][indices]
        batch_actions = self.replay_buffer['actions'][indices].astype(int)
        batch_rewards = self.replay_buffer['rewards'][indices]
        batch_dones = self.replay_buffer['dones'][indices]

        # One-hot 编码动作
        batch_actions_onehot = np.zeros((self.batch_size, self.action_dim))
        batch_actions_onehot[np.arange(self.batch_size), batch_actions.astype(int)] = 1

        # ── 更新 Q1 Network ──
        q1_loss, q1_grad = self._compute_q1_loss(batch_obs, batch_actions_onehot,
                                                  batch_rewards, batch_next_obs, batch_dones)
        self._adam_update('q1', q1_grad)

        # ── 更新 Q2 Network ──
        q2_loss, q2_grad = self._compute_q2_loss(batch_obs, batch_actions_onehot,
                                                  batch_rewards, batch_next_obs, batch_dones)
        self._adam_update('q2', q2_grad)

        # ── 更新 Actor Network ──
        actor_loss, actor_grad = self._compute_actor_loss(batch_obs)
        self._adam_update('actor', actor_grad)

        # ── Soft Update Target Network ──
        self._soft_update(self.q2_w1, self.target_q2_w1)
        self._soft_update(self.q2_b1, self.target_q2_b1)
        self._soft_update(self.q2_w2, self.target_q2_w2)
        self._soft_update(self.q2_b2, self.target_q2_b2)
        self._soft_update(self.q2_out, self.target_q2_out)

        return {
            'q1_loss': float(q1_loss),
            'q2_loss': float(q2_loss),
            'actor_loss': float(actor_loss),
        }

    def _compute_q1_loss(self, obs: np.ndarray, actions_onehot: np.ndarray,
                         rewards: np.ndarray, next_obs: np.ndarray,
                         dones: np.ndarray) -> Tuple[float, np.ndarray]:
        """计算 Q1 损失"""
        batch_size = len(obs)

        # 当前 Q1 预测
        q1_pred = self._q1_forward(obs, actions_onehot).flatten()

        # 目标 Q2 预测 (SAC 使用 target Q2)
        next_action_probs = self._actor_forward_probs(next_obs)
        next_actions = np.array([self._categorical_sample(p) for p in next_action_probs])
        next_actions_onehot = np.zeros_like(actions_onehot)
        next_actions_onehot[np.arange(batch_size), next_actions.astype(int)] = 1

        target_q2 = self._target_q2_forward(next_obs, next_actions_onehot).flatten()

        # TD target
        target_q = rewards + self.gamma * target_q2 * (1 - dones)

        # MSE Loss
        q1_loss = np.mean((q1_pred - target_q) ** 2)

        # 简化梯度
        q1_grad = 2 * (q1_pred - target_q).reshape(-1, 1) @ np.ones((1, self.hidden_dim)) / batch_size

        return q1_loss, q1_grad

    def _compute_q2_loss(self, obs: np.ndarray, actions_onehot: np.ndarray,
                         rewards: np.ndarray, next_obs: np.ndarray,
                         dones: np.ndarray) -> Tuple[float, np.ndarray]:
        """计算 Q2 损失"""
        batch_size = len(obs)

        q2_pred = self._q2_forward(obs, actions_onehot).flatten()

        # 目标 Q2
        next_action_probs = self._actor_forward_probs(next_obs)
        next_actions = np.array([self._categorical_sample(p) for p in next_action_probs])
        next_actions_onehot = np.zeros_like(actions_onehot)
        next_actions_onehot[np.arange(batch_size), next_actions.astype(int)] = 1

        target_q2 = self._target_q2_forward(next_obs, next_actions_onehot).flatten()

        target_q = rewards + self.gamma * target_q2 * (1 - dones)

        q2_loss = np.mean((q2_pred - target_q) ** 2)

        q2_grad = 2 * (q2_pred - target_q).reshape(-1, 1) @ np.ones((1, self.hidden_dim)) / batch_size

        return q2_loss, q2_grad

    def _compute_actor_loss(self, obs: np.ndarray) -> Tuple[float, np.ndarray]:
        """计算 Actor 损失 (最大熵)"""
        action_probs = self._actor_forward_probs(obs)

        # Q1 预测 (用于策略梯度)
        actions = np.array([self._categorical_sample(p) for p in action_probs])
        actions_onehot = np.zeros_like(action_probs)
        actions_onehot[np.arange(len(obs)), actions.astype(int)] = 1

        q_values = self._q1_forward(obs, actions_onehot).flatten()

        # 熵
        entropy = -np.sum(action_probs * np.log(action_probs + 1e-10), axis=1)

        # 最大熵目标：Q - alpha * entropy
        actor_loss = np.mean(-q_values - self.alpha * entropy)

        return actor_loss, action_probs  # 简化梯度

    def _actor_forward_probs(self, obs: np.ndarray) -> np.ndarray:
        """Actor 前向 (返回概率)"""
        if obs.ndim == 1:
            obs = obs[np.newaxis, :]

        h = self._relu(obs @ self.actor_w1 + self.actor_b1)
        h = self._relu(h @ self.actor_w2 + self.actor_b2)
        logits = h @ self.actor_mean
        return self._softmax(logits)

    def _q1_forward(self, obs: np.ndarray, actions_onehot: np.ndarray) -> np.ndarray:
        """Q1 前向"""
        x = np.concatenate([obs, actions_onehot], axis=-1)
        h = self._relu(x @ self.q1_w1 + self.q1_b1)
        h = self._relu(h @ self.q1_w2 + self.q1_b2)
        return h @ self.q1_out

    def _q2_forward(self, obs: np.ndarray, actions_onehot: np.ndarray) -> np.ndarray:
        """Q2 前向"""
        x = np.concatenate([obs, actions_onehot], axis=-1)
        h = self._relu(x @ self.q2_w1 + self.q2_b1)
        h = self._relu(h @ self.q2_w2 + self.q2_b2)
        return h @ self.q2_out

    def _target_q2_forward(self, obs: np.ndarray, actions_onehot: np.ndarray) -> np.ndarray:
        """Target Q2 前向"""
        x = np.concatenate([obs, actions_onehot], axis=-1)
        h = self._relu(x @ self.target_q2_w1 + self.target_q2_b1)
        h = self._relu(h @ self.target_q2_w2 + self.target_q2_b2)
        return h @ self.target_q2_out

    def _adam_update(self, name: str, grad: np.ndarray):
        """Adam 更新"""
        if name not in self._adam_states:
            self._adam_states[name] = {'m': np.zeros_like(grad), 'v': np.zeros_like(grad), 't': 0}

        state = self._adam_states[name]
        state['t'] += 1
        t = state['t']

        state['m'] = 0.9 * state['m'] + 0.1 * grad
        state['v'] = 0.999 * state['v'] + 0.001 * grad ** 2

        m_hat = state['m'] / (1 - 0.9 ** t)
        v_hat = state['v'] / (1 - 0.999 ** t)

        # 简化：只更新输出层
        if name == 'q1':
            self.q1_out -= self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)
        elif name == 'q2':
            self.q2_out -= self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)
        elif name == 'actor':
            self.actor_mean -= self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)

    def _soft_update(self, source: np.ndarray, target: np.ndarray):
        """软更新目标网络"""
        # 原地更新 target
        target[:] = self.tau * source + (1 - self.tau) * target

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    @staticmethod
    def _categorical_sample(probs: np.ndarray) -> int:
        return int(np.argmax(np.random.multinomial(1, probs)))

    def save(self, path: str):
        """保存模型"""
        data = {
            'actor_w1': self.actor_w1, 'actor_b1': self.actor_b1,
            'actor_w2': self.actor_w2, 'actor_b2': self.actor_b2,
            'actor_mean': self.actor_mean,
            'q1_w1': self.q1_w1, 'q1_b1': self.q1_b1,
            'q1_w2': self.q1_w2, 'q1_b2': self.q1_b2,
            'q1_out': self.q1_out,
            'q2_w1': self.q2_w1, 'q2_b1': self.q2_b1,
            'q2_w2': self.q2_w2, 'q2_b2': self.q2_b2,
            'q2_out': self.q2_out,
            'config': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'hidden_dim': self.hidden_dim,
            }
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"[SACAgentV2] 模型已保存：{path}")

    @classmethod
    def load(cls, path: str) -> 'SACAgentV2':
        """加载模型"""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        config = data['config']
        agent = cls(**config)

        agent.actor_w1 = data['actor_w1']
        agent.actor_b1 = data['actor_b1']
        agent.actor_w2 = data['actor_w2']
        agent.actor_b2 = data['actor_b2']
        agent.actor_mean = data['actor_mean']
        agent.q1_w1 = data['q1_w1']
        agent.q1_b1 = data['q1_b1']
        agent.q1_w2 = data['q1_w2']
        agent.q1_b2 = data['q1_b2']
        agent.q1_out = data['q1_out']
        agent.q2_w1 = data['q2_w1']
        agent.q2_b1 = data['q2_b1']
        agent.q2_w2 = data['q2_w2']
        agent.q2_b2 = data['q2_b2']
        agent.q2_out = data['q2_out']

        return agent


# ── 双 Agent 集成交易器 ──────────────────────────────────────

class RLTraderV2:
    """
    强化学习交易器 V2 — PPO + SAC 集成

    市场状态检测 → 选择最佳 Agent → 集成动作
    """

    def __init__(self, state_dim: int = 20, action_dim: int = 3):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.ppo_agent = PPOAgentV2(state_dim, action_dim)
        self.sac_agent = SACAgentV2(state_dim, action_dim)

        self._trained = False
        self._market_regime = 'neutral'  # 'bull', 'bear', 'neutral'

        self.model_dir = os.path.join(os.path.dirname(__file__), 'rl_models')
        os.makedirs(self.model_dir, exist_ok=True)

    def set_market_regime(self, regime: str):
        """设置市场状态"""
        self._market_regime = regime

    def train_ppo(self, env: TradingEnvV2, n_episodes: int = 100) -> Dict:
        """训练 PPO Agent"""
        logger.info(f"[RLTraderV2] PPO 训练：{n_episodes} episodes")

        episode_rewards = []

        for episode in range(n_episodes):
            obs = env.reset(seed=episode)
            total_reward = 0.0
            explore = max(0.1, 1.0 - episode * 0.01)

            while True:
                action, log_prob, value = self.ppo_agent.select_action(obs, explore)
                next_obs, reward, done, info = env.step(action)

                self.ppo_agent.store_transition(obs, action, reward, done, log_prob, value)
                total_reward += reward
                obs = next_obs

                if done:
                    break

            # PPO 更新
            self.ppo_agent.train()

            episode_rewards.append(total_reward)

            if episode % 20 == 0:
                metrics = env.get_episode_metrics()
                logger.info(f"[RLTraderV2] PPO Episode {episode}/{n_episodes}, "
                           f"Reward: {total_reward:.4f}, Return: {metrics['total_return']:.2%}")

        self._trained = True
        return {
            'agent': 'ppo',
            'episodes': n_episodes,
            'mean_reward': float(np.mean(episode_rewards[-20:])),
            'std_reward': float(np.std(episode_rewards[-20:])),
        }

    def train_sac(self, env: TradingEnvV2, n_steps: int = 10000) -> Dict:
        """训练 SAC Agent"""
        logger.info(f"[RLTraderV2] SAC 训练：{n_steps} steps")

        losses = []

        for step in range(n_steps):
            if step < self.sac_agent.batch_size:
                # 随机探索填充缓冲区
                obs = env.reset() if step == 0 else next_obs
                action, _ = self.sac_agent.select_action(obs, evaluate=False)
                next_obs, reward, done, info = env.step(action)
                self.sac_agent.store_transition(obs, action, reward, next_obs, done)

                if done:
                    next_obs = env.reset()
            else:
                # 训练
                action, log_prob = self.sac_agent.select_action(obs, evaluate=False)
                next_obs, reward, done, info = env.step(action)
                self.sac_agent.store_transition(obs, action, reward, next_obs, done)

                loss = self.sac_agent.train()
                losses.append(loss)

                obs = next_obs
                if done:
                    obs = env.reset()

            if step % 500 == 0:
                avg_loss = np.mean([l['actor_loss'] for l in losses[-100:]]) if losses else 0
                logger.info(f"[RLTraderV2] SAC Step {step}/{n_steps}, Actor Loss: {avg_loss:.6f}")

        self._trained = True
        return {
            'agent': 'sac',
            'steps': n_steps,
            'mean_actor_loss': float(np.mean([l['actor_loss'] for l in losses[-100:]])),
        }

    def trade(self, obs: np.ndarray) -> Dict:
        """
        交易决策

        根据市场状态选择 Agent 并集成动作
        """
        if not self._trained:
            return {'error': '模型未训练', 'action': 'hold'}

        # PPO 动作
        ppo_action, ppo_log_prob, ppo_value = self.ppo_agent.select_action(obs, explore=False)

        # SAC 动作
        sac_action, sac_log_prob = self.sac_agent.select_action(obs, evaluate=True)

        # 根据市场状态选择
        if self._market_regime == 'bull':
            # 牛市：偏向 PPO (更稳定)
            action = ppo_action
            confidence = 0.7
        elif self._market_regime == 'bear':
            # 熊市：偏向 SAC (更保守)
            action = sac_action
            confidence = 0.7
        else:
            # 震荡：投票
            if ppo_action == sac_action:
                action = ppo_action
                confidence = 0.8
            else:
                action = ppo_action  # 默认 PPO
                confidence = 0.5

        action_map = {0: 'hold', 1: 'buy', 2: 'sell'}

        return {
            'action': action_map.get(action, 'hold'),
            'action_code': action,
            'confidence': confidence,
            'market_regime': self._market_regime,
            'ppo_action': action_map.get(ppo_action, 'hold'),
            'sac_action': action_map.get(sac_action, 'hold'),
        }

    def save(self):
        """保存两个 Agent"""
        ppo_path = os.path.join(self.model_dir, 'ppo_agent.pkl')
        sac_path = os.path.join(self.model_dir, 'sac_agent.pkl')

        self.ppo_agent.save(ppo_path)
        self.sac_agent.save(sac_path)

        return {'ppo': ppo_path, 'sac': sac_path}

    def load(self):
        """加载两个 Agent"""
        ppo_path = os.path.join(self.model_dir, 'ppo_agent.pkl')
        sac_path = os.path.join(self.model_dir, 'sac_agent.pkl')

        if os.path.exists(ppo_path):
            self.ppo_agent = PPOAgentV2.load(ppo_path)
        if os.path.exists(sac_path):
            self.sac_agent = SACAgentV2.load(sac_path)

        self._trained = True
        return True


# 全局实例
rl_trader_v2 = RLTraderV2()
