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
                 reward_type: str = 'sharpe',
                 execution_model: str = 'simple',
                 limit_pct: float = 0.10):
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

        # P1 (2026-09-10, TradeMaster order_execution 思想翻译): 执行保真开关, 默认关
        #   'simple'  = 2026-07 旧行为逐字节不变 (任何非 'a_share' 值均安全降级)
        #   'a_share' = A股执行保真: 涨跌停拒单 + 成本保真 (买佣金0.03%+滑点 / 卖 +印花税0.1%; 复用 modules/execution/slippage)
        #   T+1 注记: 动作粒度=日, 买卖天然不同 bar → T+1 天然成立 (日内换手 env 才需显式 T+1)
        #   当前本 env 无活链实例方 (09-10 侦察); 升级不影响任何正在运行的链 (含 23:00 drl_agent 链)
        self.execution_model = execution_model
        self.limit_pct = float(limit_pct) if isinstance(limit_pct, (int, float)) else 0.10
        self._slippage_model = None
        if execution_model == 'a_share':
            try:
                from modules.execution.slippage import DynamicSlippageModel
                self._slippage_model = DynamicSlippageModel()
            except Exception as e:
                logger.warning(f"[TradingEnvV2] 滑点模型不可用 ({e}), 降级固定 transaction_cost")

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

        # 执行交易 (P1 2026-09-10: a_share = 涨跌停拒单 + T+1; 默认 'simple' 完全不进本分支)
        trade_size = 0.0
        rejected = None
        if self.execution_model == 'a_share':
            # 涨停拒买 (排板买不进) / 跌停拒卖 (排撤卖不出) — 与 walkforward :244/:269 BUY/SELL_REJECTED 同一语义
            # (2026-09-10 自查: T+1 在日频动作粒度天然成立, 写代码=装饰性死代码 → 不写)
            pct_chg = (current_price / prev_price - 1) if prev_price > 0 else 0.0
            if pct_chg >= self.limit_pct * 0.995 and action == 1:
                rejected = 'limit_up'
            elif pct_chg <= -self.limit_pct * 0.995 and action == 2:
                rejected = 'limit_down'

        if rejected is not None:
            target_position = self.position  # 拒单 = 无成交 (排板/排撤不出 = TradeMaster 撮合语义)
        elif action == 1:  # buy
            target_position = min(1.0, self.position + 0.2)
            trade_size = target_position - self.position
        elif action == 2:  # sell
            target_position = max(0.0, self.position - 0.2)
            trade_size = target_position - self.position
        else:  # hold
            target_position = self.position

        # 交易成本 (a_share: 买 佣金0.03%+滑点; 卖 佣金0.03%+印花税0.1%+滑点; 滑点 = DynamicSlippageModel)
        if abs(trade_size) > 0.01:
            if self.execution_model == 'a_share':
                trade_amount = abs(trade_size) * self.total_value
                slip = self.transaction_cost
                if self._slippage_model is not None:
                    win = self.prices[max(0, self._current_step - 20):self._current_step + 1]
                    if len(win) >= 5:
                        r20 = np.diff(win) / (np.abs(win[:-1]) + 1e-10)
                        vol_ann = float(np.std(r20)) * float(np.sqrt(252.0))
                    else:
                        vol_ann = 0.005
                    slip = self._slippage_model.estimate(
                        float(current_price), 0, volatility=vol_ann)['slippage_ratio']
                if trade_size > 0:
                    cost = trade_amount * (0.0003 + slip)
                else:
                    cost = trade_amount * (0.0003 + 0.001 + slip)
            else:
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
        reward = self._calculate_reward(portfolio_return, drawdown, action, rejected=rejected)

        info = {
            'total_value': self.total_value,
            'position': self.position,
            'cash': self.cash,
            'return': portfolio_return,
            'drawdown': drawdown,
            'reward': reward,
            'price': current_price,
            'rejected': rejected,
        }

        return self._get_observation(), reward, done, info

    def _calculate_reward(self, portfolio_return: float, drawdown: float, action: int,
                          rejected: Optional[str] = None) -> float:
        """计算奖励"""
        reward = portfolio_return

        # 交易成本惩罚 (P1: 拒单步无成交, 不计惩罚)
        if rejected is None and abs(action - 1) > 0.5:  # 非 hold 操作
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
        action_probs = self._softmax(action_logits).squeeze(0)

        if explore:
            # 采样
            action = self._categorical_sample(action_probs)
        else:
            # Greedy
            action = int(np.argmax(action_probs))

        log_prob = np.log(action_probs[action] + 1e-10)
        value = float(self._value_forward(state).squeeze())

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

        # 价值梯度 (正确反向传播)
        value_pred = self._value_forward(obs)
        # value_pred 是 flatten 后的 (B,), 需要恢复 (B, 1) 做 backprop
        value_pred_2d = value_pred.reshape(-1, 1)

        # 对 value_out 的梯度: d_loss/d_value_out = (value_pred - returns) * h
        value_error = (value_pred_2d - returns.reshape(-1, 1)) / batch_size  # (B, 1)

        # 重新计算前向中间值
        h1 = self._relu(obs @ self.value_w1 + self.value_b1)  # (B, hidden)
        h2 = self._relu(h1 @ self.value_w2 + self.value_b2)  # (B, hidden)

        # value_out 梯度: (hidden, 1)
        grad_value_out = h2.T @ value_error

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


# ── Attention-enhanced PPO Agent (Phase 3 增强) ──────────────────────

class AttentionPPOAgent:
    """
    带注意力机制的 PPO Agent

    新增:
    1. Self-Attention 层 — 捕捉长期依赖
    2. Regime-adaptive Action — 根据市场状态调整动作空间
    3. Online Fine-tuning — 在线微调
    """

    def __init__(self, state_dim: int, action_dim: int = 3,
                 hidden_dim: int = 128, attention_heads: int = 4,
                 lr: float = 0.0003, gamma: float = 0.99,
                 clip_coef: float = 0.2, ent_coef: float = 0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.attention_heads = attention_heads
        self.lr = lr
        self.gamma = gamma
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef

        # 经验存储
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

        # 策略网络 (带注意力)
        self.policy_w1 = np.random.randn(state_dim, hidden_dim) * np.sqrt(2.0 / state_dim)
        self.policy_b1 = np.zeros(hidden_dim)
        self.policy_w2 = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        self.policy_b2 = np.zeros(hidden_dim)

        # 注意力层
        self.attention_w = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(1.0 / hidden_dim)
        self.attention_b = np.zeros(hidden_dim)
        self.attention_out = np.random.randn(hidden_dim, attention_heads) * np.sqrt(1.0 / hidden_dim)

        # 策略输出
        self.policy_logits = np.random.randn(hidden_dim, action_dim) * np.sqrt(1.0 / hidden_dim)

        # 价值网络
        self.value_w1 = np.random.randn(state_dim, hidden_dim) * np.sqrt(2.0 / state_dim)
        self.value_b1 = np.zeros(hidden_dim)
        self.value_w2 = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        self.value_b2 = np.zeros(hidden_dim)
        self.value_out = np.random.randn(hidden_dim, 1) * np.sqrt(1.0 / hidden_dim)

        self._adam_states = {}
        self._trained = False

    def select_action(self, state: np.ndarray, explore: bool = True,
                      regime: str = 'neutral') -> Tuple[int, float, float]:
        """
        选择动作 (带 regime-adaptive 调整)
        """
        action_logits = self._policy_forward(state)

        # Regime-adaptive 调整
        if regime == 'bull':
            action_logits[:, 1] += 0.1  # 偏向 buy
        elif regime == 'bear':
            action_logits[:, 2] += 0.1  # 偏向 sell

        action_probs = self._softmax(action_logits).squeeze(0)

        if explore:
            action = self._categorical_sample(action_probs)
        else:
            action = int(np.argmax(action_probs))

        log_prob = np.log(action_probs[action] + 1e-10)
        value = float(self._value_forward(state).squeeze())
        return action, log_prob, value

    def store_transition(self, obs: np.ndarray, action: int, reward: float,
                         done: bool, log_prob: float, value: float):
        self.observations.append(obs.copy())
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def train(self) -> Dict:
        n_samples = len(self.rewards)
        if n_samples < 10:
            return {'loss': 0.0}

        obs_array = np.array(self.observations)
        actions = np.array(self.actions)
        rewards = np.array(self.rewards)
        dones = np.array(self.dones)
        old_log_probs = np.array(self.log_probs)
        old_values = np.array(self.values)

        advantages, returns = self._compute_gae(rewards, dones, old_values)
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-10)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for epoch in range(10):
            indices = np.random.permutation(n_samples)
            for start in range(0, n_samples, 64):
                end = min(start + 64, n_samples)
                idx = indices[start:end]
                batch_size = len(idx)

                batch_obs = obs_array[idx]
                batch_actions = actions[idx]
                batch_advantages = advantages[idx]
                batch_returns = returns[idx]
                batch_old_log_probs = old_log_probs[idx]

                curr_logits = self._policy_forward(batch_obs)
                curr_probs = self._softmax(curr_logits)
                curr_log_probs = np.log(curr_probs + 1e-10)

                action_log_probs = np.array([curr_log_probs[i, a] for i, a in enumerate(batch_actions)])
                ratio = np.exp(action_log_probs - batch_old_log_probs)

                surr1 = ratio * batch_advantages
                surr2 = np.clip(ratio, 1 - self.clip_coef, 1 + self.clip_coef) * batch_advantages
                policy_loss = -np.mean(np.minimum(surr1, surr2))

                value_pred = self._value_forward(batch_obs).flatten()
                value_pred_clipped = old_values[idx] + np.clip(value_pred - old_values[idx], -self.clip_coef, self.clip_coef)
                value_loss = 0.5 * np.mean(np.maximum((value_pred - batch_returns) ** 2,
                                                       (value_pred_clipped - batch_returns) ** 2))

                entropy = -np.sum(curr_probs * np.log(curr_probs + 1e-10), axis=1)
                entropy_loss = np.mean(entropy)

                loss = policy_loss + 0.5 * value_loss - self.ent_coef * entropy_loss
                total_policy_loss += policy_loss
                total_value_loss += value_loss
                total_entropy += entropy_loss
                n_updates += 1

        self._clear_buffer()
        self._trained = True
        return {
            'policy_loss': float(total_policy_loss / max(n_updates, 1)),
            'value_loss': float(total_value_loss / max(n_updates, 1)),
            'entropy': float(total_entropy / max(n_updates, 1)),
        }

    def online_fine_tune(self, env: 'TradingEnvV2', n_episodes: int = 10) -> Dict:
        """在线微调"""
        episode_rewards = []
        for _ in range(n_episodes):
            obs = env.reset()
            total_reward = 0.0
            explore = 0.1
            while True:
                action, log_prob, value = self.select_action(obs, explore=explore)
                obs, reward, done, _ = env.step(action)
                self.store_transition(obs, action, reward, done, log_prob, value)
                total_reward += reward
                if done:
                    break
            self.train()
            episode_rewards.append(total_reward)
        return {
            'mean_reward': float(np.mean(episode_rewards)),
            'std_reward': float(np.std(episode_rewards)),
            'episodes': n_episodes,
        }

    def _policy_forward(self, obs: np.ndarray) -> np.ndarray:
        if obs.ndim == 1:
            obs = obs[np.newaxis, :]
        h = self._relu(obs @ self.policy_w1 + self.policy_b1)
        h = self._relu(h @ self.policy_w2 + self.policy_b2)
        # 注意力机制
        attention_input = self._relu(h @ self.attention_w + self.attention_b)
        attention_weights = self._softmax(attention_input @ self.attention_out)
        h = h * attention_weights.mean(axis=1, keepdims=True)
        return h @ self.policy_logits

    def _value_forward(self, obs: np.ndarray) -> np.ndarray:
        if obs.ndim == 1:
            obs = obs[np.newaxis, :]
        h = self._relu(obs @ self.value_w1 + self.value_b1)
        h = self._relu(h @ self.value_w2 + self.value_b2)
        return h @ self.value_out

    def _compute_gae(self, rewards, dones, values):
        n = len(rewards)
        advantages = np.zeros(n)
        returns = np.zeros(n)
        gae = 0.0
        for t in reversed(range(n)):
            next_value = 0.0 if t == n - 1 else (values[t + 1] if not dones[t] else 0.0)
            delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * 0.95 * gae
            advantages[t] = gae
            returns[t] = gae + values[t]
        return advantages, returns

    @staticmethod
    def _relu(x): return np.maximum(0, x)

    @staticmethod
    def _softmax(x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    @staticmethod
    def _categorical_sample(probs):
        return int(np.argmax(np.random.multinomial(1, probs)))

    def _clear_buffer(self):
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()

    def save(self, path: str):
        data = {
            'policy_w1': self.policy_w1, 'policy_b1': self.policy_b1,
            'policy_w2': self.policy_w2, 'policy_b2': self.policy_b2,
            'policy_logits': self.policy_logits,
            'attention_w': self.attention_w, 'attention_b': self.attention_b,
            'attention_out': self.attention_out,
            'value_w1': self.value_w1, 'value_b1': self.value_b1,
            'value_w2': self.value_w2, 'value_b2': self.value_b2,
            'value_out': self.value_out,
            'config': {'state_dim': self.state_dim, 'action_dim': self.action_dim,
                       'hidden_dim': self.hidden_dim, 'attention_heads': self.attention_heads},
            'trained': self._trained,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"[AttentionPPOAgent] 模型已保存：{path}")

    @classmethod
    def load(cls, path: str) -> 'AttentionPPOAgent':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        config = data['config']
        agent = cls(**config)
        for key in ['policy_w1', 'policy_b1', 'policy_w2', 'policy_b2',
                     'policy_logits', 'attention_w', 'attention_b', 'attention_out',
                     'value_w1', 'value_b1', 'value_w2', 'value_b2', 'value_out']:
            setattr(agent, key, data[key])
        agent._trained = data.get('trained', False)
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
        action_probs = self._actor_forward_probs(state).squeeze(0)

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
    强化学习交易器 V2 — PPO + SAC + AttentionPPO 集成

    市场状态检测 → 选择最佳 Agent → 集成动作
    Phase 3 增强:
    1. AttentionPPOAgent — 带注意力机制的 PPO
    2. Regime-adaptive Action — 根据市场状态调整动作空间
    3. Online Fine-tuning — 在线微调
    """

    def __init__(self, state_dim: int = 20, action_dim: int = 3,
                 sac_as_default: bool = True):
        """
        初始化 RLTraderV2

        Args:
            state_dim: 状态维度
            action_dim: 动作维度 (3 = hold/buy/sell)
            sac_as_default: 是否使用 SAC 作为默认主 Agent (默认 True)
                - True:  SAC 为主力，PPO 为辅助，AttentionPPO 为趋势增强
                - False: 传统 PPO 为主，SAC 为探索增强
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.sac_as_default = sac_as_default

        self.ppo_agent = PPOAgentV2(state_dim, action_dim)
        self.sac_agent = SACAgentV2(state_dim, action_dim)
        self.attention_ppo = AttentionPPOAgent(state_dim, action_dim)

        self._trained = False
        self._market_regime = 'neutral'  # 'bull', 'bear', 'neutral'

        self.model_dir = os.path.join(os.path.dirname(__file__), 'rl_models')
        os.makedirs(self.model_dir, exist_ok=True)

    def set_market_regime(self, regime: str):
        """设置市场状态"""
        self._market_regime = regime

    def train_ppo(self, env: TradingEnvV2, n_episodes: int = 100, timeout: float = None) -> Dict:
        """训练 PPO Agent (timeout = 墙钟 deadline 时间戳, 默认 None = 无上限兼容旧调用)"""
        import time as _t
        logger.info(f"[RLTraderV2] PPO 训练：{n_episodes} episodes")

        episode_rewards = []

        for episode in range(n_episodes):
            # 2026-09-10 链修: 墙钟预算 guard (长循环无内部 timeout = 断链源, hyperparam 2400s 事故同族)
            if timeout is not None and _t.perf_counter() > timeout:
                logger.warning(f"[RLTraderV2] PPO 训练时间预算耗尽, 停在 episode {episode+1}")
                break

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
            'episodes': len(episode_rewards),
            # 空 replay guard (2026-09-10): NaN 会随 jsonify 泄漏成非法 JSON
            'mean_reward': float(np.mean(episode_rewards[-20:])) if episode_rewards else 0.0,
            'std_reward': float(np.std(episode_rewards[-20:])) if episode_rewards else 0.0,
        }

    def train(self, X, y=None, closes=None, n_episodes: int = 30,
              timeout: float = 240.0, execution_model: str = 'simple') -> Dict:
        """
        训练链入口 (2026-09-10 死链修复 — POST /api/rl/train 恒 200+success:false 史)

        断链修复: 此方法原本不存在, dl_routes.api_rl_train 直接 .train(X, y)
        → AttributeError → except 吞 → 端点从未活过 (0.285s 实测锁)。本 adapter
        把 (X, closes) 翻译成 TradingEnvV2 (P1 执行保真 env) 并驱动 train_ppo,
        断链 = train_ppo 原链身 (select_action 离散动作 × env.step 兼容已核)。

        断链三连 (2026-09-10 全修):
          ① .train 不存在 → 本方法
          ② 形状 (endpoint X 12D → env obs 17 维 ≠ agent 输入 20) → features pad 至 state_dim-5
          ③ 前端 toFixed (dl_dashboard.html 读 training_result.mean_reward) → endpoint 回包增键

        Args:
            X: (n,F) state 特征矩阵 (endpoint compute_features = 12D)
            y: 忽略 (env 不注入 label, 仅保 endpoint 签名兼容)
            closes: (m,) 收盘价链 (取后 n 条与 features 对齐; 缺省 = 恒价链降级)
            n_episodes: PPO episode 数 (默认 30 = endpoint 默认)
            timeout: 墙钟预算 (秒, episode 级 guard 防单点无限阻塞)
            execution_model: 'simple'|'a_share' → TradingEnvV2 执行保真开关 (默认关)

        Returns:
            统计 dict (agent/episodes/mean_reward/elapsed_s/…), 降级返回 {'agent','skipped'|'error'} (不抛)
        """
        import time as _t
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or len(X) < 30:
            logger.warning(f"[RLTraderV2] 训练数据不足: X shape={X.shape}")
            return {'agent': 'ppo', 'episodes': 0,
                    'skipped': f'X shape insufficient: {X.shape}'}

        # ② 形状适配: 对齐 ppo_agent 实际输入维 = state_dim-5 (obs=F+5 必须对齐)
        #    服务链 (app.py:283 load() 替换 ppo_agent) 消费者 =17 (07-15 pkl 权重) → X 12D 原生直通;
        #    singleton/默认链 =20 → pad 15。盲用 self.state_dim(恒20) = matmul 17≠20 crash (09-11 链尾实证)
        n = len(X)
        f_need = getattr(self.ppo_agent, 'state_dim', self.state_dim) - 5
        if X.shape[1] < f_need:
            X = np.pad(X, ((0, 0), (0, f_need - X.shape[1])))
        elif X.shape[1] > f_need:
            X = X[:, :f_need]

        # prices 对齐: 后 n 条与 features[i] 配对 (缺省恒价 = 降级, 非崩)
        if closes is not None:
            c = np.asarray(closes, dtype=np.float64)
            if len(c) >= n:
                prices = c[-n:]
            else:
                pad = n - len(c)
                prices = np.pad(c, (0, pad), constant_values=float(c[-1]) if len(c) else 100.0)
        else:
            prices = np.full(n, 100.0, dtype=np.float64)

        t0 = _t.perf_counter()
        deadline = t0 + max(0.05, float(timeout))  # 0 = 全预算耗尽 (unit test guard 验证路径)
        try:
            env = TradingEnvV2(prices, X, execution_model=execution_model)
            result = self.train_ppo(env, n_episodes=n_episodes, timeout=deadline)
            result['elapsed_s'] = round(_t.perf_counter() - t0, 1)
            return result
        except Exception as e:
            logger.error(f"[RLTraderV2] 训练链异常: {e}")
            return {'agent': 'ppo', 'episodes': 0, 'error': str(e)}

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
        交易决策 (Phase 3 增强 — SAC 默认主 Agent)

        架构:
            SAC 作为默认主 Agent (最大熵探索 + 稳定性)
            PPO 作为辅助 (确定性策略)
            AttentionPPO 作为趋势增强 (牛市中权重提升)

        权重配置 (sac_as_default=True):
            regime     | SAC   | PPO   | AttentionPPO
            -----------|-------|-------|-------------
            bullish    | 0.40  | 0.20  | 0.40
            bearish    | 0.60  | 0.25  | 0.15
            sideways   | 0.50  | 0.30  | 0.20

        权重配置 (sac_as_default=False, 向后兼容):
            regime     | PPO   | SAC   | AttentionPPO
            -----------|-------|-------|-------------
            bullish    | 0.40  | 0.20  | 0.40
            bearish    | 0.25  | 0.60  | 0.15
            sideways   | 0.30  | 0.30  | 0.20
        """
        if not self._trained:
            return {'error': '模型未训练', 'action': 'hold'}

        # 为每个 Agent 单独对齐观测维度
        def _align(obs, expected_dim):
            if len(obs) != expected_dim:
                if len(obs) > expected_dim:
                    return obs[:expected_dim]
                else:
                    return np.pad(obs, (0, expected_dim - len(obs)), mode='constant')
            return obs

        obs_ppo = _align(obs, self.ppo_agent.state_dim)
        obs_sac = _align(obs, self.sac_agent.state_dim)
        obs_att = _align(obs, self.attention_ppo.state_dim)

        # 各 Agent 决策
        ppo_action, ppo_log_prob, ppo_value = self.ppo_agent.select_action(obs_ppo, explore=False)
        sac_action, sac_log_prob = self.sac_agent.select_action(obs_sac, evaluate=True)
        att_action, att_log_prob, att_value = self.attention_ppo.select_action(
            obs_att, explore=False, regime=self._market_regime
        )

        # 根据 sac_as_default 和市场状态获取权重
        if self.sac_as_default:
            regime_weights = {
                'bull':   {'sac': 0.40, 'ppo': 0.20, 'att': 0.40},
                'bear':   {'sac': 0.60, 'ppo': 0.25, 'att': 0.15},
                'sideways': {'sac': 0.50, 'ppo': 0.30, 'att': 0.20},
                'neutral': {'sac': 0.50, 'ppo': 0.30, 'att': 0.20},
            }
        else:
            # 向后兼容: PPO 为主
            regime_weights = {
                'bull':   {'ppo': 0.40, 'sac': 0.20, 'att': 0.40},
                'bear':   {'ppo': 0.25, 'sac': 0.60, 'att': 0.15},
                'sideways': {'ppo': 0.30, 'sac': 0.30, 'att': 0.20},
                'neutral': {'ppo': 0.30, 'sac': 0.30, 'att': 0.20},
            }

        regime_key = self._market_regime if self._market_regime in regime_weights else 'neutral'
        weights = regime_weights[regime_key]

        # 加权投票 (one-hot 向量累加)
        action_map_inv = {'hold': 0, 'buy': 1, 'sell': 2}
        votes = {'ppo': ppo_action, 'sac': sac_action, 'att': att_action}

        weighted_scores = {0: 0.0, 1: 0.0, 2: 0.0}  # hold/buy/sell 总分
        for agent_name, action_code in votes.items():
            weighted_scores[action_code] += weights[agent_name]

        # 多样性奖励: 如果多个 Agent 意见一致，额外加分
        unique_actions = set(votes.values())
        if len(unique_actions) == 1:
            # 全票一致: +0.15 到共识动作
            consensus_action = list(unique_actions)[0]
            weighted_scores[consensus_action] += 0.15
        elif len(unique_actions) == 2:
            # 两票一致: +0.08 到多数动作
            from collections import Counter
            vote_counts = Counter(votes.values())
            if vote_counts.most_common(1)[0][1] >= 2:
                majority_action = vote_counts.most_common(1)[0][0]
                weighted_scores[majority_action] += 0.08

        # 选择得分最高的动作
        action = max(weighted_scores, key=weighted_scores.get)

        # 置信度: 基于得分分布计算
        total_score = sum(weighted_scores.values())
        if total_score > 0:
            confidence = weighted_scores[action] / total_score
        else:
            confidence = 0.33

        # 根据 regime 调整置信度上限
        regime_confidence_cap = {'bull': 0.95, 'bear': 0.90, 'sideways': 0.85, 'neutral': 0.85}
        confidence = min(confidence, regime_confidence_cap.get(regime_key, 0.85))

        action_map = {0: 'hold', 1: 'buy', 2: 'sell'}

        # 统计投票
        from collections import Counter
        vote_counts = Counter(votes.values())

        return {
            'action': action_map.get(action, 'hold'),
            'action_code': action,
            'confidence': round(confidence, 3),
            'market_regime': self._market_regime,
            'agent_weights': weights,
            'weighted_scores': {k: round(v, 4) for k, v in weighted_scores.items()},
            'ppo_action': action_map.get(ppo_action, 'hold'),
            'sac_action': action_map.get(sac_action, 'hold'),
            'attention_ppo_action': action_map.get(att_action, 'hold'),
            'agent_consensus': {action_map.get(k, k): v for k, v in vote_counts.items()},
            'sac_default': self.sac_as_default,
        }

    def online_fine_tune(self, env: TradingEnvV2, n_episodes: int = 20) -> Dict:
        """在线微调 (Phase 3 增强)"""
        logger.info(f"[RLTraderV2] 在线微调: {n_episodes} episodes")
        result = self.attention_ppo.online_fine_tune(env, n_episodes)
        return {
            'agent': 'attention_ppo',
            **result,
            'market_regime': self._market_regime,
        }

    def save(self):
        """保存三个 Agent (Phase 3 增强)"""
        ppo_path = os.path.join(self.model_dir, 'ppo_agent.pkl')
        sac_path = os.path.join(self.model_dir, 'sac_agent.pkl')
        att_path = os.path.join(self.model_dir, 'attention_ppo_agent.pkl')

        self.ppo_agent.save(ppo_path)
        self.sac_agent.save(sac_path)
        self.attention_ppo.save(att_path)

        return {'ppo': ppo_path, 'sac': sac_path, 'attention_ppo': att_path}

    def load(self):
        """加载三个 Agent (Phase 3 增强)"""
        ppo_path = os.path.join(self.model_dir, 'ppo_agent.pkl')
        sac_path = os.path.join(self.model_dir, 'sac_agent.pkl')
        att_path = os.path.join(self.model_dir, 'attention_ppo_agent.pkl')

        loaded = False
        if os.path.exists(ppo_path):
            self.ppo_agent = PPOAgentV2.load(ppo_path)
            loaded = True
        if os.path.exists(sac_path):
            self.sac_agent = SACAgentV2.load(sac_path)
            loaded = True
        if os.path.exists(att_path):
            self.attention_ppo = AttentionPPOAgent.load(att_path)
            loaded = True

        self._trained = loaded
        return loaded


# 全局实例 (SAC 作为默认主 Agent)
rl_trader_v2 = RLTraderV2(sac_as_default=True)
