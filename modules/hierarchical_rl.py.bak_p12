#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Hierarchical RL — 分层强化学习 (2026 SOTA)

高层策略 (仓位管理) + 低层策略 (执行优化):

架构:
    ┌─────────────────────────────────────────────────────┐
    │                 High-Level Policy (周级)              │
    │  输入: 市场状态 (Regime, 波动率, 趋势, 风险偏好)   │
    │  输出: 目标仓位 (满仓/3/2/空仓)                    │
    └────────────────────┬────────────────────────────────┘
                       │
                       ▼
    ┌─────────────────────────────────────────────────────┐
    │                  Low-Level Policy (日内)            │
    │  输入: 订单簿, 情绪, 技术指标, 执行成本               │
    │  输出: 动作 (市价单/限价单/止损单/观望)               │
    └─────────────────────────────────────────────────────┘

优势:
- 更接近真实交易决策结构 (先决定仓位, 再决定如何执行)
- 高层策略捕捉长期趋势, 低层策略优化执行
- 减少决策空间, 提升样本效率

参考:
- "HIRO: Data-Efficient Hierarchical Reinforcement Learning" (2018)
- "Options of Options" (2019)
- "Trading with the Intelligent Assistant" (FinRL 2023)
"""

import os
import json
import time
import copy
import threading
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque

from modules.logger import logger

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("[HierarchicalRL] PyTorch 未安装")


# ── 高层策略 (仓位管理) ─────────────────────────────────────────────────────

@dataclass
class HighLevelState:
    """高层状态 (市场状态)"""
    market_regime: str = "sideways"  # bullish/bearish/sideways/volatile
    volatility: float = 0.0  # 波动率 (ATR/STD)
    trend: float = 0.0  # 趋势强度 (-1 ~ 1)
    risk_appetite: float = 0.5  # 风险偏好 (0 ~ 1)
    portfolio_value: float = 0.0  # 组合价值
    cash_ratio: float = 1.0  # 现金比例


@dataclass
class HighLevelAction:
    """高层动作 (目标仓位)"""
    target_position: int = 2  # 0=空仓, 1=1/3仓, 2=2/3仓, 3=满仓
    confidence: float = 0.5  # 置信度 (0 ~ 1)


HIGH_LEVEL_POSITION_MAP = {
    0: "空仓",
    1: "1/3仓",
    2: "2/3仓",
    3: "满仓",
}


class HighLevelPolicy(nn.Module):
    """
    高层策略网络 (仓位管理)

    输入: 市场状态向量 (regime + volatility + trend + risk + ...)
    输出: 仓位动作 (4 选 1)
    """

    def __init__(self, state_dim: int = 64, hidden_dim: int = 128, n_actions: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, n_actions),
        )

        self.n_actions = n_actions

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (batch, state_dim) 或 (state_dim,)
        Returns:
            logits: (n_actions,)
        """
        return self.net(state)


# ── 低层策略 (执行优化) ───────────────────────────────────────────────────────

@dataclass
class LowLevelState:
    """低层状态 (日内状态)"""
    position: int = 2  # 当前仓位 (0-3)
    price: float = 0.0  # 当前价格
    bid_ask_spread: float = 0.0  # 买卖价差
    volume_imbalance: float = 0.0  # 成交量不平衡
    momentum: float = 0.0  # 动量
    execution_cost: float = 0.0  # 执行成本估计


@dataclass
class LowLevelAction:
    """低层动作 (执行动作)"""
    action_type: int = 1  # 0=市价单, 1=限价单, 2=止损单, 3=观望
    size: float = 0.0  # 数量比例 (0-1)
    price_limit: float = 0.0  # 限价 (用于限价单)


LOW_LEVEL_ACTION_MAP = {
    0: "市价单",
    1: "限价单",
    2: "止损单",
    3: "观望",
}


class LowLevelPolicy(nn.Module):
    """
    低层策略网络 (执行优化)

    输入: 日内状态向量 (position + price + spread + momentum + ...)
    输出: 执行动作 (4 选 1 + 数量)
    """

    def __init__(self, state_dim: int = 64, hidden_dim: int = 128, n_actions: int = 4):
        super().__init__()
        self.action_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, n_actions),
        )
        self.size_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),  # 数量 (0-1)
            nn.Sigmoid(),
        )

        self.n_actions = n_actions

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            state: (batch, state_dim) 或 (state_dim,)
        Returns:
            action_logits: (n_actions,)
            action_size: (1,) 或 (batch, 1)
        """
        action_logits = self.action_net(state)
        action_size = self.size_net(state)
        return action_logits, action_size


# ── 分层强化学习智能体 ─────────────────────────────────────────────────────────

class HierarchicalRLAgent:
    """
    分层强化学习智能体

    整合高层策略 (仓位管理) 和低层策略 (执行优化)

    训练流程:
    1. 训练低层策略 (高频率, 基于执行反馈)
    2. 训练高层策略 (低频率, 基于收益回测)

    使用方式:
        agent = HierarchicalRLAgent()
        position = agent.decide_position(high_state)  # 高层决策
        action = agent.decide_action(low_state)  # 低层决策
    """

    def __init__(
        self,
        high_state_dim: int = 9,  # 5 features + 4 regime one-hot
        low_state_dim: int = 6,  # 6 features
        high_hidden: int = 64,
        low_hidden: int = 64,
        n_high_actions: int = 4,
        n_low_actions: int = 4,
        lr_high: float = 1e-3,
        lr_low: float = 1e-3,
    ):
        if not HAS_TORCH:
            raise ImportError("[HierarchicalRL] PyTorch 未安装")

        self.high_state_dim = high_state_dim
        self.low_state_dim = low_state_dim
        self.n_high_actions = n_high_actions
        self.n_low_actions = n_low_actions

        # 高层策略 (仓位管理)
        self.high_policy = HighLevelPolicy(
            state_dim=high_state_dim,
            hidden_dim=high_hidden,
            n_actions=n_high_actions,
        )
        self.high_optimizer = torch.optim.Adam(self.high_policy.parameters(), lr=lr_high)

        # 低层策略 (执行优化)
        self.low_policy = LowLevelPolicy(
            state_dim=low_state_dim,
            hidden_dim=low_hidden,
            n_actions=n_low_actions,
        )
        self.low_optimizer = torch.optim.Adam(self.low_policy.parameters(), lr=lr_low)

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else
            'mps' if torch.backends.mps.is_available() else 'cpu'
        )

        self.high_policy.to(self.device)
        self.low_policy.to(self.device)

        # 经验回放
        self.high_replay = deque(maxlen=10000)
        self.low_replay = deque(maxlen=50000)

        self.trained = False
        self.model_dir = os.path.join(os.path.dirname(__file__), 'rl_models')
        os.makedirs(self.model_dir, exist_ok=True)

    def decide_position(self, state: HighLevelState) -> HighLevelAction:
        """
        高层决策 (仓位管理)

        Args:
            state: HighLevelState

        Returns:
            HighLevelAction: 目标仓位
        """
        self.high_policy.eval()

        # 构建状态向量
        state_vec = self._build_high_state(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.high_policy(state_vec)
            probs = torch.softmax(logits, dim=-1)
            action = torch.argmax(probs, dim=-1).item()
            confidence = probs[0, action].item()

        return HighLevelAction(
            target_position=action,
            confidence=confidence,
        )

    def decide_action(self, state: LowLevelState) -> LowLevelAction:
        """
        低层决策 (执行优化)

        Args:
            state: LowLevelState

        Returns:
            LowLevelAction: 执行动作
        """
        self.low_policy.eval()

        # 构建状态向量
        state_vec = self._build_low_state(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits, size = self.low_policy(state_vec)
            probs = torch.softmax(logits, dim=-1)
            action = torch.argmax(probs, dim=-1).item()
            action_size = size.item()

        return LowLevelAction(
            action_type=action,
            size=action_size,
            price_limit=state.price * (1.0 + (0.001 if action == 1 else -0.001)),
        )

    def _build_high_state(self, state: HighLevelState) -> torch.Tensor:
        """构建高层状态向量"""
        regime_map = {'bullish': [1, 0, 0, 0], 'bearish': [0, 1, 0, 0],
                       'sideways': [0, 0, 1, 0], 'volatile': [0, 0, 0, 1]}
        regime_vec = regime_map.get(state.market_regime, [0, 0, 0, 1])

        vec = [
            state.volatility,
            state.trend,
            state.risk_appetite,
            state.cash_ratio,
            state.portfolio_value / 1e6,  # 归一化
        ] + regime_vec

        return torch.tensor(vec, dtype=torch.float32)

    def _build_low_state(self, state: LowLevelState) -> torch.Tensor:
        """构建低层状态向量"""
        vec = [
            state.position / 3.0,  # 归一化
            state.price / 100.0,  # 归一化
            state.bid_ask_spread / 0.01,
            state.volume_imbalance,
            state.momentum,
            state.execution_cost / 0.001,
        ]

        return torch.tensor(vec, dtype=torch.float32)

    def train_high(
        self,
        states: List[HighLevelState],
        actions: List[int],
        rewards: List[float],
    ):
        """训练高层策略 (策略梯度)"""
        if len(states) < 32:
            return

        self.high_policy.train()

        # 构建张量
        state_tensors = [self._build_high_state(s) for s in states]
        state_batch = torch.stack(state_tensors).to(self.device)
        action_batch = torch.tensor(actions, dtype=torch.long).to(self.device)
        reward_batch = torch.tensor(rewards, dtype=torch.float32).to(self.device)

        # 标准化奖励
        reward_batch = (reward_batch - reward_batch.mean()) / (reward_batch.std() + 1e-8)

        # 策略梯度损失
        logits = self.high_policy(state_batch)
        log_probs = F.log_softmax(logits, dim=-1)
        selected_log_probs = log_probs.gather(1, action_batch.unsqueeze(1)).squeeze(1)

        loss = -(selected_log_probs * reward_batch).mean()

        self.high_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.high_policy.parameters(), max_norm=1.0)
        self.high_optimizer.step()

        logger.debug(f"[HierarchicalRL] High-level loss: {loss.item():.4f}")

    def train_low(
        self,
        states: List[LowLevelState],
        actions: List[int],
        rewards: List[float],
    ):
        """训练低层策略 (策略梯度)"""
        if len(states) < 32:
            return

        self.low_policy.train()

        # 构建张量
        state_tensors = [self._build_low_state(s) for s in states]
        state_batch = torch.stack(state_tensors).to(self.device)
        action_batch = torch.tensor(actions, dtype=torch.long).to(self.device)
        reward_batch = torch.tensor(rewards, dtype=torch.float32).to(self.device)

        # 标准化奖励
        reward_batch = (reward_batch - reward_batch.mean()) / (reward_batch.std() + 1e-8)

        # 策略梯度损失
        logits, _ = self.low_policy(state_batch)
        log_probs = F.log_softmax(logits, dim=-1)
        selected_log_probs = log_probs.gather(1, action_batch.unsqueeze(1)).squeeze(1)

        loss = -(selected_log_probs * reward_batch).mean()

        self.low_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.low_policy.parameters(), max_norm=1.0)
        self.low_optimizer.step()

        logger.debug(f"[HierarchicalRL] Low-level loss: {loss.item():.4f}")

    def save(self):
        """保存模型"""
        path = os.path.join(self.model_dir, 'hierarchical_rl.pth')
        torch.save({
            'high_policy': self.high_policy.state_dict(),
            'low_policy': self.low_policy.state_dict(),
            'high_optimizer': self.high_optimizer.state_dict(),
            'low_optimizer': self.low_optimizer.state_dict(),
        }, path)
        logger.info(f"[HierarchicalRL] 模型已保存: {path}")

    def load(self) -> bool:
        """加载模型"""
        path = os.path.join(self.model_dir, 'hierarchical_rl.pth')
        if not os.path.exists(path):
            return False

        checkpoint = torch.load(path, map_location=self.device)
        self.high_policy.load_state_dict(checkpoint['high_policy'])
        self.low_policy.load_state_dict(checkpoint['low_policy'])
        self.high_optimizer.load_state_dict(checkpoint['high_optimizer'])
        self.low_optimizer.load_state_dict(checkpoint['low_optimizer'])
        self.trained = True
        logger.info(f"[HierarchicalRL] 模型已加载: {path}")
        return True

    def get_status(self) -> Dict:
        """获取智能体状态"""
        return {
            'trained': self.trained,
            'high_state_dim': self.high_state_dim,
            'low_state_dim': self.low_state_dim,
            'n_high_actions': self.n_high_actions,
            'n_low_actions': self.n_low_actions,
            'device': str(self.device),
            'high_replay_size': len(self.high_replay),
            'low_replay_size': len(self.low_replay),
        }


# ── 全局单例 ────────────────────────────────────────────────────────────────

_instance: Optional[HierarchicalRLAgent] = None
_instance_lock = threading.Lock()


def get_hierarchical_rl_agent() -> HierarchicalRLAgent:
    """获取全局 HierarchicalRLAgent 单例"""
    global _instance
    with _instance_lock:
        if _instance is None:
            _instance = HierarchicalRLAgent()
            _instance.load()
        return _instance