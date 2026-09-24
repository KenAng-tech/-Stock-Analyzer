#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Adversarial Training — 对抗训练 (2026 SOTA)

提升模型在噪声和分布偏移下的鲁棒性:

架构:
    1. FGSM (Fast Gradient Sign Method) — 快速对抗攻击
    2. PGD (Projected Gradient Descent) — 迭代对抗攻击
    3. 对抗训练 (Adversarial Training) — 在对抗样本上训练

核心思想:
- 在输入中加入微小扰动 (对抗样本)
- 对抗样本使模型在分布外数据上表现下降
- 在对抗样本上训练可以提升模型鲁棒性

对金融时序特别重要:
- 市场噪声大 (散户情绪, 新闻噪声)
- 分布变化快 (regime shift)
- 异常事件 (黑天鹅)

参考:
- "Explaining and Harnessing Adversarial Examples" (FGSM, 2014)
- "Towards Deep Learning Models Resistant to Adversarial Attacks" (PGD, 2018)
"""

import os
import json
import time
import threading
import numpy as np
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from modules.logger import logger

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("[Adversarial] PyTorch 未安装")


# ── FGSM 对抗攻击 ─────────────────────────────────────────────────────────

def fgsm_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 0.01,
    criterion: Optional[nn.Module] = None,
) -> torch.Tensor:
    """
    FGSM (Fast Gradient Sign Method) 对抗攻击

    原理:
        x_adv = x + ε * sign(∇_x Loss(x, y))

    特点:
    - 单步攻击 (快速)
    - 在扰动边界上寻找对抗样本

    Args:
        model: 模型
        x: 输入 (batch, seq_len, n_features) 或 (batch, n_features)
        y: 目标 (batch,) 或 (n,)
        epsilon: 扰动幅度 (0.01 = 1% 扰动)

    Returns:
        x_adv: 对抗样本
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    x_adv = x.clone().detach().requires_grad_(True)

    logits = model(x_adv)
    loss = criterion(logits, y)

    model.zero_grad()
    loss.backward()

    grad = x_adv.grad
    x_adv = x_adv + epsilon * grad.sign()

    return x_adv.detach()


# ── PGD 对抗攻击 ─────────────────────────────────────────────────────────

def pgd_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 0.01,
    alpha: float = 0.003,
    k_steps: int = 7,
    criterion: Optional[nn.Module] = None,
) -> torch.Tensor:
    """
    PGD (Projected Gradient Descent) 对抗攻击

    原理:
        for t in 1..k:
            x_adv = clip(x_adv + α * sign(∇_x Loss(x_adv, y)))
            x_adv = clamp(x_adv, x - ε, x + ε)

    特点:
    - 迭代攻击 (更强大)
    - 比 FGSM 更强的对抗样本

    Args:
        model: 模型
        x: 输入
        y: 目标
        epsilon: 最大扰动幅度
        alpha: 每步扰动幅度
        k_steps: 迭代步数
        criterion: 损失函数

    Returns:
        x_adv: 对抗样本
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    x_adv = x.clone().detach()

    # 初始化 (随机扰动)
    x_adv = x + torch.randn_like(x) * epsilon * 0.1
    x_adv = torch.clamp(x_adv, x - epsilon, x + epsilon)

    for _ in range(k_steps):
        x_adv = x_adv.clone().detach().requires_grad_(True)

        logits = model(x_adv)
        loss = criterion(logits, y)

        model.zero_grad()
        loss.backward()

        grad = x_adv.grad
        x_adv = x_adv.detach() + alpha * grad.sign()
        x_adv = torch.clamp(x_adv, x - epsilon, x + epsilon)

    return x_adv.detach()


# ── 对抗训练器 ──────────────────────────────────────────────────────────────

@dataclass
class AdversarialConfig:
    """对抗训练配置"""
    method: str = "pgd"  # "fgsm" / "pgd"
    epsilon: float = 0.01  # 扰动幅度
    alpha: float = 0.003  # PGD 每步幅度
    k_steps: int = 7  # PGD 步数
    ratio: float = 0.5  # 对抗样本比例 (一半对抗, 一半原始)
    targeted: bool = False  # 是否目标攻击


class AdversarialTrainer:
    """
    对抗训练器

    训练流程:
    1. 原始样本前向传播 → 损失
    2. 生成对抗样本 (FGSM/PGD)
    3. 对抗样本前向传播 → 损失
    4. 加权合并两个损失 (原始 + 对抗)
    5. 反向传播 + 更新

    使用方式:
        trainer = AdversarialTrainer(model)
        trainer.train_step(x, y)  # 单步训练
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[AdversarialConfig] = None,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: Optional[torch.device] = None,
    ):
        if not HAS_TORCH:
            raise ImportError("[Adversarial] PyTorch 未安装")

        self.model = model
        self.config = config or AdversarialConfig()
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.optimizer = optimizer

        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else
            'mps' if torch.backends.mps.is_available() else 'cpu'
        )

        if self.model is None:
            logger.info("[Adversarial] 未提供模型，使用默认 MLP 分类器")
            self.model = nn.Sequential(
                nn.Linear(12, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 3),
            )

        self.model.to(self.device)
        self.training_history: List[Dict] = []

    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> Dict:
        """
        单步训练 (原始样本 + 对抗样本)

        Args:
            x: 输入 (batch, seq_len, n_features) 或 (batch, n_features)
            y: 目标 (batch,)

        Returns:
            Dict: 包含损失和对抗损失
        """
        x = x.to(self.device)
        y = y.to(self.device)

        self.model.train()

        # 1. 原始样本前向传播
        logits_clean = self.model(x)
        loss_clean = self.criterion(logits_clean, y)

        # 2. 生成对抗样本
        if self.config.method == "fgsm":
            x_adv = fgsm_attack(
                self.model, x, y,
                epsilon=self.config.epsilon,
                criterion=self.criterion,
            )
        else:  # pgd
            x_adv = pgd_attack(
                self.model, x, y,
                epsilon=self.config.epsilon,
                alpha=self.config.alpha,
                k_steps=self.config.k_steps,
                criterion=self.criterion,
            )

        # 3. 对抗样本前向传播
        logits_adv = self.model(x_adv)
        loss_adv = self.criterion(logits_adv, y)

        # 4. 加权合并损失 (原始 + 对抗)
        loss_total = (1 - self.config.ratio) * loss_clean + self.config.ratio * loss_adv

        # 5. 反向传播 + 更新
        if self.optimizer:
            self.optimizer.zero_grad()
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

        # 记录
        history_entry = {
            'loss_clean': loss_clean.item(),
            'loss_adv': loss_adv.item(),
            'loss_total': loss_total.item(),
            'adv_accuracy': (logits_adv.argmax(dim=-1) == y).float().mean().item(),
            'clean_accuracy': (logits_clean.argmax(dim=-1) == y).float().mean().item(),
        }
        self.training_history.append(history_entry)

        return history_entry

    def evaluate_robustness(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        epsilons: Optional[List[float]] = None,
    ) -> Dict:
        """
        评估模型鲁棒性 (不同扰动幅度下)

        Args:
            x: 测试数据
            y: 测试标签
            epsilons: 扰动幅度列表 (默认 [0.001, 0.005, 0.01, 0.02, 0.05])

        Returns:
            Dict: 各 epsilon 下的准确率
        """
        if epsilons is None:
            epsilons = [0.001, 0.005, 0.01, 0.02, 0.05]

        self.model.eval()
        results = {}

        for eps in epsilons:
            x_adv = fgsm_attack(self.model, x, y, epsilon=eps, criterion=self.criterion)
            with torch.no_grad():
                logits = self.model(x_adv)
                acc = (logits.argmax(dim=-1) == y).float().mean().item()
            results[f'epsilon_{eps}'] = acc

        return results


# ── 批量对抗训练 ────────────────────────────────────────────────────────────

def adversarial_train(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 100,
    batch_size: int = 64,
    epsilon: float = 0.01,
    method: str = "pgd",
    k_steps: int = 7,
    ratio: float = 0.5,
) -> Dict:
    """
    批量对抗训练 (便捷函数)

    Args:
        model: 模型
        X_train: 训练数据 (n_samples, n_features)
        y_train: 训练标签 (n_samples,)
        X_val: 验证数据
        y_val: 验证标签
        epochs: 训练轮数
        batch_size: 批大小
        epsilon: 扰动幅度
        method: "fgsm" / "pgd"
        k_steps: PGD 步数
        ratio: 对抗样本比例

    Returns:
        Dict: 训练历史
    """
    if not HAS_TORCH:
        raise ImportError("[Adversarial] PyTorch 未安装")

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    if model is None:
        model = nn.Sequential(
            nn.Linear(12, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 3),
        )
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    config = AdversarialConfig(
        method=method,
        epsilon=epsilon,
        k_steps=k_steps,
        ratio=ratio,
    )
    trainer = AdversarialTrainer(model, config, criterion, optimizer, device)

    history = {'loss': [], 'val_acc': [], 'robustness': []}

    for epoch in range(epochs):
        # 训练
        model.train()
        indices = np.random.permutation(len(X_train))

        epoch_losses = []
        for start in range(0, len(X_train), batch_size):
            end = min(start + batch_size, len(X_train))
            batch_idx = indices[start:end]

            x_batch = torch.tensor(X_train[batch_idx], dtype=torch.float32).to(device)
            y_batch = torch.tensor(y_train[batch_idx], dtype=torch.long).to(device)

            entry = trainer.train_step(x_batch, y_batch)
            epoch_losses.append(entry['loss_total'])

        # 验证
        model.eval()
        x_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_val_t = torch.tensor(y_val, dtype=torch.long).to(device)

        with torch.no_grad():
            logits = model(x_val_t)
            val_acc = (logits.argmax(dim=-1) == y_val_t).float().mean().item()

        # 鲁棒性评估
        if epoch % 10 == 0:
            robustness = trainer.evaluate_robustness(x_val_t[:min(500, len(x_val_t))], y_val_t[:min(500, len(y_val_t))])
        else:
            robustness = {}

        history['loss'].append(np.mean(epoch_losses))
        history['val_acc'].append(val_acc)
        history['robustness'].append(robustness)

        if epoch % 10 == 0:
            logger.info(
                f"[Adversarial] Epoch {epoch}/{epochs} | "
                f"Loss: {np.mean(epoch_losses):.4f} | "
                f"Val Acc: {val_acc:.4f}"
            )

    return history


# ── 全局单例 ──────────────────────────────────────────────────────────────

_instance: Optional[AdversarialTrainer] = None
_instance_lock = threading.Lock()


def get_adversarial_trainer(
    model: Optional[nn.Module] = None,
    config: Optional[AdversarialConfig] = None,
    criterion: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> AdversarialTrainer:
    """获取全局 AdversarialTrainer 单例"""
    global _instance
    with _instance_lock:
        if _instance is None:
            if not HAS_TORCH:
                raise ImportError("[Adversarial] PyTorch 未安装")
            _instance = AdversarialTrainer(
                model=model,
                config=config,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
            )
        return _instance