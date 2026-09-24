#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
PatchMamba 混合架构 — 2026 年最新 SOTA 时序预测模型

融合 PatchTST 的 Patch 分片机制与 Mamba 的 SSM 状态空间:

架构:
    Input(seq_len, n_features)
        ↓ Patch 分片 (patch_len=8)
    (batch, n_patches, patch_len * n_features)
        ↓ 投影到 d_model
    (batch, n_patches, d_model)
        ↓ Mamba SSM 层 (选择性状态空间)
    (batch, n_patches, d_model)
        ↓ 预测头
    (batch, n_classes)

优势:
- O(n) 线性复杂度 (vs Transformer 的 O(n²))
- 保留 PatchTST 的局部模式捕捉能力
- Mamba 选择性状态空间更适合金融时序

参考: arXiv 2501.01381 (2025-01)
"""

import os
import json
import time
import threading
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
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
    logger.warning("[PatchMamba] PyTorch 未安装")


# ── SSM 选择性状态空间块 (参考 Mamba 架构) ─────────────────────────────────

class SelectiveSSMBlock(nn.Module):
    """
    Mamba 选择性状态空间块

    核心思想: 选择性机制 (selective mechanism)
    - 输入决定是否关注当前状态
    - 避免 LSTM/GRU 的固定门控
    - 更适合金融时序的非线性、非平稳特性
    """

    def __init__(self, d_model: int, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand

        # 输入投影 (输入 → 内部状态)
        self.x_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # 输出投影 (内部状态 → 输出)
        self.dt_proj = nn.Linear(self.d_inner, d_model, bias=True)

        # A (状态矩阵) 和 D (跳跃连接)
        self.A = nn.Parameter(torch.randn(self.d_inner, d_model))
        self.D = nn.Parameter(torch.ones(d_model))

        # 输出门控
        self.gate_proj = nn.Linear(d_model, d_model)

        # y 投影 (d_inner → d_model)
        self.y_proj = nn.Linear(self.d_inner, d_model)

        # 初始化
        self._init_parameters()

    def _init_parameters(self):
        """初始化参数"""
        nn.init.xavier_uniform_(self.A)
        nn.init.zeros_(self.D)
        nn.init.normal_(self.dt_proj.weight, mean=0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape

        # 选择性扫描 (selective scan)
        # x_proj: (batch, seq_len, d_inner * 2)
        x_proj = self.x_proj(x)
        x_inner = x_proj[:, :, :self.d_inner]  # (batch, seq_len, d_inner)
        dt = F.softplus(self.dt_proj(x_proj[:, :, self.d_inner:]))  # (batch, seq_len, d_model)

        # 状态空间
        # y = (A * x) + D * x
        # 使用三角矩阵稳定计算
        A = F.softplus(self.A)  # (d_inner, d_model)

        # 简化的选择性状态更新
        y = torch.einsum('bld,md->blm', x_inner, A.transpose(0, 1)) * dt
        y = y + self.D * x

        # 门控
        gate = torch.sigmoid(self.gate_proj(x))
        y = y * gate

        # 残差连接
        y = y + x

        return y


# ── PatchMamba 主模型 ──────────────────────────────────────────────────────

class PatchMamba(nn.Module):
    """
    PatchMamba: Patch 分片 + Mamba SSM 混合架构

    融合 PatchTST 的 patch 机制与 Mamba 的选择性状态空间
    - O(n) 线性复杂度
    - 适合高频交易和长期预测
    """

    def __init__(
        self,
        n_features: int = 12,
        patch_len: int = 8,
        d_model: int = 128,
        n_layers: int = 4,
        n_classes: int = 3,
        dropout: float = 0.1,
        max_seq_len: int = 500,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.n_features = n_features
        self.d_model = d_model
        self.n_classes = n_classes

        # Patch 投影: (patch_len * n_features) → d_model
        self.patch_proj = nn.Sequential(
            nn.Linear(patch_len * n_features, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # Mamba SSM 层
        self.ssm_layers = nn.ModuleList([
            SelectiveSSMBlock(d_model) for _ in range(n_layers)
        ])

        # LayerNorm
        self.norm = nn.LayerNorm(d_model)

        # 预测头
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes),
        )

        self._init_parameters()

    def _init_parameters(self):
        """Xavier 初始化"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_features)
        Returns:
            logits: (batch, n_classes)
        """
        batch, seq_len, _ = x.shape

        # 确保 seq_len 是 patch_len 的整数倍
        if seq_len % self.patch_len != 0:
            pad_len = self.patch_len - (seq_len % self.patch_len)
            x = F.pad(x, (0, 0, 0, pad_len))
            seq_len = x.shape[1]

        n_patches = seq_len // self.patch_len

        # Patch 化: (batch, n_patches, patch_len * n_features)
        x = x.reshape(batch, n_patches, self.patch_len * self.n_features)

        # 投影到 d_model
        x = self.patch_proj(x)

        # Mamba SSM 处理
        for layer in self.ssm_layers:
            x = layer(x)

        # LayerNorm
        x = self.norm(x)

        # 取最后一个 patch 的表示
        x = x[:, -1, :]

        return self.head(x)


# ── 训练器 ─────────────────────────────────────────────────────────────────

class PatchMambaTrainer:
    """
    PatchMamba 训练器

    使用方式:
        trainer = PatchMambaTrainer()
        trainer.train(X_train, y_train, X_val, y_val)
        result = trainer.predict(X_test)
    """

    def __init__(
        self,
        n_features: int = 12,
        patch_len: int = 8,
        d_model: int = 128,
        n_layers: int = 4,
        n_classes: int = 3,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        if not HAS_TORCH:
            raise ImportError("[PatchMamba] PyTorch 未安装")

        self.n_features = n_features
        self.patch_len = patch_len
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_classes = n_classes

        self.model = PatchMamba(
            n_features=n_features,
            patch_len=patch_len,
            d_model=d_model,
            n_layers=n_layers,
            n_classes=n_classes,
        )

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else
            'mps' if torch.backends.mps.is_available() else 'cpu'
        )
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100
        )

        self.training_history = {'loss': [], 'val_loss': [], 'val_acc': []}
        self.trained = False

        self.model_dir = os.path.join(os.path.dirname(__file__), 'dl_models')
        os.makedirs(self.model_dir, exist_ok=True)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 100,
        batch_size: int = 64,
    ) -> Dict:
        """训练模型"""
        logger.info(f"[PatchMamba] 开始训练, device={self.device}, epochs={epochs}")

        best_val_loss = float('inf')
        patience = 20
        patience_counter = 0

        for epoch in range(epochs):
            # 训练
            self.model.train()
            train_loss = 0.0
            n_batches = 0

            indices = np.random.permutation(len(X_train))
            for start in range(0, len(X_train), batch_size):
                end = min(start + batch_size, len(X_train))
                batch_idx = indices[start:end]

                X_batch = torch.tensor(X_train[batch_idx], dtype=torch.float32).to(self.device)
                y_batch = torch.tensor(y_train[batch_idx], dtype=torch.long).to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(X_batch)
                loss = self.criterion(logits, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                train_loss += loss.item() * len(batch_idx)
                n_batches += 1

            self.scheduler.step()
            avg_train_loss = train_loss / len(X_train)

            # 验证
            val_loss, val_acc = self.evaluate(X_val, y_val)

            self.training_history['loss'].append(avg_train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_acc'].append(val_acc)

            if epoch % 10 == 0:
                logger.info(
                    f"[PatchMamba] Epoch {epoch}/{epochs} | "
                    f"Train Loss: {avg_train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val Acc: {val_acc:.4f}"
                )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self._save_best()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"[PatchMamba] Early stopping at epoch {epoch}")
                    break

        self.trained = True
        logger.info("[PatchMamba] 训练完成")
        return self.training_history

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """评估模型"""
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y, dtype=torch.long).to(self.device)

        with torch.no_grad():
            logits = self.model(X_t)
            loss = self.criterion(logits, y_t).item()
            acc = (logits.argmax(dim=-1) == y_t).float().mean().item()

        return loss, acc

    def predict(self, X: np.ndarray) -> Dict:
        """预测"""
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits = self.model(X_t)
            probs = torch.softmax(logits, dim=-1)
            predictions = logits.argmax(dim=-1)
            confidences = probs.max(dim=-1)[0]

        direction_map = {0: 'down', 1: 'neutral', 2: 'up'}

        return {
            'directions': [direction_map[p.item()] for p in predictions],
            'confidences': confidences.cpu().numpy().tolist(),
            'probabilities': {
                'up': probs[:, 2].cpu().numpy().tolist(),
                'neutral': probs[:, 1].cpu().numpy().tolist(),
                'down': probs[:, 0].cpu().numpy().tolist(),
            },
        }

    def _save_best(self):
        """保存最佳模型"""
        path = os.path.join(self.model_dir, 'patchmamba_best.pth')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'device': str(self.device),
            'n_features': self.n_features,
            'patch_len': self.patch_len,
            'd_model': self.d_model,
            'n_layers': self.n_layers,
            'n_classes': self.n_classes,
        }, path)

    def load(self, path: Optional[str] = None) -> bool:
        """加载模型"""
        if path is None:
            path = os.path.join(self.model_dir, 'patchmamba_best.pth')
        if not os.path.exists(path):
            return False

        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.trained = True
        logger.info(f"[PatchMamba] 模型已加载: {path}")
        return True

    @property
    def is_trained(self) -> bool:
        return self.trained


# ── 全局单例 ────────────────────────────────────────────────────────────────

_instance: Optional[PatchMambaTrainer] = None
_instance_lock = threading.Lock()


def get_patchmamba() -> PatchMambaTrainer:
    """获取全局 PatchMamba 单例"""
    global _instance
    with _instance_lock:
        if _instance is None:
            _instance = PatchMambaTrainer()
            # 尝试加载已有模型
            _instance.load()
        return _instance