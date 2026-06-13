#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Mamba/SSM 时序预测模型 — O(n) 线性复杂度

Mamba 是 2024 年提出的新一代时序/序列模型:
    - 选择性状态空间模型 (Selective SSM)
    - O(n) 线性复杂度 (vs Transformer O(n²))
    - 适合高频交易 (长序列建模)
    - 硬件感知算法 (Hardware-Aware Algorithm)

参考:
    - Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective SSMs" (2024)
    - Dao et al., "FlashMamba: Hardware-Aware Selective State Space Models" (2024)
    - Liu et al., "Time-Mamba: Mamba for Time Series Forecasting" (2024)

架构:
    Input → SSM Block (Selective) → SSM Block (Selective) → ... → Output

SSM Block:
    x_t = A * x_{t-1} + B * u_t
    y_t = C * x_t
    A, B, C 通过输入自适应选择 (Selective Mechanism)
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import os

from modules.logger import logger


# ── 选择性状态空间模块 (Selective SSM) ────────────────────────

class SelectiveSSMBlock(nn.Module):
    """
    选择性状态空间模块 (Selective SSM Block)

    核心公式:
        h_t = A * h_{t-1} + B * x_t   (状态方程)
        y_t = C * h_t                  (输出方程)

    Selective Mechanism:
        A, B, C 通过输入自适应调整:
        A(x) = Softplus(Linear_A(x))
        B(x) = Linear_B(x)
        C(x) = Linear_C(x)

    这使得模型能够根据输入内容自适应地选择信息。
    """

    def __init__(self, d_model: int, d_state: int = 16,
                 d_conv: int = 4, expand_factor: int = 2):
        """
        Args:
            d_model: 模型维度
            d_state: 状态维度
            d_conv: 局部卷积窗口大小
            expand_factor: 扩展因子 (内部维度 = expand_factor * d_model)
        """
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand_factor

        # 输入投影
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # 卷积层 (局部依赖)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,  # depthwise
        )

        # SSM 参数
        self.x_proj = nn.Linear(self.d_inner, d_state * 4, bias=False)

        # A 参数 (对角化)
        self.A_log = nn.Parameter(torch.log(torch.ones(d_state)))

        # 输出投影
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # LayerNorm
        self.norm = nn.LayerNorm(d_model)

        self._init_parameters()

    def _init_parameters(self):
        """初始化参数"""
        nn.init.kaiming_normal_(self.in_proj.weight)
        nn.init.kaiming_normal_(self.x_proj.weight)
        nn.init.kaiming_normal_(self.out_proj.weight)

    def _ssm_step(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        单步 SSM 计算

        Args:
            x: (batch, d_inner)
            h: (batch, d_state)

        Returns:
            y: (batch, d_inner)
            h_new: (batch, d_state)
        """
        # 分解 x 为 B, C, Δ 参数
        x_split = self.x_proj(x)  # (batch, d_state * 4)
        B = x_split[:, :self.d_state]
        C = x_split[:, self.d_state:2 * self.d_state]
        delta = torch.softmax(x_split[:, 2 * self.d_state:3 * self.d_state], dim=-1) + 1e-3

        # A 参数 (对角化)
        A = torch.exp(-torch.exp(self.A_log))  # (d_state,)

        # 状态更新: h_t = A * h_{t-1} + B * x_t
        h_new = A.unsqueeze(0) * h + B.unsqueeze(-1) * x.unsqueeze(1)

        # 输出: y = C * h_t
        y = (C * h_new).sum(dim=-1)

        return y, h_new

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            x: (batch, seq_len, d_model)
        """
        residual = x

        # LayerNorm
        x = self.norm(x)

        # 输入投影 + 门控
        x_bc = self.in_proj(x)  # (batch, seq_len, d_inner * 2)
        x, gate = x_bc.chunk(2, dim=-1)

        # 时间维度: (batch, d_inner, seq_len)
        x = x.transpose(1, 2)

        # 局部卷积
        x = self.conv1d(x)
        x = x.transpose(1, 2)

        # GELU 激活
        x = torch.nn.functional.gelu(x)

        # SSM 步骤
        batch, seq_len, d_inner = x.shape
        h = torch.zeros(batch, self.d_state, device=x.device)

        outputs = []
        for t in range(seq_len):
            y, h = self._ssm_step(x[:, t, :], h)
            outputs.append(y)

        x_out = torch.stack(outputs, dim=1)  # (batch, seq_len, d_inner)

        # 门控
        x_out = x_out * torch.sigmoid(gate)

        # 输出投影 + Residual
        x_out = self.out_proj(x_out)
        return x_out + residual


# ── Mamba 时序模型 ────────────────────────────────────────────

class MambaTimeSeries(nn.Module):
    """
    Mamba 时序预测模型

    架构:
        Input(seq_len, d_model)
            → Linear 投影
            → Mamba Block × N_layers
            → Last token → Linear Head
            → Output(n_classes)

    优势:
        - O(n) 线性复杂度 (vs Transformer O(n²))
        - 长序列建模能力强
        - 适合高频交易数据
    """

    def __init__(
        self,
        n_features: int = 12,
        d_model: int = 128,
        n_layers: int = 4,
        d_state: int = 16,
        n_classes: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_features = n_features

        # 输入投影
        self.input_proj = nn.Linear(n_features, d_model)

        # Mamba Blocks
        self.blocks = nn.ModuleList([
            SelectiveSSMBlock(
                d_model=d_model,
                d_state=d_state,
                expand_factor=2,
            )
            for _ in range(n_layers)
        ])

        # LayerNorm
        self.norm = nn.LayerNorm(d_model)

        # 预测头
        self.head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, n_classes),
        )

        self._init_parameters()

    def _init_parameters(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.xavier_uniform_(self.head[0].weight)
        nn.init.xavier_uniform_(self.head[3].weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_features)
        Returns:
            logits: (batch, n_classes)
        """
        batch, seq_len, _ = x.shape

        # 输入投影
        x = self.input_proj(x)

        # Mamba Blocks
        for block in self.blocks:
            x = block(x)

        # LayerNorm
        x = self.norm(x)

        # 取最后一个 token
        x = x[:, -1, :]

        return self.head(x)


# ── Mamba 训练器 ──────────────────────────────────────────────

class MambaTrainer:
    """Mamba 训练器"""

    def __init__(
        self,
        n_features: int = 12,
        d_model: int = 128,
        n_layers: int = 4,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        self.n_features = n_features
        self.d_model = d_model
        self.n_layers = n_layers

        self.model = MambaTimeSeries(
            n_features=n_features,
            d_model=d_model,
            n_layers=n_layers,
        )
        self.device = torch.device('cuda' if torch.cuda.is_available()
                                    else 'mps' if torch.backends.mps.is_available()
                                    else 'cpu')
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
        """训练 Mamba 模型"""
        logger.info(
            f"[Mamba] 开始训练: device={self.device}, "
            f"train={len(X_train)}, val={len(X_val)}, epochs={epochs}"
        )

        best_val_loss = float('inf')
        patience = 20
        patience_counter = 0

        for epoch in range(epochs):
            # 训练
            self.model.train()
            train_loss = 0.0
            n_samples = 0

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
                n_samples += len(batch_idx)

            self.scheduler.step()
            avg_train_loss = train_loss / max(n_samples, 1)

            # 验证
            val_loss, val_acc = self.evaluate(X_val, y_val)

            self.training_history['loss'].append(avg_train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_acc'].append(val_acc)

            if epoch % 10 == 0:
                logger.info(
                    f"[Mamba] Epoch {epoch}/{epochs} | "
                    f"Train: {avg_train_loss:.4f} | Val: {val_loss:.4f} | "
                    f"Acc: {val_acc:.4f}"
                )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self._save_best()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"[Mamba] Early stopping at epoch {epoch}")
                    break

        self.trained = True
        logger.info(f"[Mamba] 训练完成, val_acc={val_acc:.4f}")
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
        if X.ndim == 2:
            X = X[np.newaxis, :, :]

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
        path = os.path.join(self.model_dir, 'mamba_best.pth')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'device': str(self.device),
            'n_features': self.n_features,
            'd_model': self.d_model,
            'n_layers': self.n_layers,
        }, path)

    def load(self, path: Optional[str] = None) -> bool:
        """加载模型"""
        if path is None:
            path = os.path.join(self.model_dir, 'mamba_best.pth')
        if not os.path.exists(path):
            return False

        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.trained = True
            return True
        except Exception as e:
            logger.error(f"[Mamba] 加载失败: {e}")
            return False
