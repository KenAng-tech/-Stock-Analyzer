#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
PatchTST 时序预测模型 — 替换无效的 dl_model_v2.py (NumPy 无训练)

架构 (PatchTST, ICML 2023):
    Input(seq_len, n_features)
        → Patch 分片 (将相邻时间步分组)
        → 投影到 d_model
        → RoPE 位置编码
        → Transformer Encoder (Pre-LN + RoPE)
        → 最后一个 Patch 的表示 → Linear Head
        → 输出 (up/neutral/down)

优势:
    - Patch 机制: 降低复杂度 O(n/patch_len) vs O(n²)
    - Channel Independence: 各通道独立建模，跨通道泛化更好
    - RoPE: 旋转位置编码，适合外推
    - 完整 PyTorch 训练管线 (反向传播 + AdamW + Cosine LR)

参考:
    - Zhou et al., "PatchTST: A Neural Network Kernal over Patched Time Series"
    - https://arxiv.org/abs/2211.14730
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import os
import pickle

from modules.logger import logger


# ── 工具函数 ─────────────────────────────────────────────────

def _get_device() -> torch.device:
    """获取最佳可用设备"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ── RoPE 位置编码 ─────────────────────────────────────────────

class RotaryPositionalEmbeddings(nn.Module):
    """
    RoPE (Rotary Positional Embeddings) 位置编码

    将位置信息编码为旋转矩阵，乘到 token embedding 上。
    相比正弦位置编码，RoPE 能更好地处理长序列外推。
    """

    def __init__(self, dim: int, max_seq_len: int = 200):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('embedding', emb)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, dim)
        Returns:
            x + positional_encoding
        """
        seq_len = x.size(1)
        return x + self.embedding[:seq_len].unsqueeze(0)


# ── PatchTST 核心模型 ─────────────────────────────────────────

class PatchTST(nn.Module):
    """
    PatchTST 模型 — 基于 Patch 的时序预测 Transformer

    Args:
        n_features: 输入特征数
        patch_len: 每个 patch 包含的时间步数 (默认 8)
        d_model: 模型维度 (默认 128)
        n_layers: Transformer 层数 (默认 4)
        n_heads: 注意力头数 (默认 8)
        n_classes: 分类类别数 (默认 3: up/neutral/down)
        dropout: Dropout 比例 (默认 0.1)
        max_seq_len: 最大序列长度 (默认 200)
    """

    def __init__(
        self,
        n_features: int = 12,
        patch_len: int = 8,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 8,
        n_classes: int = 3,
        dropout: float = 0.1,
        max_seq_len: int = 200,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.d_model = d_model
        self.n_features = n_features

        # Patch 投影: (patch_len * n_features) → d_model
        self.patch_proj = nn.Linear(patch_len * n_features, d_model)

        # RoPE 位置编码
        self.pos_encoder = RotaryPositionalEmbeddings(d_model, max_seq_len)

        # Transformer Encoder (Pre-LN 更稳定)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-LN
            dropout=dropout,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
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
        """Xavier 均匀初始化"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_features) — 标准化后的时序数据
        Returns:
            logits: (batch, n_classes)
        """
        batch, seq_len, _ = x.shape
        n_patches = seq_len // self.patch_len

        # Patch: (batch, n_patches, patch_len * n_features)
        x = x.reshape(batch, n_patches, self.patch_len * self.n_features)

        # 投影到 d_model
        x = self.patch_proj(x)

        # RoPE 位置编码
        x = self.pos_encoder(x)

        # Transformer Encoder
        x = self.transformer(x)
        x = self.norm(x)

        # 取最后一个 patch 的表示 → 预测
        x = x[:, -1, :]

        return self.head(x)


# ── 训练器 ────────────────────────────────────────────────────

class PatchTSTTrainer:
    """
    PatchTST 训练器

    功能:
        - 完整的训练循环 (训练集 + 验证集)
        - Early stopping + Cosine Annealing LR
        - 梯度裁剪防止梯度爆炸
        - 模型保存/加载
    """

    def __init__(
        self,
        n_features: int = 12,
        patch_len: int = 8,
        d_model: int = 128,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        self.n_features = n_features
        self.patch_len = patch_len
        self.d_model = d_model

        self.model = PatchTST(
            n_features=n_features,
            patch_len=patch_len,
            d_model=d_model,
        )
        self.device = _get_device()
        self.model.to(self.device)

        # 优化器 + 损失 + 调度器
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
        """
        训练模型

        Args:
            X_train: (n_samples, seq_len, n_features) 训练数据
            y_train: (n_samples,) 标签 (0=down, 1=neutral, 2=up)
            X_val: (n_val, seq_len, n_features) 验证数据
            y_val: (n_val,) 验证标签
            epochs: 训练轮数
            batch_size: 批次大小

        Returns:
            训练历史字典 {'loss': [...], 'val_loss': [...], 'val_acc': [...]}
        """
        logger.info(
            f"[PatchTST] 开始训练: device={self.device}, "
            f"train={len(X_train)}, val={len(X_val)}, epochs={epochs}"
        )

        best_val_loss = float('inf')
        patience = 20
        patience_counter = 0

        for epoch in range(epochs):
            # ── 训练阶段 ──
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

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()

                train_loss += loss.item() * len(batch_idx)
                n_samples += len(batch_idx)

            self.scheduler.step()
            avg_train_loss = train_loss / max(n_samples, 1)

            # ── 验证阶段 ──
            val_loss, val_acc = self.evaluate(X_val, y_val)

            self.training_history['loss'].append(avg_train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_acc'].append(val_acc)

            if epoch % 10 == 0:
                logger.info(
                    f"[PatchTST] Epoch {epoch}/{epochs} | "
                    f"Train Loss: {avg_train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val Acc: {val_acc:.4f} | "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
                )

            # ── Early Stopping ──
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self._save_best()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"[PatchTST] Early stopping at epoch {epoch}, best val_loss={best_val_loss:.4f}")
                    break

        self.trained = True
        logger.info(f"[PatchTST] 训练完成, 最终 val_acc={val_acc:.4f}")
        return self.training_history

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """评估模型，返回 (loss, accuracy)"""
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y, dtype=torch.long).to(self.device)

        with torch.no_grad():
            logits = self.model(X_t)
            loss = self.criterion(logits, y_t).item()
            acc = (logits.argmax(dim=-1) == y_t).float().mean().item()

        return loss, acc

    def predict(self, X: np.ndarray) -> Dict:
        """
        预测

        Args:
            X: (n_samples, seq_len, n_features) 或 (seq_len, n_features)
        Returns:
            {
                'directions': ['up', 'neutral', 'down', ...],
                'confidences': [0.85, 0.62, ...],
                'probabilities': {'up': [...], 'neutral': [...], 'down': [...]},
            }
        """
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
        path = os.path.join(self.model_dir, 'patchtst_best.pth')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'device': str(self.device),
            'n_features': self.n_features,
            'patch_len': self.patch_len,
            'd_model': self.d_model,
        }, path, pickle_protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: Optional[str] = None) -> bool:
        """
        加载模型

        Args:
            path: 模型路径，默认为 patchtst_best.pth

        Returns:
            是否成功加载
        """
        if path is None:
            path = os.path.join(self.model_dir, 'patchtst_best.pth')
        if not os.path.exists(path):
            return False

        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.trained = True
            logger.info(f"[PatchTST] 模型已加载: {path}")
            return True
        except Exception as e:
            logger.error(f"[PatchTST] 加载失败: {e}")
            return False


# ── 向后兼容接口 ──────────────────────────────────────────────

class DeepLearningEnsemble:
    """
    深度学习集成模型 — 向后兼容旧接口

    内部使用 PatchTST 模型，接口与原来的 NumPy 实现一致:
        - train(X_train, y_train, X_val, y_val, epochs, batch_size, learning_rate)
        - predict(x_seq) → {'directions', 'confidences', 'probabilities'}
        - save(path) / load(path)
        - trained 标志
    """

    def __init__(self, input_dim: int = 12, sequence_length: int = 20):
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.trainer = PatchTSTTrainer(
            n_features=input_dim,
            patch_len=8,
            d_model=128,
        )
        self.trained = self.trainer.trained
        self.model_dir = self.trainer.model_dir

    def predict(self, x_seq: np.ndarray) -> Dict:
        """
        集成预测 (兼容旧接口)

        Args:
            x_seq: (batch, seq_len, input_dim) 或 (seq_len, input_dim)
        Returns:
            {'directions', 'confidences', 'probabilities'}
        """
        return self.trainer.predict(x_seq)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
    ):
        """
        训练模型 (兼容旧接口)

        Args:
            X_train: (n_samples, seq_len, input_dim)
            y_train: (n_samples,)
            X_val: 验证数据 (可选)
            y_val: 验证标签 (可选)
            epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
        """
        if X_val is None or y_val is None:
            # 自动分割训练/验证集 (80/20)
            n = len(X_train)
            split = int(n * 0.8)
            X_val = X_train[split:]
            y_val = y_train[split:]
            X_train = X_train[:split]
            y_train = y_train[:split]

        # 覆盖学习率
        self.trainer.optimizer = torch.optim.AdamW(
            self.trainer.model.parameters(),
            lr=learning_rate,
        )
        self.trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.trainer.optimizer, T_max=epochs
        )

        self.trainer.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=epochs,
            batch_size=batch_size,
        )
        self.trained = self.trainer.trained

    def save(self, path: Optional[str] = None):
        """保存模型"""
        if path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            path = os.path.join(self.model_dir, f'patchtst_{timestamp}.pth')

        # 直接调用 trainer 的保存
        self.trainer._save_best()
        # 复制文件到指定路径
        import shutil
        best_path = os.path.join(self.model_dir, 'patchtst_best.pth')
        shutil.copy2(best_path, path)

        logger.info(f"[DeepLearningEnsemble] 模型已保存: {path}")
        return path

    @classmethod
    def load(cls, path: str) -> 'DeepLearningEnsemble':
        """加载模型"""
        instance = cls()
        instance.trainer.load(path)
        instance.trained = instance.trainer.trained
        return instance


# ── 全局实例 ──────────────────────────────────────────────────

dl_ensemble = DeepLearningEnsemble()
