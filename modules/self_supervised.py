#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
自监督预训练模块 — 2026 SOTA

功能:
1. Masked Time Series Modeling (MTSM) — 预测被遮挡的时序片段
2. Contrastive Learning — SimCLR/Byol 风格对比学习
3. Data Augmentation for Time Series — 时间序列数据增强

架构:
    Input → Encoder → Latent → Decoder → Output
    Loss: MSE (重建) + Contrastive

参考:
    - Eldele et al., "TS-TCC: Time-Series Contrastive Learning" (2021)
    - Fan et al., "MaskNet: A Universal Neural Architecture" (2021)
    - Qian et al., "Mixing up Contrasting Representations" (2021)

作者: Stock Analyzer SOTA Team
"""

import os
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
import threading

from modules.logger import logger

# ── PyTorch 依赖检查 ─────────────────────────────────────────

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("[SelfSupervised] PyTorch 未安装")

# Conditional base class
ModelBase = nn.Module if HAS_TORCH else object


# ── Time Series Data Augmentation ─────────────────────────────

class TimeSeriesAugmentation:
    """
    时间序列数据增强

    方法:
    1. Jittering - 添加高斯噪声
    2. Scaling - 缩放
    3. Time Warping - 时间扭曲
    4. Magnitude Warping - 幅度扭曲
    5. Permutation - 随机打乱片段
    """

    def __init__(self):
        self.jitter_ratio = 0.1
        self.scale_ratio = 0.1
        self.warp_ratio = 0.1

    def jitter(self, x: np.ndarray, ratio: float = None) -> np.ndarray:
        """添加高斯噪声"""
        if ratio is None:
            ratio = self.jitter_ratio
        noise = np.random.normal(0, ratio * np.std(x), x.shape)
        return x + noise

    def scale(self, x: np.ndarray, ratio: float = None) -> np.ndarray:
        """随机缩放"""
        if ratio is None:
            ratio = self.scale_ratio
        scale_factor = np.random.uniform(1 - ratio, 1 + ratio)
        return x * scale_factor

    def time_warp(self, x: np.ndarray, ratio: float = None) -> np.ndarray:
        """时间扭曲 (简化版)"""
        if ratio is None:
            ratio = self.warp_ratio
        seq_len = x.shape[-1]
        warp_points = int(seq_len * ratio)
        if warp_points < 2:
            return x
        warp_start = np.random.randint(0, seq_len - warp_points)
        warp_end = warp_start + warp_points
        warp_amount = np.random.uniform(-0.5, 0.5)
        indices = np.arange(seq_len)
        warp_indices = np.linspace(0, 1, seq_len)
        warp_indices[warp_start:warp_end] += warp_amount
        warp_indices = np.clip(warp_indices, 0, 1)
        # 简化: 直接返回原数据
        return x

    def permute(self, x: np.ndarray) -> np.ndarray:
        """随机打乱片段"""
        seq_len = x.shape[-1]
        n_segments = 5
        segment_len = seq_len // n_segments
        indices = np.random.permutation(n_segments)
        result = np.zeros_like(x)
        pos = 0
        for i in indices:
            seg_start = i * segment_len
            seg_end = min(seg_start + segment_len, seq_len)
            result[..., pos:pos + (seg_end - seg_start)] = x[..., seg_start:seg_end]
            pos += (seg_end - seg_start)
        return result

    def augment(self, x: np.ndarray, methods: List[str] = None) -> np.ndarray:
        """
        应用数据增强

        Args:
            x: (n_samples, seq_len) 或 (seq_len,)
            methods: ['jitter', 'scale', 'time_warp', 'permute']

        Returns:
            增强后的数据
        """
        if methods is None:
            methods = ['jitter', 'scale']

        result = x.copy()
        for method in methods:
            if method == 'jitter':
                result = self.jitter(result)
            elif method == 'scale':
                result = self.scale(result)
            elif method == 'time_warp':
                result = self.time_warp(result)
            elif method == 'permute':
                result = self.permute(result)

        return result


# ── Masked Time Series Model (MTSM) ──────────────────────────

class MaskedTimeSeriesModel(ModelBase):
    """
    Masked Time Series Model — 预测被遮挡的片段

    架构:
        Input → Encoder (Transformer) → Latent → Decoder → Output
        Mask: 随机遮挡 30% 的时间步
    """

    def __init__(
        self,
        seq_len: int = 60,
        n_features: int = 12,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Linear(n_features, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, seq_len, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_features),
        )

        # Mask token
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_features)
            mask: (batch, seq_len) 1=masked, 0=visible

        Returns:
            (batch, seq_len, n_features) 重建的序列
        """
        # Input projection
        x = self.input_proj(x)  # (batch, seq_len, d_model)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Replace masked positions with mask token
        if mask is not None and mask.sum() > 0:
            mask_token = self.mask_token.expand(x.size(0), -1, -1)
            mask_expanded = mask.unsqueeze(-1).expand_as(x)
            x = torch.where(mask_expanded, mask_token, x)

        # Transformer
        x = self.transformer(x)

        # Decode
        output = self.decoder(x)

        return output

    def get_loss(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """计算 masked prediction loss"""
        output = self.forward(x, mask)
        # 只计算 masked 位置的 loss
        mask_expanded = mask.unsqueeze(-1).expand_as(x)
        loss = F.mse_loss(output[mask_expanded], x[mask_expanded])
        return loss


class PositionalEncoding(ModelBase):
    """Positional Encoding for Transformer"""

    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(1, max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ── Contrastive Learning (SimCLR) ─────────────────────────

class SimCLRProjection(ModelBase):
    """
    SimCLR Projection Head

    架构:
        Encoder → Latent → Projection Head → Contrastive Loss
    """

    def __init__(self, d_model: int = 64, projection_dim: int = 32):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, projection_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class ContrastiveLoss(ModelBase):
    """NT-Xent Loss for SimCLR"""

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_i: (batch, projection_dim) augmented view 1
            z_j: (batch, projection_dim) augmented view 2

        Returns:
            loss: scalar
        """
        batch_size = z_i.size(0)
        z = torch.cat([z_i, z_j], dim=0)  # (2*batch, projection_dim)

        # Normalize
        z = F.normalize(z, dim=1)

        # Similarity matrix
        sim = torch.matmul(z, z.T) / self.temperature  # (2*batch, 2*batch)

        # Mask diagonal
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(mask, float('-inf'))

        # Positive pairs
        pos_sim = torch.cat([torch.diag(sim, batch_size), torch.diag(sim, -batch_size)], dim=0)

        # Loss
        loss = -torch.log(torch.exp(pos_sim) / torch.exp(sim).sum(dim=1))
        return loss.mean()


# ── Self-Supervised Pre-Training ─────────────────────────────────

class SelfSupervisedPretrainer:
    """
    自监督预训练器

    功能:
    1. Masked Time Series Modeling
    2. Contrastive Learning (SimCLR)
    3. 预训练后可用于下游任务 (如股票预测)
    """

    def __init__(
        self,
        seq_len: int = 60,
        n_features: int = 12,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        projection_dim: int = 32,
        temperature: float = 0.5,
    ):
        self.seq_len = seq_len
        self.n_features = n_features
        self.d_model = d_model
        self.temperature = temperature

        self.device = None
        self.encoder = None
        self.projection_head = None
        self.augmentation = TimeSeriesAugmentation()
        self.trained = False
        self._lock = threading.Lock()

        self.model_dir = os.path.join(os.path.dirname(__file__), 'dl_models')
        os.makedirs(self.model_dir, exist_ok=True)

        if HAS_TORCH:
            self._init_model()
        else:
            logger.warning("[SelfSupervised] PyTorch 不可用，使用随机 fallback")

    def _init_model(self):
        """初始化模型"""
        self.device = self._get_device()
        self.encoder = MaskedTimeSeriesModel(
            seq_len=self.seq_len,
            n_features=self.n_features,
            d_model=self.d_model,
            n_heads=4,
            n_layers=2,
            dropout=0.1,
        )
        self.projection_head = SimCLRProjection(self.d_model, 32)
        self.encoder.to(self.device)
        self.projection_head.to(self.device)

    def _get_device(self) -> torch.device:
        """获取设备"""
        if torch.cuda.is_available():
            return torch.device('cuda')
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')

    def is_trained(self) -> bool:
        """检查模型是否已预训练"""
        return self.trained

    def pretrain(
        self,
        X: np.ndarray,
        epochs: int = 100,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        mask_ratio: float = 0.3,
    ) -> Dict:
        """
        预训练模型

        Args:
            X: (n_samples, seq_len, n_features) 时间序列数据
            epochs: 预训练轮数
            batch_size: 批大小
            learning_rate: 学习率
            mask_ratio: 遮挡比例

        Returns:
            training_history: {'loss': []}
        """
        if not HAS_TORCH:
            logger.error("[SelfSupervised] PyTorch 不可用，无法预训练")
            return {'loss': []}

        if X.shape[0] < 100:
            logger.warning(f"[SelfSupervised] 预训练样本不足 ({X.shape[0]})")
            return {'loss': []}

        logger.info(f"[SelfSupervised] 开始预训练: {X.shape[0]} 样本")

        self.encoder.train()
        self.projection_head.train()
        optimizer = torch.optim.AdamW(
            list(self.encoder.parameters()) + list(self.projection_head.parameters()),
            lr=learning_rate,
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        history = {'loss': []}

        for epoch in range(epochs):
            self.encoder.train()
            epoch_loss = 0.0
            n_batches = 0

            indices = np.random.permutation(len(X))
            for start in range(0, len(X), batch_size):
                end = min(start + batch_size, len(X))
                batch_idx = indices[start:end]

                X_batch = torch.tensor(X[batch_idx], dtype=torch.float32).to(self.device)

                # 数据增强 (augmented view 1)
                aug1 = self.augmentation.augment(X_batch.cpu().numpy())
                aug1 = torch.tensor(aug1, dtype=torch.float32).to(self.device)

                # 数据增强 (augmented view 2)
                aug2 = self.augmentation.augment(X_batch.cpu().numpy())
                aug2 = torch.tensor(aug2, dtype=torch.float32).to(self.device)

                # Mask
                mask = torch.rand(X_batch.shape[:2]) < mask_ratio
                mask = mask.to(self.device)

                # Forward
                z1 = self.encoder(aug1, mask)
                z2 = self.encoder(aug2, mask)

                # Projection
                p1 = self.projection_head(z1.mean(dim=1))
                p2 = self.projection_head(z2.mean(dim=1))

                # Contrastive loss
                contrastive_loss = ContrastiveLoss(self.temperature)(p1, p2)

                # Masked prediction loss
                mask_loss = self.encoder.get_loss(X_batch, mask)

                # Total loss
                loss = contrastive_loss + mask_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_loss = epoch_loss / n_batches
            history['loss'].append(avg_loss)

            if epoch % 20 == 0 or epoch == epochs - 1:
                logger.info(f"[SelfSupervised] Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f}")

        self.trained = True
        logger.info("[SelfSupervised] 预训练完成")
        return history

    def get_embeddings(self, X: np.ndarray) -> np.ndarray:
        """
        获取时间序列的 embedding

        Args:
            X: (n_samples, seq_len, n_features)

        Returns:
            embeddings: (n_samples, d_model)
        """
        if not HAS_TORCH or not self.trained:
            # Return mean of input as fallback
            return X.mean(axis=1)

        self.encoder.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            # Forward without mask
            z = self.encoder(X_t, torch.zeros_like(X_t[:, :, 0]).bool())
            # Global average pooling
            embeddings = z.mean(dim=1).cpu().numpy()

        return embeddings

    def save(self, path: Optional[str] = None):
        """保存预训练模型"""
        if path is None:
            path = os.path.join(self.model_dir, 'self_supervised.pth')

        if self.encoder is not None and self.projection_head is not None:
            torch.save({
                'encoder_state_dict': self.encoder.state_dict(),
                'projection_state_dict': self.projection_head.state_dict(),
                'device': str(self.device),
                'seq_len': self.seq_len,
                'n_features': self.n_features,
                'd_model': self.d_model,
            }, path)
            logger.info(f"[SelfSupervised] 模型已保存: {path}")

    def load(self, path: Optional[str] = None) -> bool:
        """加载预训练模型"""
        if path is None:
            path = os.path.join(self.model_dir, 'self_supervised.pth')

        if not os.path.exists(path):
            return False

        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
            if self.encoder is None:
                self._init_model()
            self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
            self.projection_head.load_state_dict(checkpoint['projection_state_dict'])
            self.trained = True
            logger.info(f"[SelfSupervised] 模型已加载: {path}")
            return True
        except Exception as e:
            logger.error(f"[SelfSupervised] 加载失败: {e}")
            return False

    def get_status(self) -> Dict:
        """获取预训练状态"""
        return {
            'trained': self.trained,
            'device': str(self.device) if self.device else 'N/A',
            'seq_len': self.seq_len,
            'n_features': self.n_features,
            'd_model': self.d_model,
        }


# ── 全局单例 ────────────────────────────────────────────────

_pretrainer_instance: Optional[SelfSupervisedPretrainer] = None
_pretrainer_lock = threading.Lock()


def get_self_supervised_pretrainer() -> SelfSupervisedPretrainer:
    """获取全局 SelfSupervisedPretrainer 实例 (线程安全)"""
    global _pretrainer_instance
    if _pretrainer_instance is None:
        with _pretrainer_lock:
            if _pretrainer_instance is None:
                _pretrainer_instance = SelfSupervisedPretrainer()
                _pretrainer_instance.load()
    return _pretrainer_instance