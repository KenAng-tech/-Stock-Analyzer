#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
自监督预训练 — Masked Time Series Modeling (MTSM)

参考:
    - BERT: Masked Language Modeling
    - TimesT: Masked Time Series Modeling
    - MAE: Masked Autoencoder for Time Series

核心思想:
    1. 随机 mask 时序中的部分时间步或特征
    2. 模型预测被 mask 的值
    3. 预训练后，微调时只需替换预测头

预训练目标:
    - 掩码时序重建 (Masked Value Reconstruction)
    - 对比学习 (Contrastive Learning)
    - 时序顺序预测 (Temporal Order Prediction)
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import os

from modules.logger import logger


# ── 数据增强 ──────────────────────────────────────────────────

class TimeSeriesAugmentor:
    """时序数据增强"""

    @staticmethod
    def jitter(x: np.ndarray, sigma: float = 0.05) -> np.ndarray:
        """高斯噪声"""
        return x + np.random.randn(*x.shape) * sigma

    @staticmethod
    def scaling(x: np.ndarray, sigma: float = 0.1) -> np.ndarray:
        """缩放"""
        factor = np.random.normal(scale=sigma, size=x.shape[-1])
        return x * (1 + factor)

    @staticmethod
    def rotation(x: np.ndarray) -> np.ndarray:
        """随机旋转"""
        theta = np.random.uniform(-np.pi, np.pi)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        return x @ rot

    @staticmethod
    def warp(x: np.ndarray, sigma: float = 0.2, magnitude: float = 0.5) -> np.ndarray:
        """时间扭曲"""
        T = len(x)
        dt = np.random.randn(T) * sigma
        w = np.cumsum(dt)
        w = (w - w.min()) / (w.max() - w.min() + 1e-10) * T
        w = np.clip(w, 0, T).astype(int)

        result = np.zeros_like(x)
        for i in range(T):
            if i < len(w):
                result[i] = x[min(w[i], T - 1)]
        return result


# ── 自监督预训练模型 ──────────────────────────────────────────

class MaskedTimeSeriesModel(nn.Module):
    """
    掩码时序预测模型 — 自监督预训练

    架构:
        Input(seq_len, n_features)
            → Mask (随机 mask 部分时间步)
            → Transformer Encoder
            → Masked Token Prediction
            → Reconstructed(seq_len, n_features)

    损失函数:
        L = L_reconstruction + λ * L_contrastive
    """

    def __init__(
        self,
        n_features: int = 12,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 8,
        mask_ratio: float = 0.3,
    ):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.mask_ratio = mask_ratio

        # 输入投影
        self.input_proj = nn.Linear(n_features, d_model)

        # RoPE 位置编码
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        t = torch.arange(200).float()
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('pos_encoding', emb)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            activation='gelu',
            batch_first=True,
            norm_first=True,
            dropout=0.1,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

        # 预测头 (重建被 mask 的值)
        self.recon_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, n_features),
        )

        # 对比学习头
        self.proj_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 32),
        )

        self._init_parameters()

    def _init_parameters(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.xavier_uniform_(self.recon_head[0].weight)
        nn.init.xavier_uniform_(self.recon_head[2].weight)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict:
        """
        Args:
            x: (batch, seq_len, n_features)
            mask: (batch, seq_len) 布尔掩码，True 表示被 mask

        Returns:
            {
                'reconstruction': (batch, seq_len, n_features),
                'mask': (batch, seq_len),
                'contrastive_repr': (batch, 32),
            }
        """
        batch, seq_len, _ = x.shape

        # 如果没有传入 mask，随机生成
        if mask is None:
            mask = torch.rand(batch, seq_len, device=x.device) < self.mask_ratio

        # 被 mask 的位置替换为 0
        x_masked = x.clone()
        x_masked[mask] = 0

        # 输入投影 + 位置编码
        x_emb = self.input_proj(x_masked) + self.pos_encoding[:seq_len]

        # Transformer Encoder
        x_out = self.encoder(x_emb)
        x_out = self.norm(x_out)

        # 重建 (只对被 mask 的位置)
        recon = self.recon_head(x_out)

        # 对比学习表示 (取所有 token 的平均)
        contrastive_repr = self.proj_head(x_out.mean(dim=1))

        return {
            'reconstruction': recon,
            'mask': mask,
            'contrastive_repr': contrastive_repr,
        }

    def create_mask(self, batch_size: int, seq_len: int) -> torch.Tensor:
        """创建随机 mask"""
        return torch.rand(batch_size, seq_len) < self.mask_ratio


# ── 自监督预训练器 ────────────────────────────────────────────

class SelfSupervisedPretrainer:
    """
    自监督预训练器

    预训练流程:
        1. 数据增强 (jitter, scaling, rotation, warp)
        2. 随机 mask
        3. 重建损失 + 对比损失
        4. 预训练完成后，加载权重到下游模型
    """

    def __init__(
        self,
        n_features: int = 12,
        d_model: int = 128,
        n_layers: int = 4,
        learning_rate: float = 1e-3,
        mask_ratio: float = 0.3,
        contrastive_weight: float = 0.5,
    ):
        self.n_features = n_features
        self.d_model = d_model
        self.mask_ratio = mask_ratio
        self.contrastive_weight = contrastive_weight

        self.model = MaskedTimeSeriesModel(
            n_features=n_features,
            d_model=d_model,
            n_layers=n_layers,
            mask_ratio=mask_ratio,
        )
        self.device = torch.device('cuda' if torch.cuda.is_available()
                                    else 'mps' if torch.backends.mps.is_available()
                                    else 'cpu')
        self.model.to(self.device)

        self.augmentor = TimeSeriesAugmentor()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=200
        )

        self.training_history = {'loss': [], 'recon_loss': [], 'contrast_loss': []}
        self.trained = False
        self.model_dir = os.path.join(os.path.dirname(__file__), 'dl_models')
        os.makedirs(self.model_dir, exist_ok=True)

    def _compute_recon_loss(self, outputs: Dict, x: torch.Tensor,
                             mask: torch.Tensor) -> torch.Tensor:
        """重建损失 (MSE)"""
        recon = outputs['reconstruction']
        # 只计算被 mask 的位置
        mask_expanded = mask.unsqueeze(-1).expand_as(recon)
        masked_x = x[mask_expanded]
        masked_recon = recon[mask_expanded]
        return nn.functional.mse_loss(masked_recon, masked_x)

    def _compute_contrastive_loss(self, outputs: Dict, batch_size: int) -> torch.Tensor:
        """对比损失 (InfoNCE)"""
        reprs = outputs['contrastive_repr']  # (batch, 32)
        reprs = nn.functional.normalize(reprs, dim=1)

        # 相似度矩阵
        sim = torch.matmul(reprs, reprs.T) / 0.1  # 温度系数

        # 对角线为负样本 (排除自身)
        labels = torch.arange(batch_size, device=self.device)
        mask = torch.eye(batch_size, device=self.device) == 0
        sim = sim[mask].view(batch_size, -1)

        return nn.functional.cross_entropy(sim, labels)

    def pretrain(
        self,
        X: np.ndarray,
        epochs: int = 200,
        batch_size: int = 64,
    ) -> Dict:
        """
        自监督预训练

        Args:
            X: (n_samples, seq_len, n_features) 原始时序数据
            epochs: 预训练轮数
            batch_size: 批次大小

        Returns:
            训练历史
        """
        logger.info(
            f"[SelfSupervised] 开始预训练: device={self.device}, "
            f"n_samples={len(X)}, epochs={epochs}"
        )

        best_loss = float('inf')
        patience = 40
        patience_counter = 0

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            n_samples = 0

            indices = np.random.permutation(len(X))
            for start in range(0, len(X), batch_size):
                end = min(start + batch_size, len(X))
                batch_idx = indices[start:end]
                bs = len(batch_idx)

                # 数据增强
                X_batch = X[batch_idx].copy()
                for i in range(bs):
                    aug_method = np.random.choice(['jitter', 'scaling', 'warp'])
                    if aug_method == 'jitter':
                        X_batch[i] = self.augmentor.jitter(X_batch[i])
                    elif aug_method == 'scaling':
                        X_batch[i] = self.augmentor.scaling(X_batch[i])
                    elif aug_method == 'warp':
                        X_batch[i] = self.augmentor.warp(X_batch[i])

                X_t = torch.tensor(X_batch, dtype=torch.float32).to(self.device)

                # 创建 mask
                mask = self.model.create_mask(bs, X_t.size(1))

                # 前向
                outputs = self.model(X_t, mask)

                # 计算损失
                recon_loss = self._compute_recon_loss(outputs, X_t, mask)
                contrast_loss = self._compute_contrastive_loss(outputs, bs)
                loss = recon_loss + self.contrastive_weight * contrast_loss

                # 反向
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                total_loss += loss.item() * bs
                n_samples += bs

            self.scheduler.step()
            avg_loss = total_loss / max(n_samples, 1)
            self.training_history['loss'].append(avg_loss)

            if epoch % 20 == 0:
                logger.info(f"[SelfSupervised] Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                self._save_best()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"[SelfSupervised] Early stopping at epoch {epoch}")
                    break

        self.trained = True
        logger.info(f"[SelfSupervised] 预训练完成, best_loss={best_loss:.4f}")
        return self.training_history

    def get_pretrained_weights(self) -> Dict:
        """获取预训练权重 (用于迁移学习)"""
        return self.model.state_dict()

    def _save_best(self):
        """保存最佳模型"""
        path = os.path.join(self.model_dir, 'selfsupervised_best.pth')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'device': str(self.device),
            'n_features': self.n_features,
            'd_model': self.d_model,
            'n_layers': len(self.model.encoder.layers),
        }, path)

    def load(self, path: Optional[str] = None) -> bool:
        """加载预训练权重"""
        if path is None:
            path = os.path.join(self.model_dir, 'selfsupervised_best.pth')
        if not os.path.exists(path):
            return False

        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.trained = True
            return True
        except Exception as e:
            logger.error(f"[SelfSupervised] 加载失败: {e}")
            return False
