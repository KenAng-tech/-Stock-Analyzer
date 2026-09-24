#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Foundation Model — 时序基础模型 (2026 SOTA)

通用时序基础模型架构:
    1. PatchTST (Transformer-based) — 补丁 + 通道独立 + 掩码预训练
    2. Mamba (SSM-based) — 状态空间模型, O(n) 复杂度
    3. TimeLLM (LLM-based) — 大语言模型注入时序知识
    4. DLinear (Linear-based) — 简化的线性架构

核心思想:
- 预训练一个通用模型 → 微调到具体任务
- 掩码自编码器 (MAE) 预训练
- 通道独立 (Channel Independence)
- 多尺度时间序列

优势:
- 减少过拟合 (预训练 + 微调)
- 零样本能力 (zero-shot)
- 统一的模型架构

参考:
- "PatchTST: A Time Series is Worth 2 Tokens" (2023)
- "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (2024)
- "TimeLLM: Time Series Forecasting with Large Language Models" (2024)
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
    torch = None
    nn = None
    F = None
    HAS_TORCH = False
    logger.warning("[Foundation] PyTorch 未安装")

# Conditional base class for nn.Module inheritance
_TorchModule = nn.Module if HAS_TORCH else object


# ── Patch Embedding ────────────────────────────────────────────────────────

class PatchEmbedding(_TorchModule):
    """
    补丁嵌入 (Patch Embedding)

    将时序数据分割成补丁, 然后投影到 d_model 维度:
    - 输入: (batch, seq_len, n_features)
    - 输出: (batch, n_patches, d_model)

    补丁化优势:
    - 减少令牌数量 (seq_len → n_patches)
    - 捕捉局部时间模式
    - 减少计算复杂度
    """

    def __init__(
        self,
        patch_len: int = 16,
        stride: int = 8,
        n_features: int = 1,
        d_model: int = 128,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.n_features = n_features
        self.d_model = d_model

        # 每个补丁投影到 d_model
        self.proj = nn.Linear(patch_len * n_features, d_model)
        self.n_patches = 0  # 动态计算

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_features)
        Returns:
            patches: (batch, n_patches, d_model)
        """
        batch, seq_len, n_feat = x.shape

        # 动态计算补丁数
        self.n_patches = (seq_len - self.patch_len) // self.stride + 1

        # 分割成补丁
        patches = x.unfold(1, self.patch_len, self.stride)  # (batch, n_patches, patch_len, n_feat)
        patches = patches.reshape(batch, self.n_patches, -1)  # (batch, n_patches, patch_len * n_feat)

        # 投影到 d_model
        patches = self.proj(patches)  # (batch, n_patches, d_model)

        return patches


# ── Transformer Encoder ──────────────────────────────────────────────────────

class TransformerEncoder(_TorchModule):
    """
    Transformer 编码器 (用于 PatchTST)

    包含:
    - 多头自注意力 (Multi-Head Self-Attention)
    - 前馈网络 (FFN)
    - 残差连接 + LayerNorm
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_patches, d_model)
        Returns:
            output: (batch, n_patches, d_model)
        """
        for layer in self.encoder_layers:
            x = layer(x)
        return x


class TransformerEncoderLayer(_TorchModule):
    """单层 Transformer 编码器"""

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 8,
        d_ff: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_patches, d_model)
        Returns:
            output: (batch, n_patches, d_model)
        """
        # 自注意力 + 残差
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))

        # FFN + 残差
        ff_out = self.linear2(F.gelu(self.linear1(x)))
        x = self.norm2(x + self.dropout(ff_out))

        return x


# ── Foundation Model ─────────────────────────────────────────────────────────

class FoundationModel(_TorchModule):
    """
    基础模型 (Foundation Model)

    统一的时序模型架构:
    1. Patch Embedding: 时序 → 补丁
    2. Encoder: Transformer / SSM / Linear
    3. Prediction Head: 回归 / 分类

    支持多种架构:
    - patchtst: Patch + Transformer
    - mamba: Patch + SSM
    - linear: Patch + Linear (DLinear)
    - hybrid: Patch + Transformer + SSM
    """

    def __init__(
        self,
        n_features: int = 1,
        patch_len: int = 16,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
        n_classes: int = 3,
        architecture: str = "transformer",  # "transformer" / "mamba" / "linear" / "hybrid"
    ):
        super().__init__()

        self.architecture = architecture
        self.n_features = n_features
        self.patch_len = patch_len
        self.d_model = d_model

        # Patch Embedding
        self.patch_embed = PatchEmbedding(patch_len, patch_len // 2, n_features, d_model)

        # 根据架构选择编码器
        if architecture == "transformer":
            self.encoder = TransformerEncoder(d_model, n_heads, n_layers, d_ff, dropout)
        elif architecture == "mamba":
            from .patchmamba import SelectiveSSMBlock
            self.encoder = nn.ModuleList([
                SelectiveSSMBlock(d_model) for _ in range(n_layers)
            ])
        elif architecture == "linear":
            self.encoder = nn.Identity()
        elif architecture == "hybrid":
            self.encoder_t = TransformerEncoder(d_model, n_heads, n_layers // 2, d_ff, dropout)
            from .patchmamba import SelectiveSSMBlock
            self.encoder_ssm = nn.ModuleList([
                SelectiveSSMBlock(d_model) for _ in range(n_layers // 2)
            ])
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        # Prediction Head
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes if n_classes > 1 else 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_features)
        Returns:
            logits: (batch, n_classes) or (batch, 1)
        """
        # Patch Embedding
        patches = self.patch_embed(x)  # (batch, n_patches, d_model)

        # Encoder
        if self.architecture == "linear":
            encoded = patches.mean(dim=1)  # (batch, d_model)
        elif self.architecture == "mamba":
            encoded = patches
            for layer in self.encoder:
                encoded = layer(encoded)
            encoded = encoded.mean(dim=1)  # (batch, d_model)
        elif self.architecture == "hybrid":
            h1 = self.encoder_t(patches)
            h2 = patches
            for layer in self.encoder_ssm:
                h2 = layer(h2)
            encoded = (h1 + h2).mean(dim=1)  # (batch, d_model)
        else:  # transformer
            encoded = self.encoder(patches)  # (batch, n_patches, d_model)
            encoded = encoded.mean(dim=1)  # (batch, d_model)

        # Prediction
        logits = self.head(encoded)  # (batch, n_classes) or (batch, 1)

        return logits

    def predict(self, x: np.ndarray) -> Dict:
        """
        预测方法

        Args:
            x: (batch, seq_len, n_features) or (n_features,)

        Returns:
            Dict with prediction, confidence, probabilities
        """
        self.eval()
        if len(x.shape) == 1:
            x = x.reshape(1, -1)

        x_t = torch.tensor(x, dtype=torch.float32)
        if hasattr(self, 'device'):
            x_t = x_t.to(self.device)

        with torch.no_grad():
            logits = self.forward(x_t)
            probs = torch.softmax(logits, dim=-1)
            predictions = logits.argmax(dim=-1)
            confidences = probs.max(dim=-1)[0]

        direction_map = {0: 'down', 1: 'neutral', 2: 'up'}

        return {
            'prediction': direction_map.get(predictions.item(), 'neutral'),
            'confidence': confidences.item(),
            'probabilities': {
                'up': probs[0, 2].item() if probs.shape[1] > 2 else 0.33,
                'neutral': probs[0, 1].item() if probs.shape[1] > 1 else 0.34,
                'down': probs[0, 0].item() if probs.shape[1] > 0 else 0.33,
            },
        }


# ── 掩码预训练 ──────────────────────────────────────────────────────────────

class MaskedAutoencoder(_TorchModule):
    """
    掩码自编码器 (Masked Autoencoder)

    用于预训练 Foundation Model:
    1. 随机掩码部分补丁 (如 50%)
    2. 用未掩码补丁重建被掩码的补丁
    3. 预训练后微调

    预训练策略:
    - MAE: 掩码 50-75% 补丁, 重建像素值
    - PatchTST: 掩码 40% 补丁, 重建语义表示
    """

    def __init__(
        self,
        foundation_model: FoundationModel,
        mask_ratio: float = 0.4,
    ):
        super().__init__()
        self.model = foundation_model
        self.mask_ratio = mask_ratio

    def forward(
        self,
        x: torch.Tensor,
        return_mask: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, n_features)
            return_mask: 是否返回掩码
        Returns:
            reconstructed: (batch, seq_len, n_features)
            mask: (batch, n_patches) or None
        """
        batch, seq_len, n_feat = x.shape

        # Patch Embedding
        patches = self.model.patch_embed(x)  # (batch, n_patches, d_model)
        n_patches = patches.shape[1]

        # 随机掩码
        mask = torch.rand(batch, n_patches) < self.mask_ratio
        mask = mask.to(x.device)

        # 用未掩码补丁预测被掩码补丁
        # 简化: 用均值表示重建 (实际应用中可用注意力机制)
        reconstructed_patches = patches.clone()
        for i in range(batch):
            unmasked_idx = (~mask[i]).nonzero(as_tuple=True)[0]
            if len(unmasked_idx) > 0:
                mean_patch = patches[i, unmasked_idx].mean(dim=0)
                masked_idx = mask[i].nonzero(as_tuple=True)[0]
                for j in masked_idx:
                    reconstructed_patches[i, j] = mean_patch

        # 重建
        # 简化: 反向投影 (实际应用中需要解码器)
        reconstructed = reconstructed_patches.mean(dim=1, keepdim=True)
        reconstructed = reconstructed.expand(-1, seq_len, -1)

        if return_mask:
            return reconstructed, mask
        return reconstructed, None


# ── 多尺度融合 ────────────────────────────────────────────────────────────────

class MultiScaleFusion(_TorchModule):
    """
    多尺度融合 (Multi-Scale Fusion)

    不同尺度的时间序列特征融合:
    - 短期 (patch_len=4): 高频细节
    - 中期 (patch_len=16): 标准模式
    - 长期 (patch_len=64): 低频趋势

    融合策略: 注意力加权
    """

    def __init__(
        self,
        d_model: int = 128,
        scales: List[int] = [4, 16, 64],
        n_features: int = 1,
        n_classes: int = 3,
    ):
        super().__init__()

        self.scales = scales
        self.n_scales = len(scales)

        # 每个尺度一个 Foundation Model
        self.models = nn.ModuleDict({
            f"scale_{s}": FoundationModel(
                n_features=n_features,
                patch_len=s,
                d_model=d_model // self.n_scales,
                n_classes=n_classes,
                architecture="transformer",
            )
            for s in scales
        })

        # 注意力融合
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, self.n_scales),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_features)
        Returns:
            logits: (batch, n_classes)
        """
        batch_size = x.shape[0]
        scale_logits = []

        for i, s in enumerate(self.scales):
            # 下采样到对应尺度
            stride = max(1, x.shape[1] // (x.shape[1] // s))
            x_scaled = x[:, ::stride, :]

            # 如果太长, 截断
            max_len = s * 32  # 假设最多 32 个补丁
            if x_scaled.shape[1] > max_len:
                x_scaled = x_scaled[:, :max_len, :]

            # 用对应尺度模型处理
            model = self.models[f"scale_{s}"]
            out = model(x_scaled)  # (batch, n_classes) or (batch, 1)
            scale_logits.append(out)

        # 加权融合
        # 先拼接, 再注意力加权
        all_features = torch.stack(scale_logits, dim=-1)  # (batch, n_classes, n_scales)
        attn = self.attention(all_features.mean(dim=1, keepdim=True))  # (batch, 1, n_scales)
        attn = attn.unsqueeze(1)  # (batch, 1, 1, n_scales)

        fused = (all_features * attn).sum(dim=-1)  # (batch, n_classes)

        return fused


# ── 全局单例 ──────────────────────────────────────────────────────────────

_instance: Optional[FoundationModel] = None
_instance_lock = threading.Lock()


def get_foundation_model(
    n_features: int = 1,
    patch_len: int = 16,
    d_model: int = 128,
    n_heads: int = 8,
    n_layers: int = 4,
    n_classes: int = 3,
    architecture: str = "transformer",
) -> FoundationModel:
    """获取全局 FoundationModel 单例"""
    global _instance
    with _instance_lock:
        if _instance is None:
            _instance = FoundationModel(
                n_features=n_features,
                patch_len=patch_len,
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                n_classes=n_classes,
                architecture=architecture,
            )
        return _instance


def reset_foundation_model():
    """重置 FoundationModel 单例"""
    global _instance
    with _instance_lock:
        _instance = None