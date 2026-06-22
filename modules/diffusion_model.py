#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Diffusion Model 概率预测模块 — 2026 SOTA

功能:
1. 使用 Score-Based Diffusion Model 进行概率预测
2. 输出预测的不确定性区间 (置信区间)
3. 适合风险管理 (知道"我不确定"比"我确定"更重要)

架构:
    Input (时序特征) → UNet (Score Network) → 噪声预测
    采样: DDPM / DDIM → 概率分布 → 置信区间

参考:
    - Ho et al., "Denoising Diffusion Probabilistic Models" (2020)
    - Song et al., "Score-Based Generative Modeling" (2021)
    - 金融应用: 股票预测的不确定性量化

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
    logger.warning("[Diffusion] PyTorch 未安装")

# Conditional base class for ScoreNetwork
UNetBase = nn.Module if HAS_TORCH else object


# ── 噪声调度器 ──────────────────────────────────────────────

class NoiseScheduler:
    """
    噪声调度器 — 管理扩散过程的噪声水平

    支持:
        - linear noise schedule (DDPM)
        - cosine noise schedule (更平滑)
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        schedule_type: str = 'linear',
    ):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.schedule_type = schedule_type

        # 预计算噪声调度
        self.betas = self._get_betas()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # 计算方差
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        # 后验方差
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance[1:], min=1e-20)
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )

    def _get_betas(self) -> torch.Tensor:
        """获取 beta 调度"""
        if self.schedule_type == 'linear':
            return torch.linspace(self.beta_start, self.beta_end, self.num_timesteps)
        elif self.schedule_type == 'cosine':
            t = torch.arange(self.num_timesteps + 1)
            alphas_cumprod = torch.cos((t / self.num_timesteps + 0.008) * np.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clamp(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")


# ── UNet Score Network ───────────────────────────────────────

class ScoreNetwork(UNetBase):
    """
    UNet Score Network — 预测噪声

    架构:
        Input → Encoder (Downsampling) → Middle → Decoder (Upsampling) → Output
    """

    def __init__(
        self,
        dim: int = 64,
        dim_mults: Tuple[int] = (1, 2, 4, 8),
        time_dim: int = 256,
        out_dim: int = 1,
    ):
        super().__init__()
        self.dim = dim
        self.time_dim = time_dim

        # 时间嵌入
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # 初始卷积
        self.init_conv = nn.Conv1d(1, dim, 3, padding=1)

        # Encoder
        self.downs = nn.ModuleList()
        current_dim = dim
        for mult in dim_mults:
            self.downs.append(nn.ModuleList([
                nn.Sequential(
                    nn.Conv1d(current_dim, current_dim * mult, 3, padding=1),
                    nn.GroupNorm(8, current_dim * mult),
                    nn.SiLU(),
                ),
                nn.Sequential(
                    nn.Conv1d(current_dim * mult, current_dim * mult, 3, padding=1),
                    nn.GroupNorm(8, current_dim * mult),
                    nn.SiLU(),
                ),
            ]))
            current_dim = current_dim * mult

        # Middle
        self.middle = nn.Sequential(
            nn.Conv1d(current_dim, current_dim, 3, padding=1),
            nn.GroupNorm(8, current_dim),
            nn.SiLU(),
            nn.Conv1d(current_dim, current_dim, 3, padding=1),
            nn.GroupNorm(8, current_dim),
            nn.SiLU(),
        )

        # Decoder
        self.ups = nn.ModuleList()
        for mult in reversed(dim_mults):
            self.ups.append(nn.ModuleList([
                nn.Sequential(
                    nn.Conv1d(current_dim, current_dim // mult, 3, padding=1),
                    nn.GroupNorm(8, current_dim // mult),
                    nn.SiLU(),
                ),
                nn.Sequential(
                    nn.Conv1d(current_dim // mult, current_dim // mult, 3, padding=1),
                    nn.GroupNorm(8, current_dim // mult),
                    nn.SiLU(),
                ),
            ]))
            current_dim = current_dim // mult

        # 输出
        self.final_conv = nn.Conv1d(dim, out_dim, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 1, seq_len) 含噪声的信号
            t: (batch, 1) 时间步

        Returns:
            (batch, 1, seq_len) 预测的噪声
        """
        # 时间嵌入
        t = t.float()
        t = self.time_mlp(t)

        # 初始卷积
        h = self.init_conv(x)

        # Encoder
        for resnet, downsample in self.downs:
            h = resnet(h)
            h = downsample(h)

        # Middle
        h = self.middle(h)

        # Decoder
        for resnet, upsample in self.ups:
            h = resnet(h)
            h = upsample(h)

        # 输出
        return self.final_conv(h)


# ── Diffusion 预测器 ──────────────────────────────────────────

class DiffusionPredictor:
    """
    Diffusion Model 概率预测器

    功能:
        - 训练: 学习 score-based model
        - 采样: DDPM / DDIM 采样
        - 预测: 输出均值 + 置信区间
    """

    def __init__(
        self,
        seq_len: int = 60,
        n_features: int = 12,
        hidden_dim: int = 64,
        num_timesteps: int = 500,
        sampling_steps: int = 50,
    ):
        """
        Args:
            seq_len: 序列长度
            n_features: 特征维度
            hidden_dim: UNet 隐藏维度
            num_timesteps: 扩散步数
            sampling_steps: 采样步数 (越小越快，越大越平滑)
        """
        self.seq_len = seq_len
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.num_timesteps = num_timesteps
        self.sampling_steps = sampling_steps

        self.device = None
        self.model = None
        self.scheduler = None
        self.trained = False
        self._lock = threading.Lock()

        self.model_dir = os.path.join(os.path.dirname(__file__), 'dl_models')
        os.makedirs(self.model_dir, exist_ok=True)

        if HAS_TORCH:
            self._init_model()
        else:
            logger.warning("[Diffusion] PyTorch 不可用，使用高斯 fallback")

    def is_trained(self) -> bool:
        """检查模型是否已训练"""
        return self.trained

    def _init_model(self):
        """初始化模型"""
        self.device = self._get_device()
        self.model = ScoreNetwork(
            dim=self.hidden_dim,
            dim_mults=(1, 2, 4),
            out_dim=self.n_features,
        )
        self.model.to(self.device)
        self.scheduler = NoiseScheduler(num_timesteps=self.num_timesteps)

    def _get_device(self) -> torch.device:
        """获取设备"""
        if torch.cuda.is_available():
            return torch.device('cuda')
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
    ) -> Dict:
        """
        训练 Diffusion Model

        Args:
            X_train: (n_samples, seq_len, n_features) 特征
            y_train: (n_samples,) 标签 (0=down, 1=neutral, 2=up)
            X_val: 可选验证特征
            y_val: 可选验证标签
            epochs: 训练轮数
            batch_size: 批大小
            learning_rate: 学习率

        Returns:
            training_history: {'loss': [], 'val_loss': []}
        """
        if not HAS_TORCH:
            logger.error("[Diffusion] PyTorch 不可用，无法训练")
            return {'loss': [], 'val_loss': []}

        if X_train.shape[0] < 50:
            logger.warning(f"[Diffusion] 训练样本不足 ({X_train.shape[0]})")
            return {'loss': [], 'val_loss': []}

        logger.info(f"[Diffusion] 开始训练: {X_train.shape[0]} 样本")

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        history = {'loss': [], 'val_loss': []}

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

                # 采样时间步
                t = torch.randint(0, self.num_timesteps, (len(X_batch), 1, 1), device=self.device)
                t_float = t.float() / self.num_timesteps

                # 添加噪声
                noise = torch.randn_like(X_batch)
                noisy_x = torch.sqrt(self.scheduler.alphas_cumprod[t][:, None, None]) * X_batch + \
                          torch.sqrt(1 - self.scheduler.alphas_cumprod[t][:, None, None]) * noise

                # 预测噪声
                predicted_noise = self.model(noisy_x, t_float)

                # 损失: 噪声预测误差
                loss = F.mse_loss(predicted_noise, noise)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_train_loss = train_loss / n_batches
            history['loss'].append(avg_train_loss)

            # 验证
            if X_val is not None:
                val_loss = self._evaluate(X_val, y_val)
                history['val_loss'].append(val_loss)

            if epoch % 20 == 0 or epoch == epochs - 1:
                logger.info(
                    f"[Diffusion] Epoch {epoch}/{epochs} | "
                    f"Train Loss: {avg_train_loss:.4f} | "
                    f"Val Loss: {history['val_loss'][-1] if history['val_loss'] else 'N/A'}"
                )

        self.trained = True
        logger.info(f"[Diffusion] 训练完成")
        return history

    def _evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """评估模型"""
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y, dtype=torch.long).to(self.device)

        with torch.no_grad():
            t = torch.randint(0, self.num_timesteps, (len(X), 1, 1), device=self.device)
            t_float = t.float() / self.num_timesteps
            noise = torch.randn_like(X_t)
            noisy_x = torch.sqrt(self.scheduler.alphas_cumprod[t][:, None, None]) * X_t + \
                      torch.sqrt(1 - self.scheduler.alphas_cumprod[t][:, None, None]) * noise
            predicted_noise = self.model(noisy_x, t_float)
            loss = F.mse_loss(predicted_noise, noise).item()

        return loss

    def predict(self, X: np.ndarray, n_samples: int = 10) -> Dict:
        """
        概率预测

        Args:
            X: (n_samples, seq_len, n_features) 特征
            n_samples: 采样次数 (越多越准确，越慢)

        Returns:
            {
                'direction': 'up' | 'neutral' | 'down',
                'confidence': 0.0-1.0,
                'probabilities': {'up': float, 'neutral': float, 'down': float},
                'uncertainty': {'lower': float, 'upper': float, 'std': float},  # 置信区间
                'all_predictions': [str, ...] (如果 n_samples > 1)
            }
        """
        if not self.trained or not HAS_TORCH:
            # Fallback: 高斯分布
            probs = {'up': 0.33, 'neutral': 0.34, 'down': 0.33}
            return {
                'direction': 'neutral',
                'confidence': 0.34,
                'probabilities': probs,
                'uncertainty': {'lower': -0.3, 'upper': 0.3, 'std': 0.2},
            }

        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)

        # 多次采样估计不确定性
        all_probs = []

        with torch.no_grad():
            for _ in range(n_samples):
                # DDIM 采样 (快速)
                x = torch.randn_like(X_t)

                for i in reversed(range(0, self.num_timesteps, self.num_timesteps // self.sampling_steps)):
                    t = torch.full((len(x), 1, 1), i / self.num_timesteps, device=self.device)
                    t_float = t.float()

                    predicted_noise = self.model(x, t_float)

                    # 重建
                    alpha = self.scheduler.alphas_cumprod[i]
                    sqrt_one_minus_alpha = self.scheduler.sqrt_one_minus_alphas_cumprod[i]

                    x = (x - sqrt_one_minus_alpha * predicted_noise) / torch.sqrt(alpha)

                # 从最终状态提取预测
                logits = x.mean(dim=-1)  # (batch, n_features) -> 简化为标量
                probs.append(torch.softmax(logits.mean() * 10, dim=0).cpu().numpy())

        # 聚合
        probs_array = np.array(all_probs)
        mean_probs = probs_array.mean(axis=0)
        std_probs = probs_array.std(axis=0)

        # 方向判断
        direction_map = {0: 'down', 1: 'neutral', 2: 'up'}
        direction_idx = np.argmax(mean_probs)
        direction = direction_map[direction_idx]
        confidence = float(mean_probs[direction_idx])

        # 不确定性
        uncertainty = {
            'lower': float(np.percentile(probs_array[:, direction_idx], 5)),
            'upper': float(np.percentile(probs_array[:, direction_idx], 95)),
            'std': float(std_probs[direction_idx]),
        }

        return {
            'direction': direction,
            'confidence': confidence,
            'probabilities': {
                'up': float(mean_probs[2]),
                'neutral': float(mean_probs[1]),
                'down': float(mean_probs[0]),
            },
            'uncertainty': uncertainty,
            'n_samples': n_samples,
        }

    def save(self, path: Optional[str] = None):
        """保存模型"""
        if path is None:
            path = os.path.join(self.model_dir, 'diffusion_model.pth')
        if self.model is not None:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'device': str(self.device),
                'seq_len': self.seq_len,
                'n_features': self.n_features,
                'hidden_dim': self.hidden_dim,
                'num_timesteps': self.num_timesteps,
            }, path)
            logger.info(f"[Diffusion] 模型已保存: {path}")

    def load(self, path: Optional[str] = None) -> bool:
        """加载模型"""
        if path is None:
            path = os.path.join(self.model_dir, 'diffusion_model.pth')
        if not os.path.exists(path):
            return False

        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
            if self.model is None:
                self._init_model()
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.trained = True
            logger.info(f"[Diffusion] 模型已加载: {path}")
            return True
        except Exception as e:
            logger.error(f"[Diffusion] 加载失败: {e}")
            return False

    def get_status(self) -> Dict:
        """获取模型状态"""
        return {
            'trained': self.trained,
            'device': str(self.device) if self.device else 'N/A',
            'n_features': self.n_features,
            'hidden_dim': self.hidden_dim,
            'num_timesteps': self.num_timesteps,
        }


# ── 全局单例 ────────────────────────────────────────────────

_diffusion_instance: Optional[DiffusionPredictor] = None
_diffusion_lock = threading.Lock()


def get_diffusion_predictor() -> DiffusionPredictor:
    """获取全局 DiffusionPredictor 实例 (线程安全)"""
    global _diffusion_instance
    if _diffusion_instance is None:
        with _diffusion_lock:
            if _diffusion_instance is None:
                _diffusion_instance = DiffusionPredictor()
                _diffusion_instance.load()
    return _diffusion_instance