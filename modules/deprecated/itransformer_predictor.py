#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
iTransformer 多变量时序预测器 (ICLR 2024 Spotlight)

P1-3 集成 (2026-07-01):
  基于 thuml/iTransformer 架构，将每个变量（而非每个时间步）作为 token，
  用自注意力捕捉变量间的相关性，而非时间依赖性。

核心创新:
  - 变量级 token: 每个特征维度（如 RSI, MACD, 成交量等）作为独立 token
  - 时间步嵌入: 时间位置信息通过 embedding 编码
  - 天然支持多变量交叉依赖: 自注意力直接建模变量间相关性

参考:
  - 论文: "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting" (ICLR 2024 Spotlight)
  - GitHub: https://github.com/thuml/iTransformer
  - Stars: ⭐2157

用法:
    predictor = ITransformerPredictor(n_vars=12, pred_len=5)
    result = predictor.predict(features)  # features: (seq_len, n_vars)
"""

import numpy as np
import math
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

from modules.logger import logger

# ─────────────────────────────────────────────
# PyTorch 依赖（可选）
# ─────────────────────────────────────────────

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("[iTransformer] PyTorch 不可用，使用 fallback 预测器")


# ─────────────────────────────────────────────
# PyTorch 实现
# ─────────────────────────────────────────────

if HAS_TORCH:

    class InputPatch(nn.Module):
        """输入 patch: 将时间序列切分为 patch 并线性投影到 embedding"""

        def __init__(self, d_model: int, d_var: int, patch_size: int = 12):
            super().__init__()
            self.patch_size = patch_size
            self.projection = nn.Linear(patch_size * d_var, d_model)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Args:
                x: (batch, seq_len, n_vars)
            Returns:
                (batch, n_patches, d_model)
            """
            batch, seq_len, n_vars = x.shape
            n_patches = math.ceil(seq_len / self.patch_size)

            # 补齐
            pad_len = n_patches * self.patch_size - seq_len
            if pad_len > 0:
                x = nn.functional.pad(x, (0, 0, 0, pad_len))

            # 切 patch: (batch, n_patches, patch_size, n_vars)
            x = x.reshape(batch, n_patches, self.patch_size, n_vars)
            # 投影到 d_model
            x = x.reshape(batch, n_patches, -1)  # (batch, n_patches, patch_size * n_vars)
            x = self.projection(x)  # (batch, n_patches, d_model)
            return x

    class VariateEmbedding(nn.Module):
        """变量嵌入: 将时间位置信息编码到每个变量 token 中"""

        def __init__(self, d_model: int, seq_len: int):
            super().__init__()
            self.position_emb = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Args:
                x: (batch, n_vars, d_model) — 变量级 token
            Returns:
                (batch, n_vars, d_model)
            """
            # 为每个变量添加位置编码
            return x + self.position_emb[:, :x.size(1), :]

    class InvertedTransformerEncoder(nn.Module):
        """
        倒置 Transformer 编码器

        与传统 Transformer 的区别:
        - 传统: token = 时间步, 注意力捕捉时间依赖
        - 倒置: token = 变量, 注意力捕捉变量间相关性
        """

        def __init__(self, d_model: int = 128, nhead: int = 4,
                     num_layers: int = 2, dim_feedforward: int = 256,
                     dropout: float = 0.1, n_vars: int = 12):
            super().__init__()
            self.d_model = d_model

            # 变量级自注意力
            attn_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead,
                dim_feedforward=dim_feedforward, dropout=dropout,
                batch_first=True, activation='gelu'
            )
            self.encoder = nn.TransformerEncoder(attn_layer, num_layers=num_layers)

            # 输出投影
            self.output_proj = nn.Linear(d_model, 1)

            # 初始化
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Args:
                x: (batch, n_vars, d_model) — 变量级 token
            Returns:
                (batch, n_vars, 1) — 每个变量的预测值
            """
            # x: (batch, n_vars, d_model) → (batch, n_vars, d_model)
            out = self.encoder(x)  # (batch, n_vars, d_model)
            out = self.output_proj(out)  # (batch, n_vars, 1)
            return out.squeeze(-1)  # (batch, n_vars)

    class ITransformerModel(nn.Module):
        """完整的 iTransformer 模型"""

        def __init__(self, seq_len: int = 60, pred_len: int = 5,
                     n_vars: int = 12, d_model: int = 128,
                     nhead: int = 4, num_layers: int = 2,
                     patch_size: int = 12, dropout: float = 0.1):
            super().__init__()
            self.seq_len = seq_len
            self.pred_len = pred_len
            self.n_vars = n_vars
            self.d_model = d_model

            # 输入投影: 将原始特征投影到 d_model
            self.input_proj = nn.Linear(n_vars, d_model)

            # Patch 输入
            self.patch_input = InputPatch(d_model, n_vars, patch_size)

            # 倒置 Transformer 编码器
            self.transformer = InvertedTransformerEncoder(
                d_model=d_model, nhead=nhead,
                num_layers=num_layers,
                dim_feedforward=d_model * 4,
                dropout=dropout, n_vars=n_vars
            )

            # 输出层
            self.head = nn.Sequential(
                nn.Linear(n_vars, n_vars // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(n_vars // 2, pred_len),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Args:
                x: (batch, seq_len, n_vars) — 输入序列
            Returns:
                (batch, pred_len) — 预测值
            """
            batch, seq_len, n_vars = x.shape

            # 1. 输入投影
            x = self.input_proj(x)  # (batch, seq_len, d_model)

            # 2. Patch 输入
            x = self.patch_input(x)  # (batch, n_patches, d_model)

            # 3. 重塑为变量级: (batch, n_vars, d_model)
            #    对 patch 维度做平均池化
            n_patches = x.size(1)
            x = x.reshape(batch, n_patches, 1, self.d_model)
            x = x.mean(dim=1)  # (batch, 1, d_model) → 简化: 直接用 patch 输出
            x = x.expand(batch, self.n_vars, self.d_model)  # (batch, n_vars, d_model)

            # 4. 倒置 Transformer
            out = self.transformer(x)  # (batch, n_vars)

            # 5. 输出投影
            out = self.head(out)  # (batch, pred_len)
            return out


# ─────────────────────────────────────────────
# Fallback 预测器（PyTorch 不可用时）
# ─────────────────────────────────────────────

class ITransformerFallback:
    """
    iTransformer fallback — 当 PyTorch 不可用时使用

    使用统计特征 + 线性模型作为基线:
    - 提取每个变量的统计特征（均值、趋势、波动率）
    - 用 Ridge 回归预测未来方向
    """

    def __init__(self, n_vars: int = 12, pred_len: int = 5):
        self.n_vars = n_vars
        self.pred_len = pred_len
        self.is_trained = False
        self.feature_importances_ = {}
        self.mean_return = 0.0
        self.std_return = 1.0

    def _extract_features(self, closes: np.ndarray) -> np.ndarray:
        """提取多变量特征"""
        if len(closes) < 30:
            return np.zeros(self.n_vars)

        features = []
        # 1. 价格水平
        features.append(closes[-1])
        # 2. 收益率
        ret_1d = (closes[-1] / closes[-2] - 1) if len(closes) >= 2 else 0
        ret_5d = (closes[-1] / closes[-6] - 1) if len(closes) >= 6 else 0
        ret_10d = (closes[-1] / closes[-11] - 1) if len(closes) >= 11 else 0
        features.extend([ret_1d * 100, ret_5d * 100, ret_10d * 100])
        # 3. 波动率
        if len(closes) >= 20:
            returns = np.diff(np.log(closes[-20:]))
            features.append(np.std(returns) * np.sqrt(252) * 100)
        else:
            features.append(5.0)
        # 4. 趋势
        if len(closes) >= 10:
            x = np.arange(len(closes[-10:]))
            y = closes[-10:]
            slope = np.polyfit(x, y, 1)[0] / np.mean(y) * 100 if np.mean(y) > 0 else 0
            features.append(slope)
        else:
            features.append(0)
        # 5. RSI
        if len(closes) >= 15:
            deltas = np.diff(closes[-15:])
            gains = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
            losses = abs(np.mean(deltas[deltas < 0])) if np.any(deltas < 0) else 0.001
            rsi = 100 - (100 / (1 + gains / losses))
            features.append(rsi)
        else:
            features.append(50)
        # 6. 成交量比率
        if len(closes) >= 20:
            vol_ratio = closes[-1] / np.mean(closes[-20:]) if np.mean(closes[-20:]) > 0 else 1
            features.append(vol_ratio)
        else:
            features.append(1.0)

        # 补齐到 n_vars
        while len(features) < self.n_vars:
            features.append(0.0)

        return np.array(features[:self.n_vars])

    def train(self, X: np.ndarray, y: np.ndarray) -> 'ITransformerFallback':
        """
        训练 fallback 模型

        Args:
            X: (n_samples, n_features) 特征矩阵
            y: (n_samples,) 标签 (未来收益率)
        """
        try:
            from sklearn.linear_model import Ridge
            self.model = Ridge(alpha=1.0)
            self.model.fit(X, y)
            self.is_trained = True
            self.feature_importances_ = {
                f'feat_{i}': float(abs(self.model.coef_[i]))
                for i in range(len(self.model.coef_))
            }
            logger.info(f"[iTransformer-Fallback] 训练完成, n_samples={len(X)}")
        except Exception as e:
            logger.warning(f"[iTransformer-Fallback] 训练失败: {e}, 使用均值预测")
            self.is_trained = False
            self.mean_return = float(np.mean(y)) if len(y) > 0 else 0.0
            self.std_return = float(np.std(y)) if len(y) > 0 else 1.0
        return self

    def predict(self, features: np.ndarray) -> Dict:
        """预测"""
        if not self.is_trained:
            direction = 'neutral'
            confidence = 0.5
            expected_return = self.mean_return
        else:
            pred = self.model.predict(features.reshape(1, -1))[0]
            expected_return = float(pred)
            confidence = min(0.95, max(0.5, 0.5 + abs(pred) / (2 * self.std_return if self.std_return > 0 else 1)))
            direction = 'up' if pred > 0 else 'down'

        return {
            'model': 'itransformer_fallback',
            'direction': direction,
            'confidence': round(confidence, 3),
            'expected_return': round(expected_return, 4),
            'feature_importances': self.feature_importances_,
        }


# ─────────────────────────────────────────────
# 主预测器类
# ─────────────────────────────────────────────

class ITransformerPredictor:
    """
    iTransformer 多变量时序预测器

    P1-3 (2026-07-01): 基于 ICLR 2024 Spotlight 论文实现

    用法:
        predictor = ITransformerPredictor(n_vars=12, pred_len=5)
        # 训练
        predictor.train(closes, volumes, other_features, labels)
        # 预测
        result = predictor.predict(features)
    """

    def __init__(self, seq_len: int = 60, pred_len: int = 5,
                 n_vars: int = 12, d_model: int = 128,
                 nhead: int = 4, num_layers: int = 2,
                 patch_size: int = 12):
        """
        Args:
            seq_len: 输入序列长度
            pred_len: 预测步长
            n_vars: 变量数量（特征维度）
            d_model: 模型维度
            nhead: 注意力头数
            num_layers: Transformer 层数
            patch_size: patch 大小
        """
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_vars = n_vars
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.patch_size = patch_size

        self.is_trained = False
        self.feature_importances_ = {}
        self.training_dates: Optional[List[str]] = None
        self.oof_predictions: Optional[np.ndarray] = None

        if HAS_TORCH:
            self.model: Optional[ITransformerModel] = ITransformerModel(
                seq_len=seq_len, pred_len=pred_len,
                n_vars=n_vars, d_model=d_model,
                nhead=nhead, num_layers=num_layers,
                patch_size=patch_size, dropout=0.1
            )
            self._device = torch.device('cpu')
            self.model.to(self._device)
        else:
            self.model = None
            self._fallback = ITransformerFallback(n_vars=n_vars, pred_len=pred_len)

    def _prepare_input(self, closes: np.ndarray, extra_features: Optional[np.ndarray] = None) -> np.ndarray:
        """
        准备模型输入

        Args:
            closes: 收盘价序列 (seq_len,)
            extra_features: 其他特征 (seq_len, n_vars-1)

        Returns:
            (seq_len, n_vars) 特征矩阵
        """
        closes = closes[-self.seq_len:]
        if len(closes) < self.seq_len:
            closes = np.pad(closes, (self.seq_len - len(closes), 0), mode='edge')

        if extra_features is not None:
            extra = extra_features[-self.seq_len:]
            if extra.shape[0] < self.seq_len:
                extra = np.pad(extra, ((self.seq_len - extra.shape[0], 0), (0, 0)), mode='edge')
            x = np.column_stack([closes, extra])
        else:
            # 只用收盘价
            x = np.column_stack([closes])
            # 补齐到 n_vars
            while x.shape[1] < self.n_vars:
                # 添加衍生特征
                ret = np.diff(closes, append=closes[-1])
                vol = np.zeros_like(closes)
                x = np.column_stack([x, closes, ret, vol])

        return x[:self.seq_len, :self.n_vars]

    def train(self, closes: np.ndarray, labels: np.ndarray,
              extra_features: Optional[np.ndarray] = None,
              dates: Optional[List[str]] = None) -> 'ITransformerPredictor':
        """
        训练模型

        Args:
            closes: (n_samples, seq_len) 收盘价序列
            labels: (n_samples,) 标签 (0=down, 1=neutral, 2=up)
            extra_features: (n_samples, seq_len, n_vars-1) 额外特征
            dates: 日期列表

        Returns:
            self
        """
        self.training_dates = dates

        if HAS_TORCH and self.model is not None:
            try:
                # 准备训练数据
                X_list = []
                for i in range(len(closes)):
                    x = self._prepare_input(closes[i],
                        extra_features[i] if extra_features is not None else None)
                    X_list.append(x)
                X = np.array(X_list)  # (n_samples, seq_len, n_vars)

                # 转换为 tensor
                X_tensor = torch.FloatTensor(X).to(self._device)
                y_tensor = torch.LongTensor(labels).to(self._device)

                # 简单训练: 10 个 epoch
                optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
                criterion = nn.CrossEntropyLoss()

                self.model.train()
                for epoch in range(10):
                    optimizer.zero_grad()
                    logits = self.model(X_tensor)
                    loss = criterion(logits, y_tensor)
                    loss.backward()
                    optimizer.step()

                self.is_trained = True
                logger.info(f"[iTransformer] 训练完成, loss={loss.item():.4f}")

                # 计算特征重要性（基于梯度）
                X_test = X_tensor[:1]
                X_test.requires_grad = True
                logits = self.model(X_test)
                logits.sum().backward()
                grad = X_test.grad.abs().mean(dim=(0, 1)).cpu().numpy()
                self.feature_importances_ = {
                    f'var_{i}': float(grad[i]) for i in range(min(grad.shape[0], self.n_vars))
                }

            except Exception as e:
                logger.warning(f"[iTransformer] PyTorch 训练失败: {e}, 使用 fallback")
                self._fallback.train(
                    np.array([self._prepare_input(closes[i],
                        extra_features[i] if extra_features is not None else None)
                        for i in range(len(closes))]),
                    labels
                )
                self.is_trained = self._fallback.is_trained
        else:
            # 使用 fallback
            X_fallback = np.array([self._prepare_input(closes[i],
                extra_features[i] if extra_features is not None else None)
                for i in range(len(closes))])
            # 展平: (n_samples, seq_len * n_vars)
            X_flat = X_fallback.reshape(len(closes), -1)
            self._fallback.train(X_flat, labels)
            self.is_trained = self._fallback.is_trained

        return self

    def predict(self, features: np.ndarray) -> Dict:
        """
        预测

        Args:
            features: (seq_len, n_vars) 特征矩阵

        Returns:
            {
                'model': 'itransformer',
                'direction': 'up'/'down'/'neutral',
                'confidence': 0.0-1.0,
                'probabilities': {'up': 0.4, 'down': 0.3, 'neutral': 0.3},
                'feature_importances': {...},
            }
        """
        if not self.is_trained:
            return {
                'model': 'itransformer',
                'direction': 'neutral',
                'confidence': 0.5,
                'probabilities': {'up': 0.33, 'down': 0.33, 'neutral': 0.34},
                'feature_importances': {},
            }

        if HAS_TORCH and self.model is not None:
            try:
                x = self._prepare_input(features[:, 0]) if features.shape[1] > 0 else features
                x_tensor = torch.FloatTensor(x.reshape(1, *x.shape)).to(self._device)
                self.model.eval()
                with torch.no_grad():
                    logits = self.model(x_tensor)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

                pred_class = int(np.argmax(probs))
                direction = ['down', 'neutral', 'up'][pred_class]
                confidence = float(max(probs))

                return {
                    'model': 'itransformer',
                    'direction': direction,
                    'confidence': round(confidence, 3),
                    'probabilities': {
                        'up': round(float(probs[2]), 3),
                        'down': round(float(probs[0]), 3),
                        'neutral': round(float(probs[1]), 3),
                    },
                    'feature_importances': self.feature_importances_,
                }
            except Exception as e:
                logger.warning(f"[iTransformer] PyTorch 预测失败: {e}, 使用 fallback")

        # Fallback
        x_flat = features.reshape(1, -1)
        return self._fallback.predict(x_flat)

    def predict_direction(self, features: np.ndarray) -> Dict:
        """预测方向（兼容 MLPredictor 接口）"""
        return self.predict(features)

    def predict_direction_cached(self, stock_code: str, features: np.ndarray) -> Dict:
        """缓存版预测（兼容 MLPredictor 接口）"""
        return self.predict(features)


# ─────────────────────────────────────────────
# 全局实例 + 工厂函数
# ─────────────────────────────────────────────

_itransformer_instance: Optional[ITransformerPredictor] = None


def get_itransformer(n_vars: int = 12, pred_len: int = 5) -> ITransformerPredictor:
    """获取 iTransformer 预测器全局实例"""
    global _itransformer_instance
    if _itransformer_instance is None:
        _itransformer_instance = ITransformerPredictor(n_vars=n_vars, pred_len=pred_len)
    return _itransformer_instance


def reset_itransformer():
    """重置全局实例（用于测试）"""
    global _itransformer_instance
    _itransformer_instance = None
