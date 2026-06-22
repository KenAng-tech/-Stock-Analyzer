#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
PatchTST 集成模块 — 替换无效的 dl_model_v2

功能:
1. 封装 PatchTST 模型为统一接口
2. 与现有 ml_predictor 集成
3. 支持 GPU/MPS/CPU 自动选择
4. 完整的训练/预测/持久化

作者: 基于 Zhou et al., "PatchTST" (ICML 2023)
参考: modules/patchtst_model.py
"""

import os
import pickle
import numpy as np
from typing import Dict, Optional, Tuple, List
import threading

from modules.logger import logger

# ── PyTorch 依赖检查 ─────────────────────────────────────────

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("[PatchTST] PyTorch 未安装，使用 fallback")


# ── PatchTST 封装 ────────────────────────────────────────────

class PatchTSTIntegrator:
    """
    PatchTST 集成器 — 统一接口

    接口设计:
        - train(X, y, X_val, y_val) → training_history
        - predict(X) → {'direction': str, 'confidence': float, 'probabilities': dict}
        - save() / load() → 持久化
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
        seq_len: int = 60,
    ):
        """
        Args:
            n_features: 输入特征维度
            patch_len: 每个 patch 的时间步数
            d_model: 模型维度
            n_layers: Transformer 层数
            n_heads: 注意力头数
            n_classes: 分类数 (3: up/neutral/down)
            dropout: Dropout 比例
            seq_len: 序列长度
        """
        self.n_features = n_features
        self.patch_len = patch_len
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_classes = n_classes
        self.dropout = dropout
        self.seq_len = seq_len

        self.model = None
        self.trained = False
        self.device = None
        self.training_history: Dict[str, List[float]] = {'loss': [], 'val_loss': [], 'val_acc': []}

        self._lock = threading.Lock()
        self.model_dir = os.path.join(os.path.dirname(__file__), 'dl_models')
        os.makedirs(self.model_dir, exist_ok=True)

        if HAS_TORCH:
            self._init_model()
        else:
            logger.warning("[PatchTST] PyTorch 不可用，使用随机预测 fallback")

    def _init_model(self):
        """初始化 PyTorch 模型"""
        self.device = self._get_device()
        logger.info(f"[PatchTST] 使用设备: {self.device}")

        # 构建模型 (延迟初始化)
        self.model = self._build_model()
        self.model.to(self.device)

    def _get_device(self) -> torch.device:
        """获取最佳设备"""
        if torch.cuda.is_available():
            return torch.device('cuda')
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')

    def _build_model(self) -> nn.Module:
        """构建 PatchTST 模型"""
        from modules.patchtst_model import PatchTST as PatchTSTModel
        return PatchTSTModel(
            n_features=self.n_features,
            patch_len=self.patch_len,
            d_model=self.d_model,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            n_classes=self.n_classes,
            dropout=self.dropout,
            max_seq_len=self.seq_len,
        )

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 100,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 20,
    ) -> Dict:
        """
        训练 PatchTST 模型

        Args:
            X_train: (n_samples, seq_len, n_features) 训练特征
            y_train: (n_samples,) 训练标签 (0=down, 1=neutral, 2=up)
            X_val: (n_val, seq_len, n_features) 验证特征
            y_val: (n_val,) 验证标签
            epochs: 训练轮数
            batch_size: 批大小
            learning_rate: 学习率
            weight_decay: 权重衰减
            patience: 早停轮数

        Returns:
            training_history: {'loss': [], 'val_loss': [], 'val_acc': []}
        """
        if not HAS_TORCH:
            logger.error("[PatchTST] PyTorch 不可用，无法训练")
            return self.training_history

        if X_train.shape[0] < 50:
            logger.warning(f"[PatchTST] 训练样本不足 ({X_train.shape[0]})，跳过训练")
            return self.training_history

        logger.info(f"[PatchTST] 开始训练: {X_train.shape[0]} 样本, {epochs} epochs")

        self.model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_val_loss = float('inf')
        patience_counter = 0
        self.training_history = {'loss': [], 'val_loss': [], 'val_acc': []}

        for epoch in range(epochs):
            # ── 训练 ──────────────────────────────────────────
            self.model.train()
            train_loss = 0.0
            n_batches = 0

            indices = np.random.permutation(len(X_train))
            for start in range(0, len(X_train), batch_size):
                end = min(start + batch_size, len(X_train))
                batch_idx = indices[start:end]

                X_batch = torch.tensor(X_train[batch_idx], dtype=torch.float32).to(self.device)
                y_batch = torch.tensor(y_train[batch_idx], dtype=torch.long).to(self.device)

                optimizer.zero_grad()
                logits = self.model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item() * len(batch_idx)
                n_batches += 1

            scheduler.step()
            avg_train_loss = train_loss / len(X_train)

            # ── 验证 ────────────────────────────────────────
            val_loss, val_acc = self._evaluate(X_val, y_val)

            self.training_history['loss'].append(avg_train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_acc'].append(val_acc)

            if epoch % 10 == 0 or epoch == epochs - 1:
                logger.info(
                    f"[PatchTST] Epoch {epoch}/{epochs} | "
                    f"Train Loss: {avg_train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val Acc: {val_acc:.4f}"
                )

            # ── 早停 ────────────────────────────────────────
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self._save_best()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"[PatchTST] Early stopping at epoch {epoch}")
                    break

        self.trained = True
        self._load_best()  # 恢复最佳模型
        logger.info(f"[PatchTST] 训练完成, best_val_loss={best_val_loss:.4f}")
        return self.training_history

    def _evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """评估模型"""
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y, dtype=torch.long).to(self.device)

        with torch.no_grad():
            logits = self.model(X_t)
            loss = nn.CrossEntropyLoss()(logits, y_t).item()
            acc = (logits.argmax(dim=-1) == y_t).float().mean().item()

        return loss, acc

    def predict(self, X: np.ndarray) -> Dict:
        """
        预测

        Args:
            X: (n_samples, seq_len, n_features) 或 (seq_len, n_features)

        Returns:
            {
                'direction': 'up' | 'neutral' | 'down',
                'confidence': 0.0-1.0,
                'probabilities': {'up': float, 'neutral': float, 'down': float},
                'all_predictions': [str, ...] (如果 n_samples > 1)
            }
        """
        if not self.trained:
            logger.warning("[PatchTST] 模型未训练，使用默认预测")
            return self._default_prediction()

        if not HAS_TORCH:
            return self._default_prediction()

        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits = self.model(X_t)
            probs = torch.softmax(logits, dim=-1)
            predictions = logits.argmax(dim=-1)

            if len(predictions.shape) == 0:
                # 单样本
                direction_map = {0: 'down', 1: 'neutral', 2: 'up'}
                direction = direction_map[predictions.item()]
                confidence = probs[0, predictions.item()].item()
                probabilities = {
                    'down': probs[0, 0].item(),
                    'neutral': probs[0, 1].item(),
                    'up': probs[0, 2].item(),
                }
                return {
                    'direction': direction,
                    'confidence': confidence,
                    'probabilities': probabilities,
                }
            else:
                # 多样本
                direction_map = {0: 'down', 1: 'neutral', 2: 'up'}
                directions = [direction_map[p.item()] for p in predictions]
                confidences = probs.max(dim=-1)[0].cpu().numpy().tolist()
                return {
                    'directions': directions,
                    'confidences': confidences,
                    'all_predictions': directions,
                }

    def _default_prediction(self) -> Dict:
        """默认预测 (模型未训练时)"""
        return {
            'direction': 'neutral',
            'confidence': 0.33,
            'probabilities': {'up': 0.33, 'neutral': 0.34, 'down': 0.33},
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
            'n_layers': self.n_layers,
            'n_heads': self.n_heads,
            'n_classes': self.n_classes,
            'dropout': self.dropout,
            'seq_len': self.seq_len,
        }, path)
        logger.info(f"[PatchTST] 模型已保存: {path}")

    def _load_best(self):
        """加载最佳模型"""
        path = os.path.join(self.model_dir, 'patchtst_best.pth')
        if not os.path.exists(path):
            return False

        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.trained = True
        logger.info(f"[PatchTST] 模型已加载: {path}")
        return True

    def save(self, path: Optional[str] = None):
        """保存完整状态"""
        if path is None:
            path = os.path.join(self.model_dir, 'patchtst_state.pkl')
        with open(path, 'wb') as f:
            pickle.dump({
                'trained': self.trained,
                'training_history': self.training_history,
            }, f)
        logger.info(f"[PatchTST] 状态已保存: {path}")

    def load(self, path: Optional[str] = None) -> bool:
        """加载完整状态"""
        if path is None:
            path = os.path.join(self.model_dir, 'patchtst_state.pkl')
        if not os.path.exists(path):
            return False

        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.trained = state.get('trained', False)
        self.training_history = state.get('training_history', {'loss': [], 'val_loss': [], 'val_acc': []})

        if self.trained:
            self._load_best()

        logger.info(f"[PatchTST] 状态已加载, trained={self.trained}")
        return True

    def is_trained(self) -> bool:
        """检查模型是否已训练"""
        return self.trained


# ── 全局单例 ────────────────────────────────────────────────

_patchtst_instance: Optional[PatchTSTIntegrator] = None
_patchtst_lock = threading.Lock()


def get_patchtst() -> PatchTSTIntegrator:
    """获取全局 PatchTST 实例 (线程安全)"""
    global _patchtst_instance
    if _patchtst_instance is None:
        with _patchtst_lock:
            if _patchtst_instance is None:
                _patchtst_instance = PatchTSTIntegrator()
                _patchtst_instance.load()  # 尝试加载已训练模型
    return _patchtst_instance


# ── 兼容性别名 ──────────────────────────────────────────────

# 导出统一接口
patchtst_integrator = PatchTSTIntegrator