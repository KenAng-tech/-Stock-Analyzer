#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
机器学习预测模块 — ML Predictor (Stacking Ensemble)

Level-0: LightGBM + XGBoost + RandomForest (异模型集成)
Level-1: Ridge 元学习器 (meta-learner)

指标:
  - IC (Information Coefficient): 预测值与真实收益的截面相关
  - ICIR (IC Information Ratio): IC 均值 / IC 标准差
  - 准确率 / 精确率 / 召回率 / F1 / AUC

前视偏差防护:
  - Purged K-Fold CV (训练/测试集之间留白区间)
  - 时间序列标签 (使用未来 horizon 收益，但仅用于标签)
  - 特征计算只使用截至当前时刻数据

升级内容:
  - P0: MACD placeholder → 真实计算
  - P0: 标签创建添加交易成本扣除
  - P1: 前视偏差修复（时间序列标签 + Purged CV）
  - P2: Stacking 集成 (LightGBM + XGBoost + RF → Ridge)
  - P3: IC / ICIR 评估指标
  - P3: 特征重要性追踪
  - P3: 模型持久化 (save/load)
"""

import hashlib
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import os
import pickle
import threading
import time
import warnings
warnings.filterwarnings('ignore')

from modules.dynamic_cache import cache
from modules.logger import logger


# ── 深度学习模型 ──────────────────────────────────────────────

class GRUModel:
    """
    GRU 深度学习模型 — 用于股票方向预测

    架构:
      Input(12) → GRU(32, dropout=0.3) → GRU(16, dropout=0.2) → Linear(8) → ReLU → Linear(3) → Softmax

    输出: 3 类分类 (上涨 / 下跌 / 震荡)

    依赖: PyTorch (可选，未安装时自动回退到 numpy 基线)
    """

    def __init__(self, input_dim: int = 12, hidden_dim: int = 32,
                 num_classes: int = 3, sequence_length: int = 20,
                 learning_rate: float = 0.001, epochs: int = 50,
                 dropout: float = 0.3, batch_size: int = 32):
        """
        Args:
            input_dim: 特征维度 (默认 12，与 prepare_features 一致)
            hidden_dim: GRU 隐藏层维度
            num_classes: 分类数 (3 = 上涨/下跌/震荡)
            sequence_length: 时间步长 (lookback window)
            learning_rate: 学习率
            epochs: 训练轮数
            dropout: Dropout 比率
            batch_size: 批大小
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.dropout = dropout
        self.batch_size = batch_size

        # PyTorch 组件
        self.use_torch = False
        self.device = None
        self.rnn = None
        self.fc = None
        self.criterion = None
        self.optimizer = None

        # 尝试导入 PyTorch
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import DataLoader, TensorDataset

            self._torch = torch
            self._nn = nn
            self._optim = optim
            self._dataset = TensorDataset
            self._dataloader = DataLoader
            self.use_torch = True

            self.device = torch.device('cpu')  # 单股预测不需要 GPU

            # ── 构建 GRU 网络 ──
            self.rnn = nn.Sequential(
                nn.GRU(input_dim, hidden_dim, batch_first=True, dropout=dropout),
                nn.GRU(hidden_dim, hidden_dim // 2, batch_first=True, dropout=0.2),
            )
            self.fc = nn.Sequential(
                nn.Linear(hidden_dim // 2, 8),
                nn.ReLU(),
                nn.Linear(8, num_classes),
            )
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = optim.Adam(self._get_params(), lr=learning_rate)
            self._prepare_fn = nn.functional.softmax
        except ImportError:
            self.use_torch = False

        self.trained = False
        self.training_history = {'loss': [], 'val_loss': []}
        self.mean_features = None
        self.std_features = None

    def _get_params(self):
        """获取所有可训练参数"""
        if not self.use_torch:
            return []
        return list(self.rnn.parameters()) + list(self.fc.parameters())

    def _normalize(self, X_seq: np.ndarray) -> np.ndarray:
        """标准化特征 (按特征维度)"""
        if self.mean_features is None:
            flat = X_seq.reshape(-1, self.input_dim)
            self.mean_features = np.mean(flat, axis=0)
            self.std_features = np.std(flat, axis=0) + 1e-8
        normalized = (X_seq - self.mean_features) / self.std_features
        return np.clip(normalized, -5, 5)  # 裁剪异常值

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        """
        训练 GRU 模型

        X: (n_samples, sequence_length, input_dim) 序列特征
        y: (n_samples,) 标签
        """
        if not self.use_torch:
            # NumPy 回退: 基于统计的简单预测
            self._numpy_fallback_fit(X, y)
            return

        # 标准化
        X_seq = self._normalize(X)
        if X_val is not None:
            X_val = self._normalize(X_val)

        y_tensor = self._torch.tensor(y, dtype=self._torch.long).to(self.device)

        # 创建 DataLoader
        dataset = self._dataset(
            self._torch.tensor(X_seq, dtype=self._torch.float32).to(self.device),
            y_tensor,
        )
        loader = self._dataloader(dataset, batch_size=self.batch_size, shuffle=True)

        best_loss = float('inf')
        patience_counter = 0
        patience = 10

        for epoch in range(self.epochs):
            self.rnn.train()
            self.fc.train()
            epoch_loss = 0.0

            for batch_X, batch_y in loader:
                self.optimizer.zero_grad()
                rnn_out, _ = self.rnn(batch_X)  # (B, T, H)
                output = self.fc(rnn_out[:, -1, :])  # 取最后一步
                loss = self.criterion(output, batch_y)
                loss.backward()
                self._torch.nn.utils.clip_grad_norm_(self._get_params(), max_norm=1.0)
                self.optimizer.step()
                epoch_loss += loss.item() * len(batch_X)

            avg_loss = epoch_loss / len(X)

            # 验证
            val_loss = 0.0
            if X_val is not None and len(X_val) > 0:
                val_X_normalized = self._normalize(X_val)
                val_X = self._torch.tensor(val_X_normalized,
                                           dtype=self._torch.float32).to(self.device)
                val_y = self._torch.tensor(y_val, dtype=self._torch.long).to(self.device)
                self.rnn.eval()
                self.fc.eval()
                with self._torch.no_grad():
                    v_out, _ = self.rnn(val_X)
                    v_pred = self.fc(v_out[:, -1, :])
                    val_loss = self.criterion(v_pred, val_y).item()

            self.training_history['loss'].append(avg_loss)
            if val_loss > 0:
                self.training_history['val_loss'].append(val_loss)

            # Early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        self.trained = True

    def _numpy_fallback_fit(self, X: np.ndarray, y: np.ndarray):
        """NumPy 回退: 基于序列统计特征的简单分类器"""
        # 从序列中提取统计特征: 均值, 标准差, 趋势
        seq_mean = np.mean(X, axis=1)       # (n, D)
        seq_std = np.std(X, axis=1)         # (n, D)
        seq_trend = (X[:, -1, :] - X[:, 0, :]) / (X[:, 0, :] + 1e-8)  # (n, D)

        self._fallback_features = np.hstack([seq_mean, seq_std, seq_trend])  # (n, 3D)
        self._fallback_labels = y

        # 计算每类的中心 (Nearest Centroid)
        self._centroids = {}
        for cls in np.unique(y):
            mask = (y == cls)
            self._centroids[int(cls)] = np.mean(self._fallback_features[mask], axis=0)

        self.trained = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测概率

        X: (n_samples, sequence_length, input_dim) 或 (n_samples, input_dim)
        Returns: (n_samples, num_classes)
        """
        if not self.trained:
            return np.ones((len(X), self.num_classes)) / self.num_classes

        if not self.use_torch:
            return self._numpy_fallback_predict(X)

        if X.ndim == 2:
            X = X.reshape(1, -1, self.input_dim)

        X_tensor = self._torch.tensor(
            self._normalize(X), dtype=self._torch.float32
        ).to(self.device)

        self.rnn.eval()
        self.fc.eval()
        with self._torch.no_grad():
            rnn_out, _ = self.rnn(X_tensor)
            logits = self.fc(rnn_out[:, -1, :])
            probs = self._prepare_fn(logits, dim=1)

        return probs.cpu().numpy()

    def _numpy_fallback_predict(self, X: np.ndarray) -> np.ndarray:
        """NumPy 回退预测: Nearest Centroid"""
        if X.ndim == 2:
            X = X.reshape(1, -1, self.input_dim)

        seq_mean = np.mean(X, axis=1)
        seq_std = np.std(X, axis=1)
        seq_trend = (X[:, -1, :] - X[:, 0, :]) / (X[:, 0, :] + 1e-8)
        features = np.hstack([seq_mean, seq_std, seq_trend])

        probs = np.zeros((len(features), self.num_classes))
        for cls, centroid in self._centroids.items():
            dist = np.sum((features - centroid) ** 2, axis=1)
            probs[:, cls] = 1.0 / (dist + 1e-6)

        # 归一化为概率
        row_sum = probs.sum(axis=1, keepdims=True)
        probs = probs / row_sum
        return probs

    def save(self, path: str):
        """保存模型"""
        data = {
            'use_torch': self.use_torch,
            'trained': self.trained,
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'num_classes': self.num_classes,
            'sequence_length': self.sequence_length,
            'training_history': self.training_history,
            'mean_features': self.mean_features,
            'std_features': self.std_features,
        }
        if self.use_torch and self.trained:
            data['rnn_state'] = self.rnn.state_dict()
            data['fc_state'] = self.fc.state_dict()
        if not self.use_torch and self.trained:
            data['fallback_features'] = self._fallback_features
            data['centroids'] = self._centroids

        with open(path, 'wb') as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: str) -> 'GRUModel':
        """加载模型"""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        model = cls(
            input_dim=data['input_dim'],
            hidden_dim=data['hidden_dim'],
            num_classes=data['num_classes'],
            sequence_length=data['sequence_length'],
        )
        model.use_torch = data['use_torch']
        model.trained = data['trained']
        model.mean_features = data.get('mean_features')
        model.std_features = data.get('std_features')
        model.training_history = data.get('training_history', {'loss': [], 'val_loss': []})

        if model.use_torch and 'rnn_state' in data:
            import torch
            import torch.nn as nn
            model.device = torch.device('cpu')
            model.rnn.load_state_dict(data['rnn_state'])
            model.fc.load_state_dict(data['fc_state'])
            model.optimizer = torch.optim.Adam(model._get_params(), lr=model.learning_rate)
            model.criterion = nn.CrossEntropyLoss()
            model._prepare_fn = nn.functional.softmax

        if not model.use_torch and 'centroids' in data:
            model._fallback_features = data['fallback_features']
            model._centroids = data['centroids']

        return model


class MLPredictor:
    """
    机器学习预测引擎 — Stacking 集成

    架构:
      Level-0: LightGBM + XGBoost + RandomForest (异模型)
      Level-1: Ridge 回归 (meta-learner)

    训练流程:
      1. Purged K-Fold CV 评估各 Level-0 模型
      2. OOF (Out-of-Fold) 预测生成 Level-1 训练数据
      3. Ridge 元学习器在 OOF 数据上训练
      4. 各 Level-0 模型在全量数据上 retrain
      5. Ridge 在 OOF 数据上最终训练
    """

    def __init__(self):
        self.models = {}                # Level-0 模型 {name: model}
        self.meta_learner = None        # Level-1 Ridge
        self.feature_names = [
            'momentum_1d', 'momentum_3d', 'momentum_5d', 'momentum_10d',
            'volume_ratio', 'volatility', 'rsi', 'macd_histogram',
            'ma5_ma20_ratio', 'price_position', 'turnover_normalized',
            'outer_inner_ratio',
        ]
        self.is_trained = False
        self.cv_score = 0.0
        self._trained_at: Optional[str] = None  # ISO format timestamp of last training
        self.model_dir = os.path.join(os.path.dirname(__file__), 'models')
        os.makedirs(self.model_dir, exist_ok=True)

        # 交易成本（A 股往返约 0.15%）
        self.transaction_cost = 0.0015

        # IC / 特征重要性
        self.factor_ic: Dict[str, float] = {}
        self.factor_ic_history: Dict[str, List[float]] = {}
        self.feature_importances: Dict[str, float] = {}
        self.oof_predictions: Optional[np.ndarray] = None  # OOF 预测 (n, n_levels)
        self.training_dates: Optional[List[str]] = None    # 训练日期索引

    # ── 特征工程 ──────────────────────────────────────────────

    def prepare_features(self, stock_data: Dict, klines: List[Dict]) -> Optional[np.ndarray]:
        """准备 ML 特征（修复前视偏差：只使用截至当前时刻的数据）"""
        if not klines or len(klines) < 30:
            return None

        closes = np.array([k['close'] for k in klines], dtype=float)
        volumes = np.array([k['volume'] for k in klines], dtype=float)

        # 动量特征（只用当前及之前的数据）
        momentum_1d = (closes[-1] / closes[-2] - 1) * 100 if len(closes) >= 2 else 0
        momentum_3d = (closes[-1] / closes[-4] - 1) * 100 if len(closes) >= 4 else 0
        momentum_5d = (closes[-1] / closes[-6] - 1) * 100 if len(closes) >= 6 else 0
        momentum_10d = (closes[-1] / closes[-11] - 1) * 100 if len(closes) >= 11 else 0

        # 成交量特征
        avg_volume = np.mean(volumes[-20:]) if len(volumes) >= 20 else np.mean(volumes)
        volume_ratio = volumes[-1] / avg_volume if avg_volume > 0 else 1.0

        # 波动率特征（已实现波动率）
        if len(closes) >= 20:
            returns = np.diff(np.log(closes[-20:]))
            volatility = float(np.std(returns) * np.sqrt(252) * 100)
        else:
            volatility = 5.0

        # RSI（真实计算）
        if len(closes) >= 15:
            deltas = np.diff(closes[-15:])
            gains = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
            losses = abs(np.mean(deltas[deltas < 0])) if np.any(deltas < 0) else 0.001
            rsi = float(100 - (100 / (1 + gains / losses)))
        else:
            rsi = 50.0

        # MACD 柱状图（真实计算，修复 placeholder）
        macd_hist = self._calculate_macd_histogram(closes)

        # MA 比率
        ma5 = np.mean(closes[-5:]) if len(closes) >= 5 else closes[-1]
        ma20 = np.mean(closes[-20:]) if len(closes) >= 20 else closes[-1]
        ma5_ma20_ratio = float(ma5 / ma20) if ma20 > 0 else 1.0

        # 价格位置
        if len(closes) >= 250:
            year_high = np.max(closes[-250:])
            year_low = np.min(closes[-250:])
        else:
            year_high = np.max(closes)
            year_low = np.min(closes)
        price_position = float((closes[-1] - year_low) / (year_high - year_low)) if (year_high - year_low) > 0 else 0.5

        # 外部数据
        turnover = stock_data.get('turnover', 100)
        outer = stock_data.get('outer_disk', 0)
        inner = stock_data.get('inner_disk', 1)
        outer_inner_ratio = outer / inner if inner > 0 else 1.0

        features = np.array([
            momentum_1d, momentum_3d, momentum_5d, momentum_10d,
            volume_ratio, volatility, rsi, macd_hist,
            ma5_ma20_ratio, price_position, turnover / 100.0, outer_inner_ratio,
        ])

        return features

    def prepare_features_batch(self, klines: List[Dict],
                                labels: np.ndarray) -> Optional[np.ndarray]:
        """
        准备完整特征矩阵（用于训练）
        返回: (n_samples, n_features) 的 numpy 数组
        每个样本使用截至该时刻的所有历史数据计算特征（无前视偏差）
        """
        if not klines or len(klines) < 30:
            return None

        closes = np.array([k['close'] for k in klines], dtype=float)
        volumes = np.array([k['volume'] for k in klines], dtype=float)
        n = len(klines)
        feature_dim = 12  # 与 prepare_features 一致
        features = np.zeros((n, feature_dim))

        for i in range(n):
            if i < 29:  # 需要至少 30 个数据点
                features[i] = self._make_default_features()
                continue

            window = closes[:i + 1]
            vol_window = volumes[:i + 1]

            # 动量特征
            features[i, 0] = (window[-1] / window[-2] - 1) * 100 if len(window) >= 2 else 0
            features[i, 1] = (window[-1] / window[-4] - 1) * 100 if len(window) >= 4 else 0
            features[i, 2] = (window[-1] / window[-6] - 1) * 100 if len(window) >= 6 else 0
            features[i, 3] = (window[-1] / window[-11] - 1) * 100 if len(window) >= 11 else 0

            # 成交量特征
            avg_vol = np.mean(vol_window[-20:]) if len(vol_window) >= 20 else np.mean(vol_window)
            features[i, 4] = vol_window[-1] / avg_vol if avg_vol > 0 else 1.0

            # 波动率
            if len(window) >= 20:
                ret = np.diff(np.log(window[-20:]))
                features[i, 5] = float(np.std(ret) * np.sqrt(252) * 100)
            else:
                features[i, 5] = 5.0

            # RSI
            if len(window) >= 15:
                d = np.diff(window[-15:])
                g = np.mean(d[d > 0]) if np.any(d > 0) else 0
                l = abs(np.mean(d[d < 0])) if np.any(d < 0) else 0.001
                features[i, 6] = float(100 - (100 / (1 + g / l)))
            else:
                features[i, 6] = 50.0

            # MACD
            features[i, 7] = self._calculate_macd_histogram(window)

            # MA 比率
            ma5 = np.mean(window[-5:]) if len(window) >= 5 else window[-1]
            ma20 = np.mean(window[-20:]) if len(window) >= 20 else window[-1]
            features[i, 8] = float(ma5 / ma20) if ma20 > 0 else 1.0

            # 价格位置
            if len(window) >= 250:
                yh = np.max(window[-250:])
                yl = np.min(window[-250:])
            else:
                yh = np.max(window)
                yl = np.min(window)
            features[i, 9] = float((window[-1] - yl) / (yh - yl)) if (yh - yl) > 0 else 0.5

            # 换手率
            features[i, 10] = klines[i].get('turnover', 100) / 100.0

            # 内外盘比
            outer = klines[i].get('outer_disk', 0)
            inner = klines[i].get('inner_disk', 1)
            features[i, 11] = outer / inner if inner > 0 else 1.0

        return features

    def _make_default_features(self) -> np.ndarray:
        """返回默认特征向量（数据不足时使用）"""
        return np.array([0.0, 0.0, 0.0, 0.0, 1.0, 5.0, 50.0, 0.0, 1.0, 0.5, 1.0, 1.0])

    def _calculate_macd_histogram(self, closes: np.ndarray) -> float:
        """计算 MACD 柱状图"""
        if len(closes) < 35:
            return 0.0

        # 需要完整 EMA 数组来计算 MACD line
        ema12_arr = self._ema_array(closes, 12)
        ema26_arr = self._ema_array(closes, 26)
        macd_line = ema12_arr - ema26_arr
        # MACD signal line (9 日 EMA 的 MACD line)
        if len(macd_line) >= 9:
            signal_line = self._ema(macd_line, 9)
        else:
            signal_line = macd_line[-1] if len(macd_line) > 0 else 0
        return float(macd_line[-1] - signal_line)

    @staticmethod
    def _ema(data: np.ndarray, period: int) -> float:
        """计算 EMA 最后一个值"""
        if len(data) < period:
            return float(np.mean(data)) if len(data) > 0 else 0
        multiplier = 2.0 / (period + 1)
        ema = float(data[0])
        for price in data[1:]:
            ema = (price - ema) * multiplier + ema
        return ema

    @staticmethod
    def _ema_array(data: np.ndarray, period: int) -> np.ndarray:
        """计算完整 EMA 数组"""
        if len(data) < period:
            return data.astype(float)
        multiplier = 2.0 / (period + 1)
        ema = np.zeros(len(data))
        ema[0] = float(data[0])
        for i in range(1, len(data)):
            ema[i] = (data[i] - ema[i - 1]) * multiplier + ema[i - 1]
        return ema

    # ── 标签创建（修复前视偏差） ──────────────────────────────

    def create_labels(self, klines: List[Dict], horizon: int = 5) -> np.ndarray:
        """创建标签（考虑交易成本，修复前视偏差）"""
        if len(klines) < horizon + 1:
            return np.array([])

        closes = np.array([k['close'] for k in klines], dtype=float)
        labels = []

        for i in range(len(closes) - horizon):
            future_return = (closes[i + horizon] - closes[i]) / closes[i]
            # 净收益 = 毛收益 - 交易成本（买入 + 卖出）
            net_return = future_return - self.transaction_cost
            label = 1 if net_return > 0.02 else (-1 if net_return < -0.02 else 0)
            labels.append(label)

        return np.array(labels)

    # ── Purged K-Fold 交叉验证 ──────────────────────────────

    def purged_kfold(self, n_samples: int, n_splits: int = 5, embargo_pct: float = 0.05) -> List[Tuple[List[int], List[int]]]:
        """
        Purged K-Fold 分割 — 防止前视偏差

        在金融时间序列 CV 中，测试集的早期样本可能与训练集的尾部样本
        有重叠（例如同一只股票的多日数据）。Purged K-Fold 在测试集之前
        移除一个 embargo 区间，确保训练数据在测试集之前完全结束。

        Args:
            n_samples: 样本总数
            n_splits: 折数
            embargo_pct: embargo 区间占每折大小的比例 (默认 5%)

        Returns:
            List of (train_indices, test_indices)
        """
        fold_size = n_samples // n_splits
        emb_size = int(fold_size * embargo_pct)

        train_indices_all = []
        test_indices_all = []

        for fold in range(n_splits):
            test_start = fold * fold_size
            test_end = test_start + fold_size if fold < n_splits - 1 else n_samples

            test_idx = list(range(test_start, test_end))
            train_idx = list(range(0, test_start)) + list(range(test_end, n_samples))

            # Purge: 从训练集中移除测试集附近的样本
            if train_idx:
                # 移除训练集中靠近测试集尾部的样本
                train_idx = [i for i in train_idx if i < test_start - max(1, emb_size)]

                # 从测试集开头移除 embargo
                if emb_size > 0 and len(test_idx) > emb_size:
                    test_idx = test_idx[emb_size:]

            train_indices_all.append(train_idx)
            test_indices_all.append(test_idx)

        return list(zip(train_indices_all, test_indices_all))

    # ── IC / ICIR 评估 ──────────────────────────────────────

    def calculate_ic(self, predicted: np.ndarray, actual_returns: np.ndarray) -> float:
        """
        计算 Information Coefficient (IC)

        IC = Spearman 相关系数 (预测排序 vs 实际收益排序)
        使用 Spearman 而非 Pearson 因为更稳健于异常值。

        Args:
            predicted: 模型预测值 (n,)
            actual_returns: 实际收益率 (n,)

        Returns:
            IC 值 (-1 ~ 1)
        """
        if len(predicted) < 10:
            return 0.0

        # Spearman 相关 = 排序后的 Pearson 相关
        rank_pred = self._rank_array(predicted)
        rank_actual = self._rank_array(actual_returns)

        mean_p = np.mean(rank_pred)
        mean_a = np.mean(rank_actual)
        cov = np.mean((rank_pred - mean_p) * (rank_actual - mean_a))
        std_p = np.std(rank_pred)
        std_a = np.std(rank_actual)

        if std_p < 1e-10 or std_a < 1e-10:
            return 0.0

        return float(cov / (std_p * std_a))

    @staticmethod
    def _rank_array(x: np.ndarray) -> np.ndarray:
        """计算排序（处理并列）"""
        ranks = np.argsort(np.argsort(x)) + 1
        # 并列处理: 取平均排名
        unique_vals = np.unique(x)
        for val in unique_vals:
            mask = (x == val)
            if np.sum(mask) > 1:
                avg_rank = np.mean(ranks[mask])
                ranks[mask] = avg_rank
        return ranks.astype(float)

    def calculate_icir(self, ic_series: List[float]) -> float:
        """
        计算 IC Information Ratio

        ICIR = mean(IC) / std(IC) * sqrt(252)

        Args:
            ic_series: 每日 IC 值序列

        Returns:
            ICIR 值
        """
        if len(ic_series) < 5:
            return 0.0
        ic_arr = np.array(ic_series)
        ic_mean = np.mean(ic_arr)
        ic_std = np.std(ic_arr)
        if ic_std < 1e-10:
            return 0.0
        return float(ic_mean / ic_std * np.sqrt(252))

    # ── Stacking 集成训练 ──────────────────────────────────────

    def train_stacking_ensemble(self, X: np.ndarray, y: np.ndarray,
                                 dates: Optional[List[str]] = None) -> bool:
        """
        Stacking 集成训练 — 核心方法

        Level-0 模型: LightGBM + XGBoost + RandomForest
        Level-1 元学习器: Ridge 回归

        流程:
          1. Purged 5-Fold CV 生成 OOF 预测
          2. Ridge 在 OOF 特征上训练
          3. 各 Level-0 模型在全量数据上 retrain
          4. Ridge 在 OOF 数据上最终训练

        Args:
            X: 特征矩阵 (n_samples, n_features)
            y: 标签数组 (n_samples,)
            dates: 日期字符串列表 (用于 IC 计算)

        Returns:
            是否训练成功
        """
        # 对齐 X 和 y 的长度（prepare_features_batch 可能返回比 labels 更多的样本）
        n = min(len(X), len(y))
        if n < 60:
            print(f"[MLPredictor] 数据不足: {n} 样本，需要至少 60")
            return False
        X = X[:n]
        y = y[:n]

        n_splits = min(5, max(3, len(X) // 50))
        folds = self.purged_kfold(len(X), n_splits=n_splits)

        # ── Step 1: 初始化 Level-0 模型（过滤不可用的） ──
        raw_models = {
            'lgb': self._make_lightgbm(),
            'xgb': self._make_xgboost(),
            'rf': self._make_random_forest(),
            'gru': self._make_gru(),
        }
        level0_models = {k: v for k, v in raw_models.items() if v is not None}

        if not level0_models:
            print("[MLPredictor] 所有模型均不可用（sklearn/lightgbm/xgboost 未安装）")
            return False

        # OOF 预测: (n_samples, n_level0_models)
        oof_preds = np.zeros((len(X), len(level0_models)))
        ic_per_fold = {name: [] for name in level0_models}
        fold_metrics = []

        # ── Step 2: Purged CV 训练 + OOF 预测 ──
        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            if len(y_train) < 30 or len(y_test) < 5:
                continue

            fold_metrics.append({
                'fold': fold_idx + 1,
                'train_size': len(X_train),
                'test_size': len(X_test),
            })

            for col_idx, (name, model) in enumerate(level0_models.items()):
                try:
                    if name == 'gru':
                        # GRU 需要序列数据
                        seq_len = 20
                        X_seq_all, all_idx = self._generate_sequences(X, seq_len)
                        y_aligned = y[all_idx]

                        # 训练集序列
                        train_mask = np.isin(all_idx, train_idx)
                        X_seq_train = X_seq_all[train_mask]
                        y_train_seq = y_aligned[train_mask]

                        # 测试集序列 (仅用训练集历史，无前视偏差)
                        test_mask = np.isin(all_idx, test_idx)
                        X_seq_test = X_seq_all[test_mask]
                        y_test_seq = y_aligned[test_mask]

                        # 用训练集统计量做标准化 (防止数据泄露)
                        if hasattr(model, '_normalize'):
                            # 临时设置统计量
                            if model.mean_features is None:
                                model._normalize(X_seq_train)
                            else:
                                # 用训练集重新计算
                                flat_train = X_seq_train.reshape(-1, X.shape[1])
                                model.mean_features = np.mean(flat_train, axis=0)
                                model.std_features = np.std(flat_train, axis=0) + 1e-8

                        model.fit(X_seq_train, y_train_seq, X_seq_test, y_test_seq)

                        # OOF 预测: 使用 "up" 类概率 (class 0) 作为连续信号
                        gru_proba = model.predict_proba(X_seq_test)
                        oof_preds[test_idx, col_idx] = gru_proba[:, 0]  # up 概率

                        if len(y_test) >= 5:
                            ic = self.calculate_ic(oof_preds[test_idx, col_idx], y_test)
                            ic_per_fold[name].append(ic)
                    else:
                        model.fit(X_train, y_train)
                        oof_preds[test_idx, col_idx] = model.predict(X_test)

                        if len(y_test) >= 5:
                            ic = self.calculate_ic(oof_preds[test_idx, col_idx], y_test)
                            ic_per_fold[name].append(ic)
                except Exception as e:
                    print(f"[MLPredictor] Fold {fold_idx+1} {name} 训练失败: {e}")

        # ── Step 3: 评估各 Level-0 模型 ──
        model_ic = {}
        for name, ics in ic_per_fold.items():
            if ics:
                model_ic[name] = {
                    'mean_ic': float(np.mean(ics)),
                    'icir': self.calculate_icir(ics),
                    'n_folds': len(ics),
                }
                print(f"[MLPredictor]   {name:6s} IC={model_ic[name]['mean_ic']:+.4f}  ICIR={model_ic[name]['icir']:.3f}")

        # ── Step 4: Ridge 元学习器在 OOF 数据上训练 ──
        try:
            from sklearn.linear_model import Ridge
            # 过滤掉 IC <= 0 的模型（无预测能力）
            valid_models = [name for name, ic_info in model_ic.items() if ic_info['mean_ic'] > 0]
            if valid_models:
                valid_cols = [list(level0_models.keys()).index(n) for n in valid_models]
                oof_valid = oof_preds[:, valid_cols]

                self.meta_learner = Ridge(alpha=1.0)
                self.meta_learner.fit(oof_valid, y)

                # 元学习器系数（反映各模型权重）
                meta_coefs = dict(zip(valid_models, self.meta_learner.coef_))
                print(f"[MLPredictor] Level-1 Ridge 权重: {', '.join(f'{k}={v:.3f}' for k, v in meta_coefs.items())}")
            else:
                print("[MLPredictor] 警告: 无 IC>0 的 Level-0 模型，使用等权平均")
                self.meta_learner = None
        except Exception as e:
            print(f"[MLPredictor] Ridge 训练失败: {e}，回退到等权平均")
            self.meta_learner = None

        # ── Step 5: 各 Level-0 模型在全量数据上 retrain ──
        for name, model in level0_models.items():
            try:
                if name == 'gru':
                    seq_len = 20
                    X_seq_all, all_idx = self._generate_sequences(X, seq_len)
                    y_full_seq = y[all_idx]
                    model.fit(X_seq_all, y_full_seq)
                else:
                    model.fit(X, y)
                self.models[name] = model
            except Exception as e:
                print(f"[MLPredictor] {name} 全量训练失败: {e}")

        # ── Step 6: 特征重要性 ──
        self._compute_feature_importances(X)

        # ── Step 7: 计算整体 CV 分数 ──
        if len(y) >= 10:
            # 使用元学习器做 OOF 预测（如果有）
            if self.meta_learner and valid_models:
                valid_cols = [list(level0_models.keys()).index(n) for n in valid_models]
                oof_valid = oof_preds[:, valid_cols]
                oof_class = self._proba_to_class(self.meta_learner.predict(oof_valid))
            else:
                # 等权平均
                oof_class = self._proba_to_class(np.mean(oof_preds, axis=1))

            # Accuracy (尝试 sklearn，回退到手动计算)
            try:
                from sklearn.metrics import accuracy_score
                self.cv_score = round(accuracy_score(y, oof_class), 4)
            except ImportError:
                self.cv_score = round(float(np.mean(y == oof_class)), 4)
        else:
            self.cv_score = 0.0

        # 保存 OOF 预测和日期
        self.oof_predictions = oof_preds
        self.training_dates = dates

        # 保存 IC 历史
        for name, ics in ic_per_fold.items():
            self.factor_ic_history[name] = ics

        self.is_trained = True
        print(f"[MLPredictor] Stacking 集成训练完成, CV准确率: {self.cv_score:.3f}")
        return True

    def _make_lightgbm(self):
        """创建 LightGBM 模型"""
        try:
            import lightgbm as lgb
            return lgb.LGBMClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.05,
                random_state=42, verbose=-1, n_jobs=1
            )
        except ImportError:
            return None

    def _make_xgboost(self):
        """创建 XGBoost 模型"""
        try:
            import xgboost as xgb
            return xgb.XGBClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                random_state=42, verbosity=0, n_jobs=1,
                use_label_encoder=False, eval_metric='logloss'
            )
        except ImportError:
            return None

    def _make_random_forest(self):
        """创建 RandomForest 模型"""
        try:
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(
                n_estimators=100, max_depth=5, random_state=42,
                n_jobs=1, class_weight='balanced'
            )
        except ImportError:
            return None

    def _make_gru(self) -> Optional[GRUModel]:
        """创建 GRU 深度学习模型 (PyTorch, 未安装则回退到 numpy)"""
        try:
            return GRUModel(
                input_dim=len(self.feature_names),
                hidden_dim=32,
                num_classes=3,
                sequence_length=20,
                learning_rate=0.001,
                epochs=50,
                dropout=0.3,
                batch_size=32,
            )
        except Exception as e:
            print(f"[MLPredictor] GRU 模型初始化失败: {e}")
            return None

    @staticmethod
    def _proba_to_class(proba: np.ndarray) -> np.ndarray:
        """将连续概率映射为类别 {-1, 0, 1}"""
        result = np.zeros_like(proba, dtype=int)
        result[proba > 0.33] = 1
        result[proba < -0.33] = -1
        return result

    def _compute_feature_importances(self, X: np.ndarray):
        """从各 Level-0 模型计算平均特征重要性"""
        if not self.models:
            return
        importances = {}
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                im = model.feature_importances_
                if len(im) == len(self.feature_names):
                    for fname, imp in zip(self.feature_names, im):
                        importances[fname] = importances.get(fname, 0) + float(imp)
        # 平均
        n = max(len(importances), 1)
        self.feature_importances = {k: v / n for k, v in importances.items()}

    @staticmethod
    def _generate_sequences(features: np.ndarray, sequence_length: int = 20
                            ) -> Tuple[np.ndarray, List[int]]:
        """
        从特征矩阵生成序列

        Args:
            features: (n_samples, n_features)
            sequence_length: 时间步长

        Returns:
            sequences: (n_sequences, sequence_length, n_features)
            valid_indices: 每个序列对应的"当前样本"索引 (即序列末尾的索引)
        """
        n_samples, n_features = features.shape
        sequences = []
        valid_indices = []

        for i in range(sequence_length - 1, n_samples):
            start = i - sequence_length + 1
            seq = features[start:i + 1]  # (sequence_length, n_features)
            # 如果开头不够，用第一个特征填充
            if start < 0:
                pad = features[0:1].repeat(abs(start), axis=0)
                seq = np.vstack([pad, features[0:i + 1]])
            sequences.append(seq)
            valid_indices.append(i)

        return np.array(sequences), valid_indices

    # ── 模型训练（兼容旧接口） ──────────────────────────────

    def train_simple_model(self, X: np.ndarray, y: np.ndarray) -> bool:
        """训练 RandomForest baseline — P0 修复: 使用时间序列交叉验证"""
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import TimeSeriesSplit
            from sklearn.metrics import accuracy_score

            self.models['rf'] = RandomForestClassifier(
                n_estimators=100, max_depth=5, random_state=42
            )

            if len(X) < 30 or len(y) < 30:
                print(f"[MLPredictor] 数据不足: {len(X)} 样本，需要至少 30")
                return False

            # P0 修复: 使用时间序列分割而非随机分割
            tscv = TimeSeriesSplit(n_splits=min(3, len(X) // 10))
            cv_scores = []
            best_model = None
            best_score = 0

            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                if len(y_train) < 10 or len(y_test) < 3:
                    continue

                temp_model = RandomForestClassifier(
                    n_estimators=100, max_depth=5, random_state=42
                )
                temp_model.fit(X_train, y_train)
                pred = temp_model.predict(X_test)
                score = accuracy_score(y_test, pred)
                cv_scores.append(score)

                if score > best_score:
                    best_score = score
                    best_model = temp_model

            if best_model is None:
                # 回退到全量训练
                self.models['rf'].fit(X, y)
            else:
                # 用最佳模型参数在全量数据上重新训练
                self.models['rf'].set_params(**best_model.get_params())
                self.models['rf'].fit(X, y)

            self.cv_score = round(float(np.mean(cv_scores)) if cv_scores else best_score, 4)
            self.is_trained = True
            print(f"[MLPredictor] RandomForest 训练完成, CV准确率: {self.cv_score:.3f}")
            return True
        except Exception as e:
            print(f"[MLPredictor] RandomForest 训练失败: {e}")
        return False

    def train_lightgbm_model(self, X: np.ndarray, y: np.ndarray) -> bool:
        """训练 LightGBM 模型 — P0 修复: 使用时间序列交叉验证

        优先使用 vnpy_workspace 的 LightGBMSignalLayer，
        如果不可用则使用本地 LightGBM。
        """
        # 尝试使用 vnpy_workspace 的 LightGBMSignalLayer
        try:
            import sys
            vnpy_path = '/Users/claw/vnpy_workspace'
            if vnpy_path not in sys.path:
                sys.path.insert(0, vnpy_path)
            from alpha_models.lightgbm_signal_layer import LightGBMSignalLayer, SignalConfig

            df = pd.DataFrame(X, columns=self.feature_names[:X.shape[1]])
            df['label'] = y

            signal_layer = LightGBMSignalLayer(SignalConfig())
            # 使用简化训练
            feature_cols = [c for c in self.feature_names[:X.shape[1]] if c in df.columns]
            if feature_cols and len(df) >= 30:
                signal_layer.train_models(df, feature_cols)
                self.models['lgb'] = signal_layer
                self.is_trained = True
                return True
        except Exception as e:
            print(f"[MLPredictor] LightGBM (vnpy) 训练失败: {e}")

        # 回退: 本地 LightGBM — P0 修复: 时间序列 CV
        try:
            import lightgbm as lgb
            from sklearn.model_selection import TimeSeriesSplit
            from sklearn.metrics import accuracy_score

            self.models['lgb'] = lgb.LGBMClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.05,
                random_state=42, verbose=-1
            )

            if len(X) < 30 or len(y) < 30:
                print(f"[MLPredictor] LightGBM 数据不足: {len(X)} 样本")
                return False

            # 时间序列交叉验证
            tscv = TimeSeriesSplit(n_splits=min(3, len(X) // 10))
            cv_scores = []

            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                if len(y_train) < 10 or len(y_test) < 3:
                    continue

                temp_model = lgb.LGBMClassifier(
                    n_estimators=200, max_depth=6, learning_rate=0.05,
                    random_state=42, verbose=-1
                )
                temp_model.fit(X_train, y_train)
                pred = temp_model.predict(X_test)
                score = accuracy_score(y_test, pred)
                cv_scores.append(score)

            self.cv_score = round(float(np.mean(cv_scores)) if cv_scores else 0.5, 4)
            # 用全量数据训练最终模型
            self.models['lgb'].fit(X, y)
            self.is_trained = True
            print(f"[MLPredictor] LightGBM 训练完成, CV准确率: {self.cv_score:.3f}")
            return True
        except ImportError:
            print("[MLPredictor] LightGBM 未安装，使用 RandomForest")
        except Exception as e:
            print(f"[MLPredictor] LightGBM 训练失败: {e}")

        return False

    # ── 预测（Stacking 集成） ────────────────────────────────────

    def predict_direction(self, features: np.ndarray) -> Dict:
        """
        Stacking 集成预测

        流程:
          1. 各 Level-0 模型分别预测 → 概率向量
          2. Ridge 元学习器加权融合
          3. 输出最终方向 + 概率

        Args:
            features: 单样本特征 (n_features,) 或 (1, n_features)
        """
        if not self.models:
            return {'direction': 'neutral', 'confidence': 0.5,
                    'probabilities': {'up': 0.33, 'down': 0.33, 'neutral': 0.34}}

        feat = features.reshape(1, -1) if features.ndim == 1 else features
        level0_probs = {}

        # ── Step 1: Level-0 预测 ──
        for name, model in self.models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(feat)[0]
                    # 映射到 {-1, 0, 1} → {up, down, neutral}
                    if len(proba) >= 3:
                        level0_probs[name] = {
                            'up': float(proba[0]),
                            'down': float(proba[1]) if len(proba) > 1 else 0.0,
                            'neutral': float(proba[2]) if len(proba) > 2 else float(proba[-1]),
                        }
                    else:
                        # 二分类: {down, up}
                        level0_probs[name] = {
                            'up': float(proba[-1]),
                            'down': float(proba[0]),
                            'neutral': 0.0,
                        }
                else:
                    # 无 predict_proba，用 predict 产生软概率
                    pred = model.predict(feat)[0]
                    level0_probs[name] = {
                        'up': 0.5 if pred == 1 else 0.1,
                        'down': 0.5 if pred == -1 else 0.1,
                        'neutral': 0.3,
                    }
            except Exception as e:
                print(f"[MLPredictor] {name} 预测失败: {e}")

        if not level0_probs:
            return {'direction': 'neutral', 'confidence': 0.5,
                    'probabilities': {'up': 0.33, 'down': 0.33, 'neutral': 0.34}}

        # ── Step 2: Level-1 融合 ──
        if self.meta_learner:
            # 提取各模型概率作为 Ridge 输入
            model_names = list(self.models.keys())
            input_vec = np.array([
                level0_probs.get(n, {'up': 0.33, 'down': 0.33, 'neutral': 0.34})['up']
                for n in model_names
            ]).reshape(1, -1)

            try:
                fused_score = float(self.meta_learner.predict(input_vec)[0])
                # Ridge 输出是连续值，映射到概率
                # 假设 Ridge 输出范围大致 [-1, 1]
                fused_up = max(0, min(1, (fused_score + 1) / 2))
                fused_down = max(0, min(1, fused_up - fused_score * 0.5))
                fused_neutral = 1.0 - fused_up + fused_down
                fused_neutral = max(0, 1.0 - fused_up - fused_down)

                final_probs = {
                    'up': round(fused_up, 4),
                    'down': round(max(0, fused_down), 4),
                    'neutral': round(fused_neutral, 4),
                }
            except Exception:
                # Ridge 预测失败 → 等权平均
                final_probs = self._average_probs(level0_probs)
        else:
            # 无元学习器 → 等权平均
            final_probs = self._average_probs(level0_probs)

        # 归一化
        total = sum(final_probs.values())
        if total > 0:
            final_probs = {k: v / total for k, v in final_probs.items()}

        # 决策
        direction = 'up' if final_probs['up'] > final_probs['down'] and final_probs['up'] > 0.4 else \
                    'down' if final_probs['down'] > final_probs['up'] and final_probs['down'] > 0.4 else 'neutral'

        return {
            'direction': direction,
            'confidence': round(max(final_probs.values()), 4),
            'probabilities': final_probs,
            'model_details': {n: dict(v) for n, v in level0_probs.items()},
        }

    @staticmethod
    def _ml_cache_key(stock_code: str, features: np.ndarray) -> str:
        """生成 ML 预测缓存键

        基于股票代码 + 特征指纹，确保特征变化时自动失效
        """
        feat_hash = hashlib.md5(
            features.tobytes()[:256]  # 取前 256 字节足够区分
        ).hexdigest()[:12]
        return f"ml_{stock_code}_{feat_hash}"

    def predict_direction_cached(self, stock_code: str,
                                  features: np.ndarray) -> Dict:
        """ML 预测（带缓存，TTL 1 分钟）

        缓存键基于 stock_code + 特征指纹
        当 realtime/kline 数据更新时，依赖链自动失效
        """
        key = self._ml_cache_key(stock_code, features)
        cached = cache.get(key, category='ml')
        if cached is not None:
            return cached

        result = self.predict_direction(features)
        cache.set(key, result, category='ml', tags={stock_code})
        return result

    @staticmethod
    def _average_probs(probs_dict: Dict) -> Dict[str, float]:
        """等权平均各模型概率"""
        avg = {'up': 0, 'down': 0, 'neutral': 0}
        n = len(probs_dict)
        if n == 0:
            return avg
        for p in probs_dict.values():
            avg['up'] += p.get('up', 0.33)
            avg['down'] += p.get('down', 0.33)
            avg['neutral'] += p.get('neutral', 0.34)
        return {k: v / n for k, v in avg.items()}

    def calculate_ensemble_prediction(self, features: np.ndarray,
                                       market_state: str = 'sideways') -> Dict:
        """集成预测（结合 ML 和市场状态调整）"""
        ml_pred = self.predict_direction(features)

        state_adjustments = {
            'bull': {'up_boost': 0.1, 'down_reduce': 0.1},
            'bear': {'up_reduce': 0.1, 'down_boost': 0.1},
            'sideways': {},
        }

        adj = state_adjustments.get(market_state, state_adjustments['sideways'])
        probs = ml_pred['probabilities'].copy()

        if 'up_boost' in adj:
            probs['up'] = min(1.0, probs['up'] + adj['up_boost'])
            probs['down'] = max(0, probs['down'] - adj['down_reduce'])
        if 'up_reduce' in adj:
            probs['up'] = max(0, probs['up'] - adj['up_reduce'])
            probs['down'] = min(1.0, probs['down'] + adj['down_boost'])

        total = sum(probs.values())
        probs = {k: v / total for k, v in probs.items()}

        direction = 'up' if probs['up'] > probs['down'] and probs['up'] > 0.4 else \
                    'down' if probs['down'] > probs['up'] and probs['down'] > 0.4 else 'neutral'

        return {
            'direction': direction,
            'confidence': max(probs.values()),
            'probabilities': probs,
            'ml_confidence': ml_pred['confidence'],
            'market_state': market_state,
        }

    # ── 模型持久化 ──────────────────────────────────────────────

    def save_model(self, path: Optional[str] = None) -> str:
        """
        保存模型到磁盘

        包括: Level-0 模型, Level-1 Ridge, 特征重要性, IC 历史

        Returns:
            保存路径
        """
        if path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            path = os.path.join(self.model_dir, f'model_{timestamp}.pkl')

        try:
            now = datetime.now().isoformat()
            self._trained_at = now
            data = {
                'models': self.models,
                'meta_learner': self.meta_learner,
                'feature_names': self.feature_names,
                'feature_importances': self.feature_importances,
                'factor_ic': self.factor_ic,
                'factor_ic_history': self.factor_ic_history,
                'cv_score': self.cv_score,
                'is_trained': self.is_trained,
                'trained_at': now,
            }
            with open(path, 'wb') as f:
                pickle.dump(data, f)
            print(f"[MLPredictor] 模型已保存: {path}")
            return path
        except Exception as e:
            print(f"[MLPredictor] 模型保存失败: {e}")
            return ''

    def load_model(self, path: str) -> bool:
        """
        从磁盘加载模型

        Returns:
            是否加载成功
        """
        try:
            if not os.path.exists(path):
                print(f"[MLPredictor] 模型文件不存在: {path}")
                return False

            with open(path, 'rb') as f:
                data = pickle.load(f)

            self.models = data.get('models', {})
            self.meta_learner = data.get('meta_learner')
            self.feature_names = data.get('feature_names', self.feature_names)
            self.feature_importances = data.get('feature_importances', {})
            self.factor_ic = data.get('factor_ic', {})
            self.factor_ic_history = data.get('factor_ic_history', {})
            self.cv_score = data.get('cv_score', 0.0)
            self.is_trained = data.get('is_trained', False)
            self._trained_at = data.get('trained_at')

            print(f"[MLPredictor] 模型已加载: {path} (CV={self.cv_score:.3f})")
            return True
        except Exception as e:
            print(f"[MLPredictor] 模型加载失败: {e}")
            return False

    def load_latest_model(self) -> bool:
        """
        自动查找并加载最新的模型文件

        Returns:
            是否加载成功
        """
        try:
            if not os.path.isdir(self.model_dir):
                return False

            model_files = [
                f for f in os.listdir(self.model_dir)
                if f.startswith('model_') and f.endswith('.pkl')
            ]
            if not model_files:
                return False

            # 按文件名排序（时间戳格式保证字典序即时间序），取最新
            latest = sorted(model_files)[-1]
            path = os.path.join(self.model_dir, latest)
            return self.load_model(path)
        except Exception as e:
            print(f"[MLPredictor] 自动加载模型失败: {e}")
            return False

    def is_model_fresh(self, hours: int = 24) -> bool:
        """
        检查模型是否仍然新鲜（是否需要重新训练）

        Args:
            hours: 新鲜度阈值（小时），默认 24

        Returns:
            True 如果模型是新鲜的（训练时间 < hours 小时前），否则 False
        """
        if not self.is_trained or self._trained_at is None:
            return False
        try:
            trained_time = datetime.fromisoformat(self._trained_at)
            elapsed = datetime.now() - trained_time
            return elapsed.total_seconds() < hours * 3600
        except (ValueError, TypeError):
            return False

    def get_model_report(self) -> Dict:
        """获取模型报告（用于 API 返回）"""
        return {
            'is_trained': self.is_trained,
            'cv_score': self.cv_score,
            'models': list(self.models.keys()),
            'feature_importances': dict(sorted(
                self.feature_importances.items(), key=lambda x: x[1], reverse=True
            )),
            'ic_history': {k: round(np.mean(v), 4) if v else 0 for k, v in self.factor_ic_history.items()},
            'trained_at': self._trained_at,
            'is_fresh': self.is_model_fresh(),
        }


class FeatureEngineering:
    """特征工程模块"""

    def __init__(self):
        self.sentiment_dict = self._load_sentiment_dict()

    def _load_sentiment_dict(self) -> Dict:
        return {
            'positive': ['上涨', '突破', '看好', '买入', '增持', '利好', '新高', '强势'],
            'negative': ['下跌', '破位', '看空', '卖出', '减持', '利空', '新低', '弱势'],
            'neutral': ['震荡', '观望', '等待', '中性', '区间']
        }

    def calculate_technical_features(self, klines: List[Dict]) -> Dict:
        """计算技术指标特征"""
        if not klines or len(klines) < 20:
            return {}

        df = pd.DataFrame(klines)
        closes = df['close'].values.astype(float)

        ema12 = self._ema_arr(closes, 12)
        ema26 = self._ema_arr(closes, 26)
        macd = ema12 - ema26
        signal = self._ema_arr(macd, 9)
        histogram = macd - signal

        rsi = self._rsi_arr(closes, 14)
        bb_upper, bb_middle, bb_lower = self._bollinger_bands(closes)

        return {
            'ema12': float(ema12[-1]),
            'ema26': float(ema26[-1]),
            'macd': float(macd[-1]),
            'macd_signal': float(signal[-1]),
            'macd_histogram': float(histogram[-1]),
            'rsi': float(rsi[-1]),
            'bb_upper': float(bb_upper[-1]),
            'bb_middle': float(bb_middle[-1]),
            'bb_lower': float(bb_lower[-1]),
        }

    @staticmethod
    def _ema_arr(data: np.ndarray, period: int) -> np.ndarray:
        ema = np.zeros_like(data)
        ema[0] = data[0]
        multiplier = 2 / (period + 1)
        for i in range(1, len(data)):
            ema[i] = (data[i] - ema[i-1]) * multiplier + ema[i-1]
        return ema

    @staticmethod
    def _rsi_arr(data: np.ndarray, period: int = 14) -> np.ndarray:
        delta = np.diff(data)
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(data)
        avg_loss = np.zeros_like(data)
        avg_gain[period] = np.mean(gains[:period])
        avg_loss[period] = np.mean(losses[:period])
        for i in range(period + 1, len(data)):
            avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i-1]) / period
        rs = avg_gain / (avg_loss + 1e-10)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _bollinger_bands(data: np.ndarray, period: int = 20, std_dev: float = 2):
        middle = pd.Series(data).rolling(window=period).mean().values
        std = pd.Series(data).rolling(window=period).std().values
        return middle + std_dev * std, middle, middle - std_dev * std


class ModelTrainingScheduler:
    """
    ML 模型训练调度器

    功能:
      - 启动时自动训练（如果模型不存在或已过期）
      - 定时重训练（默认 24 小时）
      - 按需触发训练（通过 API）
      - 后台线程执行，不阻塞主线程

    用法:
      scheduler = ModelTrainingScheduler(ml_predictor, interval_hours=24)
      scheduler.start()       # 启动后台调度
      scheduler.force_train() # 手动触发训练
      scheduler.stop()        # 停止调度
    """

    def __init__(self, predictor: MLPredictor, interval_hours: int = 24):
        """
        Args:
            predictor: MLPredictor 实例
            interval_hours: 重训练间隔（小时）
        """
        self.predictor = predictor
        self.interval_hours = interval_hours
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._last_train_result: Optional[Dict] = None
        self._is_training = False

    def start(self, on_startup: bool = True):
        """
        启动后台调度线程

        Args:
            on_startup: 是否在启动时立即检查并训练
        """
        if self._thread and self._thread.is_alive():
            logger.warning("[ModelTrainingScheduler] 调度器已在运行")
            return

        self._stop_event.clear()

        # 启动时检查: 如果模型不存在或已过期，立即训练
        if on_startup:
            if not self.predictor.is_trained or not self.predictor.is_model_fresh(self.interval_hours):
                logger.info("[ModelTrainingScheduler] 启动时触发模型训练...")
                # 在新线程中训练，不阻塞启动
                t = threading.Thread(
                    target=self._train_and_schedule,
                    name="ml-train-startup",
                    daemon=True,
                )
                t.start()
            else:
                logger.info("[ModelTrainingScheduler] 模型新鲜，跳过启动训练")
                # 设置定时器
                self._schedule_next()
        else:
            self._schedule_next()

        self._thread = threading.Thread(
            target=self._scheduler_loop,
            name="ml-train-scheduler",
            daemon=True,
        )
        self._thread.start()
        logger.info(f"[ModelTrainingScheduler] 调度器已启动，重训练间隔: {self.interval_hours}h")

    def stop(self):
        """停止后台调度"""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        logger.info("[ModelTrainingScheduler] 调度器已停止")

    def force_train(self, stock_code: str = 'sz300620') -> Dict:
        """
        手动触发模型训练

        Args:
            stock_code: 训练用的股票代码

        Returns:
            训练结果字典
        """
        with self._lock:
            if self._is_training:
                return {'success': False, 'message': '模型正在训练中，请稍后重试'}
            self._is_training = True

        try:
            logger.info(f"[ModelTrainingScheduler] 手动触发模型训练: {stock_code}")
            result = self._train_for_stock(stock_code)
            self._last_train_result = result
            # 训练成功后重新调度
            self._schedule_next()
            return result
        finally:
            with self._lock:
                self._is_training = False

    def get_status(self) -> Dict:
        """获取调度器状态"""
        with self._lock:
            is_training = self._is_training
        return {
            'running': self._thread is not None and self._thread.is_alive(),
            'interval_hours': self.interval_hours,
            'is_training': is_training,
            'is_model_trained': self.predictor.is_trained,
            'is_model_fresh': self.predictor.is_model_fresh(self.interval_hours) if self.predictor.is_trained else False,
            'trained_at': self.predictor._trained_at,
            'last_train_result': self._last_train_result,
        }

    def _scheduler_loop(self):
        """后台调度循环: 等待 interval_hours 后触发训练，然后重复"""
        while not self._stop_event.is_set():
            if self._stop_event.wait(self.interval_hours * 3600):
                break  # 收到停止信号

            if self._stop_event.is_set():
                break

            logger.info("[ModelTrainingScheduler] 定时重训练触发")
            self._train_and_schedule()

    def _schedule_next(self):
        """设置下一次训练的定时器（使用 Timer 而非 sleep，支持动态调整）"""
        # 使用 Timer 实现非阻塞定时
        timer = threading.Timer(
            self.interval_hours * 3600,
            self._on_timer_expire,
        )
        timer.daemon = True
        timer.start()

    def _on_timer_expire(self):
        """Timer 到期回调"""
        if not self._stop_event.is_set():
            logger.info("[ModelTrainingScheduler] 定时重训练触发")
            self._train_and_schedule()

    def _train_and_schedule(self):
        """训练模型并设置下一次调度"""
        try:
            # 默认训练 sz300620
            result = self._train_for_stock('sz300620')
            self._last_train_result = result
        except Exception as e:
            logger.error(f"[ModelTrainingScheduler] 训练失败: {e}")
            self._last_train_result = {'success': False, 'error': str(e)}
        finally:
            self._schedule_next()

    def _train_for_stock(self, stock_code: str) -> Dict:
        """
        为指定股票训练模型

        Args:
            stock_code: 股票代码

        Returns:
            训练结果字典
        """
        try:
            from modules.data_fetcher import StockDataFetcher

            fetcher = StockDataFetcher()
            stock_data = fetcher.get_stock_info(stock_code)
            klines = fetcher.get_kline_data(stock_code, 'daily', 300)

            if not klines or len(klines) < 60:
                return {'success': False, 'message': f'K 线数据不足: {len(klines) if klines else 0} 根'}

            if not stock_data:
                return {'success': False, 'message': '无法获取股票数据'}

            # 准备特征
            features = self.predictor.prepare_features(stock_data, klines)
            if features is None:
                return {'success': False, 'message': '特征准备失败'}

            labels = self.predictor.create_labels(klines, horizon=5)
            if len(labels) < 60:
                return {'success': False, 'message': f'标签不足: {len(labels)}'}

            full_features = self.predictor.prepare_features_batch(klines, labels)
            if full_features is None or len(full_features) < 100:
                return {'success': False, 'message': f'完整特征不足: {len(full_features) if full_features else 0}'}

            dates = [klines[i].get('date', f'day_{i}') for i in range(len(klines))]

            # 训练
            success = self.predictor.train_stacking_ensemble(full_features, labels, dates=dates)

            if success and self.predictor.is_trained:
                # 持久化
                path = self.predictor.save_model()
                return {
                    'success': True,
                    'stock_code': stock_code,
                    'cv_score': self.predictor.cv_score,
                    'models': list(self.predictor.models.keys()),
                    'path': path,
                    'trained_at': self.predictor._trained_at,
                }
            else:
                return {'success': False, 'message': '训练失败'}

        except Exception as e:
            logger.error(f"[ModelTrainingScheduler] 训练异常: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {'success': False, 'error': str(e)}


# 全局实例
ml_predictor = MLPredictor()
feature_engineering = FeatureEngineering()
model_training_scheduler = ModelTrainingScheduler(ml_predictor, interval_hours=24)
