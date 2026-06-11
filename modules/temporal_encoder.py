#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
时序特征编码器 — Temporal Encoder

用 numpy 统计方法将历史因子序列编码为固定长度向量。
捕捉"因子动量"模式: 趋势、稳定性、自相关性、偏度等。

架构:
  历史因子序列 (seq_len, n_factors)
    → 8 个时序统计量 × n_factors = 8*n_factors 维时序特征
    → 拼接最新因子值 → (9*n_factors,) 维
    → 非线性变换 → (fusion_dim,) 维

TODO: PyTorch GRU 接口预留 (见 _gru_encode 方法)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from modules.logger import logger


class TemporalEncoder:
    """
    时序特征编码器

    将历史因子序列通过统计方法编码为固定长度向量。
    每个因子计算 8 个统计量:
      mean, std, slope, autocorr, max_drawdown, sharpe, skewness, kurtosis
    """

    def __init__(self, n_factors: int = 15, seq_len: int = 20):
        self.n_factors = n_factors
        self.seq_len = seq_len
        self.temporal_dim = n_factors * 8  # 8 个统计量

    def extract_sequence(
        self,
        historical_factors: List[Dict[str, float]],
        latest_factors: Dict[str, float],
        factor_names: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        从历史因子数据提取时序特征

        Args:
            historical_factors: [{factor_name: value}, ...] 过去 N 天的因子值
            latest_factors: 最新一天的因子值
            factor_names: 因子名称列表 (可选)

        Returns:
            时序特征向量 (9 * n_factors,)
              - 前 8*n_factors: 统计量编码
              - 后 n_factors: 最新因子值
        """
        if not historical_factors:
            return np.zeros(self.temporal_dim + self.n_factors)

        if factor_names is None:
            factor_names = list(latest_factors.keys())

        n = len(factor_names)
        stats = np.zeros((self.seq_len, n))

        for i, fname in enumerate(factor_names):
            values = [h.get(fname, 0.0) for h in historical_factors[-self.seq_len:]]
            if len(values) < 2:
                values.append(latest_factors.get(fname, 0.0))

            values = np.array(values, dtype=float)
            # 处理 NaN
            valid = ~np.isnan(values)
            if np.sum(valid) < 2:
                stats[:len(values), i] = values
                continue

            stats[:len(values), i] = values
            # 后续行用 0 填充

        # 计算 8 个统计量
        temporal_feats = np.zeros(n * 8)
        for j, fname in enumerate(factor_names):
            col = stats[:, j]
            valid = col[col != 0]  # 简化: 非零值
            if len(valid) < 2:
                valid = np.array([col[-1] if col[-1] != 0 else 0.0])

            temporal_feats[j * 8: j * 8 + 8] = self._compute_statistics(valid)

        # 拼接最新因子值
        latest_vec = np.array([latest_factors.get(fname, 0.0) for fname in factor_names])
        result = np.concatenate([temporal_feats, latest_vec])

        return result

    def _compute_statistics(self, values: np.ndarray) -> np.ndarray:
        """计算 8 个时序统计量"""
        stats = np.zeros(8)

        # 1. Mean (均值)
        stats[0] = float(np.mean(values))

        # 2. Std (标准差)
        stats[1] = float(np.std(values))

        # 3. Slope (线性回归斜率)
        if len(values) >= 3:
            x = np.arange(len(values))
            x_mean = np.mean(x)
            y_mean = np.mean(values)
            numerator = np.sum((x - x_mean) * (values - y_mean))
            denominator = np.sum((x - x_mean) ** 2)
            stats[2] = float(numerator / (denominator + 1e-10))

        # 4. Autocorrelation (1阶自相关)
        if len(values) >= 4:
            centered = values - np.mean(values)
            autocorr = np.sum(centered[:-1] * centered[1:]) / (np.sum(centered ** 2) + 1e-10)
            stats[3] = float(autocorr)

        # 5. Max Drawdown (最大回撤)
        if len(values) >= 2:
            peak = values[0]
            max_dd = 0.0
            for v in values:
                if v > peak:
                    peak = v
                dd = (peak - v) / (peak + 1e-10)
                if dd > max_dd:
                    max_dd = dd
            stats[4] = float(max_dd)

        # 6. Sharpe Ratio (风险调整收益)
        if len(values) >= 2:
            mean_ret = np.mean(np.diff(values))
            std_ret = np.std(np.diff(values))
            stats[5] = float(mean_ret / (std_ret + 1e-10))

        # 7. Skewness (偏度)
        if len(values) >= 3:
            m = np.mean(values)
            s = np.std(values)
            if s > 0:
                stats[6] = float(np.mean(((values - m) / s) ** 3))

        # 8. Kurtosis (峰度)
        if len(values) >= 4:
            m = np.mean(values)
            s = np.std(values)
            if s > 0:
                stats[7] = float(np.mean(((values - m) / s) ** 4) - 3)

        return stats

    def _gru_encode(self, sequence: np.ndarray) -> np.ndarray:
        """
        TODO: PyTorch GRU 接口预留

        当前使用 numpy 统计方法。如需使用 GRU:

        ```python
        import torch
        import torch.nn as nn

        class GRUModel(nn.Module):
            def __init__(self, input_size, hidden_size=64):
                super().__init__()
                self.gru = nn.GRU(input_size, hidden_size, batch_first=True)

            def forward(self, x):
                out, hidden = self.gru(x)
                return hidden[-1]  # (batch, hidden_size)
        ```

        Args:
            sequence: (seq_len, n_factors)

        Returns:
            hidden vector (hidden_size,)
        """
        # Placeholder: 当前返回统计编码
        return self._compute_statistics(sequence[:, 0])


class TemporalFeatureFusion:
    """
    时序特征融合器

    将截面特征和时序特征拼接并通过非线性变换融合。

    架构:
      截面特征 (D1,) + 时序编码 (D2,) → 拼接 (D1+D2,)
        → Linear → GELU → (fusion_dim,)
    """

    def __init__(self, cross_sectional_dim: int = 15,
                 temporal_dim: int = 120, fusion_dim: int = 128):
        self.cross_sectional_dim = cross_sectional_dim
        self.temporal_dim = temporal_dim
        self.fusion_dim = fusion_dim
        self.input_dim = cross_sectional_dim + temporal_dim

        # 简化版: 用 PCA 降维 (numpy)
        # TODO: 可替换为 nn.Linear + GELU

    def fuse(self, cross_sectional: np.ndarray,
             temporal: np.ndarray) -> np.ndarray:
        """
        融合截面特征和时序特征

        Args:
            cross_sectional: (cross_sectional_dim,)
            temporal: (temporal_dim,)

        Returns:
            融合特征 (fusion_dim,)
        """
        # 拼接
        combined = np.concatenate([cross_sectional, temporal])

        # 简化: L2 归一化 + 截断/填充到 fusion_dim
        combined = combined / (np.linalg.norm(combined) + 1e-10)

        if len(combined) >= self.fusion_dim:
            # PCA 降维 (取前 fusion_dim 个主成分)
            return self._pca_reduce(combined, self.fusion_dim)
        else:
            # 零填充
            padded = np.zeros(self.fusion_dim)
            padded[:len(combined)] = combined
            return padded

    @staticmethod
    def _pca_reduce(x: np.ndarray, n_components: int) -> np.ndarray:
        """简化 PCA 降维"""
        if len(x) <= n_components:
            return x

        # 用随机投影近似 PCA (Johnson-Lindenstrauss)
        rng = np.random.RandomState(42)
        projection = rng.randn(len(x), n_components)
        return x @ projection / np.sqrt(n_components)

    def get_fused_dimension(self) -> int:
        return self.fusion_dim


# 全局实例
temporal_encoder = TemporalEncoder(n_factors=15, seq_len=20)
temporal_fusion = TemporalFeatureFusion(cross_sectional_dim=15, temporal_dim=120, fusion_dim=128)
