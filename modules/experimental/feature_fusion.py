#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
特征融合模块 — Feature Fusion

功能:
- 融合因子层 (21 因子) + ML 层 (12 维) + 时序编码 + 图嵌入
- 输出 100-200 维稠密特征向量
- 为后续 ML 模型提供更丰富的输入

设计原则:
- 与现有 temporal_encoder / graph_encoder 兼容
- 支持增量扩展 (新增因子/模型自动纳入)
- 输出维度可配置 (默认 128)
"""

import numpy as np
from typing import Dict, List, Optional
from modules.logger import logger


class FeatureFusion:
    """
    多源特征融合器

    融合路径:
    1. 因子层特征 (21 维, 来自 MultiFactorModelV2)
    2. ML 层特征 (12 维, 来自 MLPredictor.prepare_features)
    3. 时序统计特征 (8 维, 来自 TemporalEncoder)
    4. 图嵌入特征 (32 维, 来自 GraphFeatureExtractor)
    5. 基本面/情绪特征 (可变维度)

    最终输出: 通过线性投影 + 归一化到 fusion_dim 维
    """

    def __init__(self, fusion_dim: int = 128):
        """
        Args:
            fusion_dim: 输出特征维度, 默认 128
        """
        self.fusion_dim = fusion_dim
        self._input_dim = 0  # 将在 fuse() 时动态计算

    def fuse(
        self,
        factor_scores: Optional[Dict[str, float]] = None,
        ml_features: Optional[np.ndarray] = None,
        temporal_features: Optional[np.ndarray] = None,
        graph_features: Optional[np.ndarray] = None,
        fundamental_features: Optional[np.ndarray] = None,
        sentiment_score: Optional[float] = None,
    ) -> np.ndarray:
        """
        融合多源特征

        Args:
            factor_scores: 21 因子得分 dict {factor_name: value}
            ml_features: 12 维 ML 特征数组
            temporal_features: 时序统计特征数组 (8 维)
            graph_features: 图嵌入特征数组 (32 维)
            fundamental_features: 基本面特征数组 (可变维度)
            sentiment_score: 情感得分 (标量)

        Returns:
            融合特征向量 (fusion_dim,)
        """
        parts: List[np.ndarray] = []

        # 1. 因子层特征
        if factor_scores:
            # 按因子名称顺序排列, 保证维度一致
            default_factors = [
                'momentum_1d', 'momentum_5d', 'momentum_20d',
                'pe_value', 'pb_value', 'ps_value', 'pcf_value',
                'realized_vol', 'realized_vol_5d', 'realized_vol_20d',
                'volume_ratio', 'volume_momentum', 'turnover_level',
                'turnover_change', 'roe_quality', 'roa_quality',
                'gross_margin', 'net_margin', 'debt_ratio',
                'rsi_technical', 'macd_signal',
            ]
            vec = np.array([factor_scores.get(f, 0.0) for f in default_factors], dtype=np.float64)
            # Min-Max 标准化到 [0, 1]
            vec_min, vec_max = vec.min(), vec.max()
            if vec_max > vec_min:
                vec = (vec - vec_min) / (vec_max - vec_min)
            else:
                vec = np.zeros_like(vec)
            parts.append(vec)

        # 2. ML 层特征
        if ml_features is not None:
            parts.append(np.asarray(ml_features, dtype=np.float64))

        # 3. 时序特征
        if temporal_features is not None:
            parts.append(np.asarray(temporal_features, dtype=np.float64))

        # 4. 图嵌入特征
        if graph_features is not None:
            parts.append(np.asarray(graph_features, dtype=np.float64))

        # 5. 基本面特征
        if fundamental_features is not None:
            parts.append(np.asarray(fundamental_features, dtype=np.float64))

        # 6. 情感得分
        if sentiment_score is not None:
            parts.append(np.array([sentiment_score], dtype=np.float64))

        if not parts:
            # 全部为空, 返回零向量
            return np.zeros(self.fusion_dim)

        # 拼接所有特征
        fused = np.concatenate(parts)

        # L2 归一化
        norm = np.linalg.norm(fused)
        if norm > 1e-10:
            fused = fused / norm

        # 投影到目标维度
        if len(fused) >= self.fusion_dim:
            # 随机投影降维 (Johnson-Lindenstrauss 引理)
            return self._random_projection(fused, self.fusion_dim)
        else:
            # 零填充升维
            padded = np.zeros(self.fusion_dim)
            padded[:len(fused)] = fused
            return padded

    @staticmethod
    def _random_projection(x: np.ndarray, n_components: int) -> np.ndarray:
        """
        随机投影降维 (近似 PCA, 计算高效)

        使用 Johnson-Lindenstrauss 引理: 随机投影可以近似保持高维数据的成对距离。
        """
        # 使用固定种子保证可复现性
        rng = np.random.RandomState(42)
        projection_matrix = rng.normal(0, 1 / np.sqrt(x.shape[0]), size=(x.shape[0], n_components))
        return x @ projection_matrix

    @staticmethod
    def fuse_factor_ml_features(
        factor_scores: Dict[str, float],
        ml_features: np.ndarray,
        fusion_dim: int = 64
    ) -> np.ndarray:
        """
        快捷方法: 仅融合因子层 + ML 层特征 (最常用场景)

        Args:
            factor_scores: 21 因子得分
            ml_features: 12 维 ML 特征
            fusion_dim: 输出维度

        Returns:
            融合特征向量 (fusion_dim,)
        """
        ff = FeatureFusion(fusion_dim=fusion_dim)
        return ff.fuse(
            factor_scores=factor_scores,
            ml_features=ml_features,
        )


class FeatureSelector:
    """
    特征选择器 — 从融合特征中筛选最有价值的子集

    使用基于互信息 (MI) 和特征重要性的选择策略
    """

    @staticmethod
    def select_top_k(
        feature_importances: Dict[str, float],
        k: int = 32
    ) -> List[str]:
        """
        按特征重要性选择前 K 个特征

        Args:
            feature_importances: {feature_name: importance_score}
            k: 选择数量

        Returns:
            前 K 个特征名称列表
        """
        sorted_features = sorted(
            feature_importances.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        return [name for name, _ in sorted_features[:k]]

    @staticmethod
    def select_by_threshold(
        feature_importances: Dict[str, float],
        threshold: float = 0.01
    ) -> List[str]:
        """
        按重要性阈值选择特征

        Args:
            feature_importances: {feature_name: importance_score}
            threshold: 最小重要性阈值

        Returns:
            满足阈值的特征名称列表
        """
        return [
            name for name, imp in feature_importances.items()
            if abs(imp) >= threshold
        ]


# 全局实例 (供 dashboard_api 使用)
feature_fusion = FeatureFusion(fusion_dim=128)
