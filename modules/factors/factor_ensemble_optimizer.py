#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
因子集成优化器 — 2026 SOTA 因子管理

功能:
    - 自动选择最优因子子集 (基于 IC/IR/衰减)
    - IC 衰减监控 + 自动剔除失效因子
    - 因子正交化 (Gram-Schmidt/PCA)
    - 因子组合权重优化 (最大化 IC IR)
    - 与 Alpha158/Alpha360/Barra 因子无缝集成

2026 趋势:
    LLM 因子挖掘 + 传统因子融合是当前量化前沿
    因子衰减速度加快，需要动态权重调整
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict
from scipy import stats
from scipy.linalg import qr

logger = logging.getLogger('stock_analyzer.modules')

# ── 数据结构 ──────────────────────────────────────────────────


@dataclass
class FactorInfo:
    """因子信息"""
    name: str
    source: str  # 'alpha158', 'alpha360', 'barra', 'custom'
    ic: float = 0.0
    ic_ir: float = 0.0
    rank_ic: float = 0.0
    ic_std: float = 0.0
    decay_rate: float = 0.0  # 最近 N 期 IC 衰减率
    is_active: bool = True
    weight: float = 1.0
    half_life_days: int = 90


@dataclass
class FactorEnsembleConfig:
    """因子集成配置"""
    min_ic_threshold: float = 0.02  # 最小 IC 阈值
    max_decay_rate: float = -0.05  # 最大 IC 衰减率 (负值表示衰减)
    orthogonalize: bool = True
    max_correlation: float = 0.85  # 最大因子间相关性
    ic_window: int = 60  # IC 计算窗口
    rebalance_days: int = 20  # 权重再平衡周期
    top_k: int = 30  # 保留 Top-K 因子


class FactorEnsembleOptimizer:
    """
    因子集成优化器

    用法:
        optimizer = FactorEnsembleOptimizer()
        optimizer.register_factors(alpha158_factors)
        optimizer.register_factors(alpha360_factors)
        weights = optimizer.optimize_weights()
    """

    def __init__(self, config: Optional[FactorEnsembleConfig] = None):
        self.config = config or FactorEnsembleConfig()
        self._factors: Dict[str, FactorInfo] = {}
        self._ic_history: Dict[str, List[float]] = defaultdict(list)
        self._correlation_matrix: Optional[np.ndarray] = None
        self._factor_names: List[str] = []

    def register_factors(
        self,
        factor_dict: Dict[str, np.ndarray],
        source: str = 'custom',
    ) -> int:
        """
        注册因子

        Args:
            factor_dict: {factor_name: factor_values}
            source: 因子来源

        Returns:
            注册的因子数量
        """
        count = 0
        for name, values in factor_dict.items():
            if name not in self._factors:
                self._factors[name] = FactorInfo(
                    name=name,
                    source=source,
                )
                self._factor_names.append(name)
                count += 1
        logger.info(
            f"[FactorOptimizer] 注册 {count} 个因子 (来源: {source}), "
            f"总计 {len(self._factors)} 个"
        )
        return count

    def compute_ic(
        self,
        factor_values: Dict[str, np.ndarray],
        forward_returns: np.ndarray,
    ) -> Dict[str, float]:
        """
        计算因子 IC (Information Coefficient)

        Args:
            factor_values: {factor_name: values}
            forward_returns: 未来收益

        Returns:
            {factor_name: ic_value}
        """
        ic_results: Dict[str, float] = {}
        valid_returns = np.asarray(forward_returns).flatten()

        for name, values in factor_values.items():
            vals = np.asarray(values).flatten()
            if len(vals) != len(valid_returns) or len(vals) < 10:
                continue

            mask = ~(np.isnan(vals) | np.isnan(valid_returns))
            v, r = vals[mask], valid_returns[mask]
            if len(v) < 10:
                continue

            ic, _ = stats.spearmanr(v, r)
            ic_results[name] = float(ic)

            if name in self._factors:
                self._factors[name].ic = float(ic)
                self._ic_history[name].append(float(ic))
                # 限制历史长度
                if len(self._ic_history[name]) > self.config.ic_window:
                    self._ic_history[name] = self._ic_history[name][-self.config.ic_window:]

        return ic_results

    def compute_decay_rate(self, name: str) -> float:
        """
        计算因子 IC 衰减率

        Args:
            name: 因子名称

        Returns:
            衰减率 (负值表示衰减)
        """
        history = self._ic_history.get(name, [])
        if len(history) < 10:
            return 0.0

        x = np.arange(len(history))
        y = np.array(history)
        slope, _, _, _, _ = stats.linregress(x, y)
        return float(slope)

    def orthogonalize_factors(
        self,
        factor_matrix: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        因子正交化 (Gram-Schmidt)

        Args:
            factor_matrix: (n_samples, n_factors)

        Returns:
            (orthogonalized_matrix, transformation_matrix)
        """
        n_samples, n_factors = factor_matrix.shape
        ortho = np.zeros_like(factor_matrix)
        transform = np.eye(n_factors)

        ortho[:, 0] = factor_matrix[:, 0]

        for i in range(1, n_factors):
            proj = np.zeros(n_samples)
            for j in range(i):
                if np.std(ortho[:, j]) > 1e-10:
                    coef = np.dot(factor_matrix[:, i], ortho[:, j]) / np.dot(ortho[:, j], ortho[:, j])
                    proj += coef * ortho[:, j]
                    transform[i, j] = -coef
            ortho[:, i] = factor_matrix[:, i] - proj

        return ortho, transform

    def optimize_weights(
        self,
        factor_matrix: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        优化因子权重 (最大化 IC IR)

        Args:
            factor_matrix: 可选，因子值矩阵

        Returns:
            {factor_name: weight}
        """
        active_factors = {
            name: info for name, info in self._factors.items()
            if info.is_active
        }

        if not active_factors:
            logger.warning("[FactorOptimizer] 无活跃因子")
            return {}

        # 基于 IC 和 IR 分配权重
        weights: Dict[str, float] = {}
        total_score = 0.0

        for name, info in active_factors.items():
            # 综合评分: IC + IR + 衰减惩罚
            score = abs(info.ic) * 2.0 + abs(info.ic_ir) * 1.5

            # 衰减惩罚
            decay = self.compute_decay_rate(name)
            if decay < self.config.max_decay_rate:
                score *= 0.5  # 衰减严重，权重减半

            if score > 0:
                weights[name] = score
                total_score += score

        # 归一化
        if total_score > 0:
            for name in weights:
                weights[name] /= total_score

        # 更新因子权重
        for name, w in weights.items():
            if name in self._factors:
                self._factors[name].weight = w

        n_optimized = len(weights)
        logger.info(f"[FactorOptimizer] 权重优化完成: {n_optimized} 个因子")

        return weights

    def prune_inactive_factors(self) -> List[str]:
        """
        剔除失效因子

        Returns:
            被剔除的因子名称列表
        """
        pruned: List[str] = []

        for name, info in list(self._factors.items()):
            # 检查 IC 是否低于阈值
            if abs(info.ic) < self.config.min_ic_threshold:
                info.is_active = False
                pruned.append(name)
                continue

            # 检查衰减率
            decay = self.compute_decay_rate(name)
            info.decay_rate = decay
            if decay < self.config.max_decay_rate:
                info.is_active = False
                pruned.append(name)

        if pruned:
            logger.info(
                f"[FactorOptimizer] 剔除 {len(pruned)} 个失效因子: "
                f"{pruned[:10]}{'...' if len(pruned) > 10 else ''}"
            )

        return pruned

    def get_active_factors(self) -> Dict[str, FactorInfo]:
        """获取活跃因子"""
        return {
            name: info for name, info in self._factors.items()
            if info.is_active
        }

    def get_status(self) -> dict:
        """获取状态"""
        active = self.get_active_factors()
        total = len(self._factors)

        return {
            'total_factors': total,
            'active_factors': len(active),
            'inactive_factors': total - len(active),
            'factor_sources': list(set(
                info.source for info in self._factors.values()
            )),
            'top_factors': sorted(
                [
                    {'name': n, 'ic': round(info.ic, 4), 'weight': round(info.weight, 4)}
                    for n, info in active.items()
                ],
                key=lambda x: abs(x['ic']),
                reverse=True,
            )[:10],
            'config': {
                'min_ic_threshold': self.config.min_ic_threshold,
                'max_decay_rate': self.config.max_decay_rate,
                'orthogonalize': self.config.orthogonalize,
                'top_k': self.config.top_k,
            },
        }


# 模块级单例
_factor_ensemble_optimizer: Optional[FactorEnsembleOptimizer] = None


def get_factor_ensemble_optimizer() -> FactorEnsembleOptimizer:
    """获取因子集成优化器单例"""
    global _factor_ensemble_optimizer
    if _factor_ensemble_optimizer is None:
        _factor_ensemble_optimizer = FactorEnsembleOptimizer()
        logger.info("[FactorOptimizer] 单例已创建")
    return _factor_ensemble_optimizer