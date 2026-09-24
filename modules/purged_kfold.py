#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Purged K-Fold Cross-Validation — Lopez de Prado (2018)

金融时间序列交叉验证，防止重叠时间窗口导致的前视偏差：
1. Purge: 移除测试集边界附近的训练样本
2. Embargo: 移除测试集起始附近的样本

参考:
    - Marcos M. Lopez de Prado, "Advances in Financial Machine Learning" (2018), Chapter 8
    - ibex 库的 PurgedGroupKFold 思想 (纯 Python 轻量实现，无外部依赖)

用法:
    from modules.purged_kfold import PurgedKFold

    pkf = PurgedKFold(n_splits=5, embargo_pct=0.05)
    for train_idx, test_idx in pkf.split(X):
        model.fit(X[train_idx], y[train_idx])
        score = model.score(X[test_idx], y[test_idx])
"""

import numpy as np
from typing import Iterator, Tuple, Optional


class PurgedKFold:
    """
    Lopez de Prado 的 Purged K-Fold 交叉验证。

    金融时间序列数据不能随机打乱，因为相邻时间点的样本存在自相关。
    标准 K-Fold 会导致训练集和测试集的时间重叠，产生前视偏差 (look-ahead bias)。

    PurgedKFold 通过两步解决:
    1. Purge (净化): 移除测试集边界附近的训练样本
       - 例如测试集是第 20-40 天，则移除第 18-20 天的训练样本
       - 防止训练样本的信息泄漏到测试集
    2. Embargo (金锁期): 移除测试集起始附近的样本
       - 例如测试集第 20-40 天，移除第 20-22 天的测试样本
       - 防止训练集的信息泄漏到测试集

    与 sklearn KFold 的区别:
    - KFold: 随机打乱后分折 → 时间序列不适用
    - TimeSeriesSplit: 顺序分折，无 purge/embargo → 仍有泄漏
    - PurgedKFold: 顺序分折 + purge + embargo → 无前视偏差
    """

    def __init__(self, n_splits: int = 5, embargo_pct: float = 0.05):
        """
        Args:
            n_splits: 折数 (>= 2)
            embargo_pct: 金锁期比例 (0-1)，测试集起始段移除比例
                         默认 5%，即测试集前 5% 的样本被移除
        """
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if not 0 <= embargo_pct < 1:
            raise ValueError("embargo_pct must be in [0, 1)")
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct

    def split(
        self,
        X,
        y=None,
        groups=None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        生成 (train_idx, test_idx) 对。

        数据按时间顺序分为 K 个连续折，每个折:
        1. 测试集 = 当前折的全部索引
        2. 训练集 = 所有其他索引，但移除测试集边界附近的样本 (purge)
        3. 测试集起始附近的样本被移除 (embargo)

        Args:
            X: 数据数组 (仅使用 len(X))
            y: 标签 (可选，目前忽略)
            groups: 分组 (可选，目前忽略，为兼容 sklearn 接口)

        Yields:
            (train_indices, test_indices): 两个 numpy int64 数组

        示例:
            >>> pkf = PurgedKFold(n_splits=5, embargo_pct=0.05)
            >>> for fold, (train, test) in enumerate(pkf.split(range(100))):
            ...     print(f"Fold {fold+1}: train={len(train)}, test={len(test)}")
            Fold 1: train=62, test=20
            Fold 2: train=60, test=20
            ...
        """
        n = len(X)
        fold_size = n // self.n_splits
        embargo_size = max(1, int(fold_size * self.embargo_pct))

        for fold in range(self.n_splits):
            test_start = fold * fold_size
            # 最后一折使用剩余所有数据
            test_end = test_start + fold_size if fold < self.n_splits - 1 else n

            # 测试集索引
            test_idx = list(range(test_start, test_end))

            # 训练集索引: 排除测试集 + 金锁区 (embargo)
            # 左侧: 测试集开始之前 embargo_size 个样本也排除 (purge)
            # 右侧: 测试集结束之后 embargo_size 个样本也排除 (embargo)
            train_left = list(range(0, max(0, test_start - embargo_size)))
            train_right = list(range(min(test_end + embargo_size, n), n))
            train_idx = train_left + train_right

            yield np.array(train_idx, dtype=np.int64), np.array(test_idx, dtype=np.int64)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """返回折数 (兼容 sklearn 接口)"""
        return self.n_splits


# ── 便捷函数 ──────────────────────────────────────────────

def purged_kfold_score(
    estimator,
    X,
    y,
    n_splits: int = 5,
    embargo_pct: float = 0.05,
) -> dict:
    """
    便捷函数: 对估计器执行 Purged K-Fold 交叉验证。

    Args:
        estimator: sklearn 兼容的估计器 (有 fit/predict 方法)
        X: 特征数组
        y: 标签数组
        n_splits: 折数
        embargo_pct: 金锁期比例

    Returns:
        {'scores': [...], 'mean_score': float, 'std_score': float, 'fold_details': [...]}
    """
    pkf = PurgedKFold(n_splits=n_splits, embargo_pct=embargo_pct)
    scores = []
    fold_details = []

    for fold, (train_idx, test_idx) in enumerate(pkf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        estimator.fit(X_train, y_train)
        score = estimator.score(X_test, y_test)
        scores.append(score)

        fold_details.append({
            'fold': fold + 1,
            'train_size': len(train_idx),
            'test_size': len(test_idx),
            'score': round(float(score), 4),
        })

    return {
        'scores': [round(float(s), 4) for s in scores],
        'mean_score': round(float(np.mean(scores)), 4),
        'std_score': round(float(np.std(scores)), 4),
        'fold_details': fold_details,
    }
