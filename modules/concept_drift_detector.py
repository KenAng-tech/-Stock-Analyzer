#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
概念漂移检测 — ADWIN + KS 检验

当市场 regime 变化时，模型的预测性能会快速退化。
本模块实时检测概念漂移，触发模型重新训练。

算法:
    1. ADWIN (Adaptive Windowing):
       - 维护一个可变大小的滑动窗口
       - 当窗口可以分割为两个子窗口，且均值差异显著时，检测到漂移
       - 自动调整窗口大小，丢弃旧数据

    2. KS 检验 (Kolmogorov-Smirnov):
       - 比较特征分布的累积分布函数 (CDF)
       - 当 KS 统计量超过阈值时，检测到分布漂移

参考:
    - Bifet & Gavalda, "Adaptive Windowing for Data Stream Classification" (2007)
    - 特征分布漂移: KS 检验
"""

import numpy as np
from typing import Dict, Optional, List
from collections import deque
from scipy import stats
from modules.logger import logger


# ── ADWIN (Adaptive Windowing) ────────────────────────────────

class ADWIN:
    """
    ADWIN (Adaptive Windowing) 漂移检测算法

    核心思想:
        - 维护一个可变大小的窗口 W
        - 定期检查 W 是否可以分割为 W0 和 W1，使得 |mean(W0) - mean(W1)| > ε
        - 如果检测到显著差异，则丢弃 W0 中较老的部分
        - ε 依赖于 delta (置信度参数) 和 |W0|, |W1|

    参数:
        delta: 置信度参数 (越小越敏感，默认 0.01)
        max_window: 最大窗口大小 (默认 1000)
    """

    def __init__(self, delta: float = 0.01, max_window: int = 1000):
        self.delta = delta
        self.max_window = max_window
        self.window = deque()
        self._n_splits = 0  # 检测到漂移的次数
        self._initial_size = 30  # 最小窗口大小

    def add(self, value: float) -> bool:
        """
        添加一个新的观测值

        Args:
            value: 观测值 (如预测误差)

        Returns:
            是否检测到概念漂移
        """
        self.window.append(value)

        # 限制窗口大小
        if len(self.window) > self.max_window:
            self.window.popleft()

        # 窗口太小，不检查
        if len(self.window) < self._initial_size:
            return False

        return self._check_split()

    def _check_split(self) -> bool:
        """
        检查窗口是否可以分割为两个统计上不同的子窗口

        Returns:
            是否检测到漂移
        """
        n = len(self.window)

        # 尝试所有可能的分割点
        for cut in range(self._initial_size, n - self._initial_size):
            n0 = cut
            n1 = n - cut

            if n0 < self._initial_size or n1 < self._initial_size:
                continue

            mean0 = np.mean(self.window[:cut])
            mean1 = np.mean(self.window[cut:])

            # 理论上的 ε 界 (Hoeffding-Serfling 不等式)
            delta_prime = np.log(4.0 / self.delta)
            m = (n0 * n1) / (n0 + n1)
            epsilon = np.sqrt((delta_prime / (2.0 * m)) + (delta_prime / (6.0 * n) * np.log(4.0 / self.delta)))

            if abs(mean0 - mean1) > epsilon:
                # 检测到漂移 — 丢弃旧窗口
                self.window = deque(list(self.window)[cut:])
                self._n_splits += 1
                return True

        return False

    @property
    def n_splits(self) -> int:
        return self._n_splits

    @property
    def window_size(self) -> int:
        return len(self.window)

    @property
    def window_mean(self) -> float:
        return float(np.mean(self.window)) if self.window else 0.0

    @property
    def window_std(self) -> float:
        return float(np.std(self.window)) if self.window else 0.0


# ── 概念漂移检测器 ────────────────────────────────────────────

class ConceptDriftDetector:
    """
    概念漂移检测器 — ADWIN + KS 检验双检测

    检测两种漂移:
        1. 预测误差漂移 (ADWIN): 监控预测误差的均值是否变化
        2. 特征分布漂移 (KS 检验): 监控输入特征分布是否变化

    当检测到漂移时:
        - 返回 drift_detected=True
        - 建议触发模型重新训练
    """

    def __init__(
        self,
        ks_alpha: float = 0.01,
        adwin_delta: float = 0.01,
        adwin_max_window: int = 1000,
    ):
        self.adwin = ADWIN(delta=adwin_delta, max_window=adwin_max_window)
        self.ks_alpha = ks_alpha
        self._baseline_features: Optional[np.ndarray] = None
        self._baseline_updated = False
        self._prediction_errors = deque(maxlen=1000)
        self._drift_count = 0
        self._last_drift_time: Optional[int] = None

    def check_prediction_drift(self, prediction: float, actual: float) -> bool:
        """
        基于预测误差检测漂移

        Args:
            prediction: 预测值 (如预测收益率)
            actual: 实际值 (如实际收益率)

        Returns:
            是否检测到漂移
        """
        error = abs(prediction - actual)
        self._prediction_errors.append(error)
        return self.adwin.add(error)

    def check_feature_drift(self, X_new: np.ndarray) -> bool:
        """
        KS 检验检测特征分布漂移

        Args:
            X_new: (n_samples, n_features) 新样本的特征矩阵

        Returns:
            是否检测到漂移
        """
        if self._baseline_features is None or X_new.shape[1] != self._baseline_features.shape[1]:
            self._baseline_features = X_new.copy()
            self._baseline_updated = True
            return False

        n_features = X_new.shape[1]
        drift_found = False

        for i in range(n_features):
            col_old = self._baseline_features[:, i]
            col_new = X_new[:, i]

            # 标准化 (用基线统计量)
            mean_old = np.mean(col_old)
            std_old = np.std(col_old) + 1e-8
            col_old_norm = (col_old - mean_old) / std_old
            col_new_norm = (col_new - mean_old) / std_old

            # KS 检验
            ks_stat, ks_pvalue = stats.ks_2samp(col_old_norm, col_new_norm)

            if ks_pvalue < self.ks_alpha:
                drift_found = True
                logger.warning(
                    f"[Drift] Feature {i} distribution drift detected "
                    f"(KS stat={ks_stat:.4f}, p={ks_pvalue:.4f})"
                )

        if drift_found:
            self._drift_count += 1

        return drift_found

    def update_baseline(self, X: np.ndarray):
        """
        更新基线特征统计量

        应在模型重新训练后调用，用新数据更新基线。
        """
        self._baseline_features = X.copy()
        self._baseline_updated = True
        self.adwin = ADWIN(delta=self.adwin.delta, max_window=self.adwin.max_window)
        self._prediction_errors.clear()
        logger.info("[Drift] Baseline statistics updated")

    def get_drift_status(self) -> Dict:
        """获取漂移检测状态"""
        return {
            'adwin_splits': self.adwin.n_splits,
            'window_size': self.adwin.window_size,
            'window_mean_error': round(self.adwin.window_mean, 6),
            'window_std_error': round(self.adwin.window_std, 6),
            'total_drift_events': self._drift_count,
            'baseline_updated': self._baseline_updated,
            'recommendation': 'retrain_model' if self.adwin.n_splits > 0 else 'model_ok',
        }

    def reset(self):
        """重置检测器"""
        self.adwin = ADWIN(delta=self.adwin.delta, max_window=self.adwin.max_window)
        self._prediction_errors.clear()
        self._drift_count = 0
        self._baseline_features = None
        self._baseline_updated = False
        logger.info("[Drift] Detector reset")


# ── 全局实例 ──────────────────────────────────────────────────

drift_detector = ConceptDriftDetector()
