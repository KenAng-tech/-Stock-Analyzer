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
from modules.adwin import ADWIN


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


# ── Page-Hinkley 检测器 ─────────────────────────────────────

class PageHinkleyDetector:
    """
    Page-Hinkley 渐进漂移检测器

    通过累计和检验检测均值渐变漂移。
    适合检测缓慢衰减的 alpha。

    参考: Hawkins, S. (1993). "Implementing Change Point Detection"
    """

    def __init__(self, delta: float = 0.005, threshold: float = 50.0):
        self.delta = delta
        self.threshold = threshold
        self._sum = 0.0
        self._mean = 0.0
        self._n = 0
        self._min_cumsum = float('inf')

    def add_value(self, value: float) -> bool:
        """添加新观测值，返回是否检测到漂移"""
        self._n += 1
        self._mean += (value - self._mean) / self._n
        self._sum += value - self._mean - self.delta
        self._min_cumsum = min(self._min_cumsum, self._sum)

        ph_stat = self._sum - self._min_cumsum
        return ph_stat > self.threshold

    def get_stat(self) -> float:
        """获取当前 Page-Hinkley 统计量"""
        return self._sum - self._min_cumsum

    def reset(self):
        """重置检测器"""
        self._sum = 0.0
        self._mean = 0.0
        self._n = 0
        self._min_cumsum = float('inf')


# ── KSWIN 检测器 ────────────────────────────────────────────

class KSWINDetector:
    """
    KSWIN (Kolmogorov-Smirnov Window) 分布漂移检测器

    使用 KS 两样本检验比较两个窗口的分布差异。
    比 ADWIN 更敏感于分布形状变化。

    设计要点:
    - 单样本累积: check_error() 每次传入一个样本，累积到 mini_window 后执行 KS 检验
    - 双窗口 KS: 前半段 vs 后半段，检测分布形状变化
    - 正确实现 KS 统计量和 p-value

    参考: Drucker, H. (2014). "Improving Rudimentary Drift Detection Schemes"
    """

    def __init__(self, alpha: float = 0.01, window_size: int = 100, mini_window: int = 10):
        self.alpha = alpha
        self.window_size = window_size
        self.mini_window = mini_window  # 累积单样本的窗口大小
        self._baseline: Optional[np.ndarray] = None
        self._baseline_set = False
        self._buffer: list = []         # 单样本累积缓冲区
        self._last_ks_stat: float = 0.0 # 上次 KS 统计量
        self._last_p_value: float = 1.0 # 上次 p-value

    def check(self, new_data: np.ndarray) -> bool:
        """
        检查新数据是否与基线分布有显著差异

        支持两种调用模式:
        1. 批量模式: new_data 长度 >= mini_window → 直接执行 KS 检验
        2. 单样本模式: new_data 长度 < mini_window → 累积后延迟检测

        Args:
            new_data: 新观测值数组 (可以是一个样本 [error])

        Returns:
            是否检测到漂移
        """
        # 累积单样本
        self._buffer.extend(new_data.tolist() if hasattr(new_data, 'tolist') else [new_data])

        if len(self._buffer) < self.mini_window:
            # 缓冲区未满，不检测
            return False

        # 取累积的 mini_window 个样本
        current_window = np.array(self._buffer[:self.mini_window])
        self._buffer = self._buffer[self.mini_window:]  # 移除已处理样本

        if not self._baseline_set:
            self._baseline = current_window.copy()
            self._baseline_set = True
            self._last_ks_stat = 0.0
            self._last_p_value = 1.0
            return False

        # KS 两样本检验: 基线 vs 当前窗口
        ks_stat, p_value = stats.ks_2samp(self._baseline, current_window)
        self._last_ks_stat = float(ks_stat)
        self._last_p_value = float(p_value)

        # 更新基线 (滑动窗口)
        if self._baseline is not None:
            combined = np.concatenate([self._baseline, current_window])
            if len(combined) > self.window_size:
                self._baseline = combined[-self.window_size:]
            else:
                self._baseline = combined

        return p_value < self.alpha

    def get_stat(self) -> Dict:
        """获取当前 KS 统计量 (正确实现)"""
        if not self._baseline_set or self._baseline is None:
            return {'ks_stat': 0.0, 'p_value': 1.0, 'baseline_set': False}

        return {
            'ks_stat': round(self._last_ks_stat, 6),
            'p_value': round(self._last_p_value, 6),
            'baseline_set': True,
            'baseline_size': len(self._baseline),
        }

    def reset(self):
        """重置检测器"""
        self._baseline = None
        self._baseline_set = False


# ── 三算法共识检测器 ────────────────────────────────────────

class DriftConsensusDetector:
    """
    漂移检测三算法共识

    使用 3 种算法独立检测，需要 2/3 同意才触发警告。
    - ADWIN: 均值漂移检测 (主检测器)
    - Page-Hinkley: 渐进漂移检测
    - KSWIN: 分布级漂移检测

    参考: 2026 量化最佳实践 — 减少误报率

    使用:
        detector = DriftConsensusDetector()
        result = detector.check_error(0.05)  # 预测误差 0.05
        if result['consensus']['drifted']:
            print("漂移检测!")
    """

    def __init__(
        self,
        adwin_delta: float = 0.01,
        ph_delta: float = 0.005,
        ph_threshold: float = 50.0,
        kswin_alpha: float = 0.01,
        kswin_window: int = 100,
    ):
        self.adwin = ADWIN(delta=adwin_delta)
        self.page_hinkley = PageHinkleyDetector(delta=ph_delta, threshold=ph_threshold)
        self.kswin = KSWINDetector(alpha=kswin_alpha, window_size=kswin_window)
        self._drift_count = 0

    def check_error(self, error: float) -> Dict:
        """
        基于预测误差检测漂移

        Args:
            error: 预测误差 (绝对值)

        Returns:
            {
                'adwin': {'drifted': bool, 'splits': int},
                'page_hinkley': {'drifted': bool, 'stat': float},
                'kswin': {'drifted': bool, 'stat': float},
                'consensus': {'drifted': bool, 'votes': int},
            }
        """
        votes = 0
        adwin_drifted = False
        ph_drifted = False
        kswin_drifted = False

        # 1. ADWIN
        adwin_drifted = self.adwin.add(error)
        if adwin_drifted:
            votes += 1

        # 2. Page-Hinkley
        ph_drifted = self.page_hinkley.add_value(error)
        if ph_drifted:
            votes += 1

        # 3. KSWIN (用误差序列)
        kswin_drifted = self.kswin.check(np.array([error]))
        if kswin_drifted:
            votes += 1

        if votes >= 2:
            self._drift_count += 1

        return {
            'adwin': {'drifted': adwin_drifted, 'splits': self.adwin.n_splits},
            'page_hinkley': {'drifted': ph_drifted, 'stat': round(self.page_hinkley.get_stat(), 6)},
            'kswin': {'drifted': kswin_drifted, 'stat': round(self.kswin.get_stat().get('ks_stat', 0), 6)},
            'consensus': {'drifted': votes >= 2, 'votes': votes, 'total': 3},
            'drift_count': self._drift_count,
        }

    def check_feature(self, X_new: np.ndarray) -> Dict:
        """
        基于特征分布检测漂移

        Args:
            X_new: (n_samples,) 新特征值

        Returns:
            检测结果
        """
        kswin_drifted = self.kswin.check(X_new)
        return {
            'kswin': {'drifted': kswin_drifted},
            'consensus': {'drifted': kswin_drifted, 'votes': 1 if kswin_drifted else 0, 'total': 1},
        }

    def get_status(self) -> Dict:
        """获取检测器状态"""
        return {
            'adwin_splits': self.adwin.n_splits,
            'adwin_window_size': self.adwin.window_size,
            'page_hinkley_stat': round(self.page_hinkley.get_stat(), 6),
            'total_drift_events': self._drift_count,
        }

    def reset(self):
        """重置所有检测器"""
        self.adwin = ADWIN(delta=self.adwin.delta)
        self.page_hinkley.reset()
        self.kswin.reset()
        self._drift_count = 0


# ── 全局实例 ─────────────────────────────────────────────────

# 三算法共识检测器 (推荐用于生产)
drift_consensus = DriftConsensusDetector()
