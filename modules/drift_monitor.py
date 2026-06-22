#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
概念漂移检测集成模块 — 集成到主系统

功能:
1. 封装 ConceptDriftDetector 统一接口
2. 与 dashboard_api 集成
3. 自动触发模型重新训练
4. 实时监控预测误差和特征分布

作者: 基于 Bifet & Gavalda (2007)
参考: modules/concept_drift_detector.py
"""

import os
import pickle
import numpy as np
from typing import Dict, Optional, List, Tuple
from datetime import datetime
from collections import deque
import threading

from modules.logger import logger


# ── ADWIN 实现 ────────────────────────────────────────────────

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
        self._n_splits = 0
        self._initial_size = 30

    def add(self, value: float) -> bool:
        """添加观测值，返回是否检测到漂移"""
        self.window.append(value)

        if len(self.window) > self.max_window:
            self.window.popleft()

        if len(self.window) < self._initial_size:
            return False

        return self._check_split()

    def _check_split(self) -> bool:
        """检查窗口是否可以分割"""
        n = len(self.window)

        for cut in range(self._initial_size, n - self._initial_size):
            n0 = cut
            n1 = n - cut

            if n0 < self._initial_size or n1 < self._initial_size:
                continue

            mean0 = np.mean(self.window[:cut])
            mean1 = np.mean(self.window[cut:])

            delta_prime = np.log(4.0 / self.delta)
            m = (n0 * n1) / (n0 + n1)
            epsilon = np.sqrt((delta_prime / (2.0 * m)) + (delta_prime / (6.0 * n) * np.log(4.0 / self.delta)))

            if abs(mean0 - mean1) > epsilon:
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


# ── 概念漂移检测器 ─────────────────────────────────────────

class DriftMonitor:
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
        error_threshold: float = 0.1,
    ):
        """
        Args:
            ks_alpha: KS 检验显著性水平
            adwin_delta: ADWIN 置信度参数
            adwin_max_window: ADWIN 最大窗口大小
            error_threshold: 预测误差阈值，超过则认为异常
        """
        self.adwin = ADWIN(delta=adwin_delta, max_window=adwin_max_window)
        self.ks_alpha = ks_alpha
        self.error_threshold = error_threshold

        self._baseline_features: Optional[np.ndarray] = None
        self._feature_baseline_stats: Dict[str, float] = {}

        self._prediction_errors = deque(maxlen=1000)
        self._drift_events: List[Dict] = []
        self._last_drift_time: Optional[str] = None
        self._drift_count = 0

        self._lock = threading.Lock()
        self.model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        os.makedirs(self.model_dir, exist_ok=True)

    def check_prediction_error(self, prediction: float, actual: float) -> bool:
        """
        检查预测误差是否检测到漂移

        Args:
            prediction: 预测值 (0-1 概率)
            actual: 实际值 (0 或 1)
            target: 目标类别 (用于计算误差)

        Returns:
            是否检测到漂移
        """
        # 计算预测误差 (使用 Brier Score)
        if isinstance(actual, (int, float)) and isinstance(prediction, (int, float)):
            error = abs(prediction - actual)
        else:
            error = 0.5

        self._prediction_errors.append(error)
        drift_detected = self.adwin.add(error)

        if drift_detected:
            self._drift_count += 1
            self._last_drift_time = datetime.now().isoformat()
            self._drift_events.append({
                'timestamp': self._last_drift_time,
                'error_mean_before': float(np.mean(list(self.adwin.window)[:len(list(self.adwin.window))//2])) if len(self.adwin.window) > 10 else 0,
                'error_mean_after': float(np.mean(list(self.adwin.window))) if self.adwin.window else 0,
                'window_size': self.adwin.window_size,
            })
            logger.warning(
                f"[DriftMonitor] 概念漂移检测! "
                f"error={error:.4f}, n_splits={self.adwin.n_splits}"
            )

        return drift_detected

    def check_feature_drift(self, X_new: np.ndarray) -> bool:
        """
        KS 检验检测特征分布漂移

        Args:
            X_new: (n_features,) 或 (n_samples, n_features) 新特征

        Returns:
            是否检测到漂移
        """
        if self._baseline_features is None:
            self._update_baseline(X_new)
            return False

        # 标准化新特征
        if len(X_new.shape) == 1:
            X_new = X_new.reshape(1, -1)

        X_norm = (X_new - self._feature_baseline_stats['mean']) / (self._feature_baseline_stats['std'] + 1e-8)

        # KS 检验 (简化版: 用均值和方差的显著变化检测)
        drift_detected = False
        for i in range(min(X_norm.shape[1], 12)):  # 检查前 12 维
            col = X_norm[:, i]
            if np.abs(np.mean(col)) > 2.0 or np.abs(np.std(col) - 1.0) > 0.3:
                logger.warning(f"[DriftMonitor] Feature {i} drift detected")
                drift_detected = True

        if drift_detected:
            self._drift_count += 1
            self._last_drift_time = datetime.now().isoformat()
            self._drift_events.append({
                'timestamp': self._last_drift_time,
                'type': 'feature_drift',
                'feature_stats': {
                    'mean': float(np.mean(X_new)),
                    'std': float(np.std(X_new)),
                },
            })

        return drift_detected

    def _update_baseline(self, X: np.ndarray):
        """更新基线统计"""
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        self._baseline_features = X
        self._feature_baseline_stats = {
            'mean': np.mean(X, axis=0).tolist(),
            'std': np.std(X, axis=0).tolist(),
        }
        logger.info("[DriftMonitor] Baseline updated")

    def should_retrain(self) -> bool:
        """
        判断是否应该触发重新训练

        条件:
            - 漂移检测次数 > 3
            - 或距离上次漂移 > 7 天
        """
        if self._drift_count > 3:
            return True

        if self._last_drift_time:
            last_drift = datetime.fromisoformat(self._last_drift_time)
            days_since = (datetime.now() - last_drift).days
            if days_since > 7:
                return True

        return False

    def get_status(self) -> Dict:
        """获取漂移状态"""
        return {
            'drift_detected': self._drift_count > 0,
            'drift_count': self._drift_count,
            'last_drift_time': self._last_drift_time,
            'adwin_window_size': self.adwin.window_size,
            'adwin_n_splits': self.adwin.n_splits,
            'should_retrain': self.should_retrain(),
            'recent_drifts': self._drift_events[-5:],  # 最近 5 次漂移
        }

    def reset(self):
        """重置检测器"""
        self._drift_count = 0
        self._drift_events = []
        self._last_drift_time = None
        self._prediction_errors.clear()
        logger.info("[DriftMonitor] 重置完成")

    def save(self, path: Optional[str] = None):
        """保存状态"""
        if path is None:
            path = os.path.join(self.model_dir, 'drift_monitor.pkl')
        with open(path, 'wb') as f:
            pickle.dump({
                'drift_count': self._drift_count,
                'last_drift_time': self._last_drift_time,
                'drift_events': self._drift_events,
                'baseline_features': self._baseline_features,
                'feature_baseline_stats': self._feature_baseline_stats,
            }, f)
        logger.info(f"[DriftMonitor] 状态已保存: {path}")

    def load(self, path: Optional[str] = None) -> bool:
        """加载状态"""
        if path is None:
            path = os.path.join(self.model_dir, 'drift_monitor.pkl')
        if not os.path.exists(path):
            return False

        with open(path, 'rb') as f:
            state = pickle.load(f)
        self._drift_count = state.get('drift_count', 0)
        self._last_drift_time = state.get('last_drift_time')
        self._drift_events = state.get('drift_events', [])
        self._baseline_features = state.get('baseline_features')
        self._feature_baseline_stats = state.get('feature_baseline_stats', {})

        logger.info(f"[DriftMonitor] 状态已加载, drift_count={self._drift_count}")
        return True


# ── 全局单例 ────────────────────────────────────────────────

_drift_monitor_instance: Optional[DriftMonitor] = None
_drift_monitor_lock = threading.Lock()


def get_drift_monitor() -> DriftMonitor:
    """获取全局 DriftMonitor 实例 (线程安全)"""
    global _drift_monitor_instance
    if _drift_monitor_instance is None:
        with _drift_monitor_lock:
            if _drift_monitor_instance is None:
                _drift_monitor_instance = DriftMonitor()
                _drift_monitor_instance.load()  # 尝试加载状态
    return _drift_monitor_instance


# ── 兼容性别名 ──────────────────────────────────────────────

concept_drift_detector = DriftMonitor  # 兼容旧名称