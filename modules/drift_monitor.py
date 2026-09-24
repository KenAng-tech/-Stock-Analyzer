#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
概念漂移检测集成模块 — DEPRECATED

⚠️ 已弃用: 请使用 concept_drift_detector.py 中的 DriftConsensusDetector
此文件保留仅向后兼容，将在 2026-10-01 后删除。

新的代码应使用:
    from modules.concept_drift_detector import DriftConsensusDetector
    detector = DriftConsensusDetector()
    result = detector.check_error(0.05)
"""

import warnings
warnings.warn(
    "[DEPRECATED] modules.drift_monitor 已弃用，请使用 modules.concept_drift_detector",
    DeprecationWarning,
    stacklevel=2,
)

import os
import pickle
import numpy as np
from typing import Dict, Optional, List, Tuple
from datetime import datetime
from collections import deque
import threading
from scipy import stats as scipy_stats

from modules.logger import logger
from modules.adwin import ADWIN


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
        self._prediction_drift_count = 0
        self._feature_drift_count = 0

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
            self._prediction_drift_count += 1
            self._last_drift_time = datetime.now().isoformat()
            self._drift_events.append({
                'timestamp': self._last_drift_time,
                'type': 'prediction_drift',
                'error_mean_before': float(np.mean(list(self.adwin.window)[:len(list(self.adwin.window))//2])) if len(self.adwin.window) > 10 else 0,
                'error_mean_after': float(np.mean(list(self.adwin.window))) if self.adwin.window else 0,
                'window_size': self.adwin.window_size,
            })
            logger.warning(
                f"[DriftMonitor] 预测误差漂移检测! "
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

        drift_detected = False
        drifted_features = []

        for i in range(min(X_new.shape[1], 20)):  # 检查前 20 维
            col_old = self._baseline_features[:, i]
            col_new = X_new[:, i]

            # 标准化
            mean_old = np.mean(col_old)
            std_old = np.std(col_old) + 1e-8
            col_old_norm = (col_old - mean_old) / std_old
            col_new_norm = (col_new - mean_old) / std_old

            # KS 检验
            ks_stat, ks_pvalue = scipy_stats.ks_2samp(col_old_norm, col_new_norm)

            if ks_pvalue < self.ks_alpha:
                drift_detected = True
                drifted_features.append({'feature': i, 'ks_stat': float(ks_stat), 'p_value': float(ks_pvalue)})
                logger.warning(
                    f"[DriftMonitor] 特征 {i} 分布漂移 (KS stat={ks_stat:.4f}, p={ks_pvalue:.4f})"
                )

        if drift_detected:
            self._drift_count += 1
            self._feature_drift_count += 1
            self._last_drift_time = datetime.now().isoformat()
            self._drift_events.append({
                'timestamp': self._last_drift_time,
                'type': 'feature_drift',
                'drifted_features': drifted_features,
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
            - 或预测漂移 + 特征漂移 > 5
        """
        if self._drift_count > 3:
            return True
        if self._prediction_drift_count + self._feature_drift_count > 5:
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
            'prediction_drift_count': self._prediction_drift_count,
            'feature_drift_count': self._feature_drift_count,
            'last_drift_time': self._last_drift_time,
            'adwin_window_size': self.adwin.window_size,
            'adwin_n_splits': self.adwin.n_splits,
            'should_retrain': self.should_retrain(),
            'recent_drifts': self._drift_events[-5:],  # 最近 5 次漂移
        }

    def reset(self):
        """重置检测器"""
        self._drift_count = 0
        self._prediction_drift_count = 0
        self._feature_drift_count = 0
        self._drift_events = []
        self._last_drift_time = None
        self._prediction_errors.clear()
        self.adwin = ADWIN(delta=self.adwin.delta, max_window=self.adwin.max_window)
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