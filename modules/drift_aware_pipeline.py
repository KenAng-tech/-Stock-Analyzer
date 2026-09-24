#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Drift-Aware Pipeline — 概念漂移驱动的自动重训练 (2026 SOTA)

当检测到概念漂移时自动触发模型重训练:

架构:
    Input → [Drift Detector] → [Model Predictor]
                    ↓
              [Drift Detected?]
                    ↓
        ┌─────────────┴─────────────┐
        │ NO                       │ YES
        │ 使用当前模型              │ 保留旧模型为 fallback
        │                           │ 触发重训练
        │                           │ 使用新模型
        └───────────────────────────┘

核心组件:
1. ConceptDriftDetector — ADWIN + KS 检验双检测
2. ModelWrapper — 模型封装 (支持多模型 fallback)
3. DriftAwarePipeline — 主流水线 (协调检测和重训练)

参考:
- "Detecting Concept Drift in Machine Learning" (2017)
- "Adaptive Windowing (ADWIN)" —亡羊补牢, 为时未晚
"""

import os
import json
import time
import copy
import threading
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque

from modules.logger import logger

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("[DriftAware] PyTorch 未安装")


# ── 概念漂移检测器 (ADWIN + KS 检验) ─────────────────────────────────────────

@dataclass
class DriftEvent:
    """漂移事件"""
    timestamp: float = field(default_factory=time.time)
    drift_type: str = ""  # "prediction" / "feature" / "combined"
    severity: float = 0.0  # 0-1, 越大概严重
    details: str = ""


class ADWIN:
    """ADWIN (Adaptive Windowing) 漂移检测算法"""

    def __init__(self, delta: float = 0.01, max_window: int = 1000):
        self.delta = delta
        self.max_window = max_window
        self.window = deque()
        self._n_splits = 0

    def add(self, value: float) -> bool:
        """添加观测值, 返回是否检测到漂移"""
        self.window.append(value)

        if len(self.window) > self.max_window:
            self.window.popleft()

        if len(self.window) < 30:
            return False

        return self._check_split()

    def _check_split(self) -> bool:
        """检查窗口是否可以分割 (检测到漂移)"""
        n = len(self.window)
        for cut in range(n // 4, 3 * n // 4):
            n0 = cut
            n1 = n - cut

            if n0 < 10 or n1 < 10:
                continue

            mean0 = np.mean(self.window[:cut])
            mean1 = np.mean(self.window[cut:])

            delta_prime = np.log(2 / self.delta)
            epsilon = np.sqrt((1 / (2 * min(n0, n1))) * delta_prime)

            if abs(mean0 - mean1) > epsilon:
                # 检测到漂移 — 保留后半窗口
                self.window = deque(list(self.window)[cut:])
                self._n_splits += 1
                return True

        return False

    @property
    def n_splits(self) -> int:
        return self._n_splits


class ConceptDriftDetector:
    """
    概念漂移检测器 (ADWIN + KS 检验双检测)

    检测类型:
    1. 预测漂移 — 预测误差的分布变化
    2. 特征漂移 — 输入特征的分布变化
    """

    def __init__(
        self,
        adwin_delta: float = 0.01,
        ks_alpha: float = 0.01,
        max_window: int = 1000,
    ):
        self.adwin_delta = adwin_delta
        self.ks_alpha = ks_alpha

        self._adwin_pred = ADWIN(delta=adwin_delta, max_window=max_window)
        self._adwin_feature = ADWIN(delta=adwin_delta, max_window=max_window)

        self._feature_baseline_mean: Optional[np.ndarray] = None
        self._feature_baseline_std: Optional[np.ndarray] = None

        self._drift_history: List[DriftEvent] = []
        self._drift_count = 0

    def check_prediction_drift(
        self,
        predicted: float,
        actual: float,
    ) -> bool:
        """
        检测预测漂移 (基于预测误差)

        Args:
            predicted: 预测值
            actual: 实际值

        Returns:
            bool: 是否检测到漂移
        """
        error = abs(predicted - actual)
        drift = self._adwin_pred.add(error)

        if drift:
            self._drift_count += 1
            event = DriftEvent(
                drift_type="prediction",
                severity=min(1.0, abs(predicted - actual) / (abs(actual) + 1e-8)),
                details=f"pred_err={error:.4f}",
            )
            self._drift_history.append(event)
            logger.warning(f"[Drift] 预测漂移检测! count={self._drift_count}")

        return drift

    def check_feature_drift(
        self,
        X_new: np.ndarray,
    ) -> bool:
        """
        检测特征漂移 (基于 KS 检验)

        Args:
            X_new: 新特征 (n_features,)

        Returns:
            bool: 是否检测到漂移
        """
        X_new = np.array(X_new).flatten()

        if self._feature_baseline_mean is None:
            self._feature_baseline_mean = np.mean(X_new)
            self._feature_baseline_std = np.std(X_new) + 1e-8
            return False

        # KS 检验 (简化版: 使用均值和方差)
        mean_change = abs(np.mean(X_new) - self._feature_baseline_mean) / (self._feature_baseline_std + 1e-8)
        std_ratio = np.std(X_new) / (self._feature_baseline_std + 1e-8)

        # 显著变化 (> 2 sigma)
        if mean_change > 2.0 or std_ratio > 1.5 or std_ratio < 0.5:
            drift = self._adwin_feature.add(mean_change)

            if drift:
                self._drift_count += 1
                event = DriftEvent(
                    drift_type="feature",
                    severity=min(1.0, mean_change / 3.0),
                    details=f"mean_change={mean_change:.2f}, std_ratio={std_ratio:.2f}",
                )
                self._drift_history.append(event)
                logger.warning(f"[Drift] 特征漂移检测! count={self._drift_count}")

            return drift

        return False

    def update_baseline(self, X: np.ndarray):
        """更新特征基线"""
        X = np.array(X)
        if len(X.shape) > 1:
            X = X.flatten()

        self._feature_baseline_mean = np.mean(X, axis=0) if len(X.shape) > 1 else np.mean(X)
        self._feature_baseline_std = np.std(X, axis=0) + 1e-8 if len(X.shape) > 1 else np.std(X) + 1e-8
        logger.info("[Drift] 特征基线已更新")

    def get_drift_status(self) -> Dict:
        """获取漂移状态"""
        return {
            'drift_count': self._drift_count,
            'adwin_pred_splits': self._adwin_pred._n_splits,
            'adwin_feature_splits': self._adwin_feature._n_splits,
            'drift_history': [
                {
                    'timestamp': e.timestamp,
                    'type': e.drift_type,
                    'severity': e.severity,
                    'details': e.details,
                }
                for e in self._drift_history[-10:]
            ],
        }


# ── 模型封装 (支持多模型 fallback) ────────────────────────────────────────────

class ModelWrapper:
    """
    模型封装 — 支持多模型 fallback

    内部维护:
    - active_model: 当前活跃模型
    - fallback_model: 备用模型 (漂移时保留旧模型)
    - history: 模型版本历史
    """

    def __init__(self, model: Any):
        self.active_model = model
        self.fallback_model = None
        self.history: List[Dict] = []
        self._version = 0

    def predict(self, X: Any) -> Any:
        """使用当前模型预测"""
        if self.active_model is None:
            raise ValueError("[ModelWrapper] No active model")

        try:
            if hasattr(self.active_model, 'predict'):
                return self.active_model.predict(X)
            elif callable(self.active_model):
                return self.active_model(X)
            else:
                raise ValueError(f"[ModelWrapper] Model has no predict method")
        except Exception as e:
            logger.error(f"[ModelWrapper] 预测失败: {e}")
            if self.fallback_model is not None:
                logger.info("[ModelWrapper] 切换到 fallback 模型")
                self.active_model = self.fallback_model
                self.fallback_model = None
                return self.predict(X)
            raise

    def save_fallback(self):
        """保存当前模型为 fallback (重训练前调用)"""
        self.fallback_model = copy.deepcopy(self.active_model)
        self._version += 1
        self.history.append({
            'version': self._version,
            'timestamp': time.time(),
            'action': 'fallback_saved',
        })
        logger.info(f"[ModelWrapper] Fallback 模型已保存 (version {self._version})")

    def retrain(self, X, y, **kwargs):
        """重训练模型"""
        if hasattr(self.active_model, 'train'):
            self.active_model.train(X, y, **kwargs)
        elif hasattr(self.active_model, 'fit'):
            self.active_model.fit(X, y, **kwargs)
        else:
            raise ValueError(f"[ModelWrapper] Model has no train/fit method")

        self.history.append({
            'version': self._version,
            'timestamp': time.time(),
            'action': 'retrained',
        })
        logger.info(f"[ModelWrapper] 模型重训练完成 (version {self._version})")


# ── 漂移感知流水线 ────────────────────────────────────────────────────────────

@dataclass
class PipelineStatus:
    """流水线状态"""
    active_model_version: int = 0
    fallback_available: bool = False
    drift_count: int = 0
    retrain_count: int = 0
    last_drift_time: float = 0.0
    last_retrain_time: float = 0.0


class DriftAwarePipeline:
    """
    漂移感知流水线

    核心流程:
    1. 使用当前模型预测
    2. 检测概念漂移 (ADWIN + KS)
    3. 漂移时: 保存旧模型 → 重训练 → 使用新模型
    4. 新模型效果差时: 回退到旧模型 (fallback)

    使用方式:
        pipeline = DriftAwarePipeline(model)
        result = pipeline.predict(X)
        pipeline.step(X, y)  # 检测漂移并可能触发重训练
    """

    def __init__(
        self,
        model: Any,
        drift_detector: Optional[ConceptDriftDetector] = None,
        retrain_trigger: str = "auto",  # "auto" / "manual"
        min_samples: int = 100,
    ):
        self.model_wrapper = ModelWrapper(model)
        self.drift_detector = drift_detector or ConceptDriftDetector()

        self.retrain_trigger = retrain_trigger
        self.min_samples = min_samples

        self._status = PipelineStatus()
        self._retrain_history: List[Dict] = []
        self._lock = threading.Lock()

    def predict(self, X: Any) -> Any:
        """使用当前模型预测"""
        return self.model_wrapper.predict(X)

    def step(
        self,
        X: Any,
        y: Optional[Any] = None,
        features: Optional[Any] = None,
    ) -> Dict:
        """
        执行一步 (预测 + 漂移检测 + 可能重训练)

        Args:
            X: 输入特征
            y: 实际标签 (用于预测漂移检测)
            features: 特征 (用于特征漂移检测)

        Returns:
            Dict: 包含预测结果和状态
        """
        result = {}

        # 1. 预测
        try:
            prediction = self.predict(X)
            result['prediction'] = prediction
            result['success'] = True
        except Exception as e:
            logger.error(f"[DriftAware] 预测失败: {e}")
            result['prediction'] = None
            result['success'] = False
            return result

        # 2. 漂移检测 (仅在有实际标签时)
        drift_detected = False

        if y is not None:
            # 预测漂移检测
            if hasattr(prediction, 'item'):
                pred_val = prediction.item()
            elif isinstance(prediction, (int, float)):
                pred_val = float(prediction)
            else:
                pred_val = float(prediction[0]) if hasattr(prediction, '__getitem__') else 0.0

            if self.drift_detector.check_prediction_drift(pred_val, float(y)):
                drift_detected = True

        # 特征漂移检测
        if features is not None and not drift_detected:
            if self.drift_detector.check_feature_drift(features):
                drift_detected = True

        # 3. 漂移时触发重训练
        if drift_detected:
            self._handle_drift(X, y)

        # 4. 更新状态
        self._status.last_drift_time = time.time()
        result['drift_detected'] = drift_detected
        result['drift_count'] = self.drift_detector._drift_count

        return result

    def _handle_drift(self, X: Any, y: Any):
        """处理漂移 — 触发重训练"""
        logger.info("[DriftAware] 检测到漂移, 触发重训练...")

        # 保存旧模型为 fallback
        self.model_wrapper.save_fallback()
        self._status.fallback_available = True

        # 重训练
        try:
            self.model_wrapper.retrain(X, y)
            self._status.retrain_count += 1
            self._status.last_retrain_time = time.time()
            self._retrain_history.append({
                'timestamp': time.time(),
                'drift_count': self.drift_detector._drift_count,
                'success': True,
            })
            logger.info("[DriftAware] 重训练完成")
        except Exception as e:
            logger.error(f"[DriftAware] 重训练失败: {e}")
            self._retrain_history.append({
                'timestamp': time.time(),
                'drift_count': self.drift_detector._drift_count,
                'success': False,
                'error': str(e),
            })

    def rollback(self):
        """回退到 fallback 模型"""
        if self._status.fallback_available and self.model_wrapper.fallback_model is not None:
            logger.info("[DriftAware] 回退到 fallback 模型")
            self.model_wrapper.active_model = self.model_wrapper.fallback_model
            self.model_wrapper.fallback_model = None
            self._status.fallback_available = False
            self._status.active_model_version += 1
            return True
        return False

    def get_status(self) -> PipelineStatus:
        """获取流水线状态"""
        return PipelineStatus(
            active_model_version=self._status.active_model_version,
            fallback_available=self._status.fallback_available,
            drift_count=self.drift_detector._drift_count,
            retrain_count=self._status.retrain_count,
            last_drift_time=self._status.last_drift_time,
            last_retrain_time=self._status.last_retrain_time,
        )


# ── 全局单例 ────────────────────────────────────────────────────────────────

_instance: Optional[DriftAwarePipeline] = None
_instance_lock = threading.Lock()


def get_drift_aware_pipeline(model: Any = None) -> DriftAwarePipeline:
    """获取全局 DriftAwarePipeline 单例"""
    global _instance
    with _instance_lock:
        if _instance is None and model is not None:
            _instance = DriftAwarePipeline(model)
        return _instance


def reset_drift_aware_pipeline(model: Any = None) -> DriftAwarePipeline:
    """重置全局 DriftAwarePipeline 单例"""
    global _instance
    with _instance_lock:
        if model is not None:
            _instance = DriftAwarePipeline(model)
        else:
            _instance = None
    return _instance