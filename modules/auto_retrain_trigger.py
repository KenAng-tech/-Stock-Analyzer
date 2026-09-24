#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
自动重训练触发器 (Auto Retraining Trigger)

基于漂移检测自动触发模型重训练:
- 监控预测分布与真实分布的偏差 (KS 检验)
- 当漂移超过阈值时，自动触发重新训练
- 支持手动触发 + 定时触发
- 训练完成后自动更新模型引用

用法:
    trigger = AutoRetrainTrigger()
    trigger.check_and_retrain('patchtst', predictions, actuals)
"""

import os
import json
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict

import numpy as np

from modules.logger import logger


@dataclass
class RetrainingJob:
    """重训练任务"""
    model_name: str
    status: str  # pending/running/success/failed
    reason: str  # drift/manual/scheduled
    triggered_at: str = ''
    completed_at: str = ''
    duration_seconds: float = 0.0
    old_metrics: Dict[str, float] = field(default_factory=dict)
    new_metrics: Dict[str, float] = field(default_factory=dict)
    error: str = ''

    def to_dict(self) -> Dict:
        return asdict(self)


class AutoRetrainTrigger:
    """自动重训练触发器"""

    def __init__(
        self,
        drift_threshold: float = 0.05,
        ks_p_value: float = 0.05,
        min_data_points: int = 100,
        cooldown_seconds: int = 3600,  # 1 小时冷却
    ):
        self.drift_threshold = drift_threshold
        self.ks_p_value = ks_p_value
        self.min_data_points = min_data_points
        self.cooldown_seconds = cooldown_seconds

        self._jobs: Dict[str, RetrainingJob] = {}
        self._last_retrain_time: Dict[str, float] = {}  # model_name -> timestamp
        self._prediction_buffer: Dict[str, List[Dict]] = {}  # model_name -> [predictions]
        self._retrain_callback = None  # 回调函数: (model_name, job) -> None

    def set_retrain_callback(self, callback):
        """设置重训练回调 — 在 app.py 中注册实际的训练函数"""
        self._retrain_callback = callback

    def record_prediction(self, model_name: str, prediction: Dict, actual: Optional[float] = None):
        """
        记录预测结果，用于漂移检测

        Args:
            model_name: 模型名称
            prediction: 预测结果 {value, confidence, ...}
            actual: 实际值 (可选)
        """
        if model_name not in self._prediction_buffer:
            self._prediction_buffer[model_name] = []

        entry = {
            'prediction': prediction,
            'actual': actual,
            'timestamp': time.time(),
        }
        self._prediction_buffer[model_name].append(entry)

        # 保持缓冲区大小
        if len(self._prediction_buffer[model_name]) > 1000:
            self._prediction_buffer[model_name] = self._prediction_buffer[model_name][-500:]

    def check_and_retrain(
        self,
        model_name: str,
        predictions: np.ndarray,
        actuals: np.ndarray,
        force: bool = False,
    ) -> Optional[RetrainingJob]:
        """
        检查是否需要重训练

        Args:
            model_name: 模型名称
            predictions: 预测值数组
            actuals: 实际值数组
            force: 强制重训练 (忽略漂移检测)

        Returns:
            RetrainingJob 或 None (不需要重训练)
        """
        # 检查冷却时间
        last_time = self._last_retrain_time.get(model_name, 0)
        if not force and (time.time() - last_time) < self.cooldown_seconds:
            return None

        # 检查数据量
        if len(predictions) < self.min_data_points:
            return None

        # 漂移检测
        drifted = self._detect_drift(predictions, actuals)

        if not drifted and not force:
            return None

        # 触发重训练
        job = self._trigger_retrain(model_name, reason='drift' if drifted else 'manual')
        return job

    def manual_trigger(self, model_name: str) -> RetrainingJob:
        """手动触发重训练"""
        return self._trigger_retrain(model_name, reason='manual')

    def _detect_drift(self, predictions: np.ndarray, actuals: np.ndarray) -> bool:
        """
        使用 KS 检验检测预测分布漂移

        Returns:
            True if drift detected
        """
        if len(predictions) < 50 or len(actuals) < 50:
            return False

        # KS 检验: 预测分布 vs 实际分布
        try:
            from scipy import stats
            ks_stat, p_value = stats.ks_test(predictions, actuals)
            return p_value < self.ks_p_value
        except ImportError:
            # 无 scipy 时使用简单的统计检验
            pred_mean = np.mean(predictions)
            pred_std = np.std(predictions)
            actual_mean = np.mean(actuals)
            actual_std = np.std(actuals)

            if pred_std < 1e-8 or actual_std < 1e-8:
                return False

            # Z 检验
            z = abs(pred_mean - actual_mean) / np.sqrt(
                pred_std**2 / len(predictions) + actual_std**2 / len(actuals)
            )
            return z > 1.96  # p < 0.05

    def _trigger_retrain(self, model_name: str, reason: str = 'drift') -> RetrainingJob:
        """触发重训练"""
        logger.info(f"[AutoRetrain] 触发重训练: {model_name} (原因: {reason})")

        job = RetrainingJob(
            model_name=model_name,
            status='pending',
            reason=reason,
            triggered_at=datetime.now().isoformat(),
        )
        self._jobs[model_name] = job

        # 异步执行重训练
        def _run_retrain():
            job.status = 'running'
            start_time = time.time()
            try:
                if self._retrain_callback:
                    self._retrain_callback(model_name, job)
                job.status = 'success'
                job.completed_at = datetime.now().isoformat()
                job.duration_seconds = time.time() - start_time
                self._last_retrain_time[model_name] = time.time()
                logger.info(f"[AutoRetrain] 重训练成功: {model_name} ({job.duration_seconds:.1f}s)")
            except Exception as e:
                job.status = 'failed'
                job.error = str(e)
                job.completed_at = datetime.now().isoformat()
                job.duration_seconds = time.time() - start_time
                logger.error(f"[AutoRetrain] 重训练失败: {model_name} — {e}")

        threading.Thread(target=_run_retrain, daemon=True, name=f"Retrain-{model_name}").start()
        return job

    def get_status(self, model_name: Optional[str] = None) -> Dict:
        """获取重训练状态"""
        if model_name:
            job = self._jobs.get(model_name)
            return job.to_dict() if job else {'status': 'no_jobs'}

        return {
            'total_jobs': len(self._jobs),
            'jobs': {k: v.to_dict() for k, v in self._jobs.items()},
            'cooldown_seconds': self.cooldown_seconds,
        }

    def add_retrain_callback(self, model_name: str, callback):
        """为特定模型添加重训练回调"""
        # 包装回调，统一接口
        original = self._retrain_callback

        def wrapped(model_name_inner, job):
            if model_name_inner == model_name:
                callback(job)
            elif original:
                original(model_name_inner, job)

        self._retrain_callback = wrapped


# 全局单例
_auto_retrain: Optional[AutoRetrainTrigger] = None


def get_auto_retrain_trigger() -> AutoRetrainTrigger:
    """获取自动重训练触发器全局实例"""
    global _auto_retrain
    if _auto_retrain is None:
        _auto_retrain = AutoRetrainTrigger()
    return _auto_retrain
