#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
在线学习模块 — 概念漂移触发自动模型更新

工作流程:
1. 持续监控预测准确率
2. 当准确率下降到阈值以下时，触发概念漂移检测
3. 使用最新数据增量训练模型
4. 验证新模型性能
5. 替换旧模型 (如果新模型更好)

与现有模块集成:
- concept_drift_detector.py (ADWIN 漂移检测)
- async_training_pipeline.py (异步训练)
- model_registry.py (模型版本管理)

用法:
    online_learner = OnlineLearningManager()
    online_learner.record_prediction(stock_code, prediction, actual)
    # 后台自动检测并触发更新
"""

import os
import threading
import time
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from modules.logger import logger


@dataclass
class PredictionRecord:
    """预测记录"""
    timestamp: str
    stock_code: str
    model_name: str
    prediction: str      # buy/sell/neutral
    confidence: float
    actual: str          # up/down/flat
    kline_count: int = 0


@dataclass
class DriftAlert:
    """漂移告警"""
    stock_code: str
    model_name: str
    detected_at: str
    severity: str        # low/medium/high
    accuracy_before: float
    accuracy_after: float
    window_size: int
    recommended_action: str  # retrain/monitor/ignore


class OnlineLearningManager:
    """在线学习管理器"""

    def __init__(
        self,
        accuracy_threshold: float = 0.52,
        min_samples: int = 30,
        check_interval: int = 300,  # 5 分钟检查一次
        retrain_on_drift: bool = True,
    ):
        """
        Args:
            accuracy_threshold: 准确率阈值，低于此值触发重训练
            min_samples: 最小样本数，少于此数不检测漂移
            check_interval: 检查间隔 (秒)
            retrain_on_drift: 检测到漂移时是否自动重训练
        """
        self.accuracy_threshold = accuracy_threshold
        self.min_samples = min_samples
        self.check_interval = check_interval
        self.retrain_on_drift = retrain_on_drift

        # 预测记录 (stock_code → [PredictionRecord])
        self._predictions: Dict[str, deque] = {}
        self._lock = threading.Lock()

        # 漂移告警历史
        self._alerts: List[DriftAlert] = []

        # 监控线程
        self._monitor_thread: Optional[threading.Thread] = None
        self._running = False

        logger.info(
            f"[OnlineLearning] 管理器初始化: "
            f"threshold={accuracy_threshold}, min_samples={min_samples}"
        )

    def start_monitoring(self):
        """启动后台监控"""
        if self._running:
            return
        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name='OnlineLearningMonitor',
            daemon=True,
        )
        self._monitor_thread.start()
        logger.info("[OnlineLearning] 监控已启动")

    def stop_monitoring(self):
        """停止后台监控"""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("[OnlineLearning] 监控已停止")

    def _monitor_loop(self):
        """监控主循环"""
        while self._running:
            try:
                self._check_all_stocks()
            except Exception as e:
                logger.error(f"[OnlineLearning] 监控循环错误: {e}")
            time.sleep(self.check_interval)

    def record_prediction(
        self,
        stock_code: str,
        model_name: str,
        prediction: str,
        confidence: float,
        actual: str,
        kline_count: int = 0,
    ):
        """
        记录一次预测结果

        Args:
            stock_code: 股票代码
            model_name: 模型名称
            prediction: 预测方向 (buy/sell/neutral)
            confidence: 置信度
            actual: 实际方向 (up/down/flat)
            kline_count: K 线数据量
        """
        record = PredictionRecord(
            timestamp=datetime.now().isoformat(),
            stock_code=stock_code,
            model_name=model_name,
            prediction=prediction,
            confidence=confidence,
            actual=actual,
            kline_count=kline_count,
        )

        with self._lock:
            if stock_code not in self._predictions:
                self._predictions[stock_code] = deque(maxlen=500)
            self._predictions[stock_code].append(record)

    def _check_all_stocks(self):
        """检查所有股票的预测准确率"""
        with self._lock:
            codes = list(self._predictions.keys())

        for code in codes:
            records = self._get_predictions(code)
            if len(records) < self.min_samples:
                continue

            # 计算滑动窗口准确率
            accuracy = self._calculate_accuracy(records)
            if accuracy < self.accuracy_threshold:
                # 触发漂移检测
                alert = self._detect_drift(records)
                if alert:
                    self._alerts.append(alert)
                    logger.warning(
                        f"[OnlineLearning] 检测到概念漂移: {code} "
                        f"(accuracy={accuracy:.3f}, threshold={self.accuracy_threshold})"
                    )
                    # 触发重训练
                    if self.retrain_on_drift:
                        self._trigger_retrain(code, alert)

    def _get_predictions(self, stock_code: str) -> List[PredictionRecord]:
        """获取预测记录"""
        records = self._predictions.get(stock_code)
        if records is None:
            return []
        return list(records)

    def _calculate_accuracy(self, records: List[PredictionRecord]) -> float:
        """计算预测准确率"""
        if not records:
            return 0.0

        # 方向匹配: buy→up, sell→down, neutral→flat
        correct = 0
        for r in records:
            if (r.prediction == 'buy' and r.actual == 'up') or \
               (r.prediction == 'sell' and r.actual == 'down') or \
               (r.prediction == 'neutral' and r.actual == 'flat'):
                correct += 1

        return correct / len(records)

    def _detect_drift(self, records: List[PredictionRecord]) -> Optional[DriftAlert]:
        """
        检测概念漂移 (使用滑动窗口对比)

        将记录分为前后两半，比较准确率是否有显著差异
        """
        n = len(records)
        if n < self.min_samples * 2:
            return None

        mid = n // 2
        first_half = records[:mid]
        second_half = records[mid:]

        acc_before = self._calculate_accuracy(first_half)
        acc_after = self._calculate_accuracy(second_half)

        # 准确率下降超过 10% 视为漂移
        accuracy_drop = acc_before - acc_after
        if accuracy_drop < 0.10:
            return None

        # 确定严重度
        if accuracy_drop > 0.20:
            severity = 'high'
            action = 'retrain'
        elif accuracy_drop > 0.15:
            severity = 'medium'
            action = 'retrain'
        else:
            severity = 'low'
            action = 'monitor'

        return DriftAlert(
            stock_code=records[-1].stock_code,
            model_name=records[-1].model_name,
            detected_at=datetime.now().isoformat(),
            severity=severity,
            accuracy_before=round(acc_before, 3),
            accuracy_after=round(acc_after, 3),
            window_size=n,
            recommended_action=action,
        )

    def _trigger_retrain(self, stock_code: str, alert: DriftAlert):
        """触发模型重训练"""
        logger.info(
            f"[OnlineLearning] 触发重训练: {stock_code} "
            f"(severity={alert.severity}, action={alert.recommended_action})"
        )

        try:
            from modules.async_training_pipeline import get_training_pipeline
            pipeline = get_training_pipeline()

            # 根据模型类型选择训练类型
            model_type = 'drl'  # 默认 DRL
            try:
                from modules.model_registry import get_model_registry
                registry = get_model_registry()
                active_models = registry.list_models(status=None, active_only=False)
                for m in active_models:
                    if 'drl' in m.name.lower():
                        model_type = 'drl'
                        break
            except Exception:
                pass

            task_id = pipeline.submit(
                model_type=model_type,
                stock_code=stock_code,
                params={
                    'epochs': 50,  # 增量训练，较少轮数
                    'days': 180,   # 使用最近 6 个月数据
                    'online_learning': True,
                },
            )
            logger.info(f"[OnlineLearning] 重训练任务已提交: {task_id}")

        except Exception as e:
            logger.error(f"[OnlineLearning] 重训练触发失败: {e}")

    def get_accuracy(self, stock_code: str, window: int = 100) -> Dict:
        """获取指定股票的准确率"""
        records = self._get_predictions(stock_code)
        if not records:
            return {'stock_code': stock_code, 'n_records': 0}

        recent = list(records)[-window:]
        accuracy = self._calculate_accuracy(recent)

        # 按模型分组
        by_model: Dict[str, List[PredictionRecord]] = {}
        for r in recent:
            if r.model_name not in by_model:
                by_model[r.model_name] = []
            by_model[r.model_name].append(r)

        model_stats = {}
        for model, model_records in by_model.items():
            model_stats[model] = {
                'accuracy': round(self._calculate_accuracy(model_records), 3),
                'n_records': len(model_records),
            }

        return {
            'stock_code': stock_code,
            'n_records': len(recent),
            'accuracy': round(accuracy, 3),
            'threshold': self.accuracy_threshold,
            'below_threshold': accuracy < self.accuracy_threshold,
            'by_model': model_stats,
        }

    def get_alerts(self, limit: int = 20) -> List[Dict]:
        """获取漂移告警"""
        alerts = self._alerts[-limit:]
        return [
            {
                'stock_code': a.stock_code,
                'model_name': a.model_name,
                'detected_at': a.detected_at,
                'severity': a.severity,
                'accuracy_before': a.accuracy_before,
                'accuracy_after': a.accuracy_after,
                'recommended_action': a.recommended_action,
            }
            for a in alerts
        ]

    def get_status(self) -> Dict:
        """获取管理器状态"""
        return {
            'status': 'running' if self._running else 'stopped',
            'n_stocks_monitored': len(self._predictions),
            'n_alerts': len(self._alerts),
            'accuracy_threshold': self.accuracy_threshold,
            'min_samples': self.min_samples,
            'recent_alerts': self.get_alerts(limit=5),
        }


# 全局单例
_manager: Optional[OnlineLearningManager] = None


def get_online_learner() -> OnlineLearningManager:
    """获取在线学习管理器全局实例"""
    global _manager
    if _manager is None:
        _manager = OnlineLearningManager()
    return _manager


def reset_online_learner():
    """重置全局实例"""
    global _manager
    _manager = None
