#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
概念漂移检测 — Concept Drift Detection

功能:
- ADWIN 算法检测预测准确率漂移
- KS 检验检测特征分布漂移
- 自适应学习率调整
- 在线因子选择
- 模型健康监控
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
from modules.logger import logger


class ADWIN:
    """
    ADWIN (Adaptive Windowing) 算法

    检测数据流中的分布变化。
    参考: Bifet, G., & Gavalda, R. "Learning from Time-Changing Data." (2007)
    """

    def __init__(self, delta: float = 0.002):
        self.delta = delta
        self.window: deque = deque()
        self._m = 0  # 窗口大小

    def add(self, value: float) -> bool:
        """添加新观测, 返回是否有变化检测"""
        self.window.append(value)
        self._m = len(self.window)

        if self._m < 10:
            return False

        return self._detect()

    def _detect(self) -> bool:
        """检测窗口内是否有变化"""
        m = len(self.window)
        if m < 10:
            return False

        found_change = False
        w0_len_max = m // 2

        for w0_len in range(1, w0_len_max + 1):
            w1_len = m - w0_len
            if w0_len < 5 or w1_len < 5:
                continue

            w0 = list(self.window)[:w0_len]
            w1 = list(self.window)[w0_len:]

            mu0 = np.mean(w0)
            mu1 = np.mean(w1)
            n0, n1 = len(w0), len(w1)

            # 置信区间
            delta_prime = self.delta / np.log(m)
            epsilon = np.sqrt((n0 + n1) / (2 * n0 * n1) * np.log(4 / delta_prime))

            if abs(mu0 - mu1) > epsilon:
                found_change = True
                # 截断窗口
                self.window = deque(w1)
                break

        return found_change

    def detect_change(self) -> bool:
        """检测是否有变化 (不添加新数据)"""
        return self._detect()

    @property
    def width(self) -> int:
        return len(self.window)

    def get_average(self) -> float:
        """获取窗口内平均值"""
        if not self.window:
            return 0.0
        return float(np.mean(self.window))


class ConceptDriftDetector:
    """概念漂移检测器"""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.adwin = ADWIN(delta=0.002)
        self._predictions: deque = deque(maxlen=window_size)
        self._actuals: deque = deque(maxlen=window_size)
        self._feature_history: deque = deque(maxlen=window_size)

    def add_observation(self, prediction: float, actual: float,
                        features: Optional[np.ndarray] = None):
        """添加预测-真实对"""
        self._predictions.append(prediction)
        self._actuals.append(actual)
        if features is not None:
            self._feature_history.append(features)

        # 计算准确率并检测漂移
        correct = 1.0 if abs(prediction - actual) < 0.02 else 0.0
        self.adwin.add(correct)

    def detect_drift(self) -> Dict:
        """
        检测概念漂移

        Returns:
            {drifted: bool, method: str, confidence: float}
        """
        # ADWIN 检测
        if self.adwin.width >= 20:
            if self.adwin.detect_change():
                return {
                    'drifted': True,
                    'method': 'adwin',
                    'confidence': 0.9,
                    'recent_accuracy': self.adwin.get_average(),
                }

        # KS 检验 (特征分布变化)
        if len(self._feature_history) >= 40:
            half = len(self._feature_history) // 2
            first_half = np.array(list(self._feature_history)[:half])
            second_half = np.array(list(self._feature_history)[half:])

            if first_half.shape[1] > 0:
                # 对每个特征做 KS 检验
                max_stat = 0
                for j in range(first_half.shape[1]):
                    stat, _ = self._ks_test(first_half[:, j], second_half[:, j])
                    max_stat = max(max_stat, stat)

                if max_stat > 0.3:
                    return {
                        'drifted': True,
                        'method': 'ks_test',
                        'confidence': float(max_stat),
                        'max_statistic': float(max_stat),
                    }

        return {
            'drifted': False,
            'method': 'none',
            'confidence': 0.0,
        }

    @staticmethod
    def _ks_test(sample1: np.ndarray, sample2: np.ndarray) -> Tuple[float, float]:
        """KS 检验 (简化实现)"""
        n1, n2 = len(sample1), len(sample2)
        if n1 == 0 or n2 == 0:
            return 0.0, 1.0

        sorted1 = np.sort(sample1)
        sorted2 = np.sort(sample2)

        # 合并排序
        all_vals = np.unique(np.concatenate([sorted1, sorted2]))

        max_diff = 0
        for v in all_vals:
            cdf1 = np.sum(sorted1 <= v) / n1
            cdf2 = np.sum(sorted2 <= v) / n2
            max_diff = max(max_diff, abs(cdf1 - cdf2))

        # Kolmogorov-Smirnov p-value 近似
        n_eff = np.sqrt(n1 * n2 / (n1 + n2))
        ks_stat = max_diff
        p_value = max(0, 1 - np.exp(-2 * n_eff**2 * ks_stat**2))

        return ks_stat, p_value

    def performance_metrics(self) -> Dict:
        """获取近期性能指标"""
        if not self._predictions or not self._actuals:
            return {}

        preds = np.array(list(self._predictions))
        actuals = np.array(list(self._actuals))

        # 方向准确率
        correct = np.sum(np.sign(preds) == np.sign(actuals))
        accuracy = float(correct / max(len(preds), 1))

        # IC
        if np.std(preds) > 0 and np.std(actuals) > 0:
            ic = float(np.corrcoef(preds, actuals)[0, 1])
        else:
            ic = 0.0

        return {
            'accuracy': round(accuracy, 4),
            'ic': round(ic, 4),
            'n_samples': len(preds),
            'recent_avg_prediction': round(float(np.mean(preds)), 4),
            'recent_avg_actual': round(float(np.mean(actuals)), 4),
        }


class AdaptiveModelWrapper:
    """自适应模型包装器"""

    def __init__(self, base_model=None, drift_detector=None):
        self.base_model = base_model
        self.drift_detector = drift_detector or ConceptDriftDetector()
        self._base_lr = 0.005
        self._current_lr = self._base_lr

    def predict(self, features: np.ndarray):
        """预测"""
        if self.base_model and hasattr(self.base_model, 'predict'):
            return self.base_model.predict(features)
        return 0.0

    def update(self, prediction: float, actual: float,
               features: Optional[np.ndarray] = None):
        """更新模型 (检测漂移并调整学习率)"""
        self.drift_detector.add_observation(prediction, actual, features)

        drift_info = self.drift_detector.detect_drift()
        if drift_info['drifted']:
            # 检测到漂移 → 降低学习率
            self._current_lr = max(self._base_lr * 0.1, self._current_lr * 0.5)
            logger.warning(
                f"[AdaptiveModel] 检测到概念漂移 ({drift_info['method']}), "
                f"学习率降至 {self._current_lr:.6f}"
            )
        else:
            # 未检测到漂移 → 逐步恢复学习率
            self._current_lr = min(self._base_lr, self._current_lr * 1.01)

    def get_learning_rate(self) -> float:
        return self._current_lr


class OnlineFeatureSelector:
    """在线因子选择器"""

    def __init__(self):
        self._factor_ic_history: Dict[str, List[float]] = {}

    def update(self, factor_name: str, ic: float):
        """更新因子 IC"""
        if factor_name not in self._factor_ic_history:
            self._factor_ic_history[factor_name] = []
        self._factor_ic_history[factor_name].append(ic)
        # 保留最近 60 条
        if len(self._factor_ic_history[factor_name]) > 60:
            self._factor_ic_history[factor_name] = self._factor_ic_history[factor_name][-60:]

    def select_active_factors(self, min_icir: float = 0.5) -> List[str]:
        """选择当前有效的因子"""
        active = []
        for name, ics in self._factor_ic_history.items():
            if len(ics) >= 10:
                ic_mean = np.mean(ics)
                ic_std = np.std(ics)
                icir = ic_mean / (ic_std + 1e-10)
                if abs(icir) >= min_icir:
                    active.append(name)
        return active

    def detect_factor_decay(self) -> List[str]:
        """检测衰减的因子 (近期 IC 低于早期)"""
        decaying = []
        for name, ics in self._factor_ic_history.items():
            if len(ics) >= 20:
                early = np.mean(ics[:len(ics)//2])
                recent = np.mean(ics[len(ics)//2:])
                if abs(recent) < abs(early) * 0.5 and abs(early) > 0.02:
                    decaying.append(name)
        return decaying


class ModelHealthMonitor:
    """模型健康监控"""

    def __init__(self):
        self._metrics: Dict[str, deque] = {}
        self.alert_thresholds = {
            'accuracy': {'min': 0.55, 'max': 0.99},
            'ic': {'min': 0.02, 'max': 0.3},
            'sharpe': {'min': 0.5, 'max': 10.0},
        }

    def track_metric(self, name: str, value: float):
        """追踪指标"""
        if name not in self._metrics:
            self._metrics[name] = deque(maxlen=100)
        self._metrics[name].append(value)

    def get_trend(self, name: str) -> str:
        """获取指标趋势"""
        if name not in self._metrics or len(self._metrics[name]) < 10:
            return 'unknown'

        recent = list(self._metrics[name])[-10:]
        early = list(self._metrics[name])[:10]

        recent_mean = np.mean(recent)
        early_mean = np.mean(early)

        change = (recent_mean - early_mean) / (abs(early_mean) + 1e-10)

        if change > 0.05:
            return 'improving'
        elif change < -0.05:
            return 'degrading'
        return 'stable'

    def health_report(self) -> Dict:
        """生成整体健康报告"""
        status = 'healthy'
        alerts = []

        for name, threshold in self.alert_thresholds.items():
            if name in self._metrics and self._metrics[name]:
                current = self._metrics[name][-1]
                trend = self.get_trend(name)

                if current < threshold['min'] or current > threshold['max']:
                    status = 'warning'
                    alerts.append(f'{name}={current:.4f} 超出阈值 [{threshold["min"]}, {threshold["max"]}]')

                if trend == 'degrading':
                    if status == 'healthy':
                        status = 'warning'
                    alerts.append(f'{name} 趋势: 下降')

        return {
            'status': status,
            'metrics': {
                name: {
                    'current': float(list(vals)[-1]) if vals else 0,
                    'trend': self.get_trend(name),
                    'history_length': len(list(vals)),
                }
                for name, vals in self._metrics.items()
            },
            'alerts': alerts,
        }


# 全局实例
drift_detector = ConceptDriftDetector()
health_monitor = ModelHealthMonitor()
