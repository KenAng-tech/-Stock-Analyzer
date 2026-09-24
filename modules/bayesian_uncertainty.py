#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
贝叶斯不确定性量化 (Bayesian Uncertainty Quantification)

使用 Deep Ensemble + Monte Carlo Dropout 估计模型预测不确定性:
- Deep Ensemble: 多个模型预测的方差
- MC Dropout: 多次前向传播的方差
- 不确定性校准: Isotonic Regression
- 预警: 高不确定性时触发重新训练

用法:
    bayesian = BayesianUncertainty()
    uncertainty = bayesian.estimate(model, features)
"""

import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict

from modules.logger import logger


@dataclass
class UncertaintyEstimate:
    """不确定性估计"""
    mean: float  # 预测均值
    std: float  # 预测标准差
    ci_lower: float  # 95% 置信区间下界
    ci_upper: float  # 95% 置信区间上界
    epistemic: float  # 认知不确定性 (模型间差异)
    aleatoric: float  # 偶然不确定性 (数据噪声)
    confidence: float  # 综合置信度 0-1
    timestamp: str = ''

    def to_dict(self) -> Dict:
        return asdict(self)


class BayesianUncertainty:
    """贝叶斯不确定性量化"""

    def __init__(
        self,
        uncertainty_threshold: float = 0.15,
        calibration_window: int = 500,
    ):
        self.uncertainty_threshold = uncertainty_threshold
        self.calibration_window = calibration_window

        # 模型预测缓冲区 (用于 ensemble)
        self._model_predictions: Dict[str, List[np.ndarray]] = {}  # model -> [preds]
        # 校准数据
        self._calibration_data: List[Dict] = []
        # 不确定性历史
        self._uncertainty_history: List[UncertaintyEstimate] = []

    def estimate_ensemble(
        self,
        model_predictions: Dict[str, np.ndarray],
        features: Optional[np.ndarray] = None,
    ) -> UncertaintyEstimate:
        """
        Deep Ensemble 不确定性估计

        Args:
            model_predictions: {model_name: prediction_array}
            features: 输入特征 (可选)

        Returns:
            UncertaintyEstimate
        """
        if not model_predictions:
            return UncertaintyEstimate(
                mean=0, std=0, ci_lower=0, ci_upper=0,
                epistemic=0, aleatoric=0, confidence=0,
                timestamp=datetime.now().isoformat(),
            )

        # 收集所有预测
        all_preds = list(model_predictions.values())
        preds_array = np.array(all_preds)

        # 均值预测
        mean = float(np.mean(preds_array, axis=0))

        # 总方差 = Epistemic + Aleatoric
        # Epistemic: 模型间差异 (方差)
        epistemic = float(np.var(preds_array, axis=0))

        # Aleatoric: 模型预测的均值方差 (简化估计)
        aleatoric = float(np.mean(np.std(preds_array, axis=1)**2)) if len(preds_array) > 1 else 0

        # 总不确定性
        total_std = np.sqrt(epistemic + aleatoric)

        # 置信区间 (假设正态分布)
        ci_lower = mean - 1.96 * total_std
        ci_upper = mean + 1.96 * total_std

        # 置信度: 不确定性越低越自信
        confidence = max(0.0, min(1.0, 1.0 - total_std / max(abs(mean), 1e-8)))

        estimate = UncertaintyEstimate(
            mean=mean,
            std=float(total_std),
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
            epistemic=epistemic,
            aleatoric=aleatoric,
            confidence=confidence,
            timestamp=datetime.now().isoformat(),
        )

        self._uncertainty_history.append(estimate)
        if len(self._uncertainty_history) > self.calibration_window:
            self._uncertainty_history = self._uncertainty_history[-self.calibration_window:]

        # 记录模型预测
        for name, pred in model_predictions.items():
            if name not in self._model_predictions:
                self._model_predictions[name] = []
            self._model_predictions[name].append(pred)
            if len(self._model_predictions[name]) > 100:
                self._model_predictions[name] = self._model_predictions[name][-50:]

        if total_std > self.uncertainty_threshold:
            logger.warning(
                f"[Bayesian] 高不确定性: std={total_std:.4f}, "
                f"epistemic={epistemic:.4f}, aleatoric={aleatoric:.4f}"
            )

        return estimate

    def estimate_mc_dropout(
        self,
        predict_fn,
        features: np.ndarray,
        n_samples: int = 20,
    ) -> UncertaintyEstimate:
        """
        MC Dropout 不确定性估计

        Args:
            predict_fn: 带有 dropout 的预测函数
            features: 输入特征
            n_samples: 采样次数

        Returns:
            UncertaintyEstimate
        """
        predictions = []
        for _ in range(n_samples):
            pred = predict_fn(features, training=True)  # training=True 启用 dropout
            predictions.append(pred)

        predictions = np.array(predictions)
        mean = float(np.mean(predictions, axis=0))
        total_std = float(np.std(predictions, axis=0))

        # MC Dropout 中，方差主要来自模型不确定性
        epistemic = float(np.var(predictions, axis=0))
        aleatoric = 0.0  # MC Dropout 不直接估计 aleatoric

        ci_lower = mean - 1.96 * total_std
        ci_upper = mean + 1.96 * total_std
        confidence = max(0.0, min(1.0, 1.0 - total_std / max(abs(mean), 1e-8)))

        return UncertaintyEstimate(
            mean=mean,
            std=total_std,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            epistemic=epistemic,
            aleatoric=aleatoric,
            confidence=confidence,
            timestamp=datetime.now().isoformat(),
        )

    def calibrate(self, predictions: np.ndarray, actuals: np.ndarray) -> Dict:
        """
        不确定性校准 (Isotonic Regression 简化版)

        Args:
            predictions: 预测值
            actuals: 实际值

        Returns:
            校准结果
        """
        if len(predictions) < 10:
            return {'calibrated': False, 'reason': '数据不足'}

        # 计算预测误差
        errors = np.abs(predictions - actuals)

        # 分组: 按预测不确定性分箱
        n_bins = min(10, len(predictions) // 5)
        bin_indices = np.digitize(predictions, np.linspace(predictions.min(), predictions.max(), n_bins))

        calibration = {}
        for i in range(1, n_bins + 1):
            mask = bin_indices == i
            if mask.sum() > 0:
                bin_errors = errors[mask]
                calibration[f'bin_{i}'] = {
                    'count': int(mask.sum()),
                    'mean_error': float(np.mean(bin_errors)),
                    'max_error': float(np.max(bin_errors)),
                }

        overall_mae = float(np.mean(errors))
        overall_rmse = float(np.sqrt(np.mean(errors**2)))

        return {
            'calibrated': True,
            'mae': overall_mae,
            'rmse': overall_rmse,
            'bins': calibration,
        }

    def get_alert(self) -> Optional[Dict]:
        """获取不确定性预警"""
        if not self._uncertainty_history:
            return None

        recent = self._uncertainty_history[-10:]
        avg_std = np.mean([e.std for e in recent])

        if avg_std > self.uncertainty_threshold:
            return {
                'type': 'high_uncertainty',
                'avg_std': float(avg_std),
                'threshold': self.uncertainty_threshold,
                'timestamp': datetime.now().isoformat(),
                'recommendation': '建议重新训练模型或减少交易',
            }
        return None

    def get_summary(self) -> Dict:
        """获取摘要"""
        if not self._uncertainty_history:
            return {
                'total_estimates': 0,
                'recent_avg_std': 0.0,
                'recent_avg_confidence': 0.0,
                'high_uncertainty_count': 0,
                'alerts': [],
            }

        recent = self._uncertainty_history[-100:]
        return {
            'total_estimates': len(self._uncertainty_history),
            'recent_avg_std': float(np.mean([e.std for e in recent])),
            'recent_avg_confidence': float(np.mean([e.confidence for e in recent])),
            'high_uncertainty_count': sum(1 for e in recent if e.std > self.uncertainty_threshold),
            'alerts': [a for a in [self.get_alert()] if a],
        }


# 全局单例
_bayesian: Optional[BayesianUncertainty] = None


def get_bayesian_uncertainty() -> BayesianUncertainty:
    """获取贝叶斯不确定性量化器"""
    global _bayesian
    if _bayesian is None:
        _bayesian = BayesianUncertainty()
    return _bayesian
