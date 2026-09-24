#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
概念漂移感知集成 (Drift-Aware Ensemble)

P0 模块: 每个子模型独立监控概念漂移，检测到漂移后自动降权并回退。

核心功能:
    1. 每个子模型使用 KS 检验 + ADWIN 独立监控预测分布漂移
    2. 漂移检测 → 该模型权重降为 0
    3. 自动回退: 用最近一次非漂移模型的预测
    4. 与 factor_weight_scheduler 联动: 因子权重随 regime 动态调整

参考:
    - Bifet & Gavalda, "Adaptive Windowing for Data Stream Classification" (2007)
    - D'Adamo et al., "Combining KS-test and ADWIN for concept drift detection" (2019)
"""

from typing import Dict, List, Optional, Any, Union
from collections import deque
from datetime import datetime

import numpy as np
from scipy import stats

from modules.logger import logger
from modules.adwin import ADWIN


# ---------------------------------------------------------------------------
# 模型漂移监控器
# ---------------------------------------------------------------------------

class ModelDriftMonitor:
    """
    单个模型的漂移监控器

    监控模型预测分布的漂移:
    - 使用 KS 检验比较预测分布 vs 校准分布
    - 使用 ADWIN 监控预测准确率的时间序列

    使用示例:
        >>> monitor = ModelDriftMonitor(ks_alpha=0.05, adwin_window=100)
        >>> for pred, actual in zip(predictions, actuals):
        ...     monitor.update(pred, actual)
        >>> if monitor.is_drifted():
        ...     logger.warning("模型预测出现概念漂移")
    """

    def __init__(self, name: str = "default", ks_alpha: float = 0.05,
                 adwin_window: int = 100, adwin_delta: float = 0.01,
                 calibration_size: int = 200):
        """
        Args:
            name: 监控器名称（通常对应模型名）
            ks_alpha: KS 检验显著性水平，越小越严格
            adwin_window: ADWIN 窗口大小
            adwin_delta: ADWIN 置信度参数
            calibration_size: 校准分布的参考样本量
        """
        self.name = name
        self.ks_alpha = ks_alpha
        self._drifted = False
        self._last_ks_pvalue: float = 1.0

        # 校准分布（历史预测值），用于 KS 检验
        self._calibration: deque = deque(maxlen=calibration_size)

        # ADWIN 监控预测误差的时间序列
        self._adwin = ADWIN(delta=adwin_delta, max_window=adwin_window)

        # 历史数据
        self._predictions: deque = deque(maxlen=adwin_window)
        self._actuals: deque = deque(maxlen=adwin_window)

        # 统计信息
        self._total_updates = 0
        self._drift_count = 0

    def update(self, prediction: float, actual: float) -> bool:
        """
        更新监控器

        流程:
            1. 将预测值加入校准分布（累积阶段）
            2. 当校准样本足够后，启动 KS 检验
            3. 同时用 ADWIN 监控预测误差

        Args:
            prediction: 模型预测值
            actual: 实际值

        Returns:
            是否检测到漂移
        """
        self._total_updates += 1

        # 记录历史和误差
        self._predictions.append(prediction)
        self._actuals.append(actual)
        error = abs(prediction - actual)
        self._predictions[-1] = prediction  # 更新以便后续使用

        # 校准阶段: 先累积校准分布
        if len(self._calibration) < 50:
            self._calibration.append(prediction)
            self._adwin.add(error)
            return False

        # ADWIN 监控误差流
        adwin_drift = self._adwin.add(error)

        # KS 检验: 比较最新一批预测 vs 校准分布
        ks_drift = False
        if len(self._calibration) >= 50 and len(self._predictions) >= 30:
            recent = list(self._predictions)[-30:]
            calib = list(self._calibration)
            try:
                ks_stat, ks_pvalue = stats.ks_2samp(calib, recent)
                self._last_ks_pvalue = ks_pvalue
                if ks_pvalue < self.ks_alpha:
                    ks_drift = True
            except Exception:
                # KS 检验失败时不报错，仅记录
                self._last_ks_pvalue = 1.0

        # 漂移判定: KS 或 ADWIN 任一触发
        drifted = ks_drift or adwin_drift

        if drifted and not self._drifted:
            self._drift_count += 1
            logger.warning(
                f"[漂移检测] 模型 {self.name}: "
                f"KS p={self._last_ks_pvalue:.4f}, "
                f"校准样本={len(self._calibration)}, "
                f"ADWIN splits={self._adwin.n_splits}"
            )

        self._drifted = drifted

        # 漂移期间不更新校准分布，保持参考基准
        if not drifted:
            self._calibration.append(prediction)

        return drifted

    def is_drifted(self) -> bool:
        """
        是否检测到漂移

        Returns:
            True 表示当前模型预测出现概念漂移
        """
        return self._drifted

    def get_confidence(self) -> float:
        """
        当前置信度 (1.0 = 无漂移, 0.0 = 严重漂移)

        基于 KS p-value 和 ADWIN 状态综合计算。

        Returns:
            置信度 [0.0, 1.0]
        """
        # 基于 KS p-value 的置信度
        ks_confidence = min(self._last_ks_pvalue / self.ks_alpha, 1.0)

        # 基于 ADWIN 的置信度
        adwin_confidence = 1.0 - min(self._adwin.n_splits / 10.0, 1.0)

        # 综合置信度
        confidence = 0.6 * ks_confidence + 0.4 * adwin_confidence
        return max(0.0, min(1.0, confidence))

    def reset_calibration(self):
        """
        重置校准分布（漂移后重新校准用）
        """
        self._calibration.clear()
        self._drifted = False
        self._last_ks_pvalue = 1.0
        logger.info(f"[漂移检测] 模型 {self.name}: 校准分布已重置")

    def get_status(self) -> Dict:
        """
        获取监控器状态

        Returns:
            包含漂移状态和统计信息的字典
        """
        return {
            "name": self.name,
            "drifted": self._drifted,
            "confidence": round(self.get_confidence(), 4),
            "ks_p_value": round(self._last_ks_pvalue, 6),
            "ks_alpha": self.ks_alpha,
            "adwin_splits": self._adwin.n_splits,
            "adwin_window_size": self._adwin.window_size,
            "total_updates": self._total_updates,
            "drift_count": self._drift_count,
            "calibration_size": len(self._calibration),
        }


# ---------------------------------------------------------------------------
# 概念漂移感知集成
# ---------------------------------------------------------------------------

class DriftAwareEnsemble:
    """
    概念漂移感知集成

    1. 每个子模型独立监控概念漂移
    2. 漂移检测 -> 该模型权重降为 0
    3. 自动回退: 用最近一次非漂移模型的预测
    4. 与 factor_weight_scheduler 联动

    使用示例:
        >>> ensemble = DriftAwareEnsemble(lookback=60)
        >>> ensemble.add_model("ml_predictor", ml_model)
        >>> ensemble.add_model("technical_analyzer", tech_model)
        >>> ensemble.add_model("sentiment_model", sent_model)
        >>>
        >>> decision, report = ensemble.predict("sz300620", features)
        >>> if report.get("fallback"):
        ...     logger.info("使用回退模型预测")
    """

    def __init__(self, models: Optional[Dict[str, Any]] = None,
                 lookback: int = 60, default_weights: Optional[Dict[str, float]] = None):
        """
        Args:
            models: 模型字典 {'name': model_instance}
            lookback: 回溯天数，用于因子权重联动
            default_weights: 默认权重 {'name': weight}，如果不传则等权
        """
        self.models: Dict[str, Any] = models or {}
        self.lookback = lookback
        self._monitors: Dict[str, ModelDriftMonitor] = {}
        self._weights: Dict[str, float] = {}
        self._history: Dict[str, List[Dict]] = {}  # 历史预测记录
        self._fallback_cache: Dict[str, Any] = {}  # 回退缓存 {stock_code: prediction}
        self._drift_events: List[Dict] = []  # 漂移事件日志

        # 初始化模型和监控器
        for name, model in self.models.items():
            self._monitors[name] = ModelDriftMonitor(name=name)

        # 设置权重
        if default_weights:
            total = sum(default_weights.values())
            self._weights = {k: v / total for k, v in default_weights.items()}
        else:
            # 等权
            n = len(self.models)
            self._weights = {name: 1.0 / n for name in self.models}

        logger.info(
            f"[DriftAwareEnsemble] 初始化 {len(self.models)} 个模型: "
            f"{list(self.models.keys())}"
        )

    def add_model(self, name: str, model_instance: Any):
        """
        添加模型到集成

        Args:
            name: 模型名称（唯一标识）
            model_instance: 模型实例，需有 predict(features) 方法
        """
        if name in self.models:
            logger.warning(f"[DriftAwareEnsemble] 模型 {name} 已存在，将覆盖")

        self.models[name] = model_instance
        self._monitors[name] = ModelDriftMonitor(name=name)

        # 重新计算等权
        n = len(self.models)
        for key in self._weights:
            self._weights[key] = 1.0 / n
        self._weights[name] = 1.0 / n

        logger.info(f"[DriftAwareEnsemble] 添加模型: {name} (当前共 {n} 个)")

    def predict(self, stock_code: str, features: np.ndarray) -> tuple:
        """
        集成预测（自动排除漂移模型）

        流程:
            1. 对每个非漂移模型调用 predict
            2. 按权重加权平均
            3. 如果所有模型都漂移，使用回退缓存

        Args:
            stock_code: 股票代码
            features: 特征数组

        Returns:
            (decision, drift_report):
                - decision: 集成预测结果
                - drift_report: 漂移报告
        """
        drift_report: Dict[str, Dict] = {}
        active_predictions: Dict[str, float] = {}
        active_weights: Dict[str, float] = {}

        # 收集所有非漂移模型的预测
        for name, model in self.models.items():
            monitor = self._monitors[name]

            try:
                pred = model.predict(features)
                # 如果模型返回字典，取数值字段
                if isinstance(pred, dict):
                    pred = pred.get("prediction", pred.get("score", pred.get("value", 0)))
                pred = float(pred)
            except Exception as e:
                logger.error(f"[DriftAwareEnsemble] 模型 {name} 预测失败: {e}")
                drift_report[name] = {
                    "drifted": True,
                    "confidence": 0.0,
                    "ks_p_value": 0.0,
                    "error": str(e),
                }
                continue

            if monitor.is_drifted():
                drift_report[name] = {
                    "drifted": True,
                    "confidence": round(monitor.get_confidence(), 4),
                    "ks_p_value": round(monitor._last_ks_pvalue, 6),
                }
                logger.debug(
                    f"[DriftAwareEnsemble] 模型 {name} 已漂移，排除在集成之外"
                )
            else:
                drift_report[name] = {
                    "drifted": False,
                    "confidence": round(monitor.get_confidence(), 4),
                    "ks_p_value": round(monitor._last_ks_pvalue, 6),
                }
                active_predictions[name] = pred
                active_weights[name] = self._weights.get(name, 0.0)

        # 加权集成
        if active_predictions:
            total_weight = sum(active_weights.values())
            if total_weight > 0:
                # 归一化权重
                decision = sum(
                    pred * (w / total_weight)
                    for pred, w in zip(active_predictions.values(), active_weights.values())
                )
            else:
                # 所有活跃模型权重为 0，取平均
                decision = np.mean(list(active_predictions.values()))
        else:
            # 所有模型都漂移或不可用，使用回退
            fallback = self._fallback_cache.get(stock_code)
            if fallback is not None:
                decision = fallback
                logger.warning(
                    f"[DriftAwareEnsemble] 所有模型漂移，使用回退预测: {decision}"
                )
            else:
                # 无回退，返回中性值
                decision = 0.0
                logger.warning(
                    f"[DriftAwareEnsemble] 所有模型漂移且无回退，返回中性值"
                )

        # 记录漂移事件
        drifted_count = sum(1 for r in drift_report.values() if r.get("drifted"))
        if drifted_count > 0:
            self._drift_events.append({
                "timestamp": datetime.now().isoformat(),
                "stock_code": stock_code,
                "drifted_models": [
                    name for name, r in drift_report.items() if r.get("drifted")
                ],
                "active_models": list(active_predictions.keys()),
            })

        # 因子权重联动（如果注册了 scheduler）
        if hasattr(self, '_factor_scheduler'):
            factor_weights = self._factor_scheduler.adjust_for_drift(drift_report)
        else:
            factor_weights = {}

        return decision, {
            "decision": decision,
            "drift_report": drift_report,
            "active_models": list(active_predictions.keys()),
            "drifted_models": [
                name for name, r in drift_report.items() if r.get("drifted")
            ],
            "fallback": len(active_predictions) == 0,
            "factor_weights": factor_weights,
        }

    def update_and_check(self, stock_code: str, predictions: Dict[str, float],
                         actual: float):
        """
        更新所有模型监控器并检查漂移

        Args:
            stock_code: 股票代码
            predictions: {model_name: prediction}
            actual: 实际值
        """
        for name, pred in predictions.items():
            if name not in self._monitors:
                continue

            try:
                drifted = self._monitors[name].update(float(pred), float(actual))
            except Exception as e:
                logger.error(
                    f"[DriftAwareEnsemble] 更新监控器 {name} 失败: {e}"
                )
                continue

            # 记录历史
            if stock_code not in self._history:
                self._history[stock_code] = []
            self._history[stock_code].append({
                "timestamp": datetime.now().isoformat(),
                "model": name,
                "prediction": float(pred),
                "actual": float(actual),
                "drifted": drifted,
            })

            # 漂移后缓存该预测作为回退
            if drifted:
                self._fallback_cache[stock_code] = float(pred)
                logger.info(
                    f"[DriftAwareEnsemble] 模型 {name} 检测到漂移，"
                    f"已缓存该预测作为回退"
                )

        # 清理过期的历史记录（只保留 lookback 天）
        self._cleanup_history()

    def _cleanup_history(self):
        """清理超过 lookback 天的历史记录"""
        cutoff = datetime.now()
        for stock_code in list(self._history.keys()):
            self._history[stock_code] = [
                record for record in self._history[stock_code]
                if self._is_recent(record["timestamp"], cutoff)
            ]
            if not self._history[stock_code]:
                del self._history[stock_code]

    @staticmethod
    def _is_recent(timestamp_str: str, cutoff: datetime, max_days: int = 60) -> bool:
        """判断时间戳是否在 max_days 天内"""
        try:
            ts = datetime.fromisoformat(timestamp_str)
            return (cutoff - ts).days < max_days
        except Exception:
            return True

    def get_drift_report(self) -> Dict:
        """
        获取漂移报告

        Returns:
            {
                "models": {
                    "model_name": {
                        "drifted": bool,
                        "confidence": float,
                        "ks_p_value": float,
                        ...
                    }
                },
                "weights": {"model_name": weight},
                "drift_events_count": int,
                "active_models": [names],
            }
        """
        model_reports = {}
        active_models = []

        for name, monitor in self._monitors.items():
            status = monitor.get_status()
            model_reports[name] = status
            if not monitor.is_drifted():
                active_models.append(name)

        return {
            "models": model_reports,
            "weights": {k: round(v, 4) for k, v in self._weights.items()},
            "drift_events_count": len(self._drift_events),
            "recent_drift_events": self._drift_events[-10:],  # 最近 10 条
            "active_models": active_models,
            "fallback_cache_size": len(self._fallback_cache),
        }

    def get_fallback_model(self) -> Optional[str]:
        """
        获取回退模型（最后一个非漂移模型）

        Returns:
            非漂移模型名称，或 None（全部漂移）
        """
        for name, monitor in self._monitors.items():
            if not monitor.is_drifted():
                return name
        return None

    def register_factor_scheduler(self, scheduler: Any):
        """
        注册因子权重调度器，实现与 factor_weight_scheduler 联动

        Args:
            scheduler: 因子权重调度器实例，需有 adjust_for_drift(drift_report) 方法
        """
        self._factor_scheduler = scheduler
        logger.info(
            f"[DriftAwareEnsemble] 已注册因子权重调度器: {scheduler.__class__.__name__}"
        )

    def reset_model(self, name: str):
        """
        手动重置指定模型的漂移状态和校准分布

        Args:
            name: 模型名称
        """
        if name in self._monitors:
            self._monitors[name].reset_calibration()
            logger.info(f"[DriftAwareEnsemble] 手动重置模型 {name} 的漂移状态")
        else:
            logger.warning(f"[DriftAwareEnsemble] 模型 {name} 不存在，无法重置")

    def get_model_status(self, name: str) -> Optional[Dict]:
        """
        获取指定模型的监控状态

        Args:
            name: 模型名称

        Returns:
            监控状态字典，或 None
        """
        if name in self._monitors:
            return self._monitors[name].get_status()
        return None
