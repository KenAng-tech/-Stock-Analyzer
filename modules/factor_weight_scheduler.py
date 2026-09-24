#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
动态因子权重调度器 — Dynamic Factor Weight Scheduler

基于 IC/ICIR 表现的因子权重动态调整。
当检测到概念漂移时，自动重新校准因子权重。

算法:
    1. 计算各因子的滚动 IC/ICIR
    2. 基于 ICIR 分配权重 (权重 ∝ |ICIR|)
    3. 当检测到漂移时，增加权重调整频率
    4. 支持平滑过渡 (避免权重跳跃)

参考:
    - Grinold, R.C. "The Fundamental Law of Active Management"
    - 因子 IC 衰减分析
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
from datetime import datetime
from modules.logger import logger


class DynamicFactorWeightScheduler:
    """动态因子权重调度器"""

    def __init__(self, base_weights: Optional[Dict[str, float]] = None,
                 rebalance_threshold: float = 0.1,
                 smoothing_factor: float = 0.3):
        """
        Args:
            base_weights: 基础因子权重 (默认等权)
            rebalance_threshold: ICIR 变化超过此阈值时重新平衡
            smoothing_factor: 权重平滑因子 (0-1)，越小越平滑
        """
        self.base_weights = base_weights or {}
        self.current_weights = {}
        self.previous_weights = {}
        self.rebalance_threshold = rebalance_threshold
        self.smoothing_factor = smoothing_factor
        self._ic_history: Dict[str, deque] = {}
        self._last_rebalance: Optional[datetime] = None
        self._rebalance_count = 0
        self._drift_detected = False

        # 默认因子权重 (基于传统量化实践)
        if not self.base_weights:
            self.base_weights = {
                'momentum': 0.20,
                'value': 0.15,
                'volatility': 0.10,
                'volume': 0.10,
                'quality': 0.15,
                'technical': 0.15,
                'sentiment': 0.15,
            }

        # 初始化当前权重
        total = sum(self.base_weights.values())
        self.current_weights = {k: v / total for k, v in self.base_weights.items()}
        self.previous_weights = self.current_weights.copy()

    def update_ic(self, factor_name: str, ic_value: float):
        """
        更新因子 IC 历史

        Args:
            factor_name: 因子名称
            ic_value: IC 值
        """
        if factor_name not in self._ic_history:
            self._ic_history[factor_name] = deque(maxlen=60)
        self._ic_history[factor_name].append(ic_value)

    def compute_icir(self, factor_name: str, window: int = 20) -> float:
        """
        计算因子 ICIR (IC Information Ratio)

        Args:
            factor_name: 因子名称
            window: 滚动窗口

        Returns:
            ICIR 值
        """
        history = self._ic_history.get(factor_name, deque())
        if len(history) < window:
            return 0.0

        recent = list(history)[-window:]
        arr = np.array(recent)
        mean_ic = np.mean(arr)
        std_ic = np.std(arr)

        if std_ic < 1e-10:
            return 0.0

        return float(mean_ic / std_ic)

    def get_icir_ranking(self) -> List[Tuple[str, float]]:
        """
        获取因子 ICIR 排名

        Returns:
            [(factor_name, icir), ...] 按 ICIR 降序
        """
        icirs = []
        for factor_name in self.base_weights:
            icir = self.compute_icir(factor_name)
            icirs.append((factor_name, icir))

        # 按 ICIR 绝对值降序
        icirs.sort(key=lambda x: abs(x[1]), reverse=True)
        return icirs

    def rebalance(self) -> Dict[str, float]:
        """
        根据 ICIR 重新平衡因子权重

        Returns:
            新的因子权重
        """
        self.previous_weights = self.current_weights.copy()
        icirs = self.get_icir_ranking()

        # 基于 ICIR 绝对值分配权重
        abs_icirs = [abs(icir) for _, icir in icirs]
        total_abs = sum(abs_icirs)

        if total_abs > 0:
            new_weights = {}
            for factor_name, icir in icirs:
                weight = abs_icirs[icirs.index((factor_name, icir))] / total_abs
                new_weights[factor_name] = weight
            self.current_weights = new_weights
        else:
            # 无 IC 数据时回退到基础权重
            total = sum(self.base_weights.values())
            self.current_weights = {k: v / total for k, v in self.base_weights.items()}

        self._last_rebalance = datetime.now()
        self._rebalance_count += 1

        logger.info(f"[WeightScheduler] 因子权重已重新平衡 (第{self._rebalance_count}次)")
        for fname, w in sorted(self.current_weights.items(), key=lambda x: -x[1]):
            logger.info(f"  - {fname}: {w:.4f}")

        return self.current_weights

    def should_rebalance(self) -> bool:
        """
        判断是否需要重新平衡

        条件:
            1. 距上次平衡超过 24 小时
            2. 检测到概念漂移
            3. 最大权重变化超过阈值
        """
        if self._drift_detected:
            return True

        if self._last_rebalance is None:
            return True  # 从未平衡过

        elapsed = (datetime.now() - self._last_rebalance).total_seconds()
        if elapsed > 86400:  # 24 小时
            return True

        # 检查最大权重变化
        max_change = 0
        for fname in self.base_weights:
            current = self.current_weights.get(fname, 0)
            prev = self.previous_weights.get(fname, 0)
            max_change = max(max_change, abs(current - prev))

        return max_change > self.rebalance_threshold

    def on_drift_detected(self):
        """当检测到概念漂移时调用"""
        self._drift_detected = True
        logger.warning("[WeightScheduler] 检测到概念漂移，将加速权重调整")

    def on_drift_reset(self):
        """当漂移检测重置时调用"""
        self._drift_detected = False

    def get_smoothed_weights(self) -> Dict[str, float]:
        """
        获取平滑后的权重 (避免权重跳跃)

        Returns:
            平滑后的因子权重
        """
        smoothed = {}
        for fname in self.base_weights:
            current = self.current_weights.get(fname, 0)
            previous = self.previous_weights.get(fname, 0)
            smoothed[fname] = (1 - self.smoothing_factor) * previous + self.smoothing_factor * current
            smoothed[fname] = round(smoothed[fname], 6)

        # 归一化
        total = sum(smoothed.values())
        if total > 0:
            smoothed = {k: v / total for k, v in smoothed.items()}

        return smoothed

    def get_status(self) -> Dict:
        """获取调度器状态"""
        icirs = self.get_icir_ranking()
        return {
            'current_weights': self.current_weights,
            'smoothed_weights': self.get_smoothed_weights(),
            'base_weights': self.base_weights,
            'icir_ranking': [{'factor': f, 'icir': round(i, 4)} for f, i in icirs],
            'last_rebalance': self._last_rebalance.isoformat() if self._last_rebalance else None,
            'rebalance_count': self._rebalance_count,
            'drift_detected': self._drift_detected,
            'should_rebalance': self.should_rebalance(),
            'smoothing_factor': self.smoothing_factor,
        }

    def reduce_factor_weight(self, factor_name: str, multiplier: float = 0.5) -> Dict:
        """
        降低指定因子的权重 (用于 IC 衰减响应)

        Args:
            factor_name: 因子名称
            multiplier: 权重乘数 (0.5 = 降权 50%, 0.75 = 降权 25%)

        Returns:
            调整后的权重字典
        """
        if factor_name not in self.current_weights:
            logger.warning(f"[WeightScheduler] 因子 {factor_name} 不在当前权重中")
            return self.current_weights

        self.current_weights[factor_name] *= multiplier
        self.current_weights[factor_name] = round(self.current_weights[factor_name], 6)

        # 归一化
        total = sum(self.current_weights.values())
        if total > 0:
            self.current_weights = {k: round(v / total, 6) for k, v in self.current_weights.items()}

        logger.info(
            f"[WeightScheduler] 因子降权: {factor_name} ×{multiplier} → "
            f"{self.current_weights.get(factor_name, 0):.4f}"
        )
        return dict(self.current_weights)

    def reset(self):
        """重置调度器"""
        self._ic_history.clear()
        self._last_rebalance = None
        self._rebalance_count = 0
        self._drift_detected = False
        self.previous_weights = {}
        total = sum(self.base_weights.values())
        self.current_weights = {k: v / total for k, v in self.base_weights.items()}
        logger.info("[WeightScheduler] 调度器已重置")


# 全局实例
_factor_weight_scheduler: Optional[DynamicFactorWeightScheduler] = None


def get_factor_weight_scheduler() -> DynamicFactorWeightScheduler:
    """获取全局因子权重调度器实例 (线程安全)"""
    global _factor_weight_scheduler
    if _factor_weight_scheduler is None:
        _factor_weight_scheduler = DynamicFactorWeightScheduler()
    return _factor_weight_scheduler


def reset_factor_weight_scheduler():
    """重置因子权重调度器"""
    global _factor_weight_scheduler
    if _factor_weight_scheduler:
        _factor_weight_scheduler.reset()
    _factor_weight_scheduler = DynamicFactorWeightScheduler()
    logger.info("[WeightScheduler] 调度器已重置 (全局)")
