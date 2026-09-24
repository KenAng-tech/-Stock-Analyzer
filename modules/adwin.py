#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
ADWIN (Adaptive Windowing) 漂移检测算法

统一 ADWIN 实现，被 drift_monitor.py 和 concept_drift_detector.py 共享。
参考: Bifet & Gavalda, "Adaptive Windowing for Data Stream Classification" (2007)
"""

from collections import deque
import numpy as np
from modules.logger import logger


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

            window_list = list(self.window)
            mean0 = np.mean(window_list[:cut])
            mean1 = np.mean(window_list[cut:])

            delta_prime = np.log(4.0 / self.delta)
            m = (n0 * n1) / (n0 + n1)
            epsilon = np.sqrt(
                (delta_prime / (2.0 * m))
                + (delta_prime / (6.0 * n) * np.log(4.0 / self.delta))
            )

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
