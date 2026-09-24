#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
内存管理模块

P3 优化 (2026-07-01):
  监控系统内存使用，提供自动垃圾回收、大对象检测和内存泄漏预警。

功能:
  1. 内存使用监控（当前使用量、峰值、增长率）
  2. 大对象检测（按类型统计内存占用）
  3. 自动垃圾回收触发
  4. 内存泄漏预警（持续增长的内存使用）
"""

import gc
import sys
import threading
import tracemalloc
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from modules.logger import logger

# 启动 tracemalloc（用于内存分配追踪）
try:
    tracemalloc.start()
    TRACEMALLOC_ENABLED = True
except RuntimeError:
    TRACEMALLOC_ENABLED = False


@dataclass
class MemorySnapshot:
    """内存快照"""
    timestamp: float
    rss_mb: float           # RSS 内存（MB）
    python_heap_mb: float   # Python 堆内存（MB）
    object_count: int       # 对象总数
    gc_counts: List[int]    # GC 计数 [generation 0, 1, 2]
    top_allocators: List[Dict] = field(default_factory=list)  # tracemalloc 顶层分配

    @property
    def is_healthy(self) -> bool:
        """内存是否健康"""
        return self.rss_mb < 2048  # RSS < 2GB


class MemoryMonitor:
    """内存监控器"""

    def __init__(self, warning_threshold_mb: float = 1500,
                 critical_threshold_mb: float = 1800,
                 auto_gc_threshold: Optional[float] = None):
        """
        Args:
            warning_threshold_mb: 警告阈值（MB）
            critical_threshold_mb: 危急阈值（MB）
            auto_gc_threshold: 自动触发 GC 的阈值（MB），None 表示不自动触发
        """
        self._warning_threshold = warning_threshold_mb
        self._critical_threshold = critical_threshold_mb
        self._auto_gc_threshold = auto_gc_threshold
        self._lock = threading.Lock()

        # 历史记录（保留最近 100 个快照）
        self._history: List[MemorySnapshot] = []
        self._peak_rss_mb: float = 0.0

        # 内存泄漏检测
        self._allocation_history: Dict[str, List] = {}

        logger.info(f"[MemoryMonitor] 初始化: warning={warning_threshold_mb}MB, "
                     f"critical={critical_threshold_mb}MB, "
                     f"auto_gc={'enabled' if auto_gc_threshold else 'disabled'}")

    def take_snapshot(self) -> MemorySnapshot:
        """采集当前内存快照"""
        import psutil
        process = psutil.Process()
        rss_mb = process.memory_info().rss / 1024 / 1024

        # Python 堆内存
        if TRACEMALLOC_ENABLED:
            current, peak = tracemalloc.get_traced_memory()
            python_heap_mb = current / 1024 / 1024
            self._peak_rss_mb = max(self._peak_rss_mb, peak / 1024 / 1024)
        else:
            python_heap_mb = 0.0

        # 对象统计（在 GC 前采集）
        gc_counts = list(gc.get_count())
        object_count = sum(gc_counts)
        gc.collect()

        # tracemalloc 顶层分配
        top_allocators = []
        if TRACEMALLOC_ENABLED:
            snapshot = tracemalloc.take_snapshot()
            stats = snapshot.statistics('lineno')[:10]
            for stat in stats:
                top_allocators.append({
                    'file': str(stat.traceback),
                    'size_mb': round(stat.size / 1024 / 1024, 2),
                    'count': stat.count,
                })

        snapshot = MemorySnapshot(
            timestamp=__import__('time').time(),
            rss_mb=round(rss_mb, 2),
            python_heap_mb=round(python_heap_mb, 2),
            object_count=object_count,
            gc_counts=gc_counts,
            top_allocators=top_allocators,
        )

        # 记录历史
        with self._lock:
            self._history.append(snapshot)
            if len(self._history) > 100:
                self._history.pop(0)

        # 健康检查
        self._check_health(snapshot)

        return snapshot

    def _check_health(self, snapshot: MemorySnapshot):
        """内存健康检查"""
        rss = snapshot.rss_mb

        if rss > self._critical_threshold:
            logger.error(
                f"[MemoryMonitor] CRITICAL: RSS={rss:.0f}MB > {self._critical_threshold}MB! "
                f"对象数={snapshot.object_count}"
            )
            # 强制 GC
            gc.collect()

        elif rss > self._warning_threshold:
            logger.warning(
                f"[MemoryMonitor] WARNING: RSS={rss:.0f}MB > {self._warning_threshold}MB"
            )

        # 自动 GC
        if (self._auto_gc_threshold and rss > self._auto_gc_threshold
                and snapshot.gc_counts[0] > 100):
            logger.info(
                f"[MemoryMonitor] Auto-GC triggered: RSS={rss:.0f}MB, "
                f"gc0={snapshot.gc_counts[0]}"
            )
            gc.collect()

    def detect_leaks(self) -> Dict:
        """
        检测内存泄漏

        Returns:
            {
                'potential_leak': bool,
                'growth_rate_mb_per_min': float,
                'top_growing': List[Dict],
            }
        """
        if len(self._history) < 5:
            return {
                'potential_leak': False,
                'reason': '数据不足（需要至少 5 个快照）',
                'growth_rate_mb_per_min': 0.0,
            }

        # 计算 RSS 增长率
        recent = self._history[-10:]
        rss_values = [s.rss_mb for s in recent]
        time_diffs = [
            recent[i].timestamp - recent[0].timestamp
            for i in range(1, len(recent))
        ]

        if not time_diffs or max(time_diffs) < 10:
            return {
                'potential_leak': False,
                'reason': '时间间隔太短',
                'growth_rate_mb_per_min': 0.0,
            }

        # 简单线性回归
        n = len(rss_values)
        sum_x = sum(time_diffs)
        sum_y = sum(rss_values[1:])
        sum_xy = sum(
            time_diffs[i] * rss_values[i + 1]
            for i in range(n - 1)
        )
        sum_x2 = sum(x ** 2 for x in time_diffs)

        denominator = n * sum_x2 - sum_x ** 2
        if abs(denominator) < 1e-10:
            growth_rate = 0.0
        else:
            growth_rate = (n * sum_xy - sum_x * sum_y) / denominator  # MB/秒

        growth_rate_mb_per_min = growth_rate * 60

        # 判断是否泄漏（增长率 > 1MB/min 持续一段时间）
        potential_leak = growth_rate_mb_per_min > 1.0

        return {
            'potential_leak': potential_leak,
            'growth_rate_mb_per_min': round(growth_rate_mb_per_min, 3),
            'current_rss_mb': round(rss_values[-1], 2),
            'peak_rss_mb': round(self._peak_rss_mb, 2),
            'snapshot_count': len(self._history),
        }

    def force_gc(self) -> Dict:
        """强制垃圾回收"""
        import psutil
        process = psutil.Process()
        before = process.memory_info().rss / 1024 / 1024

        collected = gc.collect()
        gc.collect()
        gc.collect()

        after = process.memory_info().rss / 1024 / 1024

        logger.info(f"[MemoryMonitor] GC 回收了 {collected} 个对象，"
                     f"释放 {(before - after):.1f}MB")

        return {
            'collected': collected,
            'before_mb': round(before, 2),
            'after_mb': round(after, 2),
            'freed_mb': round(before - after, 2),
        }

    def get_status(self) -> Dict:
        """获取当前内存状态"""
        snapshot = self.take_snapshot()
        leak_info = self.detect_leaks()

        return {
            'rss_mb': snapshot.rss_mb,
            'python_heap_mb': snapshot.python_heap_mb,
            'object_count': snapshot.object_count,
            'gc_counts': snapshot.gc_counts,
            'peak_rss_mb': round(self._peak_rss_mb, 2),
            'snapshot_count': len(self._history),
            'health': 'healthy' if snapshot.is_healthy else 'degraded',
            'leak_detection': leak_info,
            'top_allocators': snapshot.top_allocators[:5],
        }

    def get_history(self, n: int = 20) -> List[Dict]:
        """获取最近的内存快照历史"""
        return [
            {
                'rss_mb': s.rss_mb,
                'python_heap_mb': s.python_heap_mb,
                'object_count': s.object_count,
                'timestamp': s.timestamp,
            }
            for s in self._history[-n:]
        ]


# 全局单例
_monitor: Optional[MemoryMonitor] = None


def get_memory_monitor() -> MemoryMonitor:
    """获取内存监控器全局实例"""
    global _monitor
    if _monitor is None:
        _monitor = MemoryMonitor(
            warning_threshold_mb=1500,
            critical_threshold_mb=1800,
            auto_gc_threshold=1200,
        )
    return _monitor


def reset_memory_monitor():
    """重置全局实例（用于测试）"""
    global _monitor
    _monitor = None
