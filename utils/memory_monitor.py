"""
utils/memory_monitor.py — 内存监控告警

RSS 阈值告警:
  - RSS > 2GB: WARNING
  - RSS > 3GB: ERROR + 自动 GC
  - 定期清理缓存

使用方式:
    from utils.memory_monitor import MemoryMonitor

    monitor = MemoryMonitor()

    # 检查内存状态
    status = monitor.check()
    if status['action'] == 'gc':
        import gc; gc.collect()
        logger.warning(f"[memory] RSS {status['rss_mb']:.0f}MB, 已触发 GC")

    # 定期调用 (每 5 分钟)
    monitor.check_and_maybe_gc()
"""

import os
import gc
import logging
import threading
from datetime import datetime
from typing import Optional

log = logging.getLogger(__name__)

# RSS 阈值 (MB)
RSS_WARNING_MB = 2048   # 2GB
RSS_ERROR_MB = 3072     # 3GB
RSS_CRITICAL_MB = 4096  # 4GB

# 缓存清理阈值
CACHE_CLEAR_THRESHOLD = 5000  # 缓存条目超过此数时清理


class MemoryMonitor:
    """内存监控器"""

    def __init__(self, warning_mb: int = RSS_WARNING_MB, error_mb: int = RSS_ERROR_MB):
        self.warning_mb = warning_mb
        self.error_mb = error_mb
        self._history: list[dict] = []  # 最近 100 条 RSS 记录
        self._lock = threading.Lock()

    def get_rss_mb(self) -> float:
        """获取当前进程的 RSS (MB)"""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            try:
                import resource
                # macOS: ru_maxrss 是 KB; Linux: ru_maxrss 是 KB
                usage = resource.getrusage(resource.RUSAGE_SELF)
                return usage.ru_maxrss / 1024.0
            except Exception:
                return 0.0
        except Exception:
            return 0.0

    def check(self) -> dict:
        """
        检查内存状态。

        :return: {rss_mb, action, message, level}
        """
        rss_mb = self.get_rss_mb()

        # 记录历史
        with self._lock:
            self._history.append({
                "rss_mb": rss_mb,
                "timestamp": datetime.now().isoformat(),
            })
            if len(self._history) > 100:
                self._history = self._history[-100:]

        # 判断级别
        if rss_mb >= self.error_mb:
            level = "error"
            action = "gc"
            message = f"RSS {rss_mb:.0f}MB > 阈值 {self.error_mb}MB, 触发 GC"
        elif rss_mb >= self.warning_mb:
            level = "warning"
            action = "monitor"
            message = f"RSS {rss_mb:.0f}MB > 阈值 {self.warning_mb}MB, 建议关注"
        else:
            level = "ok"
            action = "none"
            message = f"RSS {rss_mb:.0f}MB, 正常"

        # 计算趋势
        trend = self._calculate_trend()

        return {
            "rss_mb": round(rss_mb, 1),
            "level": level,
            "action": action,
            "message": message,
            "trend": trend,
        }

    def _calculate_trend(self) -> str:
        """计算 RSS 趋势"""
        with self._lock:
            if len(self._history) < 5:
                return "insufficient_data"

            recent = [h["rss_mb"] for h in self._history[-10:]]
            if len(recent) < 2:
                return "stable"

            # 简单线性趋势
            diffs = [recent[i+1] - recent[i] for i in range(len(recent)-1)]
            avg_diff = sum(diffs) / len(diffs)

            if avg_diff > 5:
                return "increasing"
            elif avg_diff < -5:
                return "decreasing"
            else:
                return "stable"

    def check_and_maybe_gc(self) -> Optional[str]:
        """
        检查内存, 如果超过阈值则触发 GC。

        :return: 动作描述或 None
        """
        status = self.check()

        if status["action"] == "gc":
            collected = gc.collect()
            log.warning(f"[memory] {status['message']} (GC collected {collected} objects)")
            return status["message"]

        if status["level"] == "warning":
            log.info(f"[memory] {status['message']}")

        return None

    def get_history(self, limit: int = 20) -> list[dict]:
        """获取 RSS 历史记录"""
        with self._lock:
            return self._history[-limit:]

    def clear_cache(self) -> int:
        """清理 Python 缓存"""
        # 清理 gc 垃圾
        collected = gc.collect()

        # 清理 sys.modules 中的过期模块
        import sys
        to_remove = [k for k in sys.modules if k.startswith('_') and k != '__main__']
        removed = 0
        for k in to_remove:
            del sys.modules[k]
            removed += 1

        log.info(f"[memory] 清理缓存: GC={collected}, sys.modules={removed}")
        return collected + removed
