"""
monitor.py - 监控告警系统

功能:
1. 请求追踪中间件 — 记录每个请求的耗时、状态码
2. 慢请求告警 — 超过阈值的请求触发告警
3. 错误率监控 — 按端点统计错误率
4. 健康检查增强 — 依赖服务状态检查
"""

import time
import threading
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Optional, Callable, Dict, Any
from modules.logger import logger

# ── 配置 ──────────────────────────────────────────────

SLOW_REQUEST_THRESHOLD_MS = 5000  # 慢请求阈值（5 秒）
ERROR_RATE_WINDOW_SEC = 300       # 错误率统计窗口（5 分钟）
MAX_REQUEST_LOG = 10000           # 请求日志最大条数
ALERT_COOLDOWN_SEC = 300          # 告警冷却期（5 分钟）


class RequestTracker:
    """请求追踪器 — 记录所有请求的指标"""

    def __init__(self, max_log: int = MAX_REQUEST_LOG):
        self.max_log = max_log
        self._lock = threading.Lock()
        # 按端点统计
        self._endpoint_stats: Dict[str, dict] = defaultdict(lambda: {
            'total': 0,
            'errors': 0,
            'total_time_ms': 0.0,
            'max_time_ms': 0.0,
            'min_time_ms': float('inf'),
        })
        # 请求日志（环形缓冲区）
        self._request_log: deque = deque(maxlen=max_log)
        # 全局指标
        self._global = {
            'total_requests': 0,
            'total_errors': 0,
            'start_time': datetime.now().isoformat(),
        }

    def record_request(self, endpoint: str, status_code: int,
                       duration_ms: float, method: str = 'GET'):
        """记录一次请求"""
        now = datetime.now().isoformat()
        is_error = status_code >= 400

        with self._lock:
            # 全局
            self._global['total_requests'] += 1
            if is_error:
                self._global['total_errors'] += 1

            # 按端点
            stats = self._endpoint_stats[endpoint]
            stats['total'] += 1
            if is_error:
                stats['errors'] += 1
            stats['total_time_ms'] += duration_ms
            stats['max_time_ms'] = max(stats['max_time_ms'], duration_ms)
            if duration_ms > 0:
                stats['min_time_ms'] = min(stats['min_time_ms'], duration_ms)

            # 请求日志
            entry = {
                'timestamp': now,
                'method': method,
                'endpoint': endpoint,
                'status_code': status_code,
                'duration_ms': round(duration_ms, 2),
                'is_error': is_error,
                'is_slow': duration_ms > SLOW_REQUEST_THRESHOLD_MS,
            }
            self._request_log.append(entry)

            # 慢请求告警
            if duration_ms > SLOW_REQUEST_THRESHOLD_MS:
                logger.warning(
                    f"[Monitor] SLOW REQUEST: {method} {endpoint} "
                    f"took {duration_ms:.0f}ms (threshold: {SLOW_REQUEST_THRESHOLD_MS}ms)"
                )

            # 高错误率告警
            if is_error:
                error_rate = self._get_error_rate(endpoint)
                if error_rate > 0.5:  # 端点错误率 > 50%
                    logger.error(
                        f"[Monitor] HIGH ERROR RATE: {method} {endpoint} "
                        f"error_rate={error_rate:.1%}"
                    )

    def _get_error_rate(self, endpoint: str) -> float:
        """获取端点错误率"""
        stats = self._endpoint_stats[endpoint]
        if stats['total'] == 0:
            return 0.0
        return stats['errors'] / stats['total']

    def get_metrics(self) -> dict:
        """获取监控指标"""
        with self._lock:
            # 计算平均值
            endpoint_metrics = {}
            for endpoint, stats in self._endpoint_stats.items():
                total = stats['total']
                avg_time = stats['total_time_ms'] / total if total > 0 else 0
                min_time = stats['min_time_ms'] if stats['min_time_ms'] != float('inf') else 0
                endpoint_metrics[endpoint] = {
                    'total_requests': total,
                    'errors': stats['errors'],
                    'error_rate': round(self._get_error_rate(endpoint), 4),
                    'avg_time_ms': round(avg_time, 2),
                    'max_time_ms': round(stats['max_time_ms'], 2),
                    'min_time_ms': round(min_time, 2),
                }

            total = self._global['total_requests']
            return {
                'global': {
                    **self._global,
                    'error_rate': round(
                        self._global['total_errors'] / total if total > 0 else 0, 4
                    ),
                },
                'endpoints': endpoint_metrics,
                'recent_errors': [
                    entry for entry in list(self._request_log)[-50:]
                    if entry['is_error']
                ][::-1],
            }

    def reset(self):
        """重置所有统计"""
        with self._lock:
            self._endpoint_stats.clear()
            self._request_log.clear()
            self._global = {
                'total_requests': 0,
                'total_errors': 0,
                'start_time': datetime.now().isoformat(),
            }


# 全局单例
_request_tracker = RequestTracker()


def get_request_tracker() -> RequestTracker:
    """获取请求追踪器单例"""
    return _request_tracker


class HealthChecker:
    """健康检查器 — 检查依赖服务状态"""

    def __init__(self):
        self._checks: Dict[str, Callable] = {}
        self._last_results: Dict[str, dict] = {}

    def register_check(self, name: str, check_fn: Callable):
        """注册健康检查函数"""
        self._checks[name] = check_fn

    def run_all(self) -> dict:
        """运行所有健康检查"""
        results = {}
        all_healthy = True

        for name, check_fn in self._checks.items():
            try:
                result = check_fn()
                healthy = result.get('healthy', True)
                results[name] = {**result, 'healthy': healthy}
                if not healthy:
                    all_healthy = False
                    logger.warning(f"[HealthCheck] {name} unhealthy: {result}")
            except Exception as e:
                results[name] = {'healthy': False, 'error': str(e)}
                all_healthy = False

        self._last_results = results
        return {
            'healthy': all_healthy,
            'checks': results,
            'timestamp': datetime.now().isoformat(),
        }

    def get_last_results(self) -> dict:
        """获取上次检查结果"""
        return self._last_results


# 全局单例
_health_checker = HealthChecker()


def get_health_checker() -> HealthChecker:
    """获取健康检查器单例"""
    return _health_checker


# ── Flask 中间件 ──────────────────────────────────────

def init_monitor_middleware(app):
    """初始化 Flask 监控中间件"""
    tracker = get_request_tracker()

    @app.before_request
    def before_request_monitor():
        """记录请求开始时间"""
        request._start_time = time.time()

    @app.after_request
    def after_request_monitor(response):
        """记录请求结束"""
        start_time = getattr(request, '_start_time', None)
        if start_time is not None:
            duration_ms = (time.time() - start_time) * 1000
            # 使用 endpoint 名称（不含 Blueprint 前缀）
            endpoint = getattr(request, 'endpoint', 'unknown')
            # 简化 endpoint 名称（去掉 blueprint 前缀）
            if '.' in endpoint:
                endpoint = endpoint.split('.', 1)[1]
            tracker.record_request(
                endpoint=endpoint,
                status_code=response.status_code,
                duration_ms=duration_ms,
                method=request.method,
            )
        return response
