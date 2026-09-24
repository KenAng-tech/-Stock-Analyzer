#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
API限流器
"""

import time
import threading
from functools import wraps
from flask import request, jsonify
from collections import defaultdict
from modules.logger import logger


class RateLimiter:
    """
    API限流器

    使用滑动窗口算法
    """

    def __init__(self, default_rate: str = "100/minute"):
        self._requests = defaultdict(list)
        self._lock = threading.Lock()
        self._cleanup_thread = None
        self._default_rate = self._parse_rate(default_rate)
        self._endpoint_limits = {}

    def _parse_rate(self, rate: str) -> tuple:
        """解析速率字符串"""
        parts = rate.split('/')
        if len(parts) != 2:
            return (100, 60)  # 默认100次/分钟

        count = int(parts[0])
        unit = parts[1].lower()

        time_map = {
            'second': 1,
            'minute': 60,
            'hour': 3600,
            'day': 86400
        }

        return (count, time_map.get(unit, 60))

    def set_endpoint_limit(self, endpoint: str, rate: str):
        """设置特定端点的限流"""
        self._endpoint_limits[endpoint] = self._parse_rate(rate)

    def _get_client_ip(self) -> str:
        """获取客户端IP"""
        return request.remote_addr

    def _clean_old_requests(self, key: str, window: int):
        """清理过期请求"""
        current_time = time.time()
        with self._lock:
            if key in self._requests:
                self._requests[key] = [
                    t for t in self._requests[key]
                    if current_time - t < window
                ]

    def is_allowed(self, endpoint: str = None) -> tuple:
        """
        检查请求是否允许

        Returns:
            (is_allowed, retry_after_seconds)
        """
        client_ip = self._get_client_ip()
        key = f"{client_ip}:{endpoint}"

        # 获取限流配置
        if endpoint and endpoint in self._endpoint_limits:
            limit, window = self._endpoint_limits[endpoint]
        else:
            limit, window = self._default_rate

        # 清理过期请求
        self._clean_old_requests(key, window)

        # 检查是否超限
        current_time = time.time()
        with self._lock:
            if len(self._requests[key]) >= limit:
                # 计算重试时间
                oldest_request = self._requests[key][0]
                retry_after = window - (current_time - oldest_request)
                return False, max(retry_after, 1)

            # 记录请求
            self._requests[key].append(current_time)
            return True, 0

    def limit(self, rate: str = None):
        """
        限流装饰器

        用法:
            @app.route('/api/test')
            @rate_limiter.limit("10/minute")
            def test():
                return jsonify({'success': True})
        """
        def decorator(f):
            @wraps(f)
            def wrapper(*args, **kwargs):
                endpoint = request.endpoint
                is_allowed, retry_after = self.is_allowed(endpoint)

                if not is_allowed:
                    return jsonify({
                        'success': False,
                        'error': 'Rate limit exceeded',
                        'retry_after': retry_after
                    }), 429

                return f(*args, **kwargs)
            return wrapper
        return decorator


# 全局限流器实例
rate_limiter = RateLimiter()