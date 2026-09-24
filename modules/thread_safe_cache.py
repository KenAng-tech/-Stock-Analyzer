#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
线程安全的缓存实现 — 双后端: cachetools.TTLCache (快速) + ThreadSafeCache (LRU)
"""

import threading
import time
from collections import OrderedDict
from typing import Any, Optional, Dict
from modules.logger import logger

try:
    from cachetools import TTLCache, LRUCache
    HAS_CACHETOOLS = True
except ImportError:
    HAS_CACHETOOLS = False


class ThreadSafeCache:
    """
    线程安全的缓存

    特性:
    - 读写锁保护
    - 自动过期
    - LRU淘汰策略
    - 缓存统计
    """

    def __init__(self, max_size: int = 10000, default_ttl: int = 300):
        self._cache: OrderedDict = OrderedDict()
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._lock = threading.RLock()
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'expirations': 0
        }

    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存值

        Args:
            key: 缓存键

        Returns:
            缓存值或None
        """
        with self._lock:
            item = self._cache.get(key)
            if item is None:
                self._stats['misses'] += 1
                return None

            # 检查过期
            if time.time() > item['expires_at']:
                del self._cache[key]
                self._stats['expirations'] += 1
                self._stats['misses'] += 1
                return None

            # 移到最近使用
            self._cache.move_to_end(key)
            self._stats['hits'] += 1
            return item['value']

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """
        设置缓存值

        Args:
            key: 缓存键
            value: 缓存值
            ttl: 过期时间(秒)，None使用默认值
        """
        ttl = ttl or self._default_ttl
        expires_at = time.time() + ttl

        with self._lock:
            # 检查容量
            if len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)
                self._stats['evictions'] += 1

            self._cache[key] = {
                'value': value,
                'expires_at': expires_at,
                'created_at': time.time()
            }

    def delete(self, key: str) -> bool:
        """删除缓存"""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self):
        """清空缓存"""
        with self._lock:
            self._cache.clear()
            self._stats = {
                'hits': 0,
                'misses': 0,
                'evictions': 0,
                'expirations': 0
            }

    def size(self) -> int:
        """获取缓存大小"""
        with self._lock:
            return len(self._cache)

    def get_stats(self) -> Dict:
        """获取缓存统计"""
        with self._lock:
            total = self._stats['hits'] + self._stats['misses']
            return {
                'size': len(self._cache),
                'max_size': self._max_size,
                'hits': self._stats['hits'],
                'misses': self._stats['misses'],
                'hit_rate': self._stats['hits'] / total if total > 0 else 0,
                'evictions': self._stats['evictions'],
                'expirations': self._stats['expirations']
            }

    def cleanup_expired(self):
        """清理过期缓存"""
        current_time = time.time()
        expired_keys = []

        with self._lock:
            for key, item in self._cache.items():
                if current_time > item['expires_at']:
                    expired_keys.append(key)

            for key in expired_keys:
                del self._cache[key]
                self._stats['expirations'] += 1

        if expired_keys:
            logger.debug(f"[Cache] 清理了 {len(expired_keys)} 个过期条目")


# 全局缓存实例
thread_safe_cache = ThreadSafeCache()


# ── TTLCache 增强 (需要 cachetools) ──────────────────────
# 按分类管理不同 TTL 的缓存

if HAS_CACHETOOLS:
    class CategoryCache:
        """按分类管理 TTLCache，支持不同 TTL 策略"""

        def __init__(self):
            self._lock = threading.RLock()
            # 分类 → TTLCache 实例
            self._categories: Dict[str, TTLCache] = {}
            # 默认 TTL (秒)
            self._default_ttl = 300
            # 最大条目数
            self._maxsize = 1000
            self._stats = {'hits': 0, 'misses': 0, 'evictions': 0}

        def _get_cache(self, category: str, ttl: int) -> TTLCache:
            """获取或创建分类缓存"""
            key = f"{category}:{ttl}"
            if key not in self._categories:
                self._categories[key] = TTLCache(
                    maxsize=self._maxsize, ttl=ttl
                )
            return self._categories[key]

        def get(self, key: str, category: str = 'default', ttl: Optional[int] = None) -> Optional[Any]:
            with self._lock:
                cache = self._get_cache(category, ttl or self._default_ttl)
                if key in cache:
                    self._stats['hits'] += 1
                    return cache[key]
                self._stats['misses'] += 1
                return None

        def set(self, key: str, value: Any, category: str = 'default', ttl: Optional[int] = None):
            with self._lock:
                cache = self._get_cache(category, ttl or self._default_ttl)
                if key not in cache and len(cache) >= cache.maxsize:
                    self._stats['evictions'] += 1
                cache[key] = value

        def delete(self, key: str, category: str = 'default') -> bool:
            with self._lock:
                for cache in self._categories.values():
                    if key in cache:
                        del cache[key]
                        return True
                return False

        def clear_category(self, category: str):
            with self._lock:
                keys_to_remove = [k for k in self._categories if k.startswith(f"{category}:")]
                for k in keys_to_remove:
                    del self._categories[k]

        def get_stats(self) -> Dict:
            with self._lock:
                total = self._stats['hits'] + self._stats['misses']
                return {
                    'categories': len(self._categories),
                    'total_entries': sum(len(c) for c in self._categories.values()),
                    'hits': self._stats['hits'],
                    'misses': self._stats['misses'],
                    'hit_rate': self._stats['hits'] / total if total > 0 else 0,
                    'evictions': self._stats['evictions'],
                }

    # 全局分类缓存实例
    category_cache = CategoryCache()
else:
    # cachetools 不可用时的降级实现
    class CategoryCache:
        """降级: 使用内置 ThreadSafeCache"""
        def __init__(self):
            self._cache = ThreadSafeCache()

        def get(self, key, category='default', ttl=None):
            return self._cache.get(key)

        def set(self, key, value, category='default', ttl=None):
            self._cache.set(key, value, ttl=ttl)

        def delete(self, key, category='default'):
            return self._cache.delete(key)

        def clear_category(self, category):
            self._cache.clear()

        def get_stats(self):
            return self._cache.get_stats()

    category_cache = CategoryCache()