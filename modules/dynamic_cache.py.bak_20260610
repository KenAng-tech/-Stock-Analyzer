"""
动态缓存管理器
分级缓存策略：实时行情 / 技术指标 / 因子计算 / ML 预测 / 基本面数据
新增：缓存命中统计 + 依赖链自动失效
"""

import time
import threading
from typing import Dict, Optional, Any, Callable, Set, List
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class CacheEntry:
    """缓存条目"""
    data: Any
    timestamp: float
    ttl: float
    access_count: int = 0
    last_access: float = 0.0
    category: str = 'default'
    tags: Set[str] = field(default_factory=set)

    def is_expired(self) -> bool:
        return time.time() - self.timestamp > self.ttl

    def touch(self):
        self.access_count += 1
        self.last_access = time.time()


class CacheStats:
    """缓存命中/未命中统计（线程安全）"""

    def __init__(self):
        self._lock = threading.Lock()
        # category -> {hits, misses}
        self._by_category: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {'hits': 0, 'misses': 0}
        )
        # key -> {hits, misses}
        self._by_key: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {'hits': 0, 'misses': 0}
        )
        # 全局累计
        self._global_hits: int = 0
        self._global_misses: int = 0

    def record_hit(self, key: str, category: str):
        with self._lock:
            self._global_hits += 1
            self._by_category[category]['hits'] += 1
            self._by_key[key]['hits'] += 1

    def record_miss(self, key: str, category: str):
        with self._lock:
            self._global_misses += 1
            self._by_category[category]['misses'] += 1
            self._by_key[key]['misses'] += 1

    def get_stats(self) -> Dict:
        with self._lock:
            total = self._global_hits + self._global_misses
            hit_rate = (self._global_hits / total * 100) if total > 0 else 0.0
            return {
                'global': {
                    'hits': self._global_hits,
                    'misses': self._global_misses,
                    'total': total,
                    'hit_rate_pct': round(hit_rate, 2),
                },
                'by_category': dict(self._by_category),
                'top_keys': self._top_keys(n=10),
            }

    def _top_keys(self, n: int = 10) -> List[Dict]:
        sorted_keys = sorted(
            self._by_key.items(),
            key=lambda x: x[1]['hits'] + x[1]['misses'],
            reverse=True,
        )
        result = []
        for key, counts in sorted_keys[:n]:
            total = counts['hits'] + counts['misses']
            rate = (counts['hits'] / total * 100) if total > 0 else 0.0
            result.append({
                'key': key,
                'hits': counts['hits'],
                'misses': counts['misses'],
                'total': total,
                'hit_rate_pct': round(rate, 2),
            })
        return result

    def reset(self):
        with self._lock:
            self._by_category.clear()
            self._by_key.clear()
            self._global_hits = 0
            self._global_misses = 0


class DynamicCache:
    """动态缓存管理器

    分级 TTL 配置:
        realtime      30s  — 实时行情/报价
        kline         30s  — K 线数据
        technical     60s  — 技术指标 (RSI/MACD/布林带)
        ml            60s  — ML 预测结果
        factor        300s — 因子计算 (15+ 因子)
        fundamental   300s — 基本面数据 (财报)
        sentiment     300s — 情感分析
        industry      600s — 行业数据
        strategy      120s — 策略结果
        default       60s  — 默认
    """

    # 预定义分类及其默认 TTL（秒）
    DEFAULT_TTLS = {
        'realtime': 30,       # 实时行情
        'kline': 30,          # K 线数据
        'technical': 60,      # 技术指标
        'ml': 60,             # ML 预测
        'factor': 300,        # 因子计算
        'fundamental': 300,   # 基本面
        'sentiment': 300,     # 情感分析
        'industry': 600,      # 行业数据
        'strategy': 120,      # 策略结果
        'default': 60,        # 默认
    }

    # 依赖关系：当某类数据更新时，自动失效的下游分类
    # 例如：realtime 数据更新 → technical / factor / ml 全部失效
    DEPENDENCY_CHAIN = {
        'realtime': ['technical', 'factor', 'ml'],
        'kline': ['technical', 'factor', 'ml'],
        'technical': ['factor', 'ml'],
        'fundamental': ['factor'],
        'sentiment': [],
        'industry': [],
        'strategy': [],
    }

    def __init__(self):
        self._store: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._ttl_overrides: Dict[str, float] = {}
        self._category_map: Dict[str, str] = {}  # key -> category（用于按分类清除）
        self._stats = CacheStats()

    # ── TTL 管理 ──────────────────────────────────────────────────

    def get_ttl(self, category: str) -> float:
        """获取某分类的 TTL"""
        return self._ttl_overrides.get(
            category, self.DEFAULT_TTLS.get(category, 60)
        )

    def set_ttl(self, category: str, ttl: float):
        """覆盖某分类的 TTL"""
        self._ttl_overrides[category] = ttl

    # ── 读写 ──────────────────────────────────────────────────────

    def get(self, key: str, category: str = 'default',
            default: Any = None) -> Any:
        """获取缓存数据，记录命中/未命中统计"""
        with self._lock:
            entry = self._store.get(key)
            if entry and not entry.is_expired():
                entry.touch()
                self._stats.record_hit(key, category)
                return entry.data
            elif entry:
                del self._store[key]
                self._category_map.pop(key, None)
            self._stats.record_miss(key, category)
            return default

    def set(self, key: str, data: Any,
            category: str = 'default', ttl: Optional[float] = None,
            tags: Optional[Set[str]] = None):
        """设置缓存数据，自动标记依赖链失效"""
        effective_ttl = ttl or self.get_ttl(category)
        with self._lock:
            # 如果写入的是 realtime/kline 等上游数据，自动使依赖链失效
            self._invalidate_dependents(category)

            self._store[key] = CacheEntry(
                data=data,
                timestamp=time.time(),
                ttl=effective_ttl,
                category=category,
                tags=tags or set(),
            )
            self._category_map[key] = category

    def invalidate(self, key: str):
        """使单个缓存键失效"""
        with self._lock:
            self._store.pop(key, None)
            self._category_map.pop(key, None)

    def invalidate_by_tags(self, tags: Set[str]):
        """按标签批量使缓存失效"""
        with self._lock:
            keys_to_remove = [
                k for k, v in self._store.items()
                if v.tags & tags
            ]
            for k in keys_to_remove:
                del self._store[k]
                self._category_map.pop(k, None)

    def invalidate_category(self, category: str):
        """使某分类所有缓存失效"""
        with self._lock:
            keys_to_remove = [
                k for k, v in self._store.items()
                if v.category == category
            ]
            for k in keys_to_remove:
                del self._store[k]
            for k in keys_to_remove:
                self._category_map.pop(k, None)

    def invalidate_key_prefix(self, prefix: str):
        """按前缀使缓存失效（如 stock_sh688981）"""
        with self._lock:
            keys_to_remove = [
                k for k in self._store if k.startswith(prefix)
            ]
            for k in keys_to_remove:
                del self._store[k]
                self._category_map.pop(k, None)

    def clear_category(self, category: str):
        """兼容旧接口：清除某分类所有缓存"""
        self.invalidate_category(category)

    def _invalidate_dependents(self, category: str):
        """使依赖链上的下游分类全部失效"""
        downstream = self.DEPENDENCY_CHAIN.get(category, [])
        for dc in downstream:
            keys_to_remove = [
                k for k, v in self._store.items()
                if v.category == dc
            ]
            for k in keys_to_remove:
                del self._store[k]
                self._category_map.pop(k, None)

    def invalidate_stock(self, stock_code: str):
        """使某只股票的所有缓存失效（含依赖链）"""
        prefix = f"stock_{stock_code}"
        self.invalidate_key_prefix(prefix)
        # 同时失效该股票的因子和 ML 缓存
        self.invalidate_key_prefix(f"factor_{stock_code}")
        self.invalidate_key_prefix(f"ml_{stock_code}")
        self.invalidate_key_prefix(f"analysis_{stock_code}")

    # ── 清理 ──────────────────────────────────────────────────────

    def cleanup(self):
        """清理过期缓存，返回清理数量"""
        with self._lock:
            expired = [k for k, v in self._store.items() if v.is_expired()]
            for k in expired:
                del self._store[k]
                self._category_map.pop(k, None)
            return len(expired)

    def cleanup_expired_by_category(self, category: str) -> int:
        """清理某分类的过期缓存"""
        with self._lock:
            expired = [
                k for k, v in self._store.items()
                if v.category == category and v.is_expired()
            ]
            for k in expired:
                del self._store[k]
                self._category_map.pop(k, None)
            return len(expired)

    # ── 统计 ──────────────────────────────────────────────────────

    def get_stats(self) -> Dict:
        """获取缓存统计（含命中率和条目信息）"""
        with self._lock:
            total = len(self._store)
            expired = sum(1 for v in self._store.values() if v.is_expired())
            active = total - expired
            total_accesses = sum(v.access_count for v in self._store.values())

            # 按分类统计条目数
            by_cat: Dict[str, int] = defaultdict(int)
            for v in self._store.values():
                by_cat[v.category] += 1

            return {
                'total_entries': total,
                'active_entries': active,
                'expired_entries': expired,
                'total_accesses': total_accesses,
                'categories': dict(self.DEFAULT_TTLS),
                'entries_by_category': dict(by_cat),
                'ttl_overrides': dict(self._ttl_overrides),
                'hit_stats': self._stats.get_stats(),
            }

    def reset_stats(self):
        """重置命中/未命中统计"""
        self._stats.reset()

    # ── 预热 ──────────────────────────────────────────────────────

    def warm_cache(self, fetcher: Callable, key: str,
                   category: str = 'default', ttl: Optional[float] = None,
                   tags: Optional[Set[str]] = None):
        """预热缓存"""
        data = fetcher()
        if data is not None:
            self.set(key, data, category, ttl, tags)
        return data

    # ── 便捷方法 ──────────────────────────────────────────────────

    def get_or_set(self, key: str, category: str,
                   fetcher: Callable, ttl: Optional[float] = None,
                   default: Any = None) -> Any:
        """获取缓存，未命中则调用 fetcher 填充"""
        result = self.get(key, category)
        if result is not None:
            return result
        data = fetcher()
        if data is not None:
            self.set(key, data, category, ttl)
        return data or default


# 全局缓存实例
cache = DynamicCache()
