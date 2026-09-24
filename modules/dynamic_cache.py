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
    # P2 优化 (2026-07-01): 区分实时数据缓存和模型缓存
    DEFAULT_TTLS = {
        'realtime': 30,         # 实时行情 — 30s
        'kline': 300,           # K 线数据 — 5min
        'technical': 300,       # 技术指标 — 5min
        'ml': 86400,            # ML 预测 — 24h (模型训练耗时数分钟)
        'factor': 3600,         # 因子计算 — 1h
        'factor_scores': 3600,  # 因子评分 — 1h
        'fundamental': 86400,   # 基本面 — 24h (财报不常变)
        'sentiment': 3600,      # 情感分析 — 1h
        'sentiment_factor': 3600,  # 情绪因子 — 1h
        'time_llm': 86400,      # Time-LLM 预测 — 24h
        'regime_switching': 86400,  # Regime-Switching — 24h
        'itransformer': 86400,    # iTransformer — 24h
        'patchtst': 86400,        # PatchTST — 24h
        'industry': 86400,        # 行业数据 — 24h
        'strategy': 300,          # 策略结果 — 5min
        'backtest': 86400,        # 回测结果 — 24h
        'default': 300,           # 默认 — 5min
    }

    # 依赖关系：当某类数据更新时，自动失效的下游分类
    DEPENDENCY_CHAIN = {
        'realtime': ['technical', 'factor', 'ml', 'time_llm', 'itransformer'],
        'kline': ['technical', 'factor', 'ml', 'time_llm', 'itransformer'],
        'technical': ['factor', 'ml'],
        'fundamental': ['factor'],
        'sentiment': ['sentiment_factor'],
        'sentiment_factor': [],
        'industry': [],
        'strategy': [],
        'time_llm': ['ml'],  # Time-LLM 更新 → ML 预测失效
        'itransformer': ['ml'],
    }

    def __init__(self, max_size: int = 10000):
        """
        动态缓存管理器

        Args:
            max_size: 最大缓存条目数，超过时触发 LRU 淘汰（默认 10000）
        """
        self.max_size = max_size
        self._store: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._ttl_overrides: Dict[str, float] = {}
        self._category_map: Dict[str, str] = {}  # key -> category（用于按分类清除）
        self._stats = CacheStats()
        self._eviction_count: int = 0  # 累计淘汰数

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
        """设置缓存数据，自动标记依赖链失效 + LRU 淘汰"""
        effective_ttl = ttl or self.get_ttl(category)
        with self._lock:
            # 如果写入的是 realtime/kline 等上游数据，自动使依赖链失效
            self._invalidate_dependents(category)

            # LRU 淘汰: 如果超过容量，删除最少访问的条目
            while len(self._store) >= self.max_size:
                self._evict_lru()

            self._store[key] = CacheEntry(
                data=data,
                timestamp=time.time(),
                ttl=effective_ttl,
                category=category,
                tags=tags or set(),
            )
            self._category_map[key] = category

    def _evict_lru(self):
        """淘汰最少访问的缓存条目 (LRU)"""
        if not self._store:
            return
        # 按 access_count 升序排序，淘汰最少的
        lru_key = min(self._store, key=lambda k: self._store[k].access_count)
        del self._store[lru_key]
        self._category_map.pop(lru_key, None)
        self._eviction_count += 1

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

            # 计算数据库文件大小
            try:
                import os
                db_path = getattr(self, '_db_path', None)
                db_size_mb = 0.0
                if db_path and os.path.exists(db_path):
                    db_size_mb = os.path.getsize(db_path) / (1024 * 1024)
            except Exception:
                db_size_mb = 0.0

            return {
                'total_entries': total,
                'active_entries': active,
                'expired_entries': expired,
                'total_accesses': total_accesses,
                'db_size_mb': round(db_size_mb, 2),
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
