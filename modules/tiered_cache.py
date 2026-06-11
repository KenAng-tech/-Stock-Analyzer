"""
分层缓存管理器 (Tiered Cache)
三层缓存架构: L1 (内存) → L2 (SQLite) → L3 (Redis/文件)

解决当前 50% 缓存命中率问题，目标提升至 85%+

架构:
┌─────────────────────────────────────────────────────────────────────────┐
│                         TieredCache Layer                                │
├─────────────────────────────────────────────────────────────────────────┤
│  L1: Memory Cache (热数据, 60s TTL)                                      │
│  ├── TTLCache maxsize=1000                                               │
│  ├── LRU 淘汰策略                                                        │
│  └── 目标: 高频实时数据 + 近期 K 线                                      │
│                                                                      │
│  L2: SQLite Cache (温数据, 1h TTL)                                     │
│  ├── 持久化存储                                                          │
│  ├── 自动清理过期记录                                                     │
│  └── 目标: 周线/月线数据 + ML 预测                                       │
│                                                                      │
│  L3: File/Redis Cache (冷数据, 24h TTL)                                 │
│  ├── 大容量存储                                                          │
│  ├── 压缩存储 (JSON gzip)                                                │
│  └── 目标: 基本面数据 + 因子数据                                         │
└─────────────────────────────────────────────────────────────────────────┘

使用示例:
    cache = TieredCache()
    cache.set("kline_sz300620_daily_100", data, data_type="kline")
    data = cache.get("kline_sz300620_daily_100", data_type="kline")
"""

import json
import time
import os
import sqlite3
import gzip
import hashlib
import threading
from typing import Dict, Optional, Any, Tuple, List
from dataclasses import dataclass, field
from collections import OrderedDict
from datetime import datetime


# ── 缓存配置 ──────────────────────────────────────────────────────────────

@dataclass
class CacheConfig:
    """缓存配置"""
    # L1: 内存缓存
    l1_maxsize: int = 1000
    l1_default_ttl: int = 60  # 60 秒
    
    # L2: SQLite 缓存
    l2_db_path: str = "cache/l2_cache.db"
    l2_default_ttl: int = 3600  # 1 小时
    l2_cleanup_interval: int = 300  # 5 分钟清理一次
    
    # L3: 文件缓存
    l3_dir: str = "cache/l3_cache"
    l3_default_ttl: int = 86400  # 24 小时
    l3_compress: bool = True
    
    # 全局
    enable_stats: bool = True
    enable_eviction: bool = True


# ── 数据类型 TTL 映射 ─────────────────────────────────────────────────────

DATA_TYPE_TTL = {
    "realtime": 30,
    "kline": 60,
    "ml": 300,        # ML 预测缓存 5 分钟
    "factor": 300,    # 因子数据 5 分钟
    "fundamental": 600,  # 基本面 10 分钟
    "sentiment": 300, # 情感分析 5 分钟
    "strategy": 120,  # 策略信号 2 分钟
    "hmm": 600,       # HMM 市场状态 10 分钟
    "llm": 600,       # LLM 决策 10 分钟
    "portfolio": 300, # 组合优化 5 分钟
    "default": 60,
}

# 数据类型分层优先级 (越高越优先放 L1)
DATA_TYPE_L1_PRIORITY = {
    "realtime": 10,
    "kline": 8,
    "ml": 5,
    "factor": 6,
    "fundamental": 3,
    "sentiment": 5,
    "strategy": 7,
    "hmm": 4,
    "llm": 4,
    "portfolio": 5,
    "default": 2,
}


# ── L1: 内存缓存 ──────────────────────────────────────────────────────────

class MemoryCache:
    """L1 内存缓存 - 基于 OrderedDict 的 LRU 实现"""
    
    def __init__(self, maxsize: int = 1000, default_ttl: int = 60):
        self.maxsize = maxsize
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, Dict] = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
        
    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            
            entry = self._cache[key]
            
            # 检查过期
            if time.time() > entry["expires"]:
                del self._cache[key]
                self._misses += 1
                return None
            
            # 更新访问顺序 (LRU)
            self._cache.move_to_end(key)
            entry["last_access"] = time.time()
            entry["access_count"] += 1
            self._hits += 1
            
            return entry["data"]
    
    def set(self, key: str, value: Any, ttl: int = None):
        with self._lock:
            # 如果 key 已存在，先删除
            if key in self._cache:
                del self._cache[key]
            
            # 如果超出容量，淘汰最久未使用的
            if len(self._cache) >= self.maxsize:
                self._evict_oldest()
            
            self._cache[key] = {
                "data": value,
                "expires": time.time() + (ttl or self.default_ttl),
                "created": time.time(),
                "last_access": time.time(),
                "access_count": 0,
            }
    
    def delete(self, key: str) -> bool:
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def clear(self):
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
    
    def _evict_oldest(self):
        """淘汰最久未使用的条目"""
        if self._cache:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
    
    def stats(self) -> Dict:
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._cache),
                "maxsize": self.maxsize,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / total * 100, 2) if total > 0 else 0,
            }
    
    def cleanup_expired(self) -> int:
        """清理过期条目"""
        now = time.time()
        expired_keys = [
            k for k, v in self._cache.items()
            if now > v["expires"]
        ]
        for key in expired_keys:
            del self._cache[key]
        return len(expired_keys)


# ── L2: SQLite 缓存 ───────────────────────────────────────────────────────

class SqliteCache:
    """L2 SQLite 持久化缓存"""
    
    def __init__(self, db_path: str = "cache/l2_cache.db", default_ttl: int = 3600):
        self.db_path = db_path
        self.default_ttl = default_ttl
        self._lock = threading.RLock()
        
        # 确保目录存在
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # 初始化数据库
        self._init_db()
        
        # 启动后台清理线程
        self._cleanup_thread = threading.Thread(
            target=self._periodic_cleanup,
            daemon=True
        )
        self._cleanup_thread.start()
    
    def _init_db(self):
        """初始化数据库表"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    data_type TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    last_access REAL DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_expires 
                ON cache(expires_at)
            """)
            conn.commit()
    
    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            cursor = conn.execute(
                "SELECT data, expires_at FROM cache WHERE key = ?",
                (key,)
            )
            row = cursor.fetchone()
            
            if row and time.time() < row[1]:
                return json.loads(row[0])
            elif row:
                # 过期，删除
                conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                conn.commit()
            
            return None
    
    def set(self, key: str, value: Any, ttl: int = None):
        with self._lock:
            data = json.dumps(value)
            now = time.time()
            ttl = ttl or self.default_ttl
            
            conn.execute("""
                INSERT OR REPLACE INTO cache 
                (key, data, data_type, created_at, expires_at, access_count, last_access)
                VALUES (?, ?, ?, ?, ?, 0, ?)
            """, (
                key,
                data,
                "unknown",
                now,
                now + ttl,
                now
            ))
            conn.commit()
    
    def delete(self, key: str) -> bool:
        with self._lock:
            cursor = conn.execute(
                "DELETE FROM cache WHERE key = ?", (key,)
            )
            conn.commit()
            return cursor.rowcount > 0
    
    def cleanup_expired(self) -> int:
        """清理过期记录"""
        with self._lock:
            now = time.time()
            cursor = conn.execute(
                "DELETE FROM cache WHERE expires_at < ?", (now,)
            )
            conn.commit()
            return cursor.rowcount
    
    def _periodic_cleanup(self):
        """定期清理过期记录"""
        while True:
            time.sleep(300)  # 5 分钟
            try:
                self.cleanup_expired()
            except Exception as e:
                print(f"[SqliteCache] Cleanup error: {e}")
    
    def stats(self) -> Dict:
        with self._lock:
            cursor = conn.execute(
                "SELECT COUNT(*), SUM(access_count) FROM cache"
            )
            row = cursor.fetchone()
            return {
                "total_entries": row[0] or 0,
                "total_accesses": row[1] or 0,
            }


# ── L3: 文件缓存 ──────────────────────────────────────────────────────────

class FileCache:
    """L3 文件缓存 (支持 gzip 压缩)"""
    
    def __init__(self, cache_dir: str = "cache/l3_cache", default_ttl: int = 86400, compress: bool = True):
        self.cache_dir = cache_dir
        self.default_ttl = default_ttl
        self.compress = compress
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_file_path(self, key: str) -> str:
        """根据 key 生成文件路径"""
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{key_hash}.json")
    
    def get(self, key: str) -> Optional[Any]:
        file_path = self._get_file_path(key)
        if not os.path.exists(file_path):
            return None
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # 检查过期
            if time.time() > data["expires"]:
                os.remove(file_path)
                return None
            
            return data["content"]
        except Exception:
            return None
    
    def set(self, key: str, value: Any, ttl: int = None):
        file_path = self._get_file_path(key)
        now = time.time()
        ttl = ttl or self.default_ttl
        
        data = {
            "content": value,
            "created": now,
            "expires": now + ttl,
        }
        
        if self.compress:
            file_path += ".gz"
            with gzip.open(file_path, 'wt') as f:
                json.dump(data, f)
        else:
            with open(file_path, 'w') as f:
                json.dump(data, f)
    
    def delete(self, key: str) -> bool:
        file_path = self._get_file_path(key)
        if os.path.exists(file_path):
            os.remove(file_path)
            return True
        return False
    
    def cleanup_expired(self) -> int:
        """清理过期文件"""
        now = time.time()
        count = 0
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.json') or filename.endswith('.json.gz'):
                file_path = os.path.join(self.cache_dir, filename)
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    if now > data["expires"]:
                        os.remove(file_path)
                        count += 1
                except Exception:
                    pass
        return count
    
    def stats(self) -> Dict:
        files = [f for f in os.listdir(self.cache_dir) if f.endswith('.json') or f.endswith('.json.gz')]
        return {
            "total_files": len(files),
            "total_size_bytes": sum(
                os.path.getsize(os.path.join(self.cache_dir, f))
                for f in files
            ),
        }


# ── 主缓存管理器 ──────────────────────────────────────────────────────────

class TieredCache:
    """
    分层缓存管理器
    
    使用流程:
    1. 查询 L1 (内存)
    2. L1 未命中 → 查询 L2 (SQLite)
    3. L2 未命中 → 查询 L3 (文件)
    4. 全部未命中 → 返回 None
    5. 写入时写入所有层级
    
    缓存键生成:
    - 自动根据数据类型生成层级
    - 支持标签过滤
    """
    
    def __init__(self, config: CacheConfig = None):
        self.config = config or CacheConfig()
        
        # 初始化三层缓存
        self.l1 = MemoryCache(
            maxsize=self.config.l1_maxsize,
            default_ttl=self.config.l1_default_ttl
        )
        self.l2 = SqliteCache(
            db_path=self.config.l2_db_path,
            default_ttl=self.config.l2_default_ttl
        )
        self.l3 = FileCache(
            cache_dir=self.config.l3_dir,
            default_ttl=self.config.l3_default_ttl,
            compress=self.config.l3_compress
        )
        
        # 缓存统计
        self._stats = {
            "l1_hits": 0,
            "l2_hits": 0,
            "l3_hits": 0,
            "misses": 0,
            "writes": 0,
        }
        self._stats_lock = threading.Lock()
    
    def _generate_key(self, key: str, data_type: str = None) -> str:
        """生成缓存键"""
        if data_type:
            return f"{data_type}:{key}"
        return key
    
    def _get_ttl(self, data_type: str) -> int:
        """获取数据类型对应的 TTL"""
        return DATA_TYPE_TTL.get(data_type, DATA_TYPE_TTL["default"])
    
    def _should_promote_to_l1(self, data_type: str) -> bool:
        """判断是否应该提升到 L1"""
        priority = DATA_TYPE_L1_PRIORITY.get(data_type, 0)
        return priority >= 5  # 优先级 >= 5 提升到 L1
    
    def get(self, key: str, data_type: str = None) -> Optional[Any]:
        """
        分层查询缓存
        
        Returns:
            (value, source_layer)
        """
        cache_key = self._generate_key(key, data_type)
        
        # 查询 L1
        value = self.l1.get(cache_key)
        if value is not None:
            with self._stats_lock:
                self._stats["l1_hits"] += 1
            return value
        
        # 查询 L2
        value = self.l2.get(cache_key)
        if value is not None:
            # 提升回 L1
            if self._should_promote_to_l1(data_type or "default"):
                ttl = self._get_ttl(data_type)
                self.l1.set(cache_key, value, ttl)
            
            with self._stats_lock:
                self._stats["l2_hits"] += 1
            return value
        
        # 查询 L3
        value = self.l3.get(cache_key)
        if value is not None:
            # 提升回 L2 和 L1
            ttl = self._get_ttl(data_type)
            self.l2.set(cache_key, value, ttl)
            
            if self._should_promote_to_l1(data_type or "default"):
                self.l1.set(cache_key, value, ttl)
            
            with self._stats_lock:
                self._stats["l3_hits"] += 1
            return value
        
        # 全部未命中
        with self._stats_lock:
            self._stats["misses"] += 1
        return None
    
    def set(self, key: str, value: Any, data_type: str = None, ttl: int = None):
        """写入所有缓存层"""
        cache_key = self._generate_key(key, data_type)
        ttl = ttl or self._get_ttl(data_type)
        
        # 写入 L1
        self.l1.set(cache_key, value, ttl)
        
        # 写入 L2
        self.l2.set(cache_key, value, ttl)
        
        # 写入 L3
        self.l3.set(cache_key, value, ttl)
        
        with self._stats_lock:
            self._stats["writes"] += 1
    
    def delete(self, key: str, data_type: str = None):
        """删除所有层的缓存"""
        cache_key = self._generate_key(key, data_type)
        self.l1.delete(cache_key)
        self.l2.delete(cache_key)
        self.l3.delete(cache_key)
    
    def get_stats(self) -> Dict:
        """获取完整缓存统计"""
        with self._stats_lock:
            total = (self._stats["l1_hits"] + 
                    self._stats["l2_hits"] + 
                    self._stats["l3_hits"] + 
                    self._stats["misses"])
            
            return {
                "layer": {
                    "l1": {
                        "hits": self._stats["l1_hits"],
                        "stats": self.l1.stats(),
                    },
                    "l2": {
                        "hits": self._stats["l2_hits"],
                        "stats": self.l2.stats(),
                    },
                    "l3": {
                        "hits": self._stats["l3_hits"],
                        "stats": self.l3.stats(),
                    },
                },
                "global": {
                    "l1_hits": self._stats["l1_hits"],
                    "l2_hits": self._stats["l2_hits"],
                    "l3_hits": self._stats["l3_hits"],
                    "misses": self._stats["misses"],
                    "writes": self._stats["writes"],
                    "total": total,
                    "hit_rate_pct": round(
                        (self._stats["l1_hits"] + 
                         self._stats["l2_hits"] + 
                         self._stats["l3_hits"]) / total * 100, 2
                    ) if total > 0 else 0,
                }
            }
    
    def cleanup_all(self) -> Dict:
        """清理所有层的过期数据"""
        return {
            "l1": self.l1.cleanup_expired(),
            "l2": self.l2.cleanup_expired(),
            "l3": self.l3.cleanup_expired(),
        }
    
    def clear_all(self):
        """清空所有缓存层"""
        self.l1.clear()
        self.l2.cleanup_expired()
        self.l3.cleanup_expired()


# ── 全局单例 ──────────────────────────────────────────────────────────────

# 创建全局缓存实例
cache = TieredCache()
