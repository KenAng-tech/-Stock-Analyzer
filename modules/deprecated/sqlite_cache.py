#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
SQLite 缓存模块

P3 优化 (2026-07-01):
  将高频查询结果持久化到 SQLite，减少重复 API 调用和计算。

功能:
  1. K 线数据缓存（按股票代码 + 周期）
  2. 技术指标缓存
  3. ML 预测结果缓存
  4. 自动过期和清理
"""

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta

from modules.logger import logger


class SQLiteCache:
    """SQLite 缓存后端"""

    def __init__(self, db_path: Optional[str] = None, default_ttl: int = 300):
        """
        Args:
            db_path: SQLite 数据库路径
            default_ttl: 默认缓存过期时间（秒）
        """
        if db_path is None:
            db_path = str(Path(__file__).parent.parent / 'cache.db')
        self._db_path = db_path
        self._default_ttl = default_ttl
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None

        # 初始化数据库
        self._init_db()
        logger.info(f"[SQLiteCache] 初始化: db={db_path}, default_ttl={default_ttl}s")

    def _get_connection(self) -> sqlite3.Connection:
        """获取数据库连接（复用连接，避免 :memory: 数据库每次创建新表）"""
        if self._conn is None:
            self._conn = sqlite3.connect(self._db_path, timeout=10)
            self._conn.execute("PRAGMA journal_mode=WAL")  # 提高并发性能
            self._conn.execute("PRAGMA busy_timeout=5000")
        return self._conn

    def _init_db(self):
        """初始化数据库表"""
        with self._lock:
            conn = self._get_connection()
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    category TEXT NOT NULL DEFAULT 'default',
                    created_at REAL NOT NULL,
                    ttl INTEGER NOT NULL,
                    access_count INTEGER NOT NULL DEFAULT 0,
                    last_access REAL NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_category ON cache(category);
                CREATE INDEX IF NOT EXISTS idx_created_at ON cache(created_at);
            """)
            conn.commit()

    def get(self, key: str, category: str = 'default') -> Optional[Any]:
        """获取缓存"""
        with self._lock:
            conn = self._get_connection()
            try:
                row = conn.execute(
                    "SELECT value, created_at, ttl, last_access FROM cache WHERE key = ?",
                    (key,),
                ).fetchone()
                if row is None:
                    return None

                value_str, created_at, ttl, last_access = row
                expire_at = created_at + ttl

                if time.time() > expire_at:
                    # 过期，删除
                    conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                    conn.commit()
                    return None

                # 更新访问统计
                conn.execute(
                    """UPDATE cache SET access_count = access_count + 1, last_access = ? WHERE key = ?""",
                    (time.time(), key),
                )
                conn.commit()

                return json.loads(value_str)
            except sqlite3.Error as e:
                logger.warning(f"[SQLiteCache] get 错误: {e}")
                return None

    def set(self, key: str, value: Any, category: str = 'default',
            ttl: Optional[int] = None):
        """设置缓存"""
        ttl = ttl or self._default_ttl
        with self._lock:
            conn = self._get_connection()
            value_str = json.dumps(value, default=str, ensure_ascii=False)
            now = time.time()
            conn.execute(
                """INSERT OR REPLACE INTO cache (key, value, category, created_at, ttl, access_count, last_access)
                   VALUES (?, ?, ?, ?, ?, 0, 0)""",
                (key, value_str, category, now, ttl),
            )
            conn.commit()

    def invalidate(self, key: str):
        """使缓存失效"""
        with self._lock:
            conn = self._get_connection()
            conn.execute("DELETE FROM cache WHERE key = ?", (key,))
            conn.commit()

    def invalidate_category(self, category: str):
        """使某分类所有缓存失效"""
        with self._lock:
            conn = self._get_connection()
            conn.execute("DELETE FROM cache WHERE category = ?", (category,))
            conn.commit()

    def invalidate_prefix(self, prefix: str):
        """按前缀使缓存失效"""
        with self._lock:
            conn = self._get_connection()
            conn.execute("DELETE FROM cache WHERE key LIKE ?", (f"{prefix}%",))
            conn.commit()

    def cleanup_expired(self) -> int:
        """清理过期缓存，返回清理数量"""
        with self._lock:
            conn = self._get_connection()
            now = time.time()
            cursor = conn.execute(
                "SELECT key FROM cache WHERE created_at + ttl < ?",
                (now,),
            )
            keys = [row[0] for row in cursor.fetchall()]
            if keys:
                placeholders = ','.join('?' * len(keys))
                conn.execute(f"DELETE FROM cache WHERE key IN ({placeholders})", keys)
                conn.commit()
            return len(keys)

    def get_stats(self) -> Dict:
        """获取缓存统计"""
        with self._lock:
            conn = self._get_connection()
            total = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
            by_category = conn.execute(
                "SELECT category, COUNT(*) FROM cache GROUP BY category"
            ).fetchall()

            total_accesses = conn.execute(
                "SELECT COALESCE(SUM(access_count), 0) FROM cache"
            ).fetchone()[0]

            # 过期数量
            now = time.time()
            expired = conn.execute(
                "SELECT COUNT(*) FROM cache WHERE created_at + ttl < ?",
                (now,),
            ).fetchone()[0]

            return {
                'total_entries': total,
                'expired_entries': expired,
                'active_entries': total - expired,
                'total_accesses': total_accesses,
                'by_category': dict(by_category),
                'db_path': self._db_path,
                'db_size_mb': round(
                    Path(self._db_path).stat().st_size / 1024 / 1024, 2
                ) if Path(self._db_path).exists() else 0,
            }

    def clear(self):
        """清空所有缓存"""
        with self._lock:
            conn = self._get_connection()
            conn.execute("DELETE FROM cache")
            conn.commit()

    def get_or_set(self, key: str, category: str,
                   fetcher, ttl: Optional[int] = None) -> Any:
        """获取缓存，未命中则调用 fetcher 填充"""
        result = self.get(key, category)
        if result is not None:
            return result
        data = fetcher()
        if data is not None:
            self.set(key, data, category, ttl)
        return data


# 全局单例
_cache: Optional[SQLiteCache] = None


def get_sqlite_cache() -> SQLiteCache:
    """获取 SQLite 缓存全局实例"""
    global _cache
    if _cache is None:
        _cache = SQLiteCache(default_ttl=300)
    return _cache


def reset_sqlite_cache():
    """重置全局缓存实例（用于测试）"""
    global _cache
    _cache = None
