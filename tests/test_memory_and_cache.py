#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
内存管理和动态缓存单元测试

P3 测试 (2026-07-01) — 2026-09-20 清尸: SQLiteCache 模块已死 (被
modules/dynamic_cache 分级缓存整体替换, 09-17 清理), 原文件 import 死模块
→ 整文件 collection error (MemoryMonitor 活链测试被陪葬)。
TestSQLiteCache 重写为 TestDynamicCache = 同一批缓存契约 (读写/miss/覆盖/
失效/前缀失效/TTL 过期/清理/stats/get_or_set) 指向活链, 不再锁尸体。

运行: python -m unittest tests.test_memory_and_cache
"""

import sys
import os
import unittest
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.memory_manager import MemoryMonitor


class TestMemoryMonitor(unittest.TestCase):
    """测试内存监控器"""

    def test_snapshot_creation(self):
        """测试内存快照采集"""
        monitor = MemoryMonitor(
            warning_threshold_mb=1500,
            critical_threshold_mb=1800,
            auto_gc_threshold=1200,
        )
        snapshot = monitor.take_snapshot()
        self.assertGreater(snapshot.rss_mb, 0)
        self.assertGreater(snapshot.object_count, 0)
        self.assertTrue(snapshot.is_healthy)

    def test_health_check_warning(self):
        """测试警告阈值触发"""
        monitor = MemoryMonitor(
            warning_threshold_mb=0.001,  # 极低阈值，确保触发
            critical_threshold_mb=0.002,
            auto_gc_threshold=None,
        )
        snapshot = monitor.take_snapshot()
        # 不应抛出异常，只记录日志
        self.assertIsNotNone(snapshot)

    def test_force_gc(self):
        """测试强制垃圾回收"""
        monitor = MemoryMonitor(
            warning_threshold_mb=1500,
            critical_threshold_mb=1800,
            auto_gc_threshold=None,
        )
        result = monitor.force_gc()
        self.assertIn('collected', result)
        self.assertIn('freed_mb', result)
        self.assertGreaterEqual(result['collected'], 0)

    def test_get_status(self):
        """测试获取内存状态"""
        monitor = MemoryMonitor(
            warning_threshold_mb=1500,
            critical_threshold_mb=1800,
            auto_gc_threshold=None,
        )
        status = monitor.get_status()
        self.assertIn('rss_mb', status)
        self.assertIn('health', status)
        self.assertIn('leak_detection', status)
        self.assertEqual(status['health'], 'healthy')

    def test_history(self):
        """测试历史记录"""
        monitor = MemoryMonitor(
            warning_threshold_mb=1500,
            critical_threshold_mb=1800,
            auto_gc_threshold=None,
        )
        for _ in range(5):
            monitor.take_snapshot()
            time.sleep(0.01)
        history = monitor.get_history()
        self.assertEqual(len(history), 5)


class TestDynamicCache(unittest.TestCase):
    """测试 dynamic_cache (SQLiteCache 继承者, 契约锁在活链)"""

    def setUp(self):
        from modules.dynamic_cache import cache
        self.cache = cache

    def test_set_and_get(self):
        """测试基本读写 roundtrip"""
        self.cache.set('_t_dc_roundtrip', {'data': 'value'}, ttl=60)
        self.assertEqual(self.cache.get('_t_dc_roundtrip'), {'data': 'value'})

    def test_get_miss(self):
        """测试未命中 → default None (非假命中)"""
        self.assertIsNone(self.cache.get('_t_dc_never_set_key'))

    def test_overwrite(self):
        """测试覆盖写入 (新锁: 09-17 修过 ttl=0 永不过期, 覆盖形同根)"""
        self.cache.set('_t_dc_ow', 'old_value', ttl=60)
        self.cache.set('_t_dc_ow', 'new_value', ttl=60)
        self.assertEqual(self.cache.get('_t_dc_ow'), 'new_value')

    def test_invalidate(self):
        """测试单键失效"""
        self.cache.set('_t_dc_inv', 'value1', ttl=60)
        self.cache.invalidate('_t_dc_inv')
        self.assertIsNone(self.cache.get('_t_dc_inv'))

    def test_invalidate_prefix(self):
        """测试按前缀失效 (真形: invalidate_key_prefix, 非旧 invalidate_prefix)"""
        self.cache.set('_t_dc_prefix_a', 'value_a', ttl=60)
        self.cache.set('_t_dc_prefix_b', 'value_b', ttl=60)
        self.cache.set('_t_dc_other', 'value_other', ttl=60)
        self.cache.invalidate_key_prefix('_t_dc_prefix')
        self.assertIsNone(self.cache.get('_t_dc_prefix_a'))
        self.assertIsNone(self.cache.get('_t_dc_prefix_b'))
        self.assertEqual(self.cache.get('_t_dc_other'), 'value_other')

    def test_expiration(self):
        """测试 TTL 过期 (0.05s 档)"""
        self.cache.set('_t_dc_exp', 'value1', ttl=0.05)
        self.assertEqual(self.cache.get('_t_dc_exp'), 'value1')
        time.sleep(0.25)
        self.assertIsNone(self.cache.get('_t_dc_exp'))

    def test_get_or_set(self):
        """测试 get_or_set 命中复用 (fetcher 只跑一次; 真签名 key,category,fetcher)"""
        call_count = [0]

        def fetcher():
            call_count[0] += 1
            return {'fetched': True}

        r1 = self.cache.get_or_set('_t_dc_gos', 'test', fetcher)
        r2 = self.cache.get_or_set('_t_dc_gos', 'test', fetcher)
        self.assertEqual(r1, {'fetched': True})
        self.assertEqual(r2, {'fetched': True})
        self.assertEqual(call_count[0], 1)   # 第二次命中 → fetcher 不重跑

    def test_cleanup(self):
        """测试清理 (cleanup 直调非负非抛, 降级非断链)"""
        self.cache.set('_t_dc_cleanup', 'value', ttl=0.05)
        time.sleep(0.25)
        self.assertIsNotNone(self.cache.cleanup())

    def test_stats(self):
        """测试统计形 (dict + total_entries 键在, cache_routes 消费面)"""
        stats = self.cache.get_stats()
        self.assertIsInstance(stats, dict)
        for k in ('total_entries', 'active_entries', 'total_accesses'):
            self.assertIn(k, stats)


if __name__ == '__main__':
    unittest.main()
