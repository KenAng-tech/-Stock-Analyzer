"""
test_ths_provider_cache.py — ThsDataProvider 缓存机制测试

验证:
- _cached() 命中时不重复调用 fetch_fn
- _cached() 过期后重新调用 fetch_fn
- clear_cache() 清除所有缓存
- 单例 get_ths_data_provider() 返回同一实例
"""

import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta


class TestThsProviderCache(unittest.TestCase):
    """ThsDataProvider 缓存机制 — 不依赖 CLI mock"""

    def setUp(self):
        """导入待测模块，mock 掉 CLI 解析"""
        self._cli_patcher = patch(
            "utils.ths_data_provider._resolve_cli",
            return_value="/fake/path/hithink-finance",
        )
        self._cli_patcher.start()

    def tearDown(self):
        self._cli_patcher.stop()
        # 重置单例
        import utils.ths_data_provider as mod
        mod._provider = None

    def test_cached_returns_data_on_first_call(self):
        """第一次调用 _cached 应返回 fetch_fn 的结果"""
        from utils.ths_data_provider import ThsDataProvider

        provider = ThsDataProvider()
        result = provider._cached("key1", lambda: {"value": 42})

        self.assertEqual(result, {"value": 42})

    def test_cached_does_not_recall_fn_on_hit(self):
        """缓存命中时不应重复调用 fetch_fn"""
        from utils.ths_data_provider import ThsDataProvider

        provider = ThsDataProvider()
        call_count = 0

        def counting_fn():
            nonlocal call_count
            call_count += 1
            return {"count": call_count}

        # 第一次调用
        r1 = provider._cached("key_count", counting_fn)
        # 第二次调用 (缓存命中)
        r2 = provider._cached("key_count", counting_fn)

        self.assertEqual(r1, {"count": 1})
        self.assertEqual(r2, {"count": 1})
        self.assertEqual(call_count, 1, "fetch_fn 应只被调用一次")

    def test_cached_expired_recalls_fn(self):
        """缓存过期后应重新调用 fetch_fn"""
        from utils.ths_data_provider import ThsDataProvider

        provider = ThsDataProvider()
        call_count = 0

        def counting_fn():
            nonlocal call_count
            call_count += 1
            return {"version": call_count}

        # 第一次调用 (ttl=0 立即过期)
        r1 = provider._cached("key_expire", counting_fn, ttl=0)
        # 第二次调用 (应过期)
        r2 = provider._cached("key_expire", counting_fn, ttl=0)

        self.assertEqual(r1, {"version": 1})
        self.assertEqual(r2, {"version": 2})
        self.assertEqual(call_count, 2, "过期后 fetch_fn 应被重新调用")

    def test_clear_cache_removes_all_entries(self):
        """clear_cache() 应清除所有缓存条目"""
        from utils.ths_data_provider import ThsDataProvider

        provider = ThsDataProvider()
        provider._cached("a", lambda: "val_a")
        provider._cached("b", lambda: "val_b")

        # 验证缓存已填充
        self.assertIn("a", provider._cache)
        self.assertIn("b", provider._cache)

        provider.clear_cache()

        self.assertNotIn("a", provider._cache)
        self.assertNotIn("b", provider._cache)
        self.assertEqual(len(provider._cache), 0)

    def test_cached_returns_none_when_fn_returns_none(self):
        """fetch_fn 返回 None 时，_cached 应返回 None 且不缓存"""
        from utils.ths_data_provider import ThsDataProvider

        provider = ThsDataProvider()
        call_count = 0

        def none_fn():
            nonlocal call_count
            call_count += 1
            return None

        r1 = provider._cached("key_none", none_fn)
        r2 = provider._cached("key_none", none_fn)

        self.assertIsNone(r1)
        self.assertIsNone(r2)
        # None 不应被缓存，所以每次都会调用
        self.assertEqual(call_count, 2, "None 结果不应被缓存")

    def test_singleton_returns_same_instance(self):
        """get_ths_data_provider() 应返回同一单例"""
        from utils.ths_data_provider import get_ths_data_provider

        a = get_ths_data_provider()
        b = get_ths_data_provider()

        self.assertIs(a, b, "单例应返回同一对象")


if __name__ == "__main__":
    unittest.main()
