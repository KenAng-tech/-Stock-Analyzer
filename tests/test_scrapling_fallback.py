"""
test_scrapling_fallback.py — ScraplingFallback 测试

验证:
- _get_session 懒创建 + 空闲超时自动重建
- eastmoney_quote 正确解析行情数据
- eastmoney_guba_news 正确解析帖子列表
- cleanup 关闭 session
"""

import unittest
from unittest.mock import patch, MagicMock, PropertyMock
from datetime import datetime, timedelta
import sys


class TestScraplingFallback(unittest.TestCase):
    """ScraplingFallback — mock scrapling session"""

    def _make_mock_session(self):
        """构造一个 mock scrapling session 对象"""
        session = MagicMock()
        session.fetch.return_value = {"data": {}}
        return session

    def setUp(self):
        """清除模块级单例和缓存的 scrapling 引用"""
        import utils.scrapling_fallback as mod
        mod._fallback = None
        # 清除 scrapling 相关缓存
        for key in list(sys.modules.keys()):
            if key.startswith("scrapling"):
                del sys.modules[key]

    def tearDown(self):
        import utils.scrapling_fallback as mod
        mod._fallback = None

    def test_get_session_lazy_creation(self):
        """_get_session 应在首次调用时创建 session"""
        # 构造一个模拟的 scrapling 模块
        mock_scrapling = MagicMock()
        mock_session_obj = MagicMock()
        mock_scrapling.open_session.return_value = mock_session_obj

        with patch.dict(sys.modules, {"scrapling": mock_scrapling}):
            from utils.scrapling_fallback import ScraplingFallback

            fallback = ScraplingFallback()
            self.assertIsNone(fallback._session)

            session = fallback._get_session()

            self.assertIsNotNone(session)
            self.assertIs(session, mock_session_obj)
            mock_scrapling.open_session.assert_called_once_with(session_type="dynamic")

    def test_get_session_reuses_existing(self):
        """已有 session 时应复用，不重复创建"""
        mock_scrapling = MagicMock()
        mock_session = self._make_mock_session()

        with patch.dict(sys.modules, {"scrapling": mock_scrapling}):
            from utils.scrapling_fallback import ScraplingFallback

            fallback = ScraplingFallback()
            fallback._session = mock_session
            fallback._last_activity = datetime.now()

            session = fallback._get_session()

            self.assertIs(session, mock_session)
            mock_scrapling.open_session.assert_not_called()

    def test_get_session_recreates_after_timeout(self):
        """session 空闲超时后应自动重建"""
        from utils.scrapling_fallback import ScraplingFallback, SESSION_IDLE_TIMEOUT

        mock_scrapling = MagicMock()
        old_session = self._make_mock_session()
        new_session = self._make_mock_session()
        mock_scrapling.open_session.return_value = new_session

        with patch.dict(sys.modules, {"scrapling": mock_scrapling}):
            fallback = ScraplingFallback()
            fallback._session = old_session
            # 模拟超过 30 分钟未活动
            fallback._last_activity = datetime.now() - timedelta(seconds=SESSION_IDLE_TIMEOUT + 60)

            session = fallback._get_session()

            self.assertIs(session, new_session)
            mock_scrapling.open_session.assert_called_once()
            old_session.close.assert_called_once()

    def test_eastmoney_quote_parses_response(self):
        """eastmoney_quote 应正确解析东方财富行情响应"""
        from utils.scrapling_fallback import ScraplingFallback

        fallback = ScraplingFallback()
        mock_session = self._make_mock_session()
        fallback._session = mock_session
        fallback._last_activity = datetime.now()

        mock_result = {
            "name": "宁德时代",
            "f43": 18550,    # 价格 (分)
            "f44": 320,      # 涨跌额 (分)
            "f45": 1.75,     # 涨跌幅
            "f46": 18800,    # 最高
            "f47": 18200,    # 最低
            "f48": 18350,    # 开盘
            "f170": 1250000, # 成交量
        }
        mock_session.fetch.return_value = {"data": mock_result}

        result = fallback.eastmoney_quote("300620")

        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "宁德时代")
        self.assertEqual(result["code"], "300620")
        self.assertEqual(result["price"], 185.50)
        self.assertEqual(result["change"], 3.20)
        self.assertEqual(result["change_pct"], 0.0175)

    def test_eastmoney_quote_returns_none_on_no_session(self):
        """session 为 None 时 eastmoney_quote 应返回 None"""
        from utils.scrapling_fallback import ScraplingFallback

        fallback = ScraplingFallback()
        result = fallback.eastmoney_quote("300620")
        self.assertIsNone(result)

    def test_eastmoney_guba_news_parses_posts(self):
        """eastmoney_guba_news 应正确解析股吧帖子列表"""
        from utils.scrapling_fallback import ScraplingFallback

        fallback = ScraplingFallback()
        mock_session = self._make_mock_session()
        fallback._session = mock_session
        fallback._last_activity = datetime.now()

        markdown_content = """
# 宁德时代讨论
**【宁德时代】Q3 业绩超预期，营收同比增长 25%** [详情](https://guba.eastmoney.com/123456)
**宁德时代：机构大幅加仓，北向资金净买入** [详情](https://guba.eastmoney.com/789012)
**宁德时代新一代电池技术发布** [详情](https://guba.eastmoney.com/345678)
"""
        mock_session.fetch.return_value = {"content": markdown_content}

        result = fallback.eastmoney_guba_news("300620", limit=10)

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 3)
        self.assertIn("宁德时代", result[0]["title"])
        self.assertIn("guba.eastmoney.com", result[0]["url"])

    def test_eastmoney_guba_news_limit(self):
        """eastmoney_guba_news 应遵守 limit 参数"""
        from utils.scrapling_fallback import ScraplingFallback

        fallback = ScraplingFallback()
        mock_session = self._make_mock_session()
        fallback._session = mock_session
        fallback._last_activity = datetime.now()

        posts = "\n".join(
            f"**帖子 {i}** [链接](https://guba.eastmoney.com/{i})"
            for i in range(10)
        )
        mock_session.fetch.return_value = {"content": posts}

        result = fallback.eastmoney_guba_news("300620", limit=3)

        self.assertEqual(len(result), 3)

    def test_cleanup_closes_session(self):
        """cleanup 应关闭 session"""
        from utils.scrapling_fallback import ScraplingFallback

        fallback = ScraplingFallback()
        mock_session = self._make_mock_session()
        fallback._session = mock_session
        fallback._last_activity = datetime.now()

        fallback.cleanup()

        mock_session.close.assert_called_once()
        self.assertIsNone(fallback._session)

    def test_eastmoney_quote_600_prefix(self):
        """600 开头股票应使用正确的 market 参数 (沪市=1)"""
        from utils.scrapling_fallback import ScraplingFallback

        fallback = ScraplingFallback()
        mock_session = self._make_mock_session()
        fallback._session = mock_session
        fallback._last_activity = datetime.now()

        mock_session.fetch.return_value = {
            "data": {"f43": 5000, "f44": 0, "f45": 0, "f46": 5100, "f47": 4900, "f48": 5000, "f170": 500000}
        }

        result = fallback.eastmoney_quote("600519")

        call_args = mock_session.fetch.call_args
        url = call_args[0][0] if call_args[0] else call_args[1].get("url", "")
        self.assertIn("secid=1.600519", url)
        self.assertEqual(result["price"], 50.00)


if __name__ == "__main__":
    unittest.main()
