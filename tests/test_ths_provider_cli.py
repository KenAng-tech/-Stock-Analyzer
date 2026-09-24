"""
test_ths_provider_cli.py — ThsDataProvider CLI 调用测试

验证:
- _run_cli 正确调用 CLI 并解析 JSON
- CLI 返回非零码时返回 None
- CLI 超时/JSON 解析失败时返回 None
- get_klines 正确处理 list 和 dict 两种 data 格式
- get_realtime 正确提取字段
"""

import unittest
import json
from unittest.mock import patch, MagicMock, call
from datetime import datetime


class TestThsDataProviderCLI(unittest.TestCase):
    """ThsDataProvider CLI 调用 — mock subprocess.run"""

    def _make_run_result(self, ok=True, data=None, error=None, returncode=0, stderr=""):
        """构造 subprocess.run 的返回对象"""
        response = {"ok": ok}
        if ok:
            response["data"] = data if data is not None else {}
            response["meta"] = {"request_id": "test-req-id"}
        else:
            response["error"] = error or "unknown error"
        return MagicMock(
            returncode=returncode,
            stdout=json.dumps(response),
            stderr=stderr,
        )

    @patch("utils.ths_data_provider._resolve_cli")
    def test_run_cli_success(self, mock_resolve):
        """_run_cli 成功时应返回解析后的数据"""
        mock_resolve.return_value = "/fake/cli"
        from utils.ths_data_provider import _run_cli

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = self._make_run_result(
                ok=True, data={"item": [{"price": 100}]}
            )
            result = _run_cli(["market", "snapshot"])

        self.assertEqual(result["ok"], True)
        self.assertEqual(result["data"]["item"][0]["price"], 100)
        mock_run.assert_called_once()

    @patch("utils.ths_data_provider._resolve_cli")
    def test_run_cli_nonzero_returncode(self, mock_resolve):
        """CLI 返回非零码时应返回 None"""
        mock_resolve.return_value = "/fake/cli"
        from utils.ths_data_provider import _run_cli

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = self._make_run_result(
                ok=False, error="not found", returncode=1
            )
            result = _run_cli(["market", "snapshot"])

        self.assertIsNone(result)

    @patch("utils.ths_data_provider._resolve_cli")
    def test_run_cli_timeout(self, mock_resolve):
        """CLI 超时应返回 None"""
        mock_resolve.return_value = "/fake/cli"
        from utils.ths_data_provider import _run_cli, subprocess

        with patch.object(subprocess, "run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd=["fake"], timeout=30)
            result = _run_cli(["market", "snapshot"])

        self.assertIsNone(result)

    @patch("utils.ths_data_provider._resolve_cli")
    def test_run_cli_bad_json(self, mock_resolve):
        """CLI 返回无效 JSON 时应返回 None"""
        mock_resolve.return_value = "/fake/cli"
        from utils.ths_data_provider import _run_cli, subprocess

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="not json at all",
                stderr="",
            )
            result = _run_cli(["market", "snapshot"])

        self.assertIsNone(result)

    @patch("utils.ths_data_provider._resolve_cli")
    def test_get_klines_remote_list_format(self, mock_resolve):
        """get_klines 应正确处理 data 为 list 的响应"""
        mock_resolve.return_value = "/fake/cli"
        from utils.ths_data_provider import ThsDataProvider

        provider = ThsDataProvider()

        with patch("utils.ths_data_provider._run_cli") as mock_run:
            mock_run.return_value = {
                "ok": True,
                "data": [
                    {"date": "2025-01-01", "open": 10, "close": 11, "high": 12, "low": 9, "volume": 1000},
                    {"date": "2025-01-02", "open": 11, "close": 12, "high": 13, "low": 10, "volume": 1200},
                ],
            }
            result = provider._get_klines_remote("300620.SZ", start="2025-01-01", end="2025-01-31")

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["close"], 11)

    @patch("utils.ths_data_provider._resolve_cli")
    def test_get_klines_remote_dict_format(self, mock_resolve):
        """get_klines 应正确处理 data 为 dict (含 items) 的响应"""
        mock_resolve.return_value = "/fake/cli"
        from utils.ths_data_provider import ThsDataProvider

        provider = ThsDataProvider()

        with patch("utils.ths_data_provider._run_cli") as mock_run:
            mock_run.return_value = {
                "ok": True,
                "data": {"items": [{"date": "2025-01-01", "close": 100}]},
            }
            result = provider._get_klines_remote("300620.SZ")

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["close"], 100)

    @patch("utils.ths_data_provider._resolve_cli")
    def test_get_klines_local_duckdb_hit_and_fallback(self, mock_resolve):
        """本地 DuckDB 实装 (2026-09-19 主链): 有库 → qfq K线 (volume 单位=股);
        无库/坏库 → None 诚实降级 (remote fallback 接管), 绝不断链。

        (09-17 写本测试时链为死链恒 None, 09-19 DuckDB 主链复活后锁新形;
        mock duckdb.connect — 不依赖真实 market.duckdb 文件与机器状态)
        """
        mock_resolve.return_value = "/fake/cli"
        from utils.ths_data_provider import ThsDataProvider
        import duckdb

        provider = ThsDataProvider()
        rows = [
            ("300620.SZ", "2025-01-02", 10.0, 12.0, 9.0, 11.0, 9.8, 1e6, 1.1e7),
            ("300620.SZ", "2025-01-03", 11.0, 12.5, 10.5, 12.0, 11.0, 1.2e6, 1.4e7),
        ]
        fake_conn = MagicMock()
        fake_conn.execute.return_value.fetchall.return_value = rows

        # 形 1: 有库 → 真形 K线 (单位契约: volume=股 无 ÷100, 同 snapshot 链相反)
        with patch.object(duckdb, "connect", return_value=fake_conn), \
             patch("utils.ths_data_provider.os.path.isfile", return_value=True):
            result = provider._get_klines_local("300620.SZ")
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["close"], 11.0)
        self.assertEqual(result[0]["volume"], 1e6)
        self.assertIn("prev_close", result[0])

        # 形 2: 无库文件 → None 诚实空态
        with patch("utils.ths_data_provider.os.path.isfile", return_value=False):
            self.assertIsNone(provider._get_klines_local("300620.SZ"))

        # 形 3: 库连接坏 (锁超时/文件损坏) → None 降级非断链
        with patch("utils.ths_data_provider.os.path.isfile", return_value=True), \
             patch.object(duckdb, "connect", side_effect=RuntimeError("db lock")):
            self.assertIsNone(provider._get_klines_local("300620.SZ"))

    @patch("utils.ths_data_provider._resolve_cli")
    def test_get_realtime_parses_fields(self, mock_resolve):
        """get_realtime 应正确解析返回字段"""
        mock_resolve.return_value = "/fake/cli"
        from utils.ths_data_provider import ThsDataProvider

        provider = ThsDataProvider()

        with patch("utils.ths_data_provider._run_cli") as mock_run:
            mock_run.return_value = {
                "ok": True,
                "data": {"item": [{"ticker": "300620", "last_price": 185.5, "price_change": 3.2}]},
                "meta": {"request_id": "req-123"},
            }
            result = provider.get_realtime("300620.SZ")

        self.assertIsNotNone(result)
        self.assertEqual(result["last_price"], 185.5)
        self.assertEqual(result["ticker"], "300620")
        self.assertIn("name", result)

    @patch("utils.ths_data_provider._resolve_cli")
    def test_get_valuation_parses_fields(self, mock_resolve):
        """get_valuation 应正确解析估值字段"""
        mock_resolve.return_value = "/fake/cli"
        from utils.ths_data_provider import ThsDataProvider

        provider = ThsDataProvider()

        with patch("utils.ths_data_provider._run_cli") as mock_run:
            mock_run.return_value = {
                "ok": True,
                "data": {"item": [{"pe_ttm": 35.2, "pb_mrq": 8.1, "ps_ttm": 5.3}]},
            }
            result = provider.get_valuation("300620.SZ")

        self.assertIsNotNone(result)
        self.assertEqual(result["pe_ttm"], 35.2)
        self.assertEqual(result["pb_mrq"], 8.1)

    @patch("utils.ths_data_provider._resolve_cli")
    def test_resolve_symbol_parses_results(self, mock_resolve):
        """resolve_symbol 应正确解析搜索结果"""
        mock_resolve.return_value = "/fake/cli"
        from utils.ths_data_provider import ThsDataProvider

        provider = ThsDataProvider()

        with patch("utils.ths_data_provider._run_cli") as mock_run:
            mock_run.return_value = {
                "ok": True,
                "data": {"item": [
                    {"thscode": "300620.SZ", "ticker": "300620", "name": "宁德时代", "exchange": "SZ"},
                    {"thscode": "300620.HK", "ticker": "300620", "name": "宁德时代-SZ", "exchange": "HK"},
                ]},
            }
            result = provider.resolve_symbol("宁德时代", limit=5)

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["name"], "宁德时代")

    @patch("utils.ths_data_provider._resolve_cli")
    def test_get_income_statement_parses_items(self, mock_resolve):
        """get_income_statement 应正确解析利润表"""
        mock_resolve.return_value = "/fake/cli"
        from utils.ths_data_provider import ThsDataProvider

        provider = ThsDataProvider()

        with patch("utils.ths_data_provider._run_cli") as mock_run:
            mock_run.return_value = {
                "ok": True,
                "data": {"items": [
                    {"period": "2024Q3", "revenue": 1000, "net_profit": 100},
                    {"period": "2024Q2", "revenue": 950, "net_profit": 95},
                ]},
            }
            result = provider.get_income_statement("300620.SZ", limit=4)

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["revenue"], 1000)

    @patch("utils.ths_data_provider._resolve_cli")
    def test_sync_data_returns_true_on_success(self, mock_resolve):
        """sync_data 成功时应返回 True"""
        mock_resolve.return_value = "/fake/cli"
        from utils.ths_data_provider import ThsDataProvider

        provider = ThsDataProvider()

        with patch("utils.ths_data_provider._run_cli") as mock_run:
            mock_run.return_value = {"ok": True, "data": {"rows_synced": 10000}}
            result = provider.sync_data(start="2025-01-01", end="2025-06-30")

        self.assertTrue(result)

    @patch("utils.ths_data_provider._resolve_cli")
    def test_sync_data_returns_false_on_failure(self, mock_resolve):
        """sync_data 失败时应返回 False"""
        mock_resolve.return_value = "/fake/cli"
        from utils.ths_data_provider import ThsDataProvider

        provider = ThsDataProvider()

        with patch("utils.ths_data_provider._run_cli") as mock_run:
            mock_run.return_value = {"ok": False}
            result = provider.sync_data()

        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
