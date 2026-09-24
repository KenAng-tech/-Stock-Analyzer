# -*- coding: utf-8 -*-
"""2026-09-21 锁链测试:
1. ths_data_provider._run_cli — 429/502/503/504 与 timeout 有界重试, 其余不重试
2. kline_data_fetcher._fetch_with_timeout — 超时/异常日志分类 (不再混标"超时")
"""
import logging
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from utils.ths_data_provider import ThsDataProvider, _run_cli
from modules.kline_data_fetcher import KlineDataFetcher


def _proc(stdout=None, returncode=0, stderr=''):
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


def _ok_envelope():
    return '{"ok": true, "data": {"price": 10.0}}'


class TestRunCliRetry(unittest.TestCase):
    """_run_cli: 可重试错误码/超时 → 重试; 其余单次; 全部耗尽 → None (非断链)"""

    def setUp(self):
        pass

    def test_ok_first_attempt_single_call(self):
        """正常响应只调用一次上游 (稳态零重试开销)"""
        with patch('utils.ths_data_provider.subprocess.run',
                   return_value=_proc(stdout=_ok_envelope())) as m:
            data = _run_cli(['snapshot', '600519.SH'], timeout=5)
        self.assertIsNotNone(data)
        self.assertTrue(data['ok'])
        self.assertEqual(m.call_count, 1)

    def test_retryable_http_429_retried_then_success(self):
        """429 首次 → 重试一次 → 第二次成功 (上游契约: 429 有界重试)"""
        seq = [_proc(returncode=1, stderr='UPSTREAM_HTTP_429: too many requests'),
               _proc(stdout=_ok_envelope())]
        with patch('utils.ths_data_provider.subprocess.run', side_effect=seq), \
             patch('utils.ths_data_provider.time.sleep'):
            data = _run_cli(['snapshot', '600519.SH'], timeout=5)
        self.assertIsNotNone(data, "429 重试链: 第二次成功应返回数据")

    def test_timeout_retried_once_then_none(self):
        """timeout → 重试 1 次仍 timeout → None + 重试日志 (链尾非静默)"""
        import subprocess as sp
        records = []

        def _slow(*a, **k):
            records.append(k.get('timeout'))
            raise sp.TimeoutExpired(cmd='hithink-finance', timeout=1)

        with patch('utils.ths_data_provider.subprocess.run', side_effect=_slow), \
             patch('utils.ths_data_provider.time.sleep'):
            result = _run_cli(['snapshot', '600519.SH'], timeout=5)
        self.assertIsNone(result)
        self.assertEqual(len(records), 2, "timeout 应重试 1 次 (共 2 次尝试)")
        self.assertEqual(records, [5, 5])

    def test_non_retryable_auth_error_single_call(self):
        """ok:false AUTH_FAILED 不可重试 → 只 1 次 (预算留给真网络抖动)"""
        envelope = ('{"ok": false, "error": {"code": "AUTH_FAILED", '
                    '"category": "auth", "hint": "bad key"}}')
        with patch('utils.ths_data_provider.subprocess.run',
                   return_value=_proc(stdout=envelope)) as m:
            with self.assertLogs('utils.ths_data_provider', level='WARNING') as logs:
                data = _run_cli(['snapshot', '600519.SH'], timeout=5)
        self.assertIsNone(data)
        self.assertEqual(m.call_count, 1, "AUTH_FAILED 非可重试类, 不得重试")
        # 错误码分类不再黑盒: 日志必须含 error.code
        self.assertIn('AUTH_FAILED', logs.output[0])

    def test_retries_zero_disables_retry(self):
        """retries=0 关闭重试 (调用方总预算受控的逃生门)"""
        with patch('utils.ths_data_provider.subprocess.run',
                   side_effect=TimeoutError('x')) as m:
            _run_cli(['snapshot', '600519.SH'], timeout=5, retries=0)
        self.assertEqual(m.call_count, 1)


class TestKlineFetchLogClass(unittest.TestCase):
    """_fetch_with_timeout: 超时=warning, 其余异常=debug (链路健康区分'上游慢'与'数据坏')"""

    def setUp(self):
        self.fetcher = KlineDataFetcher()

    def test_timeout_logged_as_warning(self):
        def _slow():
            time.sleep(0.3)
            return [1]
        with self.assertLogs('stock_analyzer', level='DEBUG') as logs:
            result = self.fetcher._fetch_with_timeout(_slow, timeout=0.05)
        self.assertIsNone(result)
        joined = '\n'.join(logs.output)
        self.assertIn('超时', joined)
        self.assertNotIn('失败', joined, "纯超时不应被标成'失败'")

    def test_worker_exception_logged_as_debug_not_warning(self):
        """子线程 ValueError (数据形状坏) ≠ 上游慢 — 分类正确才不误导排障"""
        def _broken():
            raise ValueError("shape '[1, 1, 96]' is invalid")
        with self.assertLogs('stock_analyzer', level='DEBUG') as logs:
            result = self.fetcher._fetch_with_timeout(_broken, timeout=5)
        self.assertIsNone(result)
        joined = '\n'.join(logs.output)
        self.assertIn('失败', joined)
        self.assertNotIn('超时', joined, "数据错误不得误标'超时'")

    def test_fast_path_passthrough(self):
        """正常路径透传返回值 (零开销)"""
        self.assertEqual(self.fetcher._fetch_with_timeout(lambda: [1, 2], timeout=5), [1, 2])


if __name__ == '__main__':
    unittest.main()
