#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
链式追踪测试 — trace_id 全链追踪 (2026-09-20 整合②, PanWatch 式)

契约 (modules/log_context.py + StructuredFormatter):
1. trace_id 注入 JSON 日志行 (仅在活跃时注入, 无 trace 不污染)
2. 线程隔离: threading.Thread 独立 context, 主线程 set 不泄漏到子线程
3. gen_trace_id 形态: tr-{CST日期}-{trigger}-{8hex} (可 grep 拼接)
4. 链尾拼接 = grep trace_id 跨模块串 (生成→推送 → 同 trace)

运行: python -m unittest tests.test_log_context
"""

import json
import logging
import threading
import unittest

from modules.log_context import (clear_trace, gen_trace_id, get_trace,
                                 set_trace, trace_of)
from modules.logger import StructuredFormatter


class TestTraceContext(unittest.TestCase):
    """测试: trace 上下文 (ContextVar 线程隔离)"""

    def test_set_get_clear(self):
        """set/get/clear 基本链"""
        self.assertEqual(get_trace(), '')  # 默认空
        set_trace('tr-test-001')
        self.assertEqual(get_trace(), 'tr-test-001')
        clear_trace()
        self.assertEqual(get_trace(), '')

    def test_thread_isolation(self):
        """子线程 set 不污染主线程 (5002 = Werkzeug threading 非 asyncio)"""
        set_trace('main-trace')
        seen = {}

        def worker():
            clear_trace()  # 子线程新 context, get 应为默认空
            seen['before'] = get_trace()
            set_trace('worker-trace')
            seen['after'] = get_trace()

        t = threading.Thread(target=worker)
        t.start()
        t.join(timeout=5)
        self.assertEqual(seen['before'], '', '子线程必须独立 context, 不继承主线程')
        self.assertEqual(seen['after'], 'worker-trace')
        self.assertEqual(get_trace(), 'main-trace', '主线程 trace 不受子线程影响')
        clear_trace()

    def test_gen_trace_id_shape(self):
        """生成形态可 grep 拼接: tr-YYYY-MM-DD-trigger-xxxxxxxx (8 hex)"""
        tid = gen_trace_id('auto', 'sz300620')
        self.assertRegex(tid, r'^tr-\d{4}-\d{2}-\d{2}-auto-sz300620-[0-9a-f]{8}$')

    def test_trace_of(self):
        """trace_of = 观测面 (health/链尾核对入口)"""
        self.assertIsNone(trace_of())
        set_trace('tr-x')
        self.assertEqual(trace_of(), 'tr-x')
        clear_trace()


class TestThreadPropagation(unittest.TestCase):
    """测试: with_trace 跨线程传递 (submit 段链不断: intel 3-worker/kline/fetcher)"""

    def test_submit_chain_propagates(self):
        """executor submit 包装后, worker 线程日志 = 同一 trace (跨 ContextVar)"""
        from concurrent.futures import ThreadPoolExecutor
        from modules.log_context import with_trace, get_trace
        set_trace('tr-req-sz300620-abcd1234')
        try:
            with ThreadPoolExecutor(max_workers=3, thread_name_prefix='intel') as ex:
                f1 = ex.submit(with_trace(lambda: get_trace()))
                f2 = ex.submit(get_trace)  # 未包装 = 断链基线
                tid_wrapped, tid_plain = f1.result(timeout=5), f2.result(timeout=5)
            self.assertEqual(tid_wrapped, 'tr-req-sz300620-abcd1234',
                             'with_trace 必须把请求 trace 带进 worker 线程')
            self.assertEqual(tid_plain, '', '未包装 submit = ContextVar 默认空 (断链对照)')
        finally:
            clear_trace()

    def test_with_trace_wrapper_reusable(self):
        """wrapper = 可复用的跨线程信封 (不改业务返回值/入参)"""
        from modules.log_context import with_trace, get_trace

        def job(x, y):
            return x + y, get_trace()

        set_trace('tr-job')
        try:
            r1, tid1 = with_trace(job)(1, 2)
            self.assertEqual((r1, tid1), (3, 'tr-job'))
        finally:
            clear_trace()


class TestFormatterInject(unittest.TestCase):
    """测试: JSON 日志行注入 (链尾可观测核心)"""

    def setUp(self):
        self.f = StructuredFormatter()
        self.logger = logging.getLogger('stock_analyzer.test_logctx')

    def _emit(self, msg):
        rec = self.logger.makeRecord(self.logger.name, logging.INFO, '(t)', 0, msg, (), None)
        return json.loads(self.f.format(rec))

    def test_no_trace_not_polluted(self):
        """无活跃 trace = JSON 无 trace 字段 (不污染非链式日志)"""
        clear_trace()
        line = self._emit('普通日志')
        self.assertNotIn('trace', line)
        self.assertEqual(line['message'], '普通日志')

    def test_trace_injected(self):
        """活跃 trace 注入每行 → 同 trace 多行 = 一 grep 拼全链"""
        set_trace('tr-2026-09-20-manual-e2e')
        try:
            line = self._emit('链尾: Telegram 送达 msg_id=256')
            self.assertEqual(line['trace'], 'tr-2026-09-20-manual-e2e')
        finally:
            clear_trace()


if __name__ == '__main__':
    unittest.main()
