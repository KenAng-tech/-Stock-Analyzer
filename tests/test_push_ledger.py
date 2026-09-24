#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
推送台账测试 — PanWatch 式「发送成功才标记」去重 (2026-09-20 整合①)

契约 (push_ledger.PushLedger):
1. pending: 文件在队列 = daemon 还在重试/排队 → 同 key 不重复推 (dedupe)
2. 文件消失 = daemon 协议「sendMessage 成功才 rmSync」→ confirmed = 真送达
3. 滞留超时 = 未送达 (daemon 断/token 坏) → failed + 重试计数, 允许补推
4. 重试耗尽 → blocked (推送链断, 升级人工)
5. manual 触发绕过 dedupe (用户「再推一次」合法需求)

运行: python -m unittest tests.test_push_ledger
"""

import json
import os
import tempfile
import time
import unittest

from modules.push_ledger import PushLedger


class _FixedClock:
    """可注入时钟, 摆脱真实 sleep"""

    def __init__(self, t0=1_000_000.0):
        self.t = t0

    def __call__(self):
        return self.t

    def advance(self, s):
        self.t += s


class TestPushLedgerDedupe(unittest.TestCase):
    """测试: 去重门 (dedupe gate)"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.queue = os.path.join(self.tmp.name, 'reply_queue')
        os.makedirs(self.queue)
        self.clock = _FixedClock()
        self.ledger = PushLedger(ledger_path=os.path.join(self.tmp.name, 'l.json'),
                                 queue_dir=self.queue, clock=self.clock)

    def tearDown(self):
        self.tmp.cleanup()

    def _put(self, fname, content='x'):
        with open(os.path.join(self.queue, fname), 'w') as f:
            f.write(content)

    def test_pending_blocks_duplicate(self):
        """文件在队列未过期 = 上一推未确认 → 同 key 自动链跳过 (防重复刷屏)"""
        self._put('r1.json')
        self.ledger.register('report:2026-09-20:abc', 'r1.json')
        allowed, reason, state = self.ledger.check('report:2026-09-20:abc')
        self.assertFalse(allowed, 'pending 中不应允许重复推')
        self.assertEqual(state, 'pending')
        self.assertIn('队列在途', reason)

    def test_confirmed_after_file_vanished(self):
        """文件被 daemon 消费 (rmSync) = Telegram 已接收 → confirmed (链尾=真送达)"""
        self._put('r1.json')
        self.ledger.register('report:2026-09-20:abc', 'r1.json')
        # daemon 5s 轮询消费成功 → 文件消失
        os.remove(os.path.join(self.queue, 'r1.json'))
        allowed, reason, state = self.ledger.check('report:2026-09-20:abc')
        self.assertEqual(state, 'confirmed', '文件消失必须判 confirmed')
        # 当日同内容再触发 (auto 重复) 仍去重: 已送达 ≠ 需要再发
        self.assertFalse(allowed, '已确认送达的内容不应自动重复推')
        self.assertIn('已送达', reason)

    def test_manual_bypasses_dedupe(self):
        """manual 触发绕过去重 (用户要「再推送一次」的合法形态)"""
        self._put('r1.json')
        self.ledger.register('report:2026-09-20:abc', 'r1.json')
        allowed, reason, state = self.ledger.check('report:2026-09-20:abc', force=True)
        self.assertTrue(allowed, 'manual 绕过 dedupe')
        # 无账本记录时 auto 也放行
        self.ledger = PushLedger(ledger_path=os.path.join(self.tmp.name, 'l2.json'),
                                 queue_dir=self.queue, clock=self.clock)
        allowed, _, state = self.ledger.check('report:2026-09-20:zzz')
        self.assertTrue(allowed)
        self.assertEqual(state, 'fresh')

    def test_unknown_key_allowed(self):
        """全新内容 (无账本) = 放行首推"""
        allowed, reason, state = self.ledger.check('report:2026-09-21:new')
        self.assertTrue(allowed)
        self.assertEqual(state, 'fresh')


class TestPushLedgerStale(unittest.TestCase):
    """测试: 未送达降级 (链尾可观测)"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.queue = os.path.join(self.tmp.name, 'reply_queue')
        os.makedirs(self.queue)
        self.clock = _FixedClock()
        self.ledger = PushLedger(ledger_path=os.path.join(self.tmp.name, 'l.json'),
                                 queue_dir=self.queue, clock=self.clock,
                                 ttl_pending=1800, max_retries=2)

    def tearDown(self):
        self.tmp.cleanup()

    def _put(self, fname):
        with open(os.path.join(self.queue, fname), 'w') as f:
            f.write('x')

    def test_stale_pending_becomes_failed(self):
        """滞留 >ttl = 未送达 → failed, 允许补推 (重试), 并计 retry"""
        self._put('r1.json')
        self.ledger.register('report:k', 'r1.json')
        self.clock.advance(1801)
        allowed, reason, state = self.ledger.check('report:k')
        self.assertTrue(allowed, '未送达必须允许补推 (降级非断链)')
        self.assertEqual(state, 'failed')
        self.assertIn('滞留', reason)
        stats = self.ledger.stats()
        self.assertEqual(stats['by_state']['failed'], 1)

    def test_retry_exhausted_blocks(self):
        """重试耗尽 = 推送链断, 必须 blocked + 可见 (不再无限补推)"""
        key = 'report:k'
        # 首推滞留 → 降级 failed, 补推放行
        self._put('r1.json')
        self.ledger.register(key, 'r1.json')
        self.clock.advance(1801)
        allowed, _, state = self.ledger.check(key)
        self.assertTrue(allowed, '第一次滞留应允许补推')
        self.assertEqual(state, 'failed')
        # 补推 #1 又滞留 → max_retries=2 耗尽 → 链断拦截
        self._put('r2.json')
        self.ledger.register(key, 'r2.json')
        self.clock.advance(1801)
        allowed, reason, state = self.ledger.check(key)
        self.assertFalse(allowed, '重试耗尽必须拦截')
        self.assertEqual(state, 'blocked')
        self.assertIn('链断', reason)

    def test_persistence_roundtrip(self):
        """台账落盘: 新实例读旧账 (5002 重启不丢台账)"""
        self._put('r1.json')
        self.ledger.register('report:k', 'r1.json')
        l2 = PushLedger(ledger_path=os.path.join(self.tmp.name, 'l.json'),
                        queue_dir=self.queue, clock=self.clock)
        allowed, _, state = l2.check('report:k')
        self.assertFalse(allowed, '重启后 dedupe 依然生效 (独立事实来源 = 磁盘文件)')
        self.assertEqual(state, 'pending')

    def test_stats_shape(self):
        """health 观测面: stats 形状 (T27 trace 链的基础)"""
        self._put('a.json')
        self.ledger.register('report:a', 'a.json')
        s = self.ledger.stats()
        self.assertEqual(s['by_state']['pending'], 1)
        self.assertEqual(s['pending'], 1)


if __name__ == '__main__':
    unittest.main()
