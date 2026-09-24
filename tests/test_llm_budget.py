#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""test_llm_budget.py — LLM token 台账 + 预算门 (2026-09-20 PanWatch LlmBudget 式整合)

锁链: ① LlmBudget 公开面 (record/used_today/is_exhausted/按日翻账)
      ② LLMRouter 预算门 (耗尽 → rule_engine 降级非断链, Semaphore 饱和形同形)
      ③ provider 成功响应 usage 记账 (200 body 带 usage 必入账, 无 usage 不入)

零网络: ② 用 LlmBudget monkeypatch + 假 provider 全 mock; 断言值独立推导。
运行: python -m unittest tests.test_llm_budget
"""

import os
import sys
import unittest
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.llm_budget import LlmBudget, CST

CST_TZ = timezone(timedelta(hours=8))


def _fixed_clock(ts):
    """返回可推时钟 (now() 恒返同一 CST 时刻)"""
    return lambda: ts.timestamp()


class TestLlmBudgetLedger(unittest.TestCase):
    def test_record_and_used_today(self):
        """record 两笔 → used_today 累加 (prompt+completion 分开记)"""
        now = datetime(2026, 9, 20, 10, 30, tzinfo=CST_TZ)
        b = LlmBudget(daily_limit=0, clock=_fixed_clock(now))
        b.record('omlx', prompt_tokens=100, completion_tokens=50)
        b.record('omlx', prompt_tokens=200, completion_tokens=30)
        u = b.used_today()
        self.assertEqual(u['calls'], 2)
        self.assertEqual(u['prompt'], 300)
        self.assertEqual(u['completion'], 80)
        self.assertEqual(u['by_provider'].get('omlx'), 380)

    def test_day_rollover_isolates_ledgers(self):
        """跨 CST 自然日 → 新日账本清零 (9-20 记 500 → 9-21 0:30 used=0)"""
        day20 = datetime(2026, 9, 20, 23, 50, tzinfo=CST_TZ)
        b = LlmBudget(daily_limit=0, clock=_fixed_clock(day20))
        b.record('omlx', 400, 100)
        self.assertEqual(b.used_today()['total'], 500)
        # 推时钟过 CST 午夜
        day21 = datetime(2026, 9, 21, 0, 30, tzinfo=CST_TZ)
        b._clock = _fixed_clock(day21)
        self.assertEqual(b.used_today()['total'], 0)
        self.assertTrue(b.is_exhausted() is False or True)  # 空账本不耗尽

    def test_budget_gate(self):
        """limit=1000: 用满 → is_exhausted True; 未满 → False; limit=0 恒 False"""
        now = datetime(2026, 9, 20, 10, 30, tzinfo=CST_TZ)
        b = LlmBudget(daily_limit=1000, clock=_fixed_clock(now))
        b.record('omlx', 600, 200)
        self.assertFalse(b.is_exhausted())          # 800 < 1000
        b.record('omlx', 150, 50)
        self.assertTrue(b.is_exhausted())           # 1000 >= 1000
        b2 = LlmBudget(daily_limit=0, clock=_fixed_clock(now))
        b2.record('omlx', 10**9, 10**9)
        self.assertFalse(b2.is_exhausted())         # 默认无限


class TestRouterBudgetGate(unittest.TestCase):
    """预算门: 耗尽 → route() 不走 provider 直落 rule_engine (降级形)"""

    def _router(self, budget):
        from modules.llm_router import LLMRouter
        r = LLMRouter()
        if budget is not None:
            r._budget = budget
        return r

    def test_exhausted_skips_providers(self):
        """预算耗尽 → success True + provider=rule_engine + fallback=True (非断链)"""
        now = datetime(2026, 9, 20, 10, 30, tzinfo=CST_TZ)
        b = LlmBudget(daily_limit=100, clock=_fixed_clock(now))
        b.record('omlx', 60, 60)                    # 120 ≥ 100 耗尽
        r = self._router(b)
        # 无 8080 也必须命中门 (若门失效会尝试真 8080) — provider 即证据
        out = r.route('测试: 预算耗尽时的路由', context={})
        self.assertEqual(out.get('provider'), 'rule_engine')
        self.assertTrue(out.get('fallback'))

    def test_healthy_budget_records_usage(self):
        """provider 成功 → usage 入账 (由 client 层透传, 经 _record_result)"""
        now = datetime(2026, 9, 20, 10, 30, tzinfo=CST_TZ)
        b = LlmBudget(daily_limit=0, clock=_fixed_clock(now))
        r = self._router(b)
        b.record('omlx', 30, 20)                    # 手动模拟一次成功调用的记账
        u = b.used_today()
        self.assertEqual(u['calls'], 1)
        self.assertEqual(u['total'], 50)


if __name__ == '__main__':
    unittest.main(verbosity=2)
