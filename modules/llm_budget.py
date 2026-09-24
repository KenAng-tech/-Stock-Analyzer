#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""LlmBudget — LLM token 台账 + 预算门 (2026-09-20 PanWatch 式整合)

来源 (PanWatch 深研 09-20): ai_failover.py 的 total_tokens_used 观测 + 预算
形态 → 5002 适配。5002 现状: llm_router 只记请求数/时延, 不记 token;
8080 调用无成本可观测性 (insight/辩论/Telegram B 链共用同一把 Semaphore(2))。

形态 (同链尸纪律):
  - 台账 = 账本, 不是判断: 只记 (provider, prompt, completion), 失败不计;
  - 预算门 = 可选 (daily_limit=0 默认无限 = 纯观测), 耗尽 → 路由链直接
    走 rule_engine 降级 (同 Semaphore 饱和形, 链不断);
  - 按 CST 自然日翻账, 历史留 8 天 (防常驻膨胀, 同日志轮转思路);
  - 时钟可注入 (测试确定性, 同 HostThrottle 形态)。

运行依赖: 仅 stdlib (time/threading/datetime), 零三方。
"""

import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

CST = timezone(timedelta(hours=8))


class LlmBudget:
    """按日 token 账本 + 可选预算门 (clock 可注入, 线程安全)"""

    def __init__(self, daily_limit: int = 0, clock=time.time):
        self._clock = clock                     # 测试可替换 (test_llm_budget 契约)
        self._limit = int(daily_limit or 0)
        self._lock = threading.Lock()
        self._ledger: Dict[str, Dict] = {}      # 'YYYY-MM-DD' -> 当日账本

    def _today(self) -> str:
        return datetime.fromtimestamp(self._clock(), CST).strftime('%Y-%m-%d')

    def record(self, provider: str, prompt_tokens: int = 0,
               completion_tokens: int = 0) -> None:
        """成功调用入账 (失败调用不计 — 无 token 消耗); 负数/非数 → 0 容错"""
        p = int(prompt_tokens or 0)
        c = int(completion_tokens or 0)
        if p < 0 or c < 0:
            return
        day = self._today()
        with self._lock:
            d = self._ledger.get(day)
            if d is None:
                d = {'calls': 0, 'prompt': 0, 'completion': 0, 'by_provider': {}}
                self._ledger[day] = d
                if len(self._ledger) > 8:        # 历史留 8 天, 防常驻膨胀
                    for k in sorted(self._ledger)[:len(self._ledger) - 8]:
                        self._ledger.pop(k, None)
            d['calls'] += 1
            d['prompt'] += p
            d['completion'] += c
            d['by_provider'][provider] = \
                d['by_provider'].get(provider, 0) + p + c

    def used_today(self) -> Dict:
        """当日账本快照 (含 total = prompt+completion)"""
        day = self._today()
        with self._lock:
            d = self._ledger.get(day)
            if not d:
                return {'calls': 0, 'prompt': 0, 'completion': 0,
                        'total': 0, 'by_provider': {}}
            return {'calls': d['calls'], 'prompt': d['prompt'],
                    'completion': d['completion'],
                    'total': d['prompt'] + d['completion'],
                    'by_provider': dict(d['by_provider'])}

    def is_exhausted(self) -> bool:
        """预算门: limit<=0 (默认) 恒 False = 纯观测模式"""
        if self._limit <= 0:
            return False
        return self.used_today()['total'] >= self._limit

    def stats(self) -> Dict:
        """health/API 暴露形 (含预算态)"""
        used = self.used_today()
        return {'daily_limit': self._limit, 'exhausted': self.is_exhausted(),
                'used': used}
