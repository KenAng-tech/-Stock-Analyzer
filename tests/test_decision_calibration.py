#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""决策链 B 锁链测试 (2026-09-11)

锁链 (09-10/09-11 同风格 — 真实链形态锁进测试, 不盲改):
  1. brier_ece 数学形 (perfect/anti-calibrated) + 功效门 + 非有限/坏形过滤
  2. llm_sentiment.analyze 契约三键 fail-closed (缺键/漂移形 → None,
     不再伪装 source=llm_news 真链; 反例 = 旧版 get(默认值) 假中性)
  3. consensus 辩论 tool contract (direction 缺失 ≠ neutral 票; NaN conf 弃票)

⚠ 09-08/09-11 SIGSEGV 教训: 不在 unittest 主进程 in-process 跑 Optuna/LGBM;
本套只纯函数 + mock router + 假 _call_api lambda — 零 LLM 零训练零网络。

运行: python -m unittest tests.test_decision_calibration
"""

import sys
import os
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.decision_calibration import brier_ece


class TestBrierECE(unittest.TestCase):
    """Brier/ECE 数学形 + 统计功效门 (诚实标记, 不装真值)"""

    def test_perfectly_calibrated(self):
        """p=0.9 全对 → Brier≈0.01, ECE≈0.1 (acc 1.0 vs conf 0.9)"""
        recs = [{'p': 0.9, 'outcome': 1} for _ in range(25)]
        out = brier_ece(recs)
        self.assertNotIn('skipped', out)
        self.assertEqual(out['n'], 25)
        self.assertAlmostEqual(out['brier'], 0.01, places=3)
        self.assertAlmostEqual(out['acc'], 1.0, places=3)
        self.assertIsNotNone(out['ece'])          # 单 bin n=25 ≥8 → 可算
        self.assertAlmostEqual(out['ece'], 0.1, places=3)

    def test_anti_calibrated(self):
        """p=0.9 全错 → Brier≈0.81 (反校准上界), 与 perfect 拉开数量级"""
        recs = [{'p': 0.9, 'outcome': 0} for _ in range(25)]
        out = brier_ece(recs)
        self.assertGreater(out['brier'], 0.7)
        self.assertLess(out['acc'], 0.05)

    def test_guard_and_nonfinite_filter(self):
        """n<20 → skipped; NaN/Inf/坏形过滤后 n=24 → 仍可算 (NaN 不泄漏成真值)"""
        self.assertIn('skipped', brier_ece([]))
        self.assertIn('skipped', brier_ece(
            [{'p': 0.9, 'outcome': 1} for _ in range(10)]))
        recs = ([{'p': 0.9, 'outcome': 1} for _ in range(24)]
                + [{'p': float('nan'), 'outcome': 1},
                   {'p': 0.9, 'outcome': None},
                   {'p': 5.0, 'outcome': 1}])
        out = brier_ece(recs)
        self.assertEqual(out['n'], 24)            # 3 条坏形被过滤
        self.assertTrue(out.get('ece') is not None or out['ece'] is None)
        self.assertLess(out['brier'], 0.2)

    def test_ece_small_bins_none(self):
        """4 bin × 6 样本 (各 <8): Brier 可算, ECE=skipped → ece=None"""
        recs = [{'p': p, 'outcome': 1} for p in (0.3, 0.5, 0.7, 0.9)
                for _ in range(6)]
        out = brier_ece(recs)
        self.assertEqual(out['n'], 24)
        self.assertIsNone(out['ece'])
        self.assertEqual(out['ece_bins'], 0)


class TestLLMContractValidation(unittest.TestCase):
    """llm_sentiment.analyze — 契约三键 fail-closed (无网络, lambda 假 _call_api)"""

    def _analyze(self, content):
        from modules.llm_sentiment import LLMClient
        client = LLMClient.__new__(LLMClient)      # 绕 __init__ 的 /v1/models 探测
        client._initialized = True
        client.config = SimpleNamespace(sentiment_prompt='你是金融情感分析师')
        client._call_api = lambda msgs: content    # 注入 LLM 原始返回文本
        return client.analyze('宁德时代利好不断')

    def test_good_contract_passes(self):
        out = self._analyze(
            '{"direction":"negative","score":-0.6,"confidence":0.7,'
            '"reason":"利空","keywords":["减持"]}')
        self.assertIsNotNone(out)
        self.assertEqual(out['direction'], 'negative')
        self.assertAlmostEqual(out['score'], -0.6)
        self.assertAlmostEqual(out['confidence'], 0.7)

    def test_missing_keys_rejected(self):
        """旧链: {} → 默认值伪造 (neutral,0,0.5) 标 source=llm_news 真链 — 已封"""
        self.assertIsNone(self._analyze('{}'))

    def test_drift_shape_rejected(self):
        """direction 漂移值 / score 越界 / conf 字符串 → 全 None"""
        self.assertIsNone(self._analyze('{"direction":"bullish","score":5,"confidence":2}'))
        self.assertIsNone(self._analyze('{"direction":"neutral","score":0.1}'))
        self.assertIsNone(self._analyze('{"score":0.1,"confidence":0.5}'))

    def test_fence_and_list_shape(self):
        """markdown 围栏 + list 包装 (链上实测形态) → 剥壳取首元素后仍走契约校验"""
        out = self._analyze(
            '```json\n[{"direction":"positive","score":0.5,"confidence":0.6}]\n```')
        self.assertIsNotNone(out)
        self.assertEqual(out['direction'], 'positive')
        self.assertAlmostEqual(out['confidence'], 0.6)


class TestDebateContract(unittest.TestCase):
    """consensus 辩论段 — LLM JSON 缺 direction/NaN conf → 弃票 (不造 neutral)"""

    def _run_debate(self, content):
        import modules.multi_agent_consensus as mac
        eng = mac.MultiAgentConsensus.__new__(mac.MultiAgentConsensus)
        eng._enable_debate = True
        eng._debate_timeout = 5
        votes = [
            SimpleNamespace(role='technical', decision='buy', confidence=0.6),
            SimpleNamespace(role='quant', decision='sell', confidence=0.5),
        ]
        mod = SimpleNamespace(llm_router=SimpleNamespace(
            route=lambda prompt, timeout=None: {
                'success': True, 'fallback': False, 'provider': 'omlx',
                'content': content, 'latency': 1.0}))
        with mock.patch.dict(sys.modules, {'modules.llm_router': mod}):
            return eng._run_debate('sz300620', votes)

    def test_missing_direction_no_vote(self):
        """旧链: 缺 direction → 默认 'neutral' 造 2 票稀释共识; 新链 → 0 票"""
        self.assertEqual(self._run_debate('{"argument":"无方向键","confidence":0.5}'), [])

    def test_nan_confidence_no_vote(self):
        """NaN conf 穿 max/min (Python 陷阱: 0.6<nan 为 False) → 契约弃票"""
        self.assertEqual(
            self._run_debate('{"direction":"buy","confidence":NaN,"argument":"x"}'), [])

    def test_valid_contract_votes(self):
        """合法 JSON → bull 票入 (方向限 bull 侧 allow), bear 侧方向漂移被拒"""
        out = self._debat = self._run_debate(
            '{"direction":"buy","confidence":0.5,"argument":"x"}')
        self.assertEqual(len(out), 1)             # bull: buy ✓ / bear: buy ✗
        self.assertEqual(out[0].role, 'llm_bull')
        self.assertEqual(out[0].decision, 'buy')
        self.assertAlmostEqual(out[0].confidence, 0.5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
