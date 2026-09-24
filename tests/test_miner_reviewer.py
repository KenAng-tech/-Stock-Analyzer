#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
对抗审查员 + 三级反思记忆测试 (2026-09-14 P1-8)

运行: ./venv/bin/python -m unittest tests.test_miner_reviewer -q

全链无 LLM / 无网络 (monkeypatch llm_router + 合成面板):
  1. miner_memory 三级读写 + 原子持久化 + 损坏自愈 + 教训截断
  2. 纯函数: ic_decay_ratio / compute_holdout_ic (保留窗口重执行)
  3. adversarial_review: LLM veto → observe 只标记不拦截 / enforce 拦截;
     重执行衰减 veto 不依赖 LLM; LLM 失败/规则引擎降级 → review_skipped
  4. mine_round 接线: generation 记忆 + cycle 摘要 + prompt 线索注入
"""

import os
import sys
import json
import sqlite3
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from modules.miner_memory import MinerMemory
from modules import rd_agent_miner as ram
from modules.rd_agent_miner import (
    RDAgentMiner, FactorHypothesis, ic_decay_ratio, compute_holdout_ic,
)

# 完美负相关因子: sig = -fwd → 任意窗口 IC = -1 (确定性, 无随机)
PERFECT_CODE = "df['sig'] = -df['close'].pct_change(5).shift(-5)"


def _mk_df(n: int = 120, seed: int = 3) -> pd.DataFrame:
    """合成日线面板 (随机游走 close, 含 date/open/high/low/close/volume)"""
    rng = np.random.RandomState(seed)
    close = 100.0 * np.cumprod(1.0 + rng.randn(n) * 0.01)
    return pd.DataFrame({
        'date': pd.date_range('2026-01-01', periods=n),
        'open': close, 'high': close * 1.01, 'low': close * 0.99,
        'close': close,
        'volume': rng.randint(100000, 1000000, n).astype(float),
    })


def _mk_h(ic: float = 0.05) -> FactorHypothesis:
    """固定候选: name='sig' + PERFECT_CODE (重执行 IC=-1, 衰减为负不触发 reexec veto)"""
    return FactorHypothesis(name='sig', formula='perfect neg fwd', category='momentum',
                            rationale='r', code=PERFECT_CODE, ic=ic)


class FakeRouter:
    """假 llm_router: 记录 prompt, 返回预设内容 (不发网络)"""

    def __init__(self, content='', success=True, fallback=False, provider='omlx'):
        self.content = content
        self.success = success
        self.fallback = fallback
        self.provider = provider
        self.calls = []

    def route(self, prompt, context=None, timeout=None):
        self.calls.append({'prompt': prompt, 'timeout': timeout})
        if not self.success:
            return {'success': False, 'error': 'llm down'}
        return {'success': True, 'content': self.content, 'provider': self.provider,
                'latency': 0.1, 'fallback': self.fallback}


class _MinerCase(unittest.TestCase):
    """公共夹具: tmp DB/记忆 + 禁自动挖掘线程 + 隔离环境变量"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.mem = MinerMemory(os.path.join(self.tmp.name, 'miner_memory.json'))
        with mock.patch.object(RDAgentMiner, '_start_auto_mine', lambda self: None):
            self.miner = RDAgentMiner(
                db_path=os.path.join(self.tmp.name, 'factors_test.db'))
        self._p = [
            mock.patch.object(ram, 'get_miner_memory', lambda: self.mem),
            mock.patch.dict(os.environ),
        ]
        for p in self._p:
            p.start()
            self.addCleanup(p.stop)
        os.environ.pop('MINER_ADVERSARIAL_MODE', None)

    def _patch_router(self, router):
        p = mock.patch.object(ram, 'llm_router', router)
        p.start()
        self.addCleanup(p.stop)
        return router

    def _run_round(self, router, panel=None):
        """跑一轮单类别挖掘: 生成/验证/取数全 monkeypatch, 只留审查+记忆真链"""
        self._patch_router(router)
        panel = panel if panel is not None else [_mk_df(seed=3), _mk_df(seed=5)]
        self.miner._last_panel = []
        ps = [
            mock.patch.object(self.miner, 'generate_hypothesis',
                              side_effect=lambda category='momentum': _mk_h()),
            mock.patch.object(self.miner, 'validate_factor',
                              side_effect=lambda h: (0.05, 0.3, None)),
            mock.patch.object(self.miner, '_fetch_panel', return_value=panel),
        ]
        for p in ps:
            p.start()
            self.addCleanup(p.stop)
        return self.miner.mine_round(['momentum'])


class TestMinerMemory(unittest.TestCase):
    """三级读写 + 原子持久化 + 损坏自愈"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, 'miner_memory.json')

    def test_three_tiers_roundtrip(self):
        """generation/cycle/archetype 三级写入 → 新实例读回, 计数与线索正确"""
        mem = MinerMemory(self.path)
        mem.record_generation('a', 'f1', 'momentum', 0.05, 'pass')
        mem.record_generation('b', 'f2', 'momentum', 0.02, 'reject',
                              reviewer_notes='教科书动量换皮')
        mem.summarize_cycle(['momentum'], 2, 0.5, ['momentum'],
                            lessons={'momentum': '勿再背诵20日动量'})

        mem2 = MinerMemory(self.path)  # 原子写后可持久化读回
        self.assertEqual(len(mem2._data['generations']), 2)
        self.assertEqual(len(mem2._data['cycles']), 1)
        arch = mem2._data['archetypes']['momentum']
        self.assertEqual(arch['tried_n'], 2)
        self.assertEqual(arch['pass_rate'], 0.5)
        self.assertIn('勿再背诵20日动量', arch['lessons'])

        cues = mem2.cues_for('momentum')
        self.assertIn('momentum', cues)
        self.assertIn('教训', cues)
        self.assertIn('教科书动量换皮', cues)
        self.assertIn('上轮', mem2.cues_for())  # 全局线索含 cycle 摘要行
        snap = mem2.snapshot()
        self.assertEqual(snap['n_generations'], 2)

    def test_empty_memory_cues_blank(self):
        """空记忆 → cues_for 返回 '' (调用方据此跳过注入)"""
        mem = MinerMemory(self.path)
        self.assertEqual(mem.cues_for('momentum'), '')
        self.assertEqual(mem.cues_for(), '')

    def test_corruption_selfheal(self):
        """损坏 JSON → 备份 .corrupt.* + 重建空记忆, 后续读写正常"""
        with open(self.path, 'w', encoding='utf-8') as f:
            f.write('{definitely not json!!')
        mem = MinerMemory(self.path)
        backups = [x for x in os.listdir(self.tmp.name)
                   if x.startswith('miner_memory.json.corrupt.')]
        self.assertEqual(len(backups), 1)
        mem.record_generation('x', 'f', 'value', 0.04, 'pass')
        mem2 = MinerMemory(self.path)
        self.assertEqual(len(mem2._data['generations']), 1)

    def test_lessons_capped(self):
        """archetype 教训环形截断 (≤5 条)"""
        mem = MinerMemory(self.path)
        for i in range(7):
            mem.summarize_cycle(['volume'], 1, 0.0, [], lessons={'volume': f'L{i}'})
        lessons = mem._data['archetypes']['volume']['lessons']
        self.assertEqual(len(lessons), 5)
        self.assertEqual(lessons[-1], 'L6')


class TestPureFunctions(unittest.TestCase):
    """衰减比例 + 保留窗口重执行 (纯函数, 不调 LLM)"""

    def test_ic_decay_ratio(self):
        """decay = 1 - |holdout|/|declared|; None 传播; declared≈0 → 1.0"""
        self.assertAlmostEqual(ic_decay_ratio(0.10, 0.04), 0.6)
        self.assertAlmostEqual(ic_decay_ratio(0.10, 0.06), 0.4)
        self.assertLess(ic_decay_ratio(0.05, -0.9), 0)   # holdout 更强 → 负衰减
        self.assertIsNone(ic_decay_ratio(0.10, None))
        self.assertEqual(ic_decay_ratio(0.0, 0.03), 1.0)

    def test_compute_holdout_ic_perfect(self):
        """完美因子 (sig=-fwd) → 保留窗口 IC≈-1, 两票均有效"""
        ic, n, errs = compute_holdout_ic([_mk_df(120, 3), _mk_df(120, 5)],
                                         PERFECT_CODE, 'sig', holdout_days=60)
        self.assertIsNotNone(ic)
        self.assertEqual(n, 2)
        self.assertLess(ic, -0.99)

    def test_compute_holdout_ic_degenerate(self):
        """常数因子 (零方差) → IC NaN → (None, 0); 语法错代码 → 不抛只记错"""
        ic, n, errs = compute_holdout_ic([_mk_df()], 'df["sig"] = 1.0', 'sig')
        self.assertIsNone(ic)
        self.assertEqual(n, 0)
        ic2, n2, errs2 = compute_holdout_ic([_mk_df()], 'this is not python(', 'sig')
        self.assertIsNone(ic2)
        self.assertEqual(n2, 0)
        self.assertTrue(errs2)


class TestAdversarialReview(_MinerCase):
    """审查员否决/降级路径 (monkeypatch LLM, 无网络)"""

    def test_reexec_decay_veto_no_llm(self):
        """重执行衰减 >50% → 直接 veto (source=reexec), 完全不调 LLM"""
        router = FakeRouter()
        self._patch_router(router)
        with mock.patch.object(ram, 'compute_holdout_ic',
                               return_value=(0.02, 2, [])), \
                mock.patch.object(self.miner, '_fetch_panel',
                                  return_value=[_mk_df()]):
            h = _mk_h(ic=0.10)  # 申报 0.10 → holdout 0.02 → 衰减 80%
            r = self.miner.adversarial_review(h)
        self.assertEqual(r['verdict'], 'veto')
        self.assertEqual(r['source'], 'reexec')
        self.assertAlmostEqual(r['decay'], 0.8)
        self.assertEqual(router.calls, [])  # 一票否决不依赖 LLM

    def test_llm_veto_observe_marks_not_blocks(self):
        """observe (默认): LLM veto → status=rejected_reviewer_observe, 仍入库"""
        router = FakeRouter(content=json.dumps(
            {'verdict': 'veto', 'reasons': ['教科书动量换皮', '经济逻辑缺失']}))
        r = self._run_round(router)
        self.assertEqual(r['reviewer_veto_observe'], 1)
        self.assertEqual(r['validated'], 0)
        self.assertEqual(r['adversarial_mode'], 'observe')
        h = self.miner._hypotheses['sig']
        self.assertEqual(h.status, 'rejected_reviewer_observe')
        self.assertIn('教科书动量换皮', h.reviewer_notes)
        # 不拦截入库: DB 有记录
        with sqlite3.connect(self.miner.db_path) as conn:
            row = conn.execute(
                "SELECT status FROM factors WHERE name='sig'").fetchone()
        self.assertEqual(row[0], 'rejected_reviewer_observe')

    def test_llm_veto_enforce_blocks(self):
        """enforce: LLM veto → status=rejected_reviewer (拦截 validated 入库)"""
        os.environ['MINER_ADVERSARIAL_MODE'] = 'enforce'
        router = FakeRouter(content='{"verdict": "veto", "reasons": ["数据窥探"]}')
        r = self._run_round(router)
        self.assertEqual(r['reviewer_veto_blocked'], 1)
        self.assertEqual(self.miner._hypotheses['sig'].status, 'rejected_reviewer')

    def test_llm_accept_passes(self):
        """LLM accept → 保持 validated"""
        router = FakeRouter(content='{"verdict": "accept", "reasons": ["逻辑成立"]}')
        r = self._run_round(router)
        self.assertEqual(r['validated'], 1)
        self.assertEqual(self.miner._hypotheses['sig'].status, 'validated')

    def test_llm_failure_review_skipped(self):
        """LLM 失败 → review_skipped, 绝不阻断 (保持 validated)"""
        router = FakeRouter(success=False)
        r = self._run_round(router)
        self.assertEqual(r['validated'], 1)
        self.assertEqual(r['review_skipped'], 1)
        self.assertEqual(self.miner._hypotheses['sig'].review_verdict, 'review_skipped')

    def test_rule_engine_fallback_response_skipped(self):
        """路由降级 rule_engine (决策 JSON) → 不是审查结论 → review_skipped"""
        router = FakeRouter(content='{"direction": "neutral", "confidence": 0.3}',
                            fallback=True, provider='rule_engine')
        r = self._run_round(router)
        self.assertEqual(r['review_skipped'], 1)
        self.assertEqual(self.miner._hypotheses['sig'].status, 'validated')


class TestMemoryWiring(_MinerCase):
    """mine_round 接线: generation/cycle 记忆 + prompt 线索注入"""

    def test_generation_and_cycle_recorded(self):
        """一轮挖掘后: generation 条目 + cycle 摘要 + archetype 计数落盘"""
        router = FakeRouter(content='{"verdict": "accept", "reasons": ["ok"]}')
        self._run_round(router)
        self.assertEqual(len(self.mem._data['generations']), 1)
        g = self.mem._data['generations'][0]
        self.assertEqual(g['verdict'], 'pass')
        self.assertEqual(g['family'], 'momentum')
        cycle = self.mem._data['cycles'][0]
        self.assertEqual(cycle['n_candidates'], 1)
        self.assertEqual(cycle['pass_rate'], 1.0)
        self.assertEqual(self.mem._data['archetypes']['momentum']['tried_n'], 1)

    def test_veto_recorded_as_rejected_reviewer(self):
        """veto 候选在记忆中 verdict=rejected_reviewer + 否决理由"""
        router = FakeRouter(content='{"verdict": "veto", "reasons": ["背诵嫌疑"]}')
        self._run_round(router)
        g = self.mem._data['generations'][0]
        self.assertEqual(g['verdict'], 'rejected_reviewer')
        self.assertIn('背诵嫌疑', g['reviewer_notes'])

    def test_prompt_cue_injection(self):
        """下轮 generate_hypothesis prompt 注入 cues_for 线索 (guarded)"""
        self.mem.record_generation('old_mom', 'mom20', 'momentum', 0.01,
                                   'rejected_reviewer', reviewer_notes='教科书动量')
        router = FakeRouter(content=json.dumps({
            'name': 'new_factor', 'formula': 'f', 'category': 'momentum',
            'rationale': 'r', 'code': "df['new_factor'] = df['close']"}))
        self._patch_router(router)
        h = self.miner.generate_hypothesis('momentum')
        self.assertEqual(h.name, 'new_factor')
        prompt = router.calls[0]['prompt']
        self.assertIn('历史挖掘反思线索', prompt)
        self.assertIn('教科书动量', prompt)

    def test_cue_injection_failure_not_blocking(self):
        """cues_for 抛异常 → 跳过注入, 生成链照常 (guarded)"""
        with mock.patch.object(self.mem, 'cues_for', side_effect=RuntimeError('boom')):
            router = FakeRouter(content=json.dumps({
                'name': 'f2', 'formula': 'f', 'category': 'momentum',
                'rationale': 'r', 'code': "df['f2'] = df['close']"}))
            self._patch_router(router)
            h = self.miner.generate_hypothesis('momentum')
        self.assertEqual(h.name, 'f2')


if __name__ == '__main__':
    unittest.main()
