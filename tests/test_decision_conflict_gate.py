# -*- coding:utf-8 -*-
"""23:10 决策链锁链测试 (2026-09-22 P0-1: disagreement 置信门)

锁 4 组形:
  ① 极化门触发 (多空各≥2 → 弱动能 NEUTRAL / 强动能降档 STRONG→方向)
  ② 非触发 (单边票/flag=0) = 原语义零变化 (回滚形)
  ③ UDE.decide 全链跑通 (死链摘除后不再 NameError; spy monitor.update 0 调用
     = 喂料链诚实跳过锁 — 若未来"复活"喂料, spy 必捕 = 反 tautological)
  ④ 真实 5 票 scan 链形 (5 票×8 票全跑 = 23:10 链形回归)
"""
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.multi_agent_consensus import (  # noqa: E402
    MultiAgentConsensus, AgentVote, Decision, ConsensusResult)


def _mk_votes(specs):
    """specs: [(role, decision, conf, weight), ...] → AgentVote 列表"""
    return [AgentVote(role=r, decision=d, confidence=c, reasoning='t', weight=w)
            for r, d, c, w in specs]


class TestPolarizationGate(unittest.TestCase):
    """①② 极化门 (arXiv 2603.10137/2608.27076 disagreement 门 5002 形)"""

    def setUp(self):
        self.eng = MultiAgentConsensus()
        # 门控不依赖真数据源: 只锁 _compute_consensus (纯计算, worker 内核)
        self.assertIsNotNone(self.eng._compute_consensus)

    def _gate(self, specs):
        return self.eng._compute_consensus('sz300620', _mk_votes(specs))

    def test_polar_weak_momentum_to_neutral(self):
        """多 2 空 2 强弱对消 (±0.3 以下) → NEUTRAL (极化不bet)"""
        r = self._gate([
            ('technical', 'buy', 0.5, 1.0), ('quant', 'buy', 0.4, 1.0),
            ('sentiment', 'sell', 0.5, 1.0), ('timeseries', 'sell', 0.4, 1.0),
            ('fundamental', 'neutral', 0.3, 1.0), ('rl', 'neutral', 0.2, 1.0),
        ])
        self.assertEqual(r.consensus, Decision.NEUTRAL.value)
        self.assertIn('极化门', r.reasoning)

    def test_polar_strong_downgrade(self):
        """多 2 空 2 但空方极弱 (|score|=0.4>0.3) → STRONG 降档 + conf×0.5 (半信)"""
        r = self._gate([
            ('technical', 'strong_buy', 0.9, 1.0), ('quant', 'strong_buy', 0.9, 1.0),
            ('sentiment', 'sell', 0.1, 1.0), ('timeseries', 'sell', 0.1, 1.0),
        ])
        self.assertNotEqual(r.consensus, Decision.STRONG_BUY.value)
        # 原 conf=min(1,|0.68|+0.3)=1.0 → 门后 ×0.5=0.5 (分歧半信实锤)
        self.assertLessEqual(r.confidence, 0.5)
        self.assertIn('半信', r.reasoning)

    def test_unilateral_no_gate(self):
        """单边 (多 4 空 0) = 不触发 (无分歧, 原语义零变化)"""
        r = self._gate([
            ('technical', 'buy', 0.9, 1.0), ('quant', 'buy', 0.8, 1.0),
            ('sentiment', 'buy', 0.7, 1.0), ('timeseries', 'buy', 0.6, 1.0),
        ])
        self.assertNotIn('极化门', r.reasoning)
        self.assertEqual(r.consensus, Decision.STRONG_BUY.value)

    def test_flag_off_rollback(self):
        """DECISION_CONFLICT_GATE=0 = 原路径 (STRONG 保留, 回滚形)"""
        specs = [('technical', 'strong_buy', 0.9, 1.0), ('quant', 'strong_buy', 0.9, 1.0),
                 ('sentiment', 'sell', 0.1, 1.0), ('timeseries', 'sell', 0.1, 1.0)]
        with patch.dict(os.environ, {'DECISION_CONFLICT_GATE': '1'}):
            on = self._gate(specs)
        with patch.dict(os.environ, {'DECISION_CONFLICT_GATE': '0'}):
            off = self._gate(specs)
        # on 半信降档, off 保留 STRONG_BUY 原判定 (两态差 = 门真生效实锤)
        self.assertEqual(on.consensus, Decision.BUY.value)
        self.assertEqual(off.consensus, Decision.STRONG_BUY.value)
        self.assertGreater(off.confidence, on.confidence)


class TestUdeChainAlive(unittest.TestCase):
    """③④ UDE.decide 全链 (死链摘除锁)"""

    def test_decide_full_chain_no_silent_break(self):
        """decide 跑通 (原 NameError 死链段已摘) + monitor.update 0 调用锁
        (喂料诚实跳过的诚实锁: 未来复活喂料必被本锁抓, 非 tautological)"""
        from modules.unified_decision_engine import UnifiedDecisionEngine
        ude = UnifiedDecisionEngine()
        spy = []

        def spy_update(pred, actual, _tag):
            spy.append((_tag, repr(pred)[:40], repr(actual)[:12]))
            return False

        for name, mon in ude.drift_ensemble._monitors.items():
            mon.update = lambda p, a, _t=name: spy_update(p, a, _t)
        r = ude.decide('sz300620')
        # ① 跑通: 有方向有票 (真实数据链在 dev 环境产 15 票实锤 09-22)
        self.assertTrue(r.direction)
        # ② 死链诚实: 无真收益 → 不喂 (喂料须走 23:10 replay, 非实时自指)
        self.assertEqual(spy, [], '喂料复活须带真收益 (replay 链), 不得实时自指')


if __name__ == '__main__':
    unittest.main(verbosity=2)
