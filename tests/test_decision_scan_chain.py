# -*- coding: utf-8 -*-
"""2026-09-22 锁链: 决策闭环提效 — 监控池自动决策链 (decision_routes._run_decision_scan)

背景: recalibrate 门控阈 n≥20, 但 decide 无自动产生源 (_history 仅 15 =
手动/analyze 偶发触发) → 闭环自 09-12 恒 skipped 空转。09-22 新增 23:10
replay 链尾 scan 链: 监控池 5 票串行 consensus.decide → 台账累积。
锁形: 全绿/单票坏≠链死/decide 签名/空池。
"""
import unittest
from unittest.mock import MagicMock, patch

from modules.routes.decision_routes import _run_decision_scan

POOL = 'sz300620,sh688981'


def _fake_result(consensus='hold'):
    """伪造 ConsensusResult (真形有 .consensus 属性)"""
    r = MagicMock()
    r.consensus = consensus
    return r


def _mock_eng():
    eng = MagicMock()
    eng._history = [1] * 15
    return eng


class TestRunDecisionScan(unittest.TestCase):
    """_run_decision_scan: 监控池串行 decide (09-22 链形锁)"""

    def test_all_decide_success(self):
        """全绿 → success + scanned=池大小 + n_history 直通 (观测面形)"""
        eng = _mock_eng()
        eng.decide.return_value = _fake_result('hold')
        with patch('modules.multi_agent_consensus.get_consensus_engine',
                   return_value=eng):
            out = _run_decision_scan(POOL)
        self.assertTrue(out['success'])
        self.assertEqual(out['scanned'], 2)
        self.assertEqual(out['errors'], [])
        self.assertEqual(out['n_history'], 15)

    def test_single_broken_keeps_chain(self):
        """一票抛 → success:false 诚实标 + 另一票继续跑 (链坏≠链死)"""
        def _decide(code):
            if code == 'sh688981':
                raise RuntimeError('决策队列已满 (>8 排队)')
            return _fake_result('buy')
        eng = _mock_eng()
        eng.decide.side_effect = _decide
        with patch('modules.multi_agent_consensus.get_consensus_engine',
                   return_value=eng):
            out = _run_decision_scan(POOL)
        self.assertFalse(out['success'])
        self.assertEqual(out['scanned'], 1)
        self.assertEqual(len(out['errors']), 1)
        self.assertEqual(out['errors'][0]['code'], 'sh688981')
        self.assertIn('RuntimeError', out['errors'][0]['error'])

    def test_decide_positional_signature(self):
        """decide(code) 单位置参 (09-22 签名锁: 真形无 trigger kw, 防漂移)"""
        eng = _mock_eng()
        eng.decide.return_value = _fake_result()
        with patch('modules.multi_agent_consensus.get_consensus_engine',
                   return_value=eng):
            _run_decision_scan(POOL)
        self.assertEqual(eng.decide.call_count, 2)
        for call in eng.decide.call_args_list:
            self.assertEqual(len(call.args), 1, "decide 必须单位置参")
            self.assertEqual(call.kwargs, {}, "decide 不得带 kwargs")

    def test_empty_pool_not_broken(self):
        """空池 → 不链坏 (success:true, scanned=0)"""
        eng = _mock_eng()
        with patch('modules.multi_agent_consensus.get_consensus_engine',
                   return_value=eng):
            out = _run_decision_scan('')
        self.assertTrue(out['success'])
        self.assertEqual(out['scanned'], 0)


if __name__ == '__main__':
    unittest.main()
