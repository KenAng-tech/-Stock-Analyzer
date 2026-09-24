# -*- coding:utf-8 -*-
"""P1-b 决策过程六维审计测试 (2026-09-22, arXiv 2605.05739 六维行为审计落地)

链背景: honest_eval = 回测五门 (DSR/PBO/SPA/MTRL/REGIME, 09-21 建),
exec_audit = 可成交性 (P0-c, 09-12 建) — **决策过程层审计从未建** (2605.05739
六维 = 过程质量: 证据完备/内部一致/分歧处理/不确定性量化/辩论参与/推理链)。

六维 (audit_process(result) → dims 六键, 纯计算零取数 <1ms):
  d1 evidence_completeness = (n-abstain)/n, abstain = conf≤0.25 (09-12 同形)
  d2 internal_consistency = 共识同向票/非弃权票 (neutral 票不算异议)
  d3 conflict_handling = 无分歧 1.0 | 分歧+有辩论 1.0 | 分歧+无辩论 0.5
  d4 uncertainty_quantification = 方向+异议≥0.4+conf≥0.8 → 0.5 (过度自信)
                                 neutral+conf≥0.6 → 0.5 (不 bet 但 conf 虚高)
  d5 debate_participation = bull/bear 票在 → 1.0 | 无分歧不需辩论 → 1.0
  d6 reasoning_quality = reasoning 短(<4字)票占比 (含 consensus.reasoning 空)

锁形 (expected 全独立来源手算): ① abstain 链 0.75 ② 单边满分 ③ 2v2+辩论触发
④ 过度自信 0.5 ⑤ neutral 高 conf 0.5 ⑥ 空票/不足 → skipped (断链≠链死)
⑦ 装配链 = audit_for_result 落盘跟随 DQ_PROC_PATH 重定向 (不污染真观察池)
⑧ worker 全链 smoke: 真 decide 跑通 (降级/真票皆非断链)
"""
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.decision_quality import audit_process, audit_for_result  # noqa: E402
from modules.multi_agent_consensus import ConsensusResult  # noqa: E402


def _mk(consensus='buy', conf=0.55, votes=None, reasoning='技术面与量价同向向上'):
    return ConsensusResult(
        stock_code='sz300620', consensus=consensus, confidence=conf,
        vote_count={}, agent_votes=votes if votes is not None else [],
        weights={}, conflict=False, conflict_ratio=0.3,
        timestamp='2026-09-22T23:10:00', reasoning=reasoning)


def _v(role, dec, conf, r='10日均线上方放量'):
    return {'role': role, 'decision': dec, 'confidence': conf,
            'reasoning': r, 'weight': 1.0}


class TestSixDimAudit(unittest.TestCase):

    def setUp(self):
        """落盘重定向到 tempdir (P1-a _metrics_path 同教训: 测试不污染真观察池)"""
        self._tmp = tempfile.TemporaryDirectory()
        self._env = patch.dict(os.environ, {
            'DQ_PROC_PATH': os.path.join(self._tmp.name, 'process_audit.jsonl')})
        self._env.start()

    def tearDown(self):
        self._env.stop()
        self._tmp.cleanup()

    def test_abstain_chain_and_keys(self):
        """① 六键全 + abstain 审计: 4 票 1 弃权 (conf 0.2) → d1=(4-1)/4=0.75"""
        votes = [_v('technical', 'buy', 0.6), _v('quant', 'buy', 0.2),
                 _v('sentiment', 'neutral', 0.55), _v('timeseries', 'buy', 0.6)]
        r = audit_process(_mk(votes=votes))
        self.assertNotIn('skipped', r)
        self.assertEqual(len(r['dims']), 6, '六维键全')
        # 独立手算: 弃权 1/4 → d1=0.75; 无分歧 → d3=1.0; 无比辩论但无分歧 → d5=1.0
        self.assertAlmostEqual(r['dims']['evidence_completeness'], 0.75, places=3)
        self.assertEqual(r['dims']['conflict_handling'], 1.0)
        self.assertEqual(r['dims']['debate_participation'], 1.0)

    def test_unilateral_full_marks(self):
        """② 单边 4 票 = 无分歧: 一致率 1.0 + 辩论不缺位"""
        votes = [_v('technical', 'buy', 0.9), _v('quant', 'buy', 0.8),
                 _v('sentiment', 'neutral', 0.5), _v('timeseries', 'buy', 0.7)]
        r = audit_process(_mk(consensus='strong_buy', conf=0.85, votes=votes))
        self.assertEqual(r['dims']['internal_consistency'], 1.0)
        self.assertEqual(r['dims']['conflict_handling'], 1.0)

    def test_polar_conflict_with_debate(self):
        """③ 2多2空 + bull/bear 辩论票 = 辩论链触发 (d3/d5 满分形)"""
        votes = [_v('technical', 'buy', 0.9), _v('quant', 'buy', 0.9),
                 _v('sentiment', 'sell', 0.1), _v('timeseries', 'sell', 0.1),
                 _v('bull', 'buy', 0.55), _v('bear', 'sell', 0.5)]
        r = audit_process(_mk(consensus='buy', conf=0.5, votes=votes))
        self.assertEqual(r['dims']['conflict_handling'], 1.0, '分歧有辩论 = 处理链触发')
        self.assertEqual(r['dims']['debate_participation'], 1.0)

    def test_polar_no_debate_flagged(self):
        """③b 2多2空 无辩论票 = 分歧未处理 (0.5 观察) + 异议率 2/4 + conf0.9 → d4 0.5"""
        votes = [_v('technical', 'buy', 0.9), _v('quant', 'buy', 0.9),
                 _v('sentiment', 'sell', 0.9), _v('timeseries', 'sell', 0.9)]
        r = audit_process(_mk(consensus='buy', conf=0.9, votes=votes))
        self.assertEqual(r['dims']['conflict_handling'], 0.5, '分歧无辩论 → 半信观察')
        self.assertEqual(r['dims']['debate_participation'], 0.5, '辩论缺位标记')
        self.assertEqual(r['dims']['uncertainty_quantification'], 0.5,
                         'conf0.9+异议 50% = 过度自信链')

    def test_neutral_high_conf_flagged(self):
        """⑤ neutral+conf0.75 = 不 bet 但 conf 虚高 → uncertainty 0.5 (观察)"""
        votes = [_v('technical', 'neutral', 0.5), _v('quant', 'neutral', 0.4),
                 _v('sentiment', 'buy', 0.3), _v('timeseries', 'sell', 0.3)]
        r = audit_process(_mk(consensus='neutral', conf=0.75, votes=votes))
        self.assertEqual(r['dims']['uncertainty_quantification'], 0.5)

    def test_empty_votes_skipped(self):
        """⑥ 空票/单票 = 过程审计样本不足 → skipped (诚实空, 非假六维)"""
        r0 = audit_process(_mk(votes=[]))
        self.assertIn('skipped', r0)
        r1 = audit_process(_mk(votes=[_v('technical', 'buy', 0.9)]))
        self.assertIn('skipped', r1, '单票过程审计无意义 (链坏≠链死形)')

    def test_worker_inline_chain(self):
        """⑦ 装配入口: audit_for_result 直调 = 审计+落盘, 路径跟随 DQ_PROC_PATH"""
        votes = [_v('technical', 'buy', 0.9), _v('quant', 'buy', 0.8),
                 _v('sentiment', 'neutral', 0.5), _v('timeseries', 'buy', 0.7)]
        r = audit_for_result(_mk(consensus='strong_buy', conf=0.85, votes=votes))
        self.assertEqual(len(r['dims']), 6, '六维直调出形')
        path = os.environ['DQ_PROC_PATH']
        self.assertTrue(os.path.exists(path), '落盘跟随重定向 (测试≠污染真观察池)')
        with open(path, encoding='utf-8') as f:
            self.assertEqual(len(f.readlines()), 1, '一次审计恰一行')

    def test_worker_full_chain_smoke(self):
        """⑧ worker 全链 smoke: 真 decide 跑通 (23:10 scan 形) + decision_quality 装配"""
        from modules.multi_agent_consensus import MultiAgentConsensus
        eng = MultiAgentConsensus()
        r = eng.decide('sz300620')  # 真实链 (dev 8080/数据源在位时产真票; 降级也须跑通)
        self.assertIsNotNone(r.consensus)
        # 审计装配链锁: 跑通后为 dict (六维/降权标) 或 None (降级) — 两形皆非断链
        self.assertTrue(r.decision_quality is None
                        or len(r.decision_quality.get('dims', {})) == 6
                        or 'skipped' in r.decision_quality)


if __name__ == '__main__':
    unittest.main(verbosity=2)
