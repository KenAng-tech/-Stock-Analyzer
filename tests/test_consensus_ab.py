#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
consensus 双轨 A/B + TopkDropout 防抖测试 (P1-7 + P1-10, 2026-09-14)

覆盖:
- zscore_equal_weight: 手算数值对照 / std=0 退化 / 空票 / dict 兼容
- dropout_decide: 抖动不换 / 跌出末位换 / 持有期未满不换 / tradability 拦截
- tradability: 板块化涨跌停阈值 + 封板/停牌判定
- pnl_corr: 合成数据 (完全正相关/反相关角色) + 数据不足诚实返回
- observe_dropout: 'off' 返回 None, 'observe' 写 jsonl 不改输出
"""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from modules.multi_agent_consensus import AgentVote, zscore_equal_weight
from modules.tradability import is_tradable, limit_threshold_for
from modules.pnl_corr import analyze
from modules.unified_decision_engine import (
    UnifiedDecisionEngine, dropout_decide)


class TestZscoreEqualWeight(unittest.TestCase):
    """z-score 等权旁路聚合 (手算对照)"""

    def test_hand_computed(self):
        """buy(0.9)/sell(0.5)/neutral(0.4):
        confs mean=0.6, std(ddof=0)=sqrt(0.14/3)≈0.216025
        z=[1.38873,-0.46291,-0.92582] → relu → [1.38873,0,0]
        score=(1×1.38873 + (-1)×0 + 0×0)/3 ≈ 0.46291 → buy
        """
        votes = [
            AgentVote(role='a', decision='buy', confidence=0.9),
            AgentVote(role='b', decision='sell', confidence=0.5),
            AgentVote(role='c', decision='neutral', confidence=0.4),
        ]
        out = zscore_equal_weight(votes)
        self.assertAlmostEqual(out['score'], 0.46291, places=4)
        self.assertEqual(out['direction'], 'buy')
        self.assertEqual(out['n_votes'], 3)
        self.assertAlmostEqual(out['confidence'], min(0.9, 0.4 + 0.46291), places=4)

    def test_std_zero_degenerates(self):
        """std=0 (同 confidence) → 退化 0 → neutral"""
        votes = [
            AgentVote(role='a', decision='buy', confidence=0.6),
            AgentVote(role='b', decision='sell', confidence=0.6),
        ]
        out = zscore_equal_weight(votes)
        self.assertEqual(out['score'], 0.0)
        self.assertEqual(out['direction'], 'neutral')

    def test_empty_votes(self):
        """空票 → neutral/0"""
        out = zscore_equal_weight([])
        self.assertEqual(out['direction'], 'neutral')
        self.assertEqual(out['score'], 0.0)
        self.assertEqual(out['n_votes'], 0)

    def test_dict_votes_compatible(self):
        """dict 形状票 (决策历史恢复路径) 同样可算"""
        out = zscore_equal_weight([
            {'decision': 'buy', 'confidence': 0.9},
            {'decision': 'buy', 'confidence': 0.5},
        ])
        # confs [0.9,0.5]: mean=0.7, std(ddof=0)=0.2 → z=[1,-1]→relu[1,0]
        # score=(1×1+1×0)/2=0.5
        self.assertAlmostEqual(out['score'], 0.5, places=4)
        self.assertEqual(out['direction'], 'buy')

    def test_low_conf_vote_does_not_flip(self):
        """低于组均 confidence 的票弃权 (z 截 0), 不翻转成反向票"""
        out = zscore_equal_weight([
            AgentVote(role='a', decision='buy', confidence=0.9),
            AgentVote(role='b', decision='sell', confidence=0.3),
        ])
        self.assertEqual(out['direction'], 'buy')  # 弱空票不得变成加分空票


class TestDropoutDecide(unittest.TestCase):
    """TopkDropout 防抖语义"""

    def test_jitter_no_swap(self):
        """抖动不换: 候选 C(0.29) 夹在持仓 A/B 之间, 末位是候选 D → 无卖出无换入"""
        res = dropout_decide(
            holdings_scores={'A': {'score': 0.30, 'days_held': 5},
                             'B': {'score': 0.28, 'days_held': 5}},
            candidate_scores={'C': 0.29, 'D': 0.10},
            n_drop=1, hold_thresh=2)
        self.assertEqual(res['sells'], [])
        self.assertEqual(res['buys'], [])
        self.assertEqual(sorted(res['holds']), ['A', 'B'])

    def test_drops_out_bottom_swaps(self):
        """跌出末位换: B 垫底且持有 10d ≥ 2 → 卖 B 换入榜首候选 C"""
        res = dropout_decide(
            holdings_scores={'A': {'score': 0.9, 'days_held': 10},
                             'B': {'score': 0.05, 'days_held': 10}},
            candidate_scores={'C': 0.8, 'D': 0.7},
            n_drop=1, hold_thresh=2)
        self.assertEqual(res['sells'], ['B'])
        self.assertEqual(res['buys'], ['C'])
        self.assertEqual(res['holds'], ['A'])

    def test_hold_period_not_met(self):
        """持有期未满不换: B 垫底但只持有 1d < 2 → 保留, 无换入"""
        res = dropout_decide(
            holdings_scores={'B': {'score': 0.05, 'days_held': 1}},
            candidate_scores={'C': 0.8, 'D': 0.7, 'A': 0.6},
            n_drop=1, hold_thresh=2)
        self.assertEqual(res['sells'], [])
        self.assertEqual(res['buys'], [])
        self.assertEqual(res['holds'], ['B'])
        self.assertTrue(any('防抖' in n for n in res['notes']))

    def test_tradability_blocks_buy(self):
        """换入前过 tradability: C 涨停封板 → 跳过, 顺延补 D"""
        res = dropout_decide(
            holdings_scores={'A': {'score': 0.9, 'days_held': 10},
                             'sz300620': {'score': 0.05, 'days_held': 10}},
            candidate_scores={'sz300001': 0.8, 'sz300002': 0.7},
            n_drop=1, hold_thresh=2,
            market_data={'sz300001': {'pct_change': 0.20},   # 创业板涨停封板
                         'sz300002': {'pct_change': 0.02}})
        self.assertEqual(res['sells'], ['sz300620'])
        self.assertEqual(res['buys'], ['sz300002'])
        self.assertTrue(any('tradability' in n for n in res['notes']))


class TestTradability(unittest.TestCase):
    """板块化涨跌停阈值 + 可成交性"""

    def test_thresholds_by_board(self):
        self.assertEqual(limit_threshold_for('sh688981'), 0.20)  # 科创板
        self.assertEqual(limit_threshold_for('sz300620'), 0.20)  # 创业板
        self.assertEqual(limit_threshold_for('sz301001'), 0.20)  # 创业板
        self.assertEqual(limit_threshold_for('bj920001'), 0.30)  # 北交所 920
        self.assertEqual(limit_threshold_for('830799'), 0.30)    # 北交所 8xx
        self.assertEqual(limit_threshold_for('430047'), 0.30)    # 北交所 4xx
        self.assertEqual(limit_threshold_for('sh600519'), 0.10)  # 主板
        self.assertEqual(limit_threshold_for('sh600519', 'ST某某'), 0.05)  # 主板 ST
        self.assertEqual(limit_threshold_for('sz300620', 'ST某某'), 0.20)  # 创业板 ST 沿用板块

    def test_limit_up_blocks_buy(self):
        ok, reason = is_tradable('sh600519', 0.10, 'buy')
        self.assertFalse(ok)
        self.assertIn('涨停', reason)

    def test_limit_down_blocks_sell(self):
        ok, reason = is_tradable('sz300620', -0.20, 'sell')
        self.assertFalse(ok)
        self.assertIn('跌停', reason)

    def test_normal_trade_ok(self):
        self.assertTrue(is_tradable('sh600519', 0.05, 'buy')[0])
        self.assertTrue(is_tradable('sh600519', -0.05, 'sell')[0])
        self.assertTrue(is_tradable('sh600519', 0.05, 'hold')[0])

    def test_zero_volume_suspended(self):
        ok, reason = is_tradable('sh600519', 0.0, 'buy', volume=0)
        self.assertFalse(ok)
        self.assertIn('停牌', reason)

    def test_invalid_pct(self):
        ok, _ = is_tradable('sh600519', None, 'buy')
        self.assertFalse(ok)


class TestPnlCorr(unittest.TestCase):
    """信号相关 vs PnL 相关 (合成数据)"""

    def setUp(self):
        # 两只标的 6 根 K 线 → 决策日 01-01..01-05 各有次日收益
        self.klines = {
            's1': [{'date': f'2026-01-0{i}', 'close': c} for i, c in
                   zip(range(1, 7), [10, 11, 10.5, 11.5, 12, 11.4])],
            's2': [{'date': f'2026-01-0{i}', 'close': c} for i, c in
                   zip(range(1, 7), [10, 10.5, 11, 10.8, 11.3, 11.5])],
        }
        dates = [f'2026-01-0{i}' for i in range(1, 6)]

        def rows(direction, conf):
            return [(d, s, direction, conf) for d in dates for s in ('s1', 's2')]

        # tech 与 quant 完全同向 (PnL 序列成比例 → Pearson=1);
        # anti 与 tech 完全反向 (signal_corr=-1, pnl_corr=-1, gap=0)
        self.role_decisions = {
            'tech': rows('buy', 0.8),
            'quant': rows('buy', 0.6),
            'anti': rows('sell', 0.7),
        }

    def test_synthetic_pairs(self):
        result, reason = analyze(self.role_decisions, self.klines)
        self.assertIsNone(reason)
        self.assertIsNotNone(result)
        pairs = {(p['a'], p['b']): p for p in result['pairs']}
        self.assertEqual(len(pairs), 3)  # C(3,2)
        tq = pairs[('quant', 'tech')]
        self.assertAlmostEqual(tq['signal_corr'], 1.0, places=6)
        self.assertAlmostEqual(tq['pnl_corr'], 1.0, places=6)
        self.assertAlmostEqual(tq['gap'], 0.0, places=6)
        at = pairs[('anti', 'tech')]
        self.assertAlmostEqual(at['signal_corr'], -1.0, places=6)
        self.assertAlmostEqual(at['pnl_corr'], -1.0, places=6)
        for p in result['pairs']:
            for key in ('a', 'b', 'signal_corr', 'pnl_corr', 'gap'):
                self.assertIn(key, p)

    def test_insufficient_roles(self):
        result, reason = analyze({'solo': [('2026-01-01', 's1', 'buy', 0.8)]},
                                 self.klines)
        self.assertIsNone(result)
        self.assertIsNotNone(reason)

    def test_insufficient_klines(self):
        result, reason = analyze(self.role_decisions, {})
        self.assertIsNone(result)
        self.assertIsNotNone(reason)


class TestObserveDropoutFlag(unittest.TestCase):
    """decision_dropout_mode flag: off 静默 / observe 写 jsonl 不改输出"""

    def _bare_engine(self, mode):
        """绕过 __init__ (不碰 metrics/漂移集成), 只验观察路径"""
        eng = UnifiedDecisionEngine.__new__(UnifiedDecisionEngine)
        eng.decision_dropout_mode = mode
        return eng

    def test_off_returns_none(self):
        eng = self._bare_engine('off')
        self.assertIsNone(eng.observe_dropout({'A': {'score': 0.1, 'days_held': 5}},
                                              {'B': 0.9}))

    def test_observe_writes_jsonl(self):
        eng = self._bare_engine('observe')
        holdings = {'A': {'score': 0.9, 'days_held': 10},
                    'B': {'score': 0.05, 'days_held': 10}}
        with tempfile.TemporaryDirectory() as tmp:
            with patch('modules.unified_decision_engine.PROJECT_ROOT', tmp):
                res = eng.observe_dropout(holdings, {'C': 0.8})
            self.assertEqual(res['sells'], ['B'])
            self.assertEqual(res['buys'], ['C'])
            path = os.path.join(tmp, 'runs', 'consensus_ab',
                                'dropout_observe.jsonl')
            self.assertTrue(os.path.exists(path))
            with open(path, encoding='utf-8') as f:
                rec = json.loads(f.readline())
            self.assertEqual(rec['sells'], ['B'])
            self.assertEqual(rec['mode'], 'observe')


if __name__ == '__main__':
    unittest.main()
