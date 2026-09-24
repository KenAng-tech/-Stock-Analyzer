#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""PIT 仓库 + 监管标签 + 换手评分 单元测试 (P0-1 / P2-12a/b) — 零网络"""

import os
import tempfile
import unittest

from modules import pit_store
from modules.regulatory_tags import tag_dragon_tiger
from modules.turnover_scoring import turnover_score, liquidity_grade


class PitStoreTest(unittest.TestCase):
    def setUp(self):
        fd, self.db = tempfile.mkstemp(suffix='.db')
        os.close(fd)

    def tearDown(self):
        d = os.path.dirname(self.db)
        base = os.path.basename(self.db)
        if os.path.exists(self.db):
            os.remove(self.db)
        for f in os.listdir(d):
            if f.startswith(base) and '.corrupt.' in f:
                os.remove(os.path.join(d, f))

    def test_put_and_latest(self):
        rid = pit_store.put('300620', 'roe', 15.2, period_end='2026-06-30',
                            ann_date='2026-08-20', db_path=self.db)
        self.assertIsNotNone(rid)
        rec = pit_store.latest('300620', 'roe', db_path=self.db)
        self.assertAlmostEqual(rec['value'], 15.2)

    def test_asof_boundary_inclusive(self):
        pit_store.put('300620', 'roe', 15.2, '2026-06-30', '2026-08-20', db_path=self.db)
        self.assertIsNotNone(pit_store.get_as_of('300620', 'roe', '2026-08-20', db_path=self.db))  # 边界可见

    def test_asof_rejects_future(self):
        pit_store.put('300620', 'roe', 15.2, '2026-06-30', '2026-08-20', db_path=self.db)
        self.assertIsNone(pit_store.get_as_of('300620', 'roe', '2026-08-19', db_path=self.db))

    def test_missing_ann_date_invisible_fail_closed(self):
        pit_store.put('300620', 'roe', 99.0, '2026-06-30', None, db_path=self.db)
        self.assertIsNone(pit_store.get_as_of('300620', 'roe', '2026-12-31', db_path=self.db))
        self.assertIsNotNone(pit_store.latest('300620', 'roe', db_path=self.db))  # latest 可见

    def test_revision_chain_asof_picks_latest_visible(self):
        r1 = pit_store.put('688981', 'revenue', 100.0, '2026-06-30', '2026-08-01', db_path=self.db)
        pit_store.put('688981', 'revenue', 95.0, '2026-06-30', '2026-09-15',
                      revision_of=r1, db_path=self.db)
        # 8 月只看原版
        rec = pit_store.get_as_of('688981', 'revenue', '2026-08-31', db_path=self.db)
        self.assertAlmostEqual(rec['value'], 100.0)
        self.assertIsNone(rec['revision_of'])
        # 9/15 后看修订版
        rec2 = pit_store.get_as_of('688981', 'revenue', '2026-09-20', db_path=self.db)
        self.assertAlmostEqual(rec2['value'], 95.0)
        self.assertEqual(rec2['revision_of'], r1)

    def test_history_order(self):
        pit_store.put('300502', 'np', 1.0, '2025-12-31', '2026-03-01', db_path=self.db)
        pit_store.put('300502', 'np', 2.0, '2026-06-30', '2026-08-15', db_path=self.db)
        h = pit_store.get_history('300502', 'np', db_path=self.db)
        self.assertEqual([x['ann_date'] for x in h], ['2026-03-01', '2026-08-15'])

    def test_corrupt_db_self_heal(self):
        pit_store.put('300620', 'roe', 1.0, '2026-06-30', '2026-08-01', db_path=self.db)
        with open(self.db, 'w') as f:
            f.write('not a sqlite file at all' * 100)
        rec = pit_store.latest('300620', 'roe', db_path=self.db)  # 自愈后空库 → None
        self.assertIsNone(rec)
        rid = pit_store.put('300620', 'roe', 7.0, '2026-06-30', '2026-08-01', db_path=self.db)
        self.assertIsNotNone(rid)  # 重建后可写

    def test_stats(self):
        pit_store.put('300620', 'roe', 1.0, '2026-06-30', '2026-08-01', db_path=self.db)
        pit_store.put('300620', 'roe', 2.0, '2026-06-30', None, db_path=self.db)
        s = pit_store.stats(db_path=self.db)
        self.assertEqual(s['total'], 2)
        self.assertEqual(s['asof_visible'], 1)

    def test_text_value(self):
        pit_store.put('300620', 'audit_opinion', '标准无保留', '2026-06-30',
                      '2026-08-20', db_path=self.db)
        rec = pit_store.get_as_of('300620', 'audit_opinion', '2026-09-01', db_path=self.db)
        self.assertEqual(rec['value'], '标准无保留')


class RegulatoryTagsTest(unittest.TestCase):
    def _rec(self, **kw):
        base = {'code': 'sz300620', 'pct_change': 0.02, 'turnover': 0.05,
                'net_buy': 0.0, 'amount': 1e8, 'consecutive_days': 1,
                'high': 10.3, 'low': 9.8, 'prev_close': 10.0}
        base.update(kw)
        return base

    def test_tags_always_four(self):
        tags = tag_dragon_tiger(self._rec())
        self.assertEqual(len(tags), 4)
        self.assertTrue(all('tag' in t and 'hit' in t and 'evidence' in t for t in tags))

    def test_normal_record_no_hits(self):
        self.assertFalse(any(t['hit'] for t in tag_dragon_tiger(self._rec())))

    def test_deviation_hit_chi_next_20pct(self):
        tags = {t['tag']: t['hit'] for t in tag_dragon_tiger(self._rec(pct_change=0.17))}
        self.assertTrue(tags['涨跌幅偏离'])  # 0.17 ≥ 0.20×0.8

    def test_main_board_11pct_hits(self):
        tags = {t['tag']: t['hit'] for t in tag_dragon_tiger(self._rec(code='sh600000', pct_change=0.11))}
        self.assertTrue(tags['涨跌幅偏离'])  # 0.11 ≥ 0.10×0.8

    def test_pump_tag(self):
        tags = {t['tag']: t['hit'] for t in tag_dragon_tiger(
            self._rec(pct_change=0.19, net_buy=4e7, amount=1e8))}
        self.assertTrue(tags['拉抬打压'])

    def test_dump_tag(self):
        tags = {t['tag']: t['hit'] for t in tag_dragon_tiger(
            self._rec(pct_change=-0.19, net_buy=-4e7, amount=1e8))}
        self.assertTrue(tags['拉抬打压'])

    def test_severe_by_consecutive(self):
        tags = {t['tag']: t['hit'] for t in tag_dragon_tiger(self._rec(consecutive_days=3))}
        self.assertTrue(tags['严重异常波动'])

    def test_fake_ordering_proxy(self):
        tags = {t['tag']: t['hit'] for t in tag_dragon_tiger(
            self._rec(turnover=0.35, high=11.2, low=9.5, prev_close=10.0))}
        self.assertTrue(tags['虚假申报(代理)'])  # 换手35% + 振幅17%

    def test_missing_fields_safe(self):
        tags = tag_dragon_tiger({'code': 'sz300620'})
        self.assertEqual(len(tags), 4)


class TurnoverScoringTest(unittest.TestCase):
    def test_percentile_method(self):
        hist = [0.01] * 40 + [0.20] * 10  # 50 样本, 80% ≤ 0.05
        self.assertAlmostEqual(turnover_score(0.05, hist), 80.0)

    def test_percentile_extremes(self):
        hist = [0.02] * 30
        self.assertEqual(turnover_score(0.01, hist), 0.0)
        self.assertEqual(turnover_score(0.5, hist), 100.0)

    def test_band_method_low(self):
        s = turnover_score(0.005)  # <1% → 档内 0-20
        self.assertLess(s, 20)

    def test_band_method_high(self):
        self.assertGreaterEqual(turnover_score(0.15), 70)  # >10% 落在 70-90 档下界

    def test_invalid_inputs(self):
        self.assertEqual(turnover_score(None), 0.0)
        self.assertEqual(turnover_score('abc'), 0.0)
        self.assertEqual(turnover_score(-0.1), 0.0)

    def test_short_history_falls_back_to_bands(self):
        self.assertEqual(turnover_score(0.05, [0.01, 0.02]),
                         turnover_score(0.05))

    def test_grades(self):
        self.assertEqual(liquidity_grade(85), '高')
        self.assertEqual(liquidity_grade(50), '中')
        self.assertEqual(liquidity_grade(30), '低')
        self.assertEqual(liquidity_grade(5), '劣')
        self.assertEqual(liquidity_grade(None), '劣')


if __name__ == '__main__':
    unittest.main()
