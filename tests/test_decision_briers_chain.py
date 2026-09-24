# -*- coding:utf-8 -*-
"""P0-2 Brier 持久池链测试 (2026-9-09-22, 决策链 B 校准池重启清零修复)

背景: Brier/ECE 校准原料 (calib) 原由 _recalibrate_weights 从 _history 现算,
重启/截断清零 (09-22 实锤 n16 首算→重启→n0, role_gate/brier_ece 门恒饥饿)。
P0-2 = Brier 池 SQLite 化 (decision_briers, UNIQUE(code,ts) 防重) + recalibrate
以持久池为 calib 独立源, 新 resolution 增量入池 (UNIQUE+内存 seen 双锁)。

锁形 (expected 全来自独立事实来源):
  ① append→load 往返 (形/旧→新序/字段值)
  ② UNIQUE 防重: 同 (code,ts) 双写拒收 (replay 重跑防重喂, 09-22 IC 双喂教训同链)
  ③ 持久池直接喂 brier_ece (空 _history 也可): Brier 期望手算 20×0.01+4×0.81→/24=0.1433;
     重启形 (新实例同 db) 池不清零 = P0-2 核心目标锁
  ④ _history 新 resolution 入池 + 连续 recalibrate 不重喂 (seed+新 = 2, 二次仍 2)
  ⑤ 存储断链 ≠ 决策链死 (坏路径 → load=[]/append=0, 链降级)
"""
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.decision_history_store import DecisionHistoryStore  # noqa: E402
from modules.multi_agent_consensus import (  # noqa: E402
    MultiAgentConsensus, ConsensusResult)

_NOW = datetime.now()
_OLD_TS = (_NOW - timedelta(days=7)).strftime('%Y-%m-%dT%H:%M:%S')


def _filler(code='xx9999'):
    """凑数 history 项: 空票 → resolution 链跳过 (只凑 ic_min_history 门)"""
    return ConsensusResult(stock_code=code, consensus='buy', confidence=0.5,
                           vote_count={}, agent_votes=[], weights={},
                           conflict=False, conflict_ratio=0.0, timestamp=_OLD_TS)


class TestBriersStore(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self._tmp.close()
        self.db = self._tmp.name
        self.store = DecisionHistoryStore(self.db)

    def tearDown(self):
        for suffix in ('', '-wal', '-shm'):
            try:
                os.unlink(self.db + suffix)
            except OSError:
                pass

    def test_roundtrip(self):
        """① append→load: 返回入池数 + 旧→新序 + 字段值"""
        self.assertEqual(self.store.append_briers([
            ('sz300620', _OLD_TS, 0.7, 1),
            ('sh688981', _NOW.strftime('%Y-%m-%dT%H:%M:%S'), 0.6, 0),
        ]), 2)
        got = self.store.load_briers()
        self.assertEqual([r['code'] for r in got], ['sz300620', 'sh688981'])
        self.assertEqual(got[0]['outcome'], 1)
        self.assertAlmostEqual(got[1]['p'], 0.6)

    def test_unique_dedup(self):
        """② 同 (code,ts) 双写 = 第二次 0 行 (replay 重跑防重喂锁)"""
        rows = [('sz300620', _OLD_TS, 0.7, 1)]
        self.assertEqual(self.store.append_briers(rows), 1)
        self.assertEqual(self.store.append_briers(rows), 0)
        self.assertEqual(len(self.store.load_briers()), 1)

    def test_corrupt_db_degrade(self):
        """⑤ 存储断链≠链死: 坏路径 → load=[] / append=0 (降级纯内存)"""
        bad = DecisionHistoryStore('/nonexistent_dir_xyz/a.db')
        self.assertEqual(bad.load_briers(), [])
        self.assertEqual(bad.append_briers([('sz300620', _OLD_TS, 0.5, 1)]), 0)


class TestRecalibrateBrierPool(unittest.TestCase):
    """③④ recalibrate 消费链 (持久池为 calib 源 + 增量入池)"""

    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self._tmp.close()
        self.db = self._tmp.name

    def tearDown(self):
        for suffix in ('', '-wal', '-shm'):
            try:
                os.unlink(self.db + suffix)
            except OSError:
                pass

    def test_pool_feeds_and_survives_restart(self):
        """③ 种子池 24 对 (无新 resolution) → Brier 真值; 重启形池不清零
        期望独立来源: 20×(0.9-1)² + 4×(0.9-0)² = 3.44/24 = 0.1433 (手算)"""
        store = DecisionHistoryStore(self.db)
        self.assertEqual(store.append_briers(
            [(f'sz{i:06d}',
              (_NOW - timedelta(days=5 + i % 3)).strftime('%Y-%m-%dT%H:%M:%S'),
              0.9, 1 if i < 20 else 0) for i in range(24)]), 24)

        def _fresh(store_):
            eng = MultiAgentConsensus()
            eng._store = store_  # 替换默认 store: 测试永不触碰生产 db
            eng._history = [_filler() for _ in range(20)]
            return eng

        eng = _fresh(store)
        with patch('modules.data_fetcher.StockDataFetcher') as F:
            F.return_value.get_kline_data.return_value = None
            # ③ 真值: 空 _history 持久池也喂 → n24 (原形 n0 skipped 反 tautological)
            eng._recalibrate_weights()
            rep = eng._last_calibration
            self.assertNotIn('skipped', rep)
            self.assertEqual(rep['n'], 24)
            self.assertAlmostEqual(rep['brier'], 0.1433, places=4)
            # ③b 重启形: 新实例 (独立 store 同 db) 池仍 24 → 重启不清零
            eng2 = _fresh(DecisionHistoryStore(self.db))
            eng2._recalibrate_weights()
            self.assertEqual(eng2._last_calibration.get('n'), 24,
                             'Brier 池须跨重启持久 (P0-2 核心)')

    def test_new_resolution_append_dedup(self):
        """④ 新 resolution 入池 (seed+新=2); 连续两次 recalibrate 不重喂 (双锁)"""
        store = DecisionHistoryStore(self.db)
        store.append_briers([('sz000001', _OLD_TS, 0.9, 1)])  # 种子的 (code,ts)
        h = ConsensusResult(stock_code='sz300620', consensus='buy',
                            confidence=0.8, vote_count={},
                            agent_votes=[{'role': 'fundamental',
                                          'decision': 'buy', 'confidence': 0.7}],
                            weights={}, conflict=False, conflict_ratio=0.0,
                            timestamp=_OLD_TS)
        eng = MultiAgentConsensus()
        eng._store = store
        eng._history = [_filler()] * 19 + [h]  # ≥20 过 ic_min_history 门

        # K线: 决策日 (7d 前) 收盘=10 → 末根收盘=11 (+10%, buy→hit)
        dates = [(_NOW - timedelta(days=d)).strftime('%Y-%m-%d')
                 for d in range(9, -1, -1)]
        dates += [(_NOW + timedelta(days=d)).strftime('%Y-%m-%d')
                  for d in range(1, 13)]
        klines = ([{'date': d, 'close': 10.0} for d in dates[:10]]
                  + [{'date': d, 'close': 11.0} for d in dates[10:]])

        with patch('modules.data_fetcher.StockDataFetcher') as F:
            F.return_value.get_kline_data.return_value = klines
            eng._recalibrate_weights()
            first = len(store.load_briers())
            eng._recalibrate_weights()  # replay 重跑形 (同日二次触发)
            second = len(store.load_briers())
        self.assertEqual(first, 2, '种子 1 + 新 resolution 1 应入池')
        self.assertEqual(second, 2, '二次 recalibrate 不得重喂 (UNIQUE+seen 双锁)')


if __name__ == '__main__':
    unittest.main(verbosity=2)
