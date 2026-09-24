# -*- coding: utf-8 -*-
"""2026-09-22 锁链: 预测评估面板接通 — 评估产生链 (fusion_routes + fusion_ledger)

背景: 面板三 item (准确率/Brier/漂移) 恒「尚无评估数据」= record_evaluation
全工程零调用方 (Phase 3 半成品)。09-22 接通: 23:10 replay 链尾 evaluate 段 =
①≥3 交易日老种子 90d K 对账 → record_evaluation ②监控池 predict 产新种子。
锁形: 对账/无方向跳过/单票坏≠链死/predict 形制/台账往返。
"""
import os
import tempfile
import unittest
from unittest.mock import MagicMock

from modules.fusion_ledger import FusionLedger
from modules.routes.fusion_routes import _resolve_eval_seeds, _spawn_eval_seeds

POOL = 'sz300620,sh688981'


def _kline(p0: float, p1: float):
    """90d 假 K: 倒数第 10 根收盘=p0 (种子对账日), 最后收盘=p1"""
    kl = [{'date': f'2026-09-{d:02d}', 'close': p0} for d in range(1, 22)]
    kl += [{'date': f'2026-09-{d:02d}', 'close': p1} for d in range(23, 32)]
    return kl


def _seed(code='sz300620', ts='2026-09-15 10:00:00'):
    return {'id': 1, 'ts': ts, 'code': code, 'direction': 'up', 'confidence': 0.6}


class TestResolveEvalSeeds(unittest.TestCase):
    """对账段: 老种子 → 真实收益 → record_evaluation (09-22 接通锁)"""

    def test_resolve_success_flow(self):
        """+10% 收益 → resolved + 落盘 + 标记 (单种链全链形)"""
        ledger = MagicMock()
        ledger.list_pending_seeds.return_value = [_seed()]
        fetcher = MagicMock()
        fetcher.get_kline_data.return_value = _kline(10.0, 11.0)
        fusion = MagicMock()
        fusion.record_evaluation.return_value = {'brier': 0.1, 'correct': True,
                                                 'pred_direction': 'up',
                                                 'actual_return': 0.1,
                                                 'timestamp': '2026-09-22T00:00:00'}
        out = _resolve_eval_seeds(fusion, fetcher, ledger)
        self.assertEqual(out['resolved'], 1)
        self.assertEqual(out['errors'], [])
        ledger.mark_resolved.assert_called_once_with(1)
        # 种子原形传给 record_evaluation (dict 含 direction/confidence)
        args = fusion.record_evaluation.call_args.args
        self.assertEqual(args[0]['direction'], 'up')
        self.assertAlmostEqual(args[1], 0.1, places=3)

    def test_noise_ret_skipped(self):
        """|ret|<0.005 无方向 → 只标记不写 (consensus 同形, 防重扫堆积)"""
        ledger = MagicMock()
        ledger.list_pending_seeds.return_value = [_seed()]
        fetcher = MagicMock()
        fetcher.get_kline_data.return_value = _kline(10.0, 10.02)
        fusion = MagicMock()
        out = _resolve_eval_seeds(fusion, fetcher, ledger)
        self.assertEqual(out['noise'], 1)
        fusion.record_evaluation.assert_not_called()
        ledger.mark_resolved.assert_called_once_with(1)

    def test_single_broken_keeps_chain(self):
        """一种 K 坏 → 记错不拖另一种 (链坏≠链死)"""
        s1, s2 = _seed('sz300620'), _seed('sh688981', ts='2026-09-15 11:00:00')
        s2['id'] = 2
        ledger = MagicMock()
        ledger.list_pending_seeds.return_value = [s1, s2]
        fetcher = MagicMock()
        fetcher.get_kline_data.side_effect = Exception('AKShare 断')
        fusion = MagicMock()
        out = _resolve_eval_seeds(fusion, fetcher, ledger)
        self.assertEqual(out['resolved'], 0)
        self.assertEqual(len(out['errors']), 2)
        self.assertIn('AKShare', out['errors'][0]['error'])


class TestSpawnEvalSeeds(unittest.TestCase):
    """种子段: 监控池 predict → 落盘 (23:10 链形锁)"""

    def test_spawn_all_and_unit(self):
        """predict 单位置参 + direction 空不入库 (诚实)"""
        ledger = MagicMock()
        fusion = MagicMock()
        fusion.predict.side_effect = lambda c: (
            {'stock_code': c, 'direction': None, 'confidence': 0.5}
            if c == 'sh688981' else
            {'stock_code': c, 'direction': 'up', 'confidence': 0.6})
        out = _spawn_eval_seeds(fusion, ledger, POOL)
        self.assertEqual(out['spawned'], ['sz300620'])
        self.assertEqual(len(out['errors']), 1)  # 无 direction 种诚实标错

    def test_spawn_broken_keeps_chain(self):
        """一票 predict 抛 → 记错续跑其余 (链坏≠链死)"""
        ledger = MagicMock()
        fusion = MagicMock()

        def _pred(code):
            if code == 'sh688981':
                raise TimeoutError('kline 链超时')
            return {'stock_code': code, 'direction': 'down', 'confidence': 0.55}
        fusion.predict.side_effect = _pred
        out = _spawn_eval_seeds(fusion, ledger, POOL)
        self.assertEqual(out['spawned'], ['sz300620'])
        self.assertEqual(len(out['errors']), 1)


class TestFusionLedger(unittest.TestCase):
    """FusionLedger: SQLite 往返形 (恢复链依赖)"""

    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self._tmp.close()
        self.ledger = FusionLedger(self._tmp.name)

    def tearDown(self):
        os.unlink(self._tmp.name)

    def test_seed_pending_and_dedupe(self):
        """种子: 当日不入选 (≥3 日门控) + 同日同票防重 + 标记后不再入 pending"""
        r = {'stock_code': 'sz300620', 'direction': 'up', 'confidence': 0.6}
        self.assertTrue(self.ledger.append_seed(r))
        self.assertFalse(self.ledger.append_seed(r))  # UNIQUE(code, ts) 防重
        # 直插老 ts (5 天前) → pending 可见
        with self.ledger._conn() as c:
            c.execute("INSERT INTO seeds (ts, code, direction, confidence) "
                      "VALUES ('2026-09-15 15:00:00', 'sh688981', 'down', 0.55)")
            c.commit()
        pend = self.ledger.list_pending_seeds(min_age_days=3)
        self.assertEqual([s['code'] for s in pend], ['sh688981'])
        pend_now = self.ledger.list_pending_seeds(min_age_days=0)
        self.assertEqual(len(pend_now), 2)  # 当日种子 3 天窗内也可 (手动模式)

    def test_evals_roundtrip_restore_shape(self):
        """评估落盘→读出 = _eval_history 恢复形 (重启不清零的链)"""
        ev = {'timestamp': '2026-09-22T00:00:00', 'correct': True,
              'brier': 0.12, 'pred_direction': 'up', 'actual_return': 0.02}
        self.assertTrue(self.ledger.append_eval(ev, 'sz300620'))
        loaded = self.ledger.load_evals()
        self.assertEqual(len(loaded), 1)
        self.assertEqual(set(loaded[0].keys()),
                         {'timestamp', 'correct', 'brier', 'pred_direction',
                          'actual_return'})
        self.assertTrue(loaded[0]['correct'])


if __name__ == '__main__':
    unittest.main()
