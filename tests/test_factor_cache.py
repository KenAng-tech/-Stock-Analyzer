#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""factor_cache 单元测试 (P1-5) — 零网络, 临时缓存目录"""

import json
import os
import shutil
import tempfile
import unittest

from modules.factors import factor_cache as fc


def _bars(n, start_day=1):
    return [{'date': f'2026-09-{start_day + i:02d}', 'close': 10.0 + i} for i in range(n)]


class FactorCacheTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='fcache_test_')
        self._orig_dir = fc.CACHE_DIR
        fc.CACHE_DIR = self.tmp
        fc._stats.update(hits=0, misses=0, incremental=0, full_recompute=0, fallback=0)

    def tearDown(self):
        fc.CACHE_DIR = self._orig_dir
        shutil.rmtree(self.tmp, ignore_errors=True)

    # ── 键规范化 ──
    def test_key_normalization(self):
        k1 = fc.expr_key('Mean( $Close , 20 )', 'SZ300620')
        k2 = fc.expr_key('mean($close,20)', 'sz300620')
        self.assertEqual(k1, k2)

    def test_key_diff_code_freq(self):
        self.assertNotEqual(fc.expr_key('x', 'sz300620'), fc.expr_key('x', 'sh688981'))
        self.assertNotEqual(fc.expr_key('x', 'sz300620', 'day'), fc.expr_key('x', 'sz300620', 'week'))

    # ── 指纹 ──
    def test_fingerprint_fields(self):
        fp = fc.klines_fingerprint(_bars(10))
        self.assertEqual(fp['first'], '2026-09-01')
        self.assertEqual(fp['n'], 10)
        self.assertEqual(len(fp['hash']), 16)

    def test_fingerprint_empty(self):
        self.assertEqual(fc.klines_fingerprint([])['hash'], 'empty')

    def test_fingerprint_detects_close_change(self):
        """盘中当日未收盘 K 线 close 变动 → 指纹必须变化 (防过期命中)"""
        fp1 = fc.klines_fingerprint(_bars(10))
        bars2 = [dict(b) for b in _bars(10)]
        bars2[-1]['close'] = bars2[-1]['close'] + 0.5
        self.assertNotEqual(fc.klines_fingerprint(bars2)['hash'], fp1['hash'])

    # ── 命中/未命中 ──
    def test_miss_then_hit(self):
        calls = []
        bars = _bars(20)

        def fn(k):
            calls.append(len(k))
            return {'f': [float(len(k))]}

        v1 = fc.get_or_compute('Expr A', 'sz300620', bars, fn)
        self.assertEqual(v1['f'], [20.0])
        v2 = fc.get_or_compute('Expr A', 'sz300620', bars, fn)
        self.assertEqual(v2['f'], [20.0])
        self.assertEqual(len(calls), 1)  # 第二次命中不重算
        self.assertEqual(fc.stats()['hits'], 1)

    def test_full_recompute_on_first_date_change(self):
        calls = []

        def fn(k):
            calls.append(1)
            return {'f': [1.0] * len(k)}

        fc.get_or_compute('Expr B', 'sz300620', _bars(10, start_day=1), fn)
        fc.get_or_compute('Expr B', 'sz300620', _bars(10, start_day=5), fn)
        self.assertEqual(len(calls), 2)
        self.assertEqual(fc.stats()['full_recompute'], 1)

    # ── 增量 ──
    def test_incremental_only_tail(self):
        seen_slices = []

        def fn(k):
            seen_slices.append(len(k))
            return {'f': [float(i) for i in range(len(k))]}

        old = _bars(20)
        fc.get_or_compute('Expr C', 'sz300620', old, fn)          # 全量 20
        new = _bars(23)                                            # 尾部 +3
        v = fc.get_or_compute('Expr C', 'sz300620', new, fn, overlap=3, lookback=0)
        self.assertEqual(seen_slices, [20, 6])                     # 第二次只算尾部 6 根
        self.assertEqual(fc.stats()['incremental'], 1)
        self.assertEqual(len(v['f']), 23)                          # 20-3 保留 + 6 并入? 见下
        # 对齐验证: 保留 17 + 新日期 6 (cut_idx=17, klines[17:] 共 6 根) = 23
        self.assertEqual(v['f'][:17], [float(i) for i in range(17)])

    def test_incremental_with_lookback(self):
        seen = []

        def fn(k):
            seen.append(len(k))
            return {'f': [0.0] * len(k)}

        fc.get_or_compute('Expr D', 'sz300620', _bars(20), fn)
        fc.get_or_compute('Expr D', 'sz300620', _bars(23), fn, overlap=3, lookback=10)
        self.assertEqual(seen, [20, 16])  # cut_idx=17, slice_start=7 → 16 根

    def test_scalar_values_passthrough(self):
        def fn(k):
            return {'latest': 42.0}

        v = fc.get_or_compute('Expr E', 'sz300620', _bars(5), fn)
        self.assertEqual(v['latest'], 42.0)
        v2 = fc.get_or_compute('Expr E', 'sz300620', _bars(5), fn)
        self.assertEqual(v2['latest'], 42.0)

    # ── 故障透明 ──
    def test_corrupt_file_fallback(self):
        bars = _bars(10)
        fc.get_or_compute('Expr F', 'sz300620', bars, lambda k: {'f': [1.0]})
        key = fc.expr_key('Expr F', 'sz300620')
        with open(fc._path(key), 'w') as f:
            f.write('{broken json')
        calls = []

        def fn(k):
            calls.append(1)
            return {'f': [2.0]}

        v = fc.get_or_compute('Expr F', 'sz300620', bars, fn)
        self.assertEqual(v['f'], [2.0])   # 透明回源
        self.assertEqual(len(calls), 1)
        self.assertTrue(any(x.endswith('.corrupt.1') or '.corrupt.' in x
                            for x in os.listdir(self.tmp)))

    def test_compute_fn_error_propagates(self):
        def bad(k):
            raise ValueError('boom')

        with self.assertRaises(ValueError):
            fc.get_or_compute('Expr G', 'sz300620', _bars(5), bad)

    def test_hash_string_degrades_to_full(self):
        calls = []

        def fn(k):
            calls.append(1)
            return {'f': [1.0]}

        fc.get_or_compute('Expr H', 'sz300620', 'hash-aaa', fn)
        fc.get_or_compute('Expr H', 'sz300620', 'hash-bbb', fn)
        self.assertEqual(len(calls), 2)  # 无日期信息 → 全量重算

    # ── killswitch (FACTOR_SEMANTIC_CACHE) ──
    def test_killswitch_bypass(self):
        calls = []

        def fn(k):
            calls.append(1)
            return {'f': [1.0]}

        bars = _bars(10)
        fc.get_or_compute('Expr L', 'sz300620', bars, fn)  # 预热缓存
        fc._ENABLED = False
        try:
            v = fc.get_or_compute('Expr L', 'sz300620', bars, fn)
        finally:
            fc._ENABLED = True
        self.assertEqual(v['f'], [1.0])
        self.assertEqual(len(calls), 2)  # killswitch 下不读缓存, compute 恒被直调

    # ── stats / clear ──
    def test_stats_counts(self):
        bars = _bars(10)
        fc.get_or_compute('Expr I', 'sz300620', bars, lambda k: {'f': [1.0]})
        fc.get_or_compute('Expr I', 'sz300620', bars, lambda k: {'f': [1.0]})
        s = fc.stats()
        self.assertEqual(s['misses'], 1)
        self.assertEqual(s['hits'], 1)
        self.assertEqual(s['entries'], 1)
        self.assertGreater(s['hit_rate'], 0)

    def test_clear_removes_stale(self):
        bars = _bars(10)
        fc.get_or_compute('Expr J', 'sz300620', bars, lambda k: {'f': [1.0]})
        self.assertEqual(fc.clear(stale_days=30), 0)       # 刚写入不删
        self.assertEqual(fc.clear(stale_days=0), 1)        # 0 天阈值全删
        self.assertEqual(fc.stats()['entries'], 0)

    def test_atomic_save_no_tmp_leftover(self):
        fc.get_or_compute('Expr K', 'sz300620', _bars(5), lambda k: {'f': [1.0]})
        self.assertFalse(any(f.endswith('.tmp') for f in os.listdir(self.tmp)))


if __name__ == '__main__':
    unittest.main()
