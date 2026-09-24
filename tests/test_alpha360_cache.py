#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""test_alpha360_cache — Alpha360 语义缓存旁路单测 (P1-5, 2026-09-15)

覆盖: ①缓存结果 == 直算 (全窗口 lookback → 命中返回/变化全量重算)
②同数据二次调用命中 ③盘中 close 变动 → 不命中过期缓存 ④killswitch 全关
⑤无代码跳过缓存。零网络: 合成随机游走 K 线 + 临时缓存目录。
"""

import random
import shutil
import tempfile
import unittest

from modules.factors import factor_cache as fc
from modules.factors.alpha360_calculator import Alpha360Calculator


def _bars(n, seed=42):
    """确定性随机游走 K 线 (零网络, 可复现)"""
    rng = random.Random(seed)
    bars, price = [], 100.0
    for i in range(n):
        o = price
        c = o * (1 + rng.uniform(-0.03, 0.03))
        h = max(o, c) * (1 + rng.uniform(0, 0.01))
        l = min(o, c) * (1 - rng.uniform(0, 0.01))
        bars.append({'date': f'2026-{1 + i // 28:02d}-{1 + i % 28:02d}',
                     'open': o, 'high': h, 'low': l, 'close': c,
                     'volume': 1e6 + i * 1000})
        price = c
    return bars


class Alpha360CacheTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='a360_cache_')
        self._orig_dir = fc.CACHE_DIR
        self._orig_en = fc._ENABLED
        fc.CACHE_DIR = self.tmp
        fc._ENABLED = True
        fc._stats.update(hits=0, misses=0, incremental=0, full_recompute=0,
                         fallback=0)
        self.calc = Alpha360Calculator()

    def tearDown(self):
        fc.CACHE_DIR = self._orig_dir
        fc._ENABLED = self._orig_en
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_cached_matches_uncached(self):
        """缓存路径结果必须与非缓存直算一致 (全窗口 lookback = 语义无损)"""
        kl = _bars(120)
        direct = self.calc.calculate_all(kl)
        cached = self.calc.calculate_all(kl, stock_code='sz300620',
                                         use_cache=True)
        self.assertGreater(len(direct), 100)
        self.assertEqual(set(direct), set(cached))
        for k in direct:
            self.assertAlmostEqual(direct[k], cached[k], delta=1e-9, msg=k)

    def test_second_call_hits_cache(self):
        kl = _bars(120)
        self.calc.calculate_all(kl, stock_code='sz300620', use_cache=True)
        cached2 = self.calc.calculate_all(kl, stock_code='sz300620',
                                          use_cache=True)
        direct = self.calc.calculate_all(kl)
        self.assertEqual(set(cached2), set(direct))
        s = fc.stats()
        self.assertEqual(s['misses'], 1)
        self.assertEqual(s['hits'], 1)

    def test_intraday_close_change_not_stale_hit(self):
        """盘中当日 K 线 close 变动 → 指纹变化, 绝不返回过期值 (09-15 补丁核心)"""
        kl = _bars(120)
        self.calc.calculate_all(kl, stock_code='sz300620', use_cache=True)
        kl2 = [dict(b) for b in kl]
        kl2[-1]['close'] *= 1.02  # 模拟当日未收盘 K 线价格漂移
        v = self.calc.calculate_all(kl2, stock_code='sz300620', use_cache=True)
        s = fc.stats()
        self.assertEqual(s['hits'], 0)           # 不允许过期命中
        self.assertEqual(s['full_recompute'], 1)  # 数据变化 → 全量重算
        direct = self.calc.calculate_all(kl2)
        self.assertAlmostEqual(direct['macd_hist'], v['macd_hist'], delta=1e-9)

    def test_no_stock_code_skips_cache(self):
        """无代码 → 无法作缓存键, 必须走直算 (与 alpha158 行为一致)"""
        kl = _bars(60)
        v = self.calc.calculate_all(kl, use_cache=True)
        self.assertGreater(len(v), 100)
        self.assertEqual(fc.stats()['misses'], 0)

    def test_killswitch_bypass(self):
        """FACTOR_SEMANTIC_CACHE killswitch → 完全绕过缓存层"""
        fc._ENABLED = False
        kl = _bars(60)
        v = self.calc.calculate_all(kl, stock_code='sz300620', use_cache=True)
        self.assertGreater(len(v), 100)
        self.assertEqual(fc.stats()['misses'], 0)  # 未触缓存层


if __name__ == '__main__':
    unittest.main()
