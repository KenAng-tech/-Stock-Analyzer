# -*- coding: utf-8 -*-
"""2026-09-21 锁链: PatchTST 时序输入形制 (截面 (8,1) 恒抛 → 窗口形修复)

背景: analysis_engine 旧形把 prepare_features 截面直喂 predict →
'[1, 1, 96]' is invalid for input of size 8 恒抛 (每 analyze 2 WARNING +
集成静默缺 patchtst)。修复 = build_seq_features_12 (端点实证形移植)。
"""
import unittest

import numpy as np

from modules.models.patchtst_integrator import (
    HAS_TORCH, build_seq_features_12, get_patchtst)


def _random_walk(n: int = 200, seed: int = 42):
    rng = np.random.default_rng(seed)
    return list(10.0 * np.cumprod(1 + rng.normal(0, 0.02, n)))


class TestBuildSeqFeatures(unittest.TestCase):
    """build_seq_features_12: 形制/守卫 (09-21 修复锁)"""

    def test_shape_and_no_nan(self):
        """100 K → (60,12) 全有限 (修复前此形直喂 predict 恒抛)"""
        X = build_seq_features_12(_random_walk(100), [1e6] * 100, seq_len=60)
        self.assertIsNotNone(X)
        self.assertEqual(X.shape, (60, 12))
        self.assertTrue(np.isfinite(X).all(), "特征矩阵不得含 NaN/Inf")

    def test_short_kline_returns_none(self):
        """短 K (<30) 守卫 → None (诚实跳过, 不喂半窗)"""
        self.assertIsNone(build_seq_features_12(_random_walk(10), [1e6] * 10))

    def test_none_volumes_zero_fallback(self):
        """无 volume → 0 兜底 (量能特征走 0, 非抛非 None)"""
        X = build_seq_features_12(_random_walk(80), None, seq_len=60)
        self.assertIsNotNone(X)
        self.assertEqual(X.shape, (60, 12))

    @unittest.skipUnless(HAS_TORCH, 'torch 不可用')
    def test_predict_accepts_built_window(self):
        """predict 全链锁: 窗口形喂进模型不抛 (旧截面 (8,1) 在此抛 '[1,1,96]')"""
        pt = get_patchtst()
        X = build_seq_features_12(_random_walk(200), [1e6] * 200)
        r = pt.predict(X)
        self.assertIsInstance(r, dict)
        self.assertIn('direction', r)
        self.assertIn(r['direction'], ('up', 'neutral', 'down'))


if __name__ == '__main__':
    unittest.main()
