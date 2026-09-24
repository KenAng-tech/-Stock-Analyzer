#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
selection-bias 诊断测试 (2026-09-11 #30)

锁链 (09-10/09-11 同风格 — 真实链形态锁进测试, 不盲改):
  1. selection_bias_check 数学形: deflation γ=Φ⁻¹(1-1/N)·√(2/N), deflated=mean−γ·std
  2. guard: n<2 skipped (跨 trial 方差无定义) + 非有限值过滤
  3. Orchestrator 端到端: Optuna study trials 采集 → summary (Optuna 未装 = 网格回退跳 study)

运行: python -m unittest tests.test_selection_bias
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.hyperparam_optimizer import (
    selection_bias_check, HyperParamOrchestrator,
)

try:
    import optuna  # noqa: F401
    OPTUNA = True
except ImportError:
    OPTUNA = False


class TestSelectionBiasCheck(unittest.TestCase):
    """deflated SR/IC 数学 + guard"""

    def test_noise_chain_flags_suspect(self):
        """噪声 trial (mean≈0, 离散度大) → deflated<0 → selection_suspect=True"""
        vals = list(np.random.RandomState(7).randn(12) * 0.01)
        out = selection_bias_check(vals)
        self.assertNotIn('skipped', out)
        self.assertEqual(out['n_trials'], 12)
        self.assertTrue(out['selection_suspect'])
        self.assertLess(out['deflated_ic'], 0)

    def test_real_skill_chain_not_suspect(self):
        """真实 skill trial (mean≫0, 离散度小) → deflated>0 → 非 suspect"""
        vals = list(0.2 + np.random.RandomState(7).randn(12) * 0.001)
        out = selection_bias_check(vals)
        self.assertFalse(out['selection_suspect'])
        self.assertGreater(out['deflated_ic'], 0)

    def test_guard_lt2_and_nonfinite(self):
        """n<2 → skipped (无跨 trial 方差); NaN/Inf 过滤后再判 (无 NaN 泄漏)"""
        self.assertIn('skipped', selection_bias_check([0.5]))
        self.assertIn('skipped', selection_bias_check([]))
        out = selection_bias_check([0.1, 0.2, 0.1, float('nan'), float('inf')])
        self.assertEqual(out['n_trials'], 3)
        self.assertTrue(np.isfinite(out['deflated_ic']))



class TestPboCscv(unittest.TestCase):
    """多候选 CSCV PBO (2026-09-11 #30 标准形升级)

    ⚠ 09-08/09-11 SIGSEGV 教训: 不在 unittest 主进程 in-process 跑 Optuna/
    LGBM — 本类只验 pbo_cscv_check 数学形 + 功效门 + summary 空形。
    """

    def test_real_skill_matrix_not_suspect(self):
        """前 6 候选整体 IC 高且稳 → IS 最优常在 OOS 不衰减 → pbo≤0.5"""
        from modules.hyperparam_optimizer import pbo_cscv_check
        rng = np.random.RandomState(42)
        good = 0.10 + rng.randn(6, 8) * 0.02
        bad = rng.randn(6, 8) * 0.05
        matrix = np.vstack([good, bad]).tolist()
        out = pbo_cscv_check(matrix)
        self.assertNotIn('skipped', out)
        self.assertEqual(out['n_candidates'], 12)
        self.assertEqual(out['n_blocks'], 8)
        self.assertEqual(out['combos'], 70)          # C(8,4)
        self.assertLessEqual(out['pbo'], 0.5)
        self.assertFalse(out['selection_suspect'])

    def test_guard_power_gate(self):
        """统计功效门: N<8 / blocks<4 → skipped (诚实标记, 不装真值)"""
        from modules.hyperparam_optimizer import pbo_cscv_check
        self.assertIn('skipped', pbo_cscv_check([[0.1] * 8] * 3))     # N=3
        self.assertIn('skipped', pbo_cscv_check([[0.1] * 3] * 12))    # blocks=3
        self.assertIn('skipped', pbo_cscv_check([]))                  # 空
        # 全 NaN 行 → 过滤后 n=0 → skipped (NaN 不泄漏成假 PBO)
        self.assertIn('skipped', pbo_cscv_check(
            [[float('nan')] * 8] * 12))

    def test_summary_shape_empty(self):
        """selection_summary 空形: 无 study 时 pbo 键完整 (快照/日志消费)"""
        orch = HyperParamOrchestrator(n_trials=12, cv_splits=8)
        s = orch.selection_summary()
        self.assertEqual(s['studies'], 0)
        self.assertEqual(s['pbo'], {'studies': 0, 'evaluated': 0, 'suspect': 0,
                                    'pbo_max': None, 'method': 'pbo_cscv_v1'})


if __name__ == '__main__':
    unittest.main()
