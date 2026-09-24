#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
超参调优链完整性验证 (2026-09-08 诚实边界四项)

背景: 23:00 链与 POST /hyperparams 的调参结果此前无消费方 —
apply_to_predictor 是 pass 桩 (记"已应用"日志但什么都不做),
模型工厂硬编码默认参数, 调参 = 装饰。本轮接通:
  Optuna 调参 (全局 + per-regime 分段) → custom_params →
  模型工厂 (主进程) / 子进程 trainer 脚本 合并消费

测试策略 (合成+噪声双测验):
1. segment_regimes — 趋势/下跌/横盘分段正确性 (仅用过去窗口)
2. apply_to_predictor — 空参数 False (诚实) / 注入 True / regime 优先
3. 模型工厂 — custom_params 合并生效 (num_leaves 等)
4. predict_direction margin gating — 边际信号降级 neutral, gate=0 行为不变
5. worker 端到端 — run_training 带 custom_params 子进程不崩 (SIGSEGV 边界内)
"""

import os
import unittest
import numpy as np

from modules.hyperparam_optimizer import (
    HyperParamOrchestrator, segment_regimes, optimize_all_isolated,
)


def _trend_klines(n: int, daily: float) -> list:
    """构造单调趋势 K 线 (daily=日收益率)"""
    return [{'close': 100.0 * (1 + daily) ** i} for i in range(n)]


class TestSegmentRegimes(unittest.TestCase):
    """regime 分段 — 仅用过去窗口, 无前视"""

    def test_uptrend_is_bull(self):
        """持续 20 日 +17% 趋势 → bull"""
        self.assertEqual(segment_regimes(_trend_klines(60, 0.008))[-1], 'bull')

    def test_downtrend_is_bear(self):
        """持续 20 日 -15% 趋势 → bear"""
        self.assertEqual(segment_regimes(_trend_klines(60, -0.008))[-1], 'bear')

    def test_flat_is_sideways(self):
        """横盘 (±0.1% 锯齿) → sideways"""
        klines = [{'close': 100 + (i % 3) * 0.1} for i in range(60)]
        self.assertEqual(segment_regimes(klines)[-1], 'sideways')

    def test_leading_samples_are_sideways(self):
        """前导样本 (<lookback) 保守归 sideways (不参与 per-regime 调参)"""
        labels = segment_regimes(_trend_klines(60, 0.008), lookback=20)
        self.assertTrue(all(l == 'sideways' for l in labels[:20]))

    def test_alignment_and_types(self):
        """输出与 klines 等长且值域合法"""
        klines = _trend_klines(80, 0.002)
        labels = segment_regimes(klines)
        self.assertEqual(len(labels), len(klines))
        self.assertTrue(set(labels) <= {'bull', 'bear', 'sideways'})


class TestApplyToPredictor(unittest.TestCase):
    """apply_to_predictor — pass 桩 → 真实链"""

    def test_empty_params_returns_false(self):
        """无调参结果 → False (不再谎报已应用)"""
        class FakePredictor:
            custom_params = {}
        orch = HyperParamOrchestrator(n_trials=2)
        self.assertFalse(orch.apply_to_predictor(FakePredictor()))

    def test_injection_writes_custom_params(self):
        """全局参数注入 → predictor.custom_params"""
        class FakePredictor:
            custom_params = {}
        orch = HyperParamOrchestrator(n_trials=2)
        orch._best_params = {'lgb': {'num_leaves': 64}, 'rf': {'max_depth': 8}}
        p = FakePredictor()
        self.assertTrue(orch.apply_to_predictor(p))
        self.assertEqual(p.custom_params['lgb']['num_leaves'], 64)

    def test_regime_params_take_priority(self):
        """current_regime 段参数优先于全局参数"""
        class FakePredictor:
            custom_params = {}
        orch = HyperParamOrchestrator(n_trials=2)
        orch._best_params = {'lgb': {'num_leaves': 64}}
        orch._regime_params = {'bull': {'lgb': {'num_leaves': 128}}}
        p = FakePredictor()
        orch.apply_to_predictor(p, current_regime='bull')
        self.assertEqual(p.custom_params['lgb']['num_leaves'], 128)

    def test_get_best_params_regime_fallback(self):
        """get_best_params(model, regime) — 无该段参数回退全局"""
        orch = HyperParamOrchestrator(n_trials=2)
        orch._best_params = {'xgb': {'max_depth': 4}}
        orch._regime_params = {'bull': {'xgb': {'max_depth': 7}}}
        self.assertEqual(orch.get_best_params('xgb', regime='bull')['max_depth'], 7)
        self.assertEqual(orch.get_best_params('xgb', regime='bear')['max_depth'], 4)
        self.assertEqual(orch.get_best_params('xgb')['max_depth'], 4)


class TestFactoryConsumesCustomParams(unittest.TestCase):
    """模型工厂消费点 — custom_params 合并进构造参数"""

    def setUp(self):
        try:
            import lightgbm  # noqa: F401
            import xgboost  # noqa: F401
            self.has_c_ext = True
        except ImportError:
            self.has_c_ext = False

    def test_factories_merge_tuned_params(self):
        from modules.ml_predictor import MLPredictor
        ml = MLPredictor()
        ml.custom_params = {'lgb': {'num_leaves': 64, 'max_depth': 3},
                            'xgb': {'max_depth': 4},
                            'rf': {'max_depth': 8}}
        rf = ml._make_random_forest()
        self.assertEqual(rf.get_params()['max_depth'], 8)
        if self.has_c_ext:
            self.assertEqual(ml._make_lightgbm().get_params()['num_leaves'], 64)
            self.assertEqual(ml._make_xgboost().get_params()['max_depth'], 4)

    def test_worker_accepts_custom_params(self):
        """worker 透传 custom_params → 子进程建模不崩 (SIGSEGV 边界内)"""
        import tempfile, shutil
        import os
        from modules import ml_training_worker
        rng = np.random.default_rng(42)
        X = rng.normal(size=(200, 12))
        y = rng.integers(0, 3, 200) - 1
        tmp = tempfile.mkdtemp()
        try:
            result = ml_training_worker.run_training(
                X, y, [f'f{i}' for i in range(12)], tmp, timeout=180,
                custom_params={'lgb': {'n_estimators': 50, 'num_leaves': 64},
                               'xgb': {'n_estimators': 50, 'max_depth': 4},
                               'rf': {'n_estimators': 50, 'max_depth': 8}})
            self.assertTrue(result.get('success'),
                            f"worker 带参训练失败: {result.get('message')}")
            self.assertTrue(result.get('models_trained'))
        finally:
            shutil.rmtree(tmp)


class TestPredictDirectionGating(unittest.TestCase):
    """推理侧利润阈值 gating — 概率边际 < gate → neutral"""

    class FixedProbaModel:
        """predict_proba 列序 = sklearn 排序类别 [down, neutral, up]"""

        def __init__(self, proba):
            self.p = np.array([proba])

        def predict_proba(self, X):
            return self.p

        def predict(self, X):
            return np.array([2])

    def _predict(self, proba, gate=0.10):
        from modules.ml_predictor import MLPredictor
        ml = MLPredictor()
        ml.signal_margin_gate = gate
        ml.models = {'rf': self.FixedProbaModel(proba)}
        rng = np.random.default_rng(7)
        return ml.predict_direction(rng.normal(size=29))

    def test_marginal_signal_gated(self):
        """up 概率与次高差 0 → 降级 neutral (盖不过成本的不确定下注)"""
        result = self._predict([0.16, 0.42, 0.42])
        self.assertEqual(result['direction'], 'neutral')

    def test_strong_signal_preserved(self):
        """up 概率优势 0.50 → 保留 up"""
        self.assertEqual(self._predict([0.10, 0.20, 0.70])['direction'], 'up')

    def test_strong_down_preserved(self):
        """down 概率优势 0.25 → 保留 down"""
        self.assertEqual(self._predict([0.55, 0.30, 0.15])['direction'], 'down')

    def test_gate_zero_restores_old_behavior(self):
        """gate=0 → 与修复前行为一致 (0.4 下限保留)"""
        self.assertEqual(self._predict([0.16, 0.42, 0.42], gate=0.0)['direction'], 'up')

    def test_default_gate_is_positive(self):
        """默认阈值 0.10 (非 0 = 默认启用 gating)"""
        from modules.ml_predictor import MLPredictor
        self.assertGreater(MLPredictor().signal_margin_gate, 0)


class TestIsolatedTuning(unittest.TestCase):
    """optimize_all_isolated — 子进程调参链 (SIGSEGV 边界)"""

    def test_isolated_tuning_end_to_end(self):
        """子进程跑通: 返回 best_params + per-regime 结构合法"""
        rng = np.random.default_rng(42)
        n = 300
        X = rng.normal(size=(n, 12))
        y = rng.integers(0, 3, n) - 1
        # 前段上涨 (bull) / 后段下跌 (bear) — 制造非全 sideways 的分段
        closes = np.concatenate([np.cumprod(np.concatenate([[100.0], np.full(150, 1.006)])),
                                 np.cumprod(np.concatenate([[100.0], np.full(149, 0.994)]))])
        regimes = segment_regimes([{'close': float(c)} for c in closes])
        result = optimize_all_isolated(X, y, regimes=regimes, n_trials=2, timeout=300)
        self.assertIsNotNone(result, '子进程调参失败 (应返回 dict)')
        self.assertTrue(result.get('success'))
        self.assertIn('lgb', result['best_params'])
        self.assertIn('current_regime', result)
        for regime, params in result.get('regime_params', {}).items():
            self.assertIn(regime, ('bull', 'bear', 'sideways'))
            for model_name, p in params.items():
                self.assertIn(model_name, ('lgb', 'xgb', 'rf'))
                self.assertIsInstance(p, dict)


class TestParamSnapshot(unittest.TestCase):
    """2026-09-08: 02:00 调参链快照持久化 + 22:00 重训/GET 消费链 (时间预算闭环)"""

    def setUp(self):
        import modules.hyperparam_optimizer as ho
        import tempfile
        self.ho = ho
        self._orig_path = ho.SNAPSHOT_PATH
        self._tmp = tempfile.TemporaryDirectory()
        ho.SNAPSHOT_PATH = os.path.join(self._tmp.name, 'snap.json')

    def tearDown(self):
        self.ho.SNAPSHOT_PATH = self._orig_path
        self._tmp.cleanup()

    def test_snapshot_roundtrip(self):
        """写入 → 读回 = 调参参数 (22:00 重训 + GET 展示的入口)"""
        from datetime import datetime
        snap = {
            'generated_at': datetime.now().isoformat(),
            'stock_code': 'sz300620',
            'best_params': {'lgb': {'n_estimators': 200}, 'xgb': {'max_depth': 5},
                            'rf': {'n_estimators': 100}},
        }
        self.assertTrue(self.ho.save_param_snapshot(snap))
        loaded = self.ho.load_param_snapshot()
        self.assertIsNotNone(loaded, '新鲜快照应可读回')
        self.assertIn('lgb', loaded)
        self.assertIn('xgb', loaded)

    def test_expired_snapshot_rejected(self):
        """>7 天快照 → None (市场状态迁移快于模型迭代, 过期参数不如默认)"""
        from datetime import datetime, timedelta
        snap = {
            'generated_at': (datetime.now() - timedelta(days=8)).isoformat(),
            'best_params': {'lgb': {'n_estimators': 200}, 'xgb': {'max_depth': 5},
                            'rf': {'n_estimators': 100}},
        }
        self.assertTrue(self.ho.save_param_snapshot(snap))
        self.assertIsNone(self.ho.load_param_snapshot(), '过期快照必须弃用')

    def test_corrupt_snapshot_rejected(self):
        """损坏/空文件 → None (降级默认参数, 不抛异常)"""
        with open(self.ho.SNAPSHOT_PATH, 'w') as f:
            f.write('not json {{{')
        self.assertIsNone(self.ho.load_param_snapshot())


if __name__ == '__main__':
    unittest.main(verbosity=2)
