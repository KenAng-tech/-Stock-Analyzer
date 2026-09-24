#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
ml_processors 流水线 + model_registry 版本流水线测试 (P1-6/P1-9/P2-13, 2026-09-14)

覆盖:
1. CSZScoreNorm — fit 窗口外数据用冻结参数变换 (防前视纪律); 缺窗口参数 assert
2. learn/infer 守卫 — DropnaLabel (is_for_infer=False) 进 infer 抛 TypeError
3. ProcessInf / Fillna — inf→NaN / NaN 填充
4. regime_sample_weights — 手算例 + 归一化均值 1 + 退化兜底 + 维度校验
5. ModelVersionRegistry — 注册/online 切换/list/days_since + 并发 5 线程
   set_online 原子性 (不产生双 online)
"""

import json
import os
import tempfile
import threading
import unittest

import numpy as np
import pandas as pd

from modules.ml_processors import (
    CSZScoreNorm, ProcessInf, Fillna, DropnaLabel,
    apply_pipeline, regime_sample_weights,
)
from modules.model_registry import ModelVersionRegistry


def _df():
    """10 天合成数据: x=0..9, label 第 3 天 NaN, inf 在第 5 天"""
    return pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=10).strftime('%Y-%m-%d'),
        'x': [0.0, 1.0, 2.0, 3.0, 4.0, np.inf, 6.0, 7.0, 8.0, 9.0],
        'label': [1, 0, np.nan, 1, 0, 1, 0, 1, 0, 1],
    })


class TestCSZScoreNorm(unittest.TestCase):
    """z-score 标准化 — 统计量冻结于 fit 窗口"""

    def test_frozen_params_outside_window(self):
        """窗口 [01-01, 01-05] 内 mean=2, std=√2; 窗口外 x=100 用冻结参数变换"""
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=6).strftime('%Y-%m-%d'),
            'x': [0.0, 1.0, 2.0, 3.0, 4.0, 100.0],
        })
        p = CSZScoreNorm().fit(df, fit_start_time='2024-01-01', fit_end_time='2024-01-05')
        out = p.transform(df)
        exp_mean, exp_std = 2.0, np.sqrt(2.0)
        self.assertAlmostEqual(out['x'].iloc[0], (0.0 - exp_mean) / exp_std, places=9)
        self.assertAlmostEqual(out['x'].iloc[5], (100.0 - exp_mean) / exp_std, places=9)
        # 冻结参数不随 transform 输入变化 (再 transform 一次极端值, 参数不变)
        self.assertAlmostEqual(p.mean_['x'], exp_mean, places=9)
        self.assertAlmostEqual(p.std_['x'], exp_std, places=9)

    def test_fit_requires_window(self):
        """缺 fit_start_time/fit_end_time → assert (qlib 硬约束)"""
        df = _df()
        p = CSZScoreNorm()
        with self.assertRaises(AssertionError):
            p.fit(df)
        with self.assertRaises(AssertionError):
            p.fit(df, fit_start_time='2024-01-01')  # 缺 end 同样拒绝

    def test_transform_before_fit_raises(self):
        """未 fit 就 transform → RuntimeError (参数不存在禁止猜测)"""
        with self.assertRaises(RuntimeError):
            CSZScoreNorm().transform(_df())

    def test_picklable(self):
        """fit 参数存普通属性 → 整处理器可 pickle (qlib 要求)"""
        import pickle
        df = _df().replace([np.inf, -np.inf], np.nan)
        p = CSZScoreNorm().fit(df, fit_start_time='2024-01-01', fit_end_time='2024-01-05')
        p2 = pickle.loads(pickle.dumps(p))
        pd.testing.assert_series_equal(p.transform(df)['x'], p2.transform(df)['x'])


class TestLearnInferGuard(unittest.TestCase):
    """learn/infer 守卫 (qlib handler.py:534-535 同形)"""

    def _pipeline(self):
        # 同实例复用 (qlib 纪律: learn fit 出参数, infer 用冻结参数);
        # DropnaLabel 在 Fillna 前 — 否则 label NaN 先被填掉, 守卫形同虚设
        return [ProcessInf(), CSZScoreNorm(), DropnaLabel(), Fillna()]

    def test_infer_mode_rejects_dropna_label(self):
        """learn 先 fit, 同实例 infer 再跑 → 遇 DropnaLabel 抛 TypeError"""
        df = _df()
        procs = self._pipeline()
        apply_pipeline(df, procs, mode='learn',
                       fit_start_time='2024-01-01', fit_end_time='2024-01-08')
        with self.assertRaises(TypeError):
            apply_pipeline(df, procs, mode='infer')

    def test_learn_mode_drops_nan_label(self):
        """learn 模式: inf→NaN, DropnaLabel 丢标签 NaN 行 (10→9), Fillna 清特征残 NaN"""
        df = _df()
        out = apply_pipeline(df, self._pipeline(), mode='learn',
                             fit_start_time='2024-01-01', fit_end_time='2024-01-08')
        self.assertEqual(len(out), 9)
        self.assertFalse(out['label'].isna().any())
        self.assertTrue(np.isfinite(out['x']).all())

    def test_invalid_mode(self):
        with self.assertRaises(ValueError):
            apply_pipeline(_df(), [], mode='validate')


class TestStatelessProcessors(unittest.TestCase):

    def test_process_inf(self):
        out = ProcessInf().transform(_df())
        self.assertTrue(np.isnan(out['x'].iloc[5]))

    def test_fillna(self):
        df = pd.DataFrame({'a': [1.0, np.nan, 3.0]})
        out = Fillna(fill_value=-1.0).transform(df)
        self.assertEqual(list(out['a']), [1.0, -1.0, 3.0])

    def test_is_for_infer_flags(self):
        self.assertTrue(ProcessInf().is_for_infer())
        self.assertTrue(Fillna().is_for_infer())
        self.assertTrue(CSZScoreNorm().is_for_infer())
        self.assertFalse(DropnaLabel().is_for_infer())


class TestRegimeSampleWeights(unittest.TestCase):
    """regime 相似性加权 — 手算例"""

    def test_onehot_current_hand_computed(self):
        """current=bull, posteriors bull 概率 [0.8, 0.2] → 权重 [1.6, 0.4] (均值 1)"""
        w = regime_sample_weights(
            {'bull': 1.0, 'bear': 0.0},
            np.array([[0.8, 0.2], [0.2, 0.8]]),
            regime_order=['bull', 'bear'],
        )
        np.testing.assert_allclose(w, [1.6, 0.4])
        self.assertAlmostEqual(float(w.mean()), 1.0, places=12)

    def test_uniform_current_gives_uniform_weights(self):
        """current 均匀分布 → 每行内积相同 → 全 1"""
        w = regime_sample_weights(
            {'bull': 0.5, 'bear': 0.5},
            np.array([[0.8, 0.2], [0.2, 0.8]]),
            regime_order=['bull', 'bear'],
        )
        np.testing.assert_allclose(w, [1.0, 1.0])

    def test_list_of_dicts_alignment(self):
        """list[dict] 形态自动按 regime_order 对齐"""
        w = regime_sample_weights(
            {'bull': 1.0},
            [{'bull': 0.6, 'bear': 0.4}, {'bull': 0.2, 'bear': 0.8}],
            regime_order=['bull', 'bear'],
        )
        np.testing.assert_allclose(w, [0.6 / 0.4, 0.2 / 0.4])  # mean=0.4 → [1.5, 0.5]

    def test_all_zero_falls_back_to_ones(self):
        w = regime_sample_weights({'bull': 1.0}, np.zeros((3, 1)), regime_order=['bull'])
        np.testing.assert_allclose(w, [1.0, 1.0, 1.0])

    def test_dim_mismatch_raises(self):
        with self.assertRaises(ValueError):
            regime_sample_weights({'a': 1.0}, np.ones((4, 2)), regime_order=['a'])


class TestModelVersionRegistry(unittest.TestCase):
    """版本流水线 — online 原子切换 + 并发安全"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._tmp.name, 'model_versions.json')
        self.reg = ModelVersionRegistry(registry_path=self.path)

    def tearDown(self):
        self._tmp.cleanup()

    def _register(self, name='patchtst'):
        return self.reg.register_version(name, '/tmp/x.pth', {'val_acc': 0.4},
                                         ['2025-01-01', '2025-09-01'])

    def test_register_and_online_switch(self):
        v1 = self._register()
        v2 = self._register()
        self.assertTrue(self.reg.set_online('patchtst', v1))
        self.assertEqual(self.reg.get_online('patchtst')['version_id'], v1)
        # 原子切换: v2 上线后 v1 必 offline (旧版本保留可回滚)
        self.assertTrue(self.reg.set_online('patchtst', v2))
        online = self.reg.get_online('patchtst')
        self.assertEqual(online['version_id'], v2)
        versions = self.reg.list_versions('patchtst')
        self.assertEqual(len(versions), 2)
        statuses = {v['version_id']: v['status'] for v in versions}
        self.assertEqual(statuses[v1], 'offline')
        self.assertEqual(statuses[v2], 'online')
        # 回滚 = 对旧版本再 set_online
        self.assertTrue(self.reg.set_online('patchtst', v1))
        self.assertEqual(self.reg.get_online('patchtst')['version_id'], v1)

    def test_set_online_unknown_version(self):
        self.assertFalse(self.reg.set_online('patchtst', 'patchtst@nope'))

    def test_days_since_last_train(self):
        self.assertIsNone(self.reg.days_since_last_train('ghost'))
        self._register()
        days = self.reg.days_since_last_train('patchtst')
        self.assertIsNotNone(days)
        self.assertGreaterEqual(days, 0.0)
        self.assertLess(days, 0.01)  # 刚注册 < 15 分钟

    def test_models_isolated(self):
        """不同模型的 online 互不影响"""
        vp = self._register('patchtst')
        vm = self.reg.register_version('mamba', '/tmp/m.pth', {}, [])
        self.reg.set_online('patchtst', vp)
        self.reg.set_online('mamba', vm)
        self.assertEqual(self.reg.get_online('patchtst')['version_id'], vp)
        self.assertEqual(self.reg.get_online('mamba')['version_id'], vm)

    def test_concurrent_set_online_single_online(self):
        """并发 5 线程 set_online 不同版本 → 恰好 1 个 online (原子性)"""
        vids = [self._register() for _ in range(5)]
        errors = []

        def worker(vid):
            try:
                self.reg.set_online('patchtst', vid)
            except Exception as e:  # pragma: no cover
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(v,)) for v in vids]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        self.assertEqual(errors, [])
        with open(self.path, encoding='utf-8') as f:
            data = json.load(f)
        online = [v for v in data['versions']
                  if v['model_name'] == 'patchtst' and v['status'] == 'online']
        self.assertEqual(len(online), 1)
        self.assertIn(online[0]['version_id'], vids)

    def test_corrupted_file_rebuilds(self):
        """损坏 JSON → 读侧兜底重建, 不抛"""
        with open(self.path, 'w') as f:
            f.write('{ not json')
        self.assertIsNone(self.reg.get_online('patchtst'))
        vid = self._register()  # 读空表 → 追加 → 原子覆盖
        self.assertIsNotNone(vid)
        self.assertEqual(len(self.reg.list_versions('patchtst')), 1)


if __name__ == '__main__':
    unittest.main()
