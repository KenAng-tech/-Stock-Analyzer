#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
集成学习测试 — Temporal Stacking / BMA / Hybrid Ensemble
"""

import unittest
import numpy as np
from modules.models.ensemble_learning import (
    TemporalStackingEnsemble,
    BayesianModelAverager,
    HybridEnsemble,
    EnsembleResult,
)


class TestTemporalStackingEnsemble(unittest.TestCase):
    """Temporal Stacking 测试"""

    def setUp(self):
        self.ensemble = TemporalStackingEnsemble(n_lags=3)
        self.level0_preds = {
            'lgb': np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.85, 0.75, 0.65]),
            'rf': np.array([0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 0.9, 0.8, 0.7]),
            'xgb': np.array([0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.8, 0.7, 0.6]),
        }
        self.y = np.array([1, 1, 0, -1, 1, 0, 1, 1, 0, -1])

    def test_generate_lagged_features(self):
        """测试滞后特征生成"""
        features, names = self.ensemble.generate_lagged_features(self.level0_preds)
        self.assertGreater(len(features), 0)
        self.assertGreater(len(names), 0)
        # 应该有 3 模型 * 3 滞后 = 9 个特征
        self.assertEqual(features.shape[1], 9)

    def test_train_insufficient_data(self):
        """测试数据不足"""
        small_preds = {'lgb': np.array([0.5, 0.6])}
        features, _ = self.ensemble.generate_lagged_features(small_preds)
        result = self.ensemble.train(features, self.y[:len(features)], ['lgb'])
        self.assertFalse(result)

    def test_train_success(self):
        """测试训练成功"""
        features, names = self.ensemble.generate_lagged_features(self.level0_preds)
        result = self.ensemble.train(features, self.y[:len(features)], ['lgb', 'rf', 'xgb'])
        self.assertTrue(result)
        self.assertTrue(self.ensemble.is_trained)

    def test_predict_trained(self):
        """测试训练后预测"""
        features, names = self.ensemble.generate_lagged_features(self.level0_preds)
        self.ensemble.train(features, self.y[:len(features)], ['lgb', 'rf', 'xgb'])
        result = self.ensemble.predict({'lgb': 0.6, 'rf': 0.7, 'xgb': 0.5})
        self.assertIn(result.direction, ['up', 'down', 'neutral'])
        self.assertIn(result.method, ['temporal_stacking'])

    def test_predict_untrained(self):
        """测试未训练时预测"""
        result = self.ensemble.predict({'lgb': 0.6, 'rf': 0.7, 'xgb': 0.5})
        self.assertEqual(result.direction, 'neutral')
        self.assertAlmostEqual(result.confidence, 0.33, places=2)


class TestBayesianModelAverager(unittest.TestCase):
    """BMA 测试"""

    def setUp(self):
        self.bma = BayesianModelAverager(prior_type='ic_weighted', temperature=1.0)

    def test_update_performance(self):
        """测试性能更新"""
        self.bma.update_model_performance('lgb', ic=0.05)
        self.bma.update_model_performance('rf', ic=0.08)
        self.bma.update_model_performance('xgb', ic=0.02)

    def test_compute_weights(self):
        """测试权重计算"""
        self.bma.update_model_performance('lgb', ic=0.05,
                                          ic_series=[0.03, 0.05, 0.07, 0.04, 0.06])
        self.bma.update_model_performance('rf', ic=0.08,
                                          ic_series=[0.06, 0.08, 0.10, 0.07, 0.09])
        self.bma.update_model_performance('xgb', ic=0.02,
                                          ic_series=[0.01, 0.02, 0.03, 0.02, 0.01])
        weights = self.bma.compute_weights()
        # 权重和应为 1
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=4)
        # 每个权重应在 [0, 1] 范围内
        for w in weights.values():
            self.assertGreaterEqual(w, 0)
            self.assertLessEqual(w, 1)

    def test_average(self):
        """测试模型平均"""
        self.bma.update_model_performance('lgb', ic=0.05)
        self.bma.update_model_performance('rf', ic=0.08)
        self.bma.update_model_performance('xgb', ic=0.02)
        self.bma.compute_weights()

        avg = self.bma.average({'lgb': 0.6, 'rf': 0.7, 'xgb': 0.4})
        self.assertIn('up', avg)
        self.assertIn('down', avg)
        self.assertIn('neutral', avg)
        self.assertAlmostEqual(sum(avg.values()), 1.0, places=4)

    def test_average_no_weights(self):
        """测试无权重时的平均"""
        avg = self.bma.average({'lgb': 0.6})
        self.assertAlmostEqual(avg['up'], 0.33, places=2)

    def test_icir_calculation(self):
        """测试 ICIR 计算"""
        self.bma.update_model_performance('lgb', ic=0.05,
                                          ic_series=[0.03, 0.05, 0.07, 0.04, 0.06,
                                                     0.05, 0.06, 0.04, 0.07, 0.05])
        icir = self.bma._compute_icir('lgb')
        self.assertGreaterEqual(icir, 0)

    def test_get_summary(self):
        """测试摘要"""
        self.bma.update_model_performance('lgb', ic=0.05)
        self.bma.compute_weights()
        summary = self.bma.get_summary()
        self.assertEqual(summary['method'], 'bma')
        self.assertTrue(summary['is_trained'])
        self.assertEqual(summary['n_models'], 1)

    def test_uniform_prior(self):
        """测试均匀先验"""
        bma = BayesianModelAverager(prior_type='uniform')
        bma.update_model_performance('a', ic=0.1)
        bma.update_model_performance('b', ic=0.1)
        weights = bma.compute_weights()
        self.assertAlmostEqual(weights['a'], weights['b'], places=4)

    def test_temperature_effect(self):
        """测试温度参数影响"""
        bma_low = BayesianModelAverager(temperature=0.5)
        bma_high = BayesianModelAverager(temperature=2.0)

        bma_low.update_model_performance('a', ic=0.1)
        bma_low.update_model_performance('b', ic=0.05)
        bma_high.update_model_performance('a', ic=0.1)
        bma_high.update_model_performance('b', ic=0.05)

        w_low = bma_low.compute_weights()
        w_high = bma_high.compute_weights()

        # 低温下权重更集中 (好的模型获得更多权重)
        self.assertGreater(w_low['a'], w_high['a'])


class TestHybridEnsemble(unittest.TestCase):
    """混合集成测试"""

    def setUp(self):
        self.ensemble = HybridEnsemble(n_lags=3, temperature=1.0)
        self.level0_preds = {
            'lgb': np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.85, 0.75, 0.65]),
            'rf': np.array([0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 0.9, 0.8, 0.7]),
            'xgb': np.array([0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.8, 0.7, 0.6]),
        }
        self.y = np.array([1, 1, 0, -1, 1, 0, 1, 1, 0, -1])

    def test_init(self):
        """测试初始化"""
        self.assertEqual(self.ensemble.temporal.n_lags, 3)
        self.assertEqual(self.ensemble.bma.temperature, 1.0)

    def test_train(self):
        """测试训练"""
        result = self.ensemble.train(self.level0_preds, self.y, ['lgb', 'rf', 'xgb'])
        self.assertTrue(result)

    def test_predict(self):
        """测试预测"""
        self.ensemble.train(self.level0_preds, self.y, ['lgb', 'rf', 'xgb'])
        result = self.ensemble.predict({'lgb': 0.6, 'rf': 0.7, 'xgb': 0.5})
        self.assertIn(result.direction, ['up', 'down', 'neutral'])
        self.assertIn(result.method, ['hybrid'])
        self.assertGreater(result.confidence, 0)
        self.assertLessEqual(result.confidence, 1)

    def test_predict_consistent(self):
        """测试一致预测提高置信度"""
        self.ensemble.train(self.level0_preds, self.y, ['lgb', 'rf', 'xgb'])
        # 所有模型都预测上涨
        result = self.ensemble.predict({'lgb': 0.8, 'rf': 0.85, 'xgb': 0.75})
        self.assertEqual(result.direction, 'up')
        self.assertGreater(result.confidence, 0.5)

    def test_predict_disagreement(self):
        """测试分歧预测"""
        self.ensemble.train(self.level0_preds, self.y, ['lgb', 'rf', 'xgb'])
        # 模型间有分歧
        result = self.ensemble.predict({'lgb': 0.8, 'rf': 0.3, 'xgb': 0.6})
        self.assertIn(result.direction, ['up', 'down', 'neutral'])

    def test_to_dict(self):
        """测试结果转字典"""
        self.ensemble.train(self.level0_preds, self.y, ['lgb', 'rf', 'xgb'])
        result = self.ensemble.predict({'lgb': 0.6, 'rf': 0.7, 'xgb': 0.5})
        d = result.to_dict()
        self.assertIn('direction', d)
        self.assertIn('probabilities', d)
        self.assertIn('model_weights', d)
        self.assertIn('timestamp', d)


class TestEnsembleResult(unittest.TestCase):
    """EnsembleResult 测试"""

    def test_to_dict(self):
        """测试序列化"""
        result = EnsembleResult(
            direction='up',
            confidence=0.75,
            probabilities={'up': 0.7, 'down': 0.15, 'neutral': 0.15},
            method='hybrid',
            model_weights={'lgb': 0.4, 'rf': 0.35, 'xgb': 0.25},
            timestamp='2026-01-01T00:00:00',
        )
        d = result.to_dict()
        self.assertAlmostEqual(d['confidence'], 0.75, places=4)
        self.assertAlmostEqual(d['probabilities']['up'], 0.7, places=4)


if __name__ == '__main__':
    unittest.main()
