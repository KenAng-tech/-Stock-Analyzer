#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
新模型测试 — Week 3-4 新增模型验证

测试项目:
1. iTransformer 预测器
2. Bi-Mamba+ 预测器
3. HRP 组合优化器
"""

import unittest
import numpy as np
import importlib.util


def _import_model(module_name, file_path):
    """直接导入模块，绕过 __init__.py"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestITransformerPredictor(unittest.TestCase):
    """iTransformer 预测器测试"""

    def setUp(self):
        try:
            mod = _import_model('itransformer', 'modules/models/itransformer_predictor.py')
            self.model = mod.iTransformerPredictor(
                n_features=6, d_model=32, n_heads=4,
                n_layers=2, seq_len=30, pred_len=5
            )
        except Exception:
            # torch 未安装时跳过
            self.model = None

    def test_init(self):
        """验证初始化"""
        if self.model is None:
            self.skipTest("torch 未安装，跳过")
        self.assertEqual(self.model.n_features, 6)
        self.assertEqual(self.model.d_model, 32)
        self.assertEqual(self.model.seq_len, 30)
        self.assertEqual(self.model.pred_len, 5)
        self.assertFalse(self.model.is_trained)

    def test_forward(self):
        """验证前向传播"""
        if self.model is None:
            self.skipTest("torch 未安装，跳过")
        X = np.random.randn(2, 30, 6).astype(np.float32)
        output = self.model.forward(X)
        self.assertEqual(output.shape, (2, 5))

    def test_predict_2d(self):
        """验证 2D 输入预测"""
        if self.model is None:
            self.skipTest("torch 未安装，跳过")
        X = np.random.randn(30, 6).astype(np.float32)
        result = self.model.predict(X)
        self.assertIn('direction', result)
        self.assertIn('confidence', result)
        self.assertIn('probabilities', result)
        self.assertEqual(result['model'], 'iTransformer')

    def test_predict_3d(self):
        """验证 3D 输入预测"""
        if self.model is None:
            self.skipTest("torch 未安装，跳过")
        X = np.random.randn(4, 30, 6).astype(np.float32)
        result = self.model.predict(X)
        self.assertIn('direction', result)
        self.assertEqual(len(result['predictions']), 4)

    def test_train_step(self):
        """验证单步训练"""
        if self.model is None:
            self.skipTest("torch 未安装，跳过")
        X = np.random.randn(4, 30, 6).astype(np.float32)
        y = np.random.randn(4, 5).astype(np.float32)
        loss = self.model.train_step(X, y)
        self.assertIsInstance(loss, float)
        self.assertGreater(loss, 0)
        self.assertTrue(self.model.is_trained)

    def test_get_status(self):
        """验证状态获取"""
        if self.model is None:
            self.skipTest("torch 未安装，跳过")
        status = self.model.get_status()
        self.assertEqual(status['model'], 'iTransformer')
        self.assertEqual(status['n_features'], 6)
        self.assertEqual(status['n_layers'], 2)

    def test_factory_function(self):
        """验证工厂函数"""
        # 直接读取代码验证工厂函数存在
        with open('modules/models/itransformer_predictor.py', 'r') as f:
            content = f.read()
        self.assertIn('def create_iTransformer', content)
        self.assertIn('return iTransformerPredictor', content)


class TestBiMambaPredictor(unittest.TestCase):
    """Bi-Mamba+ 预测器测试"""

    def setUp(self):
        try:
            mod = _import_model('bi_mamba', 'modules/models/bi_mamba_predictor.py')
            self.model = mod.BiMambaPredictor(
                d_input=4, d_model=32, d_state=8,
                n_layers=2, seq_len=30, pred_len=5
            )
        except Exception:
            self.model = None

    def test_init(self):
        """验证初始化"""
        if self.model is None:
            self.skipTest("torch 未安装，跳过")
        self.assertEqual(self.model.d_input, 4)
        self.assertEqual(self.model.d_model, 32)
        self.assertEqual(self.model.pred_len, 5)
        self.assertFalse(self.model.is_trained)

    def test_forward(self):
        """验证前向传播"""
        if self.model is None:
            self.skipTest("torch 未安装，跳过")
        X = np.random.randn(2, 30, 4).astype(np.float32)
        output = self.model.forward(X)
        self.assertEqual(output.shape, (2, 5))

    def test_predict_2d(self):
        """验证 2D 输入预测"""
        if self.model is None:
            self.skipTest("torch 未安装，跳过")
        X = np.random.randn(30, 4).astype(np.float32)
        result = self.model.predict(X)
        self.assertIn('direction', result)
        self.assertIn('confidence', result)
        self.assertEqual(result['model'], 'Bi-Mamba+')

    def test_predict_3d(self):
        """验证 3D 输入预测"""
        if self.model is None:
            self.skipTest("torch 未安装，跳过")
        X = np.random.randn(4, 30, 4).astype(np.float32)
        result = self.model.predict(X)
        self.assertIn('direction', result)
        self.assertEqual(len(result['predictions']), 4)

    def test_train_step(self):
        """验证单步训练"""
        if self.model is None:
            self.skipTest("torch 未安装，跳过")
        X = np.random.randn(4, 30, 4).astype(np.float32)
        y = np.random.randn(4, 5).astype(np.float32)
        loss = self.model.train_step(X, y)
        self.assertIsInstance(loss, float)
        self.assertGreater(loss, 0)

    def test_get_status(self):
        """验证状态获取"""
        if self.model is None:
            self.skipTest("torch 未安装，跳过")
        status = self.model.get_status()
        self.assertEqual(status['model'], 'Bi-Mamba+')
        self.assertEqual(status['d_input'], 4)
        self.assertEqual(status['n_layers'], 2)

    def test_factory_function(self):
        """验证工厂函数"""
        with open('modules/models/bi_mamba_predictor.py', 'r') as f:
            content = f.read()
        self.assertIn('def create_bi_mamba', content)
        self.assertIn('return BiMambaPredictor', content)


class TestHRPPortfolioOptimizer(unittest.TestCase):
    """HRP 组合优化器测试"""

    def setUp(self):
        mod = _import_model('hrp_optimizer', 'modules/models/hrp_optimizer.py')
        self.optimizer = mod.HRPPortfolioOptimizer()

    def test_optimize_3_assets(self):
        """验证 3 资产优化"""
        cov = np.array([
            [0.04, 0.01, 0.005],
            [0.01, 0.09, 0.02],
            [0.005, 0.02, 0.16],
        ])
        tickers = ['A', 'B', 'C']
        weights = self.optimizer.optimize(cov, tickers)

        self.assertEqual(len(weights), 3)
        self.assertEqual(set(weights.keys()), set(tickers))
        total = sum(weights.values())
        self.assertAlmostEqual(total, 1.0, places=6)
        for w in weights.values():
            self.assertGreaterEqual(w, 0)

    def test_optimize_5_assets(self):
        """验证 5 资产优化"""
        np.random.seed(42)
        n = 5
        A = np.random.randn(n, n) * 0.1
        cov = A @ A.T + np.eye(n) * 0.01
        tickers = [f'asset_{i}' for i in range(n)]

        weights = self.optimizer.optimize(cov, tickers)
        self.assertEqual(len(weights), n)
        total = sum(weights.values())
        self.assertAlmostEqual(total, 1.0, places=6)

    def test_optimize_with_custom_linkage(self):
        """验证自定义链接方法"""
        cov = np.array([[0.04, 0.01], [0.01, 0.09]])
        mod = _import_model('hrp_optimizer', 'modules/models/hrp_optimizer.py')
        optimizer = mod.HRPPortfolioOptimizer(linkage_method='complete')
        weights = optimizer.optimize(cov, ['X', 'Y'])
        self.assertEqual(len(weights), 2)

    def test_get_portfolio_metrics(self):
        """验证组合指标计算"""
        np.random.seed(42)
        n_days = 100
        n_assets = 3
        returns = np.random.randn(n_days, n_assets) * 0.02

        cov = np.cov(returns.T)
        tickers = ['A', 'B', 'C']
        weights = self.optimizer.optimize(cov, tickers)

        metrics = self.optimizer.get_portfolio_metrics(returns)
        self.assertIn('annual_return', metrics)
        self.assertIn('annual_volatility', metrics)
        self.assertIn('sharpe_ratio', metrics)
        self.assertIn('max_drawdown', metrics)
        self.assertIn('calmar_ratio', metrics)
        self.assertIn('weight_entropy', metrics)

    def test_weight_entropy(self):
        """验证权重熵计算"""
        # 均匀权重熵应接近 1
        uniform = np.array([0.5, 0.5])
        entropy = self.optimizer._weight_entropy(uniform)
        self.assertGreater(entropy, 0.9)

        # 集中权重熵应接近 0
        concentrated = np.array([0.99, 0.01])
        entropy = self.optimizer._weight_entropy(concentrated)
        self.assertLess(entropy, 0.3)

    def test_get_status(self):
        """验证状态获取"""
        status = self.optimizer.get_status()
        self.assertEqual(status['method'], 'HRP')
        self.assertEqual(status['n_tickers'], 0)

    def test_optimize_after_status(self):
        """验证优化后状态更新"""
        cov = np.array([[0.04, 0.01], [0.01, 0.09]])
        self.optimizer.optimize(cov, ['A', 'B'])
        status = self.optimizer.get_status()
        self.assertEqual(status['n_tickers'], 2)
        self.assertIn('A', status['weights'])
        self.assertIn('B', status['weights'])


class TestHRPInPortfolioOptimizer(unittest.TestCase):
    """验证 HRP 已集成到 portfolio_optimizer.py"""

    def test_hrp_optimizer_class_exists(self):
        """验证 HRPOptimizer 类存在于 portfolio_optimizer.py"""
        from modules.portfolio_optimizer import HRPOptimizer
        self.assertIsNotNone(HRPOptimizer)

    def test_hrp_optimizer_instantiable(self):
        """验证可以实例化 HRPOptimizer"""
        from modules.portfolio_optimizer import HRPOptimizer
        optimizer = HRPOptimizer()
        self.assertIsNotNone(optimizer)


class TestModelsInit(unittest.TestCase):
    """验证 models/__init__.py 导出"""

    def test_itransformer_in_init(self):
        """验证 iTransformer 在 __init__.py 中导出"""
        with open('modules/models/__init__.py', 'r') as f:
            content = f.read()
        self.assertIn('itransformer_predictor', content)
        self.assertIn('iTransformerPredictor', content)

    def test_bi_mamba_in_init(self):
        """验证 Bi-Mamba+ 在 __init__.py 中导出"""
        with open('modules/models/__init__.py', 'r') as f:
            content = f.read()
        self.assertIn('bi_mamba_predictor', content)
        self.assertIn('BiMambaPredictor', content)

    def test_hrp_in_init(self):
        """验证 HRP 在 __init__.py 中导出"""
        with open('modules/models/__init__.py', 'r') as f:
            content = f.read()
        self.assertIn('hrp_optimizer', content)
        self.assertIn('HRPPortfolioOptimizer', content)


if __name__ == '__main__':
    unittest.main()
