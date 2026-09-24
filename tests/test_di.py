#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
DI Container 测试
"""

import unittest
from modules.di import (
    DIContainer, Lifetime, get_container, reset_container,
    register, resolve, register_ml_predictor, register_stock_fetcher,
)


class TestDIContainer(unittest.TestCase):
    """DI Container 基本测试"""

    def setUp(self):
        reset_container()
        self.container = DIContainer()

    def test_register_instance(self):
        """测试实例注册"""
        self.container.register_instance(str, 'hello')
        self.assertEqual(self.container.resolve(str), 'hello')

    def test_register_singleton(self):
        """测试 Singleton 生命周期"""
        self.container.register(str, lifetime=Lifetime.SINGLETON)
        a = self.container.resolve(str)
        b = self.container.resolve(str)
        self.assertIs(a, b)

    def test_register_transient(self):
        """测试 Transient 生命周期"""
        self.container.register(list, lifetime=Lifetime.TRANSIENT)
        a = self.container.resolve(list)
        b = self.container.resolve(list)
        self.assertIsNot(a, b)

    def test_register_factory(self):
        """测试工厂注册"""
        self.container.register(dict, lifetime=Lifetime.SINGLETON, factory=lambda: {'key': 'value'})
        result = self.container.resolve(dict)
        self.assertEqual(result, {'key': 'value'})
        # 第二次应该返回同一实例
        result2 = self.container.resolve(dict)
        self.assertIs(result, result2)

    def test_has(self):
        """测试 has 方法"""
        self.container.register(str)
        self.assertTrue(self.container.has(str))
        self.assertFalse(self.container.has(int))

    def test_get_instance(self):
        """测试 get_instance"""
        self.container.register(str, lifetime=Lifetime.SINGLETON)
        self.assertIsNone(self.container.get_instance(str))
        self.container.resolve(str)
        self.assertIsNotNone(self.container.get_instance(str))

    def test_reset_single(self):
        """测试重置单个服务"""
        self.container.register(str, lifetime=Lifetime.SINGLETON)
        self.container.resolve(str)
        self.container.reset(str)
        self.assertIsNone(self.container.get_instance(str))

    def test_reset_all(self):
        """测试重置所有服务"""
        self.container.register(str, lifetime=Lifetime.SINGLETON)
        self.container.register(list, lifetime=Lifetime.SINGLETON)
        self.container.resolve(str)
        self.container.resolve(list)
        self.container.reset()
        self.assertIsNone(self.container.get_instance(str))
        self.assertIsNone(self.container.get_instance(list))

    def test_get_status(self):
        """测试状态获取"""
        self.container.register(str, lifetime=Lifetime.SINGLETON)
        self.container.resolve(str)
        status = self.container.get_status()
        self.assertEqual(status['name'], 'root')
        self.assertIn(str, status['registered'])
        self.assertIn(str, status['resolved'])

    def test_direct_instantiation(self):
        """测试直接实例化 (未注册)"""
        result = self.container.resolve(str)
        self.assertEqual(result, '')

    def test_chain_registration(self):
        """测试链式注册"""
        result = (self.container
                  .register(str, lifetime=Lifetime.SINGLETON)
                  .register(list, lifetime=Lifetime.TRANSIENT))
        self.assertIs(result, self.container)

    def test_scoped_lifetime(self):
        """测试 Scoped 生命周期"""
        self.container.register(dict, lifetime=Lifetime.SCOPED)
        a = self.container.resolve(dict)
        b = self.container.resolve(dict)
        self.assertIs(a, b)


class TestGlobalContainer(unittest.TestCase):
    """全局容器测试"""

    def setUp(self):
        reset_container()

    def test_get_container_singleton(self):
        """测试全局容器单例"""
        gc1 = get_container()
        gc2 = get_container()
        self.assertIs(gc1, gc2)

    def test_reset_container(self):
        """测试重置全局容器"""
        gc1 = get_container()
        reset_container()
        gc2 = get_container()
        self.assertIsNot(gc1, gc2)

    def test_register_resolve(self):
        """测试快捷注册/解析"""
        register(list, lifetime=Lifetime.TRANSIENT)
        a = resolve(list)
        b = resolve(list)
        self.assertIsNot(a, b)

    def test_register_ml_predictor(self):
        """测试 ML 预测器注册"""
        register_ml_predictor()
        from modules.ml_predictor import MLPredictor
        p = resolve(MLPredictor)
        self.assertIsInstance(p, MLPredictor)

    def test_register_stock_fetcher(self):
        """测试股票数据获取器注册"""
        register_stock_fetcher()
        from modules.data_fetcher import StockDataFetcher
        sf = resolve(StockDataFetcher)
        self.assertIsNotNone(sf)

    def test_register_all(self):
        """测试全部注册"""
        from modules.di import register_all
        register_all()
        from modules.ml_predictor import MLPredictor
        from modules.data_fetcher import StockDataFetcher
        p = resolve(MLPredictor)
        sf = resolve(StockDataFetcher)
        self.assertIsInstance(p, MLPredictor)
        self.assertIsNotNone(sf)


if __name__ == '__main__':
    unittest.main()
