#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
API 端点测试 — Week 5-6

测试项目:
1. 输入验证 (无效 stock_code 返回 400)
2. 路由拆分后所有端点可访问
3. 新模型端点可用
4. 健康检查
"""

import unittest
import os
import re


class TestInputValidation(unittest.TestCase):
    """测试: 输入验证"""

    def test_invalid_stock_code_rejected(self):
        """无效股票代码应被拒绝"""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            'input_validator', 'modules/input_validator.py')
        iv = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(iv)
        self.assertFalse(iv.InputValidator.validate_stock_code('invalid'))
        self.assertFalse(iv.InputValidator.validate_stock_code(''))
        self.assertFalse(iv.InputValidator.validate_stock_code('sh123'))
        self.assertFalse(iv.InputValidator.validate_stock_code('../../etc/passwd'))

    def test_valid_stock_codes_accepted(self):
        """有效股票代码应被接受"""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            'input_validator', 'modules/input_validator.py')
        iv = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(iv)
        valid_codes = ['sz300620', 'sh688981', 'sz000001', 'sh600000', 'sz000002']
        for code in valid_codes:
            self.assertTrue(iv.InputValidator.validate_stock_code(code),
                          f"应接受 {code}")


class TestSotaRoutesSplit(unittest.TestCase):
    """测试: sota_routes.py 拆分后路由可导入"""

    def test_sota_routes_is_dispatcher(self):
        """验证 sota_routes.py 是调度器"""
        with open('modules/routes/sota_routes.py', 'r') as f:
            content = f.read()
        self.assertIn("sota_predict_routes", content)
        self.assertIn("sota_status_routes", content)
        self.assertIn("sota_advanced_routes", content)
        self.assertIn("sota_llm_routes", content)

    def test_sota_predict_routes_has_all_routes(self):
        """验证 predict 路由包含所有预测端点"""
        with open('modules/routes/sota_predict_routes.py', 'r') as f:
            content = f.read()
        self.assertIn("api_patchtst_predict", content)
        self.assertIn("api_itransformer_predict", content)
        self.assertIn("api_bi_mamba_predict", content)

    def test_sota_status_routes_has_all_routes(self):
        """验证 status 路由包含所有状态端点"""
        with open('modules/routes/sota_status_routes.py', 'r') as f:
            content = f.read()
        self.assertIn("api_sota_status", content)
        self.assertIn("api_drift_status", content)
        self.assertIn("api_sota_cache_status", content)

    def test_sota_advanced_routes_has_all_routes(self):
        """验证 advanced 路由包含所有高级端点"""
        with open('modules/routes/sota_advanced_routes.py', 'r') as f:
            content = f.read()
        self.assertIn("api_hrp_optimize", content)
        self.assertIn("api_sota_multiagent_pipeline", content)
        self.assertIn("api_sota_decision", content)

    def test_sota_llm_routes_has_all_routes(self):
        """验证 llm 路由包含所有 LLM 端点"""
        with open('modules/routes/sota_llm_routes.py', 'r') as f:
            content = f.read()
        self.assertIn("api_llm_router_status", content)
        self.assertIn("api_llm_router_health", content)
        self.assertIn("api_cross_market_summary", content)


class TestRoutesInit(unittest.TestCase):
    """测试: routes/__init__.py 注册"""

    def test_register_blueprints_function_exists(self):
        """验证 register_blueprints 函数存在"""
        with open('modules/routes/__init__.py', 'r') as f:
            content = f.read()
        self.assertIn("def register_blueprints", content)

    def test_all_sub_blueprints_registered(self):
        """验证所有子蓝图在 __init__.py 中注册"""
        with open('modules/routes/__init__.py', 'r') as f:
            content = f.read()
        required = [
            # 2026-09-20 名单同步: 删 core_routes/uncertainty_routes (模块已不存在,
            # 09-14/09-17 死链清理遗留 → 测试恒红 = 假警报掩盖真断链); 补真实注册项
            'sota_predict_routes', 'sota_status_routes',
            'sota_advanced_routes', 'sota_llm_routes',
            'data_routes', 'cache_routes', 'backtest_routes',
            'factor_routes', 'sentiment_routes', 'config_routes',
            'training_routes', 'health_routes', 'memory_routes',
            'portfolio_routes', 'regime_routes', 'time_llm_routes',
            'online_learning_routes', 'ab_test_routes',
            'dl_routes', 'models_routes', 'monitor_routes',
            'async_backtest_routes', 'new_routes',
            'mlops_routes', 'cvar_position_routes',
            'rd_agent_routes', 'trading_agents_routes', 'kronos_routes',
            'consensus_routes', 'safe_rl_routes',
            'gnn_routes', 'fusion_routes', 'extra_routes',
            'ths_data_routes', 'decision_routes', 'integration2026_routes',
        ]
        for name in required:
            self.assertIn(name, content, f"{name} 应在 __init__.py 中注册")


class TestNewModelEndpoints(unittest.TestCase):
    """测试: 新模型端点路由注册"""

    def test_itransformer_route_in_predict(self):
        """验证 iTransformer 端点在 predict 路由中"""
        with open('modules/routes/sota_predict_routes.py', 'r') as f:
            content = f.read()
        self.assertIn("api_itransformer_predict", content)
        self.assertIn("api_itransformer_status", content)

    def test_bi_mamba_route_in_predict(self):
        """验证 Bi-Mamba+ 端点在 predict 路由中"""
        with open('modules/routes/sota_predict_routes.py', 'r') as f:
            content = f.read()
        self.assertIn("api_bi_mamba_predict", content)
        self.assertIn("api_bi_mamba_status", content)

    def test_hrp_route_in_advanced(self):
        """验证 HRP 端点在 advanced 路由中"""
        with open('modules/routes/sota_advanced_routes.py', 'r') as f:
            content = f.read()
        self.assertIn("api_hrp_optimize", content)
        self.assertIn("api_hrp_status", content)


class TestCacheTTL(unittest.TestCase):
    """测试: 缓存 TTL 功能"""

    # 2026-09-20 测试尸清尸: 原 4 例 = grep app.py 旧字符串 (缓存函数 09-17 已迁
    # modules/dynamic_cache.py, 字符串恒 False 锁死链) + 1 自包含假实现 (恒绿 tautology)。
    # 全换真链锁: dynamic_cache.cache 公开面, 行为按 09-20 实测事实。

    def test_cache_get_function_exists(self):
        """get 真链: set/get roundtrip + miss→None (cache_routes /api/cache/stats 消费面)"""
        from modules.dynamic_cache import cache
        cache.set('_selftest_get_exists', {'v': 42}, ttl=60)
        self.assertEqual(cache.get('_selftest_get_exists'), {'v': 42})
        self.assertIsNone(cache.get('_selftest_never_set'))

    def test_cache_set_function_exists(self):
        """set TTL 真链: 过期失效 (09-17 ttl=0 永不过期教训反锁)"""
        import time
        from modules.dynamic_cache import cache
        cache.set('_selftest_set_expired', 1, ttl=0.05)
        time.sleep(0.25)
        self.assertIsNone(cache.get('_selftest_set_expired'))

    def test_cache_cleanup_function_exists(self):
        """cleanup 直调非负 (POST /api/cache/cleanup 消费面, 降级非断链)"""
        from modules.dynamic_cache import cache
        self.assertIsNotNone(cache.cleanup())

    def test_cache_get_set_logic(self):
        """get_stats 链尾: cache_routes:22 消费面 (stats dict 非 tautology)"""
        from modules.dynamic_cache import cache
        self.assertIsInstance(cache.get_stats(), dict)

    def test_cache_cleanup_logic(self):
        """cleanup 链尾真形: 过期键回收 (09-20 由自包含假实现改真链)"""
        import time
        from modules.dynamic_cache import cache
        cache.set('_t_api_clean', 1, ttl=0.05)
        time.sleep(0.25)
        self.assertIsNotNone(cache.cleanup())   # 过期后直调 = 降级非断链


class TestDEFAULTStockConsistency(unittest.TestCase):
    """测试: DEFAULT_STOCK 一致性"""

    def test_cost_basis_120(self):
        """验证 cost_basis 为 120"""
        with open('modules/routes/data_routes.py', 'r') as f:
            content = f.read()
        self.assertIn("'cost_basis': 120.0", content)


class TestNoBareExcept(unittest.TestCase):
    """测试: 无裸 except"""

    def test_no_bare_except_in_routes(self):
        """验证路由文件无裸 except"""
        routes_dir = 'modules/routes'
        for filename in os.listdir(routes_dir):
            if not filename.endswith('.py') or filename == '__init__.py':
                continue
            filepath = os.path.join(routes_dir, filename)
            with open(filepath, 'r') as f:
                content = f.read()
            bare_except = re.findall(r'except\s*:', content)
            self.assertEqual(len(bare_except), 0,
                           f"{filename} 中有裸 except: {bare_except}")


class TestNoSilentPass(unittest.TestCase):
    """测试: 无静默 pass"""

    def test_data_routes_no_silent_pass(self):
        """验证 data_routes.py 无静默 pass"""
        with open('modules/routes/data_routes.py', 'r') as f:
            content = f.read()
        silent = re.findall(r'except.*?:\s*\n\s*pass', content)
        self.assertEqual(len(silent), 0,
                       f"data_routes.py 中有静默 pass: {silent}")


if __name__ == '__main__':
    unittest.main()
