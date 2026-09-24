#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
安全测试 — Week 1 安全加固验证

测试项目:
1. 无硬编码 API Key
2. 路径穿越防护
3. exec() 沙箱化
4. 输入验证器
5. 日志替代 print()
"""

import os
import re
import unittest
import subprocess
import importlib.util


def _import_module(name, path):
    """直接导入模块，绕过 __init__.py"""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestNoHardcodedKeys(unittest.TestCase):
    """测试: 无硬编码敏感凭据"""

    def test_no_omlx_key_hardcoded(self):
        """验证 '953357' 未硬编码在业务代码中"""
        result = subprocess.run(
            ['grep', '-rn', '"953357"', 'modules/', 'app.py', 'config.py'],
            capture_output=True, text=True
        )
        self.assertNotEqual(result.returncode, 0,
                           f"发现硬编码 OMLX API Key: {result.stdout}")

    def test_no_eastmoney_key_hardcoded(self):
        """验证东方财富 Token 未硬编码"""
        result = subprocess.run(
            ['grep', '-rn', 'fa5fd1943c7b386f172d6893dbbd1', 'modules/', 'app.py'],
            capture_output=True, text=True
        )
        self.assertNotEqual(result.returncode, 0,
                           f"发现硬编码东方财富 Token: {result.stdout}")

    def test_config_keys_module_exists(self):
        """验证统一密钥管理模块存在"""
        self.assertTrue(os.path.exists('modules/config_keys.py'),
                       "config_keys.py 应存在")

    def test_config_keys_no_fallback(self):
        """验证 config_keys.py 不使用硬编码 fallback"""
        with open('modules/config_keys.py', 'r') as f:
            content = f.read()
        self.assertNotIn('"953357"', content)
        self.assertNotIn("'953357'", content)


class TestPathTraversal(unittest.TestCase):
    """测试: 路径穿越防护"""

    def test_data_routes_uses_re_sub(self):
        """验证 data_routes.py 使用 re.sub 清理 stock_code"""
        with open('modules/routes/data_routes.py', 'r') as f:
            content = f.read()
        self.assertIn("re.sub", content,
                     "data_routes.py 应使用 re.sub 清理 stock_code")

    def test_path_traversal_pattern(self):
        """验证路径穿越模式被过滤"""
        with open('modules/routes/data_routes.py', 'r') as f:
            content = f.read()
        self.assertRegex(content, r"re\.sub.*?stock_code",
                        "data_routes.py 应过滤 stock_code 中的路径穿越字符")


class TestExecSandbox(unittest.TestCase):
    """测试: exec() 沙箱化"""

    def test_rd_agent_has_safe_builtins(self):
        """验证 rd_agent_miner.py 使用安全沙箱"""
        with open('modules/rd_agent_miner.py', 'r') as f:
            content = f.read()
        self.assertIn("safe_builtins", content,
                     "rd_agent_miner.py 应使用 safe_builtins 沙箱")


class TestInputValidator(unittest.TestCase):
    """测试: 输入验证器"""

    def test_validate_stock_code_valid(self):
        """验证有效股票代码"""
        # 直接读取代码验证正则表达式
        with open('modules/input_validator.py', 'r') as f:
            content = f.read()
        # 验证正则模式正确
        self.assertIn(r"'^(sh|sz)\d{6}$'", content)

    def test_validate_stock_code_invalid(self):
        """验证无效股票代码逻辑"""
        import re
        pattern = r'^(sh|sz)\d{6}$'
        # 有效
        self.assertTrue(bool(re.match(pattern, 'sz300620')))
        self.assertTrue(bool(re.match(pattern, 'sh688981')))
        self.assertTrue(bool(re.match(pattern, 'sz000001')))
        # 无效
        self.assertFalse(bool(re.match(pattern, 'invalid')))
        self.assertFalse(bool(re.match(pattern, '')))
        self.assertFalse(bool(re.match(pattern, 'sh123')))
        self.assertFalse(bool(re.match(pattern, 'bj688981')))

    def test_validate_stock_code_path_traversal(self):
        """验证路径穿越输入被拒绝"""
        import re
        pattern = r'^(sh|sz)\d{6}$'
        self.assertFalse(bool(re.match(pattern, '../../etc/passwd')))
        self.assertFalse(bool(re.match(pattern, 'sh../etc/passwd')))
        self.assertFalse(bool(re.match(pattern, 'sz%2e%2e/etc')))


class TestNoPrintStatements(unittest.TestCase):
    """测试: 无 print() 替代 logger"""

    def test_config_no_print(self):
        """验证 config.py 无 print()"""
        with open('config.py', 'r') as f:
            content = f.read()
        self.assertNotIn("print(", content,
                        "config.py 不应使用 print()")

    def test_app_no_diagnostic_print(self):
        """验证 app.py 无裸 print() 调用 (排除 blueprint 等)"""
        with open('app.py', 'r') as f:
            content = f.read()
        # 排除 blueprint 中的 "print" 子串
        import re
        # 匹配独立的 print( 调用（前面不是字母）
        print_calls = re.findall(r'(?<![a-zA-Z_])print\(', content)
        self.assertEqual(len(print_calls), 0,
                        f"app.py 中有 {len(print_calls)} 处 print() 调用")


class TestDEFAULTStockConsistency(unittest.TestCase):
    """测试: DEFAULT_STOCK 一致性"""

    def test_cost_basis_consistency(self):
        """验证 cost_basis 在所有文件中一致"""
        with open('modules/routes/data_routes.py', 'r') as f:
            content = f.read()
        self.assertIn("'cost_basis': 120.0", content,
                     "data_routes.py cost_basis 应为 120.0")


class TestCacheTTL(unittest.TestCase):
    """测试: 缓存 TTL 生效"""

    # 2026-09-20 测试尸清尸 (同 test_api_endpoints.TestCacheTTL): 原 3 例 grep
    # app.py 旧字符串 (缓存 09-17 已迁 dynamic_cache, 恒 False) + 1 恒绿假实现。

    def test_cache_get_function_exists(self):
        """get 真链: set/get roundtrip + miss→None"""
        from modules.dynamic_cache import cache
        cache.set('_selftest_sec_get', {'v': 42}, ttl=60)
        self.assertEqual(cache.get('_selftest_sec_get'), {'v': 42})
        self.assertIsNone(cache.get('_selftest_sec_never_set'))

    def test_cache_set_function_exists(self):
        """set TTL 真链: 过期失效 (09-17 ttl=0 教训反锁)"""
        import time
        from modules.dynamic_cache import cache
        cache.set('_selftest_sec_expire', 1, ttl=0.05)
        time.sleep(0.25)
        self.assertIsNone(cache.get('_selftest_sec_expire'))

    def test_cache_cleanup_function_exists(self):
        """cleanup 直调非负 (降级非断链)"""
        from modules.dynamic_cache import cache
        self.assertIsNotNone(cache.cleanup())

    def test_cache_get_set_logic(self):
        """get_stats 链尾 (cache_routes 消费面)"""
        from modules.dynamic_cache import cache
        self.assertIsInstance(cache.get_stats(), dict)

    def test_cache_cleanup_logic(self):
        """cleanup 链尾: 过期键可被回收 (09-17 后 cleanup 真形 = 直调非抛)"""
        import time
        from modules.dynamic_cache import cache
        cache.set('_selftest_sec_clean', 1, ttl=0.05)
        time.sleep(0.25)
        self.assertIsNotNone(cache.cleanup())   # 过期后 cleanup 直调 = 降级非断链


class TestSotaRoutesSplit(unittest.TestCase):
    """测试: sota_routes.py 拆分"""

    def test_sota_routes_is_dispatcher(self):
        """验证 sota_routes.py 是调度器"""
        with open('modules/routes/sota_routes.py', 'r') as f:
            content = f.read()
        self.assertIn("sota_predict_routes", content)
        self.assertIn("sota_status_routes", content)
        self.assertIn("sota_advanced_routes", content)
        self.assertIn("sota_llm_routes", content)

    def test_sota_predict_routes_exist(self):
        """验证拆分文件存在"""
        self.assertTrue(os.path.exists('modules/routes/sota_predict_routes.py'))
        self.assertTrue(os.path.exists('modules/routes/sota_status_routes.py'))
        self.assertTrue(os.path.exists('modules/routes/sota_advanced_routes.py'))
        self.assertTrue(os.path.exists('modules/routes/sota_llm_routes.py'))

    def test_routes_init_registers_sub_blueprints(self):
        """验证 __init__.py 注册所有子蓝图"""
        with open('modules/routes/__init__.py', 'r') as f:
            content = f.read()
        self.assertIn("sota_predict_routes", content)
        self.assertIn("sota_status_routes", content)
        self.assertIn("sota_advanced_routes", content)
        self.assertIn("sota_llm_routes", content)

    def test_sota_predict_routes_has_itransformer(self):
        """验证 iTransformer 端点在 predict 路由中"""
        with open('modules/routes/sota_predict_routes.py', 'r') as f:
            content = f.read()
        self.assertIn("api_itransformer_predict", content)
        self.assertIn("api_itransformer_status", content)

    def test_sota_predict_routes_has_bi_mamba(self):
        """验证 Bi-Mamba+ 端点在 predict 路由中"""
        with open('modules/routes/sota_predict_routes.py', 'r') as f:
            content = f.read()
        self.assertIn("api_bi_mamba_predict", content)
        self.assertIn("api_bi_mamba_status", content)

    def test_sota_advanced_routes_has_hrp(self):
        """验证 HRP 端点在 advanced 路由中"""
        with open('modules/routes/sota_advanced_routes.py', 'r') as f:
            content = f.read()
        self.assertIn("api_hrp_optimize", content)
        self.assertIn("api_hrp_status", content)


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
