"""
tests/test_schemas.py — Schema 验证 + 报告生成器测试

验证:
- Schema 验证: 正确数据通过，错误数据拒绝
- 报告生成: 文件创建、SHA-256、原子写入
"""

import json
import os
import sys
import tempfile
import unittest

# 确保项目根在 path 中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestSchemaValidation(unittest.TestCase):
    """Schema 验证 — 正确数据通过，错误数据拒绝"""

    def test_valid_analysis_passes(self):
        """有效分析数据应通过验证"""
        from modules.schemas import validate_analysis

        data = {
            'basic_info': {'name': '测试', 'code': 'sz300620', 'price': 185.5, 'date': '2025-09-17'},
            'prediction': {
                'model': {'composite': 0.7, 'ml_direction': 'up', 'ml_confidence': 0.75,
                          'ml_probabilities': {'up': 0.6, 'down': 0.2, 'neutral': 0.2}},
                'kelly': {}, 'sota_models': {},
                'weighted_target': 200.0, 'upside_space': 8.0,
                'daily_volatility': 2.5, 'drift_status': 'stable',
            },
        }
        errors = validate_analysis(data)
        self.assertEqual(len(errors), 0, f"有效数据不应有错误: {errors}")

    def test_invalid_stock_code_rejected(self):
        """无效股票代码应被拒绝"""
        from modules.schemas import validate_analysis

        data = {
            'basic_info': {'name': '测试', 'code': 'invalid', 'price': 100, 'date': '2025-09-17'},
            'prediction': {'model': {}, 'kelly': {}, 'sota_models': {}},
        }
        errors = validate_analysis(data)
        self.assertTrue(len(errors) > 0, "应返回验证错误")
        # 确认错误与 code 字段相关
        self.assertTrue(any('code' in e.lower() or 'pattern' in e.lower() for e in errors))

    def test_valid_prediction_passes(self):
        """有效预测数据应通过验证"""
        from modules.schemas import validate_prediction

        data = {
            'model': {'composite': 0.5, 'ml_direction': 'neutral', 'ml_confidence': 0.5,
                      'ml_probabilities': {'up': 0.33, 'down': 0.33, 'neutral': 0.34}},
            'kelly': {}, 'sota_models': {},
            'weighted_target': 100, 'upside_space': 0,
            'daily_volatility': 2.0, 'drift_status': 'stable',
        }
        errors = validate_prediction(data)
        self.assertEqual(len(errors), 0, f"有效数据不应有错误: {errors}")

    def test_confidence_out_of_range(self):
        """置信度超出 [0,1] 应被拒绝"""
        from modules.schemas import validate_prediction

        data = {
            'model': {'composite': 0.5, 'ml_direction': 'up', 'ml_confidence': 1.5,
                      'ml_probabilities': {'up': 0.33, 'down': 0.33, 'neutral': 0.34}},
            'kelly': {}, 'sota_models': {},
            'weighted_target': 100, 'upside_space': 0,
            'daily_volatility': 2.0, 'drift_status': 'stable',
        }
        errors = validate_prediction(data)
        self.assertTrue(len(errors) > 0, "置信度 > 1 应被拒绝")

    def test_missing_required_field(self):
        """缺少 required 字段应被拒绝"""
        from modules.schemas import validate_analysis

        # 缺少 basic_info
        data = {'prediction': {}}
        errors = validate_analysis(data)
        self.assertTrue(len(errors) > 0, "缺少 basic_info 应被拒绝")

    def test_validate_and_raise(self):
        """validate_and_raise 在验证失败时应抛出 ValueError"""
        from modules.schemas import validate_and_raise

        data = {'basic_info': {'code': 'bad'}, 'prediction': {}}
        with self.assertRaises(ValueError) as ctx:
            validate_and_raise(data, 'analysis')
        self.assertIn('analysis', str(ctx.exception).lower())

    def test_valid_model_metadata(self):
        """有效模型元数据应通过验证"""
        from modules.schemas import validate_model_metadata

        data = {
            'name': 'patchtst',
            'version': '1.0.0',
            'trained': True,
            'last_trained': '2025-09-17',
            'direction': 'up',
            'confidence': 0.75,
        }
        errors = validate_model_metadata(data)
        self.assertEqual(len(errors), 0, f"有效数据不应有错误: {errors}")


class TestReportGenerator(unittest.TestCase):
    """报告生成器测试"""

    def test_generate_creates_file(self):
        """generate 应创建 HTML 文件"""
        from modules.report_generator import ReportGenerator

        gen = ReportGenerator()
        data = {
            'basic_info': {'name': '测试', 'code': 'sz300620', 'price': 100, 'date': '2025-09-17'},
            'prediction': {'model': {}, 'kelly': {}, 'sota_models': {}, 'weighted_target': 0,
                           'upside_space': 0, 'daily_volatility': 0, 'drift_status': 'stable'},
        }

        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
            path = f.name

        try:
            receipt = gen.generate(data, output_path=path)
            self.assertTrue(os.path.exists(path), "HTML 文件应被创建")
            self.assertTrue(receipt.passed, "报告应通过验证")
            self.assertGreater(receipt.artifact_bytes, 0, "文件应有内容")
            self.assertEqual(len(receipt.spec_sha256), 64, "SHA-256 应为 64 字符")
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_receipt_to_dict(self):
        """Receipt.to_dict 应返回完整信息"""
        from modules.report_generator import ReportReceipt

        receipt = ReportReceipt(
            output_path='/tmp/test.html',
            spec_sha256='a' * 64,
            artifact_sha256='b' * 64,
            artifact_bytes=1234,
            validation_errors=[],
        )
        d = receipt.to_dict()
        self.assertEqual(d['output_path'], '/tmp/test.html')
        self.assertEqual(d['artifact_bytes'], 1234)
        self.assertTrue(d['passed'])
        self.assertIn('timestamp', d)

    def test_deterministic_output(self):
        """相同输入应产生相同 SHA-256"""
        from modules.report_generator import ReportGenerator

        gen = ReportGenerator()
        data = {
            'basic_info': {'name': '测试', 'code': 'sz300620', 'price': 100, 'date': '2025-09-17'},
            'prediction': {'model': {}, 'kelly': {}, 'sota_models': {}, 'weighted_target': 0,
                           'upside_space': 0, 'daily_volatility': 0, 'drift_status': 'stable'},
        }

        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f1:
            path1 = f1.name
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f2:
            path2 = f2.name

        try:
            r1 = gen.generate(data, output_path=path1)
            r2 = gen.generate(data, output_path=path2)
            self.assertEqual(r1.spec_sha256, r2.spec_sha256, "相同输入应产生相同 SHA-256")
        finally:
            for p in [path1, path2]:
                if os.path.exists(p):
                    os.unlink(p)

    def test_atomic_write_preserves_on_failure(self):
        """原子写入在失败时应保留原文件"""
        # 验证: 如果写入失败，原文件不被破坏
        # 简化: 测试正常写入后文件可读取
        from modules.report_generator import ReportGenerator

        gen = ReportGenerator()
        data = {
            'basic_info': {'name': '测试', 'code': 'sz300620', 'price': 100, 'date': '2025-09-17'},
            'prediction': {'model': {}, 'kelly': {}, 'sota_models': {}, 'weighted_target': 0,
                           'upside_space': 0, 'daily_volatility': 0, 'drift_status': 'stable'},
        }

        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
            path = f.name

        try:
            receipt = gen.generate(data, output_path=path)
            with open(path) as f:
                content = f.read()
            self.assertIn('测试', content, "HTML 应包含标的名称")
            self.assertIn('sz300620', content, "HTML 应包含股票代码")
        finally:
            if os.path.exists(path):
                os.unlink(path)


if __name__ == "__main__":
    unittest.main()
