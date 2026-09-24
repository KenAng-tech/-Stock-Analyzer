#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
模块重组验证测试

验证重组后的导入路径是否正确工作。
"""

import unittest


class TestModuleRestructure(unittest.TestCase):
    """模块重组导入测试"""

    def test_root_imports(self):
        """测试根目录模块导入"""
        from modules.data_fetcher import StockDataFetcher
        from modules.analysis_engine import AnalysisEngine
        from modules.ml_predictor import MLPredictor
        from modules.kline_signal_analyzer import KlineSignalAnalyzer
        from modules.alert_engine import AlertEngine
        self.assertIsNotNone(StockDataFetcher)
        self.assertIsNotNone(AnalysisEngine)
        self.assertIsNotNone(MLPredictor)

    def test_factors_imports(self):
        """测试 factors/ 子模块导入"""
        from modules.factors import (
            Alpha158Calculator,
            Alpha360Calculator,
            BarraCNE6Calculator,
            FactorICMonitor,
            FactorOrthogonalizer,
            MultiFactorModelV2,
        )
        self.assertIsNotNone(Alpha158Calculator)
        self.assertIsNotNone(FactorICMonitor)
        self.assertIsNotNone(MultiFactorModelV2)

    def test_models_imports(self):
        """测试 models/ 子模块导入"""
        from modules.models import (
            ChronosPredictor,
            DiffusionPredictor,
            DRLTradingAgent,
            FoundationModel,
            GNNPredictor,
            MambaClassifier,
            MoiraiPredictor,
            PatchTSTIntegrator,
            ConformalPredictor,
            TimeMoEPredictor,
            TimesFMPredictor,
            TimesNet,
        )
        self.assertIsNotNone(ChronosPredictor)
        self.assertIsNotNone(DiffusionPredictor)
        self.assertIsNotNone(MoiraiPredictor)

    def test_execution_imports(self):
        """测试 execution/ 子模块导入"""
        from modules.execution import (
            VWAPExecutor,
            TWAPExecutor,
            ImpactCostModel,
            DynamicSlippageModel,
        )
        self.assertIsNotNone(VWAPExecutor)
        self.assertIsNotNone(TWAPExecutor)
        self.assertIsNotNone(ImpactCostModel)
        self.assertIsNotNone(DynamicSlippageModel)

    def test_di_imports(self):
        """测试 di/ 子模块导入"""
        from modules.di import (
            DIContainer,
            Lifetime,
            get_container,
            reset_container,
            register,
            resolve,
        )
        self.assertIsNotNone(DIContainer)
        self.assertIsNotNone(Lifetime)

    def test_main_init_imports(self):
        """测试 modules/__init__.py 的重新导出"""
        from modules import (
            Alpha158Calculator,
            FactorICMonitor,
            FactorOrthogonalizer,
            VWAPExecutor,
            TWAPExecutor,
            ImpactCostModel,
            DynamicSlippageModel,
            MLPredictor,
            ml_predictor,
        )
        self.assertIsNotNone(Alpha158Calculator)
        self.assertIsNotNone(MLPredictor)
        self.assertIsNotNone(ml_predictor)


if __name__ == '__main__':
    unittest.main()
