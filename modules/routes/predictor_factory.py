"""
predictor_factory.py — 预测器工厂 v2 (2026-08-12 增强版)

统一管理所有预测器的注册、初始化和调用。
支持按名称获取预测器实例，支持懒加载和故障转移。

## 架构设计

### 为什么需要工厂模式
- app.py 中有 25+ 个 SOTA 模型变量，routes 文件通过 `import app as app_module` 直接访问
- 这导致: 循环导入风险、测试困难、无法统一故障转移
- 工厂模式提供统一接口: `predictor_factory.get('patchtst')` → 预测器实例

### 设计原则
1. **懒加载**: 模型只在首次被请求时初始化
2. **故障转移**: 初始化失败返回 None，不阻断服务
3. **单例**: 每个模型只初始化一次
4. **状态可观测**: `get_status()` 返回所有模型状态

### 使用方式
    from modules.routes.predictor_factory import predictor_factory

    # 获取预测器
    patchtst = predictor_factory.get('patchtst')
    if patchtst:
        result = patchtst.predict(features)

    # 获取所有模型状态
    status = predictor_factory.get_status()

    # 检查模型是否可用
    if predictor_factory.is_available('diffusion'):
        do_something()
"""

import sys
from typing import Optional, Callable, Any, Dict, List
from datetime import datetime
from modules.logger import logger


class PredictorRegistry:
    """预测器注册表 — 管理所有预测器的工厂"""

    def __init__(self):
        self._factories: Dict[str, Callable] = {}
        self._instances: Dict[str, Any] = {}
        self._lock = None  # 延迟初始化（避免导入时 threading 未就绪）

    def _get_lock(self):
        """延迟获取锁"""
        if self._lock is None:
            import threading
            self._lock = threading.Lock()
        return self._lock

    def register(self, name: str, factory: Callable):
        """
        注册预测器工厂函数。

        Args:
            name: 预测器名称（如 'patchtst', 'mamba'）
            factory: 工厂函数，返回预测器实例
        """
        self._factories[name] = factory
        logger.info(f"[PredictorFactory] Registered: {name}")

    def get(self, name: str, force_refresh: bool = False) -> Optional[Any]:
        """
        获取预测器实例（单例模式 + 懒加载）。

        Args:
            name: 预测器名称
            force_refresh: 强制重新创建实例

        Returns:
            预测器实例，或 None（初始化失败）
        """
        if force_refresh:
            with self._get_lock():
                self._instances.pop(name, None)

        if name in self._instances:
            return self._instances[name]

        factory = self._factories.get(name)
        if factory is None:
            logger.warning(f"[PredictorFactory] No factory for: {name}")
            return None

        try:
            instance = factory()
            with self._get_lock():
                self._instances[name] = instance
            if instance is not None:
                logger.info(f"[PredictorFactory] Initialized: {name}")
            return instance
        except Exception as e:
            logger.error(f"[PredictorFactory] Failed to init {name}: {e}")
            return None

    def is_available(self, name: str) -> bool:
        """检查预测器是否可用"""
        return name in self._factories and self.get(name) is not None

    def get_status(self) -> Dict:
        """获取所有预测器状态"""
        status = {}
        for name in self._factories:
            try:
                instance = self.get(name)
                if instance is None:
                    status[name] = {'available': False, 'initialized': False}
                else:
                    status[name] = {
                        'available': True,
                        'initialized': True,
                        'status': getattr(instance, 'get_status', lambda: {})(),
                    }
            except Exception as e:
                status[name] = {'available': False, 'error': str(e)}
        return status

    def list_names(self) -> List[str]:
        """列出所有已注册的预测器名称"""
        return list(self._factories.keys())

    def list_available(self) -> List[str]:
        """列出所有可用的预测器名称"""
        return [name for name in self._factories if self.is_available(name)]


def _get_app_module():
    """获取 app 模块（全局变量定义在 app.py 中）"""
    # 优先从 sys.modules['app'] 获取（run_server.py 启动时 __main__ 是 run_server）
    return sys.modules.get('app') or sys.modules.get('__main__')


def _make_factory(attr_name: str, module_name: str = None):
    """
    通用工厂工厂 — 从 app.py 模块变量创建工厂函数。

    Args:
        attr_name: app.py 中的变量名（如 'patchtst_integrator'）
        module_name: 可选，从哪个模块导入（如果 app.py 中没有）
    """
    def _factory():
        app_module = _get_app_module()
        if app_module and hasattr(app_module, attr_name):
            return getattr(app_module, attr_name)
        # 回退: 尝试直接从模块导入
        if module_name:
            try:
                mod = __import__(module_name, fromlist=[''])
                return getattr(mod, attr_name, None)
            except Exception:
                pass
        return None
    return _factory


# ═══════════════════════════════════════════════════════════
# 全局单例
# ═══════════════════════════════════════════════════════════
predictor_factory = PredictorRegistry()

# ── 核心预测器 (Core Predictors) ─────────────────────────────
predictor_factory.register('patchtst', _make_factory('patchtst_integrator'))
predictor_factory.register('timesfm', _make_factory('timesfm_predictor'))
predictor_factory.register('mamba', _make_factory('mamba_hft'))
predictor_factory.register('diffusion', _make_factory('diffusion_predictor'))
predictor_factory.register('gnn', _make_factory('gnn_predictor'))
predictor_factory.register('conformal', _make_factory('conformal_predictor'))
predictor_factory.register('rl', _make_factory('rl_trader_v2'))

# ── 时间序列模型 (Time Series Models) ────────────────────────
predictor_factory.register('timesnet', _make_factory('timesnet_trainer'))
predictor_factory.register('patchmamba', _make_factory('patchmamba_model'))

# ── 因子与风控 (Factors & Risk) ──────────────────────────────
predictor_factory.register('alpha158', _make_factory('alpha158_calculator'))
predictor_factory.register('factor_weight', _make_factory('factor_weight_scheduler'))
predictor_factory.register('cvar', _make_factory('cvar_analyzer'))
predictor_factory.register('factor_ic', _make_factory('factor_ic_monitor'))
predictor_factory.register('cross_market', _make_factory('cross_market_factors'))

# ── 市场状态与漂移 (Regime & Drift) ──────────────────────────
predictor_factory.register('regime', _make_factory('regime_switching'))
predictor_factory.register('drift', _make_factory('drift_monitor'))
predictor_factory.register('drift_pipeline', _make_factory('drift_aware_pipeline'))

# ── 情感与 NLP (Sentiment & NLP) ─────────────────────────────
predictor_factory.register('sentiment', _make_factory('sentiment_engine'))
predictor_factory.register('llm_sentiment', _make_factory('llm_sentiment_analyzer'))

# ── 高级模型 (Advanced Models) ───────────────────────────────
predictor_factory.register('multi_agent', _make_factory('multi_agent_coordinator'))
predictor_factory.register('hierarchical_rl', _make_factory('hierarchical_rl'))
predictor_factory.register('adversarial', _make_factory('adversarial_trainer'))
predictor_factory.register('foundation', _make_factory('foundation_model'))
predictor_factory.register('dynamic_ensemble', _make_factory('dynamic_ensemble'))
predictor_factory.register('causal', _make_factory('causal_discovery_engine'))

# ── 自监督与预训练 (Self-Supervised) ─────────────────────────
predictor_factory.register('self_supervised', _make_factory('self_supervised_pretrainer'))

# ── 日志注册完成 ─────────────────────────────────────────────
logger.info(f"[PredictorFactory] {len(predictor_factory._factories)} factories registered")
