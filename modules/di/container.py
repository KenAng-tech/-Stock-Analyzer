#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
依赖注入容器 (DI Container)

提供统一的依赖管理:
- 注册/解析服务
- 支持 singleton / transient / scoped 生命周期
- 自动依赖解析
- 向后兼容 (旧代码仍可直接实例化)

用法:
    container = DIContainer()
    container.register(MLPredictor, lifetime=Singleton)
    container.register(StockDataFetcher, lifetime=Singleton)
    predictor = container.resolve(MLPredictor)
"""

import functools
import threading
from typing import Any, Callable, Dict, Optional, Type, TypeVar
from enum import Enum
from datetime import datetime

T = TypeVar('T')


class Lifetime(Enum):
    """服务生命周期"""
    SINGLETON = 'singleton'    # 全局单例，首次解析时创建
    TRANSIENT = 'transient'    # 每次请求创建新实例
    SCOPED = 'scoped'          # 在同一 scope 内共享 (当前仅支持 root scope)


class DIContainer:
    """
    轻量级 DI 容器

    支持:
    1. 类型注册: container.register(MLPredictor, ...))
    2. 实例注册: container.register_instance(ml_predictor, ...))
    3. 工厂注册: container.register_factory(lambda: MLPredictor(), ...))
    4. 自动解析: container.resolve(MLPredictor)
    5. 延迟初始化: 仅在首次 resolve 时创建实例
    """

    def __init__(self, name: str = 'root'):
        self.name = name
        self._registry: Dict[Type, '_Registration'] = {}
        self._instances: Dict[Type, Any] = {}
        self._lock = threading.Lock()
        self._factories: Dict[Type, Callable] = {}

    def register(self, service_type: Type[T],
                 lifetime: Lifetime = Lifetime.TRANSIENT,
                 factory: Optional[Callable] = None) -> 'DIContainer':
        """
        注册服务类型

        Args:
            service_type: 服务类型 (类)
            lifetime: 生命周期
            factory: 可选的工厂函数 (覆盖默认构造函数)

        Returns:
            self (链式调用)
        """
        with self._lock:
            self._registry[service_type] = _Registration(
                service_type=service_type,
                lifetime=lifetime,
                factory=factory,
            )
        return self

    def register_instance(self, service_type: Type[T],
                          instance: T,
                          lifetime: Lifetime = Lifetime.SINGLETON) -> 'DIContainer':
        """
        注册已有实例

        Args:
            service_type: 服务类型
            instance: 实例
            lifetime: 生命周期

        Returns:
            self
        """
        with self._lock:
            self._registry[service_type] = _Registration(
                service_type=service_type,
                lifetime=lifetime,
            )
            self._instances[service_type] = instance
        return self

    def resolve(self, service_type: Type[T]) -> T:
        """
        解析服务

        Args:
            service_type: 服务类型

        Returns:
            服务实例
        """
        with self._lock:
            reg = self._registry.get(service_type)
            if reg is None:
                # 未注册，尝试直接实例化
                return service_type()

            if reg.lifetime == Lifetime.SINGLETON:
                if service_type not in self._instances:
                    if reg.factory:
                        self._instances[service_type] = reg.factory()
                    else:
                        self._instances[service_type] = service_type()
                return self._instances[service_type]

            elif reg.lifetime == Lifetime.TRANSIENT:
                if reg.factory:
                    return reg.factory()
                return service_type()

            else:  # SCOPED
                if service_type not in self._instances:
                    if reg.factory:
                        self._instances[service_type] = reg.factory()
                    else:
                        self._instances[service_type] = service_type()
                return self._instances[service_type]

    def resolve_optional(self, service_type: Type[T]) -> Optional[T]:
        """解析服务，如果未注册则返回 None"""
        with self._lock:
            if service_type in self._registry:
                return self.resolve(service_type)
            return None

    def has(self, service_type: Type) -> bool:
        """检查是否已注册"""
        return service_type in self._registry

    def get_instance(self, service_type: Type[T]) -> Optional[T]:
        """获取已解析的实例 (不创建新实例)"""
        return self._instances.get(service_type)

    def reset(self, service_type: Optional[Type] = None):
        """
        重置服务实例

        Args:
            service_type: 要重置的服务类型，None 则重置所有
        """
        with self._lock:
            if service_type:
                self._instances.pop(service_type, None)
            else:
                self._instances.clear()

    def get_status(self) -> Dict:
        """获取容器状态"""
        return {
            'name': self.name,
            'registered': list(self._registry.keys()),
            'resolved': list(self._instances.keys()),
            'timestamp': datetime.now().isoformat(),
        }


# ── 全局容器 ──────────────────────────────────────────────

_global_container: Optional[DIContainer] = None
_container_lock = threading.Lock()


def get_container() -> DIContainer:
    """获取全局 DI 容器 (懒初始化)"""
    global _global_container
    if _global_container is None:
        with _container_lock:
            if _global_container is None:
                _global_container = DIContainer(name='root')
    return _global_container


def reset_container():
    """重置全局容器 (用于测试)"""
    global _global_container
    with _container_lock:
        _global_container = None


def register(service_type: Type[T],
             lifetime: Lifetime = Lifetime.SINGLETON,
             factory: Optional[Callable] = None) -> DIContainer:
    """快捷注册函数"""
    return get_container().register(service_type, lifetime, factory)


def resolve(service_type: Type[T]) -> T:
    """快捷解析函数"""
    return get_container().resolve(service_type)


def register_ml_predictor(container: Optional[DIContainer] = None):
    """注册 ML 预测器"""
    from modules.ml_predictor import MLPredictor
    c = container or get_container()
    c.register_instance(MLPredictor, MLPredictor(), lifetime=Lifetime.SINGLETON)
    return c


def register_stock_fetcher(container: Optional[DIContainer] = None):
    """注册股票数据获取器"""
    from modules.data_fetcher import StockDataFetcher
    c = container or get_container()
    c.register_instance(StockDataFetcher, StockDataFetcher(), lifetime=Lifetime.SINGLETON)
    return c


def register_all(container: Optional[DIContainer] = None):
    """注册所有常用服务"""
    c = container or get_container()

    # 基础设施
    from modules.data_fetcher import StockDataFetcher
    from modules.ml_predictor import MLPredictor
    from modules.analysis_engine import AnalysisEngine
    from modules.kline_signal_analyzer import KlineSignalAnalyzer
    from modules.alert_engine import AlertEngine
    from modules.report_generator import ReportGenerator
    from modules.heatmap_generator import HeatmapGenerator
    from modules.portfolio_optimizer import PortfolioOptimizer

    c.register_instance(StockDataFetcher, StockDataFetcher(), lifetime=Lifetime.SINGLETON)
    c.register_instance(MLPredictor, MLPredictor(), lifetime=Lifetime.SINGLETON)
    c.register_instance(AnalysisEngine, AnalysisEngine(), lifetime=Lifetime.SINGLETON)
    c.register_instance(KlineSignalAnalyzer, KlineSignalAnalyzer(), lifetime=Lifetime.SINGLETON)
    c.register_instance(AlertEngine, AlertEngine(), lifetime=Lifetime.SINGLETON)
    c.register_instance(ReportGenerator, ReportGenerator(), lifetime=Lifetime.SINGLETON)
    c.register_instance(HeatmapGenerator, HeatmapGenerator(), lifetime=Lifetime.SINGLETON)
    c.register_instance(PortfolioOptimizer, PortfolioOptimizer(), lifetime=Lifetime.SINGLETON)

    return c


class _Registration:
    """服务注册信息"""
    __slots__ = ('service_type', 'lifetime', 'factory')

    def __init__(self, service_type: Type,
                 lifetime: Lifetime,
                 factory: Optional[Callable] = None):
        self.service_type = service_type
        self.lifetime = lifetime
        self.factory = factory
