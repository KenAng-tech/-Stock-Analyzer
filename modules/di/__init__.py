"""
di/ - 依赖注入子模块

提供统一的依赖管理容器。
"""

from .container import (
    DIContainer,
    Lifetime,
    get_container,
    reset_container,
    register,
    resolve,
    register_ml_predictor,
    register_stock_fetcher,
    register_all,
)

__all__ = [
    'DIContainer',
    'Lifetime',
    'get_container',
    'reset_container',
    'register',
    'resolve',
    'register_ml_predictor',
    'register_stock_fetcher',
    'register_all',
]
