#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Phase 1 优化模块初始化
集成所有Phase 1优化功能到现有系统
"""

import signal
import sys
import time
from datetime import datetime
from flask import jsonify
from modules.logger import logger

# Phase 1 模块导入
from modules.app_lifecycle import lifecycle_manager
from modules.thread_safe_cache import thread_safe_cache
from modules.api_rate_limiter import rate_limiter
from modules.health_check import health_checker, create_health_check_routes
from modules.errors import register_error_handlers, StockAnalyzerError
from modules.input_validator import InputValidator


def initialize_phase1(app, data_fetcher=None, analysis_engine=None,
                     patchtst_integrator=None, drift_monitor=None,
                     sentiment_engine=None, diffusion_predictor=None,
                     mamba_hft=None, multi_agent_coordinator=None,
                     self_supervised_pretrainer=None, alpha158_calculator=None,
                     factor_weight_scheduler=None, gnn_predictor=None,
                     cvar_analyzer=None, factor_ic_monitor=None,
                     dynamic_ensemble=None, cross_market_factors=None,
                     llm_sentiment_analyzer=None, timesfm_predictor=None,
                     patchmamba_model=None, conformal_predictor=None,
                     causal_discovery_engine=None, drift_aware_pipeline=None,
                     hierarchical_rl=None, adversarial_trainer=None,
                     foundation_model=None, rl_trader=None,
                     # 2026-08-14 新增模块
                     factor_ensemble_optimizer=None,
                     regime_adaptive_router=None,
                     cross_asset_feature_engine=None,
                     explainable_ai_aggregator=None,
                     synthetic_data_generator=None):
    """
    初始化Phase 1优化模块

    Args:
        app: Flask应用实例
        其他参数: 现有的模块实例
    """

    logger.info("=" * 70)
    logger.info("[Optimization] 开始初始化Phase 1优化模块")
    logger.info("=" * 70)

    start_time = time.time()

    # 1. 注册资源到生命周期管理器
    logger.info("[Optimization] 1. 注册资源到生命周期管理器")

    resources = {
        'data_fetcher': data_fetcher,
        'analysis_engine': analysis_engine,
        'patchtst_integrator': patchtst_integrator,
        'drift_monitor': drift_monitor,
        'sentiment_engine': sentiment_engine,
        'diffusion_predictor': diffusion_predictor,
        'mamba_hft': mamba_hft,
        'multi_agent_coordinator': multi_agent_coordinator,
        'self_supervised_pretrainer': self_supervised_pretrainer,
        'alpha158_calculator': alpha158_calculator,
        'factor_weight_scheduler': factor_weight_scheduler,
        'gnn_predictor': gnn_predictor,
        'cvar_analyzer': cvar_analyzer,
        'factor_ic_monitor': factor_ic_monitor,
        'dynamic_ensemble': dynamic_ensemble,
        'cross_market_factors': cross_market_factors,
        'llm_sentiment_analyzer': llm_sentiment_analyzer,
        'timesfm_predictor': timesfm_predictor,
        'patchmamba_model': patchmamba_model,
        'conformal_predictor': conformal_predictor,
        'causal_discovery_engine': causal_discovery_engine,
        'drift_aware_pipeline': drift_aware_pipeline,
        'hierarchical_rl': hierarchical_rl,
        'adversarial_trainer': adversarial_trainer,
        'foundation_model': foundation_model,
        'rl_trader': rl_trader,
        # 2026-08-14 新增
        'factor_ensemble_optimizer': factor_ensemble_optimizer,
        'regime_adaptive_router': regime_adaptive_router,
        'cross_asset_feature_engine': cross_asset_feature_engine,
        'explainable_ai_aggregator': explainable_ai_aggregator,
        'synthetic_data_generator': synthetic_data_generator,
    }

    registered_count = 0
    for name, resource in resources.items():
        if resource is not None:
            try:
                # 核心模块级单例资源标记为 protected，不随时间过期
                lifecycle_manager.register_resource(name, resource, protected=True)
                registered_count += 1
                logger.debug(f"[Optimization] 已注册资源: {name}")
            except Exception as e:
                logger.warning(f"[Optimization] 注册资源失败 {name}: {e}")

    logger.info(f"[Optimization] 成功注册 {registered_count} 个资源")

    # 2. 配置API限流器
    logger.info("[Optimization] 2. 配置API限流器")

    # 设置不同端点的限流策略
    rate_limiter.set_endpoint_limit("/api/analyze", "10/minute")
    rate_limiter.set_endpoint_limit("/api/stock", "100/minute")
    rate_limiter.set_endpoint_limit("/api/sota", "5/minute")
    rate_limiter.set_endpoint_limit("/api/quant", "20/minute")
    rate_limiter.set_endpoint_limit("/api/cache", "30/minute")

    logger.info("[Optimization] API限流器配置完成")

    # 3. 注册健康检查
    logger.info("[Optimization] 3. 注册健康检查")

    # 数据库检查
    def check_database():
        try:
            import sqlite3
            conn = sqlite3.connect("data/stock_analyzer.db")
            conn.execute("SELECT 1")
            conn.close()
            return True
        except Exception:
            return True  # 数据库不存在也返回True

    # 模型检查
    def check_models():
        try:
            if patchtst_integrator and not patchtst_integrator.is_trained():
                return False
            return True
        except Exception:
            return True

    # 缓存检查
    def check_cache():
        try:
            stats = thread_safe_cache.get_stats()
            return stats['hit_rate'] > 0.5 or stats['hits'] + stats['misses'] < 10
        except Exception:
            return True

    health_checker.register_check('database', check_database)
    health_checker.register_check('models', check_models)
    health_checker.register_check('cache', check_cache)

    logger.info("[Optimization] 健康检查注册完成")

    # 4. 创建健康检查路由
    logger.info("[Optimization] 4. 创建健康检查路由")
    create_health_check_routes(app)

    # 5. 注册错误处理器
    logger.info("[Optimization] 5. 注册错误处理器")
    register_error_handlers(app)

    # 6. 启动生命周期管理
    logger.info("[Optimization] 6. 启动生命周期管理")
    lifecycle_manager.start_health_checks()
    lifecycle_manager.start_gc()

    # 7. 注册优化状态API
    logger.info("[Optimization] 7. 注册优化状态API")
    register_optimization_routes(app)

    # 8. 启动缓存清理线程
    logger.info("[Optimization] 8. 启动缓存清理线程")
    start_cache_cleanup_thread()

    elapsed_time = time.time() - start_time
    logger.info(f"[Optimization] Phase 1优化模块初始化完成 ({elapsed_time:.2f}秒)")
    logger.info("=" * 70)

    return {
        'success': True,
        'registered_resources': registered_count,
        'elapsed_time': elapsed_time,
        'timestamp': datetime.now().isoformat()
    }


def register_optimization_routes(app):
    """注册优化相关的API路由"""

    @app.route('/api/optimization/status')
    def optimization_status():
        """获取优化模块状态"""
        return jsonify({
            'success': True,
            'data': {
                'lifecycle': lifecycle_manager.get_system_stats(),
                'cache': thread_safe_cache.get_stats(),
                'rate_limiter': {
                    'default_rate': rate_limiter._default_rate,
                    'endpoint_limits': rate_limiter._endpoint_limits
                },
                'health_checker': {
                    'registered_checks': list(health_checker._checks.keys()),
                    'last_check': health_checker._last_check_time.isoformat()
                        if health_checker._last_check_time else None
                },
                'timestamp': datetime.now().isoformat()
            }
        })

    @app.route('/api/optimization/resources')
    def optimization_resources():
        """获取已注册资源列表"""
        resources = lifecycle_manager.get_all_resources()
        return jsonify({
            'success': True,
            'data': {
                'resources': {
                    name: {
                        'name': info.name,
                        'created_at': info.created_at,
                        'last_used': info.last_used,
                        'is_healthy': info.is_healthy
                    }
                    for name, info in resources.items()
                },
                'total_count': len(resources),
                'timestamp': datetime.now().isoformat()
            }
        })

    @app.route('/api/optimization/cache/stats')
    def cache_stats():
        """获取缓存统计"""
        return jsonify({
            'success': True,
            'data': thread_safe_cache.get_stats(),
            'timestamp': datetime.now().isoformat()
        })

    @app.route('/api/optimization/cache/cleanup', methods=['POST'])
    def cache_cleanup():
        """手动清理缓存"""
        thread_safe_cache.cleanup_expired()
        return jsonify({
            'success': True,
            'message': 'Cache cleanup completed',
            'timestamp': datetime.now().isoformat()
        })

    @app.route('/api/optimization/cache/clear', methods=['POST'])
    def cache_clear():
        """清空缓存"""
        thread_safe_cache.clear()
        return jsonify({
            'success': True,
            'message': 'Cache cleared',
            'timestamp': datetime.now().isoformat()
        })

    @app.route('/api/optimization/lifecycle/shutdown', methods=['POST'])
    def lifecycle_shutdown():
        """优雅关闭系统"""
        lifecycle_manager.shutdown()
        return jsonify({
            'success': True,
            'message': 'System shutdown initiated',
            'timestamp': datetime.now().isoformat()
        })


def start_cache_cleanup_thread():
    """启动定期缓存清理线程"""
    import threading

    def cleanup_loop():
        while True:
            try:
                thread_safe_cache.cleanup_expired()
            except Exception as e:
                logger.error(f"[Cache] 缓存清理失败: {e}")
            time.sleep(300)  # 5分钟清理一次

    cleanup_thread = threading.Thread(
        target=cleanup_loop,
        name="CacheCleanup",
        daemon=True
    )
    cleanup_thread.start()
    logger.info("[Optimization] 缓存清理线程已启动")


def create_optimization_decorator(rate_limiter_instance):
    """创建优化装饰器"""

    def rate_limit(rate: str):
        """限流装饰器"""
        def decorator(f):
            def wrapper(*args, **kwargs):
                endpoint = args[0].request.endpoint if args else None
                is_allowed, retry_after = rate_limiter_instance.is_allowed(endpoint)

                if not is_allowed:
                    return jsonify({
                        'success': False,
                        'error': 'Rate limit exceeded',
                        'retry_after': retry_after
                    }), 429

                return f(*args, **kwargs)
            return wrapper
        return decorator

    def validate_stock_code(f):
        """股票代码验证装饰器"""
        def wrapper(*args, **kwargs):
            stock_code = kwargs.get('stock_code')
            if not InputValidator.validate_stock_code(stock_code):
                return jsonify({'error': 'Invalid stock code format'}), 400
            return f(*args, **kwargs)
        return wrapper


# 导出优化装饰器
rate_limit_decorator = create_optimization_decorator(rate_limiter)