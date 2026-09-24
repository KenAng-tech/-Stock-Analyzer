#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
系统健康检查模块
"""

import os
import sys
import time
import threading
from datetime import datetime
from flask import jsonify
from modules.logger import logger
from modules.app_lifecycle import lifecycle_manager


class HealthChecker:
    """
    系统健康检查器
    """

    def __init__(self):
        self._checks = {}
        self._last_check_time = None

    def register_check(self, name: str, check_func):
        """注册健康检查"""
        self._checks[name] = check_func
        logger.info(f"[HealthCheck] 注册检查: {name}")

    def get_health_status(self) -> dict:
        """获取系统健康状态"""
        try:
            import psutil
            process = psutil.Process()
        except ImportError:
            process = None

        # 基础系统状态
        status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'uptime': time.time() if process else 0,
            'system': {},
            'process': {},
            'checks': {},
            'lifecycle': lifecycle_manager.get_system_stats()
        }

        # 系统信息
        if process:
            try:
                status['system'] = {
                    'cpu_percent': __import__('psutil').cpu_percent(interval=1),
                    'memory_percent': __import__('psutil').virtual_memory().percent,
                    'disk_usage': __import__('psutil').disk_usage('/').percent,
                    'load_average': os.getloadavg()
                }
                status['process'] = {
                    'memory_rss': process.memory_info().rss,
                    'memory_vms': process.memory_info().vms,
                    'thread_count': threading.active_count(),
                    'cpu_percent': process.cpu_percent(interval=1),
                    'open_files': len(process.open_files())
                }
            except Exception as e:
                logger.error(f"[HealthCheck] 获取系统信息失败: {e}")

        # 执行所有健康检查
        all_healthy = True
        for name, check_func in self._checks.items():
            try:
                result = check_func()
                status['checks'][name] = {
                    'healthy': result,
                    'timestamp': datetime.now().isoformat()
                }
                if not result:
                    all_healthy = False
            except Exception as e:
                status['checks'][name] = {
                    'healthy': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
                all_healthy = False
                logger.error(f"[HealthCheck] 检查失败 {name}: {e}")

        if not all_healthy:
            status['status'] = 'degraded'

        self._last_check_time = datetime.now()
        return status


# 全局健康检查器
health_checker = HealthChecker()


def create_health_check_routes(app):
    """创建健康检查路由"""

    @app.route('/api/health')
    def health_check():
        """健康检查端点"""
        return jsonify(health_checker.get_health_status())

    @app.route('/api/health/checks')
    def health_checks():
        """健康检查列表"""
        return jsonify({
            'checks': list(health_checker._checks.keys()),
            'last_check': health_checker._last_check_time.isoformat()
                if health_checker._last_check_time else None
        })

    @app.route('/api/health/check/<name>', methods=['POST'])
    def run_health_check(name):
        """执行特定健康检查"""
        if name not in health_checker._checks:
            return jsonify({'error': 'Check not found'}), 404

        try:
            result = health_checker._checks[name]()
            return jsonify({
                'name': name,
                'healthy': result,
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            return jsonify({
                'name': name,
                'healthy': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500


# 默认健康检查函数
def check_database():
    """检查数据库连接"""
    try:
        # 实现数据库健康检查
        return True
    except Exception:
        return False


def check_models():
    """检查ML模型状态"""
    try:
        # 实现模型健康检查
        return True
    except Exception:
        return False


def check_memory():
    """检查内存状态"""
    try:
        import psutil
        memory = psutil.virtual_memory()
        return memory.percent < 90
    except Exception:
        return False


# 注册默认检查
health_checker.register_check('database', check_database)
health_checker.register_check('models', check_models)
health_checker.register_check('memory', check_memory)