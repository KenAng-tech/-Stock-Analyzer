#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
应用生命周期管理器
管理所有资源的创建、使用和销毁
"""

import gc
import time
import threading
from typing import List, Callable, Dict, Optional
from dataclasses import dataclass, field
from modules.logger import logger


@dataclass
class ResourceInfo:
    """资源信息"""
    name: str
    created_at: float
    last_used: float
    is_healthy: bool = True
    memory_usage: int = 0


class ApplicationLifecycle:
    """
    应用生命周期管理器

    功能:
    - 资源注册与追踪
    - 自动垃圾回收
    - 健康检查
    - 优雅关闭
    """

    _instance: Optional['ApplicationLifecycle'] = None
    _lock = threading.Lock()

    def __init__(self):
        self._resources: Dict[str, ResourceInfo] = {}
        self._cleanup_functions: List[Callable] = []
        self._health_check_interval = 300  # 5分钟
        self._max_resource_age = 86400  # 24小时
        self._gc_interval = 600  # 10分钟
        self._is_shutting_down = False
        self._health_check_thread = None
        self._gc_thread = None
        self._resource_lock = threading.RLock()
        self._protected_resources: set = set()  # 启动时注册的核心资源，不过期

    @classmethod
    def get_instance(cls) -> 'ApplicationLifecycle':
        """获取单例实例"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
                logger.info("[Lifecycle] 应用生命周期管理器已初始化")
            return cls._instance

    def register_resource(self, name: str, resource=None, cleanup_func: Callable = None, protected: bool = False):
        """
        注册资源

        Args:
            name: 资源名称
            resource: 资源对象
            cleanup_func: 清理函数
            protected: 是否为核心资源（不过期，仅健康检查）
        """
        with self._resource_lock:
            self._resources[name] = ResourceInfo(
                name=name,
                created_at=time.time(),
                last_used=time.time(),
                is_healthy=True
            )
            if cleanup_func:
                self._cleanup_functions.append(cleanup_func)
            if protected:
                self._protected_resources.add(name)
            logger.debug(f"[Lifecycle] 资源已注册: {name}" + (" [protected]" if protected else ""))

    def mark_resource_used(self, name: str):
        """标记资源最近使用"""
        with self._resource_lock:
            if name in self._resources:
                self._resources[name].last_used = time.time()

    def update_resource_health(self, name: str, is_healthy: bool):
        """更新资源健康状态"""
        with self._resource_lock:
            if name in self._resources:
                self._resources[name].is_healthy = is_healthy
                if not is_healthy:
                    logger.warning(f"[Lifecycle] 资源不健康: {name}")

    def get_resource_info(self, name: str) -> Optional[ResourceInfo]:
        """获取资源信息"""
        with self._resource_lock:
            return self._resources.get(name)

    def get_all_resources(self) -> Dict[str, ResourceInfo]:
        """获取所有资源信息"""
        with self._resource_lock:
            return self._resources.copy()

    def get_resource_count(self) -> int:
        """获取资源数量"""
        with self._resource_lock:
            return len(self._resources)

    def start_health_checks(self):
        """启动定期健康检查"""
        if self._health_check_thread is None:
            self._health_check_thread = threading.Thread(
                target=self._health_check_loop,
                name="HealthCheck",
                daemon=True
            )
            self._health_check_thread.start()
            logger.info("[Lifecycle] 健康检查线程已启动")

    def start_gc(self):
        """启动定期垃圾回收"""
        if self._gc_thread is None:
            self._gc_thread = threading.Thread(
                target=self._gc_loop,
                name="GarbageCollector",
                daemon=True
            )
            self._gc_thread.start()
            logger.info("[Lifecycle] 垃圾回收线程已启动")

    def _health_check_loop(self):
        """健康检查循环"""
        while not self._is_shutting_down:
            try:
                self._perform_health_check()
            except Exception as e:
                logger.error(f"[Lifecycle] 健康检查失败: {e}")
            time.sleep(self._health_check_interval)

    def _gc_loop(self):
        """垃圾回收循环"""
        while not self._is_shutting_down:
            try:
                self._perform_gc()
            except Exception as e:
                logger.error(f"[Lifecycle] 垃圾回收失败: {e}")
            time.sleep(self._gc_interval)

    def _perform_health_check(self):
        """执行健康检查"""
        with self._resource_lock:
            current_time = time.time()
            stale_resources = []

            for name, info in self._resources.items():
                # 核心资源（启动时注册）不过期，仅检查健康状态
                if name in self._protected_resources:
                    if not info.is_healthy:
                        logger.warning(f"[Lifecycle] 核心资源不健康: {name}")
                    continue

                # 检查资源是否过期
                if current_time - info.last_used > self._max_resource_age:
                    stale_resources.append(name)
                    info.is_healthy = False
                    logger.warning(f"[Lifecycle] 资源过期: {name}")

            # 清理过期资源
            for name in stale_resources:
                self.unregister_resource(name)

    def _perform_gc(self):
        """执行垃圾回收"""
        gc_count = gc.collect()
        logger.debug(f"[Lifecycle] 垃圾回收完成: {gc_count} 个对象")

    def unregister_resource(self, name: str):
        """注销资源"""
        with self._resource_lock:
            if name in self._resources:
                del self._resources[name]
                logger.debug(f"[Lifecycle] 资源已注销: {name}")

    def shutdown(self):
        """优雅关闭"""
        logger.info("[Lifecycle] 开始优雅关闭")
        self._is_shutting_down = True

        # 等待线程结束
        if self._health_check_thread:
            self._health_check_thread.join(timeout=5)
        if self._gc_thread:
            self._gc_thread.join(timeout=5)

        # 执行清理函数
        for func in self._cleanup_functions:
            try:
                func()
            except Exception as e:
                logger.error(f"[Lifecycle] 清理函数执行失败: {e}")

        # 强制垃圾回收
        gc.collect()
        logger.info("[Lifecycle] 优雅关闭完成")

    def get_system_stats(self) -> Dict:
        """获取系统统计信息"""
        try:
            import psutil
            process = psutil.Process()

            return {
                'resource_count': self.get_resource_count(),
                'memory_usage': process.memory_info().rss,
                'cpu_percent': process.cpu_percent(interval=1),
                'thread_count': threading.active_count(),
                'gc_stats': gc.get_stats(),
                'is_shutting_down': self._is_shutting_down
            }
        except Exception as e:
            logger.error(f"[Lifecycle] 获取系统统计失败: {e}")
            return {
                'resource_count': self.get_resource_count(),
                'thread_count': threading.active_count(),
                'is_shutting_down': self._is_shutting_down
            }


# 全局实例
lifecycle_manager = ApplicationLifecycle.get_instance()