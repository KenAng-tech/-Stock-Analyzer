#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
异步任务管理器
"""

import asyncio
import uuid
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, Optional, Callable, Any
from enum import Enum
from modules.logger import logger


class TaskStatus(Enum):
    """任务状态"""
    PENDING = 'pending'
    RUNNING = 'running'
    COMPLETED = 'completed'
    FAILED = 'failed'
    CANCELLED = 'cancelled'


@dataclass
class Task:
    """任务"""
    id: str
    name: str
    func: Callable
    args: tuple
    kwargs: dict
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[Exception] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None


class AsyncTaskManager:
    """
    异步任务管理器
    """

    def __init__(self, max_workers: int = 8):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._tasks: Dict[str, Task] = {}
        self._lock = asyncio.Lock()

    async def submit_task(self, name: str, func: Callable, *args, **kwargs) -> str:
        """提交异步任务"""
        task_id = str(uuid.uuid4())
        task = Task(
            id=task_id,
            name=name,
            func=func,
            args=args,
            kwargs=kwargs,
            status=TaskStatus.RUNNING
        )

        async with self._lock:
            self._tasks[task_id] = task

        # 异步执行
        asyncio.create_task(self._execute_task(task))

        logger.info(f"[AsyncTask] 任务已提交: {task_id} ({name})")
        return task_id

    async def _execute_task(self, task: Task):
        """执行任务"""
        try:
            task.started_at = time.time()
            task.result = await asyncio.get_event_loop().run_in_executor(
                self.executor, task.func, *task.args, **task.kwargs
            )
            task.status = TaskStatus.COMPLETED
        except Exception as e:
            task.error = e
            task.status = TaskStatus.FAILED
            logger.error(f"[AsyncTask] 任务执行失败 {task.id}: {e}")
        finally:
            task.completed_at = time.time()

    async def get_task_status(self, task_id: str) -> Optional[Dict]:
        """获取任务状态"""
        async with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return None

            return {
                'id': task.id,
                'name': task.name,
                'status': task.status.value,
                'result': task.result,
                'error': str(task.error) if task.error else None,
                'created_at': task.created_at,
                'started_at': task.started_at,
                'completed_at': task.completed_at
            }

    async def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        async with self._lock:
            task = self._tasks.get(task_id)
            if task and task.status == TaskStatus.PENDING:
                task.status = TaskStatus.CANCELLED
                return True
        return False


# 全局任务管理器
async_task_manager = AsyncTaskManager()