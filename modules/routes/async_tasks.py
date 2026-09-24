"""
async_tasks.py - 异步任务管理器

用于将耗时 API 端点（训练、回测、预测）改为后台执行，
通过 SocketIO 推送进度，前端轮询任务状态。

使用方式:
    # 提交任务
    task_id = submit_task(
        func=training_function,
        args=(stock_code,),
        kwargs={'epochs': 100},
        task_type='training',
    )

    # 查询进度
    status = get_task_status(task_id)

    # 取消任务
    cancel_task(task_id)
"""

import threading
import uuid
import traceback
from datetime import datetime
from typing import Callable, Any, Optional, Dict
from modules.dynamic_cache import cache
from modules.logger import logger

# 任务存储: task_id -> task_info
_tasks: Dict[str, dict] = {}
_lock = threading.Lock()


class AsyncTaskManager:
    """异步任务管理器 — 后台线程池 + 进度追踪"""

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self._queue = []
        self._active = 0

    def submit(self, func: Callable, args: tuple = (), kwargs: dict = None,
               task_type: str = 'generic', stock_code: str = '') -> str:
        """提交后台任务，返回 task_id"""
        task_id = f"task_{uuid.uuid4().hex[:12]}"
        kwargs = kwargs or {}

        task_info = {
            'task_id': task_id,
            'status': 'pending',
            'task_type': task_type,
            'stock_code': stock_code,
            'progress': 0.0,
            'result': None,
            'error': None,
            'started_at': None,
            'completed_at': None,
            'thread': None,
            'func': func,
            'args': args,
            'kwargs': kwargs,
        }

        with _lock:
            _tasks[task_id] = task_info

        # 启动后台线程
        thread = threading.Thread(
            target=self._run_task,
            args=(task_info,),
            daemon=True,
            name=f"async-{task_type}-{task_id[:8]}",
        )
        thread.start()
        task_info['thread'] = thread
        task_info['status'] = 'running'
        task_info['started_at'] = datetime.now().isoformat()

        logger.info(f"[AsyncTask] Submitted: {task_id} type={task_type} stock={stock_code}")
        return task_id

    def _run_task(self, task_info: dict):
        """在后台线程中执行任务"""
        task_id = task_info['task_id']
        func = task_info['func']
        args = task_info['args']
        kwargs = task_info['kwargs']

        try:
            # 执行时传入 progress_callback
            def progress_callback(pct: float):
                self.update_progress(task_id, pct)

            result = func(*args, **kwargs, _progress=progress_callback)
            task_info['status'] = 'completed'
            task_info['result'] = result
            task_info['progress'] = 100.0
        except Exception as e:
            task_info['status'] = 'failed'
            task_info['error'] = traceback.format_exc()
            logger.error(f"[AsyncTask] {task_id} failed: {e}")
        finally:
            task_info['completed_at'] = datetime.now().isoformat()

        # 通知 SocketIO
        try:
            from flask_socketio import emit
            emit('task_progress', {
                'task_id': task_id,
                'status': task_info['status'],
                'progress': task_info['progress'],
            })
        except Exception:
            pass  # SocketIO 不可用时忽略

    def update_progress(self, task_id: str, progress: float):
        """更新任务进度"""
        with _lock:
            if task_id in _tasks:
                _tasks[task_id]['progress'] = min(100.0, max(0.0, progress))

    def get_status(self, task_id: str) -> Optional[dict]:
        """获取任务状态"""
        with _lock:
            task = _tasks.get(task_id)
            if task is None:
                return None
            return {
                'task_id': task['task_id'],
                'status': task['status'],
                'task_type': task['task_type'],
                'stock_code': task['stock_code'],
                'progress': task['progress'],
                'result': task['result'],
                'error': task['error'],
                'started_at': task['started_at'],
                'completed_at': task['completed_at'],
            }

    def cancel(self, task_id: str) -> bool:
        """取消任务（仅标记，实际线程无法强制终止）"""
        with _lock:
            task = _tasks.get(task_id)
            if task and task['status'] == 'running':
                task['status'] = 'cancelled'
                return True
        return False

    def list_tasks(self, task_type: str = None) -> list:
        """列出所有任务"""
        with _lock:
            result = []
            for t in _tasks.values():
                if task_type and t['task_type'] != task_type:
                    continue
                result.append({
                    'task_id': t['task_id'],
                    'status': t['status'],
                    'task_type': t['task_type'],
                    'stock_code': t['stock_code'],
                    'progress': t['progress'],
                    'started_at': t['started_at'],
                    'completed_at': t['completed_at'],
                })
            return result


# 全局单例
_async_manager = AsyncTaskManager(max_workers=4)


def submit_task(func, args=(), kwargs=None, task_type='generic', stock_code=''):
    """快捷函数：提交异步任务"""
    return _async_manager.submit(func, args, kwargs, task_type, stock_code)


def get_task_status(task_id):
    """快捷函数：获取任务状态"""
    return _async_manager.get_status(task_id)


def cancel_task(task_id):
    """快捷函数：取消任务"""
    return _async_manager.cancel(task_id)


def list_tasks(task_type=None):
    """快捷函数：列出任务"""
    return _async_manager.list_tasks(task_type)
