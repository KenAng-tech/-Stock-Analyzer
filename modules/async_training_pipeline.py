#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
异步训练管线 — 后台训练新模型，不阻塞 API 服务

功能:
- 训练任务队列 (FIFO)
- 后台线程执行训练
- 训练进度追踪
- 训练完成/失败通知 (通过 WebSocket)
- 任务状态管理 (pending/running/completed/failed/cancelled)

用法:
    pipeline = AsyncTrainingPipeline()
    task_id = pipeline.submit('drl', 'sz300620', epochs=100)
    status = pipeline.get_status(task_id)
"""

import os
import sys
import json
import uuid
import threading
import traceback
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from queue import Queue

from modules.logger import logger


class TaskStatus(Enum):
    """任务状态"""
    PENDING = 'pending'
    RUNNING = 'running'
    COMPLETED = 'completed'
    FAILED = 'failed'
    CANCELLED = 'cancelled'


@dataclass
class TrainingTask:
    """训练任务"""
    task_id: str
    model_type: str
    stock_code: str
    params: Dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0
    started_at: str = ''
    completed_at: str = ''
    error: str = ''
    result: Dict = field(default_factory=dict)
    thread: Optional[threading.Thread] = None
    cancel_flag: bool = False

    def to_dict(self) -> Dict:
        # 手动构建 dict，避免 asdict 尝试序列化 thread 对象
        return {
            'task_id': self.task_id,
            'model_type': self.model_type,
            'stock_code': self.stock_code,
            'params': self.params,
            'status': self.status.value,
            'progress': self.progress,
            'started_at': self.started_at,
            'completed_at': self.completed_at,
            'error': self.error,
            'result': self.result,
        }


class AsyncTrainingPipeline:
    """异步训练管线"""

    def __init__(self, max_concurrent: int = 2, queue_size: int = 100):
        self.max_concurrent = max_concurrent
        self.queue_size = queue_size

        # 任务存储
        self._tasks: Dict[str, TrainingTask] = {}
        self._task_queue: Queue = Queue(maxsize=queue_size)

        # 运行中的任务数
        self._running_count = 0
        self._lock = threading.Lock()

        # 调度器线程
        self._scheduler_thread: Optional[threading.Thread] = None
        self._running = False

        # 进度回调
        self._progress_callbacks: Dict[str, Callable] = {}

        logger.info(
            f"[AsyncTraining] 管线初始化: max_concurrent={max_concurrent}, "
            f"queue_size={queue_size}"
        )
        self._start_scheduler()

    def _start_scheduler(self):
        """启动任务调度器"""
        self._running = True
        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            name='TrainingScheduler',
            daemon=True,
        )
        self._scheduler_thread.start()
        logger.info("[AsyncTraining] 调度器已启动")

    def _scheduler_loop(self):
        """调度器主循环 — 从队列取任务并启动训练线程"""
        while self._running:
            try:
                # 检查是否有空闲槽位
                with self._lock:
                    if self._running_count >= self.max_concurrent:
                        import time
                        time.sleep(1)
                        continue

                # 从队列取任务 (阻塞 1 秒)
                try:
                    task = self._task_queue.get(timeout=1)
                except Exception:
                    continue

                if task.status == TaskStatus.CANCELLED:
                    continue

                # 启动训练线程
                task.status = TaskStatus.RUNNING
                task.started_at = datetime.now().isoformat()
                self._running_count += 1

                thread = threading.Thread(
                    target=self._run_task,
                    args=(task,),
                    name=f'Train-{task.model_type}-{task.stock_code[:8]}',
                    daemon=True,
                )
                task.thread = thread
                thread.start()

            except Exception as e:
                logger.error(f"[AsyncTraining] 调度器错误: {e}")

    def _run_task(self, task: TrainingTask):
        """执行训练任务"""
        try:
            logger.info(
                f"[AsyncTraining] 开始训练: {task.model_type} "
                f"({task.stock_code}) epochs={task.params.get('epochs', 'N/A')}"
            )

            # 根据模型类型执行不同的训练
            result = self._execute_training(task)
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now().isoformat()
            task.progress = 100.0

            logger.info(
                f"[AsyncTraining] 训练完成: {task.task_id} "
                f"({task.model_type} {task.stock_code})"
            )

            # 触发完成回调
            self._notify_progress(task.task_id, task)

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.now().isoformat()
            logger.error(f"[AsyncTraining] 训练失败: {task.task_id} — {e}")
            logger.debug(traceback.format_exc())

        finally:
            with self._lock:
                self._running_count -= 1

    def _execute_training(self, task: TrainingTask) -> Dict:
        """执行具体训练逻辑"""
        model_type = task.model_type
        stock_code = task.stock_code
        params = task.params

        if model_type == 'drl':
            return self._train_drl(stock_code, params, task)
        elif model_type == 'patchtst':
            return self._train_patchtst(stock_code, params)
        elif model_type == 'moirai':
            return self._train_moirai(stock_code, params)
        else:
            raise ValueError(f"未知的模型类型: {model_type}")

    def _train_drl(self, stock_code: str, params: Dict,
                   task: Optional[TrainingTask] = None) -> Dict:
        """训练 DRL 模型"""
        # 动态导入避免 torch 依赖问题
        try:
            from modules.models.drl_agent import DRLTradingAgent
            from modules.kline_data_fetcher import KlineDataFetcher
            from modules.data_fetcher import StockDataFetcher
            from modules.technical_indicators import calculate_rsi, calculate_macd
        except ImportError as e:
            raise ImportError(f"训练依赖未安装: {e}")

        epochs = params.get('epochs', 100)
        days = params.get('days', 365)
        batch_size = params.get('batch_size', 64)

        # 获取 K 线数据
        count = min(days * 1.5, 500)
        fetcher = KlineDataFetcher()
        klines = fetcher.fetch_kline(stock_code, period='daily', count=int(count))

        if not klines or len(klines) < 50:
            raise ValueError(f"数据不足: {len(klines) if klines else 0} 条")

        # 进度回调
        if task and task.task_id in self._progress_callbacks:
            self._progress_callbacks[task.task_id](task.task_id, 10, "获取数据完成")

        # 提取特征 (2026-09-08: closes 必须是 np.array — 下方 sma5/sma20 用
        # .mean()/.std(), list 切片会抛 AttributeError 使任务静默失败)
        closes = np.array([k['close'] for k in klines])
        features = []
        labels = []
        for i in range(29, len(klines) - 5):
            rsi = calculate_rsi(closes[:i+1], 14) if i >= 13 else 50.0
            macd_r = calculate_macd(closes[:i+1]) if i >= 25 else {'macd': 0.0}
            sma5 = closes[max(0,i-4):i+1].mean() / (closes[i] + 1e-8) - 1
            sma20 = closes[max(0,i-19):i+1].mean() / (closes[i] + 1e-8) - 1
            vol_ratio = klines[i].get('volume', 0) / (
                sum(klines[j].get('volume', 0) for j in range(max(0,i-4), i+1)) / 5 + 1e-8
            )
            ret_5 = closes[i] / closes[max(0,i-4)] - 1 if i >= 4 else 0
            vol_20 = closes[max(0,i-19):i+1].std() / (closes[max(0,i-19):i+1].mean() + 1e-8)

            state = [
                rsi / 100, macd_r.get('macd', 0), 0, 0, 0, 0, 0,
                sma5, sma20, min(vol_ratio / 5, 1),
                min(ret_5 * 10, 1), min(vol_20 * 10, 1),
                0.5, 0.3, 0.2, min(vol_20 * 10, 1),
                0.5, 0.5, 0.5, min(ret_5 * 2, 1),
                min(vol_ratio * 0.5, 1), 0, 0, 0,
            ]
            features.append(state)

            if i + 5 < len(closes):
                labels.append(float(np.tanh((closes[i+5] / closes[i] - 1) * 5)))
            else:
                labels.append(0.0)

        features = np.array(features)
        labels = np.array(labels)

        # 训练 — 2026-09-08: 复用 DRLTradingAgent.train (轨迹窗口采样 + PnL 对齐 reward)。
        # 原内联循环 = 随机散点 (s,s') 采样 + 动作无关 reward (tanh(label*0.5)),
        # 与 train_sota_models 的同源断链 (09-03 教训: 修链≠链绿, 同类断链要横扫)
        agent = DRLTradingAgent()
        if task and task.cancel_flag:
            raise KeyboardInterrupt("训练已取消")

        env_data = {
            'features': features,
            'labels': labels,
            'seq_len': 20,
            'n_features': features.shape[1] if features.ndim == 2 else features.shape[2],
        }
        agent.train(env_data, epochs=epochs, batch_size=batch_size)

        # 更新进度
        if task and task.task_id in self._progress_callbacks:
            self._progress_callbacks[task.task_id](task.task_id, 90, "DRL 训练完成")

        # 保存模型 (train() 已置 _trained=True, save 序列化该标志)
        agent.save()

        return {
            'status': 'success',
            'epochs': epochs,
            'n_samples': len(labels),
            'model_path': os.path.join(
                os.path.dirname(__file__), 'models', 'dl_models', 'drl_agent.npy'
            ),
        }

    def _train_patchtst(self, stock_code: str, params: Dict) -> Dict:
        """训练 PatchTST 模型 (fallback: 记录任务)

        ⚠️ 进程隔离警告: 如果未来实现此方法，必须使用 subprocess 隔离训练。
        PyTorch C 扩展在线程中可能引发 SIGSEGV。
        正确模式:
            import subprocess
            script = os.path.join(os.path.dirname(__file__), '_train_patchtst_worker.py')
            subprocess.run([sys.executable, script, stock_code, json.dumps(params)])
        """
        return {
            'status': 'skipped',
            'reason': 'PyTorch not available',
            'note': 'Install PyTorch to enable PatchTST training',
        }

    def _train_moirai(self, stock_code: str, params: Dict) -> Dict:
        """训练 Moirai 模型 (fallback: 记录任务)

        ⚠️ 进程隔离警告: 同 _train_patchtst，必须使用 subprocess 隔离。
        """
        return {
            'status': 'skipped',
            'reason': 'Moirai training not yet implemented',
        }

    def submit(self, model_type: str, stock_code: str,
               params: Optional[Dict] = None) -> str:
        """
        提交训练任务

        Args:
            model_type: 模型类型 (drl/patchtst/moirai)
            stock_code: 股票代码
            params: 训练参数

        Returns:
            task_id
        """
        if self._task_queue.full():
            raise RuntimeError(f"训练队列已满 ({self.queue_size})")

        task_id = str(uuid.uuid4())[:8]
        task = TrainingTask(
            task_id=task_id,
            model_type=model_type,
            stock_code=stock_code,
            params=params or {},
            status=TaskStatus.PENDING,
        )

        self._tasks[task_id] = task
        self._task_queue.put(task)

        logger.info(
            f"[AsyncTraining] 提交任务: {task_id} "
            f"({model_type} {stock_code})"
        )
        return task_id

    def get_status(self, task_id: str) -> Optional[Dict]:
        """获取任务状态"""
        task = self._tasks.get(task_id)
        if task is None:
            return None
        return task.to_dict()

    def list_tasks(self, status: Optional[TaskStatus] = None) -> List[Dict]:
        """列出任务"""
        tasks = list(self._tasks.values())
        if status:
            tasks = [t for t in tasks if t.status == status]
        return sorted([t.to_dict() for t in tasks], key=lambda x: x.get('started_at', ''), reverse=True)

    def cancel(self, task_id: str) -> bool:
        """取消任务"""
        task = self._tasks.get(task_id)
        if task and task.status in (TaskStatus.PENDING, TaskStatus.RUNNING):
            task.cancel_flag = True
            task.status = TaskStatus.CANCELLED
            logger.info(f"[AsyncTraining] 取消任务: {task_id}")
            return True
        return False

    def register_progress_callback(self, task_id: str, callback: Callable):
        """注册进度回调"""
        self._progress_callbacks[task_id] = callback

    def _notify_progress(self, task_id: str, task: TrainingTask):
        """通知进度更新"""
        callback = self._progress_callbacks.get(task_id)
        if callback:
            try:
                callback(task_id, task.progress, task.status.value)
            except Exception as e:
                logger.warning(f"[AsyncTraining] 回调通知失败: {e}")

    def stop(self):
        """停止调度器"""
        self._running = False
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5)
        logger.info("[AsyncTraining] 调度器已停止")


# 全局单例
_pipeline: Optional[AsyncTrainingPipeline] = None


def get_training_pipeline() -> AsyncTrainingPipeline:
    """获取异步训练管线全局实例"""
    global _pipeline
    if _pipeline is None:
        _pipeline = AsyncTrainingPipeline()
    return _pipeline


def reset_training_pipeline():
    """重置全局实例"""
    global _pipeline
    _pipeline = None
