#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
模型推理队列
控制并发推理请求数量
"""

import asyncio
import time
import threading
from asyncio import Semaphore, Queue, QueueFull
from dataclasses import dataclass
from typing import Optional, Callable, Any
from modules.logger import logger


@dataclass
class InferenceRequest:
    """推理请求"""
    id: str
    model: str
    data: Any
    callback: Optional[Callable] = None
    created_at: float = 0.0


@dataclass
class InferenceResult:
    """推理结果"""
    request_id: str
    result: Any
    error: Optional[Exception] = None
    inference_time: float = 0.0


class ModelInferenceQueue:
    """
    模型推理队列

    控制并发推理请求数量
    """

    def __init__(self, max_concurrent: int = 4, queue_size: int = 1000):
        self.semaphore = Semaphore(max_concurrent)
        self.queue = Queue(maxsize=queue_size)
        self._workers = []
        self._is_running = False
        self._max_concurrent = max_concurrent

    async def start(self):
        """启动推理队列"""
        self._is_running = True
        for i in range(self._max_concurrent):
            worker = asyncio.create_task(self._worker())
            self._workers.append(worker)
        logger.info(f"[InferenceQueue] 推理队列已启动，{self._max_concurrent} 个worker")

    async def stop(self):
        """停止推理队列"""
        self._is_running = False
        for worker in self._workers:
            worker.cancel()
        logger.info("[InferenceQueue] 推理队列已停止")

    async def submit(self, model: str, data: Any, callback: Callable = None) -> str:
        """提交推理请求"""
        import uuid
        request_id = str(uuid.uuid4())

        request = InferenceRequest(
            id=request_id,
            model=model,
            data=data,
            callback=callback,
            created_at=time.time()
        )

        await self.queue.put(request)
        logger.debug(f"[InferenceQueue] 请求已提交: {request_id}")
        return request_id

    async def _worker(self):
        """Worker协程"""
        while self._is_running:
            try:
                request = await self.queue.get()
                async with self.semaphore:
                    start_time = time.time()
                    try:
                        # 执行推理
                        result = await self._execute_inference(request)

                        # 回调
                        if request.callback:
                            await request.callback(result)
                    except Exception as e:
                        logger.error(f"[InferenceQueue] 推理失败: {e}")
                        if request.callback:
                            await request.callback(InferenceResult(
                                request_id=request.id,
                                error=e,
                                inference_time=time.time() - start_time
                            ))
                    finally:
                        self.queue.task_done()
            except asyncio.CancelledError:
                break

    async def _execute_inference(self, request: InferenceRequest) -> InferenceResult:
        """执行模型推理"""
        start_time = time.time()

        # TODO: 实现具体的模型推理逻辑
        result = await request.model.predict(request.data)

        return InferenceResult(
            request_id=request.id,
            result=result,
            inference_time=time.time() - start_time
        )

    async def get_queue_size(self) -> int:
        """获取队列大小"""
        return self.queue.qsize()


# 全局推理队列
inference_queue = ModelInferenceQueue()