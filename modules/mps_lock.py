"""
mps_lock.py — Metal/MPS 计算串行化锁 (2026-09-02)

背景: 多 worker 线程并发使用 MPS/Metal 会触发
`A command encoder is already encoding to this command buffer` 断言,
整个进程 abort (2026-07-31 SIGSEGV 与 2026-09-02 两次共识链崩溃同根因)。

规则: 所有 MPS 重计算 (后台训练 / 共识决策链) 必须经本锁串行化。
- 训练侧: blocking acquire (训练是兜底任务, 可以等)
- 决策侧: acquire(timeout=45), 拿不到 → 诚实返回"训练中/排队"占位结果,
  页面不卡死, 进程不崩溃。
"""

import threading

# 全局 MPS 计算锁 (RLock: 允许同链嵌套持有)
MPS_LOCK = threading.RLock()


def mps_busy() -> bool:
    """当前是否有其他 MPS 重计算在跑 (非阻塞探测, 用于占位提示)"""
    if MPS_LOCK.acquire(blocking=False):
        MPS_LOCK.release()
        return False
    return True
