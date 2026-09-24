# -*- coding: utf-8 -*-
"""log_context — 链式追踪上下文 (2026-09-20 整合②, PanWatch log_context 式适配)

PanWatch 形态: ContextVar 持 trace_id → JSON 日志行注入 → 一条链 (取数→LLM→
推送→送达) 的所有日志共享同一 trace, 一 grep 串全链 = 链尾可观测。

5002 适配 (Werkzeug threading 非 asyncio):
  - threading.Thread 不自动继承 ContextVar → 新线程默认空, 无跨线程泄漏;
    后台线程链 (晨报/复盘) 在链头显式 set, 不依赖请求 context。
  - StructuredFormatter 仅在有活跃 trace 时注入 "trace" 字段, 非链式日志零污染。

链形态: tr-{CST日期}-{trigger}-{target}-{8hex} — 人眼可 grep, 可进 health。
"""
import contextvars
import hashlib
import os
from datetime import datetime, timezone, timedelta

_CST = timezone(timedelta(hours=8))

_trace: contextvars.ContextVar = contextvars.ContextVar('log_trace_id', default='')


def get_trace() -> str:
    """当前线程/上下文的活跃 trace (无 = '')。"""
    return _trace.get()


def trace_of():
    """观测面入口 (health/链尾核对): 无活跃 trace 返 None。"""
    return _trace.get() or None


def set_trace(trace_id: str):
    """绑定链尾 (链头调用一次, 线程内后续日志自动携带)。"""
    _trace.set(trace_id)


def clear_trace():
    _trace.set('')


def gen_trace_id(trigger: str = 'auto', target: str = '') -> str:
    """生成可 grep 的链 ID: tr-YYYY-MM-DD-{trigger}-{target}-{8hex}。"""
    date = datetime.now(_CST).strftime('%Y-%m-%d')
    tgt = f'-{target}' if target else ''
    return f"tr-{date}-{trigger}{tgt}-{hashlib.sha1(os.urandom(8)).hexdigest()[:8]}"


def with_trace(fn):
    """跨线程链信封: 包装线程池 submit 的函数, 子线程执行时绑回提交点的 trace。

    ContextVar 在 threading.Thread 不自动继承 → submit 裸函数进 worker 后 get
    默认空 = 日志断链 (intel 3-worker/kline/fetcher 超时包共 4 处 submit 段)。
    用法: ex.submit(with_trace(job)) — worker 内所有 logger 行随之带上请求 trace。
    """
    tid = _trace.get()

    def wrapper(*a, **k):
        old = _trace.get()
        _trace.set(tid)
        try:
            return fn(*a, **k)
        finally:
            _trace.set(old)

    return wrapper
