#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
DEPRECATED — 已弃用，请使用 event_backtester.py + walkforward_backtester.py

此文件仅为向后兼容保留。
所有新的代码应使用:
  - 批量回测: from modules.walkforward_backtester import WalkForwardBacktester
  - 事件驱动: from modules.event_backtester import EventDrivenBacktester

保留时间: 2026-07-01 之后 3 个月
"""

import warnings

warnings.warn(
    "[DEPRECATED] modules.backtester 已弃用，请使用 modules.event_backtester 或 modules.walkforward_backtester",
    DeprecationWarning,
    stacklevel=2,
)

from modules.event_backtester import EventDrivenBacktester
from modules.walkforward_backtester import WalkForwardBacktester

__all__ = ['EventDrivenBacktester', 'WalkForwardBacktester']
