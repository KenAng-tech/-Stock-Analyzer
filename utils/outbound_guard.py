#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""outbound_guard — 外发取数链保护 (2026-09-20, 参照 PanWatch market_http/kline_collector 模式)

三件套 (一个模块, 三个 seam):
- HostThrottle: 按 host 串行节流, 治「批量链并发突发触发上游限流」
  (腾讯 0.15s / 东财 0.2s / Scrapling 抓取 1.0s, 含 jitter 防同频共振)
- CooldownTable: per-key (source:code) 失败冷却, 治「一标的接口故障烧全 provider 链」
  (盘中 60s / 非盘中 900s, 半开自动放行)
- flight_fetch: 同 key 并发合并 (singleflight), 治「并发同标的重复打上游」
  (leader 取数, followers 等待复用; leader 超时/失败 → followers 自取, 永不阻塞)

设计原则 (2026-09-20 链修原则):
- 保护链永不阻断取数: 冷却返回「跳过」, 合并超时返回 None, 调用方自行降级
- 时间全可注入 (clock/sleep), 并发全线程安全 (告警 30s 链 + 晨报 TP 执行链 + Telegram 触发并发)
"""

import random
import threading
import time
from typing import Any, Callable, Dict, Optional

# host 最小间隔 (秒); 未列 host 用 default_min_interval
_DEFAULT_MIN_INTERVAL = {
    "push2delay.eastmoney.com": 0.2,
    "push2.eastmoney.com": 0.2,
    "datacenter-web.eastmoney.com": 0.2,
    "qt.gtimg.cn": 0.15,
    "money.finance.sina.com.cn": 0.15,
    "scrapling": 1.0,   # 浏览器抓取链: 反爬+限频双高, 间隔收紧
}


# ── 交易时段 (CST, 节假日简化 = 周末不盘中, 假日盘中按盘中处理=短冷却) ──

def is_trading_time(now: Optional[float] = None) -> bool:
    """A股交易时段: 工作日 9:30-11:30 / 13:00-15:00 (本地 CST 时区语义)."""
    lt = time.localtime(now if now is not None else time.time())
    if lt.tm_wday >= 5:
        return False
    h = lt.tm_hour + lt.tm_min / 60.0
    return 9.5 <= h < 11.5 or 13.0 <= h < 15.0


# ── HostThrottle ────────────────────────────────────────────────────

class HostThrottle:
    """进程级按 host 串行节流 (线程安全, 阻塞式排队不抛弃)."""

    def __init__(self, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep,
                 default_min_interval: float = 0.15):
        self._clock = clock
        self._sleep = sleep
        self._default = default_min_interval
        self._last: Dict[str, float] = {}
        self._lock = threading.Lock()

    def min_interval(self, host_key: str) -> float:
        return _DEFAULT_MIN_INTERVAL.get(host_key, self._default)

    def acquire(self, host: str, host_key: Optional[str] = None,
                jitter: bool = True) -> float:
        """阻塞直到允许对 host 发请求; 返回实际等待秒数 (≤2s 防积压雪崩).

        host=逻辑名 (日志/默认表命中), host_key=节流键 (默认=host)。
        """
        key = host_key or host
        with self._lock:
            now = self._clock()
            last = self._last.get(key)
            mi = self.min_interval(key)
            wait = 0.0
            if last is not None:
                gap = now - last
                if gap < mi:
                    wait = min(mi - gap, 2.0)   # 积压上限: 队列永不雪崩
            if jitter and wait > 0:
                wait += random.uniform(0, 0.2 * mi)
            self._last[key] = now + wait
        if wait > 0:
            self._sleep(wait)
        return wait


# ── CooldownTable ───────────────────────────────────────────────────

class CooldownTable:
    """per-key 失败冷却表 (key 例: 'akshare:sz300620').

    冷却窗口 = 该源对该标的的冷却: 故障期内 (source,code) 取数直接跳过该源。
    半开语义: 窗口过后自动放行 (下次请求允许探测)。
    """

    def __init__(self, clock: Callable[[], float] = time.time):
        self._clock = clock
        self._until: Dict[str, float] = {}
        self._lock = threading.Lock()

    def _default_ttl(self, now: float) -> float:
        return 60.0 if is_trading_time(now) else 900.0

    def is_cooling(self, key: str, now: Optional[float] = None) -> bool:
        with self._lock:
            until = self._until.get(key)
            if until is None:
                return False
            now = self._clock() if now is None else now
            return now < until

    def open(self, key: str, ttl: Optional[float] = None,
             now: Optional[float] = None) -> None:
        """打开/延长冷却 (重新计时)."""
        with self._lock:
            now = self._clock() if now is None else now
            ttl = self._default_ttl(now) if ttl is None else ttl
            self._until[key] = now + ttl

    def clear(self, key: str) -> None:
        with self._lock:
            self._until.pop(key, None)


# ── flight_fetch (singleflight) ─────────────────────────────────────

class _Flight:
    __slots__ = ("done", "value", "event")

    def __init__(self):
        self.done = False
        self.value = None
        self.event = threading.Event()


_flights: Dict[str, _Flight] = {}
_registry_lock = threading.Lock()


def flight_fetch(key: str, fn: Callable[[], Any], timeout: float = 25.0) -> Any:
    """同 key 并发合并: leader 执行 fn, followers 等待复用其结果.

    - leader 成功 → followers 直接拿同一结果 (不重复打上游)
    - leader 抛异常 → 槽位释放, 等待者超时后自取 (失败不阻塞链尾)
    - followers 等待超时 (默认 25s, 对齐既有 timeout 预算) → 自取, 防链尾永久阻塞
    """
    with _registry_lock:
        flight = _flights.get(key)
        is_leader = flight is None
        if is_leader:
            flight = _Flight()
            _flights[key] = flight

    if not is_leader:
        flight.event.wait(timeout)
        return flight.value if flight.done else None

    try:
        result = fn()
    except Exception:
        with _registry_lock:
            _flights.pop(key, None)
        flight.event.set()   # 唤醒等待者 (未完成态 → 等待者超时后自取)
        return None

    flight.value = result
    flight.done = True
    flight.event.set()
    with _registry_lock:
        _flights.pop(key, None)
    return result


# ── 全局单例 (KlineDataFetcher 多实例共享同一节流/冷却面) ──────────

host_throttle = HostThrottle()
cooldowns = CooldownTable()
