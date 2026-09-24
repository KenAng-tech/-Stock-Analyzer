#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""outbound_guard 单元测试 (2026-09-20 TDD)

5002 外发链三件套 (参照 PanWatch 模式):
1. HostThrottle — 按 host 串行节流 (批量链触发上游限流)
2. CooldownTable — per-key 失败冷却 (一标的故障不烧全链; 盘中 60s/其余 900s)
3. flight_fetch — 同 key 并发合并 (singleflight, 重复上游请求=0)

时间全注入 (FakeClock/固定时刻), 不依赖真实 sleep 时长。
"""
import calendar
import threading
import time

from utils.outbound_guard import CooldownTable, HostThrottle, flight_fetch


class FakeClock:
    """假时钟: sleep 快进, 测节流串行化不真等."""
    def __init__(self):
        self.t = 1000.0
        self.lock = threading.Lock()

    def now(self):
        with self.lock:
            return self.t

    def sleep(self, s):
        with self.lock:
            self.t += s


# ── HostThrottle ─────────────────────────────────────────────────────

def test_host_throttle_serializes_same_host():
    """同 host 连续 3 次 acquire, 假时钟至少推进 2×0.15s (两次节流间隔)."""
    clk = FakeClock()
    th = HostThrottle(clock=clk.now, sleep=clk.sleep, default_min_interval=0.15)
    th.acquire("sina", host_key="money.finance.sina.com.cn")
    th.acquire("sina", host_key="money.finance.sina.com.cn")
    th.acquire("sina", host_key="money.finance.sina.com.cn")
    assert clk.t >= 1000.30 - 1e-9, f"节流未串行化: t={clk.t}"


def test_host_throttle_different_hosts_independent():
    """不同 host 互不等待 (并行取数不互相拖慢)."""
    clk = FakeClock()
    th = HostThrottle(clock=clk.now, sleep=clk.sleep)
    t_before = clk.t
    th.acquire("sina")
    th.acquire("eastmoney")
    th.acquire("tencent")
    assert clk.t - t_before < 0.01, f"跨 host 被误串行: +{clk.t - t_before}"


def test_host_throttle_per_host_interval():
    """eastmoney 默认 0.2s, 其他 0.15s (per-host 间隔表)."""
    clk = FakeClock()
    th = HostThrottle(clock=clk.now, sleep=clk.sleep)
    th.acquire("x", host_key="push2delay.eastmoney.com")
    th.acquire("y", host_key="push2delay.eastmoney.com")  # 第二次必须被节流 0.2s
    assert clk.t >= 1000.2 - 1e-9, f"eastmoney 间隔不足 0.2: +{clk.t - 1000}"


# ── CooldownTable ────────────────────────────────────────────────────

def test_cooldown_per_key_isolation():
    """一 key 冷却不牵连其他 key; 冷却内重复 open 重新计时."""
    clk = FakeClock()
    cd = CooldownTable(clock=clk.now)
    cd.open("akshare:sz300620", ttl=60)
    assert cd.is_cooling("akshare:sz300620") is True
    assert cd.is_cooling("akshare:sh600519") is False
    clk.t += 30
    cd.open("akshare:sz300620", ttl=60)   # 重新计时 → 再 60s
    clk.t += 30                            # 第二个 60s 窗口未过
    assert cd.is_cooling("akshare:sz300620") is True


def test_cooldown_window_expiry():
    """窗口过后自动放行 (半开探测语义)."""
    clk = FakeClock()
    cd = CooldownTable(clock=clk.now)
    cd.open("sina:sh600519", ttl=60)
    clk.t += 61
    assert cd.is_cooling("sina:sh600519") is False


# 2026-09-21 周一 CST 时刻 → 注入 clock 用 UTC 秒 (localtime 转 CST):
_TRADING_TS = calendar.timegm((2026, 9, 21, 2, 30, 0, 0, 0, -1))    # CST 10:30 盘中
_CLOSED_TS = calendar.timegm((2026, 9, 21, 15, 30, 0, 0, 0, -1))    # CST 23:30 收盘


def test_cooldown_default_ttl_trading_hours():
    """默认冷却: 盘中 60s (CST 10:30 注入)."""
    cd = CooldownTable(clock=lambda: float(_TRADING_TS))
    cd.open("sina:x")                                   # ttl=None → 默认
    assert cd.is_cooling("sina:x", now=_TRADING_TS + 59) is True
    assert cd.is_cooling("sina:x", now=_TRADING_TS + 61) is False


def test_cooldown_default_ttl_closed_hours():
    """默认冷却: 收盘 900s (CST 23:30 注入, 重试无意义→长冷却)."""
    cd = CooldownTable(clock=lambda: float(_CLOSED_TS))
    cd.open("sina:y")
    assert cd.is_cooling("sina:y", now=_CLOSED_TS + 899) is True
    assert cd.is_cooling("sina:y", now=_CLOSED_TS + 901) is False


def test_cooldown_clear_on_success():
    """成功清冷却 (下一请求立刻可再试该源)."""
    cd = CooldownTable()
    cd.open("sina:z")
    cd.clear("sina:z")
    assert cd.is_cooling("sina:z") is False


# ── flight_fetch (singleflight) ──────────────────────────────────────

def test_flight_merges_concurrent_same_key():
    """同 key 并发只打一次上游: leader 取数, followers 合并复用."""
    calls = []
    out = []

    def upstream():
        calls.append(1)
        time.sleep(0.3)
        return ["bar", "bar"]

    def worker():
        out.append(flight_fetch("kline_sz300620_daily_250", upstream, timeout=5))

    ts = [threading.Thread(target=worker) for _ in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert len(calls) == 1, f"上游被打 {len(calls)} 次"
    assert all(r == ["bar", "bar"] for r in out), f"8 线程未共享: {out}"


def test_flight_different_keys_parallel():
    """不同 key 并发各自执行, 互不阻塞."""
    def slow_upstream():
        time.sleep(0.2)
        return "v"

    outs = {}

    def worker(key):
        outs[key] = flight_fetch(key, slow_upstream, timeout=5)

    ts = [threading.Thread(target=worker, args=(k,)) for k in ("k1", "k2", "k3", "k4")]
    t0 = time.time()
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert outs == {"k1": "v", "k2": "v", "k3": "v", "k4": "v"}
    assert time.time() - t0 < 0.45, f"不同 key 被串行化"


def test_flight_error_not_merged_poisonously():
    """leader 抛异常 → 槽位释放, 下批自取 (失败不阻塞链尾)."""
    state = {"n": 0}

    def flaky():
        state["n"] += 1
        if state["n"] == 1:
            raise RuntimeError("RemoteDisconnected")
        return ["ok"]

    r1 = flight_fetch("k", flaky, timeout=1)
    r2 = flight_fetch("k", flaky, timeout=1)
    assert r1 is None and r2 == ["ok"]
    assert state["n"] == 2
