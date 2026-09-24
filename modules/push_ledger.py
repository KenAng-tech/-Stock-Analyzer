# -*- coding: utf-8 -*-
"""push_ledger — Telegram 推送台账 (2026-09-20 整合①, PanWatch「发送成功才标记」式)

链背景:
  5002 晨报链 _push_to_telegram → 写 ~/.claude/channels/telegram/reply_queue/*.json
  → telegram daemon (bun server.ts, 0.0.6 插件) 5s 轮询消费:
      sendMessage 成功 → rmSync 删文件 (链尾 = Telegram 已接收)
      发送失败 → 文件滞留重试 (永久 retry, 无放弃)
  旧形态两缺口: ① 同内容同日多触发 (08:00 auto / 手动 POST / TG「报」) 重复推 ② 写队列
  即算"已推送" — daemon 断或 token 坏时无限滞留 = 链尾失明 (09-18 案模式)。

本模块 = 生产侧 (producer) 台账, 不碰 daemon (其在位协议即"发送成功才标记"真形):
  check(key)     推前门: sweep 扫账 (滞留超时降级) + 判定放行/拦截
  register(key)  记一次入队尝试 (auto/manual 都记, 失败可重试 ≤max_retries 次)
  stats()        health 观测面 (pending/confirmed/failed/blocked)

状态机 (per key):
  fresh     无账本 → 放行首推
  pending   队列文件在 + 未超时 → 拦截 (上一推在途, daemon 还在重试/排队)
  confirmed 文件被 daemon 删除 = Telegram 已接收 → 拦截 auto 重复 (force/manual 放行)
  failed    滞留>ttl, 尝试次数未耗尽 → 放行补推 (降级非断链)
  blocked   滞留>ttl, 尝试次数耗尽 → 拦截 + warning (推送链断, 升级人工)

台账文件损坏/不可写 → 跳过台账照常推 (记账失败 ≠ 推送链失败, 同 search_ledger 形)。
线程: 单锁 (generate 链本身持 _lock, 此处双保险)。时钟可注入 (测试免 sleep)。
"""
import json
import os
import tempfile
import threading
import time

from modules.logger import logger

DEFAULT_QUEUE_DIR = os.path.expanduser('~/.claude/channels/telegram/reply_queue')


class PushLedger:
    """Telegram 推送去重台账: 去重门 + 送达确认 + 断链可见。"""

    def __init__(self, ledger_path=None, queue_dir=None, clock=time.time,
                 ttl_pending=1800, max_retries=3):
        self._queue_dir = queue_dir or DEFAULT_QUEUE_DIR
        self._path = ledger_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'push_ledger.json')
        self._clock = clock
        self._ttl = ttl_pending
        self._max = max_retries
        self._lock = threading.Lock()
        self._load()

    # ── 台账读写 ──────────────────────────────────────────

    def _load(self):
        try:
            with open(self._path, 'r', encoding='utf-8') as f:
                d = json.load(f)
            self._ledger = d if isinstance(d, dict) else {}
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            self._ledger = {}

    def _persist(self):
        try:
            os.makedirs(os.path.dirname(self._path), exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=os.path.dirname(self._path),
                                       prefix='.push_ledger.', suffix='.tmp')
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump(self._ledger, f, ensure_ascii=False, indent=1)
            os.replace(tmp, self._path)
        except OSError as e:
            logger.warning(f"[PushLedger] 台账落盘失败 (不影响推送): {e}")

    # ── 状态机 ────────────────────────────────────────────

    def _file_alive(self, fname, now):
        """队列文件还在且比记账时刻新 = daemon 尚未消费 (含 daemon 重启后重扫)。"""
        p = os.path.join(self._queue_dir, fname)
        try:
            return os.path.getmtime(p) >= (now - 86400 * 8)
        except OSError:
            return False

    def check(self, key, force=False, now=None):
        """推前门: 返回 (allowed, reason, state)。force = manual 绕过去重。"""
        now = self._clock() if now is None else now
        with self._lock:
            self._sweep(now)
            ent = self._ledger.get(key)
            if ent is None:
                return True, '新内容', 'fresh'
            st = ent['state']
            if force:
                return True, f'manual 绕过 ({st})', st
            if st == 'pending':
                return False, f'队列在途 (daemon 5s 重试中, {now - ent["ts"]:.0f}s)', 'pending'
            if st == 'confirmed':
                return False, '已送达 (当日同内容去重)', 'confirmed'
            if st == 'failed':
                return True, (f'上一推滞留未送达 (尝试{ent["attempts"]}/{self._max}), '
                              f'补推放行'), 'failed'
            if st == 'blocked':
                logger.warning(f"[PushLedger] 推送链疑似断: key={key} 连续{ent['attempts']}次未送达, 需人工查 daemon/token")
                return False, f'链断 (连续{ent["attempts"]}次未送达, 升级人工)', 'blocked'
            return True, f'未知态 {st}', st

    def register(self, key, fname, now=None):
        """记一次入队尝试 (写队列文件成功后调用)。同一 key 的滞留失败后重推 = 尝试+1。"""
        now = self._clock() if now is None else now
        with self._lock:
            old = self._ledger.get(key) or {}
            attempts = old.get('attempts', 0) + (1 if old.get('state') == 'failed' else 0)
            self._ledger[key] = {'state': 'pending', 'file': fname,
                                 'ts': now, 'attempts': max(1, attempts)}
            if len(self._ledger) > 500:  # 防胀: 裁最老
                for k in sorted(self._ledger, key=lambda k: self._ledger[k]['ts'])[:100]:
                    del self._ledger[k]
            self._persist()

    def _sweep(self, now):
        """扫账: pending → confirmed (文件被 daemon 删 = 送达) / failed (滞留) / blocked (耗尽)。"""
        changed = False
        for key, ent in list(self._ledger.items()):
            if ent['state'] != 'pending':
                continue
            if not self._file_alive(ent['file'], now):
                ent['state'] = 'confirmed'
                changed = True
            elif now - ent['ts'] > self._ttl:
                if ent['attempts'] >= self._max:
                    ent['state'] = 'blocked'
                else:
                    ent['state'] = 'failed'
                changed = True
        if changed:
            self._persist()

    # ── 观测面 (health) ──────────────────────────────────

    def stats(self, now=None):
        now = self._clock() if now is None else now
        with self._lock:
            self._sweep(now)
            out = {'ttl_pending': self._ttl, 'max_retries': self._max, 'total': len(self._ledger)}
            by = {}
            for ent in self._ledger.values():
                by[ent['state']] = by.get(ent['state'], 0) + 1
            for k in ('pending', 'confirmed', 'failed', 'blocked'):
                out[k] = by.get(k, 0)
            out['by_state'] = by
            return out
