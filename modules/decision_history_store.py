"""
decision_history_store.py — 共识决策历史 SQLite 持久化 (2026-09-03, Task #14)

背景: 决策历史原先只在内存 list (len>1000 截 500) — 重启即全失,
IC 权重闭环的 resolution 数据源也只能看到最近 ≤500 条。

设计 (参考 sqlite_cache 教训: 禁共享连接, 每操作独立连接 + 文件锁):
  - WAL + busy_timeout 2500ms: 决策 worker (串行) 与同步链并发写安全
  - 每操作新建 sqlite3 连接 (无 self._conn 共享 → 无跨调用连接失效)
  - 上限 50000 行: append 事务内 prune 旧行 (24/7 自动跑 ~百行/日, 够用数月)
  - 全路径吞异常 + logger.debug: 存储永不拖垮决策主链 (决策优先于审计)
"""

import json
import os
import sqlite3
import threading
from typing import Dict, List

from modules.logger import logger

_DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
_DEFAULT_DB = os.path.join(_DB_DIR, 'decision_history.db')
_MAX_ROWS = 50000


class DecisionHistoryStore:
    """决策历史 SQLite 存储 (线程安全: 每操作独立连接 + 文件级锁)"""

    def __init__(self, db_path: str = None):
        self._db_path = db_path or _DEFAULT_DB
        self._lock = threading.RLock()
        try:
            os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
            conn = self._conn()
            try:
                conn.execute('PRAGMA journal_mode=WAL')
                conn.execute('''CREATE TABLE IF NOT EXISTS decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    code TEXT NOT NULL,
                    consensus TEXT,
                    confidence REAL,
                    agent_votes TEXT
                )''')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_dh_ts ON decisions(ts)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_dh_code ON decisions(code)')
                # 09-22 P0-2: Brier 池持久 (决策链 B 校准报告原料, 重启不清零)
                # UNIQUE(code, ts) = 同日重跑 replay 防重喂 (decision_brier_pool 形)
                conn.execute('''CREATE TABLE IF NOT EXISTS decision_briers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    code TEXT NOT NULL,
                    p REAL,
                    outcome INTEGER,
                    UNIQUE(code, ts)
                )''')
                conn.commit()
            finally:
                conn.close()
            logger.info(f"[DecisionHistory] SQLite 就绪: {self._db_path}")
        except Exception as e:
            logger.warning(f"[DecisionHistory] 初始化失败 (持久化降级为内存): {e}")
            self._db_path = None

    def _conn(self) -> sqlite3.Connection:
        """独立连接 (无共享): 线程安全靠 SQLite 文件锁 + busy_timeout"""
        conn = sqlite3.connect(self._db_path, timeout=2.5)
        conn.execute('PRAGMA busy_timeout=2500')
        return conn

    # ── 写 ──

    def append(self, result) -> bool:
        """落库一条 ConsensusResult。任何异常吞掉 (审计失败不拖决策主链)。

        Args:
            result: ConsensusResult (stock_code/timestamp/consensus/confidence/agent_votes)
        """
        if not self._db_path:
            return False
        try:
            with self._lock:
                conn = self._conn()
                try:
                    votes_json = json.dumps(result.agent_votes or [],
                                            ensure_ascii=False, default=str)
                    conn.execute(
                        'INSERT INTO decisions(ts, code, consensus, confidence, agent_votes)'
                        ' VALUES (?,?,?,?,?)',
                        (result.timestamp or '', result.stock_code,
                         result.consensus, float(result.confidence or 0), votes_json))
                    conn.execute(
                        'DELETE FROM decisions WHERE id <= '
                        '(SELECT MAX(id) FROM decisions) - ?', (_MAX_ROWS,))
                    conn.commit()
                    return True
                finally:
                    conn.close()
        except Exception as e:
            logger.debug(f"[DecisionHistory] append 失败 (已吞): {e}")
            return False

    # ── 读 ──

    def load_recent(self, limit: int = 1000) -> List[Dict]:
        """最近 limit 条 (按 id 倒序 → 调用方按需反转)。失败返回 []。

        Returns:
            [{timestamp, stock_code, consensus, confidence, agent_votes: List[Dict]}]
        """
        if not self._db_path:
            return []
        try:
            conn = self._conn()
            try:
                rows = conn.execute(
                    'SELECT ts, code, consensus, confidence, agent_votes '
                    'FROM decisions ORDER BY id DESC LIMIT ?', (limit,)).fetchall()
            finally:
                conn.close()
            out = []
            for ts, code, cons, conf, votes_json in rows:
                try:
                    votes = json.loads(votes_json) if votes_json else []
                except (ValueError, TypeError):
                    votes = []
                out.append({'timestamp': ts, 'stock_code': code,
                            'consensus': cons, 'confidence': conf,
                            'agent_votes': votes})
            return out
        except Exception as e:
            logger.debug(f"[DecisionHistory] load 失败 (已吞): {e}")
            return []

    def load_briers(self, limit: int = 5000) -> List[Dict]:
        """读 Brier 持久池 (决策链 B 校准报告原料)。失败返回 [] (断链≠链死)。

        Returns:
            [{'code','ts','p','outcome'}, ...] — 最近 limit 条, 旧→新序 (id ASC)
        """
        if not self._db_path:
            return []
        try:
            conn = self._conn()
            try:
                rows = conn.execute(
                    'SELECT code, ts, p, outcome FROM decision_briers'
                    ' ORDER BY id DESC LIMIT ?', (limit,)).fetchall()
            finally:
                conn.close()
            return [{'code': c, 'ts': t, 'p': p, 'outcome': o}
                    for c, t, p, o in reversed(rows)]
        except Exception as e:
            logger.debug(f"[DecisionHistory] briers 读失败 (已吞): {e}")
            return []

    def append_briers(self, rows: List[tuple]) -> int:
        """Brier resolution 批量落库 (row=(code, ts, p, outcome))。

        UNIQUE(code, ts) 防重 = INSERT OR IGNORE 拒重复键 (同日 replay 重跑
        防重喂, 09-22 IC 双喂失真同教训)。失败吞掉返回 0 (不拖决策主链)。

        Returns:
            实际插入条数 (防重拒入/断链 = 0)
        """
        if not self._db_path or not rows:
            return 0
        try:
            conn = self._conn()
            try:
                cur = conn.executemany(
                    'INSERT OR IGNORE INTO decision_briers(code, ts, p, outcome)'
                    ' VALUES (?,?,?,?)', rows)
                conn.commit()
                return cur.rowcount or 0
            finally:
                conn.close()
        except Exception as e:
            logger.debug(f"[DecisionHistory] briers 写失败 (已吞): {e}")
            return 0

    def size(self) -> int:
        if not self._db_path:
            return 0
        try:
            conn = self._conn()
            try:
                n = conn.execute('SELECT COUNT(*) FROM decisions').fetchone()[0]
            finally:
                conn.close()
            return int(n or 0)
        except Exception:
            return 0
