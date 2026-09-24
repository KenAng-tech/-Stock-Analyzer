"""fusion_ledger.py — 多模态融合预测评估 SQLite 台账 (2026-09-22, 预测评估面板接通)

背景 (Phase 3 半成品接通): 预测评估面板三 item (准确率/Brier/漂移) 数据源
= MultiModalFusion._eval_history, 但全工程无 record_evaluation 调用方 →
n_records 恒 0 = 产生链从未建。本模块 = 产生链载体 (scan 链同宗):
  1. 种子持久: predict 结果落盘 (内存 _history 重启清零 → 次日无法对账)
  2. 对账: ≥3 交易日的种子 → 90d K 线真实收益 → record_evaluation → 评估落盘
  (窗口/|ret|<0.005 过滤/串行防重 = decision_replay consensus resolution 形制复用)

设计 (照搬 decision_history_store 教训形):
  - WAL + busy_timeout 2500ms, 每操作独立连接 (无跨调用连接失效)
  - UNIQUE(code, date) 防重: 同日同票一条种子 (手动重入不重复)
  - 全路径吞异常 + logger.debug: 台账永不拖垮评估主链
"""

import os
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from modules.logger import logger

_DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
_DEFAULT_DB = os.path.join(_DB_DIR, 'fusion_ledger.db')
_MAX_ROWS = 20000


class FusionLedger:
    """融合预测种子/评估 SQLite 台账 (线程安全: 每操作独立连接 + 文件锁)"""

    def __init__(self, db_path: Optional[str] = None):
        self._db_path = db_path or _DEFAULT_DB
        self._lock = threading.RLock()
        try:
            os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
            with self._conn() as conn:
                conn.execute('PRAGMA journal_mode=WAL')
                conn.execute('''CREATE TABLE IF NOT EXISTS seeds (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    code TEXT NOT NULL,
                    direction TEXT,
                    confidence REAL,
                    resolved INTEGER DEFAULT 0
                )''')
                # 同日同票防重 (唯一索引在表达式列上; ts 含秒恒异不能当约束)
                conn.execute('CREATE UNIQUE INDEX IF NOT EXISTS uq_seed_day '
                             'ON seeds(code, substr(ts, 1, 10))')
                conn.execute('''CREATE TABLE IF NOT EXISTS evals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    code TEXT NOT NULL,
                    pred_direction TEXT,
                    actual_return REAL,
                    correct INTEGER,
                    brier REAL
                )''')
                # 09-22 ① 拍板: IC 衰减监控样本持久 (重启不清零, 09-22 三因之一)
                conn.execute('''CREATE TABLE IF NOT EXISTS factor_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    code TEXT NOT NULL,
                    factor TEXT NOT NULL,
                    fval REAL,
                    ret REAL,
                    UNIQUE(code, factor, ts)
                )''')
        except Exception as e:
            logger.debug(f"[FusionLedger] 初始化失败 (内存降级): {e}")

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=2.5)
        conn.row_factory = sqlite3.Row
        return conn

    # ── 种子 ──────────────────────────────────────────────

    def append_seed(self, result: Dict) -> bool:
        """predict 结果落盘为种子 (同日同票防重; 失败只 debug, 不拖预测链)"""
        try:
            code = result.get('stock_code') or result.get('code')
            direction = result.get('direction')
            if not code or not direction:
                return False
            ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            with self._lock, self._conn() as conn:
                cur = conn.execute(
                    'INSERT OR IGNORE INTO seeds (ts, code, direction, confidence) '
                    'VALUES (?, ?, ?, ?)',
                    (ts, code, direction, float(result.get('confidence') or 0.5)))
                inserted = cur.rowcount > 0  # OR IGNORE 撞同日唯一 = 0 行 (诚实 False)
                conn.execute('DELETE FROM seeds WHERE rowid NOT IN '
                             '(SELECT rowid FROM seeds ORDER BY id DESC LIMIT ?)',
                             (_MAX_ROWS,))
                conn.commit()
            return inserted
        except Exception as e:
            logger.debug(f"[FusionLedger] 种子落盘跳过: {e}")
            return False

    def list_pending_seeds(self, min_age_days: int = 3,
                           limit: int = 50) -> List[Dict]:
        """≥min_age_days 交易日前且未对账的种子 (23:10 replay 链形)"""
        try:
            cutoff = (datetime.now() - timedelta(days=min_age_days)).strftime('%Y-%m-%d %H:%M:%S')
            with self._lock, self._conn() as conn:
                rows = conn.execute(
                    'SELECT id, ts, code, direction, confidence FROM seeds '
                    'WHERE resolved = 0 AND ts <= ? ORDER BY id LIMIT ?',
                    (cutoff, limit)).fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            logger.debug(f"[FusionLedger] 种子读取跳过: {e}")
            return []

    def mark_resolved(self, seed_id: int):
        """标记已对账 (含 |ret|<0.005 无方向样本 — 不回填, 防重扫堆积)"""
        try:
            with self._lock, self._conn() as conn:
                conn.execute('UPDATE seeds SET resolved = 1 WHERE id = ?', (seed_id,))
                conn.commit()
        except Exception as e:
            logger.debug(f"[FusionLedger] 对账标记跳过: {e}")

    # ── 评估 ──────────────────────────────────────────────

    def append_eval(self, eval_entry: Dict, code: str) -> bool:
        with self._lock, self._conn() as conn:
            conn.execute(
                'INSERT INTO evals (ts, code, pred_direction, actual_return, correct, brier) '
                'VALUES (?, ?, ?, ?, ?, ?)',
                (eval_entry.get('timestamp', datetime.now().isoformat()), code,
                 eval_entry.get('pred_direction'), eval_entry.get('actual_return'),
                 int(bool(eval_entry.get('correct'))), eval_entry.get('brier')))
            conn.execute('DELETE FROM evals WHERE rowid NOT IN '
                         '(SELECT rowid FROM evals ORDER BY id DESC LIMIT ?)',
                         (_MAX_ROWS,))
            conn.commit()
        return True

    def load_evals(self, limit: int = 500) -> List[Dict]:
        """重启恢复: 评估历史读出 (MultiModalFusion._eval_history 形制)"""
        try:
            with self._lock, self._conn() as conn:
                rows = conn.execute(
                    'SELECT ts, code, pred_direction, actual_return, correct, brier '
                    'FROM evals ORDER BY id DESC LIMIT ?', (limit,)).fetchall()
            return [{'timestamp': r['ts'], 'correct': bool(r['correct']),
                     'brier': r['brier'], 'pred_direction': r['pred_direction'],
                     'actual_return': r['actual_return']} for r in reversed(rows)]
        except Exception as e:
            logger.debug(f"[FusionLedger] 评估恢复跳过: {e}")
            return []


    # ── IC 衰减监控 (09-22 ① 拍板: 样本持久, 仿 decisions 教训形) ──

    def append_factor_records(self, code: str, date: str,
                              records: Dict[str, tuple]) -> bool:
        """批量落盘 ({factor: (fval, ret)}) — UNIQUE(code,factor,ts) 防重。"""
        try:
            ts = f"{date} 23:10:00"
            with self._lock, self._conn() as conn:
                conn.executemany(
                    'INSERT OR IGNORE INTO factor_records (ts, code, factor, fval, ret) '
                    'VALUES (?, ?, ?, ?, ?)',
                    [(ts, code, f, float(v[0]), float(v[1]))
                     for f, v in records.items() if len(v) == 2])
                # 23:10 链 + analyze 链双喂 ≈690 行/日 → 20w 行封顶 (~1 年滚动窗)
                conn.execute('DELETE FROM factor_records WHERE rowid NOT IN '
                             '(SELECT rowid FROM factor_records '
                             'ORDER BY id DESC LIMIT 200000)')
                conn.commit()
            return True
        except Exception as e:
            logger.debug(f"[FusionLedger] IC 样本落盘跳过: {e}")
            return False

    def load_factor_records(self, limit: int = 50000) -> Dict:
        """读出 = _factor_returns 形 (code → {factor: [(date, fval, ret)]})"""
        try:
            with self._lock, self._conn() as conn:
                rows = conn.execute(
                    'SELECT ts, code, factor, fval, ret FROM factor_records '
                    'ORDER BY id LIMIT ?', (limit,)).fetchall()
            out: Dict[str, Dict[str, list]] = {}
            for r in rows:
                out.setdefault(r['code'], {}).setdefault(r['factor'], []).append(
                    (r['ts'][:10], r['fval'], r['ret']))
            return out
        except Exception as e:
            logger.debug(f"[FusionLedger] IC 样本读出跳过: {e}")
            return {}


_ledger: Optional[FusionLedger] = None
_ledger_lock = threading.Lock()


_ledger_fail_logged = False


def get_fusion_ledger() -> Optional[FusionLedger]:
    """台账单例; 初始化失败 → None (调用侧 None 检查降级, 链坏≠链死)"""
    global _ledger, _ledger_fail_logged
    if _ledger is None:
        with _ledger_lock:
            if _ledger is None:
                try:
                    _ledger = FusionLedger()
                except Exception as e:
                    if not _ledger_fail_logged:
                        _ledger_fail_logged = True
                        logger.error(f"[FusionLedger] 台账初始化失败 (评估链降级跳过): {e}")
                    return None
    return _ledger
