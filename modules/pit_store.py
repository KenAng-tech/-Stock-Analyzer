#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
pit_store.py — 点时 (point-in-time) 基本面仓库 (P0-1, 2026-09-15)

结构性反前视 (arXiv 2608.27734: 反前视必须做在数据层, 不能靠流程自觉)。
qlib 思想移植 (qlib/data/data.py:748-826 as-of 查询 + pit.py 拒未来引用):
回测任意历史日 as_of 只能看到 ann_date ≤ as_of 的披露版本 — 报告期
(period_end) 再新, 没公告就不可见 (fail-closed)。

- 修订链: revision_of 指向前身记录 id, 财报更正可追溯; as-of 按 ann_date
  排序天然取最新可见修订
- ann_date 缺失的记录 get_as_of 永不可见 (绝不拿 period_end 冒充可见性)
- SQLite 落盘 data/pit_store.db; 损坏自愈 (备份 .corrupt.{ts} 重建)
- 全部函数 guarded: 出错返回 None/[]/False, 绝不上抛
"""

import logging
import os
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger('stock_analyzer.pit_store')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, 'data', 'pit_store.db')

_lock = threading.RLock()
_SCHEMA = """
CREATE TABLE IF NOT EXISTS fundamentals (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    code        TEXT NOT NULL,
    field       TEXT NOT NULL,
    value       REAL,
    value_text  TEXT,
    period_end  TEXT,
    ann_date    TEXT,
    revision_of INTEGER,
    updated_ts  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pit_asof ON fundamentals(code, field, ann_date);
CREATE INDEX IF NOT EXISTS idx_pit_period ON fundamentals(code, field, period_end);
"""


def _connect(path: str = None) -> Optional[sqlite3.Connection]:
    """打开库; 损坏则备份重建后重开一次; 失败返回 None"""
    p = path or DB_PATH
    try:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        conn = sqlite3.connect(p, timeout=10)
        conn.executescript(_SCHEMA)
        return conn
    except sqlite3.DatabaseError as e:
        logger.error(f"[PIT] 库损坏, 备份重建: {e}")
        try:
            os.replace(p, p + f'.corrupt.{int(time.time())}')
            conn = sqlite3.connect(p, timeout=10)
            conn.executescript(_SCHEMA)
            return conn
        except Exception as e2:
            logger.error(f"[PIT] 重建失败: {e2}")
            return None
    except Exception as e:
        logger.error(f"[PIT] 打开失败: {e}")
        return None


def put(code: str, field: str, value: Any, period_end: Optional[str] = None,
        ann_date: Optional[str] = None, revision_of: Optional[int] = None,
        db_path: str = None) -> Optional[int]:
    """
    写入一条披露记录。ann_date=None 的记录 get_as_of 永不可见 (fail-closed)。

    Returns: 新记录 id, 失败 None
    """
    if not code or not field:
        return None
    conn = _connect(db_path)
    if conn is None:
        return None
    try:
        with _lock:
            num = float(value) if isinstance(value, (int, float)) or (
                isinstance(value, str) and value.replace('.', '', 1).replace('-', '', 1).isdigit()) else None
            cur = conn.execute(
                'INSERT INTO fundamentals(code, field, value, value_text, period_end, ann_date, revision_of, updated_ts) '
                'VALUES (?,?,?,?,?,?,?,?)',
                (str(code), str(field), num, None if num is not None else str(value),
                 period_end, ann_date, revision_of,
                 time.strftime('%Y-%m-%dT%H:%M:%S')))
            conn.commit()
            return int(cur.lastrowid)
    except Exception as e:
        logger.error(f"[PIT] put 失败 {code}/{field}: {e}")
        return None
    finally:
        conn.close()


def get_as_of(code: str, field: str, as_of_date: str, db_path: str = None) -> Optional[Dict]:
    """
    as-of 查询 (qlib searchsorted as-of 思想的 SQL 等价): 只返回
    ann_date ≤ as_of_date 的最新可见修订; 无则 None (拒未来引用)。
    """
    if not as_of_date:
        return None
    conn = _connect(db_path)
    if conn is None:
        return None
    try:
        row = conn.execute(
            'SELECT id, value, value_text, period_end, ann_date, revision_of '
            'FROM fundamentals WHERE code=? AND field=? AND ann_date IS NOT NULL AND ann_date<=? '
            'ORDER BY ann_date DESC, id DESC LIMIT 1',
            (str(code), str(field), str(as_of_date))).fetchone()
        if row is None:
            return None
        return {'id': row[0], 'code': code, 'field': field,
                'value': row[1] if row[1] is not None else row[2],
                'period_end': row[3], 'ann_date': row[4], 'revision_of': row[5]}
    except Exception as e:
        logger.error(f"[PIT] get_as_of 失败 {code}/{field}: {e}")
        return None
    finally:
        conn.close()


def latest(code: str, field: str, db_path: str = None) -> Optional[Dict]:
    """最新记录 (含 ann_date 缺失的; 仅供非回测路径, 回测禁用)"""
    conn = _connect(db_path)
    if conn is None:
        return None
    try:
        row = conn.execute(
            'SELECT id, value, value_text, period_end, ann_date, revision_of '
            'FROM fundamentals WHERE code=? AND field=? ORDER BY id DESC LIMIT 1',
            (str(code), str(field))).fetchone()
        if row is None:
            return None
        return {'id': row[0], 'code': code, 'field': field,
                'value': row[1] if row[1] is not None else row[2],
                'period_end': row[3], 'ann_date': row[4], 'revision_of': row[5]}
    except Exception as e:
        logger.error(f"[PIT] latest 失败: {e}")
        return None
    finally:
        conn.close()


def get_history(code: str, field: str, db_path: str = None) -> List[Dict]:
    """全部披露+修订历史 (审计用), 按 ann_date/id 升序"""
    conn = _connect(db_path)
    if conn is None:
        return []
    try:
        rows = conn.execute(
            'SELECT id, value, value_text, period_end, ann_date, revision_of, updated_ts '
            'FROM fundamentals WHERE code=? AND field=? ORDER BY COALESCE(ann_date, updated_ts), id',
            (str(code), str(field))).fetchall()
        return [{'id': r[0], 'value': r[1] if r[1] is not None else r[2],
                 'period_end': r[3], 'ann_date': r[4], 'revision_of': r[5], 'updated_ts': r[6]}
                for r in rows]
    except Exception as e:
        logger.error(f"[PIT] get_history 失败: {e}")
        return []
    finally:
        conn.close()


def stats(db_path: str = None) -> Dict[str, Any]:
    """记录数 / 可见率 (面板用)"""
    conn = _connect(db_path)
    if conn is None:
        return {'error': '库不可用'}
    try:
        total, visible = conn.execute(
            'SELECT COUNT(*), SUM(CASE WHEN ann_date IS NOT NULL THEN 1 ELSE 0 END) '
            'FROM fundamentals').fetchone()
        return {'total': int(total or 0), 'asof_visible': int(visible or 0),
                'db': DB_PATH}
    except Exception as e:
        logger.error(f"[PIT] stats 失败: {e}")
        return {'error': str(e)}
    finally:
        conn.close()
