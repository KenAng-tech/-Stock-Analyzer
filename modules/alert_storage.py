"""
告警持久化模块 — SQLite 存储

所有告警事件持久化到 data/alerts.db，服务器重启不丢失。
自动清理 7 天前的旧告警。
"""

import json
import os
import sqlite3
import time
from typing import Dict, List, Optional


class AlertStorage:
    """SQLite 告警持久化存储"""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or os.path.join('data', 'alerts.db')
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")  # 提高并发写入性能
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self):
        """初始化数据库表结构"""
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    stock_code TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    message TEXT,
                    severity TEXT DEFAULT 'medium',
                    category TEXT DEFAULT 'fund_flow',
                    timestamp TEXT NOT NULL,
                    data_json TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_alerts_code_time
                ON alerts(stock_code, timestamp DESC)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_alerts_type
                ON alerts(alert_type)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_alerts_severity
                ON alerts(severity)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_alerts_ts
                ON alerts(timestamp DESC)
            """)

    # ── 写入 ────────────────────────────────────────────────────

    def save_alert(self, alert: Dict):
        """保存单条告警"""
        with self._get_conn() as conn:
            conn.execute(
                """INSERT INTO alerts
                   (stock_code, alert_type, message, severity, category, timestamp, data_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    alert.get('stock_code', ''),
                    alert.get('type', alert.get('alert_type', 'unknown')),
                    alert.get('message', ''),
                    alert.get('severity', 'medium'),
                    alert.get('category', 'fund_flow'),
                    alert.get('timestamp', time.strftime('%Y-%m-%d %H:%M:%S')),
                    json.dumps(alert.get('data', {}), ensure_ascii=False),
                ),
            )

    def save_alerts(self, alerts: List[Dict], stock_code: str = ''):
        """批量保存告警"""
        if not alerts:
            return
        with self._get_conn() as conn:
            conn.executemany(
                """INSERT INTO alerts
                   (stock_code, alert_type, message, severity, category, timestamp, data_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        alert.get('stock_code', stock_code),
                        alert.get('type', alert.get('alert_type', 'unknown')),
                        alert.get('message', ''),
                        alert.get('severity', 'medium'),
                        alert.get('category', 'fund_flow'),
                        alert.get('timestamp', time.strftime('%Y-%m-%d %H:%M:%S')),
                        json.dumps(alert.get('data', {}), ensure_ascii=False),
                    )
                    for alert in alerts
                ],
            )

    # ── 读取 ────────────────────────────────────────────────────

    def get_alerts(self, stock_code: str = None, limit: int = 50) -> List[Dict]:
        """获取最近告警，可选按股票代码过滤"""
        with self._get_conn() as conn:
            if stock_code:
                rows = conn.execute(
                    """SELECT id, stock_code, alert_type, message, severity,
                              category, timestamp, data_json
                       FROM alerts
                       WHERE stock_code = ?
                       ORDER BY timestamp DESC
                       LIMIT ?""",
                    (stock_code, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT id, stock_code, alert_type, message, severity,
                              category, timestamp, data_json
                       FROM alerts
                       ORDER BY timestamp DESC
                       LIMIT ?""",
                    (limit,),
                ).fetchall()
            return [
                {
                    'id': r[0],
                    'stock_code': r[1],
                    'type': r[2],
                    'message': r[3],
                    'severity': r[4],
                    'category': r[5],
                    'timestamp': r[6],
                    'data': json.loads(r[7]) if r[7] else {},
                }
                for r in rows
            ]

    def get_recent_count(self, stock_code: str = None, minutes: int = 5) -> int:
        """获取最近 N 分钟的告警数量"""
        cutoff = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() - minutes * 60))
        with self._get_conn() as conn:
            if stock_code:
                row = conn.execute(
                    "SELECT COUNT(*) FROM alerts WHERE stock_code=? AND timestamp>=?",
                    (stock_code, cutoff),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT COUNT(*) FROM alerts WHERE timestamp>=?",
                    (cutoff,),
                ).fetchone()
            return row[0] if row else 0

    def get_summary(self, stock_code: str = None) -> Dict:
        """获取告警统计摘要"""
        with self._get_conn() as conn:
            where = "WHERE stock_code=?" if stock_code else ""
            params = (stock_code,) if stock_code else ()

            # 总数
            total = conn.execute(
                f"SELECT COUNT(*) FROM alerts {where}", params
            ).fetchone()[0]

            # 按严重度
            by_severity = {}
            for sev in ['high', 'medium', 'low']:
                count = conn.execute(
                    f"SELECT COUNT(*) FROM alerts {where} AND severity=?",
                    params + (sev,),
                ).fetchone()[0]
                by_severity[sev] = count

            # 按类型
            rows = conn.execute(
                f"SELECT alert_type, COUNT(*) FROM alerts {where} GROUP BY alert_type",
                params,
            ).fetchall()
            by_type = {row[0]: row[1] for row in rows}

            # 最近 5 / 15 分钟
            ts_5 = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() - 300))
            ts_15 = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() - 900))
            recent_5 = conn.execute(
                f"SELECT COUNT(*) FROM alerts {where} AND timestamp>=?",
                params + (ts_5,),
            ).fetchone()[0]
            recent_15 = conn.execute(
                f"SELECT COUNT(*) FROM alerts {where} AND timestamp>=?",
                params + (ts_15,),
            ).fetchone()[0]

            return {
                'total_alerts': total,
                'by_type': by_type,
                'by_severity': by_severity,
                'recent_5min': recent_5,
                'recent_15min': recent_15,
            }

    # ── 清理 ────────────────────────────────────────────────────

    def cleanup_old(self, days: int = 7):
        """删除 N 天前的旧告警"""
        cutoff = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() - days * 86400))
        with self._get_conn() as conn:
            result = conn.execute(
                "DELETE FROM alerts WHERE timestamp < ?", (cutoff,)
            )
            deleted = result.rowcount
        if deleted > 0:
            print(f"[AlertStorage] Cleaned up {deleted} old alerts")
