#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""搜索全记账账本 — 每一次搜索评估落库 (P0-3, 2026-09-14)

背景 (arXiv 2608.27734 / Bailey & López de Prado 选择偏系列):
DSR (Deflated Sharpe Ratio) 与 PBO 的收缩幅度必须按 **真实 trial 数** 计算,
而系统现有 selection_bias_check / pbo_cscv_check (2026-09-11 #30, 观察模式)
的 trial 数只覆盖单次 study, 跨进程/跨夜的累计搜索预算无单一事实来源。
本模块提供独立 SQLite 账本: 任何搜索进程 (02:00 调参 worker、POST 端点
worker、未来 RD-Agent 挖掘链) 每完成一次评估即 record_trial, 下游按
trial_count(source) 取真实 N 收缩 DSR/PBO。

设计纪律:
- 独立库 data/search_ledger.db, 不与 SQLiteCache/业务库共享连接;
- 每次调用新开连接即用即关 (SQLiteCache 共享连接 bug 教训, 07-08);
- WAL + busy_timeout — 多进程 (worker 子进程 / Flask 主进程) 并发写安全;
- 轻量: 仅 stdlib (sqlite3/json/math), 不 import torch/akshare 等项目重模块,
  可被任何进程安全调用;
- record_trial 绝不抛异常打断调用方 (内部 try/except + logger.error,
  返回 bool) — 记账失败 ≠ 搜索链失败; 其余查询 API 同样防御式降级。

API:
    record_trial(source, params, metric_name, metric_value, run_id, note) -> bool
    trial_count(source=None, since_days=None) -> int
    recent_trials(source=None, limit=50, since_days=None) -> list[dict]
    source_summary() -> list[dict]   # 每 source 的 trials 数/最优值/最近时间 (面板用)
    prune(keep_days=90) -> int       # 保留策略: 删除过期行, 返回删除数
"""

import json
import math
import os
import sqlite3
from datetime import datetime, timedelta

# 项目 logger 优先 (项目规范); 独立进程外调用时回退 std logging — 双路皆轻量
try:
    from modules.logger import logger
except Exception:  # pragma: no cover - 项目外独立运行回退
    import logging
    logger = logging.getLogger('search_ledger')

__all__ = ['record_trial', 'trial_count', 'recent_trials',
           'source_summary', 'prune', 'DEFAULT_DB_PATH']

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DB_PATH = os.path.join(PROJECT_ROOT, 'data', 'search_ledger.db')

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trials (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           TEXT    NOT NULL,
    source       TEXT    NOT NULL,
    run_id       TEXT,
    params_json  TEXT,
    metric_name  TEXT,
    metric_value REAL,
    note         TEXT
);
CREATE INDEX IF NOT EXISTS idx_trials_source_ts ON trials(source, ts);
"""


def _resolve_db_path(db_path=None) -> str:
    """解析库路径: 显式参数 > 模块全局 (测试可 monkeypatch DEFAULT_DB_PATH)"""
    return db_path or DEFAULT_DB_PATH


def _connect(db_path=None) -> sqlite3.Connection:
    """每调用新建连接 (即用即关, 绝不跨调用复用), WAL + busy_timeout 多进程安全

    Raises:
        sqlite3.Error / OSError: 路径不可写等 — 由调用方防御式捕获
    """
    path = _resolve_db_path(db_path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    try:
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute('PRAGMA busy_timeout=5000')
        conn.executescript(_SCHEMA)
        conn.commit()
    except Exception:
        conn.close()
        raise
    return conn


def _now_iso() -> str:
    """统一时间戳格式 (秒级 ISO, 字典序 = 时间序, 供 since_days/prune 字符串比较)"""
    return datetime.now().isoformat(timespec='seconds')


def record_trial(source, params, metric_name, metric_value,
                 run_id=None, note=None, db_path=None) -> bool:
    """记账一次搜索评估。绝不抛异常 — 记账失败只记日志, 不打断调用方搜索链。

    Args:
        source: 搜索来源标识 ('hyperparam' / 'rdagent' / ...), 面板按此分组
        params: 参数字典 (json 可序列化; 不可序列化字段经 default=str 降级)
        metric_name: 指标名 (如 'cv_ic'), 与 metric_value 配对
        metric_value: 指标值; 非有限值 (NaN/inf) 存 NULL, 不污染 MAX/收缩统计
        run_id: 可选, 关联一次完整运行 (run_artifact 的 run / worker run_id)
        note: 可选备注 (如 'lgb/global study#0 trial#3')
        db_path: 可选, 覆盖库路径 (测试隔离用)

    Returns:
        bool: 写入成功与否。任何内部异常 → logger.error + False。
    """
    try:
        try:
            params_json = json.dumps(params if params is not None else {},
                                     ensure_ascii=False, default=str)
        except Exception:
            # default=str 仍失败 (如 __str__ 本身抛异常) → 类型名降级, 不丢行
            params_json = json.dumps({'__unserializable__': repr(type(params))})
        val = None
        if metric_value is not None:
            v = float(metric_value)   # 非数值在此抛 ValueError → 外层捕获
            val = v if math.isfinite(v) else None
        conn = _connect(db_path)
        try:
            conn.execute(
                'INSERT INTO trials (ts, source, run_id, params_json, '
                'metric_name, metric_value, note) VALUES (?,?,?,?,?,?,?)',
                (_now_iso(), str(source), run_id, params_json,
                 str(metric_name) if metric_name is not None else None, val, note))
            conn.commit()
        finally:
            conn.close()
        return True
    except Exception as e:
        logger.error(f"[search_ledger] record_trial 失败 (已吞, 不影响调用方): "
                     f"source={source} {type(e).__name__}: {e}")
        return False


def trial_count(source=None, since_days=None, db_path=None) -> int:
    """真实 trial 数 — DSR/PBO 收缩的 N 来源。

    Args:
        source: 过滤来源; None = 全部
        since_days: 仅统计最近 N 天; None = 全历史
        db_path: 覆盖库路径

    Returns:
        int: 行数; 查询失败 → 0 (防御式, 面板/链上不可因账本挂而炸)
    """
    try:
        where, args = [], []
        if source is not None:
            where.append('source = ?')
            args.append(str(source))
        if since_days is not None:
            where.append('ts >= ?')
            args.append((datetime.now() - timedelta(days=float(since_days)))
                        .isoformat(timespec='seconds'))
        sql = 'SELECT COUNT(*) FROM trials'
        if where:
            sql += ' WHERE ' + ' AND '.join(where)
        conn = _connect(db_path)
        try:
            return int(conn.execute(sql, args).fetchone()[0])
        finally:
            conn.close()
    except Exception as e:
        logger.error(f"[search_ledger] trial_count 失败 (返回 0): {e}")
        return 0


def recent_trials(source=None, limit=50, since_days=None, db_path=None) -> list:
    """最近 trials 明细 (新→旧)。params_json 解析回 dict 返回。

    Args:
        source: 过滤来源; since_days: 最近 N 天; limit: 最大条数
        db_path: 覆盖库路径

    Returns:
        list[dict]: [{id, ts, source, run_id, params, metric_name,
                      metric_value, note}]; 失败/空 → []
    """
    try:
        where, args = [], []
        if source is not None:
            where.append('source = ?')
            args.append(str(source))
        if since_days is not None:
            where.append('ts >= ?')
            args.append((datetime.now() - timedelta(days=float(since_days)))
                        .isoformat(timespec='seconds'))
        sql = ('SELECT id, ts, source, run_id, params_json, metric_name, '
               'metric_value, note FROM trials')
        if where:
            sql += ' WHERE ' + ' AND '.join(where)
        sql += ' ORDER BY id DESC LIMIT ?'
        args.append(int(limit))
        conn = _connect(db_path)
        try:
            rows = conn.execute(sql, args).fetchall()
        finally:
            conn.close()
        out = []
        for (rid, ts, src, run_id, params_json, mname, mval, note) in rows:
            try:
                params = json.loads(params_json) if params_json else {}
            except Exception:
                params = {'__raw__': params_json}
            out.append({'id': rid, 'ts': ts, 'source': src, 'run_id': run_id,
                        'params': params, 'metric_name': mname,
                        'metric_value': mval, 'note': note})
        return out
    except Exception as e:
        logger.error(f"[search_ledger] recent_trials 失败 (返回 []): {e}")
        return []


def source_summary(db_path=None) -> list:
    """按 source 汇总 (面板用): trials 数 / 最优值 / 最近时间。

    Returns:
        list[dict]: [{source, trials, best_value, best_metric_name, last_ts,
                      metrics: [{metric_name, trials, best_value, last_ts}]}],
                    按 trials 降序; 失败 → []
    """
    try:
        conn = _connect(db_path)
        try:
            rows = conn.execute(
                'SELECT source, metric_name, COUNT(*), MAX(metric_value), MAX(ts) '
                'FROM trials GROUP BY source, metric_name').fetchall()
        finally:
            conn.close()
        by_source = {}
        for src, mname, cnt, best, last_ts in rows:
            s = by_source.setdefault(src, {'source': src, 'trials': 0,
                                           'best_value': None,
                                           'best_metric_name': None,
                                           'last_ts': None, 'metrics': []})
            s['trials'] += int(cnt)
            s['metrics'].append({'metric_name': mname, 'trials': int(cnt),
                                 'best_value': best, 'last_ts': last_ts})
            if last_ts and (s['last_ts'] is None or last_ts > s['last_ts']):
                s['last_ts'] = last_ts
            if best is not None and (s['best_value'] is None or best > s['best_value']):
                s['best_value'] = best
                s['best_metric_name'] = mname
        return sorted(by_source.values(), key=lambda x: -x['trials'])
    except Exception as e:
        logger.error(f"[search_ledger] source_summary 失败 (返回 []): {e}")
        return []


def prune(keep_days=90, db_path=None) -> int:
    """保留策略: 删除 keep_days 之前的行。

    Args:
        keep_days: 保留天数 (负数按 0 处理 = 全删)
        db_path: 覆盖库路径

    Returns:
        int: 删除行数; 失败 → 0
    """
    try:
        days = max(0, float(keep_days))
        cutoff = (datetime.now() - timedelta(days=days)).isoformat(timespec='seconds')
        conn = _connect(db_path)
        try:
            cur = conn.execute('DELETE FROM trials WHERE ts < ?', (cutoff,))
            conn.commit()
            return int(cur.rowcount)
        finally:
            conn.close()
    except Exception as e:
        logger.error(f"[search_ledger] prune 失败 (返回 0): {e}")
        return 0
