#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
factor_cache.py — 因子语义缓存 (P1-5, 2026-09-15)

qlib ExpressionCache 思想移植 (qlib/data/cache.py:502-644): 以**表达式字符串**
为键 (规范化后哈希), 全历史单文件落盘, 增量只补尾部, 按右端截尾重算防复权漂移。

设计要点:
- expr_key: 表达式去空白+小写 → sha256, 异串同键 (空格/大小写差异命中同一缓存)
- get_or_compute: 命中直接返回; 右端落后 → 丢最后 overlap 根重叠 bar 后只重算
  尾部 (lookback 根上下文供滚动窗口); 首日期变了 → 全量重算
- 缓存任何异常 → 透明回源 (compute_fn 直调), 缓存坏绝不拖垮因子计算
- 2026-09-15 接线: alpha360/multi_factor 请求路径 use_cache=True 默认开,
  环境变量 FACTOR_SEMANTIC_CACHE=0 一键关闭 (killswitch, 回到全量直算)
"""

import hashlib
import json
import logging
import os
import re
import threading
import time
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger('stock_analyzer.factor_cache')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE_DIR = os.path.join(PROJECT_ROOT, 'data', 'factor_cache')

_lock = threading.RLock()
_stats = {'hits': 0, 'misses': 0, 'incremental': 0, 'full_recompute': 0,
          'fallback': 0}

# 语义缓存主开关 (killswitch, 2026-09-15): 默认开, 仅 FACTOR_SEMANTIC_CACHE
# 设为 0/false/off/空 时全关 — get_or_compute 入口直接绕过, 行为=接线前全量直算
_ENABLED = os.environ.get('FACTOR_SEMANTIC_CACHE', '1').strip().lower() not in (
    '0', 'false', 'off', '')


def _normalize_expr(expr: str) -> str:
    """表达式规范化: 去所有空白 + 小写 (qlib 表达式串作键的前提)"""
    return re.sub(r'\s+', '', str(expr)).lower()


def expr_key(expr: str, code: str, freq: str = 'day') -> str:
    """规范化表达式 + 代码 + 频率 → sha256 键"""
    payload = f"{_normalize_expr(expr)}|{str(code).strip().lower()}|{str(freq).strip().lower()}"
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def klines_fingerprint(klines: List[Dict]) -> Dict[str, Any]:
    """
    K 线数据指纹: 首末日期 + 根数 + 末根 close hash。

    hash 含末根 close: 盘中当日未收盘 K 线 close 变动会使指纹变化 →
    缓存不会命中过期值 (仅日期+根数无法感知盘中变动, 同 code 盘中
    重复请求会全命中过期缓存)。指纹变化时走增量/全量重算, 正确性优先。

    Returns: {'first': str, 'last': str, 'n': int, 'hash': str}
    """
    if not klines:
        return {'first': '', 'last': '', 'n': 0, 'hash': 'empty'}
    def _d(k):
        return str(k.get('date') or k.get('time') or str(k.get('datetime', ''))[:10])
    first, last = _d(klines[0]), _d(klines[-1])
    n = len(klines)
    last_close = round(float(klines[-1].get('close', 0) or 0), 4)
    h = hashlib.sha256(f'{first}|{last}|{n}|{last_close}'.encode()).hexdigest()[:16]
    return {'first': first, 'last': last, 'n': n, 'hash': h}


def _path(key: str) -> str:
    return os.path.join(CACHE_DIR, f'{key}.json')


def _serialize(values: Dict[str, Any]) -> Dict[str, Any]:
    """np.ndarray/list/标量 → JSON 安全值"""
    out = {}
    for k, v in values.items():
        if hasattr(v, 'tolist'):
            out[k] = v.tolist()
        elif isinstance(v, (list, tuple)):
            out[k] = list(v)
        else:
            out[k] = v
    return out


def _load(key: str) -> Optional[Dict]:
    p = _path(key)
    if not os.path.exists(p):
        return None
    try:
        with open(p, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f'[FactorCache] 缓存文件损坏 {key[:12]}: {e} — 回源重算')
        try:
            os.replace(p, p + f'.corrupt.{int(time.time())}')
        except OSError:
            pass
        return None


def _save(key: str, envelope: Dict) -> bool:
    try:
        with _lock:
            os.makedirs(CACHE_DIR, exist_ok=True)
            tmp = _path(key) + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(envelope, f, ensure_ascii=False)
            os.replace(tmp, _path(key))
        return True
    except Exception as e:
        logger.error(f'[FactorCache] 缓存写入失败 {key[:12]}: {e}')
        return False


def get_or_compute(expr: str, code: str, klines_fingerprint_or_bars,
                   compute_fn: Callable[[List[Dict]], Dict[str, Any]],
                   freq: str = 'day', overlap: int = 3,
                   lookback: int = 0) -> Dict[str, Any]:
    """
    语义缓存主入口。

    Args:
        expr: 因子表达式 (语义键本体, 如 "Mean($close, 20)")
        code: 股票代码
        klines_fingerprint_or_bars: klines_fingerprint() 结果 dict, 或直接传
            K 线列表 (内部算指纹并用于尾部切片); 传纯 hash 串则退化为
            全量命中/全量重算 (无日期信息无法增量)
        compute_fn: callable(klines_slice) -> Dict[str, series|scalar]
        overlap: 增量时丢弃缓存尾部重叠 bar 数 (防复权漂移, qlib remove_n 思想)
        lookback: 尾部切片额外多带的历史上下文根数 (滚动窗口因子需要)

    缓存任何异常透明回源, 绝不抛出。
    """
    fp_input = klines_fingerprint_or_bars
    klines = None
    if isinstance(fp_input, list):
        klines = fp_input
        fp = klines_fingerprint(klines)
    elif isinstance(fp_input, dict):
        fp = fp_input
    else:
        fp = {'first': '', 'last': '', 'n': 0, 'hash': str(fp_input)}

    if not _ENABLED:
        # killswitch: 完全绕过缓存 (不计 stats), 回到接线前的全量直算行为
        return compute_fn(klines if klines is not None else [])

    key = expr_key(expr, code, freq)
    try:
        cached = _load(key)
        if cached is None:
            with _lock:
                _stats['misses'] += 1
            values = compute_fn(klines if klines is not None else [])
            _save(key, {'meta': {'expr': expr, 'code': code, 'freq': freq,
                                 'fp': fp, 'created': time.strftime('%Y-%m-%dT%H:%M:%S')},
                        'dates': _dates(klines) if klines else [],
                        'values': _serialize(values)})
            return values

        cfp = cached.get('meta', {}).get('fp', {})
        if cfp.get('hash') == fp.get('hash'):
            with _lock:
                _stats['hits'] += 1
            return cached.get('values', {})

        # 指纹不同: 首日期一致且能切片 → 尾部增量; 否则全量
        dates = cached.get('dates') or []
        can_incr = (klines is not None and dates and fp.get('first')
                    and cfp.get('first') == fp.get('first')
                    and len(dates) > overlap)
        if can_incr:
            cut_idx = len(dates) - overlap  # 丢弃缓存尾部 overlap 根 (复权漂移区)
            slice_start = max(0, cut_idx - lookback)
            tail = _serialize(compute_fn(klines[slice_start:]))
            new_dates = dates[:cut_idx] + _dates(klines)[cut_idx:]
            n_append = len(klines) - cut_idx  # 尾部需并入的根数
            merged = {}
            for kname, vals in cached.get('values', {}).items():
                tv = tail.get(kname)
                if isinstance(vals, list) and isinstance(tv, list):
                    merged[kname] = vals[:cut_idx] + tv[len(tv) - n_append:]
                else:
                    merged[kname] = tv if tv is not None else vals
            _save(key, {'meta': {**cached['meta'], 'fp': fp},
                        'dates': new_dates, 'values': merged})
            with _lock:
                _stats['incremental'] += 1
            return merged

        with _lock:
            _stats['full_recompute'] += 1
        values = compute_fn(klines if klines is not None else [])
        _save(key, {'meta': {**cached.get('meta', {}), 'fp': fp},
                    'dates': _dates(klines) if klines else [],
                    'values': _serialize(values)})
        return values
    except Exception as e:
        # 透明回源: 缓存层任何故障不拖垮因子计算
        logger.error(f'[FactorCache] 缓存故障回源 {key[:12]}: {e}', exc_info=True)
        with _lock:
            _stats['fallback'] += 1
        try:
            return compute_fn(klines if klines is not None else [])
        except Exception:
            raise


def _dates(klines: List[Dict]) -> List[str]:
    return [str(k.get('date') or k.get('time') or str(k.get('datetime', ''))[:10])
            for k in klines]


def stats() -> Dict[str, Any]:
    """命中率与磁盘占用 (面板/巡检用)"""
    with _lock:
        s = dict(_stats)
    total = s['hits'] + s['misses'] + s['incremental'] + s['full_recompute']
    s['hit_rate'] = round(s['hits'] / total, 4) if total else None
    try:
        files = os.listdir(CACHE_DIR) if os.path.isdir(CACHE_DIR) else []
        s['entries'] = len([f for f in files if f.endswith('.json')])
        s['disk_mb'] = round(sum(
            os.path.getsize(os.path.join(CACHE_DIR, f)) for f in files
            if f.endswith('.json')) / 1048576.0, 2)
    except OSError:
        s['entries'], s['disk_mb'] = 0, 0.0
    return s


def clear(stale_days: int = 30) -> int:
    """清理 stale_days 天未更新的缓存条目; 返回删除数"""
    removed = 0
    try:
        if not os.path.isdir(CACHE_DIR):
            return 0
        cutoff = time.time() - stale_days * 86400
        with _lock:
            for f in os.listdir(CACHE_DIR):
                if not f.endswith('.json'):
                    continue
                p = os.path.join(CACHE_DIR, f)
                try:
                    if os.path.getmtime(p) < cutoff:
                        os.remove(p)
                        removed += 1
                except OSError:
                    continue
    except Exception as e:
        logger.error(f'[FactorCache] clear 失败: {e}')
    return removed
