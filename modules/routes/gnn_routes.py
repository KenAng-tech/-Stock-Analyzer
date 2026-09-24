"""
gnn_routes.py - /api/gnn/*

图神经网络路由 (2026-09-03: stub → RealTimeGNN 真链)

旧 stub 问题 (页面空转):
  - /realtime/status 硬编码 graph 0/0, 页面永远"0 节点 0 边"
  - /realtime/update 1.1ms 返回假 "GNN update triggered", 无任何建图
真链:
  - /realtime/update: 抓取 watchlist K线 (每票 12s 超时保护) → 真实收益率
    → RealTimeGNN.update_graph() 相关性建图 (纯 numpy, 无 MPS)
  - /realtime/status: 读图真实状态 (节点/边/市场/跨市场信号/更新时间)
"""

import threading
import time
from datetime import datetime

import numpy as np
from flask import Blueprint, request, jsonify

from modules.logger import logger

bp = Blueprint('gnn', __name__)

# 默认 watchlist (K线源恒可用; POST body {"stock_codes": [...]} 可覆盖, 上限 12)
_WATCHLIST = [
    ('sz300620', 'cn', 'New Energy'),
    ('sh688981', 'cn', 'Semiconductor'),
    ('sh600519', 'cn', 'Consumer'),
    ('sh601318', 'cn', 'Finance'),
    ('sz000001', 'cn', 'Banking'),
]

_gnn_lock = threading.RLock()
_gnn = None  # RealTimeGNN 惰性单例


def _get_gnn():
    """RealTimeGNN 单例 (线程安全)"""
    global _gnn
    with _gnn_lock:
        if _gnn is None:
            from modules.models.realtime_gnn import RealTimeGNN
            _gnn = RealTimeGNN()
        return _gnn


@bp.route('/api/gnn/realtime/status', methods=['GET'])
def api_gnn_realtime_status():
    """GNN 实时图状态 (只读, 0ms 级)"""
    try:
        with _gnn_lock:
            gnn = _get_gnn()
            adj = gnn._adjacency
            n_nodes = len(gnn._node_ids)
            n_edges = 0
            if adj is not None and getattr(adj, 'size', 0) > 0:
                n_edges = int(np.count_nonzero(adj) // 2)
            markets = []
            if gnn._nodes:
                markets = sorted({gnn._nodes[c].market for c in gnn._node_ids
                                  if c in gnn._nodes})
            last_upd = (gnn._last_update.isoformat()
                        if hasattr(gnn._last_update, 'isoformat') else None)
            n_signals = len(gnn._cross_signals)

        # 响应键 = 前端 loadGNNRealtimeStatus 期望 shape (兼容 webgui dashboard.js)
        return jsonify({
            'success': True,
            'status': 'ready' if last_upd else 'idle',
            'data': {
                'n_nodes': n_nodes,
                'n_edges': n_edges,
                'markets': markets,
                'cross_signals': n_signals,
                'last_update': last_upd,
            },
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"[GNN] status error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)[:150],
            'data': {'n_nodes': 0, 'n_edges': 0, 'markets': [], 'cross_signals': 0},
        }), 200


@bp.route('/api/gnn/realtime/update', methods=['GET', 'POST'])
def api_gnn_realtime_update():
    """
    实时建图 (真链): watchlist/body K线 → returns → 相关性图。

    - K线抓取: 每票 thread join 12s 超时 (miss 跳过, 不阻塞)
    - 纯 numpy (RealTimeGNN.update_graph), 无 MPS 锁需求
    - 并发: _gnn_lock 串行 (重复触发排队, 第二个直接复用刚建好的图)
    """
    try:
        body = request.get_json(silent=True) or {}
        codes = body.get('stock_codes') or [c for c, _, _ in _WATCHLIST]
        codes = list(dict.fromkeys(codes))[:12]
        market_map = {c: (m, ind) for c, m, ind in _WATCHLIST}

        from modules.data_fetcher import StockDataFetcher
        fetcher = StockDataFetcher()

        t0 = time.time()
        with _gnn_lock:
            gnn = _get_gnn()

            added, skipped = [], []
            for code in codes:
                holder, errh = [None], [None]

                def _fetch(c=code):
                    try:
                        holder[0] = fetcher.get_kline_data(c, 'daily', 60)
                    except Exception as e:
                        errh[0] = e

                th = threading.Thread(target=_fetch, daemon=True)
                th.start()
                th.join(timeout=12)
                kl = holder[0]
                if th.is_alive() or not kl or len(kl) < 30:
                    skipped.append(code)
                    continue

                closes = np.array([k.get('close', 0) or 0 for k in kl], dtype=float)
                with np.errstate(divide='ignore', invalid='ignore'):
                    rets = np.diff(closes) / np.where(closes[:-1] == 0, np.nan, closes[:-1])
                rets = np.nan_to_num(rets, nan=0.0)
                if len(rets) < 30 or not np.any(rets != 0):
                    skipped.append(code)
                    continue

                market, industry = market_map.get(code, ('cn', 'Auto'))
                if code not in gnn._node_ids:
                    # features: 末 12 日收益率 (GraphNode.features 维度展示 + 后续扩展用)
                    feats = rets[-12:] if len(rets) >= 12 else np.pad(rets, (0, 12 - len(rets)))
                    gnn.add_stock(code, market, industry, feats)
                gnn.update_returns(code, list(rets))
                added.append(code)

            adj = gnn.update_graph() if len(gnn._node_ids) >= 2 else None

            n_nodes = len(gnn._node_ids)
            n_edges = 0
            if adj is not None and getattr(adj, 'size', 0) > 0:
                n_edges = int(np.count_nonzero(adj) // 2)

            dt = time.time() - t0
            logger.info(f"[GNN] 建图完成: {n_nodes} 节点 {n_edges} 边, "
                        f"新增 {len(added)}, 跳过 {len(skipped)}, 耗时 {dt:.1f}s")

        return jsonify({
            'success': True,
            'message': f'图已更新: {n_nodes} 节点 / {n_edges} 边'
                       + (f' (跳过 {len(skipped)}: {skipped[:4]})' if skipped else ''),
            'data': {'n_nodes': n_nodes, 'n_edges': n_edges,
                     'added': added, 'elapsed': round(dt, 2)},
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"[GNN] update error: {e}")
        return jsonify({'success': False, 'error': str(e)[:150]}), 500
