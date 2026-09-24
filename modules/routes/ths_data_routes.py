"""
modules/routes/ths_data_routes.py — 同花顺数据 API 端点

提供基于 hithink-finance CLI 的数据查询端点:
  - GET /api/ths/realtime/<code>     实时行情
  - GET /api/ths/klines/<code>       历史 K 线
  - GET /api/ths/valuation/<code>    估值快照
  - GET /api/ths/resolve?q=          标的消歧
  - GET /api/ths/status              数据状态

使用方式:
    curl 'http://127.0.0.1:5002/api/ths/realtime/300620.SZ'
    curl 'http://127.0.0.1:5002/api/ths/klines/300620.SZ?start=2026-09-01'
    curl 'http://127.0.0.1:5002/api/ths/resolve?q=宁德时代'
"""

import logging
from flask import Blueprint, request, jsonify
from datetime import datetime

logger = logging.getLogger('stock_analyzer.ths')
bp = Blueprint('ths_data', __name__, url_prefix='/api/ths')


def _json_response(data, meta=None, request_id=None):
    """统一 JSON 响应格式 (含数据溯源三要素)"""
    response = {"ok": True}
    if data is not None:
        response["data"] = data
    # 数据溯源: 来源 + 时间戳 + 复权方式
    source_meta = {
        "source": meta.get("source", "hithink-finance") if meta else "hithink-finance",
        "timestamp": datetime.now().isoformat(),
        "adjusted": meta.get("adjusted", "forward"),
    }
    if meta:
        source_meta.update(meta)
    response["meta"] = source_meta
    if request_id:
        response["request_id"] = request_id
    return jsonify(response), 200


def _error_response(code, hint, status=400, debug=None):
    """统一错误响应格式"""
    error = {"code": code, "hint": hint}
    if debug:
        error["debug"] = debug
    return jsonify({"ok": False, "error": error}), status


@bp.route('/realtime/<thscode>')
def api_ths_realtime(thscode):
    """实时行情"""
    try:
        from utils.ths_data_provider import get_ths_data_provider
        provider = get_ths_data_provider()
        data = provider.get_realtime(thscode)
        if data:
            return _json_response(data, meta={"source": "hithink-finance", "type": "realtime"})
        return _error_response("NO_DATA", f"未获取到 {thscode} 的实时行情")
    except Exception as e:
        return _error_response("FETCH_ERROR", str(e), debug=str(e))


@bp.route('/klines-batch')
def api_ths_klines_batch():
    """批量 K 线 (本地 DuckDB 单 SQL, 2026-09-19)

    用法: GET /api/ths/klines-batch?codes=600519.SH,300620.SZ&start=2026-09-01
    纪律: codes 上限 50 (防大结果撑爆 stdout); 全市场面板请走 CLI market panel --output
    """
    try:
        from utils.ths_data_provider import get_ths_data_provider
        codes_raw = request.args.get('codes', '').strip()
        if not codes_raw:
            return _error_response('BAD_REQUEST', '缺参数: ?codes=600519.SH,300620.SZ')
        clist = [c.strip() for c in codes_raw.split(',') if c.strip()][:50]
        provider = get_ths_data_provider()
        t0 = datetime.now()
        data = provider.get_klines_batch(
            clist, request.args.get('start'), request.args.get('end'))
        ms = (datetime.now() - t0).total_seconds() * 1000
        return _json_response(data, {
            "source": f"duckdb_local_batch:{ms:.0f}ms",
            "codes_requested": len(clist),
            "codes_hit": len(data),
        })
    except Exception as e:
        logger.error(f"[ths_batch] K线批量查询失败: {e}", exc_info=True)
        return _error_response('INTERNAL_ERROR', str(e), 500, debug=str(e))


@bp.route('/klines/<thscode>')
def api_ths_klines(thscode):
    """历史 K 线"""
    try:
        from utils.ths_data_provider import get_ths_data_provider
        provider = get_ths_data_provider()

        start = request.args.get('start')
        end = request.args.get('end')

        data = provider.get_klines(thscode, start=start, end=end)
        if data:
            meta = {
                "source": getattr(provider, "last_source", "hithink-finance"),
                "type": "klines",
                "count": len(data),
            }
            if start:
                meta["start"] = start
            if end:
                meta["end"] = end
            return _json_response(data, meta=meta)
        return _error_response("NO_DATA", f"未获取到 {thscode} 的 K 线数据")
    except Exception as e:
        return _error_response("FETCH_ERROR", str(e), debug=str(e))


@bp.route('/valuation/<thscode>')
def api_ths_valuation(thscode):
    """估值快照"""
    try:
        from utils.ths_data_provider import get_ths_data_provider
        provider = get_ths_data_provider()
        data = provider.get_valuation(thscode)
        if data:
            return _json_response(data, meta={"source": "hithink-finance", "type": "valuation"})
        return _error_response("NO_DATA", f"未获取到 {thscode} 的估值数据")
    except Exception as e:
        return _error_response("FETCH_ERROR", str(e), debug=str(e))


@bp.route('/resolve')
def api_ths_resolve():
    """标的消歧"""
    try:
        query = request.args.get('q', '').strip()
        if not query:
            return _error_response("MISSING_QUERY", "请提供 q 参数")

        limit = int(request.args.get('limit', 5))

        from utils.ths_data_provider import get_ths_data_provider
        provider = get_ths_data_provider()
        data = provider.resolve_symbol(query, limit=limit)
        if data:
            return _json_response(data, meta={"source": "hithink-finance", "type": "resolve", "query": query})
        return _error_response("NO_RESULTS", f"未找到与 '{query}' 匹配的标的")
    except Exception as e:
        return _error_response("FETCH_ERROR", str(e), debug=str(e))


@bp.route('/status')
def api_ths_status():
    """数据状态"""
    try:
        from utils.ths_data_provider import get_ths_data_provider
        provider = get_ths_data_provider()
        status = provider.get_data_status()
        if status:
            return _json_response(status, meta={"source": "hithink-finance", "type": "data_status"})
        return _error_response("NO_DATA", "无法获取数据状态")
    except Exception as e:
        return _error_response("FETCH_ERROR", str(e), debug=str(e))


@bp.route('/sync', methods=['POST'])
def api_ths_sync():
    """同步数据 (POST)"""
    try:
        from utils.ths_data_provider import get_ths_data_provider
        provider = get_ths_data_provider()

        start = request.json.get('start') if request.json else None
        end = request.json.get('end') if request.json else None

        success = provider.sync_data(start=start, end=end)
        if success:
            return _json_response({"synced": True}, meta={"source": "hithink-finance", "type": "sync"})
        return _error_response("SYNC_FAILED", "数据同步失败")
    except Exception as e:
        return _error_response("SYNC_ERROR", str(e), debug=str(e))


# ── 特色数据端点 ──────────────────────────────────────────────

@bp.route('/dragon-tiger')
def api_ths_dragon_tiger():
    """龙虎榜数据"""
    try:
        from utils.ths_data_provider import get_ths_data_provider
        provider = get_ths_data_provider()
        limit = int(request.args.get('limit', 20))
        data = provider.get_dragon_tiger(limit=limit)
        if data:
            return _json_response(data, meta={"source": "hithink-finance", "type": "dragon_tiger"})
        return _json_response([], meta={"source": "hithink-finance", "type": "dragon_tiger", "note": "暂无数据"})
    except Exception as e:
        return _error_response("FETCH_ERROR", str(e), debug=str(e))


@bp.route('/limit-up-pool')
def api_ths_limit_up():
    """涨停池"""
    try:
        from utils.ths_data_provider import get_ths_data_provider
        provider = get_ths_data_provider()
        date = request.args.get('date')
        data = provider.get_limit_up_pool(date=date)
        if data:
            return _json_response(data, meta={"source": "hithink-finance", "type": "limit_up_pool"})
        return _json_response([], meta={"source": "hithink-finance", "type": "limit_up_pool", "note": "暂无数据"})
    except Exception as e:
        return _error_response("FETCH_ERROR", str(e), debug=str(e))


@bp.route('/hot-stocks')
def api_ths_hot_stocks():
    """热股榜"""
    try:
        from utils.ths_data_provider import get_ths_data_provider
        provider = get_ths_data_provider()
        limit = int(request.args.get('limit', 20))
        data = provider.get_hot_stocks(limit=limit)
        if data:
            return _json_response(data, meta={"source": "hithink-finance", "type": "hot_stocks"})
        return _json_response([], meta={"source": "hithink-finance", "type": "hot_stocks", "note": "暂无数据"})
    except Exception as e:
        return _error_response("FETCH_ERROR", str(e), debug=str(e))
