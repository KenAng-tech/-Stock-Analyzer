"""
SOTA LLM Routes - LLM sentiment, router, and cross-market endpoints

Extracted from sota_routes.py — LLM-focused endpoints.
"""

from flask import Blueprint, request, jsonify
from datetime import datetime
import numpy as np
import threading  # 2026-09-10 修复: 本文件 :52 一直 NameError — 路由层从模块抽取时漏了 import (AH premium 永久 NameError→硬编码假数据冒充真实行情)

from modules.logger import logger
from modules.data_fetcher import StockDataFetcher
from .shared_utils import get_stock_data, ema
from .predictor_factory import predictor_factory

bp = Blueprint('sota_llm', __name__)

@bp.route('/api/sota/cross-market/summary', methods=['GET'])
def api_cross_market_summary():
    """跨市场分析摘要"""
    try:
        from modules.cross_market import get_cross_market_analyzer
        analyzer = get_cross_market_analyzer()
        return jsonify({'data': analyzer.get_summary()})
    except Exception as e:
        logger.error(f"[API] Cross-market summary error: {e}")
        return jsonify({'error': str(e)}), 500


@bp.route('/api/sota/cross-market/ah-premium', methods=['GET'])
def api_cross_market_ah_premium():
    """AH 溢价分析 (带缓存 + 超时保护)"""
    try:
        from modules.dynamic_cache import cache

        # 1. 检查缓存 (TTL=300s)
        cached = cache.get('ah_premium_sz300620_hk00700', category='strategy')
        if cached:
            return jsonify({'data': cached})

        # 2. 带超时执行
        result = [None]
        error = [None]
        def _run():
            try:
                from modules.cross_market import get_cross_market_analyzer
                analyzer = get_cross_market_analyzer()
                result[0] = analyzer.calculate_ah_premium('sz300620', 'hk00700')
            except Exception as e:
                error[0] = e

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        thread.join(timeout=10)

        if thread.is_alive() or error[0]:
            raise TimeoutError('Cross-market data fetch timed out')
        data = result[0]

        # 3. Fallback (2026-09-10 修复: 原返回硬编码假溢价 -14.6% 冒充真实行情,
        # 决策链/面板把假数当真 → no_data 诚实返回空, 不再给假数)
        if not data or data.get('signal') == 'no_data':
            logger.warning("[API] AH premium: 实时行情不可得 (akshare/tencent), 诚实降级")
            return jsonify({'data': {'premium_pct': None, 'signal': 'no_data'},
                            'source': 'no_data'})

        # 4. 缓存结果
        cache.set('ah_premium_sz300620_hk00700', data, category='strategy', ttl=300)
        return jsonify({'data': data})
    except Exception as e:
        logger.error(f"[API] AH premium error: {e}")
        return jsonify({'data': {'premium_pct': None, 'signal': 'fetch_error'},
                        'source': 'fallback_error', 'error': str(e)})


@bp.route('/api/sota/llm-router/status', methods=['GET'])
def api_llm_router_status():
    """LLM 路由状态 API"""
    try:
        llm_router = None
        try:
            from modules.llm_router import llm_router
        except Exception as e:
            return jsonify({'success': False, 'error': f'LLM Router 加载失败: {e}', 'timestamp': datetime.now().isoformat()}), 503

        status = llm_router.get_status()
        return jsonify({'success': True, 'data': status, 'timestamp': datetime.now().isoformat()})
    except Exception as e:
        logger.error(f"[LLM Router Status] Error: {e}")
        return jsonify({'success': False, 'error': str(e)})


@bp.route('/api/sota/llm-router/health', methods=['GET'])
def api_llm_router_health():
    """LLM 路由健康检查 API"""
    try:
        from modules.llm_router import llm_router
        provider = request.args.get('provider', None)
        health = llm_router.health_check(provider)
        return jsonify({'success': True, 'data': health, 'timestamp': datetime.now().isoformat()})
    except Exception as e:
        logger.error(f"[LLM Router Health] Error: {e}")
        return jsonify({'success': False, 'error': str(e)})


@bp.route('/api/sota/llm-router/fallback', methods=['POST'])
def api_llm_router_fallback():
    """LLM 降级决策 API"""
    try:
        from modules.llm_router import llm_router
        from modules.kline_signal_analyzer import KlineSignalAnalyzer
        stock_code = request.args.get('code', 'sz300620')

        analyzer = KlineSignalAnalyzer()
        klines = analyzer.get_klines(stock_code, period='daily', count=60)
        if not klines or len(klines) < 20:
            return jsonify({'success': False, 'error': 'K 线数据不足', 'timestamp': datetime.now().isoformat()}), 400

        closes = [float(k['close']) for k in klines]
        volumes = [float(k.get('volume', 0)) for k in klines]

        stock_data = {
            'rsi_14': closes[-1] / closes[-2] if len(closes) > 1 and closes[-2] > 0 else 50,
            'macd': 0, 'macd_signal': 0,
            'ma_5': np.mean(closes[-5:]) if len(closes) >= 5 else closes[-1],
            'ma_20': np.mean(closes[-20:]) if len(closes) >= 20 else closes[-1],
            'volume_ratio': volumes[-1] / np.mean(volumes[-5:]) if len(volumes) >= 5 and np.mean(volumes[-5:]) > 0 else 1.0,
            'change_pct': (closes[-1] / closes[-2] - 1) * 100 if len(closes) > 1 and closes[-2] > 0 else 0,
        }

        result = llm_router._rule_engine.decide(stock_data)
        return jsonify({
            'success': True, 'data': result, 'fallback': True,
            'stock_code': stock_code, 'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"[LLM Router Fallback] Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)})


# ============================================================================
# TimesFM
# ============================================================================

