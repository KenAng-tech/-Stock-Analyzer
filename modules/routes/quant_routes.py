"""
Quant Routes - /api/quant/*

Extracted from app.py (lines 1735-1923).
"""

from flask import Blueprint, request, jsonify
from datetime import datetime
import numpy as np

from modules.data_fetcher import StockDataFetcher
from modules.strategy_engine import StrategyEngine
from modules.logger import logger

bp = Blueprint('quant', __name__)

data_fetcher = StockDataFetcher()
strategy_engine = StrategyEngine()

DEFAULT_STOCK = {
    'code': 'sz300620',
    'name': '光库科技',
    'industry': '光通信',
    'cost_basis': 200.0,
}

_quant_signals = []
_quant_positions = []
_quant_stocks = []
_quant_performance = {}

# Data cache alias (mirrors app.py's data_cache usage)
data_cache = {}


def _generate_quant_signals(stock_data: dict) -> list:
    """Generate quant trading signals"""
    price = stock_data.get('price', 0)
    change_pct = stock_data.get('change_pct', 0)
    turnover = stock_data.get('turnover', 0)
    code = stock_data.get('code', DEFAULT_STOCK['code'])
    name = stock_data.get('name', DEFAULT_STOCK['name'])

    signals = []

    if change_pct < -5:
        signals.append({
            'symbol': code,
            'direction': 'BUY',
            'name': '超跌反弹',
            'strength': '强',
            'confidence': 0.75,
            'price': price,
            'reason': f'跌幅{abs(change_pct):.1f}%，超卖区域',
            'timestamp': datetime.now().isoformat()
        })

    if turnover > 200:
        signals.append({
            'symbol': code,
            'direction': 'BUY',
            'name': '放量突破',
            'strength': '中',
            'confidence': 0.65,
            'price': price,
            'reason': f'换手率{turnover:.1f}%，资金活跃',
            'timestamp': datetime.now().isoformat()
        })

    if change_pct > 3:
        signals.append({
            'symbol': code,
            'direction': 'HOLD',
            'name': '趋势持有',
            'strength': '中',
            'confidence': 0.60,
            'price': price,
            'reason': f'涨幅{change_pct:.1f}%，趋势向上',
            'timestamp': datetime.now().isoformat()
        })

    # If no signals generated, add a default one so the dashboard isn't empty
    if not signals:
        signals.append({
            'symbol': code,
            'direction': 'HOLD',
            'name': '观望',
            'strength': '弱',
            'confidence': 0.40,
            'price': price,
            'reason': '暂无明显信号',
            'timestamp': datetime.now().isoformat()
        })

    return signals


def _generate_quant_positions(stock_data: dict, kline_signals: dict) -> list:
    """Generate quant positions"""
    price = stock_data.get('price', 0)
    cost_basis = stock_data.get('cost_basis', 200.0)
    pnl = round(((price - cost_basis) / cost_basis * 100) if cost_basis > 0 else 0, 2)
    return [{
        'symbol': DEFAULT_STOCK['code'],
        'name': DEFAULT_STOCK['name'],
        'price': price,
        'unrealized_pnl': pnl,
        'side': 'LONG',
        'position_pct': strategy_engine.calculate_dynamic_kelly_position(stock_data)['position_pct'],
        'kline_score': kline_signals.get('total_score', 0),
        'trend': kline_signals.get('overall_trend', '震荡'),
        'timestamp': datetime.now().isoformat()
    }]


def _generate_quant_performance() -> dict:
    """Generate quant performance metrics"""
    return {
        'total_return': 0.155,       # 15.5% as decimal
        'sharpe_ratio': 1.2,
        'max_drawdown': -0.083,      # -8.3% as decimal
        'win_rate': 0.585,           # 58.5% as decimal
        'profit_factor': 1.8,
        'total_trades': 127,
        'timestamp': datetime.now().isoformat()
    }


def _generate_quant_stocks() -> list:
    """Generate monitored stocks"""
    return [{
        'code': DEFAULT_STOCK['code'],
        'name': DEFAULT_STOCK['name'],
        'price': data_cache.get(f"stock_{DEFAULT_STOCK['code']}", {}).get('data', {}).get('price', 0),
        'volume': data_cache.get(f"stock_{DEFAULT_STOCK['code']}", {}).get('data', {}).get('volume', 0),
        'sector': DEFAULT_STOCK.get('industry', ''),
        'timestamp': datetime.now().isoformat()
    }]


def get_stock_data(stock_code: str = None) -> dict:
    """Get stock data with dynamic caching (mirrors app.py)"""
    code = stock_code or DEFAULT_STOCK['code']
    from modules.dynamic_cache import cache
    cache_key = f"stock_{code}"
    cached = cache.get(cache_key, category='realtime')
    if cached:
        return cached
    stock_data = data_fetcher.get_stock_info(code)
    if stock_data:
        cache.set(cache_key, stock_data, category='realtime')
    return stock_data


@bp.route('/api/quant/signals')
def api_quant_signals():
    """Get quant trading signals"""
    stock_data = get_stock_data()
    if stock_data:
        signals = _generate_quant_signals(stock_data)
        return jsonify(signals)
    return jsonify([])


@bp.route('/api/quant/positions')
def api_quant_positions():
    """Get quant positions — 2026-09-09: 真实持仓优先 (portfolio_store), 无配置回退 mock"""
    # 1) 真实持仓 (~/.stock_analyzer_portfolio.json, 用户个人配置)
    try:
        from modules.portfolio_store import portfolio_store

        def _live_price(code):
            sd = get_stock_data(code) or {}
            return sd.get('price') or 0

        snap = portfolio_store.get_snapshot(price_lookup=_live_price)
        if snap and snap.get('positions'):
            return jsonify(snap['positions'])
    except Exception as e:
        logger.error(f"[QuantPositions] 真实持仓读取失败, 回退 mock: {e}")

    # 2) 回退: 原 DEFAULT_STOCK mock 单行 (保页面不断链)
    stock_data = get_stock_data()
    if stock_data:
        return jsonify(_generate_quant_positions(stock_data, {}))
    return jsonify([])


@bp.route('/api/quant/performance')
def api_quant_performance():
    """Get quant performance metrics"""
    return jsonify(_generate_quant_performance())


@bp.route('/api/quant/stocks')
def api_quant_stocks():
    """Get monitored stocks"""
    return jsonify(_generate_quant_stocks())


@bp.route('/api/quant/refresh')
def api_quant_refresh():
    """Refresh quant data"""
    global _quant_signals, _quant_positions, _quant_stocks, _quant_performance
    stock_data = get_stock_data()
    if stock_data:
        _quant_signals = _generate_quant_signals(stock_data)
        _quant_positions = _generate_quant_positions(stock_data, {})
        _quant_stocks = _generate_quant_stocks()
        _quant_performance = _generate_quant_performance()
    return jsonify({
        'success': True,
        'signals': _quant_signals,
        'positions': _quant_positions,
        'performance': _quant_performance,
        'stocks': _quant_stocks,
        'timestamp': datetime.now().isoformat()
    })


@bp.route('/api/quant/kline-scores')
def api_quant_kline_scores():
    """Get K-line scores for quant analysis"""
    stock_code = request.args.get('stock_code', DEFAULT_STOCK['code'])
    try:
        from modules.kline_signal_analyzer import KlineSignalAnalyzer
        kline_analyzer = KlineSignalAnalyzer()
        stock_data = get_stock_data(stock_code)
        if not stock_data:
            return jsonify({'success': False, 'error': 'Failed to fetch stock data'}), 404

        kline_data_dict = {}
        try:
            kline_data_dict['daily'] = data_fetcher.get_kline_data(stock_code, 'daily', 100)
            kline_data_dict['weekly'] = data_fetcher.get_kline_data(stock_code, 'weekly', 50)
        except Exception:
            pass

        signals = kline_analyzer.generate_kline_signals(stock_data, kline_data_dict)
        return jsonify({
            'success': True,
            'data': signals,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"K-line scores error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
