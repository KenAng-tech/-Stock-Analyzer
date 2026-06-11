"""
Stock Analyzer - Main Flask Application (Optimized V2)
Web-based stock analysis system with real-time data and report generation

Enhanced with:
- Dynamic caching
- K-line data integration
- Dynamic Kelly position sizing
- Bayesian signal weight updating
- ATR trend-aware stop loss
- RSI multi-cycle analysis
- Volatility target position sizing
- MACD + Bollinger Bands
- Sector rotation
- Comprehensive backtesting
- Structured logging
"""

import os
import sys
import json
import time
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
from flask_socketio import SocketIO
from flask_cors import CORS

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.data_fetcher import StockDataFetcher
from modules.analysis_engine import AnalysisEngine
from modules.report_generator import ReportGenerator
from modules.strategy_engine import StrategyEngine
from modules.strategy_engine_v2 import StrategyEngineV2, strategy_engine_v2
from modules.kline_signal_analyzer import KlineSignalAnalyzer
from modules.atr_calculator import ATRCalculator, ADXCalculator
from modules.atr_trend_filter import ATR_TrendFilter, atr_trend_filter
from modules.rsi_multicycle import RSI_MultiCycle, rsi_multicycle
from modules.volatility_target import VolatilityTarget, volatility_target
from modules.macd_bollinger import MACD_Bollinger, macd_bollinger
from modules.sector_rotation import SectorRotation, sector_rotation
from modules.kelly_optimizer import KellyOptimizer, kelly_optimizer
from modules.websocket_handler import WebSocketFundFlowHandler
from modules.heatmap_generator import HeatmapGenerator
from modules.hmm_market_detector import MarketRegimeDetector
from modules.factor_orthogonalizer import FactorOrthogonalizer
from modules.transaction_cost_model import TransactionCostModel
from modules.alert_engine import AlertEngine
from modules.backtester import Backtester
from modules.enhanced_backtester import EnhancedBacktester
from modules.dynamic_cache import cache
from modules.logger import logger
from config import config

# Initialize Flask app
app = Flask(__name__, 
            template_folder='templates',
            static_folder='static')
CORS(app)

# Initialize modules
data_fetcher = StockDataFetcher()
analysis_engine = AnalysisEngine()
report_generator = ReportGenerator()
strategy_engine = StrategyEngine()
strategy_engine_v2_instance = strategy_engine_v2
kline_analyzer = KlineSignalAnalyzer()
atr_calculator = ATRCalculator()
atr_trend = atr_trend_filter
rsi_analyzer = rsi_multicycle
vol_target = volatility_target
macd_bb = macd_bollinger
sector_rot = sector_rotation
kelly_opt = kelly_optimizer
socketio = SocketIO(app, cors_allowed_origins="*")
websocket_handler = WebSocketFundFlowHandler(socketio)
heatmap_generator = HeatmapGenerator()
alert_engine = AlertEngine()

# New optimization modules
hmm_detector = MarketRegimeDetector(n_states=3)
factor_orthogonalizer = FactorOrthogonalizer()
transaction_cost_model = TransactionCostModel()
adx_calculator = ADXCalculator()

# Default stock configuration
DEFAULT_STOCK = {
    'code': 'sz300620',
    'name': '光库科技',
    'industry': '光通信',
    'cost_basis': 120
}

# Data cache
data_cache = {}
CACHE_TTL = 60


def get_stock_data(stock_code: str = None) -> dict:
    """Get stock data with dynamic caching"""
    code = stock_code or DEFAULT_STOCK['code']
    cache_key = f"stock_{code}"
    
    # Try dynamic cache first
    cached = cache.get(cache_key, category='realtime')
    if cached:
        return cached
    
    stock_data = data_fetcher.get_stock_info(code)
    if stock_data:
        cache.set(cache_key, stock_data, category='realtime')
    
    return stock_data


@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route("/webgui.html")
def webgui():
    """Quant webgui page"""
    return send_file("webgui.html")


@app.route('/api/stock/<stock_code>')
def api_get_stock(stock_code):
    """Get stock data API"""
    stock_data = get_stock_data(stock_code)
    if stock_data:
        return jsonify({
            'success': True,
            'data': stock_data,
            'timestamp': datetime.now().isoformat()
        })
    return jsonify({'success': False, 'error': 'Failed to fetch stock data'}), 500


@app.route('/api/stock/enhanced/<stock_code>')
def api_get_enhanced_stock(stock_code):
    """Get enhanced stock data with K-line stats"""
    stock_data = data_fetcher.get_enhanced_stock_info(stock_code)
    if stock_data:
        return jsonify({
            'success': True,
            'data': stock_data,
            'timestamp': datetime.now().isoformat()
        })
    return jsonify({'success': False, 'error': 'Failed to fetch stock data'}), 500


@app.route('/api/analyze/<stock_code>')
def api_analyze_stock(stock_code):
    """Get comprehensive analysis"""
    stock_data = get_stock_data(stock_code)
    if not stock_data:
        return jsonify({'success': False, 'error': 'Failed to fetch stock data'}), 500
    
    industry = DEFAULT_STOCK.get('industry', '光通信')
    cost_basis = request.args.get('cost_basis', DEFAULT_STOCK['cost_basis'], type=float)
    
    # Perform analysis
    analysis = analysis_engine.comprehensive_analysis(stock_data, industry, cost_basis)
    
    # Generate strategies
    strategies = strategy_engine.generate_strategy_recommendation(
        stock_data, analysis, cost_basis
    )
    
    # Start mock alert generation
    alert_engine.start_mock_alert_generation(stock_code, interval=15)
    
    # K-line signal analysis
    kline_signals = kline_analyzer.generate_kline_signals(stock_data)
    
    # Generate report
    report = report_generator.generate_report(analysis)
    
    # Save report to file
    report_filename = f"report_{stock_code}_{int(time.time())}.md"
    report_path = os.path.join('data', report_filename)
    os.makedirs('data', exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(report)
    
    return jsonify({
        'success': True,
        'analysis': analysis,
        'strategies': strategies,
        'kline_signals': kline_signals,
        'report_path': report_filename,
        'timestamp': datetime.now().isoformat()
    })


# ============================================================================
# Enhanced Strategy API Endpoints (V2)
# ============================================================================

@app.route('/api/strategy/enhanced/<stock_code>')
def api_strategy_enhanced(stock_code):
    """Get enhanced strategy recommendation using V2 engine"""
    stock_data = get_stock_data(stock_code)
    if not stock_data:
        return jsonify({'success': False, 'error': 'Failed to fetch stock data'}), 500
    
    industry = DEFAULT_STOCK.get('industry', '光通信')
    cost_basis = request.args.get('cost_basis', DEFAULT_STOCK['cost_basis'], type=float)
    
    # Get K-line data
    klines = data_fetcher.get_kline_data(stock_code, 'daily', 60)
    
    # Enhanced strategy
    enhanced = strategy_engine_v2.generate_comprehensive_signal(
        stock_data, klines=klines, industry=industry
    )
    
    return jsonify({
        'success': True,
        'strategy': enhanced,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/strategy/kelly/<stock_code>')
def api_strategy_kelly(stock_code):
    """Get Kelly-based strategy recommendation"""
    stock_data = get_stock_data(stock_code)
    if not stock_data:
        return jsonify({'success': False, 'error': 'Failed to fetch stock data'}), 500
    
    kelly_result = kelly_opt.get_position_size(stock_data)
    kelly_params = kelly_opt.get_optimal_kelly_params()
    
    return jsonify({
        'success': True,
        'kelly_position': kelly_result,
        'kelly_params': kelly_params,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/strategy/atr/<stock_code>')
def api_strategy_atr(stock_code):
    """Get ATR trend-aware strategy"""
    stock_data = get_stock_data(stock_code)
    if not stock_data:
        return jsonify({'success': False, 'error': 'Failed to fetch stock data'}), 500
    
    klines = data_fetcher.get_kline_data(stock_code, 'daily', 60)
    atr_result = atr_trend.calculate_trend_aware_stop(stock_data, klines)
    adx_info = atr_trend.calculate_adx(klines)
    
    return jsonify({
        'success': True,
        'atr_stop': atr_result,
        'adx': adx_info,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/strategy/rsi/<stock_code>')
def api_strategy_rsi(stock_code):
    """Get RSI multi-cycle analysis"""
    stock_data = get_stock_data(stock_code)
    if not stock_data:
        return jsonify({'success': False, 'error': 'Failed to fetch stock data'}), 500
    
    klines = data_fetcher.get_kline_data(stock_code, 'daily', 60)
    if klines:
        prices = [k['close'] for k in klines]
        rsi_data = rsi_analyzer.calculate_all_periods(prices)
        rsi_summary = rsi_analyzer.get_multi_cycle_summary(rsi_data)
        rsi_signal = rsi_analyzer.get_rsi_signal(rsi_data.get('rsi_14', 50))
    else:
        rsi_data = {'rsi_14': 50, 'rsi_25': 50, 'rsi_60': 50}
        rsi_summary = {'trend': 'neutral', 'strength': 'weak'}
        rsi_signal = {'signal': 'neutral', 'action': 'hold'}
    
    return jsonify({
        'success': True,
        'rsi': {
            'values': rsi_data,
            'summary': rsi_summary,
            'signal': rsi_signal
        },
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/strategy/volatility/<stock_code>')
def api_strategy_volatility(stock_code):
    """Get volatility target position sizing"""
    stock_data = get_stock_data(stock_code)
    if not stock_data:
        return jsonify({'success': False, 'error': 'Failed to fetch stock data'}), 500
    
    vol_result = vol_target.calculate_target_position(stock_data)
    vol_trend = vol_target.get_volatility_trend()
    
    return jsonify({
        'success': True,
        'volatility': vol_result,
        'vol_trend': vol_trend,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/strategy/macd-bollinger/<stock_code>')
def api_strategy_macd_bb(stock_code):
    """Get MACD + Bollinger Bands strategy"""
    stock_data = get_stock_data(stock_code)
    if not stock_data:
        return jsonify({'success': False, 'error': 'Failed to fetch stock data'}), 500
    
    klines = data_fetcher.get_kline_data(stock_code, 'daily', 60)
    if klines:
        prices = [k['close'] for k in klines]
        macd_data = macd_bb.calculate_macd(prices)
        bb_data = macd_bb.calculate_bollinger_bands(prices)
        combined = macd_bb.get_combined_signal(macd_data, bb_data)
    else:
        macd_data = {'macd': 0, 'signal': 0, 'histogram': 0, 'trend': 'neutral'}
        bb_data = {'upper': 0, 'middle': 0, 'lower': 0, 'percent_b': 0.5}
        combined = {'combined_signal': 'HOLD', 'combined_strength': 'weak'}
    
    return jsonify({
        'success': True,
        'macd': macd_data,
        'bollinger': bb_data,
        'combined': combined,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/strategy/sector/<stock_code>')
def api_strategy_sector(stock_code):
    """Get sector rotation analysis"""
    stock_data = get_stock_data(stock_code)
    if not stock_data:
        return jsonify({'success': False, 'error': 'Failed to fetch stock data'}), 500
    
    industry = DEFAULT_STOCK.get('industry', '光通信')
    sector_result = sector_rot.get_sector_recommendation(stock_data, industry)
    momentum = sector_rot.calculate_sector_momentum(stock_data, industry)
    trend = sector_rot.get_sector_trend(stock_data, industry)
    
    return jsonify({
        'success': True,
        'sector': sector_result,
        'momentum': momentum,
        'trend': trend,
        'timestamp': datetime.now().isoformat()
    })


# ============================================================================
# Backtest API Endpoints
# ============================================================================

@app.route('/api/backtest/<stock_code>')
def api_backtest(stock_code):
    """Run backtest for stock"""
    stock_data = get_stock_data(stock_code)
    if not stock_data:
        return jsonify({'success': False, 'error': 'Failed to fetch stock data'}), 500
    
    klines = data_fetcher.get_kline_data(stock_code, 'daily', 250)
    if not klines:
        return jsonify({'success': False, 'error': 'Failed to fetch K-line data'}), 500
    
    # Run enhanced backtest
    bt = EnhancedBacktester()
    result = bt.run_walk_forward(klines, strategy_engine_v2._default_strategy)
    
    # Monte Carlo simulation
    if 'equity_curves' in result and result['equity_curves']:
        mc_result = bt.run_monte_carlo(result['equity_curves'][0], n_simulations=1000)
        result['monte_carlo'] = mc_result
    
    return jsonify({
        'success': True,
        'backtest': result,
        'timestamp': datetime.now().isoformat()
    })


# ============================================================================
# K-line Scores API Endpoint (for webgui.html)
# ============================================================================

@app.route('/api/quant/kline-scores')
def api_quant_kline_scores():
    """Get K-line scores: morphology, volume, position, RSI"""
    stock_data = get_stock_data()
    if not stock_data:
        return jsonify({'success': False, 'error': 'Failed to fetch stock data'}), 500
    
    kline_signals = kline_analyzer.generate_kline_signals(stock_data)
    technical = analysis_engine.technical_analysis(stock_data)
    
    response = {
        'success': True,
        'stock': {
            'code': stock_data.get('code', ''),
            'name': stock_data.get('name', ''),
            'price': stock_data.get('price', 0),
            'timestamp': stock_data.get('timestamp', '')
        },
        'scores': {
            'morphology_score': kline_signals.get('candlestick_patterns', []),
            'volume_score': kline_signals.get('volume_signals', []),
            'position_score': kline_signals.get('price_position', ''),
            'rsi_score': kline_signals.get('rsi', 0),
            'total_score': kline_signals.get('total_score', 0),
            'trend': kline_signals.get('overall_trend', ''),
            'trend_strength': kline_signals.get('trend_strength', ''),
            'bullish_signals': kline_signals.get('bullish_signals', []),
            'bearish_signals': kline_signals.get('bearish_signals', []),
            'multi_cycle': kline_signals.get('multi_cycle', {})
        },
        'technical': {
            'rsi': technical.get('indicators', {}).get('rsi', 0),
            'macd': technical.get('indicators', {}).get('macd', ''),
            'kdj': technical.get('indicators', {}).get('kdj', ''),
            'short_term_trend': technical.get('trend', {}).get('short_term', ''),
            'medium_term_trend': technical.get('trend', {}).get('medium_term', '')
        },
        'timestamp': datetime.now().isoformat()
    }
    
    return jsonify(response)


# ============================================================================
# ATR API Endpoint (for webgui.html)
# ============================================================================

@app.route('/api/atr/<stock_code>')
def api_get_atr(stock_code):
    """Get ATR-based analysis"""
    stock_data = get_stock_data(stock_code)
    if not stock_data:
        return jsonify({'success': False, 'error': 'Failed to fetch stock data'}), 500
    
    atr_stop = atr_calculator.calculate_atr_stop_loss(stock_data)
    atr_profit = atr_calculator.calculate_atr_stop_gain(stock_data)
    atr_support_resistance = atr_calculator.calculate_dynamic_support_resistance(stock_data)
    
    return jsonify({
        'success': True,
        'atr_stop_loss': atr_stop,
        'atr_take_profit': atr_profit,
        'atr_support_resistance': atr_support_resistance,
        'timestamp': datetime.now().isoformat()
    })


# ============================================================================
# Health Check & Additional API Endpoints (for webgui.html)
# ============================================================================

@app.route('/api/health')
def api_health():
    """Health check endpoint for webgui.html"""
    try:
        stock_data = get_stock_data()
        return jsonify({
            'success': True,
            'version': '2.0.0',
            'timestamp': datetime.now().isoformat(),
            'data_fresh': stock_data is not None,
            'cache_stats': cache.get_stats()
        })
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({
            'success': True,
            'version': '2.0.0',
            'timestamp': datetime.now().isoformat(),
            'data_fresh': False,
            'error': str(e)
        })


@app.route('/api/heatmap')
def api_get_heatmap():
    """Get industry heatmap data"""
    try:
        heatmap = heatmap_generator.generate_industry_heatmap()
        return jsonify({
            'success': True,
            'data': heatmap,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Heatmap error: {e}")
        return jsonify({
            'success': True,
            'data': [],
            'timestamp': datetime.now().isoformat()
        })


@app.route('/api/alerts/<stock_code>')
def api_get_alerts(stock_code):
    """Get alerts for a stock"""
    try:
        alerts = alert_engine.get_recent_alerts(stock_code)
        summary = alert_engine.get_alert_summary(stock_code)
        return jsonify({
            'success': True,
            'alerts': alerts,
            'summary': summary,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Alerts error: {e}")
        return jsonify({
            'success': True,
            'alerts': [],
            'summary': {},
            'timestamp': datetime.now().isoformat()
        })
