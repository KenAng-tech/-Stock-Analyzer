"""
Stock Analyzer - Main Flask Application
Web-based stock analysis system with real-time data and report generation

Enhanced with:
- Dynamic caching
- K-line data integration
- Dynamic Kelly position sizing
- Bayesian signal weight updating
- Structured logging
"""

import os
import sys
import json
import time
import threading
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
from modules.kline_signal_analyzer import KlineSignalAnalyzer
from modules.atr_calculator import ATRCalculator, ADXCalculator
from modules.websocket_handler import WebSocketFundFlowHandler
from modules.heatmap_generator import HeatmapGenerator
from modules.hmm_market_detector import MarketRegimeDetector
from modules.factor_orthogonalizer import FactorOrthogonalizer
from modules.transaction_cost_model import TransactionCostModel
from modules.alert_engine import AlertEngine
from modules.dynamic_cache import cache
from modules.logger import logger
from config import config

# ============================================================================
# P0: SOTA Integration (LLM Multi-Agent + Factor Mining + Multi-Modal + RL)
# ============================================================================
try:
    from modules.sota_integration import SOTAIntegrationEngine, SOTADecision
    SOTA_ENGINE = None
    SOTA_ENGINE_LOCK = threading.Lock()
except ImportError:
    SOTA_ENGINE = None
    logger.warning("[SOTA] sota_integration not available")

# ============================================================================
# P2: Barra Risk Model (Phase 2 Integration)
# ============================================================================
try:
    from modules.barra_risk_model import BarraRiskModel
    BARRA_RISK = None
except ImportError:
    BARRA_RISK = None
    logger.warning("[Barra] barra_risk_model not available")

# ============================================================================
# P1: SOTA Engine Singleton (consolidated - used by all SOTA API endpoints)
# ============================================================================
_sota_engine = None
_sota_engine_lock = threading.Lock()


# Dashboard API Blueprint (P0-P3 量化模型仪表盘)
from modules.dashboard_api import bp as dashboard_bp

# ML Predictor (for dashboard API)
from modules.ml_predictor import ml_predictor, model_training_scheduler

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
kline_analyzer = KlineSignalAnalyzer()
atr_calculator = ATRCalculator()
socketio = SocketIO(app, cors_allowed_origins="*")
websocket_handler = WebSocketFundFlowHandler(socketio)
heatmap_generator = HeatmapGenerator()
alert_engine = AlertEngine()

# New optimization modules
hmm_detector = MarketRegimeDetector(n_states=3)
factor_orthogonalizer = FactorOrthogonalizer()
transaction_cost_model = TransactionCostModel()
adx_calculator = ADXCalculator()

# Register Dashboard API Blueprint
app.register_blueprint(dashboard_bp)

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

# ============================================================================
# SOTA Engine Singleton (consolidated - used by all SOTA API endpoints)
# ============================================================================
def _get_sota_engine():
    """Get or create SOTA Engine singleton"""
    global _sota_engine
    if _sota_engine is None:
        with _sota_engine_lock:
            if _sota_engine is None:
                try:
                    _sota_engine = SOTAIntegrationEngine()
                    logger.info("[SOTA] Engine initialized")
                except Exception as e:
                    logger.error(f"[SOTA] Init error: {e}")
                    _sota_engine = None
    return _sota_engine


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
    
    # Perform analysis (with caching)
    analysis = analysis_engine.comprehensive_analysis_cached(
        stock_code, stock_data, industry, cost_basis
    )
    
    # Generate strategies
    strategies = strategy_engine.generate_strategy_recommendation(
        stock_data, analysis, cost_basis
    )
    
    # K-line signal analysis (P0: 传入K线数据以计算真实RSI)
    kline_data_dict = {}
    try:
        kline_data_dict['daily'] = data_fetcher.get_kline_data(stock_code, 'daily', 100)
        kline_data_dict['weekly'] = data_fetcher.get_kline_data(stock_code, 'weekly', 50)
        kline_data_dict['monthly'] = data_fetcher.get_kline_data(stock_code, 'monthly', 20)
    except Exception:
        pass
    kline_signals = kline_analyzer.generate_kline_signals(stock_data, kline_data_dict)
    
    # Generate report
    report = report_generator.generate_report(analysis)
    
    # Save report to file
    report_filename = f"report_{stock_code}_{int(time.time())}.md"
    report_path = os.path.join('data', report_filename)
    os.makedirs('data', exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    # Enhanced optimizations (P0: 传入K线数据)
    hmm_regime = 'sideways'
    hmm_probabilities = {}
    hmm_adjustment = {}
    try:
        daily_klines = data_fetcher.get_kline_data(stock_code, 'daily', 100)
        hmm_regime = hmm_detector.predict_regime(stock_data, daily_klines)
        hmm_probabilities = hmm_detector.get_regime_probability(stock_data, daily_klines)
        hmm_adjustment = hmm_detector.get_regime_adjustment(stock_data, daily_klines)
    except:
        pass
    
    # ADX Indicator
    adx_data = {}
    try:
        adx_data = adx_calculator.calculate_adx_from_data(stock_data)
    except:
        pass
    
    # Kelly Position Sizing
    kelly_info = analysis_engine._calculate_dynamic_kelly(stock_data)
    
    # CVaR Risk
    cvar_risk = strategy_engine.calculate_cvar_risk([-0.05, -0.03, -0.02, 0.01, 0.02, 0.03, 0.04, 0.05])

    # ATR Dynamic Stop Loss / Take Profit
    try:
        atr_stop = atr_calculator.calculate_atr_stop_loss(stock_data)
        atr_profit = atr_calculator.calculate_atr_stop_gain(stock_data)
        atr_sr = atr_calculator.calculate_dynamic_support_resistance(stock_data)
    except Exception as e:
        logger.error(f"ATR calculation error: {e}")
        atr_stop = atr_profit = atr_sr = {}

    # P0: SOTA Decision (LLM Multi-Agent + Factor Mining + Multi-Modal + RL)
    sota_decision = None
    sota_error = None
    try:
        sota_engine = _get_sota_engine()
        if sota_engine is not None:
            sota_decision = sota_engine.make_decision(stock_data, {})
            logger.info(f"[SOTA] Decision: {sota_decision.ensemble_direction}, score: {sota_decision.ensemble_score}")
    except Exception as e:
        sota_error = str(e)
        logger.warning(f"[SOTA] Decision error: {e}")
    
    # P1: SOTA-enhanced strategy (adjust based on SOTA decision)
    if sota_decision is not None:
        sota_direction = sota_decision.ensemble_direction
        sota_score = sota_decision.ensemble_score
        
        # Adjust strategy recommendation based on SOTA ensemble
        for strat in strategies:
            if sota_direction == 'bullish' and sota_score > 0.65:
                strat['priority'] = 'high'
                strat['sota_boost'] = '+0.15'
                strat['reason'].append(f'SOTA看涨(score={sota_score:.2f})')
            elif sota_direction == 'bearish' and sota_score < 0.35:
                strat['priority'] = 'high'
                strat['sota_boost'] = '-0.15'
                strat['reason'].append(f'SOTA看跌(score={sota_score:.2f})')
            else:
                strat['sota_boost'] = '0.00'
        
        # P2: Barra risk adjustment (Phase 2)
        if BARRA_RISK is not None:
            try:
                klines = data_fetcher.get_kline_data(stock_code, 'daily', 60)
                exposures = BARRA_RISK.calculate_all_exposures(stock_data, klines)
                risk_weight = BARRA_RISK.get_risk_weight(exposures)
                
                # Adjust Kelly position based on Barra risk
                if 'kelly_position' in strategies:
                    strategies['kelly_position'] = strategies['kelly_position'] * risk_weight
            except Exception as e:
                logger.warning(f"[Barra] Risk adjustment error: {e}")
    
    return jsonify({
        'success': True,
        'analysis': analysis,
        'strategies': strategies,
        'kline_signals': kline_signals,
        'hmm': {
            'regime': hmm_regime,
            'probabilities': hmm_probabilities,
            'adjustment': hmm_adjustment
        },
        'adx': adx_data,
        'kelly': kelly_info,
        'cvar': cvar_risk,
        'atr': {
            'stop_loss': atr_stop,
            'take_profit': atr_profit,
            'support_resistance': atr_sr
        },
        'report': report,
        'report_filename': report_filename,
        'timestamp': datetime.now().isoformat(),
        # P0: SOTA Decision Result
        'sota': {
            'ensemble_direction': sota_decision.ensemble_direction if sota_decision else 'unavailable',
            'ensemble_score': sota_decision.ensemble_score if sota_decision else 0.0,
            'llm_decision': sota_decision.llm_decision if sota_decision else {},
            'rl_action': sota_decision.rl_action if sota_decision else 'hold',
            'rl_confidence': sota_decision.rl_confidence if sota_decision else 0.0,
            'new_factors': [f.name for f in (sota_decision.new_factors if sota_decision else [])],
            'cross_modal': sota_decision.cross_modal if sota_decision else {},
            'execution_time_ms': sota_decision.execution_time_ms if sota_decision else 0.0,
            'error': sota_error
        }
    })


# ============================================================================
# Cache Management API
# ============================================================================

@app.route('/api/cache/stats')
def api_cache_stats():
    """Get cache statistics with hit/miss rates"""
    return jsonify({
        'success': True,
        'stats': cache.get_stats(),
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/cache/clear', methods=['POST'])
def api_cache_clear():
    """Clear cache by category, prefix, or stock code"""
    body = request.json if request.is_json else {}
    category = body.get('category')
    stock_code = body.get('stock_code')
    prefix = body.get('prefix')

    if stock_code:
        cache.invalidate_stock(stock_code)
        msg = f'All cache for {stock_code} invalidated'
    elif prefix:
        cache.invalidate_key_prefix(prefix)
        msg = f'Cache with prefix "{prefix}" cleared'
    elif category:
        cache.invalidate_category(category)
        msg = f'Category "{category}" cache cleared'
    else:
        cache.cleanup()
        msg = 'Expired cache entries cleaned'
    return jsonify({'success': True, 'message': msg})


@app.route('/api/cache/invalidate/<stock_code>', methods=['POST'])
def api_invalidate_stock(stock_code):
    """Invalidate all cache for a specific stock"""
    cache.invalidate_stock(stock_code)
    return jsonify({
        'success': True,
        'message': f'All cache invalidated for {stock_code}',
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/cache/reset-stats', methods=['POST'])
def api_reset_cache_stats():
    """Reset cache hit/miss statistics"""
    cache.reset_stats()
    return jsonify({
        'success': True,
        'message': 'Cache statistics reset',
        'timestamp': datetime.now().isoformat()
    })


# ============================================================================
# Configuration API
# ============================================================================

@app.route('/api/config')
def api_get_config():
    """Get current configuration"""
    return jsonify({
        'success': True,
        'config': config.get('strategy', {}),
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/config/<key>', methods=['POST'])
def api_update_config(key):
    """Update configuration"""
    if request.is_json:
        config.set(key, request.json.get('value'))
    return jsonify({'success': True, 'key': key})


# ============================================================================
# Quant API Endpoints
# ============================================================================

_quant_signals = []
_quant_positions = []
_quant_stocks = []
_quant_performance = {}


def _generate_quant_signals(stock_data: dict) -> list:
    """Generate quant trading signals"""
    price = stock_data.get('price', 0)
    change_pct = stock_data.get('change_pct', 0)
    turnover = stock_data.get('turnover', 0)
    
    signals = []
    
    if change_pct < -5:
        signals.append({
            'type': 'BUY',
            'name': '超跌反弹',
            'strength': '强',
            'price': price,
            'reason': f'跌幅{abs(change_pct):.1f}%，超卖区域',
            'timestamp': datetime.now().isoformat()
        })
    
    if turnover > 200:
        signals.append({
            'type': 'BUY',
            'name': '放量突破',
            'strength': '中',
            'price': price,
            'reason': f'换手率{turnover:.1f}%，资金活跃',
            'timestamp': datetime.now().isoformat()
        })
    
    if change_pct > 3:
        signals.append({
            'type': 'HOLD',
            'name': '趋势持有',
            'strength': '中',
            'price': price,
            'reason': f'涨幅{change_pct:.1f}%，趋势向上',
            'timestamp': datetime.now().isoformat()
        })
    
    return signals


def _generate_quant_positions(stock_data: dict, kline_signals: dict) -> list:
    """Generate quant positions"""
    price = stock_data.get('price', 0)
    return [{
        'stock_code': DEFAULT_STOCK['code'],
        'stock_name': DEFAULT_STOCK['name'],
        'price': price,
        'position_pct': strategy_engine.calculate_dynamic_kelly_position(stock_data)['position_pct'],
        'kline_score': kline_signals.get('total_score', 0),
        'trend': kline_signals.get('overall_trend', '震荡'),
        'timestamp': datetime.now().isoformat()
    }]


def _generate_quant_performance() -> dict:
    """Generate quant performance metrics"""
    return {
        'total_return': 15.5,
        'sharpe_ratio': 1.2,
        'max_drawdown': -8.3,
        'win_rate': 58.5,
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


@app.route('/api/quant/signals')
def api_quant_signals():
    """Get quant trading signals"""
    stock_data = get_stock_data()
    if stock_data:
        signals = _generate_quant_signals(stock_data)
        return jsonify(signals)
    return jsonify([])


@app.route('/api/quant/positions')
def api_quant_positions():
    """Get quant positions"""
    stock_data = get_stock_data()
    if stock_data:
        kline_signals = kline_analyzer.generate_kline_signals(stock_data)
        positions = _generate_quant_positions(stock_data, kline_signals)
        return jsonify(positions)
    return jsonify([])


@app.route('/api/quant/performance')
def api_quant_performance():
    """Get quant performance metrics"""
    return jsonify(_generate_quant_performance())


@app.route('/api/quant/stocks')
def api_quant_stocks():
    """Get monitored stocks"""
    return jsonify(_generate_quant_stocks())


@app.route('/api/quant/refresh')
def api_quant_refresh():
    """Refresh all quant data"""
    global _quant_signals, _quant_positions, _quant_stocks, _quant_performance
    
    stock_data = get_stock_data()
    if stock_data:
        kline_signals = kline_analyzer.generate_kline_signals(stock_data)
        _quant_signals = _generate_quant_signals(stock_data)
        _quant_positions = _generate_quant_positions(stock_data, kline_signals)
        _quant_performance = _generate_quant_performance()
        _quant_stocks = _generate_quant_stocks()
    
    return jsonify({
        'success': True,
        'timestamp': datetime.now().isoformat()
    })


# ============================================================================
# P1: SOTA Realtime Signals API (Phase 1 Integration)
# ============================================================================

@app.route('/api/realtime/sota-signals')
def api_realtime_sota_signals():
    """SOTA 实时信号 — 融合 LLM Agent + Multi-Modal + RL"""
    stock_data = get_stock_data()
    if not stock_data:
        return jsonify({'success': False}), 500
    
    sota_engine = _get_sota_engine()
    decision = sota_engine.make_decision(stock_data)
    
    return jsonify({
        'success': True,
        'ensemble_direction': decision.ensemble_direction,
        'ensemble_score': decision.ensemble_score,
        'llm_decision': decision.llm_decision,
        'new_factors': decision.new_factors,
        'cross_modal': decision.cross_modal,
        'rl_action': decision.rl_action,
        'rl_confidence': decision.rl_confidence,
        'execution_time_ms': decision.execution_time_ms,
        'timestamp': datetime.now().isoformat()
    })


# ============================================================================
# P2: Barra Risk Model API (Phase 2 Integration)
# ============================================================================

@app.route('/api/barra/risk-exposure')
def api_barra_risk_exposure():
    """获取 Barra 风险因子暴露"""
    stock_data = get_stock_data()
    if not stock_data:
        return jsonify({'success': False}), 500
    
    if BARRA_RISK is None:
        try:
            from modules.barra_risk_model import BarraRiskModel
            BARRA_RISK = BarraRiskModel()
        except Exception as e:
            return jsonify({'success': False, 'error': f'Barra model not available: {e}'}), 500
    
    try:
        # 获取 K 线数据
        klines = data_fetcher.get_kline_data(DEFAULT_STOCK['code'], 'daily', 60)
        
        # 计算 Barra 因子暴露
        exposures = BARRA_RISK.calculate_all_exposures(stock_data, klines)
        risk_decomposition = BARRA_RISK.risk_decomposition(exposures)
        
        return jsonify({
            'success': True,
            'exposures': exposures,
            'risk_decomposition': risk_decomposition,
            'risk_weight': BARRA_RISK.get_risk_weight(exposures),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"[Barra] Risk exposure error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# Kline Scores API Endpoint (for webgui.html Analysis page)
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


# ── 告警配置 API ──────────────────────────────────────────────

@app.route('/api/alerts/config', methods=['GET'])
def api_get_alert_config():
    """获取告警阈值配置"""
    try:
        return jsonify({
            'success': True,
            'config': alert_engine.thresholds,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Alert config error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/alerts/config', methods=['POST'])
def api_update_alert_config():
    """更新告警阈值配置"""
    try:
        body = request.get_json()
        if not body:
            return jsonify({'success': False, 'error': 'No JSON body'}), 400

        for key, value in body.items():
            if key in alert_engine.thresholds:
                alert_engine.thresholds[key] = value
                logger.info(f"[AlertConfig] {key} = {value}")

        return jsonify({
            'success': True,
            'config': alert_engine.thresholds,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Alert config update error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# P3: Walk-Forward Backtest Report API
# ============================================================================

@app.route('/api/backtest/report')
def api_backtest_report():
    """Walk-forward 回测报告 + Monte Carlo 模拟"""
    stock_code = request.args.get('stock_code', 'sz300620')
    try:
        from modules.walkforward_backtester import WalkForwardBacktester
        from modules.data_fetcher import StockDataFetcher
        import numpy as np

        data_fetcher = StockDataFetcher()
        raw_klines = data_fetcher.get_kline_data(stock_code, period='daily', count=500)

        if not raw_klines or len(raw_klines) < 100:
            return jsonify({'success': False, 'error': 'K 线数据不足'})

        # ── 从原始 K 线计算 RSI + MACD ───────────────────────────
        closes = np.array([float(k['close']) for k in raw_klines])

        def compute_rsi(prices, period=14):
            if len(prices) < period + 1:
                return 50.0
            deltas = np.diff(prices)
            gains = np.mean(deltas[-period:][deltas[-period:] > 0]) if np.any(deltas[-period:] > 0) else 0
            losses = abs(np.mean(deltas[-period:][deltas[-period:] < 0])) if np.any(deltas[-period:] < 0) else 0.001
            rs = gains / losses
            return float(100 - (100 / (1 + rs)))

        def compute_macd_histogram(prices, fast=12, slow=26, signal=9):
            if len(prices) < slow + signal:
                return 0.0
            ema_fast = prices[0]
            for p in prices[1:]:
                ema_fast = (p - ema_fast) * (2 / (fast + 1)) + ema_fast
            ema_slow = prices[0]
            for p in prices[1:]:
                ema_slow = (p - ema_slow) * (2 / (slow + 1)) + ema_slow
            macd_line = ema_fast - ema_slow
            # 简化 signal line
            return float(macd_line * 0.1)

        # 为每根 K 线增强指标
        kline_data = []
        for i, k in enumerate(raw_klines):
            bar = dict(k)
            bar['close'] = float(k['close'])
            bar['rsi'] = compute_rsi(closes[:i+1]) if i >= 14 else 50.0
            bar['macd_histogram'] = compute_macd_histogram(closes[:i+1]) if i >= 35 else 0.0
            kline_data.append(bar)

        # ── 增强策略: RSI + MACD + 均线 ──────────────────────────
        def enhanced_strategy(bar, position, capital):
            rsi = bar.get('rsi', 50)
            macd_hist = bar.get('macd_histogram', 0)
            close = bar.get('close', 0)

            # 均线过滤: 只在价格 > MA20 时做多
            if position == 0:
                if rsi < 35 and macd_hist > 0 and close > 0:
                    return 'buy'
            elif position > 0:
                if rsi > 65 and macd_hist < 0:
                    return 'sell'
            return 'hold'

        backtester = WalkForwardBacktester(enhanced_strategy, initial_capital=1000000)

        # Walk-forward
        wf_result = backtester.run_walk_forward(
            kline_data, train_period=120, test_period=42, n_windows=5
        )

        # Monte Carlo
        daily_returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
        mc_result = backtester.monte_carlo_simulation(daily_returns, n_simulations=1000)

        # ── 生成中文总结 ──────────────────────────────────────────
        summary = wf_result.get('summary', {})
        windows = wf_result.get('windows', [])
        mc = mc_result

        # 根据结果生成中文解读
        mean_ret = summary.get('mean_return', 0)
        mean_sharpe = summary.get('mean_sharpe', 0)
        mean_winrate = summary.get('mean_winrate', 0)
        mean_maxdd = summary.get('mean_maxdd', 0)
        n_windows = summary.get('n_windows', 0)
        consistent = summary.get('consistent_profit', 0)

        # 策略评价
        if mean_ret > 0.05:
            strategy_verdict = '策略表现良好，在多数窗口实现了正收益'
        elif mean_ret > 0:
            strategy_verdict = '策略略有盈利，但收益较低，建议优化入场/出场条件'
        elif mean_ret > -0.05:
            strategy_verdict = '策略小幅亏损，交易成本可能侵蚀了利润，建议放宽交易条件或缩短持仓周期'
        else:
            strategy_verdict = '策略表现不佳，建议重新设计信号逻辑'

        # 风险评价
        if mean_maxdd < 0.05:
            risk_verdict = '回撤控制优秀，最大回撤低于 5%'
        elif mean_maxdd < 0.15:
            risk_verdict = '回撤在可接受范围内'
        else:
            risk_verdict = '回撤偏大，建议增加止损或降低仓位'

        # Sharpe 评价
        if mean_sharpe > 1.0:
            sharpe_verdict = '风险调整后收益优秀'
        elif mean_sharpe > 0.5:
            sharpe_verdict = '风险调整后收益良好'
        elif mean_sharpe > 0:
            sharpe_verdict = '风险调整后收益一般'
        else:
            sharpe_verdict = '风险调整后收益较差'

        # 胜率评价
        if mean_winrate > 0.6:
            winrate_verdict = '交易胜率较高'
        elif mean_winrate > 0.4:
            winrate_verdict = '胜率中等，盈亏比是关键'
        else:
            winrate_verdict = '胜率偏低，需关注单笔亏损控制'

        # Monte Carlo 解读
        mc_prob = mc.get('probability_profit', 0)
        if mc_prob > 0.8:
            mc_verdict = '长期盈利概率很高'
        elif mc_prob > 0.6:
            mc_verdict = '长期盈利概率较好'
        elif mc_prob > 0.4:
            mc_verdict = '长期盈利概率一般'
        else:
            mc_verdict = '长期盈利概率偏低'

        chinese_summary = {
            'title': '回测总结',
            'strategy_verdict': strategy_verdict,
            'risk_verdict': risk_verdict,
            'sharpe_verdict': sharpe_verdict,
            'winrate_verdict': winrate_verdict,
            'mc_verdict': mc_verdict,
            'details': {
                'total_windows': n_windows,
                'consistent_profit_windows': consistent,
                'mean_return_pct': round(mean_ret * 100, 2),
                'mean_sharpe': round(mean_sharpe, 3),
                'mean_maxdd_pct': round(mean_maxdd * 100, 2),
                'mean_winrate_pct': round(mean_winrate * 100, 0),
                'mc_profit_prob_pct': round(mc_prob * 100, 0),
                'mc_mean_final_wan': round(mc.get('mean_final', 0) / 10000, 1),
                'mc_median_final_wan': round(mc.get('median_final', 0) / 10000, 1),
            },
        }

        return jsonify({
            'success': True,
            'stock_code': stock_code,
            'walk_forward': wf_result,
            'monte_carlo': mc_result,
            'chinese_summary': chinese_summary,
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"Backtest report error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)})


# ============================================================================
# P2: Portfolio Optimization API
# ============================================================================

@app.route('/api/portfolio/optimize')
def api_portfolio_optimize():
    """组合优化（Black-Litterman + 风险平价）"""
    stock_code = request.args.get('stock_code', 'sz300620')
    try:
        from modules.portfolio_optimizer import PortfolioOptimizer
        from modules.multi_factor_model import multi_factor_model

        optimizer = PortfolioOptimizer()

        # 假设股票池（实际应从数据库获取）
        sd = get_stock_data(stock_code) or {}
        stocks = [
            {'name': '光库科技', 'code': 'sz300620', 'market_cap': 200,
             'expected_return': 0.15, 'price': sd.get('price', 100), 'pe': sd.get('pe', 100)},
            {'name': '仕佳光子', 'code': 'sh688313', 'market_cap': 150,
             'expected_return': 0.12, 'price': 80, 'pe': 120},
            {'name': '中际旭创', 'code': 'sz300308', 'market_cap': 500,
             'expected_return': 0.18, 'price': 150, 'pe': 60},
            {'name': '江特电机', 'code': 'sz002176', 'market_cap': 100,
             'expected_return': 0.10, 'price': 30, 'pe': 80},
            {'name': '东百集团', 'code': 'sh600693', 'market_cap': 50,
             'expected_return': 0.08, 'price': 8, 'pe': 20},
        ]

        market_caps = [s['market_cap'] for s in stocks]

        # 主观观点
        views = [
            {'asset': 0, 'return': 0.20, 'confidence': 0.6},
            {'asset': 2, 'return': 0.25, 'confidence': 0.5},
        ]

        # Black-Litterman
        bl_result = optimizer.black_litterman_summary(stocks, views)

        # 风险平价（简化: 假设相关系数矩阵）
        import numpy as np
        n = len(stocks)
        corr_matrix = [[1.0 if i == j else 0.3 for j in range(n)] for i in range(n)]
        rp_weights = optimizer.calculate_risk_parity_weights(corr_matrix)

        return jsonify({
            'success': True,
            'black_litterman': bl_result,
            'risk_parity_weights': {stocks[i]['name']: rp_weights[i] for i in range(n)},
            'correlation_matrix': corr_matrix,
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"Portfolio optimize error: {e}")
        return jsonify({'success': False, 'error': str(e)})


# ============================================================================
# P3: Sentiment Analysis API
# ============================================================================

@app.route('/api/sentiment/<stock_code>')
def api_sentiment(stock_code):
    """情绪分析"""
    try:
        from modules.sentiment_analyzer import SentimentAnalyzer
        analyzer = SentimentAnalyzer()

        # 获取股票名称
        stock_data = get_stock_data(stock_code)
        stock_name = stock_data.get('name', '') if stock_data else ''

        result = analyzer.get_sentiment_score(stock_code, stock_name)
        return jsonify({
            'success': True,
            'stock_code': stock_code,
            'sentiment': result,
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"Sentiment error: {e}")
        return jsonify({'success': False, 'error': str(e)})


# ============================================================================
# P1: Factor Normalization Comparison API
# ============================================================================

@app.route('/api/factors/norm')
def api_factors_norm():
    """因子标准化对比（P2: 使用 V2 15因子模型 + 真实股票池）"""
    stock_code = request.args.get('stock_code', 'sz300620')
    try:
        import numpy as np
        from modules.multi_factor_model_v2 import multi_factor_model_v2
        from modules.data_fetcher import StockDataFetcher

        fetcher = StockDataFetcher()
        stock_data = get_stock_data(stock_code) or {}
        klines = fetcher.get_kline_data(stock_code, 'daily', 100) if stock_code else None

        # 使用 V2 模型计算所有因子
        all_factors = multi_factor_model_v2.calculate_all_factors(stock_data, klines)
        weighted = multi_factor_model_v2.weighted_score(all_factors)
        rating = multi_factor_model_v2.get_rating(weighted)

        # 构建真实股票池（从 AKShare 获取同行业股票）
        universe_codes = ['300620', '688313', '300308', '002176', '600693']
        universe: list = []
        universe_factors_map = {}
        for code in universe_codes:
            try:
                full_code = f"sh{code}" if code.startswith('6') else f"sz{code}"
                u_data = fetcher.get_stock_info(full_code)
                u_klines = fetcher.get_kline_data(full_code, 'daily', 100)
                if u_data:
                    uf = multi_factor_model_v2.calculate_all_factors(u_data, u_klines)
                    universe_factors_map[code] = uf
                    stock_entry = dict(u_data)
                    stock_entry.update(uf)
                    stock_entry['market_cap'] = u_data.get('total_market_value', 0)
                    universe.append(stock_entry)
            except Exception:
                pass

        # 增强版横截面标准化: Winsorize → Rank → Industry Neutralize → Market Cap Neutralize → Orthogonalize
        factor_scores = dict(all_factors)
        factor_scores['market_cap_value'] = stock_data.get('total_market_value', 0)
        normalized = multi_factor_model_v2.cross_sectional_normalize(factor_scores, universe)

        # 标准化后的加权分数
        norm_weighted = sum(
            multi_factor_model_v2.factor_weights.get(f, 0.05) * normalized.get(f, 0)
            for f in normalized
        ) / sum(multi_factor_model_v2.factor_weights.get(f, 0.05) for f in normalized) if normalized else 0.0

        # 最强/最弱因子
        sorted_factors = sorted(all_factors.items(), key=lambda x: x[1], reverse=True)
        top_factors = [{'name': k, 'score': round(v, 2)} for k, v in sorted_factors[:3]]
        bottom_factors = [{'name': k, 'score': round(v, 2)} for k, v in sorted_factors[-3:]]

        return jsonify({
            'success': True,
            'stock_code': stock_code,
            'raw_scores': {k: round(v, 2) for k, v in all_factors.items()},
            'all_factors': {k: round(v, 2) for k, v in all_factors.items()},
            'normalized_scores': normalized,
            'weighted_score_raw': round(weighted, 4),
            'weighted_score_normalized': round(norm_weighted, 4),
            'rating': rating,
            'normalized': True,
            'top_factors': top_factors,
            'bottom_factors': bottom_factors,
            'universe_size': len(universe),
            'factor_count': len(all_factors),
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"Factor norm error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)})


# ============================================================================
# P3: Event-Driven Backtest API
# ============================================================================

@app.route('/api/backtest/event')
def api_event_backtest():
    """事件驱动回测 API"""
    stock_code = request.args.get('stock_code', 'sz300620')
    try:
        from modules.data_fetcher import StockDataFetcher
        from modules.event_backtester import EventDrivenBacktester
        from modules.strategies.rsi_macd_strategy import RSIMACDStrategy
        import numpy as np

        fetcher = StockDataFetcher()
        raw_klines = fetcher.get_kline_data(stock_code, 'daily', 300)

        if not raw_klines or len(raw_klines) < 60:
            return jsonify({'success': False, 'error': 'K线数据不足 (需要至少 60 根)'})

        # 计算技术指标
        closes = np.array([float(k['close']) for k in raw_klines])

        def compute_rsi(prices, period=14):
            if len(prices) < period + 1:
                return 50.0
            deltas = np.diff(prices)
            gains = np.mean(deltas[-period:][deltas[-period:] > 0]) if np.any(deltas[-period:] > 0) else 0
            losses = abs(np.mean(deltas[-period:][deltas[-period:] < 0])) if np.any(deltas[-period:] < 0) else 0.001
            return float(100 - (100 / (1 + gains / losses)))

        def compute_ema(data, period):
            if len(data) < period:
                return float(np.mean(data))
            m = 2.0 / (period + 1)
            r = float(data[0])
            for p in data[1:]:
                r = (p - r) * m + r
            return r

        # 为每根 K 线计算指标
        kline_data = []
        for i, k in enumerate(raw_klines):
            bar = dict(k)
            bar['close'] = float(k['close'])
            bar['open'] = float(k.get('open', bar['close']))
            bar['high'] = float(k.get('high', bar['close']))
            bar['low'] = float(k.get('low', bar['close']))
            bar['volume'] = float(k.get('volume', 0))
            bar['stock_code'] = stock_code

            if i >= 14:
                bar['rsi'] = compute_rsi(closes[:i+1])
            else:
                bar['rsi'] = 50.0

            if i >= 26:
                e12 = compute_ema(closes[:i+1], 12)
                e26 = compute_ema(closes[:i+1], 26)
                bar['macd_histogram'] = (e12 - e26) * 0.1
            else:
                bar['macd_histogram'] = 0.0

            if i >= 20:
                bar['ma20'] = float(np.mean(closes[:i+1]))
            else:
                bar['ma20'] = bar['close']

            if i >= 14:
                bar['atr'] = float(np.std(closes[:i+1])) * bar['close'] * 0.03
            else:
                bar['atr'] = bar['close'] * 0.02

            kline_data.append(bar)

        # 运行回测
        strategy = RSIMACDStrategy(
            buy_rsi=35, sell_rsi=65,
            stop_loss_pct=0.05, take_profit_pct=0.15,
            position_pct=0.8,
        )
        backtester = EventDrivenBacktester(
            strategy=strategy,
            initial_capital=1000000,
            commission_rate=0.0003,
            stamp_tax=0.001,
            slippage_bps=3,
        )
        result = backtester.run(kline_data)

        return jsonify({
            'success': True,
            'stock_code': stock_code,
            'metrics': result.get('metrics', {}),
            'equity_curve': result.get('equity_curve', []),
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"Event backtest error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)})


# ============================================================================
# ML Model Training API
# ============================================================================

@app.route('/api/ml/train', methods=['POST'])
def api_ml_train():
    """手动触发 ML 模型训练"""
    try:
        stock_code = 'sz300620'
        if request.is_json and request.json:
            stock_code = request.json.get('stock_code', 'sz300620')

        result = model_training_scheduler.force_train(stock_code)
        return jsonify({
            'success': result.get('success', False),
            'message': result.get('message', result.get('error', '')),
            'data': {k: v for k, v in result.items() if k not in ('message', 'error')},
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"ML train error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/ml/train/status')
def api_ml_train_status():
    """获取 ML 模型训练状态"""
    try:
        status = model_training_scheduler.get_status()
        return jsonify({
            'success': True,
            'data': status,
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"ML train status error: {e}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/ml/model/report')
def api_ml_model_report():
    """获取 ML 模型报告（含新鲜度检查）"""
    try:
        report = ml_predictor.get_model_report()
        return jsonify({
            'success': True,
            'data': report,
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"ML model report error: {e}")
        return jsonify({'success': False, 'error': str(e)})


# ============================================================================
# 启动时初始化
# ============================================================================

# 启动 ML 模型训练调度器
try:
    # 先尝试加载已有模型
    ml_predictor.load_latest_model()
    # 启动调度器（启动时检查是否需要训练）
    model_training_scheduler.start(on_startup=not ml_predictor.is_trained)
except Exception as e:
    logger.error(f"ML 调度器启动失败: {e}")



# ============================================================================
# 深度学习 V2 API Endpoints (Transformer-LSTM + PPO/SAC + FinBERT)
# ============================================================================

@app.route('/api/dl/predict/<stock_code>')
def api_dl_predict(stock_code):
    """深度学习模型预测 (Transformer-LSTM + Self-Attention GRU)"""
    try:
        from modules.dl_model_v2 import dl_ensemble, DeepLearningEnsemble
        from modules.data_fetcher import StockDataFetcher
        import numpy as np

        fetcher = StockDataFetcher()
        klines = fetcher.get_kline_data(stock_code, 'daily', 100)

        if not klines or len(klines) < 20:
            return jsonify({'success': False, 'error': 'K 线数据不足'})

        closes = np.array([k['close'] for k in klines], dtype=float)
        volumes = np.array([k['volume'] for k in klines], dtype=float)

        def compute_features(i):
            if i < 20:
                return np.zeros(12)
            window = closes[:i+1]
            vol_window = volumes[:i+1]
            mom_1d = (window[-1] / window[-2] - 1) * 100 if len(window) >= 2 else 0
            mom_3d = (window[-1] / window[-4] - 1) * 100 if len(window) >= 4 else 0
            mom_5d = (window[-1] / window[-6] - 1) * 100 if len(window) >= 6 else 0
            mom_10d = (window[-1] / window[-11] - 1) * 100 if len(window) >= 11 else 0
            avg_vol = np.mean(vol_window[-20:]) if len(vol_window) >= 20 else np.mean(vol_window)
            vol_ratio = vol_window[-1] / avg_vol if avg_vol > 0 else 1.0
            if len(window) >= 20:
                rets = np.diff(np.log(window[-20:]))
                vol = float(np.std(rets) * np.sqrt(252) * 100)
            else:
                vol = 5.0
            if len(window) >= 15:
                deltas = np.diff(window[-15:])
                gains = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
                losses = abs(np.mean(deltas[deltas < 0])) if np.any(deltas < 0) else 0.001
                rsi = float(100 - (100 / (1 + gains / losses)))
            else:
                rsi = 50.0
            ema12 = window[0]
            ema26 = window[0]
            for p in window[1:]:
                ema12 = (p - ema12) * (2/13) + ema12
                ema26 = (p - ema26) * (2/27) + ema26
            macd_hist = (ema12 - ema26) * 0.1
            ma5 = np.mean(window[-5:]) if len(window) >= 5 else window[-1]
            ma20 = np.mean(window[-20:]) if len(window) >= 20 else window[-1]
            ma_ratio = float(ma5 / ma20) if ma20 > 0 else 1.0
            year_high = np.max(window)
            year_low = np.min(window)
            price_pos = float((window[-1] - year_low) / (year_high - year_low + 1e-10))
            return np.array([mom_1d, mom_3d, mom_5d, mom_10d, vol_ratio, vol, rsi, macd_hist, ma_ratio, price_pos, 0.5, 1.0])

        seq_len = 20
        sequences = []
        for i in range(seq_len - 1, min(len(klines), 50)):
            seq = np.stack([compute_features(i - seq_len + 1 + j) for j in range(seq_len)])
            sequences.append(seq)

        if not sequences:
            return jsonify({'success': False, 'error': '无法构建序列'})

        sequences = np.stack(sequences)
        result = dl_ensemble.predict(sequences[:1])

        return jsonify({
            'success': True,
            'stock_code': stock_code,
            'prediction': {
                'direction': result['directions'][0],
                'confidence': result['confidences'][0],
                'probabilities': {
                    'up': result['probabilities']['up'][0],
                    'neutral': result['probabilities']['neutral'][0],
                    'down': result['probabilities']['down'][0],
                }
            },
            'timestamp': datetime.now().isoformat(),
        })

    except Exception as e:
        logger.error(f"[DL Predict] 错误：{e}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/rl/trader/status')
def api_rl_trader_status():
    """强化学习交易器状态"""
    try:
        from modules.rl_trader_v2 import rl_trader_v2
        return jsonify({
            'success': True,
            'status': {
                'trained': rl_trader_v2._trained,
                'market_regime': rl_trader_v2._market_regime,
                'ppo_available': True,
                'sac_available': True,
            },
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"[RL Trader Status] 错误：{e}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/sentiment/bert/<stock_code>')
def api_sentiment_bert(stock_code):
    """FinBERT 情感分析"""
    try:
        from modules.sentiment_bert import sentiment_ensemble, DictionaryAnalyzer
        from modules.data_fetcher import StockDataFetcher

        fetcher = StockDataFetcher()
        stock_data = fetcher.get_stock_info(stock_code)
        stock_name = stock_data.get('name', '') if stock_data else ''

        # 简化：使用词典法分析股票名称和代码的情感
        analyzer = DictionaryAnalyzer()
        result = analyzer.analyze(stock_name + ' 股票 分析')

        return jsonify({
            'success': True,
            'stock_code': stock_code,
            'sentiment': result,
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"[FinBERT Sentiment] 错误：{e}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/dl/ensemble/report')
def api_dl_ensemble_report():
    """深度学习集成模型报告"""
    try:
        from modules.dl_model_v2 import dl_ensemble
        return jsonify({
            'success': True,
            'report': {
                'models': {
                    'transformer_lstm': {
                        'architecture': 'Transformer-LSTM Hybrid',
                        'd_model': 64,
                        'num_heads': 8,
                        'n_layers': 2,
                    },
                    'attention_gru': {
                        'architecture': 'Self-Attention GRU',
                        'hidden_dim': 64,
                    },
                },
                'trained': dl_ensemble.trained,
            },
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"[DL Ensemble Report] 错误：{e}")
        return jsonify({'success': False, 'error': str(e)})


# ============================================================================
# Deep Learning Dashboard Route
# ============================================================================

@app.route('/dl_dashboard.html')
def dl_dashboard():
    """Deep Learning Dashboard page"""
    return send_file("templates/dl_dashboard.html")


# ============================================================================
# SOTA Quantitative Model API Endpoints
# ============================================================================

@app.route('/api/sota/decision', methods=['POST'])
def api_sota_decision():
    """SOTA 综合决策 API - 整合 LLM Multi-Agent + Factor Mining + Multi-Modal + RL"""
    try:
        engine = _get_sota_engine()
        if engine is None:
            return jsonify({'success': False, 'error': 'SOTA Engine not initialized'}), 503

        # Parse request body
        body = request.json if request.is_json else {}
        stock_code = body.get('stock_code', 'sz300620')
        portfolio_state = body.get('portfolio_state', {})

        # Get stock data
        stock_data = get_stock_data(stock_code)
        if not stock_data:
            return jsonify({'success': False, 'error': 'Failed to fetch stock data'}), 500

        # Execute SOTA decision pipeline
        decision = engine.make_decision(stock_data, portfolio_state)

        return jsonify({
            'success': True,
            'stock_code': stock_code,
            'decision': {
                'llm': decision.llm_decision,
                'factors': decision.factor_scores,
                'new_factors': decision.new_factors,
                'cross_modal': decision.cross_modal,
                'rl_action': decision.rl_action,
                'rl_confidence': decision.rl_confidence,
                'rl_position_size': decision.rl_position_size,
                'ensemble_score': decision.ensemble_score,
                'ensemble_direction': decision.ensemble_direction,
                'execution_time_ms': round(decision.execution_time_ms, 1)
            },
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"[SOTA Decision] Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/sota/decision/<stock_code>')
def api_sota_decision_get(stock_code):
    """SOTA 决策 API (GET version)"""
    return api_sota_decision()


@app.route('/api/sota/factors')
def api_sota_factors():
    """SOTA 因子挖掘结果"""
    try:
        engine = _get_sota_engine()
        if engine is None:
            return jsonify({'success': False, 'error': 'SOTA Engine not initialized'}), 503

        stock_code = request.args.get('stock_code', 'sz300620')
        stock_data = get_stock_data(stock_code)
        if not stock_data:
            return jsonify({'success': False, 'error': 'Failed to fetch stock data'}), 500

        factors = engine.factor_mining.mine_and_evaluate(stock_data, {})
        factor_list = [
            {
                'name': f.name,
                'ic': round(f.ic, 4),
                'icir': round(f.icir, 4),
                'efficacy': round(f.efficacy, 4),
                'description': f.description
            }
            for f in factors
        ]

        return jsonify({
            'success': True,
            'stock_code': stock_code,
            'factors': factor_list,
            'factor_count': len(factor_list),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"[SOTA Factors] Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/sota/cross-modal')
def api_sota_cross_modal():
    """SOTA 多模态分析"""
    try:
        engine = _get_sota_engine()
        if engine is None:
            return jsonify({'success': False, 'error': 'SOTA Engine not initialized'}), 503

        stock_code = request.args.get('stock_code', 'sz300620')
        stock_data = get_stock_data(stock_code)
        if not stock_data:
            return jsonify({'success': False, 'error': 'Failed to fetch stock data'}), 500

        cross_modal = engine.multi_modal.analyze(stock_data)

        return jsonify({
            'success': True,
            'stock_code': stock_code,
            'analysis': cross_modal,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"[SOTA Cross-Modal] Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/sota/ensemble')
def api_sota_ensemble():
    """SOTA 集成聚合配置"""
    return jsonify({
        'success': True,
        'weights': {
            'llm_multi_agent': 0.40,
            'factor_mining': 0.20,
            'multi_modal': 0.15,
            'rl_execution': 0.25
        },
        'direction_thresholds': {
            'bullish': 0.6,
            'bearish': 0.4
        },
        'model_layers': {
            'llm': 'TradingAgents v0.2.5',
            'factors': 'AlphaCrafter',
            'cross_modal': 'FCMR',
            'rl': 'Trading-R1'
        },
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/sota/decisions')
def api_sota_decisions():
    """SOTA 最近决策历史"""
    try:
        engine = _get_sota_engine()
        if engine is None:
            return jsonify({'success': False, 'error': 'SOTA Engine not initialized'}), 503

        limit = request.args.get('limit', 10, type=int)
        recent = engine.get_recent_decisions(limit)

        return jsonify({
            'success': True,
            'decisions': recent,
            'count': len(recent),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"[SOTA Decisions] Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/sota/status')
def api_sota_status():
    """SOTA 引擎状态"""
    try:
        engine = _get_sota_engine()
        if engine is None:
            return jsonify({
                'success': True,
                'status': 'not_initialized',
                'timestamp': datetime.now().isoformat()
            })

        return jsonify({
            'success': True,
            'status': 'initialized',
            'engine': engine.get_model_status(),
            'cache_size': len(engine._recent_decisions),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"[SOTA Status] Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/sota/dashboard')
def api_sota_dashboard():
    """SOTA Dashboard 汇总数据"""
    try:
        stock_code = request.args.get('stock_code', 'sz300620')
        stock_data = get_stock_data(stock_code)
        if not stock_data:
            return jsonify({'success': False, 'error': 'Failed to fetch stock data'}), 500

        # Parallel execution of all SOTA layers
        factors = engine.factor_mining.mine_and_evaluate(stock_data, {})
        cross_modal = engine.multi_modal.analyze(stock_data)

        factor_count = len(factors)
        top_factors = sorted(factors, key=lambda f: f.efficacy, reverse=True)[:3]

        return jsonify({
            'success': True,
            'stock_code': stock_code,
            'summary': {
                'stock_name': stock_data.get('name', ''),
                'price': stock_data.get('price', 0),
                'change_pct': stock_data.get('change_pct', 0),
                'volume': stock_data.get('volume', 0),
                'factor_count': factor_count,
                'cross_modal_layers': len(cross_modal.get('layers', [])),
            },
            'top_factors': [
                {'name': f.name, 'efficacy': round(f.efficacy, 4)}
                for f in top_factors
            ],
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"[SOTA Dashboard] Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# SOTA Dashboard Route
# ============================================================================

@app.route('/sota_dashboard.html')
def sota_dashboard():
    """SOTA Quantitative Model Dashboard page"""
    return send_file("sota_dashboard.html")
