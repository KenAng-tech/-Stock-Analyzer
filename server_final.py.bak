#!/usr/bin/env python3
"""
Stock Analyzer - Final Robust Server
Uses simple Flask + Werkzeug without eventlet
"""

import sys
import os
import signal
import atexit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Signal handling
def cleanup(sig, frame):
    print(f"\nReceived signal {sig}, shutting down...")
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

# Disable eventlet completely
os.environ['EVENTLET_NO_GREENDNS'] = '1'
os.environ['EVENTLET_NO_SELECT'] = '1'

# Import Flask and required modules
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import json
import time
from datetime import datetime

# Import stock analyzer modules
from modules.data_fetcher import StockDataFetcher
from modules.analysis_engine import AnalysisEngine
from modules.report_generator import ReportGenerator
from modules.strategy_engine import StrategyEngine
from modules.kline_signal_analyzer import KlineSignalAnalyzer
from modules.heatmap_generator import HeatmapGenerator
from modules.alert_engine import AlertEngine

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
heatmap_generator = HeatmapGenerator()
alert_engine = AlertEngine()

# Data cache
data_cache = {}
CACHE_TTL = 60

def get_stock_data(stock_code=None):
    """Get stock data with caching"""
    code = stock_code or 'sz300620'
    cache_key = f"stock_{code}"
    
    if cache_key in data_cache and time.time() - data_cache[cache_key]['time'] < CACHE_TTL:
        return data_cache[cache_key]['data']
    
    stock_data = data_fetcher.get_stock_info(code)
    if stock_data:
        data_cache[cache_key] = {'data': stock_data, 'time': time.time()}
    
    return stock_data

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/stock/<stock_code>')
def api_get_stock(stock_code):
    stock_data = get_stock_data(stock_code)
    if stock_data:
        return jsonify({'success': True, 'data': stock_data})
    return jsonify({'success': False, 'error': 'Stock data not found'}), 404

@app.route('/api/heatmap')
def api_get_heatmap():
    heatmap = heatmap_generator.generate_industry_heatmap()
    return jsonify({'success': True, 'data': heatmap})

@app.route('/api/alerts/<stock_code>')
def api_get_alerts(stock_code):
    alerts = alert_engine.get_recent_alerts(stock_code)
    summary = alert_engine.get_alert_summary(stock_code)
    return jsonify({'success': True, 'alerts': alerts, 'summary': summary})

@app.route('/api/presets')
def api_get_presets():
    presets = {
        'sz300620': {'code': 'sz300620', 'name': '光库科技', 'industry': '光通信', 'cost_basis': 120},
        'sh688313': {'code': 'sh688313', 'name': '仕佳光子', 'industry': '光通信', 'cost_basis': 85},
        'sz300308': {'code': 'sz300308', 'name': '中际旭创', 'industry': '光通信', 'cost_basis': 150},
        'sz002176': {'code': 'sz002176', 'name': '江特电机', 'industry': '锂电', 'cost_basis': 25},
        'sh600869': {'code': 'sh600869', 'name': '远东股份', 'industry': '电缆', 'cost_basis': 12},
    }
    preset_name = request.args.get('preset', 'sz300620')
    return jsonify(presets.get(preset_name, {}))

# Start server
if __name__ == '__main__':
    print("=" * 70)
    print("  Stock Analyzer System - FINAL SERVER")
    print("=" * 70)
    print(f"  Server: http://127.0.0.1:5002")
    print(f"  API: http://127.0.0.1:5002/api/stock/sz300620")
    print(f"  Mode: Flask + Werkzeug (NO eventlet)")
    print("=" * 70)
    
    app.run(host='0.0.0.0', port=5002, debug=False, threaded=True, use_reloader=False)
