#!/usr/bin/env python3
"""
Stock Analyzer - Robust Server Startup
Uses Werkzeug instead of eventlet for reliable macOS operation
"""

import sys
import os
import signal
import atexit
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logger = logging.getLogger('stock_analyzer')

# Signal handling for clean shutdown
def signal_handler(sig, frame):
    print("\nShutting down server...")
    websocket_handler.stop_periodic_polling()
    alert_engine.stop_monitoring()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Disable eventlet completely
os.environ['EVENTLET_NO_GREENDNS'] = '1'
os.environ['EVENTLET_NO_SELECT'] = '1'

# Patch socketio before import
import flask_socketio
original_init = flask_socketio.SocketIO.__init__

def patched_init(self, app=None, *args, **kwargs):
    kwargs['async_mode'] = 'threading'
    kwargs['ping_timeout'] = 60
    kwargs['ping_interval'] = 25
    return original_init(self, app, *args, **kwargs)

flask_socketio.SocketIO.__init__ = patched_init

# Import app
from flask import request
import app as app_module
from app import app, data_fetcher, analysis_engine, report_generator
from modules.websocket_handler import WebSocketFundFlowHandler
from modules.heatmap_generator import HeatmapGenerator
from modules.alert_engine import AlertEngine
from modules.alert_storage import AlertStorage

# Reinitialize socketio with threading
socketio = flask_socketio.SocketIO(app, cors_allowed_origins="*", async_mode='threading')
# 同步更新 app 模块中的引用，确保 API 路由和动态导入都使用新实例
alert_storage = AlertStorage()
app_module.socketio = socketio
app_module.websocket_handler = WebSocketFundFlowHandler(socketio)
app_module.alert_engine = AlertEngine(storage=alert_storage)
websocket_handler = app_module.websocket_handler
alert_engine = app_module.alert_engine
heatmap_generator = HeatmapGenerator()

# ── SocketIO 事件处理器 ──────────────────────────────────────

@socketio.on('connect')
def handle_connect():
    """客户端连接"""
    logger.info(f"[WebSocket] Client connected: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    """客户端断开 — 自动清理所有订阅"""
    sid = request.sid
    websocket_handler.unsubscribe_all(sid)
    logger.info(f"[WebSocket] Client disconnected: {sid}")

@socketio.on('subscribe_stock')
def handle_subscribe(data):
    """订阅某只股票的实时数据"""
    sid = request.sid
    stock_code = data.get('stock_code', '')
    if not stock_code:
        return
    websocket_handler.subscribe(sid, stock_code)
    # 立即推送一次最新数据
    websocket_handler.push_realtime_data(stock_code)
    logger.info(f"[WebSocket] {sid} subscribed to {stock_code}")

@socketio.on('unsubscribe_stock')
def handle_unsubscribe(data):
    """取消订阅"""
    sid = request.sid
    stock_code = data.get('stock_code', '')
    if stock_code:
        websocket_handler.unsubscribe(sid, stock_code)
    else:
        websocket_handler.unsubscribe_all(sid)
    logger.info(f"[WebSocket] {sid} unsubscribed from {stock_code}")

# ── 启动监控线程 ─────────────────────────────────────────────

# WebSocket 智能轮询（只轮询有订阅者的股票）
websocket_handler.start_periodic_polling(interval=5)
logger.info("[AlertEngine] WebSocket 智能轮询已启动 (5s)")

# 告警实时监控
try:
    from modules.kline_signal_analyzer import KlineSignalAnalyzer
    kline_analyzer = KlineSignalAnalyzer()
except Exception:
    kline_analyzer = None

if kline_analyzer:
    alert_engine.start_realtime_monitoring(
        fetcher=data_fetcher,
        kline_analyzer=kline_analyzer,
        interval=30,
    )
    logger.info("[AlertEngine] 实时告警监控已启动 (30s)")
else:
    logger.warning("[AlertEngine] KlineSignalAnalyzer 加载失败，告警监控跳过")

# ── 启动服务器 ───────────────────────────────────────────────

# Log startup
print("=" * 70)
print("  Stock Analyzer System - Robust Startup")
print("=" * 70)
print(f"  Server: http://127.0.0.1:5002")
print(f"  API: http://127.0.0.1:5002/api/stock/sz300620")
print(f"  WebSocket: Enabled (threading mode)")
print(f"  Async Mode: threading (NOT eventlet)")
print(f"  WebSocket Polling: 5s (smart subscriber-based)")
print(f"  Alert Monitoring: 30s (SQLite persistence)")
print("=" * 70)

# Start server with Werkzeug
print("\nStarting server...")
try:
    socketio.run(app, host='0.0.0.0', port=5002, debug=False,
                 allow_unsafe_werkzeug=True, use_reloader=False)
except Exception as e:
    print(f"Error starting server: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
