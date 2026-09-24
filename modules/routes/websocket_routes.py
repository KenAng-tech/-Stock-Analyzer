"""WebSocket 事件处理器 — 从 app.py 提取"""

from flask_socketio import emit
from flask import request, jsonify
from modules.logger import logger
from modules.websocket_handler import WebSocketFundFlowHandler
from modules.sota_predict_pusher import SOTAPredictPusher

# Module-level placeholders — assigned by init_socketio()
websocket_handler = None
sota_predict_pusher = None


def init_socketio(socketio):
    """注册所有 WebSocket 事件处理器"""
    global websocket_handler, sota_predict_pusher
    websocket_handler = WebSocketFundFlowHandler(socketio)
    sota_predict_pusher = SOTAPredictPusher(socketio, interval=30)

    @socketio.on("connect")
    def handle_connect():
        """客户端连接"""
        logger.info(f"[WebSocket] Client connected: {request.sid}")

    @socketio.on("disconnect")
    def handle_disconnect():
        """客户端断开"""
        sid = request.sid
        websocket_handler.unsubscribe_all(sid)
        sota_predict_pusher.unsubscribe(sid)
        logger.info(f"[WebSocket] Client disconnected: {sid}")

    @socketio.on("subscribe_fund_flow")
    def handle_subscribe_fund_flow(data):
        """订阅资金流数据"""
        stock_code = data.get("stock_code", "")
        if stock_code:
            websocket_handler.subscribe(request.sid, stock_code)

    @socketio.on("unsubscribe_fund_flow")
    def handle_unsubscribe_fund_flow(data):
        """取消订阅资金流数据"""
        stock_code = data.get("stock_code", "")
        if stock_code:
            websocket_handler.unsubscribe(request.sid, stock_code)

    @socketio.on("subscribe_sota")
    def handle_subscribe_sota(data):
        """订阅 SOTA 预测更新"""
        stock_code = data.get("stock_code", "")
        if stock_code:
            sota_predict_pusher.subscribe(request.sid, stock_code)

    @socketio.on("unsubscribe_sota")
    def handle_unsubscribe_sota(data):
        """取消订阅 SOTA 预测更新"""
        stock_code = data.get("stock_code", "")
        if stock_code:
            sota_predict_pusher.unsubscribe(request.sid, stock_code)

    @socketio.on("get_sota_status")
    def handle_sota_status():
        """获取 SOTA 推送器状态"""
        return jsonify({
            "success": True,
            "data": sota_predict_pusher.get_status(),
        })
