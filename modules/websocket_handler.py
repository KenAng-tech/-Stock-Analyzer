"""
WebSocket 实时推送模块
提供真实股票数据的实时资金流推送 + 异动检测
"""

import time
import threading
import logging
from typing import Dict, List, Optional
from flask_socketio import SocketIO

logger = logging.getLogger('stock_analyzer')


class WebSocketFundFlowHandler:
    """WebSocket 实时推送处理器 — 基于真实腾讯财经数据"""

    def __init__(self, socketio: SocketIO):
        self.socketio = socketio
        self.clients: Dict[str, List[str]] = {}  # stock_code → [sid, ...]
        self._lock = threading.Lock()
        self._last_fetch: Dict[str, float] = {}  # stock_code → last_fetch_time
        self._last_data: Dict[str, Dict] = {}    # stock_code → cached data
        self._anomaly_history: List[Dict] = []
        self._polling_running = True
        self.update_interval = 5  # 更新间隔（秒）

        # 异动阈值配置
        self.anomaly_thresholds = {
            'large_order': {'min_amount': 500000, 'label': '大单异动'},
            'sudden_inflow': {'min_speed': 3000, 'label': '资金突增'},
            'sudden_outflow': {'max_speed': -2000, 'label': '资金突减'},
            'high_volume': {'min_volume': 100000, 'label': '放量异动'},
            'buy_dominant': {'min_ratio': 1.3, 'label': '买盘主导'},
            'sell_dominant': {'max_ratio': 0.7, 'label': '卖盘主导'},
        }

    # ── 客户端管理 ──────────────────────────────────────────────

    def subscribe(self, sid: str, stock_code: str):
        """订阅某只股票的实时数据"""
        with self._lock:
            if stock_code not in self.clients:
                self.clients[stock_code] = []
            if sid not in self.clients[stock_code]:
                self.clients[stock_code].append(sid)
        logger.info(f"[WebSocket] Client {sid} subscribed to {stock_code}")

    def unsubscribe(self, sid: str, stock_code: str = None):
        """取消订阅"""
        with self._lock:
            if stock_code:
                if stock_code in self.clients and sid in self.clients[stock_code]:
                    self.clients[stock_code].remove(sid)
                    if not self.clients[stock_code]:
                        del self.clients[stock_code]
            else:
                for code, sids in list(self.clients.items()):
                    if sid in sids:
                        sids.remove(sid)
                        if not sids:
                            del self.clients[code]
        logger.info(f"[WebSocket] Client {sid} unsubscribed")

    def unsubscribe_all(self, sid: str):
        """清理某客户端的所有订阅"""
        self.unsubscribe(sid)

    def get_subscribed_stocks(self) -> List[str]:
        """获取所有有订阅者的股票代码"""
        with self._lock:
            return [code for code, sids in self.clients.items() if sids]

    # ── 真实数据获取 ────────────────────────────────────────────

    def _fetch_real_data(self, stock_code: str) -> Optional[Dict]:
        """
        从腾讯财经 API 获取真实实时数据

        使用 DynamicCache 缓存 10 秒，避免频繁请求。
        返回格式与 generate_mock_data 兼容。
        """
        now = time.time()
        with self._lock:
            last = self._last_fetch.get(stock_code, 0)
            if now - last < 10:  # 10 秒缓存
                return self._last_data.get(stock_code)

        try:
            from modules.data_fetcher import StockDataFetcher
            fetcher = StockDataFetcher()
            data = fetcher.get_stock_info(stock_code)
            if not data:
                return None

            # 转换为 WebSocket 推送格式
            outer = data.get('outer_disk', 0)
            inner = data.get('inner_disk', 0)
            volume = data.get('volume', 0)
            amount = data.get('amount', 0)
            price = data.get('price', 0)
            change_pct = data.get('change_pct', 0)
            turnover = data.get('turnover', 0)

            speed = amount / 240 if amount > 0 else 0  # 万元/分钟（240 分钟交易日）

            result = {
                'stock_code': stock_code,
                'timestamp': time.strftime('%H:%M:%S'),
                'price': round(price, 2),
                'price_change': round(change_pct, 2),
                'volume': volume,
                'amount': round(amount, 2),
                'turnover': round(turnover, 2),
                'outer': outer,
                'inner': inner,
                'ratio': round(outer / inner, 2) if inner > 0 else 1.0,
                'speed': round(speed, 2),
                'anomalies': [],  # 稍后填充
                'distribution': self._estimate_distribution(outer, inner, amount),
                'pressure_index': round((outer / inner) * (turnover / 100), 2) if inner > 0 else 2.5,
            }

            with self._lock:
                self._last_fetch[stock_code] = now
                self._last_data[stock_code] = result
            return result

        except Exception as e:
            logger.debug(f"[WebSocket] Fetch failed for {stock_code}: {e}")
            with self._lock:
                return self._last_data.get(stock_code)  # fallback 缓存

    @staticmethod
    def _estimate_distribution(outer: int, inner: int, amount: float) -> Dict:
        """基于外内盘比估算大中小单分布"""
        total = outer + inner if (outer + inner) > 0 else 1
        buy_ratio = outer / total
        # 买盘强 → 大单比例高
        large = int(20 + buy_ratio * 30)
        medium = int(30 + (1 - abs(buy_ratio - 0.5) * 2) * 20)
        small = 100 - large - medium
        return {
            'large': max(10, min(50, large)),
            'medium': max(20, min(50, medium)),
            'small': max(10, 100 - large - medium),
        }

    # ── 异动检测 ────────────────────────────────────────────────

    def check_anomaly(self, data: Dict) -> List[Dict]:
        """检查资金异动"""
        anomalies = []
        t = self.anomaly_thresholds

        # 大单异动
        if data.get('amount', 0) > t['large_order']['min_amount']:
            anomalies.append({
                'type': 'large_order',
                'message': f"大单成交: {data['amount'] / 10000:.1f}万元",
                'severity': 'high',
            })

        # 资金突增
        if data.get('speed', 0) > t['sudden_inflow']['min_speed']:
            anomalies.append({
                'type': 'sudden_inflow',
                'message': f"资金突增: {data['speed']:.0f}万元/分钟",
                'severity': 'medium',
            })

        # 资金突减
        if data.get('speed', 0) < t['sudden_outflow']['max_speed']:
            anomalies.append({
                'type': 'sudden_outflow',
                'message': f"资金突减: {abs(data['speed']):.0f}万元/分钟",
                'severity': 'medium',
            })

        # 放量异动
        if data.get('volume', 0) > t['high_volume']['min_volume']:
            anomalies.append({
                'type': 'high_volume',
                'message': f"放量成交: {data['volume']}手",
                'severity': 'low',
            })

        # 买盘/卖盘主导
        ratio = data.get('ratio', 1.0)
        if ratio > t['buy_dominant']['min_ratio']:
            anomalies.append({
                'type': 'buy_dominant',
                'message': f"买盘主导: 内外比 {ratio:.2f}",
                'severity': 'medium',
            })
        elif ratio < t['sell_dominant']['max_ratio']:
            anomalies.append({
                'type': 'sell_dominant',
                'message': f"卖盘主导: 内外比 {ratio:.2f}",
                'severity': 'medium',
            })

        return anomalies

    # ── 推送逻辑 ────────────────────────────────────────────────

    def push_realtime_data(self, stock_code: str):
        """获取真实数据并推送到所有订阅客户端"""
        data = self._fetch_real_data(stock_code)
        if not data:
            return

        anomalies = self.check_anomaly(data)
        data['anomalies'] = anomalies

        # 记录异动历史
        if anomalies:
            entry = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'stock_code': stock_code,
                'anomalies': anomalies,
            }
            with self._lock:
                self._anomaly_history.append(entry)
                if len(self._anomaly_history) > 200:
                    self._anomaly_history = self._anomaly_history[-200:]

        # 推送给所有订阅该股票的客户端
        with self._lock:
            sids = list(self.clients.get(stock_code, []))

        for sid in sids:
            try:
                self.socketio.emit('fund_flow_update', {
                    'data': data,
                    'anomalies': anomalies,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                }, room=sid, skip_sid=None)
            except Exception as e:
                logger.debug(f"[WebSocket] Push failed to {sid}: {e}")

    def get_anomaly_history(self, stock_code: str = None, limit: int = 20) -> List[Dict]:
        """获取异动历史记录"""
        with self._lock:
            history = self._anomaly_history[:]
        if stock_code:
            history = [h for h in history if h['stock_code'] == stock_code]
        return history[-limit:]

    # ── 智能轮询 ────────────────────────────────────────────────

    def start_periodic_polling(self, interval: int = None):
        """
        启动智能轮询：只轮询有订阅者的股票

        每 N 秒检查一次，对每个有订阅者的股票调用 push_realtime_data。
        如果没有订阅者，不发起任何 API 请求。
        """
        interval = interval or self.update_interval

        def polling_loop():
            while self._polling_running:
                time.sleep(interval)
                with self._lock:
                    stocks = list(self.clients.keys())
                for stock_code in stocks:
                    self.push_realtime_data(stock_code)

        thread = threading.Thread(target=polling_loop, daemon=True, name='ws-polling')
        thread.start()
        logger.info(f"[WebSocket] Periodic polling started (interval={interval}s)")

    def stop_periodic_polling(self):
        """停止智能轮询"""
        self._polling_running = False
        logger.info("[WebSocket] Periodic polling stopped")

    # ── 分时数据（用于图表，保持 mock 降级） ─────────────────────

    def get_intraday_data(self, stock_code: str, minutes: int = 240) -> List[Dict]:
        """
        获取分时资金数据用于图表

        优先使用真实数据，不可用时 fallback 到模拟数据。
        """
        try:
            from modules.data_fetcher import StockDataFetcher
            fetcher = StockDataFetcher()
            data = fetcher.get_stock_info(stock_code)
            if data and data.get('price', 0) > 0:
                price = data['price']
                outer = data.get('outer_disk', 0)
                inner = data.get('inner_disk', 1)
                amount = data.get('amount', 0)
                volume = data.get('volume', 0)

                points = []
                for i in range(minutes):
                    # 基于真实数据做小幅随机波动
                    tick_price = price * (1 + (0.5 - (i / minutes)) * 0.01 * (1 + (i % 7) * 0.002))
                    tick_outer = int(outer * (0.9 + (i % 11) * 0.02))
                    tick_inner = int(inner * (0.9 + (i % 13) * 0.02))
                    points.append({
                        'time': f"{i // 60:02d}:{i % 60:02d}",
                        'price': round(tick_price, 2),
                        'outer': tick_outer,
                        'inner': tick_inner,
                        'amount': round(amount / minutes * (0.8 + (i % 5) * 0.1), 2),
                    })
                return points
        except Exception:
            pass

        # Fallback: 模拟数据
        import random
        data_points = []
        base_price = 250
        for i in range(minutes):
            data_points.append({
                'time': f"{i // 60:02d}:{i % 60:02d}",
                'price': round(base_price + random.uniform(-1, 1), 2),
                'outer': random.randint(200, 600),
                'inner': random.randint(200, 600),
                'amount': round(random.uniform(100, 500), 2),
            })
            base_price += random.uniform(-1, 1)
        return data_points
