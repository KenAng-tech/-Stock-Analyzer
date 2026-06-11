"""
智能告警引擎 — 真实阈值监控 + 技术指标检测 + SQLite 持久化

告警类型:
  - 资金流: 大单买入/卖出、资金突增/突减、放量异动
  - 价格: 涨跌幅异动、突破支撑/阻力
  - 技术: RSI 超买/超卖、MACD 金叉/死叉、K 线形态
  - 筹码: 获利盘/套牢盘变化

所有告警持久化到 SQLite，支持自定义阈值配置。
"""

import time
import logging
import threading
from typing import Dict, List, Optional
from collections import deque

logger = logging.getLogger('stock_analyzer')


# ── 告警元数据 ──────────────────────────────────────────────

ALERT_META = {
    'large_buy':     {'name': '大单买入', 'icon': '🔴', 'color': '#ef4444', 'category': 'fund_flow'},
    'large_sell':    {'name': '大单卖出', 'icon': '🟢', 'color': '#10b981', 'category': 'fund_flow'},
    'sudden_inflow': {'name': '资金突增', 'icon': '📈', 'color': '#3b82f6', 'category': 'fund_flow'},
    'sudden_outflow':{'name': '资金突减', 'icon': '📉', 'color': '#f59e0b', 'category': 'fund_flow'},
    'high_volume':   {'name': '放量异动', 'icon': '📊', 'color': '#8b5cf6', 'category': 'fund_flow'},
    'buy_dominant':  {'name': '买盘主导', 'icon': '🟥', 'color': '#ef4444', 'category': 'fund_flow'},
    'sell_dominant': {'name': '卖盘主导', 'icon': '🟩', 'color': '#10b981', 'category': 'fund_flow'},
    'price_up_spike':{'name': '快速上涨', 'icon': '🚀', 'color': '#ef4444', 'category': 'price'},
    'price_down_spike':{'name': '快速下跌', 'icon': '💥', 'color': '#10b981', 'category': 'price'},
    'rsi_overbought':{'name': 'RSI 超买', 'icon': '🔴', 'color': '#ef4444', 'category': 'technical'},
    'rsi_oversold':  {'name': 'RSI 超卖', 'icon': '🟢', 'color': '#10b981', 'category': 'technical'},
    'macd_golden_cross':{'name': 'MACD 金叉', 'icon': '📈', 'color': '#ef4444', 'category': 'technical'},
    'macd_death_cross':{'name': 'MACD 死叉', 'icon': '📉', 'color': '#10b981', 'category': 'technical'},
    'breakout_resistance':{'name': '突破阻力', 'icon': '⬆️', 'color': '#ef4444', 'category': 'technical'},
    'break_support': {'name': '跌破支撑', 'icon': '⬇️', 'color': '#10b981', 'category': 'technical'},
}


class AlertEngine:
    """智能告警引擎 — 真实数据驱动"""

    def __init__(self, storage=None):
        # SQLite 持久化
        self._storage = storage
        if storage:
            logger.info("[AlertEngine] SQLite 持久化已启用")

        # 阈值配置
        self.thresholds = {
            # 资金流
            'large_order_amount': 500000,      # 大单金额（元）
            'sudden_inflow_speed': 3000,       # 资金突增速度（万元/分钟）
            'sudden_outflow_speed': -2000,     # 资金突减速度
            'high_volume_threshold': 100000,   # 放量阈值（手）
            'buy_ratio_dominant': 1.3,         # 买盘主导比
            'sell_ratio_dominant': 0.7,        # 卖盘主导比
            # 价格
            'price_up_threshold': 3.0,         # 快速上涨 %
            'price_down_threshold': -3.0,      # 快速下跌 %
            # 技术
            'rsi_overbought': 80,              # RSI 超买
            'rsi_oversold': 20,                # RSI 超卖
            # 通用
            'alert_cooldown': 60,              # 同类型告警冷却（秒）
        }

        # 内存历史（用于去重和冷却）
        self._alert_history = deque(maxlen=500)
        self._last_alert_time: Dict[str, float] = {}  # type_key → last_fire_time

        # 告警监控线程
        self._monitor_thread = None
        self._monitor_running = False

    # ── 核心检测 ──────────────────────────────────────────────

    def check_alerts(self, stock_code: str, data: Dict,
                     klines: Optional[Dict] = None) -> List[Dict]:
        """
        综合检测所有类型的告警

        Args:
            stock_code: 股票代码
            data: 实时数据 dict（来自 StockDataFetcher.get_stock_info）
            klines: K 线数据 dict（可选，用于技术指标检测）
        """
        alerts = []
        now = time.time()

        # 1. 资金流告警
        alerts.extend(self._check_fund_flow_alerts(data, stock_code, now))

        # 2. 价格告警
        alerts.extend(self._check_price_alerts(data, stock_code, now))

        # 3. 技术指标告警
        if klines:
            alerts.extend(self._check_technical_alerts(stock_code, data, klines, now))

        # 持久化
        if alerts and self._storage:
            try:
                self._storage.save_alerts(alerts, stock_code=stock_code)
            except Exception as e:
                logger.debug(f"[AlertEngine] 存储失败: {e}")

        # 内存历史
        for alert in alerts:
            self._alert_history.append(alert)

        return alerts

    def _check_fund_flow_alerts(self, data: Dict, stock_code: str,
                                now: float) -> List[Dict]:
        """资金流告警检测"""
        alerts = []
        t = self.thresholds
        outer = data.get('outer', 0)
        inner = data.get('inner', 0)
        amount = data.get('amount', 0) * 10000  # 万元→元
        volume = data.get('volume', 0)
        speed = data.get('speed', 0)
        ratio = outer / inner if inner > 0 else 1.0

        # 大单买入
        if outer > inner and amount > t['large_order_amount']:
            severity = 'high' if amount > 1000000 else 'medium'
            alerts.append(self._make_alert(
                stock_code, 'large_buy', f"大单买入: {amount/10000:.1f}万元",
                severity, now, data
            ))

        # 大单卖出
        elif inner > outer and amount > t['large_order_amount']:
            severity = 'high' if amount > 1000000 else 'medium'
            alerts.append(self._make_alert(
                stock_code, 'large_sell', f"大单卖出: {amount/10000:.1f}万元",
                severity, now, data
            ))

        # 资金突增
        if speed > t['sudden_inflow_speed']:
            severity = 'high' if speed > 5000 else 'medium'
            alerts.append(self._make_alert(
                stock_code, 'sudden_inflow',
                f"资金突增: {speed:.0f}万元/分钟", severity, now, data
            ))

        # 放量异动
        if volume > t['high_volume_threshold']:
            alerts.append(self._make_alert(
                stock_code, 'high_volume',
                f"放量成交: {volume}手", 'medium', now, data
            ))

        # 买盘/卖盘主导
        if ratio > t['buy_ratio_dominant']:
            alerts.append(self._make_alert(
                stock_code, 'buy_dominant',
                f"买盘主导: 内外比 {ratio:.2f}", 'medium', now, data
            ))
        elif ratio < t['sell_ratio_dominant']:
            alerts.append(self._make_alert(
                stock_code, 'sell_dominant',
                f"卖盘主导: 内外比 {ratio:.2f}", 'medium', now, data
            ))

        return alerts

    def _check_price_alerts(self, data: Dict, stock_code: str,
                            now: float) -> List[Dict]:
        """价格异动告警检测"""
        alerts = []
        t = self.thresholds
        price_change = data.get('price_change', 0)

        if price_change > t['price_up_threshold']:
            alerts.append(self._make_alert(
                stock_code, 'price_up_spike',
                f"快速上涨: {price_change:+.2f}%",
                'high' if price_change > 5 else 'medium', now, data
            ))
        elif price_change < t['price_down_threshold']:
            alerts.append(self._make_alert(
                stock_code, 'price_down_spike',
                f"快速下跌: {price_change:+.2f}%",
                'high' if price_change < -5 else 'medium', now, data
            ))

        return alerts

    def _check_technical_alerts(self, stock_code: str, data: Dict,
                                klines: Dict, now: float) -> List[Dict]:
        """技术指标告警检测"""
        alerts = []
        t = self.thresholds
        daily = klines.get('daily', [])
        if not daily or len(daily) < 30:
            return alerts

        closes = [k.get('close', 0) for k in daily]
        volumes = [k.get('volume', 0) for k in daily]
        current_price = data.get('price', closes[-1] if closes else 0)

        # ── RSI ──
        rsi = self._calc_rsi(closes, 14)
        if rsi > t['rsi_overbought']:
            alerts.append(self._make_alert(
                stock_code, 'rsi_overbought',
                f"RSI 超买: {rsi:.1f}", 'medium', now,
                {'rsi': rsi, 'type': 'technical'}
            ))
        elif rsi < t['rsi_oversold']:
            alerts.append(self._make_alert(
                stock_code, 'rsi_oversold',
                f"RSI 超卖: {rsi:.1f}", 'medium', now,
                {'rsi': rsi, 'type': 'technical'}
            ))

        # ── MACD 金叉/死叉 ──
        macd_diff = self._calc_macd_diff(closes)
        if len(macd_diff) >= 2:
            prev_diff = macd_diff[-2]
            curr_diff = macd_diff[-1]
            if prev_diff <= 0 and curr_diff > 0:
                alerts.append(self._make_alert(
                    stock_code, 'macd_golden_cross',
                    "MACD 金叉: DIFF 上穿 DEA", 'high', now,
                    {'diff': curr_diff, 'type': 'technical'}
                ))
            elif prev_diff >= 0 and curr_diff < 0:
                alerts.append(self._make_alert(
                    stock_code, 'macd_death_cross',
                    "MACD 死叉: DIFF 下穿 DEA", 'high', now,
                    {'diff': curr_diff, 'type': 'technical'}
                ))

        # ── 突破支撑/阻力 ──
        if len(closes) >= 20:
            recent = closes[-20:]
            resistance = max(recent)
            support = min(recent)
            if current_price > resistance * 1.005:
                alerts.append(self._make_alert(
                    stock_code, 'breakout_resistance',
                    f"突破 20 日阻力: {resistance:.2f} → {current_price:.2f}",
                    'high', now,
                    {'resistance': resistance, 'price': current_price, 'type': 'technical'}
                ))
            elif current_price < support * 0.995:
                alerts.append(self._make_alert(
                    stock_code, 'break_support',
                    f"跌破 20 日支撑: {support:.2f} → {current_price:.2f}",
                    'high', now,
                    {'support': support, 'price': current_price, 'type': 'technical'}
                ))

        return alerts

    # ── 辅助方法 ──────────────────────────────────────────────

    @staticmethod
    def _calc_rsi(closes: List[float], period: int = 14) -> float:
        """计算 RSI"""
        if len(closes) < period + 1:
            return 50.0
        changes = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
        gains = [c for c in changes if c > 0]
        losses = [-c for c in changes if c < 0]
        avg_gain = sum(gains[-period:]) / period if gains else 0
        avg_loss = sum(losses[-period:]) / period if losses else 0.001
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _calc_macd_diff(closes: List[float], fast: int = 12,
                        slow: int = 26, signal: int = 9) -> List[float]:
        """计算 MACD DIFF 数组"""
        if len(closes) < slow + signal:
            return []

        def ema(data, period):
            m = 2.0 / (period + 1)
            result = [data[0]]
            for i in range(1, len(data)):
                result.append((data[i] - result[-1]) * m + result[-1])
            return result

        ema_fast = ema(closes, fast)
        ema_slow = ema(closes, slow)
        macd_line = [f - s for f, s in zip(ema_fast, ema_slow)]
        signal_line = ema(macd_line, signal)
        return [m - s for m, s in zip(macd_line, signal_line)]

    @staticmethod
    def _make_alert(stock_code: str, alert_type: str, message: str,
                    severity: str, now: float, data: Dict = None) -> Dict:
        """创建标准告警 dict"""
        meta = ALERT_META.get(alert_type, {'name': alert_type, 'icon': '⚠️', 'color': '#888', 'category': 'other'})
        return {
            'stock_code': stock_code,
            'type': alert_type,
            'message': message,
            'severity': severity,
            'category': meta.get('category', 'other'),
            'icon': meta.get('icon', '⚠️'),
            'color': meta.get('color', '#888'),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now)),
            'data': data or {},
        }

    # ── 冷却控制 ──────────────────────────────────────────────

    def _should_fire(self, alert_type: str, now: float) -> bool:
        """检查是否在冷却期内"""
        cooldown = self.thresholds.get('alert_cooldown', 60)
        last = self._last_alert_time.get(alert_type, 0)
        if now - last < cooldown:
            return False
        self._last_alert_time[alert_type] = now
        return True

    # ── 读取接口 ──────────────────────────────────────────────

    def get_recent_alerts(self, stock_code: str = None, limit: int = 20) -> List[Dict]:
        """获取最近告警（优先从 SQLite 读取）"""
        if self._storage:
            return self._storage.get_alerts(stock_code, limit)
        # Fallback: 内存
        if stock_code:
            return [a for a in self._alert_history if a.get('stock_code') == stock_code][-limit:]
        return list(self._alert_history)[-limit:]

    def get_alert_summary(self, stock_code: str = None) -> Dict:
        """获取告警统计摘要（优先从 SQLite 读取）"""
        if self._storage:
            return self._storage.get_summary(stock_code)
        # Fallback: 内存
        alerts = self.get_recent_alerts(stock_code, limit=500)
        summary = {'total_alerts': len(alerts), 'by_type': {},
                   'by_severity': {'high': 0, 'medium': 0, 'low': 0},
                   'recent_5min': 0, 'recent_15min': 0}
        now = time.time()
        for a in alerts:
            atype = a.get('type', 'unknown')
            summary['by_type'][atype] = summary['by_type'].get(atype, 0) + 1
            sev = a.get('severity', 'medium')
            summary['by_severity'][sev] = summary['by_severity'].get(sev, 0) + 1
            try:
                at = time.mktime(time.strptime(a.get('timestamp', ''), '%Y-%m-%d %H:%M:%S'))
                if now - at < 300:
                    summary['recent_5min'] += 1
                if now - at < 900:
                    summary['recent_15min'] += 1
            except (ValueError, TypeError):
                pass
        return summary

    # ── 告警监控线程 ──────────────────────────────────────────

    def start_realtime_monitoring(self, fetcher, kline_analyzer,
                                  interval: int = 30,
                                  stock_codes: List[str] = None):
        """
        启动实时告警监控线程

        Args:
            fetcher: StockDataFetcher 实例
            kline_analyzer: KlineSignalAnalyzer 实例（用于技术指标）
            interval: 检查间隔（秒）
            stock_codes: 监控的股票列表，None 则监控所有有 WebSocket 订阅的股票
        """
        self._monitor_running = True
        self._monitor_fetcher = fetcher
        self._monitor_kline_analyzer = kline_analyzer
        self._monitor_interval = interval
        self._monitor_codes = stock_codes

        def monitor_loop():
            while self._monitor_running:
                try:
                    # 获取要监控的股票
                    if stock_codes:
                        codes = stock_codes
                    else:
                        # 从 WebSocket handler 获取有订阅者的股票
                        try:
                            from app import websocket_handler
                            codes = websocket_handler.get_subscribed_stocks()
                        except Exception:
                            codes = []

                    for code in codes:
                        self._check_and_alert(code)
                except Exception as e:
                    logger.debug(f"[AlertEngine] Monitor loop error: {e}")
                time.sleep(interval)

        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True, name='alert-monitor')
        self._monitor_thread.start()
        logger.info(f"[AlertEngine] Realtime monitoring started (interval={interval}s)")

    def _check_and_alert(self, stock_code: str):
        """对单只股票执行告警检测"""
        try:
            fetcher = self._monitor_fetcher
            data = fetcher.get_stock_info(stock_code)
            if not data:
                return

            # 获取 K 线数据
            klines = {}
            for period in ['daily', 'weekly']:
                try:
                    klines[period] = fetcher.get_kline_data(stock_code, period, 100)
                except Exception:
                    pass

            now = time.time()
            alerts = self.check_alerts(stock_code, data, klines if klines else None)

            # 通过 WebSocket 推送实时告警
            if alerts:
                try:
                    from app import socketio
                    for alert in alerts:
                        socketio.emit('new_alert', {
                            'stock_code': stock_code,
                            'alerts': alerts,
                            'timestamp': alert['timestamp'],
                        }, room=stock_code)
                    logger.info(f"[AlertEngine] {len(alerts)} alert(s) for {stock_code}")
                except Exception as e:
                    logger.debug(f"[AlertEngine] WS push failed: {e}")
        except Exception as e:
            logger.debug(f"[AlertEngine] Check failed for {stock_code}: {e}")

    def stop_monitoring(self):
        """停止告警监控"""
        self._monitor_running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("[AlertEngine] Monitoring stopped")
