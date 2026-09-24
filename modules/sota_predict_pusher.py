#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
SOTA 预测实时推送模块

功能:
1. 定时重新运行 SOTA 预测
2. 通过 WebSocket 推送更新到客户端
3. 智能轮询: 只轮询有订阅者的股票
4. 支持预测方向突变告警
"""

import time
import threading
import numpy as np
from typing import Dict, List, Optional
from flask_socketio import SocketIO

from modules.logger import logger


class SOTAPredictPusher:
    """SOTA 预测实时推送器"""

    def __init__(self, socketio: SocketIO, interval: int = 30):
        self.socketio = socketio
        self.interval = interval
        self.clients: Dict[str, List[str]] = {}  # stock_code → [sid, ...]
        self._lock = threading.Lock()
        self._polling_running = False
        self._last_predictions: Dict[str, Dict] = {}  # 缓存上次预测结果

    def subscribe(self, sid: str, stock_code: str):
        """订阅某只股票的 SOTA 预测更新"""
        with self._lock:
            if stock_code not in self.clients:
                self.clients[stock_code] = []
            if sid not in self.clients[stock_code]:
                self.clients[stock_code].append(sid)
        logger.info(f"[SOTA Push] Client {sid} subscribed to {stock_code}")

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
                            del self[code]
        logger.info(f"[SOTA Push] Client {sid} unsubscribed")

    def get_subscribed_stocks(self) -> List[str]:
        """获取所有有订阅者的股票代码"""
        with self._lock:
            return [code for code, sids in self.clients.items() if sids]

    def _run_sota_prediction(self, stock_code: str) -> Optional[Dict]:
        """
        运行 SOTA 预测 (简化版，直接调用 API 逻辑)

        返回预测结果字典，格式与 /api/sota/decision/<code> 一致
        """
        try:
            # 导入必要的模块
            import sys
            if '/Users/claw/stock_analyzer' not in sys.path:
                sys.path.insert(0, '/Users/claw/stock_analyzer')

            from modules.factors.multi_factor_model_v2 import MultiFactorModelV2
            from modules.ml_predictor import MLPredictor
            from modules.sentiment_analyzer_v2 import SentimentEngine
            from modules.kline_signal_analyzer import KlineSignalAnalyzer

            # 获取股票数据 (通过 AKShare)
            try:
                import akshare as ak
                df = ak.stock_zh_a_hist(symbol=stock_code.replace('sh', '').replace('sz', ''),
                                        period="daily", start_date="20240101",
                                        end_date="20261231", adjust="qfq")
                if df is None or len(df) < 30:
                    return None
            except Exception as e:
                logger.debug(f"[SOTA Push] 数据获取失败: {e}")
                return None

            closes = df['收盘'].values.astype(float)
            volumes = df['成交量'].values.astype(float)

            # 1. 多因子模型
            try:
                mf = MultiFactorModelV2()
                mf_features = mf.compute_factors(df)
                mf_pred = mf.predict(mf_features) if mf_features is not None else None
            except Exception as e:
                logger.debug(f"[SOTA Push] 多因子预测失败: {e}")
                mf_pred = None

            # 2. ML 预测
            try:
                ml = MLPredictor()
                ml_pred = ml.predict(df.tail(60).values) if len(df) >= 60 else None
            except Exception as e:
                logger.debug(f"[SOTA Push] ML 预测失败: {e}")
                ml_pred = None

            # 3. 情感分析
            try:
                sent = SentimentEngine()
                sent_pred = {'direction': 'neutral', 'score': 0.0, 'confidence': 0.3}
            except Exception as e:
                logger.debug(f"[SOTA Push] 情感分析失败: {e}")
                sent_pred = None

            # 4. K 线信号
            try:
                kline = KlineSignalAnalyzer()
                kline_pred = kline.analyze(closes[-60:]) if len(closes) >= 60 else None
            except Exception as e:
                logger.debug(f"[SOTA Push] K 线分析失败: {e}")
                kline_pred = None

            # 5. 集成决策 (简单平均)
            directions = []
            scores = []

            if mf_pred and isinstance(mf_pred, dict):
                directions.append(mf_pred.get('direction', 'neutral'))
                scores.append(mf_pred.get('score', 0.0))
            if ml_pred and isinstance(ml_pred, dict):
                directions.append(ml_pred.get('direction', 'neutral'))
                scores.append(ml_pred.get('score', 0.0))
            if sent_pred:
                directions.append(sent_pred.get('direction', 'neutral'))
                scores.append(sent_pred.get('score', 0.0))

            # 多数投票
            from collections import Counter
            if directions:
                ensemble_dir = Counter(directions).most_common(1)[0][0]
                ensemble_score = np.mean(scores) if scores else 0.0
            else:
                ensemble_dir = 'neutral'
                ensemble_score = 0.0

            # 检测方向突变
            direction_change = False
            prev = self._last_predictions.get(stock_code)
            if prev and prev.get('ensemble_direction') != ensemble_dir:
                direction_change = True
                logger.info(
                    f"[SOTA Push] {stock_code} 方向突变: "
                    f"{prev.get('ensemble_direction')} → {ensemble_dir}"
                )

            self._last_predictions[stock_code] = {
                'ensemble_direction': ensemble_dir,
                'ensemble_score': float(ensemble_score),
            }

            return {
                'stock_code': stock_code,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'ensemble_direction': ensemble_dir,
                'ensemble_score': round(float(ensemble_score), 4),
                'direction_change': direction_change,
                'models': {
                    'multi_factor': mf_pred,
                    'ml': ml_pred,
                    'sentiment': sent_pred,
                    'kline': kline_pred,
                },
            }

        except Exception as e:
            logger.error(f"[SOTA Push] 预测失败 for {stock_code}: {e}")
            return None

    def push_prediction(self, stock_code: str):
        """运行预测并推送给所有订阅客户端"""
        result = self._run_sota_prediction(stock_code)
        if not result:
            return

        # 推送给所有订阅该股票的客户端
        with self._lock:
            sids = list(self.clients.get(stock_code, []))

        for sid in sids:
            try:
                self.socketio.emit('sota_prediction_update', result, room=sid, skip_sid=None)
            except Exception as e:
                logger.debug(f"[SOTA Push] Push failed to {sid}: {e}")

    def start_periodic_polling(self):
        """启动定时轮询"""
        if self._polling_running:
            logger.warning("[SOTA Push] 轮询已在运行")
            return

        self._polling_running = True

        def polling_loop():
            while self._polling_running:
                time.sleep(self.interval)
                with self._lock:
                    stocks = list(self.clients.keys())
                for stock_code in stocks:
                    self.push_prediction(stock_code)

        thread = threading.Thread(target=polling_loop, daemon=True, name='sota-polling')
        thread.start()
        logger.info(f"[SOTA Push] Periodic polling started (interval={self.interval}s)")

    def stop_periodic_polling(self):
        """停止轮询"""
        self._polling_running = False
        logger.info("[SOTA Push] Periodic polling stopped")

    def get_status(self) -> Dict:
        """获取推送器状态"""
        with self._lock:
            return {
                'running': self._polling_running,
                'interval': self.interval,
                'subscribed_stocks': list(self.clients.keys()),
                'total_subscribers': sum(len(sids) for sids in self.clients.values()),
                'cached_predictions': len(self._last_predictions),
            }
