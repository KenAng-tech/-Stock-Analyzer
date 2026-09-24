"""
shared_utils.py - 共享工具函数

从 route 文件中提取的重复函数，避免在每个文件中重复定义。
所有 route 文件应从此模块导入，而非自行定义。
"""

import functools
import threading
import numpy as np
from flask import jsonify
from modules.dynamic_cache import cache
from modules.data_fetcher import StockDataFetcher


def timeout_handler(seconds):
    """超时处理装饰器 — 超时返回 504
    注意: 由于 Flask 上下文限制, 超时时返回 stub 响应而非原始错误.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = [None]
            exception = [None]

            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e

            thread = threading.Thread(target=target, daemon=True)
            thread.start()
            thread.join(timeout=seconds)

            if thread.is_alive():
                # 超时: 返回 stub 而非调用 jsonify (可能无 Flask 上下文)
                return {
                    'success': False,
                    'error': f'Request timed out after {seconds}s',
                    'timeout': seconds,
                }

            if exception[0] is not None:
                raise exception[0]
            return result[0]
        return wrapper
    return decorator


def get_stock_data(stock_code: str):
    """获取股票数据（带缓存）"""
    cache_key = f"stock_{stock_code}"
    cached = cache.get(cache_key, category='realtime')
    if cached:
        return cached
    fetcher = StockDataFetcher()
    stock_data = fetcher.get_stock_info(stock_code)
    if stock_data:
        cache.set(cache_key, stock_data, category='realtime')
    return stock_data


def build_features(stock_data: dict, klines: list = None) -> list:
    """构建特征向量（价格、涨跌幅、成交量、换手率、PE、PB、短期收益率）"""
    if not stock_data:
        return []
    features = []
    features.append(stock_data.get('price', 0))
    features.append(stock_data.get('change_pct', 0))
    features.append(stock_data.get('volume', 0))
    features.append(stock_data.get('turnover', 0))
    features.append(stock_data.get('pe', 0))
    features.append(stock_data.get('pb', 0))
    if klines:
        closes = [k['close'] for k in klines[-20:]]
        if len(closes) >= 2:
            features.append((closes[-1] - closes[-2]) / closes[-2] * 100)
        else:
            features.append(0)
    return features


def prepare_training_data(stock_code: str, n_samples: int = 500) -> tuple:
    """准备训练数据（20 日窗口分类）"""
    fetcher = StockDataFetcher()
    klines = fetcher.get_kline_data(stock_code, 'daily', n_samples)
    if not klines or len(klines) < 50:
        return None, None
    closes = [k['close'] for k in klines]
    X = []
    y = []
    for i in range(20, len(closes)):
        window = closes[i-20:i]
        X.append(window)
        y.append(1 if closes[i] > closes[i-1] else 0)
    return np.array(X), np.array(y)


def ema(data: list, period: int) -> list:
    """指数移动平均线"""
    if len(data) == 0:
        return []
    m = 2.0 / (period + 1)
    result = [float(data[0])]
    for i in range(1, len(data)):
        result.append((float(data[i]) - result[-1]) * m + result[-1])
    return result


def compute_rsi(prices: np.ndarray, period: int = 14) -> float:
    """计算 RSI (Relative Strength Index)"""
    if len(prices) < period + 1:
        return 50.0
    deltas = np.diff(prices)
    gains = np.mean(deltas[-period:][deltas[-period:] > 0]) if np.any(deltas[-period:] > 0) else 0
    losses = abs(np.mean(deltas[-period:][deltas[-period:] < 0])) if np.any(deltas[-period:] < 0) else 0.001
    rs = gains / losses
    return float(100 - (100 / (1 + rs)))


def compute_macd_histogram(prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> float:
    """计算 MACD 柱状图（简化版）"""
    if len(prices) < slow + signal:
        return 0.0
    ema_fast = prices[0]
    for p in prices[1:]:
        ema_fast = (p - ema_fast) * (2 / (fast + 1)) + ema_fast
    ema_slow = prices[0]
    for p in prices[1:]:
        ema_slow = (p - ema_slow) * (2 / (slow + 1)) + ema_slow
    macd_line = ema_fast - ema_slow
    return float(macd_line * 0.1)


def compute_ema(data: np.ndarray, period: int) -> float:
    """计算 EMA（返回单个值）"""
    if len(data) < period:
        return float(np.mean(data))
    m = 2.0 / (period + 1)
    r = float(data[0])
    for p in data[1:]:
        r = (p - r) * m + r
    return r
