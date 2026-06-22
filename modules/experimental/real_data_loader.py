#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Real Data Loader — 真实 OHLCV 数据加载器

使用 akshare 获取真实 A 股历史行情数据，支持磁盘缓存（parquet），
24 小时 TTL，避免重复请求。

替代方案：
  - 优先 akshare（免费）
  - 回退到东方财富 push2 API
  - 缓存目录: data/ohlcv_cache/
"""

import os
import json
import datetime
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── 缓存配置 ──────────────────────────────────────────────
CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'ohlcv_cache')
CACHE_TTL_HOURS = 24
os.makedirs(CACHE_DIR, exist_ok=True)


def _code_to_akshare(code: str) -> str:
    """将 sz300620 / sh688313 格式转换为 akshare 需要的格式。"""
    code = code.strip().lower()
    if code.startswith('sz'):
        return code[2:]  # 300620
    if code.startswith('sh'):
        return code[2:]  # 688313
    return code


def _cache_path(code: str, period: str) -> str:
    safe = code.replace('.', '_').replace('-', '_')
    return os.path.join(CACHE_DIR, f'{safe}_{period}.parquet')


def _cache_meta_path(code: str, period: str) -> str:
    safe = code.replace('.', '_').replace('-', '_')
    return os.path.join(CACHE_DIR, f'{safe}_{period}.meta.json')


def _is_cache_valid(meta_path: str) -> bool:
    """检查缓存是否过期。"""
    if not os.path.exists(meta_path):
        return False
    try:
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        cached_at = datetime.datetime.fromisoformat(meta['cached_at'])
        return (datetime.datetime.now() - cached_at).total_seconds() < CACHE_TTL_HOURS * 3600
    except Exception:
        return False


# ── 数据获取 ──────────────────────────────────────────────

def _fetch_via_akshare(symbol: str, period: str = 'daily') -> Optional[pd.DataFrame]:
    """通过 akshare 获取真实数据。"""
    try:
        import akshare as ak
    except ImportError:
        return None

    try:
        if period == 'daily':
            # 获取最近 2 年的日 K 线
            end_date = datetime.datetime.now().strftime('%Y%m%d')
            start_date = (datetime.datetime.now() - datetime.timedelta(days=730)).strftime('%Y%m%d')
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period=start_date + '-' + end_date,
                adjust='qfq'  # 前复权
            )
            if df is not None and len(df) > 0:
                df = df.rename(columns={
                    '日期': 'date',
                    '开盘': 'open',
                    '最高': 'high',
                    '最低': 'low',
                    '收盘': 'close',
                    '成交量': 'volume',
                    '成交额': 'amount',
                    '振幅': 'amplitude',
                    '涨跌幅': 'change_pct',
                    '涨跌额': 'change',
                    '换手率': 'turnover',
                })
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.sort_values('date').reset_index(drop=True)
                # 确保数值类型
                for col in ['open', 'high', 'low', 'close', 'volume', 'amount', 'change_pct', 'turnover']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                return df
    except Exception as e:
        logger.warning(f"akshare daily fetch failed for {symbol}: {e}")

    return None


def _fetch_via_eastmoney(code6: str) -> Optional[pd.DataFrame]:
    """通过东方财富 push2 API 获取日 K 线（akshare 不可用时的回退）。"""
    try:
        import urllib.request
        import json as json_mod
    except ImportError:
        return None

    try:
        # 判断市场
        suffix = ''
        if code6.startswith(('6', '9')):
            secid = f'1.{code6}'
        else:
            secid = f'0.{code6}'

        # 获取 500 根 K 线
        url = (
            f'https://push2his.eastmoney.com/api/qt/stock/kline/get?'
            f'secid={secid}&fields1=f1,f2,f3,f4,f5,f6&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61'
            f'&klt=101&fqt=1&beg=0&end=20500101&lmt=500'
        )
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0',
            'Referer': 'https://quote.eastmoney.com/'
        })
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json_mod.loads(resp.read().decode('utf-8'))

        klines = data.get('data', {}).get('klines', [])
        if not klines:
            return None

        rows = []
        for k in klines:
            parts = k.split(',')
            rows.append({
                'date': pd.Timestamp(parts[0]),
                'open': float(parts[1]),
                'close': float(parts[2]),
                'high': float(parts[3]),
                'low': float(parts[4]),
                'volume': float(parts[5]),
                'amount': float(parts[6]),
                'change_pct': float(parts[7]) if len(parts) > 7 else 0,   # P2 修复: 边界检查
                'turnover': float(parts[8]) if len(parts) > 8 else 0,     # P2 修复: 边界检查
            })

        df = pd.DataFrame(rows)
        df = df.sort_values('date').reset_index(drop=True)
        return df
    except Exception as e:
        logger.warning(f"eastmoney fetch failed for {code6}: {e}")
        return None


# ── 主类 ──────────────────────────────────────────────────

class RealDataLoader:
    """真实 OHLCV 数据加载器。

    用法:
        loader = RealDataLoader()
        df = loader.load_klines('sz300620', lookback=250)
        # df 列: date, open, high, low, close, volume, amount, change_pct, turnover
    """

    def __init__(self, cache_ttl_hours: int = CACHE_TTL_HOURS):
        self.cache_ttl = cache_ttl_hours

    def load_klines(self, stock_code: str, lookback: int = 250) -> pd.DataFrame:
        """获取最近 N 根日 K 线数据。

        Args:
            stock_code: 股票代码，如 'sz300620', 'sh688313'
            lookback: 需要的 K 线数量

        Returns:
            DataFrame with columns: date, open, high, low, close, volume, amount, change_pct, turnover
        """
        symbol = _code_to_akshare(stock_code)
        period = f'kline_{lookback}'

        # 尝试从缓存读取
        meta_path = _cache_meta_path(stock_code, period)
        cache_file = _cache_path(stock_code, period)
        if _is_cache_valid(meta_path):
            try:
                df = pd.read_parquet(cache_file)
                if len(df) >= lookback:
                    logger.debug(f"Cache hit for {stock_code} ({len(df)} rows)")
                    return df.head(lookback)
            except Exception as e:
                logger.warning(f"Cache read failed for {stock_code}: {e}")

        # 尝试 akshare
        df = _fetch_via_akshare(symbol, 'daily')

        # 回退到东方财富
        if df is None or len(df) < lookback:
            df = _fetch_via_eastmoney(symbol)

        if df is None or len(df) == 0:
            logger.error(f"Failed to fetch data for {stock_code}")
            return pd.DataFrame()

        # 确保足够的行数
        if len(df) > lookback:
            df = df.tail(lookback).reset_index(drop=True)

        # 保存缓存
        try:
            df.to_parquet(cache_file, index=False)
            with open(meta_path, 'w') as f:
                json.dump({
                    'stock_code': stock_code,
                    'cached_at': datetime.datetime.now().isoformat(),
                    'rows': len(df),
                }, f)
        except Exception as e:
            logger.warning(f"Cache write failed for {stock_code}: {e}")

        return df

    def load_daily(self, stock_code: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """获取指定日期范围的日 K 线数据。

        Args:
            stock_code: 股票代码
            start_date: 开始日期 'YYYY-MM-DD'，默认 2 年前
            end_date: 结束日期 'YYYY-MM-DD'，默认今天

        Returns:
            DataFrame with OHLCV data
        """
        symbol = _code_to_akshare(stock_code)

        if end_date is None:
            end_date = datetime.datetime.now().strftime('%Y-%m-%d')
        if start_date is None:
            start_date = (datetime.datetime.now() - datetime.timedelta(days=730)).strftime('%Y-%m-%d')

        # 尝试 akshare
        try:
            import akshare as ak
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period=f'{start_date} {end_date}',
                adjust='qfq'
            )
            if df is not None and len(df) > 0:
                df = df.rename(columns={
                    '日期': 'date', '开盘': 'open', '最高': 'high',
                    '最低': 'low', '收盘': 'close', '成交量': 'volume',
                    '成交额': 'amount', '涨跌幅': 'change_pct', '换手率': 'turnover',
                })
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                return df.sort_values('date').reset_index(drop=True)
        except Exception as e:
            logger.warning(f"akshare date range fetch failed: {e}")

        # 回退：使用 load_klines 获取最近数据
        return self.load_klines(stock_code, lookback=500)

    def get_price_history(self, stock_code: str, days: int = 60) -> List[float]:
        """获取最近 N 天的收盘价列表。

        用于技术指标计算（RSI、MACD、MA 等）的便捷方法。
        """
        df = self.load_klines(stock_code, lookback=max(days, 250))
        if df.empty:
            return []
        return df['close'].tolist()[-days:]

    def get_full_history(self, stock_code: str) -> Dict:
        """获取完整历史数据 + 计算好的技术指标。

        Returns:
            Dict with 'klines' (DataFrame), 'closes', 'volumes', 'indicators'
        """
        df = self.load_klines(stock_code, lookback=500)
        if df.empty:
            return {'klines': [], 'closes': [], 'volumes': [], 'indicators': {}}

        closes = df['close'].tolist()
        volumes = df['volume'].tolist()

        # 预计算常用技术指标
        indicators = {}
        if len(closes) >= 5:
            indicators['ma5'] = np.mean(closes[-5:])
        if len(closes) >= 10:
            indicators['ma10'] = np.mean(closes[-10:])
        if len(closes) >= 20:
            indicators['ma20'] = np.mean(closes[-20:])
        if len(closes) >= 60:
            indicators['ma60'] = np.mean(closes[-60:])

        return {
            'klines': df.to_dict('records'),
            'closes': closes,
            'volumes': volumes,
            'indicators': indicators,
        }
