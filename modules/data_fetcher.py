"""
Stock Data Fetcher Module
Fetches real-time stock data from multiple sources:
- Tencent Stock API (qt.gtimg.cn)
- East Money API (push2.eastmoney.com)
- K-line data (日/周/月线)

Fixed (2026-06-02):
- year_high/low: swapped parts[44]<>parts[45] (Tencent API: 44=年低点, 45=年高点)
- close: use parts[3] (current price) instead of parts[38] (adjusted close)
- Added prev_close from parts[38] for accurate change_pct calculation
"""

import requests
import json
import time
from typing import Dict, Optional, List

from modules.dynamic_cache import cache
from modules.logger import logger
from modules.kline_data_fetcher import KlineDataFetcher


class StockDataFetcher:
    """Fetches stock data from various APIs"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
        })
        self.cache = {}
        self.cache_ttl = 30  # seconds
        self.kline_fetcher = KlineDataFetcher()
    
    def fetch_tencent_stock(self, stock_code: str) -> Dict:
        """Fetch stock data from Tencent API"""
        cache_key = f"tencent_{stock_code}"
        cached = cache.get(cache_key, category='realtime')
        if cached:
            return cached
        
        try:
            url = f"https://qt.gtimg.cn/q={stock_code}"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                text = response.content.decode('gbk', errors='replace').strip()
                data_str = text.split('=', 1)[1].strip('"')
                parts = data_str.split('~')
                
                if len(parts) > 50:
                    stock_data = self._parse_tencent_data(parts, stock_code)
                    cache.set(cache_key, stock_data, category='realtime')
                    logger.info(f"Tencent数据获取成功: {stock_code}", 
                               extra={'code': stock_code, 'price': stock_data.get('price')})
                    return stock_data
        except Exception as e:
            logger.error(f"Tencent API error: {e}", extra={'code': stock_code})
        
        return None
    
    def fetch_eastmoney_stock(self, market: str, code: str) -> Dict:
        """Fetch stock data from East Money API"""
        cache_key = f"eastmoney_{market}_{code}"
        cached = cache.get(cache_key, category='realtime')
        if cached:
            return cached
        
        try:
            url = f"https://push2.eastmoney.com/api/qt/stock/get"
            params = {
                'secid': f"{market}.{code}",
                'fields': 'f43,f44,f45,f46,f47,f48,f49,f50,f57,f58,f170',
                'ut': 'fa5fd1943c7b386f172d6893dbbd1'
            }
            headers = {'Referer': 'https://quote.eastmoney.com/'}
            response = self.session.get(url, params=params, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'data' in data:
                    stock_data = self._parse_eastmoney_data(data['data'], market, code)
                    cache.set(cache_key, stock_data, category='realtime')
                    return stock_data
        except Exception as e:
            logger.error(f"East Money API error: {e}")
        
        return None
    
    def _parse_tencent_data(self, parts: List[str], stock_code: str) -> Dict:
        """Parse Tencent API response

        Tencent API 字段说明 (2026-06-05 验证):
        - parts[3]  = 最新价 (current price)
        - parts[6]  = 成交量 (volume, 手)
        - parts[7]  = 外盘 (active buys, 主动买入)
        - parts[8]  = 内盘 (active sells, 主动卖出)
        - parts[30] = 时间戳 (timestamp)
        - parts[31] = 涨跌额 (change amount)
        - parts[32] = 涨跌幅 (change %)
        - parts[33] = 开盘价 (open)
        - parts[34] = 最高价 (high)
        - parts[35] = 最低价/成交量/成交额 (low/vol/amt, 用 / 分隔)
        - parts[37] = 成交额 (amount, 万元)
        - parts[38] = 昨收 (yesterday's close / adjusted close)
        - parts[39] = 换手率 (turnover rate)
        - parts[44] = 年低点 (year low)
        - parts[45] = 年高点 (year high)
        - parts[46] = 流通市值 (circulating cap, 亿元)
        - parts[47] = 总市值 (total market cap, 亿元)
        - parts[48] = 市盈率 (PE ratio)
        """
        # 解析 parts[35] = "low/volume/amount"
        low_str = parts[35].split('/')[0] if len(parts) > 35 else '0'
        low = float(low_str) if low_str else 0
        
        # 核心修复: close 使用 parts[3] (最新价) 而非 parts[38] (复权价)
        # parts[38] 在除权后可能与最新价差异很大 (如 光库科技: 8.10 vs 278.65)
        current_price = float(parts[3]) if len(parts) > 3 else 0
        prev_close = float(parts[38]) if len(parts) > 38 else current_price
        
        # 如果 prev_close 异常小 (可能是复权价), 用 current_price 代替
        if prev_close > 0 and prev_close < current_price * 0.1:
            prev_close = current_price
        
        return {
            'source': 'tencent',
            'code': stock_code,
            'name': parts[1] if len(parts) > 1 else '',
            'price': current_price,
            'open': float(parts[33]) if len(parts) > 33 else current_price,
            'high': float(parts[34]) if len(parts) > 34 else current_price,
            'low': low,
            'close': current_price,  # 修复: 使用最新价
            'prev_close': prev_close,  # 新增: 昨收价
            'change': float(parts[31]) if len(parts) > 31 else 0,
            'change_pct': float(parts[32]) if len(parts) > 32 else 0,
            'volume': int(parts[6]) if len(parts) > 6 else 0,
            'amount': float(parts[37]) if len(parts) > 37 else 0,
            'turnover': float(parts[39]) if len(parts) > 39 else 0,
            'pe': float(parts[48]) if len(parts) > 48 else 0,
            'market_cap': float(parts[47]) if len(parts) > 47 else 0,
            'circulating_cap': float(parts[46]) if len(parts) > 46 else 0,
            'year_low': float(parts[44]) if len(parts) > 44 else 0,    # 修复: 年低点
            'year_high': float(parts[45]) if len(parts) > 45 else 0,    # 修复: 年高点
            'outer_disk': int(parts[7]) if len(parts) > 7 else 0,   # 外盘 = 主动买入 = parts[7]
            'inner_disk': int(parts[8]) if len(parts) > 8 else 0,   # 内盘 = 主动卖出 = parts[8]
            'timestamp': parts[30] if len(parts) > 30 else '',
        }
    
    def _parse_eastmoney_data(self, data: Dict, market: str, code: str) -> Dict:
        """Parse East Money API response"""
        return {
            'source': 'eastmoney',
            'code': f"{market}.{code}",
            'name': data.get('f170', ''),
            'price': data.get('f43', 0),
            'change': data.get('f44', 0),      # 涨跌额 (absolute change)
            'change_pct': data.get('f49', 0),  # 涨跌幅 (percentage change) — P2 修复: 原用 f44 (涨跌额) 错误
            'high': data.get('f45', 0),
            'low': data.get('f46', 0),
            'open': data.get('f47', 0),
            'close': data.get('f43', 0),       # P2 修复: 原用 f48 (昨收) 错误，f43=最新价
            'prev_close': data.get('f48', 0),  # 新增: 昨收价
            'volume': data.get('f50', 0),
            'amount': data.get('f57', 0),
            'pe': data.get('f58', 0),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
    
    def get_stock_info(self, stock_code: str) -> Dict:
        """Get comprehensive stock information"""
        data = self.fetch_tencent_stock(stock_code)
        if not data:
            market = 0 if stock_code.startswith('sz') else 1
            code = stock_code[2:]
            data = self.fetch_eastmoney_stock(market, code)
        return data
    
    def get_kline_data(self, stock_code: str, period: str = 'daily', 
                       count: int = 250) -> List[Dict]:
        """获取历史K线数据"""
        cache_key = f"kline_{stock_code}_{period}_{count}"
        cached = cache.get(cache_key, category='kline')
        if cached:
            return cached
        
        klines = self.kline_fetcher.fetch_kline(stock_code, period, count)
        if klines:
            cache.set(cache_key, klines, category='kline')
        return klines
    
    def get_enhanced_stock_info(self, stock_code: str) -> Dict:
        """获取增强版股票信息（含K线统计）"""
        stock_data = self.get_stock_info(stock_code)
        if not stock_data:
            return stock_data
        
        # 获取K线统计
        kline_stats = self.kline_fetcher.get_kline_stats(stock_code, 'daily', 250)
        
        # 合并数据
        if 'error' not in kline_stats:
            stock_data['kline_stats'] = kline_stats
            stock_data['historical_high'] = kline_stats.get('highest', 0)
            stock_data['historical_low'] = kline_stats.get('lowest', 0)
            stock_data['avg_volume'] = kline_stats.get('avg_volume', 0)
            stock_data['std_dev'] = kline_stats.get('std_dev', 0)
        
        return stock_data
    
    def clear_cache(self):
        """Clear all caches"""
        self.cache.clear()
        cache.cleanup()
