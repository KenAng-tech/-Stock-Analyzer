"""
K线数据获取模块
从新浪/东方财富获取历史K线数据（日/周/月）
"""

import requests
import json
import time
from typing import Dict, List, Optional
from datetime import datetime


class KlineDataFetcher:
    """历史K线数据获取器"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
        })
        self.cache = {}
        self.cache_ttl = 300  # 5分钟缓存
    
    def fetch_kline(self, stock_code: str, period: str = 'daily', 
                    count: int = 250) -> List[Dict]:
        """
        获取历史K线数据
        
        Args:
            stock_code: 股票代码 (sz300620)
            period: 周期 (daily/weekly/monthly)
            count: 获取K线数量
        
        Returns:
            K线数据列表
        """
        cache_key = f"kline_{stock_code}_{period}_{count}"
        if cache_key in self.cache and time.time() - self.cache[cache_key]['time'] < self.cache_ttl:
            return self.cache[cache_key]['data']

        # 优先使用 AKShare（有复权数据）
        klines = self._fetch_akshare_kline(stock_code, period, count)
        if klines:
            self.cache[cache_key] = {'data': klines, 'time': time.time()}
            return klines

        # 备用1: 东方财富K线（有复权）
        klines = self._fetch_eastmoney_kline(stock_code, period, count)
        if klines:
            self.cache[cache_key] = {'data': klines, 'time': time.time()}
            return klines

        # 备用2: 新浪K线（无复权）
        return self._fetch_sina_kline(stock_code, period, count)
    
    def _fetch_sina_kline(self, stock_code: str, period: str, 
                          count: int) -> List[Dict]:
        """新浪K线数据"""
        try:
            # 新浪K线API
            url = 'http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData'
            
            # 根据周期设置scale
            scale_map = {'daily': '240', 'weekly': '1200', 'monthly': '4800'}
            scale = scale_map.get(period, '240')
            
            params = {
                'symbol': stock_code,
                'scale': scale,
                'ma': 'no',
                'datalen': str(count),
            }
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                klines = []
                for item in data:
                    klines.append({
                        'date': item.get('day', ''),
                        'open': float(item.get('open', 0)),
                        'close': float(item.get('close', 0)),
                        'high': float(item.get('high', 0)),
                        'low': float(item.get('low', 0)),
                        'volume': float(item.get('volume', 0)),
                        'amount': 0,  # 新浪K线不提供成交额
                        'stock_code': stock_code,
                        'adjusted': False,  # 新浪K线为原始价格，未复权
                    })
                return klines
        except Exception as e:
            print(f"[KlineFetcher] 新浪K线错误: {e}")
        
        return []
    
    def _fetch_eastmoney_kline(self, stock_code: str, period: str, 
                                count: int) -> List[Dict]:
        """东方财富K线数据"""
        try:
            market = 1 if stock_code.startswith('sh') else 0
            code = stock_code[2:]
            period_map = {'daily': '101', 'weekly': '102', 'monthly': '103'}
            period_code = period_map.get(period, '101')
            
            url = 'https://push2.eastmoney.com/api/qt/stock/kline/get'
            params = {
                'secid': f'{market}.{code}',
                'fields1': 'f1,f2,f3,f4,f5,f6',
                'fields2': 'f51,f52,f53,f54,f55,f56,f57',  # P2 修复: 移除未使用的 f58~f61
                'klt': period_code,
                'fqt': 1,
                'beg': '',
                'end': '20500101',
                'lmt': count,
            }
            headers = {'Referer': 'https://quote.eastmoney.com/'}
            response = self.session.get(url, params=params, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('data') and data['data'].get('klines'):
                    klines = []
                    for line in data['data']['klines']:
                        parts = line.split(',')
                        klines.append({
                            'date': parts[0],
                            'open': float(parts[1]),
                            'close': float(parts[2]),
                            'high': float(parts[3]),
                            'low': float(parts[4]),
                            'volume': float(parts[5]),
                            'amount': float(parts[6]),
                            'stock_code': stock_code,
                            'adjusted': True,  # 东方财富 fqt=1 表示前复权
                        })
                    return klines
        except Exception as e:
            print(f"[KlineFetcher] 东方财富K线错误: {e}")

        # 备用: AKShare K 线
        return self._fetch_akshare_kline(stock_code, period, count)

    def _fetch_akshare_kline(self, stock_code: str, period: str,
                              count: int) -> List[Dict]:
        """AKShare K 线数据 (备用数据源)"""
        try:
            import akshare as ak
            # 提取纯数字代码
            code_num = stock_code[2:]
            df = ak.stock_zh_a_hist(symbol=code_num, period='daily', adjust='qfq')
            if df is None or df.empty:
                return []

            # 只取最近 count 条
            df = df.tail(count)
            klines = []
            for _, row in df.iterrows():
                klines.append({
                    'date': str(row.get('日期', '')),
                    'open': float(row.get('开盘', 0)),
                    'close': float(row.get('收盘', 0)),
                    'high': float(row.get('最高', 0)),
                    'low': float(row.get('最低', 0)),
                    'volume': float(row.get('成交量', 0)),
                    'amount': float(row.get('成交额', 0)),
                    'stock_code': stock_code,
                    'adjusted': True,  # AKShare 使用 adjust='qfq' 返回前复权数据
                })
            return klines
        except ImportError:
            logger.warning("[KlineFetcher] AKShare 未安装")
        except Exception as e:
            print(f"[KlineFetcher] AKShare K 线错误: {e}")

        return []
    
    def get_kline_stats(self, stock_code: str, period: str = 'daily', 
                        count: int = 250) -> Dict:
        """获取K线统计数据"""
        klines = self.fetch_kline(stock_code, period, count)
        if not klines:
            return {'error': 'No data'}
        
        closes = [k['close'] for k in klines if k['close'] > 0]
        volumes = [k['volume'] for k in klines if k['volume'] > 0]
        
        import numpy as np
        
        stats = {
            'count': len(klines),
            'latest': klines[-1] if klines else {},
            'highest': max(closes) if closes else 0,
            'lowest': min(closes) if closes else 0,
            'avg_volume': float(np.mean(volumes)) if volumes else 0,
            'avg_close': float(np.mean(closes)) if closes else 0,
            'std_dev': float(np.std(closes)) if len(closes) > 1 else 0,
            'klines': klines[-30:],  # 最近30根
        }
        return stats


if __name__ == '__main__':
    fetcher = KlineDataFetcher()
    klines = fetcher.fetch_kline('sz300620', 'daily', 30)
    print(f'获取 {len(klines)} 根K线')
    if klines:
        print(f'最新: {klines[-1]}')
    stats = fetcher.get_kline_stats('sz300620', 'daily', 250)
    print(f'统计: {stats}')
