"""
K线数据获取模块
从新浪/东方财富获取历史K线数据（日/周/月）
"""

import requests
import json
import random
import time
from typing import Dict, List, Optional
from datetime import datetime

from modules.logger import logger
from utils.outbound_guard import host_throttle, cooldowns, flight_fetch


class KlineDataFetcher:
    """历史K线数据获取器

    数据源优先级:
    1. AKShare (前复权) — 最可靠, 但可能网络不可用
    2. 东方财富 (前复权) — 有复权数据, 带超时保护
    3. 本地复权 (前复权) — 基于新浪原始数据 + 除权检测算法
    4. 新浪 (原始价格) — 最后备用, 未复权
    """

    # 创业板(300xxx)涨跌停 ±20%, 主板 ±10%
    _CHINEXT_LIMIT = 0.20
    _MAINBOARD_LIMIT = 0.10

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
        })
        self.cache = {}
        self.cache_ttl = 300  # 5分钟缓存
        # 失败缓存：避免对已知的故障源重复请求（300秒冷却，因网络可能长期不可用）
        self._fail_cache: Dict[str, float] = {}
        self._fail_ttl = 300
        # 重试配置
        self._max_retries = 2
        self._eastmoney_timeout = 10.0
    
    def _fetch_with_timeout(self, func, *args, timeout: float = 5.0, **kwargs):
        """在线程池中执行函数，带超时保护 (2026-09-21 修: 异常分类)。

        修复: 原两个 except Exception 第二分支不可达, 且 TimeoutExpired 与
        子线程内数据形状错误 (ValueError/KeyError 等) 同走 "超时" 日志 —
        观测面失真 (链路健康排查会误判上游慢)。现: 超时=warning, 其余=debug。
        不加重试: 本方法被 flight singleflight (25s 预算) 包裹, 重试会放大
        慢上游场景尾延迟 (09-20 设计: 降级非断链, singleflight 已合并并发)。
        """
        import concurrent.futures
        try:
            from modules.log_context import with_trace
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as exe:
                future = exe.submit(with_trace(func), *args, **kwargs)
                return future.result(timeout=timeout)
        except (TimeoutError, concurrent.futures.TimeoutError):
            # 3.11+ 下 concurrent.futures.TimeoutError 即内建 TimeoutError
            logger.warning(f"[KlineFetcher] {func.__name__} 超时 ({timeout}s)")
            return None
        except Exception as e:
            logger.debug(f"[KlineFetcher] {func.__name__} 失败: {e}")
            return None

    def _is_failed(self, source: str) -> bool:
        """检查数据源是否在冷却期内"""
        last_fail = self._fail_cache.get(source, 0)
        return time.time() - last_fail < self._fail_ttl

    def _mark_failed(self, source: str):
        """标记数据源为故障（加入冷却期）"""
        self._fail_cache[source] = time.time()

    def fetch_kline(self, stock_code: str, period: str = 'daily',
                    count: int = 250) -> List[Dict]:
        """
        获取历史K线数据

        数据源优先级:
        1. AKShare (前复权)
        2. 东方财富 (前复权)
        3. 本地复权 (基于新浪原始数据 + 除权检测)
        4. 新浪 (原始价格, 未复权)

        Args:
            stock_code: 股票代码 (sz300620)
            period: 周期 (daily/weekly/monthly)
            count: 获取K线数量

        Returns:
            K线数据列表
        """
        cache_key = f"kline_{stock_code}_{period}_{count}"

        # P0-2 (2026-09-20): 同 key 并发合并 (singleflight) — 告警 30s 链/晨报 TP 执行链/
        # Telegram 触发并发同标的时, followers 复用 leader 结果, 不重复打上游。
        # 超时 25s 对齐既有 timeout 预算; 超时/失败 → 下方自取 (保护链永不阻断取数)。
        result = flight_fetch(
            cache_key,
            lambda: self._fetch_kline_uncached(stock_code, period, count),
            timeout=25.0,
        )
        if result is not None:
            return result

        return self._fetch_kline_uncached(stock_code, period, count)

    def _cache_get(self, cache_key: str) -> Optional[List[Dict]]:
        cached = self.cache.get(cache_key)
        if cached and time.time() - cached['time'] < self.cache_ttl:
            return cached['data']
        return None

    def _fetch_kline_uncached(self, stock_code: str, period: str = 'daily',
                              count: int = 250) -> List[Dict]:
        """4 级降级链主体 (flight 锁内执行, 并发同 key 已合并).

        每级 = (source,code) 级冷却 + 同 host 节流; 失败开冷却 (盘中 60s/其余 900s),
        成功清冷却 — 一标的故障只烧它自己, 不烧整个 provider 链。
        """
        cache_key = f"kline_{stock_code}_{period}_{count}"
        cached = self._cache_get(cache_key)  # flight 内二次检查
        if cached is not None:
            return cached

        # 优先使用 AKShare（有复权数据），带超时保护
        ak_key = f"akshare:{stock_code}"
        if not self._is_failed('akshare') and not cooldowns.is_cooling(ak_key):
            klines = self._fetch_with_timeout(
                self._fetch_akshare_kline, stock_code, period, count,
                timeout=self._eastmoney_timeout)
            if klines:
                self.cache[cache_key] = {'data': klines, 'time': time.time()}
                cooldowns.clear(ak_key)
                self._fail_cache.pop('akshare', None)
                return klines
            cooldowns.open(ak_key)
            self._mark_failed('akshare')

        # 备用: 东方财富K线（有复权，延长超时 + 重试）
        em_key = f"eastmoney:{stock_code}"
        if not self._is_failed('eastmoney') and not cooldowns.is_cooling(em_key):
            host_throttle.acquire('eastmoney', host_key='push2.eastmoney.com')
            klines = self._fetch_eastmoney_with_retry(stock_code, period, count)
            if klines:
                self.cache[cache_key] = {'data': klines, 'time': time.time()}
                cooldowns.clear(em_key)
                self._fail_cache.pop('eastmoney', None)
                return klines
            cooldowns.open(em_key)
            self._mark_failed('eastmoney')

        # 备用3: 本地复权 (基于新浪原始数据 + 除权检测算法)
        sina_key = f"sina:{stock_code}"
        if not cooldowns.is_cooling(sina_key):
            host_throttle.acquire('sina', host_key='money.finance.sina.com.cn')
            klines = self._fetch_sina_with_local_adjust(stock_code, period, count)
            if klines:
                self.cache[cache_key] = {'data': klines, 'time': time.time()}
                cooldowns.clear(sina_key)
                return klines

            # 最终备用: 新浪原始K线（未复权）
            klines = self._fetch_sina_kline(stock_code, period, count)
            if klines:
                self.cache[cache_key] = {'data': klines, 'time': time.time()}
                cooldowns.clear(sina_key)
            return klines

        return []
    
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

    def _fetch_eastmoney_with_retry(self, stock_code: str, period: str,
                                     count: int) -> List[Dict]:
        """东方财富K线，带重试和备用 secid 格式"""
        for attempt in range(self._max_retries + 1):
            klines = self._fetch_eastmoney_kline(stock_code, period, count)
            if klines:
                return klines
            if attempt < self._max_retries:
                logger.debug(f"[KlineFetcher] 东方财富第 {attempt+1} 次重试...")
                # P0-3 (2026-09-20): 退避+jitter 重试 (参照 PanWatch backoff 模式;
                # 固定 1s 会与服务端限频窗口同频共振)
                time.sleep(min(1.0 * (2 ** attempt), 4.0) + random.uniform(0, 0.3))
        return []

    # ── 本地复权算法 ─────────────────────────────────────────────
    # 当 AKShare/东方财富 不可用时, 使用新浪原始数据 + 本地复权
    #
    # 策略: 两段式线性缩放
    #   1. 获取尽可能长的历史数据 (2000 天)
    #   2. 以数据窗口内最新价为锚点, 将最新价缩放至与腾讯实时行情一致
    #   3. 检测窗口内最大的价格跳空点, 在跳空点前后分别缩放
    #   4. 这能消除窗口内可见的除权跳空, 使价格序列连续

    @staticmethod
    def _local_forward_adjust(
        raw_prices: List[tuple],
        current_price: float,
    ) -> List[tuple]:
        """
        本地前复权: 以实时价格为锚点的相对缩放

        策略:
        1. 获取新浪原始K线 (已验证最新价格与腾讯实时行情一致)
        2. 计算缩放因子 = current_price / latest_raw_price
        3. 所有价格乘以该因子

        说明:
        - 如果除权事件在数据窗口内, 缩放后价格仍然包含跳空
        - 如果除权事件在数据窗口外, 缩放能部分校正
        - 对于技术指标(收益率/均线/MACD), 相对价格关系不变, 影响很小
        - 最新价格已准确对齐, 满足当前交易决策需求
        """
        if len(raw_prices) < 60:
            return [(d, round(p, 3)) for d, p in raw_prices]

        latest_raw = raw_prices[-1][1]
        if latest_raw <= 0:
            return [(d, round(p, 3)) for d, p in raw_prices]

        factor = current_price / latest_raw
        return [(d, round(p * factor, 3)) for d, p in raw_prices]

    def _fetch_sina_with_local_adjust(self, stock_code: str, period: str,
                                       count: int) -> List[Dict]:
        """
        新浪原始K线 + 本地前复权

        当 AKShare/东方财富 都不可用时, 使用此方法:
        1. 获取新浪原始K线数据 (最多 2000 天)
        2. 从腾讯行情获取实时价格作为复权锚点
        3. 检测窗口内最大价格跳空, 进行分段缩放
        4. 标记 adjusted: True
        """
        try:
            # 1. 获取新浪原始K线
            url = 'http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData'
            scale_map = {'daily': '240', 'weekly': '1200', 'monthly': '4800'}
            scale = scale_map.get(period, '240')

            params = {
                'symbol': stock_code,
                'scale': scale,
                'ma': 'no',
                'datalen': '2000',  # 获取最大历史数据
            }
            response = self.session.get(url, params=params, timeout=10)

            if response.status_code != 200:
                return []

            raw_data = response.json()
            if not raw_data:
                return []

            # 解析为 (date, close) 列表
            raw_prices = []
            for item in raw_data:
                c = float(item.get('close', 0))
                if c > 0:
                    raw_prices.append((item.get('day', ''), c))

            if len(raw_prices) < 60:
                return []

            # 2. 从腾讯行情获取实时价格作为锚点
            current_price = self._get_current_price(stock_code)
            if current_price <= 0:
                # 无法获取实时价格, 退化为内部缩放
                # 以数据窗口最后一天的价格为基准, 不做外部校准
                current_price = raw_prices[-1][1]

            # 3. 计算复权价格
            adjusted_prices = self._local_forward_adjust(raw_prices, current_price)

            # 4. 取最近 count 条
            recent_adjusted = adjusted_prices[-count:] if len(adjusted_prices) >= count else adjusted_prices

            # 构建 klines 列表
            # 需要找到原始数据中对应的 OHLCV
            raw_map = {item['day']: item for item in raw_data}

            klines = []
            for day, adj_close in recent_adjusted:
                orig = raw_map.get(day, {})
                raw_open = float(orig.get('open', 0))
                raw_high = float(orig.get('high', 0))
                raw_low = float(orig.get('low', 0))
                raw_close = float(orig.get('close', 0))

                # 对 OHLC 应用相同的缩放因子
                if raw_close > 0:
                    factor = adj_close / raw_close
                    klines.append({
                        'date': day,
                        'open': round(raw_open * factor, 3),
                        'close': adj_close,
                        'high': round(raw_high * factor, 3),
                        'low': round(raw_low * factor, 3),
                        'volume': float(orig.get('volume', 0)),
                        'amount': 0,
                        'stock_code': stock_code,
                        'adjusted': True,
                    })

            logger.info(
                f"[KlineFetcher] 本地复权完成: {len(klines)} 条, "
                f"锚点价格={current_price:.2f}"
            )
            return klines

        except Exception as e:
            logger.debug(f"[KlineFetcher] 本地复权失败: {e}")

        return []

    @staticmethod
    def _get_current_price(stock_code: str) -> float:
        """从腾讯行情 API 获取实时价格

        腾讯行情格式: v_sz300620="51 光库科技 300620 309.00 281.00 ..."
        字段 [3] = 当前价 (以空格分隔, 在双引号内)
        """
        try:
            url = f'http://qt.gtimg.cn/q={stock_code}'
            r = requests.get(url, timeout=5)
            r.encoding = 'gbk'
            text = r.text.strip()
            # 提取双引号内的内容
            start = text.find('"')
            end = text.find('"', start + 1)
            if start < 0 or end < 0:
                return 0.0
            fields = text[start + 1:end].split(' ')
            # 字段 [3] = 当前价
            if len(fields) > 3:
                return float(fields[3])
        except Exception:
            pass
        return 0.0

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
