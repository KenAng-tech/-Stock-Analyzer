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

import os
import requests
import json
import time
from typing import Dict, Optional, List

import threading

from modules.dynamic_cache import cache
from modules.logger import logger
from utils.outbound_guard import host_throttle, cooldowns
from modules.kline_data_fetcher import KlineDataFetcher

# East Money API token (从环境变量读取，不设置则 None)
from modules.config_keys import get_eastmoney_ut
EASTMONEY_UT = get_eastmoney_ut()

# ThsData 主链熔断 (2026-09-19): 连续失败 10 次 → 600s 内跳过主链 (hithink 故障不拖慢主链)
_THS_GATE = {"fails": 0, "reset_at": 0.0}

# 模块级锁 (用于单例初始化)
_lock_init = threading.Lock()


class StockDataFetcher:
    """Fetches stock data from various APIs"""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # 防止重复初始化
        if hasattr(self, '_initialized') and self._initialized:
            return
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
        })
        self.cache = {}
        self.cache_ttl = 30  # seconds
        self.kline_fetcher = KlineDataFetcher()
        self._initialized = True

    def close(self):
        """关闭 HTTP 会话，释放文件描述符"""
        if self.session:
            self.session.close()
    
    def fetch_tencent_stock(self, stock_code: str) -> Dict:
        """Fetch stock data from Tencent API"""
        cache_key = f"tencent_{stock_code}"
        cached = cache.get(cache_key, category='realtime')
        if cached:
            return cached
        
        try:
            host_throttle.acquire('tencent', host_key='qt.gtimg.cn')  # P0-3: 同 host 节流 (2026-09-20)
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
            host_throttle.acquire('eastmoney', host_key='push2.eastmoney.com')  # P0-3: 同 host 节流 (2026-09-20)
            url = f"https://push2.eastmoney.com/api/qt/stock/get"
            params = {
                'secid': f"{market}.{code}",
                'fields': 'f43,f44,f45,f46,f47,f48,f49,f50,f57,f58,f170',
                'ut': EASTMONEY_UT
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

        Tencent API 字段说明 (2026-08-13 验证):
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
        - parts[46] = 流通市值 (circulating cap, 亿元) ⚠️ 不可靠
        - parts[47] = 总市值 (total market cap, 亿元) ⚠️ 不可靠
        - parts[48] = 市盈率 (PE ratio) ⚠️ 不可靠

        注意: 腾讯 API 的市值/PE 字段对部分股票(如中芯国际)返回严重错误数据.
        get_stock_info() 会使用 AkShare (东方财富) 数据覆盖这些字段.
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

        # 解析 year_high/low — 腾讯 API 返回的是未复权原始值
        # 当 current_price 远大于原始昨收时, 说明发生了除权, year_high/low 需要校正
        raw_year_high = float(parts[45]) if len(parts) > 45 else 0
        raw_year_low = float(parts[44]) if len(parts) > 44 else 0

        year_high = raw_year_high
        year_low = raw_year_low

        # 检测: 如果 year_high 和 year_low 都远高于当前价, 说明是未复权旧价格
        # 通过 K 线数据计算真实的前复权年高年低 (仅用最近 250 交易日 ≈ 1 年)
        # 需要检测并排除股票拆分/除权产生的异常低价
        if current_price > 0 and raw_year_low > current_price * 1.5:
            try:
                klines = self.kline_fetcher.fetch_kline(stock_code, 'daily', 250)
                if klines:
                    # 过滤掉远低于当前价的异常价格 (可能是未正确处理拆分的旧数据)
                    # 只保留 >= current_price * 0.5 的价格 (排除拆分前的低价)
                    threshold = current_price * 0.5
                    filtered_highs = [k['high'] for k in klines if k.get('high', 0) > threshold]
                    filtered_lows = [k['low'] for k in klines if k.get('low', 0) > threshold]
                    if filtered_highs and filtered_lows:
                        year_high = max(filtered_highs)
                        year_low = min(filtered_lows)
                    elif klines:
                        # 如果过滤后没有数据, 用收盘价范围
                        filtered_closes = [k['close'] for k in klines if k.get('close', 0) > threshold]
                        if filtered_closes:
                            year_high = max(filtered_closes)
                            year_low = min(filtered_closes)
            except Exception:
                pass  # K 线获取失败, 保持原始值

        # 校正 high < open 的异常情况 (腾讯 API 缓存问题)
        adj_open = float(parts[33]) if len(parts) > 33 else current_price
        adj_high = float(parts[34]) if len(parts) > 34 else current_price
        if adj_high < adj_open and adj_open > 0:
            # high < open 不可能, 用 open 作为 high
            adj_high = adj_open

        return {
            'source': 'tencent',
            'code': stock_code,
            'name': parts[1] if len(parts) > 1 else '',
            'price': current_price,
            'open': adj_open,
            'high': adj_high,
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
            # 腾讯 API 原始字段 (用于市值校验)
            '_raw_total_shares_yi': float(parts[62]) if len(parts) > 62 else 0,
            '_raw_circulating_shares_yi': float(parts[63]) if len(parts) > 63 else 0,
            'year_low': year_low,
            'year_high': year_high,
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
    
    def fetch_akshare_market_data(self, stock_code: str) -> Dict:
        """从 AkShare (东方财富) 获取可靠的市值和 PE 数据

        腾讯 API 的 parts[46/47/48] (市值/PE) 字段位置不稳定,
        对部分股票(如中芯国际)返回严重错误的数据.
        使用 AkShare 的 stock_zh_a_spot_em 作为市值数据源.
        """
        cache_key = f"akshare_market_{stock_code}"
        cached = cache.get(cache_key, category='realtime')
        if cached:
            return cached

        try:
            import akshare as ak
            import concurrent.futures
            # 超时保护: AkShare API 可能永久挂起
            # 2026-09-11: 原 with 块 exit 会 shutdown(wait=True) 无限等 worker (79858 探针挂 1.5d 根因)
            # → 改 shutdown(wait=False, cancel_futures=True): 主链永不被外网挂死, 仅进程退出时 join
            from modules.log_context import with_trace
            exe = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            try:
                future = exe.submit(with_trace(ak.stock_zh_a_spot_em))
                df = future.result(timeout=3.0)
            finally:
                exe.shutdown(wait=False, cancel_futures=True)
            row = df[df['代码'] == stock_code.replace('sh', '').replace('sz', '')]
            if row is not None and len(row) > 0:
                r = row.iloc[0]
                market_data = {
                    'source': 'akshare',
                    'pe': float(r.get('市盈率-动态', 0)),
                    'market_cap': float(r.get('总市值', 0)),  # 单位: 元
                    'circulating_cap': float(r.get('流通市值', 0)),  # 单位: 元
                }
                cache.set(cache_key, market_data, category='realtime')
                return market_data
        except Exception as e:
            logger.warning(f"AkShare 市值数据获取失败 ({stock_code}): {e}")

        return {}

    def fetch_akshare_cached_market_data(self, stock_code: str) -> Dict:
        """仅查询 AkShare 缓存, 不触发下载 (2026-08-17 新增)

        AkShare 断连接/下载全市场 A 股表很慢, get_stock_info 不应每次
        都等 3s+ 挂依赖. 仅在缓存已命中时返回, 否则空 {} (走腾讯校验).
        后台有 fetch_akshare_market_data 触发下载并填充缓存.
        """
        cache_key = f"akshare_market_{stock_code}"
        cached = cache.get(cache_key, category='realtime')
        if cached:
            return cached
        return {}

    def _validate_market_cap(self, data: Dict) -> Dict:
        """校验腾讯 API 返回的市值数据是否合理

        腾讯 API 的 parts[46/47] (市值/流通市值) 对部分股票返回错误数据.
        通过 price * _raw_total_shares_yi 计算期望市值进行校验.
        仅当 shares 数据合理 (ratio 0.5-2.0) 时才用计算值替换.
        否则保留 API 值但标记警告.
        """
        price = data.get('price', 0)
        reported_mc = data.get('market_cap', 0)
        raw_shares = data.get('_raw_total_shares_yi', 0)

        if price > 0 and raw_shares > 0 and reported_mc > 0:
            # 计算期望市值 (price * total_shares_亿股 = 亿元)
            expected_mc = price * raw_shares
            ratio = reported_mc / expected_mc if expected_mc > 0 else 1.0

            # 仅当 ratio 在合理范围 (0.5-2.0) 时才信任计算值
            if 0.5 <= ratio <= 2.0:
                data['market_cap'] = expected_mc
                data['market_source'] = 'tencent_recomputed'
                logger.info(
                    f"市值数据已确认 ({data.get('code')}): "
                    f"API={reported_mc:.2f} 亿, 计算值={expected_mc:.2f} 亿, "
                    f"比率={ratio:.2f} (可信)"
                )
            else:
                # ratio 异常: shares 数据本身不可靠 (如单位错误/负值)
                # 保留 API 值，标记警告
                data['_market_cap_warning'] = True
                logger.warning(
                    f"市值数据异常 ({data.get('code')}): "
                    f"API={reported_mc:.2f} 亿, 计算值≈{expected_mc:.2f} 亿, "
                    f"比率={ratio:.4f} (shares 数据不可信, 保留 API 值)"
                )
        elif raw_shares <= 0:
            # shares 数据为负或零，明显错误
            data['_market_cap_warning'] = True
            logger.warning(
                f"市值校验跳过 ({data.get('code')}): "
                f"shares 数据异常 (_raw_total_shares_yi={raw_shares})"
            )

        data['market_source'] = data.get('market_source', 'tencent')
        return data

    def get_stock_info(self, stock_code: str) -> Dict:
        """Get comprehensive stock information

        数据链 (2026-09-19 主链重排): ThsData primary (timeout 6s) → Tencent (10s) → 东财源 → AkShare cache
        - 主链命中: 专业源真值 (单位换算与旧链一致); 不可用自动降级副链 (never hangs)
        - 主链连续失败 10 次 → 熔断 600s (hithink 故障期不拖慢主链)
        """
        # ── 主链: ThsData 实时行情 (2026-09-19 双源互证 E2E 通过) ──
        primary = self._ths_primary(stock_code)
        if primary:
            ak = self.fetch_akshare_cached_market_data(stock_code)
            if ak:
                primary['pe'] = ak.get('pe', 0)
                primary['market_cap'] = ak.get('market_cap', 0) / 100000000
                primary['circulating_cap'] = ak.get('circulating_cap', 0) / 100000000
            primary['market_source'] = 'akshare_eastmoney' if ak else 'ths_data_primary'
            logger.debug(f"[ThsPrimary] {stock_code} 主链命中 (hithink 专业源)")
            return primary

        # ── 副链 (原链不动): Tencent 10s → AkShare 缓存命中才覆盖 ──
        # 1. 先获取腾讯实时行情 (价格/成交量等)
        tencent_data = self.fetch_tencent_stock(stock_code)
        # 2. AkShare 市值/PE 为可选增强: 仅当缓存已命中时覆盖 (避免每次触发全市场下载挂死)
        # 修复 2026-08-17: 回退无条件 AkShare 调用 — AkShare 断连接/慢会放大到每次请求
        akshare_market = self.fetch_akshare_cached_market_data(stock_code)

        if tencent_data:
            if akshare_market:
                # AkShare 的市值单位是 元, 需要转换为 亿元 (仅缓存命中, 不阻塞)
                tencent_data['pe'] = akshare_market.get('pe', tencent_data.get('pe', 0))
                tencent_data['market_cap'] = akshare_market.get('market_cap', 0) / 100000000
                tencent_data['circulating_cap'] = akshare_market.get('circulating_cap', 0) / 100000000
                tencent_data['market_source'] = 'akshare_eastmoney'
            else:
                # AkShare 未缓存/不可用, 用腾讯数据 + 校验 (不等待下载)
                tencent_data = self._validate_market_cap(tencent_data)
            return tencent_data

        # 腾讯失败, 尝试东方财富
        market = 0 if stock_code.startswith('sz') else 1
        code = stock_code[2:]
        em_data = self.fetch_eastmoney_stock(market, code)
        if em_data:
            akshare_market = self.fetch_akshare_cached_market_data(stock_code)
            if akshare_market:
                em_data['pe'] = akshare_market.get('pe', em_data.get('pe', 0))
                em_data['market_cap'] = akshare_market.get('market_cap', 0) / 100000000
                em_data['circulating_cap'] = akshare_market.get('circulating_cap', 0) / 100000000
                em_data['market_source'] = 'akshare_eastmoney'
            else:
                em_data['market_source'] = 'eastmoney'
            return em_data

        return None
    
    def _fetch_hithink_snapshot(self, stock_code: str, timeout: int = 6) -> Dict:
        """hithink CLI 实时快照 (2026-09-19 并行源, 仅观察标记不参与决策)。

        :param stock_code: 腾讯格式代码 (如 sz300620)
        :param timeout: CLI 超时秒数 (短超时防拖慢主降级链)
        :return: snapshot dict 或 {} (永不抛异常)
        """
        try:
            if len(stock_code) != 8 or stock_code[:2].lower() not in ("sh", "sz", "bj"):
                return {}
            thscode = f"{stock_code[2:]}.{stock_code[:2].upper()}"
            from utils.ths_data_provider import get_ths_data_provider
            provider = get_ths_data_provider()
            item = provider.get_realtime(thscode, timeout=timeout)
            if item and item.get("last_price"):
                return item
            return {}
        except Exception as e:
            logger.debug(f"hithink snapshot 获取失败 {stock_code}: {e}")
            return {}

    def _ths_primary(self, stock_code: str, timeout: int = 6) -> Optional[Dict]:
        """ThsData 主链: 实时行情快照 → 主链同构 dict (2026-09-19 主链重排)。

        单位换算 (E2E 双源互证 2026-09-19): volume 股→手 (÷100), turnover 元→万元 (÷1e4)。
        熔断: 连续失败 ≥10 且 600s 窗口内 → 跳过主链 (hithink 故障期); 窗口过自动半开重试。
        :return: 同构 dict 或 None (降级副链, never raises)
        """
        now = time.time()
        if _THS_GATE["fails"] >= 10 and now - _THS_GATE["reset_at"] < 600:
            return None
        try:
            if len(stock_code) < 6 or stock_code[:2].lower() not in ("sh", "sz", "bj"):
                return None
            from utils.ths_data_provider import get_ths_data_provider
            thscode = f"{stock_code[2:]}.{stock_code[:2].upper()}"
            snap = get_ths_data_provider().get_realtime(thscode, timeout=timeout)
            if not snap or not snap.get("last_price"):
                _THS_GATE["fails"] += 1
                _THS_GATE["reset_at"] = now
                if _THS_GATE["fails"] in (1, 10):
                    logger.warning(f"[ThsPrimary] 主链故障 (连续{_THS_GATE['fails']}次) → 副链: {stock_code}")
                return None
            _THS_GATE["fails"] = 0
            price = float(snap.get("last_price") or 0)
            vol_shares = float(snap.get("volume") or 0)
            amt_yuan = float(snap.get("turnover") or 0)
            return {
                'source': 'ths_data_primary',
                'code': stock_code,
                'name': snap.get('name') or stock_code[2:],
                'price': price,
                'open': float(snap.get("open_price") or 0),
                'high': float(snap.get("high_price") or 0),
                'low': float(snap.get("low_price") or 0),
                'close': price,
                'prev_close': float(snap.get("prev_price") or 0) or price,
                'change': float(snap.get("price_change") or 0),
                'change_pct': float(snap.get("price_change_ratio_pct") or 0),
                'volume': int(vol_shares / 100),      # 股→手 (主链旧单位保持)
                'amount': amt_yuan / 10000,           # 元→万元 (主链旧单位保持)
                'turnover': 0,                        # 换手率: 腾讯 API 独有字段, hithink 无
                'pe': 0, 'market_cap': 0, 'circulating_cap': 0,  # AkShare 缓存命中时注入
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            }
        except Exception as e:
            _THS_GATE["fails"] += 1
            _THS_GATE["reset_at"] = now
            logger.warning(f"[ThsPrimary] 主链异常 (→ 副链): {stock_code}: {e}")
            return None

    def get_kline_data(self, stock_code: str, period: str = 'daily',
                       count: int = 250) -> List[Dict]:
        """获取历史K线数据

        数据链 (2026-09-19 主链重排): daily → DuckDB v_daily_qfq (毫秒级本地, qfq 语义
        与 AKShare 一致, 双跑 E2E 价格/量级互证) → 原 4 源降级链 (AKShare/东财/新浪, 页面爬取)。
        weekly/monthly → 原链 (DuckDB 仅日K)。单位与旧链一致 (股), 消费者统计特征零漂移。
        """
        cache_key = f"kline_{stock_code}_{period}_{count}"
        cached = cache.get(cache_key, category='kline')
        if cached:
            return cached

        klines = None
        if period == 'daily' and len(stock_code) >= 6 and stock_code[:2].lower() in ("sh", "sz", "bj"):
            try:
                from utils.ths_data_provider import get_ths_data_provider
                thscode = f"{stock_code[2:]}.{stock_code[:2].upper()}"
                raw = get_ths_data_provider().get_klines_qfq(thscode, count)
                if raw:
                    klines = [{**k, "stock_code": stock_code, "adjusted": True,
                               "source": "duckdb_qfq"} for k in raw][-count:]
            except Exception as e:
                logger.debug(f"get_kline_data duckdb 主链失败: {stock_code} {e}")

        if not klines:
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
    
    def get_stock_news(self, stock_code: str) -> List[Dict]:
        """
        获取股票相关新闻 (东方财富)

        Args:
            stock_code: 股票代码

        Returns:
            List of news items with title, date, etc.
        """
        cache_key = f"news_{stock_code}"
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            if time.time() - cached.get('time', 0) < 300:  # 5 min cache
                return cached['data']

        result: List[Dict] = []
        try:
            # 东方财富新闻 API — ⚠ 2026-09-11 尸检: 此旧版参数+旧版解析键
            # (data.list) 已双死, 恒 200 空 body = 链死非链慢 (llm-sentiment
            # n:0 / 共识 sentiment 票 0.3 即其产物)。保留调用仅为"接口若复活"
            # 兜底; 真正取数靠下方 Bypass (决策链 B 后续, 实测 10 条真新闻)。
            news_url = f"https://search-api-web.eastmoney.com/search/jsonp?cb=jQuery&type=0&param=%7B%22uid%22:%22%22,%22keyword%22:%22{stock_code}%22,%22page%22:1,%22num%22:20,%22product%22:%22news%22,%22clientin%22:%22web%22,%22clienttype%22:%22web%22,%22clientversion%22:%2210.58%22,%22ispartial%22:true,%22searchrange%22:%22%22,%22order%22:%220%22,%22reqtrace%22:%22%22%7D&_=1720000000000"
            import urllib.request
            req = urllib.request.Request(news_url, headers={'Referer': 'https://guba.eastmoney.com/'})
            with urllib.request.urlopen(req, timeout=10) as resp:
                raw = resp.read().decode('utf-8', errors='ignore')
                # 去除 JSONP callback wrapper
                if raw.startswith('jQuery'):
                    raw = raw[raw.index('(') + 1:raw.rindex(')')]
                data = json.loads(raw)
                news_items = data.get('data', {}).get('list', [])
                result = []
                for item in news_items[:20]:
                    result.append({
                        'title': item.get('title', ''),
                        'date': item.get('date', ''),
                        'url': item.get('url', ''),
                    })
        except Exception as e:
            logger.debug(f"[DataFetcher] 获取新闻失败: {e}")
            result = []

        if not result:
            # 决策链 B 后续 (2026-09-11): 主链空 → search-api-web 新版参数 JSONP
            # 旁路二次取数 (akshare 上游同款参数; <em> 剥标签自实现避开
            # akshare 在 Py3.14 的 re 兼容炸弹)。数据源故障 → [] 不拖决策链。
            try:
                from modules.eastmoney_news_fetcher import fetch_news_eastmoney
                result = fetch_news_eastmoney(stock_code)
            except Exception as e:
                logger.debug(f"[DataFetcher] 新闻旁路也空: {e}")
                result = []

        self.cache[cache_key] = {'data': result, 'time': time.time()}
        return result

    def get_stock_posts(self, stock_code: str) -> List[Dict]:
        """
        获取股吧帖子 (东方财富)

        Args:
            stock_code: 股票代码

        Returns:
            List of post items with title, content, etc.
        """
        cache_key = f"posts_{stock_code}"
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            if time.time() - cached.get('time', 0) < 300:
                return cached['data']

        try:
            # 东方财富股吧帖子 API
            posts_url = f"https://guba.eastmoney.com/list,{stock_code}.html"
            import urllib.request
            req = urllib.request.Request(posts_url, headers={'Referer': 'https://guba.eastmoney.com/'})
            with urllib.request.urlopen(req, timeout=10) as resp:
                raw = resp.read().decode('utf-8', errors='ignore')

                # 提取帖子标题 (简化解析)
                import re
                titles = re.findall(r'<a[^>]+class="b[^"]*"[^>]*>([^<]+)</a>', raw)
                result = [{'title': t.strip(), 'content': ''} for t in titles[:20]]
                self.cache[cache_key] = {'data': result, 'time': time.time()}
                return result
        except Exception as e:
            logger.debug(f"[DataFetcher] 获取股吧帖子失败: {e}")
            return []

    def clear_cache(self):
        """Clear all caches"""
        self.cache.clear()
        cache.cleanup()
