"""
utils/ths_data_provider.py — 同花顺数据提供者

基于 hithink-finance CLI (0.1.12) 的统一数据访问层。
替代 AKShare 作为主数据源，提供：
  - 本地 DuckDB 查询 (K 线、复权因子)
  - 远端 API 实时行情
  - 财务报表 (利润表/资产负债表/现金流量表)
  - 估值快照 (PE/PB/PS/PCF)
  - 特色数据 (龙虎榜/涨停池/热股榜)

使用方式:
    from utils.ths_data_provider import ThsDataProvider

    provider = ThsDataProvider()

    # 实时行情
    snapshot = provider.get_realtime("300620.SZ")

    # 历史 K 线 (本地 DuckDB)
    klines = provider.get_klines("300620.SZ", start="2025-01-01", end="2026-09-17")

    # 财务报表
    income = provider.get_income_statement("300620.SZ")

    # 估值
    valuation = provider.get_valuation("300620.SZ")
"""

import subprocess
import json
import logging
import os
import random
import threading
import time
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta

log = logging.getLogger(__name__)

# CLI 路径 (优先环境变量，其次默认路径)
CLI_PATH = os.environ.get("HITHINK_FINANCE_CLI", "~/.local/bin/hithink-finance")

# 2026-09-21: 可重试上游错误码 (上游 global-rules 契约: 429/502/503/504 有界重试)
_RETRYABLE_STATUS = ('UPSTREAM_HTTP_429', 'UPSTREAM_HTTP_502',
                    'UPSTREAM_HTTP_503', 'UPSTREAM_HTTP_504')
_CLI_RETRIES = 1  # 单发调用超时/可重试错误时最多再试 1 次 (退避 1.0s+抖动)

# 2026-09-19: 本地 DuckDB 数据层 (5564 标的 / 1029 万行日K, 与 remote history 同构)
_DUCKDB_PATH = os.path.expanduser(
    "~/Library/Application Support/hithink-finance/data/market.duckdb")
_db_lock = threading.Lock()  # 防并发查库 (Flask threading 模式)


def _resolve_cli() -> str:
    """解析 CLI 实际路径"""
    path = CLI_PATH.replace("~", str(os.path.expanduser("~")))
    if os.path.isfile(path):
        return path
    # Fallback: 从 PATH 查找
    for dir_path in os.environ.get("PATH", "").split(os.pathsep):
        candidate = os.path.join(dir_path, "hithink-finance")
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        f"hithink-finance CLI not found at {path} or in PATH. "
        "Install via: npm install -g @hithink-tech/hithink-finance-cli"
    )


def _run_cli(args: List[str], timeout: int = 30,
             retries: int = _CLI_RETRIES) -> Optional[Dict[str, Any]]:
    """
    执行 hithink-finance CLI 命令 (2026-09-21 升级: 有界重试 + 错误码分类)。

    上游 global-rules 契约: HTTP 429/502/503/504 优先于业务错误信封做
    有界重试 (UPSTREAM_HTTP_<status> 错误码); 重试耗尽返回 None (非断链,
    由调用方走降级)。重试间隔 1.0s + U(0,0.3) 抖动 (同 K 线链退避形态)。

    :param args: 命令参数列表 (不含 CLI 路径)
    :param timeout: 单次尝试超时秒数 (重试时总耗时 ≤ timeout×(1+retries)+1)
    :param retries: 可重试错误 (超时/429/502/503/504) 的追加尝试数
    :return: JSON 解析后的 dict 或 None
    """
    cli = _resolve_cli()
    cmd = [cli, *args, "--format", "json"]
    attempts = 1 + max(0, retries)
    short_cmd = ' '.join(cmd[:3])

    for attempt in range(attempts):
        if attempt:
            time.sleep(1.0 + random.uniform(0, 0.3))
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if result.returncode != 0:
                err_snip = (result.stderr or '')[:200]
                retryable = any(code in err_snip for code in _RETRYABLE_STATUS)
                if attempt + 1 < attempts and retryable:
                    log.warning(f"[ThsDataProvider] CLI 可重试错误码 {result.returncode} "
                                f"(重试 {attempt + 1}/{retries}): {short_cmd}")
                    continue
                log.warning(f"[ThsDataProvider] CLI 返回非零码 {result.returncode}: {err_snip}")
                return None

            data = json.loads(result.stdout)
            if not data.get("ok"):
                err = data.get("error") or {}
                code = str(err.get("code", ""))
                if attempt + 1 < attempts and any(c in code for c in _RETRYABLE_STATUS):
                    log.warning(f"[ThsDataProvider] CLI 可重试 ok=false {code} "
                                f"(重试 {attempt + 1}/{retries}): {short_cmd}")
                    continue
                # 错误码分类入日志 (error.code/category 供链路区分, 不再黑盒)
                log.warning(f"[ThsDataProvider] CLI ok=false: code={code or '?'} "
                            f"category={err.get('category', '?')} hint={str(err.get('hint', ''))[:80]}")
                return None

            return data

        except subprocess.TimeoutExpired:
            if attempt + 1 < attempts:
                log.warning(f"[ThsDataProvider] CLI 超时 {timeout}s "
                            f"(重试 {attempt + 1}/{retries}): {short_cmd}")
                continue
            log.error(f"[ThsDataProvider] CLI 超时 ({timeout}s×{attempts} 尝试): {' '.join(cmd)}")
            return None
        except json.JSONDecodeError as e:
            log.error(f"[ThsDataProvider] CLI JSON 解析失败: {e}")
            return None
        except FileNotFoundError as e:
            log.error(f"[ThsDataProvider] CLI 未找到: {e}")
            return None
        except Exception as e:
            log.error(f"[ThsDataProvider] CLI 未知错误: {e}")
            return None

    return None


class ThsDataProvider:
    """同花顺数据提供者 — 统一访问 DuckDB + 远端 API"""

    def __init__(self):
        self._cli = _resolve_cli()
        self._cache: Dict[str, tuple] = {}  # key -> (data, timestamp)
        self._cache_ttl = 60  # 缓存 60 秒

    def _cached(self, key: str, fetch_fn, ttl: int = None) -> Optional[Any]:
        """简单内存缓存"""
        ttl = self._cache_ttl if ttl is None else ttl
        now = datetime.now()
        if key in self._cache:
            data, ts = self._cache[key]
            # .seconds 在间隔≥24h 时返回"当日秒数"(跨天误判永不过期), 用 total_seconds()
            if ttl > 0 and (now - ts).total_seconds() < ttl:
                return data
        data = fetch_fn()
        if data is not None:
            self._cache[key] = (data, now)
        return data

    # ── 实时行情 ──────────────────────────────────────────────

    def get_realtime(self, thscode: str, timeout: int = 30) -> Optional[Dict[str, Any]]:
        """
        获取实时/最近行情快照 (远端 API)。

        :param thscode: 同花顺代码 (如 "300620.SZ")
        :param timeout: CLI 超时秒数 (主链并行观察用短超时, 避免拖慢降级链)
        :return: {thscode, ticker, name, last_price, price_change, ...} 或 None
        """
        def _fetch():
            data = _run_cli(["market", "snapshot", "--thscodes", thscode],
                            timeout=timeout)
            if data and data.get("data", {}).get("item"):
                item = data["data"]["item"][0]
                # 快照返回不含 name，从 thscode 提取 ticker 作为名称占位
                item["name"] = item.get("name", "") or item.get("ticker", thscode)
                item["timestamp"] = data.get("meta", {}).get("request_id", "")
                return item
            return None

        return self._cached(f"realtime:{thscode}", _fetch, ttl=30)

    def get_klines_qfq(self, thscode: str, limit: int = 250) -> Optional[List[Dict[str, Any]]]:
        """本地 v_daily_qfq 视图取最近 limit 条 (2026-09-19, 与 AKShare qfq 同语义)。

        :return: 按时间正序的 K线 list (date/open/high/low/close/volume/amount), 失败 None
        """
        if not os.path.isfile(_DUCKDB_PATH):
            return None
        if not _db_lock.acquire(timeout=8):
            log.warning(f"[ThsDataProvider] DuckDB 锁超时 (qfq): {thscode}")
            return None
        try:
            import duckdb
            conn = duckdb.connect(_DUCKDB_PATH, read_only=True)
            try:
                rows = conn.execute(
                    "SELECT CAST(date AS VARCHAR), open, high, low, close, "
                    "volume, amount FROM v_daily_qfq WHERE thscode=? "
                    "ORDER BY date DESC LIMIT ?",
                    [thscode, limit + 5]).fetchall()
            finally:
                conn.close()
            if not rows:
                return None
            out = [{
                "date": r[0], "open": r[1], "high": r[2], "low": r[3],
                "close": r[4], "volume": r[5], "amount": r[6],
            } for r in reversed(rows)]
            return out
        except Exception as e:
            log.warning(f"[ThsDataProvider] v_daily_qfq 查询失败: {e}")
            return None
        finally:
            _db_lock.release()

    def get_klines_batch(self, thscodes: List[str], start: str = None,
                         end: str = None, timeout: int = 15) -> Dict[str, List[Dict[str, Any]]]:
        """批量取 K 线 (2026-09-19: 本地 DuckDB 单 SQL, 无 API 成本/无 subprocess)。

        :return: {thscode: [kline dict...]}, 缺数据的 code 不含; 失败返 {} (never hangs)
        """
        if not thscodes or not os.path.isfile(_DUCKDB_PATH):
            return {}
        if not _db_lock.acquire(timeout=timeout):
            log.warning(f"[ThsDataProvider] DuckDB 批量锁超时 ({len(thscodes)} codes)")
            return {}
        try:
            import duckdb
            placeholders = ",".join("?" for _ in thscodes)
            sql = (f"SELECT thscode, CAST(date AS VARCHAR) AS date, open, high, low, "
                   f"close, prev_close, volume, amount FROM v_daily "
                   f"WHERE thscode IN ({placeholders}) AND date>=? AND date<=? "
                   f"ORDER BY thscode, date")
            end_d = end or datetime.now().strftime("%Y-%m-%d")
            start_d = start or "1990-01-01"
            conn = duckdb.connect(_DUCKDB_PATH, read_only=True)
            try:
                rows = conn.execute(sql, [*thscodes, start_d, end_d]).fetchall()
            finally:
                conn.close()
            out: Dict[str, List[Dict[str, Any]]] = {}
            for r in rows:
                out.setdefault(r[0], []).append(dict(zip(
                    ("thscode", "date", "open", "high", "low",
                     "close", "prev_close", "volume", "amount"), r)))
            return out
        except Exception as e:
            log.warning(f"[ThsDataProvider] DuckDB 批量查询失败: {e}")
            return {}
        finally:
            _db_lock.release()

    def get_realtime_batch(self, thscodes: List[str]) -> List[Dict[str, Any]]:
        """批量获取实时行情 (最多 100 个)"""
        codes_str = ",".join(thscodes[:100])
        data = _run_cli(["market", "snapshot", "--thscodes", codes_str])
        if data and data.get("data", {}).get("item"):
            for item in data["data"]["item"]:
                item["name"] = item.get("name", "") or item.get("ticker", "")
            return data["data"]["item"]
        return []

    # ── 历史 K 线 ──────────────────────────────────────────────

    def get_klines(self, thscode: str, start: str = None, end: str = None,
                   frequency: str = "daily") -> Optional[List[Dict[str, Any]]]:
        """
        获取历史 K 线数据。

        策略:
        1. 先查本地 DuckDB (快, 毫秒级, 2026-9-19 实装)
        2. 本地无数据/锁超时/异常 → 远端 API fallback (never hangs)

        :param thscode: 同花顺代码
        :param start: 开始日期 (YYYY-MM-DD)
        :param end: 结束日期 (YYYY-MM-DD)
        :param frequency: K 线周期 (daily/weekly/monthly)
        :return: K 线列表 或 None; self.last_source 记录命中链
        """
        t0 = time.perf_counter()
        klines = self._get_klines_local(thscode, start, end)
        if klines:
            self.last_source = f"duckdb_local:{time.perf_counter() - t0:.3f}s"
            return klines

        # Fallback: 远端 API
        data = self._get_klines_remote(thscode, start, end)
        if data:
            self.last_source = f"hithink_remote:{time.perf_counter() - t0:.1f}s"
        return data

    def _get_klines_local(self, thscode: str, start: str = None, end: str = None) -> Optional[List[Dict]]:
        """本地 DuckDB v_daily 毫秒查询 (2026-09-19 实装)。

        断链安全: 无锁等待死锁 (acquire timeout 8s), 任何异常 → None → remote fallback。
        """
        if not os.path.isfile(_DUCKDB_PATH):
            return None
        if not _db_lock.acquire(timeout=8):
            log.warning(f"[ThsDataProvider] DuckDB 锁超时 8s (并发查库): {thscode}")
            return None
        try:
            import duckdb
            sql = ("SELECT thscode, CAST(date AS VARCHAR) AS date, open, high, low, "
                   "close, prev_close, volume, amount FROM v_daily "
                   "WHERE thscode=? AND date>=? AND date<=? ORDER BY date")
            end_d = end or datetime.now().strftime("%Y-%m-%d")
            start_d = start or "1990-01-01"
            conn = duckdb.connect(_DUCKDB_PATH, read_only=True)
            try:
                rows = conn.execute(sql, [thscode, start_d, end_d]).fetchall()
            finally:
                conn.close()
            if rows:
                return [dict(zip(("thscode", "date", "open", "high", "low",
                                   "close", "prev_close", "volume", "amount"), r))
                        for r in rows]
            return None
        except Exception as e:
            log.warning(f"[ThsDataProvider] DuckDB 本地查询失败 (降级 remote): {e}")
            return None
        finally:
            _db_lock.release()

    def _get_klines_remote(self, thscode: str, start: str = None, end: str = None) -> Optional[List[Dict]]:
        """从远端 API 获取 K 线 (使用毫秒时间戳)"""
        from datetime import datetime
        args = ["market", "history", "--thscode", thscode]

        if start:
            # 日期字符串 → 毫秒时间戳
            try:
                dt = datetime.strptime(start, "%Y-%m-%d")
                args.extend(["--start-ms", str(int(dt.timestamp() * 1000))])
            except ValueError:
                pass

        if end:
            try:
                dt = datetime.strptime(end, "%Y-%m-%d")
                args.extend(["--end-ms", str(int(dt.timestamp() * 1000))])
            except ValueError:
                pass

        data = _run_cli(args)
        if data and data.get("ok"):
            # data 可能是 list (直接返回) 或 dict (嵌套在 items 中)
            raw = data.get("data", [])
            if isinstance(raw, list):
                return raw
            elif isinstance(raw, dict):
                return raw.get("items", [])
        return None

    # ── 财务报表 ──────────────────────────────────────────────

    def get_income_statement(self, thscode: str, limit: int = 4) -> Optional[List[Dict]]:
        """获取利润表"""
        data = _run_cli(["financials", "income", "--thscode", thscode])
        if data and data.get("data", {}).get("items"):
            return data["data"]["items"][:limit]
        return None

    def get_balance_sheet(self, thscode: str, limit: int = 4) -> Optional[List[Dict]]:
        """获取资产负债表"""
        data = _run_cli(["financials", "balance-sheet", "--thscode", thscode])
        if data and data.get("data", {}).get("items"):
            return data["data"]["items"][:limit]
        return None

    def get_cash_flow(self, thscode: str, limit: int = 4) -> Optional[List[Dict]]:
        """获取现金流量表"""
        data = _run_cli(["financials", "cash-flow", "--thscode", thscode])
        if data and data.get("data", {}).get("items"):
            return data["data"]["items"][:limit]
        return None

    def get_financial_indicators(self, thscode: str) -> Optional[Dict]:
        """获取财务指标 (ROE/毛利率/净利率等)"""
        data = _run_cli(["financials", "indicators", "--thscode", thscode])
        if data and data.get("data", {}).get("items"):
            return data["data"]["items"][0]
        return None

    # ── 估值 ──────────────────────────────────────────────────

    def get_valuation(self, thscode: str) -> Optional[Dict]:
        """
        获取估值快照 (PE/PB/PS/PCF)。

        :return: {pe_ttm, pb_mrq, ps_ttm, pcf_ttm} 或 None
        """
        data = _run_cli(["valuation", "snapshot", "--thscodes", thscode])
        if data and data.get("data", {}).get("item"):
            return data["data"]["item"][0]
        return None

    def get_valuation_batch(self, thscodes: List[str]) -> List[Dict]:
        """批量获取估值 (最多 100 个)"""
        codes_str = ",".join(thscodes[:100])
        data = _run_cli(["valuation", "snapshot", "--thscodes", codes_str])
        if data and data.get("data", {}).get("item"):
            return data["data"]["item"]
        return []

    # ── 特色数据 ──────────────────────────────────────────────

    def get_dragon_tiger(self, limit: int = 20) -> Optional[List[Dict]]:
        """获取龙虎榜数据"""
        data = _run_cli(["special", "dragon-tiger"])
        if data and data.get("data", {}).get("items"):
            return data["data"]["items"][:limit]
        return None

    def get_limit_up_pool(self, date: str = None) -> Optional[List[Dict]]:
        """获取涨停池"""
        args = ["special", "limit-up-pool"]
        if date:
            args.extend(["--date", date])
        data = _run_cli(args)
        if data and data.get("data", {}).get("items"):
            return data["data"]["items"]
        return None

    def get_hot_stocks(self, limit: int = 20) -> Optional[List[Dict]]:
        """获取热股榜"""
        data = _run_cli(["special", "hot-stock"])
        if data and data.get("data", {}).get("items"):
            return data["data"]["items"][:limit]
        return None

    # ── 标的消歧 ──────────────────────────────────────────────

    def resolve_symbol(self, query: str, limit: int = 5) -> Optional[List[Dict]]:
        """
        股票名称/代码搜索消歧。

        :param query: 搜索关键词 (如 "宁德时代" 或 "300620")
        :param limit: 返回数量
        :return: [{thscode, ticker, name, exchange, asset_type}, ...]
        """
        data = _run_cli(["symbol", "search", "--q", query])
        if data and data.get("data", {}).get("item"):
            return data["data"]["item"][:limit]
        return None

    # ── 交易日历 ──────────────────────────────────────────────

    def get_trading_calendar(self, year: int = None) -> Optional[List[str]]:
        """获取交易日历"""
        if year is None:
            year = datetime.now().year
        data = _run_cli(["market", "calendar"])
        if data and data.get("data", {}).get("days"):
            return data["data"]["days"]
        return None

    # ── 复权因子 ──────────────────────────────────────────────

    def get_adjustment_factors(self, thscode: str, start: str = None,
                               end: str = None) -> Optional[List[Dict]]:
        """获取复权因子 (本地 DuckDB)"""
        args = ["market", "adjustment-factors", "--thscode", thscode]
        if start:
            args.extend(["--start", start])
        if end:
            args.extend(["--end", end])
        data = _run_cli(args)
        if data and data.get("data", {}).get("items"):
            return data["data"]["items"]
        return None

    # ── 数据状态 ──────────────────────────────────────────────

    def get_data_status(self) -> Optional[Dict]:
        """获取本地 DuckDB 状态"""
        data = _run_cli(["data", "status"])
        return data.get("data") if data else None

    def sync_data(self, start: str = None, end: str = None) -> bool:
        """
        同步市场数据到本地 DuckDB。

        :param start: 开始日期
        :param end: 结束日期
        :return: 是否成功
        """
        args = ["data", "sync"]
        if start:
            args.extend(["--start", start])
        if end:
            args.extend(["--end", end])
        result = _run_cli(args, timeout=300)
        return result is not None and result.get("ok")

    def clear_cache(self):
        """清除内存缓存"""
        self._cache.clear()
        log.info("[ThsDataProvider] 缓存已清除")


# ── 模块级单例 ──────────────────────────────────────────────────

_provider = None


def get_ths_data_provider() -> ThsDataProvider:
    """获取 ThsDataProvider 单例"""
    global _provider
    if _provider is None:
        _provider = ThsDataProvider()
    return _provider
