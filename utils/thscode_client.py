"""
utils/thscode_client.py — 同花顺 thscode 消歧客户端

从 hithink-finance-symbol Skill 借鉴的标的消歧流程:
  用户输入: "贵州茅台" / "600519" / "Kweichow Moutai"
    → symbol search → thscode (如 600519.SH)
    → 使用 thscode 查询所有数据源

thscode 格式:
  - SH (上海): 600519.SH, 688981.SH
  - SZ (深圳): 000001.SZ, 300620.SZ
  - BJ (北京): 830799.BJ

使用方式:
    from utils.thscode_client import ThsCodeClient

    client = ThsCodeClient()

    # 名称/代码 → thscode
    thscode = client.resolve("贵州茅台")
    # → "600519.SH"

    # 批量消歧
    codes = client.resolve_batch(["宁德时代", "中芯国际", "000001"])
    # → {"宁德时代": "300750.SZ", "中芯国际": "688981.SH", "000001": "0.000001"}

    # 缓存查询 (避免重复 API 调用)
    code = client.get_or_resolve("贵州茅台")
"""

import os
import json
import subprocess
import logging
import threading
from typing import Optional
from pathlib import Path
from datetime import datetime, timedelta

log = logging.getLogger(__name__)

# 缓存 TTL: 24 小时
CACHE_TTL = timedelta(hours=24)
CACHE_FILE = Path.home() / ".appdata" / "stock_analyzer" / "thscode_cache.json"


class ThsCodeClient:
    """
    同花顺 thscode 消歧客户端。

    通过 hithink-finance CLI 的 symbol search 命令将名称/ticker/thscode
    转换为统一的 thscode 格式。
    """

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or os.environ.get("HITHINK_FINANCE_API_KEY")
        self._cache: dict = {}
        self._cache_loaded = False
        self._lock = threading.Lock()
        self._load_cache()

    def _load_cache(self) -> None:
        """加载 thscode 缓存"""
        try:
            if CACHE_FILE.exists():
                with open(CACHE_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # 只保留未过期的缓存
                now = datetime.now()
                self._cache = {
                    k: v for k, v in data.items()
                    if now < datetime.fromisoformat(v.get("_expires", ""))
                }
                log.debug(f"[thscode] 加载缓存: {len(self._cache)} 条")
        except (OSError, json.JSONDecodeError) as e:
            log.warning(f"[thscode] 加载缓存失败: {e}")
            self._cache = {}

    def _save_cache(self) -> None:
        """保存 thscode 缓存"""
        try:
            CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            # 移除过期标记
            export = {k: {kk: vv for kk, vv in v.items() if kk != "_expires"}
                      for k, v in self._cache.items()}
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(export, f, ensure_ascii=False, indent=2)
        except OSError as e:
            log.warning(f"[thscode] 保存缓存失败: {e}")

    def _run_cli(self, args: list[str]) -> dict | None:
        """运行 hithink-finance CLI 命令"""
        if not self._api_key:
            log.error("[thscode] HITHINK_FINANCE_API_KEY 未配置")
            return None

        cmd = [
            "hithink-finance",
            "--api-key-stdin",
            "symbol", "search",
            "--format", "json",
        ] + args

        try:
            proc = subprocess.run(
                cmd,
                input=self._api_key,
                capture_output=True,
                text=True,
                timeout=15,
            )
            if proc.returncode != 0:
                result = json.loads(proc.stdout) if proc.stdout else {}
                log.warning(f"[thscode] CLI 错误: {result.get('error', {})}")
                return None
            result = json.loads(proc.stdout)
            if result.get("ok") is True:
                return result
            else:
                log.warning(f"[thscode] 业务错误: {result.get('error', {})}")
                return None
        except subprocess.TimeoutExpired:
            log.error("[thscode] CLI 超时 (15s)")
            return None
        except (json.JSONDecodeError, OSError) as e:
            log.error(f"[thscode] CLI 执行失败: {e}")
            return None

    def resolve(self, query: str, limit: int = 5) -> Optional[str]:
        """
        将名称/ticker/thscode 转换为唯一的 thscode。

        :param query: 查询词 (如 "贵州茅台", "600519", "Kweichow Moutai")
        :param limit: 最大返回结果数
        :return: thscode (如 "600519.SH") 或 None
        """
        # 先查缓存
        with self._lock:
            if query in self._cache:
                return self._cache[query]["thscode"]

        # CLI 查询
        result = self._run_cli(["--q", query, "--limit", str(limit)])
        if not result:
            return None

        items = result.get("data", {}).get("item", [])
        if not items:
            return None

        # 取最高分
        best = items[0]
        thscode = best.get("thscode")

        # 缓存
        with self._lock:
            self._cache[query] = {
                "thscode": thscode,
                "name": best.get("name", ""),
                "ticker": best.get("ticker", ""),
                "exchange": best.get("exchange", ""),
                "_expires": (datetime.now() + CACHE_TTL).isoformat(),
            }
            self._save_cache()

        return thscode

    def resolve_batch(self, queries: list[str]) -> dict[str, Optional[str]]:
        """
        批量消歧。

        :param queries: 查询词列表
        :return: {query: thscode} 字典
        """
        results = {}
        for q in queries:
            results[q] = self.resolve(q)
        return results

    def get_or_resolve(self, query: str) -> Optional[str]:
        """
        缓存优先的消歧: 缓存命中直接返回, 否则 CLI 查询。

        :param query: 查询词
        :return: thscode 或 None
        """
        return self.resolve(query)

    def clear_cache(self) -> None:
        """清除缓存"""
        with self._lock:
            self._cache.clear()
            try:
                CACHE_FILE.unlink(missing_ok=True)
            except OSError:
                pass
        log.info("[thscode] 缓存已清除")
