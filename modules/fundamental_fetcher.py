#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
基本面数据获取模块 — P0 修复 (2026-06-04)

从 AKShare 获取真实财务数据，替代 analysis_engine.py 中的硬编码值。

数据来源:
- 利润表/资产负债表/现金流量表: AKShare stock_financial_abstract_ths
- 个股信息: AKShare stock_individual_info_em (备用)
- 新浪/腾讯行情: 已有 data_fetcher.py

用法:
    from modules.fundamental_fetcher import FundamentalFetcher
    fetcher = FundamentalFetcher()
    data = fetcher.get_financial_data("300620")
"""

import re
import time
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime

from modules.dynamic_cache import cache
from modules.logger import logger

# 缓存 TTL（基本面数据变化慢，缓存 1 小时）
FUNDAMENTAL_CACHE_TTL = 3600


def _parse_percent(s: str) -> float:
    """解析百分比字符串，如 '163.76%' -> 163.76, 'False' -> 0.0"""
    if not s or s == "False" or s == "true":
        return 0.0
    s = s.strip()
    m = re.match(r"([-+]?\d+\.?\d*)\s*%", s)
    if m:
        return float(m.group(1))
    # 处理 "1.31亿" 格式
    m2 = re.match(r"([-+]?\d+\.?\d*)\s*(亿|万)", s)
    if m2:
        val = float(m2.group(1))
        if m2.group(2) == "亿":
            return val * 10000
        return val
    try:
        return float(s)
    except (ValueError, TypeError):
        return 0.0


def _parse_money(s: str) -> float:
    """解析金额字符串，如 '1.77亿' -> 177000000"""
    if not s or s == "False":
        return 0.0
    s = s.strip()
    m = re.match(r"([-+]?\d+\.?\d*)\s*(亿|万)", s)
    if m:
        val = float(m.group(1))
        if m.group(2) == "亿":
            return val * 100000000
        return val * 10000
    try:
        return float(s)
    except (ValueError, TypeError):
        return 0.0


class FundamentalFetcher:
    """基本面数据获取器 — 使用 AKShare 获取真实财务数据"""

    def __init__(self):
        self._akshare_available = False
        self._import_akshare()

    def _import_akshare(self):
        """延迟导入 AKShare"""
        try:
            import akshare as ak  # noqa: F811
            self._ak = ak
            self._akshare_available = True
            logger.info("[FundamentalFetcher] AKShare 加载成功")
        except ImportError:
            self._akshare_available = False
            logger.warning("[FundamentalFetcher] AKShare 未安装，基本面数据将使用 fallback")

    def get_financial_data(self, stock_code: str, stock_name: str = "") -> Dict:
        """
        获取股票完整财务数据

        Args:
            stock_code: 股票代码（不带市场前缀，如 '300620'）
            stock_name: 股票名称（用于日志）

        Returns:
            财务数据字典
        """
        cache_key = f"fundamental_{stock_code}"
        cached = cache.get(cache_key, category='fundamental')
        if cached:
            return cached

        if not self._akshare_available:
            data = self._fallback_financial_data(stock_code, stock_name)
            cache.set(cache_key, data, category='fundamental', ttl=FUNDAMENTAL_CACHE_TTL)
            return data

        try:
            # 获取利润表数据（所有报表共用同一接口）
            df = self._ak.stock_financial_abstract_ths(symbol=stock_code, indicator="利润表")

            if df is None or len(df) == 0:
                raise ValueError("AKShare 返回空数据")

            # 取最新一期数据（最后一行）
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) >= 2 else None

            financial_data = self._parse_financial_row(latest, prev)

            # 计算趋势指标
            financial_data["trends"] = self._calculate_trends(df)

            cache.set(cache_key, financial_data, category='fundamental', ttl=FUNDAMENTAL_CACHE_TTL)
            logger.info(
                f"[FundamentalFetcher] 财务数据获取成功: {stock_code} {stock_name}",
                extra={"code": stock_code, "roe": financial_data.get("roe")},
            )
            return financial_data

        except Exception as e:
            logger.error(f"[FundamentalFetcher] AKShare 获取失败: {e}，使用 fallback")
            data = self._fallback_financial_data(stock_code, stock_name)
            cache.set(cache_key, data, category='fundamental', ttl=FUNDAMENTAL_CACHE_TTL)
            return data

    def _parse_financial_row(self, latest, prev) -> Dict:
        """解析单行财务数据"""
        # 营收和利润
        revenue = _parse_money(str(latest.get("营业总收入", 0)))
        net_profit = _parse_money(str(latest.get("净利润", 0)))
        deducted_net_profit = _parse_money(str(latest.get("扣非净利润", 0)))

        # 增长率
        revenue_growth = _parse_percent(str(latest.get("营业总收入同比增长率", 0)))
        profit_growth = _parse_percent(str(latest.get("净利润同比增长率", 0)))
        deducted_profit_growth = _parse_percent(str(latest.get("扣非净利润同比增长率", 0)))

        # 盈利能力
        gross_margin = _parse_percent(str(latest.get("销售毛利率", 0)))
        net_margin = _parse_percent(str(latest.get("销售净利率", 0)))
        roe = _parse_percent(str(latest.get("净资产收益率-摊薄", 0)))
        roe_weighted = _parse_percent(str(latest.get("净资产收益率", 0)))

        # 偿债能力
        debt_ratio = _parse_percent(str(latest.get("资产负债率", 0)))
        current_ratio = _parse_percent(str(latest.get("流动比率", 0)))
        quick_ratio = _parse_percent(str(latest.get("速动比率", 0)))

        # 每股数据
        eps = _parse_percent(str(latest.get("基本每股收益", 0)))
        bvps = _parse_percent(str(latest.get("每股净资产", 0)))
        oprcfps = _parse_percent(str(latest.get("每股经营现金流", 0)))
        undistributed = _parse_percent(str(latest.get("每股未分配利润", 0)))

        # 运营效率
        inventory_turnover = _parse_percent(str(latest.get("存货周转率", 0)))
        ar_turnover_days = _parse_percent(str(latest.get("应收账款周转天数", 0)))

        # 如果 prev 存在，计算同比变化
        revenue_growth_yoy_change = 0.0
        profit_growth_yoy_change = 0.0
        if prev is not None:
            prev_rev_growth = _parse_percent(str(prev.get("营业总收入同比增长率", 0)))
            prev_profit_growth = _parse_percent(str(prev.get("净利润同比增长率", 0)))
            revenue_growth_yoy_change = revenue_growth - prev_rev_growth
            profit_growth_yoy_change = profit_growth - prev_profit_growth

        return {
            # 营收利润
            "revenue": round(revenue, 2),
            "net_profit": round(net_profit, 2),
            "deducted_net_profit": round(deducted_net_profit, 2),
            "revenue_growth": round(revenue_growth, 2),
            "profit_growth": round(profit_growth, 2),
            "deducted_profit_growth": round(deducted_profit_growth, 2),
            "revenue_growth_yoy_change": round(revenue_growth_yoy_change, 2),
            "profit_growth_yoy_change": round(profit_growth_yoy_change, 2),
            # 盈利能力
            "gross_margin": round(gross_margin, 2),
            "net_margin": round(net_margin, 2),
            "roe": round(roe, 2),
            "roe_weighted": round(roe_weighted, 2),
            # 偿债能力
            "debt_ratio": round(debt_ratio, 2),
            "current_ratio": round(current_ratio, 2),
            "quick_ratio": round(quick_ratio, 2),
            # 每股数据
            "eps": round(eps, 4),
            "bvps": round(bvps, 4),
            "oprcfps": round(oprcfps, 4),
            "undistributed": round(undistributed, 4),
            # 运营效率
            "inventory_turnover": round(inventory_turnover, 2),
            "ar_turnover_days": round(ar_turnover_days, 2),
            # 报告期
            "report_period": str(latest.get("报告期", "")),
        }

    def _calculate_trends(self, df) -> Dict:
        """计算多期趋势指标"""
        if len(df) < 2:
            return {}

        # 取最近 4 期（1 年）数据
        recent = df.tail(4)
        revenues = []
        profits = []
        roes = []

        for _, row in recent.iterrows():
            revenues.append(_parse_percent(str(row.get("营业总收入同比增长率", 0))))
            profits.append(_parse_percent(str(row.get("净利润同比增长率", 0))))
            roes.append(_parse_percent(str(row.get("净资产收益率-摊薄", 0))))

        return {
            "revenue_growth_avg": round(np.mean(revenues), 2),
            "revenue_growth_std": round(np.std(revenues), 2),
            "profit_growth_avg": round(np.mean(profits), 2),
            "profit_growth_std": round(np.std(profits), 2),
            "roe_avg": round(np.mean(roes), 2),
            "roe_trend": "改善" if roes[-1] > roes[0] else "恶化",
        }

    def _fallback_financial_data(self, stock_code: str, stock_name: str = "") -> Dict:
        """
        Fallback 财务数据 — 当 AKShare 不可用时使用。
        基于公开信息估算（仅作为临时方案）。
        """
        logger.warning(f"[FundamentalFetcher] 使用 fallback 数据: {stock_code}")

        # 根据股票代码给出不同的估算值
        # 注意: 这些是估算值，实际使用时应确保 AKShare 可用
        defaults = {
            "300620": {  # 光库科技
                "revenue": 1474000000,
                "net_profit": 177000000,
                "revenue_growth": 47.56,
                "profit_growth": 163.76,
                "gross_margin": 34.66,
                "net_margin": 12.24,
                "roe": 8.14,
                "debt_ratio": 40.37,
                "eps": 0.709,
                "bvps": 8.71,
                "report_period": "2025",
            },
            "688313": {  # 仕佳光子
                "revenue": 850000000,
                "net_profit": 85000000,
                "revenue_growth": 15.0,
                "profit_growth": 20.0,
                "gross_margin": 28.0,
                "net_margin": 10.0,
                "roe": 6.5,
                "debt_ratio": 25.0,
                "eps": 0.15,
                "bvps": 2.3,
                "report_period": "2025",
            },
        }

        base = defaults.get(stock_code, {
            "revenue": 500000000,
            "net_profit": 50000000,
            "revenue_growth": 10.0,
            "profit_growth": 10.0,
            "gross_margin": 30.0,
            "net_margin": 10.0,
            "roe": 8.0,
            "debt_ratio": 30.0,
            "eps": 0.3,
            "bvps": 3.0,
            "report_period": "latest",
        })

        return {
            **base,
            "deducted_net_profit": base.get("net_profit", 0) * 0.85,
            "deducted_profit_growth": base.get("profit_growth", 0) * 0.9,
            "roe_weighted": base.get("roe", 0) * 1.05,
            "current_ratio": 2.5,
            "quick_ratio": 1.8,
            "oprcfps": 0.5,
            "undistributed": 1.5,
            "inventory_turnover": 2.5,
            "ar_turnover_days": 90.0,
            "revenue_growth_yoy_change": 0.0,
            "profit_growth_yoy_change": 0.0,
            "trends": {},
            "_fallback": True,  # 标记为 fallback 数据
        }

    def get_valuation_level(self, pe: float, industry_avg_pe: float) -> Dict:
        """
        基于真实行业均值的估值判断

        Args:
            pe: 当前 PE
            industry_avg_pe: 行业平均 PE

        Returns:
            估值判断字典
        """
        if pe <= 0:
            return {"level": "无法判断", "percentile": 0, "description": "亏损或 PE 异常"}

        ratio = pe / industry_avg_pe if industry_avg_pe > 0 else 1.0

        if ratio > 1.5:
            level = "极高"
            description = f"PE 为行业均值的 {ratio:.1f} 倍，估值显著偏高"
        elif ratio > 1.0:
            level = "偏高"
            description = f"PE 为行业均值的 {ratio:.1f} 倍，估值略偏高"
        elif ratio > 0.7:
            level = "合理"
            description = f"PE 为行业均值的 {ratio:.1f} 倍，估值合理"
        elif ratio > 0.4:
            level = "偏低"
            description = f"PE 为行业均值的 {ratio:.1f} 倍，估值偏低"
        else:
            level = "极低"
            description = f"PE 为行业均值的 {ratio:.1f} 倍，估值显著偏低"

        return {
            "level": level,
            "pe_ratio": round(ratio, 2),
            "percentile": min(100, int(ratio * 100)),
            "description": description,
        }


# 全局实例
fundamental_fetcher = FundamentalFetcher()
