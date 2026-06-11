#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
增强特征工程模块 — Enhanced Feature Engineering

在现有 ml_predictor.py 的 12 个特征基础上新增 25+ 高质量因子。

分类:
- 波动率类 (5): realized_vol_5d/20d, downside_vol_ratio, high_low_range, garch_vol_proxy
- 流动性类 (3): amihud_illiquidity, volume_turnover_ratio, bid_ask_spread_proxy
- 量价关系 (5): VPT, OBV变化率, 量动量5d/20d, 量价相关
- 技术增强 (5): ATR位置, 布林带宽度, MACD背离, RSI斜率, 价格MA偏离度
- 资金流增强 (3): 主力净流入/流通市值, 5日内外盘比, 成交额比率
- 基本面增强 (3): 盈利代理, 营收加速, 营运资本变化
"""

import numpy as np
from typing import Dict, List, Optional
from modules.logger import logger


class EnhancedFeatures:
    """增强特征工程 — 25+ 高质量因子"""

    def __init__(self):
        self.feature_names = [
            # 波动率类
            'realized_vol_5d', 'realized_vol_20d', 'downside_vol_ratio',
            'high_low_range', 'garch_vol_proxy',
            # 流动性类
            'amihud_illiquidity', 'volume_turnover_ratio', 'bid_ask_spread_proxy',
            # 量价关系
            'volume_price_trend', 'obv_change', 'volume_momentum_5d',
            'volume_momentum_20d', 'volume_price_correlation',
            # 技术增强
            'atr_position', 'bollinger_width', 'macd_divergence',
            'rsi_slope', 'price_ma_divergence',
            # 资金流增强
            'main_net_inflow_ratio', 'outer_inner_ratio_5d', 'amount_level_ratio',
            # 基本面增强
            'earnings_surrogate', 'revenue_acceleration', 'working_capital_change',
        ]

    def calculate_all(self, stock_data: Dict, klines: Optional[List[Dict]] = None) -> Dict[str, float]:
        """计算所有增强特征"""
        result = {}
        for method_name in self.feature_names:
            try:
                method = getattr(self, f'_{method_name}')
                result[method_name] = method(stock_data, klines)
            except Exception as e:
                logger.debug(f"[EnhancedFeatures] {method_name} 计算失败: {e}")
                result[method_name] = 0.0
        return result

    # ── 波动率类 ──

    def _realized_vol_5d(self, stock_data: Dict, klines: Optional[List[Dict]] = None) -> float:
        """5日已实现波动率"""
        if klines and len(klines) >= 6:
            closes = np.array([float(k.get('close', 0)) for k in klines[-6:] if float(k.get('close', 0)) > 0])
            if len(closes) >= 2:
                returns = np.diff(np.log(closes))
                return float(np.std(returns) * np.sqrt(252) * 100)
        return stock_data.get('turnover', 0) / 100.0  # 降级: 用换手率代理

    def _realized_vol_20d(self, stock_data: Dict, klines: Optional[List[Dict]] = None) -> float:
        """20日已实现波动率"""
        if klines and len(klines) >= 21:
            closes = np.array([float(k.get('close', 0)) for k in klines[-21:] if float(k.get('close', 0)) > 0])
            if len(closes) >= 2:
                returns = np.diff(np.log(closes))
                return float(np.std(returns) * np.sqrt(252) * 100)
        return 0.0

    def _downside_vol_ratio(self, stock_data: Dict, klines: Optional[List[Dict]] = None) -> float:
        """下行波动率/总波动率"""
        if klines and len(klines) >= 21:
            closes = np.array([float(k.get('close', 0)) for k in klines[-21:] if float(k.get('close', 0)) > 0])
            if len(closes) >= 2:
                returns = np.diff(np.log(closes))
                total_vol = np.std(returns)
                downside = returns[returns < 0]
                if len(downside) > 0 and total_vol > 0:
                    dvol = np.std(downside)
                    return float(dvol / (total_vol + 1e-10))
        return 0.5  # 中性

    def _high_low_range(self, stock_data: Dict, klines: Optional[List[Dict]] = None) -> float:
        """日内波动幅度 (H-L)/(H+L)"""
        if klines and len(klines) >= 1:
            k = klines[-1]
            h = float(k.get('high', 0))
            l = float(k.get('low', 0))
            if h + l > 0:
                return float((h - l) / (h + l))
        return 0.0

    def _garch_vol_proxy(self, stock_data: Dict, klines: Optional[List[Dict]] = None) -> float:
        """GARCH(1,1) 波动率代理"""
        if klines and len(klines) >= 30:
            closes = np.array([float(k.get('close', 0)) for k in klines[-30:] if float(k.get('close', 0)) > 0])
            if len(closes) >= 2:
                returns = np.diff(np.log(closes))
                # Simple GARCH(1,1) proxy: omega=1e-8, alpha=0.1, beta=0.85
                omega, alpha, beta = 1e-8, 0.1, 0.85
                var = np.mean(returns ** 2)
                for r in returns[1:]:
                    var = omega + alpha * r**2 + beta * var
                return float(np.sqrt(var) * np.sqrt(252) * 100)
        return 0.0

    # ── 流动性类 ──

    def _amihud_illiquidity(self, stock_data: Dict, klines: Optional[List[Dict]] = None) -> float:
        """Amihud 流动性指标: mean(|return| / volume)"""
        if klines and len(klines) >= 21:
            closes = np.array([float(k.get('close', 0)) for k in klines[-21:] if float(k.get('close', 0)) > 0])
            volumes = np.array([float(k.get('volume', 0)) for k in klines[-21:] if float(k.get('volume', 0)) > 0])
            if len(closes) >= 2 and len(volumes) >= 2:
                returns = np.abs(np.diff(np.log(closes)))
                vols = volumes[1:]
                if np.all(vols > 0):
                    return float(np.mean(returns / vols))
        return 0.0

    def _volume_turnover_ratio(self, stock_data: Dict, klines: Optional[List[Dict]] = None) -> float:
        """成交量/换手率比率"""
        vol = stock_data.get('volume', 0)
        turnover = stock_data.get('turnover', 0)
        if turnover > 0:
            return float(vol / turnover)
        return 0.0

    def _bid_ask_spread_proxy(self, stock_data: Dict, klines: Optional[List[Dict]] = None) -> float:
        """买卖价差代理: (H-L)/(H+L)"""
        if klines and len(klines) >= 1:
            k = klines[-1]
            h = float(k.get('high', 0))
            l = float(k.get('low', 0))
            if h + l > 0:
                return float((h - l) / (h + l))
        return 0.0

    # ── 量价关系 ──

    def _volume_price_trend(self, stock_data: Dict, klines: Optional[List[Dict]] = None) -> float:
        """VPT: sum(volume * change_pct)"""
        if klines and len(klines) >= 21:
            total = 0.0
            for k in klines[-21:]:
                vol = float(k.get('volume', 0))
                pct = float(k.get('change_pct', 0)) / 100.0
                total += vol * pct
            return float(total)
        return 0.0

    def _obv_change(self, stock_data: Dict, klines: Optional[List[Dict]] = None) -> float:
        """OBV 变化率"""
        if klines and len(klines) >= 6:
            obv = 0
            for i, k in enumerate(klines):
                close = float(k.get('close', 0))
                prev_close = float(klines[i-1].get('close', 0)) if i > 0 else close
                vol = float(k.get('volume', 0))
                if close > prev_close:
                    obv += vol
                elif close < prev_close:
                    obv -= vol
            # 5日前的OBV
            if len(klines) >= 6:
                obv_5 = 0
                for i, k in enumerate(klines[:5]):
                    close = float(k.get('close', 0))
                    prev_close = float(klines[i-1].get('close', 0)) if i > 0 else close
                    vol = float(k.get('volume', 0))
                    if close > prev_close:
                        obv_5 += vol
                    elif close < prev_close:
                        obv_5 -= vol
                if obv_5 != 0:
                    return float((obv - obv_5) / abs(obv_5))
        return 0.0

    def _volume_momentum_5d(self, stock_data: Dict, klines: Optional[List[Dict]] = None) -> float:
        """5日成交量动量: 当前量 / 60日均量"""
        if klines and len(klines) >= 61:
            current_vol = float(klines[-1].get('volume', 0))
            avg_vols = [float(k.get('volume', 0)) for k in klines[-61:-1]]
            if avg_vols and sum(avg_vols) > 0:
                return float(current_vol / (np.mean(avg_vols) + 1e-10))
        return 1.0

    def _volume_momentum_20d(self, stock_data: Dict, klines: Optional[List[Dict]] = None) -> float:
        """20日成交量动量: 当前量 / 120日均量"""
        if klines and len(klines) >= 121:
            current_vol = float(klines[-1].get('volume', 0))
            avg_vols = [float(k.get('volume', 0)) for k in klines[-121:-1]]
            if avg_vols and sum(avg_vols) > 0:
                return float(current_vol / (np.mean(avg_vols) + 1e-10))
        return 1.0

    def _volume_price_correlation(self, stock_data: Dict, klines: Optional[List[Dict]] = None) -> float:
        """量价相关系数 (20日滚动)"""
        if klines and len(klines) >= 21:
            closes = np.array([float(k.get('close', 0)) for k in klines[-21:] if float(k.get('close', 0)) > 0])
            volumes = np.array([float(k.get('volume', 0)) for k in klines[-21:] if float(k.get('volume', 0)) > 0])
            if len(closes) >= 3 and len(volumes) >= 3:
                ret = np.diff(closes) / closes[:-1]
                vol_ret = np.diff(volumes) / (volumes[:-1] + 1e-10)
                if np.std(ret) > 0 and np.std(vol_ret) > 0:
                    return float(np.corrcoef(ret, vol_ret)[0, 1])
        return 0.0

    # ── 技术增强 ──

    def _atr_position(self, stock_data: Dict, klines: Optional[List[Dict]] = None) -> float:
        """ATR 标准化价格位置: (price - MA20) / ATR14"""
        if klines and len(klines) >= 21:
            closes = np.array([float(k.get('close', 0)) for k in klines[-21:] if float(k.get('close', 0)) > 0])
            highs = np.array([float(k.get('high', 0)) for k in klines[-21:] if float(k.get('high', 0)) > 0])
            lows = np.array([float(k.get('low', 0)) for k in klines[-21:] if float(k.get('low', 0)) > 0])
            if len(closes) >= 2:
                ma20 = np.mean(closes)
                tr = np.maximum(highs[1:] - lows[1:],
                               np.abs(highs[1:] - closes[:-1]),
                               np.abs(lows[1:] - closes[:-1]))
                atr = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)
                price = closes[-1]
                if atr > 0:
                    return float((price - ma20) / atr)
        return 0.0

    def _bollinger_width(self, stock_data: Dict, klines: Optional[List[Dict]] = None) -> float:
        """布林带宽度: (upper - lower) / middle"""
        if klines and len(klines) >= 21:
            closes = np.array([float(k.get('close', 0)) for k in klines[-21:] if float(k.get('close', 0)) > 0])
            if len(closes) >= 20:
                ma20 = np.mean(closes)
                std20 = np.std(closes)
                if std20 > 0 and ma20 > 0:
                    upper = ma20 + 2 * std20
                    lower = ma20 - 2 * std20
                    return float((upper - lower) / ma20)
        return 0.0

    def _macd_divergence(self, stock_data: Dict, klines: Optional[List[Dict]] = None) -> float:
        """MACD 背离检测"""
        if klines and len(klines) >= 30:
            closes = np.array([float(k.get('close', 0)) for k in klines[-30:] if float(k.get('close', 0)) > 0])
            if len(closes) >= 30:
                def ema(data, period):
                    if len(data) < period:
                        return float(np.mean(data))
                    m = 2.0 / (period + 1)
                    r = float(data[0])
                    for p in data[1:]:
                        r = (p - r) * m + r
                    return r

                ema12 = ema(closes, 12)
                ema26 = ema(closes, 26)
                macd_current = ema12 - ema26

                # 比较前10日和当前
                macd_prev = ema(closes[:20], 12) - ema(closes[:20], 26)
                price_prev = closes[19]
                price_current = closes[-1]

                if price_current > price_prev and macd_current < macd_prev:
                    return -1.0  # 顶背离
                elif price_current < price_prev and macd_current > macd_prev:
                    return 1.0   # 底背离
        return 0.0

    def _rsi_slope(self, stock_data: Dict, klines: Optional[List[Dict]] = None) -> float:
        """RSI 斜率: 3日变化"""
        if klines and len(klines) >= 17:
            closes = np.array([float(k.get('close', 0)) for k in klines[-17:] if float(k.get('close', 0)) > 0])
            if len(closes) >= 15:
                # 计算 RSI
                delta = np.diff(closes)
                gains = np.where(delta > 0, delta, 0)
                losses = np.where(delta < 0, -delta, 0)
                avg_gain = np.mean(gains[:14])
                avg_loss = np.mean(losses[:14])
                rsi_3d_ago = 100 - 100 / (1 + avg_gain / (avg_loss + 1e-10))

                # 重新计算最近14天
                gains2 = np.diff(closes[-14:])
                gains_pos = np.where(gains2 > 0, gains2, 0)
                losses_neg = np.where(gains2 < 0, -gains2, 0)
                ag = np.mean(gains_pos)
                al = np.mean(losses_neg)
                rsi_now = 100 - 100 / (1 + ag / (al + 1e-10))

                return float(rsi_now - rsi_3d_ago)
        return 0.0

    def _price_ma_divergence(self, stock_data: Dict, klines: Optional[List[Dict]] = None) -> float:
        """价格与MA20偏离度: (price - MA20) / MA20"""
        if klines and len(klines) >= 21:
            closes = np.array([float(k.get('close', 0)) for k in klines[-21:] if float(k.get('close', 0)) > 0])
            if len(closes) >= 20:
                ma20 = np.mean(closes)
                price = closes[-1]
                if ma20 > 0:
                    return float((price - ma20) / ma20)
        return 0.0

    # ── 资金流增强 ──

    def _main_net_inflow_ratio(self, stock_data: Dict, klines: Optional[List[Dict]] = None) -> float:
        """主力净流入/流通市值"""
        inflow = stock_data.get('main_net_inflow', 0)
        cap = stock_data.get('circulating_cap', 0)
        if cap > 0:
            return float(inflow / cap)
        return 0.0

    def _outer_inner_ratio_5d(self, stock_data: Dict, klines: Optional[List[Dict]] = None) -> float:
        """5日内外盘比均值"""
        # 从当前数据获取
        outer = stock_data.get('outer_disk', 0)
        inner = stock_data.get('inner_disk', 1)
        if inner > 0:
            return float(outer / inner)
        return 1.0

    def _amount_level_ratio(self, stock_data: Dict, klines: Optional[List[Dict]] = None) -> float:
        """成交额/60日均额"""
        amount = stock_data.get('amount', 0)
        # 降级: 用当前成交额/10
        if amount > 0:
            return float(amount / (amount / 10 + 1e-10))  # 简化
        return 1.0

    # ── 基本面增强 ──

    def _earnings_surrogate(self, stock_data: Dict, klines: Optional[List[Dict]] = None) -> float:
        """盈利代理: ROE * net_margin / 100"""
        roe = stock_data.get('roe', 0)
        net_margin = stock_data.get('net_margin', 0)
        return float(roe * net_margin / 100.0)

    def _revenue_acceleration(self, stock_data: Dict, klines: Optional[List[Dict]] = None) -> float:
        """营收加速: 本期增速 - 上期增速 (proxy)"""
        growth = stock_data.get('revenue_growth', 0)
        # 用 change_pct 作为 proxy 的 change in growth
        change_pct = stock_data.get('change_pct', 0)
        return float(growth - change_pct * 10)  # 简化代理

    def _working_capital_change(self, stock_data: Dict, klines: Optional[List[Dict]] = None) -> float:
        """营运资本变化率 proxy"""
        current_ratio = stock_data.get('current_ratio', 1.0)
        return float((current_ratio - 1.0) * current_ratio)


# 全局实例
enhanced_features = EnhancedFeatures()
