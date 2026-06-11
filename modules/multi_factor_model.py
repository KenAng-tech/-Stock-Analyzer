#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
多因子模型模块（增强版）

实现 8 大因子 + 横截面 Z-Score 标准化（P1 修复）:
  - 动量、价值、波动率、成交量、流动性、质量、情绪、技术

升级内容:
  - P1: 横截面 Z-Score 标准化
  - P1: 质量因子补充 ROE、营收增速
  - P1: 动态因子权重（基于 IC/ICIR）
"""

import numpy as np
from typing import Dict, List, Optional
from datetime import datetime


class MultiFactorModel:
    """多因子选股模型（增强版）"""

    def __init__(self):
        # 因子权重（基准权重，可动态调整）
        self.factor_weights = {
            'momentum': 0.20,
            'value': 0.15,
            'volatility': 0.10,
            'volume': 0.15,
            'liquidity': 0.10,
            'quality': 0.10,
            'sentiment': 0.10,
            'technical': 0.10,
        }
        # 因子 IC 历史（用于动态权重）
        self._factor_ic_history: Dict[str, List[float]] = {}

    # ── 因子计算 ──────────────────────────────────────────────

    def momentum_factor(self, stock_data: Dict) -> float:
        """动量因子 - 基于价格趋势和动量"""
        change_pct = stock_data.get('change_pct', 0)
        price = stock_data.get('price', 0)

        short_momentum = min(10, max(0, change_pct / 5))

        year_high = stock_data.get('year_high', 0)
        year_low = stock_data.get('year_low', 0)
        if year_high > 0 and year_low > 0 and year_high != year_low:
            price_position = (price - year_low) / (year_high - year_low)
        else:
            price_position = 0.5

        return min(10, max(0, short_momentum + price_position * 5))

    def value_factor(self, stock_data: Dict) -> float:
        """价值因子 - 基于 PE 和 PB"""
        pe = stock_data.get('pe', 100)
        pb = stock_data.get('pb', 5)
        market_cap = stock_data.get('market_cap', 0)

        pe_score = max(0, 10 - pe / 30) if pe > 0 else 5
        pb_score = max(0, 10 - pb / 5) if pb > 0 else 5

        if market_cap > 0:
            if 50 < market_cap < 500:
                market_score = 8
            elif market_cap >= 500:
                market_score = 6
            else:
                market_score = 5
        else:
            market_score = 5

        return (pe_score + pb_score + market_score) / 3

    def volatility_factor(self, stock_data: Dict) -> float:
        """波动率因子 - 低波动率更优"""
        turnover = stock_data.get('turnover', 100)
        change_pct = stock_data.get('change_pct', 0)

        turnover_vol = max(0, 10 - turnover / 30)
        price_vol = max(0, 10 - abs(change_pct) / 2)

        return (turnover_vol + price_vol) / 2

    def volume_factor(self, stock_data: Dict) -> float:
        """成交量因子 - 放量上涨为正向信号"""
        outer = stock_data.get('outer_disk', 0)
        inner = stock_data.get('inner_disk', 0)

        outer_inner_ratio = outer / inner if inner > 0 else 1
        if outer_inner_ratio > 1.2:
            return 8
        elif outer_inner_ratio > 1.0:
            return 6
        return 4

    def liquidity_factor(self, stock_data: Dict) -> float:
        """流动性因子 - 基于换手率和成交额"""
        turnover = stock_data.get('turnover', 100)
        amount = stock_data.get('amount', 0)

        turnover_score = 8 if turnover > 200 else (6 if turnover > 100 else 4)
        amount_score = 8 if amount > 500000 else (6 if amount > 200000 else 4)

        return (turnover_score + amount_score) / 2

    def quality_factor(self, stock_data: Dict) -> float:
        """质量因子（增强: P1 补充 ROE、营收增速）"""
        pe = stock_data.get('pe', 100)
        market_cap = stock_data.get('market_cap', 0)
        roe = stock_data.get('roe', 15)  # ROE
        revenue_growth = stock_data.get('revenue_growth', 10)  # 营收增速%

        # PE 质量
        pe_quality = 8 if 20 < pe < 100 else (7 if pe <= 20 else 5)
        # 市值质量
        market_quality = 8 if market_cap > 100 else (6 if market_cap > 50 else 4)
        # ROE 质量（新增）
        roe_quality = 9 if roe > 20 else (7 if roe > 15 else (5 if roe > 10 else 3))
        # 营收增速质量（新增）
        growth_quality = 8 if revenue_growth > 25 else (6 if revenue_growth > 15 else (4 if revenue_growth > 5 else 2))

        return (pe_quality + market_quality + roe_quality + growth_quality) / 4

    def sentiment_factor(self, stock_data: Dict) -> float:
        """情绪因子 - 基于市场情绪指标"""
        change_pct = stock_data.get('change_pct', 0)
        turnover = stock_data.get('turnover', 100)

        price_sentiment = 8 if change_pct > 5 else (6 if change_pct > 0 else (4 if change_pct > -5 else 2))
        volume_sentiment = 7 if turnover > 300 else (6 if turnover > 150 else 5)

        return (price_sentiment + volume_sentiment) / 2

    def technical_factor(self, stock_data: Dict) -> float:
        """技术因子 - 基于技术指标"""
        change_pct = stock_data.get('change_pct', 0)
        volume = stock_data.get('volume', 0)

        rsi = 55 + (change_pct * 2)
        rsi_score = max(0, min(10, rsi))
        volume_score = 8 if volume > 200000 else (6 if volume > 100000 else 4)

        return (rsi_score + volume_score) / 2

    # ── 横截面 Z-Score 标准化（P1 修复） ──────────────────────

    @staticmethod
    def cross_sectional_normalize(factor_scores: Dict[str, float],
                                    universe: List[Dict[str, float]]) -> Dict[str, float]:
        """横截面 Z-Score 标准化

        将因子评分转换为横截面上的 Z-Score，消除不同量纲的影响。

        Args:
            factor_scores: 目标股票的因子评分 {factor_name: score}
            universe: 股票池，每只股票是一个 {factor_name: score} 的字典

        Returns:
            标准化后的因子评分 {factor_name: z_score}
        """
        normalized = {}
        all_scores = {}

        # 收集所有股票在每个因子上的评分
        for factor_name in factor_scores:
            values = [us[factor_name] for us in universe if factor_name in us]
            values.append(factor_scores[factor_name])
            all_scores[factor_name] = values

        # 计算 Z-Score
        for factor_name, values in all_scores.items():
            if len(values) < 2:
                normalized[factor_name] = 0.0
                continue
            mean = np.mean(values)
            std = np.std(values)
            if std > 1e-10:
                normalized[factor_name] = (factor_scores[factor_name] - mean) / std
            else:
                normalized[factor_name] = 0.0

        return normalized

    # ── 综合评分 ──────────────────────────────────────────────

    def calculate_scores(self, stock_data: Dict,
                          universe: Optional[List[Dict]] = None) -> Dict:
        """计算所有因子评分（支持横截面标准化）"""
        raw_scores = {
            'momentum': self.momentum_factor(stock_data),
            'value': self.value_factor(stock_data),
            'volatility': self.volatility_factor(stock_data),
            'volume': self.volume_factor(stock_data),
            'liquidity': self.liquidity_factor(stock_data),
            'quality': self.quality_factor(stock_data),
            'sentiment': self.sentiment_factor(stock_data),
            'technical': self.technical_factor(stock_data),
        }

        # 如果有股票池，进行横截面 Z-Score 标准化
        if universe and len(universe) >= 3:
            normalized = self.cross_sectional_normalize(raw_scores, universe)
            # 使用标准化后的分数加权
            weighted_score = sum(
                self.factor_weights[f] * normalized[f]
                for f in normalized
            )
            return {
                'factors': normalized,
                'raw_factors': raw_scores,
                'weighted_score': round(weighted_score, 4),
                'rating': self._get_rating(normalized),
                'normalized': True,
            }

        # 无股票池时使用原始分数
        weighted_score = sum(
            self.factor_weights[f] * raw_scores[f]
            for f in raw_scores
        )
        return {
            'factors': raw_scores,
            'raw_factors': raw_scores,
            'weighted_score': round(weighted_score, 2),
            'rating': self._get_rating(raw_scores),
            'normalized': False,
        }

    def _get_rating(self, score) -> str:
        """评级"""
        # 如果是 dict，计算加权平均
        if isinstance(score, dict):
            weights = self.factor_weights
            denom = sum(weights.get(f, 1 / len(weights)) for f in score)
            if denom > 0:
                avg = sum(weights.get(f, 1 / len(weights)) * v for f, v in score.items()) / denom
            else:
                avg = np.mean(list(score.values()))
            score = avg

        # Z-Score 标准
        if isinstance(score, (int, float)) and abs(score) > 1:
            if score > 1.5:
                return '强烈推荐'
            elif score > 0.5:
                return '推荐'
            elif score > -0.5:
                return '中性'
            else:
                return '观望'
        # 原始分数标准
        if score >= 8:
            return '强烈推荐'
        elif score >= 6:
            return '推荐'
        elif score >= 4:
            return '中性'
        return '观望'

    def get_factor_exposure(self, stock_data: Dict) -> Dict:
        """获取因子暴露"""
        scores = self.calculate_scores(stock_data)
        exposure = {
            f: round(scores['factors'][f] - np.mean(list(scores['factors'].values())), 2)
            for f in scores['factors']
        }
        return {
            'exposure': exposure,
            'dominant_factor': max(exposure, key=exposure.get) if exposure else '',
            'weakest_factor': min(exposure, key=exposure.get) if exposure else '',
        }

    def update_factor_weights(self, ic_data: Dict[str, float]):
        """基于 IC 更新因子权重（动态权重）

        Args:
            ic_data: {factor_name: IC_value}
        """
        abs_ics = {f: abs(v) for f, v in ic_data.items() if v != 0}
        total = sum(abs_ics.values())
        if total > 0:
            self.factor_weights = {f: v / total for f, v in abs_ics.items()}


# 全局实例
multi_factor_model = MultiFactorModel()
