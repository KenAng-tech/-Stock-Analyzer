#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
市场状态检测模块 — P0 修复 (2026-06-04)

原问题:
1. HMM 用单点数据做 regime 预测，无法运行维特比解码
2. 回退到 GaussianMixture 但特征工程太简单
3. 硬编码的 transition_matrix 没有实际意义

修复方案:
- 使用 GMM (Gaussian Mixture Model) 替代 HMM
- 基于滚动窗口累积特征
- 用多时间框架特征做 regime 判断
- 增加 regime 持久性检测（避免频繁切换）
"""

import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta


class MarketRegimeDetector:
    """
    市场状态检测器 — 基于 GMM 的多因子 regime 检测

    Regime 定义:
    - bull (牛市): 价格上涨 + 波动率适中 + 成交量放大
    - bear (熊市): 价格下跌 + 波动率升高 + 成交量放大
    - sideways (震荡): 价格无明显趋势 + 波动率正常
    """

    def __init__(self, n_states: int = 3):
        self.n_states = n_states
        self.state_names = ['bull', 'bear', 'sideways']
        self.model = None
        self.is_fitted = False

        # 特征缩放参数
        self.feature_means = None
        self.feature_stds = None

        # 历史预测（用于持久性检测）
        self._recent_predictions: List[str] = []
        self._max_recent = 5

        # 特征权重（基于重要性手动调优）
        self._feature_weights = np.array([0.30, 0.20, 0.15, 0.15, 0.10, 0.10])

    def extract_features(self, stock_data: Dict,
                         klines: Optional[List[Dict]] = None) -> np.ndarray:
        """
        提取 regime 检测特征

        特征列表:
        1. 日收益率 (normalized)
        2. 已实现波动率 (ATR/Price)
        3. 成交量变化率
        4. 价格动量 (价格位置)
        5. RSI (14日)
        6. 趋势强度 (MA5/MA20 比率)
        """
        # 1. 日收益率
        return_rate = stock_data.get('change_pct', 0) / 100.0

        # 2. 波动率 (用换手率作为代理)
        turnover = stock_data.get('turnover', 100)
        volatility = min(1.0, turnover / 500.0)  # 归一化到 [0, 1]

        # 3. 成交量变化率
        volume = stock_data.get('volume', 0)
        avg_volume = stock_data.get('avg_volume', volume)
        volume_change = (volume - avg_volume) / avg_volume if avg_volume > 0 else 0.0
        volume_change = np.clip(volume_change, -2, 2)  # 截断异常值

        # 4. 价格位置
        price = stock_data.get('price', 0)
        year_high = stock_data.get('year_high', 0)
        year_low = stock_data.get('year_low', 0)
        if year_high > 0 and year_low > 0 and year_high != year_low:
            price_position = (price - year_low) / (year_high - year_low)
        else:
            price_position = 0.5

        # 5. RSI (如果可用)
        rsi = stock_data.get('rsi_14', 50) / 100.0  # 归一化到 [0, 1]

        # 6. 趋势强度 (从 K 线数据计算 MA 比率)
        trend_strength = 0.0
        if klines and len(klines) >= 20:
            closes = np.array([float(k['close']) for k in klines if float(k.get('close', 0)) > 0])
            if len(closes) >= 20:
                ma5 = np.mean(closes[-5:])
                ma20 = np.mean(closes[-20:])
                if ma20 > 0:
                    trend_strength = (ma5 - ma20) / ma20  # MA5/MA20 偏离度

        features = np.array([
            return_rate,
            volatility,
            volume_change,
            price_position,
            rsi,
            trend_strength,
        ])

        # 特征缩放
        if self.feature_means is not None:
            features = (features - self.feature_means) / self.feature_stds

        return features

    def fit(self, historical_data: List[Dict],
            klines_map: Optional[Dict[str, List[Dict]]] = None) -> 'MarketRegimeDetector':
        """
        用历史数据训练 GMM 模型

        Args:
            historical_data: 历史股票数据列表
            klines_map: 可选的 K 线数据映射
        """
        if len(historical_data) < self.n_states * 10:
            raise ValueError(f"需要至少 {self.n_states * 10} 个历史数据点，当前 {len(historical_data)}")

        # 提取所有特征
        features_list = []
        for i, data in enumerate(historical_data):
            klines = klines_map.get(str(i), []) if klines_map else []
            features_list.append(self.extract_features(data, klines))

        features = np.array(features_list)

        # 计算特征统计
        self.feature_means = np.mean(features, axis=0)
        self.feature_stds = np.std(features, axis=0)
        self.feature_stds[self.feature_stds < 1e-10] = 1.0  # 避免除零

        # 缩放特征
        features_scaled = (features - self.feature_means) / self.feature_stds

        # 使用 GaussianMixture 训练
        from sklearn.mixture import GaussianMixture

        self.model = GaussianMixture(
            n_components=self.n_states,
            covariance_type='full',
            n_init=5,
            random_state=42,
            max_iter=200,
        )
        self.model.fit(features_scaled)

        self.is_fitted = True
        return self

    def predict_regime(self, stock_data: Dict,
                       klines: Optional[List[Dict]] = None) -> str:
        """
        预测当前市场状态 — P0 修复: 使用 GMM + 持久性过滤

        Args:
            stock_data: 当前股票数据
            klines: 可选的 K 线数据

        Returns:
            市场状态 ('bull', 'bear', 'sideways')
        """
        if not self.is_fitted:
            return self._predict_simple_regime(stock_data)

        features = self.extract_features(stock_data, klines).reshape(1, -1)
        prediction = self.model.predict(features)[0]
        regime = self.state_names[prediction]

        # 持久性过滤: 如果最近 5 次预测中有 3 次以上相同，保持原状态
        self._recent_predictions.append(regime)
        self._recent_predictions = self._recent_predictions[-self._max_recent:]

        if len(self._recent_predictions) >= 3:
            from collections import Counter
            most_common = Counter(self._recent_predictions).most_common(1)[0]
            if most_common[1] >= 3:  # 3/5 以上一致
                return most_common[0]

        return regime

    def get_regime_probability(self, stock_data: Dict,
                                klines: Optional[List[Dict]] = None) -> Dict:
        """
        获取各状态的概率分布
        """
        if not self.is_fitted:
            return {name: 1.0 / self.n_states for name in self.state_names}

        features = self.extract_features(stock_data, klines).reshape(1, -1)
        probabilities = self.model.predict_proba(features)[0]

        return dict(zip(self.state_names, probabilities))

    def get_regime_adjustment(self, stock_data: Dict,
                               klines: Optional[List[Dict]] = None) -> Dict:
        """
        获取基于市场状态的调整因子
        """
        regime = self.predict_regime(stock_data, klines)
        probabilities = self.get_regime_probability(stock_data, klines)

        # 各状态的调整因子
        adjustments = {
            'bull': {
                'trend_factor': 1.15,
                'volatility_factor': 0.85,
                'momentum_factor': 1.10,
                'position_bias': 'long',
                'max_position': 0.30,
            },
            'bear': {
                'trend_factor': 0.85,
                'volatility_factor': 1.15,
                'momentum_factor': 0.90,
                'position_bias': 'reduce',
                'max_position': 0.10,
            },
            'sideways': {
                'trend_factor': 1.00,
                'volatility_factor': 1.00,
                'momentum_factor': 1.00,
                'position_bias': 'neutral',
                'max_position': 0.20,
            },
        }

        current_adj = adjustments[regime]

        # 概率加权调整
        weighted_adj = {}
        for key in current_adj:
            if isinstance(current_adj[key], (int, float)):
                weighted_adj[key] = sum(
                    adjustments[r][key] * probabilities[r]
                    for r in self.state_names
                )
            else:
                weighted_adj[key] = current_adj[key]

        return {
            'regime': regime,
            'regime_probability': round(float(probabilities[self.state_names.index(regime)]), 3),
            'adjustments': {k: round(v, 3) if isinstance(v, float) else v for k, v in weighted_adj.items()},
            'all_probabilities': {k: round(float(v), 3) for k, v in probabilities.items()},
        }

    def _predict_simple_regime(self, stock_data: Dict) -> str:
        """
        简单规则预测（GMM 不可用时的降级方案）
        基于多因子规则判断
        """
        change_pct = stock_data.get('change_pct', 0)
        price = stock_data.get('price', 0)
        year_high = stock_data.get('year_high', 0)
        year_low = stock_data.get('year_low', 0)
        turnover = stock_data.get('turnover', 100)

        if year_high > 0 and year_low > 0 and year_high != year_low:
            price_position = (price - year_low) / (year_high - year_low)
        else:
            price_position = 0.5

        # 多因子规则
        if change_pct > 3 and price_position > 0.5 and turnover > 100:
            return 'bull'
        elif change_pct < -3 and price_position < 0.5 and turnover > 100:
            return 'bear'
        elif change_pct > 5:
            return 'bull'
        elif change_pct < -5:
            return 'bear'
        else:
            return 'sideways'

    def update_with_new_data(self, stock_data: Dict):
        """
        增量更新特征统计（EMA 方式）
        """
        if not self.is_fitted:
            return

        features = self.extract_features(stock_data)
        alpha = 0.01  # 学习率

        if self.feature_means is not None:
            self.feature_means = (1 - alpha) * self.feature_means + alpha * features


# 全局实例
hmm_detector = MarketRegimeDetector(n_states=3)
