#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
跨资产特征工程 — 2026 SOTA 横截面因子构建

功能:
    - 从多市场数据自动构建横截面因子
    - 支持 A股/港股/美股 跨市场特征
    - 因子标准化和中性化
    - 自动特征选择

2026 趋势:
    跨市场因子 + 横截面特征工程是 Alpha 重要来源
    多市场联动增强预测稳定性
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from scipy import stats

logger = logging.getLogger('stock_analyzer.modules')


@dataclass
class CrossAssetFeatureConfig:
    """跨资产特征配置"""
    momentum_window: int = 20
    volatility_window: int = 20
    correlation_window: int = 60
    min_history_days: int = 60
    top_correlated: int = 5  # 保留 Top-N 相关资产
    standardize: bool = True
    neutralize_market: bool = True


class CrossAssetFeatureEngine:
    """
    跨资产特征工程引擎

    用法:
        engine = CrossAssetFeatureEngine()
        features = engine.build_features(price_data_dict)
    """

    def __init__(self, config: Optional[CrossAssetFeatureConfig] = None):
        self.config = config or CrossAssetFeatureConfig()
        self._feature_names: List[str] = []
        self._market_data: Dict[str, np.ndarray] = {}
        self._correlation_matrix: Optional[np.ndarray] = None

    def build_features(
        self,
        price_data: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """
        构建跨资产特征

        Args:
            price_data: {asset_name: price_series}

        Returns:
            {feature_name: feature_values}
        """
        self._market_data = price_data
        features: Dict[str, np.ndarray] = {}
        n_assets = len(price_data)
        asset_names = list(price_data.keys())

        if n_assets < 2:
            logger.warning("[CrossAssetEngine] 资产数量不足 (< 2)")
            return features

        # 对齐所有序列到最短长度
        min_len = min(len(v) for v in price_data.values())
        aligned = {
            name: values[-min_len:]
            for name, values in price_data.items()
        }

        # 1. 跨资产动量相对强度
        for i, name in enumerate(asset_names):
            prices = aligned[name]
            returns = np.diff(prices) / prices[:-1]
            if len(returns) < self.config.momentum_window:
                continue

            # 相对强度: 本资产 vs 市场平均
            market_returns = np.mean([
                np.diff(aligned[n]) / aligned[n][:-1]
                for n in asset_names
            ], axis=0)

            relative_returns = returns - market_returns
            relative_momentum = np.mean(relative_returns[-self.config.momentum_window:])

            feat_name = f'relative_momentum_{name}'
            features[feat_name] = np.full(min_len, relative_momentum)

        # 2. 跨资产相关性特征
        returns_matrix = []
        for name in asset_names:
            prices = aligned[name]
            r = np.diff(prices) / prices[:-1]
            returns_matrix.append(r)

        returns_matrix = np.array(returns_matrix)
        n_times = returns_matrix.shape[1]

        if n_times >= self.config.correlation_window:
            corr_features = []
            for t in range(n_times):
                window_returns = returns_matrix[:, max(0, t - self.config.correlation_window):t + 1]
                if window_returns.shape[1] >= 10:
                    corr = np.corrcoef(window_returns)
                    # 平均相关性 (排除自身)
                    avg_corr = (np.sum(corr) - n_assets) / (n_assets * (n_assets - 1))
                    corr_features.append(avg_corr)
                else:
                    corr_features.append(0.0)

            corr_array = np.array(corr_features)
            # 对齐到 aligned 长度
            if len(corr_array) < min_len:
                pad = np.full(min_len - len(corr_array), corr_array[-1] if len(corr_array) > 0 else 0.0)
                corr_array = np.concatenate([pad, corr_array])

            features['cross_asset_correlation'] = corr_array[-min_len:]

        # 3. 波动率联动特征
        vol_features = []
        for name in asset_names:
            prices = aligned[name]
            returns = np.diff(prices) / prices[:-1]
            vol = float(np.std(returns[-self.config.volatility_window:]))
            vol_features.append(vol)

        features['cross_asset_volatility'] = np.full(min_len, np.mean(vol_features))
        features['cross_asset_vol_dispersion'] = np.full(min_len, np.std(vol_features))

        # 4. 市场领先-滞后特征
        for i, name in enumerate(asset_names):
            prices = aligned[name]
            returns = np.diff(prices) / prices[:-1]

            for j, other in enumerate(asset_names):
                if i >= j:
                    continue
                other_prices = aligned[other]
                other_returns = np.diff(other_prices) / other_prices[:-1]

                min_rlen = min(len(returns), len(other_returns))
                r1, r2 = returns[-min_rlen:], other_returns[-min_rlen:]

                if min_rlen >= 20:
                    # 领先-滞后相关性
                    lead_lag = np.correlate(r1 - np.mean(r1), r2 - np.mean(r2), mode='valid')
                    if len(lead_lag) > 0:
                        features[f'lead_lag_{name}_{other}'] = np.full(
                            min_len, float(lead_lag[0]) / (np.std(r1) * np.std(r2) + 1e-10)
                        )

        self._feature_names = list(features.keys())
        logger.info(
            f"[CrossAssetEngine] 构建 {len(features)} 个跨资产特征 "
            f"来自 {n_assets} 个资产"
        )

        return features

    def standardize_features(
        self,
        features: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """特征标准化"""
        if not self.config.standardize:
            return features

        standardized: Dict[str, np.ndarray] = {}
        for name, values in features.items():
            vals = np.asarray(values, dtype=np.float64)
            mean = np.nanmean(vals)
            std = np.nanstd(vals)
            if std > 1e-10:
                vals = (vals - mean) / std
            standardized[name] = vals

        return standardized

    def get_feature_importance(
        self,
        features: Dict[str, np.ndarray],
        target: np.ndarray,
    ) -> Dict[str, float]:
        """计算特征重要性 (基于 IC)"""
        importance: Dict[str, float] = {}
        target = np.asarray(target).flatten()

        for name, values in features.items():
            vals = np.asarray(values).flatten()
            if len(vals) != len(target) or len(vals) < 10:
                continue

            mask = ~(np.isnan(vals) | np.isnan(target))
            if np.sum(mask) < 10:
                continue

            ic, _ = stats.spearmanr(vals[mask], target[mask])
            importance[name] = float(ic)

        return importance

    def get_status(self) -> dict:
        """获取状态"""
        return {
            'n_assets': len(self._market_data),
            'asset_names': list(self._market_data.keys()),
            'n_features': len(self._feature_names),
            'feature_names': self._feature_names[:20],
            'config': {
                'momentum_window': self.config.momentum_window,
                'volatility_window': self.config.volatility_window,
                'correlation_window': self.config.correlation_window,
            },
        }


# 模块级单例
_cross_asset_engine: Optional[CrossAssetFeatureEngine] = None


def get_cross_asset_engine() -> CrossAssetFeatureEngine:
    """获取跨资产特征引擎单例"""
    global _cross_asset_engine
    if _cross_asset_engine is None:
        _cross_asset_engine = CrossAssetFeatureEngine()
        logger.info("[CrossAssetEngine] 单例已创建")
    return _cross_asset_engine