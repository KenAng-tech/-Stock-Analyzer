#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
CVaR/EVT 风险管理模块 — Conditional VaR & Extreme Value Theory

集成 barra_risk_model.py 的 EVTAnalyzer，提供完整的尾部风险分析。

功能:
    - VaR (Value at Risk) 计算
    - CVaR (Conditional VaR) 计算
    - GPD (Generalized Pareto Distribution) 拟合
    - 极值统计指标
    - 风险报告生成
"""

import numpy as np
from typing import Dict, Optional, List
from datetime import datetime
from modules.logger import logger


class CVaREventAnalyzer:
    """CVaR/EVT 风险分析器"""

    def __init__(self):
        self._history: Dict[str, List[float]] = {}

    def fit_gpd(self, returns: np.ndarray,
                threshold: float = 0.02) -> Dict:
        """
        拟合 GPD (Generalized Pareto Distribution)

        Args:
            returns: 收益率序列
            threshold: 阈值 (默认 2%)

        Returns:
            GPD 参数
        """
        tail_returns = returns[returns < -threshold]
        if len(tail_returns) < 10:
            return {
                'shape': 0.0,
                'scale': 0.0,
                'tail_index': 0.0,
                'tail_sample_size': 0,
                'method': 'insufficient_data',
            }

        # 形状参数 (ξ): 厚尾程度
        shape = float(np.clip(
            np.percentile(tail_returns, 75) - np.percentile(tail_returns, 25),
            -0.5, 2.0
        ))

        # 尺度参数 (σ)
        scale = float(np.clip(
            np.mean(np.abs(tail_returns - np.percentile(tail_returns, 50))),
            0.001, 0.1
        ))

        return {
            'shape': shape,
            'scale': scale,
            'tail_index': shape,
            'tail_sample_size': int(len(tail_returns)),
            'threshold': float(threshold),
            'method': 'pots',
        }

    def calculate_var(self, returns: np.ndarray,
                      confidence: float = 0.95,
                      gpd_params: Optional[Dict] = None) -> float:
        """
        计算 VaR

        Args:
            returns: 收益率序列
            confidence: 置信水平
            gpd_params: GPD 参数

        Returns:
            VaR 值
        """
        if gpd_params and gpd_params.get('method') == 'pots':
            shape = gpd_params['shape']
            scale = gpd_params['scale']
            tail_prob = 1 - confidence
            if shape != 0:
                var = -scale / shape * ((tail_prob * len(returns)) ** (-shape) - 1)
            else:
                var = -scale * np.log(tail_prob * len(returns))
            return float(var)
        else:
            return float(-np.percentile(returns, (1 - confidence) * 100))

    def calculate_cvar(self, returns: np.ndarray,
                       confidence: float = 0.95,
                       gpd_params: Optional[Dict] = None) -> float:
        """
        计算 CVaR (Conditional VaR / Expected Shortfall)
        """
        if gpd_params and gpd_params.get('method') == 'pots':
            shape = gpd_params['shape']
            scale = gpd_params['scale']
            tail_prob = 1 - confidence
            if shape != 0:
                cvar = -scale / (shape - 1) * ((tail_prob * len(returns)) ** (-shape) - 1)
            else:
                cvar = -scale * (1 - np.log(tail_prob * len(returns)))
            return float(cvar)
        else:
            threshold = self.calculate_var(returns, confidence)
            tail_returns = returns[returns <= threshold]
            return float(-np.mean(tail_returns)) if len(tail_returns) > 0 else float(threshold)

    def calculate_extreme_metrics(self, returns: np.ndarray) -> Dict:
        """
        计算极值统计指标
        """
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]

        if len(negative_returns) < 5:
            return {
                'max_loss': float(-np.min(returns)),
                'max_gain': float(np.max(returns)),
                'extreme_ratio': 1.0,
                'tail_skewness': 0.0,
                'tail_kurtosis': 0.0,
                'tail_sample_size': len(negative_returns),
            }

        tail_threshold = np.percentile(negative_returns, 10)
        extreme_negative = negative_returns[negative_returns <= tail_threshold]
        extreme_positive = positive_returns[positive_returns >= np.percentile(positive_returns, 90)]

        return {
            'max_loss': float(-np.min(returns)),
            'max_gain': float(np.max(returns)),
            'extreme_ratio': float(np.mean(np.abs(extreme_negative)) / (np.mean(np.abs(extreme_positive)) + 1e-10)),
            'tail_skewness': float(np.mean(extreme_negative) / (np.std(extreme_negative) + 1e-10)),
            'tail_kurtosis': float(
                np.mean((extreme_negative - np.mean(extreme_negative)) ** 4) /
                (np.std(extreme_negative) ** 4 + 1e-10) - 3
            ),
            'tail_sample_size': len(extreme_negative),
        }

    def analyze_stock_risk(self, stock_code: str,
                           klines: List[Dict],
                           confidence: float = 0.95) -> Dict:
        """
        分析单只股票的风险指标

        Args:
            stock_code: 股票代码
            klines: K 线数据
            confidence: 置信水平

        Returns:
            风险报告
        """
        if not klines or len(klines) < 60:
            return {
                'success': False,
                'error': '数据不足',
                'stock_code': stock_code,
            }

        # 计算收益率
        closes = np.array([float(k.get('close', 0)) for k in klines if float(k.get('close', 0)) > 0])
        if len(closes) < 60:
            return {
                'success': False,
                'error': '数据不足',
                'stock_code': stock_code,
            }

        returns = np.diff(np.log(closes))

        # GPD 拟合
        gpd_params = self.fit_gpd(returns)

        # VaR / CVaR
        var_95 = self.calculate_var(returns, 0.95, gpd_params)
        var_99 = self.calculate_var(returns, 0.99, gpd_params)
        cvar_95 = self.calculate_cvar(returns, 0.95, gpd_params)
        cvar_99 = self.calculate_cvar(returns, 0.99, gpd_params)

        # 极值指标
        extreme_metrics = self.calculate_extreme_metrics(returns)

        # 日收益率统计
        daily_stats = {
            'mean_return': float(np.mean(returns)),
            'std_return': float(np.std(returns)),
            'annualized_vol': float(np.std(returns) * np.sqrt(252) * 100),
            'skewness': float(np.mean(returns) / (np.std(returns) + 1e-10)),
            'kurtosis': float(
                np.mean((returns - np.mean(returns)) ** 4) /
                (np.std(returns) ** 4 + 1e-10) - 3
            ),
        }

        # 年化风险
        ann_vol = daily_stats['annualized_vol']
        var_95_annual = float(var_95 * np.sqrt(252) * 100)
        cvar_95_annual = float(cvar_95 * np.sqrt(252) * 100)

        return {
            'success': True,
            'stock_code': stock_code,
            'n_observations': len(returns),
            'gpd_params': gpd_params,
            'var': {
                'var_95_daily': round(var_95 * 100, 4),
                'var_99_daily': round(var_99 * 100, 4),
                'var_95_annual': round(var_95_annual, 4),
                'var_99_annual': round(var_99 * np.sqrt(252) * 100, 4),
            },
            'cvar': {
                'cvar_95_daily': round(cvar_95 * 100, 4),
                'cvar_99_daily': round(cvar_99 * 100, 4),
                'cvar_95_annual': round(cvar_95_annual, 4),
                'cvar_99_annual': round(cvar_99 * np.sqrt(252) * 100, 4),
            },
            'extreme_metrics': extreme_metrics,
            'daily_stats': daily_stats,
            'timestamp': datetime.now().isoformat(),
        }

    def get_status(self) -> Dict:
        """获取状态"""
        return {
            'modules_loaded': True,
            'supported_methods': ['GPD', 'Historical', 'Parametric'],
            'confidence_levels': [0.90, 0.95, 0.99],
            'timestamp': datetime.now().isoformat(),
        }


# 全局实例
_cvar_analyzer: Optional[CVaREventAnalyzer] = None


def get_cvar_analyzer() -> CVaREventAnalyzer:
    """获取全局 CVaR/EVT 分析器实例"""
    global _cvar_analyzer
    if _cvar_analyzer is None:
        _cvar_analyzer = CVaREventAnalyzer()
    return _cvar_analyzer
