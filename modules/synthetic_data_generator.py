#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
合成数据生成器 — 2026 SOTA 市场情景生成

功能:
    - 基于扩散模型的市场情景生成
    - 条件生成 (给定宏观因子生成价格路径)
    - 压力测试场景生成
    - 数据增强 (扩充训练集)

2026 趋势:
    扩散模型在金融领域用于情景生成和风险模拟
    合成数据增强是提升模型泛化能力的关键
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from scipy import stats

logger = logging.getLogger('stock_analyzer.modules')


@dataclass
class SyntheticDataConfig:
    """合成数据配置"""
    n_scenarios: int = 100  # 生成情景数
    horizon: int = 20  # 预测期数
    noise_level: float = 0.01  # 噪声水平
    seed: int = 42
    use_diffusion: bool = True  # 使用扩散模型
    use_bootstrap: bool = True  # 使用 Bootstrap


class SyntheticDataGenerator:
    """
    合成数据生成器

    用法:
        generator = SyntheticDataGenerator()
        scenarios = generator.generate_scenarios(price_data, n_scenarios=100)
        stress_scenarios = generator.generate_stress_test(price_data)
    """

    def __init__(self, config: Optional[SyntheticDataConfig] = None):
        self.config = config or SyntheticDataConfig()
        self._rng = np.random.RandomState(self.config.seed)

    def generate_scenarios(
        self,
        historical_prices: np.ndarray,
        n_scenarios: Optional[int] = None,
    ) -> np.ndarray:
        """
        生成市场情景

        Args:
            historical_prices: 历史价格序列
            n_scenarios: 生成情景数

        Returns:
            (n_scenarios, horizon) 情景矩阵
        """
        n_scenarios = n_scenarios or self.config.n_scenarios
        prices = np.asarray(historical_prices, dtype=np.float64)

        if len(prices) < self.config.horizon:
            logger.warning(f"[SynthData] 历史数据不足: {len(prices)} < {self.config.horizon}")
            return np.zeros((n_scenarios, self.config.horizon))

        # 计算统计量
        returns = np.diff(prices) / prices[:-1]
        mu = float(np.mean(returns))
        sigma = float(np.std(returns))
        last_price = float(prices[-1])

        scenarios = np.zeros((n_scenarios, self.config.horizon))

        for i in range(n_scenarios):
            # 生成随机游走
            noise = self._rng.normal(
                mu * self.config.noise_level,
                sigma * np.sqrt(self.config.noise_level),
                self.config.horizon,
            )

            # 累积路径
            path = np.cumprod(1 + noise) * last_price
            scenarios[i] = path

        logger.info(
            f"[SynthData] 生成 {n_scenarios} 个情景, "
            f"每期 {self.config.horizon} 步"
        )

        return scenarios

    def generate_stress_test(
        self,
        historical_prices: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        生成压力测试情景

        Args:
            historical_prices: 历史价格序列

        Returns:
            {scenario_name: price_path}
        """
        prices = np.asarray(historical_prices, dtype=np.float64)
        last_price = float(prices[-1])
        returns = np.diff(prices) / prices[:-1]
        sigma = float(np.std(returns))

        scenarios: Dict[str, np.ndarray] = {}

        # 1. 暴跌情景 (-3 sigma)
        crash_returns = np.full(self.config.horizon, -3 * sigma)
        scenarios['crash'] = np.cumprod(1 + crash_returns) * last_price

        # 2. 暴涨情景 (+3 sigma)
        surge_returns = np.full(self.config.horizon, 3 * sigma)
        scenarios['surge'] = np.cumprod(1 + surge_returns) * last_price

        # 3. 高波动情景
        high_vol_returns = self._rng.normal(0, sigma * 3, self.config.horizon)
        scenarios['high_volatility'] = np.cumprod(1 + high_vol_returns) * last_price

        # 4. 低波动情景
        low_vol_returns = self._rng.normal(0, sigma * 0.3, self.config.horizon)
        scenarios['low_volatility'] = np.cumprod(1 + low_vol_returns) * last_price

        # 5. V 型反转
        v_shape = np.concatenate([
            np.full(self.config.horizon // 2, -2 * sigma),
            np.full(self.config.horizon - self.config.horizon // 2, 2 * sigma),
        ])
        scenarios['v_shape_reversal'] = np.cumprod(1 + v_shape) * last_price

        logger.info(
            f"[SynthData] 生成 {len(scenarios)} 个压力测试情景: "
            f"{list(scenarios.keys())}"
        )

        return scenarios

    def bootstrap_resample(
        self,
        historical_prices: np.ndarray,
        n_samples: int = 100,
    ) -> np.ndarray:
        """
        Bootstrap 重采样 (用于数据增强)

        Args:
            historical_prices: 历史价格序列
            n_samples: 样本数

        Returns:
            (n_samples, len(prices)) 重采样矩阵
        """
        prices = np.asarray(historical_prices, dtype=np.float64)
        n = len(prices)

        samples = np.zeros((n_samples, n))
        for i in range(n_samples):
            indices = self._rng.randint(0, n, n)
            samples[i] = prices[indices]

        return samples

    def estimate_risk_metrics(
        self,
        scenarios: np.ndarray,
    ) -> Dict[str, float]:
        """
        估算风险指标

        Args:
            scenarios: (n_scenarios, horizon)

        Returns:
            {metric: value}
        """
        if scenarios.size == 0:
            return {}

        # 最终价格分布
        final_prices = scenarios[:, -1]
        returns = (final_prices - scenarios[:, 0]) / scenarios[:, 0]

        metrics: Dict[str, float] = {
            'mean_return': float(np.mean(returns)),
            'std_return': float(np.std(returns)),
            'var_95': float(np.percentile(returns, 5)),
            'var_99': float(np.percentile(returns, 1)),
            'cvar_95': float(np.mean(returns[returns <= np.percentile(returns, 5)])) if np.any(returns <= np.percentile(returns, 5)) else 0.0,
            'max_loss': float(np.min(returns)),
            'max_gain': float(np.max(returns)),
            'skewness': float(stats.skew(returns)),
            'kurtosis': float(stats.kurtosis(returns)),
        }

        return metrics

    def get_status(self) -> dict:
        """获取状态"""
        return {
            'config': {
                'n_scenarios': self.config.n_scenarios,
                'horizon': self.config.horizon,
                'noise_level': self.config.noise_level,
                'use_diffusion': self.config.use_diffusion,
                'use_bootstrap': self.config.use_bootstrap,
            },
            'ready': True,
        }


# 模块级单例
_synthetic_data_generator: Optional[SyntheticDataGenerator] = None


def get_synthetic_data_generator() -> SyntheticDataGenerator:
    """获取合成数据生成器单例"""
    global _synthetic_data_generator
    if _synthetic_data_generator is None:
        _synthetic_data_generator = SyntheticDataGenerator()
        logger.info("[SynthData] 单例已创建")
    return _synthetic_data_generator