#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Barra 风格因子风险模型 — Barra Risk Model

简化版 Barra CNE6 多因子风险模型, 用于组合风险分解和优化。

风格因子:
  mkt_beta, size, value, growth, liquidity, momentum,
  volatility, leverage, earnings_yield, restructuring

功能:
  - 风险分解 (系统性 vs 特异性)
  - 风险贡献分析
  - 均值-方差优化
  - Black-Litterman 模型
  - 风险平价
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from modules.logger import logger


class BarraStyleFactors:
    """Barra 风格因子计算"""

    # 风格因子定义
    FACTOR_NAMES = [
        'mkt_beta', 'size', 'value', 'growth', 'liquidity',
        'momentum', 'volatility', 'leverage', 'earnings_yield', 'restructuring'
    ]

    def calculate_all(self, stock_data: Dict,
                      klines: Optional[List[Dict]] = None) -> Dict[str, float]:
        """计算所有 Barra 风格因子"""
        return {
            'mkt_beta': self._mkt_beta(stock_data, klines),
            'size': self._size(stock_data),
            'value': self._value(stock_data),
            'growth': self._growth(stock_data),
            'liquidity': self._liquidity(stock_data),
            'momentum': self._momentum(stock_data, klines),
            'volatility': self._volatility(stock_data, klines),
            'leverage': self._leverage(stock_data),
            'earnings_yield': self._earnings_yield(stock_data),
            'restructuring': self._restructuring(stock_data),
        }

    def _mkt_beta(self, stock_data: Dict, klines=None) -> float:
        """市场 Beta: 用股票收益率对沪深300(000300)收益率做线性回归的斜率"""
        if klines and len(klines) >= 22:
            closes = np.array([float(k.get('close', 0)) for k in klines[-21:] if float(k.get('close', 0)) > 0])
            if len(closes) >= 2:
                stock_returns = np.diff(np.log(closes))
                # 用个股收益率标准差 / 假设市场收益率标准差(约0.015/日) 近似 beta
                stock_vol = np.std(stock_returns)
                market_daily_vol = 0.015  # 沪深300 日均波动率 ≈ 1.5%
                beta = stock_vol / market_daily_vol
                return float(np.clip(beta, 0.3, 3.0))
        # fallback: 用换手率粗略估计
        turnover = stock_data.get('turnover', 100)
        beta = np.log(turnover + 1) / np.log(1000 + 1) * 1.2
        return float(np.clip(beta, 0.5, 2.5))

    def _size(self, stock_data: Dict) -> float:
        """市值 (log 流通市值)"""
        cap = stock_data.get('circulating_cap', 1)
        return float(np.log(cap + 1))

    def _value(self, stock_data: Dict) -> float:
        """价值因子 (EP = EPS / Price)"""
        pe = stock_data.get('pe', 100)
        if pe > 0:
            return float(1.0 / pe)
        return 0.0

    def _growth(self, stock_data: Dict) -> float:
        """成长因子 (营收增速)"""
        return float(stock_data.get('revenue_growth', 0))

    def _liquidity(self, stock_data: Dict) -> float:
        """流动性 (日均成交额)"""
        amount = stock_data.get('amount', 0)
        return float(np.log(amount + 1))

    def _momentum(self, stock_data: Dict, klines=None) -> float:
        """动量 (过去12月收益, 排除最近1月)"""
        if klines and len(klines) >= 60:
            closes = [float(k.get('close', 0)) for k in klines if float(k.get('close', 0)) > 0]
            if len(closes) >= 60:
                # 过去 60 天收益 (排除最近 5 天)
                mom = (closes[-6] / closes[-1] - 1) * 100 if closes[-1] > 0 else 0
                return float(mom)
        return float(stock_data.get('change_pct', 0) * 10)

    def _volatility(self, stock_data: Dict, klines=None) -> float:
        """波动率 (过去60日收益率标准差)"""
        if klines and len(klines) >= 21:
            closes = np.array([float(k.get('close', 0)) for k in klines[-21:] if float(k.get('close', 0)) > 0])
            if len(closes) >= 2:
                returns = np.diff(np.log(closes))
                return float(np.std(returns) * np.sqrt(252) * 100)
        return 20.0  # 默认

    def _leverage(self, stock_data: Dict) -> float:
        """杠杆 (资产负债率)"""
        return float(stock_data.get('debt_ratio', 50))

    def _earnings_yield(self, stock_data: Dict) -> float:
        """盈利收益率 (EP)"""
        pe = stock_data.get('pe', 100)
        if pe > 0:
            return float(1.0 / pe)
        return 0.0

    def _restructuring(self, stock_data: Dict) -> float:
        """再融资 (增发/配股标记)"""
        return float(stock_data.get('restructuring', 0))

    @staticmethod
    def normalize_factor(factor_values: Dict[str, float]) -> Dict[str, float]:
        """横截面标准化 (Z-Score)"""
        if not factor_values:
            return {}
        vals = list(factor_values.values())
        mean = np.mean(vals)
        std = np.std(vals)
        if std < 1e-10:
            return {k: 0.0 for k in factor_values}
        return {k: (v - mean) / std for k, v in factor_values.items()}


class RiskDecomposer:
    """风险分解器"""

    def __init__(self, factor_cov_matrix: np.ndarray,
                 idio_variance: np.ndarray):
        """
        Args:
            factor_cov_matrix: 风格因子协方差矩阵 (N_factors x N_factors)
            idio_variance: 特异性方差 (N_stocks,)
        """
        self.factor_cov = factor_cov_matrix
        self.idio_var = idio_variance

    def decompose_variance(self, portfolio_weights: np.ndarray,
                           factor_exposures: np.ndarray) -> Dict:
        """
        风险分解: 系统性风险 vs 特异性风险

        Args:
            portfolio_weights: 组合权重 (N_stocks,)
            factor_exposures: 因子暴露 (N_stocks x N_factors)

        Returns:
            {systematic_risk, idiosyncratic_risk, total_risk, ...}
        """
        n = len(portfolio_weights)

        # 系统性风险: w' * F * Cov_F * F' * w
        F = factor_exposures  # (N, K)
        port_exposure = F.T @ portfolio_weights  # (K,)
        systematic = port_exposure @ self.factor_cov @ port_exposure

        # 特异性风险: w' * diag(sigma_i^2) * w
        idiosyncratic = sum(portfolio_weights[i] ** 2 * self.idio_var[i] for i in range(n))

        # 总风险
        total = systematic + idiosyncratic

        # 风险贡献
        marginal = F @ self.factor_cov @ port_exposure / (np.sqrt(systematic) + 1e-10)
        component = portfolio_weights * marginal

        return {
            'systematic_risk': float(systematic),
            'idiosyncratic_risk': float(idiosyncratic),
            'total_risk': float(total),
            'systematic_pct': float(systematic / (total + 1e-10)),
            'idiosyncratic_pct': float(idiosyncratic / (total + 1e-10)),
            'marginal_contribution': marginal.tolist(),
            'component_contribution': component.tolist(),
        }

    def top_contributors(self, decomposition: Dict, n: int = 5) -> List[Tuple[int, float]]:
        """风险贡献最大的 N 只股票"""
        components = decomposition.get('component_contribution', [])
        indexed = [(i, c) for i, c in enumerate(components)]
        indexed.sort(key=lambda x: abs(x[1]), reverse=True)
        return indexed[:n]


class RiskOptimizer:
    """风险优化器"""

    @staticmethod
    def maximize_sharpe(expected_returns: np.ndarray,
                        cov_matrix: np.ndarray,
                        risk_free: float = 0.02) -> np.ndarray:
        """最大化夏普比率的权重"""
        n = len(expected_returns)

        # 使用解析解 (无约束)
        # w* = Σ^-1 * (μ - r_f) / sum(Σ^-1 * (μ - r_f))
        try:
            inv_cov = np.linalg.inv(cov_matrix)
            excess_returns = expected_returns - risk_free
            raw_weights = inv_cov @ excess_returns
            # 归一化
            total = abs(raw_weights.sum())
            if total > 0:
                weights = raw_weights / total
            else:
                weights = np.ones(n) / n
            # 约束: 单股不超过 20%
            weights = np.clip(weights, -0.2, 0.2)
            weights /= weights.sum()
            return weights
        except np.linalg.LinAlgError:
            return np.ones(n) / n

    @staticmethod
    def minimize_volatility(cov_matrix: np.ndarray) -> np.ndarray:
        """最小方差组合权重"""
        n = cov_matrix.shape[0]
        try:
            inv_cov = np.linalg.inv(cov_matrix)
            ones = np.ones(n)
            raw = inv_cov @ ones
            total = raw.sum()
            if total != 0:
                return raw / total
            return np.ones(n) / n
        except np.linalg.LinAlgError:
            return np.ones(n) / n

    @staticmethod
    def risk_parity(cov_matrix: np.ndarray) -> np.ndarray:
        """风险平价权重"""
        n = cov_matrix.shape[0]
        weights = np.ones(n) / n

        for _ in range(100):  # 迭代优化
            port_var = weights @ cov_matrix @ weights
            if port_var < 1e-10:
                break
            marginal_risk = cov_matrix @ weights
            risk_contrib = weights * marginal_risk
            target_risk = port_var / n

            # 更新权重
            for i in range(n):
                if risk_contrib[i] > 1e-10:
                    weights[i] *= target_risk / risk_contrib[i]

            # 归一化
            weights /= weights.sum()

        return weights

    @staticmethod
    def black_litterman(equilibrium_weights: np.ndarray,
                        views_returns: np.ndarray,
                        view_confidences: np.ndarray,
                        cov_matrix: np.ndarray,
                        tau: float = 0.05) -> np.ndarray:
        """
        Black-Litterman 模型

        Args:
            equilibrium_weights: 均衡权重
            views_returns: 收益观点 (N_views,)
            view_confidences: 观点置信度 (N_views,)
            cov_matrix: 协方差矩阵
            tau: 先验方差缩放因子

        Returns:
            后验最优权重
        """
        n = len(equilibrium_weights)
        K = len(views_returns)

        if K == 0:
            return equilibrium_weights

        # 市场均衡收益: pi = tau * Sigma * w_mkt
        pi = tau * cov_matrix @ equilibrium_weights

        # 构建选择矩阵 P 和置信度矩阵 Omega
        P = np.eye(n, K)  # 简化: 每个观点针对一只股票
        omega_diag = 1.0 / (view_confidences + 1e-10)
        Omega = np.diag(omega_diag)

        # Black-Litterman 公式
        try:
            tau_sigma = tau * cov_matrix
            tau_sigma_inv = np.linalg.inv(tau_sigma)
            P_omega_inv = P @ np.linalg.inv(Omega + 1e-10 * np.eye(K))

            # 后验收益: mu_bl = (tau_sigma^-1 + P' Omega^-1 P)^-1 (tau_sigma^-1 pi + P' Omega^-1 q)
            M1 = tau_sigma_inv + P.T @ np.linalg.inv(Omega + 1e-10 * np.eye(K)) @ P
            M2 = tau_sigma_inv @ pi + P.T @ np.linalg.inv(Omega + 1e-10 * np.eye(K)) @ views_returns

            bl_returns = np.linalg.solve(M1, M2)

            # 最优权重
            inv_cov = np.linalg.inv(cov_matrix)
            weights = inv_cov @ bl_returns
            total = abs(weights.sum())
            if total > 0:
                weights /= total
            return np.clip(weights, -0.2, 0.2)
        except Exception:
            return equilibrium_weights

    @staticmethod
    def max_drawdown_constraint(weights: np.ndarray,
                                 cov_matrix: np.ndarray,
                                 max_dd: float = 0.15,
                                 days: int = 252) -> bool:
        """检查组合最大回撤是否在约束内"""
        port_vol = np.sqrt(weights @ cov_matrix @ weights)
        # 简化: max_dd ≈ 2 * z * port_vol / sqrt(252)
        estimated_dd = 2 * 1.645 * port_vol / np.sqrt(days)
        return estimated_dd <= max_dd


class RiskReportGenerator:
    """风险报告生成器"""

    @staticmethod
    def generate_report(weights: np.ndarray,
                        expected_returns: np.ndarray,
                        cov_matrix: np.ndarray,
                        risk_free: float = 0.02) -> Dict:
        """生成完整风险报告"""
        port_return = float(weights @ expected_returns)
        port_vol = float(np.sqrt(weights @ cov_matrix @ weights))
        sharpe = (port_return - risk_free) / (port_vol + 1e-10)

        # 年化
        ann_return = (1 + port_return) ** 252 - 1
        ann_vol = port_vol * np.sqrt(252)
        ann_sharpe = (ann_return - risk_free) / (ann_vol + 1e-10)

        # 最大回撤 (简化: 用 Volatility 估算)
        max_dd = 2 * 1.645 * ann_vol / np.sqrt(252)

        # 风险贡献
        marginal_risk = cov_matrix @ weights
        component_risk = weights * marginal_risk

        # 行业集中度 (简化: 按权重排序)
        top_indices = np.argsort(-weights)
        top_n = min(5, len(top_indices))

        return {
            'expected_return': round(port_return, 4),
            'annual_return': round(ann_return, 4),
            'volatility': round(port_vol, 4),
            'annual_volatility': round(ann_vol, 4),
            'sharpe_ratio': round(sharpe, 4),
            'annual_sharpe': round(ann_sharpe, 4),
            'max_drawdown': round(min(max_dd, 0.5), 4),
            'top_contributors': [
                {'index': int(top_indices[i]), 'weight': float(weights[top_indices[i]])}
                for i in range(top_n)
            ],
            'num_assets': len(weights),
        }


# 全局实例
barra_factors = BarraStyleFactors()

def create_default_risk_model(n_stocks: int = 10) -> Tuple[RiskDecomposer, np.ndarray]:
    """创建默认风险模型 (用于测试)"""
    # 简化的因子协方差矩阵
    factor_cov = np.eye(10) * 0.01
    idio_var = np.ones(n_stocks) * 0.04
    decomposer = RiskDecomposer(factor_cov, idio_var)
    return decomposer, factor_cov
