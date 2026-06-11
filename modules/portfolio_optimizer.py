"""
Portfolio Optimizer Module - Long-term Optimization
Implements portfolio optimization with risk parity and diversification
"""

import math
from typing import Dict, List, Optional
from datetime import datetime, timedelta


class PortfolioOptimizer:
    """Portfolio optimization engine with risk parity and diversification"""
    
    def __init__(self):
        # Risk budget parameters
        self.risk_budget = {
            'max_single_stock_risk': 0.15,
            'max_sector_risk': 0.30,
            'min_diversification_score': 0.4,
            'target_volatility': 0.15
        }
        
        # Optimization parameters
        self.opt_params = {
            'max_iterations': 1000,
            'convergence_threshold': 1e-6,
            'risk_free_rate': 0.02
        }
    
    def calculate_portfolio_variance(self, weights: List[float], 
                                      correlation_matrix: List[List[float]]) -> float:
        """Calculate portfolio variance using covariance matrix"""
        n = len(weights)
        variance = 0
        
        for i in range(n):
            for j in range(n):
                variance += weights[i] * weights[j] * correlation_matrix[i][j]
        
        return variance
    
    def calculate_portfolio_volatility(self, weights: List[float],
                                        correlation_matrix: List[List[float]]) -> float:
        """Calculate portfolio volatility"""
        variance = self.calculate_portfolio_variance(weights, correlation_matrix)
        return math.sqrt(variance)
    
    def calculate_risk_parity_weights(self, correlation_matrix: List[List[float]],
                                       target_volatility: float = 0.15) -> List[float]:
        """
        Calculate risk parity weights
        Each asset contributes equally to portfolio risk
        """
        n = len(correlation_matrix)
        
        # Initialize with equal weights
        weights = [1.0 / n] * n
        
        # Iterative optimization
        for _ in range(self.opt_params['max_iterations']):
            # Calculate portfolio volatility
            port_vol = self.calculate_portfolio_volatility(weights, correlation_matrix)
            
            # Calculate marginal risk contribution
            marginal_risk = []
            for i in range(n):
                mr = 0
                for j in range(n):
                    mr += weights[j] * correlation_matrix[i][j]
                marginal_risk.append(mr)
            
            # Calculate risk contribution
            risk_contribution = [weights[i] * marginal_risk[i] for i in range(n)]
            total_risk = sum(risk_contribution)
            
            # Calculate risk parity error
            risk_parity_error = 0
            for i in range(n):
                if total_risk > 0:
                    risk_share = risk_contribution[i] / total_risk
                    risk_parity_error += (risk_share - 1/n) ** 2
            
            # Check convergence
            if risk_parity_error < self.opt_params['convergence_threshold']:
                break
            
            # Update weights
            new_weights = []
            for i in range(n):
                if total_risk > 0:
                    risk_share = risk_contribution[i] / total_risk
                    new_weights.append(weights[i] * (1/n) / risk_share)
                else:
                    new_weights.append(weights[i])
            
            # Normalize weights
            total_weight = sum(new_weights)
            if total_weight > 0:
                weights = [w / total_weight for w in new_weights]
        
        return [round(w, 4) for w in weights]
    
    def optimize_portfolio(self, stocks: List[Dict], 
                           correlation_analysis: Dict) -> Dict:
        """
        Main portfolio optimization function
        Combines risk parity with diversification constraints
        """
        if not stocks:
            return {'optimized_weights': [], 'portfolio_metrics': {}}
        
        n = len(stocks)
        correlation_matrix = correlation_analysis.get('correlation_matrix', [])
        
        # Calculate risk parity weights
        risk_parity_weights = self.calculate_risk_parity_weights(correlation_matrix)
        
        # Apply diversification constraints
        constrained_weights = []
        for i in range(n):
            weight = risk_parity_weights[i]
            # Apply max single stock risk constraint
            weight = min(weight, self.risk_budget['max_single_stock_risk'])
            constrained_weights.append(weight)
        
        # Normalize constrained weights
        total_weight = sum(constrained_weights)
        if total_weight > 0:
            constrained_weights = [w / total_weight for w in constrained_weights]
        
        # Calculate portfolio metrics
        portfolio_volatility = self.calculate_portfolio_volatility(
            constrained_weights, correlation_matrix
        )
        diversification_score = correlation_analysis.get('diversification_score', 0.5)
        
        # Calculate expected return (simplified)
        avg_expected_return = sum(s.get('expected_return', 0.1) for s in stocks) / n
        weighted_return = sum(constrained_weights[i] * stocks[i].get('expected_return', 0.1) 
                             for i in range(n))
        
        # Calculate Sharpe ratio
        sharpe_ratio = (weighted_return - self.opt_params['risk_free_rate']) / portfolio_volatility \
            if portfolio_volatility > 0 else 0
        
        return {
            'optimized_weights': constrained_weights,
            'risk_parity_weights': risk_parity_weights,
            'portfolio_metrics': {
                'volatility': round(portfolio_volatility, 4),
                'expected_return': round(weighted_return, 4),
                'sharpe_ratio': round(sharpe_ratio, 3),
                'diversification_score': diversification_score,
                'num_stocks': n
            }
        }
    
    def calculate_position_allocation(self, total_capital: float,
                                       stock_weights: List[float]) -> List[Dict]:
        """Calculate position allocation based on optimized weights"""
        allocations = []
        for i, weight in enumerate(stock_weights):
            position_value = total_capital * weight
            allocations.append({
                'index': i,
                'weight': round(weight, 4),
                'position_value': round(position_value, 2),
                'allocation_pct': round(weight * 100, 2)
            })
        return allocations

    # ── Black-Litterman 组合优化（P2 增强） ─────────────────────

    def black_litterman_weights(self, market_caps: List[float],
                                 views: List[Dict],
                                 tau: float = 0.05,
                                 risk_aversion: float = 2.0) -> List[float]:
        """
        Black-Litterman 组合优化

        Args:
            market_caps: 各股票的市值列表
            views: 主观观点列表，每项 {'asset': 索引, 'return': 预期收益率, 'confidence': 0-1}
            tau: 市场均衡协方差缩放因子
            risk_aversion: 风险厌恶系数

        Returns:
            优化后的权重列表
        """
        import numpy as np

        n = len(market_caps)
        total_cap = sum(market_caps)
        if total_cap == 0:
            return [1.0 / n] * n

        # 市场隐含均衡收益 pi = delta * Sigma * w_mkt
        w_mkt = np.array([mc / total_cap for mc in market_caps])

        # 简化: 假设对角协方差矩阵
        vols = np.array([0.3] * n)  # 假设 30% 波动率
        Sigma = np.diag(vols ** 2)

        pi = risk_aversion * Sigma @ w_mkt  # 均衡收益

        # 处理主观观点
        if views:
            P = np.zeros((len(views), n))  # 观点矩阵
            Q = np.zeros(len(views))  # 观点收益率
            omega_diag = []  # 观点不确定性

            for idx, view in enumerate(views):
                asset = view.get('asset', 0)
                ret = view.get('return', 0)
                conf = view.get('confidence', 0.5)

                P[idx, asset] = 1.0
                Q[idx] = ret
                # 不确定性 = (1/conf - 1) * tau * P @ Sigma @ P.T
                omega_val = (1.0 / max(conf, 0.01) - 1) * tau * (P[idx] @ Sigma @ P[idx])
                omega_diag.append(max(omega_val, 1e-10))

            Omega = np.diag(omega_diag)

            # Black-Litterman 公式
            tau_Sigma = tau * Sigma
            M = np.linalg.inv(np.linalg.inv(tau_Sigma) + P.T @ np.linalg.inv(Omega) @ P)

            # 后验期望收益
            mu_bl = M @ (np.linalg.inv(tau_Sigma) @ pi + P.T @ np.linalg.inv(Omega) @ Q)

            # 后验协方差
            Sigma_bl = M + mu_bl.reshape(-1, 1) @ mu_bl.reshape(1, -1)

            # 最优权重 w* = (1/delta) * Sigma_bl^{-1} * mu_bl
            try:
                w_opt = (1.0 / risk_aversion) * np.linalg.inv(Sigma_bl) @ mu_bl
                # 确保正权重并归一化
                w_opt = np.maximum(w_opt, 0)
                total_w = w_opt.sum()
                if total_w > 0:
                    w_opt = w_opt / total_w
                else:
                    w_opt = w_mkt  # 回退到市场权重
            except Exception:
                w_opt = w_mkt
        else:
            # 无观点，使用市场权重
            w_opt = w_mkt

        return [round(float(w), 4) for w in w_opt]

    def black_litterman_summary(self, stocks: List[Dict],
                                 views: List[Dict]) -> Dict:
        """Black-Litterman 优化摘要"""
        market_caps = [s.get('market_cap', 100) for s in stocks]
        weights = self.black_litterman_weights(market_caps, views)

        return {
            'method': 'Black-Litterman',
            'weights': {stocks[i]['name']: weights[i] for i in range(len(stocks))},
            'num_views': len(views),
            'total_weight': round(sum(weights), 4),
        }
