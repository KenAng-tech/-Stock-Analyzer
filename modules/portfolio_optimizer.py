"""
Portfolio Optimizer Module - Long-term Optimization
Implements portfolio optimization with risk parity and diversification
"""

import math
import numpy as np
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

    # ── Ledoit-Wolf 协方差矩阵估计（Phase 3 增强） ─────────────────────

    def ledoit_wolf_covariance(self, returns: np.ndarray,
                                shrinkage_target: str = 'constant_variance') -> np.ndarray:
        """
        Ledoit-Wolf 收缩估计

        Args:
            returns: (T, n) 收益率矩阵
            shrinkage_target: 'constant_variance' | 'single_factor' | 'identity'

        Returns:
            收缩后的协方差矩阵
        """
        n = returns.shape[1]
        T = returns.shape[0]

        # 样本协方差
        Sigma_hat = np.cov(returns, rowvar=False)

        if shrinkage_target == 'constant_variance':
            # 目标: 对角矩阵 (各资产方差相等)
            delta = np.mean(np.diag(Sigma_hat))
            target = delta * np.eye(n)
        elif shrinkage_target == 'single_factor':
            # 目标: 单因子模型
            beta = returns.mean(axis=0) / (np.var(returns[:, 0]) + 1e-10)
            target = np.outer(beta, beta) * np.var(np.mean(returns, axis=1))
            target += delta * np.eye(n)
        else:
            target = np.diag(np.diag(Sigma_hat))

        # 计算收缩系数
        y = Sigma_hat - target
        y_sq = y @ y
        phi = np.sum(y ** 2) - np.sum(y_sq) / T
        c = np.sum(y_sq) / T
        shrinkage = max(0, min(1, phi / (c + 1e-10)))

        # 收缩估计
        Sigma_shrunk = shrinkage * target + (1 - shrinkage) * Sigma_hat

        # 确保对称正定
        Sigma_shrunk = (Sigma_shrunk + Sigma_shrunk.T) / 2
        eigvals = np.linalg.eigvalsh(Sigma_shrunk)
        if np.min(eigvals) < 1e-10:
            Sigma_shrunk += (1e-10 - np.min(eigvals)) * np.eye(n)

        return Sigma_shrunk

    # ── CVaR 优化（Phase 3 增强） ─────────────────────

    def calculate_cvar(self, returns: np.ndarray,
                       weights: np.ndarray,
                       confidence: float = 0.95) -> float:
        """
        CVaR (Conditional Value at Risk)

        Args:
            returns: (T, n) 收益率
            weights: (n,) 权重
            confidence: 置信水平 (0.95 = 95%)

        Returns:
            CVaR 值 (负数表示损失)
        """
        portfolio_returns = returns @ weights
        var = np.percentile(portfolio_returns, (1 - confidence) * 100)
        cvar = portfolio_returns[portfolio_returns <= var].mean()
        return float(cvar)

    def cvar_optimization(self, returns: np.ndarray,
                          target_return: float = 0.001,
                          confidence: float = 0.95) -> Dict:
        """
        CVaR 最小化优化

        最小化 CVaR 约束预期收益 ≥ target_return
        """
        n = returns.shape[1]
        T = returns.shape[0]

        # 使用简单网格搜索
        best_weights = np.ones(n) / n
        best_cvar = float('inf')

        # 生成随机权重样本
        n_samples = 10000
        random_weights = np.random.dirichlet(np.ones(n), n_samples)

        for weights in random_weights:
            port_ret = returns @ weights
            expected_ret = np.mean(port_ret)

            if expected_ret >= target_return:
                cvar = self.calculate_cvar(returns, weights, confidence)
                if cvar < best_cvar:
                    best_cvar = cvar
                    best_weights = weights.copy()

        return {
            'weights': best_weights.tolist(),
            'cvar': round(best_cvar, 6),
            'expected_return': float(np.mean(returns @ best_weights)),
            'confidence': confidence,
            'n_assets': n,
        }

    # ── 风险平价（协方差矩阵版，供综合优化调用） ─────────────────────

    def risk_parity_weights(self, covariance_matrix: np.ndarray,
                            risk_budgets: Optional[np.ndarray] = None,
                            max_weight: float = 0.25) -> np.ndarray:
        """
        风险平价权重（基于协方差矩阵）

        每个资产对组合风险的贡献与其风险预算成比例
        默认等风险预算 → 经典风险平价

        Args:
            covariance_matrix: (n, n) 协方差矩阵
            risk_budgets: (n,) 风险预算 (默认等权重)
            max_weight: 单个资产最大权重约束

        Returns:
            优化后的权重 (n,)
        """
        n = covariance_matrix.shape[0]
        if risk_budgets is None:
            risk_budgets = np.ones(n) / n

        risk_budgets = risk_budgets / risk_budgets.sum()  # 归一化
        weights = np.ones(n) / n

        for _ in range(100):
            marginal_risk = covariance_matrix @ weights
            risk_contrib = weights * marginal_risk
            total_risk = risk_contrib.sum()

            target = risk_budgets * total_risk
            new_weights = weights * (target / (risk_contrib + 1e-10))

            # 最大权重约束
            new_weights = np.minimum(new_weights, max_weight)
            new_weights /= new_weights.sum()

            if np.max(np.abs(new_weights - weights)) < 1e-8:
                break
            weights = new_weights

        return weights

    # ── 最大分散化（Maximum Diversification） ─────────────────────

    def max_diversification_weights(self, covariance_matrix: np.ndarray,
                                     max_weight: float = 0.25) -> np.ndarray:
        """
        最大分散化组合优化 (Maillard et al. 2010)

        最大化分散化比率 DR(w) = (w^T sigma) / sqrt(w^T Sigma w)
        其中 sigma 为各资产波动率向量

        采用多起点搜索 + 局部梯度上升:
        1. 从多个候选权重 (随机/逆波动率/风险平价) 启动
        2. 对每个候选做投影梯度上升
        3. 选择分散化比率最高的结果

        Args:
            covariance_matrix: (n, n) 协方差矩阵
            max_weight: 单个资产最大权重

        Returns:
            优化后的权重 (n,)
        """
        n = covariance_matrix.shape[0]
        vols = np.sqrt(np.diag(covariance_matrix))

        def diversification_ratio(w):
            num = vols @ w
            den = np.sqrt(w @ covariance_matrix @ w)
            return num / (den + 1e-10)

        def project(w):
            w = np.maximum(w, 0)
            w = np.minimum(w, max_weight)
            s = w.sum()
            return w / s if s > 0 else np.ones(n) / n

        def local_search(init_w, n_iter=200, lr=0.01):
            w = init_w.copy()
            best_w = w.copy()
            best_dr = diversification_ratio(w)
            for _ in range(n_iter):
                port_vol = np.sqrt(w @ covariance_matrix @ w)
                Sigma_w = covariance_matrix @ w
                # DR 梯度
                grad = vols / (port_vol + 1e-10) - \
                       (Sigma_w @ vols) / (port_vol ** 3 + 1e-10) * (w @ vols)
                new_w = w + lr * grad
                new_w = project(new_w)
                dr = diversification_ratio(new_w)
                if dr > best_dr:
                    best_dr = dr
                    best_w = new_w.copy()
                    w = new_w
                else:
                    lr *= 0.5  # 退火
                    if lr < 1e-8:
                        break
            return best_w

        # 候选起点
        candidates = []

        # 1. 等权 + 随机扰动 (多次)
        for _ in range(5):
            noise = np.random.dirichlet(np.ones(n), 1)[0]
            candidates.append(project(np.ones(n) / n * 0.8 + noise * 0.2))

        # 2. 逆波动率权重
        inv_vol = 1.0 / (vols + 1e-10)
        candidates.append(project(inv_vol / inv_vol.sum()))

        # 3. 逆方差权重
        inv_var = 1.0 / (vols ** 2 + 1e-10)
        candidates.append(project(inv_var / inv_var.sum()))

        # 4. 最小方差权重 (解析解)
        try:
            Sigma_inv = np.linalg.inv(covariance_matrix)
            inv_var_vec = Sigma_inv.sum(axis=1)
            candidates.append(project(inv_var_vec))
        except Exception:
            pass

        # 5. 风险平价权重
        rp = self.risk_parity_weights(covariance_matrix)
        candidates.append(rp)

        # 6. 等权
        candidates.append(np.ones(n) / n)

        # 对每个候选做局部搜索
        best_weights = np.ones(n) / n
        best_dr = diversification_ratio(best_weights)

        for cand in candidates:
            w = local_search(cand)
            dr = diversification_ratio(w)
            if dr > best_dr:
                best_dr = dr
                best_weights = w

        return best_weights

    # ── 层次风险平价（Hierarchical Risk Parity, HRP） ─────────────────────

    def hierarchical_risk_parity(self, covariance_matrix: np.ndarray,
                                  max_weight: float = 0.25) -> np.ndarray:
        """
        层次风险平价 (HRP) — Lillo et al. 2017

        步骤:
          1. 对资产相关性做层次聚类
          2. 按聚类树状图递归分配权重 (quasi-diversification)

        优点: 不需要矩阵求逆，对估计误差鲁棒，计算高效

        Args:
            covariance_matrix: (n, n) 协方差矩阵
            max_weight: 单个资产最大权重

        Returns:
            HRP 权重 (n,)
        """
        n = covariance_matrix.shape[0]
        if n == 1:
            return np.array([1.0])

        # 1. 计算距离矩阵 (基于相关系数)
        corr = covariance_matrix.copy()
        np.fill_diagonal(corr, 1.0)
        # 相关系数转距离
        dist = np.sqrt(0.5 * (1 - corr))
        # 确保对称非负
        dist = (dist + dist.T) / 2
        np.fill_diagonal(dist, 0.0)
        dist = np.maximum(dist, 0.0)

        # 2. 层次聚类 (最近邻链接)
        order = self._clustered_order(dist)

        # 3. 递归二分分配权重
        weights = np.ones(n) / n
        weights = self._hrp_recursive(covariance_matrix, order, 0, n - 1, weights)

        # 应用最大权重约束
        weights = np.minimum(weights, max_weight)
        weights /= weights.sum()

        return weights

    def _clustered_order(self, dist_matrix: np.ndarray) -> List[int]:
        """
        基于距离矩阵的层次聚类，返回聚类顺序的索引列表

        使用最近邻链接 (single linkage)，递归二分
        """
        n = dist_matrix.shape[0]
        if n <= 2:
            return list(range(n))

        n_assets = dist_matrix.shape[0]
        clusters = {i: [i] for i in range(n_assets)}
        # 预分配足够大的距离矩阵 (最多 n-1 次合并)
        max_id = 2 * n
        d = np.full((max_id, max_id), np.inf)
        d[:n, :n] = dist_matrix.copy()
        active = list(range(n_assets))
        merge_history = []

        while len(active) > 1:
            min_dist = float('inf')
            i_best, j_best = active[0], active[1]
            for i_idx in range(len(active)):
                for j_idx in range(i_idx + 1, len(active)):
                    i, j = active[i_idx], active[j_idx]
                    if d[i][j] < min_dist:
                        min_dist = d[i][j]
                        i_best, j_best = i, j

            new_cluster = clusters[i_best] + clusters[j_best]
            new_id = n_assets + len(merge_history)
            clusters[new_id] = new_cluster
            merge_history.append((i_best, j_best, new_id))

            # 更新距离 (最近邻)
            for k in active:
                if k != i_best and k != j_best:
                    d[k][new_id] = d[new_id][k] = min(d[i_best][k], d[j_best][k])

            active.remove(i_best)
            active.remove(j_best)
            active.append(new_id)

        def get_order(node_id):
            merged = None
            for a, b, c in merge_history:
                if c == node_id:
                    merged = (a, b)
                    break
            if merged is None:
                return clusters[node_id]
            a, b = merged
            return get_order(a) + get_order(b)

        final_cluster = active[0]
        return get_order(final_cluster)

    def _hrp_recursive(self, cov: np.ndarray, order: List[int],
                       low: int, high: int, weights: np.ndarray) -> np.ndarray:
        """
        HRP 递归二分权重分配 (quasi-diversification)

        在每次递归中，将当前簇分为两半，按逆波动率比例分配权重
        """
        if low >= high:
            return weights

        mid = (low + high) // 2
        cluster_left = order[low:mid + 1]
        cluster_right = order[mid + 1:high + 1]

        # 计算每个子簇的方差 (基于子集协方差)
        def cluster_variance(indices):
            sub = cov[np.ix_(indices, indices)]
            # 等权权重下的方差
            w = np.ones(len(indices)) / len(indices)
            return float(w @ sub @ w)

        var_left = cluster_variance(cluster_left)
        var_right = cluster_variance(cluster_right)

        # 逆方差加权分配 (1/var 权重)
        total_var = var_left + var_right
        if total_var < 1e-10:
            alloc_left = 0.5
        else:
            alloc_left = var_right / total_var  # 逆方差: 方差大的分配少

        # 分配权重
        for i in cluster_left:
            weights[i] *= alloc_left
        for i in cluster_right:
            weights[i] *= (1 - alloc_left)

        # 递归
        self._hrp_recursive(cov, order, low, mid, weights)
        self._hrp_recursive(cov, order, mid + 1, high, weights)

        return weights

    # ── 风险预算优化（Phase 3 增强） ─────────────────────

    def risk_budget_optimization(self, covariance_matrix: np.ndarray,
                                  risk_budgets: Optional[np.ndarray] = None,
                                  max_weight: float = 0.25) -> np.ndarray:
        """
        风险预算优化

        每个资产按指定风险预算分配权重

        Args:
            covariance_matrix: (n, n) 协方差矩阵
            risk_budgets: (n,) 风险预算 (默认等权重)
            max_weight: 单个资产最大权重

        Returns:
            优化后的权重
        """
        n = covariance_matrix.shape[0]
        if risk_budgets is None:
            risk_budgets = np.ones(n) / n

        # 迭代优化
        weights = np.ones(n) / n
        max_iter = 100
        tol = 1e-8

        for _ in range(max_iter):
            # 边际风险贡献
            marginal_risk = covariance_matrix @ weights
            risk_contrib = weights * marginal_risk
            total_risk = risk_contrib.sum()

            # 目标风险贡献
            target_risk = risk_budgets * total_risk

            # 更新权重
            new_weights = weights * (target_risk / (risk_contrib + 1e-10))

            # 应用最大权重约束
            new_weights = np.minimum(new_weights, max_weight)

            # 归一化
            new_weights /= new_weights.sum()

            # 收敛检查
            if np.max(np.abs(new_weights - weights)) < tol:
                break

            weights = new_weights

        return weights

    # ── 综合优化摘要（Phase 3 增强） ─────────────────────

    def comprehensive_optimization(self, stocks: List[Dict],
                                    returns: Optional[np.ndarray] = None,
                                    views: Optional[List[Dict]] = None,
                                    risk_budgets: Optional[np.ndarray] = None) -> Dict:
        """
        综合优化: 结合 Risk Parity + Black-Litterman + CVaR + Risk Budget

        Returns:
            包含所有优化结果的字典
        """
        n = len(stocks)
        if n == 0:
            return {'error': 'No stocks'}

        # 计算收益率矩阵
        if returns is None:
            # 简化: 使用历史收益率
            market_caps = np.array([s.get('market_cap', 100) for s in stocks])
            returns = np.random.randn(252, n) * 0.02  # 模拟

        # 1. Ledoit-Wolf 协方差
        cov_matrix = self.ledoit_wolf_covariance(returns)

        # 2. Risk Parity
        rp_weights = self.calculate_risk_parity_weights(cov_matrix.tolist())
        rp_weights = np.array(rp_weights)

        # 3. Risk Budget
        rb_weights = self.risk_budget_optimization(cov_matrix, risk_budgets)

        # 4. CVaR
        cvar_result = self.cvar_optimization(returns)

        # 5. Black-Litterman
        market_caps = [s.get('market_cap', 100) for s in stocks]
        bl_weights = self.black_litterman_weights(market_caps, views or [])

        # 6. 综合权重 (等权平均)
        all_weights = np.array([rp_weights, rb_weights, cvar_result['weights'], bl_weights])
        composite_weights = np.mean(all_weights, axis=0)

        # 计算综合指标
        port_vol = np.sqrt(composite_weights @ cov_matrix @ composite_weights)
        port_return = np.mean(returns @ composite_weights)
        sharpe = (port_return - self.opt_params['risk_free_rate']) / port_vol if port_vol > 0 else 0
        cvar = self.calculate_cvar(returns, composite_weights)

        return {
            'method': 'Comprehensive (RP + RB + CVaR + BL)',
            'weights': {stocks[i]['name']: round(float(w), 4) for i, w in enumerate(composite_weights)},
            'risk_parity_weights': rp_weights.tolist(),
            'risk_budget_weights': rb_weights.tolist(),
            'cvar_weights': cvar_result['weights'],
            'black_litterman_weights': bl_weights,
            'portfolio_metrics': {
                'volatility': round(float(port_vol), 4),
                'expected_return': round(float(port_return), 4),
                'sharpe_ratio': round(float(sharpe), 3),
                'cvar': round(cvar, 6),
                'num_assets': n,
            },
            'covariance_matrix': cov_matrix.tolist(),
        }


# ── HRP 层次聚类组合优化 ─────────────────────────────────────────

try:
    from scipy.cluster.hierarchy import linkage, fcluster, leaves_list
    from scipy.spatial.distance import squareform
    HAS_SCIPY_HIER = True
except ImportError:
    HAS_SCIPY_HIER = False


class HRPOptimizer:
    """
    Hierarchical Risk Parity — 层次聚类风险平价

    集成自 modules/models/hrp_optimizer.py
    作为 PortfolioOptimizer 的补充方法
    """

    def __init__(self, linkage_method='ward'):
        self.linkage_method = linkage_method
        self.hierarchy = None
        self.order = None
        self.weights = None

    def _correlation_matrix(self, cov_matrix):
        """从协方差矩阵计算相关系数矩阵"""
        diag = np.diag(cov_matrix)
        std_dev = np.sqrt(np.outer(diag, diag))
        std_dev = np.where(std_dev == 0, 1, std_dev)
        return cov_matrix / std_dev

    def optimize(self, cov_matrix, tickers=None):
        """
        执行 HRP 优化

        Args:
            cov_matrix: 协方差矩阵
            tickers: 资产名称列表

        Returns:
            权重字典
        """
        if not HAS_SCIPY_HIER:
            raise ImportError("HRP 需要 scipy.cluster.hierarchy")

        n = cov_matrix.shape[0]
        if tickers is None:
            tickers = [f'asset_{i}' for i in range(n)]

        # 层次聚类
        corr = self._correlation_matrix(cov_matrix)
        dist = np.sqrt(0.5 * (1 - corr))
        dist_cond = squareform(np.clip(dist, 0, None))
        self.hierarchy = linkage(dist_cond, method=self.linkage_method)
        self.order = leaves_list(self.hierarchy)

        # 递归二分
        reordered_cov = cov_matrix[np.ix_(self.order, self.order)]
        ordered_tickers = [tickers[i] for i in self.order]
        weights = self._recursive_bisect(reordered_cov, ordered_tickers)

        # 恢复原始顺序
        self.weights = {ticker: weights.get(ticker, 0.0) for ticker in tickers}
        return self.weights

    def _recursive_bisect(self, cov_matrix, tickers):
        """递归二分法分配权重"""
        n = len(tickers)
        weights = {t: 0.0 for t in tickers}

        if n == 1:
            weights[tickers[0]] = 1.0
            return weights

        mid = n // 2
        left_tickers = tickers[:mid]
        right_tickers = tickers[mid:]

        left_var = self._group_variance(cov_matrix, list(range(mid)))
        right_var = self._group_variance(cov_matrix, list(range(mid, n)))

        alpha = 1 - (left_var + right_var) / (left_var ** 2 + right_var ** 2 + 1e-10)
        alpha = max(0, min(1, alpha))  # 防止负权重

        left_weights = self._recursive_bisect(cov_matrix, left_tickers)
        right_weights = self._recursive_bisect(cov_matrix, right_tickers)

        for t, w in left_weights.items():
            weights[t] = w * alpha
        for t, w in right_weights.items():
            weights[t] = w * (1 - alpha)

        return weights

    @staticmethod
    def _group_variance(cov_matrix, idx):
        """计算资产组的逆方差"""
        sub = cov_matrix[np.ix_(idx, idx)]
        n = len(idx)
        ones = np.ones(n)
        var = ones @ sub @ ones / (n ** 2)
        return var if var > 0 else 1.0

    def get_status(self):
        return {
            'method': 'HRP',
            'linkage_method': self.linkage_method,
            'weights': self.weights or {},
            'n_tickers': len(self.weights) if self.weights else 0,
        }
