#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
cvar_position_manager.py — CVaR 约束仓位管理器 (2026-08-12)

整合 CVaR/EVT 尾部风险 + Kelly 公式 + Black-Litterman 贝叶斯估计，
实现风险约束下的最优仓位管理。

核心功能:
    1. Kelly 公式计算最优投注比例
    2. CVaR 约束缩减 Kelly 仓位（防止尾部风险过度暴露）
    3. Black-Litterman 贝叶斯估计市场均衡收益
    4. CVaR 约束下的 Black-Litterman 组合优化
    5. 动态仓位调整（基于市场波动率）

使用方式:
    from modules.cvar_position_manager import cvar_position_manager

    # 计算 CVaR 约束仓位
    position = cvar_position_manager.calculate_position(
        stock_code='sz300620',
        klines=kline_data,
        predictions=prediction_dict,
        portfolio_value=1000000,
    )
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from modules.logger import logger


class CVaRPositionManager:
    """
    CVaR 约束仓位管理器

    算法流程:
    1. 基于 Kelly 公式计算理论最优仓位
    2. 基于 CVaR 约束缩减仓位（Kelly Fraction = min(Kelly, CVaR_Ratio)）
    3. Black-Litterman 贝叶斯估计调整预期收益
    4. 动态波动率调整（高波动时降仓）
    """

    def __init__(self, kelly_fraction: float = 0.25, max_cvar_constraint: float = 0.02):
        """
        Args:
            kelly_fraction: Kelly 仓位比例（默认 25%，避免全 Kelly 过于激进）
            max_cvar_constraint: 最大 CVaR 约束（默认 2%，单尾最大损失）
        """
        self.kelly_fraction = kelly_fraction
        self.max_cvar_constraint = max_cvar_constraint
        self._risk_free_rate = 0.02  # 无风险利率

    def calculate_kelly(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        计算 Kelly 最优投注比例

        Kelly 公式: f* = (p * b - q) / b = (p * (b + 1) - 1) / b
        其中 p = 胜率, q = 1-p, b = 盈亏比 (avg_win / avg_loss)

        Args:
            win_rate: 胜率 (0-1)
            avg_win: 平均盈利
            avg_loss: 平均亏损 (绝对值)

        Returns:
            Kelly 仓位比例 (0-1)
        """
        if avg_loss == 0:
            return 0.0

        b = avg_win / avg_loss  # 盈亏比
        q = 1.0 - win_rate  # 败率
        f = (win_rate * b - q) / b  # 标准 Kelly 公式

        # 半 Kelly（更保守）
        f = f * self.kelly_fraction

        # 限制在 [0, 1]
        return float(np.clip(f, 0, 1.0))

    def calculate_cvar_constraint(self, returns: np.ndarray,
                                   confidence: float = 0.95) -> float:
        """
        计算 CVaR 约束比例

        CVaR 约束: 仓位 <= (最大可接受损失 - CVaR) / 仓位规模

        Args:
            returns: 收益率序列
            confidence: 置信水平

        Returns:
            CVaR 约束比例 (0-1)
        """
        if len(returns) < 20:
            return 1.0  # 数据不足，不约束

        # 历史 CVaR
        tail_returns = np.sort(returns)
        tail_idx = int(len(tail_returns) * (1 - confidence))
        if tail_idx <= 0:
            return 1.0

        cvar = -np.mean(tail_returns[:tail_idx])

        # CVaR 约束: 仓位与 CVaR 成反比
        if cvar <= 0:
            return 1.0

        constraint = self.max_cvar_constraint / cvar
        return float(np.clip(constraint, 0, 1.0))

    def black_litterman_equilibrium(self, market_caps: List[float],
                                      risk_aversion: float = 2.5) -> np.ndarray:
        """
        计算市场均衡收益 (Black-Litterman Implied Returns)

        pi = delta * Sigma * w_mkt

        Args:
            market_caps: 市值权重 [w_mkt]
            risk_aversion: 风险厌恶系数 (默认 2.5)

        Returns:
            均衡收益向量
        """
        n = len(market_caps)
        total_cap = sum(market_caps)
        if total_cap == 0:
            return np.zeros(n)

        w_mkt = np.array(market_caps) / total_cap
        # 简化的协方差矩阵（用历史波动率近似）
        # 实际应用中应使用历史协方差矩阵
        Sigma = np.diag([0.2 / np.sqrt(252)] * n)  # 假设 20% 年化波动
        pi = risk_aversion * Sigma @ w_mkt
        return pi

    def black_litterman_returns(self, equilibrium_returns: np.ndarray,
                                  views: Dict[int, float],
                                  view_confidences: Optional[Dict[int, float]] = None,
                                  tau: float = 0.05) -> np.ndarray:
        """
        Black-Litterman 贝叶斯估计

        Args:
            equilibrium_returns: 均衡收益向量
            views: 观点 {view_index: expected_return}
            view_confidences: 观点置信度 {view_index: confidence}
            tau: 先验不确定性缩放因子

        Returns:
            后验预期收益
        """
        n = len(equilibrium_returns)

        # 构建 P 矩阵 (pick matrix)
        k = len(views)  # 观点数量
        if k == 0:
            return equilibrium_returns

        P = np.zeros((k, n))
        q = np.zeros(k)
        omega_diag = []

        for i, (view_idx, view_return) in enumerate(views.items()):
            if view_idx < n:
                P[i, view_idx] = 1.0
                q[i] = view_return
                conf = (view_confidences or {}).get(view_idx, 0.5)
                # omega = 置信度的倒数（置信度越高，omega 越小）
                omega_diag.append(tau * (1 - conf) / max(conf, 0.01))

        P = np.array(P)
        q = np.array(q)
        Omega = np.diag(omega_diag)

        # Black-Litterman 公式
        # M = (tau * Sigma)^-1
        # Sigma_bl = (P' Omega^-1 P + (tau * Sigma)^-1)^-1
        # mu_bl = Sigma_bl @ (P' Omega^-1 q + (tau * Sigma)^-1 @ pi)
        try:
            tau_sigma = tau * np.eye(n)
            tau_sigma_inv = np.linalg.inv(tau_sigma)
            omega_inv = np.linalg.inv(Omega)

            # 后验协方差
            M = tau_sigma_inv
            posterior_cov = np.linalg.inv(P.T @ omega_inv @ P + M)

            # 后验期望收益
            posterior_mean = posterior_cov @ (P.T @ omega_inv @ q + M @ equilibrium_returns)
            return posterior_mean
        except np.linalg.LinAlgError:
            logger.warning("[CVaRPositionManager] Black-Litterman 矩阵奇异，返回均衡收益")
            return equilibrium_returns

    def calculate_position(self, stock_code: str,
                           klines: List[Dict],
                           predictions: Dict,
                           portfolio_value: float = 1000000,
                           confidence: float = 0.95) -> Dict:
        """
        计算 CVaR 约束下的最优仓位

        Args:
            stock_code: 股票代码
            klines: K 线数据
            predictions: 预测结果 {direction, confidence, target_price, ...}
            portfolio_value: 组合总价值
            confidence: 置信水平

        Returns:
            仓位建议
        """
        if not klines or len(klines) < 60:
            return {
                'success': False,
                'error': '数据不足',
                'stock_code': stock_code,
            }

        # 1. 计算收益率
        closes = np.array([float(k.get('close', 0)) for k in klines if float(k.get('close', 0)) > 0])
        if len(closes) < 60:
            return {
                'success': False,
                'error': '数据不足',
                'stock_code': stock_code,
            }

        returns = np.diff(np.log(closes))

        # 2. Kelly 计算
        daily_pnl = returns * np.sign(returns)
        positive_days = daily_pnl[daily_pnl > 0]
        negative_days = daily_pnl[daily_pnl < 0]

        win_rate = len(positive_days) / len(daily_pnl) if len(daily_pnl) > 0 else 0.5
        avg_win = float(np.mean(positive_days)) if len(positive_days) > 0 else 0.01
        avg_loss = float(np.mean(np.abs(negative_days))) if len(negative_days) > 0 else 0.01

        kelly = self.calculate_kelly(win_rate, avg_win, avg_loss)

        # 3. CVaR 约束
        cvar_constraint = self.calculate_cvar_constraint(returns, confidence)

        # 4. 综合仓位（取 Kelly 和 CVaR 约束的较小值）
        raw_position = min(kelly, cvar_constraint)

        # 5. 预测置信度调整
        pred_confidence = predictions.get('confidence', 0.5)
        adjusted_position = raw_position * pred_confidence

        # 6. 波动率调整（高波动时降仓）
        annualized_vol = np.std(returns) * np.sqrt(252)
        vol_adjustment = 1.0
        if annualized_vol > 0.4:  # 波动率 > 40%
            vol_adjustment = 0.5
        elif annualized_vol > 0.3:
            vol_adjustment = 0.75

        final_position = adjusted_position * vol_adjustment
        final_position = float(np.clip(final_position, 0, 1.0))

        # 7. CVaR 计算
        tail_returns = np.sort(returns)
        tail_idx = int(len(tail_returns) * (1 - confidence))
        cvar = -np.mean(tail_returns[:max(tail_idx, 1)]) if tail_idx > 0 else 0.0

        return {
            'success': True,
            'stock_code': stock_code,
            'timestamp': datetime.now().isoformat(),
            'position': {
                'kelly_fraction': round(kelly, 4),
                'cvar_constraint': round(cvar_constraint, 4),
                'raw_position': round(raw_position, 4),
                'confidence_adjusted': round(adjusted_position, 4),
                'vol_adjusted': round(final_position, 4),
                'portfolio_value': portfolio_value,
                'position_value': round(final_position * portfolio_value, 2),
            },
            'risk': {
                'cvar_95': round(cvar * 100, 4),
                'annualized_vol': round(annualized_vol * 100, 2),
                'win_rate': round(win_rate * 100, 2),
                'avg_win': round(avg_win * 100, 4),
                'avg_loss': round(avg_loss * 100, 4),
            },
            'signal': {
                'direction': predictions.get('direction', 'hold'),
                'confidence': pred_confidence,
                'target_price': predictions.get('target_price'),
            },
        }

    def get_status(self) -> Dict:
        """获取状态"""
        return {
            'module': 'CVaRPositionManager',
            'kelly_fraction': self.kelly_fraction,
            'max_cvar_constraint': self.max_cvar_constraint,
            'risk_free_rate': self._risk_free_rate,
            'status': 'active',
            'timestamp': datetime.now().isoformat(),
        }


# 全局单例
cvar_position_manager = CVaRPositionManager()
