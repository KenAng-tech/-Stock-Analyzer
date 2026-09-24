#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
因果推断因子选择器 — Causal Factor Selector (2026 SOTA)

从相关性分析升级到因果推断:

架构:
    1. Double ML 估计每个因子的因果效应
    2. PC 算法做因果发现 (构建因子因果图)
    3. 过滤: 因果效应显著 + IC > 阈值的因子
    4. 识别伪因子: IC 高但因果效应不显著

核心思想:
- 相关性: X 和 Y 相关 (不知道方向, 可能是伪相关)
- 因果性: X → Y (X 导致 Y, 控制其他变量后仍然显著)

应用场景:
- 因子选择: 只保留因果因子, 淘汰伪因子
- 因子去重: 因果链上的冗余因子合并
- 风险管理: 因果效应突变的因子告警

参考:
- "Double/Debiased Machine Learning for Treatment Effects" (Chernozhukov et al. 2018)
- "Causal Discovery and Inference in Time Series" (JMLR 2023)
- "From Correlation to Causation in Factor Investing" (2024)
"""

import os
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime

from modules.logger import logger

# 尝试导入 scipy
try:
    from scipy import stats
    from scipy.linalg import lstsq
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    logger.warning("[CausalFactor] scipy 未安装, 因果推断功能受限")

# 尝试导入 statsmodels
try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    logger.warning("[CausalFactor] statsmodels 未安装, 使用 numpy 回退")


# ── 数据结构 ─────────────────────────────────────────────────────────

@dataclass
class CausalEffectResult:
    """单个因子的因果效应结果"""
    factor_name: str = ""
    causal_effect: float = 0.0      # 因果效应大小
    p_value: float = 1.0            # 显著性 p-value
    is_causal: bool = False          # 是否显著因果
    direct_effect: bool = False      # 是否为直接效应因子
    ic: float = 0.0                  # 相关 IC (信息系数)
    ir: float = 0.0                  # 信息比率 IC/σ
    is_pseudo: bool = False          # 是否为伪因子 (IC 高但因果不显著)
    timestamp: float = field(default_factory=time.time)


@dataclass
class CausalReport:
    """因果分析报告"""
    stock_code: str = ""
    causal_factors: List[str] = field(default_factory=list)       # 因果因子
    pseudo_factors: List[str] = field(default_factory=list)       # 伪因子
    all_effects: Dict[str, CausalEffectResult] = field(
        default_factory=lambda: {}
    )
    n_factors: int = 0
    n_causal: int = 0
    n_pseudo: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# ── Double ML 估计器 ──────────────────────────────────────────────────────

class DoubleMLEstimator:
    """
    Double/Debiased Machine Learning 估计器

    用于估计单个因子对收益率的因果效应, 同时控制其他混淆因子:

    步骤:
    1. M1: X_j ~ ML(X_{-j}) → 得到残差 π_j = X_j - E[X_j|X_{-j}]
    2. M2: Y ~ ML(X_{-j}) → 得到残差 η = Y - E[Y|X_{-j}]
    3. 回归 η ~ π_j → 系数 = 因果效应 θ_j

    参考: Chernozhukov et al. (2018) "Double/Debiased Machine Learning"
    """

    def __init__(self, use_lasso: bool = True):
        """
        Args:
            use_lasso: 是否使用 Lasso 做变量选择 (减少混淆变量维度)
        """
        self.use_lasso = use_lasso
        self._fitted = False

    def estimate(
        self,
        X: np.ndarray,          # 目标因子 (n,)
        y: np.ndarray,          # 收益率 (n,)
        Z: np.ndarray,          # 混淆变量/控制变量 (n, m)
    ) -> Tuple[float, float]:
        """
        Double ML 估计

        Args:
            X: 目标因子序列
            y: 收益率序列
            Z: 混淆变量矩阵

        Returns:
            (causal_effect, p_value)
        """
        if not HAS_SCIPY:
            logger.warning("[CausalFactor] scipy 不可用, 使用简单相关回退")
            corr = np.corrcoef(X, y)[0, 1]
            return float(corr), 0.5

        n = len(X)
        if n < 30:
            logger.warning(f"[CausalFactor] 样本不足 (n={n}), 跳过 Double ML")
            return 0.0, 1.0

        # 步骤 1: 用 Z 预测 X (得到残差 π)
        if Z.shape[1] > 0:
            # 使用岭回归防止过拟合
            alpha = 0.1 * np.var(Z)
            X_residual = self._residualize(X, Z, ridge_alpha=alpha)
        else:
            X_residual = X - np.mean(X)

        # 步骤 2: 用 Z 预测 y (得到残差 η)
        if Z.shape[1] > 0:
            y_residual = self._residualize(y, Z, ridge_alpha=alpha)
        else:
            y_residual = y - np.mean(y)

        # 步骤 3: 回归 η ~ π (单变量回归)
        if np.std(X_residual) < 1e-10:
            return 0.0, 1.0

        # 简单线性回归
        beta = np.sum(y_residual * X_residual) / (np.sum(X_residual ** 2) + 1e-10)

        # 计算标准误
        residuals = y_residual - beta * X_residual
        se = np.sqrt(np.sum(residuals ** 2) / (n - 2) / (np.sum(X_residual ** 2) + 1e-10))

        if se < 1e-10:
            return float(beta), 0.0

        # t 检验
        t_stat = beta / se
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 2))

        return float(beta), float(min(p_value, 1.0))

    def _residualize(self, target: np.ndarray, controls: np.ndarray,
                     ridge_alpha: float = 0.1) -> np.ndarray:
        """用岭回归得到残差"""
        try:
            # 岭回归系数
            XtX = controls.T @ controls + ridge_alpha * np.eye(controls.shape[1])
            Xty = controls.T @ target
            coeffs = np.linalg.solve(XtX, Xty)
            return target - controls @ coeffs
        except Exception:
            return target - np.mean(target)


# ── PC 算法因果发现 ───────────────────────────────────────────────────────

class PCAlgorithm:
    """
    PC 算法 (Constraint-based Causal Discovery)

    通过条件独立检验构建因果图:
    1. 从完全图开始
    2. 逐步移除不满足条件独立的边
    3. 定向有向边 (v-structures)

    使用偏相关系数做条件独立检验。
    """

    def __init__(self, alpha: float = 0.05):
        """
        Args:
            alpha: 条件独立检验显著性水平
        """
        self.alpha = alpha
        self.causal_graph: Dict[str, Set[str]] = {}  # 因子 → 直接因果因子

    def run(
        self,
        factor_matrix: np.ndarray,
        factor_names: Optional[List[str]] = None,
    ) -> Dict[str, Set[str]]:
        """
        运行 PC 算法

        Args:
            factor_matrix: (n_days, n_factors)
            factor_names: 因子名列表

        Returns:
            causal_graph: {factor_name: set_of_causal_parents}
        """
        n_factors = factor_matrix.shape[1]
        if factor_names is None:
            factor_names = [f"f{i}" for i in range(n_factors)]

        # 初始化: 完全无向图
        edges = set()
        for i in range(n_factors):
            for j in range(i + 1, n_factors):
                edges.add((i, j))

        # 逐步移除边 (按条件集大小递增)
        for cond_size in range(n_factors - 2):
            removed = True
            while removed:
                removed = False
                to_remove = []
                for (i, j) in edges:
                    # 找到 i 和 j 的共同邻居
                    neighbors_i = set()
                    for (a, b) in edges:
                        if a == i:
                            neighbors_i.add(b)
                        if b == i:
                            neighbors_i.add(a)
                    condition_set = neighbors_i - {j}

                    if len(condition_set) > cond_size:
                        continue

                    # 取 cond_size 大小的子集
                    condition_set = list(condition_set)[:cond_size]
                    if len(condition_set) == 0:
                        partial_corr = self._partial_corr(factor_matrix, i, j, [])
                    else:
                        partial_corr = self._partial_corr(
                            factor_matrix, i, j, condition_set
                        )

                    # 条件独立 → 移除边
                    if abs(partial_corr) < self._corr_to_pvalue_threshold(
                        abs(partial_corr), factor_matrix.shape[0]
                    ):
                        to_remove.append((i, j))

                if to_remove:
                    removed = True
                    for edge in to_remove:
                        edges.discard(edge)

        # 构建因果图
        self.causal_graph = {}
        for name in factor_names:
            self.causal_graph[name] = set()

        for (i, j) in edges:
            self.causal_graph[factor_names[i]].add(factor_names[j])
            self.causal_graph[factor_names[j]].add(factor_names[i])

        return dict(self.causal_graph)

    def _partial_corr(
        self,
        X: np.ndarray,
        i: int,
        j: int,
        condition: List[int],
    ) -> float:
        """计算偏相关系数"""
        if not condition:
            return float(np.corrcoef(X[:, i], X[:, j])[0, 1])

        # 回归 X_i ~ X_condition → 残差
        Z = X[:, condition]
        Z = sm.add_constant(Z) if HAS_STATSMODELS else Z
        try:
            coeffs_i = np.linalg.lstsq(Z, X[:, i], rcond=None)[0]
            resid_i = X[:, i] - Z @ coeffs_i
            coeffs_j = np.linalg.lstsq(Z, X[:, j], rcond=None)[0]
            resid_j = X[:, j] - Z @ coeffs_j
            return float(np.corrcoef(resid_i, resid_j)[0, 1])
        except Exception:
            return 0.0

    def _corr_to_pvalue_threshold(self, abs_corr: float, n: int) -> float:
        """将相关系数阈值转换为 p-value 的临界值"""
        # 简化: 使用 Fisher z-transform
        if abs(abs_corr) >= 1.0:
            return 1.0
        z = 0.5 * np.log((1 + abs_corr) / (1 - abs_corr))
        se = 1.0 / np.sqrt(n - 3)
        # z > 1.96*se → p < 0.05
        return min(z / (1.96 * se + 1e-10), 1.0)


# ── 主类: 因果因子选择器 ──────────────────────────────────────────────────

class CausalFactorSelector:
    """
    因果推断因子选择器

    使用 Double ML 估计每个因子对收益率的因果效应:
    1. 对每个因子 X, 用 ML 模型预测 X ~ 其他因子 (得到残差)
    2. 用 ML 模型预测 Y ~ 其他因子 (得到残差)
    3. 残差回归的系数 = 因果效应
    4. 因果效应显著 (p < 0.05) 的因子保留

    同时使用 PC 算法做因果发现:
    - 构建因子因果图
    - 识别直接效应 vs 间接效应因子

    与 factor_ic_monitor.py 联动:
    - IC 高 + 因果效应显著 → 核心因子 (高权重)
    - IC 高 + 因果效应不显著 → 伪因子 (降权/淘汰)
    - IC 低 + 因果效应显著 → 潜在因子 (关注)
    - IC 低 + 因果效应不显著 → 噪声因子 (淘汰)
    """

    def __init__(
        self,
        causal_threshold: float = 0.05,
        ic_threshold: float = 0.03,
        use_pc_algorithm: bool = True,
        use_lasso: bool = True,
    ):
        """
        Args:
            causal_threshold: 因果效应显著性阈值 (p-value)
            ic_threshold: IC 最低阈值
            use_pc_algorithm: 是否运行 PC 算法
            use_lasso: 是否用 Lasso 做变量选择
        """
        self.causal_threshold = causal_threshold
        self.ic_threshold = ic_threshold
        self.use_pc_algorithm = use_pc_algorithm
        self.use_lasso = use_lasso

        self.estimator = DoubleMLEstimator(use_lasso=use_lasso)
        self.pc_algorithm = PCAlgorithm(alpha=causal_threshold)

        # 缓存因果分析报告
        self._last_report: Optional[CausalReport] = None
        self._report_path = os.path.join(
            os.path.dirname(__file__), 'dl_models', 'causal_report.json'
        )

        logger.info(
            f"[CausalFactor] 因果因子选择器初始化: "
            f"threshold={causal_threshold}, pc={use_pc_algorithm}"
        )

    def estimate_causal_effects(
        self,
        factor_matrix: np.ndarray,
        returns: np.ndarray,
        factor_names: Optional[List[str]] = None,
    ) -> Dict[str, CausalEffectResult]:
        """
        估计每个因子的因果效应

        Args:
            factor_matrix: np.ndarray (n_days, n_factors), 标准化后的因子值
            returns: np.ndarray (n_days,), 收益率
            factor_names: 因子名列表 (可选)

        Returns:
            Dict: {factor_name: CausalEffectResult}
        """
        n_factors = factor_matrix.shape[1]
        if factor_names is None:
            factor_names = [f"f{i}" for i in range(n_factors)]

        logger.info(
            f"[CausalFactor] 估计 {n_factors} 个因子的因果效应 "
            f"(n={factor_matrix.shape[0]})"
        )

        results = {}

        # 先计算所有因子的 IC (用于伪因子检测)
        ic_scores = {}
        for i in range(n_factors):
            corr = np.corrcoef(factor_matrix[:, i], returns)[0, 1]
            ic_scores[factor_names[i]] = float(corr)

        # 对每个因子做 Double ML
        for i in range(n_factors):
            X = factor_matrix[:, i]
            # 其他因子作为混淆变量
            Z = np.delete(factor_matrix, i, axis=1)

            # Double ML 估计
            causal_effect, p_value = self.estimator.estimate(X, returns, Z)

            # IC 和 IR
            ic = ic_scores[factor_names[i]]
            ir = ic / (np.std([ic_scores[f] for f in ic_scores]) + 1e-8)

            is_causal = p_value < self.causal_threshold
            is_pseudo = abs(ic) > self.ic_threshold and not is_causal

            results[factor_names[i]] = CausalEffectResult(
                factor_name=factor_names[i],
                causal_effect=causal_effect,
                p_value=p_value,
                is_causal=is_causal,
                ic=ic,
                ir=ir,
                is_pseudo=is_pseudo,
            )

        # PC 算法因果发现 (可选, 计算量大)
        if self.use_pc_algorithm and n_factors <= 50:
            try:
                causal_graph = self.pc_algorithm.run(factor_matrix, factor_names)
                # 标记直接效应因子
                for name, parents in causal_graph.items():
                    if name in results:
                        results[name].direct_effect = len(parents) > 0
            except Exception as e:
                logger.warning(f"[CausalFactor] PC 算法失败: {e}")

        logger.info(
            f"[CausalFactor] 因果效应估计完成: "
            f"{sum(1 for r in results.values() if r.is_causal)}/{n_factors} 显著"
        )

        return results

    def filter_causal_factors(
        self,
        factor_df: pd.DataFrame,
        returns: pd.Series,
        factor_names: Optional[List[str]] = None,
    ) -> Tuple[List[str], CausalReport]:
        """
        过滤出因果因子

        Args:
            factor_df: DataFrame with factor columns
            returns: 收益率序列
            factor_names: 因子名列表 (可选)

        Returns:
            causal_factors: 因果因子名列表
            causal_report: 因果分析报告
        """
        if factor_names is None:
            factor_names = list(factor_df.columns)

        # 处理 NaN
        clean_data = factor_df.dropna()
        clean_returns = returns.loc[clean_data.index]

        if len(clean_data) < 30:
            logger.warning("[CausalFactor] 有效样本不足, 返回所有因子")
            report = CausalReport(
                causal_factors=list(factor_names),
                all_effects={
                    name: CausalEffectResult(
                        factor_name=name, is_causal=True, ic=0.0, ir=0.0
                    )
                    for name in factor_names
                },
                n_factors=len(factor_names),
                n_causal=len(factor_names),
            )
            return list(factor_names), report

        effects = self.estimate_causal_effects(
            clean_data.values, clean_returns.values, factor_names
        )

        causal_factors = [
            name for name, effect in effects.items() if effect.is_causal
        ]
        pseudo_factors = [
            name for name, effect in effects.items() if effect.is_pseudo
        ]

        report = CausalReport(
            causal_factors=causal_factors,
            pseudo_factors=pseudo_factors,
            all_effects=effects,
            n_factors=len(factor_names),
            n_causal=len(causal_factors),
            n_pseudo=len(pseudo_factors),
        )

        self._last_report = report
        self._save_report(report)

        logger.info(
            f"[CausalFactor] 因果因子选择完成: "
            f"{len(causal_factors)}/{len(factor_names)} 因果, "
            f"{len(pseudo_factors)} 伪因子"
        )

        return causal_factors, report

    def detect_pseudo_factors(
        self,
        ic_scores: Dict[str, float],
        causal_effects: Dict[str, CausalEffectResult],
    ) -> List[str]:
        """
        检测伪因子: IC 高但因果效应不显著

        伪因子是量化中的常见陷阱:
        - 历史回测表现好, 但实盘失效
        - 与收益率相关但不是因果关系 (可能是幸存者偏差、数据窥探等)

        Args:
            ic_scores: {factor_name: ic_value}
            causal_effects: {factor_name: CausalEffectResult}

        Returns:
            pseudo_factors: 伪因子名列表
        """
        pseudo = []
        for name, effect in causal_effects.items():
            ic = ic_scores.get(name, 0)
            if abs(ic) > self.ic_threshold and not effect.is_causal:
                pseudo.append(name)
                logger.warning(
                    f"[CausalFactor] 检测到伪因子: {name} "
                    f"(IC={ic:.4f}, p={effect.p_value:.4f})"
                )

        return pseudo

    def get_factor_category(self, factor_name: str) -> str:
        """
        对因子进行分类 (基于 IC 和因果效应)

        Returns:
            'core' (核心因子) / 'pseudo' (伪因子) /
            'potential' (潜在因子) / 'noise' (噪声因子)
        """
        if self._last_report is None or factor_name not in self._last_report.all_effects:
            return 'unknown'

        effect = self._last_report.all_effects[factor_name]
        ic = abs(effect.ic)

        if effect.is_causal and ic > self.ic_threshold:
            return 'core'
        elif effect.is_pseudo:
            return 'pseudo'
        elif effect.is_causal and ic <= self.ic_threshold:
            return 'potential'
        else:
            return 'noise'

    def _save_report(self, report: CausalReport):
        """保存因果分析报告到 JSON"""
        try:
            os.makedirs(os.path.dirname(self._report_path), exist_ok=True)
            serializable = {
                'stock_code': report.stock_code,
                'n_factors': report.n_factors,
                'n_causal': report.n_causal,
                'n_pseudo': report.n_pseudo,
                'causal_factors': report.causal_factors,
                'pseudo_factors': report.pseudo_factors,
                'timestamp': report.timestamp,
                'effects': {
                    name: {
                        'causal_effect': r.causal_effect,
                        'p_value': r.p_value,
                        'is_causal': r.is_causal,
                        'ic': r.ic,
                        'ir': r.ir,
                        'is_pseudo': r.is_pseudo,
                    }
                    for name, r in report.all_effects.items()
                },
            }
            with open(self._report_path, 'w') as f:
                json.dump(serializable, f, indent=2, ensure_ascii=False)
            logger.info(f"[CausalFactor] 因果报告已保存: {self._report_path}")
        except Exception as e:
            logger.error(f"[CausalFactor] 保存报告失败: {e}")

    def load_report(self) -> Optional[CausalReport]:
        """加载上次保存的因果分析报告"""
        if not os.path.exists(self._report_path):
            return None
        try:
            with open(self._report_path) as f:
                data = json.load(f)
            effects = {}
            for name, info in data.get('effects', {}).items():
                effects[name] = CausalEffectResult(
                    factor_name=name,
                    causal_effect=info['causal_effect'],
                    p_value=info['p_value'],
                    is_causal=info['is_causal'],
                    ic=info['ic'],
                    ir=info['ir'],
                    is_pseudo=info['is_pseudo'],
                )
            return CausalReport(
                stock_code=data.get('stock_code', ''),
                causal_factors=data.get('causal_factors', []),
                pseudo_factors=data.get('pseudo_factors', []),
                all_effects=effects,
                n_factors=data.get('n_factors', 0),
                n_causal=data.get('n_causal', 0),
                n_pseudo=data.get('n_pseudo', 0),
                timestamp=data.get('timestamp', ''),
            )
        except Exception as e:
            logger.error(f"[CausalFactor] 加载报告失败: {e}")
            return None
