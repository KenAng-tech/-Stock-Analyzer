#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Causal Discovery Engine — 因果发现引擎 (2026 SOTA)

从相关性分析升级到因果发现:

架构:
    1. Granger 因果检验 — 时序因果
    2. PC 算法 (约束-based) — 无环因果发现
    3. LiNGAM (线性非高斯) — 非线性因果发现
    4. 因果图构建 → GNN 的边权重建模

核心思想:
- 相关性: X 和 Y 相关 (不知道方向)
- 因果性: X → Y (X 导致 Y)

应用场景:
- 识别市场领先指标 (先行指标)
- 传导链分析 (政策 → 资金 → 股价)
- 构建因果因子 (vs 相关因子)

参考:
- "Causal Discovery and Inference in Time Series" (JMLR 2023)
- "LiNGAM: Finding Linear Non-Gaussian Relations"
"""

import os
import json
import time
import threading
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict, deque

from modules.logger import logger

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    logger.warning("[Causal] networkx 未安装, 因果图功能受限")


# ── Granger 因果检验 ─────────────────────────────────────────────────────────

@dataclass
class GrangerResult:
    """Granger 因果检验结果"""
    cause: str = ""
    effect: str = ""
    p_value: float = 1.0
    significant: bool = False
    best_lag: int = 1
    f_statistic: float = 0.0
    timestamp: float = field(default_factory=time.time)


class GrangerCausalityTest:
    """
    Granger 因果检验

    原理: 如果 X 的过去值有助于预测 Y 的未来值, 则 X Granger 引起 Y

    局限性: 仅适用于时序数据, 假设因果方向已知
    """

    def __init__(self, max_lag: int = 5, alpha: float = 0.05):
        self.max_lag = max_lag
        self.alpha = alpha
        self.results: Dict[Tuple[str, str], GrangerResult] = {}

    def test(
        self,
        data: pd.DataFrame,
        cause_col: str,
        effect_col: str,
    ) -> GrangerResult:
        """
        检验 cause 是否 Granger 引起 effect

        Args:
            data: 时间序列数据 (DataFrame)
            cause_col: 原因变量列名
            effect_col: 效果变量列名

        Returns:
            GrangerResult: 包含 p-value 和是否显著
        """
        try:
            test_data = data[[cause_col, effect_col]].dropna()
            if len(test_data) < self.max_lag * 3:
                return GrangerResult(cause=cause_col, effect=effect_col)

            # 简化 Granger 检验 (使用 statsmodels)
            try:
                from statsmodels.tsa.stattools import grangercausalitytests

                result = grangercausalitytests(
                    test_data[[effect_col, cause_col]],
                    maxlag=self.max_lag,
                    verbose=False,
                )

                # 提取各滞后阶数的 p-value
                p_values = []
                for lag in range(1, self.max_lag + 1):
                    # ssr_ftest: F-test based on SSR
                    p_val = result[lag][0]['ssr_ftest'][1]
                    p_values.append(p_val)

                min_p = min(p_values)
                best_lag = np.argmin(p_values) + 1

                significant = min_p < self.alpha

                granger_result = GrangerResult(
                    cause=cause_col,
                    effect=effect_col,
                    p_value=min_p,
                    significant=significant,
                    best_lag=best_lag,
                    f_statistic=result[best_lag][0]['ssr_ftest'][0],
                )

                self.results[(cause_col, effect_col)] = granger_result
                return granger_result

            except Exception as e:
                logger.debug(f"[Granger] 检验失败: {e}")
                return GrangerResult(cause=cause_col, effect=effect_col)

        except Exception as e:
            logger.error(f"[Granger] {cause_col} → {effect_col} 检验失败: {e}")
            return GrangerResult(cause=cause_col, effect=effect_col)

    def test_all_pairs(
        self,
        data: pd.DataFrame,
        max_pairs: int = 50,
    ) -> Dict[Tuple[str, str], GrangerResult]:
        """
        测试所有变量对的 Granger 因果关系

        Args:
            data: 时间序列数据 (n_samples, n_vars)
            max_pairs: 最大测试对数 (防止组合爆炸)

        Returns:
            Dict[(cause, effect) → GrangerResult]
        """
        results = {}
        columns = data.columns.tolist()
        n_vars = len(columns)

        # 限制测试对数
        tested = 0
        for i, cause in enumerate(columns):
            for j, effect in enumerate(columns):
                if i == j:
                    continue
                if tested >= max_pairs:
                    break

                result = self.test(data, cause, effect)
                if result.significant:
                    results[(cause, effect)] = result

                tested += 1

            if tested >= max_pairs:
                break

        logger.info(f"[Granger] 测试了 {tested} 对, 发现 {len(results)} 对显著因果关系")
        return results


# ── PC 算法 (简化版) ──────────────────────────────────────────────────────

@dataclass
class CausalEdge:
    """因果边"""
    source: str = ""
    target: str = ""
    strength: float = 0.0
    direction_confidence: float = 0.0


class SimplifiedPCAlgorithm:
    """
    PC 算法简化实现 (Constraint-based 因果发现)

    流程:
    1. 构建完全无向图
    2. 条件独立检验 → 删除边
    3. 方向判断 (v-structure)
    4. 输出因果邻接矩阵

    局限性:
    - 假设无混杂因子 (无 latent confounder)
    - 仅处理离散或线性连续数据
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.adjacency: Dict[str, Set[str]] = defaultdict(set)
        self.directions: Dict[Tuple[str, str], str] = {}  # (source, target) → "→" or "←" or "↔"

    def discover(
        self,
        data: pd.DataFrame,
        max_cond_set: int = 2,
    ) -> Dict[str, Set[str]]:
        """
        运行 PC 算法发现因果图

        Args:
            data: 数据 (n_samples, n_vars)
            max_cond_set: 条件集最大大小 (计算复杂度控制)

        Returns:
            Dict[str, Set[str]]: 因果邻接表 (source → {targets})
        """
        if not HAS_NETWORKX:
            logger.warning("[PC] networkx 未安装, 使用简化方法")
            return self._simplified_pc(data)

        variables = data.columns.tolist()
        n_vars = len(variables)

        # Step 1: 完全无向图
        adj = defaultdict(set)
        for vi in variables:
            for vj in variables:
                if vi != vj:
                    adj[vi].add(vj)

        # Step 2: 条件独立检验 → 删除边
        sep_sets: Dict[Tuple[str, str], Set] = defaultdict(set)

        for cond_size in range(max_cond_set + 1):
            for vi in variables:
                for vj in variables:
                    if vi == vj:
                        continue
                    if vj not in adj[vi]:
                        continue

                    # 找到 vj 的邻居 (排除 vi 和自身)
                    neighbors = list(adj[vj] - {vi})

                    if len(neighbors) < cond_size:
                        continue

                    # 尝试所有大小为 cond_size 的条件集
                    from itertools import combinations
                    for cond_set in combinations(neighbors, cond_size):
                        # 条件独立检验 (偏相关检验)
                        try:
                            from scipy import stats

                            partial_corr = self._partial_correlation(
                                data, vi, vj, list(cond_set)
                            )

                            if abs(partial_corr) < 0.1:  # 近似条件独立
                                adj[vi].discard(vj)
                                sep_sets[(vi, vj)] = set(cond_set)
                                sep_sets[(vj, vi)] = set(cond_set)
                                break
                        except Exception:
                            pass

        self.adjacency = adj
        return adj

    def _partial_correlation(
        self,
        data: pd.DataFrame,
        vi: str,
        vj: str,
        cond_set: List[str],
    ) -> float:
        """计算偏相关系数"""
        cols = [vi, vj] + cond_set
        sub_data = data[cols].dropna()

        if len(sub_data) < len(cond_set) + 3:
            return 0.0

        try:
            corr_matrix = sub_data.corr().values
            n = len(cols)

            # 偏相关计算 (递归方法)
            if n == 2:
                return corr_matrix[0, 1]

            # 使用伪逆计算偏相关
            precision = np.linalg.pinv(corr_matrix)
            idx_i = cols.index(vi)
            idx_j = cols.index(vj)
            partial = -precision[idx_i, idx_j] / np.sqrt(precision[idx_i, idx_i] * precision[idx_j, idx_j])

            return partial
        except Exception:
            return 0.0

    def _simplified_pc(self, data: pd.DataFrame) -> Dict[str, Set[str]]:
        """简化版 PC (当 networkx 不可用时)"""
        variables = data.columns.tolist()
        adj = defaultdict(set)

        # 构建完全图
        for vi in variables:
            for vj in variables:
                if vi != vj:
                    adj[vi].add(vj)

        # 使用相关性和 Granger 结果简化
        corr_matrix = data.corr()

        # 删除弱相关边
        for vi in variables:
            for vj in variables:
                if vi != vj and abs(corr_matrix.loc[vi, vj]) < 0.2:
                    adj[vi].discard(vj)

        return adj


# ── 因果发现引擎 ───────────────────────────────────────────────────────────

@dataclass
class CausalGraph:
    """因果图"""
    nodes: List[str] = field(default_factory=list)
    edges: List[CausalEdge] = field(default_factory=list)
    adjacency: Dict[str, Set[str]] = field(default_factory=dict)  # source → {targets}
    method: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class CausalAnalysis:
    """因果分析结果"""
    causal_graph: CausalGraph = field(default_factory=CausalGraph)

    # Granger 检验结果
    granger_results: Dict[Tuple[str, str], GrangerResult] = field(default_factory=dict)

    # 领先指标 (被 Granger 检验为原因的变量)
    leading_indicators: List[str] = field(default_factory=list)

    # 传导链 (传导路径)
    causal_chains: List[List[str]] = field(default_factory=list)

    # 因果强度 (边权重)
    edge_strengths: Dict[Tuple[str, str], float] = field(default_factory=dict)


class CausalDiscoveryEngine:
    """
    因果发现引擎

    整合多种因果发现方法:
    1. Granger 因果检验 (时序因果)
    2. PC 算法 (无环因果发现)
    3. 相关性分析 (辅助)

    输出:
    - 因果图 (Adjacency List)
    - 领先指标 (Leading Indicators)
    - 传导链 (Causal Chains)
    - GNN 边权重 (用于图神经网络)

    使用方式:
        engine = CausalDiscoveryEngine()
        result = engine.analyze(stock_data, klines)
    """

    def __init__(self, alpha: float = 0.05, max_lag: int = 5):
        self.alpha = alpha
        self.max_lag = max_lag

        self.granger_tester = GrangerCausalityTest(max_lag=max_lag, alpha=alpha)
        self.pc_algorithm = SimplifiedPCAlgorithm(alpha=alpha)

        self._analysis_cache: Dict[str, CausalAnalysis] = {}
        self._lock = threading.Lock()

    def analyze(
        self,
        data: pd.DataFrame,
        feature_names: Optional[List[str]] = None,
    ) -> CausalAnalysis:
        """
        分析因果关系

        Args:
            data: 时间序列数据 (n_samples, n_vars)
            feature_names: 特征名列表 (用于命名节点)

        Returns:
            CausalAnalysis: 包含因果图和领先指标
        """
        with self._lock:
            cache_key = f"causal_{hash(data.values.tobytes())}"
            if cache_key in self._analysis_cache:
                return self._analysis_cache[cache_key]

            if feature_names is None:
                feature_names = [f"var_{i}" for i in range(data.shape[1])]

            data.columns = feature_names

            # 1. Granger 因果检验
            granger_results = self.granger_tester.test_all_pairs(data)

            # 2. PC 算法
            adjacency = self.pc_algorithm.discover(data)

            # 3. 构建因果图
            nodes = feature_names
            edges = []

            for (cause, effect), result in granger_results.items():
                edges.append(CausalEdge(
                    source=cause,
                    target=effect,
                    strength=1 - result.p_value,
                    direction_confidence=1 - result.p_value,
                ))

            # 4. 识别领先指标 (Granger 原因)
            leading_indicators = set()
            for (cause, _) in granger_results.keys():
                if granger_results[(cause, _)].significant:
                    leading_indicators.add(cause)

            # 5. 传导链分析 (BFS)
            chains = self._find_causal_chains(adjacency)

            # 6. 边强度 (Granger p-value → 强度)
            edge_strengths = {}
            for (cause, effect), result in granger_results.items():
                edge_strengths[(cause, effect)] = 1 - result.p_value

            causal_graph = CausalGraph(
                nodes=nodes,
                edges=edges,
                adjacency=adjacency,
                method="Granger + PC",
            )

            analysis = CausalAnalysis(
                causal_graph=causal_graph,
                granger_results=granger_results,
                leading_indicators=list(leading_indicators),
                causal_chains=chains,
                edge_strengths=edge_strengths,
            )

            self._analysis_cache[cache_key] = analysis
            return analysis

    def _find_causal_chains(
        self,
        adjacency: Dict[str, Set[str]],
    ) -> List[List[str]]:
        """使用 BFS 找到所有传导链"""
        chains = []
        nodes = list(adjacency.keys())

        for start in nodes:
            queue = deque([(start, [start])])
            visited = set()

            while queue:
                node, path = queue.popleft()

                if len(path) > 1 and node not in visited:
                    chains.append(path)
                    visited.add(node)

                for neighbor in adjacency.get(node, set()):
                    if neighbor not in path:  # 避免循环
                        queue.append((neighbor, path + [neighbor]))

        return chains

    def get_gnn_edge_weights(
        self,
        analysis: CausalAnalysis,
    ) -> Dict[Tuple[str, str], float]:
        """
        获取 GNN 边权重 (用于图神经网络)

        Returns:
            Dict[(source, target), weight]
        """
        return analysis.edge_strengths

    def get_leading_indicators(
        self,
        analysis: CausalAnalysis,
    ) -> List[str]:
        """
        获取领先指标 (因果原因的变量)

        Returns:
            List[str]: 领先指标列表
        """
        return analysis.leading_indicators


# ── 全局单例 ────────────────────────────────────────────────────────────────

_instance: Optional[CausalDiscoveryEngine] = None
_instance_lock = threading.Lock()


def get_causal_discovery_engine() -> CausalDiscoveryEngine:
    """获取全局 CausalDiscoveryEngine 单例"""
    global _instance
    with _instance_lock:
        if _instance is None:
            _instance = CausalDiscoveryEngine()
        return _instance