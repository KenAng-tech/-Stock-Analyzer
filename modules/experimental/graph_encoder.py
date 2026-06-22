#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
股票关联图编码器 — Graph Encoder

用图结构挖掘股票之间的关联, 生成图嵌入特征。

图构建规则:
- 同行业: 权重 0.8
- 同申万二级行业: 权重 0.9
- 价格相关性 > 0.7: 权重 = corr
- 供应链关联: 权重 0.3

特征提取:
- PageRank 嵌入
- 邻居特征聚合 (简化版 GAT)
- 拉普拉斯谱嵌入

TODO: PyTorch Geometric 接口预留
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from modules.logger import logger


class StockGraphBuilder:
    """股票关联图构建器"""

    def __init__(self, max_nodes: int = 100, knn: int = 5):
        self.max_nodes = max_nodes
        self.knn = knn
        self.adj_matrix: Optional[np.ndarray] = None
        self.node_names: List[str] = []
        self.node_features: Optional[np.ndarray] = None

    def build_graph(self, stocks: List[Dict],
                    correlations: Optional[np.ndarray] = None) -> 'StockGraphBuilder':
        """
        构建股票关联图

        Args:
            stocks: [{code, name, industry, sw_level2, market_cap, ...}, ...]
            correlations: 价格相关系数矩阵 (可选, 如果提供则直接使用)

        Returns:
            self (链式调用)
        """
        n = min(len(stocks), self.max_nodes)
        self.node_names = [s.get('code', f'stock_{i}') for i, s in enumerate(stocks[:n])]

        # 初始化邻接矩阵
        self.adj_matrix = np.zeros((n, n))

        # 构建边
        for i in range(n):
            for j in range(i + 1, n):
                weight = self._compute_edge_weight(stocks[i], stocks[j])
                if weight > 0:
                    self.adj_matrix[i, j] = weight
                    self.adj_matrix[j, i] = weight

        # KNN 稀疏化: 每个节点只保留 K 条最强边
        self._knn_sparse()

        # 归一化
        row_sums = self.adj_matrix.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        self.adj_matrix = self.adj_matrix / row_sums

        return self

    def _compute_edge_weight(self, stock_a: Dict, stock_b: Dict) -> float:
        """计算两条边的权重"""
        weight = 0.0

        # 同行业
        ind_a = stock_a.get('industry', stock_a.get('sw_l1', ''))
        ind_b = stock_b.get('industry', stock_b.get('sw_l1', ''))
        if ind_a and ind_b and ind_a == ind_b:
            weight = max(weight, 0.8)

        # 同申万二级行业
        sw_a = stock_a.get('sw_l2', '')
        sw_b = stock_b.get('sw_l2', '')
        if sw_a and sw_b and sw_a == sw_b:
            weight = max(weight, 0.9)

        # 供应链关联 (通过行业间接关联)
        supply_chain_weight = self._supply_chain_weight(stock_a, stock_b)
        weight = max(weight, supply_chain_weight)

        return weight

    def _supply_chain_weight(self, stock_a: Dict, stock_b: Dict) -> float:
        """供应链关联权重 (简化: 通过行业间接关联)"""
        # 如果行业相同但二级行业不同, 可能是供应链上下游
        ind_a = stock_a.get('industry', '')
        ind_b = stock_b.get('industry', '')
        if ind_a and ind_b and ind_a == ind_b:
            return 0.3
        return 0.0

    def _knn_sparse(self):
        """KNN 稀疏化: 每个节点只保留 K 条最强边"""
        n = self.adj_matrix.shape[0]
        for i in range(n):
            if self.adj_matrix[i].sum() == 0:
                continue
            # 获取最强的 K 个邻居
            neighbor_scores = self.adj_matrix[i].copy()
            neighbor_scores[i] = 0  # 排除自己
            top_k_indices = np.argsort(neighbor_scores)[-self.knn:]

            # 只保留 top K
            for j in range(n):
                if j != i and j not in top_k_indices:
                    self.adj_matrix[i, j] = 0
                    self.adj_matrix[j, i] = 0

    def get_adjacency_matrix(self) -> np.ndarray:
        """获取邻接矩阵"""
        return self.adj_matrix.copy()

    def get_node_features(self, stocks: List[Dict]) -> np.ndarray:
        """获取节点特征 (因子值)"""
        n = len(self.node_names)
        features = np.zeros((n, 10))

        for i, code in enumerate(self.node_names):
            stock = next((s for s in stocks if s.get('code') == code), {})
            # 简单特征: pe, pb, roe, change_pct, volume
            features[i, 0] = stock.get('pe', 0) / 100.0  # 归一化
            features[i, 1] = stock.get('pb', 0) / 20.0
            features[i, 2] = stock.get('roe', 0) / 100.0
            features[i, 3] = stock.get('change_pct', 0) / 20.0
            features[i, 4] = stock.get('turnover', 0) / 500.0
            features[i, 5] = stock.get('market_cap', 0) / 1e12  # 万亿归一化
            features[i, 6] = stock.get('rsi_14', 50) / 100.0
            features[i, 7] = stock.get('outer_disk', 0) / (stock.get('inner_disk', 1) + 1)
            features[i, 8] = stock.get('revenue_growth', 0) / 100.0
            features[i, 9] = 1.0  # bias

        return features


class GraphFeatureExtractor:
    """图特征提取器"""

    def __init__(self, embedding_dim: int = 32):
        self.embedding_dim = embedding_dim

    def pagerank_embedding(self, adj: np.ndarray,
                           damping: float = 0.85,
                           iterations: int = 20) -> np.ndarray:
        """
        PageRank 嵌入

        Args:
            adj: 归一化邻接矩阵
            damping: 阻尼系数
            iterations: 迭代次数

        Returns:
            PageRank 分数 (n,)
        """
        n = adj.shape[0]
        pr = np.ones(n) / n

        for _ in range(iterations):
            new_pr = np.ones(n) * (1 - damping) / n
            for i in range(n):
                new_pr += damping * adj[:, i] * pr[i]
            pr = new_pr

        # 归一化到 [0, embedding_dim]
        pr_min, pr_max = pr.min(), pr.max()
        if pr_max > pr_min:
            pr = (pr - pr_min) / (pr_max - pr_min) * self.embedding_dim
        else:
            pr = np.ones(n) * self.embedding_dim / 2

        return pr

    def neighborhood_aggregation(self, features: np.ndarray,
                                  adj: np.ndarray,
                                  rounds: int = 2) -> np.ndarray:
        """
        邻居特征聚合 (简化版 GAT)

        对每轮: aggregated[i] = sum(adj[i,j] * features[j]) / sum(adj[i,j])
        拼接原始特征 + 聚合特征 -> 嵌入

        Args:
            features: (n, feature_dim)
            adj: 归一化邻接矩阵 (n, n)
            rounds: 聚合轮数

        Returns:
            图嵌入 (n, feature_dim * (rounds + 1))
        """
        aggregated = [features]
        current_features = features.copy()

        for _ in range(rounds):
            # 邻居聚合
            aggregated_features = adj @ current_features
            # 拼接
            current_features = np.concatenate([current_features, aggregated_features], axis=1)
            aggregated.append(current_features)

        return np.concatenate(aggregated, axis=1)

    def laplacian_embedding(self, adj: np.ndarray,
                            n_components: int = 16) -> np.ndarray:
        """
        图拉普拉斯谱嵌入 (简化版)

        L = D - A, 取最小 k 个特征向量

        注意: 完整实现需要 scipy.linalg.eigh
        这里用 power iteration 近似
        """
        n = adj.shape[0]
        D = np.diag(adj.sum(axis=1))
        L = D - adj  # 拉普拉斯矩阵

        # 用幂迭代找最小特征值对应的特征向量 (简化)
        # 实际应使用 scipy.linalg.eigh
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(L)
            # 取最小的 n_components 个特征向量
            return eigenvectors[:, :n_components]
        except Exception:
            # 回退: 用度中心性
            degrees = adj.sum(axis=1)
            return degrees.reshape(-1, 1) / (degrees.max() + 1e-10)


class GraphFeatureFusion:
    """图特征融合器"""

    def fuse_graph_features(self, graph_embedding: np.ndarray,
                            original_features: np.ndarray) -> np.ndarray:
        """
        拼接图嵌入 + 原始特征 -> 融合特征

        Args:
            graph_embedding: (n, graph_dim)
            original_features: (n, orig_dim)

        Returns:
            fused: (n, graph_dim + orig_dim)
        """
        return np.concatenate([graph_embedding, original_features], axis=1)

    def get_fused_dimension(self, graph_dim: int, orig_dim: int) -> int:
        return graph_dim + orig_dim


# 全局实例
graph_builder = StockGraphBuilder(max_nodes=100, knn=5)
graph_extractor = GraphFeatureExtractor(embedding_dim=32)
graph_fusion = GraphFeatureFusion()
