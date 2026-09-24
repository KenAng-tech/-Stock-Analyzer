#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
可解释 AI 聚合器 — 2026 SOTA 模型解释统一输出

功能:
    - 统一 SHAP/LIME/特征重要性输出格式
    - 多模型解释对比
    - 时序特征重要性追踪
    - 生成可读的解释报告

2026 趋势:
    模型可解释性是量化合规要求
    SHAP + 特征重要性 + 时序分析三合一
"""

import logging
import numpy as np
import json
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger('stock_analyzer.modules')


@dataclass
class ExplanationResult:
    """解释结果"""
    feature_name: str
    importance: float
    shap_value: Optional[float] = None
    lime_weight: Optional[float] = None
    direction: str = 'neutral'  # 'positive', 'negative', 'neutral'
    confidence: float = 0.0


@dataclass
class ModelExplanation:
    """模型解释"""
    model_name: str
    feature_importances: Dict[str, float]
    shap_values: Optional[Dict[str, float]] = None
    lime_weights: Optional[Dict[str, float]] = None
    top_features: List[str] = field(default_factory=list)
    explanation_text: str = ''


class ExplainableAIAggregator:
    """
    可解释 AI 聚合器

    用法:
        aggregator = ExplainableAIAggregator()
        aggregator.add_model_explanation('lgbm', importances, shap_values)
        report = aggregator.generate_report()
    """

    def __init__(self):
        self._explanations: Dict[str, ModelExplanation] = {}
        self._history: Dict[str, List[Dict[str, float]]] = defaultdict(list)

    def add_model_explanation(
        self,
        model_name: str,
        feature_importances: Dict[str, float],
        shap_values: Optional[Dict[str, float]] = None,
        lime_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        添加模型解释

        Args:
            model_name: 模型名称
            feature_importances: {feature: importance}
            shap_values: {feature: shap_value}
            lime_weights: {feature: lime_weight}
        """
        top_features = sorted(
            feature_importances,
            key=feature_importances.get,
            reverse=True,
        )[:10]

        explanation = ModelExplanation(
            model_name=model_name,
            feature_importances=feature_importances,
            shap_values=shap_values,
            lime_weights=lime_weights,
            top_features=top_features,
        )

        self._explanations[model_name] = explanation

        # 记录历史
        for feat, imp in feature_importances.items():
            self._history[feat].append({model_name: imp})

        logger.info(
            f"[XAI] 添加模型 '{model_name}' 解释: "
            f"{len(feature_importances)} 个特征"
        )

    def get_consensus_features(self, top_k: int = 10) -> List[str]:
        """
        获取多模型共识的重要特征

        Args:
            top_k: 返回 Top-K 特征

        Returns:
            共识特征列表
        """
        if not self._explanations:
            return []

        # 汇总所有模型的特征重要性
        all_features: Dict[str, List[float]] = defaultdict(list)

        for exp in self._explanations.values():
            for feat, imp in exp.feature_importances.items():
                all_features[feat].append(imp)

        # 计算平均重要性
        avg_importance = {
            feat: np.mean(imps)
            for feat, imps in all_features.items()
        }

        # 排序
        sorted_features = sorted(
            avg_importance,
            key=avg_importance.get,
            reverse=True,
        )

        return sorted_features[:top_k]

    def compare_models(self) -> Dict[str, Any]:
        """
        多模型解释对比

        Returns:
            对比结果
        """
        comparison: Dict[str, Any] = {
            'models': list(self._explanations.keys()),
            'n_models': len(self._explanations),
            'common_features': [],
            'model_details': {},
        }

        if not self._explanations:
            return comparison

        # 找共同特征
        feature_sets = [
            set(exp.feature_importances.keys())
            for exp in self._explanations.values()
        ]
        common = set.intersection(*feature_sets) if feature_sets else set()
        comparison['common_features'] = sorted(common)

        # 各模型详情
        for name, exp in self._explanations.items():
            comparison['model_details'][name] = {
                'n_features': len(exp.feature_importances),
                'top_features': exp.top_features[:5],
                'has_shap': exp.shap_values is not None,
                'has_lime': exp.lime_weights is not None,
            }

        return comparison

    def generate_report(self) -> Dict[str, Any]:
        """
        生成可读解释报告

        Returns:
            报告字典
        """
        report: Dict[str, Any] = {
            'summary': '',
            'consensus_features': [],
            'model_comparison': {},
            'feature_trends': {},
        }

        if not self._explanations:
            report['summary'] = '无模型解释数据'
            return report

        # 共识特征
        consensus = self.get_consensus_features(10)
        report['consensus_features'] = [
            {
                'name': feat,
                'avg_importance': round(
                    np.mean([
                        exp.feature_importances.get(feat, 0)
                        for exp in self._explanations.values()
                    ]), 4
                ),
            }
            for feat in consensus
        ]

        # 模型对比
        report['model_comparison'] = self.compare_models()

        # 生成摘要
        n_models = len(self._explanations)
        n_features = len(set(
            feat
            for exp in self._explanations.values()
            for feat in exp.feature_importances
        ))
        report['summary'] = (
            f"分析 {n_models} 个模型, {n_features} 个特征. "
            f"共识 Top-3 特征: "
            f"{', '.join(consensus[:3]) if consensus else '无'}"
        )

        return report

    def get_status(self) -> dict:
        """获取状态"""
        return {
            'n_models': len(self._explanations),
            'model_names': list(self._explanations.keys()),
            'n_features_total': len(set(
                feat
                for exp in self._explanations.values()
                for feat in exp.feature_importances
            )),
            'n_features_tracked': len(self._history),
        }


# 模块级单例
_xai_aggregator: Optional[ExplainableAIAggregator] = None


def get_xai_aggregator() -> ExplainableAIAggregator:
    """获取 XAI 聚合器单例"""
    global _xai_aggregator
    if _xai_aggregator is None:
        _xai_aggregator = ExplainableAIAggregator()
        logger.info("[XAI] 单例已创建")
    return _xai_aggregator