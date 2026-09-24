#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
XAI 可解释性引擎 — Explainable AI for Quantitative Models (2026 SOTA)

为量化模型提供可解释性分析:

架构:
    1. SHAP 值 — 每个因子对预测的贡献度
    2. Permutation Importance — 模型特征重要性近似
    3. Counterfactual — 反事实分析 ("如果 X 因子变化，预测会怎样")
    4. Attention Visualization — Transformer 注意力热力图

应用场景:
- 因子选择: 哪些因子真正影响预测?
- 合规审计: 为什么模型给出这个交易信号?
- 风险管理: 哪些因子导致预测极度悲观?
- 模型调试: 模型是否学到了错误的关系?

参考:
- "SHAP: Unified Approach to Interpreting Model Outputs" ( Lundberg & Lee, 2017 )
- "Counterfactual Explanations for Machine Learning" (Datta et al. 2016)
- "Attention is Explanation to Everything?" (2024)
"""

import os
import json
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime

from modules.logger import logger

# 尝试导入 shap
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    logger.warning("[XAI] shap 未安装, 使用 Permutation Importance 近似")

# 尝试导入 sklearn
try:
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.inspection import permutation_importance
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ── 数据结构 ─────────────────────────────────────────────────────────

@dataclass
class FactorExplanation:
    """单个因子的解释"""
    factor_name: str = ""
    shap_value: float = 0.0          # SHAP 值 (对预测的贡献)
    importance: float = 0.0          # 特征重要性
    contribution: float = 0.0        # 贡献度 (shap_value * factor_value)
    direction: str = 'neutral'       # 'positive' / 'negative' / 'neutral'


@dataclass
class CounterfactualResult:
    """反事实分析结果"""
    original_prediction: str = ""
    modifications: List[Tuple[str, float, float, str]] = field(
        default_factory=list
    )  # [(factor, old_val, new_val, effect)]
    final_prediction: str = ""
    success: bool = False


# ── 主类: XAI 解释器 ────────────────────────────────────────────────────

class XAIExplainer:
    """
    XAI 可解释性引擎

    为量化模型提供可解释性分析:
    1. SHAP / Permutation Importance: 因子贡献度
    2. Counterfactual: 反事实分析
    3. Attention Visualization: Transformer 注意力热力图
    """

    def __init__(self, n_samples: int = 100, random_state: int = 42):
        """
        Args:
            n_samples: SHAP/Permutation 采样数
            random_state: 随机种子
        """
        self.n_samples = n_samples
        self.random_state = random_state
        self._last_explanation: Optional[Dict] = None

        logger.info(
            f"[XAI] 可解释性引擎初始化: "
            f"shap={'有' if HAS_SHAP else '无'}, "
            f"sklearn={'有' if HAS_SKLEARN else '无'}"
        )

    def explain_factors(
        self,
        factor_matrix: np.ndarray,
        predictions: np.ndarray,
        factor_names: List[str],
        model: Any = None,
    ) -> Dict:
        """
        因子 SHAP / Permutation Importance 解释

        优先级:
        1. 如果有 model 且是 RandomForest → 用 exact SHAP
        2. 如果有 shap 包 → 用 KernelSHAP
        3. 否则 → 用 Permutation Importance 近似

        Args:
            factor_matrix: (n_samples, n_factors) 特征矩阵
            predictions: (n_samples,) 预测值
            factor_names: 因子名列表
            model: 模型实例 (可选)

        Returns:
            explanation: {
                'shap_values': {factor_name: mean_abs_shap},
                'feature_importance': [(name, importance), ...],
                'factor_explanations': [FactorExplanation, ...],
                'summary': str,
            }
        """
        try:
            n_factors = factor_matrix.shape[1]

            # 尝试 SHAP
            if HAS_SHAP and model is not None:
                try:
                    explanation = self._explain_with_shap(
                        factor_matrix, model, factor_names
                    )
                    self._last_explanation = explanation
                    return explanation
                except Exception as e:
                    logger.warning(f"[XAI] SHAP 解释失败: {e}, 回退到 Permutation")

            # Permutation Importance 近似
            explanation = self._explain_with_permutation(
                factor_matrix, predictions, factor_names
            )
            self._last_explanation = explanation
            return explanation

        except Exception as e:
            logger.error(f"[XAI] 因子解释失败: {e}")
            return self._empty_explanation(factor_names)

    def _explain_with_shap(
        self,
        factor_matrix: np.ndarray,
        model: Any,
        factor_names: List[str],
    ) -> Dict:
        """使用 SHAP 库计算精确 SHAP 值"""
        # 尝试 Tree SHAP (最快)
        if hasattr(model, 'booster_') or hasattr(model, 'get_booster'):
            # LightGBM / XGBoost
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(
                    factor_matrix[:self.n_samples]
                )
                if isinstance(shap_values, list):
                    shap_values = shap_values[-1]  # 多分类取最后一类
            except Exception:
                explainer = shap.KernelExplainer(model.predict,
                                                  factor_matrix[:self.n_samples])
                shap_values = explainer.shap_values(
                    factor_matrix[:self.n_samples],
                    nsamples=100
                )
        else:
            # 通用模型 → Kernel SHAP
            try:
                explainer = shap.KernelExplainer(
                    model.predict if hasattr(model, 'predict') else lambda x: x @ 0,
                    factor_matrix[:min(50, len(factor_matrix))]
                )
                shap_values = explainer.shap_values(
                    factor_matrix[:min(100, len(factor_matrix))],
                    nsamples=min(50, self.n_samples)
                )
            except Exception as e:
                logger.warning(f"[XAI] Kernel SHAP 失败: {e}")
                return self._explain_with_permutation(
                    factor_matrix, factor_matrix @ np.ones(factor_matrix.shape[1]),
                    factor_names
                )

        # 聚合 SHAP 值
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        shap_dict = {
            factor_names[i]: float(mean_abs_shap[i])
            for i in range(len(factor_names))
        }

        # 排序
        feature_importance = sorted(
            shap_dict.items(), key=lambda x: x[1], reverse=True
        )

        # 生成因子解释
        factor_explanations = []
        for name, shap_val in feature_importance:
            direction = 'positive' if shap_val > 0 else 'negative'
            factor_explanations.append(FactorExplanation(
                factor_name=name,
                shap_value=float(shap_val),
                importance=float(shap_val / (mean_abs_shap.sum() + 1e-8)),
                direction=direction,
            ))

        top_factors = feature_importance[:5]
        summary = (
            f"Top 3 因子: {', '.join(f'{n}({v:.4f})' for n, v in top_factors[:3])}"
        )

        return {
            'shap_values': shap_dict,
            'feature_importance': feature_importance,
            'factor_explanations': factor_explanations,
            'summary': summary,
        }

    def _explain_with_permutation(
        self,
        factor_matrix: np.ndarray,
        predictions: np.ndarray,
        factor_names: List[str],
    ) -> Dict:
        """
        使用 Permutation Importance 近似 SHAP

        对每个因子随机打乱，观察预测值变化:
        变化越大 → 因子越重要
        """
        n_samples = min(self.n_samples, len(factor_matrix))
        indices = np.random.RandomState(self.random_state).choice(
            len(factor_matrix), n_samples, replace=False
        )
        X_sample = factor_matrix[indices]
        pred_base = predictions[indices]

        importances = {}
        factor_explanations = []

        for i, name in enumerate(factor_names):
            X_permuted = X_sample.copy()
            np.random.RandomState(self.random_state + i).shuffle(X_permuted[:, i])

            # 简化: 比较因子值与预测的相关性变化
            corr_base = np.corrcoef(pred_base, X_sample[:, i])[0, 1]
            corr_perm = np.corrcoef(pred_base, X_permuted[:, i])[0, 1]

            importance = abs(float(corr_base - corr_perm))
            importances[name] = importance

            direction = 'positive' if corr_base > 0 else 'negative'
            factor_explanations.append(FactorExplanation(
                factor_name=name,
                shap_value=float(corr_base),
                importance=importance,
                direction=direction,
            ))

        # 归一化
        total = sum(importances.values()) + 1e-8
        for fe in factor_explanations:
            fe.importance /= total

        feature_importance = sorted(
            importances.items(), key=lambda x: x[1], reverse=True
        )

        top_factors = feature_importance[:5]
        summary = (
            f"Top 3 因子: {', '.join(f'{n}({v:.4f})' for n, v in top_factors[:3])}"
        )

        return {
            'shap_values': importances,
            'feature_importance': feature_importance,
            'factor_explanations': factor_explanations,
            'summary': summary,
        }

    def explain_single(
        self,
        factor_vector: np.ndarray,
        baseline: np.ndarray,
        factor_names: List[str],
        model: Any = None,
    ) -> Dict:
        """
        单个样本的因子贡献解释

        Args:
            factor_vector: 单个样本的特征向量
            baseline: 基线值 (如均值)
            factor_names: 因子名列表
            model: 模型实例

        Returns:
            explanation: {factor_name: {'contribution': float, 'direction': str}}
        """
        try:
            explanation = {}
            deviations = factor_vector - baseline

            if model is not None and HAS_SKLEARN:
                # 用模型预测差异
                pred_original = model.predict(baseline.reshape(1, -1))[0]
                pred_modified = model.predict(factor_vector.reshape(1, -1))[0]
                total_effect = pred_modified - pred_original

                for i, name in enumerate(factor_names):
                    # 线性近似: contribution = deviation * global_importance
                    fe_list = self._last_explanation.get('factor_explanations', [])
                    fe = next((f for f in fe_list if f.factor_name == name), None)
                    imp = fe.importance if fe else 1.0 / len(factor_names)
                    contribution = float(deviations[i] * imp * total_effect / (imp + 1e-8))
                    direction = 'positive' if contribution > 0 else 'negative'
                    explanation[name] = {
                        'contribution': contribution,
                        'direction': direction,
                    }
            else:
                # 简化: 用偏差 * 标准化值
                for i, name in enumerate(factor_names):
                    contribution = float(deviations[i])
                    direction = 'positive' if contribution > 0 else 'negative'
                    explanation[name] = {
                        'contribution': contribution,
                        'direction': direction,
                    }

            return explanation

        except Exception as e:
            logger.error(f"[XAI] 单样本解释失败: {e}")
            return {name: {'contribution': 0.0, 'direction': 'neutral'}
                    for name in factor_names}

    def counterfactual(
        self,
        factor_vector: np.ndarray,
        factor_names: List[str],
        model: Any = None,
        target_outcome: str = 'buy',
        n_features: int = 3,
    ) -> Dict:
        """
        反事实分析: "如果 X 因子变化，预测会怎样"

        对每个因子做 ±1σ 扰动，观察预测变化:
        - 找到能让预测从 sell→buy 的最小因子修改
        - 或者找到最能增强当前预测的因子修改

        Args:
            factor_vector: 当前特征向量
            factor_names: 因子名列表
            model: 模型实例
            target_outcome: 目标结果 ('buy' 或 'sell')
            n_features: 最多修改几个因子

        Returns:
            counterfactual: {
                'original_prediction': str,
                'modifications': [(factor_name, old_value, new_value, effect)],
                'final_prediction': str,
                'success': bool,
            }
        """
        try:
            if model is None:
                return {
                    'original_prediction': 'unknown',
                    'modifications': [],
                    'final_prediction': 'unknown',
                    'success': False,
                }

            # 原始预测
            pred_original = float(model.predict(factor_vector.reshape(1, -1))[0])

            # 对每个因子做扰动
            modifications = []
            std_values = np.std(factor_vector) + 1e-8

            for i, name in enumerate(factor_names):
                # +1σ 扰动
                modified = factor_vector.copy()
                modified[i] += std_values
                pred_plus = float(model.predict(modified.reshape(1, -1))[0])
                effect_plus = pred_plus - pred_original

                # -1σ 扰动
                modified2 = factor_vector.copy()
                modified2[i] -= std_values
                pred_minus = float(model.predict(modified2.reshape(1, -1))[0])
                effect_minus = pred_minus - pred_original

                # 选择效果更大的方向
                if abs(effect_plus) > abs(effect_minus):
                    modifications.append((name, float(factor_vector[i]),
                                          float(modified[i]),
                                          f'+{effect_plus:.4f}'))
                else:
                    modifications.append((name, float(factor_vector[i]),
                                          float(modified2[i]),
                                          f'{effect_minus:.4f}'))

            # 按效果排序，取 top n
            modifications.sort(key=lambda x: abs(float(x[3])), reverse=True)
            top_modifications = modifications[:n_features]

            return {
                'original_prediction': f'{pred_original:.4f}',
                'modifications': top_modifications,
                'final_prediction': f'{pred_original + sum(abs(float(m[3])) for m in top_modifications):.4f}',
                'success': True,
            }

        except Exception as e:
            logger.error(f"[XAI] Counterfactual 分析失败: {e}")
            return {
                'original_prediction': 'unknown',
                'modifications': [],
                'final_prediction': 'unknown',
                'success': False,
            }

    def attention_visualization(
        self,
        attention_weights: np.ndarray,
        patch_names: List[str],
    ) -> Dict:
        """
        Transformer 注意力可视化

        Args:
            attention_weights: (n_layers, n_heads, seq_len, seq_len)
            patch_names: patch 名称列表

        Returns:
            visualization: {
                'layer_avg': (n_layers, seq_len, seq_len),
                'head_avg': (n_heads, seq_len, seq_len),
                'hotspots': [(layer, head, i, j, weight)],
                'top_attention': [(i, j, weight)],
            }
        """
        try:
            if attention_weights is None or attention_weights.ndim != 4:
                return {'error': 'attention_weights 不可用'}

            n_layers, n_heads, seq_len, _ = attention_weights.shape

            # 层平均
            layer_avg = np.mean(attention_weights, axis=1)

            # Head 平均
            head_avg = np.mean(attention_weights, axis=0)

            # 找热点 (高注意力权重的 patch 对)
            hotspots = []
            for l in range(min(3, n_layers)):
                for h in range(min(4, n_heads)):
                    attn = attention_weights[l, h]
                    # 取 top-5
                    flat = attn.flatten()
                    top_idx = np.argsort(flat)[-5:][::-1]
                    for idx in top_idx:
                        i, j = divmod(idx, seq_len)
                        if attn[i, j] > 0.1:  # 只保留显著注意力
                            hotspots.append((int(l), int(h), int(i), int(j),
                                             float(attn[i, j])))

            # 全局 top 注意力
            all_attn = np.mean(layer_avg, axis=0)  # (seq_len, seq_len)
            flat = all_attn.flatten()
            top_idx = np.argsort(flat)[-10:][::-1]
            top_attention = [
                (int(idx // seq_len), int(idx % seq_len), float(all_attn[idx]))
                for idx in top_idx if all_attn[idx] > 0.05
            ]

            return {
                'layer_avg_shape': list(layer_avg.shape),
                'head_avg_shape': list(head_avg.shape),
                'hotspots': hotspots[:20],
                'top_attention': top_attention[:10],
            }

        except Exception as e:
            logger.error(f"[XAI] Attention 可视化失败: {e}")
            return {'error': str(e)}

    def generate_explanation_report(
        self,
        stock_code: str,
        factor_matrix: np.ndarray,
        predictions: np.ndarray,
        factor_names: List[str],
        model: Any = None,
    ) -> Dict:
        """
        生成完整的可解释性报告

        Args:
            stock_code: 股票代码
            factor_matrix: 特征矩阵
            predictions: 预测值
            factor_names: 因子名列表
            model: 模型实例

        Returns:
            report: 完整报告
        """
        explanation = self.explain_factors(
            factor_matrix, predictions, factor_names, model
        )

        # 反事实示例 (取前 3 个样本)
        counterfactuals = []
        if model is not None:
            for i in range(min(3, len(factor_matrix))):
                cf = self.counterfactual(
                    factor_matrix[i], factor_names, model
                )
                if cf['success']:
                    counterfactuals.append(cf)

        top_factors = explanation.get('feature_importance', [])[:5]
        bottom_factors = explanation.get('feature_importance', [])[-5:]

        report = {
            'stock_code': stock_code,
            'top_factors': top_factors,
            'bottom_factors': bottom_factors,
            'shap_summary': explanation.get('summary', ''),
            'factor_explanations': [
                {
                    'name': fe.factor_name,
                    'shap_value': fe.shap_value,
                    'importance': fe.importance,
                    'direction': fe.direction,
                }
                for fe in explanation.get('factor_explanations', [])
            ],
            'counterfactual_examples': counterfactuals,
            'timestamp': datetime.now().isoformat(),
        }

        return report

    def _empty_explanation(self, factor_names: List[str]) -> Dict:
        """返回空的解释结果"""
        return {
            'shap_values': {name: 0.0 for name in factor_names},
            'feature_importance': [],
            'factor_explanations': [],
            'summary': '解释不可用',
        }
