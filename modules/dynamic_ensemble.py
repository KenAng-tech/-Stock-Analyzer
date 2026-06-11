#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
动态集成策略 — Dynamic Ensemble

根据市场状态 (牛市/熊市/震荡/高波动) 动态调整模型权重。

市场状态检测:
- MA 排列: 多头=bull, 空头=bear
- 波动率: >3%/日 = volatile
- 其他 = sideways

模型权重配置:
- bull: LGBM 0.35 / XGB 0.30 / RF 0.20 / temporal 0.15
- bear: LGBM 0.30 / XGB 0.35 / RF 0.20 / temporal 0.15
- sideways: LGBM 0.25 / XGB 0.25 / RF 0.30 / temporal 0.20
- volatile: LGBM 0.30 / XGB 0.30 / RF 0.25 / temporal 0.15
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from modules.logger import logger


class MarketRegimeDetector:
    """市场状态检测器"""

    def detect_regime(self, stock_data: Dict,
                      klines: Optional[List[Dict]] = None) -> str:
        """
        检测单只股票的市场状态

        Returns:
            'bull' / 'bear' / 'sideways' / 'volatile'
        """
        if not klines or len(klines) < 60:
            return 'sideways'

        closes = np.array([float(k.get('close', 0)) for k in klines if float(k.get('close', 0)) > 0])
        if len(closes) < 60:
            return 'sideways'

        # 计算移动平均线
        ma5 = np.mean(closes[-5:])
        ma20 = np.mean(closes[-20:])
        ma60 = np.mean(closes[-60:])

        # 计算 20 日波动率
        returns = np.diff(np.log(closes[-20:]))
        daily_vol = np.std(returns)

        # 判断波动状态
        if daily_vol > 0.03:  # 日波动 > 3%
            return 'volatile'

        # 判断趋势
        if ma5 > ma20 > ma60:
            # 多头排列 + 确认趋势
            if closes[-1] > ma5:
                return 'bull'
            return 'sideways'
        elif ma5 < ma20 < ma60:
            if closes[-1] < ma5:
                return 'bear'
            return 'sideways'
        else:
            # 均线交织 → 震荡
            return 'sideways'

    def detect_weighted(self, codes: List[str],
                        fetcher=None) -> Dict[str, str]:
        """
        多股票加权市场状态检测

        Returns:
            {stock_code: regime}
        """
        results = {}
        for code in codes[:10]:  # 最多检测 10 只
            try:
                if fetcher:
                    data = fetcher.get_stock_info(code)
                    results[code] = self.detect_regime(data)
            except Exception as e:
                logger.debug(f"[MarketRegimeDetector] {code} 检测失败: {e}")
                results[code] = 'sideways'
        return results


class ModelWeightScheduler:
    """模型权重调度器"""

    # 各市场状态下的模型权重
    STATE_WEIGHTS = {
        'bull': {'lgb': 0.35, 'xgb': 0.30, 'rf': 0.20, 'temporal': 0.15},
        'bear': {'lgb': 0.30, 'xgb': 0.35, 'rf': 0.20, 'temporal': 0.15},
        'sideways': {'lgb': 0.25, 'xgb': 0.25, 'rf': 0.30, 'temporal': 0.20},
        'volatile': {'lgb': 0.30, 'xgb': 0.30, 'rf': 0.25, 'temporal': 0.15},
    }

    def __init__(self):
        self._current_weights = None
        self._recent_ic: Dict[str, List[float]] = {}

    def get_weights(self, regime: str) -> Dict[str, float]:
        """获取指定状态下的模型权重"""
        return self.STATE_WEIGHTS.get(regime, self.STATE_WEIGHTS['sideways']).copy()

    def smooth_transition(self, prev_weights: Dict[str, float],
                          new_weights: Dict[str, float],
                          alpha: float = 0.3) -> Dict[str, float]:
        """
        平滑权重切换: new = alpha * new_weights + (1-alpha) * prev_weights
        """
        if not prev_weights:
            return new_weights

        smoothed = {}
        all_keys = set(prev_weights.keys()) | set(new_weights.keys())
        for key in all_keys:
            smoothed[key] = alpha * new_weights.get(key, 0) + (1 - alpha) * prev_weights.get(key, 0)

        # 归一化
        total = sum(smoothed.values())
        if total > 0:
            smoothed = {k: v / total for k, v in smoothed.items()}

        return smoothed

    def adaptive_adjustment(self, recent_ic: Dict[str, float]) -> Dict[str, float]:
        """
        根据近期 IC 动态调整权重

        IC 高的模型获得更多权重
        """
        if not recent_ic:
            return {}

        # 绝对 IC 加权
        abs_ics = {k: abs(v) for k, v in recent_ic.items()}
        total = sum(abs_ics.values())
        if total > 0:
            return {k: v / total for k, v in abs_ics.items()}
        return recent_ic.copy()

    def record_ic(self, model_name: str, ic: float):
        """记录模型 IC"""
        if model_name not in self._recent_ic:
            self._recent_ic[model_name] = []
        self._recent_ic[model_name].append(ic)
        # 只保留最近 30 条
        if len(self._recent_ic[model_name]) > 30:
            self._recent_ic[model_name] = self._recent_ic[model_name][-30:]

    def get_avg_ic(self, model_name: str) -> float:
        """获取模型平均 IC"""
        if model_name in self._recent_ic and self._recent_ic[model_name]:
            return float(np.mean(self._recent_ic[model_name]))
        return 0.0


class DynamicEnsemblePredictor:
    """动态集成预测器"""

    def __init__(self, predictor=None, regime_detector=None,
                 weight_scheduler=None):
        self.predictor = predictor
        self.regime_detector = regime_detector or MarketRegimeDetector()
        self.weight_scheduler = weight_scheduler or ModelWeightScheduler()

    def predict(self, features: np.ndarray,
                stock_data: Dict,
                klines: Optional[List[Dict]] = None) -> Dict:
        """
        动态集成预测

        流程:
        1. 检测市场状态
        2. 获取该状态的模型权重
        3. 加权融合各模型概率
        4. 返回最终预测
        """
        # Step 1: 检测市场状态
        regime = self.regime_detector.detect_regime(stock_data, klines)

        # Step 2: 获取权重
        weights = self.weight_scheduler.get_weights(regime)

        # Step 3: 获取各模型预测
        model_probs = self._get_model_predictions(features)

        if not model_probs:
            return {
                'direction': 'neutral',
                'confidence': 0.5,
                'probabilities': {'up': 0.33, 'down': 0.33, 'neutral': 0.34},
                'regime': regime,
                'weights': weights,
            }

        # Step 4: 加权融合
        fused_probs = {'up': 0, 'down': 0, 'neutral': 0}
        for model_name, probs in model_probs.items():
            w = weights.get(model_name, 0)
            fused_probs['up'] += w * probs.get('up', 0.33)
            fused_probs['down'] += w * probs.get('down', 0.33)
            fused_probs['neutral'] += w * probs.get('neutral', 0.34)

        # 归一化
        total = sum(fused_probs.values())
        if total > 0:
            fused_probs = {k: v / total for k, v in fused_probs.items()}

        # 决策
        direction = 'up' if fused_probs['up'] > fused_probs['down'] and fused_probs['up'] > 0.4 else \
                    'down' if fused_probs['down'] > fused_probs['up'] and fused_probs['down'] > 0.4 else 'neutral'

        return {
            'direction': direction,
            'confidence': round(max(fused_probs.values()), 4),
            'probabilities': fused_probs,
            'regime': regime,
            'weights': weights,
            'model_details': {n: dict(v) for n, v in model_probs.items()},
        }

    def _get_model_predictions(self, features: np.ndarray) -> Dict[str, Dict]:
        """获取各模型预测概率"""
        if not self.predictor:
            return {}

        try:
            result = self.predictor.predict_direction(features)
            return {
                'ensemble': {
                    'up': result.get('probabilities', {}).get('up', 0.33),
                    'down': result.get('probabilities', {}).get('down', 0.33),
                    'neutral': result.get('probabilities', {}).get('neutral', 0.34),
                }
            }
        except Exception as e:
            logger.error(f"[DynamicEnsemble] 预测失败: {e}")
            return {}

    def update_weights_from_performance(
        self, recent_predictions: List[Dict], actual_outcomes: List[int]
    ):
        """根据近期预测准确率调整权重"""
        if len(recent_predictions) < 5:
            return

        model_scores = {}
        for pred, actual in zip(recent_predictions, actual_outcomes):
            model_details = pred.get('model_details', {})
            for model_name, probs in model_details.items():
                if model_name not in model_scores:
                    model_scores[model_name] = {'correct': 0, 'total': 0}
                model_scores[model_name]['total'] += 1
                # 简单判断: 预测概率最高的方向是否正确
                max_dir = max(probs, key=probs.get)
                direction = pred.get('direction', '')
                if max_dir == 'up' and actual == 1:
                    model_scores[model_name]['correct'] += 1
                elif max_dir == 'down' and actual == -1:
                    model_scores[model_name]['correct'] += 1

        # 更新权重调度器
        for model_name, scores in model_scores.items():
            if scores['total'] > 0:
                accuracy = scores['correct'] / scores['total']
                self.weight_scheduler.record_ic(model_name, accuracy * 2 - 1)  # 转换为 IC 范围


class ModelPerformanceTracker:
    """模型性能追踪器"""

    def __init__(self):
        self._records: Dict[str, List[Dict]] = {}

    def record_prediction(self, model_name: str, pred: str, actual: str, confidence: float = 0.5):
        """记录单次预测"""
        if model_name not in self._records:
            self._records[model_name] = []
        self._records[model_name].append({
            'pred': pred, 'actual': actual, 'confidence': confidence,
            'correct': pred == actual
        })

    def get_performance(self, model_name: str) -> Dict:
        """获取模型性能指标"""
        if model_name not in self._records or not self._records[model_name]:
            return {'accuracy': 0, 'total': 0, 'win_rate': 0}

        records = self._records[model_name]
        total = len(records)
        correct = sum(1 for r in records if r['correct'])
        avg_confidence = np.mean([r['confidence'] for r in records])

        return {
            'accuracy': round(correct / total, 4) if total > 0 else 0,
            'total': total,
            'win_rate': round(correct / total, 4) if total > 0 else 0,
            'avg_confidence': round(float(avg_confidence), 4),
        }

    def get_ranking(self) -> List[Tuple[str, float]]:
        """获取模型性能排名 (按准确率)"""
        rankings = []
        for name, perf in self._records.items():
            if perf:
                acc = sum(1 for r in perf if r['correct']) / len(perf)
                rankings.append((name, acc))
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings


# 全局实例
dynamic_ensemble = DynamicEnsemblePredictor()
model_tracker = ModelPerformanceTracker()
