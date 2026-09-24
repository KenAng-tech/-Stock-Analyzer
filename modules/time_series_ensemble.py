"""
时序基础模型投票集成

集成 Moirai (ICLR 2024) + TimesFM (Google) + PatchTST (ICML 2024)
三个时序基础模型的预测，通过加权投票生成最终决策。

架构:
    Input -> [Moirai]  ---
                          |---> Weighted Vote ---> Final Decision
    Input -> [TimesFM]  ---
    Input -> [PatchTST] ---

权重策略:
    - 初始: 等权重 (各 1/3)
    - 动态: 基于最近 N 日 IC (Information Coefficient) 加权
    - 回退: 单个模型失败时自动降级

用法:
    >>> ensemble = TimeSeriesEnsemble()
    >>> result = ensemble.predict('sz300620', history=prices, lookforward=20)
    >>> print(result['direction'], result['confidence'], result['consensus'])
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

import numpy as np

from modules.logger import logger


# ---------------------------------------------------------------------------
# 条件导入: 三个时序基础模型 (torch 可能不可用)
# ---------------------------------------------------------------------------

_moirai_available = False
_timesfm_available = False
_patchtst_available = False

try:
    from modules.models.moirai_predictor import MoiraiPredictor
    _moirai_available = True
except (ImportError, ModuleNotFoundError) as exc:
    logger.debug(f"MoiraiPredictor 不可用: {exc}")

try:
    # TimesFM 使用独立的包名
    import timesfm  # type: ignore  # noqa: F401
    _timesfm_available = True
except (ImportError, ModuleNotFoundError) as exc:
    logger.debug(f"TimesFM 不可用: {exc}")

try:
    from modules.patchtst_predictor import PatchTSTPredictor
    _patchtst_available = True
except (ImportError, ModuleNotFoundError) as exc:
    logger.debug(f"PatchTSTPredictor 不可用: {exc}")

# 可用的模型清单 (常量)
AVAILABLE_MODELS = {
    'moirai': _moirai_available,
    'timesfm': _timesfm_available,
    'patchtst': _patchtst_available,
}

# 默认等权重
_DEFAULT_WEIGHTS = {'moirai': 1.0 / 3, 'timesfm': 1.0 / 3, 'patchtst': 1.0 / 3}

# 方向映射
_DIR_MAP = {'buy': 1.0, 'sell': -1.0, 'neutral': 0.0}
_INV_DIR_MAP = {v: k for k, v in _DIR_MAP.items()}


def _safe_direction(p: Dict) -> float:
    """从模型预测结果中提取方向分数 [-1, 1]，失败时返回 0。"""
    try:
        return _DIR_MAP.get(p.get('direction', 'neutral'), 0.0)
    except (AttributeError, TypeError):
        return 0.0


def _safe_direction_label(score: float) -> str:
    """方向分数 -> 方向标签。"""
    if score > 0.15:
        return 'buy'
    elif score < -0.15:
        return 'sell'
    return 'neutral'


# ---------------------------------------------------------------------------
# 主类
# ---------------------------------------------------------------------------

class TimeSeriesEnsemble:
    """
    时序基础模型投票集成

    集成 Moirai (ICLR 2024) + TimesFM (Google) + PatchTST (ICML 2024)
    三个基础模型的预测，通过加权投票生成最终决策。

    权重策略:
        - 初始: 等权重 (各 1/3)
        - 动态: 基于最近 N 日 IC 加权 (IC-IR 归一化)
        - 回退: 单个模型失败时自动降级

    示例:
        >>> ensemble = TimeSeriesEnsemble(weights=None, lookback=30)
        >>> result = ensemble.predict('sz300620', history=prices, lookforward=20)
        >>> print(result['direction'])      # 'buy' / 'sell' / 'neutral'
        >>> print(result['confidence'])     # 0.0 - 1.0
        >>> print(result['consensus'])      # 模型间一致性 0.0 - 1.0
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        lookback: int = 30,
    ):
        """
        初始化集成。

        Args:
            weights: 模型权重 {'moirai': w1, 'timesfm': w2, 'patchtst': w3}。
                     若为 None 则使用等权重。权重会自动归一化。
            lookback: 回溯天数，用于动态权重计算 (IC 计算窗口)。
        """
        self.lookback = lookback
        self._models: Dict[str, Any] = {}   # name -> model_instance
        self.models: Dict[str, Any] = {}    # 公开接口 (别名)
        self._weights: Dict[str, float] = dict(_DEFAULT_WEIGHTS)

        if weights is not None:
            for name, w in weights.items():
                if name in self._weights:
                    self._weights[name] = float(w)

        # 归一化权重
        total = sum(self._weights.values())
        if total > 0:
            self._weights = {k: v / total for k, v in self._weights.items()}

        logger.info(
            f"TimeSeriesEnsemble 初始化完成 | "
            f"权重={self._weights} | 可用模型={[k for k, v in AVAILABLE_MODELS.items() if v]}"
        )

    def add_model(self, name: str, model_instance: Any):
        """
        添加模型到集成。

        Args:
            name: 模型标识名 ('moirai', 'timesfm', 'patchtst')。
            model_instance: 模型实例，需支持 predict() 方法。
        """
        if name not in _DEFAULT_WEIGHTS:
            logger.warning(f"未知模型名 '{name}'，已添加但不参与投票")
            # 仍加入以便 fallback 使用
        self._models[name] = model_instance
        logger.info(f"模型 '{name}' 已添加到集成 (实例: {type(model_instance).__name__})")

    # ------------------------------------------------------------------
    # 核心预测
    # ------------------------------------------------------------------

    def predict(
        self,
        stock_code: str,
        history: Optional[np.ndarray] = None,
        lookforward: int = 20,
    ) -> Dict:
        """
        集成预测 — 调用所有可用模型，加权投票生成最终决策。

        流程:
            1. 遍历所有已注册的模型，逐个调用 predict()
            2. 单个模型失败不影响其他模型
            3. 计算加权方向分数和一致性
            4. 返回结构化结果

        Args:
            stock_code: 股票代码 (如 'sz300620')。
            history: 历史价格数据，形状 (n, 5) [open, high, low, close, volume]。
            lookforward: 预测未来天数。

        Returns:
            result: {
                'direction': 'buy'/'sell'/'neutral',
                'confidence': float (0-1),
                'consensus': float (0-1, 模型间一致性),
                'model_predictions': {
                    name: {'direction': str, 'confidence': float, 'forecast': np.ndarray}
                },
                'weights': {name: float},
                'weighted_score': float (-1 to 1),
                'n_models': int (调用成功的模型数),
            }

            若无任何可用模型:
                {'direction': 'neutral', 'confidence': 0.0, ...}
        """
        model_predictions: Dict[str, Dict] = {}
        available_names: List[str] = []

        for name in _DEFAULT_WEIGHTS:
            if name not in self._models:
                continue
            model = self._models[name]
            try:
                pred = model.predict(
                    stock_code=stock_code,
                    history=history,
                    lookforward=lookforward,
                )
                # 标准化预测结果格式
                direction = pred.get('direction', 'neutral')
                confidence = float(pred.get('confidence', 0.5))
                forecast = pred.get('forecast', None)

                model_predictions[name] = {
                    'direction': direction,
                    'confidence': confidence,
                    'forecast': forecast,
                }
                available_names.append(name)
            except Exception as exc:
                logger.error(
                    f"模型 '{name}' 预测失败 ({stock_code}): {exc}",
                    extra={'stock_code': stock_code, 'model': name},
                )

        # 无可用模型
        if not available_names:
            logger.warning(f"所有模型均不可用，返回中性预测 ({stock_code})")
            return {
                'direction': 'neutral',
                'confidence': 0.0,
                'consensus': 0.0,
                'model_predictions': {},
                'weights': dict(self._weights),
                'weighted_score': 0.0,
                'n_models': 0,
            }

        # 计算加权方向分数
        weighted_score = 0.0
        weight_sum = 0.0
        for name in available_names:
            w = self._weights.get(name, 1.0 / len(available_names))
            d = _safe_direction(model_predictions[name])
            weighted_score += w * d
            weight_sum += w

        if weight_sum > 0:
            weighted_score /= weight_sum

        # 方向
        direction = _safe_direction_label(weighted_score)

        # 一致性 & 置信度
        consensus = self.get_consensus(model_predictions)
        confidence = self._compute_confidence(weighted_score, consensus, len(available_names))

        logger.info(
            f"集成预测 ({stock_code}): direction={direction}, "
            f"confidence={confidence:.3f}, consensus={consensus:.3f}, "
            f"score={weighted_score:.3f}, models={len(available_names)}"
        )

        return {
            'direction': direction,
            'confidence': confidence,
            'consensus': consensus,
            'model_predictions': model_predictions,
            'weights': {k: v for k, v in self._weights.items()},
            'weighted_score': float(weighted_score),
            'n_models': len(available_names),
        }

    # ------------------------------------------------------------------
    # 动态权重 (IC-IR)
    # ------------------------------------------------------------------

    def update_weights_ic(
        self,
        stock_code: str,
        predictions: Dict[str, np.ndarray],
        actual_returns: np.ndarray,
    ):
        """
        基于 IC (Information Coefficient) 更新模型权重。

        IC = Pearson 相关系数(预测值, 实际收益率)
        IC-IR = IC / std(IC) (信息比率)

        权重更新:
            1. 计算每个模型在 lookback 窗口内的 IC
            2. 将 IC 归一化为权重 (负 IC 截断为 0)
            3. 自动归一化使权重之和为 1

        Args:
            stock_code: 股票代码 (用于日志)。
            predictions: {model_name: predicted_returns}，每个值为 np.ndarray。
            actual_returns: 实际收益率序列 np.ndarray。
        """
        if not predictions:
            logger.warning("update_weights_ic: 无预测数据")
            return

        if len(actual_returns) == 0:
            logger.warning("update_weights_ic: 实际收益率为空")
            return

        n = min(self.lookback, len(actual_returns))
        actual_slice = actual_returns[-n:]

        ic_scores: Dict[str, float] = {}
        for name, preds in predictions.items():
            if len(preds) < 2:
                ic_scores[name] = 0.0
                continue
            pred_slice = preds[-n:]
            # 对齐长度
            min_len = min(len(pred_slice), len(actual_slice))
            if min_len < 2:
                ic_scores[name] = 0.0
                continue
            p = pred_slice[:min_len]
            a = actual_slice[:min_len]
            # Pearson 相关系数
            if np.std(p) < 1e-10 or np.std(a) < 1e-10:
                ic_scores[name] = 0.0
            else:
                corr = np.corrcoef(p, a)[0, 1]
                ic_scores[name] = float(np.nan_to_num(corr, nan=0.0))

        # 负 IC 截断为 0 (负相关模型不应获得权重)
        positive_ic = {k: max(v, 0.0) for k, v in ic_scores.items()}
        total = sum(positive_ic.values())

        if total > 0:
            new_weights = {k: v / total for k, v in positive_ic.items()}
        else:
            # 全部 IC <= 0，回退等权重
            n_models = len(predictions)
            new_weights = {k: 1.0 / n_models for k in predictions}

        # 只更新已注册的模型权重
        for name in _DEFAULT_WEIGHTS:
            if name in new_weights:
                self._weights[name] = new_weights.get(name, self._weights.get(name, 1.0 / 3))

        total_w = sum(self._weights.values())
        if total_w > 0:
            self._weights = {k: v / total_w for k, v in self._weights.items()}

        logger.info(
            f"IC 权重更新 ({stock_code}): IC={ic_scores} | "
            f"weights={self._weights}"
        )

    # ------------------------------------------------------------------
    # 一致性计算
    # ------------------------------------------------------------------

    def get_consensus(self, model_predictions: Dict[str, Dict]) -> float:
        """
        计算模型间一致性 (0-1)。

        方法:
            1. 将每个模型的方向转换为分数 [-1, 1]
            2. consensus = 1 - (std(directions) / max_possible_std)
            3. 1 = 所有模型方向一致, 0 = 完全分歧

        Args:
            model_predictions: {name: {'direction': str, ...}}

        Returns:
            consensus: float [0, 1]
        """
        if len(model_predictions) <= 1:
            return 1.0

        directions = np.array([_safe_direction(p) for p in model_predictions.values()])

        if directions.shape[0] < 2:
            return 1.0

        std_val = float(np.std(directions))
        # 最大可能标准差: 当一半为 -1 一半为 +1 时
        # 对于二元方向 (buy/sell)，max_std = 1.0
        max_std = 1.0

        consensus = max(0.0, min(1.0, 1.0 - std_val / max_std))
        return consensus

    # ------------------------------------------------------------------
    # 回退预测
    # ------------------------------------------------------------------

    def get_fallback_prediction(
        self,
        stock_code: str,
        available_models: List[str],
    ) -> Optional[Dict]:
        """
        当部分模型失败时的回退预测。

        Args:
            stock_code: 股票代码。
            available_models: 可用的模型名列表。

        Returns:
            若无可用的集成模型则返回 None；
            否则返回与 predict() 相同格式的结果 (但仅基于可用模型)。
        """
        if not available_models:
            logger.warning(f"回退预测: 无可用的模型 ({stock_code})")
            return None

        # 使用可用模型子集重新计算权重
        fallback_weights = {}
        for name in _DEFAULT_WEIGHTS:
            if name in available_models:
                fallback_weights[name] = self._weights.get(name, 1.0 / len(available_models))

        total = sum(fallback_weights.values())
        if total > 0:
            fallback_weights = {k: v / total for k, v in fallback_weights.items()}

        logger.info(
            f"回退预测 ({stock_code}): 使用模型={available_models}, "
            f"权重={fallback_weights}"
        )
        return {
            'direction': 'neutral',
            'confidence': 0.3,
            'consensus': 1.0,
            'model_predictions': {},
            'weights': fallback_weights,
            'weighted_score': 0.0,
            'n_models': len(available_models),
            'fallback': True,
        }

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_confidence(
        weighted_score: float,
        consensus: float,
        n_models: int,
    ) -> float:
        """
        综合置信度 = f(加权方向强度, 模型一致性, 模型数量)。

        Args:
            weighted_score: 加权方向分数 [-1, 1]。
            consensus: 模型间一致性 [0, 1]。
            n_models: 参与投票的模型数量。

        Returns:
            confidence: float [0, 1]
        """
        # 方向强度 (绝对值)
        intensity = abs(weighted_score)

        # 模型数量因子 (1个模型信心低, 3个模型信心高)
        n_factor = min(1.0, n_models / 3.0)

        # 综合置信度: 一致性主导, 方向强度辅助
        confidence = 0.6 * consensus + 0.25 * intensity + 0.15 * n_factor
        return float(max(0.0, min(1.0, confidence)))

    def get_status(self) -> Dict:
        """
        获取集成状态 (用于监控/API)。

        Returns:
            {
                'available_models': {'moirai': True/False, ...},
                'registered_models': ['name', ...],
                'weights': {...},
                'lookback': int,
            }
        """
        return {
            'available_models': dict(AVAILABLE_MODELS),
            'registered_models': list(self._models.keys()),
            'weights': dict(self._weights),
            'lookback': self.lookback,
        }

    def __repr__(self) -> str:
        return (
            f"TimeSeriesEnsemble(weights={self._weights}, "
            f"models={list(self._models.keys())}, "
            f"lookback={self.lookback})"
        )
