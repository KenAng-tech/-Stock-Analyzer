#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
市场状态自适应路由器 — 2026 SOTA 动态策略选择

功能:
    - 集成 HMM/ADWIN 市场状态检测结果
    - 根据市场状态自动切换预测模型/策略
    - 支持模型权重动态调整
    - 状态转换平滑过渡 (避免频繁切换)

2026 趋势:
    市场状态切换加速，需要自适应路由
    HMM + ADWIN 双检测 + 融合决策是 SOTA
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

logger = logging.getLogger('stock_analyzer.modules')


class MarketRegime(Enum):
    """市场状态"""
    BULL = 'bull'           # 牛市
    BEAR = 'bear'           # 熊市
    RANGING = 'ranging'     # 震荡
    HIGH_VOL = 'high_vol'   # 高波动
    LOW_VOL = 'low_vol'     # 低波动
    TRENDING = 'trending'   # 趋势
    UNKNOWN = 'unknown'     # 未知


@dataclass
class ModelRoute:
    """模型路由配置"""
    model_name: str
    regimes: List[MarketRegime]  # 适用市场状态
    weight: float = 1.0
    priority: int = 0  # 优先级 (高优先)
    min_confidence: float = 0.3  # 最小置信度
    performance: Dict[str, float] = field(default_factory=dict)  # 各状态下的表现


@dataclass
class RegimeRouterConfig:
    """路由器配置"""
    transition_smooth_window: int = 5  # 状态转换平滑窗口
    min_regime_duration: int = 3  # 最小状态持续期
    confidence_threshold: float = 0.4  # 状态置信度阈值
    enable_hmm: bool = True
    enable_adwin: bool = True
    fallback_regime: MarketRegime = MarketRegime.UNKNOWN


class RegimeAdaptiveRouter:
    """
    市场状态自适应路由器

    用法:
        router = RegimeAdaptiveRouter()
        router.register_model('patchtst', model_a, [MarketRegime.TRENDING])
        router.register_model('mamba', model_b, [MarketRegime.HIGH_VOL])
        regime = router.detect_regime(price_data)
        model = router.route(regime)
    """

    def __init__(self, config: Optional[RegimeRouterConfig] = None):
        self.config = config or RegimeRouterConfig()
        self._routes: List[ModelRoute] = []
        self._regime_history: deque = deque(maxlen=20)
        self._current_regime: MarketRegime = MarketRegime.UNKNOWN
        self._regime_duration: int = 0
        self._hmm_detector = None
        self._adwin_detector = None

    def register_model(
        self,
        name: str,
        model: Any,
        regimes: List[MarketRegime],
        priority: int = 0,
    ) -> None:
        """
        注册模型路由

        Args:
            name: 模型名称
            model: 模型实例
            regimes: 适用市场状态列表
            priority: 优先级
        """
        route = ModelRoute(
            model_name=name,
            regimes=regimes,
            priority=priority,
        )
        self._routes.append(route)
        logger.info(
            f"[RegimeRouter] 注册模型 '{name}' "
            f"适用状态: {[r.value for r in regimes]}"
        )

    def detect_regime(
        self,
        prices: np.ndarray,
        volumes: Optional[np.ndarray] = None,
    ) -> MarketRegime:
        """
        检测当前市场状态

        Args:
            prices: 价格序列
            volumes: 成交量序列

        Returns:
            市场状态
        """
        if len(prices) < 20:
            return MarketRegime.UNKNOWN

        # 1. HMM 检测
        hmm_regime = None
        if self.config.enable_hmm:
            try:
                from modules.hmm_market_detector import MarketRegimeDetector
                if self._hmm_detector is None:
                    self._hmm_detector = MarketRegimeDetector()
                hmm_result = self._hmm_detector.detect_regime(prices)
                if isinstance(hmm_result, dict):
                    hmm_regime = hmm_result.get('regime', 'unknown')
            except Exception as e:
                logger.debug(f"[RegimeRouter] HMM 检测失败: {e}")

        # 2. ADWIN 概念漂移检测
        adwin_drift = False
        if self.config.enable_adwin:
            try:
                from modules.adwin import ADWIN
                if self._adwin_detector is None:
                    self._adwin_detector = ADWIN()
                returns = np.diff(prices) / prices[:-1]
                for r in returns:
                    self._adwin_detector.update(r)
                adwin_drift = self._adwin_detector.drift_detected
            except Exception as e:
                logger.debug(f"[RegimeRouter] ADWIN 检测失败: {e}")

        # 3. 技术指标辅助判断
        returns = np.diff(prices) / prices[:-1]
        volatility = float(np.std(returns)) if len(returns) > 1 else 0.0
        trend = float(np.mean(returns[-5:])) if len(returns) >= 5 else 0.0

        # 4. 融合决策
        regime = self._fuse_signals(
            hmm_regime=hmm_regime,
            adwin_drift=adwin_drift,
            trend=trend,
            volatility=volatility,
        )

        # 5. 状态平滑
        self._regime_history.append(regime)
        regime = self._smooth_regime(regime)

        self._current_regime = regime
        return regime

    def _fuse_signals(
        self,
        hmm_regime: Optional[str],
        adwin_drift: bool,
        trend: float,
        volatility: float,
    ) -> MarketRegime:
        """融合多信号判断市场状态"""
        scores: Dict[MarketRegime, float] = {
            r: 0.0 for r in MarketRegime
        }

        # HMM 信号
        if hmm_regime:
            regime_map = {
                'bull': MarketRegime.BULL,
                'bear': MarketRegime.BEAR,
                'ranging': MarketRegime.RANGING,
                'high_vol': MarketRegime.HIGH_VOL,
                'low_vol': MarketRegime.LOW_VOL,
            }
            mapped = regime_map.get(hmm_regime)
            if mapped:
                scores[mapped] += 2.0

        # ADWIN 信号
        if adwin_drift:
            scores[MarketRegime.HIGH_VOL] += 1.5
            scores[MarketRegime.BEAR] += 1.0
        else:
            scores[MarketRegime.LOW_VOL] += 1.0

        # 趋势信号
        if trend > 0.005:
            scores[MarketRegime.BULL] += 1.5
            scores[MarketRegime.TRENDING] += 1.0
        elif trend < -0.005:
            scores[MarketRegime.BEAR] += 1.5
            scores[MarketRegime.TRENDING] += 1.0
        else:
            scores[MarketRegime.RANGING] += 1.5

        # 波动率信号
        if volatility > 0.02:
            scores[MarketRegime.HIGH_VOL] += 1.0
        else:
            scores[MarketRegime.LOW_VOL] += 1.0

        # 取最高分
        best_regime = max(scores, key=scores.get)
        best_score = scores[best_regime]

        if best_score < self.config.confidence_threshold:
            return MarketRegime.UNKNOWN

        return best_regime

    def _smooth_regime(self, new_regime: MarketRegime) -> MarketRegime:
        """状态平滑 (避免频繁切换)"""
        if new_regime == self._current_regime:
            self._regime_duration += 1
            return new_regime

        # 检查历史状态
        if len(self._regime_history) >= self.config.transition_smooth_window:
            recent = list(self._regime_history)[-self.config.transition_smooth_window:]
            mode_regime = max(set(recent), key=recent.count)

            # 如果新模式不是主流，等待确认
            if mode_regime != new_regime and self._regime_duration < self.config.min_regime_duration:
                return self._current_regime

        # 状态切换
        self._regime_duration = 1
        return new_regime

    def route(self, regime: Optional[MarketRegime] = None) -> List[ModelRoute]:
        """
        根据市场状态路由到最佳模型

        Args:
            regime: 市场状态 (None 使用当前状态)

        Returns:
            匹配的模型路由列表 (按优先级排序)
        """
        target_regime = regime or self._current_regime

        # 筛选适用模型
        matched = [
            route for route in self._routes
            if target_regime in route.regimes
        ]

        # 按优先级和权重排序
        matched.sort(key=lambda r: (r.priority, r.weight), reverse=True)

        if not matched:
            logger.info(
                f"[RegimeRouter] 状态 '{target_regime.value}' 无匹配模型, "
                f"使用全部模型"
            )
            return self._routes

        return matched

    def update_model_performance(
        self,
        model_name: str,
        regime: MarketRegime,
        sharpe: float,
    ) -> None:
        """更新模型在各状态下的表现"""
        for route in self._routes:
            if route.model_name == model_name:
                route.performance[str(regime.value)] = sharpe
                break

    def get_status(self) -> dict:
        """获取状态"""
        return {
            'current_regime': self._current_regime.value,
            'regime_duration': self._regime_duration,
            'total_routes': len(self._routes),
            'routes': [
                {
                    'model': r.model_name,
                    'regimes': [reg.value for reg in r.regimes],
                    'priority': r.priority,
                    'weight': r.weight,
                }
                for r in self._routes
            ],
            'history': [r.value for r in self._regime_history],
            'config': {
                'smooth_window': self.config.transition_smooth_window,
                'min_duration': self.config.min_regime_duration,
                'confidence_threshold': self.config.confidence_threshold,
            },
        }


# 模块级单例
_regime_router: Optional[RegimeAdaptiveRouter] = None


def get_regime_router() -> RegimeAdaptiveRouter:
    """获取路由器单例"""
    global _regime_router
    if _regime_router is None:
        _regime_router = RegimeAdaptiveRouter()
        logger.info("[RegimeRouter] 单例已创建")
    return _regime_router