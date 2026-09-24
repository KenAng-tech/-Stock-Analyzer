#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
增强型 Regime 检测器 (Phase 2 增强)

新增功能 (2026-07-07):
1. ADX (Average Directional Index) — 趋势强度
2. Bollinger Band Width — 波动率 regime
3. Regime 转移矩阵 (Markov Transition Matrix)
4. 多指标融合 Regime 评分
5. 过渡平滑 (HMM-like smoothing)

架构:
    Price → Technical Indicators → Regime Scores → Markov Smoothing → Final Regime
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque, Counter
from datetime import datetime, timedelta

from modules.logger import logger


# ── Regime 数据类 ──────────────────────────────────────────────

@dataclass
class RegimeSignal:
    """Regime 信号"""
    regime: str = 'sideways'
    confidence: float = 0.5
    scores: Dict[str, float] = field(default_factory=dict)
    indicators: Dict[str, float] = field(default_factory=dict)
    transition_prob: Dict[str, float] = field(default_factory=dict)
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0:
            self.timestamp = datetime.now().timestamp()


@dataclass
class RegimeTransitionMatrix:
    """Regime 转移矩阵"""
    states: List[str] = field(default_factory=lambda: ['bull', 'bear', 'sideways', 'volatile'])
    transition_matrix: np.ndarray = field(default_factory=lambda: np.eye(4) * 0.6 + 0.1)
    observation_counts: Dict[str, int] = field(default_factory=dict)
    observation_transitions: Dict[str, int] = field(default_factory=dict)

    def update(self, prev_regime: str, curr_regime: str):
        """更新转移矩阵"""
        if prev_regime not in self.states or curr_regime not in self.states:
            return
        prev_idx = self.states.index(prev_regime)
        curr_idx = self.states.index(curr_regime)
        self.observation_counts[prev_regime] = self.observation_counts.get(prev_regime, 0) + 1
        self.observation_transitions[f"{prev_regime}->{curr_regime}"] = \
            self.observation_transitions.get(f"{prev_regime}->{curr_regime}", 0) + 1
        self.transition_matrix[prev_idx, curr_idx] += 0.05
        # 归一化
        row_sum = self.transition_matrix[prev_idx].sum()
        if row_sum > 0:
            self.transition_matrix[prev_idx] /= row_sum

    def get_transition_prob(self, regime: str) -> Dict[str, float]:
        """获取从某 regime 转移的概率"""
        if regime not in self.states:
            return {s: 1.0 / len(self.states) for s in self.states}
        idx = self.states.index(regime)
        probs = {self.states[j]: float(self.transition_matrix[idx, j]) for j in range(len(self.states))}
        return probs


# ── 技术指标计算 ──────────────────────────────────────────────

class TechnicalIndicatorCalculator:
    """技术指标计算器 (增强版)"""

    @staticmethod
    def adx(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray,
            period: int = 14) -> float:
        """
        ADX (Average Directional Index)
        衡量趋势强度:
        - ADX > 25: 强趋势
        - ADX < 20: 弱趋势/震荡
        - 20-25: 中等趋势
        """
        if len(closes) < period + 1:
            return 20.0  # 默认中等

        # 计算方向运动 (DM)
        up_move = np.diff(highs)
        down_move = -np.diff(lows)

        pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        # 平滑
        atr = np.zeros(len(closes))
        tr = np.maximum(highs[1:] - lows[1:],
                        np.abs(highs[1:] - closes[:-1]),
                        np.abs(lows[1:] - closes[:-1]))
        atr[1:] = TechnicalIndicatorCalculator._smooth(tr, period)

        smoothed_pos = TechnicalIndicatorCalculator._smooth(pos_dm, period)
        smoothed_neg = TechnicalIndicatorCalculator._smooth(neg_dm, period)

        # DI+ 和 DI-
        dx = np.zeros(len(closes))
        for i in range(period, len(closes)):
            if atr[i] > 0:
                dx[i] = 100 * abs(smoothed_pos[i - period] - smoothed_neg[i - period]) / atr[i]

        if len(dx) > period:
            adx = np.mean(dx[-period:])
        else:
            adx = 20.0

        return float(adx)

    @staticmethod
    def bollinger_width(closes: np.ndarray, period: int = 20,
                        num_std: float = 2.0) -> float:
        """
        Bollinger Band Width
        (Upper - Lower) / Middle
        - 宽: 高波动
        - 窄: 低波动 (可能即将突破)
        """
        if len(closes) < period:
            return 0.05

        ma = np.mean(closes[-period:])
        std = np.std(closes[-period:])
        upper = ma + num_std * std
        lower = ma - num_std * std
        return float((upper - lower) / ma) if ma > 0 else 0.05

    @staticmethod
    def _smooth(values: np.ndarray, period: int) -> np.ndarray:
        """平滑 (EMA-like)"""
        result = np.zeros_like(values)
        alpha = 2.0 / (period + 1)
        result[0] = values[0]
        for i in range(1, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i - 1]
        return result

    @staticmethod
    def momentum(closes: np.ndarray, period: int = 10) -> float:
        """动量指标"""
        if len(closes) < period + 1:
            return 0.0
        return float((closes[-1] / closes[-period] - 1) * 100)

    @staticmethod
    def volatility(closes: np.ndarray, period: int = 20) -> float:
        """波动率 (年化)"""
        if len(closes) < period:
            return 0.2
        returns = np.diff(np.log(closes[-period:]))
        return float(np.std(returns) * np.sqrt(252) * 100)

    @staticmethod
    def volume_trend(volumes: np.ndarray, short: int = 5, long: int = 20) -> float:
        """成交量趋势"""
        if len(volumes) < long:
            return 0.0
        short_avg = np.mean(volumes[-short:])
        long_avg = np.mean(volumes[-long:])
        return float((short_avg / long_avg - 1) * 100) if long_avg > 0 else 0.0


# ── 增强型 Regime 检测器 ──────────────────────────────────────

class EnhancedRegimeDetectorV2:
    """
    增强型 Regime 检测器 V2

    综合多个指标进行 Regime 检测:
    1. MA 趋势 (5/20/60)
    2. ADX 趋势强度
    3. Bollinger Band Width 波动率
    4. 动量
    5. 成交量趋势
    6. Markov 转移平滑

    输出:
    - regime: 'bull' / 'bear' / 'sideways' / 'volatile'
    - confidence: 置信度
    - scores: 各维度得分
    - transition_prob: 转移概率
    """

    def __init__(self, persistence: int = 3, smoothing_window: int = 5):
        self.persistence = persistence
        self.smoothing_window = smoothing_window
        self.indicator_calculator = TechnicalIndicatorCalculator()

        # 标记
        self._bull_threshold = 0.3
        self._bear_threshold = -0.3
        self._adx_strong = 25.0
        self._bbw_high = 0.08
        self._bbw_low = 0.03

        # Regime 历史
        self._regime_history: deque = deque(maxlen=persistence * 2)
        self._current_regime = 'sideways'
        self._regime_count = 0
        self._last_regime = None

        # Markov 转移矩阵
        self.transition_matrix = RegimeTransitionMatrix()

    def detect(self, stock_data: Dict,
               klines: Optional[List[Dict]] = None) -> RegimeSignal:
        """
        检测当前市场状态

        Args:
            stock_data: 股票数据
            klines: K线数据

        Returns:
            RegimeSignal
        """
        if not klines or len(klines) < 60:
            return self._default_signal()

        closes = np.array([float(k.get('close', 0)) for k in klines if float(k.get('close', 0)) > 0])
        highs = np.array([float(k.get('high', 0)) for k in klines if float(k.get('high', 0)) > 0])
        lows = np.array([float(k.get('low', 0)) for k in klines if float(k.get('low', 0)) > 0])
        volumes = np.array([float(k.get('volume', 0)) for k in klines if float(k.get('volume', 0)) > 0])

        if len(closes) < 60:
            return self._default_signal()

        # 计算各指标
        ma5 = np.mean(closes[-5:])
        ma20 = np.mean(closes[-20:])
        ma60 = np.mean(closes[-60:])

        adx = self.indicator_calculator.adx(closes, highs, lows)
        bbw = self.indicator_calculator.bollinger_width(closes)
        momentum = self.indicator_calculator.momentum(closes)
        vol = self.indicator_calculator.volatility(closes)
        vol_trend = self.indicator_calculator.volume_trend(volumes)

        # ── Regime 评分 ──
        scores = {}

        # 趋势得分 (MA 排列 + 价格位置)
        trend_score = 0.0
        if ma5 > ma20 > ma60:
            trend_score = 0.6 + 0.4 * (closes[-1] > ma5)
        elif ma5 < ma20 < ma60:
            trend_score = -0.6 - 0.4 * (closes[-1] < ma5)
        else:
            trend_score = 0.0
        scores['trend'] = trend_score

        # 趋势强度得分 (ADX)
        if adx > self._adx_strong:
            strength_score = 0.8 if trend_score > 0 else -0.8
        elif adx > 15:
            strength_score = 0.4 * np.sign(trend_score)
        else:
            strength_score = 0.0
        scores['strength'] = strength_score

        # 波动率得分
        if bbw > self._bbw_high:
            vol_score = 0.6
        elif bbw < self._bbw_low:
            vol_score = -0.6  # 窄幅 = 可能即将突破
        else:
            vol_score = 0.0
        scores['volatility'] = vol_score

        # 动量得分
        scores['momentum'] = np.clip(momentum / 10, -1, 1)

        # 成交量得分
        scores['volume'] = np.clip(vol_trend / 20, -1, 1)

        # ── 综合 Regime 判定 ──
        weighted_score = (
            0.25 * scores['trend'] +
            0.20 * scores['strength'] +
            0.20 * scores['volatility'] +
            0.20 * scores['momentum'] +
            0.15 * scores['volume']
        )

        # 判定 Regime
        if weighted_score > self._bull_threshold and adx > 20:
            regime = 'bull'
        elif weighted_score < self._bear_threshold and adx > 20:
            regime = 'bear'
        elif vol > 35:  # 年化波动率 > 35%
            regime = 'volatile'
        else:
            regime = 'sideways'

        # ── 平滑 (Persistence) ──
        self._regime_history.append(regime)
        recent = list(self._regime_history)[-self.persistence:]
        if len(recent) >= self.persistence:
            recent_votes = Counter(recent)
            most_common = recent_votes.most_common(1)[0]
            if most_common[1] >= self.persistence * 0.6:
                regime = most_common[0]

        self._current_regime = regime
        self._last_regime = self._last_regime or regime
        self.transition_matrix.update(self._last_regime, regime)
        self._last_regime = regime

        # ── 转移概率 ──
        transition_prob = self.transition_matrix.get_transition_prob(regime)

        # ── 置信度 ──
        confidence = min(1.0, abs(weighted_score) * 1.2 + 0.2)

        return RegimeSignal(
            regime=regime,
            confidence=round(confidence, 4),
            scores=scores,
            indicators={
                'adx': round(adx, 2),
                'bollinger_width': round(bbw, 4),
                'momentum': round(momentum, 2),
                'volatility': round(vol, 2),
                'volume_trend': round(vol_trend, 2),
            },
            transition_prob=transition_prob,
        )

    def get_regime_probability(self, stock_data: Dict,
                               klines: Optional[List[Dict]] = None) -> Dict[str, float]:
        """
        获取各 Regime 的概率 (基于 Markov 转移 + 当前信号)
        """
        signal = self.detect(stock_data, klines)
        transition = signal.transition_prob

        # 当前信号作为观测似然
        scores = signal.scores
        trend_score = scores.get('trend', 0)
        vol_score = scores.get('volatility', 0)

        # 观测似然
        likelihood = {
            'bull': max(0.1, 0.3 + trend_score * 0.5 + vol_score * 0.2),
            'bear': max(0.1, 0.3 - trend_score * 0.5 + vol_score * 0.2),
            'sideways': max(0.1, 0.3 + abs(trend_score) * 0.3 + abs(vol_score) * 0.1),
            'volatile': max(0.1, 0.2 + abs(vol_score) * 0.5 + (1 - abs(trend_score)) * 0.2),
        }

        # Bayes 规则: P(regime|obs) ∝ P(obs|regime) * P(regime)
        posterior = {}
        for regime in ['bull', 'bear', 'sideways', 'volatile']:
            prior = transition.get(regime, 0.25)
            posterior[regime] = likelihood[regime] * prior

        # 归一化
        total = sum(posterior.values())
        if total > 0:
            posterior = {k: round(v / total, 4) for k, v in posterior.items()}

        return posterior

    def _default_signal(self) -> RegimeSignal:
        return RegimeSignal(
            regime='sideways',
            confidence=0.5,
            scores={},
            indicators={},
            transition_prob={},
        )

    def get_regime_summary(self) -> Dict:
        """获取 Regime 摘要"""
        return {
            'current_regime': self._current_regime,
            'confidence': min(1.0, abs(self._regime_count) * 0.1 + 0.5) if self._regime_count > 0 else 0.5,
            'transition_matrix': self.transition_matrix.transition_matrix.tolist() if hasattr(self.transition_matrix, 'transition_matrix') else [],
            'regime_history': list(self._regime_history)[-10:],
        }


# ── 全局单例 ──────────────────────────────────────────────

_enhanced_detector: Optional[EnhancedRegimeDetectorV2] = None
_detector_lock = __import__('threading').Lock()


def get_enhanced_regime_detector(persistence: int = 3) -> EnhancedRegimeDetectorV2:
    """获取全局增强型 Regime 检测器"""
    global _enhanced_detector
    if _enhanced_detector is None:
        with _detector_lock:
            if _enhanced_detector is None:
                _enhanced_detector = EnhancedRegimeDetectorV2(persistence=persistence)
    return _enhanced_detector
