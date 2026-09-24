#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
多模态融合预测 (Multi-Modal Fusion Prediction)

融合三类独立信号源，基于市场状态 (Regime) 动态权重加权：
- 价格/技术指标信号 (Price/Technical Signal)
- 情感分析信号 (Sentiment Signal)
- 基本面信号 (Fundamental Signal)

不同 regime 下最优模态不同：
- bullish (牛市): 价格信号权重高
- bearish (熊市): 基本面信号权重高
- sideways (震荡): 价格信号中等，情感信号提升
- volatile (高波动): 基本面 + 价格各半

参考:
- 2026 量化最佳实践: Multi-Modal Fusion + Regime-Switching Weights
- Renaissance: 跨模态注意力机制 (简化版: 加权投票)
- Citadel: 分层融合架构 (独立信号 → 加权融合 → 统一决策)

使用:
    from modules.multi_modal_fusion import MultiModalFusion
    fusion = MultiModalFusion()
    result = fusion.predict('sz300620')
"""

import numpy as np
from typing import Dict, List, Optional
from datetime import datetime

from modules.logger import logger
from modules.dynamic_cache import cache

# ── Regime 定义 ──────────────────────────────────────────────

REGIMES = ['bullish', 'bearish', 'sideways', 'volatile']

# 各 regime 下的模态权重 (price, sentiment, fundamental)
# 权重总和为 1.0
REGIME_MODALITY_WEIGHTS = {
    'bullish': {
        'price': 0.60,       # 趋势市中价格信号最强
        'sentiment': 0.25,   # 情绪偏正面，辅助确认趋势
        'fundamental': 0.15, # 基本面权重较低
    },
    'bearish': {
        'price': 0.35,       # 熊市中价格信号噪音大
        'sentiment': 0.25,   # 情绪偏负面，需警惕
        'fundamental': 0.40, # 质量因子在熊市中最重要
    },
    'sideways': {
        'price': 0.50,       # 震荡市价格信号中等
        'sentiment': 0.30,   # 情绪在震荡市中更有区分度
        'fundamental': 0.20, # 基本面中等权重
    },
    'volatile': {
        'price': 0.40,       # 波动市中价格信号噪音大但仍重要
        'sentiment': 0.20,   # 情绪在波动市中不可靠
        'fundamental': 0.40, # 基本面提供稳定锚定
    },
}


# ── IC 自适应权重 ──────────────────────────────────────────────

class ModalityICTracker:
    """
    模态 IC 追踪器 — 基于历史预测 + 实际收益计算各模态 IC。

    原理: 每个模态的输出可以看作一个"因子"，IC = 因子值与未来收益的 Pearson 相关系数。
    IC 高的模态权重应该更高，IC 为负则权重为 0 (该模态无效)。

    用法:
        tracker = ModalityICTracker()
        tracker.record(modalities, actual_return, date)
        weights = tracker.get_adaptive_weights()  # 返回 {price, sentiment, fundamental}
    """

    def __init__(self, max_history: int = 200):
        self._history: List[Dict] = []  # [{date, modalities, return}, ...]
        self.max_history = max_history

    def record(self, modalities: Dict[str, Dict], actual_return: float, date: str = '') -> None:
        """记录一次预测 + 实际收益"""
        entry = {
            'date': date or datetime.now().strftime('%Y-%m-%d'),
            'modalities': modalities,
            'return': actual_return,
        }
        self._history.append(entry)
        if len(self._history) > self.max_history:
            self._history.pop(0)

    def get_adaptive_weights(self, min_samples: int = 20) -> Optional[Dict[str, float]]:
        """
        基于 IC 计算自适应权重。

        权重 ∝ |IC|，IC 为负则权重为 0 (该模态无效)。
        如果样本不足，返回 None (调用方回退到 regime 默认权重)。
        """
        if len(self._history) < min_samples:
            return None

        # 计算各模态的 IC (Pearson 相关系数)
        ic_scores = {}
        for modality in ['price', 'sentiment', 'fundamental']:
            factor_values = []
            returns = []
            for entry in self._history:
                m = entry['modalities'].get(modality, {})
                direction = m.get('direction', 'neutral')
                confidence = m.get('confidence', 0.5)
                # 将 direction 映射为数值: up=1, neutral=0, down=-1
                val = 1.0 if direction == 'up' else (-1.0 if direction == 'down' else 0.0)
                factor_values.append(val * confidence)
                returns.append(entry['return'])

            if len(factor_values) >= min_samples:
                f_arr = np.array(factor_values)
                r_arr = np.array(returns)
                f_std = np.std(f_arr)
                r_std = np.std(r_arr)
                if f_std > 1e-10 and r_std > 1e-10:
                    ic = float(np.corrcoef(f_arr, r_arr)[0, 1])
                    ic_scores[modality] = ic
                else:
                    ic_scores[modality] = 0.0
            else:
                ic_scores[modality] = 0.0

        # 权重 ∝ |IC|，IC < 0 的模态权重保底 0.01 防全零
        abs_ics = {k: max(abs(v), 0.01) for k, v in ic_scores.items()}
        total = sum(abs_ics.values())
        adaptive_weights = {k: round(v / total, 4) for k, v in abs_ics.items()}

        return adaptive_weights

    def get_ic_summary(self) -> Dict:
        """获取 IC 摘要 (用于 API 端点)"""
        return {
            'n_records': len(self._history),
            'max_history': self.max_history,
            'recent_returns': [round(e['return'], 6) for e in self._history[-10:]],
        }


# ── 信号提取器 ──────────────────────────────────────────────

class TechnicalSignalExtractor:
    """
    价格/技术指标信号提取器

    从 K 线数据中提取技术指标，生成独立的涨跌预测信号。
    基于 RSI、MACD、布林带、均线系统综合判断。
    """

    def extract(self, klines: List[Dict]) -> Dict:
        """
        从 K 线数据提取技术信号

        Args:
            klines: K 线数据列表，每个元素含 'open','high','low','close','volume'

        Returns:
            {
                'direction': 'up' | 'down' | 'neutral',
                'confidence': 0.0 ~ 1.0,
                'details': {
                    'rsi': float,
                    'macd_signal': str,
                    'ma_alignment': str,
                    'bollinger_position': str,
                    'bullish_count': int,
                    'bearish_count': int,
                }
            }
        """
        if not klines or len(klines) < 20:
            return {
                'direction': 'neutral',
                'confidence': 0.5,
                'details': {'error': 'K线数据不足 (至少 20 根)'},
            }

        closes = np.array([float(k.get('close', 0)) for k in klines])
        volumes = np.array([float(k.get('volume', 0)) for k in klines])

        # 1. RSI 信号
        rsi = self._calculate_rsi(closes, 14)
        rsi_signal = self._rsi_to_signal(rsi)

        # 2. MACD 信号
        macd_signal = self._calculate_macd_signal(closes)

        # 3. 均线对齐信号
        ma_alignment = self._check_ma_alignment(closes)

        # 4. 布林带位置
        boll_position = self._bollinger_position(closes)

        # 5. 动量信号 (短期 vs 长期均线)
        momentum_signal = 'up' if closes[-1] > closes[-5] else 'down'

        # 综合投票
        signals = [rsi_signal, macd_signal, ma_alignment, boll_position, momentum_signal]
        bullish_count = sum(1 for s in signals if s == 'up')
        bearish_count = sum(1 for s in signals if s == 'down')
        neutral_count = sum(1 for s in signals if s == 'neutral')

        # 决策
        if bullish_count > bearish_count + 1:
            direction = 'up'
        elif bearish_count > bullish_count + 1:
            direction = 'down'
        else:
            direction = 'neutral'

        # 置信度 = 最高票数的比例
        max_votes = max(bullish_count, bearish_count, neutral_count)
        confidence = max_votes / len(signals)

        return {
            'direction': direction,
            'confidence': round(confidence, 4),
            'details': {
                'rsi': round(rsi, 2),
                'macd_signal': macd_signal,
                'ma_alignment': ma_alignment,
                'bollinger_position': boll_position,
                'momentum': momentum_signal,
                'bullish_count': bullish_count,
                'bearish_count': bearish_count,
                'neutral_count': neutral_count,
                'total_signals': len(signals),
            },
        }

    @staticmethod
    def _calculate_rsi(closes: np.ndarray, period: int = 14) -> float:
        """计算 RSI 指标"""
        if len(closes) < period + 1:
            return 50.0
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def _rsi_to_signal(rsi: float) -> str:
        """RSI 转信号"""
        if rsi < 30:
            return 'up'       # 超卖 → 看涨
        elif rsi > 70:
            return 'down'      # 超买 → 看跌
        elif rsi < 45:
            return 'up'        # 偏低 → 偏多
        elif rsi > 55:
            return 'down'      # 偏高 → 偏空
        else:
            return 'neutral'   # 中性区域

    @staticmethod
    def _calculate_macd_signal(closes: np.ndarray) -> str:
        """计算 MACD 信号"""
        if len(closes) < 26:
            return 'neutral'
        ema12 = TechnicalSignalExtractor._ema(closes, 12)
        ema26 = TechnicalSignalExtractor._ema(closes, 26)
        macd = ema12 - ema26
        signal_line = TechnicalSignalExtractor._ema(macd, 9)
        if macd[-1] > signal_line[-1]:
            return 'up'
        elif macd[-1] < signal_line[-1]:
            return 'down'
        return 'neutral'

    @staticmethod
    def _ema(data: np.ndarray, period: int) -> np.ndarray:
        """计算 EMA"""
        multiplier = 2.0 / (period + 1)
        ema = np.zeros_like(data)
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = (data[i] - ema[i - 1]) * multiplier + ema[i - 1]
        return ema

    @staticmethod
    def _check_ma_alignment(closes: np.ndarray) -> str:
        """检查均线对齐"""
        if len(closes) < 60:
            return 'neutral'
        ma5 = np.mean(closes[-5:])
        ma10 = np.mean(closes[-10:])
        ma20 = np.mean(closes[-20:])
        ma60 = np.mean(closes[-60:])

        if ma5 > ma10 > ma20 > ma60:
            return 'up'       # 多头排列
        elif ma5 < ma10 < ma20 < ma60:
            return 'down'     # 空头排列
        return 'neutral'

    @staticmethod
    def _bollinger_position(closes: np.ndarray) -> str:
        """布林带位置判断"""
        if len(closes) < 20:
            return 'neutral'
        ma20 = np.mean(closes[-20:])
        std20 = np.std(closes[-20:])
        upper = ma20 + 2 * std20
        lower = ma20 - 2 * std20
        current = closes[-1]
        if current > upper:
            return 'down'      # 突破上轨 → 可能回调
        elif current < lower:
            return 'up'        # 跌破下轨 → 可能反弹
        return 'neutral'


class SentimentSignalExtractor:
    """
    情感分析信号提取器

    从 sentiment_engine 获取情感分数，转换为涨跌预测信号。
    """

    def extract(self, stock_code: str) -> Dict:
        """
        从情感分析引擎提取信号

        Args:
            stock_code: 股票代码 (如 'sz300620')

        Returns:
            {
                'direction': 'up' | 'down' | 'neutral',
                'confidence': 0.0 ~ 1.0,
                'details': {
                    'score': float,        # -1.0 ~ +1.0
                    'label': str,          # 'positive' | 'negative' | 'neutral'
                    'confidence': float,
                    'n_articles': int,
                }
            }
        """
        try:
            from modules.sentiment_engine import get_sentiment_engine

            engine = get_sentiment_engine()
            sentiment = engine.get_sentiment_score(stock_code)

            # 2026-09-10: get_sentiment_score 已补契约 (此前此调用恒 AttributeError
            # → 情绪通道 3 路融合永久被 0.5 占位噪声顶替); no_data 诚实降级
            if not sentiment or sentiment.get('method') == 'no_data':
                return self._fallback_signal()

            score = sentiment.get('score', 0.0)
            label = sentiment.get('label', 'neutral')
            confidence = sentiment.get('confidence', 0.5)
            n_articles = sentiment.get('n_articles', 0)

            # 情感分数 → 信号
            if score > 0.15:
                direction = 'up'
            elif score < -0.15:
                direction = 'down'
            else:
                direction = 'neutral'

            # 置信度 = |score| × 文章数量系数
            raw_confidence = abs(score)
            if n_articles >= 5:
                raw_confidence = min(1.0, raw_confidence * 1.2)
            elif n_articles >= 2:
                raw_confidence = min(1.0, raw_confidence * 1.1)

            return {
                'direction': direction,
                'confidence': round(max(raw_confidence, confidence * 0.5), 4),
                'details': {
                    'score': round(score, 4),
                    'label': label,
                    'confidence': round(confidence, 4),
                    'n_articles': n_articles,
                },
            }

        except Exception as e:
            logger.warning(f"[SentimentSignalExtractor] 情感信号提取失败: {e}")
            return self._fallback_signal()

    @staticmethod
    def _fallback_signal() -> Dict:
        """无数据时的降级信号"""
        return {
            'direction': 'neutral',
            'confidence': 0.5,
            'details': {'error': '情感分析不可用', 'score': 0.0},
        }


class FundamentalSignalExtractor:
    """
    基本面信号提取器

    从财务数据中提取基本面信号，基于 ROE、营收增长、利润增长等指标。
    """

    def extract(self, stock_code: str, stock_name: str = '') -> Dict:
        """
        从基本面数据提取信号

        Args:
            stock_code: 股票代码 (不带市场前缀，如 '300620')
            stock_name: 股票名称 (可选)

        Returns:
            {
                'direction': 'up' | 'down' | 'neutral',
                'confidence': 0.0 ~ 1.0,
                'details': {
                    'roe': float,
                    'revenue_growth': float,
                    'profit_growth': float,
                    'score': float,        # 综合基本面得分 -1 ~ +1
                }
            }
        """
        try:
            from modules.fundamental_fetcher import FundamentalFetcher

            fetcher = FundamentalFetcher()
            data = fetcher.get_financial_data(stock_code, stock_name)

            if not data:
                return self._fallback_signal()

            # 提取关键指标
            roe = data.get('roe', 0.0) or 0.0
            revenue_growth = data.get('revenue_growth', 0.0) or 0.0
            profit_growth = data.get('profit_growth', 0.0) or 0.0
            gross_margin = data.get('gross_margin', 0.0) or 0.0
            debt_ratio = data.get('debt_ratio', 0.0) or 0.0

            # 综合得分 (-1 ~ +1)
            score = self._calculate_fundamental_score(
                roe, revenue_growth, profit_growth, gross_margin, debt_ratio
            )

            # 得分 → 信号
            if score > 0.15:
                direction = 'up'
            elif score < -0.15:
                direction = 'down'
            else:
                direction = 'neutral'

            # 置信度 = |score|
            confidence = min(1.0, abs(score))

            return {
                'direction': direction,
                'confidence': round(confidence, 4),
                'details': {
                    'roe': round(roe, 2),
                    'revenue_growth': round(revenue_growth, 2),
                    'profit_growth': round(profit_growth, 2),
                    'gross_margin': round(gross_margin, 2),
                    'debt_ratio': round(debt_ratio, 2),
                    'score': round(score, 4),
                },
            }

        except Exception as e:
            logger.warning(f"[FundamentalSignalExtractor] 基本面信号提取失败: {e}")
            return self._fallback_signal()

    @staticmethod
    def _calculate_fundamental_score(
        roe: float,
        revenue_growth: float,
        profit_growth: float,
        gross_margin: float,
        debt_ratio: float,
    ) -> float:
        """
        计算基本面综合得分 (-1 ~ +1)

        评分标准:
        - ROE > 15%: +0.25, > 10%: +0.15, > 5%: +0.05
        - 营收增长 > 20%: +0.25, > 10%: +0.15, > 0%: +0.05
        - 利润增长 > 20%: +0.25, > 10%: +0.15, > 0%: +0.05
        - 毛利率 > 40%: +0.15, > 25%: +0.10, > 10%: +0.05
        - 负债率 < 30%: +0.10, < 50%: +0.05, > 70%: -0.10
        """
        score = 0.0

        # ROE 评分
        if roe > 15:
            score += 0.25
        elif roe > 10:
            score += 0.15
        elif roe > 5:
            score += 0.05
        elif roe < 0:
            score -= 0.15

        # 营收增长评分
        if revenue_growth > 20:
            score += 0.25
        elif revenue_growth > 10:
            score += 0.15
        elif revenue_growth > 0:
            score += 0.05
        elif revenue_growth < -10:
            score -= 0.15

        # 利润增长评分
        if profit_growth > 20:
            score += 0.25
        elif profit_growth > 10:
            score += 0.15
        elif profit_growth > 0:
            score += 0.05
        elif profit_growth < -10:
            score -= 0.15

        # 毛利率评分
        if gross_margin > 40:
            score += 0.15
        elif gross_margin > 25:
            score += 0.10
        elif gross_margin > 10:
            score += 0.05

        # 负债率评分
        if debt_ratio < 30:
            score += 0.10
        elif debt_ratio < 50:
            score += 0.05
        elif debt_ratio > 70:
            score -= 0.10

        # 截断到 [-1, +1]
        return float(np.clip(score, -1.0, 1.0))

    @staticmethod
    def _fallback_signal() -> Dict:
        """无数据时的降级信号"""
        return {
            'direction': 'neutral',
            'confidence': 0.5,
            'details': {'error': '基本面数据不可用', 'score': 0.0},
        }


# ── 融合引擎 ─────────────────────────────────────────────────

class FusionEngine:
    """
    多模态融合引擎

    基于 regime 动态权重，融合三个独立信号源。
    采用加权投票法: 每个模态输出 {direction, confidence}，
    按 regime 权重加权后累加 up/down/neutral 得分。

    IC 自适应: 如果设置了 _ic_tracker，优先使用自适应权重。
    """

    def __init__(self):
        self._ic_tracker: Optional[ModalityICTracker] = None

    def set_ic_tracker(self, tracker: ModalityICTracker) -> None:
        """设置 IC 追踪器 (由 MultiModalFusion 调用)"""
        self._ic_tracker = tracker

    def fuse(
        self,
        price_signal: Dict,
        sentiment_signal: Dict,
        fundamental_signal: Dict,
        regime: str = 'sideways',
    ) -> Dict:
        """
        融合三个模态信号

        Args:
            price_signal: 技术信号 {direction, confidence, details}
            sentiment_signal: 情感信号 {direction, confidence, details}
            fundamental_signal: 基本面信号 {direction, confidence, details}
            regime: 当前市场状态

        Returns:
            {
                'direction': 'up' | 'down' | 'neutral',
                'confidence': 0.0 ~ 1.0,
                'regime': str,
                'modality_weights': {price, sentiment, fundamental},
                'modalities': {
                    'price': {...},
                    'sentiment': {...},
                    'fundamental': {...},
                },
                'scores': {'up': float, 'down': float, 'neutral': float},
                'consensus': float,      # 模态间一致度 0-1
                'timestamp': str,
            }
        """
        if regime not in REGIME_MODALITY_WEIGHTS:
            regime = 'sideways'

        # 尝试 IC 自适应权重 (如果样本充足)
        adaptive_weights = None
        if self._ic_tracker:
            adaptive_weights = self._ic_tracker.get_adaptive_weights()

        if adaptive_weights:
            price_w = adaptive_weights['price']
            sentiment_w = adaptive_weights['sentiment']
            fundamental_w = adaptive_weights['fundamental']
            logger.debug(
                f"[FusionEngine] 使用 IC 自适应权重: "
                f"price={price_w:.3f}, sentiment={sentiment_w:.3f}, fundamental={fundamental_w:.3f}"
            )
        else:
            # 回退到 regime 默认权重
            weights = REGIME_MODALITY_WEIGHTS[regime]
            price_w = weights['price']
            sentiment_w = weights['sentiment']
            fundamental_w = weights['fundamental']

        # 初始化累加器
        scores = {'up': 0.0, 'down': 0.0, 'neutral': 0.0}

        # 加权累加各模态信号
        for signal, w in [
            (price_signal, price_w),
            (sentiment_signal, sentiment_w),
            (fundamental_signal, fundamental_w),
        ]:
            direction = signal.get('direction', 'neutral')
            confidence = signal.get('confidence', 0.5)
            scores[direction] += w * confidence

        # 归一化得分
        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}

        # 最终决策
        direction = max(scores, key=scores.get)
        confidence = scores[direction]

        # 计算模态间一致度 (0 = 完全分歧, 1 = 完全一致)
        consensus = self._calculate_consensus(
            price_signal, sentiment_signal, fundamental_signal
        )

        return {
            'direction': direction,
            'confidence': round(confidence, 4),
            'regime': regime,
            'modality_weights': {
                'price': round(price_w, 4),
                'sentiment': round(sentiment_w, 4),
                'fundamental': round(fundamental_w, 4),
            },
            'modalities': {
                'price': {
                    'direction': price_signal.get('direction'),
                    'confidence': price_signal.get('confidence'),
                    'score_contribution': round(scores.get('price', 0), 4),
                },
                'sentiment': {
                    'direction': sentiment_signal.get('direction'),
                    'confidence': sentiment_signal.get('confidence'),
                    'score_contribution': round(scores.get('sentiment', 0), 4),
                },
                'fundamental': {
                    'direction': fundamental_signal.get('direction'),
                    'confidence': fundamental_signal.get('confidence'),
                    'score_contribution': round(scores.get('fundamental', 0), 4),
                },
            },
            'scores': {k: round(v, 4) for k, v in scores.items()},
            'consensus': round(consensus, 4),
            'timestamp': datetime.now().isoformat(),
        }

    @staticmethod
    def _calculate_consensus(
        price_signal: Dict,
        sentiment_signal: Dict,
        fundamental_signal: Dict,
    ) -> float:
        """
        计算模态间一致度

        0 = 完全分歧 (三个方向都不同), 1 = 完全一致 (三个方向相同)
        """
        directions = [
            price_signal.get('direction', 'neutral'),
            sentiment_signal.get('direction', 'neutral'),
            fundamental_signal.get('direction', 'neutral'),
        ]

        if len(set(directions)) == 1:
            return 1.0  # 完全一致
        elif len(set(directions)) == 2:
            return 0.5  # 两个一致
        else:
            return 0.0  # 完全分歧


# ── 主融合器 ─────────────────────────────────────────────────

class MultiModalFusion:
    """
    多模态融合预测器

    整合价格、情感、基本面三类信号，基于 regime 动态权重融合。
    """

    def __init__(self):
        self._price_extractor = TechnicalSignalExtractor()
        self._sentiment_extractor = SentimentSignalExtractor()
        self._fundamental_extractor = FundamentalSignalExtractor()
        self._fusion_engine = FusionEngine()
        self._current_regime: str = 'sideways'
        self._history: List[Dict] = []
        # IC 自适应权重
        self._ic_tracker = ModalityICTracker()
        self._fusion_engine.set_ic_tracker(self._ic_tracker)
        # 评估历史
        self._eval_history: List[Dict] = []
        # 2026-09-22 预测评估面板接通: SQLite 台账恢复 (重启不清零, scan 链同宗)
        try:
            from modules.fusion_ledger import get_fusion_ledger
            ledger = get_fusion_ledger()
            if ledger:
                self._eval_history = ledger.load_evals()
        except Exception as e:
            logger.debug(f"[MultiModalFusion] 台账恢复跳过: {e}")

    def predict(
        self,
        stock_code: str,
        regime: str = 'sideways',
        use_cache: bool = True,
    ) -> Dict:
        """
        执行多模态融合预测

        Args:
            stock_code: 股票代码 (如 'sz300620' 或 'sz300620')
            regime: 市场状态 (自动检测时传 None)
            use_cache: 是否使用缓存

        Returns:
            融合预测结果
        """
        # 尝试缓存
        if use_cache:
            cache_key = f"fusion_{stock_code}_{regime}"
            cached = cache.get(cache_key, category='fusion')
            if cached:
                return cached

        # 标准化 stock_code
        if not stock_code.startswith(('sh', 'sz')):
            full_code = f"sz{stock_code}" if stock_code.startswith('3') or stock_code.startswith('0') else f"sh{stock_code}"
        else:
            full_code = stock_code

        # 提取股票代码 (不带前缀) 和名称
        code_only = full_code[2:]  # 去掉 sh/sz 前缀

        # 获取股票数据
        try:
            from modules.data_fetcher import StockDataFetcher
            fetcher = StockDataFetcher()
            stock_data = fetcher.get_stock_info(full_code) or {}
            klines = fetcher.get_kline_data(full_code, 'daily', 120) or []
        except Exception as e:
            logger.warning(f"[MultiModalFusion] 数据获取失败: {e}")
            stock_data = {}
            klines = []

        # 1. 提取技术信号
        price_signal = self._price_extractor.extract(klines)

        # 2. 提取情感信号
        stock_name = stock_data.get('name', '')
        sentiment_signal = self._sentiment_extractor.extract(full_code)

        # 3. 提取基本面信号
        fundamental_signal = self._fundamental_extractor.extract(code_only, stock_name)

        # 4. 融合
        result = self._fusion_engine.fuse(
            price_signal, sentiment_signal, fundamental_signal, regime
        )

        # 添加股票信息
        result['stock_code'] = full_code
        result['stock_name'] = stock_name

        # 记录历史
        self._history.append(result)
        if len(self._history) > 100:
            self._history = self._history[-100:]

        # 缓存 (TTL 30 秒)
        if use_cache:
            cache.set(cache_key, result, category='fusion', ttl=30)

        logger.info(
            f"[MultiModalFusion] 融合预测: {full_code} "
            f"→ {result['direction']} (confidence={result['confidence']}, "
            f"regime={regime}, consensus={result['consensus']})"
        )

        # 种子落盘 (2026-09-22 接通): ≥3 交易日后 23:10 replay 对账依赖种子持久
        try:
            from modules.fusion_ledger import get_fusion_ledger
            ledger = get_fusion_ledger()
            if ledger:
                ledger.append_seed(result)
        except Exception as e:
            logger.debug(f"[MultiModalFusion] 种子落盘跳过: {e}")

        return result

    def record_evaluation(self, result: Dict, actual_return: float) -> Dict:
        """
        评估单次融合预测的准确性 (延迟评估，实际收益次日才能知道)。

        Args:
            result: predict() 返回的结果
            actual_return: 实际收益率 (如次日涨跌幅)

        Returns:
            {correct: bool, brier: float, pred_direction, actual_return}
        """
        pred_direction = result.get('direction', 'neutral')
        pred_confidence = result.get('confidence', 0.5)

        # 方向准确率
        pred_numeric = 1.0 if pred_direction == 'up' else (-1.0 if pred_direction == 'down' else 0.0)
        actual_direction = 1.0 if actual_return > 0.001 else (-1.0 if actual_return < -0.001 else 0.0)
        correct = (pred_numeric * actual_direction) > 0  # 同号即正确

        # Brier score (将 direction 映射为概率)
        if pred_direction == 'up':
            prob_up = pred_confidence
            prob_down = 1 - pred_confidence
        elif pred_direction == 'down':
            prob_up = 1 - pred_confidence
            prob_down = pred_confidence
        else:
            prob_up = 0.5
            prob_down = 0.5

        actual_binary = 1.0 if actual_direction > 0 else 0.0
        brier = (prob_up - actual_binary) ** 2 + (prob_down - (1 - actual_binary)) ** 2

        # 记录到评估历史
        eval_entry = {
            'correct': correct,
            'brier': float(brier),
            'pred_direction': pred_direction,
            'actual_return': actual_return,
            'timestamp': datetime.now().isoformat(),
        }
        self._eval_history.append(eval_entry)
        if len(self._eval_history) > 500:
            self._eval_history = self._eval_history[-500:]

        # 漂移检测: 最近 50 条 vs 之前 50 条准确率
        drift_alert = False
        if len(self._eval_history) >= 100:
            recent = self._eval_history[-50:]
            older = self._eval_history[-100:-50]
            recent_acc = sum(1 for e in recent if e['correct']) / len(recent)
            older_acc = sum(1 for e in older if e['correct']) / len(older)
            if older_acc - recent_acc > 0.15:  # 准确率下降 > 15%
                logger.warning(
                    f"[MultiModalFusion] 预测准确率漂移! "
                    f"近期={recent_acc:.2%} vs 前期={older_acc:.2%}"
                )
                drift_alert = True

        return {
            'correct': correct,
            'brier': round(float(brier), 4),
            'pred_direction': pred_direction,
            'actual_return': round(float(actual_return), 6),
            'drift_alert': drift_alert,
        }

    def compare(self, stock_code: str) -> Dict:
        """
        对比各模态独立信号

        Args:
            stock_code: 股票代码

        Returns:
            各模态独立信号对比
        """
        if not stock_code.startswith(('sh', 'sz')):
            full_code = f"sz{stock_code}" if stock_code.startswith('3') or stock_code.startswith('0') else f"sh{stock_code}"
        else:
            full_code = stock_code

        code_only = full_code[2:]

        try:
            from modules.data_fetcher import StockDataFetcher
            fetcher = StockDataFetcher()
            stock_data = fetcher.get_stock_info(full_code) or {}
            klines = fetcher.get_kline_data(full_code, 'daily', 120) or []
        except Exception as e:
            logger.warning(f"[MultiModalFusion] 数据获取失败: {e}")
            klines = []
            stock_data = {}

        stock_name = stock_data.get('name', '')

        return {
            'stock_code': full_code,
            'stock_name': stock_name,
            'modalities': {
                'price': self._price_extractor.extract(klines),
                'sentiment': self._sentiment_extractor.extract(full_code),
                'fundamental': self._fundamental_extractor.extract(code_only, stock_name),
            },
            'regime_weights': dict(REGIME_MODALITY_WEIGHTS),
            'timestamp': datetime.now().isoformat(),
        }

    def get_status(self) -> Dict:
        """获取融合器状态"""
        return {
            'current_regime': self._current_regime,
            'history_length': len(self._history),
            'regime_weights': REGIME_MODALITY_WEIGHTS,
            'modalities': ['price', 'sentiment', 'fundamental'],
            'timestamp': datetime.now().isoformat(),
        }

    def update_regime(self, regime: str):
        """更新当前市场状态"""
        if regime in REGIMES:
            self._current_regime = regime
            logger.info(f"[MultiModalFusion] Regime 更新: {regime}")


# ── 全局单例 ─────────────────────────────────────────────────

multi_modal_fusion = MultiModalFusion()


def get_multi_modal_fusion() -> MultiModalFusion:
    """获取全局 MultiModalFusion 实例"""
    return multi_modal_fusion
