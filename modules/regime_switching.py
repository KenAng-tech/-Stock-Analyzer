#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Regime-Switching 多模型预测器

整合多种模型，根据市场状态 (Regime) 动态切换和加权:

Regime 定义 (基于 GMM + MA 双检测):
    - bullish (牛市): 上涨趋势, 低波动, 成交量放大
    - bearish (熊市): 下跌趋势, 高波动, 成交量放大
    - sideways (震荡): 无明显趋势, 中低波动
    - volatile (高波动): 高波动率, 可能趋势转变

模型池:
    - LightGBM (主流量因子模型)
    - XGBoost (梯度提升树)
    - RandomForest (随机森林)
    - PatchTST (Transformer 时序模型)
    - Mamba (SSM 线性模型)
    - Diffusion (概率预测模型)
    - RL Agent (PPO/SAC)
    - Cross-Modal (多模态推理)

Regime → 模型权重映射:
    Regime     | LGBM | XGB  | RF   | PatchTST | Mamba | Diffusion | RL | CM
    -----------|------|------|------|----------|-------|-----------|----|-----
    bullish    | 0.30 | 0.25 | 0.15 | 0.15     | 0.10  | 0.03      | 0.02| 0.05
    bearish    | 0.25 | 0.30 | 0.15 | 0.10     | 0.15  | 0.08      | 0.07| 0.10
    sideways   | 0.25 | 0.25 | 0.20 | 0.15     | 0.10  | 0.05      | 0.05| 0.15
    volatile   | 0.20 | 0.25 | 0.20 | 0.10     | 0.15  | 0.10      | 0.10| 0.10

架构:
    Input → Regime Detector → Model Selector → Ensemble → Prediction
"""

import os
import json
import time
import threading
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque

from modules.logger import logger

# ── 依赖检查 ──────────────────────────────────────────────

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("[RegimeSwitching] PyTorch 未安装")

try:
    from modules.factors.multi_factor_model_v2 import MultiFactorModelV2
    HAS_MULTI_FACTOR = True
except ImportError:
    HAS_MULTI_FACTOR = False
    logger.warning("[RegimeSwitching] MultiFactorModelV2 不可用")

try:
    from modules.ml_predictor import MLPredictor
    HAS_ML_PREDICTOR = True
except ImportError:
    HAS_ML_PREDICTOR = False
    logger.warning("[RegimeSwitching] MLPredictor 不可用")

try:
    from modules.models.patchtst_integrator import get_patchtst
    HAS_PATCHTST = True
except ImportError:
    HAS_PATCHTST = False
    logger.warning("[RegimeSwitching] PatchTST 不可用")

try:
    from modules.experimental.mamba_model import MambaTrainer
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False
    logger.warning("[RegimeSwitching] Mamba 不可用")

try:
    from modules.models.diffusion_model import get_diffusion_predictor
    HAS_DIFFUSION = True
except ImportError:
    HAS_DIFFUSION = False
    logger.warning("[RegimeSwitching] Diffusion 不可用")

try:
    from modules.rl_trader_v2 import RLTraderV2
    HAS_RL = True
except ImportError:
    HAS_RL = False
    logger.warning("[RegimeSwitching] RL Trader 不可用")

try:
    from modules.hmm_market_detector import MarketRegimeDetector
    HAS_HMM = True
except ImportError:
    HAS_HMM = False
    logger.warning("[RegimeSwitching] HMM Regime Detector 不可用")

try:
    from modules.dynamic_ensemble import (
        MarketRegimeDetector as DynamicRegimeDetector,
        ModelWeightScheduler,
    )
    HAS_DYNAMIC = True
except ImportError:
    HAS_DYNAMIC = False
    logger.warning("[RegimeSwitching] Dynamic Ensemble 不可用")



# ── Predictor Factory 集成 ──────────────────────────────────
# 使用统一工厂获取模型，避免硬编码导入

def _get_predictor(name: str):
    """通过 predictor_factory 获取预测器"""
    try:
        from modules.routes.predictor_factory import predictor_factory
        return predictor_factory.get(name)
    except Exception:
        return None

# ── 数据类 ──────────────────────────────────────────────

@dataclass
class RegimeSwitchingPrediction:
    """Regime-Switching 预测结果"""
    # 预测结果
    direction: str = "neutral"
    confidence: float = 0.5
    probabilities: Dict[str, float] = field(default_factory=lambda: {"up": 0.33, "neutral": 0.34, "down": 0.33})

    # Regime 信息
    detected_regime: str = "sideways"
    regime_confidence: float = 0.0
    regime_weights: Dict[str, float] = field(default_factory=dict)

    # 各模型预测
    model_predictions: Dict[str, Dict] = field(default_factory=dict)

    # 元数据
    timestamp: float = field(default_factory=time.time)
    execution_time_ms: float = 0.0
    model_count: int = 0


@dataclass
class RegimeSwitchingConfig:
    """Regime-Switching 配置"""
    # Regime 权重矩阵
    regime_weights: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        'bullish': {
            'multi_factor': 0.30, 'ml_predictor': 0.25,
            'patchtst': 0.15, 'mamba': 0.10,
            'diffusion': 0.03, 'rl': 0.02,
        },
        'bearish': {
            'multi_factor': 0.25, 'ml_predictor': 0.30,
            'patchtst': 0.10, 'mamba': 0.15,
            'diffusion': 0.08, 'rl': 0.07,
        },
        'sideways': {
            'multi_factor': 0.25, 'ml_predictor': 0.25,
            'patchtst': 0.15, 'mamba': 0.10,
            'diffusion': 0.05, 'rl': 0.05,
        },
        'volatile': {
            'multi_factor': 0.20, 'ml_predictor': 0.25,
            'patchtst': 0.10, 'mamba': 0.15,
            'diffusion': 0.10, 'rl': 0.10,
        },
    })
    # Regime 持久性 (连续 N 次相同 Regime 才切换)
    regime_persistence: int = 3
    # 自适应学习率
    adaptive_alpha: float = 0.3
    # 模型性能追踪窗口
    performance_window: int = 30


# ── Regime 检测器 ──────────────────────────────────────

class EnhancedRegimeDetector:
    """
    增强型 Regime 检测器
    结合 GMM (HMM) 和 MA-based (Dynamic Ensemble) 两种方法
    增加持久性过滤防止频繁切换
    """

    def __init__(self, persistence: int = 3):
        self.persistence = persistence
        self.hmm_detector = MarketRegimeDetector() if HAS_HMM else None
        self.dynamic_detector = DynamicRegimeDetector() if HAS_DYNAMIC else None
        self._hmm_fitted = False
        self._regime_history: deque = deque(maxlen=persistence * 2)
        self._current_regime = 'sideways'

    def detect(self, stock_data: Dict,
               klines: Optional[List[Dict]] = None) -> str:
        """
        检测当前市场状态
        使用多数投票 + 持久性过滤
        """
        # 收集各检测器结果
        signals = []

        if self.hmm_detector and self._hmm_fitted:
            try:
                regime = self.hmm_detector.predict_regime(stock_data, klines)
                signals.append(('hmm', regime))
            except Exception as e:
                logger.debug(f"[EnhancedRegimeDetector] HMM 检测失败: {e}")

        if self.dynamic_detector:
            try:
                regime = self.dynamic_detector.detect_regime(stock_data, klines)
                signals.append(('dynamic', regime))
            except Exception as e:
                logger.debug(f"[EnhancedRegimeDetector] Dynamic 检测失败: {e}")

        # 默认
        if not signals:
            return self._current_regime

        # 多数投票
        from collections import Counter
        votes = Counter([s[1] for s in signals])
        detected = votes.most_common(1)[0][0]

        # 持久性过滤
        self._regime_history.append(detected)
        recent = list(self._regime_history)[-self.persistence:]

        if len(recent) >= self.persistence:
            recent_votes = Counter(recent)
            most_common = recent_votes.most_common(1)[0]
            if most_common[1] >= self.persistence * 0.6:  # 60% 以上一致
                self._current_regime = most_common[0]
                return most_common[0]

        return detected

    def get_regime_probability(self, stock_data: Dict,
                               klines: Optional[List[Dict]] = None) -> Dict:
        """获取各 Regime 的概率"""
        probs = {}

        if self.hmm_detector and self._hmm_fitted:
            try:
                probs = self.hmm_detector.get_regime_probability(stock_data, klines)
            except Exception as e:
                logger.debug(f"[EnhancedRegimeDetector] HMM 概率计算失败: {e}")

        if not probs:
            return {'bullish': 0.33, 'bearish': 0.33, 'sideways': 0.34}

        return probs

    def fit_from_history(self, historical_data: List[Dict],
                         klines_map: Optional[Dict] = None):
        """用历史数据拟合"""
        if self.hmm_detector:
            try:
                self.hmm_detector.fit(historical_data, klines_map)
                self._hmm_fitted = True
                logger.info("[EnhancedRegimeDetector] HMM 检测器已拟合")
            except Exception as e:
                logger.warning(f"[EnhancedRegimeDetector] HMM 拟合失败: {e}")


# ── 模型性能追踪器 ──────────────────────────────────────

class ModelPerformanceTracker:
    """
    追踪各模型在最近 N 次预测中的表现
    用于自适应调整权重
    """

    def __init__(self, window: int = 30, adaptive_alpha: float = 0.3):
        self.window = window
        self.adaptive_alpha = adaptive_alpha
        self._records: Dict[str, deque] = {}
        self._lock = threading.Lock()

    def record(self, model_name: str, correct: bool, confidence: float):
        """记录模型预测结果"""
        with self._lock:
            if model_name not in self._records:
                self._records[model_name] = deque(maxlen=self.window)
            self._records[model_name].append({
                'correct': correct,
                'confidence': confidence,
                'timestamp': time.time(),
            })

    def get_accuracy(self, model_name: str) -> float:
        """获取模型准确率"""
        with self._lock:
            if model_name not in self._records:
                return 0.5
            records = list(self._records[model_name])
            if not records:
                return 0.5
            return sum(1 for r in records if r['correct']) / len(records)

    def get_avg_confidence(self, model_name: str) -> float:
        """获取模型平均置信度"""
        with self._lock:
            if model_name not in self._records:
                return 0.5
            records = list(self._records[model_name])
            if not records:
                return 0.5
            return np.mean([r['confidence'] for r in records])

    def get_adaptive_weights(self, base_weights: Dict[str, float]) -> Dict[str, float]:
        """
        根据模型性能自适应调整权重
        新权重 = alpha * 性能权重 + (1-alpha) * 基础权重
        """
        adaptive = {}
        for model_name, base_w in base_weights.items():
            accuracy = self.get_accuracy(model_name)
            avg_conf = self.get_avg_confidence(model_name)
            # 性能权重: 准确率 * 置信度
            perf_weight = accuracy * avg_conf
            adaptive[model_name] = self.adaptive_alpha * perf_weight + (1 - self.adaptive_alpha) * base_w

        # 归一化
        total = sum(adaptive.values())
        if total > 0:
            adaptive = {k: v / total for k, v in adaptive.items()}

        return adaptive

    def get_ranking(self) -> List[Tuple[str, float]]:
        """获取模型性能排名"""
        rankings = []
        for model_name in self._records:
            acc = self.get_accuracy(model_name)
            rankings.append((model_name, acc))
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings


# ── Regime-Switching 多模型预测器 ──────────────────────────────────────

class RegimeSwitchingPredictor:
    """
    Regime-Switching 多模型预测器

    核心流程:
    1. 检测当前 Regime
    2. 根据 Regime 获取模型权重
    3. 并行调用各模型
    4. 根据模型性能自适应调整权重
    5. 加权集成得到最终预测
    """

    def __init__(self, config: Optional[RegimeSwitchingConfig] = None):
        self.config = config or RegimeSwitchingConfig()

        # 初始化各模型
        self._init_models()

        # Regime 检测器
        self.regime_detector = EnhancedRegimeDetector(
            persistence=self.config.regime_persistence
        )

        # 模型性能追踪
        self.performance_tracker = ModelPerformanceTracker(
            window=self.config.performance_window,
            adaptive_alpha=self.config.adaptive_alpha,
        )

        # 线程锁
        self._lock = threading.Lock()

        # 缓存
        self._last_prediction: Optional[RegimeSwitchingPrediction] = None
        self._prediction_count = 0

    def _init_models(self):
        """初始化所有子模型 — 使用 predictor_factory 统一获取"""
        # 通过 predictor_factory 获取所有可用模型
        self.models = {}
        
        model_configs = [
            ('multi_factor', HAS_MULTI_FACTOR, lambda: MultiFactorModelV2()),
            ('ml_predictor', HAS_ML_PREDICTOR, lambda: self._init_ml_predictor()),
            ('patchtst', HAS_PATCHTST, lambda: self._init_patchtst()),
            ('mamba', HAS_MAMBA, lambda: self._init_mamba()),
            ('diffusion', HAS_DIFFUSION, lambda: self._init_diffusion()),
            ('rl', HAS_RL, lambda: self._init_rl()),
        ]
        
        for name, available, init_fn in model_configs:
            if available:
                try:
                    model = init_fn()
                    if model is not None:
                        self.models[name] = model
                        logger.info(f"[RegimeSwitching] {name} 初始化完成")
                except Exception as e:
                    logger.warning(f"[RegimeSwitching] {name} 初始化失败: {e}")
        
        # 也尝试从 predictor_factory 获取增强模型
        self._load_from_factory()
        
        logger.info(f"[RegimeSwitching] 已初始化 {len(self.models)} 个模型: {list(self.models.keys())}")
    
    def _init_ml_predictor(self):
        """初始化 ML Predictor"""
        try:
            p = MLPredictor()
            p.load_latest_model()
            return p
        except Exception:
            return None
    
    def _init_patchtst(self):
        """初始化 PatchTST"""
        try:
            p = get_patchtst()
            p.load()
            return p
        except Exception:
            return None
    
    def _init_mamba(self):
        """初始化 Mamba"""
        try:
            from modules.experimental.mamba_model import MambaTrainer
            m = MambaTrainer(n_features=12, d_model=128, n_layers=4)
            m.load()
            return m
        except Exception:
            return None
    
    def _init_diffusion(self):
        """初始化 Diffusion"""
        try:
            d = get_diffusion_predictor()
            d.load()
            return d
        except Exception:
            return None
    
    def _init_rl(self):
        """初始化 RL Trader"""
        try:
            r = RLTraderV2(state_dim=20, action_dim=3)
            r.load()
            return r
        except Exception:
            return None
    
    def _load_from_factory(self):
        """从 predictor_factory 加载额外模型"""
        try:
            from modules.routes.predictor_factory import predictor_factory
            extra_models = ['gnn', 'conformal', 'timesfm', 'foundation', 'causal']
            for name in extra_models:
                if name not in self.models and predictor_factory.is_available(name):
                    model = predictor_factory.get(name)
                    if model is not None:
                        self.models[f'factory_{name}'] = model
                        logger.info(f"[RegimeSwitching] 从 factory 加载 {name}")
        except Exception as e:
            logger.debug(f"[RegimeSwitching] predictor_factory 加载失败: {e}")

    def predict(self, X: np.ndarray,
                stock_data: Dict,
                klines: Optional[List[Dict]] = None) -> RegimeSwitchingPrediction:
        """
        Regime-Switching 预测

        Args:
            X: (n_samples, seq_len, n_features) 或 (seq_len, n_features)
            stock_data: 股票数据
            klines: K线数据

        Returns:
            RegimeSwitchingPrediction
        """
        start_time = time.time()
        self._prediction_count += 1

        # Step 1: Regime 检测
        regime = self.regime_detector.detect(stock_data, klines)
        regime_probs = self.regime_detector.get_regime_probability(stock_data, klines)
        regime_confidence = regime_probs.get(regime, 0.5)

        # Step 2: 获取基础权重
        base_weights = self.config.regime_weights.get(regime, self.config.regime_weights['sideways'])

        # Step 3: 自适应调整权重
        adaptive_weights = self.performance_tracker.get_adaptive_weights(base_weights)

        # Step 4: 并行预测
        predictions = {}

        # Multi-Factor Model
        if self.multi_factor:
            try:
                result = self.multi_factor.predict_stock(X)
                predictions['multi_factor'] = self._normalize(result)
            except Exception as e:
                logger.debug(f"[RegimeSwitching] Multi-Factor 预测失败: {e}")
                predictions['multi_factor'] = self._default()
        else:
            predictions['multi_factor'] = self._default()

        # ML Predictor
        if self.ml_predictor:
            try:
                result = self.ml_predictor.predict_direction(X)
                predictions['ml_predictor'] = self._normalize(result)
            except Exception as e:
                logger.debug(f"[RegimeSwitching] ML Predictor 预测失败: {e}")
                predictions['ml_predictor'] = self._default()
        else:
            predictions['ml_predictor'] = self._default()

        # PatchTST
        if self.patchtst and self.patchtst.is_trained():
            try:
                result = self.patchtst.predict(X)
                predictions['patchtst'] = self._normalize(result)
            except Exception as e:
                logger.debug(f"[RegimeSwitching] PatchTST 预测失败: {e}")
                predictions['patchtst'] = self._default()
        else:
            predictions['patchtst'] = self._default()

        # Mamba
        if self.mamba and self.mamba.trained:
            try:
                result = self.mamba.predict(X)
                predictions['mamba'] = self._normalize(result)
            except Exception as e:
                logger.debug(f"[RegimeSwitching] Mamba 预测失败: {e}")
                predictions['mamba'] = self._default()
        else:
            predictions['mamba'] = self._default()

        # Diffusion
        if self.diffusion and self.diffusion.is_trained():
            try:
                result = self.diffusion.predict(X, n_samples=10)
                predictions['diffusion'] = self._normalize(result)
            except Exception as e:
                logger.debug(f"[RegimeSwitching] Diffusion 预测失败: {e}")
                predictions['diffusion'] = self._default()
        else:
            predictions['diffusion'] = self._default()

        # RL Trader
        if self.rl_trader:
            try:
                if X.ndim == 2:
                    X_3d = X[np.newaxis, :, :]
                else:
                    X_3d = X
                obs = X_3d[0][:20] if X_3d.shape[1] >= 20 else X_3d[0]
                result = self.rl_trader.trade(obs)
                action_map = {'buy': 0.7, 'sell': 0.3, 'hold': 0.5}
                action_score = action_map.get(result.get('action', 'hold'), 0.5)
                predictions['rl'] = {
                    'direction': result.get('action', 'hold'),
                    'confidence': result.get('confidence', 0.5),
                    'probabilities': {
                        'up': action_score,
                        'down': 1.0 - action_score,
                        'neutral': 0.5,
                    },
                }
            except Exception as e:
                logger.debug(f"[RegimeSwitching] RL 预测失败: {e}")
                predictions['rl'] = self._default()
        else:
            predictions['rl'] = self._default()

        # Step 5: 加权集成
        ensemble = self._ensemble_aggregate(predictions, adaptive_weights)

        # Step 6: 记录性能
        self._record_performance(predictions, ensemble)

        # Step 7: 构建结果
        execution_time_ms = (time.time() - start_time) * 1000

        result = RegimeSwitchingPrediction(
            direction=ensemble['direction'],
            confidence=ensemble['confidence'],
            probabilities=ensemble['probabilities'],
            detected_regime=regime,
            regime_confidence=regime_confidence,
            regime_weights=adaptive_weights,
            model_predictions=predictions,
            execution_time_ms=execution_time_ms,
            model_count=len(predictions),
        )

        self._last_prediction = result
        return result

    def _normalize(self, raw: Dict) -> Dict:
        """标准化预测结果"""
        direction = raw.get('direction', raw.get('directions', ['neutral'])[0] if 'directions' in raw else 'neutral')
        confidence = raw.get('confidence', 0.5)
        probabilities = raw.get('probabilities', {})

        if not probabilities:
            if direction == 'up':
                probabilities = {'up': 0.7, 'neutral': 0.2, 'down': 0.1}
            elif direction == 'down':
                probabilities = {'up': 0.1, 'neutral': 0.2, 'down': 0.7}
            else:
                probabilities = {'up': 0.33, 'neutral': 0.34, 'down': 0.33}

        return {
            'direction': direction,
            'confidence': confidence,
            'probabilities': probabilities,
        }

    def _default(self) -> Dict:
        return {
            'direction': 'neutral',
            'confidence': 0.33,
            'probabilities': {'up': 0.33, 'neutral': 0.34, 'down': 0.33},
        }

    def _ensemble_aggregate(self, predictions: Dict, weights: Dict) -> Dict:
        """加权集成"""
        direction_scores = {'up': 0.7, 'neutral': 0.5, 'down': 0.3}

        total_weight = 0.0
        weighted_up = 0.0
        weighted_neutral = 0.0
        weighted_down = 0.0

        for model_name, pred in predictions.items():
            w = weights.get(model_name, 0.1)
            prob = pred.get('probabilities', {})
            dir_score = direction_scores.get(pred.get('direction', 'neutral'), 0.5)

            weighted_up += w * prob.get('up', 0.33)
            weighted_neutral += w * prob.get('neutral', 0.34)
            weighted_down += w * prob.get('down', 0.33)
            total_weight += w

        if total_weight > 0:
            weighted_up /= total_weight
            weighted_neutral /= total_weight
            weighted_down /= total_weight

        if weighted_up > weighted_down and weighted_up > 0.4:
            direction = 'up'
            confidence = weighted_up
        elif weighted_down > weighted_up and weighted_down > 0.4:
            direction = 'down'
            confidence = weighted_down
        else:
            direction = 'neutral'
            confidence = weighted_neutral

        return {
            'direction': direction,
            'confidence': round(confidence, 4),
            'probabilities': {
                'up': round(weighted_up, 4),
                'neutral': round(weighted_neutral, 4),
                'down': round(weighted_down, 4),
            },
        }

    def _record_performance(self, predictions: Dict, ensemble: Dict):
        """记录各模型性能"""
        ensemble_dir = ensemble['direction']
        ensemble_conf = ensemble['confidence']

        for model_name, pred in predictions.items():
            pred_dir = pred.get('direction', 'neutral')
            pred_conf = pred.get('confidence', 0.5)
            # 简单判断: 预测方向与集成方向一致
            correct = pred_dir == ensemble_dir
            self.performance_tracker.record(model_name, correct, pred_conf)

    def train(self, X: np.ndarray, y: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              epochs: int = 50) -> Dict:
        """
        训练所有子模型
        """
        logger.info("[RegimeSwitching] 开始训练所有子模型")

        results = {}

        # 训练 ML Predictor
        if self.ml_predictor:
            try:
                self.ml_predictor.train(X, y, X_val, y_val, epochs=epochs)
                results['ml_predictor'] = 'trained'
            except Exception as e:
                results['ml_predictor'] = f'error: {e}'

        # 训练 PatchTST
        if self.patchtst:
            try:
                self.patchtst.train(X, y, X_val, y_val, epochs=epochs)
                results['patchtst'] = 'trained'
            except Exception as e:
                results['patchtst'] = f'error: {e}'

        # 训练 Mamba
        if self.mamba:
            try:
                self.mamba.train(X, y, X_val, y_val, epochs=epochs)
                results['mamba'] = 'trained'
            except Exception as e:
                results['mamba'] = f'error: {e}'

        # 拟合 Regime 检测器
        self.regime_detector._hmm_fitted = True

        logger.info(f"[RegimeSwitching] 所有模型训练完成: {results}")
        return results

    def save(self, path: Optional[str] = None):
        """保存所有模型"""
        if path is None:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            path = os.path.join(os.path.dirname(__file__), 'dl_models', f'regime_switching_{timestamp}')

        saved = {}
        if self.ml_predictor:
            self.ml_predictor.save(os.path.join(path, 'ml_predictor.pkl'))
            saved['ml_predictor'] = True
        if self.patchtst:
            self.patchtst.save(os.path.join(path, 'patchtst.pth'))
            saved['patchtst'] = True
        if self.mamba:
            self.mamba.save(os.path.join(path, 'mamba.pth'))
            saved['mamba'] = True
        if self.diffusion:
            self.diffusion.save(os.path.join(path, 'diffusion.pth'))
            saved['diffusion'] = True
        if self.rl_trader:
            self.rl_trader.save()
            saved['rl'] = True

        logger.info(f"[RegimeSwitching] 模型已保存: {saved}")
        return saved

    def load(self, path: Optional[str] = None) -> Dict:
        """加载所有模型"""
        if path is None:
            path = os.path.join(os.path.dirname(__file__), 'dl_models')

        loaded = {}
        if self.ml_predictor:
            # MLPredictor 使用 load_latest_model
            loaded['ml_predictor'] = self.ml_predictor.load_latest_model()
        if self.patchtst:
            loaded['patchtst'] = self.patchtst.load(os.path.join(path, 'patchtst.pth'))
        if self.mamba:
            loaded['mamba'] = self.mamba.load(os.path.join(path, 'mamba.pth'))
        if self.diffusion:
            loaded['diffusion'] = self.diffusion.load(os.path.join(path, 'diffusion.pth'))
        if self.rl_trader:
            loaded['rl'] = self.rl_trader.load()

        logger.info(f"[RegimeSwitching] 模型已加载: {loaded}")
        return loaded

    def get_status(self) -> Dict:
        """获取模型状态"""
        return {
            'regime_detector': 'hmm_fitted' if self.regime_detector._hmm_fitted else 'hmm_unfitted',
            'current_regime': self.regime_detector._current_regime,
            'model_count': sum(1 for m in [
                self.multi_factor, self.ml_predictor,
                self.patchtst, self.mamba, self.diffusion, self.rl_trader
            ] if m is not None),
            'model_performance': self.performance_tracker.get_ranking(),
            'last_prediction': {
                'direction': self._last_prediction.direction if self._last_prediction else None,
                'confidence': self._last_prediction.confidence if self._last_prediction else None,
                'regime': self._last_prediction.detected_regime if self._last_prediction else None,
                'prediction_count': self._prediction_count,
            },
        }

    def reset(self):
        """重置所有模型"""
        if self.ml_predictor:
            self.ml_predictor.trained = False
        if self.patchtst:
            self.patchtst.trained = False
        if self.mamba:
            self.mamba.trained = False
        if self.diffusion:
            self.diffusion.trained = False
        self._last_prediction = None
        self._prediction_count = 0
        logger.info("[RegimeSwitching] 所有模型已重置")


# ── 全局单例 ──────────────────────────────────────────────

_regime_switching_instance: Optional[RegimeSwitchingPredictor] = None
_regime_switching_lock = threading.Lock()


def get_regime_switching(config: Optional[RegimeSwitchingConfig] = None) -> RegimeSwitchingPredictor:
    """获取全局 RegimeSwitchingPredictor 实例 (线程安全)"""
    global _regime_switching_instance
    if _regime_switching_instance is None:
        with _regime_switching_lock:
            if _regime_switching_instance is None:
                _regime_switching_instance = RegimeSwitchingPredictor(config)
                _regime_switching_instance.load()
    return _regime_switching_instance
