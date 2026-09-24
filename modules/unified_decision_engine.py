#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
统一决策引擎 — 加权集成 18 个 SOTA 模型

集成以下模型预测:
1. SOTA 决策引擎 (内部集成)
2. 多因子模型 V2
3. ITransformer (PatchTST 变体)
4. Mamba/SSM
5. Diffusion 模型
6. Self-Supervised 模型
7. DRL (PPO)
8. Moirai 基础模型
9. Conformal 预测
10. GNN 图神经网络
11. Alpha158 因子
12. Time-LLM
13. Regime-Switching
14. CVaR 风险模型
15. 跨市场融合
16. 多智能体
17. TimesFM
18. 动态权重

权重策略:
- 初始: 等权重
- 更新: 基于最近 N 日预测准确率 (IC/Directional Accuracy)
- 衰减: 指数加权移动平均 (EWMA), 近期表现权重更高

用法:
    engine = UnifiedDecisionEngine()
    decision = engine.decide('sz300620')
"""

import os
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from modules.logger import logger
from modules.tradability import is_tradable

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# P0: 漂移感知集成
try:
    from modules.drift_aware_ensemble import DriftAwareEnsemble, ModelDriftMonitor
    HAS_DRIFT_AWARE = True
except Exception:
    HAS_DRIFT_AWARE = False
    logger.warning("[UnifiedDecision] DriftAwareEnsemble 加载失败")


class Direction(Enum):
    """预测方向"""
    STRONG_BUY = 'strong_buy'
    BUY = 'buy'
    NEUTRAL = 'neutral'
    SELL = 'sell'
    STRONG_SELL = 'strong_sell'


@dataclass
class UnifiedDecision:
    """统一交易决策"""
    direction: Direction = Direction.NEUTRAL
    confidence: float = 0.0
    consensus: float = 0.0       # 共识度 (0-1, 越高说明模型间越一致)
    n_models: int = 0            # 参与决策的模型数
    weighted_score: float = 0.0  # 加权得分 [-1, 1]
    model_votes: Dict[str, str] = field(default_factory=dict)  # 各模型投票
    model_weights: Dict[str, float] = field(default_factory=dict)  # 各模型权重
    buy_count: int = 0
    sell_count: int = 0
    neutral_count: int = 0
    timestamp: str = ''
    stock_code: str = ''
    reason: str = ''


def dropout_decide(holdings_scores: Dict[str, Dict],
                   candidate_scores: Dict[str, float],
                   n_drop: int = 1, hold_thresh: int = 2,
                   market_data: Optional[Dict[str, Dict]] = None) -> Dict:
    """
    qlib TopkDropoutStrategy 防抖语义 (P1-10, 2026-09-14, 纯函数)。

    背景: 朴素「评分低于最优候选即换仓」在信号抖动时会频繁换股 (换手率
    爆炸 + 交易成本)。qlib 的对策: 持仓只有跌出 (持仓∪候选) 合并榜倒数
    n_drop 名才卖, 且需已持有 ≥ hold_thresh 天; 卖出几个名额就按合并榜
    从高到低补几个非持仓候选, 换入前过 tradability (涨跌停/停牌)。

    Args:
        holdings_scores: {code: {'score': float, 'days_held': int}} 当前持仓
        candidate_scores: {code: float} 候选池评分
        n_drop: 合并榜末位淘汰名额数
        hold_thresh: 最短持有天数 (未满不卖, 防抖动)
        market_data: 可选 {code: {'pct_change','volume','name'}} —
            提供时对换入候选过 modules.tradability.is_tradable, 不可成交跳过

    Returns:
        {'sells': [code], 'buys': [code], 'holds': [code],
         'ranking': [(code, score) 降序], 'notes': [str]}
    """
    notes: List[str] = []
    merged = sorted(
        [(c, float(info.get('score', 0.0))) for c, info in holdings_scores.items()]
        + [(c, float(s)) for c, s in candidate_scores.items()],
        key=lambda x: -x[1])
    held = set(holdings_scores)
    bottom = {c for c, _ in merged[-n_drop:]} if (merged and n_drop > 0) else set()

    sells: List[str] = []
    for code, info in holdings_scores.items():
        if code not in bottom:
            continue
        days = int(info.get('days_held', 0) or 0)
        if days >= hold_thresh:
            sells.append(code)
        else:
            notes.append(f'{code} 跌出合并榜末{n_drop}但持有 {days}d < '
                         f'{hold_thresh}d → 不换 (防抖)')

    # 卖出几个名额 → 从合并榜高分端补几个非持仓候选 (tradability 门控)
    buys: List[str] = []
    slots = len(sells)
    if slots:
        for code, _ in merged:
            if slots <= 0:
                break
            if code in held:
                continue
            md = (market_data or {}).get(code)
            if md is not None:
                ok, reason = is_tradable(
                    code, md.get('pct_change', 0.0), 'buy',
                    volume=md.get('volume'), name=md.get('name'))
                if not ok:
                    notes.append(f'{code} 换入被 tradability 拦截: {reason}')
                    continue
            buys.append(code)
            slots -= 1

    return {
        'sells': sells,
        'buys': buys,
        'holds': [c for c in holdings_scores if c not in sells],
        'ranking': merged,
        'notes': notes,
    }


class UnifiedDecisionEngine:
    """统一决策引擎 — 加权集成所有 SOTA 模型"""

    # 所有可用模型
    AVAILABLE_MODELS = [
        'multi_factor', 'itransformer', 'mamba', 'diffusion',
        'self_supervised', 'drl', 'moirai', 'conformal',
        'gnn', 'alpha158', 'time_llm', 'regime_switching',
        'cvar', 'cross_market', 'multi_agent', 'timesfm',
        'sota_decision', 'dynamic_weights',
    ]

    # 默认权重 (等权重, 保留用于重置)
    DEFAULT_WEIGHTS = {m: 1.0 / 18.0 for m in AVAILABLE_MODELS}

    # 2026 SOTA 权重配置 (基于 IC-IR 和实证研究)
    # Mamba 在金融时序上的表现被多项 2025 研究证明不如 Transformer
    # 因此 Mamba 权重降至 0.5x，Transformer 系列权重提升
    BASE_WEIGHTS = {
        'multi_factor': 1.0,           # 基础因子模型
        'itransformer': 1.5,           # ITransformer (ICLR 2024 Spotlight)
        'mamba': 0.5,                  # Mamba 降权 (金融时序表现 < Transformer)
        'diffusion': 1.0,              # Diffusion 模型
        'self_supervised': 1.2,        # 自监督预训练
        'drl': 1.0,                    # DRL (PPO)
        'moirai': 1.5,                 # Moirai (ICLR 2024 基础模型)
        'conformal': 1.0,              # Conformal Prediction (不确定性量化)
        'gnn': 1.0,                    # GNN 图神经网络
        'alpha158': 1.2,               # Alpha158 因子
        'time_llm': 0.8,               # Time-LLM (降低权重，可能超时)
        'regime_switching': 1.0,       # Regime-Switching
        'cvar': 0.8,                   # CVaR 风险模型 (偏保守)
        'cross_market': 1.0,           # 跨市场融合
        'multi_agent': 1.0,            # 多智能体
        'timesfm': 1.5,                # TimesFM (Google 基础模型)
        'sota_decision': 1.0,          # SOTA 决策引擎
        'dynamic_weights': 1.0,        # 动态权重
    }

    # 模型预测到方向/得分的映射
    # 每个模型返回不同的格式，需要统一解析
    MODEL_PARSERS = {
        'multi_factor': '_parse_multi_factor',
        'itransformer': '_parse_direction',
        'mamba': '_parse_direction',
        'diffusion': '_parse_direction',
        'self_supervised': '_parse_direction',
        'drl': '_parse_drl',
        'moirai': '_parse_direction',
        'conformal': '_parse_conformal',
        'gnn': '_parse_direction',
        'alpha158': '_parse_alpha158',
        'time_llm': '_parse_direction',
        'regime_switching': '_parse_regime',
        'cvar': '_parse_cvar',
        'cross_market': '_parse_cross_market',
        'multi_agent': '_parse_direction',
        'timesfm': '_parse_timesfm',
        'sota_decision': '_parse_sota_decision',
        'dynamic_weights': '_parse_direction',
    }

    def __init__(self,
                 lookback: int = 60,
                 ewma_alpha: float = 0.1,
                 min_consensus: float = 0.6,
                 min_models: int = 3,
                 decision_dropout_mode: str = 'observe'):
        """
        Args:
            lookback: 回溯天数 (用于计算准确率)
            ewma_alpha: EWMA 衰减因子 (0-1, 越大近期权重越高)
            min_consensus: 最小共识度阈值
            min_models: 最小模型参与数
            decision_dropout_mode: P1-10 防抖观察 flag —
                'observe' (默认) observe_dropout() 只 logger.info + 写
                runs/consensus_ab/dropout_observe.jsonl, 不改任何实际输出;
                'off' 时 observe_dropout() 直接返回 None。
        """
        self.lookback = lookback
        self.ewma_alpha = ewma_alpha
        self.min_consensus = min_consensus
        self.min_models = min_models
        self.decision_dropout_mode = decision_dropout_mode

        # 使用 IC-IR 加权 (初始用 BASE_WEIGHTS 作为先验)
        self.base_weights = dict(self.BASE_WEIGHTS)
        self.weights = {k: self.base_weights.get(k, 1.0) for k in self.AVAILABLE_MODELS}
        self._normalize_weights()

        # 历史准确率 (stock_code → {date → accuracy})
        self._accuracy_history: Dict[str, Dict[str, float]] = {}

        # 最近预测记录 (用于在线更新准确率)
        self._recent_predictions: Dict[str, List[Dict]] = {}

        # P1-a (2026-09-22): performative 反馈链路径 (seed 落盘 + consumed 防重喂)
        self._seed_path = os.path.join(
            PROJECT_ROOT, 'runs', 'consensus_ab', 'model_seed.jsonl')
        self._feed_path = os.path.join(
            PROJECT_ROOT, 'runs', 'consensus_ab', 'seed_consumed.jsonl')

        # 性能指标缓存
        self._metrics_path = os.path.join(
            os.path.dirname(__file__), 'dl_models', 'decision_engine_metrics.json'
        )
        self._load_metrics()

        # 2026-09-23 待拍板S2: bps 持久消费 (flag=1/2 → 载观察档 bps 权重 merge,
        # 重启不丢 = 23:10 段7 观察→自动喂料闭环; 默认 0 = 纯观察回滚形;
        # '2' = auto_flip 门控档: 重启同样载 = 持久链与 '1' 同形, 差异只在触发端)
        if os.environ.get('UDE_BPS_WEIGHT', '0') in ('1', '2'):
            try:
                from modules.bps_synthesis import load_bps_weights
                _wp = os.path.join(PROJECT_ROOT, 'runs', 'consensus_ab',
                                   'bps_weights.jsonl')
                _w, _i = load_bps_weights(self.weights, _wp)
                if _i:
                    self.weights = _w
                    logger.info(f"[UnifiedDecision] bps 权重已载 ({_i})")
            except Exception as e:
                logger.warning(f"[UnifiedDecision] bps 加载失败 (默认链续): {e}")

        logger.info(
            f"[UnifiedDecision] 决策引擎初始化: "
            f"models={len(self.AVAILABLE_MODELS)}, "
            f"lookback={self.lookback}, ewma_alpha={self.ewma_alpha}"
        )

        # P0: 漂移感知集成
        self.drift_ensemble = None
        if HAS_DRIFT_AWARE:
            try:
                self.drift_ensemble = DriftAwareEnsemble(lookback=lookback)
                # 注册所有可用模型
                for model in self.AVAILABLE_MODELS:
                    self.drift_ensemble.add_model(model, None)  # 占位
                logger.info("[UnifiedDecision] 漂移感知集成已启用")
            except Exception as e:
                logger.warning(f"[UnifiedDecision] 漂移感知集成初始化失败: {e}")

    def _load_metrics(self):
        """加载历史性能指标"""
        if os.path.exists(self._metrics_path):
            try:
                with open(self._metrics_path) as f:
                    metrics = json.load(f)
                self.weights = metrics.get('weights', self.weights)
                self._accuracy_history = metrics.get('accuracy_history', {})
                logger.info(f"[UnifiedDecision] 已加载性能指标: {len(self.weights)} 个权重")
            except Exception as e:
                logger.warning(f"[UnifiedDecision] 加载性能指标失败: {e}，使用默认权重")

    def _save_metrics(self):
        """保存性能指标"""
        try:
            os.makedirs(os.path.dirname(self._metrics_path), exist_ok=True)
            with open(self._metrics_path, 'w') as f:
                json.dump({
                    'weights': self.weights,
                    'accuracy_history': self._accuracy_history,
                    'updated_at': datetime.now().isoformat(),
                }, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"[UnifiedDecision] 保存性能指标失败: {e}")

    def _normalize_weights(self):
        """归一化权重到和为 1"""
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

    def _update_weights_ic_ir(self, stock_code: str, lookback: Optional[int] = None):
        """
        基于滚动 IC-IR 更新模型权重

        IC (Information Coefficient): 预测值与实际收益的相关系数
        IR (Information Ratio): IC / IC 的标准差

        权重 ∝ |IR|，保证正负 IR 的模型都有参与

        Args:
            stock_code: 股票代码
            lookback: 回溯天数
        """
        lookback = lookback or self.lookback
        history = self._accuracy_history.get(stock_code, {})

        if len(history) < 10:
            logger.debug(f"[UnifiedDecision] 历史数据不足 ({len(history)} < 10)，使用先验权重")
            self.weights = {k: self.base_weights.get(k, 1.0) for k in self.AVAILABLE_MODELS}
            self._normalize_weights()
            return

        # 计算每个模型的 IC 和 IR
        ir_scores = {}
        for model in self.AVAILABLE_MODELS:
            accuracies = [v.get(model, 0) for v in history.values()]
            if len(accuracies) < 5:
                continue

            # IC = 准确率 - 0.5 (随机猜测为 0.5)
            ic = np.mean(accuracies) - 0.5
            # IR = IC / std(IC)
            ic_std = np.std(accuracies) + 1e-8
            ir = ic / ic_std

            ir_scores[model] = ir

        # 权重 ∝ |IR|，用 softmax 缩放
        if ir_scores:
            abs_irs = np.array([abs(ir_scores.get(m, 0)) for m in self.AVAILABLE_MODELS])
            # 用 exp 缩放避免权重过小
            exp_irs = np.exp(abs_irs * 2)  # 温度参数 2
            # 结合先验权重
            combined = np.array([
                exp_irs[i] * self.base_weights.get(self.AVAILABLE_MODELS[i], 1.0)
                for i in range(len(self.AVAILABLE_MODELS))
            ])
            self.weights = {
                self.AVAILABLE_MODELS[i]: combined[i]
                for i in range(len(self.AVAILABLE_MODELS))
            }
            self._normalize_weights()
            logger.info(f"[UnifiedDecision] 权重已更新 (IC-IR): {dict(list(self.weights.items())[:5])}...")

    def _parse_direction(self, result: Dict) -> Tuple[str, float]:
        """解析返回 direction 字段的模型 (buy/sell/neutral + confidence)"""
        direction = result.get('direction', result.get('signal', 'neutral')).lower()
        confidence = result.get('confidence', result.get('probability', 0.5))

        if direction in ('buy', 'bullish', 'up', 'strong_buy'):
            return 'buy', confidence
        elif direction in ('sell', 'bearish', 'down', 'strong_sell'):
            return 'sell', confidence
        return 'neutral', confidence

    def _parse_multi_factor(self, result: Dict) -> Tuple[str, float]:
        """解析多因子模型结果"""
        score = result.get('score', result.get('weighted_score', 0))
        # 因子得分通常 [-1, 1] 或 [0, 100]
        if -1 <= score <= 1:
            if score > 0.2:
                return 'buy', min(abs(score) * 2, 1.0)
            elif score < -0.2:
                return 'sell', min(abs(score) * 2, 1.0)
        elif score > 60:
            return 'buy', min((score - 50) / 50, 1.0)
        elif score < 40:
            return 'sell', min((50 - score) / 50, 1.0)
        return 'neutral', 0.3

    def _parse_drl(self, result: Dict) -> Tuple[str, float]:
        """解析 DRL 结果"""
        action = result.get('action', 'hold')
        confidence = result.get('confidence', 0.5)

        if action in ('buy', 'strong_buy'):
            return 'buy', confidence
        elif action in ('sell', 'strong_sell'):
            return 'sell', confidence
        return 'neutral', confidence

    def _parse_conformal(self, result: Dict) -> Tuple[str, float]:
        """解析共形预测结果"""
        prediction = result.get('prediction', result.get('direction', 'neutral'))
        confidence = result.get('confidence', 0.5)

        if isinstance(prediction, str):
            prediction = prediction.lower()
        if prediction in ('buy', 'bullish', 'up'):
            return 'buy', confidence
        elif prediction in ('sell', 'bearish', 'down'):
            return 'sell', confidence
        return 'neutral', confidence

    def _parse_alpha158(self, result: Dict) -> Tuple[str, float]:
        """解析 Alpha158 结果"""
        # Alpha158 返回 top_factors 列表
        factors = result.get('factors', result.get('top_factors', []))
        if isinstance(factors, list) and len(factors) > 0:
            # 根据因子方向统计
            buy_score = sum(1 for f in factors[:5] if f.get('direction', '').lower() in ('buy', 'positive', 'bullish'))
            sell_score = sum(1 for f in factors[:5] if f.get('direction', '').lower() in ('sell', 'negative', 'bearish'))
            if buy_score > sell_score + 1:
                return 'buy', min(buy_score / 5, 1.0)
            elif sell_score > buy_score + 1:
                return 'sell', min(sell_score / 5, 1.0)
        return 'neutral', 0.3

    def _parse_regime(self, result: Dict) -> Tuple[str, float]:
        """解析 Regime-Switching 结果"""
        regime = result.get('regime', result.get('current_regime', 'neutral'))
        confidence = result.get('confidence', result.get('probability', 0.5))

        if regime in ('bull', 'trending', 'up'):
            return 'buy', confidence
        elif regime in ('bear', 'crash', 'down'):
            return 'sell', confidence
        return 'neutral', confidence

    def _parse_cvar(self, result: Dict) -> Tuple[str, float]:
        """解析 CVaR 风险模型"""
        # CVaR 主要评估风险，低 CVaR 意味着高风险 → sell
        cvar = result.get('cvar', result.get('cvar_value', 0))
        var = result.get('var', result.get('var_value', 0))

        if cvar < 0:
            # 负 CVaR = 极端损失风险高
            return 'sell', min(abs(cvar), 1.0)
        elif cvar > 0.05:
            return 'buy', min(cvar, 1.0)
        return 'neutral', 0.3

    def _parse_cross_market(self, result: Dict) -> Tuple[str, float]:
        """解析跨市场融合结果"""
        direction = result.get('direction', result.get('signal', 'neutral'))
        confidence = result.get('confidence', result.get('fusion_score', 0.5))
        return self._parse_direction({'direction': direction, 'confidence': confidence})

    def _parse_timesfm(self, result: Dict) -> Tuple[str, float]:
        """解析 TimesFM 结果"""
        direction = result.get('direction', result.get('prediction', 'neutral'))
        confidence = result.get('confidence', result.get('probability', 0.5))
        return self._parse_direction({'direction': direction, 'confidence': confidence})

    def _parse_sota_decision(self, result: Dict) -> Tuple[str, float]:
        """解析 SOTA 决策引擎结果"""
        direction = result.get('direction', result.get('signal', 'neutral'))
        confidence = result.get('confidence', result.get('probability', 0.5))
        return self._parse_direction({'direction': direction, 'confidence': confidence})

    def _get_model_prediction(self, model: str, stock_code: str,
                               klines: Optional[List[Dict]] = None) -> Optional[Tuple[str, float]]:
        """获取单个模型的预测"""
        try:
            # 尝试从各 API 端点获取预测
            import urllib.request
            import urllib.error

            # 根据模型选择 API 端点
            api_endpoints = {
                'multi_factor': f'/api/sota/factors?stock_code={stock_code}',
                'itransformer': f'/api/sota/patchtst/predict?stock_code={stock_code}',
                'mamba': f'/api/sota/mamba/predict?stock_code={stock_code}',
                'diffusion': f'/api/sota/diffusion/predict?stock_code={stock_code}',
                'self_supervised': f'/api/sota/self-supervised/status',
                'drl': f'/api/sota/drl/decide/{stock_code}',
                'moirai': f'/api/sota/moirai/predict?stock_code={stock_code}',
                'conformal': f'/api/sota/conformal/predict?stock_code={stock_code}',
                'gnn': f'/api/sota/gnn/predict?stock_code={stock_code}',
                'alpha158': f'/api/sota/alpha158/predict',
                'time_llm': f'/api/time-llm/predict/{stock_code}',
                'regime_switching': f'/api/regime-switching/predict/{stock_code}',
                'cvar': f'/api/sota/cvar/risk?code={stock_code}',
                'cross_market': f'/api/sota/cross-market/fusion?code={stock_code}',
                'multi_agent': f'/api/sota/multiagent/pipeline',
                'timesfm': f'/api/sota/timesfm/predict/{stock_code}',
                'sota_decision': f'/api/sota/decision/{stock_code}',
                'dynamic_weights': f'/api/sota/factor-weights/status',
            }

            url = api_endpoints.get(model, '')
            if not url:
                return None

            full_url = f'http://127.0.0.1:5002{url}'
            req = urllib.request.Request(full_url, method='GET')
            req.add_header('User-Agent', 'UnifiedDecisionEngine/1.0')

            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())

            # 解析结果
            parser = self.MODEL_PARSERS.get(model, self._parse_direction)
            result = parser(data) if callable(parser) else self._parse_direction(data)
            return result

        except Exception as e:
            logger.debug(f"[UnifiedDecision] 模型 {model} 预测失败: {e}")
            return None

    def decide(self, stock_code: str,
               klines: Optional[List[Dict]] = None) -> UnifiedDecision:
        """
        做出统一交易决策

        Args:
            stock_code: 股票代码
            klines: K 线数据 (可选，部分模型需要)

        Returns:
            UnifiedDecision
        """
        timestamp = datetime.now().isoformat()
        votes: Dict[str, Tuple[str, float]] = {}
        weights = {}

        # 0. 基于 IC-IR 更新权重
        self._update_weights_ic_ir(stock_code)

        # 0.5: 漂移检测 — 排除漂移模型
        if self.drift_ensemble is not None:
            try:
                drift_report = self.drift_ensemble.get_drift_report()
                drifted_models = [
                    name for name, info in drift_report.get('models', {}).items()
                    if info.get('drifted', False)
                ]
                if drifted_models:
                    logger.warning(
                        f"[UnifiedDecision] 检测到漂移模型: {drifted_models}，已排除"
                    )
                    # 从权重中移除漂移模型
                    for m in drifted_models:
                        self.weights[m] = 0
            except Exception as e:
                logger.debug(f"[UnifiedDecision] 漂移检测失败: {e}")

        # 1. 收集所有模型预测
        for model in self.AVAILABLE_MODELS:
            if model not in self.weights or self.weights[model] <= 0:
                continue

            pred = self._get_model_prediction(model, stock_code, klines)
            if pred is not None:
                votes[model] = pred
                weights[model] = self.weights.get(model, 0)

        # 2. 计算投票统计
        buy_score = 0.0
        sell_score = 0.0
        buy_count = 0
        sell_count = 0
        neutral_count = 0

        for model, (direction, confidence) in votes.items():
            w = self.weights.get(model, 0)
            if direction == 'buy':
                buy_score += w * confidence
                buy_count += 1
            elif direction == 'sell':
                sell_score += w * confidence
                sell_count += 1
            else:
                neutral_count += 1

        total_weight = buy_score + sell_score + 0.001
        weighted_score = (buy_score - sell_score) / total_weight  # [-1, 1]

        # 3. 计算共识度
        n_models = len(votes)
        max_vote = max(buy_count, sell_count, neutral_count)
        consensus = max_vote / n_models if n_models > 0 else 0

        # 4. 确定方向
        if weighted_score > 0.15 and consensus >= self.min_consensus and n_models >= self.min_models:
            direction = Direction.BUY if weighted_score < 0.5 else Direction.STRONG_BUY
        elif weighted_score < -0.15 and consensus >= self.min_consensus and n_models >= self.min_models:
            direction = Direction.SELL if weighted_score > -0.5 else Direction.STRONG_SELL
        else:
            direction = Direction.NEUTRAL

        confidence = min(abs(weighted_score) * consensus * 2, 1.0)

        # 5. 构建原因
        reasons = []
        if n_models > 0:
            reasons.append(f"{buy_count}买/{sell_count}卖/{neutral_count}中性")
        reasons.append(f"共识度={consensus:.2f}")
        reasons.append(f"加权得分={weighted_score:.3f}")
        reasons.append(f"参与模型={n_models}/{len(self.AVAILABLE_MODELS)}")

        decision = UnifiedDecision(
            direction=direction,
            confidence=round(confidence, 3),
            consensus=round(consensus, 3),
            n_models=n_models,
            weighted_score=round(weighted_score, 3),
            model_votes={m: d for m, (d, c) in votes.items()},
            model_weights=dict(self.weights),
            buy_count=buy_count,
            sell_count=sell_count,
            neutral_count=neutral_count,
            timestamp=timestamp,
            stock_code=stock_code,
            reason='; '.join(reasons),
        )

        logger.info(
            f"[UnifiedDecision] {stock_code}: {direction.value} "
            f"(conf={confidence:.2f}, consensus={consensus:.2f}, "
            f"score={weighted_score:.3f}, models={n_models})"
        )

        # P1-a: model seed 落盘 (23:10 replay accuracy_feed 对账原料; 非实时自指)
        if os.environ.get('UDE_SEED_LOG', '1') == '1' and votes:
            try:
                self._seed_log(stock_code, timestamp, dict(votes))
            except Exception as e:
                logger.debug(f"[UnifiedDecision] seed 落盘跳过: {e}")

        # 5: (摘除 2026-09-22) 旧「漂移监控器更新」段 = NameError 静默死链
        # (dict 推导引用未定义 score, 被 debug 吞) + 喂料自指 (actual=weighted_score
        # 近似 = 模型自评非真收益) → 漂移监控 KS/ADWIN 上线以来 0 喂料
        # (09-22 spy 实验实锤: 18 monitor 全 0 入; 自检失明第 4 例, 09-22 evaluate
        # 链同宗)。连带: 段 0.5 drift_exclude 门 (is_drifted) 恒 False = 漂移排除
        # 从未真触发。摘除非伪造: 复活需「模型原生连续 pred 落盘 + 23:10 replay
        # 真收益回填」两段链 (18 模型输出现为 (dir,conf) 形, 改形过大待拍板);
        # 决策质量层缺口已由 MAC 极化门 (conflict_ratio≥2v2 降档) 部分补位。
        return decision

    # ── P1-a (2026-09-22, arXiv 2412.10545) performative 反馈链 ──────
    # decide 尾 seed 落盘 (每模型票一行) + 23:10 replay accuracy_feed:
    # ≥3 交易日成熟 seed → K线真收益 → update_accuracy 喂入 (死入口复活:
    # 每模型 EWMA acc → softmax 权重自动降档)。非实时自指 = replay 真 outcome 形。

    def _seed_log(self, stock_code: str, ts: str,
                  votes: Dict[str, Tuple[str, float]]):
        """decide 尾 seed 落盘: 每模型票 (dir, conf) 一行 jsonl。
        链尾 flag UDE_SEED_LOG=0 全关 (默认 1); 写失败吞 (观察不拖决策)。"""
        try:
            with open(self._seed_path, 'a', encoding='utf-8') as f:
                for model, (d, c) in (votes or {}).items():
                    f.write(json.dumps({'ts': ts, 'code': stock_code,
                                        'model': model, 'dir': d,
                                        'conf': round(float(c), 3)}) + '\n')
        except Exception as e:
            logger.debug(f"[UnifiedDecision] seed 落盘跳过: {e}")

    def accuracy_feed(self) -> Dict:
        """23:10 replay 段: 成熟 seed (≥3 交易日) → 真收益对账 → update_accuracy。

        形同 _recalibrate_weights (MAC 侧): 非实时自指, 真 outcome 才喂;
        consumed.jsonl (code, ts) 防重喂 (replay 手动重跑防同 seed 双稀释,
        09-22 IC 双喂失真同教训); 无成熟 seed/链坏 → skipped (诚实, 链坏≠链死)。
        """
        if os.environ.get('UDE_ACC_FEED', '1') != '1':
            return {'skipped': 'UDE_ACC_FEED=0'}
        try:
            if not os.path.exists(self._seed_path):
                return {'skipped': 'model_seed.jsonl 未建 (今晚 23:10 链起积累)'}
            groups: Dict = {}   # (code, ts) → [(model, dir, conf), ...]
            with open(self._seed_path, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        s = json.loads(line)
                        groups.setdefault((s['code'], s['ts']), []).append(
                            (s['model'], s['dir'], float(s['conf'])))
                    except (KeyError, ValueError, TypeError):
                        continue
            consumed = set()
            if os.path.exists(self._feed_path):
                with open(self._feed_path, encoding='utf-8') as f:
                    for line in f:
                        try:
                            consumed.add(tuple(json.loads(line)))
                        except (ValueError, TypeError):
                            continue
            from modules.data_fetcher import StockDataFetcher
            fetcher = StockDataFetcher()
            klines: Dict = {}
            fed, unripe = 0, 0
            detail: Dict = {}
            new_consumed: List[str] = []
            for (code, ts), preds in groups.items():
                if (code, ts) in consumed:
                    continue   # 第二锁: 本 seed 已喂过 (replay 重跑防稀释)
                try:
                    ts_dt = datetime.fromisoformat(ts)
                except ValueError:
                    continue
                if (datetime.now() - ts_dt).days < 3:
                    unripe += 1
                    continue   # 未成熟 (同 recalibrate 3 交易日门)
                if code not in klines:
                    try:
                        klines[code] = fetcher.get_kline_data(code, 'daily', 90)
                    except Exception:
                        klines[code] = None
                kl = klines.get(code) or []
                if len(kl) < 10:
                    unripe += 1
                    continue
                p0 = None
                for k in reversed(kl):
                    d = str(k.get('date', ''))[:10]
                    if d and d <= ts_dt.strftime('%Y-%m-%d'):
                        p0 = k.get('close')
                        break
                p1 = kl[-1].get('close')
                if not p0 or not p1 or p0 <= 0:
                    unripe += 1
                    continue
                ret = (p1 - p0) / p0
                if abs(ret) < 0.005:
                    unripe += 1
                    continue   # 噪声区无方向语义 (recalibrate 同门)
                actual = 'up' if ret > 0 else 'down'
                self.update_accuracy(
                    code,
                    [{'model': m, 'direction': d, 'date': ts[:10]}
                     for m, d, _c in preds],
                    [actual] * len(preds))
                new_consumed.append(json.dumps([code, ts]))
                fed += len(preds)
                detail[f"{code}@{ts[:10]}"] = len(preds)
            if new_consumed:
                with open(self._feed_path, 'a', encoding='utf-8') as f:
                    f.write('\n'.join(new_consumed) + '\n')
            if fed == 0:
                return {'skipped': f'无成熟 seed ({unripe} 组未成熟/坏线已跳)'}
            return {'fed_pairs': fed, 'groups': detail}
        except Exception as e:
            logger.debug(f"[UnifiedDecision] accuracy_feed 跳过: {e}")
            return {'skipped': f'feed 失败: {type(e).__name__}: {str(e)[:100]}'}

    # ── P1-10 TopkDropout 防抖观察 (2026-09-14) ───────────────────

    def observe_dropout(self, holdings_scores: Dict[str, Dict],
                        candidate_scores: Dict[str, float],
                        market_data: Optional[Dict[str, Dict]] = None,
                        n_drop: int = 1, hold_thresh: int = 2) -> Optional[Dict]:
        """
        dropout 语义观察入口 (纯旁路, 不改任何实际决策输出)。

        对比「dropout 防抖决策」与「朴素阈值换仓」(评分低于最优候选的持仓
        即卖) 的差异: logger.info 记录分歧 + 追加
        runs/consensus_ab/dropout_observe.jsonl。decision_dropout_mode
        ='off' 时直接返回 None。内部异常降级为 warning (观察不拖主链)。

        Args 同 dropout_decide。

        Returns:
            dropout_decide 的结果 dict (mode='off' 或异常 → None)
        """
        if self.decision_dropout_mode == 'off':
            return None
        try:
            res = dropout_decide(holdings_scores, candidate_scores,
                                 n_drop=n_drop, hold_thresh=hold_thresh,
                                 market_data=market_data)
            # 朴素基线: 评分低于最优候选的持仓即卖 (无防抖, 抖动全换)
            best_cand = max(candidate_scores.values()) if candidate_scores else None
            naive_sells = ([c for c, info in holdings_scores.items()
                            if best_cand is not None
                            and float(info.get('score', 0.0)) < best_cand]
                           if best_cand is not None else [])
            diff = sorted(set(naive_sells) - set(res['sells']))
            logger.info(
                f"[UnifiedDecision] dropout 观察: sells={res['sells']} "
                f"buys={res['buys']} holds={res['holds']} | 朴素阈值换仓会卖 "
                f"{naive_sells} (dropout 防抖省掉 {diff})"
                + (f" | notes={res['notes']}" if res['notes'] else ''))
            rec = {
                'ts': datetime.now().isoformat(),
                'mode': 'observe',
                'n_drop': n_drop, 'hold_thresh': hold_thresh,
                'sells': res['sells'], 'buys': res['buys'],
                'holds': res['holds'], 'naive_sells': naive_sells,
                'debounced_sells': diff, 'notes': res['notes'],
            }
            ab_dir = os.path.join(PROJECT_ROOT, 'runs', 'consensus_ab')
            os.makedirs(ab_dir, exist_ok=True)
            with open(os.path.join(ab_dir, 'dropout_observe.jsonl'),
                      'a', encoding='utf-8') as f:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            return res
        except Exception as e:
            logger.warning(f"[UnifiedDecision] dropout 观察跳过: {e}")
            return None

    def update_accuracy(self, stock_code: str, predictions: List[Dict],
                        actual_outcomes: List[str]):
        """
        更新模型准确率 (在线学习)

        Args:
            stock_code: 股票代码
            predictions: [{'model': 'xxx', 'direction': 'buy', 'date': '2026-07-20'}, ...]
            actual_outcomes: ['up', 'down', ...] 实际涨跌
        """
        if stock_code not in self._accuracy_history:
            self._accuracy_history[stock_code] = {}

        # 按模型分组计算准确率
        model_correct = {}
        model_total = {}

        for pred, actual in zip(predictions, actual_outcomes):
            model = pred.get('model', '')
            pred_dir = pred.get('direction', '')

            model_total[model] = model_total.get(model, 0) + 1

            # 判断预测是否正确
            if (pred_dir == 'buy' and actual == 'up') or \
               (pred_dir == 'sell' and actual == 'down') or \
               (pred_dir == 'neutral' and actual == 'flat'):
                model_correct[model] = model_correct.get(model, 0) + 1

        # 更新 EWMA 准确率
        for model in model_total:
            accuracy = model_correct.get(model, 0) / model_total[model]
            if model in self._accuracy_history[stock_code]:
                prev = self._accuracy_history[stock_code][model]
                self._accuracy_history[stock_code][model] = (
                    self.ewma_alpha * accuracy + (1 - self.ewma_alpha) * prev
                )
            else:
                self._accuracy_history[stock_code][model] = accuracy

        # 基于准确率更新权重
        self._update_weights_from_accuracy(stock_code)
        self._save_metrics()

    def _update_weights_from_accuracy(self, stock_code: str):
        """基于准确率重新分配权重"""
        accuracies = self._accuracy_history.get(stock_code, {})

        if not accuracies:
            # 无历史数据，使用默认权重
            self.weights = dict(self.DEFAULT_WEIGHTS)
            return

        # 将准确率映射为权重 (软最大化)
        exp_scores = {}
        for model, acc in accuracies.items():
            exp_scores[model] = np.exp(acc * 10)  # 放大差异

        total = sum(exp_scores.values())
        if total > 0:
            for model in self.AVAILABLE_MODELS:
                if model in exp_scores:
                    self.weights[model] = exp_scores[model] / total
                else:
                    self.weights[model] = 0.001  # 最小权重

        # 归一化
        total_w = sum(self.weights.values())
        if total_w > 0:
            for m in self.weights:
                self.weights[m] /= total_w

        logger.info(
            f"[UnifiedDecision] 权重已更新: "
            f"top3={sorted(self.weights.items(), key=lambda x: -x[1])[:3]}"
        )

    def decide_from_predictions(
        self,
        stock_code: str,
        predictions: Dict[str, Dict],
        klines: Optional[List[Dict]] = None,
    ) -> UnifiedDecision:
        """
        从预计算的预测结果做出统一决策 (消除 HTTP 自调)

        替代 decide() 的 HTTP 自调路径，直接消费 analysis_engine.py
        的 sota_models dict。

        Args:
            stock_code: 股票代码
            predictions: {model_name: {'direction': 'up', 'confidence': 0.7, ...}, ...}
            klines: K 线数据 (可选)

        Returns:
            UnifiedDecision
        """
        timestamp = datetime.now().isoformat()
        votes: Dict[str, Tuple[str, float]] = {}
        weights = {}

        # 1. 解析每个模型的预测结果
        for model_name, pred in predictions.items():
            if model_name not in self.BASE_WEIGHTS:
                continue
            weights[model_name] = self.BASE_WEIGHTS.get(model_name, 1.0)

            # 解析预测结果
            parser = self.MODEL_PARSERS.get(model_name, self._parse_direction)
            try:
                result = parser(pred) if callable(parser) else self._parse_direction(pred)
                if result:
                    votes[model_name] = result
            except Exception as e:
                logger.debug(f"[UnifiedDecision] 解析模型 {model_name} 失败: {e}")

        # 2. 加权投票聚合
        buy_score = 0.0
        sell_score = 0.0
        buy_count = 0
        sell_count = 0
        neutral_count = 0

        for model, (direction, confidence) in votes.items():
            w = weights.get(model, 1.0)
            if direction == 'buy':
                buy_score += w * confidence
                buy_count += 1
            elif direction == 'sell':
                sell_score += w * confidence
                sell_count += 1
            else:
                neutral_count += 1

        total_weight = sum(weights.values()) or 1.0
        n_models = len(votes)

        # 3. 计算加权得分 [-1, 1]
        weighted_score = (buy_score - sell_score) / total_weight

        # 4. 计算共识度
        max_vote = max(buy_count, sell_count, neutral_count)
        consensus = max_vote / n_models if n_models > 0 else 0

        # 5. 确定方向 (与 decide() 一致的阈值)
        if weighted_score > 0.15 and consensus >= self.min_consensus and n_models >= self.min_models:
            direction = 'buy' if weighted_score < 0.5 else 'strong_buy'
        elif weighted_score < -0.15 and consensus >= self.min_consensus and n_models >= self.min_models:
            direction = 'sell' if weighted_score > -0.5 else 'strong_sell'
        else:
            direction = 'hold'

        confidence = min(abs(weighted_score) * consensus * 2, 1.0)

        # 6. 构建原因
        reasons = []
        if n_models > 0:
            reasons.append(f"{buy_count}买/{sell_count}卖/{neutral_count}中性")
        reasons.append(f"共识度={consensus:.2f}")
        reasons.append(f"加权得分={weighted_score:.3f}")
        reasons.append(f"参与模型={n_models}/{len(self.AVAILABLE_MODELS)}")

        decision = UnifiedDecision(
            direction=direction,
            confidence=round(confidence, 3),
            consensus=round(consensus, 3),
            n_models=n_models,
            weighted_score=round(weighted_score, 3),
            model_votes={m: d for m, (d, c) in votes.items()},
            model_weights=weights,
            buy_count=buy_count,
            sell_count=sell_count,
            neutral_count=neutral_count,
            timestamp=timestamp,
            stock_code=stock_code,
            reason='; '.join(reasons),
        )

        logger.info(
            f"[UnifiedDecision] {stock_code}: {direction} "
            f"(conf={confidence:.2f}, consensus={consensus:.2f}, "
            f"score={weighted_score:.3f}, models={n_models}) [管线模式]"
        )

        return decision

    def get_status(self) -> Dict:
        """获取引擎状态"""
        return {
            'status': 'running',
            'n_models': len(self.AVAILABLE_MODELS),
            'n_trained': len(self._accuracy_history),
            'weights': {k: round(v, 4) for k, v in sorted(
                self.weights.items(), key=lambda x: -x[1]
            )},
            'lookback': self.lookback,
            'ewma_alpha': self.ewma_alpha,
        }

    def reset_weights(self):
        """重置权重为等权重"""
        self.weights = dict(self.DEFAULT_WEIGHTS)
        self._save_metrics()
        logger.info("[UnifiedDecision] 权重已重置为等权重")


# 全局单例
_engine: Optional[UnifiedDecisionEngine] = None


def get_decision_engine() -> UnifiedDecisionEngine:
    """获取统一决策引擎全局实例"""
    global _engine
    if _engine is None:
        _engine = UnifiedDecisionEngine()
    return _engine


# 别名 (与 analysis_engine.py 中的导入一致)
get_unified_decision_engine = get_decision_engine


def reset_decision_engine():
    """重置全局实例"""
    global _engine
    _engine = None
