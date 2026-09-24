#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
多智能体共识决策引擎 (Multi-Agent Consensus Engine)

六个独立智能体从不同维度分析, 加权投票生成最终共识。

智能体角色与数据源 (2026-09-02 P0 修复):
- technical: 统一决策引擎 (SOTA/GNN/Regime/CVaR 模型组)
- fundamental: FundamentalFetcher (AKShare 财务数据)
- sentiment: 新闻+股吧 → SentimentEngine (FinBERT/词典)
- quant: 统一决策引擎 (多因子/Alpha158/CVaR/跨市场模型组)
- timeseries: 统一决策引擎 (Moirai/TimesFM/PatchTST/Mamba/Diffusion/SSL/Time-LLM)
- rl: 统一决策引擎 (DRL/多智能体/Conformal 模型组)

修复背景: 旧版 6 条链直连各模块, 其中 5 条因 API 不匹配抛异常
被 except 静默吞掉 (analyze 方法不存在/类名错/模块路径是 .bak/缺参)
→ 每轮实际仅 fundamental 有票, "共识" 为空转; 且 set_dependencies()
注入的 unified_decision_engine 从未被调用。
新版: 4 个 agent 消费 UnifiedDecisionEngine (18 模型投票 + IC-IR 权重
+ 漂移门控), 数据源不可用时投诚实的 abstain 票 (neutral/0.2, reasoning
带原因) 而不是静默无票 — 页面可见 "哪个 agent 掉了"。

共识机制:
- 加权投票 (role weight × 组内权重 × 方向)
- 冲突检测 (多空分歧 >30% 标记 conflict)
- single-flight: 同票并发串行化 (per-stock 锁, 等待上限 25s)

用法:
    engine = MultiAgentConsensus()
    engine.set_dependencies(unified_decision_engine, analysis_engine)
    result = engine.decide('sz300620')
"""

import os
import queue
import threading
import time
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum

from modules.logger import logger

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 决策方向 → 数值 (与 _compute_consensus 的 decision_values 同尺度)
_DECISION_VALUES = {
    'strong_buy': 2, 'buy': 1, 'neutral': 0, 'sell': -1, 'strong_sell': -2,
}


def _side_of(decision: str) -> int:
    """方向 → 侧向符号: 多=+1, 空=-1, 中性/未知=0 (A/B 一致性判定用)"""
    d = str(decision).lower()
    if d in ('buy', 'strong_buy'):
        return 1
    if d in ('sell', 'strong_sell'):
        return -1
    return 0


def zscore_equal_weight(votes: List[Union['AgentVote', Dict]]) -> Dict:
    """
    旁路聚合: 组内 z-score 后等权 (P1-7, 2026-09-14, arXiv 2609.09588 对照轨)。

    主链共识是 weight×confidence 加权 (IC/Brier 门控), 隐含「confidence
    跨角色可比」假设 — 2609.09588 证明该假设下 IC 相关 ≠ PnL 相关。
    qlib 工业实践: 信号先做横截面 z-score 再等权。本函数把 z-score 等权
    落到同批票上, 仅作 A/B 观察, 不参与主决策。

    算法:
    1. 对同批票的 confidence 做组内 z-score (ddof=0); std=0 → 全 0 (退化);
    2. 负 z 截断为 0 — 低于组均的票弃权而非翻转方向 (避免低置信票变反向票);
    3. 等权平均 score = mean(decision_value × z⁺);
    4. score > 0.15 → buy, < -0.15 → sell, 否则 neutral。

    Args:
        votes: AgentVote 列表 (或含 decision/confidence 键的 dict 列表)

    Returns:
        {'direction', 'confidence', 'score', 'n_votes'}
    """
    items = []
    for v in votes or []:
        if isinstance(v, dict):
            items.append((str(v.get('decision', 'neutral')),
                          float(v.get('confidence', 0.0) or 0.0)))
        else:
            items.append((str(getattr(v, 'decision', 'neutral')),
                          float(getattr(v, 'confidence', 0.0) or 0.0)))
    if not items:
        return {'direction': 'neutral', 'confidence': 0.0, 'score': 0.0,
                'n_votes': 0}

    confs = np.array([c for _, c in items], dtype=float)
    mean, std = float(confs.mean()), float(confs.std())
    if std < 1e-12:
        z = np.zeros(len(confs))          # std=0 → 退化 0 (组内无分歧信息)
    else:
        z = np.maximum((confs - mean) / std, 0.0)
    score = float(np.mean([_DECISION_VALUES.get(d, 0) * zz
                           for (d, _), zz in zip(items, z)]))
    if score > 0.15:
        direction = 'buy'
    elif score < -0.15:
        direction = 'sell'
    else:
        direction = 'neutral'
    return {
        'direction': direction,
        'confidence': round(min(0.9, 0.4 + abs(score)), 4),
        'score': round(score, 4),
        'n_votes': len(items),
    }


class AgentRole(Enum):
    """智能体角色"""
    TECHNICAL = 'technical'
    FUNDAMENTAL = 'fundamental'
    SENTIMENT = 'sentiment'
    QUANT = 'quant'
    TIMESERIES = 'timeseries'
    RL = 'rl'


class Decision(Enum):
    """交易决策"""
    STRONG_BUY = 'strong_buy'
    BUY = 'buy'
    NEUTRAL = 'neutral'
    SELL = 'sell'
    STRONG_SELL = 'strong_sell'


@dataclass
class AgentVote:
    """智能体投票"""
    role: str
    decision: str
    confidence: float  # 0-1
    reasoning: str = ''
    weight: float = 1.0  # 智能体权重

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'AgentVote':
        return cls(
            role=data['role'],
            decision=data['decision'],
            confidence=data.get('confidence', 0.5),
            reasoning=data.get('reasoning', ''),
            weight=data.get('weight', 1.0),
        )


@dataclass
class ConsensusResult:
    """共识结果"""
    stock_code: str
    consensus: str  # strong_buy/buy/neutral/sell/strong_sell
    confidence: float  # 0-1
    vote_count: Dict[str, int]  # 各决策的票数
    agent_votes: List[Dict]  # 各智能体投票
    weights: Dict[str, float]  # 各角色权重
    conflict: bool  # 是否存在冲突 (多空分歧大)
    conflict_ratio: float  # 多空比例
    timestamp: str = ''
    reasoning: str = ''
    # P0-c (2026-09-12): 可成交性审计块 (观察层, 不改变共识/投票) — 默认 None
    # = 审计未运行; 形状见 modules/exec_audit.py audit_execution()
    exec_audit: Optional[Dict] = None
    # P1-b (2026-09-22): 决策过程六维审计块 (arXiv 2605.05739, 观察层非门控)
    # 形状见 modules/decision_quality.py audit_process(); 与 exec_audit 同 worker 内联
    decision_quality: Optional[Dict] = None

    def to_dict(self) -> Dict:
        d = asdict(self)
        d['consensus'] = self.consensus
        return d


# UnifiedDecisionEngine 的 18 模型 → 6 维 agent 归属 (按维度分组)
ROLE_MODEL_GROUPS = {
    AgentRole.TECHNICAL.value: ('sota_decision', 'gnn', 'regime_switching', 'cvar'),
    AgentRole.QUANT.value: ('multi_factor', 'alpha158', 'cross_market', 'dynamic_weights'),
    AgentRole.TIMESERIES.value: ('moirai', 'timesfm', 'itransformer',
                                 'diffusion', 'mamba', 'self_supervised', 'time_llm'),
    AgentRole.RL.value: ('drl', 'multi_agent', 'conformal'),
}


class MultiAgentConsensus:
    """多智能体共识决策引擎"""

    # 默认角色权重
    DEFAULT_WEIGHTS = {
        'technical': 1.0,
        'fundamental': 1.0,
        'sentiment': 0.8,
        'quant': 1.2,
        'timeseries': 1.1,
        'rl': 0.9,
    }

    def __init__(self, weights: Optional[Dict[str, float]] = None,
                 consensus_ab_mode: str = 'observe'):
        """
        Args:
            weights: 角色权重 (默认 DEFAULT_WEIGHTS)
            consensus_ab_mode: P1-7 双轨 A/B 观察 flag —
                'observe' (默认) 每轮把 z-score 等权旁路结果写
                runs/consensus_ab/{date}.jsonl; 'off' 完全不写。
                任何取值都不改变主决策。
        """
        self.weights = weights or dict(self.DEFAULT_WEIGHTS)
        self.consensus_ab_mode = consensus_ab_mode
        self._history: List[ConsensusResult] = []
        # 决策链 B (2026-09-11): Brier/ECE 校准观察键 (recalibrate 每 50 决策刷新; 空 {} = 未算)
        self._last_calibration: Dict = {}
        # 2026-09-03 Task #14: 决策历史 SQLite 持久化 — 重启从 DB 恢复最近 1000 条
        # (跨重启审计/IC resolution 样本不断档); 存储故障 → 降级纯内存 (原语义)
        try:
            from modules.decision_history_store import DecisionHistoryStore
            self._store = DecisionHistoryStore()
            restored = self._store.load_recent(limit=1000)
            self._history = [ConsensusResult(
                stock_code=r['stock_code'], consensus=r['consensus'],
                confidence=float(r['confidence'] or 0), vote_count={},
                agent_votes=r['agent_votes'], weights={},
                conflict=False, conflict_ratio=0.0,
                timestamp=r['timestamp'] or '') for r in reversed(restored)]
            if self._history:
                logger.info(f"[MultiAgent] 决策历史从 SQLite 恢复 {len(self._history)} 条")
        except Exception as e:
            self._store = None
            logger.warning(f"[MultiAgent] 决策历史存储不可用 (降级纯内存): {e}")
        self._decision_engine = None   # UnifiedDecisionEngine (app.py 注入)
        self._analysis_engine = None   # AnalysisEngine (app.py 注入)
        # 2026-09-02 P1: LLM Bull/Bear 辩论层 (TradingAgents 式) — 仅多空分歧时触发,
        # llm_router 降级 (rule_engine/fallback/超时) → 跳过辩论, 不伪造票
        self._enable_debate = True
        self._debate_timeout = 60.0  # 2026-09-15: 25→60, 8080 实测 17s+ 辩论常超时被跳过
        # 2026-09-02 P1: IC 权重闭环 — 每 50 次决策用历史决策 vs 真实收益重算角色权重
        self._decide_count = 0
        self._ic_min_history = 20
        # 2026-09-02: 单后台 worker 串行化决策 (防 MPS/Metal C 扩展与训练调度器
        # 并发冲突崩溃 + 不占用 Flask threading worker)。queue 满 → 直接拒绝。
        self._queue: 'queue.Queue' = queue.Queue(maxsize=8)
        self._worker = threading.Thread(
            target=self._worker_loop, daemon=True, name='consensus-worker',
        )
        self._worker.start()

    def set_dependencies(self, decision_engine, analysis_engine):
        """设置依赖 — 在 app.py 初始化后调用"""
        self._decision_engine = decision_engine
        self._analysis_engine = analysis_engine

    # ── 主流程 ────────────────────────────────────────────────────

    def decide(self, stock_code: str, history: Optional[Any] = None) -> ConsensusResult:
        """
        多智能体共识决策 (提交给后台 worker 串行执行, queue maxsize=8)

        Flask 请求线程最多等 45s; 排队满/等待超时 → 诚实的 abstain 结果。
        同票并发在 worker 中排队, 不阻塞也不冻结页面。
        """
        slot = threading.Event()
        box: Dict[str, Any] = {}
        try:
            self._queue.put((stock_code, slot, box), timeout=2.0)
        except queue.Full:
            logger.warning(f"[MultiAgent] {stock_code} 决策队列已满, 本轮跳过")
            return self._placeholder_result(stock_code, '决策队列已满 (>8 排队)')

        if not slot.wait(timeout=45):
            logger.warning(f"[MultiAgent] {stock_code} 决策排队超时 (45s)")
            return self._placeholder_result(stock_code, '决策排队超时 (>45s)')

        result = box.get('result')
        if result is None:
            return self._placeholder_result(stock_code, f'worker 异常: {box.get("error", "unknown")}')
        return result

    def _worker_loop(self):
        """后台决策 worker (串行执行 + MPS 锁, 防 Metal 并发断言崩溃)"""
        from modules.mps_lock import MPS_LOCK
        while True:
            stock_code, slot, box = self._queue.get()
            try:
                # MPS 重计算与后台训练调度器串行化 (训练持锁时本轮诚实跳过)
                if not MPS_LOCK.acquire(timeout=20):
                    logger.warning(f"[MultiAgent] {stock_code} MPS 忙 (训练中), 跳过本轮")
                    box['result'] = self._placeholder_result(stock_code, 'MPS 训练中, 本轮跳过')
                    continue
                try:
                    logger.info(f"[MultiAgent] 开始共识决策: {stock_code}")
                    votes = self._collect_votes(stock_code)
                    abstain = sum(1 for v in votes if v.confidence <= 0.25)
                    if abstain:
                        logger.warning(f"[MultiAgent] {abstain}/{len(votes)} 个智能体无有效数据源")
                    # P1: 多空分歧时 LLM Bull/Bear 辩论 (LLM 降级 → 跳过, 不伪造)
                    votes += self._run_debate(stock_code, votes)
                    result = self._compute_consensus(stock_code, votes)

                    # P1-7 (2026-09-14, arXiv 2609.09588): consensus 双轨 A/B —
                    # z-score 等权旁路聚合与主链 (IC/Brier 加权) 并行记录,
                    # 纯观察不改票/不改共识; 写失败只 warning, 不拖决策链
                    try:
                        self._log_consensus_ab(stock_code, votes, result)
                    except Exception as e:
                        logger.warning(f"[MultiAgent] consensus A/B 观察跳过: {e}")

                    # P0-c (2026-09-12, 2026 调研「执行假设审计」): 决策→执行
                    # 间的假设审计 (触板/流动性/成本) — 纯观察层, 不改票;
                    # 取数 join 5s 硬超时 (同 09-09 辩论超时保护形), 最坏跳过
                    try:
                        from modules.exec_audit import audit_for_stock
                        result.exec_audit = audit_for_stock(
                            stock_code, result.consensus, result.confidence,
                            timeout=5.0)
                    except Exception as e:
                        logger.debug(f"[MultiAgent] 执行审计跳过: {e}")

                    # P1-b (2026-09-22, arXiv 2605.05739 六维行为审计): 决策过程
                    # 六维审计 (纯计算零取数 <1ms; 证据/一致/分歧/不确定/辩论/推理)
                    # — 观察层非门控, 链坏≠链死 (审计异常吞=模块内+外层双兜底)
                    try:
                        from modules.decision_quality import audit_for_result
                        result.decision_quality = audit_for_result(result)
                    except Exception as e:
                        logger.debug(f"[MultiAgent] 过程审计跳过: {e}")

                    self._history.append(result)
                    if len(self._history) > 1000:
                        self._history = self._history[-500:]
                    if self._store:
                        self._store.append(result)  # 落库失败内部已吞 (审计不拖决策)

                    # P1: IC 权重闭环 — 每 50 次决策, 用真实收益重算角色权重
                    self._decide_count += 1
                    if self._decide_count % 50 == 0:
                        self._recalibrate_weights()

                    box['result'] = result
                    logger.info(
                        f"[MultiAgent] 共识: {result.consensus} "
                        f"(置信度 {result.confidence:.2f}, 冲突 {result.conflict})"
                    )
                finally:
                    MPS_LOCK.release()
            except Exception as e:
                logger.error(f"[MultiAgent] 决策 worker 异常: {e}")
                box['error'] = str(e)[:120]
            finally:
                slot.set()

    # ── LLM Bull/Bear 辩论 (P1, TradingAgents 式) ──────────────────

    _DEBATE_BULL = (
        '你是股票多头研究员。股票 {code} 的多维量化证据: {summary}\n'
        '请构建当前最强的**看涨**论证 (可结合市场常识), 即使证据偏空也要给'
        '出最强多头视角, 但 confidence 必须诚实 (论证弱则给低值)。\n'
        '只输出 JSON, 格式 {{"direction":"buy或neutral","confidence":0.35,"argument":"30字内理由"}}'
    )

    _DEBATE_BEAR = (
        '你是股票空头研究员。股票 {code} 的多维量化证据: {summary}\n'
        '请构建当前最强的**看跌**论证 (可结合市场常识), 即使证据偏多也要给'
        '出最强空头视角, 但 confidence 必须诚实 (论证弱则给低值)。\n'
        '只输出 JSON, 格式 {{"direction":"sell或neutral","confidence":0.35,"argument":"30字内理由"}}'
    )

    def _run_debate(self, stock_code: str, votes: List[AgentVote]) -> List[AgentVote]:
        """
        多空分歧时触发 Bull/Bear 辩论 (每路 llm_router 单调用, 串行)。

        返回 0/1/2 张辩论票:
        - 无分歧 (全体同向) → 0 票, 省 LLM 资源
        - LLM 降级/超时/解析失败 → 0 票 (诚实跳过, 不伪造)
        - 正常 → Bull 票 + Bear 票 (conf ≤0.6, weight 0.6, 不压过模型票)
        """
        if not self._enable_debate:
            return []
        bullish = sum(1 for v in votes if v.decision in (Decision.BUY.value, Decision.STRONG_BUY.value))
        bearish = sum(1 for v in votes if v.decision in (Decision.SELL.value, Decision.STRONG_SELL.value))
        if bullish == 0 or bearish == 0:
            return []  # 一致意见不辩论

        try:
            from modules.llm_router import llm_router as router
        except Exception as e:
            logger.warning(f"[MultiAgent] llm_router 不可用, 跳过辩论: {e}")
            return []

        summary = '; '.join(f'{v.role}={v.decision}({v.confidence:.2f})' for v in votes)[:400]
        t0 = time.time()
        extra: List[AgentVote] = []
        try:
            for side, prompt_tpl, allow in (
                ('bull', self._DEBATE_BULL, ('buy', 'neutral')),
                ('bear', self._DEBATE_BEAR, ('sell', 'neutral')),
            ):
                resp = router.route(
                    prompt_tpl.format(code=stock_code, summary=summary),
                    timeout=self._debate_timeout,
                )
                if not resp.get('success') or resp.get('fallback') or resp.get('provider') == 'rule_engine':
                    logger.info(f"[MultiAgent] {side} 辩论跳过 (provider={resp.get('provider')}, fallback)")
                    continue
                content = (resp.get('content') or '').strip()
                if content.startswith('```'):
                    content = content.strip('`').lstrip('json').strip()
                try:
                    data = json.loads(content)
                except Exception:
                    start = content.find('{')
                    end = content.rfind('}')
                    if start < 0 or end <= start:
                        continue
                    try:
                        data = json.loads(content[start:end + 1])
                    except Exception:
                        continue
                # ── tool contract (2026-09-11 决策链 B): schema fail-closed ──
                # 缺 direction 不再默认 neutral (中性票会稀释共识分母);
                # conf 非有限 (NaN 能穿过 max/min) 或缺失 → 无票 (诚实跳过)
                direction = data.get('direction')
                if direction not in allow:
                    continue
                conf = data.get('confidence')
                try:
                    conf = float(conf)
                    if conf != conf:      # NaN → 弃票 (!= 自身即 NaN)
                        continue
                except (TypeError, ValueError):
                    continue
                conf = max(0.1, min(0.6, conf))
                extra.append(AgentVote(
                    role=f'llm_{side}',
                    decision=direction,
                    confidence=conf,
                    reasoning=f"LLM辩论 ({resp.get('provider', '?')}, {resp.get('latency', 0):.1f}s): "
                              f"{str(data.get('argument', ''))[:60]}",
                    weight=0.6,
                ))
            if extra:
                logger.info(f"[MultiAgent] 辩论完成: {len(extra)} 票, 耗时 {time.time()-t0:.1f}s")
            return extra
        except Exception as e:
            logger.error(f"[MultiAgent] 辩论失败 (跳过): {e}")
            return []

    def _collect_votes(self, stock_code: str) -> List[AgentVote]:
        """收集 6 个 agent 投票 (每路保证有票, 数据源坏 → abstain 票)"""
        ude_result, ude_err = self._call_decision_engine(stock_code)

        votes = [
            self._vote_from_ude(AgentRole.TECHNICAL.value, ude_result, ude_err),
            self._vote_fundamental(stock_code),
            self._vote_sentiment(stock_code),
            self._vote_from_ude(AgentRole.QUANT.value, ude_result, ude_err),
            self._vote_from_ude(AgentRole.TIMESERIES.value, ude_result, ude_err),
            self._vote_from_ude(AgentRole.RL.value, ude_result, ude_err),
        ]
        return votes

    # ── 统一决策引擎 (technical/quant/timeseries/rl 共用) ─────────

    def _call_decision_engine(self, stock_code: str):
        """调 UnifiedDecisionEngine.decide() — 18 模型投票+IC权重+漂移门控"""
        if self._decision_engine is None:
            return None, 'unified_decision_engine 未注入'
        try:
            result = self._decision_engine.decide(stock_code)
            return result, None
        except Exception as e:
            logger.error(f"[MultiAgent] 统一决策引擎失败: {e}")
            return None, f'统一决策引擎: {str(e)[:100]}'

    def _vote_from_ude(self, role: str, ude_result, ude_err: Optional[str]) -> AgentVote:
        """把 UDE 模型组投票折算成角色票 (组内加权方向)"""
        weight = self.weights.get(role, 1.0)
        if ude_err or ude_result is None:
            return AgentVote(
                role=role, decision=Decision.NEUTRAL.value, confidence=0.2,
                reasoning=f'无数据源: {ude_err or "未注入"} → abstain',
                weight=weight,
            )

        try:
            model_votes = getattr(ude_result, 'model_votes', None) or {}
            model_weights = getattr(ude_result, 'model_weights', None) or {}
            group = [m for m in ROLE_MODEL_GROUPS.get(role, ()) if m in model_votes]
            if not group:
                return AgentVote(
                    role=role, decision=Decision.NEUTRAL.value, confidence=0.2,
                    reasoning='模型组无有效票 → abstain', weight=weight,
                )

            net = 0.0
            total_w = 0.0
            n_buy = n_sell = 0
            for m in group:
                d = model_votes[m]
                w = max(0.05, float(model_weights.get(m, 1.0)))
                total_w += w
                if d == 'buy':
                    net += w
                    n_buy += 1
                elif d == 'sell':
                    net -= w
                    n_sell += 1

            net /= max(total_w, 1e-9)
            if net > 0.67:
                decision = Decision.STRONG_BUY.value
            elif net > 0.34:
                decision = Decision.BUY.value
            elif net < -0.67:
                decision = Decision.STRONG_SELL.value
            elif net < -0.34:
                decision = Decision.SELL.value
            else:
                decision = Decision.NEUTRAL.value

            confidence = min(0.85, 0.45 + abs(net) * 0.5)
            reasoning = f'{n_buy}买/{n_sell}卖/{len(group) - n_buy - n_sell}中 (组净票 {net:+.2f})'
            return AgentVote(
                role=role, decision=decision, confidence=confidence,
                reasoning=reasoning, weight=weight,
            )
        except Exception as e:
            logger.error(f"[MultiAgent] {role} 折算失败: {e}")
            return AgentVote(
                role=role, decision=Decision.NEUTRAL.value, confidence=0.2,
                reasoning=f'投票折算异常: {str(e)[:80]} → abstain', weight=weight,
            )

    # ── fundamental (原链路保留 — 唯一原本就通的链) ────────────────

    def _vote_fundamental(self, stock_code: str) -> AgentVote:
        """基本面分析师: 基于财务数据 (ROE/营收增长/毛利率)"""
        weight = self.weights.get('fundamental', 1.0)
        try:
            from modules.fundamental_fetcher import FundamentalFetcher
            fetcher = FundamentalFetcher()
            data = fetcher.get_financial_data(stock_code)

            if not data:
                return AgentVote(
                    role=AgentRole.FUNDAMENTAL.value, decision=Decision.NEUTRAL.value,
                    confidence=0.3, reasoning='无财务数据 → abstain', weight=weight,
                )

            score = 0
            reasons = []
            roe = data.get('roe', 0)
            if roe > 15:
                score += 1
                reasons.append(f'ROE优秀({roe:.1f}%)')
            elif roe < 5:
                score -= 1
                reasons.append(f'ROE差({roe:.1f}%)')

            revenue_growth = data.get('revenue_growth', 0)
            if revenue_growth > 20:
                score += 1
                reasons.append(f'营收增长{revenue_growth:.0f}%')
            elif revenue_growth < 0:
                score -= 1
                reasons.append(f'营收下滑{revenue_growth:.0f}%')

            gross_margin = data.get('gross_margin', 0)
            if gross_margin > 50:
                score += 1
                reasons.append(f'毛利率高({gross_margin:.1f}%)')

            if score >= 2:
                decision = Decision.BUY.value
                confidence = 0.65
            elif score <= -1:
                decision = Decision.SELL.value
                confidence = 0.6
            else:
                decision = Decision.NEUTRAL.value
                confidence = 0.5

            return AgentVote(
                role=AgentRole.FUNDAMENTAL.value, decision=decision,
                confidence=confidence, reasoning='; '.join(reasons) or '财务指标中性',
                weight=weight,
            )
        except Exception as e:
            logger.error(f"[MultiAgent] 基本面分析师失败: {e}")
            return AgentVote(
                role=AgentRole.FUNDAMENTAL.value, decision=Decision.NEUTRAL.value,
                confidence=0.2, reasoning=f'基本面数据链异常: {str(e)[:80]} → abstain',
                weight=weight,
            )

    # ── sentiment (新链: 新闻/股吧 → SentimentEngine) ──────────────

    def _vote_sentiment(self, stock_code: str) -> AgentVote:
        """情绪分析师: 新闻+股吧 → SentimentEngine (FinBERT/词典)"""
        weight = self.weights.get('sentiment', 0.8)
        try:
            from modules.data_fetcher import StockDataFetcher
            from modules.sentiment_engine import get_sentiment_engine

            fetcher = StockDataFetcher()
            news = fetcher.get_stock_news(stock_code) or []
            posts = fetcher.get_stock_posts(stock_code) or []

            texts = [(n.get('title', '') if isinstance(n, dict) else str(n)) for n in news]
            texts += [(p.get('content', '') if isinstance(p, dict) else str(p)) for p in posts]
            texts = [t.strip() for t in texts if t and t.strip()][:16]

            if not texts:
                return AgentVote(
                    role=AgentRole.SENTIMENT.value, decision=Decision.NEUTRAL.value,
                    confidence=0.3,
                    reasoning=f'无舆情数据 (新闻0/股吧0) → abstain', weight=weight,
                )

            engine = get_sentiment_engine()
            results = engine.analyze_batch(texts)
            scores = [r.get('score', 0.0) for r in results if isinstance(r, dict)]
            if not scores:
                return AgentVote(
                    role=AgentRole.SENTIMENT.value, decision=Decision.NEUTRAL.value,
                    confidence=0.3, reasoning='情感分析无有效输出 → abstain', weight=weight,
                )

            avg = float(np.mean(scores))
            n_pos = sum(1 for s in scores if s > 0.1)
            n_neg = sum(1 for s in scores if s < -0.1)
            if avg > 0.3:
                decision = Decision.BUY.value
                confidence = min(0.8, 0.5 + abs(avg))
            elif avg < -0.3:
                decision = Decision.SELL.value
                confidence = min(0.8, 0.5 + abs(avg))
            else:
                decision = Decision.NEUTRAL.value
                confidence = 0.4

            return AgentVote(
                role=AgentRole.SENTIMENT.value, decision=decision,
                confidence=confidence,
                reasoning=f'舆情均值 {avg:+.2f} (新闻{len(news)}/股吧{len(posts)}, 多{n_pos}空{n_neg})',
                weight=weight,
            )
        except Exception as e:
            logger.error(f"[MultiAgent] 情绪分析师失败: {e}")
            return AgentVote(
                role=AgentRole.SENTIMENT.value, decision=Decision.NEUTRAL.value,
                confidence=0.2, reasoning=f'情感链异常: {str(e)[:80]} → abstain',
                weight=weight,
            )

    # ── 共识计算 ─────────────────────────────────────────────────

    def _placeholder_result(self, stock_code: str, reason: str) -> ConsensusResult:
        """排队超时/不可用时: 诚实的 abstain 结果 (可重试)"""
        votes = [AgentVote(
            role=r.value, decision=Decision.NEUTRAL.value, confidence=0.2,
            reasoning=f'本轮作废: {reason}',
            weight=self.weights.get(r.value, 1.0),
        ) for r in AgentRole]
        result = self._compute_consensus(stock_code, votes)
        self._history.append(result)
        return result

    def _compute_consensus(self, stock_code: str, votes: List[AgentVote]) -> ConsensusResult:
        """
        计算加权共识:
        consensus = Σ(weight × confidence × vote_value) / Σ(weight × confidence)
        """
        if not votes:
            return ConsensusResult(
                stock_code=stock_code,
                consensus=Decision.NEUTRAL.value,
                confidence=0.0,
                vote_count={},
                agent_votes=[],
                weights=self.weights,
                conflict=False,
                conflict_ratio=0.5,
                timestamp=datetime.now().isoformat(),
                reasoning='无有效投票',
            )

        decision_values = {
            Decision.STRONG_BUY.value: 2,
            Decision.BUY.value: 1,
            Decision.NEUTRAL.value: 0,
            Decision.SELL.value: -1,
            Decision.STRONG_SELL.value: -2,
        }

        weighted_score = 0.0
        total_weight = 0.0
        vote_count = {}
        agent_votes_data = []

        for vote in votes:
            val = decision_values.get(vote.decision, 0)
            weight = vote.weight
            confidence = vote.confidence

            weighted_score += val * weight * confidence
            total_weight += weight * confidence

            vote_count[vote.decision] = vote_count.get(vote.decision, 0) + 1
            agent_votes_data.append(vote.to_dict())

        if total_weight > 0:
            normalized_score = weighted_score / total_weight
        else:
            normalized_score = 0.0

        if normalized_score > 0.8:
            consensus = Decision.STRONG_BUY.value
        elif normalized_score > 0.3:
            consensus = Decision.BUY.value
        elif normalized_score > -0.3:
            consensus = Decision.NEUTRAL.value
        elif normalized_score > -0.8:
            consensus = Decision.SELL.value
        else:
            consensus = Decision.STRONG_SELL.value

        confidence = min(1.0, abs(normalized_score) + 0.3)

        bullish = vote_count.get(Decision.BUY.value, 0) + vote_count.get(Decision.STRONG_BUY.value, 0)
        bearish = vote_count.get(Decision.SELL.value, 0) + vote_count.get(Decision.STRONG_SELL.value, 0)
        total_votes = len(votes)

        if bullish > 0 and bearish > 0:
            conflict_ratio = min(bullish, bearish) / total_votes
            conflict = conflict_ratio > 0.3
        else:
            conflict = False
            conflict_ratio = 0.0

        reasons = [f'{v.role}: {v.decision} ({v.confidence:.2f})' for v in votes]

        # 分歧极化门 (2026-09-22, arXiv 2603.10137/2608.27076: 集成高分歧期
        # 方向票胜率坍塌): 多空各≥2 = 极化 (conflict_ratio>0.3 旧标记仅观察)。
        # 两档: 弱动能 (|score|<0.3) → 不bet 降 NEUTRAL; 有方向 (2v2 对冲仍能
        # 凑出 |score|≥0.3, 原 conf=|score|+0.3 公式与之脱钩 = 假安全) → 降档
        # 半信 (STRONG→普通 + conf×0.5)。flag DECISION_CONFLICT_GATE=0 回滚。
        if (bullish >= 2 and bearish >= 2
                and os.environ.get('DECISION_CONFLICT_GATE', '1') == '1'):
            if abs(normalized_score) < 0.3:
                consensus = Decision.NEUTRAL.value
                confidence = min(confidence, 0.25)
                reasons.insert(0, f'⊘ 极化门: 多{bullish}空{bearish} 强弱对消 → 不bet')
                logger.warning(f"[MultiAgent] {stock_code} 极化门: 多{bullish}/空{bearish} "
                               f"强弱对消 |score|={abs(normalized_score):.2f} → 不bet")
            else:
                if consensus == Decision.STRONG_BUY.value:
                    consensus = Decision.BUY.value
                elif consensus == Decision.STRONG_SELL.value:
                    consensus = Decision.SELL.value
                confidence = round(confidence * 0.5, 3)
                reasons.insert(0, f'⊘ 极化门: 多{bullish}空{bearish} 对冲方向半信 '
                               f'(conf×0.5={confidence:.2f})')
                logger.warning(f"[MultiAgent] {stock_code} 极化门: 多{bullish}/空{bearish} "
                               f"对冲方向半信 conf→{confidence:.2f}")

        return ConsensusResult(
            stock_code=stock_code,
            consensus=consensus,
            confidence=confidence,
            vote_count=vote_count,
            agent_votes=agent_votes_data,
            weights=self.weights,
            conflict=conflict,
            conflict_ratio=conflict_ratio,
            timestamp=datetime.now().isoformat(),
            reasoning='\n'.join(reasons),
        )

    # ── P1-7 双轨 A/B 观察 (2026-09-14) ───────────────────────────

    def _log_consensus_ab(self, stock_code: str, votes: List[AgentVote],
                          result: ConsensusResult) -> None:
        """
        z-score 等权旁路结果写观察日志 runs/consensus_ab/{date}.jsonl。

        每行: {ts, symbol, primary:{direction,confidence}, zscore:{...},
        agreement:bool} — agreement 按侧向 (多/空/中性) 判定, 分歧时另打
        info 日志方便 grep。consensus_ab_mode='off' 时直接返回; 本方法内
        异常向上抛由 worker 钩子降级为 warning, 不影响主链。
        """
        if self.consensus_ab_mode == 'off':
            return
        z = zscore_equal_weight(votes)
        agreement = _side_of(result.consensus) == _side_of(z['direction'])
        rec = {
            'ts': datetime.now().isoformat(),
            'symbol': stock_code,
            'primary': {'direction': result.consensus,
                        'confidence': round(float(result.confidence), 4)},
            'zscore': z,
            'agreement': agreement,
        }
        ab_dir = os.path.join(PROJECT_ROOT, 'runs', 'consensus_ab')
        os.makedirs(ab_dir, exist_ok=True)
        path = os.path.join(
            ab_dir, f"{datetime.now().strftime('%Y-%m-%d')}.jsonl")
        with open(path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        if not agreement:
            logger.info(
                f"[MultiAgent] A/B 分歧: {stock_code} 主链={result.consensus} "
                f"zscore等权={z['direction']}(score={z['score']}) — 证据积累中")

    # ── 查询/维护 ────────────────────────────────────────────────

    def _recalibrate_weights(self):
        """
        IC 权重闭环: 历史决策 resolution → 角色权重重标定。

        对每条可 resolution 的历史决策 (决策日 ≥3 个交易日前, 收益已成形):
          对每张方向票, 若决策方向与其后真实收益同号 → hit, 反号 → miss;
          每角色 ic_proxy = (hit - miss) / (hit + miss), 样本 ≥10 才参与,
          交 update_weights 重映射权重至 [0.5, 2.0]。
        在 worker 线程内联执行 (每 50 次决策一次, 纯 pandas/无 MPS)。
        """
        if len(self._history) < self._ic_min_history:
            return
        try:
            from modules.data_fetcher import StockDataFetcher
            fetcher = StockDataFetcher()
        except Exception as e:
            logger.warning(f"[MultiAgent] IC 闭环跳过: 数据源不可用 {e}")
            return

        try:
            # 每标的一次取数 (K线含 cache, 串行)
            codes = sorted({r.stock_code for r in self._history})
            klines: Dict[str, Optional[List[Dict]]] = {}
            for code in codes:
                try:
                    klines[code] = fetcher.get_kline_data(code, 'daily', 90)
                except Exception as e:
                    logger.debug(f"[MultiAgent] IC 闭环 K线失败 {code}: {e}")
                    klines[code] = None

            role_stats: Dict[str, List[int]] = {}  # role -> [hits, misses]
            # P0-2 (2026-09-22): Brier 持久池 = calib 独立源 (重启/截断不清零)。
            # 原内存形每次从 _history 现算 → 09-22 实锤 n16 首算→重启→n0,
            # role_gate/brier_ece 门恒饥饿。seen_briers = (code,ts) 防重喂第一锁
            # (第二锁 = decision_briers UNIQUE 键); 09-22 IC 双喂失真同教训。
            calib: List[Dict] = []   # 决策链 B: Brier/ECE 观察池 (持久池+本轮新算)
            seen_briers: set = set()
            new_briers: List[tuple] = []   # (code, ts, p, outcome) → 出循环批量落库
            if self._store:
                for _b in self._store.load_briers(5000):
                    calib.append({'p': _b['p'], 'outcome': _b['outcome']})
                    seen_briers.add((_b['code'], _b['ts']))
            for r in self._history:
                kl = klines.get(r.stock_code)
                if not kl or len(kl) < 10:
                    continue
                try:
                    ts = datetime.fromisoformat(r.timestamp)
                except (ValueError, TypeError):
                    continue
                if not r.agent_votes:
                    continue

                # 决策日收盘 → 当前收盘 的真实收益
                p0 = None
                # 取决策时刻之前最近一个交易日收盘 (含当日)
                for k in reversed(kl):
                    d = str(k.get('date', ''))[:10]
                    if d and d <= ts.strftime('%Y-%m-%d'):
                        p0 = k.get('close')
                        break
                p1 = kl[-1].get('close')
                if not p0 or not p1 or p0 <= 0:
                    continue
                # 至少 3 个新交易日, 收益才有方向信息
                if (datetime.now() - ts).days < 3:
                    continue
                ret = (p1 - p0) / p0
                if abs(ret) < 0.005:
                    continue

                # 决策链 B (2026-09-11): Brier/ECE 观察记录 — 仅方向性共识入池
                # (neutral/abstain 票无方向语义; |ret|<0.005 已在上文 continue)
                if r.consensus in ('buy', 'strong_buy', 'sell', 'strong_sell'):
                    _up = r.consensus in ('buy', 'strong_buy')
                    _p = float(r.confidence or 0.0)
                    _y = int((ret > 0) == _up)
                    _key = (r.stock_code, r.timestamp)
                    if _key not in seen_briers:
                        # 新算 resolution: 入本轮 calib + 落库池 (下次 recalc 从池读,
                        # 不再依赖 _history 幸存 — 截断/重启不断链)
                        seen_briers.add(_key)
                        calib.append({'p': _p, 'outcome': _y})
                        new_briers.append((r.stock_code, r.timestamp, _p, _y))

                for v in r.agent_votes:
                    role = v.get('role', '')
                    if role not in self.weights:
                        continue
                    d = v.get('decision', 'neutral')
                    if d in (Decision.BUY.value, Decision.STRONG_BUY.value):
                        hit = ret > 0
                    elif d in (Decision.SELL.value, Decision.STRONG_SELL.value):
                        hit = ret < 0
                    else:
                        continue
                    stat = role_stats.setdefault(role, [0, 0, []])
                    stat[0 if hit else 1] += 1
                    # P0-a (2026-09-12): 角色级 (conf, hit) 对 — role_gate 的原料
                    try:
                        stat[2].append((float(v.get('confidence') or 0),
                                        int(hit)))
                    except (TypeError, ValueError):
                        pass

            ic_data = {}
            for role, stat in role_stats.items():
                hits, misses = stat[0], stat[1]
                total = hits + misses
                if total >= 10:
                    ic_data[role] = (hits - misses) / total
            if ic_data:
                self.update_weights('batch', ic_data)
                logger.info(f"[MultiAgent] IC 闭环完成: { {k: round(v, 3) for k, v in ic_data.items()} }")

            # ── 决策链 B (2026-09-11): Brier/ECE 校准报告 (观察模式) ──
            # n<20 → skipped (诚实积累证据); 不参与任何门控/权重更新
            # P0-2: 新 resolution 落库 (INSERT OR IGNORE, UNIQUE+seen 双锁) —
            # 跨重启/跨触发累积, 下次 recalc 从池直读; 存储断链不拖决策 (09-03 同形)
            if self._store and new_briers:
                try:
                    self._store.append_briers(new_briers)
                except Exception as _e:
                    logger.debug(f"[MultiAgent] Brier 落库跳过: {_e}")

            from modules.decision_calibration import brier_ece, role_gate
            report = brier_ece(calib)

            # ── P0-a (2026-09-12, 2026 调研"可证伪化"向): 角色级 Brier 校准门 ──
            # 与 IC 闭环正交: IC=方向命中率, Brier=conf 校准质量 — 高 conf 连错
            # (校准漂移) IC 不敏感, 此处抓。n≥20 且 Brier>0.55 → weight×0.7 (floor
            # 0.5, IC 重映射 [0.5,2.0] 已先行); 无样本恒不动作 (诚实门控, 不空转)。
            gate = role_gate({r: s[2] for r, s in role_stats.items()
                              if len(s) >= 3 and s[2]})
            gated = {r: g for r, g in gate.items() if g.get('gated')}
            for r, g in gated.items():
                old = self.weights.get(r, 1.0)
                self.weights[r] = max(0.5, old * 0.7)
                logger.info(f"[MultiAgent] ⚠ 校准门: {r} Brier={g['brier']} "
                            f"(n={g['n']}, 阈值 0.55) → weight {old:.2f}→"
                            f"{self.weights[r]:.2f} (conf 校准漂移, 自动降权)")
            report['role_gate'] = gate
            report['gated_roles'] = sorted(gated)
            self._last_calibration = report
            if report.get('skipped'):
                logger.debug(f"[MultiAgent] 校准观察: {report['skipped']} — 证据积累中")
            else:
                logger.info(f"[MultiAgent] 校准观察: Brier={report['brier']} "
                            f"acc={report['acc']} ECE={report['ece']} n={report['n']} "
                            f"(门控: {sorted(gated) or '无'})")
        except Exception as e:
            logger.error(f"[MultiAgent] IC 闭环失败: {e}")

    def update_weights(self, stock_code: str, ic_data: Dict[str, float]):
        """
        基于 IC (Information Coefficient) 更新角色权重

        Args:
            stock_code: 股票代码
            ic_data: {role: ic_value}
        """
        for role, ic in ic_data.items():
            if role in self.weights:
                # IC 越高权重越高，范围 [0.5, 2.0]
                self.weights[role] = max(0.5, min(2.0, 1.0 + ic * 0.5))
        logger.info(f"[MultiAgent] 更新权重: {self.weights}")

    def get_history(self, limit: int = 10) -> List[Dict]:
        """获取历史共识结果"""
        return [h.to_dict() for h in self._history[-limit:]]

    def get_summary(self) -> Dict:
        """获取引擎摘要"""
        return {
            'total_decisions': len(self._history),
            'current_weights': self.weights,
            'recent_consensus': self.get_history(limit=5),
            # 决策链 B: 校准观察键 ({} = 未算; {skipped:…} = 证据积累中;
            # {n,brier,ece,…} = 有统计 — 只观测, 权重/门控未消费)
            'calibration': dict(self._last_calibration),
        }


# 全局单例
_consensus_engine: Optional[MultiAgentConsensus] = None


def get_consensus_engine() -> MultiAgentConsensus:
    """获取多智能体共识引擎全局实例"""
    global _consensus_engine
    if _consensus_engine is None:
        _consensus_engine = MultiAgentConsensus()
    return _consensus_engine
