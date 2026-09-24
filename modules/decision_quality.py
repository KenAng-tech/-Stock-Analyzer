"""decision_quality.py — 决策过程六维审计 (2026-09-22, arXiv 2605.05739 六维行为审计 5002 化)

三层并列 (不造轮子, 分层): honest_eval 五门=回测稳健 (DSR/PBO/SPA/MTRL/REGIME),
exec_audit=可成交性 (触板/流动性/26bp), 本模块=**决策过程质量**。纯函数零取数
(<1ms, worker 串行 20s MPS 预算不占); 输出只观察 (不改票/共识/门控);
链坏≠链死 (全路径吞异常 → skipped)。

六维 (每维 [0,1], 独立事实来源):
  evidence_completeness  证据完备: (n-abstain)/n, abstain=conf≤0.25 (09-12 worker 同形)
  internal_consistency   内部一致: 方向共识→1-异议率 (反向票/live); neutral→双向对异议标 0.5
  conflict_handling      分歧处理: 无极化 1.0 | 极化+bull/bear 辩论 1.0 (讨论过) | 极化无辩论 0.5
  uncertainty_quantification 不确定性量化: 方向+异议≥40%+conf≥0.8→0.5 (过度自信);
                         neutral+conf≥0.6→0.5 (不 bet 但 conf 虚高)
  debate_participation   辩论参与: bull/bear 票在→1.0 (触发); 无极化→1.0 (不需论证); 极化缺辩论→0.5
  reasoning_quality      推理链在: reasoning≥4字 满票占比 (空 consensus.reasoning 再×0.5)

落盘 = runs/consensus_ab/process_audit.jsonl (每次 decide 一行, A/B jsonl 与
dropout_observe 同宗) + GET /api/decision/process_quality (面板观察消费)。
flags=观察标记非门控 — 决策门控已在 P0-1 极化门 (降档) + P0-a Brier 门 (降权),
本层=证据积累 (六维趋势 → 未来门控候选, 待拍板)。
"""

import json
import os
from datetime import datetime
from typing import Dict, Optional

from modules.logger import logger

_PROC_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'runs', 'consensus_ab', 'process_audit.jsonl')

_DEBATE_ROLES = ('bull', 'bear')


def _vote_attr(v, key, default=None):
    """dict/AgentVote 双容 (audit 输入 = ConsensusResult.agent_votes 形状)"""
    if isinstance(v, dict):
        return v.get(key, default)
    return getattr(v, key, default)


def audit_process(result) -> Dict:
    """六维过程审计 (纯计算, 零取数; 输入 ConsensusResult)。

    Returns:
        {'ts','code','version','dims':{6 键}} 或 {'skipped': reason}
        — 样本不足诚实 skipped (≠链坏; 与 Brier n<20 同形)
    """
    votes = list(result.agent_votes or [])
    n = len(votes)
    if n < 2:
        return {'skipped': f'过程审计样本不足 n={n} (需≥2, 诚实空非链坏)'}

    live = [v for v in votes if (_vote_attr(v, 'confidence') or 0) > 0.25]
    if not live:
        return {'skipped': f'全弃权 {n} 票 (conf≤0.25, 无过程信息)'}

    cons = (result.consensus or '').lower()
    conf = float(result.confidence or 0.0)
    up = ('buy', 'strong_buy')
    down = ('sell', 'strong_sell')

    n_buy = sum(1 for v in live if _vote_attr(v, 'decision') in up)
    n_sell = sum(1 for v in live if _vote_attr(v, 'decision') in down)
    polarized = n_buy >= 2 and n_sell >= 2          # 2v2 极化 (P0-1 门输入同形)
    has_debate = any(_vote_attr(v, 'role') in _DEBATE_ROLES for v in votes)

    # d2 内部一致: 方向共识→反向票占 live; neutral→双向对异议 0.5 标记
    if cons in up:
        dissent = n_sell / max(len(live), 1)
    elif cons in down:
        dissent = n_buy / max(len(live), 1)
    else:
        dissent = 0.5 if (n_buy >= 1 and n_sell >= 1) else 0.0
    d2 = round(1.0 - dissent, 3)

    # d3/d5 分歧处理 & 辩论参与 (bull/bear 辩论链 = 讨论过证据)
    if not polarized:
        d3, d5 = 1.0, 1.0          # 单边/单向 = 无分歧, 不需论证 (不缺位)
    elif has_debate:
        d3, d5 = 1.0, 1.0          # 分歧 + 论证链触发
    else:
        d3, d5 = 0.5, 0.5          # 分歧无论证 = 观察标记 (非门控)

    # d4 不确定性量化 (过程自评质量: 异议占比 + conf 配平)
    if cons == 'neutral':
        d4 = 0.5 if conf >= 0.6 else 1.0
    elif dissent >= 0.4 and conf >= 0.8:
        d4 = 0.5                   # 高 conf + 半数异议 = 过度自信观察
    else:
        d4 = 1.0

    # d6 推理链在 (非短 reasoning 票占比; consensus.reasoning 空再减半)
    thin = sum(1 for v in votes
               if len(str(_vote_attr(v, 'reasoning') or '').strip()) < 4)
    d6 = round(1.0 - thin / n, 3)
    if not (result.reasoning or '').strip():
        d6 = round(d6 * 0.5, 3)

    return {
        'ts': datetime.now().isoformat(timespec='seconds'),
        'code': result.stock_code,
        'version': 'decision_quality_v1',
        'dims': {
            'evidence_completeness': round(len(live) / n, 3),
            'internal_consistency': d2,
            'conflict_handling': d3,
            'uncertainty_quantification': d4,
            'debate_participation': d5,
            'reasoning_quality': d6,
        },
    }


def audit_for_result(result) -> Optional[Dict]:
    """worker 内联入口 (exec_audit audit_for_stock 同宗): 审计 + jsonl 落盘。

    链坏≠链死: 审计异常/落盘失败吞掉 (观察不拖决策); None = 本轮未审计。
    """
    try:
        r = audit_process(result)
        if 'dims' not in r:
            return r          # skipped 形不落盘 (空样本无信息, 不占 jsonl)
        path = os.environ.get('DQ_PROC_PATH') or _PROC_PATH   # 测试/自定义重定向
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
        return r
    except Exception as e:
        logger.debug(f"[DecisionQuality] 过程审计跳过: {e}")
        return None
