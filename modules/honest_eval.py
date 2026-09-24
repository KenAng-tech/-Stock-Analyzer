#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
honest_eval.py — 五门诚实评估审计层 (P2-11b)

2026 研究依据:
- MinervaScore (arXiv 2608.23808): DSR + PBO + SPA + 最小跟踪记录长度(MTRL)
  + regime 稳定性 五门合一的 post-selection 稳健性分级。其作者自证该分数
  在真实市场无前瞻预测力 (rho=0.013) — 因此本模块定位为 **可审计的验证
  报告层**, 不是盈利预测; 输出必须带此免责声明。
- 诚实评估框架 (arXiv 2608.27734): 收缩必须使用搜索的真实 trial 数 —
  本模块优先从 modules.search_ledger 读取真实 N, 读不到才退回传入值。

五门:
  1. DSR   Deflated Sharpe Ratio (Bailey & López de Prado 闭式)
  2. PBO   回测过拟合概率 (CSCV; 优先复用 hyperparam_optimizer.pbo_cscv_check)
  3. SPA    Superior Predictive Ability (Hansen 简化 bootstrap vs 基准)
  4. MTRL  最小跟踪记录长度 (达到观测 Sharpe 所需最少样本量检验)
  5. REGIME 分段稳定性 (三段切分 Sharpe 同号且最差段 > 0)
"""

import math
import logging
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger('stock_analyzer.honest_eval')

TRADING_DAYS = 252
DISCLAIMER = (
    '五门分数度量统计支持强度 (post-selection robustness), '
    '不是未来盈利概率; 参照 MinervaScore 作者自证: 此类分数在真实市场'
    '前瞻相关性不显著, 仅作可审计验证报告层。'
)


def _sharpe(returns: Sequence[float]) -> float:
    """日收益年化 Sharpe (无风险利率按 0)"""
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < 2:
        return float('nan')
    sd = r.std(ddof=1)
    if sd < 1e-12:
        return float('nan')
    return float(r.mean() / sd * math.sqrt(TRADING_DAYS))


def deflated_sharpe(returns: Sequence[float], n_trials: int) -> Dict:
    """
    Deflated Sharpe Ratio (Bailey & López de Prado 2014 闭式)。

    DSR = P( SR_true > 0 | 观测 SR, trial 数 N, 收益偏度/峰度 ),
    期望最大值 E[max SR] 由 N 的 Gumbel 近似给出。

    Args:
        returns: 日收益序列
        n_trials: 搜索的真实评估次数 (来自 search_ledger, 关键输入)

    Returns:
        {'dsr': float(0-1), 'sr_observed': float, 'sr_benchmark': float, 'n_trials': int}
    """
    try:
        from scipy.stats import skew, kurtosis, norm
        r = pd.Series(returns).dropna().to_numpy(dtype=float)
        n = len(r)
        if n < 30 or n_trials < 1:
            return {'error': f'样本不足 (n={n}, 需≥30) 或 trial 数非法'}
        sr_daily = r.mean() / r.std(ddof=1)
        g1 = float(skew(r))
        g2 = float(kurtosis(r, fisher=False))  # 非 Fisher: 正态=3
        # Bailey-LdP: SR* = E[max of N 个零假设样本 SR] ≈ sqrt(2 ln N) × σ(SR̂),
        # σ(SR̂) ≈ 1/sqrt(n-1) (日频单位) → SR*_daily = sqrt(2 ln N)/sqrt(n-1)
        sr_star_daily = math.sqrt(2.0 * math.log(max(n_trials, 2))) / math.sqrt(n - 1)
        denom = math.sqrt(max(1e-12, 1.0 - g1 * sr_daily + (g2 - 1.0) / 4.0 * sr_daily ** 2))
        z = (sr_daily - sr_star_daily) * math.sqrt(n - 1) / denom
        dsr = float(norm.cdf(z))
        return {
            'dsr': round(dsr, 4),
            'sr_observed': round(_sharpe(r), 4),
            'sr_benchmark_annualized': round(sr_star_daily * math.sqrt(TRADING_DAYS), 4),
            'n_trials': n_trials,
        }
    except Exception as e:
        logger.error(f"[HonestEval] DSR 计算失败: {e}", exc_info=True)
        return {'error': str(e)}


def mtrl_check(returns: Sequence[float], target_sharpe: float = 1.0, confidence: float = 0.95) -> Dict:
    """
    最小跟踪记录长度 (Minimum Track-record Length)。

    以观测到的偏度/峰度修正, 估计「以置信度 confidence 证明真实 Sharpe ≥
    target_sharpe 所需的最少年数」, 与实际年数比较。观测 Sharpe 越接近
    目标所需 Sharpe, 需要的记录越长 (Bailey & López de Prado)。
    """
    try:
        from scipy.stats import norm, skew, kurtosis
        r = pd.Series(returns).dropna().to_numpy(dtype=float)
        n = len(r)
        if n < 30:
            return {'error': f'样本不足 (n={n})'}
        sr = _sharpe(r)
        if math.isnan(sr):
            return {'error': 'Sharpe 不可计算'}
        g1 = float(skew(r))
        g2 = float(kurtosis(r, fisher=False))
        za, zb = norm.ppf(confidence), norm.ppf(0.5)  # 单侧置信, power 0.5 近似
        sr_t = target_sharpe / math.sqrt(TRADING_DAYS)  # 折日频
        sr_hat = sr / math.sqrt(TRADING_DAYS)
        numer = (za + zb) ** 2 * (1 - g1 * sr_hat + (g2 - 1) / 4.0 * sr_hat ** 2)
        denom = (sr_hat - sr_t) ** 2
        if sr_hat <= sr_t:
            return {'mtrl_years': float('inf'), 'actual_years': round(n / TRADING_DAYS, 2),
                    'pass': False, 'note': f'观测 SR {sr:.2f} ≤ 目标 {target_sharpe}, 任何记录长度都不够'}
        mtrl = numer / denom / TRADING_DAYS
        return {'mtrl_years': round(mtrl, 2), 'actual_years': round(n / TRADING_DAYS, 2),
                'pass': bool(n / TRADING_DAYS >= mtrl)}
    except Exception as e:
        logger.error(f"[HonestEval] MTRL 失败: {e}", exc_info=True)
        return {'error': str(e)}


def spa_check(returns: pd.Series, benchmark_returns: pd.Series, n_bootstrap: int = 500, seed: int = 42) -> Dict:
    """
    Superior Predictive Ability (Hansen 2005 简化 block-bootstrap 版)。

    检验「策略相对基准的超额均值 > 0」在新息自相关下的稳健性: 移动块
    bootstrap 重采样超额收益, 报告 p 值。n_bootstrap=500 足够审计用途。
    """
    try:
        df = pd.concat([returns, benchmark_returns], axis=1, join='inner').dropna()
        if len(df) < 60:
            return {'error': f'对齐样本不足 ({len(df)})'}
        excess = (df.iloc[:, 0] - df.iloc[:, 1]).to_numpy(dtype=float)
        obs = excess.mean()
        if obs <= 0:
            return {'p_value': 1.0, 'pass': False, 'excess_mean_daily': round(float(obs), 6),
                    'note': '超额均值为负, 无需 bootstrap'}
        rng = np.random.default_rng(seed)
        block = max(5, int(round(len(excess) ** 0.33)))
        n_blocks = len(excess) // block + 1
        nulls = np.empty(n_bootstrap)
        for b in range(n_bootstrap):
            starts = rng.integers(0, len(excess) - block + 1, size=n_blocks)
            sample = np.concatenate([excess[s:s + block] for s in starts])[:len(excess)]
            nulls[b] = sample.mean() - obs
        p = float((nulls >= 0).mean())
        return {'p_value': round(p, 4), 'pass': bool(p < 0.05),
                'excess_mean_daily': round(float(obs), 6), 'n_bootstrap': n_bootstrap}
    except Exception as e:
        logger.error(f"[HonestEval] SPA 失败: {e}", exc_info=True)
        return {'error': str(e)}


def regime_stability(returns: pd.Series, n_splits: int = 3) -> Dict:
    """
    分段稳定性: 等分 n 段, 各段年化 Sharpe 同号且最差段 > 0 才通过。
    (MinervaScore 用 regime 标签; 此处用时间分段作稳健代理, 避免依赖
    regime 模型的滞后标签。)
    """
    try:
        r = pd.Series(returns).dropna()
        if len(r) < n_splits * 40:
            return {'error': f'样本不足以切 {n_splits} 段'}
        shs = [_sharpe(seg) for seg in np.array_split(r.to_numpy(), n_splits)]
        valid = [s for s in shs if not math.isnan(s)]
        if len(valid) < 2:
            return {'error': '分段 Sharpe 不可计算'}
        same_sign = all(s > 0 for s in valid) or all(s < 0 for s in valid)
        return {'segment_sharpes': [round(s, 3) for s in shs],
                'same_sign': bool(same_sign),
                'worst_segment': round(min(valid), 3),
                'pass': bool(same_sign and min(valid) > 0)}
    except Exception as e:
        logger.error(f"[HonestEval] regime 稳定性失败: {e}", exc_info=True)
        return {'error': str(e)}


def resolve_trial_count(source: str, fallback: Optional[int] = None) -> Dict:
    """
    从 search_ledger 解析真实 trial 数 (2608.27734: 收缩必须用真实 N)。

    Returns: {'n_trials': int|None, 'source': 'ledger'|'fallback'|'none', 'window_days': 90}
    """
    try:
        from modules.search_ledger import trial_count
        n = trial_count(source=source, since_days=90)
        if n and n > 0:
            return {'n_trials': int(n), 'source': 'ledger', 'window_days': 90}
    except ImportError:
        logger.warning("[HonestEval] search_ledger 不可用, trial 数走 fallback")
    except Exception as e:
        logger.error(f"[HonestEval] ledger 读取失败: {e}")
    if fallback:
        return {'n_trials': int(fallback), 'source': 'fallback', 'window_days': None}
    return {'n_trials': None, 'source': 'none', 'window_days': None}


def five_gate_audit(
    returns: Sequence[float],
    benchmark_returns: Optional[Sequence[float]] = None,
    source: str = 'backtest',
    n_trials_fallback: Optional[int] = None,
    target_sharpe: float = 1.0,
) -> Dict:
    """
    五门合一审计 (MinervaScore 结构, 诚实定位)。

    Args:
        returns: 策略日收益
        benchmark_returns: 基准日收益 (缺省则 SPA 门记 skipped 并降满分上限)
        source: search_ledger 来源标签 (读真实 trial 数)
        n_trials_fallback: ledger 不可用时的 trial 数
        target_sharpe: MTRL 目标年化 Sharpe

    Returns:
        {'gates': {...5 门...}, 'score': 0-100, 'seal': bool, 'passed': n,
         'trial_info': {...}, 'disclaimer': str}
    """
    try:
        r = pd.Series(returns)
        trial_info = resolve_trial_count(source, n_trials_fallback)
        gates: Dict[str, Dict] = {}

        gates['dsr'] = deflated_sharpe(r, trial_info['n_trials'] or 1) if trial_info['n_trials'] else {'error': 'trial 数未知'}
        if 'dsr' in gates['dsr']:
            gates['dsr']['pass'] = bool(gates['dsr']['dsr'] > 0.95)

        # PBO: 需要多 trial × 多窗矩阵, 单序列不可算 → 记 skipped 并说明
        try:
            from modules.hyperparam_optimizer import pbo_cscv_check  # noqa: F401
            gates['pbo'] = {'status': 'skipped',
                            'note': 'PBO 需 trials×windows 矩阵, 请经 walkforward 链传入 matrix 调 pbo_cscv_check; 单序列审计不含此门'}
        except ImportError:
            gates['pbo'] = {'status': 'skipped', 'note': 'hyperparam_optimizer 不可用'}

        if benchmark_returns is not None:
            gates['spa'] = spa_check(r, pd.Series(benchmark_returns))
        else:
            gates['spa'] = {'status': 'skipped', 'note': '未提供基准收益'}

        gates['mtrl'] = mtrl_check(r, target_sharpe=target_sharpe)
        gates['regime'] = regime_stability(r)

        def _passed(g):
            return bool(g.get('pass', False))

        evaluable = [k for k, g in gates.items() if 'pass' in g]
        passed = sum(1 for k in evaluable if _passed(gates[k]))
        skipped = [k for k in gates if k not in evaluable]

        score = round(100.0 * passed / max(len(evaluable), 1)) if evaluable else 0
        seal = bool(evaluable) and passed == len(evaluable) and not skipped and trial_info['source'] == 'ledger'

        return {
            'gates': gates,
            'score': score,
            'seal': seal,
            'passed': passed,
            'evaluable_gates': len(evaluable),
            'skipped_gates': skipped,
            'trial_info': trial_info,
            'disclaimer': DISCLAIMER,
        }
    except Exception as e:
        logger.error(f"[HonestEval] 五门审计失败: {e}", exc_info=True)
        return {'error': str(e)}
