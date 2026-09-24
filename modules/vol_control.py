#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
vol_control.py — 闭环波动率目标控制 (P2-11a)

2026 研究依据: arXiv 2603.01298 (Boyd/Candès/Hastie, Stanford, 2026-03)。
传统开环波动率目标 (exposure = target/forecast_vol) 有三病: 高换手、
杠杆尖峰、对估计误差敏感。本文提出比例反馈控制: 仓位增量正比于
「目标波动 - 已实现波动」的跟踪误差, 显式纠错而非盲从预测。

本模块为纯观察/建议层: 输出建议仓位序列与跟踪统计, 供决策链与面板
参考; 不直接改写任何现有仓位逻辑。
"""

import math
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger('stock_analyzer.vol_control')

TRADING_DAYS = 252


def _annualized_vol(returns: pd.Series, window: int) -> pd.Series:
    """滚动年化波动率 (min_periods 放宽避免起步 NaN)"""
    return returns.rolling(window, min_periods=max(3, window // 3)).std() * math.sqrt(TRADING_DAYS)


def closed_loop_vol_target(
    returns: pd.Series,
    target_vol: float = 0.15,
    kp: float = 0.10,
    max_leverage: float = 1.0,
    min_leverage: float = 0.0,
    vol_window: int = 20,
    ema_span: int = 5,
) -> Dict:
    """
    闭环比例控制波动率目标 (qlib/论文思想移植)。

    控制律 (论文为加性比例控制, 不是乘性):
        w_{t+1} = clip( w_t + kp * (target_vol - realized_vol_ema) / target_vol,
                        min_leverage, max_leverage )
    仓位增量正比于相对跟踪误差, 误差为 0 即停 — 收敛到
    w* ≈ target/asset_vol 的不动点; 乘性律会指数衰减到边界 (已修正)。
    realized_vol 用 EMA 平滑降低估计噪声 (论文: 少参数、可解释)。

    Args:
        returns: 策略/标的日收益序列 (pd.Series, 按日期升序)
        target_vol: 目标年化波动率 (默认 0.15)
        kp: 比例增益 (默认 0.10, 越大跟踪越快但可能振荡)
        max_leverage / min_leverage: 仓位上下限
        vol_window: 已实现波动滚动窗口 (交易日)
        ema_span: 波动率 EMA 平滑 span

    Returns:
        dict: {
          'exposure': pd.Series 建议仓位序列,
          'strategy_returns': pd.Series 控制后收益 (exposure.shift(1)*returns),
          'tracking_mae': 年化波动跟踪平均绝对误差,
          'realized_vol': 控制后组合年化波动,
          'turnover_mean': 日均仓位变动,
          'open_loop': 开环对照 {tracking_mae, turnover_mean, max_leverage_used},
        }
    """
    try:
        if returns is None or len(returns) < vol_window + 5:
            return {'error': f'样本不足 ({0 if returns is None else len(returns)} 条, 需 ≥{vol_window + 5})'}

        r = returns.dropna().astype(float)
        vol_raw = _annualized_vol(r, vol_window)
        vol_ema = vol_raw.ewm(span=ema_span, adjust=False).mean()

        exposures: List[float] = []
        w = min(1.0, max_leverage)
        for i in range(len(r)):
            exposures.append(w)
            v_asset = vol_ema.iloc[i]
            if v_asset is not None and not (isinstance(v_asset, float) and math.isnan(v_asset)) and v_asset > 1e-6:
                # 反馈信号 = 受控组合的已实现波动 (w_prev×资产波动),
                # 不是原始资产波动 —— 控制器必须看到自身效果, 否则积分饱和
                v_port = w * v_asset
                # 加性比例反馈 (论文式): 增量 ∝ 相对跟踪误差
                w = w + kp * (target_vol - v_port) / target_vol
                w = float(np.clip(w, min_leverage, max_leverage))

        exp = pd.Series(exposures, index=r.index, name='exposure')
        strat = exp.shift(1).fillna(exp.iloc[0]) * r

        realized = float(_annualized_vol(strat, min(len(strat), vol_window * 3)).iloc[-1])
        tracking_mae = None
        if len(strat) > vol_window:
            tracking_mae = float((_annualized_vol(strat, vol_window) - target_vol).abs().mean())

        # 开环对照: exposure = clip(target/forecast_vol)
        open_exp = (target_vol / vol_ema).clip(min_leverage, max_leverage).fillna(0.0)
        open_strat = open_exp.shift(1).fillna(0.0) * r
        open_mae = None
        if len(open_strat) > vol_window:
            open_mae = float((_annualized_vol(open_strat, vol_window) - target_vol).abs().mean())

        return {
            'exposure': exp,
            'strategy_returns': strat,
            'realized_vol': round(realized, 4) if realized and not math.isnan(realized) else None,
            'tracking_mae': round(tracking_mae, 4) if tracking_mae is not None and not math.isnan(tracking_mae) else None,
            'turnover_mean': round(float(exp.diff().abs().mean()), 4),
            'open_loop': {
                'tracking_mae': round(open_mae, 4) if open_mae is not None and not math.isnan(open_mae) else None,
                'turnover_mean': round(float(open_exp.diff().abs().mean()), 4),
                'max_leverage_used': round(float(open_exp.max()), 4),
            },
            'params': {'target_vol': target_vol, 'kp': kp, 'vol_window': vol_window,
                       'ema_span': ema_span, 'max_leverage': max_leverage},
        }
    except Exception as e:
        logger.error(f"[VolControl] 闭环波动率控制失败: {e}", exc_info=True)
        return {'error': str(e)}


def current_suggested_exposure(
    returns: pd.Series,
    target_vol: float = 0.15,
    kp: float = 0.10,
    max_leverage: float = 1.0,
) -> Dict:
    """
    给决策链的即时建议: 以当前为终态跑一遍闭环, 返回最新建议仓位。

    Returns:
        {'suggested_exposure': float, 'realized_vol': float, 'note': str}
    """
    res = closed_loop_vol_target(returns, target_vol=target_vol, kp=kp, max_leverage=max_leverage)
    if 'error' in res:
        return {'suggested_exposure': None, 'error': res['error']}
    return {
        'suggested_exposure': round(float(res['exposure'].iloc[-1]), 4),
        'realized_vol': res['realized_vol'],
        'tracking_mae': res['tracking_mae'],
        'open_loop_mae': res['open_loop']['tracking_mae'],
        'note': '闭环反馈控制 (arXiv 2603.01298); 建议层, 不自动执行',
    }
