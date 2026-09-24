#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
ML 数据处理器流水线 (qlib handler/processor 思想, P1-9, 2026-09-14)

qlib 数据流的两条核心纪律移植到本项目:
1. **learn/infer 守卫** — 训练期专用处理器 (如 DropnaLabel) 在推理流水线上
   是数据泄露/静默丢行的来源, qlib handler.py:534-535 在 infer 模式遇到
   is_for_infer()==False 的处理器直接 raise; 本模块同形复刻。
2. **fit 窗口硬约束** — 需要拟合统计量的处理器 (如 CSZScoreNorm) 必须显式
   传 fit_start_time/fit_end_time, 参数只在窗口内估计、窗口外冻结使用
   (防前视: 标准化统计量不得来自未来数据); 缺参数直接 assert。

处理器契约:
    Processor.fit(df, fit_start_time, fit_end_time) -> self   (幂等)
    Processor.transform(df) -> df                              (不改原 df 语义)
    Processor.is_for_infer() -> bool

用法:
    from modules.ml_processors import (CSZScoreNorm, ProcessInf, Fillna,
                                       DropnaLabel, apply_pipeline)
    processors = [ProcessInf(), Fillna(), CSZScoreNorm(), DropnaLabel()]
    train_df = apply_pipeline(df, processors, mode='learn',
                              fit_start_time='2024-01-01', fit_end_time='2025-01-01')
    infer_df = apply_pipeline(df, processors, mode='infer')  # DropnaLabel → TypeError

regime 样本加权 (P2-13):
    regime_sample_weights(current_regime_probs, historical_posteriors)
    → 每历史样本权重 ∝ P(当前 regime | 该日), 归一化均值 1 (纯函数)。
"""

from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

try:
    from modules.logger import logger
except Exception:  # pragma: no cover - 项目外独立运行回退
    import logging
    logger = logging.getLogger('ml_processors')

__all__ = [
    'Processor', 'CSZScoreNorm', 'ProcessInf', 'Fillna', 'DropnaLabel',
    'apply_pipeline', 'regime_sample_weights',
]


def _window_mask(df: pd.DataFrame, fit_start_time, fit_end_time) -> pd.Series:
    """取 [fit_start_time, fit_end_time] 闭窗的行掩码。

    时间列优先级: 'date'/'datetime' 列 > DatetimeIndex。ISO 字符串列按字典序
    比较即时间序, 无需解析。
    """
    if len(df) == 0:
        return pd.Series(False, index=df.index)
    col = None
    for cand in ('date', 'datetime'):
        if cand in df.columns:
            col = df[cand]
            break
    if col is None:
        if isinstance(df.index, pd.DatetimeIndex):
            col = pd.Series(df.index, index=df.index)
        else:
            raise ValueError("df 需含 'date'/'datetime' 列或 DatetimeIndex 才能按窗口 fit")
    start = pd.to_datetime(fit_start_time)
    end = pd.to_datetime(fit_end_time)
    times = pd.to_datetime(col)
    return (times >= start) & (times <= end)


class Processor:
    """处理器基类 — 契约: fit (可幂等) / transform / is_for_infer

    子类约定:
    - 需要拟合统计量的置 need_fit=True, fit 内 assert fit_start_time/fit_end_time
      (qlib 硬约束: 标准化参数只能来自显式历史窗口, 防前视);
    - fit 结果只存普通属性 (self._xxx), 保证整条流水线可 pickle;
    - 训练期专用 (会丢样本/用标签) 的覆写 is_for_infer() 返回 False。
    """

    need_fit: bool = False

    def __init__(self):
        self._is_fitted = False

    def is_for_infer(self) -> bool:
        """推理流水线是否可用 (False = 训练期专用, infer 模式被 apply_pipeline 拦截)"""
        return True

    def fit(self, df: pd.DataFrame, fit_start_time=None, fit_end_time=None) -> 'Processor':
        """拟合参数 (默认无状态, 幂等返回 self)"""
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """变换 (默认恒等)"""
        return df

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.transform(df)


class ProcessInf(Processor):
    """把 ±inf 替换为 NaN (qlib ProcessInf 同形) — 无状态, learn/infer 通用"""

    def __init__(self, fields_group: Optional[Sequence[str]] = None):
        super().__init__()
        self.fields_group = list(fields_group) if fields_group else None

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        cols = self.fields_group or [c for c in out.columns
                                     if pd.api.types.is_numeric_dtype(out[c])]
        cols = [c for c in cols if c in out.columns]
        if cols:
            out[cols] = out[cols].replace([np.inf, -np.inf], np.nan)
        return out


class Fillna(Processor):
    """NaN 填充 (qlib Fillna 同形) — 无状态, learn/infer 通用"""

    def __init__(self, fields_group: Optional[Sequence[str]] = None,
                 fill_value: float = 0.0):
        super().__init__()
        self.fields_group = list(fields_group) if fields_group else None
        self.fill_value = fill_value

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        cols = self.fields_group or list(out.columns)
        cols = [c for c in cols if c in out.columns]
        if cols:
            out[cols] = out[cols].fillna(self.fill_value)
        return out


class CSZScoreNorm(Processor):
    """z-score 标准化 (qlib CSZScoreNorm 思想) — 统计量只在 fit 窗口内估计。

    硬约束 (qlib): fit 必须显式传 fit_start_time/fit_end_time, 否则 assert —
    标准化参数禁止来自全量/未来数据 (前视偏差)。transform 用冻结参数,
    窗口外数据同样按 fit 时估计的 mean/std 变换; std==0 列按 1 处理防除零。
    """

    need_fit = True

    def __init__(self, fields_group: Optional[Sequence[str]] = None):
        super().__init__()
        self.fields_group = list(fields_group) if fields_group else None
        self.mean_: Dict[str, float] = {}
        self.std_: Dict[str, float] = {}

    def fit(self, df: pd.DataFrame, fit_start_time=None, fit_end_time=None) -> 'CSZScoreNorm':
        assert fit_start_time is not None and fit_end_time is not None, \
            "CSZScoreNorm.fit 必须显式传 fit_start_time/fit_end_time (qlib 硬约束: 统计量只能来自历史窗口, 防前视)"
        cols = self.fields_group or [c for c in df.columns
                                     if pd.api.types.is_numeric_dtype(df[c])]
        cols = [c for c in cols if c in df.columns]
        window = df.loc[_window_mask(df, fit_start_time, fit_end_time), cols]
        if len(window) == 0:
            logger.warning(f"[CSZScoreNorm] fit 窗口 [{fit_start_time}, {fit_end_time}] 内无数据, "
                           f"参数冻结为空 (transform 将透传)")
            self.mean_, self.std_, self._is_fitted = {}, {}, True
            return self
        self.mean_ = {}
        self.std_ = {}
        for c in cols:
            m = float(window[c].mean())
            s = float(window[c].std(ddof=0))
            self.mean_[c] = m if np.isfinite(m) else 0.0
            self.std_[c] = s if (np.isfinite(s) and s > 0) else 1.0
        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("CSZScoreNorm 未 fit 就 transform — 请先 fit(df, fit_start_time, fit_end_time)")
        out = df.copy()
        for c in self.mean_:
            if c in out.columns:
                out[c] = (out[c] - self.mean_[c]) / self.std_[c]
        return out


class DropnaLabel(Processor):
    """丢弃标签为 NaN 的样本 (qlib DropnaLabel 同形) — 训练期专用。

    is_for_infer()=False: 推理时标签本就不存在/不可丢 (丢行会错位预测),
    apply_pipeline(mode='infer') 遇之 raise TypeError (qlib handler.py:534-535 守卫)。
    """

    def __init__(self, label_col: str = 'label'):
        super().__init__()
        self.label_col = label_col

    def is_for_infer(self) -> bool:
        return False

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.label_col not in df.columns:
            logger.warning(f"[DropnaLabel] 列 '{self.label_col}' 不存在, 透传")
            return df
        return df.dropna(subset=[self.label_col])


def apply_pipeline(df: pd.DataFrame, processors: Sequence[Processor],
                   mode: str = 'learn', fit_start_time=None,
                   fit_end_time=None) -> pd.DataFrame:
    """按序应用处理器流水线, mode 守卫 learn/infer 两条链。

    Args:
        df: 输入 DataFrame (含 'date'/'datetime' 列或 DatetimeIndex)
        processors: Processor 序列
        mode: 'learn' — 先 fit(窗口参数) 再 transform;
              'infer' — 只 transform (用冻结参数), 遇 is_for_infer()==False
                        的处理器立即 raise TypeError (qlib 同形守卫)
        fit_start_time / fit_end_time: learn 模式传给需要 fit 的处理器

    Returns:
        变换后的 DataFrame (不修改原 df)
    """
    if mode not in ('learn', 'infer'):
        raise ValueError(f"mode 必须是 'learn' 或 'infer', 收到: {mode!r}")
    for proc in processors:
        if mode == 'infer' and not proc.is_for_infer():
            raise TypeError(
                f"{type(proc).__name__} 是训练期专用处理器 (is_for_infer=False), "
                f"infer 流水线禁止使用 (qlib handler 守卫同形)")
        if mode == 'learn' and proc.need_fit:
            proc.fit(df, fit_start_time=fit_start_time, fit_end_time=fit_end_time)
        df = proc.transform(df)
    return df


def regime_sample_weights(current_regime_probs: Dict[str, float],
                          historical_posteriors: Union[np.ndarray, Sequence[Dict[str, float]]],
                          regime_order: Optional[List[str]] = None) -> np.ndarray:
    """regime 相似性样本加权 (P2-13) — 纯函数, 无副作用。

    思想: 模型该更重视"与当前市场状态相似"的历史样本。每个历史样本 t 的权重
    ∝ P(当前 regime 状态 | 该日) = Σ_r P_cur(r) · P(r | day t), 即当前 regime
    分布与该日后验分布的内积; 最后归一化为均值 1 (不改变总有效样本量)。

    Args:
        current_regime_probs: 当前 regime 概率 {regime名: prob} (建议归一化,
            未归一化时按其原样计算 — 权重只要求相对比例)
        historical_posteriors: 每历史样本的 regime 后验, 两种形态:
            - np.ndarray shape (n_days, n_regimes), 列序 = regime_order;
            - list[dict], 每元素 {regime名: prob} (自动对齐 regime_order)
        regime_order: 列序对应的 regime 名列表; historical_posteriors 为
            ndarray 时必填语义 (缺省取 current_regime_probs 键的排序, 要求维度吻合)

    Returns:
        np.ndarray shape (n_days,), 均值 1 (全零退化时返回全 1)

    Raises:
        ValueError: 维度不匹配
    """
    if isinstance(historical_posteriors, (list, tuple)) and len(historical_posteriors) > 0 \
            and isinstance(historical_posteriors[0], dict):
        order = list(regime_order) if regime_order else sorted(current_regime_probs.keys())
        post = np.array([[float(d.get(r, 0.0)) for r in order]
                         for d in historical_posteriors], dtype=float)
    else:
        post = np.asarray(historical_posteriors, dtype=float)
        if post.ndim == 1:
            post = post.reshape(-1, 1)
        order = list(regime_order) if regime_order else sorted(current_regime_probs.keys())
    if post.ndim != 2:
        raise ValueError(f"historical_posteriors 需为 2-D (n_days, n_regimes), 收到 ndim={post.ndim}")
    if post.shape[1] != len(order):
        raise ValueError(
            f"regime 维度不匹配: posteriors {post.shape[1]} 列 vs regime_order {len(order)} 项")
    cur = np.array([float(current_regime_probs.get(r, 0.0)) for r in order], dtype=float)

    weights = post @ cur
    weights = np.clip(weights, 0.0, None)  # 数值噪声负值截 0
    mean = weights.mean() if len(weights) else 0.0
    if len(weights) == 0 or not np.isfinite(mean) or mean <= 0:
        logger.warning("[regime_sample_weights] 权重全零/退化, 返回均匀权重")
        return np.ones(len(weights))
    return weights / mean
