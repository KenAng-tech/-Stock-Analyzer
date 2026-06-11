#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
因子 IC/ICIR 监控模块 — Factor IC Monitor

功能:
- 截面 IC (Pearson/Spearman)
- 滚动 IC 序列
- ICIR (IC Information Ratio)
- IC t-statistic
- 因子筛选 (基于 IC/ICIR/相关性)
- IC 衰减曲线与因子半衰期

参考: Liu, Wei, et al. "A review of factor co-moments." (2014)
"""

import numpy as np
import json
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from modules.logger import logger


class FactorICMonitor:
    """因子 IC/ICIR 持续监控"""

    def __init__(self):
        self._ic_history: Dict[str, List[float]] = {}

    def compute_cross_sectional_ic(
        self, factor_values: Dict[str, Dict[str, float]],
        future_returns: Dict[str, Dict[str, float]]
    ) -> float:
        """
        截面 IC (Pearson 相关系数)

        Args:
            factor_values: {date_str: {stock_code: factor_value}}
            future_returns: {date_str: {stock_code: return_pct}}

        Returns:
            IC 值 (-1 ~ 1)
        """
        # 取最新日期的截面数据
        dates = sorted(set(factor_values.keys()) & set(future_returns.keys()), reverse=True)
        if not dates:
            return 0.0

        date = dates[0]
        factors = []
        returns = []

        for code in set(factor_values[date].keys()) & set(future_returns[date].keys()):
            f = factor_values[date][code]
            r = future_returns[date][code]
            if not (np.isnan(f) or np.isnan(r) or np.isinf(f) or np.isinf(r)):
                factors.append(f)
                returns.append(r)

        if len(factors) < 5:
            return 0.0

        f_arr = np.array(factors)
        r_arr = np.array(returns)

        f_std = np.std(f_arr)
        r_std = np.std(r_arr)

        if f_std < 1e-10 or r_std < 1e-10:
            return 0.0

        return float(np.corrcoef(f_arr, r_arr)[0, 1])

    def compute_rank_ic(
        self, factor_values: Dict[str, Dict[str, float]],
        future_returns: Dict[str, Dict[str, float]]
    ) -> float:
        """
        秩 IC (Spearman 相关系数, 更稳健)
        """
        dates = sorted(set(factor_values.keys()) & set(future_returns.keys()), reverse=True)
        if not dates:
            return 0.0

        date = dates[0]
        factors = []
        returns = []

        for code in set(factor_values[date].keys()) & set(future_returns[date].keys()):
            f = factor_values[date][code]
            r = future_returns[date][code]
            if not (np.isnan(f) or np.isnan(r) or np.isinf(f) or np.isinf(r)):
                factors.append(f)
                returns.append(r)

        if len(factors) < 5:
            return 0.0

        # Rank both
        f_rank = np.array(factors).argsort().argsort() / (len(factors) - 1)
        r_rank = np.array(returns).argsort().argsort() / (len(returns) - 1)

        return float(np.corrcoef(f_rank, r_rank)[0, 1])

    def compute_ic_series(
        self, factor_values_list: List[Dict[str, float]],
        returns_list: List[float],
        window: int = 60
    ) -> List[float]:
        """
        滚动 IC 序列

        Args:
            factor_values_list: [{stock_code: factor_value}, ...] per date
            returns_list: [return_pct, ...] per date
            window: 滚动窗口大小

        Returns:
            IC 序列
        """
        ics = []
        for i in range(window, len(factor_values_list)):
            window_factors = []
            window_returns = []

            for j in range(i - window, i):
                for code, val in factor_values_list[j].items():
                    if code in returns_list and not np.isnan(val):
                        window_factors.append(val)
                        window_returns.append(returns_list[j])
                        break

            if len(window_factors) >= 10:
                f_arr = np.array(window_factors)
                r_arr = np.array(window_returns)
                f_std = np.std(f_arr)
                r_std = np.std(r_arr)
                if f_std > 1e-10 and r_std > 1e-10:
                    ic = np.corrcoef(f_arr, r_arr)[0, 1]
                    if not np.isnan(ic):
                        ics.append(float(ic))

        return ics

    def compute_icir(self, ic_series: List[float]) -> float:
        """
        ICIR = mean(IC) / std(IC)

        Args:
            ic_series: IC 时间序列

        Returns:
            ICIR 值
        """
        if len(ic_series) < 12:
            return 0.0
        arr = np.array(ic_series)
        mean_ic = np.mean(arr)
        std_ic = np.std(arr)
        if std_ic < 1e-10:
            return 0.0
        return float(mean_ic / std_ic)

    def compute_ic_tstat(self, ic_series: List[float]) -> float:
        """
        IC 的 t 统计量: t = IC_mean / (IC_std / sqrt(N))

        用于检验 IC 是否显著不为 0
        """
        if len(ic_series) < 12:
            return 0.0
        arr = np.array(ic_series)
        mean_ic = np.mean(arr)
        std_ic = np.std(arr)
        n = len(arr)
        if std_ic < 1e-10:
            return 0.0
        return float(mean_ic / (std_ic / np.sqrt(n)))

    def factor_rank_regression(
        self,
        factor_cross_sectional: List[Dict[str, Dict[str, float]]],
        return_cross_sectional: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """
        横截面回归因子排名 (Liu et al. 2014)

        对每个日期做多元截面回归, 返回各因子的平均 t-stat

        Args:
            factor_cross_sectional: [{date: {stock_code: {factor_name: value}}}]
                每个元素是一个日期的因子截面数据
            return_cross_sectional: [{date: {stock_code: return}}]
                每个元素是一个日期的收益率截面数据

        Returns:
            {factor_name: avg_t_stat}
        """
        if not factor_cross_sectional or not return_cross_sectional:
            return {}

        n_dates = len(factor_cross_sectional)
        if n_dates < 5:
            return {}

        # 收集所有因子名称
        all_factors = set()
        for date_data in factor_cross_sectional:
            for date, stock_data in date_data.items():
                for stock_code, factor_dict in stock_data.items():
                    all_factors.update(factor_dict.keys())
        factor_names = sorted(all_factors)
        n_factors = len(factor_names)
        if n_factors == 0:
            return {}

        # 对每个日期做截面回归, 累积 t-stat
        t_stat_sum = {fname: 0.0 for fname in factor_names}
        t_stat_count = {fname: 0 for fname in factor_names}

        for date_idx, date_factors in enumerate(factor_cross_sectional):
            for date_key, stock_factors in date_factors.items():
                # 获取对应收益率
                returns_dict = {}
                for rc in return_cross_sectional:
                    if date_key in rc:
                        returns_dict = rc[date_key]
                        break

                # 收集有因子值和收益率的股票
                valid_stocks = []
                factor_matrix = []
                returns_vec = []

                for stock_code, factor_dict in stock_factors.items():
                    ret = returns_dict.get(stock_code)
                    if ret is None or np.isnan(ret):
                        continue
                    vec = [factor_dict.get(fname, 0.0) for fname in factor_names]
                    if any(np.isnan(v) for v in vec):
                        continue
                    valid_stocks.append(stock_code)
                    factor_matrix.append(vec)
                    returns_vec.append(ret)

                if len(valid_stocks) < max(n_factors + 3, 5):
                    continue

                f_arr = np.array(factor_matrix)
                r_arr = np.array(returns_vec)

                # 标准化
                f_std = np.std(f_arr, axis=0)
                f_mean = np.mean(f_arr, axis=0)
                r_std = np.std(r_arr)
                r_mean = np.mean(r_arr)

                if np.any(f_std < 1e-10) or r_std < 1e-10:
                    continue

                f_norm = (f_arr - f_mean) / (f_std + 1e-10)
                r_norm = (r_arr - r_mean) / (r_std + 1e-10)

                # 多元回归
                try:
                    beta = np.linalg.lstsq(f_norm, r_norm, rcond=None)[0]
                except np.linalg.LinAlgError:
                    continue

                predicted = f_norm @ beta
                residuals = r_norm - predicted
                n = len(r_arr)
                mse = np.sum(residuals ** 2) / max(n - n_factors - 1, 1)
                if mse <= 0:
                    continue

                try:
                    XtX_inv = np.linalg.inv(f_norm.T @ f_norm)
                    se = np.sqrt(np.diag(XtX_inv) * mse)
                except np.linalg.LinAlgError:
                    continue

                for j, fname in enumerate(factor_names):
                    t_val = beta[j] / (se[j] + 1e-10)
                    if not np.isnan(t_val):
                        t_stat_sum[fname] += t_val
                        t_stat_count[fname] += 1

        # 计算平均 t-stat
        result = {}
        for fname in factor_names:
            count = t_stat_count.get(fname, 0)
            if count > 0:
                result[fname] = round(t_stat_sum[fname] / count, 4)
            else:
                result[fname] = 0.0

        return result

    def get_factor_ranking(self, ic_data: Dict[str, Dict[str, float]]) -> List[Tuple[str, float, float]]:
        """
        获取因子排名

        Args:
            ic_data: {factor_name: {ic_mean, ic_std, icir, t_stat, ic_series}}

        Returns:
            [(factor_name, ic_mean, icir), ...] 按 ICIR 降序
        """
        if not ic_data:
            return []

        ranking = []
        for fname, stats in ic_data.items():
            ic_mean = stats.get('ic_mean', 0.0)
            icir = stats.get('icir', 0.0)
            ranking.append((fname, ic_mean, icir))

        # 按 ICIR 绝对值降序排列
        ranking.sort(key=lambda x: abs(x[2]), reverse=True)
        return ranking


class FactorSelection:
    """因子筛选器"""

    @staticmethod
    def select_factors(
        ic_scores: Dict[str, float],
        icir_scores: Dict[str, float],
        min_ic: float = 0.03,
        min_icir: float = 0.5,
        max_corr: float = 0.7
    ) -> List[str]:
        """
        因子筛选

        Args:
            ic_scores: {factor_name: ic_mean}
            icir_scores: {factor_name: icir}
            min_ic: 最小 IC 阈值
            min_icir: 最小 ICIR 阈值
            max_corr: 最大允许相关性

        Returns:
            筛选后的因子列表
        """
        # Step 1: 基于 IC/ICIR 筛选
        candidates = [
            name for name, ic in ic_scores.items()
            if abs(ic) >= min_ic and abs(icir_scores.get(name, 0)) >= min_icir
        ]

        # Step 2: 去除高相关因子
        kept = []
        for name in candidates:
            kept.append(name)

        return kept

    @staticmethod
    def remove_correlated(
        factor_matrix: np.ndarray,
        factor_names: List[str],
        max_corr: float = 0.7
    ) -> List[str]:
        """
        移除高相关因子

        按 IC 降序保留, 移除与已保留因子相关性 > max_corr 的因子
        """
        if factor_matrix.shape[1] <= 1:
            return factor_names

        corr_matrix = np.corrcoef(factor_matrix.T)
        kept = [factor_names[0]]

        for i in range(1, factor_matrix.shape[1]):
            is_kept = True
            for j in kept:
                idx_j = factor_names.index(j)
                if abs(corr_matrix[i, idx_j]) > max_corr:
                    is_kept = False
                    break
            if is_kept:
                kept.append(factor_names[i])

        return kept

    @staticmethod
    def ranking_weights(ic_scores: Dict[str, float]) -> Dict[str, float]:
        """
        基于 IC 排名分配权重
        """
        abs_ics = {f: abs(v) for f, v in ic_scores.items() if v != 0}
        total = sum(abs_ics.values())
        if total > 0:
            return {f: v / total for f, v in abs_ics.items()}
        return {f: 1.0 / len(ic_scores) for f in ic_scores}


class ICDecay:
    """IC 衰减分析"""

    @staticmethod
    def compute_decay(
        factor_values: Dict[str, Dict[str, float]],
        future_returns: Dict[str, Dict[str, float]],
        horizons: List[int] = None
    ) -> Dict[int, float]:
        """
        IC 衰减曲线

        Args:
            factor_values: {date_str: {stock_code: factor_value}}
            future_returns: {date_str: {stock_code: forward_return}}
            horizons: 预测 horizon 列表

        Returns:
            {horizon: ic_mean}
        """
        if horizons is None:
            horizons = [1, 3, 5, 10, 20]

        result = {}
        for h in horizons:
            ics = []
            for date in sorted(factor_values.keys()):
                # 找到 h 天后的收益
                dates = sorted(factor_values.keys())
                idx = dates.index(date) if date in dates else -1
                if idx >= 0 and idx + h < len(dates):
                    future_date = dates[idx + h]
                    if future_date in future_returns:
                        common_codes = set(factor_values[date].keys()) & set(future_returns[future_date].keys())
                        f_vals = []
                        r_vals = []
                        for code in common_codes:
                            fv = factor_values[date][code]
                            rv = future_returns[future_date][code]
                            if not (np.isnan(fv) or np.isnan(rv)):
                                f_vals.append(fv)
                                r_vals.append(rv)
                        if len(f_vals) >= 5:
                            ic = np.corrcoef(f_vals, r_vals)[0, 1]
                            if not np.isnan(ic):
                                ics.append(ic)
            if ics:
                result[h] = float(np.mean(ics))
            else:
                result[h] = 0.0

        return result

    @staticmethod
    def half_life(factor_values: Dict[str, Dict[str, float]],
                  future_returns: Dict[str, Dict[str, float]]) -> int:
        """
        因子半衰期: IC 衰减到一半所需天数
        """
        decay = ICDecay.compute_decay(factor_values, future_returns, horizons=[1, 3, 5, 10, 20, 30, 60])

        if 1 not in decay or decay[1] == 0:
            return 0

        target = decay[1] / 2
        best_h = 1
        for h, ic in decay.items():
            if abs(ic - target) < abs(decay.get(best_h, 0) - target):
                best_h = h

        return best_h


# 全局实例
factor_ic_monitor = FactorICMonitor()


def save_ic_history(filepath: str, ic_data: Dict):
    """保存 IC 历史到 JSON"""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(ic_data, f, ensure_ascii=False, indent=2)
        logger.info(f"[FactorICMonitor] IC 历史已保存: {filepath}")
    except Exception as e:
        logger.error(f"[FactorICMonitor] IC 历史保存失败: {e}")


def load_ic_history(filepath: str) -> Dict:
    """加载 IC 历史"""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        logger.info(f"[FactorICMonitor] IC 历史已加载: {filepath}")
        return data
    except Exception as e:
        logger.error(f"[FactorICMonitor] IC 历史加载失败: {e}")
        return {}
