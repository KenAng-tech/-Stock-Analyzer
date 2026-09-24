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

# 延迟导入 SHAP，避免启动慢
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

# 09-22: 具体因子名 → 7 组名映射 (scheduler.current_weights 键形 = 7 组)。
# 键空间实锤自 alpha158_calculator 因子族 (69 具体名), 归类理由见行内。
# 无前缀匹配 = None 诚实跳过 (不硬映射, 防错误组降权)。
_IC_GROUP_PREFIX = (
    ('momentum_', 'momentum'), ('reversal_', 'momentum'),
    ('volatility_', 'volatility'),
    ('upside_volatility', 'volatility'), ('downside_volatility', 'volatility'),
    ('up_down_vol_ratio', 'volatility'),
    ('amihud_', 'volatility'),  # 非流动性 = 收益波动族
    ('return_autocorr', 'volatility'), ('return_kurtosis', 'volatility'),
    ('return_skewness', 'volatility'),
    ('volume_ratio_', 'volume'), ('volume_momentum_', 'volume'),
    ('volume_position_', 'volume'), ('volume_price_corr_', 'volume'),
    ('rsi_', 'technical'), ('macd', 'technical'), ('boll_position', 'technical'),
    ('atr_', 'technical'), ('price_position_', 'technical'),
)


def _factor_to_group(factor_name: str) -> Optional[str]:
    """具体因子名 → 7 组名 (前缀表, 无匹配 None = 诚实跳过)。"""
    for prefix, group in _IC_GROUP_PREFIX:
        if factor_name.startswith(prefix):
            return group
    return None


class FactorICMonitor:
    """因子 IC/ICIR 持续监控

    支持两种模式:
    1. 单截面 IC: 单日的截面相关系数 (噪声大, 仅作参考)
    2. 多日 ICIR: 多日 IC 序列的均值/标准差 (稳健, 推荐)
    """

    def __init__(self):
        self._ic_history: Dict[str, List[float]] = {}
        # Alpha 衰减监控: {stock_code: {factor_name: [(date, factor_val, return_val), ...]}}
        self._factor_returns: Dict[str, Dict[str, List[Tuple[str, float, float]]]] = {}
        # 因子权重调度器 (由 Alpha 衰减闭环激活)
        self._weight_scheduler = None
        try:
            from modules.factor_weight_scheduler import get_factor_weight_scheduler
            self._weight_scheduler = get_factor_weight_scheduler()
            logger.info("[FactorIC] 因子权重调度器已激活")
        except Exception as e:
            logger.warning(f"[FactorIC] 因子权重调度器初始化失败: {e}")
        # 09-22 持久化 (decision_history_store 教训形): 样本从台账恢复 → 重启不清零
        # (此前 _factor_returns 纯内存, 每次重启 IC 统计从 0 重攒; 23:10 链形制)
        try:
            from modules.fusion_ledger import get_fusion_ledger
            _led = get_fusion_ledger()
            if _led is not None:
                _restored = _led.load_factor_records()
                if _restored:
                    self._factor_returns = _restored
                    logger.info(f"[FactorIC] IC 样本台账恢复: {len(_restored)} 票")
        except Exception as e:
            logger.debug(f"[FactorIC] 台账恢复跳过 (内存模式): {e}")

    def record_factor_return(
        self,
        stock_code: str,
        factors: Dict[str, float],
        return_value: float,
        date: Optional[str] = None,
    ) -> bool:
        """
        记录一次因子值 + 实际收益率，用于 IC 衰减监控

        每次 analysis_engine.quantitative_prediction() 后调用，
        将刚计算的 Alpha158 因子值 + 实际收益喂入监控。

        Args:
            stock_code: 股票代码
            factors: {factor_name: factor_value, ...}
            return_value: 实际收益率 (如次日涨跌幅)
            date: 可选日期字符串 (默认今天)

        Returns:
            True 如果触发了衰减告警
        """
        date = date or datetime.now().strftime('%Y-%m-%d')

        if stock_code not in self._factor_returns:
            self._factor_returns[stock_code] = {}

        # 同键 (factor, date) 内存防重 (09-22 实锤: 回填/重跑双喂同历史对 →
        # 重复样本曾致 IC 失真 35→0 跳变; 台账侧 OR IGNORE 已防, 内存侧补)
        for fname, fval in factors.items():
            lst = self._factor_returns[stock_code].setdefault(fname, [])
            if not any(r[0] == date and r[1] == fval for r in lst[-8:]):
                lst.append((date, fval, return_value))

        # IC 样本落盘 (09-22): 整批进台账 (INSERT OR IGNORE 防重) → 重启不清零
        try:
            from modules.fusion_ledger import get_fusion_ledger
            _led = get_fusion_ledger()
            if _led is not None:
                _led.append_factor_records(
                    stock_code, date,
                    {f: (v, return_value) for f, v in factors.items()})
        except Exception as e:
            logger.debug(f"[FactorIC] IC 样本落盘失败 (不拖监控): {e}")

        # 如果缓冲区足够大，计算 IC 并检查衰减
        if len(self._factor_returns[stock_code]) >= 5:
            alerts = self.get_ic_decay_alerts_from_returns(stock_code)
            if alerts:
                logger.warning(f"[FactorIC] 因子衰减告警 ({stock_code}): {len(alerts)} 个因子")
                # 衰减→降权闭环 (09-22 实修): 旧形喂具体因子名 ('rsi_7' 等 69 个)
                # 进 scheduler 7 组键 = 100% warning 空转。修 = 具体名→组名映射 +
                # 同组 ≥2 告警才触发 (单因子衰减≠整组衰减) + type 判级
                # (critical 0.5 / warning 0.75)。无映射名跳过。flag 可关。
                if self._weight_scheduler and \
                        os.environ.get('DECAY_AUTO_WEIGHT', '1') == '1':
                    grp: Dict[str, List[float]] = {}
                    for alert in alerts:
                        g = _factor_to_group(alert.get('factor', ''))
                        if g:
                            grp.setdefault(g, []).append(
                                0.5 if alert.get('type') == 'critical' else 0.75)
                    for g, mults in grp.items():
                        if len(mults) >= 2:
                            m = min(mults)
                            logger.warning(f"[FactorIC] 组衰减 ({stock_code}) {g}: "
                                           f"{len(mults)} 因子 IC 衰减 → 权重 ×{m}")
                            self._weight_scheduler.reduce_factor_weight(g, m)
                return True

        return False

    def get_ic_decay_alerts_from_returns(
        self,
        stock_code: str,
        min_samples: int = 20,
    ) -> List[Dict]:
        """
        从因子+收益记录中计算 IC 衰减告警

        对每个因子，计算因子值与实际收益的 Pearson 相关系数 (IC)，
        然后检查是否低于衰减阈值。

        Args:
            stock_code: 股票代码
            min_samples: 最小样本数

        Returns:
            告警列表
        """
        alerts = []
        stock_data = self._factor_returns.get(stock_code, {})

        for fname, records in stock_data.items():
            if len(records) < min_samples:
                continue

            # 提取因子值和收益率
            f_vals = [r[1] for r in records]
            r_vals = [r[2] for r in records]

            f_arr = np.array(f_vals)
            r_arr = np.array(r_vals)

            # 计算 IC (Pearson 相关系数)
            f_std = float(np.std(f_arr))
            r_std = float(np.std(r_arr))

            if f_std < 1e-10 or r_std < 1e-10:
                continue

            ic = float(np.corrcoef(f_arr, r_arr)[0, 1])
            if np.isnan(ic):
                continue

            # 检查衰减阈值
            if abs(ic) < IC_DECAY_THRESHOLDS['critical']:
                alerts.append({
                    'type': 'critical',
                    'factor': fname,
                    'ic_mean': round(ic, 4),
                    'n_samples': len(records),
                    'message': f'因子 {fname} IC={ic:.4f} 低于严重阈值',
                })
            elif abs(ic) < IC_DECAY_THRESHOLDS['warning']:
                alerts.append({
                    'type': 'warning',
                    'factor': fname,
                    'ic_mean': round(ic, 4),
                    'n_samples': len(records),
                    'message': f'因子 {fname} IC={ic:.4f} 低于警告阈值',
                })

        return alerts

    def compute_cross_sectional_ic(
        self, factor_values: Dict[str, Dict[str, float]],
        future_returns: Dict[str, Dict[str, float]]
    ) -> float:
        """
        截面 IC (Pearson 相关系数) — 多日截面 + ICIR

        真正的截面 IC 应该跨多个交易日计算，然后取 IC 序列的均值/ICIR。
        单天截面 IC 完全是噪声，不可靠。

        Returns:
            {'ic': float, 'icir': float, 'n_days': int, 'ic_series': List[float]}
        """
        common_dates = sorted(set(factor_values.keys()) & set(future_returns.keys()))
        if len(common_dates) < 2:
            return {'ic': 0.0, 'icir': 0.0, 'n_days': len(common_dates), 'ic_series': []}

        ic_series = []
        for date in common_dates:
            ic = self._single_day_ic(factor_values.get(date, {}), future_returns.get(date, {}))
            if ic is not None:
                ic_series.append(ic)

        if len(ic_series) < 2:
            mean_ic = ic_series[0] if ic_series else 0.0
            return {'ic': mean_ic, 'icir': 0.0, 'n_days': len(common_dates), 'ic_series': ic_series}

        mean_ic = float(np.mean(ic_series))
        std_ic = float(np.std(ic_series))
        icir = mean_ic / std_ic if std_ic > 1e-10 else 0.0

        return {'ic': mean_ic, 'icir': icir, 'n_days': len(common_dates), 'ic_series': ic_series}

    def _single_day_ic(self, factors: Dict[str, float], returns: Dict[str, float]) -> Optional[float]:
        """单日截面 IC"""
        factors_list = []
        returns_list = []

        for code in set(factors.keys()) & set(returns.keys()):
            f, r = factors[code], returns[code]
            try:
                f, r = float(f), float(r)
            except (ValueError, TypeError):
                continue
            if not (np.isnan(f) or np.isnan(r) or np.isinf(f) or np.isinf(r)):
                factors_list.append(f)
                returns_list.append(r)

        if len(factors_list) < 5:
            return None

        f_arr = np.array(factors_list)
        r_arr = np.array(returns_list)

        f_std = np.std(f_arr)
        r_std = np.std(r_arr)

        if f_std < 1e-10 or r_std < 1e-10:
            return None

        corr = np.corrcoef(f_arr, r_arr)[0, 1]
        if np.isnan(corr) or np.isinf(corr):
            return None
        return float(corr)

    def compute_rank_ic(
        self, factor_values: Dict[str, Dict[str, float]],
        future_returns: Dict[str, Dict[str, float]]
    ) -> float:
        """
        秩 IC (Spearman 相关系数, 更稳健) — 多日截面 + ICIR
        """
        result = self.compute_cross_sectional_ic(factor_values, future_returns)
        # 复用单截面逻辑，但用秩相关
        common_dates = sorted(set(factor_values.keys()) & set(future_returns.keys()))
        if len(common_dates) < 2:
            return 0.0

        ic_series = []
        for date in common_dates:
            ic = self._single_day_rank_ic(
                factor_values.get(date, {}), future_returns.get(date, {})
            )
            if ic is not None:
                ic_series.append(ic)

        if not ic_series:
            return 0.0

        mean_ic = float(np.mean(ic_series))
        std_ic = float(np.std(ic_series)) if len(ic_series) > 1 else 1.0
        return mean_ic / std_ic if std_ic > 1e-10 else mean_ic

    def _single_day_rank_ic(self, factors: Dict[str, float], returns: Dict[str, float]) -> Optional[float]:
        """单日截面秩 IC (Spearman)"""
        factors_list = []
        returns_list = []

        for code in set(factors.keys()) & set(returns.keys()):
            f, r = factors[code], returns[code]
            try:
                f, r = float(f), float(r)
            except (ValueError, TypeError):
                continue
            if not (np.isnan(f) or np.isnan(r) or np.isinf(f) or np.isinf(r)):
                factors_list.append(f)
                returns_list.append(r)

        if len(factors_list) < 5:
            return None

        f_arr = np.array(factors_list)
        r_arr = np.array(returns_list)

        f_rank = f_arr.argsort().argsort() / (len(f_arr) - 1)
        r_rank = r_arr.argsort().argsort() / (len(r_arr) - 1)

        corr = np.corrcoef(f_rank, r_rank)[0, 1]
        if np.isnan(corr) or np.isinf(corr):
            return None
        return float(corr)

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

    def compute_shap_importance(
        self,
        factor_matrix: Dict[str, Dict[str, float]],
        returns: Dict[str, float],
        model_type: str = 'lightgbm',
    ) -> Dict[str, float]:
        """
        基于 IC (Information Coefficient) 的因子重要性

        计算每个因子与未来收益率的 Spearman 秩相关系数绝对值作为重要性。
        这是因子分析中最标准的重要性度量，比 ML 方法更稳定可靠。

        Args:
            factor_matrix: {date_key: {factor_name: value}}
            returns: {date_key: return}

        Returns:
            {factor_name: |IC|} 因子重要性排序
        """
        try:
            # 收集每个因子的时间序列
            factor_series: Dict[str, List[float]] = {}
            for date_key, factors in factor_matrix.items():
                if date_key not in returns:
                    continue
                for fname, val in factors.items():
                    if fname not in factor_series:
                        factor_series[fname] = []
                    try:
                        factor_series[fname].append(float(val))
                    except (ValueError, TypeError):
                        pass

            ret_series = [returns[d] for d in factor_matrix.keys() if d in returns]

            if len(ret_series) < 10:
                return {}

            importances = {}
            for fname, fvals in factor_series.items():
                if len(fvals) != len(ret_series):
                    continue
                # Spearman 秩相关
                try:
                    from scipy.stats import spearmanr
                    corr, _ = spearmanr(fvals, ret_series)
                    importances[fname] = round(float(abs(corr)), 6) if not np.isnan(corr) else 0.0
                except Exception:
                    importances[fname] = 0.0

            # 按重要性排序
            sorted_factors = sorted(importances.items(), key=lambda x: x[1], reverse=True)

            logger.info(
                f"[Importance] 因子重要性计算完成 ({len(sorted_factors)} 个因子, "
                f"{len(ret_series)} 样本)"
            )

            return dict(sorted_factors)

        except Exception as e:
            logger.error(f"[Importance] 因子重要性计算失败: {e}")
            return {}

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


class ICQQPlot:
    """
    IC QQ-图数据生成 (Alphalens 风格)

    用于检验 IC 分布是否服从正态分布。
    QQ-图: 分位数-分位数图，比较样本分位数与理论分位数。
    """

    @staticmethod
    def compute_qq_data(ic_series: List[float], n_points: int = 50) -> Dict[str, List[float]]:
        """
        计算 QQ-图数据

        Args:
            ic_series: IC 时间序列
            n_points: QQ-图点数

        Returns:
            {
                'theoretical_quantiles': [float],  # 理论正态分位数
                'sample_quantiles': [float],       # 样本分位数
            }
        """
        if len(ic_series) < 5:
            return {'theoretical_quantiles': [], 'sample_quantiles': []}

        arr = np.array(ic_series)
        mean = np.mean(arr)
        std = np.std(arr) + 1e-10

        # 样本分位数
        sorted_vals = np.sort(arr)
        probs = np.linspace(1 / (2 * len(sorted_vals)), 1 - 1 / (2 * len(sorted_vals)), n_points)
        sample_q = np.percentile(arr, probs * 100)

        # 理论正态分位数
        from math import sqrt, log
        def _norm_ppf(p: float) -> float:
            """近似正态分位数函数"""
            if p <= 0 or p >= 1:
                return 0.0
            # Rational approximation
            if p < 0.5:
                t = sqrt(-2 * log(p))
                z = -(t - (2.515517 + 0.802853 * t + 0.010328 * t ** 2) /
                        (1 + 1.432788 * t + 0.189269 * t ** 2 + 0.001308 * t ** 3))
            else:
                t = sqrt(-2 * log(1 - p))
                z = t - (2.515517 + 0.802853 * t + 0.010328 * t ** 2) / \
                      (1 + 1.432788 * t + 0.189269 * t ** 2 + 0.001308 * t ** 3)
            return z * std + mean

        theoretical_q = [_norm_ppf(float(p)) for p in probs]

        return {
            'theoretical_quantiles': [round(float(x), 6) for x in theoretical_q],
            'sample_quantiles': [round(float(x), 6) for x in sample_q],
        }

    @staticmethod
    def normality_test(ic_series: List[float]) -> Dict[str, float]:
        """
        正态性检验 (Jarque-Bera 检验)

        Returns:
            {
                'skewness': float,
                'kurtosis': float,
                'jb_stat': float,
                'jb_pvalue': float,  # 近似
                'is_normal': bool,   # p > 0.05 → 正态
            }
        """
        if len(ic_series) < 20:
            return {'skewness': 0, 'kurtosis': 0, 'jb_stat': 0, 'jb_pvalue': 1.0, 'is_normal': True}

        arr = np.array(ic_series)
        n = len(arr)
        skew = float(pd.Series(arr).skew())
        kurt = float(pd.Series(arr).kurtosis())

        # Jarque-Bera 统计量
        jb_stat = n * (skew ** 2 / 6 + kurt ** 2 / 24)

        # 近似 p-value (chi-squared with 2 df)
        jb_pvalue = float(np.exp(-jb_stat / 2))
        jb_pvalue = min(1.0, max(0.0, jb_pvalue))

        return {
            'skewness': round(skew, 4),
            'kurtosis': round(kurt, 4),
            'jb_stat': round(jb_stat, 4),
            'jb_pvalue': round(jb_pvalue, 4),
            'is_normal': jb_pvalue > 0.05,
        }


class FactorTurnover:
    """
    因子换手率分析 (Alphalens 风格)

    因子换手率 = 权重变化绝对值之和 / 2

    用于评估因子组合的交易成本。
    """

    @staticmethod
    def compute_turnover(weight_series: List[Dict[str, float]],
                         top_n: int = 10) -> Dict[str, float]:
        """
        计算因子换手率

        Args:
            weight_series: [{stock_code: weight}, ...] 每期权重
            top_n: 仅计算 Top N 权重的换手率

        Returns:
            {
                'mean_turnover': float,       # 平均换手率
                'std_turnover': float,        # 换手率标准差
                'max_turnover': float,        # 最大换手率
                'turnover_pct_0_5': float,    # 换手率 < 5% 的比例
                'turnover_pct_5_10': float,   # 换手率 5-10% 的比例
                'turnover_pct_10_20': float,  # 换手率 10-20% 的比例
                'turnover_pct_gt_20': float,  # 换手率 > 20% 的比例
            }
        """
        if len(weight_series) < 2:
            return {
                'mean_turnover': 0, 'std_turnover': 0, 'max_turnover': 0,
                'turnover_pct_0_5': 1.0, 'turnover_pct_5_10': 0,
                'turnover_pct_10_20': 0, 'turnover_pct_gt_20': 0,
            }

        turnovers = []
        for i in range(1, len(weight_series)):
            prev = weight_series[i - 1]
            curr = weight_series[i]

            # Top N 权重
            if top_n > 0:
                prev_top = dict(sorted(prev.items(), key=lambda x: x[1], reverse=True)[:top_n])
                curr_top = dict(sorted(curr.items(), key=lambda x: x[1], reverse=True)[:top_n])
            else:
                prev_top = prev
                curr_top = curr

            all_codes = set(prev_top.keys()) | set(curr_top.keys())
            turnover = 0
            for code in all_codes:
                w_prev = prev_top.get(code, 0)
                w_curr = curr_top.get(code, 0)
                turnover += abs(w_curr - w_prev)

            turnover /= 2  # 换手率定义
            turnovers.append(turnover)

        if not turnovers:
            return {
                'mean_turnover': 0, 'std_turnover': 0, 'max_turnover': 0,
                'turnover_pct_0_5': 1.0, 'turnover_pct_5_10': 0,
                'turnover_pct_10_20': 0, 'turnover_pct_gt_20': 0,
            }

        arr = np.array(turnovers)

        # 换手率区间分布
        n = len(arr)
        return {
            'mean_turnover': round(float(np.mean(arr)), 6),
            'std_turnover': round(float(np.std(arr)), 6),
            'max_turnover': round(float(np.max(arr)), 6),
            'turnover_pct_0_5': round(float(np.sum(arr < 0.05)) / n, 4),
            'turnover_pct_5_10': round(float(np.sum((arr >= 0.05) & (arr < 0.10))) / n, 4),
            'turnover_pct_10_20': round(float(np.sum((arr >= 0.10) & (arr < 0.20))) / n, 4),
            'turnover_pct_gt_20': round(float(np.sum(arr >= 0.20)) / n, 4),
        }

    @staticmethod
    def compute_rank_turnover(rank_series: List[Dict[str, int]]) -> float:
        """
        排名换手率: 排名变化绝对值 / 总排名数

        Args:
            rank_series: [{stock_code: rank}, ...]

        Returns:
            平均排名换手率
        """
        if len(rank_series) < 2:
            return 0.0

        rank_changes = []
        for i in range(1, len(rank_series)):
            prev = rank_series[i - 1]
            curr = rank_series[i]
            all_codes = set(prev.keys()) | set(curr.keys())
            if not all_codes:
                continue
            total_change = sum(abs(curr.get(c, 0) - prev.get(c, 0)) for c in all_codes)
            avg_change = total_change / len(all_codes)
            rank_changes.append(avg_change)

        return float(np.mean(rank_changes)) if rank_changes else 0.0


class LongShortAnalyzer:
    """
    多空分析 (Alphalens 风格)

    将股票按因子值分成分位数组合 (如 Top/Bottom 20%)，
    分析多空组合的表现。
    """

    @staticmethod
    def compute_long_short(factor_values: Dict[str, float],
                           future_returns: Dict[str, float],
                           n_groups: int = 5) -> Dict[str, float]:
        """
        计算多空组合表现

        Args:
            factor_values: {stock_code: factor_value}
            future_returns: {stock_code: forward_return}
            n_groups: 分组数 (默认 5 组: Q1-Q5)

        Returns:
            {
                'long_return': float,          # Top 组平均收益
                'short_return': float,         # Bottom 组平均收益
                'long_short_return': float,    # 多空收益差
                'long_short_sharpe': float,    # 多空 Sharpe (近似)
                'win_rate': float,             # 多空收益为正的频率
                'information_ratio': float,    # 信息比率
            }
        """
        common_codes = set(factor_values.keys()) & set(future_returns.keys())
        if len(common_codes) < n_groups * 2:
            return {
                'long_return': 0, 'short_return': 0, 'long_short_return': 0,
                'long_short_sharpe': 0, 'win_rate': 0.5, 'information_ratio': 0,
            }

        codes = list(common_codes)
        f_vals = np.array([factor_values[c] for c in codes])
        r_vals = np.array([future_returns[c] for c in codes])

        # 按因子值分组
        n = len(codes)
        group_size = max(1, n // n_groups)
        sorted_indices = np.argsort(f_vals)

        results = {}
        for g in range(n_groups):
            start = g * group_size
            end = start + group_size if g < n_groups - 1 else n
            group_returns = r_vals[sorted_indices[start:end]]
            results[f'group_{g + 1}_return'] = float(np.mean(group_returns))

        # Top vs Bottom
        top_returns = r_vals[sorted_indices[-group_size:]]
        bottom_returns = r_vals[sorted_indices[:group_size]]

        long_ret = float(np.mean(top_returns))
        short_ret = float(np.mean(bottom_returns))
        ls_ret = long_ret - short_ret

        # 多空 Sharpe (近似)
        ls_std = float(np.std(top_returns - bottom_returns) + 1e-10)
        ls_sharpe = ls_ret / ls_std

        # 信息比率
        ir = ls_ret / (np.std(r_vals) + 1e-10)

        return {
            'long_return': round(long_ret, 6),
            'short_return': round(short_ret, 6),
            'long_short_return': round(ls_ret, 6),
            'long_short_sharpe': round(ls_sharpe, 4),
            'win_rate': round(float(np.mean(r_vals > 0)), 4),
            'information_ratio': round(ir, 4),
            **{f'group_{g + 1}_return': round(results.get(f'group_{g + 1}_return', 0), 6)
               for g in range(n_groups)},
        }

    @staticmethod
    def compute_cumulative_return(factor_values_list: List[Dict[str, float]],
                                   returns_list: List[Dict[str, float]],
                                   n_groups: int = 5) -> List[float]:
        """
        计算分位数组合累计收益

        Args:
            factor_values_list: [{stock_code: factor_value}, ...] 每期截面
            returns_list: [{stock_code: return}, ...] 每期收益
            n_groups: 分组数

        Returns:
            [cumulative_return_per_group_1, ..., cumulative_return_per_group_n]
        """
        if len(factor_values_list) < 2:
            return [0.0] * n_groups

        n_dates = len(factor_values_list)
        group_cum_returns = {g: [1.0] for g in range(1, n_groups + 1)}

        for i in range(1, n_dates):
            fv = factor_values_list[i - 1]
            ret = returns_list[i - 1] if i - 1 < len(returns_list) else {}

            common_codes = set(fv.keys()) & set(ret.keys())
            if len(common_codes) < n_groups * 2:
                for g in range(1, n_groups + 1):
                    group_cum_returns[g].append(group_cum_returns[g][-1])
                continue

            codes = list(common_codes)
            f_vals = np.array([fv[c] for c in codes])
            r_vals = np.array([ret[c] for c in codes])

            sorted_indices = np.argsort(f_vals)
            group_size = max(1, len(codes) // n_groups)

            for g in range(n_groups):
                start = g * group_size
                end = start + group_size if g < n_groups - 1 else len(codes)
                group_ret = np.mean(r_vals[sorted_indices[start:end]])
                prev_cum = group_cum_returns[g + 1][-1]
                group_cum_returns[g + 1].append(prev_cum * (1 + group_ret))

        # 返回最后一期累计值
        return [round(group_cum_returns[g][-1] - 1, 6) for g in range(1, n_groups + 1)]


class FactorGroupIC:
    """
    因子组 IC 分析 (Alphalens 风格)

    按因子类别计算组内平均 IC。
    """

    @staticmethod
    def compute_group_ic(factor_data: Dict[str, Dict[str, float]],
                         future_returns: Dict[str, float],
                         factor_groups: Dict[str, List[str]]) -> Dict[str, float]:
        """
        计算每个因子组的平均 IC

        Args:
            factor_data: {date_str: {stock_code: {factor_name: value}}}
            future_returns: {date_str: {stock_code: return}}
            factor_groups: {group_name: [factor_names]}

        Returns:
            {group_name: {ic_mean, ic_std, icir, n_dates}}
        """
        result = {}

        for group_name, factor_names in factor_groups.items():
            ics = []

            for date in sorted(set(factor_data.keys()) & set(future_returns.keys())):
                fd = factor_data[date]
                fr = future_returns.get(date, {})

                # 收集该组所有因子的截面数据
                all_f = []
                all_r = []
                for code in set(fd.keys()) & set(fr.keys()):
                    for fname in factor_names:
                        if fname in fd[code] and not np.isnan(fd[code][fname]):
                            all_f.append(fd[code][fname])
                            all_r.append(fr[code])
                            break  # 每个股票只计一次

                if len(all_f) >= 10:
                    f_arr = np.array(all_f)
                    r_arr = np.array(all_r)
                    f_std = np.std(f_arr)
                    r_std = np.std(r_arr)
                    if f_std > 1e-10 and r_std > 1e-10:
                        ic = np.corrcoef(f_arr, r_arr)[0, 1]
                        if not np.isnan(ic):
                            ics.append(ic)

            if ics:
                arr = np.array(ics)
                result[group_name] = {
                    'ic_mean': round(float(np.mean(arr)), 6),
                    'ic_std': round(float(np.std(arr)), 6),
                    'icir': round(float(np.mean(arr) / (np.std(arr) + 1e-10)), 4),
                    'n_dates': len(ics),
                }
            else:
                result[group_name] = {
                    'ic_mean': 0, 'ic_std': 0, 'icir': 0, 'n_dates': 0,
                }

        return result


# ── IC 衰减分析 ────────────────────────────────────────────────────

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
factor_turnover = FactorTurnover()
long_short_analyzer = LongShortAnalyzer()
factor_group_ic = FactorGroupIC()

# ── IC Decay 主动告警 ──────────────────────────────────────────

# IC 衰减告警阈值
IC_DECAY_THRESHOLDS = {
    'warning': 0.03,      # IC 均值 < 0.03 时警告
    'critical': 0.01,     # IC 均值 < 0.01 时严重
    'icir_warning': 0.5,  # ICIR < 0.5 时警告
    'icir_critical': 0.3, # ICIR < 0.3 时严重
}


def get_ic_decay_alerts(ic_data: Dict[str, List[float]]) -> List[Dict]:
    """
    检查 IC 衰减告警

    Args:
        ic_data: {factor_name: [ic_values]}

    Returns:
        告警列表
    """
    alerts = []

    for factor_name, ic_series in ic_data.items():
        if len(ic_series) < 12:
            continue

        arr = np.array(ic_series)
        mean_ic = float(np.mean(arr))
        std_ic = float(np.std(arr))
        icir = mean_ic / std_ic if std_ic > 1e-10 else 0.0

        # 检查 IC 衰减
        if mean_ic < IC_DECAY_THRESHOLDS['critical']:
            alerts.append({
                'type': 'critical',
                'factor': factor_name,
                'metric': 'ic_mean',
                'value': round(mean_ic, 4),
                'threshold': IC_DECAY_THRESHOLDS['critical'],
                'message': f'因子 {factor_name} IC 均值 {mean_ic:.4f} 低于严重阈值 {IC_DECAY_THRESHOLDS["critical"]}',
                'action': '建议停用或重新校准该因子',
            })
        elif mean_ic < IC_DECAY_THRESHOLDS['warning']:
            alerts.append({
                'type': 'warning',
                'factor': factor_name,
                'metric': 'ic_mean',
                'value': round(mean_ic, 4),
                'threshold': IC_DECAY_THRESHOLDS['warning'],
                'message': f'因子 {factor_name} IC 均值 {mean_ic:.4f} 低于警告阈值 {IC_DECAY_THRESHOLDS["warning"]}',
                'action': '建议降低该因子权重',
            })

        # 检查 ICIR
        if icir < IC_DECAY_THRESHOLDS['icir_critical']:
            alerts.append({
                'type': 'critical',
                'factor': factor_name,
                'metric': 'icir',
                'value': round(icir, 4),
                'threshold': IC_DECAY_THRESHOLDS['icir_critical'],
                'message': f'因子 {factor_name} ICIR {icir:.4f} 低于严重阈值',
                'action': 'IC 不稳定，建议停用',
            })
        elif icir < IC_DECAY_THRESHOLDS['icir_warning']:
            alerts.append({
                'type': 'warning',
                'factor': factor_name,
                'metric': 'icir',
                'value': round(icir, 4),
                'threshold': IC_DECAY_THRESHOLDS['icir_warning'],
                'message': f'因子 {factor_name} ICIR {icir:.4f} 低于警告阈值',
                'action': '建议监控 IC 稳定性',
            })

    # 按严重程度排序
    alerts.sort(key=lambda x: (0 if x['type'] == 'critical' else 1, x['value']))
    return alerts


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


# ── 模块级单例 ──────────────────────────────────────────────
# 避免每次请求新建实例导致缓存失效
factor_ic_monitor = FactorICMonitor()
