#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
因子监控模块 — Factor Monitor (2026-07-24)

为因子分析提供监控 API:
1. IC Decay 曲线 — 因子 IC 随持有期的衰减
2. 因子相关性矩阵 — 因子间相关性热力图数据
3. 衰减警告 — IC 低于阈值或衰减过快的因子
4. 滚动 IC 趋势 — 因子 IC 随时间的变化趋势

后端依赖:
- FactorICMonitor: 截面 IC 计算
- ICDecayAnalyzer: IC Decay 分析
- Alpha158/Alpha360: 因子值计算

使用方式:
    from modules.factors.factor_monitor import factor_monitor
    result = factor_monitor.get_ic_decay(stock_code)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field

from modules.logger import logger

# 延迟导入，避免启动慢
try:
    from modules.factors.factor_ic_monitor import FactorICMonitor, ICDecay
    HAS_FACTOR_IC = True
except ImportError:
    HAS_FACTOR_IC = False
    FactorICMonitor = None
    ICDecay = None

try:
    from modules.factor_mining import ICDecayAnalyzer
    HAS_IC_DECAY_ANALYZER = True
except ImportError:
    HAS_IC_DECAY_ANALYZER = False
    ICDecayAnalyzer = None

try:
    from modules.adwin import ADWIN
    HAS_ADWIN = True
except ImportError:
    HAS_ADWIN = False
    ADWIN = None

try:
    from modules.concept_drift_detector import ConceptDriftDetector
    HAS_DRIFT_DETECTOR = True
except ImportError:
    HAS_DRIFT_DETECTOR = False
    ConceptDriftDetector = None


# ── 数据结构 ──────────────────────────────────────────────────────────

@dataclass
class FactorICData:
    """因子 IC 数据"""
    name: str
    ic_mean: float = 0.0
    ic_std: float = 0.0
    icir: float = 0.0
    t_stat: float = 0.0
    rank_ic_mean: float = 0.0
    n_obs: int = 0
    decay_stability: str = "unknown"  # stable/moderate/unstable


@dataclass
class DecayWarning:
    """衰减警告"""
    factor_name: str
    warning_type: str  # low_ic / fast_decay / unstable
    severity: str  # low/medium/high
    message: str
    current_ic: float = 0.0
    threshold: float = 0.0


@dataclass
class FactorCorrelation:
    """因子相关性"""
    name_a: str
    name_b: str
    correlation: float


@dataclass
class FactorMonitorResult:
    """因子监控结果"""
    stock_code: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    ic_decay: Dict[str, Dict] = field(default_factory=dict)
    multi_scale_ic: Dict[str, Dict] = field(default_factory=dict)  # 60D/120D/252D
    factor_ranking: List[Dict] = field(default_factory=list)
    warnings: List[Dict] = field(default_factory=list)
    rolling_ic: Dict[str, List[float]] = field(default_factory=dict)
    correlation_matrix: List[List[float]] = field(default_factory=list)
    factor_names: List[str] = field(default_factory=list)
    half_life: int = 0
    decay_interpretation: str = ""
    drift_consensus: Dict = field(default_factory=dict)  # ADWIN/Page-Hinkley/DDM 共识
    success: bool = True
    error: str = ""


# ── 因子监控器 ───────────────────────────────────────────────────────

class FactorMonitor:
    """
    因子监控器 — 提供因子分析 API

    功能:
    1. IC Decay 曲线: 计算因子 IC 随持有期的衰减
    2. 因子相关性矩阵: 计算因子间相关性
    3. 衰减警告: 检测 IC 过低或衰减过快的因子
    4. 滚动 IC 趋势: 计算因子 IC 随时间的变化
    """

    def __init__(
        self,
        ic_low_threshold: float = 0.02,
        decay_warning_threshold: float = 0.5,
        correlation_high_threshold: float = 0.7,
    ):
        """
        Args:
            ic_low_threshold: IC 低于此值触发警告
            decay_warning_threshold: IC 衰减到此比例触发警告
            correlation_high_threshold: 相关性高于此值标记为高相关
        """
        self.ic_low_threshold = ic_low_threshold
        self.decay_warning_threshold = decay_warning_threshold
        self.correlation_high_threshold = correlation_high_threshold

        self._ic_history: Dict[str, List[float]] = {}  # 因子名 → IC 历史
        self._warnings: List[DecayWarning] = []

    def compute_ic_decay(
        self,
        factor_values: Dict[str, Dict[str, float]],
        future_returns: Dict[str, Dict[str, float]],
        horizons: List[int] = None,
    ) -> Dict:
        """
        计算 IC Decay 曲线

        Args:
            factor_values: {date_str: {stock_code: {factor_name: value}}}
            future_returns: {date_str: {stock_code: return}}
            horizons: 持有期列表

        Returns:
            {
                'decay': {horizon: ic_mean},
                'half_life': int,
                'interpretation': str,
            }
        """
        if horizons is None:
            horizons = [1, 3, 5, 10, 20]

        # 使用 ICDecay.compute_decay
        try:
            if ICDecay is not None:
                decay = ICDecay.compute_decay(factor_values, future_returns, horizons)
            else:
                # Fallback: 手动计算
                decay = self._compute_decay_fallback(factor_values, future_returns, horizons)
        except (TypeError, ValueError):
            # NumPy isnan 不支持 Python float 类型时 fallback
            decay = self._compute_decay_fallback(factor_values, future_returns, horizons)

        # 计算半衰期
        half_life = self._estimate_half_life(decay)

        # 解释
        interpretation = self._interpret_decay(half_life, decay)

        return {
            'decay': {str(k): round(v, 4) for k, v in decay.items()},
            'half_life': half_life,
            'decay_rate': round(0.5 / max(half_life, 1), 4),
            'interpretation': interpretation,
        }

    def _compute_decay_fallback(
        self,
        factor_values: Dict[str, Dict[str, float]],
        future_returns: Dict[str, Dict[str, float]],
        horizons: List[int],
    ) -> Dict[int, float]:
        """手动计算 IC Decay (当 ICDecay 不可用时)"""
        result = {}
        dates = sorted(factor_values.keys())

        for h in horizons:
            ics = []
            for i in range(len(dates) - h):
                date = dates[i]
                future_date = dates[i + h]
                if future_date not in future_returns:
                    continue

                common = set(factor_values[date].keys()) & set(future_returns[future_date].keys())
                f_vals, r_vals = [], []
                for code in common:
                    fv = factor_values[date][code]
                    rv = future_returns[future_date][code]
                    try:
                        fv, rv = float(fv), float(rv)
                        if not (np.isnan(fv) or np.isnan(rv)) and not (fv == float('inf') or fv == float('-inf') or rv == float('inf') or rv == float('-inf')):
                            f_vals.append(fv)
                            r_vals.append(rv)
                    except (ValueError, TypeError):
                        continue

                if len(f_vals) >= 10:
                    ic = np.corrcoef(f_vals, r_vals)[0, 1]
                    if not np.isnan(ic):
                        ics.append(ic)

            result[h] = float(np.mean(ics)) if ics else 0.0

        return result

    def _estimate_half_life(self, decay: Dict[int, float]) -> int:
        """
        估计 IC 半衰期

        半衰期 = IC 衰减到初始值一半所需的持有期
        """
        if 1 not in decay or decay[1] == 0:
            return 0

        target = decay[1] / 2
        best_h = 1
        best_diff = abs(decay[1] - target)

        for h, ic in decay.items():
            if h > 1:
                diff = abs(ic - target)
                if diff < best_diff:
                    best_diff = diff
                    best_h = h

        return best_h

    def _interpret_decay(self, half_life: int, decay: Dict[int, float]) -> str:
        """解释 IC 衰减"""
        if half_life == 0:
            return "因子无预测能力 (IC=0)"

        # 获取衰减率
        decay_rates = []
        keys = sorted(decay.keys())
        for i in range(1, len(keys)):
            if decay[keys[i - 1]] > 0:
                decay_rates.append(decay[keys[i]] / decay[keys[i - 1]])

        avg_decay = np.mean(decay_rates) if decay_rates else 1.0

        if half_life >= 20:
            return f"因子稳定 (半衰期 {half_life} 天), IC 衰减缓慢"
        elif half_life >= 5:
            return f"因子中等稳定 (半衰期 {half_life} 天), 适合中短期使用"
        elif half_life >= 1:
            return f"因子衰减较快 (半衰期 {half_life} 天), 适合短期交易"
        else:
            return f"因子衰减极快 (半衰期 < 1 天), 仅适合超短期"

    def compute_factor_correlation(
        self,
        factor_matrix: Dict[str, Dict[str, float]],
    ) -> Dict:
        """
        计算因子相关性矩阵

        Args:
            factor_matrix: {stock_code: {factor_name: value}}

        Returns:
            {
                'names': [factor_names],
                'matrix': [[corr_values]],
                'high_corr_pairs': [(name_a, name_b, corr)],
            }
        """
        if not factor_matrix:
            return {'names': [], 'matrix': [], 'high_corr_pairs': []}

        # 收集所有因子名
        all_factors = set()
        for stock_factors in factor_matrix.values():
            all_factors.update(stock_factors.keys())
        factor_names = sorted(all_factors)

        if not factor_names:
            return {'names': [], 'matrix': [], 'high_corr_pairs': []}

        # 构建因子矩阵 (stocks x factors)
        stocks = sorted(factor_matrix.keys())
        n_stocks = len(stocks)
        n_factors = len(factor_names)
        matrix = np.zeros((n_stocks, n_factors))

        for i, stock in enumerate(stocks):
            for j, fname in enumerate(factor_names):
                val = factor_matrix[stock].get(fname)
                if isinstance(val, (int, float)) and not np.isnan(val):
                    matrix[i, j] = val
                else:
                    matrix[i, j] = np.nan

        # 计算相关性矩阵
        corr_matrix = np.corrcoef(matrix, rowvar=False)

        # 找出高相关因子对
        high_corr_pairs = []
        for i in range(n_factors):
            for j in range(i + 1, n_factors):
                corr = corr_matrix[i, j]
                if not np.isnan(corr) and abs(corr) > self.correlation_high_threshold:
                    high_corr_pairs.append({
                        'name_a': factor_names[i],
                        'name_b': factor_names[j],
                        'correlation': round(float(corr), 4),
                    })

        # 处理 NaN
        corr_clean = np.nan_to_num(corr_matrix, nan=0.0).tolist()

        return {
            'names': factor_names,
            'matrix': corr_clean,
            'high_corr_pairs': high_corr_pairs,
        }

    def get_decay_warnings(
        self,
        ic_decay_result: Dict,
        factor_names: List[str] = None,
    ) -> List[Dict]:
        """
        获取衰减警告

        Args:
            ic_decay_result: compute_ic_decay 的返回结果
            factor_names: 因子名列表

        Returns:
            警告列表
        """
        warnings = []
        decay = ic_decay_result.get('decay', {})
        half_life = ic_decay_result.get('half_life', 0)

        # 检查 IC 过低
        for horizon, ic in decay.items():
            if abs(ic) < self.ic_low_threshold:
                warnings.append({
                    'factor_name': 'all_factors',
                    'warning_type': 'low_ic',
                    'severity': 'high' if abs(ic) < self.ic_low_threshold / 2 else 'medium',
                    'message': f"因子整体 IC 过低 (h={horizon}d, IC={ic:.4f} < {self.ic_low_threshold})",
                    'current_ic': ic,
                    'threshold': self.ic_low_threshold,
                })

        # 检查衰减过快
        if half_life > 0 and half_life < 3:
            warnings.append({
                'factor_name': 'all_factors',
                'warning_type': 'fast_decay',
                'severity': 'high' if half_life < 1 else 'medium',
                'message': f"因子衰减过快 (半衰期 {half_life}d < 3d)",
                'current_ic': half_life,
                'threshold': 3,
            })

        # 检查不稳定
        interpretation = ic_decay_result.get('interpretation', '')
        if '衰减极快' in interpretation:
            warnings.append({
                'factor_name': 'all_factors',
                'warning_type': 'unstable',
                'severity': 'high',
                'message': '因子极度不稳定，建议仅用于超短期交易',
                'current_ic': 0.0,
                'threshold': 0.0,
            })

        return warnings

    def compute_rolling_ic(
        self,
        factor_values_list: List[Dict[str, float]],
        returns_list: List[float],
        window: int = 20,
    ) -> Dict[str, List[float]]:
        """
        计算滚动 IC 趋势

        Args:
            factor_values_list: [{stock_code: factor_value}, ...] per date
            returns_list: [return_pct, ...] per date
            window: 滚动窗口大小

        Returns:
            {factor_name: [ic_values]}
        """
        if not HAS_FACTOR_IC or FactorICMonitor is None:
            return {}

        monitor = FactorICMonitor()
        ics = monitor.compute_ic_series(factor_values_list, returns_list, window)

        return {'all_factors': ics}

    def compute_multi_scale_ic(
        self,
        factor_values: Dict[str, Dict[str, float]],
        future_returns: Dict[str, Dict[str, float]],
    ) -> Dict:
        """
        多尺度 IC 监控 (60D/120D/252D)

        参考: 2026 量化最佳实践 — 三窗口同时监控
        - 60D (3个月): 快速检测短期衰减，但有噪声
        - 120D (6个月): 平衡灵敏度和稳定性
        - 252D (1年): 最稳定，检测长期趋势

        Warning 阈值 (行业标准):
            Green: IC120 > 0.05, ICIR120 > 0.10, HitRate > 55%
            Yellow: IC120 0.02-0.05, ICIR120 0.05-0.10, HitRate 50-55%
            Red: IC120 < 0.02, ICIR120 < 0.05, HitRate < 50%

        Returns:
            {
                '60D': {ic_mean, ic_std, icir, hit_rate, status},
                '120D': {...},
                '252D': {...},
                'overall_status': 'green/yellow/red',
            }
        """
        dates = sorted(factor_values.keys())
        if len(dates) < 30:
            return {'error': '数据不足 (至少需要 30 天)'}

        windows = {
            '60D': min(60, len(dates)),
            '120D': min(120, len(dates)),
            '252D': min(252, len(dates)),
        }

        result = {}
        all_statuses = []

        for scale, window_size in windows.items():
            window_dates = dates[-window_size:]

            # 计算该窗口内的 IC
            ics = []
            for i in range(len(window_dates) - 1):
                date = window_dates[i]
                next_date = window_dates[i + 1]

                if next_date not in future_returns:
                    continue

                common = set(factor_values[date].keys()) & set(future_returns[next_date].keys())
                f_vals, r_vals = [], []
                for code in common:
                    try:
                        fv = float(factor_values[date][code])
                        rv = float(future_returns[next_date][code])
                        if not (np.isnan(fv) or np.isnan(rv)):
                            f_vals.append(fv)
                            r_vals.append(rv)
                    except (ValueError, TypeError):
                        continue

                if len(f_vals) >= 5:
                    ic = np.corrcoef(f_vals, r_vals)[0, 1]
                    if not np.isnan(ic):
                        ics.append(ic)

            if not ics:
                result[scale] = {'ic_mean': 0.0, 'ic_std': 0.0, 'icir': 0.0, 'hit_rate': 0.0, 'n_obs': 0, 'status': 'no_data'}
                continue

            ic_mean = float(np.mean(ics))
            ic_std = float(np.std(ics))
            icir = ic_mean / (ic_std + 1e-10)
            hit_rate = float(np.mean([abs(ic) > 0.01 for ic in ics]))  # 绝对 IC > 0.01 算有效

            # 状态判定
            if ic_mean > 0.05 and icir > 0.10 and hit_rate > 0.55:
                status = 'green'
            elif ic_mean < 0.02 or icir < 0.05 or hit_rate < 0.50:
                status = 'red'
            else:
                status = 'yellow'

            all_statuses.append(status)

            result[scale] = {
                'ic_mean': round(ic_mean, 4),
                'ic_std': round(ic_std, 4),
                'icir': round(icir, 4),
                'hit_rate': round(hit_rate, 4),
                'n_obs': len(ics),
                'status': status,
            }

        # 总体状态
        if 'red' in all_statuses:
            overall = 'red'
        elif 'yellow' in all_statuses:
            overall = 'yellow'
        else:
            overall = 'green'

        result['overall_status'] = overall
        result['scale_count'] = len(windows)
        result['dates_available'] = len(dates)

        return result

    def compute_drift_consensus(
        self,
        ic_series: List[float],
    ) -> Dict:
        """
        漂移检测三算法共识

        使用 3 种算法独立检测，需要 2/3 同意才触发警告。
        - ADWIN: 主检测器，变量窗口大小
        - Page-Hinkley: 累计和检验，检测渐变漂移
        - DDM (Dynamic Drift Detection): PAC-learning 基础

        参考: 2026 量化最佳实践 — 减少误报率

        Args:
            ic_series: IC 时间序列 [ic_t1, ic_t2, ...]

        Returns:
            {
                'adwin': {'drifted': bool, 'splits': int},
                'page_hinkley': {'drifted': bool, 'stat': float},
                'ddm': {'drifted': bool},
                'consensus': {'drifted': bool, 'votes': int},
            }
        """
        if len(ic_series) < 20:
            return {'error': '数据不足 (至少需要 20 个观测)', 'consensus': {'drifted': False, 'votes': 0}}

        result = {}
        votes = 0

        # 1. ADWIN
        if HAS_ADWIN and ADWIN is not None:
            adwin = ADWIN(delta=0.01)
            for val in ic_series:
                try:
                    adwin.add(float(val))
                except (ValueError, TypeError):
                    continue
            result['adwin'] = {
                'drifted': adwin.n_splits > 0,
                'splits': adwin.n_splits,
                'window_size': adwin.window_size,
            }
            if adwin.n_splits > 0:
                votes += 1
        else:
            result['adwin'] = {'drifted': False, 'splits': 0, 'note': 'ADWIN 未安装'}

        # 2. Page-Hinkley Test
        if len(ic_series) >= 10:
            ph_values = [float(x) for x in ic_series if not np.isnan(x)]
            if len(ph_values) >= 10:
                mean_val = np.mean(ph_values)
                cumsum = np.cumsum(np.array(ph_values) - mean_val - 0.005)  # 0.005 容忍阈值
                ph_stat = float(cumsum[-1] - np.min(cumsum)) if len(cumsum) > 0 else 0.0
                # 如果累计和超过阈值，检测漂移
                ph_drifted = ph_stat > 0.1 * (np.std(ph_values) + 1e-10) * len(ph_values)
                result['page_hinkley'] = {
                    'drifted': bool(ph_drifted),
                    'stat': round(ph_stat, 6),
                }
                if ph_drifted:
                    votes += 1
            else:
                result['page_hinkley'] = {'drifted': False, 'note': '数据不足'}
        else:
            result['page_hinkley'] = {'drifted': False, 'note': '数据不足'}

        # 3. DDM (Drift Detection Method)
        if len(ic_series) >= 20:
            ddm_values = [float(x) for x in ic_series if not np.isnan(x)]
            if len(ddm_values) >= 20:
                # 简化的 DDM: 比较前后两段的标准差
                mid = len(ddm_values) // 2
                early = ddm_values[:mid]
                late = ddm_values[mid:]
                std_early = np.std(early)
                std_late = np.std(late)
                mean_all = np.mean(ddm_values)

                # 如果后期标准差显著大于早期，说明不稳定增加
                ddm_drifted = std_late > std_early * 1.5 and len(late) > 5
                result['ddm'] = {
                    'drifted': bool(ddm_drifted),
                    'std_early': round(float(std_early), 6),
                    'std_late': round(float(std_late), 6),
                }
                if ddm_drifted:
                    votes += 1
            else:
                result['ddm'] = {'drifted': False, 'note': '数据不足'}
        else:
            result['ddm'] = {'drifted': False, 'note': '数据不足'}

        result['consensus'] = {
            'drifted': votes >= 2,
            'votes': votes,
            'total': 3,
            'threshold': 2,
        }

        return result

    def get_monitoring_result(
        self,
        stock_code: str,
        factor_data: Dict,
        future_returns: Dict,
    ) -> FactorMonitorResult:
        """
        获取完整的因子监控结果

        Args:
            stock_code: 股票代码
            factor_data: {date_str: {stock_code: {factor_name: value}}}
            future_returns: {date_str: {stock_code: return}}

        Returns:
            FactorMonitorResult
        """
        try:
            # 1. IC Decay
            ic_decay = self.compute_ic_decay(factor_data, future_returns)

            # 2. 多尺度 IC (60D/120D/252D)
            multi_scale_ic = self.compute_multi_scale_ic(factor_data, future_returns)

            # 3. 因子相关性
            agg_factors = {}
            for date, stocks in factor_data.items():
                for code, factors in stocks.items():
                    if code not in agg_factors:
                        agg_factors[code] = {}
                    agg_factors[code].update(factors)

            correlation = self.compute_factor_correlation(agg_factors)

            # 4. 衰减警告
            warnings = self.get_decay_warnings(ic_decay)

            # 5. 因子排名 (基于 IC 绝对值)
            decay_dict = ic_decay.get('decay', {})
            ranking = sorted(
                [{'name': h, 'ic': v} for h, v in decay_dict.items()],
                key=lambda x: abs(x['ic']),
                reverse=True,
            )

            # 6. 漂移检测三算法共识
            all_ics = []
            for h, v in decay_dict.items():
                all_ics.append(v)
            drift_consensus = self.compute_drift_consensus(all_ics) if all_ics else {'consensus': {'drifted': False, 'votes': 0}}

            return FactorMonitorResult(
                stock_code=stock_code,
                ic_decay=ic_decay,
                multi_scale_ic=multi_scale_ic,
                factor_ranking=ranking,
                warnings=[
                    {
                        'factor_name': w['factor_name'],
                        'warning_type': w['warning_type'],
                        'severity': w['severity'],
                        'message': w['message'],
                    }
                    for w in warnings
                ],
                correlation_matrix=correlation.get('matrix', []),
                factor_names=correlation.get('names', []),
                half_life=ic_decay.get('half_life', 0),
                decay_interpretation=ic_decay.get('interpretation', ''),
                drift_consensus=drift_consensus,
            )

        except Exception as e:
            logger.error(f"[FactorMonitor] 监控失败: {e}")
            return FactorMonitorResult(
                stock_code=stock_code,
                success=False,
                error=str(e),
            )


# ── 全局单例 ───────────────────────────────────────────────────────

factor_monitor = FactorMonitor()
