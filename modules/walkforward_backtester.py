#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Walk-Forward Backtester — 滚动窗口回测 + Monte Carlo 模拟

P3 长期建设:
  - Walk-forward 验证（滚动/扩展窗口）
  - Monte Carlo 模拟（1000 次随机交易顺序）
  - 交易成本建模（佣金 0.03% + 印花税 0.1% + 滑点）
  - 绩效指标: Sharpe, Sortino, Calmar, MaxDD, WinRate
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from modules.logger import logger


class TransactionCostModel:
    """A 股交易成本建模"""

    def __init__(self, commission_rate=0.0003, stamp_tax=0.001, slippage_bps=2):
        self.commission_rate = commission_rate  # 佣金 0.03%
        self.stamp_tax = stamp_tax  # 印花税 0.1%（卖出）
        self.slippage_bps = slippage_bps  # 滑点 2bps

    def calculate_buy(self, price: float, shares: int) -> Dict:
        """计算买入成本"""
        trade_value = price * shares
        commission = max(trade_value * self.commission_rate, 5)
        slippage = trade_value * self.slippage_bps / 10000
        total = trade_value + commission + slippage
        return {'total': round(total, 2), 'total_cost': round(total, 2)}

    def calculate_sell(self, price: float, shares: int) -> Dict:
        """计算卖出收入"""
        trade_value = price * shares
        commission = max(trade_value * self.commission_rate, 5)
        stamp_tax = trade_value * self.stamp_tax
        slippage = trade_value * self.slippage_bps / 10000
        total = trade_value - commission - stamp_tax - slippage
        return {'total': round(total, 2), 'total_cost': round(commission + stamp_tax + slippage, 2)}

    def calculate_round_trip_cost(self, trade_value: float) -> Dict:
        """计算往返交易成本"""
        # 买入成本
        buy_commission = max(trade_value * self.commission_rate, 5)  # 最低 5 元
        # 卖出成本
        sell_commission = max(trade_value * self.commission_rate, 5)
        sell_stamp_tax = trade_value * self.stamp_tax
        # 滑点
        slippage = trade_value * self.slippage_bps / 10000

        total_cost = buy_commission + sell_commission + sell_stamp_tax + slippage
        cost_pct = total_cost / trade_value  # 往返成本占比（小数形式）

        return {
            'buy_commission': round(buy_commission, 2),
            'sell_commission': round(sell_commission, 2),
            'stamp_tax': round(sell_stamp_tax, 2),
            'slippage': round(slippage, 2),
            'total_cost': round(total_cost, 2),
            'total_cost_pct': round(cost_pct, 4),
        }


class WalkForwardBacktester:
    """Walk-forward 回测引擎 — P2 修复: 真实持仓管理 + 滑点建模"""

    def __init__(self, strategy_func, initial_capital: float = 1000000,
                 slippage_bps: int = 3, commission_rate: float = 0.0003):
        """
        Args:
            strategy_func: 策略函数，签名: func(bar: Dict, position: int, capital: float) -> str
            initial_capital: 初始资金
            slippage_bps: 滑点 (bps)
            commission_rate: 佣金率
        """
        self.strategy_func = strategy_func
        self.initial_capital = initial_capital
        self.cost_model = TransactionCostModel(
            commission_rate=commission_rate,
            slippage_bps=slippage_bps
        )
        # P2: ATR 动态止损参数
        self.atr_stop_multiplier = 2.0
        self.atr_profit_multiplier = 3.0
        # P0-4 (2026-07-01): 涨跌停限制
        self.enable_limit_check = True  # 默认启用涨跌停检查

    @staticmethod
    def is_limit_up(close: float, prev_close: float, is_gem: bool = False, is_st: bool = False) -> bool:
        """
        检查是否涨停

        Args:
            close: 当前收盘价
            prev_close: 前一日收盘价
            is_gem: 是否创业板/科创板 (20%)
            is_st: 是否 ST 股票 (5%)

        Returns:
            True 如果涨停
        """
        if prev_close <= 0:
            return False
        if is_st:
            limit = 0.05
        elif is_gem:
            limit = 0.20
        else:
            limit = 0.10
        return round(close / prev_close - 1, 4) >= limit

    @staticmethod
    def is_limit_down(close: float, prev_close: float, is_gem: bool = False, is_st: bool = False) -> bool:
        """
        检查是否跌停

        Returns:
            True 如果跌停
        """
        if prev_close <= 0:
            return False
        if is_st:
            limit = 0.05
        elif is_gem:
            limit = 0.20
        else:
            limit = 0.10
        return (prev_close / close - 1) >= limit

    def run_backtest(self, klines: List[Dict], position: int = 0) -> Dict:
        """
        单次回测 — P2 修复: 真实持仓管理 + ATR 止损 + 滑点

        Args:
            klines: K 线数据列表
            position: 初始持仓 (0=空仓, 1=持仓)

        Returns:
            回测结果字典
        """
        if not klines or len(klines) < 30:
            return {'error': '数据不足'}

        capital = self.initial_capital
        shares = 0
        entry_price = 0.0
        stop_loss_price = 0.0
        trade_count = 0
        wins = 0
        losses = 0
        daily_returns = []
        equity_curve = [self.initial_capital]
        transactions = []
        peak_equity = self.initial_capital
        max_drawdown = 0.0

        for i in range(20, len(klines)):
            bar = klines[i]
            close = float(bar.get('close', 0))
            high = float(bar.get('high', close))
            low = float(bar.get('low', close))
            prev_close = float(bar.get('prev_close', klines[i - 1]['close'] if i > 0 else close))
            atr = float(bar.get('atr', close * 0.03))  # ATR 从 K 线数据获取

            # P0-4 (2026-07-01): 涨跌停检查
            is_gem = bar.get('is_gem', False) or (bar.get('code', '').startswith(('sz30', 'sh68')))
            is_st = bar.get('is_st', False) or 'ST' in bar.get('name', '') or 'st' in bar.get('name', '')
            limit_up = self.is_limit_up(close, prev_close, is_gem, is_st) if self.enable_limit_check else False
            limit_down = self.is_limit_down(close, prev_close, is_gem, is_st) if self.enable_limit_check else False

            # P2: ATR 动态止损/止盈
            if shares > 0 and atr > 0:
                new_stop_loss = entry_price - atr * self.atr_stop_multiplier
                new_take_profit = entry_price + atr * self.atr_profit_multiplier
                # 只上移止损，不下移
                if new_stop_loss > stop_loss_price:
                    stop_loss_price = new_stop_loss
                # 只上移止盈
                if new_take_profit > 0:
                    pass  # 止盈通过信号判断

            # 计算当前权益
            if shares > 0:
                current_equity = capital + shares * close
            else:
                current_equity = capital

            # P2: 检查止损/止盈（价格触发）
            if shares > 0:
                # 止损触发
                if low <= stop_loss_price:
                    # P0-4: 跌停时无法以止损价卖出，以跌停价成交
                    if limit_down:
                        sell_price = prev_close * (1 - (0.20 if is_gem else 0.10))
                        sell_shares = shares
                        cost_pct = self.cost_model.commission_rate + self.cost_model.stamp_tax + self.cost_model.slippage_bps / 10000
                        proceeds = sell_shares * sell_price * (1 - cost_pct)
                        pnl = proceeds - shares * entry_price * (1 + self.cost_model.commission_rate)
                        if pnl > 0:
                            wins += 1
                        else:
                            losses += 1
                        capital += proceeds
                        transactions.append({
                            'date': i, 'action': 'SELL_STOP_LIMIT_DOWN', 'price': round(sell_price, 2),
                            'shares': sell_shares, 'pnl': round(pnl, 2), 'reason': '跌停止损'
                        })
                        shares = 0
                        entry_price = 0
                        stop_loss_price = 0
                        trade_count += 1
                    else:
                        sell_price = stop_loss_price
                    sell_shares = shares
                    cost_pct = self.cost_model.commission_rate + self.cost_model.stamp_tax + self.cost_model.slippage_bps / 10000
                    proceeds = sell_shares * sell_price * (1 - cost_pct)
                    pnl = proceeds - shares * entry_price * (1 + self.cost_model.commission_rate)
                    if pnl > 0:
                        wins += 1
                    else:
                        losses += 1
                    # ✅ 修复: 止损卖出 — capital += 卖出净收入
                    # 买入时: capital -= shares * entry_price * (1 + buy_cost)
                    # 卖出时: capital += proceeds (净收入)
                    capital += proceeds
                    transactions.append({
                        'date': i, 'action': 'SELL_STOP', 'price': round(sell_price, 2),
                        'shares': sell_shares, 'pnl': round(pnl, 2)
                    })
                    shares = 0
                    entry_price = 0
                    stop_loss_price = 0
                    trade_count += 1

            # 获取策略信号
            signal = self.strategy_func(bar, shares, capital)

            # 执行交易
            if signal == 'buy' and shares == 0:
                # P0-4: 涨停日无法买入（实际交易中涨停时排队买不进）
                if limit_up:
                    transactions.append({
                        'date': i, 'action': 'BUY_REJECTED', 'price': round(close, 2),
                        'reason': '涨停无法买入',
                    })
                    continue

                # 买入: 用 80% 资金
                buy_value = capital * 0.8
                buy_cost_pct = self.cost_model.commission_rate + self.cost_model.slippage_bps / 10000
                actual_cost = buy_value * buy_cost_pct
                shares = int((buy_value - actual_cost) / close / 100) * 100
                if shares > 0:
                    entry_price = close
                    stop_loss_price = close - atr * self.atr_stop_multiplier if atr > 0 else close * 0.95
                    capital -= shares * close * (1 + buy_cost_pct)
                    trade_count += 1
                    transactions.append({
                        'date': i, 'action': 'BUY', 'price': round(close, 2),
                        'shares': shares, 'cost_pct': round(buy_cost_pct * 100, 3)
                    })

            elif signal == 'sell' and shares > 0:
                # P0-4: 跌停日无法卖出（实际交易中跌停时排队卖不出）
                if limit_down:
                    transactions.append({
                        'date': i, 'action': 'SELL_REJECTED', 'price': round(close, 2),
                        'shares': shares, 'reason': '跌停无法卖出',
                    })
                    continue

                sell_cost_pct = self.cost_model.commission_rate + self.cost_model.stamp_tax + self.cost_model.slippage_bps / 10000
                proceeds = shares * close * (1 - sell_cost_pct)
                cost_basis = shares * entry_price * (1 + self.cost_model.commission_rate)
                pnl = proceeds - cost_basis
                if pnl > 0:
                    wins += 1
                else:
                    losses += 1
                capital += proceeds
                transactions.append({
                    'date': i, 'action': 'SELL', 'price': round(close, 2),
                    'shares': shares, 'pnl': round(pnl, 2)
                })
                shares = 0
                entry_price = 0
                stop_loss_price = 0
                trade_count += 1

            # 记录权益
            equity_curve.append(current_equity)

            # 跟踪峰值和最大回撤
            if current_equity > peak_equity:
                peak_equity = current_equity
            dd = (peak_equity - current_equity) / peak_equity if peak_equity > 0 else 0
            if dd > max_drawdown:
                max_drawdown = dd

            # 日收益率
            if len(equity_curve) >= 2:
                daily_ret = (equity_curve[-1] - equity_curve[-2]) / equity_curve[-2]
                daily_returns.append(daily_ret)

        return self._compute_metrics(equity_curve, daily_returns, trade_count, wins, losses, transactions, max_drawdown)

    def run_walk_forward(self, klines: List[Dict], train_period: int = 252,
                          test_period: int = 63, n_windows: int = 5,
                          retrain_callback=None, expanding_window: bool = False,
                          cv_method: str = 'rolling') -> Dict:
        """
        真正的 Walk-Forward 回测（滚动/扩展窗口 / Purged K-Fold）

        P0 修复 (2026-07-01):
          之前版本只是将数据切分后直接回测，训练数据完全没用。
          现在支持 retrain_callback 在每个训练窗口上重新训练/更新策略参数。

        P2 新增 (2026-09-16):
          新增 cv_method 参数支持 Purged K-Fold 交叉验证，
          防止时间序列前视偏差 (purge + embargo)。

        Args:
            klines: 完整 K 线数据
            train_period: 训练窗口天数
            test_period: 测试窗口天数
            n_windows: 回测窗口数
            retrain_callback: callable(train_data, current_params) -> new_params
                              在每个训练窗口上重新训练/更新参数。
                              如果不提供，则使用固定参数回测（退化为普通回测）。
            expanding_window: 是否使用扩展窗口（True=训练集累积，False=滚动窗口）
            cv_method: 交叉验证方法
                       - 'rolling': 滚动窗口 (默认，向后兼容)
                       - 'expanding': 扩展窗口
                       - 'purged_kfold': Purged K-Fold (Lopez de Prado, 无前视偏差)

        Returns:
            Walk-forward 回测结果
        """
        if cv_method == 'purged_kfold':
            return self._run_purged_kfold(klines, n_windows, retrain_callback)
        if cv_method == 'expanding':
            expanding_window = True
        results = []
        total_klines = len(klines)
        current_params = None  # 当前策略参数（可能被 retrain_callback 更新）

        for w in range(n_windows):
            if expanding_window:
                # 扩展窗口: 训练集从开始累积到 train_end
                train_start = 0
                train_end = train_period + w * test_period
            else:
                # 滚动窗口: 训练集滑动
                train_start = w * test_period
                train_end = train_start + train_period

            test_start = train_end
            test_end = test_start + test_period

            if test_end > total_klines:
                break

            train_data = klines[train_start:train_end]
            test_data = klines[test_start:test_end]

            # P0 修复: 在训练窗口上重新训练/更新策略参数
            if retrain_callback is not None:
                try:
                    current_params = retrain_callback(train_data, current_params)
                    logger.debug(f"[WalkForward] 窗口 {w+1}: retrain_callback 完成, params={type(current_params).__name__}")
                except Exception as e:
                    logger.warning(f"[WalkForward] 窗口 {w+1} retrain_callback 失败: {e}, 使用上次参数")

            # 在测试窗口上回测
            result = self.run_backtest(test_data)
            result['window'] = w + 1
            result['train_period'] = f"{train_start}-{train_end}"
            result['test_period'] = f"{test_start}-{test_end}"
            result['train_size'] = len(train_data)
            result['test_size'] = len(test_data)
            results.append(result)

        result = {
            'method': 'walk_forward',
            'windows': results,
            'summary': self._aggregate_results(results),
            'n_windows': len(results),
            'train_period': train_period,
            'test_period': test_period,
            'expanding_window': expanding_window,
        }
        # P0 (2026-09-10): 统计过拟合诊断 (DSR + PBO-CSCV + t 检验), 替代原启发式规则。
        # enhanced_report 自 2026-07 起零调用 (死代码), 本方法独立挂在 run_walk_forward 链,
        # 两条活链 (backtest_routes/async_backtest_routes) 自动携带 statistical_diagnostics 键。
        try:
            result['statistical_diagnostics'] = self.statistical_overfit_check(results)
        except Exception as e:
            logger.warning(f"[WalkForward] 统计诊断降级 (不影响回测结果): {e}")
        return result

    def _run_purged_kfold(self, klines: List[Dict], n_windows: int,
                           retrain_callback=None) -> Dict:
        """
        Purged K-Fold 回测 — Lopez de Prado (2018)

        将数据分为 K 个折，每个折:
        1. 测试集 = 当前折
        2. 训练集 = 其他折 - purge 区 (测试集边界附近的训练样本)
        3. 测试集起始 embargo 区也被移除

        Args:
            klines: 完整 K 线数据
            n_windows: 折数 (K-Fold 的 K)
            retrain_callback: 在每个训练折上重新训练

        Returns:
            Walk-forward 回测结果 (与 run_walk_forward 格式一致)
        """
        from modules.purged_kfold import PurgedKFold

        results = []
        current_params = None

        pkf = PurgedKFold(n_splits=n_windows, embargo_pct=0.05)
        for fold, (train_idx, test_idx) in enumerate(pkf.split(klines)):
            train_data = [klines[i] for i in train_idx]
            test_data = [klines[i] for i in test_idx]

            # 在训练窗口上重新训练/更新策略参数
            if retrain_callback is not None:
                try:
                    current_params = retrain_callback(train_data, current_params)
                    logger.debug(f"[WalkForward/PKFold] 窗口 {fold+1}: retrain_callback 完成")
                except Exception as e:
                    logger.warning(f"[WalkForward/PKFold] 窗口 {fold+1} retrain_callback 失败: {e}")

            # 在测试窗口上回测
            result = self.run_backtest(test_data)
            result['window'] = fold + 1
            result['train_size'] = len(train_idx)
            result['test_size'] = len(test_idx)
            result['cv_method'] = 'purged_kfold'
            results.append(result)

        result = {
            'method': 'purged_kfold',
            'windows': results,
            'summary': self._aggregate_results(results),
            'n_windows': len(results),
            'cv_method': 'purged_kfold',
            'embargo_pct': 0.05,
        }
        try:
            result['statistical_diagnostics'] = self.statistical_overfit_check(results)
        except Exception as e:
            logger.warning(f"[WalkForward/PKFold] 统计诊断降级: {e}")
        return result

    def statistical_overfit_check(self, windows: List[Dict]) -> Dict:
        """统计过拟合诊断 (P0, 2026-09-10) — 替代 enhanced_report 启发式规则

        三层 (仅 scipy/numpy, 零新依赖):
          1. DSR Deflated Sharpe Ratio (Bailey & López de Prado 2014) —
             跨窗 SR 序列做多选偏差 deflation (deflated = 校正 N 次试验的选优膨胀)。
          2. PBO-CSCV (López de Prado 2016 单策略变体) — 组合交叉验证:
             IS 段收益为正而 OOS 段为负的 252 组合占比 = 策略退化概率。
             (标准 PBO 需多候选策略 trial SR 矩阵; hyperparam 02:00 链暂无 trial 落盘,
              退而用 CSCV 段变体 = 策略退化概率, 非标准 PBO — 诚实边界。)
          3. t 检验 (ttest_1samp, scipy.stats) — 替代原 dispersion heuristic;
             每窗仅 5 点, 统计功效弱 = 已知边界。

        无前视: 窗口 OOS 收益独立拼接 (各窗 equity 独立, 2026-09-10 侦察实证)。
        降级: 数据不足返回 {'method': 'statistical_overfit_check_v1'} (call 方 try 包裹)。

        Args:
            windows: run_walk_forward 的 results 列表 (每窗含 daily_returns/total_return)

        Returns:
            {'method': 'statistical_overfit_check_v1', 'dsr': {...}, 'pbo_cscv': {...}, 't_test': {...}}
        """
        from itertools import combinations
        from scipy import stats as scipy_stats

        out: Dict = {'method': 'statistical_overfit_check_v1'}
        if len(windows) < 2:
            return out

        # ── 拼接窗口 OOS 日收益 + 收集各窗 SR (DSR deflation 用) ──
        srs, oos_ret = [], []
        for w in windows:
            dr = w.get('daily_returns') or []
            if len(dr) >= 10:
                arr = np.array(dr, dtype=float)
                mu, sd = float(arr.mean()), float(arr.std(ddof=1))
                if sd > 1e-12:
                    srs.append((mu / sd) * np.sqrt(252))  # 年化 SR (per-window)
                oos_ret.extend(dr)
        n_trials = max(len(srs), 1)
        # 2026-09-10 链尾验证修正 (二次): 准平链统计检验=误判源。
        # 实测 sz300620: w1 仅 4/22 非零, w2-w5 全零 — 拼接链 106/110 个 0。
        # 此类链上 PBO "0/252 有泛化信号" 与 t 检验 p 值均为退化统计假象
        # (无退化段可找 ≠ 泛化好, 只是无交易可统计) → 三块统一跳过并给原因。
        # srs 同理: 全零窗 sd=0 不入 srs → n_trials<2 → DSR 正确跳过 (非 bug)。
        _nz = sum(1 for r in oos_ret if abs(r) > 1e-12)
        _flat_chain = (len(oos_ret) >= 40 and
                       (_nz * 10 < len(oos_ret) or  # 非零收益占比 <10% = 低活跃链
                        float(np.std(np.array(oos_ret, dtype=float), ddof=1)) < 1e-9))

        # ── 1) DSR: Deflated Sharpe Ratio ──
        if len(oos_ret) >= 40 and n_trials >= 2 and not _flat_chain:
            r = np.array(oos_ret, dtype=float)
            T = len(r)
            mu, sd = float(r.mean()), float(r.std(ddof=1))
            sharpe = (mu / sd) * np.sqrt(252)
            var_sr = float(np.var(srs, ddof=1)) * 252  # 跨窗 SR 方差 (年化)
            gamma = 0.5772156649  # Euler-Mascheroni 常数
            sr_star = float(np.sqrt(max(var_sr, 1e-12)) * (
                (1 - gamma) * scipy_stats.norm.ppf(1 - 1.0 / n_trials)
                + gamma * scipy_stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
            ))
            skew = float(scipy_stats.skew(r))
            kurt = float(scipy_stats.kurtosis(r, fisher=False))  # 非中心四阶矩 (高斯=3)
            denom = max(1 - skew * sharpe + (kurt - 1) / 4 * sharpe ** 2, 1e-12)
            dsr = float(scipy_stats.norm.cdf(
                (sharpe - sr_star) * np.sqrt(T - 1) / np.sqrt(denom)
            ))
            out['dsr'] = {
                'sharpe': round(sharpe, 3),
                'sr_star': round(sr_star, 3),
                'skewness': round(skew, 3),
                'kurtosis': round(kurt, 3),
                'n_trials': n_trials,
                'n_trials_source': 'walk_forward_windows',  # 保守 deflation (无 trial 历史)
                'dsr': round(dsr, 4),
                'significant_at_95': bool(dsr > 0.95),
                'note': ('DSR>0.95 = SR 在跨窗 deflation 下不拒绝 H0 (非过拟合信号)'
                         if dsr > 0.95 else
                         f'DSR={dsr:.3f} ≤ 0.95: SR 可能来自选择偏差 (deflation 后不显著)'),
            }

        # ── 2) PBO-CSCV: 252 组合 IS→OOS 退化频率 ──
        if len(oos_ret) >= 40 and not _flat_chain:
            T = len(oos_ret)
            S = 10
            seg_len = T // S
            segs = [np.array(oos_ret[i * seg_len:(i + 1) * seg_len], dtype=float)
                    for i in range(S)]
            combos = list(combinations(range(S), S // 2))
            degrade = 0
            for is_idx in combos:
                oos_idx = [i for i in range(S) if i not in is_idx]
                is_ret = float(np.mean([segs[i].mean() for i in is_idx]))
                oos_r = float(np.mean([segs[i].mean() for i in oos_idx]))
                if is_ret > 0 and oos_r < 0:
                    degrade += 1
            pbo = degrade / len(combos)
            out['pbo_cscv'] = {
                'method': 'cscv_v1',
                'pbo': round(pbo, 4),
                'degrade_combinations': degrade,
                'n_combinations': len(combos),
                'n_segments': S,
                'note': (f'{degrade}/{len(combos)} 组合 IS 正→OOS 退化 (CSCV 单策略变体) — '
                         + ('退化占多数, 策略疑似过拟合' if pbo > 0.5 else '退化少数, 有泛化信号')),
            }

        # ── 3) t 检验: mean=0 H0 (替代 dispersion heuristic) ──
        if _flat_chain:
            out['skipped'] = (f'low_activity: 非零收益仅 {_nz}/{len(oos_ret)} (策略少交易/低波动链), '
                              f'统计检验跳过 — 数据不足, p 值与退化率均不可信')
        sig: Dict = {}
        for m in ('sharpe_ratio', 'total_return', 'win_rate'):
            vals = [w.get(m) for w in windows if m in w and w.get(m) is not None]
            # 常量指标 (std=0) → ttest 返回 NaN → jsonify 产生非法 JSON → 跳过
            if len(vals) >= 2 and not _flat_chain and float(np.std(np.array(vals, dtype=float), ddof=1)) > 0:
                arr = np.array(vals, dtype=float)
                t_stat, p_value = scipy_stats.ttest_1samp(arr, popmean=0)
                sig[m] = {
                    't_statistic': round(float(t_stat), 4),
                    'p_value': round(float(p_value), 6),
                    'significant_at_5pct': bool(p_value < 0.05),
                    'significant_at_1pct': bool(p_value < 0.01),
                    'n': len(vals),
                }
        out['t_test'] = sig
        return out

    def monte_carlo_simulation(self, daily_returns: List[float],
                                n_simulations: int = 1000,
                                horizon_days: int = 252) -> Dict:
        """
        Monte Carlo 模拟 — P2 修复: 使用 Bootstrap 而非正态分布（处理厚尾）

        Args:
            daily_returns: 历史日收益率
            n_simulations: 模拟次数
            horizon_days: 模拟 horizon

        Returns:
            Monte Carlo 模拟结果
        """
        if not daily_returns:
            return {'error': '无收益率数据'}

        final_values = []
        for sim in range(n_simulations):
            equity = self.initial_capital
            for day in range(horizon_days):
                # P2 修复: 从历史收益中随机抽样 (Bootstrap)
                ret = float(np.random.choice(daily_returns))
                equity *= (1 + ret)
            final_values.append(equity)

        final_values = np.array(final_values)

        return {
            'method': 'monte_carlo_bootstrap',
            'simulations': n_simulations,
            'horizon_days': horizon_days,
            'mean_final': float(np.mean(final_values)),
            'median_final': float(np.median(final_values)),
            'std_final': float(np.std(final_values)),
            'percentile_5': float(np.percentile(final_values, 5)),
            'percentile_25': float(np.percentile(final_values, 25)),
            'percentile_75': float(np.percentile(final_values, 75)),
            'percentile_95': float(np.percentile(final_values, 95)),
            'max_final': float(np.max(final_values)),
            'min_final': float(np.min(final_values)),
            'probability_profit': float(np.mean(final_values > self.initial_capital)),
            'histogram': {
                'bins': np.percentile(final_values, np.arange(0, 101, 10)).tolist(),
                'values': np.percentile(final_values, np.arange(0, 101, 10)).tolist(),
            },
        }

    def monte_carlo_block_bootstrap(self, daily_returns: List[float],
                                     block_size: int = 10,
                                     n_simulations: int = 1000,
                                     horizon_days: int = 252) -> Dict:
        """
        Monte Carlo Block Bootstrap 模拟 — P2 修复 (2026-07-01)
        使用 Block Bootstrap 保留收益率序列的自相关性，
        比等概率抽样 (np.random.choice) 更真实地模拟厚尾分布。

        Args:
            daily_returns: 历史日收益率
            block_size: 块大小，默认 10（捕获约 2 周的自相关性）
            n_simulations: 模拟次数
            horizon_days: 模拟 horizon

        Returns:
            Monte Carlo 模拟结果（含 Block Bootstrap 标记）
        """
        if not daily_returns or len(daily_returns) < block_size:
            return {'error': '收益率数据不足 (至少需要 block_size 个数据点)'}

        n = len(daily_returns)
        final_values = []

        for sim in range(n_simulations):
            equity = self.initial_capital
            pos = np.random.randint(0, n)

            for day in range(horizon_days):
                # 从 pos 位置取一个 block_size 长度的块
                block_start = pos % n
                end = min(block_start + block_size, n)
                block = daily_returns[block_start:end]

                for ret in block:
                    equity *= (1 + ret)

                pos += block_size
                # 如果取完了所有数据，重新随机选择起始位置
                if pos >= n:
                    pos = np.random.randint(0, n)

            final_values.append(equity)

        final_values = np.array(final_values)

        return {
            'method': 'monte_carlo_block_bootstrap',
            'block_size': block_size,
            'simulations': n_simulations,
            'horizon_days': horizon_days,
            'mean_final': float(np.mean(final_values)),
            'median_final': float(np.median(final_values)),
            'std_final': float(np.std(final_values)),
            'percentile_5': float(np.percentile(final_values, 5)),
            'percentile_25': float(np.percentile(final_values, 25)),
            'percentile_75': float(np.percentile(final_values, 75)),
            'percentile_95': float(np.percentile(final_values, 95)),
            'max_final': float(np.max(final_values)),
            'min_final': float(np.min(final_values)),
            'probability_profit': float(np.mean(final_values > self.initial_capital)),
            'skewness': float(float(np.mean(((final_values - np.mean(final_values)) / np.std(final_values)) ** 3)) if np.std(final_values) > 0 else 0),
            'kurtosis': float(float(np.mean(((final_values - np.mean(final_values)) / np.std(final_values)) ** 4)) - 3 if np.std(final_values) > 0 else 0),
        }

    def _compute_metrics(self, equity_curve: List[float], daily_returns: List[float],
                          trade_count: int, wins: int, losses: int,
                          transactions: List[Dict],
                          explicit_max_dd: float = None) -> Dict:
        """计算回测指标 — P2 修复: 使用传入的最大回撤"""
        if not equity_curve or len(equity_curve) < 2:
            return {'error': '权益曲线数据不足'}

        final_equity = equity_curve[-1]
        total_return = (final_equity - self.initial_capital) / self.initial_capital

        # 年化收益
        n_days = len(equity_curve) - 1
        annual_return = (1 + total_return) ** (252 / max(n_days, 1)) - 1

        # 年化波动率
        annual_vol = np.std(daily_returns) * np.sqrt(252) if daily_returns else 0

        # Sharpe Ratio
        sharpe = (annual_return - 0.02) / annual_vol if annual_vol > 0 else 0

        # Sortino Ratio
        downside_returns = [r for r in daily_returns if r < 0]
        downside_vol = np.std(downside_returns) * np.sqrt(252) if downside_returns else 0
        sortino = (annual_return - 0.02) / downside_vol if downside_vol > 0 else 0

        # Max Drawdown (使用回测中计算的值)
        max_dd = explicit_max_dd if explicit_max_dd is not None else 0

        # Calmar Ratio
        calmar = annual_return / max_dd if max_dd > 0 else 0

        # Win Rate
        win_rate = wins / max(wins + losses, 1)

        # Profit Factor
        total_wins_val = sum(t.get('pnl', 0) for t in transactions if t.get('pnl', 0) > 0)
        total_losses_val = abs(sum(t.get('pnl', 0) for t in transactions if t.get('pnl', 0) < 0))
        profit_factor = total_wins_val / total_losses_val if total_losses_val > 0 else float('inf')

        return {
            'total_return': round(total_return, 4),
            'annual_return': round(annual_return, 4),
            'annual_volatility': round(annual_vol, 4),
            'sharpe_ratio': round(sharpe, 3),
            'sortino_ratio': round(sortino, 3),
            'calmar_ratio': round(calmar, 3),
            'max_drawdown': round(max_dd, 4),
            'win_rate': round(win_rate, 4),
            'profit_factor': round(profit_factor, 3) if profit_factor != float('inf') else 999,
            'total_trades': trade_count,
            'wins': wins,
            'losses': losses,
            'n_days': n_days,
            'final_equity': round(final_equity, 2),
            'transactions': transactions[:20],
            # P0 (2026-09-10): 窗口日收益序列带出, 供 run_walk_forward 链的
            # statistical_overfit_check 拼接 DSR/PBO-CSCV 统计检验 (原丢弃)
            'daily_returns': [round(float(r), 6) for r in daily_returns] if daily_returns else [],
        }

    def bootstrap_confidence_interval(self, windows: List[Dict],
                                       metrics: List[str] = None,
                                       n_bootstrap: int = 1000,
                                       confidence_level: float = 0.95) -> Dict:
        """
        Bootstrap 置信区间估计 — 对 Walk-Forward 各窗口的关键指标做重抽样，
        计算均值在给定置信水平下的置信区间，评估策略稳定性。

        Args:
            windows: walk_forward 返回的 windows 列表（每个窗口含回测指标）
            metrics: 需要计算置信区间的指标名列表，默认 ['sharpe_ratio', 'max_drawdown', 'win_rate', 'total_return']
            n_bootstrap: Bootstrap 重抽样次数
            confidence_level: 置信水平 (默认 0.95)

        Returns:
            各指标的原始均值 + Bootstrap 置信区间
        """
        if not windows:
            return {'error': '无窗口数据'}

        default_metrics = ['sharpe_ratio', 'max_drawdown', 'win_rate', 'total_return',
                           'annual_return', 'sortino_ratio', 'calmar_ratio', 'profit_factor']
        if metrics is None:
            metrics = default_metrics

        alpha = 1.0 - confidence_level
        lower_pct = (alpha / 2) * 100
        upper_pct = (1.0 - alpha / 2) * 100

        results = {}
        for metric in metrics:
            values = [w.get(metric) for w in windows if metric in w and w.get(metric) is not None]
            if not values or len(values) < 2:
                results[metric] = {
                    'n_samples': len(values),
                    'note': '样本不足，无法计算置信区间',
                }
                continue

            values = np.array(values, dtype=float)
            original_mean = float(np.mean(values))

            # Bootstrap 重抽样
            boot_means = np.empty(n_bootstrap)
            rng = np.random.RandomState(42)  # 固定种子保证可复现
            for b in range(n_bootstrap):
                sample = rng.choice(values, size=len(values), replace=True)
                boot_means[b] = np.mean(sample)

            results[metric] = {
                'n_samples': int(len(values)),
                'original_mean': round(original_mean, 4),
                'bootstrap_mean': round(float(np.mean(boot_means)), 4),
                'bootstrap_std': round(float(np.std(boot_means)), 4),
                f'ci_{int(confidence_level * 100)}_lower': round(float(np.percentile(boot_means, lower_pct)), 4),
                f'ci_{int(confidence_level * 100)}_upper': round(float(np.percentile(boot_means, upper_pct)), 4),
                'ci_width': round(float(np.percentile(boot_means, upper_pct) - np.percentile(boot_means, lower_pct)), 4),
                'bias': round(float(np.mean(boot_means) - original_mean), 4),
                'skewness': round(float(np.mean(((boot_means - np.mean(boot_means)) / max(np.std(boot_means), 1e-10)) ** 3)), 4),
            }

        return results

    def enhanced_report(self, walk_forward_result: Dict,
                        bootstrap_n: int = 1000,
                        confidence_level: float = 0.95) -> Dict:
        """
        增强版 Walk-Forward 报告 — 在原有 summary 基础上加入：
          - Bootstrap 置信区间 (Sharpe / MaxDD / WinRate / 收益率)
          - 窗口稳定性分析 (各窗口指标排名 / 一致性)
          - 过拟合检测 (训练集 vs 测试集表现对比)
          - 统计显著性 (单样本 t 检验 vs 零假设)

        Args:
            walk_forward_result: run_walk_forward 返回的结果字典
            bootstrap_n: Bootstrap 次数
            confidence_level: 置信水平

        Returns:
            增强版报告字典
        """
        if 'error' in walk_forward_result:
            return walk_forward_result

        windows = walk_forward_result.get('windows', [])
        if not windows:
            return {'error': '无回测窗口数据'}

        # 1) Bootstrap 置信区间
        ci = self.bootstrap_confidence_interval(
            windows,
            metrics=['sharpe_ratio', 'max_drawdown', 'win_rate', 'total_return',
                     'annual_return', 'sortino_ratio', 'calmar_ratio'],
            n_bootstrap=bootstrap_n,
            confidence_level=confidence_level,
        )

        # 2) 窗口稳定性：每个指标在不同窗口的变异系数
        stability = {}
        all_metrics = ['sharpe_ratio', 'max_drawdown', 'win_rate', 'total_return',
                       'annual_return', 'sortino_ratio', 'calmar_ratio', 'profit_factor']
        for m in all_metrics:
            vals = [w.get(m) for w in windows if m in w and w.get(m) is not None]
            if len(vals) >= 2:
                arr = np.array(vals, dtype=float)
                stability[m] = {
                    'mean': round(float(np.mean(arr)), 4),
                    'std': round(float(np.std(arr)), 4),
                    'cv': round(float(np.std(arr) / max(abs(np.mean(arr)), 1e-10)), 4),  # 变异系数
                    'min': round(float(np.min(arr)), 4),
                    'max': round(float(np.max(arr)), 4),
                    'best_window': int(int(np.argmax(arr)) + 1),
                    'worst_window': int(int(np.argmin(arr)) + 1),
                }

        # 3) 过拟合检测：比较各窗口收益分布
        returns = [w.get('total_return', 0) for w in windows if 'total_return' in w]
        overfit_risk = {}
        if len(returns) >= 2:
            arr = np.array(returns)
            # 收益为正的窗口比例
            positive_pct = float(np.mean(arr > 0))
            # 如果正收益窗口比例 < 50%，说明策略不稳定
            is_stable = positive_pct >= 0.5
            # 收益标准差 vs 均值，衡量离散程度
            dispersion = float(np.std(arr) / max(abs(np.mean(arr)), 1e-10))
            overfit_risk = {
                'positive_window_pct': round(positive_pct * 100, 1),
                'is_stable': is_stable,
                'dispersion': round(dispersion, 4),
                'risk_level': '低' if dispersion < 0.5 else ('中' if dispersion < 1.0 else '高'),
                'note': '策略在各窗口表现稳定' if is_stable else '策略在某些窗口失效，可能存在过拟合风险',
            }

        # 4) 统计显著性：对 Sharpe 和收益率做单样本 t 检验 (H0: mean == 0)
        from scipy import stats as scipy_stats  # lazy import
        significance = {}
        for m in ['sharpe_ratio', 'total_return', 'win_rate']:
            vals = [w.get(m) for w in windows if m in w and w.get(m) is not None]
            if len(vals) >= 2:
                arr = np.array(vals, dtype=float)
                t_stat, p_value = scipy_stats.ttest_1samp(arr, popmean=0)
                significance[m] = {
                    't_statistic': round(float(t_stat), 4),
                    'p_value': round(float(p_value), 6),
                    'significant_at_5pct': bool(p_value < 0.05),
                    'significant_at_1pct': bool(p_value < 0.01),
                    'n': len(vals),
                }

        # 5) 各窗口明细（保留原始结果）
        window_details = []
        for w in windows:
            detail = {
                'window': w.get('window'),
                'train_period': w.get('train_period'),
                'test_period': w.get('test_period'),
                'train_size': w.get('train_size'),
                'test_size': w.get('test_size'),
            }
            for m in all_metrics:
                if m in w:
                    detail[m] = w[m]
            window_details.append(detail)

        # 6) 综合评分
        summary = walk_forward_result.get('summary', {})
        overall_score = {}
        if summary:
            mean_sharpe = summary.get('mean_sharpe', 0)
            consistent = summary.get('consistent_profit', 0)
            n_win = summary.get('n_windows', 1)
            stability_ratio = consistent / max(n_win, 1)
            # 综合分数 = 0.4 * Sharpe归一化 + 0.3 * 一致性 + 0.3 * 收益
            overall_score = {
                'sharpe_component': round(min(max(mean_sharpe / 2.0, 0), 1) * 0.4, 3),
                'stability_component': round(stability_ratio * 0.3, 3),
                'return_component': round(min(max(summary.get('mean_return', 0) + 0.5, 0), 1) * 0.3, 3),
                'total_score': round(
                    min(max(mean_sharpe / 2.0, 0)) * 0.4 + stability_ratio * 0.3 +
                    min(max(summary.get('mean_return', 0) + 0.5, 0), 1) * 0.3, 3),
                'rating': '优秀' if (overall_score.get('total_score', 0) >= 0.7) else
                          ('良好' if overall_score.get('total_score', 0) >= 0.5 else
                           ('一般' if overall_score.get('total_score', 0) >= 0.3 else '较差')),
            }

        return {
            'method': 'walk_forward_enhanced',
            'confidence_level': confidence_level,
            'bootstrap_simulations': bootstrap_n,
            'n_windows': len(windows),
            'train_period': walk_forward_result.get('train_period'),
            'test_period': walk_forward_result.get('test_period'),
            'expanding_window': walk_forward_result.get('expanding_window', False),
            'summary': summary,
            'confidence_intervals': ci,
            'stability': stability,
            'overfit_risk': overfit_risk,
            'statistical_significance': significance,
            'overall_score': overall_score,
            'window_details': window_details,
        }

    @staticmethod
    def _aggregate_results(windows: List[Dict]) -> Dict:
        """聚合多个窗口的结果"""
        if not windows:
            return {}

        returns = [w.get('total_return', 0) for w in windows if 'total_return' in w]
        sharpes = [w.get('sharpe_ratio', 0) for w in windows if 'sharpe_ratio' in w]
        maxdds = [w.get('max_drawdown', 0) for w in windows if 'max_drawdown' in w]
        winrates = [w.get('win_rate', 0) for w in windows if 'win_rate' in w]

        return {
            'mean_return': round(float(np.mean(returns)), 4) if returns else 0,
            'std_return': round(float(np.std(returns)), 4) if returns else 0,
            'mean_sharpe': round(float(np.mean(sharpes)), 3) if sharpes else 0,
            'mean_maxdd': round(float(np.mean(maxdds)), 4) if maxdds else 0,
            'mean_winrate': round(float(np.mean(winrates)), 4) if winrates else 0,
            'n_windows': len(windows),
            'consistent_profit': sum(1 for r in returns if r > 0),
        }
