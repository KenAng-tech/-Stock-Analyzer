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


class TransactionCostModel:
    """A 股交易成本建模"""

    def __init__(self, commission_rate=0.0003, stamp_tax=0.001, slippage_bps=2):
        self.commission_rate = commission_rate  # 佣金 0.03%
        self.stamp_tax = stamp_tax  # 印花税 0.1%（卖出）
        self.slippage_bps = slippage_bps  # 滑点 2bps

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
        cost_pct = total_cost / trade_value * 100

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
            atr = float(bar.get('atr', close * 0.03))  # ATR 从 K 线数据获取

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
                          test_period: int = 63, n_windows: int = 5) -> Dict:
        """
        Walk-forward 回测（滚动窗口）

        Args:
            klines: 完整 K 线数据
            train_period: 训练窗口天数
            test_period: 测试窗口天数
            n_windows: 回测窗口数

        Returns:
            Walk-forward 回测结果
        """
        results = []
        total_klines = len(klines)

        for w in range(n_windows):
            train_start = w * test_period
            train_end = train_start + train_period
            test_start = train_end
            test_end = test_start + test_period

            if test_end > total_klines:
                break

            train_data = klines[train_start:train_end]
            test_data = klines[test_start:test_end]

            # 用训练数据训练策略（简化: 直接回测）
            result = self.run_backtest(test_data)
            result['window'] = w + 1
            result['train_period'] = f"{train_start}-{train_end}"
            result['test_period'] = f"{test_start}-{test_end}"
            results.append(result)

        return {
            'method': 'walk_forward',
            'windows': results,
            'summary': self._aggregate_results(results),
        }

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
