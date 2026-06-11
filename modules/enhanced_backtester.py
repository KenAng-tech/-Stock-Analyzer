"""
增强回测引擎 - Enhanced Backtester
支持Walk-forward分析、参数优化、Monte Carlo模拟
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable, Tuple
from datetime import datetime, timedelta
from itertools import product


class EnhancedBacktester:
    """增强回测引擎"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.results = {}
    
    def run_walk_forward(self, klines: List[Dict], 
                         strategy_func: Callable,
                         train_window: int = 60,
                         test_window: int = 20,
                         min_train: int = 30) -> Dict:
        """
        Walk-forward分析
        
        Args:
            klines: K线数据
            strategy_func: 策略函数 (bar, position -> 'buy'/'sell'/'hold')
            train_window: 训练窗口 (天)
            test_window: 测试窗口 (天)
            min_train: 最小训练数据量
        
        Returns:
            回测结果统计
        """
        if len(klines) < train_window + test_window:
            return {'error': '数据不足'}
        
        results = []
        equity_curves = []
        
        i = train_window
        while i + test_window <= len(klines):
            train_data = klines[i - train_window:i]
            test_data = klines[i:i + test_window]
            
            # 使用训练数据优化参数 (简化版)
            optimal_params = self._optimize_params(train_data, strategy_func)
            
            # 回测测试数据
            equity, trades = self._backtest_segment(
                test_data, strategy_func, optimal_params
            )
            
            results.append({
                'start_date': test_data[0]['date'],
                'end_date': test_data[-1]['date'],
                'return': (equity[-1] / equity[0] - 1) * 100,
                'sharpe': self._calculate_sharpe(equity),
                'max_dd': self._calculate_max_dd(equity),
                'num_trades': len(trades),
                'params': optimal_params
            })
            
            equity_curves.append(equity)
            i += test_window
        
        return self._aggregate_results(results, equity_curves)
    
    def _optimize_params(self, train_data: List[Dict], 
                         strategy_func: Callable) -> Dict:
        """参数优化 (网格搜索)"""
        # 简化参数网格
        param_grid = {
            'stop_loss_pct': [0.03, 0.05, 0.07],
            'take_profit_pct': [0.06, 0.10, 0.15],
            'ma_period': [10, 20, 30]
        }
        
        best_sharpe = -999
        best_params = {}
        
        for params in product(*param_grid.values()):
            p = dict(zip(param_grid.keys(), params))
            equity, _ = self._backtest_segment(train_data, strategy_func, p)
            
            sharpe = self._calculate_sharpe(equity)
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = p
        
        return best_params
    
    def _backtest_segment(self, data: List[Dict], strategy_func: Callable,
                          params: Dict) -> Tuple[List[float], List[Dict]]:
        """回测一个阶段"""
        capital = self.initial_capital
        position = 0
        entry_price = 0
        trades = []
        equity = [capital]
        
        for bar in data:
            signal = strategy_func(bar, position, capital, params)
            price = bar['close']
            
            if signal == 'buy' and position == 0 and capital > price * 100:
                # 买入
                shares = int(capital * 0.95 / price)
                cost = shares * price * 1.0003  # 佣金
                capital -= cost
                position = shares
                entry_price = price
                trades.append({'type': 'buy', 'price': price, 'date': bar['date']})
            
            elif signal == 'sell' and position > 0:
                # 止损/止盈检查
                pnl_pct = (price - entry_price) / entry_price
                
                if pnl_pct <= -params['stop_loss_pct'] or \
                   pnl_pct >= params['take_profit_pct']:
                    revenue = position * price * 0.9987  # 佣金+印花税
                    capital += revenue
                    trades.append({
                        'type': 'sell', 'price': price, 'date': bar['date'],
                        'pnl_pct': pnl_pct * 100
                    })
                    position = 0
            
            equity.append(capital + position * price)
        
        return equity, trades
    
    def _calculate_sharpe(self, equity: List[float]) -> float:
        """计算Sharpe比率"""
        if len(equity) < 2:
            return 0
        returns = np.diff(equity) / equity[:-1]
        if np.std(returns) == 0:
            return 0
        return np.mean(returns) / np.std(returns) * np.sqrt(252)
    
    def _calculate_max_dd(self, equity: List[float]) -> float:
        """计算最大回撤"""
        peak = equity[0]
        max_dd = 0
        for v in equity:
            if v > peak:
                peak = v
            dd = (peak - v) / peak * 100
            if dd > max_dd:
                max_dd = dd
        return max_dd
    
    def _aggregate_results(self, results: List[Dict], 
                           equity_curves: List[List[float]]) -> Dict:
        """汇总结果"""
        if not results:
            return {'error': '无结果'}
        
        returns = [r['return'] for r in results]
        sharpes = [r['sharpe'] for r in results]
        max_dds = [r['max_dd'] for r in results]
        
        # 合并权益曲线
        combined_equity = []
        for eq in equity_curves:
            combined_equity.extend(eq[1:])  # 避免重复
        
        return {
            'num_segments': len(results),
            'avg_return': round(np.mean(returns), 2),
            'std_return': round(np.std(returns), 2),
            'avg_sharpe': round(np.mean(sharpes), 2),
            'avg_max_dd': round(np.mean(max_dds), 2),
            'win_rate': round(len([r for r in results if r['return'] > 0]) / len(results) * 100, 1),
            'total_return': round((combined_equity[-1] / self.initial_capital - 1) * 100, 2) if combined_equity else 0,
            'segment_results': results
        }
    
    def run_monte_carlo(self, equity_curve: List[float], 
                        n_simulations: int = 1000) -> Dict:
        """
        Monte Carlo模拟
        
        Args:
            equity_curve: 权益曲线
            n_simulations: 模拟次数
        
        Returns:
            模拟结果
        """
        if len(equity_curve) < 2:
            return {'error': '数据不足'}
        
        returns = np.diff(equity_curve) / equity_curve[:-1]
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        simulations = []
        for _ in range(n_simulations):
            # 随机采样收益率
            simulated_returns = np.random.choice(returns, size=len(returns))
            simulated_equity = [1.0]
            for r in simulated_returns:
                simulated_equity.append(simulated_equity[-1] * (1 + r))
            simulations.append(simulated_equity[-1])
        
        simulations = np.array(simulations) * self.initial_capital
        
        return {
            'mean_final': round(np.mean(simulations), 2),
            'std_final': round(np.std(simulations), 2),
            'percentile_5': round(np.percentile(simulations, 5), 2),
            'percentile_50': round(np.percentile(simulations, 50), 2),
            'percentile_95': round(np.percentile(simulations, 95), 2),
            'prob_profit': round(len([s for s in simulations if s > self.initial_capital]) / n_simulations * 100, 1)
        }


class ParameterOptimizer:
    """参数优化器"""
    
    def __init__(self):
        self.best_params = {}
        self.optimization_history = []
    
    def grid_search(self, param_grid: Dict, 
                    objective_func: Callable) -> Dict:
        """
        网格搜索优化
        
        Args:
            param_grid: 参数网格
            objective_func: 目标函数
        
        Returns:
            最优参数
        """
        best_score = -999
        best_params = {}
        
        for params in product(*param_grid.values()):
            p = dict(zip(param_grid.keys(), params))
            score = objective_func(p)
            
            self.optimization_history.append({
                'params': p,
                'score': score
            })
            
            if score > best_score:
                best_score = score
                best_params = p
        
        self.best_params = best_params
        return best_params
    
    def bayesian_optimization(self, param_bounds: Dict,
                              objective_func: Callable,
                              n_iterations: int = 20) -> Dict:
        """贝叶斯优化 (简化版)"""
        # 简化实现: 使用随机采样 + 梯度上升
        best_score = -999
        best_params = {}
        
        for _ in range(n_iterations):
            # 随机采样参数
            params = {k: np.random.uniform(v[0], v[1]) 
                     for k, v in param_bounds.items()}
            
            score = objective_func(params)
            
            if score > best_score:
                best_score = score
                best_params = params
        
        self.best_params = best_params
        return best_params


# 全局实例
enhanced_backtester = EnhancedBacktester()
parameter_optimizer = ParameterOptimizer()
