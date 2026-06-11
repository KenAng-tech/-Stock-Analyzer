"""
回测引擎模块
支持多股票回测、参数敏感性分析、Sharpe比率等指标
"""

import numpy as np
from typing import Dict, List, Optional, Callable
from datetime import datetime


class Backtester:
    """回测引擎"""
    
    def __init__(self, strategy: Callable, data: List[Dict], 
                 initial_capital: float = 100000,
                 commission: float = 0.0003):
        self.strategy = strategy
        self.data = data
        self.initial_capital = initial_capital
        self.commission = commission
        self.positions = []
        self.trades = []
        self.equity_curve = []
    
    def run(self) -> Dict:
        """运行回测"""
        capital = self.initial_capital
        position = 0
        entry_price = 0
        entry_date = None
        
        for bar in self.data:
            price = bar['close']
            date = bar['date']
            
            # 获取策略信号
            signal = self.strategy(bar, position, capital)
            
            if signal == 'BUY' and capital > 0:
                # 买入
                shares = int(capital * 0.95 / price)  # 95%仓位
                if shares > 0:
                    cost = shares * price * (1 + self.commission)
                    capital -= cost
                    position = shares
                    entry_price = price
                    entry_date = date
                    self.trades.append({
                        'date': date,
                        'type': 'BUY',
                        'price': price,
                        'shares': shares,
                        'value': cost
                    })
            
            elif signal == 'SELL' and position > 0:
                # 卖出
                revenue = position * price * (1 - self.commission)
                capital += revenue
                self.trades.append({
                    'date': date,
                    'type': 'SELL',
                    'price': price,
                    'shares': position,
                    'value': revenue,
                    'profit': revenue - position * entry_price
                })
                position = 0
            
            # 记录权益
            total_value = capital + position * price
            self.equity_curve.append({
                'date': date,
                'value': total_value,
                'return': (total_value / self.initial_capital - 1) * 100
            })
        
        return self.generate_report()
    
    def generate_report(self) -> Dict:
        """生成回测报告"""
        if not self.equity_curve:
            return {'error': 'No data'}
        
        values = [e['value'] for e in self.equity_curve]
        returns = [e['return'] for e in self.equity_curve]
        
        # 计算指标
        total_return = (values[-1] / self.initial_capital - 1) * 100
        daily_returns = np.diff(values) / values[:-1]
        sharpe_ratio = (np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
                       if np.std(daily_returns) > 0 else 0)
        
        # 最大回撤
        peak = values[0]
        max_dd = 0
        for v in values:
            if v > peak:
                peak = v
            dd = (peak - v) / peak * 100
            if dd > max_dd:
                max_dd = dd
        
        # 胜率
        sell_trades = [t for t in self.trades if t['type'] == 'SELL']
        wins = sum(1 for t in sell_trades if t.get('profit', 0) > 0)
        win_rate = (wins / len(sell_trades) * 100) if sell_trades else 0
        
        # 盈亏比
        profits = [t['profit'] for t in sell_trades if t.get('profit', 0) > 0]
        losses = [t['profit'] for t in sell_trades if t.get('profit', 0) <= 0]
        avg_profit = np.mean(profits) if profits else 0
        avg_loss = abs(np.mean(losses)) if losses else 1
        profit_factor = avg_profit / avg_loss if avg_loss > 0 else 0
        
        return {
            'total_return': round(total_return, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'max_drawdown': round(max_dd, 2),
            'win_rate': round(win_rate, 1),
            'profit_factor': round(profit_factor, 2),
            'total_trades': len(self.trades),
            'buy_trades': sum(1 for t in self.trades if t['type'] == 'BUY'),
            'sell_trades': len(sell_trades),
            'final_value': round(values[-1], 2),
            'equity_curve': self.equity_curve[-50:],  # 最近50条
        }


class ParameterSensitivity:
    """参数敏感性分析"""
    
    def __init__(self, strategy: Callable, data: List[Dict]):
        self.strategy = strategy
        self.data = data
    
    def analyze(self, param_name: str, param_range: List[float],
                **kwargs) -> Dict:
        """分析参数敏感性"""
        results = []
        for value in param_range:
            kwargs[param_name] = value
            bt = Backtester(self.strategy, self.data, **kwargs)
            report = bt.run()
            results.append({
                param_name: value,
                'sharpe_ratio': report.get('sharpe_ratio', 0),
                'max_drawdown': report.get('max_drawdown', 0),
                'total_return': report.get('total_return', 0),
            })
        
        # 找到最优参数
        best = max(results, key=lambda x: x.get('sharpe_ratio', 0))
        
        return {
            'param_name': param_name,
            'results': results,
            'best': best,
        }
