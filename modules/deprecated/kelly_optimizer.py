"""
Kelly Criterion Optimizer - V2
Enhanced Kelly position sizing with real win rate, CVaR constraint,
and volatility targeting.
"""

import math
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta


class KellyOptimizer:
    """Enhanced Kelly Criterion Optimizer"""
    
    def __init__(self):
        # Historical performance tracking
        self._trade_history: List[Dict] = []
        self._signal_history: Dict[str, Dict] = {}
        
        # Kelly parameters
        self.kelly_fraction = 0.5  # Fractional Kelly (half Kelly is optimal)
        self.max_position = 0.25   # Max 25% position
        self.min_position = 0.05   # Min 5% position
        self.cvar_confidence = 0.95
        self.cvar_limit = 0.05     # 5% CVaR limit
        
    def add_trade(self, signal_type: str, pnl_pct: float, is_win: bool):
        """Record a trade for Kelly calculation"""
        self._trade_history.append({
            'signal_type': signal_type,
            'pnl_pct': pnl_pct,
            'is_win': is_win,
            'timestamp': datetime.now()
        })
        self.update_signal_history(signal_type, is_win)
    
    def update_signal_history(self, signal_type: str, is_win: bool):
        """Update Bayesian signal history"""
        if signal_type not in self._signal_history:
            self._signal_history[signal_type] = {'wins': 0, 'total': 0, 'pnl': []}
        
        self._signal_history[signal_type]['total'] += 1
        if is_win:
            self._signal_history[signal_type]['wins'] += 1
        self._signal_history[signal_type]['pnl'].append(is_win)
    
    def calculate_real_win_rate(self, lookback_days: int = 60) -> float:
        """Calculate real win rate from recent trades"""
        if not self._trade_history:
            return 0.55  # Default
        
        cutoff = datetime.now() - timedelta(days=lookback_days)
        recent_trades = [t for t in self._trade_history if t['timestamp'] >= cutoff]
        
        if not recent_trades:
            return 0.55
        
        wins = sum(1 for t in recent_trades if t['is_win'])
        return wins / len(recent_trades)
    
    def calculate_real_win_loss_ratio(self, lookback_days: int = 60) -> float:
        """Calculate real win/loss ratio from recent trades"""
        if not self._trade_history:
            return 2.0
        
        cutoff = datetime.now() - timedelta(days=lookback_days)
        recent_trades = [t for t in self._trade_history if t['timestamp'] >= cutoff]
        
        if not recent_trades:
            return 2.0
        
        wins = [t['pnl_pct'] for t in recent_trades if t['is_win']]
        losses = [abs(t['pnl_pct']) for t in recent_trades if not t['is_win'] and t['pnl_pct'] != 0]
        
        avg_win = np.mean(wins) if wins else 1.0
        avg_loss = np.mean(losses) if losses else 1.0
        
        return avg_win / avg_loss if avg_loss > 0 else 2.0
    
    def calculate_kelly_fraction(self, 
                                  win_rate: Optional[float] = None,
                                  win_loss_ratio: Optional[float] = None,
                                  volatility: float = 0.15) -> float:
        """
        Calculate Kelly fraction using real parameters
        
        Kelly % = W - [(1-W)/R]
        where W = win rate, R = win/loss ratio
        
        Uses fractional Kelly (half Kelly) for safety.
        """
        if win_rate is None:
            win_rate = self.calculate_real_win_rate()
        if win_loss_ratio is None:
            win_loss_ratio = self.calculate_real_win_loss_ratio()
        
        # Kelly formula
        kelly = win_rate - ((1 - win_rate) / win_loss_ratio)
        
        # Apply fractional Kelly (0.5x is optimal per theory)
        kelly *= self.kelly_fraction
        
        # Adjust for volatility (higher vol -> smaller position)
        vol_adjustment = 1.0 - (volatility - 0.10) * 2.0
        kelly *= max(0.5, min(1.5, vol_adjustment))
        
        # Clamp to valid range
        kelly = max(self.min_position, min(self.max_position, kelly))
        
        return round(kelly, 3)
    
    def calculate_cvar_risk(self, returns: List[float], 
                             confidence: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (CVaR)"""
        if not returns:
            return 0.05
        
        sorted_returns = sorted(returns)
        cutoff = int(len(sorted_returns) * (1 - confidence))
        cvar_returns = sorted_returns[:cutoff]
        
        if not cvar_returns:
            return 0.05
        
        return abs(np.mean(cvar_returns))
    
    def get_position_size(self, stock_data: Dict, 
                           total_capital: float = 100000) -> Dict:
        """
        Get optimal position size using Kelly + CVaR
        
        Returns:
            Dict with position size, risk metrics, and recommendations
        """
        price = stock_data.get('price', 0)
        volatility = stock_data.get('std_dev', 0) / price if price > 0 else 0.15
        
        # Calculate Kelly position
        kelly_frac = self.calculate_kelly_fraction(
            volatility=volatility
        )
        
        # Calculate CVaR constraint
        # Simulate returns based on current volatility
        simulated_returns = np.random.normal(
            loc=0.02,  # Expected daily return
            scale=volatility / math.sqrt(252),  # Daily vol
            size=1000
        )
        cvar_risk = self.calculate_cvar_risk(simulated_returns.tolist())
        
        # CVaR adjustment (reduce position if CVaR is high)
        if cvar_risk > self.cvar_limit:
            cvar_adj = 1.0 - (cvar_risk - self.cvar_limit) * 2.0
            kelly_frac *= max(0.5, min(1.0, cvar_adj))
        
        # Calculate position value
        position_value = total_capital * kelly_frac
        shares = int(position_value / price) if price > 0 else 0
        
        # Calculate stop loss using ATR
        atr = stock_data.get('atr', 0)
        if atr > 0:
            stop_loss_price = price - (atr * 2.0)
        else:
            stop_loss_price = price * 0.95
        
        # Calculate take profit
        take_profit_price = price + (atr * 3.0) if atr > 0 else price * 1.10
        
        return {
            'kelly_fraction': float(kelly_frac),
            'position_value': round(float(position_value), 2),
            'shares': int(shares),
            'stop_loss': round(float(stop_loss_price), 2),
            'take_profit': round(float(take_profit_price), 2),
            'cvar_risk': round(float(cvar_risk), 4),
            'cvar_within_limit': bool(cvar_risk <= self.cvar_limit),
            'win_rate': round(float(self.calculate_real_win_rate()), 3),
            'win_loss_ratio': round(float(self.calculate_real_win_loss_ratio()), 2),
            'volatility': round(float(volatility), 4)
        }
    
    def get_optimal_kelly_params(self) -> Dict:
        """Get optimal Kelly parameters based on historical data"""
        win_rate = self.calculate_real_win_rate()
        win_loss_ratio = self.calculate_real_win_loss_ratio()
        
        # Kelly formula: W - (1-W)/R
        kelly = win_rate - ((1 - win_rate) / win_loss_ratio)
        fractional_kelly = kelly * self.kelly_fraction
        
        return {
            'win_rate': round(float(win_rate), 3),
            'win_loss_ratio': round(float(win_loss_ratio), 2),
            'kelly_fraction': round(float(kelly), 3),
            'fractional_kelly': round(float(fractional_kelly), 3),
            'recommended_position': round(float(fractional_kelly), 3),
            'confidence': 'high' if win_rate > 0.55 else 'medium' if win_rate > 0.50 else 'low'
        }


# Global instance
kelly_optimizer = KellyOptimizer()
