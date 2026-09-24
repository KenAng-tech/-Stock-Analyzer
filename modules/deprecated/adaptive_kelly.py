"""
Adaptive Kelly Criterion Module - V1
Enhanced Kelly position sizing with volatility and drawdown adjustments.

Improvement over standard Kelly:
1. Volatility adjustment: Higher vol -> smaller position
2. Drawdown adjustment: Higher drawdown -> smaller position
3. Adaptive fraction: Adjusts based on market conditions
"""

import math
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta


class AdaptiveKelly:
    """
    Adaptive Kelly Criterion
    
    Kelly % = W - [(1-W)/R]
    
    Adjustments:
    - Volatility: Kelly * (1 / (1 + vol))
    - Drawdown: Kelly * (1 / (1 + max_drawdown * 2))
    - Adaptive fraction: k varies based on win rate confidence
    """
    
    def __init__(self, base_kelly_fraction: float = 0.5):
        self.base_kelly_fraction = base_kelly_fraction
        self.min_position = 0.05
        self.max_position = 0.50
        self.default_win_rate = 0.55
        self.default_win_loss_ratio = 2.0
        
        # Historical trade tracking
        self._trade_history: List[Dict] = []
        self._win_rate_history: List[float] = []
    
    def calculate_kelly_fraction(self,
                                  win_rate: float = None,
                                  win_loss_ratio: float = None,
                                  volatility: float = 0.15,
                                  max_drawdown: float = 0.10) -> float:
        """
        Calculate adaptive Kelly fraction
        
        Args:
            win_rate: Historical win rate (default: 0.55)
            win_loss_ratio: Win/Loss ratio (default: 2.0)
            volatility: Annualized volatility (default: 0.15)
            max_drawdown: Maximum drawdown (default: 0.10)
        
        Returns:
            Kelly fraction (0.05 to 0.50)
        """
        if win_rate is None:
            win_rate = self.default_win_rate
        if win_loss_ratio is None:
            win_loss_ratio = self.default_win_loss_ratio
        
        # Base Kelly formula
        base_kelly = win_rate - ((1 - win_rate) / win_loss_ratio)
        
        # Apply fractional Kelly (0.5x is optimal)
        kelly = base_kelly * self.base_kelly_fraction
        
        # Volatility adjustment: Kelly * (1 / (1 + vol))
        vol_adj = 1.0 / (1.0 + volatility)
        kelly *= vol_adj
        
        # Drawdown adjustment: Kelly * (1 / (1 + max_drawdown * 2))
        dd_adj = 1.0 / (1.0 + max_drawdown * 2)
        kelly *= dd_adj
        
        # Clamp to valid range
        kelly = max(self.min_position, min(self.max_position, kelly))
        
        return round(kelly, 3)
    
    def calculate_adaptive_fraction(self, win_rate: float, 
                                     lookback: int = 60) -> float:
        """
        Calculate adaptive Kelly fraction based on win rate confidence
        
        - High win rate confidence (many trades): use full fraction
        - Low win rate confidence (few trades): use conservative fraction
        """
        n = len(self._trade_history) if self._trade_history else 10
        
        # Confidence score based on sample size
        confidence = min(1.0, n / lookback)
        
        # Adaptive fraction: range from 0.25 (conservative) to 0.75 (aggressive)
        adaptive_fraction = 0.25 + 0.50 * confidence
        
        # Adjust for win rate
        if win_rate > 0.60:
            adaptive_fraction *= 1.1  # Bonus for high win rate
        elif win_rate < 0.45:
            adaptive_fraction *= 0.9  # Penalty for low win rate
        
        return round(min(0.75, max(0.25, adaptive_fraction)), 3)
    
    def get_position_size(self, stock_data: Dict,
                           total_capital: float = 100000,
                           win_rate: float = None,
                           win_loss_ratio: float = None,
                           volatility: float = 0.15,
                           max_drawdown: float = 0.10) -> Dict:
        """
        Get optimal position size using adaptive Kelly
        
        Returns comprehensive position sizing with all adjustments.
        """
        price = stock_data.get('price', 0)
        
        # Calculate adaptive Kelly fraction
        if win_rate is None:
            win_rate = self.calculate_real_win_rate()
        if win_loss_ratio is None:
            win_loss_ratio = self.calculate_real_win_loss_ratio()
        
        kelly_fraction = self.calculate_kelly_fraction(
            win_rate=win_rate,
            win_loss_ratio=win_loss_ratio,
            volatility=volatility,
            max_drawdown=max_drawdown
        )
        
        adaptive_fraction = self.calculate_adaptive_fraction(win_rate)
        
        # Calculate position value
        position_value = total_capital * kelly_fraction
        shares = int(position_value / price) if price > 0 else 0
        
        # Risk metrics
        expected_return = win_rate * win_loss_ratio - (1 - win_rate)
        risk_adjusted_return = expected_return * kelly_fraction
        
        return {
            'kelly_fraction': kelly_fraction,
            'adaptive_fraction': adaptive_fraction,
            'win_rate': round(win_rate, 3),
            'win_loss_ratio': round(win_loss_ratio, 2),
            'position_value': round(position_value, 2),
            'shares': int(shares),
            'volatility': round(volatility, 4),
            'max_drawdown': round(max_drawdown, 4),
            'expected_return': round(expected_return, 3),
            'risk_adjusted_return': round(risk_adjusted_return, 3),
            'recommendation': self._get_recommendation(kelly_fraction)
        }
    
    def calculate_real_win_rate(self, lookback_days: int = 60) -> float:
        """Calculate real win rate from recent trades"""
        if not self._trade_history:
            return self.default_win_rate
        
        cutoff = datetime.now() - timedelta(days=lookback_days)
        recent = [t for t in self._trade_history if t['timestamp'] >= cutoff]
        
        if not recent:
            return self.default_win_rate
        
        wins = sum(1 for t in recent if t['is_win'])
        return wins / len(recent)
    
    def calculate_real_win_loss_ratio(self, lookback_days: int = 60) -> float:
        """Calculate real win/loss ratio"""
        if not self._trade_history:
            return self.default_win_loss_ratio
        
        cutoff = datetime.now() - timedelta(days=lookback_days)
        recent = [t for t in self._trade_history if t['timestamp'] >= cutoff]
        
        if not recent:
            return self.default_win_loss_ratio
        
        wins = [t['pnl_pct'] for t in recent if t['is_win']]
        losses = [abs(t['pnl_pct']) for t in recent if not t['is_win'] and t['pnl_pct'] != 0]
        
        avg_win = np.mean(wins) if wins else 1.0
        avg_loss = np.mean(losses) if losses else 1.0
        
        return avg_win / avg_loss if avg_loss > 0 else 2.0
    
    def add_trade(self, signal_type: str, pnl_pct: float, is_win: bool):
        """Record a trade"""
        self._trade_history.append({
            'signal_type': signal_type,
            'pnl_pct': pnl_pct,
            'is_win': is_win,
            'timestamp': datetime.now()
        })
    
    def _get_recommendation(self, kelly_fraction: float) -> str:
        """Get position recommendation"""
        if kelly_fraction >= 0.35:
            return '积极建仓'
        elif kelly_fraction >= 0.20:
            return '逐步建仓'
        elif kelly_fraction >= 0.10:
            return '轻仓试探'
        else:
            return '观望等待'


# Global instance
adaptive_kelly = AdaptiveKelly()
