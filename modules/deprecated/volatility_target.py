"""
Volatility Target Position Sizing Module
Implements volatility targeting for optimal position sizing.
"""

import math
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta


class VolatilityTarget:
    """Volatility targeting for position sizing"""
    
    def __init__(self, target_vol: float = 0.15, lookback: int = 60):
        self.target_vol = target_vol  # Target annual volatility (15%)
        self.lookback = lookback
        self.volatility_history: List[float] = []
    
    def calculate_realized_vol(self, prices: List[float]) -> float:
        """Calculate realized volatility from price series"""
        if len(prices) < 2:
            return self.target_vol
        returns = np.diff(np.log(prices))
        return np.std(returns) * math.sqrt(252)
    
    def calculate_historical_vol(self, stock_data: Dict) -> float:
        """Calculate historical volatility from stock data"""
        std_dev = stock_data.get('std_dev', 0)
        price = stock_data.get('price', 1)
        if price > 0:
            return (std_dev / price) * math.sqrt(252)
        return self.target_vol
    
    def calculate_implied_vol(self, stock_data: Dict) -> float:
        """Estimate implied volatility from ATR"""
        atr = stock_data.get('atr', 0)
        price = stock_data.get('price', 1)
        if atr > 0 and price > 0:
            daily_vol = atr / price
            return daily_vol * math.sqrt(252)
        return self.target_vol
    
    def get_volatility_level(self, vol: float) -> str:
        """Classify volatility level"""
        if vol < 0.10:
            return 'low'
        elif vol < 0.20:
            return 'medium'
        elif vol < 0.35:
            return 'high'
        else:
            return 'very_high'
    
    def calculate_target_position(self, stock_data: Dict, 
                                   total_capital: float = 100000,
                                   kelly_position: float = 0.15) -> Dict:
        """
        Calculate optimal position size using volatility targeting
        
        Position = Kelly_Position * (Target_Vol / Realized_Vol)
        """
        price = stock_data.get('price', 0)
        klines = stock_data.get('klines', [])
        
        # Calculate realized volatility
        if klines:
            prices = [k['close'] for k in klines[-self.lookback:]]
            realized_vol = self.calculate_realized_vol(prices)
        else:
            realized_vol = self.calculate_historical_vol(stock_data)
        
        # Volatility adjustment
        vol_adj = self.target_vol / realized_vol if realized_vol > 0 else 1.0
        vol_adj = max(0.5, min(2.0, vol_adj))  # Clamp between 0.5x and 2.0x
        
        # Volatility-adjusted position
        vol_position = kelly_position * vol_adj
        vol_position = max(0.05, min(0.50, vol_position))
        
        # Calculate position value
        position_value = total_capital * vol_position
        shares = int(position_value / price) if price > 0 else 0
        
        # Risk metrics
        expected_pnl = position_value * (realized_vol * math.sqrt(1/252))
        var_95 = position_value * 1.645 * realized_vol / math.sqrt(252)
        
        return {
            'realized_vol': round(realized_vol, 4),
            'target_vol': self.target_vol,
            'vol_adjustment': round(vol_adj, 3),
            'kelly_position': kelly_position,
            'vol_position': round(vol_position, 3),
            'position_value': round(position_value, 2),
            'shares': shares,
            'volatility_level': self.get_volatility_level(realized_vol),
            'expected_pnl': round(expected_pnl, 2),
            'var_95': round(var_95, 2),
            'risk_adjusted': True
        }
    
    def update_volatility_history(self, vol: float):
        """Update volatility history"""
        self.volatility_history.append(vol)
        if len(self.volatility_history) > 100:
            self.volatility_history = self.volatility_history[-100:]
    
    def get_volatility_trend(self) -> str:
        """Get volatility trend (increasing/decreasing/stable)"""
        if len(self.volatility_history) < 10:
            return 'stable'
        recent = np.mean(self.volatility_history[-5:])
        older = np.mean(self.volatility_history[-20:-5])
        if recent > older * 1.1:
            return 'increasing'
        elif recent < older * 0.9:
            return 'decreasing'
        return 'stable'


# Global instance
volatility_target = VolatilityTarget()
