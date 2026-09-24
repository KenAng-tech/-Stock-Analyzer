"""
Absolute Momentum Filter Module - V1
Implements absolute momentum (trend) filter to reduce positions
when price is below long-term moving average.

Based on QuantPedia's Dual Momentum GTAA Strategy research.
"""

import numpy as np
from typing import Dict, List, Optional


class AbsoluteMomentumFilter:
    """
    Absolute Momentum Filter
    
    Key concept: When price < long-term MA, reduce position size
    even if other signals are bullish.
    
    - MA periods: 50, 100, 200 (configurable)
    - Momentum score: (price - MA) / MA
    - Filter strength: 0.0 (full short) to 1.5 (strong long)
    """
    
    def __init__(self, ma_periods: List[int] = None):
        self.ma_periods = ma_periods or [50, 100, 200]
        self.default_momentum_threshold = 0.0
    
    def calculate_momentum(self, price: float, ma: float) -> float:
        """Calculate momentum score"""
        if ma == 0:
            return 0.5
        return (price - ma) / ma
    
    def get_momentum_signal(self, momentum: float) -> Dict:
        """Get momentum signal based on score"""
        if momentum > 0.05:
            return {
                'signal': 'BULLISH',
                'strength': 'strong' if momentum > 0.10 else 'moderate',
                'position_multiplier': 1.2,
                'description': f'趋势向上 (动量 {momentum:.1%})'
            }
        elif momentum > 0.0:
            return {
                'signal': 'SLIGHTLY_BULLISH',
                'strength': 'weak',
                'position_multiplier': 1.0,
                'description': f'轻微向上 (动量 {momentum:.1%})'
            }
        elif momentum > -0.05:
            return {
                'signal': 'SLIGHTLY_BEARISH',
                'strength': 'weak',
                'position_multiplier': 0.8,
                'description': f'轻微向下 (动量 {momentum:.1%})'
            }
        else:
            return {
                'signal': 'BEARISH',
                'strength': 'strong' if momentum < -0.10 else 'moderate',
                'position_multiplier': 0.5,
                'description': f'趋势向下 (动量 {momentum:.1%})'
            }
    
    def calculate_ma(self, prices: List[float], period: int) -> float:
        """Calculate Simple Moving Average"""
        if len(prices) < period:
            return np.mean(prices) if prices else 0
        return np.mean(prices[-period:])
    
    def calculate_all_mas(self, prices: List[float]) -> Dict:
        """Calculate MAs for all periods"""
        result = {}
        for period in self.ma_periods:
            result[f'ma_{period}'] = self.calculate_ma(prices, period)
        return result
    
    def get_absolute_momentum(self, stock_data: Dict, klines: List[Dict] = None) -> Dict:
        """
        Get absolute momentum signal
        
        Returns comprehensive momentum analysis with:
        - Multiple MA levels
        - Momentum score
        - Position multiplier
        - Signal strength
        """
        price = stock_data.get('price', 0)
        
        if klines:
            closes = [k['close'] for k in klines if k['close'] > 0]
        else:
            closes = [price]
        
        # Calculate MAs
        ma_dict = self.calculate_all_mas(closes)
        
        # Calculate momentum for each MA
        momentums = {}
        for period in self.ma_periods:
            ma = ma_dict.get(f'ma_{period}', 0)
            momentums[period] = self.calculate_momentum(price, ma)
        
        # Overall momentum (weighted average)
        weights = {50: 0.2, 100: 0.3, 200: 0.5}
        overall_momentum = sum(
            momentums.get(p, 0) * weights.get(p, 0)
            for p in self.ma_periods
        )
        
        # Get signal
        signal = self.get_momentum_signal(overall_momentum)
        
        # Position multiplier based on momentum
        pos_multiplier = signal['position_multiplier']
        
        # Additional adjustment: if price above ALL MAs, add bonus
        above_all_mas = all(
            price > ma_dict.get(f'ma_{p}', 0)
            for p in self.ma_periods
        )
        if above_all_mas:
            pos_multiplier = min(1.5, pos_multiplier * 1.1)
            signal['description'] += '，站上所有均线'
        
        return {
            'momentum': round(overall_momentum, 4),
            'signal': signal['signal'],
            'strength': signal['strength'],
            'position_multiplier': round(pos_multiplier, 3),
            'description': signal['description'],
            'ma_levels': {
                'ma_50': round(ma_dict.get('ma_50', 0), 2),
                'ma_100': round(ma_dict.get('ma_100', 0), 2),
                'ma_200': round(ma_dict.get('ma_200', 0), 2),
            },
            'momentums': {
                p: round(momentums.get(p, 0), 4)
                for p in self.ma_periods
            },
            'above_all_mas': above_all_mas
        }


# Global instance
absolute_momentum = AbsoluteMomentumFilter()
