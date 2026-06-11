"""
MACD + Bollinger Bands Strategy Module
Enhanced with divergence detection and multi-timeframe analysis.
"""

import math
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta


class MACD_Bollinger:
    """Combined MACD and Bollinger Bands strategy"""
    
    def __init__(self):
        # MACD parameters
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        
        # Bollinger Band parameters
        self.bb_period = 20
        self.bb_std = 2.0
    
    def calculate_ema(self, prices: List[float], period: int) -> List[float]:
        """Calculate EMA series"""
        if not prices:
            return []
        multiplier = 2 / (period + 1)
        ema = [prices[0]]
        for price in prices[1:]:
            ema.append((price - ema[-1]) * multiplier + ema[-1])
        return ema
    
    def calculate_macd(self, prices: List[float]) -> Dict:
        """Calculate MACD indicators"""
        if len(prices) < self.macd_slow + self.macd_signal:
            return {'macd': 0, 'signal': 0, 'histogram': 0, 'trend': 'neutral'}
        
        ema_fast = self.calculate_ema(prices, self.macd_fast)
        ema_slow = self.calculate_ema(prices, self.macd_slow)
        
        macd_line = [f - s for f, s in zip(ema_fast, ema_slow)]
        signal_line = self.calculate_ema(macd_line[-(self.macd_slow + self.macd_signal):], self.macd_signal)
        
        macd = macd_line[-1]
        signal = signal_line[-1]
        histogram = macd - signal
        
        # Determine trend
        if histogram > 0 and macd > signal:
            trend = 'bullish'
        elif histogram < 0 and macd < signal:
            trend = 'bearish'
        else:
            trend = 'neutral'
        
        # Detect crossover
        prev_histogram = macd_line[-2] - signal_line[-2]
        crossover = 'none'
        if prev_histogram <= 0 and histogram > 0:
            crossover = 'golden_cross'
        elif prev_histogram >= 0 and histogram < 0:
            crossover = 'death_cross'
        
        return {
            'macd': round(macd, 4),
            'signal': round(signal, 4),
            'histogram': round(histogram, 4),
            'trend': trend,
            'crossover': crossover,
            'divergence': self.detect_macd_divergence(prices, macd_line)
        }
    
    def detect_macd_divergence(self, prices: List[float], macd_line: List[float], lookback: int = 20) -> Dict:
        """Detect MACD divergence"""
        if len(prices) < lookback * 2:
            return {'type': 'none', 'strength': 'weak'}
        
        recent_prices = prices[-lookback:]
        recent_macd = macd_line[-lookback:]
        prev_prices = prices[-lookback*2:-lookback]
        prev_macd = macd_line[-lookback*2:-lookback]
        
        curr_high = max(recent_prices)
        curr_low = min(recent_prices)
        prev_high = max(prev_prices)
        prev_low = min(prev_prices)
        
        curr_macd_high = max(recent_macd)
        curr_macd_low = min(recent_macd)
        prev_macd_high = max(prev_macd)
        prev_macd_low = min(prev_macd)
        
        if curr_low < prev_low and curr_macd_low > prev_macd_low:
            return {'type': 'bullish', 'strength': 'strong' if (prev_low - curr_low) > (curr_macd_low - prev_macd_low) else 'moderate'}
        if curr_high > prev_high and curr_macd_high < prev_macd_high:
            return {'type': 'bearish', 'strength': 'strong' if (curr_high - prev_high) > (prev_macd_high - curr_macd_high) else 'moderate'}
        return {'type': 'none', 'strength': 'weak'}
    
    def calculate_bollinger_bands(self, prices: List[float]) -> Dict:
        """Calculate Bollinger Bands"""
        if len(prices) < self.bb_period:
            return {'upper': 0, 'middle': 0, 'lower': 0, 'bandwidth': 0, 'percent_b': 0.5}
        
        recent = prices[-self.bb_period:]
        middle = np.mean(recent)
        std = np.std(recent)
        upper = middle + self.bb_std * std
        lower = middle - self.bb_std * std
        
        current_price = prices[-1]
        bandwidth = (upper - lower) / middle if middle > 0 else 0
        percent_b = (current_price - lower) / (upper - lower) if upper != lower else 0.5
        
        # Bandwidth classification
        if bandwidth < 0.05:
            width_class = 'narrow'  # Squeeze
        elif bandwidth < 0.15:
            width_class = 'normal'
        else:
            width_class = 'wide'
        
        return {
            'upper': round(upper, 2),
            'middle': round(middle, 2),
            'lower': round(lower, 2),
            'bandwidth': round(bandwidth, 4),
            'percent_b': round(percent_b, 3),
            'width_class': width_class,
            'current_price': round(current_price, 2)
        }
    
    def get_macd_signal(self, macd_data: Dict) -> Dict:
        """Get MACD trading signal"""
        histogram = macd_data.get('histogram', 0)
        trend = macd_data.get('trend', 'neutral')
        crossover = macd_data.get('crossover', 'none')
        divergence = macd_data.get('divergence', {'type': 'none'})
        
        if crossover == 'golden_cross':
            return {'signal': 'BUY', 'strength': 'strong', 'reason': 'MACD金叉'}
        elif crossover == 'death_cross':
            return {'signal': 'SELL', 'strength': 'strong', 'reason': 'MACD死叉'}
        elif trend == 'bullish' and histogram > 0:
            return {'signal': 'BUY', 'strength': 'moderate', 'reason': 'MACD多头趋势'}
        elif trend == 'bearish' and histogram < 0:
            return {'signal': 'SELL', 'strength': 'moderate', 'reason': 'MACD空头趋势'}
        elif divergence['type'] == 'bullish':
            return {'signal': 'BUY', 'strength': divergence['strength'], 'reason': 'MACD看涨背离'}
        elif divergence['type'] == 'bearish':
            return {'signal': 'SELL', 'strength': divergence['strength'], 'reason': 'MACD看跌背离'}
        else:
            return {'signal': 'HOLD', 'strength': 'weak', 'reason': 'MACD中性'}
    
    def get_bollinger_signal(self, bb_data: Dict) -> Dict:
        """Get Bollinger Bands trading signal"""
        percent_b = bb_data.get('percent_b', 0.5)
        width_class = bb_data.get('width_class', 'normal')
        
        if percent_b > 0.9:
            signal = 'SELL'
            strength = 'strong'
            reason = '触及上轨，超买'
        elif percent_b > 0.8:
            signal = 'SELL'
            strength = 'moderate'
            reason = '接近上轨，注意回调'
        elif percent_b < 0.1:
            signal = 'BUY'
            strength = 'strong'
            reason = '触及下轨，超卖'
        elif percent_b < 0.2:
            signal = 'BUY'
            strength = 'moderate'
            reason = '接近下轨，关注买入'
        elif width_class == 'narrow':
            signal = 'HOLD'
            strength = 'moderate'
            reason = '布林带收窄，等待突破'
        else:
            signal = 'HOLD'
            strength = 'weak'
            reason = '布林带正常'
        
        return {'signal': signal, 'strength': strength, 'reason': reason, 'percent_b': percent_b}
    
    def get_combined_signal(self, macd_data: Dict, bb_data: Dict) -> Dict:
        """Get combined MACD + Bollinger signal"""
        macd_signal = self.get_macd_signal(macd_data)
        bb_signal = self.get_bollinger_signal(bb_data)
        
        # Combine signals
        if macd_signal['signal'] == bb_signal['signal']:
            combined_signal = macd_signal['signal']
            combined_strength = 'strong'
        elif macd_signal['signal'] == 'HOLD':
            combined_signal = bb_signal['signal']
            combined_strength = bb_signal['strength']
        elif bb_signal['signal'] == 'HOLD':
            combined_signal = macd_signal['signal']
            combined_strength = macd_signal['strength']
        else:
            combined_signal = 'HOLD'
            combined_strength = 'moderate'
        
        return {
            'combined_signal': combined_signal,
            'combined_strength': combined_strength,
            'macd': macd_signal,
            'bollinger': bb_signal,
            'description': f'MACD: {macd_signal["reason"]}, 布林带: {bb_signal["reason"]}'
        }


# Global instance
macd_bollinger = MACD_Bollinger()
