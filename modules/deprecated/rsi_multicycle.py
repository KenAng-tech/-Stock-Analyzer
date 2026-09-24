"""
RSI Multi-Cycle Module
Enhanced RSI calculation with multiple timeframes and divergence detection.
"""

import math
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta


class RSI_MultiCycle:
    """Enhanced RSI with multi-cycle support"""
    
    def __init__(self):
        self.periods = {
            'rsi_6': 6,
            'rsi_14': 14,
            'rsi_25': 25,
            'rsi_60': 60,
        }
    
    def calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate RSI using Wilder's method"""
        if len(prices) < period + 1:
            return 50.0
        changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [max(0, c) for c in changes]
        losses = [max(0, -c) for c in changes]
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        return round(rsi, 2)
    
    def calculate_rsi_series(self, prices: List[float], period: int = 14) -> List[float]:
        """Calculate RSI series"""
        if len(prices) < period + 1:
            return [50.0] * len(prices)
        rsi_values = [50.0] * period
        changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [max(0, c) for c in changes]
        losses = [max(0, -c) for c in changes]
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            if avg_loss == 0:
                rsi = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            rsi_values.append(round(rsi, 2))
        return rsi_values
    
    def calculate_all_periods(self, prices: List[float]) -> Dict[str, float]:
        """Calculate RSI for all configured periods"""
        result = {}
        for name, period in self.periods.items():
            result[name] = self.calculate_rsi(prices, period)
        return result
    
    def detect_divergence(self, prices: List[float], rsi_values: List[float], lookback: int = 20) -> Dict:
        """Detect RSI divergence (bullish/bearish)"""
        if len(prices) < lookback * 2:
            return {'type': 'none', 'strength': 'weak'}
        recent_prices = prices[-lookback:]
        recent_rsi = rsi_values[-lookback:]
        prev_prices = prices[-lookback*2:-lookback]
        prev_rsi = rsi_values[-lookback*2:-lookback]
        curr_high = max(recent_prices)
        curr_low = min(recent_prices)
        prev_high = max(prev_prices)
        prev_low = min(prev_prices)
        curr_rsi_high = max(recent_rsi)
        curr_rsi_low = min(recent_rsi)
        prev_rsi_high = max(prev_rsi)
        prev_rsi_low = min(prev_rsi)
        if curr_low < prev_low and curr_rsi_low > prev_rsi_low:
            return {'type': 'bullish', 'strength': 'strong' if (prev_low - curr_low) > (curr_rsi_low - prev_rsi_low) else 'moderate', 'price_low': round(curr_low, 2), 'rsi_low': round(curr_rsi_low, 2)}
        if curr_high > prev_high and curr_rsi_high < prev_rsi_high:
            return {'type': 'bearish', 'strength': 'strong' if (curr_high - prev_high) > (prev_rsi_high - curr_rsi_high) else 'moderate', 'price_high': round(curr_high, 2), 'rsi_high': round(curr_rsi_high, 2)}
        return {'type': 'none', 'strength': 'weak'}
    
    def get_rsi_signal(self, rsi: float) -> Dict:
        """Get RSI signal based on value"""
        if rsi >= 80:
            return {'signal': 'overbought', 'action': 'consider_sell', 'strength': 'strong' if rsi >= 85 else 'moderate', 'description': '超买区域，考虑减仓'}
        elif rsi >= 70:
            return {'signal': 'approaching_overbought', 'action': 'watch', 'strength': 'moderate', 'description': '接近超买，注意回调风险'}
        elif rsi >= 30:
            return {'signal': 'neutral', 'action': 'hold', 'strength': 'neutral', 'description': '中性区域，持有观望'}
        elif rsi >= 20:
            return {'signal': 'approaching_oversold', 'action': 'watch', 'strength': 'moderate', 'description': '接近超卖，关注买入机会'}
        else:
            return {'signal': 'oversold', 'action': 'consider_buy', 'strength': 'strong' if rsi <= 15 else 'moderate', 'description': '超卖区域，考虑买入'}
    
    def calculate_rsi_score(self, rsi: float) -> float:
        """Convert RSI to a 0-100 score"""
        return round(rsi, 1)
    
    def get_multi_cycle_summary(self, rsi_data: Dict[str, float]) -> Dict:
        """Get summary of multi-cycle RSI"""
        rsi_14 = rsi_data.get('rsi_14', 50)
        rsi_25 = rsi_data.get('rsi_25', 50)
        rsi_60 = rsi_data.get('rsi_60', 50)
        if rsi_14 > 50 and rsi_25 > 50 and rsi_60 > 50:
            trend = 'strong_bullish'
            strength = 'strong'
        elif rsi_14 > 50 and rsi_25 > 50:
            trend = 'bullish'
            strength = 'moderate'
        elif rsi_14 > 50:
            trend = 'slightly_bullish'
            strength = 'weak'
        elif rsi_14 < 50 and rsi_25 < 50 and rsi_60 < 50:
            trend = 'strong_bearish'
            strength = 'strong'
        elif rsi_14 < 50 and rsi_25 < 50:
            trend = 'bearish'
            strength = 'moderate'
        else:
            trend = 'neutral'
            strength = 'weak'
        return {'rsi_14': rsi_14, 'rsi_25': rsi_25, 'rsi_60': rsi_60, 'trend': trend, 'strength': strength, 'consensus': 'bullish' if rsi_14 > 50 else 'bearish'}


# Global instance
rsi_multicycle = RSI_MultiCycle()
