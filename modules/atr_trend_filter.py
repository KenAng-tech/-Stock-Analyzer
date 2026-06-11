"""
ATR Trend Filter Module - V2
Enhanced ATR stop loss with trend direction filtering.
Uses ADX for trend strength and EMA for trend direction.
"""

import math
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta


class ATR_TrendFilter:
    """Enhanced ATR with trend direction filtering"""
    
    def __init__(self, atr_period: int = 14, adx_period: int = 14):
        self.atr_period = atr_period
        self.adx_period = adx_period
        self.trend_threshold = 25  # ADX > 25 indicates strong trend
        self.ema_periods = [5, 10, 20, 60, 120, 250]
    
    def calculate_ema(self, prices: List[float], period: int) -> float:
        """Calculate Exponential Moving Average"""
        if not prices:
            return 0
        multiplier = 2 / (period + 1)
        ema = prices[0]
        for price in prices[1:]:
            ema = (price - ema) * multiplier + ema
        return ema
    
    def calculate_ema_series(self, prices: List[float], period: int) -> List[float]:
        """Calculate EMA series"""
        if not prices:
            return []
        multiplier = 2 / (period + 1)
        ema_series = [prices[0]]
        for price in prices[1:]:
            ema = (price - ema_series[-1]) * multiplier + ema_series[-1]
            ema_series.append(ema)
        return ema_series
    
    def calculate_atr(self, klines: List[Dict]) -> float:
        """Calculate ATR from K-line data"""
        if len(klines) < self.atr_period + 1:
            return 0
        
        # Calculate True Range for each period
        tr_values = []
        for i in range(1, len(klines)):
            high = klines[i]['high']
            low = klines[i]['low']
            prev_close = klines[i-1]['close']
            
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            tr_values.append(tr)
        
        # Wilder's smoothing
        atr = np.mean(tr_values[:self.atr_period])
        for tr in tr_values[self.atr_period:]:
            atr = (atr * (self.atr_period - 1) + tr) / self.atr_period
        
        return atr
    
    def calculate_adx(self, klines: List[Dict]) -> Dict:
        """Calculate ADX for trend strength"""
        if len(klines) < self.adx_period + 1:
            return {'adx': 25, 'trend': 'neutral', 'strength': 'weak'}
        
        # Calculate +DM and -DM
        plus_dm = []
        minus_dm = []
        for i in range(1, len(klines)):
            high_diff = klines[i]['high'] - klines[i-1]['high']
            low_diff = klines[i-1]['low'] - klines[i]['low']
            
            if high_diff > low_diff and high_diff > 0:
                plus_dm.append(high_diff)
                minus_dm.append(0)
            elif low_diff > high_diff and low_diff > 0:
                plus_dm.append(0)
                minus_dm.append(low_diff)
            else:
                plus_dm.append(0)
                minus_dm.append(0)
        
        # Smooth DM values
        plus_dm_smooth = self._smooth_series(plus_dm, self.adx_period)
        minus_dm_smooth = self._smooth_series(minus_dm, self.adx_period)
        
        # Calculate TR
        tr_values = []
        for i in range(1, len(klines)):
            high = klines[i]['high']
            low = klines[i]['low']
            prev_close = klines[i-1]['close']
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            tr_values.append(tr)
        
        tr_smooth = self._smooth_series(tr_values, self.adx_period)
        
        # Calculate +DI and -DI
        plus_di = [(p / t * 100) if t > 0 else 50 
                   for p, t in zip(plus_dm_smooth, tr_smooth)]
        minus_di = [(m / t * 100) if t > 0 else 50 
                    for m, t in zip(minus_dm_smooth, tr_smooth)]
        
        # Calculate DX and ADX
        dx_values = []
        for p_di, m_di in zip(plus_di, minus_di):
            di_sum = p_di + m_di
            dx = abs(p_di - m_di) / di_sum * 100 if di_sum > 0 else 0
            dx_values.append(dx)
        
        adx = np.mean(dx_values[-self.adx_period:]) if dx_values else 25
        
        # Determine trend
        latest_plus_di = plus_di[-1] if plus_di else 50
        latest_minus_di = minus_di[-1] if minus_di else 50
        
        if adx > 50:
            strength = 'very_strong'
        elif adx > 25:
            strength = 'strong'
        elif adx > 20:
            strength = 'moderate'
        else:
            strength = 'weak'
        
        if latest_plus_di > latest_minus_di + 5:
            direction = 'bullish'
        elif latest_minus_di > latest_plus_di + 5:
            direction = 'bearish'
        else:
            direction = 'neutral'
        
        return {
            'adx': round(adx, 2),
            'plus_di': round(latest_plus_di, 2),
            'minus_di': round(latest_minus_di, 2),
            'trend': direction,
            'strength': strength
        }
    
    def _smooth_series(self, values: List[float], period: int) -> List[float]:
        """Smooth a series using Wilder's method"""
        if not values:
            return []
        
        smoothed = [np.mean(values[:period])]
        for i in range(period, len(values)):
            new_smooth = (smoothed[-1] * (period - 1) + values[i]) / period
            smoothed.append(new_smooth)
        
        # Pad beginning
        while len(smoothed) < len(values):
            smoothed.insert(0, smoothed[0])
        
        return smoothed
    
    def calculate_trend_aware_stop(self, stock_data: Dict, 
                                    klines: List[Dict],
                                    atr_multiplier: float = 2.0) -> Dict:
        """
        Calculate stop loss with trend awareness
        
        - Bullish trend: stop loss is tighter (attracts more)
        - Bearish trend: stop loss is wider (more room)
        - Neutral: standard ATR stop
        """
        price = stock_data.get('price', 0)
        atr = self.calculate_atr(klines) if klines else 0
        
        if atr == 0:
            atr = price * 0.02  # Default 2% ATR
        
        adx_info = self.calculate_adx(klines) if klines else {'adx': 25, 'trend': 'neutral'}
        
        # Trend-aware ATR multiplier
        if adx_info['trend'] == 'bullish' and adx_info['strength'] in ['strong', 'very_strong']:
            atr_mult = atr_multiplier * 0.85  # Tighter stop in strong uptrend
        elif adx_info['trend'] == 'bearish' and adx_info['strength'] in ['strong', 'very_strong']:
            atr_mult = atr_multiplier * 1.15  # Wider stop in strong downtrend
        else:
            atr_mult = atr_multiplier
        
        # Calculate stop loss
        stop_loss = price - (atr * atr_mult)
        
        # Calculate trailing stop (for bullish trends)
        if adx_info['trend'] == 'bullish':
            trailing_stop = price - (atr * atr_mult * 0.5)
        else:
            trailing_stop = stop_loss
        
        # Calculate take profit
        take_profit = price + (atr * atr_multiplier * 1.5)
        
        return {
            'stop_loss_price': round(stop_loss, 2),
            'stop_loss_pct': round((price - stop_loss) / price * 100, 2),
            'trailing_stop': round(trailing_stop, 2),
            'take_profit_price': round(take_profit, 2),
            'take_profit_pct': round((take_profit - price) / price * 100, 2),
            'atr': round(atr, 2),
            'atr_multiplier': round(atr_mult, 2),
            'trend': adx_info['trend'],
            'trend_strength': adx_info['strength'],
            'adx': adx_info['adx'],
            'type': 'trend_aware'
        }
    
    def calculate_dynamic_trailing_stop(self, price: float, 
                                         atr: float,
                                         highest_since_entry: float,
                                         atr_mult: float = 2.0) -> float:
        """
        Calculate dynamic trailing stop based on highest price since entry
        """
        if highest_since_entry == 0:
            return price - (atr * atr_mult)
        
        # Trailing stop from highest price
        trailing = highest_since_entry - (atr * atr_mult)
        
        # Never move stop loss below entry price
        return max(trailing, price * 0.95)


# Global instance
atr_trend_filter = ATR_TrendFilter()
