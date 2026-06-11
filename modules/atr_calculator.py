"""
ATR (Average True Range) Calculator Module - Enhanced V3
Provides dynamic stop loss, support/resistance levels based on volatility
with Wilder's EMA smoothing, ADX trend strength, and volatility targeting
"""

import math
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta


class ATRCalculator:
    """Enhanced ATR Calculator with Wilder's EMA and ADX"""
    
    def __init__(self, atr_period: int = 14):
        self.atr_period = atr_period
        # Volatility-adaptive multiplier ranges (refined)
        self.volatility_multipliers = {
            'low': {'stop': 1.5, 'profit': 2.5},
            'medium': {'stop': 2.0, 'profit': 3.0},
            'high': {'stop': 2.5, 'profit': 3.5}
        }
        # Time-based stop loss adjustment
        self.time_decay_rates = {
            '1d': 0.02,
            '5d': 0.05,
            '10d': 0.08,
            '20d': 0.12,
            '60d': 0.20
        }
        # Volatility target for position sizing
        self.vol_target = 0.15
    
    def calculate_atr_ema(self, stock_data: Dict, prev_atr: float = None) -> float:
        """
        Calculate ATR using Wilder's EMA smoothing
        ATR = (PrevATR * (period - 1) + TrueRange) / period
        """
        high = stock_data.get('high', 0)
        low = stock_data.get('low', 0)
        close = stock_data.get('close', 0)
        prev_close = stock_data.get('prev_close', close)
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - prev_close) if prev_close > 0 else 0
        tr3 = abs(low - prev_close) if prev_close > 0 else 0
        true_range = max(tr1, tr2, tr3)
        
        # Wilder's EMA smoothing
        if prev_atr is not None:
            atr = (prev_atr * (self.atr_period - 1) + true_range) / self.atr_period
        else:
            atr = true_range
        
        return round(atr, 2)
    
    def assess_volatility_level(self, stock_data: Dict) -> str:
        """
        Assess current volatility level using multiple metrics
        """
        turnover = stock_data.get('turnover', 100)
        atr = self.calculate_atr_ema(stock_data)
        price = stock_data.get('price', 1)
        
        # Volatility metrics
        turnover_vol = turnover / 100
        atr_pct = (atr / price * 100) if price > 0 else 0
        
        # Bollinger Band Width proxy
        bb_width = (stock_data.get('high', 0) - stock_data.get('low', 0)) / price * 100 if price > 0 else 0
        
        # Combined volatility score (weighted)
        volatility_score = turnover_vol * 0.4 + atr_pct * 0.35 + bb_width * 0.25
        
        if volatility_score > 3.0:
            return 'high'
        elif volatility_score > 1.8:
            return 'medium'
        else:
            return 'low'
    
    def get_adaptive_multiplier(self, stock_data: Dict, stop_type: str = 'stop') -> float:
        """
        Get adaptive ATR multiplier based on current volatility
        Uses continuous function instead of discrete classification
        """
        vol_level = self.assess_volatility_level(stock_data)
        
        # Continuous adjustment based on volatility level
        base_multiplier = {
            'low': 1.5,
            'medium': 2.0,
            'high': 2.5
        }
        
        multiplier = base_multiplier[vol_level]
        
        # Fine-tune based on exact volatility score
        turnover = stock_data.get('turnover', 100)
        if turnover > 400:
            multiplier += 0.3 if stop_type == 'stop' else 0.5
        elif turnover > 300:
            multiplier += 0.15 if stop_type == 'stop' else 0.25
        
        return round(multiplier, 2)
    
    def calculate_atr_stop_loss(self, stock_data: Dict, atr_multiplier: Optional[float] = None) -> Dict:
        """
        Calculate dynamic stop loss based on ATR with volatility adaptation
        """
        price = stock_data.get('price', 0)
        atr = self.calculate_atr_ema(stock_data)
        
        if atr_multiplier is None:
            atr_multiplier = self.get_adaptive_multiplier(stock_data, 'stop')
        
        stop_loss = price - (atr * atr_multiplier)
        stop_loss_pct = ((price - stop_loss) / price * 100) if price > 0 else 0
        
        vol_level = self.assess_volatility_level(stock_data)
        
        return {
            'stop_loss_price': round(stop_loss, 2),
            'stop_loss_pct': round(stop_loss_pct, 2),
            'atr': atr,
            'atr_multiplier': atr_multiplier,
            'volatility_level': vol_level,
            'type': 'dynamic_ema'
        }
    
    def calculate_atr_stop_gain(self, stock_data: Dict, atr_multiplier: Optional[float] = None) -> Dict:
        """
        Calculate dynamic take profit based on ATR with volatility adaptation
        """
        price = stock_data.get('price', 0)
        atr = self.calculate_atr_ema(stock_data)
        
        if atr_multiplier is None:
            atr_multiplier = self.get_adaptive_multiplier(stock_data, 'profit')
        
        take_profit = price + (atr * atr_multiplier)
        take_profit_pct = ((take_profit - price) / price * 100) if price > 0 else 0
        
        vol_level = self.assess_volatility_level(stock_data)
        
        return {
            'take_profit_price': round(take_profit, 2),
            'take_profit_pct': round(take_profit_pct, 2),
            'atr': atr,
            'atr_multiplier': atr_multiplier,
            'volatility_level': vol_level,
            'type': 'dynamic_ema'
        }
    
    def calculate_trailing_stop(self, entry_price: float, current_price: float, 
                                trail_percent: float = 5.0,
                                time_in_position_days: int = 0) -> Dict:
        """
        Calculate trailing stop loss with time-based adjustments
        """
        time_factor = 1.0
        for threshold, rate in self.time_decay_rates.items():
            days = int(threshold.replace('d', ''))
            if time_in_position_days >= days:
                time_factor -= rate
        
        adjusted_trail = trail_percent * (1 + time_factor)
        
        if current_price > entry_price:
            trailing_stop = current_price * (1 - adjusted_trail / 100)
        else:
            trailing_stop = entry_price * (1 - adjusted_trail / 100)
        
        return {
            'trailing_stop': round(trailing_stop, 2),
            'trail_percent': trail_percent,
            'adjusted_trail_percent': round(adjusted_trail, 2),
            'time_in_position': time_in_position_days,
            'time_adjustment': round(time_factor, 3),
            'distance_from_current': round((current_price - trailing_stop) / current_price * 100, 2) if current_price > 0 else 0
        }
    
    def calculate_dynamic_support_resistance(self, stock_data: Dict, atr_multiplier: float = 2.0) -> Dict:
        """
        Calculate dynamic support and resistance levels based on ATR
        """
        high = stock_data.get('high', 0)
        low = stock_data.get('low', 0)
        price = stock_data.get('price', 0)
        atr = self.calculate_atr_ema(stock_data)
        
        support = low - atr * atr_multiplier
        resistance = high + atr * atr_multiplier
        
        return {
            'support': round(support, 2),
            'resistance': round(resistance, 2),
            'atr': atr,
            'distance_to_support': round((price - support) / price * 100, 2) if price > 0 else 0,
            'distance_to_resistance': round((resistance - price) / price * 100, 2) if price > 0 else 0
        }
    
    def calculate_vol_target_position(self, total_capital: float,
                                       current_volatility: float = 0.20) -> float:
        """
        Calculate position size based on volatility targeting
        Position = Target Volatility / Current Volatility
        """
        vol_ratio = self.vol_target / current_volatility if current_volatility > 0 else 1.0
        vol_ratio = max(0.3, min(1.5, vol_ratio))
        
        return total_capital * vol_ratio


class ADXCalculator:
    """ADX (Average Directional Index) Calculator for trend strength"""
    
    def __init__(self, adx_period: int = 14):
        self.adx_period = adx_period
    
    def calculate_dm(self, high: float, low: float, 
                     prev_high: float, prev_low: float) -> Tuple[float, float]:
        """
        Calculate +DM and -DM (Directional Movement)
        
        +DM = Current High - Previous High (if positive and > -DM)
        -DM = Previous Low - Current Low (if positive and > +DM)
        """
        plus_dm = 0
        minus_dm = 0
        
        if high > prev_high:
            plus_dm = high - prev_high
        if low < prev_low:
            minus_dm = prev_low - low
        
        # Filter: only take the larger one
        if plus_dm > minus_dm and plus_dm > 0:
            minus_dm = 0
        elif minus_dm > plus_dm and minus_dm > 0:
            plus_dm = 0
        else:
            # If both are equal or both zero
            if plus_dm == 0 and minus_dm == 0:
                pass
            elif plus_dm > 0:
                minus_dm = 0
            elif minus_dm > 0:
                plus_dm = 0
        
        return plus_dm, minus_dm
    
    def calculate_di(self, plus_dm: float, minus_dm: float) -> Tuple[float, float]:
        """
        Calculate +DI and -DI (Directional Indicators)
        
        +DI = (+DM / TR) * 100
        -DI = (-DM / TR) * 100
        """
        tr = plus_dm + minus_dm
        if tr > 0:
            plus_di = (plus_dm / tr) * 100
            minus_di = (minus_dm / tr) * 100
        else:
            plus_di = 50  # Neutral
            minus_di = 50
        
        return plus_di, minus_di
    
    def calculate_dx(self, plus_di: float, minus_di: float) -> float:
        """
        Calculate DX (Directional Index)
        
        DX = |(+DI - -DI)| / (+DI + -DI) * 100
        """
        di_sum = plus_di + minus_di
        if di_sum > 0:
            dx = abs(plus_di - minus_di) / di_sum * 100
        else:
            dx = 0
        
        return dx
    
    def calculate_adx(self, dx: float, prev_adx: float = None) -> float:
        """
        Calculate ADX using Wilder's smoothing
        
        ADX = (PrevADX * (period - 1) + DX) / period
        """
        if prev_adx is not None:
            adx = (prev_adx * (self.adx_period - 1) + dx) / self.adx_period
        else:
            adx = dx
        
        return adx
    
    def calculate_adx_from_data(self, stock_data: Dict, 
                                 prev_plus_di: float = 25,
                                 prev_minus_di: float = 25,
                                 prev_adx: float = 25) -> Dict:
        """
        Calculate ADX from stock data
        
        Args:
            stock_data: Stock data dictionary
            prev_plus_di: Previous +DI value
            prev_minus_di: Previous -DI value
            prev_adx: Previous ADX value
        
        Returns:
            Dictionary with ADX values and trend information
        """
        high = stock_data.get('high', 0)
        low = stock_data.get('low', 0)
        prev_high = stock_data.get('prev_high', high)
        prev_low = stock_data.get('prev_low', low)
        
        # Calculate directional movement
        plus_dm, minus_dm = self.calculate_dm(high, low, prev_high, prev_low)
        
        # Calculate directional indicators
        plus_di, minus_di = self.calculate_di(plus_dm, minus_dm)
        
        # Calculate DX
        dx = self.calculate_dx(plus_di, minus_di)
        
        # Calculate ADX
        adx = self.calculate_adx(dx, prev_adx)
        
        # Determine trend strength
        if adx > 50:
            trend_strength = 'very_strong'
        elif adx > 25:
            trend_strength = 'strong'
        elif adx > 20:
            trend_strength = 'moderate'
        elif adx > 15:
            trend_strength = 'weak'
        else:
            trend_strength = 'very_weak'
        
        # Determine trend direction
        if plus_di > minus_di + 5:
            trend_direction = 'bullish'
        elif minus_di > plus_di + 5:
            trend_direction = 'bearish'
        else:
            trend_direction = 'neutral'
        
        return {
            'adx': round(adx, 2),
            'plus_di': round(plus_di, 2),
            'minus_di': round(minus_di, 2),
            'dx': round(dx, 2),
            'trend_strength': trend_strength,
            'trend_direction': trend_direction,
            'prev_plus_di': prev_plus_di,
            'prev_minus_di': prev_minus_di,
            'prev_adx': prev_adx
        }
    
    def get_adx_adjustment(self, adx_data: Dict) -> Dict:
        """
        Get market adjustment factors based on ADX
        
        Args:
            adx_data: ADX calculation results
        
        Returns:
            Dictionary with adjustment factors
        """
        adx = adx_data['adx']
        trend_direction = adx_data['trend_direction']
        trend_strength = adx_data['trend_strength']
        
        # Strength-based adjustment
        strength_factors = {
            'very_strong': 1.2,
            'strong': 1.1,
            'moderate': 1.0,
            'weak': 0.9,
            'very_weak': 0.8
        }
        
        # Direction-based adjustment
        direction_factors = {
            'bullish': 1.1,
            'bearish': 0.9,
            'neutral': 1.0
        }
        
        # Combine adjustments
        trend_factor = strength_factors[trend_strength] * direction_factors[trend_direction]
        
        return {
            'trend_factor': round(trend_factor, 3),
            'strength': trend_strength,
            'direction': trend_direction,
            'adx_level': 'strong' if adx > 25 else 'weak'
        }
