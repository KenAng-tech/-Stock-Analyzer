"""
Dynamic RSI Module - V1
Enhanced RSI with dynamic thresholds based on volatility.

Improvement over standard RSI:
1. Volatility-adjusted thresholds (not fixed 30/70)
2. RSI momentum (rate of change)
3. Trend-adjusted RSI (different thresholds in uptrend vs downtrend)
"""

import math
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta


class DynamicRSI:
    """
    Dynamic RSI with volatility-adjusted thresholds
    
    Key improvements over standard RSI:
    1. Dynamic thresholds based on volatility
    2. RSI momentum (rate of change)
    3. Trend-adjusted thresholds
    4. Multiple periods with consensus
    """
    
    def __init__(self, default_periods: List[int] = None):
        self.default_periods = default_periods or [6, 14, 25, 60]
        self.base_oversold = 30
        self.base_overbought = 70
    
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
            return 100.0
        
        rs = avg_gain / avg_loss
        return round(100 - (100 / (1 + rs)), 2)
    
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
    
    def calculate_rsi_momentum(self, rsi_series: List[float], 
                                lookback: int = 5) -> float:
        """
        Calculate RSI momentum (rate of change)
        
        Positive momentum = RSI rising
        Negative momentum = RSI falling
        """
        if len(rsi_series) < lookback + 1:
            return 0.0
        
        current_rsi = rsi_series[-1]
        past_rsi = rsi_series[-lookback - 1]
        
        return round(current_rsi - past_rsi, 2)
    
    def calculate_volatility(self, prices: List[float]) -> float:
        """Calculate rolling volatility from prices"""
        if len(prices) < 2:
            return 0.15
        returns = np.diff(np.log(prices))
        return np.std(returns) * math.sqrt(252)
    
    def get_dynamic_thresholds(self, volatility: float,
                                trend: str = 'neutral') -> Dict:
        """
        Get dynamic RSI thresholds based on volatility and trend
        
        Higher volatility -> wider thresholds (30-70 becomes 25-75)
        Lower volatility -> narrower thresholds (30-70 becomes 35-65)
        Uptrend -> oversold threshold lower (more room to dip)
        Downtrend -> overbought threshold higher (more room to rise)
        """
        # Volatility adjustment
        if volatility > 0.25:
            vol_adj = -5  # Wider thresholds
        elif volatility < 0.10:
            vol_adj = 5   # Narrower thresholds
        else:
            vol_adj = 0
        
        # Trend adjustment
        trend_adj = 0
        if trend == 'uptrend':
            trend_adj = 5  # Oversold threshold lower
        elif trend == 'downtrend':
            trend_adj = -5  # Overbought threshold lower
        
        oversold = self.base_oversold + vol_adj + trend_adj
        overbought = self.base_overbought + vol_adj + trend_adj
        
        return {
            'oversold': round(oversold, 1),
            'overbought': round(overbought, 1),
            'volatility': round(volatility, 4),
            'trend': trend
        }
    
    def get_rsi_signal(self, rsi: float, 
                        oversold: float = 30,
                        overbought: float = 70,
                        momentum: float = 0) -> Dict:
        """Get RSI signal with dynamic thresholds"""
        if rsi >= overbought:
            signal = 'OVERBOUGHT'
            action = '考虑减仓'
            strength = 'strong' if rsi >= overbought + 5 else 'moderate'
        elif rsi >= overbought - 5:
            signal = 'APPROACHING_OVERBOUGHT'
            action = '注意回调风险'
            strength = 'moderate'
        elif rsi <= oversold:
            signal = 'OVERSOLD'
            action = '考虑买入'
            strength = 'strong' if rsi <= oversold - 5 else 'moderate'
        elif rsi <= oversold + 5:
            signal = 'APPROACHING_OVERSOLD'
            action = '关注买入机会'
            strength = 'moderate'
        else:
            signal = 'NEUTRAL'
            action = '持有观望'
            strength = 'neutral'
        
        # Momentum adjustment
        if momentum > 3:
            action += ' (动量向上)'
        elif momentum < -3:
            action += ' (动量向下)'
        
        return {
            'signal': signal,
            'action': action,
            'strength': strength,
            'rsi': rsi,
            'momentum': momentum,
            'oversold': oversold,
            'overbought': overbought
        }
    
    def get_multi_cycle_rsi(self, prices: List[float]) -> Dict:
        """
        Get RSI for multiple periods with consensus
        """
        rsi_values = {}
        for period in self.default_periods:
            rsi_values[period] = self.calculate_rsi(prices, period)
        
        # Consensus
        avg_rsi = np.mean(list(rsi_values.values()))
        rsi_std = np.std(list(rsi_values.values()))
        
        # Determine trend from RSI values
        if avg_rsi > 55:
            trend = 'uptrend'
        elif avg_rsi < 45:
            trend = 'downtrend'
        else:
            trend = 'neutral'
        
        return {
            'rsi_values': rsi_values,
            'average_rsi': round(avg_rsi, 2),
            'rsi_std': round(rsi_std, 2),
            'trend': trend,
            'consensus': 'bullish' if avg_rsi > 50 else 'bearish',
            'divergence': 'high' if rsi_std > 10 else 'low' if rsi_std > 5 else 'none'
        }
    
    def analyze(self, stock_data: Dict, klines: List[Dict] = None) -> Dict:
        """
        Comprehensive RSI analysis
        
        Returns:
            Dict with RSI values, thresholds, signals, and momentum
        """
        price = stock_data.get('price', 0)
        
        if klines:
            closes = [k['close'] for k in klines if k['close'] > 0]
        else:
            closes = [price]
        
        # Calculate RSI series
        rsi_14_series = self.calculate_rsi_series(closes, 14)
        
        # Calculate volatility
        volatility = self.calculate_volatility(closes)
        
        # Get trend from price position
        year_high = stock_data.get('year_high', 0)
        year_low = stock_data.get('year_low', 0)
        if year_high > 0 and year_low > 0:
            price_position = (price - year_low) / (year_high - year_low)
        else:
            price_position = 0.5
        
        if price_position > 0.6:
            trend = 'uptrend'
        elif price_position < 0.4:
            trend = 'downtrend'
        else:
            trend = 'neutral'
        
        # Dynamic thresholds
        thresholds = self.get_dynamic_thresholds(volatility, trend)
        
        # RSI signals
        rsi_14 = rsi_14_series[-1] if rsi_14_series else 50
        rsi_momentum = self.calculate_rsi_momentum(rsi_14_series)
        
        signal = self.get_rsi_signal(rsi_14, thresholds['oversold'], 
                                      thresholds['overbought'], rsi_momentum)
        
        # Multi-cycle RSI
        multi_cycle = self.get_multi_cycle_rsi(closes)
        
        return {
            'rsi_14': round(rsi_14, 2),
            'rsi_momentum': rsi_momentum,
            'signal': signal,
            'thresholds': thresholds,
            'volatility': round(volatility, 4),
            'trend': trend,
            'multi_cycle': multi_cycle,
            'rsi_series': rsi_14_series[-20:],  # Last 20 values
            'recommendation': self._get_recommendation(signal, rsi_momentum)
        }
    
    def _get_recommendation(self, signal: Dict, momentum: float) -> str:
        """Get trading recommendation"""
        if signal['signal'] == 'OVERSOLD' and momentum > 0:
            return '超卖反弹，积极买入'
        elif signal['signal'] == 'OVERSOLD':
            return '超卖区域，关注买入'
        elif signal['signal'] == 'OVERBOUGHT' and momentum < 0:
            return '超买回落，积极减仓'
        elif signal['signal'] == 'OVERBOUGHT':
            return '超买区域，注意回调'
        elif momentum > 3:
            return '动量向上，持有或加仓'
        elif momentum < -3:
            return '动量向下，减仓观望'
        else:
            return '中性区域，持有观望'


# Global instance
dynamic_rsi = DynamicRSI()
