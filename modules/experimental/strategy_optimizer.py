"""
Strategy Optimizer - V2
Comprehensive strategy optimization combining all modules.
"""

import math
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

from modules.kelly_optimizer import KellyOptimizer
from modules.atr_trend_filter import ATR_TrendFilter
from modules.rsi_multicycle import RSI_MultiCycle
from modules.volatility_target import VolatilityTarget
from modules.macd_bollinger import MACD_Bollinger
from modules.sector_rotation import SectorRotation
from modules.backtester import Backtester
from modules.enhanced_backtester import EnhancedBacktester


class StrategyOptimizer:
    """Comprehensive strategy optimizer"""
    
    def __init__(self):
        self.kelly = KellyOptimizer()
        self.atr_filter = ATR_TrendFilter()
        self.rsi = RSI_MultiCycle()
        self.vol_target = VolatilityTarget()
        self.macd_bb = MACD_Bollinger()
        self.sector = SectorRotation()
        
        # Strategy parameters
        self.params = {
            'kelly_fraction': 0.5,
            'atr_multiplier': 2.0,
            'rsi_period': 14,
            'vol_target': 0.15,
            'stop_loss_pct': 0.05,
            'take_profit_pct': 0.10,
            'max_position': 0.25,
        }
    
    def generate_comprehensive_signal(self, stock_data: Dict,
                                       klines: Optional[List[Dict]] = None,
                                       industry: str = '光通信') -> Dict:
        """
        Generate comprehensive trading signal using all modules
        
        Returns:
            Dict with signal, strength, and detailed analysis
        """
        price = stock_data.get('price', 0)
        
        # 1. Kelly position sizing
        kelly_result = self.kelly.get_position_size(stock_data)
        
        # 2. ATR trend-aware stop loss
        atr_result = self.atr_filter.calculate_trend_aware_stop(stock_data, klines or [])
        
        # 3. RSI multi-cycle analysis
        rsi_data = self.rsi.calculate_all_periods(
            [k['close'] for k in klines[-60:]] if klines else []
        )
        rsi_summary = self.rsi.get_multi_cycle_summary(rsi_data)
        rsi_signal = self.rsi.get_rsi_signal(rsi_data.get('rsi_14', 50))
        
        # 4. Volatility target position
        vol_result = self.vol_target.calculate_target_position(stock_data)
        
        # 5. MACD + Bollinger Bands
        if klines:
            prices = [k['close'] for k in klines]
            macd_data = self.macd_bb.calculate_macd(prices)
            bb_data = self.macd_bb.calculate_bollinger_bands(prices)
            combined_signal = self.macd_bb.get_combined_signal(macd_data, bb_data)
        else:
            macd_data = {'macd': 0, 'signal': 0, 'histogram': 0, 'trend': 'neutral'}
            bb_data = {'upper': price * 1.1, 'middle': price, 'lower': price * 0.9, 'percent_b': 0.5}
            combined_signal = {'combined_signal': 'HOLD', 'combined_strength': 'weak'}
        
        # 6. Sector rotation
        sector_result = self.sector.get_sector_recommendation(stock_data, industry)
        
        # 7. Calculate composite score
        composite_score = self._calculate_composite_score(
            rsi_data, atr_result, kelly_result, vol_result, combined_signal, sector_result
        )
        
        # 8. Final recommendation
        recommendation = self._generate_recommendation(
            composite_score, rsi_signal, atr_result, kelly_result, combined_signal
        )
        
        return {
            'recommendation': recommendation['recommendation'],
            'action': recommendation['action'],
            'confidence': recommendation['confidence'],
            'composite_score': round(composite_score, 2),
            'kelly': kelly_result,
            'atr_stop': atr_result,
            'rsi': {
                'rsi_14': rsi_data.get('rsi_14', 0),
                'rsi_25': rsi_data.get('rsi_25', 0),
                'rsi_60': rsi_data.get('rsi_60', 0),
                'summary': rsi_summary,
                'signal': rsi_signal
            },
            'volatility': vol_result,
            'macd_bb': combined_signal,
            'sector': sector_result,
            'timestamp': datetime.now().isoformat()
        }
    
    def _calculate_composite_score(self, rsi_data: Dict, atr_result: Dict,
                                    kelly_result: Dict, vol_result: Dict,
                                    combined_signal: Dict, sector_result: Dict) -> float:
        """Calculate composite score (0-100)"""
        # RSI score (0-100)
        rsi_score = rsi_data.get('rsi_14', 50)
        
        # ATR score (based on trend strength)
        atr_trend = atr_result.get('trend', 'neutral')
        atr_trend_score = {'bullish': 70, 'neutral': 50, 'bearish': 30}.get(atr_trend, 50)
        
        # Kelly score (based on position size)
        kelly_score = kelly_result.get('kelly_fraction', 0.15) * 100 / self.params['max_position']
        
        # Volatility score (lower vol = higher score)
        vol_score = max(0, 100 - vol_result.get('realized_vol', 0.15) * 200)
        
        # MACD+BB score
        macd_bb_score = {'BUY': 75, 'SELL': 35, 'HOLD': 50}.get(combined_signal.get('combined_signal', 'HOLD'), 50)
        
        # Sector score
        sector_score = {'强烈推荐': 90, '推荐': 70, '中性': 50, '观望': 30}.get(sector_result.get('rating', '中性'), 50)
        
        # Weighted composite
        weights = {
            'rsi': 0.20,
            'atr': 0.20,
            'kelly': 0.15,
            'vol': 0.10,
            'macd_bb': 0.20,
            'sector': 0.15
        }
        
        composite = (
            rsi_score * weights['rsi'] +
            atr_trend_score * weights['atr'] +
            kelly_score * weights['kelly'] +
            vol_score * weights['vol'] +
            macd_bb_score * weights['macd_bb'] +
            sector_score * weights['sector']
        )
        
        return composite
    
    def _generate_recommendation(self, composite_score: float,
                                  rsi_signal: Dict, atr_result: Dict,
                                  kelly_result: Dict, combined_signal: Dict) -> Dict:
        """Generate final recommendation"""
        if composite_score >= 75:
            recommendation = '强烈买入'
            action = '建议加仓至Kelly目标仓位'
            confidence = 'high'
        elif composite_score >= 60:
            recommendation = '买入'
            action = '建议逐步建仓'
            confidence = 'medium'
        elif composite_score >= 45:
            recommendation = '持有'
            action = '保持现有仓位'
            confidence = 'medium'
        elif composite_score >= 30:
            recommendation = '减仓'
            action = '建议减仓至Kelly目标仓位'
            confidence = 'medium'
        else:
            recommendation = '卖出'
            action = '建议清仓或大幅减仓'
            confidence = 'high'
        
        # Add stop loss and take profit
        stop_loss = atr_result.get('stop_loss_price', 0)
        take_profit = atr_result.get('take_profit_price', 0)
        
        return {
            'recommendation': recommendation,
            'action': action,
            'confidence': confidence,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_reward_ratio': round((take_profit - atr_result.get('stop_loss_price', 0)) / 
                                       (atr_result.get('stop_loss_price', 0) - atr_result.get('stop_loss_price', 0) + 1), 2)
        }
    
    def run_backtest(self, klines: List[Dict], strategy_func=None) -> Dict:
        """Run backtest using the strategy"""
        if strategy_func is None:
            strategy_func = self._default_strategy
        
        bt = EnhancedBacktester()
        result = bt.run_walk_forward(klines, strategy_func)
        return result
    
    def _default_strategy(self, bar: Dict, position: int, capital: float,
                          params: Optional[Dict] = None) -> str:
        """Default strategy function for backtesting"""
        if params is None:
            params = self.params
        
        price = bar['close']
        rsi = bar.get('rsi', 50)
        macd = bar.get('macd', 0)
        
        if position == 0 and rsi < 35 and macd > 0:
            return 'buy'
        elif position > 0 and rsi > 65 and macd < 0:
            return 'sell'
        return 'hold'
    
    def optimize_params(self, klines: List[Dict]) -> Dict:
        """Optimize strategy parameters"""
        param_grid = {
            'kelly_fraction': [0.3, 0.5, 0.7],
            'atr_multiplier': [1.5, 2.0, 2.5],
            'rsi_period': [10, 14, 20],
            'vol_target': [0.10, 0.15, 0.20],
        }
        
        best_params = {}
        best_score = -999
        
        for kelly_f in param_grid['kelly_fraction']:
            for atr_m in param_grid['atr_multiplier']:
                for rsi_p in param_grid['rsi_period']:
                    for vol_t in param_grid['vol_target']:
                        # Calculate score for this parameter combination
                        score = self._parameter_score(kelly_f, atr_m, rsi_p, vol_t)
                        if score > best_score:
                            best_score = score
                            best_params = {
                                'kelly_fraction': kelly_f,
                                'atr_multiplier': atr_m,
                                'rsi_period': rsi_p,
                                'vol_target': vol_t,
                            }
        
        return {
            'best_params': best_params,
            'best_score': round(best_score, 2),
            'all_combinations': len(param_grid['kelly_fraction']) * 
                               len(param_grid['atr_multiplier']) *
                               len(param_grid['rsi_period']) *
                               len(param_grid['vol_target'])
        }
    
    def _parameter_score(self, kelly_f: float, atr_m: float, 
                         rsi_p: int, vol_t: float) -> float:
        """Calculate score for a parameter combination"""
        # Simple scoring based on parameter reasonableness
        kelly_score = 10 - abs(kelly_f - 0.5) * 10
        atr_score = 10 - abs(atr_m - 2.0) * 2
        rsi_score = 10 - abs(rsi_p - 14) * 0.5
        vol_score = 10 - abs(vol_t - 0.15) * 10
        
        return kelly_score + atr_score + rsi_score + vol_score


# Global instance
strategy_optimizer = StrategyOptimizer()
