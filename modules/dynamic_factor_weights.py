"""
Dynamic Factor Weights Module - Long-term Optimization
Implements adaptive factor weighting based on market conditions
"""

import math
from typing import Dict, List, Optional
from datetime import datetime, timedelta


class DynamicFactorWeights:
    """Dynamic factor weight adjustment based on market conditions"""
    
    def __init__(self):
        # Base factor weights
        self.base_weights = {
            'momentum': 0.25,
            'mean_reversion': 0.20,
            'trend_following': 0.20,
            'volatility': 0.15,
            'volume': 0.10,
            'sentiment': 0.10
        }
        
        # Factor performance tracking
        self.factor_performance = {
            'momentum': {'win_rate': 0.55, 'sharpe': 0.8, 'avg_return': 0.02},
            'mean_reversion': {'win_rate': 0.50, 'sharpe': 0.6, 'avg_return': 0.015},
            'trend_following': {'win_rate': 0.45, 'sharpe': 0.7, 'avg_return': 0.025},
            'volatility': {'win_rate': 0.60, 'sharpe': 0.5, 'avg_return': 0.01},
            'volume': {'win_rate': 0.52, 'sharpe': 0.65, 'avg_return': 0.018},
            'sentiment': {'win_rate': 0.48, 'sharpe': 0.55, 'avg_return': 0.012}
        }
        
        # Market regime performance adjustments
        self.regime_adjustments = {
            'uptrend': {
                'momentum': 1.2,
                'trend_following': 1.3,
                'mean_reversion': 0.8,
                'volatility': 0.9
            },
            'downtrend': {
                'momentum': 0.8,
                'trend_following': 0.7,
                'mean_reversion': 1.3,
                'volatility': 1.2
            },
            'sideways': {
                'momentum': 1.0,
                'trend_following': 0.9,
                'mean_reversion': 1.2,
                'volatility': 1.1
            }
        }
    
    def calculate_factor_score(self, factor_name: str, market_state: Dict) -> float:
        """Calculate composite score for a factor"""
        perf = self.factor_performance[factor_name]
        
        # Score components
        win_rate_score = perf['win_rate'] * 40  # Max 40 points
        sharpe_score = perf['sharpe'] * 20      # Max 20 points (normalized)
        return_score = min(perf['avg_return'] * 1000, 40)  # Max 40 points
        
        # Market regime adjustment
        trend = market_state.get('trend', 'sideways')
        regime_adj = self.regime_adjustments.get(trend, {}).get(factor_name, 1.0)
        
        # Calculate final score
        base_score = win_rate_score + sharpe_score + return_score
        adjusted_score = base_score * regime_adj
        
        return round(adjusted_score, 2)
    
    def get_dynamic_weights(self, market_state: Dict) -> Dict:
        """Get dynamically adjusted factor weights"""
        # Calculate scores for all factors
        factor_scores = {}
        for factor_name in self.base_weights:
            factor_scores[factor_name] = self.calculate_factor_score(factor_name, market_state)
        
        # Normalize scores to weights
        total_score = sum(factor_scores.values())
        dynamic_weights = {
            factor: round(score / total_score, 3)
            for factor, score in factor_scores.items()
        }
        
        return {
            'dynamic_weights': dynamic_weights,
            'base_weights': self.base_weights,
            'factor_scores': factor_scores,
            'market_regime': market_state.get('trend', 'sideways')
        }
    
    def calculate_weight_entropy(self, weights: Dict) -> float:
        """Calculate entropy of weight distribution (measure of concentration)"""
        import math
        entropy = 0
        for weight in weights.values():
            if weight > 0:
                entropy -= weight * math.log(weight)
        return round(entropy, 3)
    
    def get_weight_confidence(self, weights: Dict, market_state: Dict) -> float:
        """Calculate confidence in current weight configuration"""
        entropy = self.calculate_weight_entropy(weights)
        max_entropy = math.log(len(weights))
        
        # Lower entropy = higher concentration = higher confidence
        concentration = 1 - (entropy / max_entropy)
        
        # Adjust for market volatility
        volatility = market_state.get('volatility', 'medium')
        volatility_factor = {'low': 1.1, 'medium': 1.0, 'high': 0.9}.get(volatility, 1.0)
        
        return round(concentration * volatility_factor, 3)
