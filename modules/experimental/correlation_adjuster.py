"""
Correlation Adjuster Module - Long-term Optimization
Handles market correlation adjustments for portfolio diversification
"""

import math
from typing import Dict, List, Optional
from datetime import datetime, timedelta


class CorrelationAdjuster:
    """Market correlation adjustment for portfolio optimization"""
    
    def __init__(self):
        # Sector correlation matrix (simplified)
        self.sector_correlations = {
            '科技': {'科技': 1.0, '金融': 0.4, '消费': 0.5, '医药': 0.3, '能源': 0.35},
            '金融': {'科技': 0.4, '金融': 1.0, '消费': 0.45, '医药': 0.35, '能源': 0.5},
            '消费': {'科技': 0.5, '金融': 0.45, '消费': 1.0, '医药': 0.4, '能源': 0.4},
            '医药': {'科技': 0.3, '金融': 0.35, '消费': 0.4, '医药': 1.0, '能源': 0.3},
            '能源': {'科技': 0.35, '金融': 0.5, '消费': 0.4, '医药': 0.3, '能源': 1.0},
        }
        
        # Market regime correlation shifts
        self.regime_correlation_shifts = {
            'bull_market': {'shift': -0.1, 'description': '牛市相关性下降'},
            'bear_market': {'shift': 0.15, 'description': '熊市相关性上升'},
            'volatile_market': {'shift': 0.2, 'description': '高波动相关性上升'},
            'stable_market': {'shift': -0.05, 'description': '稳定市场相关性略降'},
        }
    
    def calculate_sector_correlation(self, stock1_sector: str, stock2_sector: str) -> float:
        """Calculate correlation between two sectors"""
        if stock1_sector in self.sector_correlations and stock2_sector in self.sector_correlations[stock1_sector]:
            return self.sector_correlations[stock1_sector][stock2_sector]
        return 0.5  # Default correlation
    
    def get_regime_correlation_shift(self, market_state: Dict) -> float:
        """Get correlation shift based on market regime"""
        volatility = market_state.get('volatility', 'medium')
        trend = market_state.get('trend', 'sideways')
        
        if volatility == 'high' and trend == 'downtrend':
            regime = 'bear_market'
        elif volatility == 'low' and trend == 'uptrend':
            regime = 'bull_market'
        elif volatility == 'high':
            regime = 'volatile_market'
        else:
            regime = 'stable_market'
        
        return self.regime_correlation_shifts[regime]['shift']
    
    def adjust_correlation_for_portfolio(self, portfolio_stocks: List[Dict], 
                                          market_state: Dict) -> Dict:
        """
        Adjust correlation matrix for portfolio optimization
        Returns adjusted correlation matrix and diversification score
        """
        n = len(portfolio_stocks)
        if n == 0:
            return {'correlation_matrix': [], 'diversification_score': 0}
        
        # Get regime shift
        regime_shift = self.get_regime_correlation_shift(market_state)
        
        # Build correlation matrix
        correlation_matrix = []
        for i in range(n):
            row = []
            for j in range(n):
                if i == j:
                    row.append(1.0)
                else:
                    base_corr = self.calculate_sector_correlation(
                        portfolio_stocks[i].get('sector', '科技'),
                        portfolio_stocks[j].get('sector', '科技')
                    )
                    adjusted_corr = max(0, min(1, base_corr + regime_shift))
                    row.append(round(adjusted_corr, 2))
            correlation_matrix.append(row)
        
        # Calculate diversification score
        avg_correlation = sum(sum(row) for row in correlation_matrix) / (n * n)
        diversification_score = 1 - avg_correlation
        
        return {
            'correlation_matrix': correlation_matrix,
            'diversification_score': round(diversification_score, 2),
            'regime_shift': round(regime_shift, 2),
            'num_stocks': n
        }
    
    def calculate_portfolio_risk_adjustment(self, portfolio_stocks: List[Dict],
                                             market_state: Dict) -> float:
        """Calculate risk adjustment factor based on portfolio correlation"""
        portfolio_analysis = self.adjust_correlation_for_portfolio(portfolio_stocks, market_state)
        diversification_score = portfolio_analysis['diversification_score']
        
        # Higher diversification = lower risk adjustment needed
        risk_adjustment = 1 - (diversification_score * 0.3)
        
        return round(risk_adjustment, 2)
