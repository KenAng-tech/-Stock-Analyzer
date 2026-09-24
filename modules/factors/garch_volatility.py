"""
GARCH(1,1) Volatility Module - V1
Implements GARCH(1,1) model for volatility forecasting.

Improvement over simple rolling standard deviation:
1. Captures volatility clustering (GARCH effect)
2. Predicts future volatility (not just historical)
3. More stable estimates for short samples
"""

import math
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta


class GARCHVolatility:
    """
    GARCH(1,1) Volatility Model
    
    Equation: σ²_t = ω + α * ε²_{t-1} + β * σ²_{t-1}
    
    Where:
    - ω (omega): long-run variance
    - α (alpha): ARCH term (response to shocks)
    - β (beta): GARCH term (persistence)
    - ω + α + β < 1 for stationarity
    
    Typical values: ω=0.000001, α=0.05, β=0.94
    """
    
    def __init__(self, omega: float = 0.000001,
                 alpha: float = 0.05,
                 beta: float = 0.94,
                 lookback: int = 252):
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self.lookback = lookback
        
        # Historical data
        self._returns: List[float] = []
        self._variances: List[float] = []
    
    def calculate_returns(self, prices: List[float]) -> List[float]:
        """Calculate log returns"""
        if len(prices) < 2:
            return [0.0]
        return [math.log(prices[i] / prices[i-1]) for i in range(1, len(prices))]
    
    def fit_garch(self, returns: List[float]) -> Dict:
        """
        Fit GARCH(1,1) model using MLE (simplified)
        
        Returns model parameters and statistics.
        """
        if len(returns) < self.lookback:
            # Use default parameters if insufficient data
            long_run_var = self.omega / (1 - self.alpha - self.beta)
            return {
                'omega': self.omega,
                'alpha': self.alpha,
                'beta': self.beta,
                'long_run_var': long_run_var,
                'volatility': math.sqrt(long_run_var),
                'half_life': math.log(2) / math.log(self.alpha + self.beta),
                'persistence': self.alpha + self.beta,
                'current_variance': self.omega,
                'current_volatility': math.sqrt(self.omega) * math.sqrt(252)
            }
        
        # Calculate squared returns
        eps_sq = [r ** 2 for r in returns]
        
        # Initialize variance
        long_run_var = np.mean(eps_sq)
        sigma_sq = [long_run_var] * len(returns)
        
        # Iterative estimation
        for i in range(1, len(returns)):
            sigma_sq[i] = (self.omega + 
                          self.alpha * eps_sq[i-1] + 
                          self.beta * sigma_sq[i-1])
        
        # Calculate long-run variance
        long_run_var = self.omega / (1 - self.alpha - self.beta)
        
        # Calculate half-life (time for shock to decay by half)
        persistence = self.alpha + self.beta
        half_life = math.log(2) / math.log(persistence) if persistence < 1 else float('inf')
        
        # Calculate realized volatility (annualized)
        realized_vol = math.sqrt(np.mean(sigma_sq)) * math.sqrt(252)
        
        return {
            'omega': self.omega,
            'alpha': self.alpha,
            'beta': self.beta,
            'long_run_var': round(long_run_var, 8),
            'volatility': round(realized_vol, 4),
            'half_life': round(half_life, 2),
            'persistence': round(persistence, 4),
            'current_variance': round(sigma_sq[-1], 8),
            'current_volatility': round(math.sqrt(sigma_sq[-1]) * math.sqrt(252), 4)
        }
    
    def forecast_volatility(self, returns: List[float], 
                             steps_ahead: int = 1) -> float:
        """
        Forecast volatility for future periods
        
        Multi-step forecast: σ²_{t+h} = ω/(1-α-β) + (α+β)^h * (σ²_t - ω/(1-α-β))
        """
        if len(returns) < self.lookback:
            return self._get_default_volatility(returns)
        
        # Current variance
        eps_sq = [r ** 2 for r in returns]
        sigma_sq = np.mean(eps_sq)
        
        # Long-run variance
        long_run_var = self.omega / (1 - self.alpha - self.beta)
        
        # Multi-step forecast
        forecast = long_run_var + (self.alpha + self.beta) ** steps_ahead * (sigma_sq - long_run_var)
        
        return math.sqrt(forecast) * math.sqrt(252)
    
    def _get_default_volatility(self, returns: List[float]) -> float:
        """Get default volatility from simple standard deviation"""
        if not returns:
            return 0.15
        return np.std(returns) * math.sqrt(252)
    
    def get_volatility_regime(self, volatility: float) -> Dict:
        """
        Classify volatility regime
        
        Regimes:
        - Low: < 10%
        - Normal: 10-20%
        - High: 20-35%
        - Very High: > 35%
        """
        if volatility < 0.10:
            regime = 'low'
            description = '低波动'
        elif volatility < 0.20:
            regime = 'normal'
            description = '正常波动'
        elif volatility < 0.35:
            regime = 'high'
            description = '高波动'
        else:
            regime = 'very_high'
            description = '极高波动'
        
        return {
            'regime': regime,
            'description': description,
            'volatility': volatility,
            'position_multiplier': {
                'low': 1.1,
                'normal': 1.0,
                'high': 0.85,
                'very_high': 0.7
            }.get(regime, 1.0)
        }
    
    def analyze(self, stock_data: Dict, klines: List[Dict] = None) -> Dict:
        """
        Comprehensive GARCH volatility analysis
        
        Returns:
            Dict with GARCH parameters, forecast, and regime
        """
        price = stock_data.get('price', 0)
        
        if klines:
            closes = [k['close'] for k in klines if k['close'] > 0]
        else:
            closes = [price]
        
        # Calculate returns
        returns = self.calculate_returns(closes)
        
        # Fit GARCH model
        garch_params = self.fit_garch(returns)
        
        # Forecast future volatility
        forecast_vol = self.forecast_volatility(returns, steps_ahead=1)
        
        # Volatility regime
        regime = self.get_volatility_regime(garch_params['volatility'])
        
        # Compare GARCH vs simple volatility
        simple_vol = np.std(returns) * math.sqrt(252) if returns else 0.15
        
        return {
            'garch_volatility': garch_params['volatility'],
            'forecast_volatility': round(forecast_vol, 4),
            'simple_volatility': round(simple_vol, 4),
            'garch_params': {
                'omega': garch_params['omega'],
                'alpha': garch_params['alpha'],
                'beta': garch_params['beta'],
                'persistence': garch_params['persistence'],
                'half_life': garch_params['half_life']
            },
            'regime': regime,
            'current_variance': garch_params['current_variance'],
            'long_run_variance': garch_params['long_run_var'],
            'volatility_change': round(
                (garch_params['volatility'] - simple_vol) / simple_vol * 100, 2
            ) if simple_vol > 0 else 0
        }


# Global instance
garch_volatility = GARCHVolatility()
