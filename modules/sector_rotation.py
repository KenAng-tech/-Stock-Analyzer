"""
Sector Rotation Module
Implements sector rotation strategy based on industry momentum.
"""

import math
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta


class SectorRotation:
    """Sector rotation strategy"""
    
    def __init__(self):
        # Industry profiles with typical characteristics
        self.industry_profiles = {
            '光通信': {'avg_pe': 150, 'growth_rate': '20-35%', 'volatility': 'high', 'beta': 1.3},
            '半导体': {'avg_pe': 80, 'growth_rate': '15-25%', 'volatility': 'high', 'beta': 1.2},
            '新能源': {'avg_pe': 40, 'growth_rate': '25-40%', 'volatility': 'medium', 'beta': 1.1},
            '医药': {'avg_pe': 60, 'growth_rate': '10-20%', 'volatility': 'medium', 'beta': 0.9},
            '消费': {'avg_pe': 50, 'growth_rate': '10-15%', 'volatility': 'low', 'beta': 0.8},
            '金融': {'avg_pe': 15, 'growth_rate': '5-10%', 'volatility': 'low', 'beta': 0.7},
            '科技': {'avg_pe': 100, 'growth_rate': '20-30%', 'volatility': 'high', 'beta': 1.4},
            '制造': {'avg_pe': 40, 'growth_rate': '10-20%', 'volatility': 'medium', 'beta': 1.0},
            '能源': {'avg_pe': 20, 'growth_rate': '5-15%', 'volatility': 'medium', 'beta': 0.9},
            '地产': {'avg_pe': 15, 'growth_rate': '5-10%', 'volatility': 'medium', 'beta': 0.8},
        }
    
    def calculate_sector_momentum(self, stock_data: Dict, industry: str = '光通信') -> Dict:
        """Calculate sector momentum score"""
        pe = stock_data.get('pe', 100)
        change_pct = stock_data.get('change_pct', 0)
        turnover = stock_data.get('turnover', 100)
        
        profile = self.industry_profiles.get(industry, {})
        avg_pe = profile.get('avg_pe', 100)
        beta = profile.get('beta', 1.0)
        
        # PE relative to industry average
        pe_ratio = pe / avg_pe if avg_pe > 0 else 1.0
        pe_score = max(0, 10 - abs(pe_ratio - 1) * 5)
        
        # Momentum score
        momentum_score = min(10, max(0, change_pct * 2 + 5))
        
        # Volume score
        volume_score = min(10, max(0, turnover / 30))
        
        # Beta-adjusted momentum
        beta_adjusted_momentum = momentum_score * beta
        
        # Combined score
        combined = pe_score * 0.3 + momentum_score * 0.4 + volume_score * 0.3
        
        return {
            'pe_ratio': round(pe_ratio, 2),
            'pe_score': round(pe_score, 1),
            'momentum_score': round(momentum_score, 1),
            'volume_score': round(volume_score, 1),
            'beta_adjusted': round(beta_adjusted_momentum, 1),
            'combined_score': round(combined, 1),
            'rating': self._get_rating(combined)
        }
    
    def _get_rating(self, score: float) -> str:
        """Get sector rating"""
        if score >= 8:
            return '强烈推荐'
        elif score >= 6:
            return '推荐'
        elif score >= 4:
            return '中性'
        else:
            return '观望'
    
    def get_sector_trend(self, stock_data: Dict, industry: str = '光通信') -> str:
        """Determine sector trend"""
        momentum = stock_data.get('change_pct', 0)
        turnover = stock_data.get('turnover', 100)
        
        if momentum > 5 and turnover > 200:
            return '强势上涨'
        elif momentum > 2:
            return '温和上涨'
        elif momentum > -2:
            return '震荡整理'
        elif momentum > -5:
            return '温和下跌'
        else:
            return '强势下跌'
    
    def get_sector_recommendation(self, stock_data: Dict, industry: str = '光通信') -> Dict:
        """Get sector-based recommendation"""
        momentum = stock_data.get('change_pct', 0)
        pe = stock_data.get('pe', 100)
        turnover = stock_data.get('turnover', 100)
        
        profile = self.industry_profiles.get(industry, {})
        avg_pe = profile.get('avg_pe', 100)
        
        # Sector-specific rules
        if pe < avg_pe * 0.7:
            recommendation = '低估买入'
            reason = f'PE低于行业均值{avg_pe}的70%，具备安全边际'
        elif pe > avg_pe * 1.5:
            recommendation = '高估持有'
            reason = f'PE高于行业均值{avg_pe}的1.5倍，注意估值风险'
        else:
            recommendation = '合理持有'
            reason = f'PE处于行业合理区间'
        
        # Add momentum adjustment
        if momentum > 5 and turnover > 300:
            recommendation = '强势加仓'
            reason += '，量价齐升'
        elif momentum < -5 and turnover > 200:
            recommendation = '逢低布局'
            reason += '，超跌机会'
        
        return {
            'recommendation': recommendation,
            'reason': reason,
            'industry': industry,
            'industry_avg_pe': avg_pe,
            'current_pe': pe,
            'pe_vs_industry': round(pe / avg_pe, 2) if avg_pe > 0 else 0,
            'momentum': momentum,
            'turnover': turnover
        }


# Global instance
sector_rotation = SectorRotation()
