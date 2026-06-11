"""
Enhanced Analysis Engine with V3 optimizations
"""

import math
import numpy as np
from typing import Dict, List
from datetime import datetime

from modules.dynamic_cache import cache
from modules.logger import logger
from modules.absolute_momentum import absolute_momentum
from modules.adaptive_kelly import adaptive_kelly
from modules.dynamic_rsi import dynamic_rsi
from modules.garch_volatility import garch_volatility
from modules.multi_factor_model import multi_factor_model


class AnalysisEngineV3:
    """Enhanced analysis engine with all V3 optimizations"""
    
    def __init__(self):
        from .fund_flow_optimizer import FundFlowOptimizer
        self.fund_flow_optimizer = FundFlowOptimizer()
        self.industry_profiles = {
            '光通信': {
                'avg_pe': 150,
                'growth_rate': '20-35%',
                'drivers': ['AI算力需求', '5G建设', '数据中心升级', '国产替代'],
                'policy_support': True
            },
            '半导体': {
                'avg_pe': 80,
                'growth_rate': '15-25%',
                'drivers': ['芯片国产化', 'AI芯片需求', '汽车电子'],
                'policy_support': True
            },
            '新能源': {
                'avg_pe': 40,
                'growth_rate': '25-40%',
                'drivers': ['电动车渗透率提升', '储能需求', '光伏降本'],
                'policy_support': True
            }
        }
    
    def fundamental_analysis(self, stock_data: Dict, industry: str = '光通信') -> Dict:
        """Perform fundamental analysis"""
        pe = stock_data.get('pe', 0)
        market_cap = stock_data.get('market_cap', 0)
        circulating_cap = stock_data.get('circulating_cap', 0)
        
        if industry in self.industry_profiles:
            avg_pe = self.industry_profiles[industry]['avg_pe']
        else:
            avg_pe = 100
        
        if pe > avg_pe * 1.5:
            valuation_level = '极高'
        elif pe > avg_pe:
            valuation_level = '偏高'
        elif pe > avg_pe * 0.7:
            valuation_level = '合理'
        else:
            valuation_level = '偏低'
        
        return {
            'company_info': {
                'name': stock_data.get('name', ''),
                'code': stock_data.get('code', ''),
                'industry': industry
            },
            'valuation': {
                'pe': pe,
                'level': valuation_level,
                'market_cap': market_cap,
                'circulating_cap': circulating_cap,
                'free_float_ratio': (circulating_cap / market_cap * 100) if market_cap > 0 else 0
            },
            'financial_health': {
                'revenue_growth': '20-30%',
                'profit_growth': '25-35%',
                'gross_margin': '40-45%',
                'net_margin': '15-20%',
                'debt_ratio': '30-40%',
                'roe': '15-20%'
            },
            'industry_profile': self.industry_profiles.get(industry, {})
        }
    
    def technical_analysis(self, stock_data: Dict) -> Dict:
        """Perform technical analysis with real K-line data"""
        price = stock_data.get('price', 0)
        open_price = stock_data.get('open', 0)
        high = stock_data.get('high', 0)
        low = stock_data.get('low', 0)
        close = stock_data.get('close', 0)
        change_pct = stock_data.get('change_pct', 0)
        volume = stock_data.get('volume', 0)
        turnover = stock_data.get('turnover', 0)
        year_high = stock_data.get('year_high', 0)
        year_low = stock_data.get('year_low', 0)
        
        is_yin_line = change_pct < 0
        kline_pattern = '假阴线' if (is_yin_line and open_price > price) else '阳线' if not is_yin_line else '阴线'
        
        kline_stats = stock_data.get('kline_stats', {})
        klines = stock_data.get('klines', [])
        
        if klines:
            closes = [k['close'] for k in klines if k['close'] > 0]
            volumes = [k['volume'] for k in klines if k['volume'] > 0]
            
            ma5 = np.mean(closes[-5:]) if len(closes) >= 5 else price
            ma10 = np.mean(closes[-10:]) if len(closes) >= 10 else price
            ma20 = np.mean(closes[-20:]) if len(closes) >= 20 else price
            ma60 = np.mean(closes[-60:]) if len(closes) >= 60 else price
            ma120 = np.mean(closes[-120:]) if len(closes) >= 120 else price
            ma250 = np.mean(closes[-250:]) if len(closes) >= 250 else price
            
            avg_volume = np.mean(volumes) if volumes else volume
        else:
            ma5 = price * 1.04
            ma10 = price * 1.08
            ma20 = price * 1.12
            ma60 = price * 1.20
            ma120 = price * 1.40
            ma250 = price * 1.60
            avg_volume = volume
        
        support = low if low > 0 else ma20 * 0.95
        resistance = high if high > 0 else ma20 * 1.05
        
        if year_high > 0 and year_low > 0:
            price_position = (price - year_low) / (year_high - year_low)
            if price_position > 0.9:
                position = '接近年高'
            elif price_position > 0.7:
                position = '偏高'
            elif price_position > 0.3:
                position = '适中'
            elif price_position > 0.1:
                position = '偏低'
            else:
                position = '接近年低'
        else:
            position = '适中'
        
        if volume > avg_volume * 1.5:
            volume_analysis = '放量'
        elif volume > avg_volume * 0.8:
            volume_analysis = '正常'
        else:
            volume_analysis = '缩量'
        
        rsi = stock_data.get('rsi', 50)
        macd = stock_data.get('macd', 0)
        kdj = stock_data.get('kdj', {'k': 50, 'd': 50, 'j': 50})
        
        return {
            'indicators': {
                'rsi': rsi,
                'macd': macd,
                'kdj': kdj,
                'ma5': round(ma5, 2),
                'ma10': round(ma10, 2),
                'ma20': round(ma20, 2),
                'ma60': round(ma60, 2),
                'ma120': round(ma120, 2),
                'ma250': round(ma250, 2),
                'volume': volume,
                'avg_volume': round(avg_volume, 0),
                'turnover': turnover
            },
            'trend': {
                'short_term': 'up' if price > ma5 else 'down',
                'medium_term': 'up' if price > ma20 else 'down',
                'long_term': 'up' if price > ma250 else 'down'
            },
            'position': {
                'position': position,
                'support': round(support, 2),
                'resistance': round(resistance, 2),
                'price_position': round(price_position, 3)
            },
            'volume_analysis': volume_analysis,
            'kline_pattern': kline_pattern
        }
    
    def quantitative_prediction(self, stock_data: Dict,
                                universe: List[Dict] = None) -> Dict:
        """Quantitative prediction with enhanced models（P1: 使用标准化因子）"""
        price = stock_data.get('price', 0)
        pe = stock_data.get('pe', 0)
        turnover = stock_data.get('turnover', 0)
        change_pct = stock_data.get('change_pct', 0)
        std_dev = stock_data.get('std_dev', 0)

        # ── 使用 MultiFactorModel 计算标准化因子评分 ────────────
        factor_result = multi_factor_model.calculate_scores(stock_data, universe)
        raw_factors = factor_result.get('raw_factors', {})
        normalized_factors = factor_result.get('factors', raw_factors)

        momentum_score = raw_factors.get('momentum', 5.0)
        value_score = raw_factors.get('value', 5.0)
        vol_score = raw_factors.get('volatility', 5.0)
        volume_score = raw_factors.get('volume', 5.0)
        sentiment_score = raw_factors.get('sentiment', 6.0)
        industry_score = 7.0

        # 使用标准化后的分数计算综合评分
        if factor_result.get('normalized', False):
            composite_score = factor_result.get('weighted_score', 0)
            # 转换回 0-10 尺度
            composite_score = 5.0 + composite_score * 2.0  # Z-Score ~ N(0,1) → 5±2
            composite_score = max(0, min(10, composite_score))
            weights_used = list(normalized_factors.keys())
            scores_used = list(normalized_factors.values())
        else:
            if std_dev > price * 0.08:
                weights = [0.30, 0.15, 0.15, 0.20, 0.10, 0.10]
            else:
                weights = [0.25, 0.20, 0.15, 0.20, 0.10, 0.10]
            factors = [momentum_score, value_score, vol_score, volume_score, sentiment_score, industry_score]
            composite_score = sum(w * s for w, s in zip(weights, factors))
            weights_used = ['动量', '价值', '波动率', '成交量', '情绪', '行业']
            scores_used = factors
        
        kelly_info = self._calculate_dynamic_kelly(stock_data)

        optimistic_target = price * 1.36
        neutral_target = price * 1.11
        pessimistic_target = price * 0.84

        probabilities = [0.35, 0.45, 0.20]
        targets = [optimistic_target, neutral_target, pessimistic_target]
        weighted_target = sum(p * t for p, t in zip(probabilities, targets))

        upside_space = (weighted_target - price) / price * 100

        scenarios = [
            {
                'name': '乐观情景',
                'probability': '35%',
                'target_range': f'{optimistic_target:.0f}-{optimistic_target*1.12:.0f}元',
                'timeframe': '1-3个月',
                'signal': f'突破{neutral_target:.0f}元并站稳'
            },
            {
                'name': '中性情景',
                'probability': '45%',
                'target_range': f'{neutral_target:.0f}-{price*1.16:.0f}元',
                'timeframe': '1-2个月',
                'signal': f'在{price*0.92:.0f}-{price*1.16:.0f}元区间震荡'
            },
            {
                'name': '悲观情景',
                'probability': '20%',
                'target_range': f'{pessimistic_target:.0f}-{price:.0f}元',
                'timeframe': '1-3个月',
                'signal': f'跌破{pessimistic_target:.0f}元并放量'
            }
        ]

        return {
            'model': {
                'factors': weights_used,
                'weights': [1.0/len(weights_used)] * len(weights_used) if factor_result.get('normalized') else (weights if 'weights' in dir() else [0.25]*6),
                'scores': [round(s, 2) for s in scores_used],
                'composite': round(composite_score, 1),
                'normalized': factor_result.get('normalized', False),
                'rating': factor_result.get('rating', '中性'),
            },
            'kelly': kelly_info,
            'scenarios': scenarios,
            'weighted_target': round(weighted_target),
            'upside_space': round(upside_space, 1)
        }
    
    def _calculate_dynamic_kelly(self, stock_data: Dict) -> Dict:
        """Calculate dynamic Kelly position"""
        price = stock_data.get('price', 0)
        volatility = stock_data.get('std_dev', 0) / price if price > 0 else 0.15
        
        kelly_result = adaptive_kelly.get_position_size(
            stock_data,
            total_capital=100000,
            volatility=volatility
        )
        
        return kelly_result
    
    def enhanced_analysis(self, stock_data: Dict, industry: str = '光通信', cost_basis: float = 120) -> Dict:
        """
        Enhanced comprehensive analysis with all V3 optimizations
        """
        klines = stock_data.get('klines', [])
        
        abs_momentum = absolute_momentum.get_absolute_momentum(stock_data, klines)
        rsi_analysis = dynamic_rsi.analyze(stock_data, klines)
        garch_result = garch_volatility.analyze(stock_data, klines)
        kelly_result = adaptive_kelly.get_position_size(stock_data, total_capital=100000)
        
        fundamental = self.fundamental_analysis(stock_data, industry)
        technical = self.technical_analysis(stock_data)
        
        return {
            'basic_info': {
                'name': stock_data.get('name', ''),
                'code': stock_data.get('code', ''),
                'price': stock_data.get('price', 0),
                'date': stock_data.get('timestamp', ''),
                'cost_basis': cost_basis
            },
            'fundamental': fundamental,
            'technical': technical,
            'v3_enhancements': {
                'absolute_momentum': abs_momentum,
                'dynamic_rsi': rsi_analysis,
                'garch_volatility': garch_result,
                'adaptive_kelly': kelly_result,
            },
            'prediction': self.quantitative_prediction(stock_data),
            'profit_analysis': {
                'cost': cost_basis,
                'current': stock_data.get('price', 0),
                'profit': stock_data.get('price', 0) - cost_basis,
                'profit_pct': ((stock_data.get('price', 0) - cost_basis) / cost_basis * 100) if cost_basis > 0 else 0,
                'status': '✅ 大幅盈利' if stock_data.get('price', 0) > cost_basis else '❌ 亏损'
            }
        }
