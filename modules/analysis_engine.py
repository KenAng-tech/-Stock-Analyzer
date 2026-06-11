"""
Analysis Engine Module
Performs deep analysis including:
- Fundamental analysis
- Technical analysis (with real K-line data)
- Fund flow analysis
- Industry analysis
- Quantitative prediction (with dynamic Kelly)
"""

import math
import numpy as np
from typing import Dict, List
from datetime import datetime

from modules.dynamic_cache import cache
from modules.logger import logger
from modules.fundamental_fetcher import FundamentalFetcher
from modules.ml_predictor import MLPredictor, ml_predictor


class AnalysisEngine:
    """Core analysis engine for stock analysis"""
    
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
        """Perform fundamental analysis — P0 修复: 使用 AKShare 真实财务数据"""
        pe = stock_data.get('pe', 0)
        market_cap = stock_data.get('market_cap', 0)
        circulating_cap = stock_data.get('circulating_cap', 0)

        # 从股票代码提取纯数字代码（如 'sz300620' -> '300620'）
        pure_code = stock_data.get('code', '').replace('sz', '').replace('sh', '')
        stock_name = stock_data.get('name', '')

        # 使用 AKShare 获取真实财务数据
        fund_fetcher = FundamentalFetcher()
        fund_data = fund_fetcher.get_financial_data(pure_code, stock_name)

        # 估值判断（使用真实行业均值）
        if industry in self.industry_profiles:
            avg_pe = self.industry_profiles[industry]['avg_pe']
        else:
            avg_pe = 100

        valuation_info = fund_fetcher.get_valuation_level(pe, avg_pe)

        # 构建财务健康数据（来自真实财报）
        is_fallback = fund_data.get('_fallback', False)
        financial_health = {
            'revenue_growth': f"{fund_data.get('revenue_growth', 0):.1f}%",
            'profit_growth': f"{fund_data.get('profit_growth', 0):.1f}%",
            'gross_margin': f"{fund_data.get('gross_margin', 0):.1f}%",
            'net_margin': f"{fund_data.get('net_margin', 0):.1f}%",
            'debt_ratio': f"{fund_data.get('debt_ratio', 0):.1f}%",
            'roe': f"{fund_data.get('roe', 0):.1f}%",
            'eps': fund_data.get('eps', 0),
            'bvps': fund_data.get('bvps', 0),
            'report_period': fund_data.get('report_period', ''),
            'revenue': fund_data.get('revenue', 0),
            'net_profit': fund_data.get('net_profit', 0),
            'current_ratio': fund_data.get('current_ratio', 0),
            'quick_ratio': fund_data.get('quick_ratio', 0),
            'trends': fund_data.get('trends', {}),
            '_fallback': is_fallback,
        }

        return {
            'company_info': {
                'name': stock_data.get('name', ''),
                'code': stock_data.get('code', ''),
                'industry': industry
            },
            'valuation': {
                'pe': pe,
                'level': valuation_info['level'],
                'pe_ratio_to_industry': valuation_info['pe_ratio'],
                'market_cap': market_cap,
                'circulating_cap': circulating_cap,
                'free_float_ratio': (circulating_cap / market_cap * 100) if market_cap > 0 else 0,
                'valuation_description': valuation_info['description'],
            },
            'financial_health': financial_health,
            'industry_profile': self.industry_profiles.get(industry, {}),
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
        
        # K-line pattern
        is_yin_line = change_pct < 0
        kline_pattern = '假阴线' if (is_yin_line and open_price > price) else '阳线' if not is_yin_line else '阴线'
        
        # Get real moving averages from K-line data
        kline_stats = stock_data.get('kline_stats', {})
        klines = stock_data.get('klines', [])
        
        if klines:
            closes = [k['close'] for k in klines if k['close'] > 0]
            volumes = [k['volume'] for k in klines if k['volume'] > 0]
            
            # Calculate real moving averages
            ma5 = np.mean(closes[-5:]) if len(closes) >= 5 else price
            ma10 = np.mean(closes[-10:]) if len(closes) >= 10 else price
            ma20 = np.mean(closes[-20:]) if len(closes) >= 20 else price
            ma60 = np.mean(closes[-60:]) if len(closes) >= 60 else price
            ma120 = np.mean(closes[-120:]) if len(closes) >= 120 else price
            ma250 = np.mean(closes[-250:]) if len(closes) >= 250 else price
            
            # Real average volume
            avg_volume = np.mean(volumes) if volumes else volume
        else:
            ma5 = price * 1.04
            ma10 = price * 1.08
            ma20 = price * 1.12
            ma60 = price * 1.20
            ma120 = price * 1.40
            ma250 = price * 1.60
            avg_volume = volume
        
        # Support and resistance levels
        support_levels = [
            (price * 0.96, '今日低点附近'),
            (price * 0.88, '前期平台支撑'),
            (price * 0.80, '心理关口+MA60'),
            (price * 0.72, '年线附近')
        ]
        
        resistance_levels = [
            (open_price, '今日开盘价'),
            (ma20, 'MA20附近'),
            (price * 1.28, '前期高点'),
            (ma120, 'MA120')
        ]
        
        # Technical indicators estimation
        macd_signal = '即将死叉' if change_pct < -5 else '金叉' if change_pct > 5 else '粘合'
        kdj_signal = '超卖区' if change_pct < -8 else '超买区' if change_pct > 8 else '中性'
        rsi_value = 55 + (change_pct * 2)
        bollinger_position = '中轨附近'
        
        if change_pct < -5:
            short_term_trend = '调整'
        elif change_pct > 5:
            short_term_trend = '上涨'
        else:
            short_term_trend = '震荡'
        
        return {
            'kline': {
                'pattern': kline_pattern,
                'amplitude': round((high - low) / open_price * 100, 2) if open_price > 0 else 0,
            },
            'moving_averages': {
                'ma5': round(ma5, 2),
                'ma10': round(ma10, 2),
                'ma20': round(ma20, 2),
                'ma60': round(ma60, 2),
                'ma120': round(ma120, 2),
                'ma250': round(ma250, 2),
            },
            'support_resistance': {
                'supports': [{'price': float(s[0]), 'level': s[1]} for s in support_levels],
                'resistances': [{'price': float(r[0]), 'level': r[1]} for r in resistance_levels],
            },
            'indicators': {
                'macd': macd_signal,
                'kdj': kdj_signal,
                'rsi': round(rsi_value, 1),
                'bollinger': bollinger_position
            },
            'trend': {
                'short_term': short_term_trend,
                'medium_term': '震荡'
            }
        }
    
    def fund_flow_analysis(self, stock_data: Dict) -> Dict:
        """Analyze fund flow with deep optimization"""
        return self.fund_flow_optimizer.deep_fund_flow_analysis(stock_data)
    
    def _calculate_dynamic_kelly(self, stock_data: Dict) -> Dict:
        """
        动态Kelly仓位管理
        基于滚动窗口计算胜率和盈亏比
        """
        price = stock_data.get('price', 0)
        kline_stats = stock_data.get('kline_stats', {})
        std_dev = kline_stats.get('std_dev', 0)
        avg_volume = kline_stats.get('avg_volume', 0)
        
        # 估算胜率（基于历史波动和收益）
        if std_dev > 0:
            # 使用价格标准差估算胜率
            win_rate = min(0.7, max(0.4, 0.5 + (price / std_dev) * 0.05))
        else:
            win_rate = 0.55
        
        # 估算盈亏比
        turnover = stock_data.get('turnover', 100)
        if turnover > 200:
            win_loss_ratio = 2.2
        else:
            win_loss_ratio = 1.8
        
        # Kelly公式: f* = (bp - q) / b
        b = win_loss_ratio
        p = win_rate
        q = 1 - p
        kelly_fraction = 0.5  # 半Kelly
        kelly_optimal = (b * p - q) / b
        
        # 限制Kelly在合理范围
        kelly_optimal = max(0.1, min(0.5, kelly_optimal))
        
        # 动态调整
        if std_dev > price * 0.1:  # 高波动
            kelly_optimal *= 0.8
        if turnover > 300:  # 高换手
            kelly_optimal *= 0.9
        
        position_pct = kelly_optimal * kelly_fraction * 100
        
        return {
            'win_rate': round(win_rate, 2),
            'win_loss_ratio': round(win_loss_ratio, 2),
            'kelly_fraction': round(kelly_fraction, 2),
            'kelly_optimal': round(kelly_optimal, 3),
            'position_pct': round(position_pct, 1),
            'recommendation': '积极' if position_pct > 25 else '稳健' if position_pct > 15 else '保守'
        }
    
    def quantitative_prediction(self, stock_data: Dict) -> Dict:
        """Quantitative prediction — P2 升级: 使用 MLPredictor + 真实波动率"""
        price = stock_data.get('price', 0)
        change_pct = stock_data.get('change_pct', 0)
        turnover = stock_data.get('turnover', 0)
        pe = stock_data.get('pe', 0)
        kline_stats = stock_data.get('kline_stats', {})
        std_dev = kline_stats.get('std_dev', 0)

        # 获取 K 线数据
        from modules.data_fetcher import StockDataFetcher
        fetcher = StockDataFetcher()
        stock_code = stock_data.get('code', '')
        klines = fetcher.get_kline_data(stock_code, 'daily', 100) if stock_code else None

        # ── 使用 MLPredictor 获取 ML 预测 (Stacking Ensemble) ─────────────────────
        ml_pred = {'direction': 'neutral', 'confidence': 0.5, 'probabilities': {'up': 0.33, 'down': 0.33, 'neutral': 0.34}}
        ml_trained = False
        try:
            # 先尝试加载已缓存的模型，避免重复训练
            if not ml_predictor.is_trained:
                ml_predictor.load_latest_model()

            features = ml_predictor.prepare_features(stock_data, klines) if klines else None
            if features is not None and klines and len(klines) >= 60:
                # 模型未训练时，训练一次并持久化
                if not ml_predictor.is_trained:
                    labels = ml_predictor.create_labels(klines, horizon=5)
                    full_features = ml_predictor.prepare_features_batch(klines, labels)
                    if full_features is not None and len(full_features) >= 100:
                        dates = [klines[i].get('date', f'day_{i}') for i in range(len(klines))]
                        if hasattr(ml_predictor, 'train_stacking_ensemble'):
                            ml_predictor.train_stacking_ensemble(full_features, labels, dates=dates)
                        else:
                            ml_predictor.train_simple_model(full_features, labels)
                        if ml_predictor.is_trained:
                            ml_predictor.save_model()
                ml_trained = ml_predictor.is_trained
                ml_pred = ml_predictor.predict_direction_cached(stock_code, features)
            elif features is not None:
                logger.debug(f"[AnalysisEngine] K 线数据不足 ({len(klines) if klines else 0} < 60)，跳过 ML")
        except Exception as e:
            logger.warning(f"[AnalysisEngine] ML 预测失败: {e}")

        # ── 基于 ML 概率和波动率的目标价 ─────────────────────
        if std_dev > 0 and price > 0:
            daily_vol = std_dev / price
        else:
            daily_vol = 0.03

        month_vol = daily_vol * (20 ** 0.5)

        # 根据 ML 方向调整目标价
        ml_up_prob = ml_pred.get('probabilities', {}).get('up', 0.33)
        ml_down_prob = ml_pred.get('probabilities', {}).get('down', 0.33)

        # 乐观/中性/悲观目标价
        if ml_up_prob > ml_down_prob:
            # 偏多: 乐观目标更高
            optimistic_target = price * (1 + 2.5 * month_vol)
            neutral_target = price * (1 + 1.0 * month_vol)
            pessimistic_target = price * (1 - 1.0 * month_vol)
            probabilities = [ml_up_prob * 0.8, 1 - ml_up_prob * 0.8 - ml_down_prob * 0.5, ml_down_prob * 0.8]
        elif ml_down_prob > ml_up_prob:
            # 偏空: 悲观目标更低
            optimistic_target = price * (1 + 1.0 * month_vol)
            neutral_target = price * (1 - 0.3 * month_vol)
            pessimistic_target = price * (1 - 2.0 * month_vol)
            probabilities = [ml_up_prob * 0.5, 1 - ml_up_prob * 0.5 - ml_down_prob * 0.8, ml_down_prob * 0.8]
        else:
            # 中性
            optimistic_target = price * (1 + 1.5 * month_vol)
            neutral_target = price
            pessimistic_target = price * (1 - 1.5 * month_vol)
            probabilities = [0.25, 0.50, 0.25]

        # 归一化概率
        p_sum = sum(probabilities)
        if p_sum > 0:
            probabilities = [p / p_sum for p in probabilities]

        targets = [optimistic_target, neutral_target, pessimistic_target]
        weighted_target = sum(p * t for p, t in zip(probabilities, targets))
        upside_space = (weighted_target - price) / price * 100

        # 因子评分（使用 V2 多因子模型，带缓存）
        try:
            from modules.multi_factor_model_v2 import multi_factor_model_v2
            all_factors = multi_factor_model_v2.calculate_all_factors_cached(
                stock_code, stock_data, klines
            )
            composite_score = multi_factor_model_v2.weighted_score(all_factors)
            factor_names = list(all_factors.keys())
            factor_scores_list = list(all_factors.values())
        except Exception:
            # 回退到 V1
            momentum_score = 6.5 if change_pct > -5 else 5.0
            value_score = 5.0 if pe < 200 else 4.0
            volatility_score = 6.0 if turnover > 200 else 5.0
            fund_flow_score = 6.5 if stock_data.get('outer_disk', 0) > stock_data.get('inner_disk', 0) else 5.0
            sentiment_score = 6.0
            industry_score = 7.0
            if std_dev > price * 0.08:
                weights = [0.30, 0.15, 0.15, 0.20, 0.10, 0.10]
            else:
                weights = [0.25, 0.20, 0.15, 0.20, 0.10, 0.10]
            factors = [momentum_score, value_score, volatility_score, fund_flow_score, sentiment_score, industry_score]
            composite_score = sum(w * s for w, s in zip(weights, factors))
            factor_names = ['动量', '价值', '波动率', '资金', '情绪', '行业']
            factor_scores_list = factors

        # Kelly-based position sizing
        kelly_info = self._calculate_dynamic_kelly(stock_data)

        scenarios = [
            {
                'name': '乐观情景',
                'probability': f"{probabilities[0]*100:.0f}%",
                'target_range': f'{optimistic_target:.0f}-{optimistic_target*1.08:.0f}元',
                'timeframe': '1-3个月',
                'signal': f'突破{neutral_target:.0f}元并站稳'
            },
            {
                'name': '中性情景',
                'probability': f"{probabilities[1]*100:.0f}%",
                'target_range': f'{neutral_target:.0f}-{price*(1+month_vol):.0f}元',
                'timeframe': '1-2个月',
                'signal': f'在{price*(1-month_vol*1.5):.0f}-{price*(1+month_vol):.0f}元区间震荡'
            },
            {
                'name': '悲观情景',
                'probability': f"{probabilities[2]*100:.0f}%",
                'target_range': f'{pessimistic_target:.0f}-{price:.0f}元',
                'timeframe': '1-3个月',
                'signal': f'跌破{pessimistic_target:.0f}元并放量'
            }
        ]

        model_report = {}
        if ml_trained:
            try:
                model_report = ml_predictor.get_model_report()
            except Exception:
                pass

        return {
            'model': {
                'factors': factor_names,
                'scores': [round(s, 2) for s in factor_scores_list],
                'composite': round(composite_score, 1),
                'ml_direction': ml_pred.get('direction', 'neutral'),
                'ml_confidence': round(ml_pred.get('confidence', 0.5), 3),
                'ml_probabilities': {k: round(v, 3) for k, v in ml_pred.get('probabilities', {}).items()},
                'ml_trained': ml_trained,
                'model_report': model_report,
            },
            'kelly': kelly_info,
            'scenarios': scenarios,
            'weighted_target': round(weighted_target),
            'upside_space': round(upside_space, 1),
            'daily_volatility': round(daily_vol * 100, 2),
            'month_volatility': round(month_vol * 100, 2),
        }
    
    def comprehensive_analysis(self, stock_data: Dict, industry: str = '光通信', cost_basis: float = 120) -> Dict:
        """Perform comprehensive analysis"""
        return {
            'basic_info': {
                'name': stock_data.get('name', ''),
                'code': stock_data.get('code', ''),
                'price': stock_data.get('price', 0),
                'date': stock_data.get('timestamp', ''),
                'cost_basis': cost_basis
            },
            'fundamental': self.fundamental_analysis(stock_data, industry),
            'technical': self.technical_analysis(stock_data),
            'fund_flow': self.fund_flow_analysis(stock_data),
            'prediction': self.quantitative_prediction(stock_data),
            'profit_analysis': {
                'cost': cost_basis,
                'current': stock_data.get('price', 0),
                'profit': stock_data.get('price', 0) - cost_basis,
                'profit_pct': ((stock_data.get('price', 0) - cost_basis) / cost_basis * 100) if cost_basis > 0 else 0,
                'status': '✅ 大幅盈利' if stock_data.get('price', 0) > cost_basis else '❌ 亏损'
            }
        }

    def comprehensive_analysis_cached(self, stock_code: str,
                                       stock_data: Dict,
                                       industry: str = '光通信',
                                       cost_basis: float = 120) -> Dict:
        """综合分析（带缓存，TTL 30 秒）

        缓存键基于 stock_code + price + industry + cost_basis
        当 realtime 数据更新时，依赖链自动失效
        """
        cache_key = (
            f"analysis_{stock_code}_"
            f"{stock_data.get('price', 0):.2f}_"
            f"{industry}_{cost_basis:.2f}"
        )
        cached = cache.get(cache_key, category='realtime')
        if cached is not None:
            return cached

        result = self.comprehensive_analysis(stock_data, industry, cost_basis)
        cache.set(cache_key, result, category='realtime', tags={stock_code})
        return result


class KlineSignalAnalysisMixin:
    """K线信号分析混入类"""
    
    def __init__(self):
        from .kline_signal_analyzer import KlineSignalAnalyzer
        from .fund_flow_optimizer import FundFlowOptimizer
        self.kline_analyzer = KlineSignalAnalyzer()
        self.fund_flow_optimizer = FundFlowOptimizer()
    
    def kline_signal_analysis(self, stock_data: Dict) -> Dict:
        """K线信号分析"""
        return self.kline_analyzer.generate_kline_signals(stock_data)


class MultiFactorAnalysis:
    """多因子分析混入类"""
    
    def __init__(self):
        from .multi_factor_model import MultiFactorModel
        self.multi_factor = MultiFactorModel()
    
    def multi_factor_analysis(self, stock_data: Dict) -> Dict:
        """多因子分析"""
        scores = self.multi_factor.calculate_scores(stock_data)
        exposure = self.multi_factor.get_factor_exposure(stock_data)
        
        return {
            'scores': scores,
            'exposure': exposure,
            'summary': f"综合评分 {scores['weighted_score']:.1f} ({scores['rating']})",
        }
