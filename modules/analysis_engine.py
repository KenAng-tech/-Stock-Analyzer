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
import threading
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from modules.dynamic_cache import cache
from modules.logger import logger
from modules.fundamental_fetcher import FundamentalFetcher
from modules.ml_predictor import MLPredictor, ml_predictor
from modules.technical_indicators import comprehensive_technical_analysis

# Time-LLM 统一预测器 (P0: 集成到主分析流程)
try:
    from modules.models.time_llm import get_time_llm, TimeLLM
    HAS_TIME_LLM = True
except ImportError:
    HAS_TIME_LLM = False
    logger.warning("[AnalysisEngine] Time-LLM 不可用")

# Regime-Switching 多模型预测器 (P0: 集成到主分析流程)
try:
    from modules.regime_switching import get_regime_switching, RegimeSwitchingPredictor
    HAS_REGIME_SWITCHING = True
except ImportError:
    HAS_REGIME_SWITCHING = False
    logger.warning("[AnalysisEngine] Regime-Switching 不可用")

# ── SOTA 模型导入 (可选，失败时自动降级) ─────────────────────
HAS_PATCHTST = False
HAS_MAMBA = False
HAS_DIFFUSION = False
HAS_DRL = False
HAS_CONFORMAL = False
HAS_ALPHA158 = False
HAS_GNN = False
HAS_CROSS_MARKET = False
HAS_SENTIMENT = False

try:
    from modules.models.patchtst_integrator import PatchTSTIntegrator
    HAS_PATCHTST = True
    logger.info("[AnalysisEngine] PatchTST 可用")
except ImportError:
    logger.debug("[AnalysisEngine] PatchTST 不可用")

try:
    from modules.models.hft_mamba import MambaHFTPredictor
    HAS_MAMBA = True
    logger.info("[AnalysisEngine] Mamba HFT 可用")
except ImportError:
    logger.debug("[AnalysisEngine] Mamba HFT 不可用")

try:
    from modules.models.diffusion_model import DiffusionPredictor
    HAS_DIFFUSION = True
    logger.info("[AnalysisEngine] Diffusion 可用")
except ImportError:
    logger.debug("[AnalysisEngine] Diffusion 不可用")

try:
    from modules.models.drl_agent import DRLTradingAgent
    HAS_DRL = True
    logger.info("[AnalysisEngine] DRL 可用")
except ImportError:
    logger.debug("[AnalysisEngine] DRL 不可用")

try:
    from modules.models.conformal_predictor import ConformalPredictor
    HAS_CONFORMAL = True
    logger.info("[AnalysisEngine] Conformal Prediction 可用")
except ImportError:
    logger.debug("[AnalysisEngine] Conformal Prediction 不可用")

try:
    from modules.factors.alpha158_calculator import Alpha158Calculator
    HAS_ALPHA158 = True
    logger.info("[AnalysisEngine] Alpha158 可用")
except ImportError:
    logger.debug("[AnalysisEngine] Alpha158 不可用")

try:
    from modules.models.gnn_predictor import GNNPredictor
    HAS_GNN = True
    logger.info("[AnalysisEngine] GNN 可用")
except ImportError:
    logger.debug("[AnalysisEngine] GNN 不可用")

try:
    from modules.cross_market import CrossMarketAnalyzer
    HAS_CROSS_MARKET = True
    logger.info("[AnalysisEngine] Cross-Market 可用")
except ImportError:
    logger.debug("[AnalysisEngine] Cross-Market 不可用")

try:
    from modules.sentiment_engine import SentimentEngine
    HAS_SENTIMENT = True
    logger.info("[AnalysisEngine] Sentiment 可用")
except ImportError:
    logger.debug("[AnalysisEngine] Sentiment 不可用")


class AnalysisEngine:
    """Core analysis engine for stock analysis"""

    def __init__(self):
        from .fund_flow_optimizer import FundFlowOptimizer
        self.fund_flow_optimizer = FundFlowOptimizer()

        # 概念漂移检测器 (P0: 集成到主流程; 2026-09-08 断链修复)
        # 旧断链: 私有 ConceptDriftDetector 与全局 drift_monitor (app.py:268,
        # sota_status_routes 读它) 互不相通 → 面板恒显"无漂移"。统一用全局
        # DriftMonitor 单例: 喂源 (分析主流程) / 展示 (/api/sota/drift/status) /
        # 消费 (scheduler.should_retrain) 三端共享同一实例
        from modules.drift_monitor import get_drift_monitor
        self.drift_detector = get_drift_monitor()
        self._pending_drift_pair = None  # (date, up_prob) 跨时配对队列

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

        # 计算标准 PE（Price / AKShare EPS），用于验证腾讯动态 PE 的准确性
        price = stock_data.get('price', 0)
        eps = fund_data.get('eps', 0)
        pe_standard = round(price / eps, 2) if eps and eps > 0 and price > 0 else 0

        return {
            'company_info': {
                'name': stock_data.get('name', ''),
                'code': stock_data.get('code', ''),
                'industry': industry
            },
            'valuation': {
                'pe': pe,
                'pe_source': 'stock_data_pe',  # 2026-09-19: 主链 hithink/副链腾讯均可能无 PE, 不猜来源
                'pe_standard': pe_standard,  # 标准 PE = Price / AKShare 年度 EPS
                'eps': eps,  # AKShare 年度 EPS
                'level': valuation_info.get('level', ''),
                'pe_ratio_to_industry': valuation_info.get('pe_ratio', 0),
                'market_cap': market_cap,
                'circulating_cap': circulating_cap,
                'free_float_ratio': (circulating_cap / market_cap * 100) if market_cap > 0 else 0,
                'valuation_description': valuation_info.get('description', ''),
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
        
        # P0 修复 (2026-07-01): 使用真实技术指标计算，替代虚假估算
        # 如果 K 线数据充足，使用 comprehensive_technical_analysis
        if klines and len(klines) >= 15:
            tech_result = comprehensive_technical_analysis(klines)
            if 'error' not in tech_result:
                indicators = {
                    'rsi': {
                        'value': tech_result['rsi']['value'],
                        'signal': tech_result['rsi']['signal'],
                    },
                    'macd': {
                        'macd': tech_result['macd']['macd'],
                        'signal_line': tech_result['macd']['signal'],
                        'histogram': tech_result['macd']['histogram'],
                        'signal_type': tech_result['macd']['signal_type'],
                        'divergence': tech_result['macd'].get('divergence', '无'),
                    },
                    'kdj': {
                        'k': tech_result['kdj']['k'],
                        'd': tech_result['kdj']['d'],
                        'j': tech_result['kdj']['j'],
                        'signal': tech_result['kdj']['signal'],
                    },
                    'bollinger': {
                        'upper': tech_result['bollinger']['upper'],
                        'middle': tech_result['bollinger']['middle'],
                        'lower': tech_result['bollinger']['lower'],
                        'width': tech_result['bollinger']['width'],
                        'position': tech_result['bollinger']['position'],
                    },
                    'atr': tech_result.get('atr', 0),
                }
                short_term_trend = tech_result['summary']['overall_trend'] if 'summary' in tech_result else '震荡'
                medium_term = '偏多' if tech_result['summary']['bullish_signals'] > tech_result['summary']['bearish_signals'] else '偏空' if tech_result['summary']['bearish_signals'] > tech_result['summary']['bullish_signals'] else '震荡'
            else:
                # K 线数据不足 15 根，fallback 到简单估算
                indicators = {
                    'rsi': {'value': round(55 + change_pct * 2, 1), 'signal': '中性'},
                    'macd': {'signal_type': '粘合', 'divergence': '无'},
                    'kdj': {'signal': '中性'},
                    'bollinger': {'position': '中轨附近'},
                    'atr': 0,
                }
                short_term_trend = '调整' if change_pct < -3 else '上涨' if change_pct > 3 else '震荡'
                medium_term = '震荡'
        else:
            # 无 K 线数据，fallback 到简单估算
            indicators = {
                'rsi': {'value': round(55 + change_pct * 2, 1), 'signal': '中性'},
                'macd': {'signal_type': '粘合', 'divergence': '无'},
                'kdj': {'signal': '中性'},
                'bollinger': {'position': '中轨附近'},
                'atr': 0,
            }
            short_term_trend = '调整' if change_pct < -3 else '上涨' if change_pct > 3 else '震荡'
            medium_term = '震荡'

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
            'indicators': indicators,
            'trend': {
                'short_term': short_term_trend,
                'medium_term': medium_term
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

        # 获取 K 线数据 — 优先使用路由层预取的数据（避免重复网络调用）
        stock_code = stock_data.get('code', '')
        _klines = stock_data.get('_klines_daily')  # 路由层预注入
        if _klines:
            klines = _klines
        elif stock_code:
            from modules.data_fetcher import StockDataFetcher
            fetcher = StockDataFetcher()
            klines = fetcher.get_kline_data(stock_code, 'daily', 100)
        else:
            klines = None

        # ── 使用 MLPredictor 获取 ML 预测 (Stacking Ensemble) ─────────────────────
        # 注意：不在 analyze 请求中触发训练（太慢会导致请求超时）
        # 模型训练由 ModelTrainingScheduler 后台调度（每天 23:00 + API 手动触发）
        ml_pred = {'direction': 'neutral', 'confidence': 0.5, 'probabilities': {'up': 0.33, 'down': 0.33, 'neutral': 0.34}}
        ml_trained = False
        try:
            features = ml_predictor.prepare_features(stock_data, klines) if klines else None
            if features is not None and ml_predictor.is_trained:
                ml_trained = True
                ml_pred = ml_predictor.predict_direction_cached(stock_code, features)
            elif features is not None:
                logger.debug("[AnalysisEngine] ML 模型未训练，返回中性预测")
        except Exception as e:
            logger.warning(f"[AnalysisEngine] ML 预测失败: {e}")

        # ── Time-LLM 统一预测 — 已禁用（LLM 调用 35s+ 导致超时 + C 扩展 SIGSEGV） ──
        # 该模块在 HTTP 请求上下文中不可靠，改为后台定时推理
        time_llm_pred = None
        time_llm_regime = None
        time_llm_regime_weights = None
        time_llm_models_used = []
        # if HAS_TIME_LLM and klines and len(klines) >= 60:
        #     try:
        #         ... 原 TimeLLM 推理代码（35s+ 耗时，可能触发 SIGSEGV）
        #     except Exception as e:
        #         logger.warning(f"[AnalysisEngine] Time-LLM 预测失败: {e}")

        # ── Regime-Switching 多模型预测 ─────────────────────
        regime_pred = None
        regime_detected = None
        regime_weights = None
        # ── Regime-Switching — 已禁用（rs.predict() 触发 MPS SIGSEGV） ──
        # if HAS_REGIME_SWITCHING and klines and len(klines) >= 60:
        #     try:
        #         ... 原 Regime-Switching 推理代码（触发 MPS SIGSEGV）
        #     except Exception as e:
        #         logger.warning(f"[AnalysisEngine] Regime-Switching 预测失败: {e}")

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
            from modules.factors.multi_factor_model_v2 import multi_factor_model_v2
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

        # ── 概念漂移检测 (P0: 集成到主流程; 2026-09-08 配对修复) ─────────────────────
        # 旧断链: _last_prediction 每次请求被覆盖 + _last_actual 仅首次计算 →
        # 第二次起永远在比较"新预测 vs 旧 actual"(时间错位, 误差对失真)。
        # 修复: 跨时配对 — 本次分析把 (最后 K 线日期, up_prob) 记为 pending,
        # 下次分析 (若出现更新 K 线) 把 pending 预测 vs 预测日次日实际收益喂给
        # 全局 drift_monitor (Brier 语义: up 概率 vs 是否真涨)
        drift_status = {}
        try:
            # ① 喂源: 上一对 pending 配对 vs 预测日次日实际收益 (跨时, 无前视)
            if self._pending_drift_pair is not None and klines and len(klines) >= 2:
                pend_date, pend_up_prob = self._pending_drift_pair
                bars = [(k.get('date', ''), float(k.get('close', 0))) for k in klines]
                bars = [(d, c) for d, c in bars if d and c > 0]
                dates = [d for d, _ in bars]
                if pend_date in dates:
                    i = dates.index(pend_date)
                    if i + 1 < len(bars):  # 预测日之后至少出现一根新 K 线
                        next_ret = (bars[i + 1][1] - bars[i][1]) / bars[i][1]
                        self.drift_detector.check_prediction_error(
                            pend_up_prob, 1 if next_ret > 0 else 0
                        )
                        logger.debug(
                            f"[AnalysisEngine] 漂移喂源: pred_up={pend_up_prob:.3f} "
                            f"vs actual={next_ret:+.4f} ({pend_date})"
                        )

            # ② 配对: 本次分析记为新 pending (下次分析验证)
            if ml_up_prob is not None and klines:
                last_date = klines[-1].get('date', '')
                if last_date:
                    self._pending_drift_pair = (last_date, ml_up_prob)

            drift_status = self.drift_detector.get_status()
            if drift_status.get('drift_detected'):
                logger.warning("[AnalysisEngine] 概念漂移检测，ML 预测性能可能已退化")
        except Exception as e:
            logger.debug(f"[AnalysisEngine] 漂移检测异常: {e}")

        # ── SOTA 模型集成 ─────────────────────
        sota_models = {}

        # 0. PatchTST 预测 (ICLR 2024 时序基础模型; 2026-09-21 形制修复)
        # 修复: 旧形直喂 prepare_features 截面特征 reshape (n_feat,1) → 模型
        # reshape '[1,1,96]' 恒抛 = 每 analyze 2 WARNING + 集成静默缺 patchtst
        # (自检失明)。现与 sota_predict /api/sota/patchtst/predict 端点共用
        # (seq_len, 12) 时序形制: build_seq_features_12 从日 K 构窗 (端点实证形)。
        if HAS_PATCHTST and klines and len(klines) >= 60:
            try:
                from modules.models.patchtst_integrator import (
                    get_patchtst, build_seq_features_12)
                patchtst = get_patchtst()
                closes = [float(k['close']) for k in klines
                          if k.get('close') is not None]
                vols = [float(k.get('volume') or 0.0) for k in klines]
                patchtst_X = build_seq_features_12(
                    closes, vols, seq_len=min(60, len(closes)))
                if patchtst_X is not None:
                    result = patchtst.predict(patchtst_X)
                    if result:
                        sota_models['patchtst'] = {
                            'direction': result.get('direction', 'neutral'),
                            'confidence': round(result.get('confidence', 0.5), 3),
                            'probabilities': result.get('probabilities', {}),
                            'trained': patchtst.trained if hasattr(patchtst, 'trained') else False,
                        }
                        logger.info(f"[AnalysisEngine] PatchTST 预测: {sota_models['patchtst']['direction']} (conf={sota_models['patchtst']['confidence']:.3f})")
                else:
                    logger.debug(f"[AnalysisEngine] PatchTST 跳过 (K 线形制不足, 非断链)")
            except Exception as e:
                logger.warning(f"[AnalysisEngine] PatchTST 预测失败 (跳过, 非断链): {e}")

        # 1. Mamba HFT 预测 (使用 ml_predictor.prepare_features)
        if HAS_MAMBA and klines and len(klines) >= 60:
            try:
                from modules.models.hft_mamba import get_mamba_hft_predictor
                mamba = get_mamba_hft_predictor()
                features = ml_predictor.prepare_features(stock_data, klines)
                if features is not None:
                    if len(features.shape) == 1:
                        features = features.reshape(1, 1, -1)
                    elif len(features.shape) == 2:
                        features = features.reshape(features.shape[0], 1, -1)
                    result = mamba.predict(features)
                    sota_models['mamba'] = {
                        'direction': result.get('direction', 'neutral'),
                        'confidence': round(result.get('confidence', 0.5), 3),
                        'probabilities': result.get('probabilities', {}),
                        'trained': mamba.trained if hasattr(mamba, 'trained') else False,
                    }
                    logger.info(f"[AnalysisEngine] Mamba HFT 预测: {sota_models['mamba']['direction']} (conf={sota_models['mamba']['confidence']:.3f})")
            except Exception as e:
                logger.warning(f"[AnalysisEngine] Mamba HFT 预测失败: {e}")

        # 2. Diffusion 预测
        if HAS_DIFFUSION and klines and len(klines) >= 60:
            try:
                from modules.models.diffusion_model import get_diffusion_predictor
                diffusion = get_diffusion_predictor()
                features = ml_predictor.prepare_features(stock_data, klines)
                if features is not None:
                    if len(features.shape) == 1:
                        features = features.reshape(1, 1, -1)
                    elif len(features.shape) == 2:
                        features = features.reshape(features.shape[0], 1, -1)
                    result = diffusion.predict(features)
                    uncertainty_raw = result.get('uncertainty', {})
                    sota_models['diffusion'] = {
                        'direction': result.get('direction', 'neutral'),
                        'confidence': round(result.get('confidence', 0.5), 3),
                        'probabilities': result.get('probabilities', {}),
                        'uncertainty': float(uncertainty_raw.get('std', 0.1)) if isinstance(uncertainty_raw, dict) else round(uncertainty_raw, 4),
                        'trained': diffusion.trained if hasattr(diffusion, 'trained') else False,
                    }
                    logger.info(f"[AnalysisEngine] Diffusion 预测: {sota_models['diffusion']['direction']} (conf={sota_models['diffusion']['confidence']:.3f})")
            except Exception as e:
                logger.warning(f"[AnalysisEngine] Diffusion 预测失败: {e}")

        # 3. DRL 决策
        if HAS_DRL and klines and len(klines) >= 60:
            try:
                from modules.models.drl_agent import get_drl_agent, PortfolioState
                drl = get_drl_agent()
                # ── 自动训练：如果未训练，则从 klines 提取特征并训练 ──
                if not getattr(drl, '_trained', False):
                    try:
                        labels = ml_predictor.create_labels(klines, horizon=5)
                        features = ml_predictor.prepare_features_batch(klines, labels)
                        if features is not None and len(features) >= 30:
                            env_data = {
                                'features': features,
                                'labels': labels,
                                'seq_len': len(features),
                                'n_features': features.shape[1],
                            }
                            drl.train(env_data, epochs=50, batch_size=32)
                            logger.info(f"[AnalysisEngine] DRL 自动训练完成: {len(features)} 样本, 50 epochs")
                    except Exception as e:
                        logger.warning(f"[AnalysisEngine] DRL 自动训练失败: {e}")
                portfolio = PortfolioState()
                drl_decision = drl.decide(klines, portfolio, stock_code)
                if drl_decision:
                    # TradingDecision 是 dataclass，不是 dict
                    action = getattr(drl_decision, 'action', 'hold')
                    confidence = getattr(drl_decision, 'confidence', 0.5)
                    target_weight = getattr(drl_decision, 'target_weight', 0.0)
                    sota_models['drl'] = {
                        'action': action,
                        'confidence': round(float(confidence), 3),
                        'position_size': round(float(target_weight), 3),
                        'trained': getattr(drl, '_trained', False),
                    }
                    logger.info(f"[AnalysisEngine] DRL 决策: {sota_models['drl']['action']} (conf={sota_models['drl']['confidence']:.3f})")
            except Exception as e:
                logger.warning(f"[AnalysisEngine] DRL 决策失败: {e}")

        # 4. Conformal Prediction 区间 (使用 PatchTST/ML 预测作为基础)
        if HAS_CONFORMAL and klines and len(klines) >= 120:
            try:
                from modules.models.conformal_predictor import get_conformal_predictor
                closes = np.array([float(k.get('close', 0)) for k in klines if float(k.get('close', 0)) > 0])
                if len(closes) >= 120:
                    # 用历史收益率校准: 取前 150 根 K 线的收益率作为校准集 (约 149 样本)
                    # 之前的 closes[-100:-40] 只有 60 个收盘价 → np.diff 后仅 59 样本
                    cal_window = min(150, len(closes) - 10)
                    cal_returns = np.diff(closes[:cal_window]) / closes[:cal_window][:-1]
                    test_closes = closes[-10:]
                    test_returns = np.diff(test_closes) / test_closes[:-1]
                    if len(cal_returns) >= 50 and len(test_returns) > 0:
                        conformal = get_conformal_predictor()
                        X_cal = np.zeros((len(cal_returns), 1))
                        conformal.calibrate(X_cal, cal_returns)
                        conformal_result = conformal.predict_with_interval(test_returns[-1].reshape(1, 1))
                        if conformal_result:
                            sota_models['conformal'] = {
                                'point_prediction': round(float(conformal_result.prediction), 6),
                                'lower_bound': round(float(conformal_result.lower), 6),
                                'upper_bound': round(float(conformal_result.upper), 6),
                                'confidence_interval': round(float(conformal_result.upper - conformal_result.lower), 6),
                                'coverage_probability': round(1.0 - conformal.alpha, 2),
                                'calibrated': conformal_result.calibrated if hasattr(conformal_result, 'calibrated') else True,
                            }
                            logger.info(f"[AnalysisEngine] Conformal 预测区间: [{sota_models['conformal']['lower_bound']:.4f}, {sota_models['conformal']['upper_bound']:.4f}]")
            except Exception as e:
                logger.warning(f"[AnalysisEngine] Conformal Prediction 失败: {e}")

        # 5. Alpha158 因子计算
        if HAS_ALPHA158 and klines and len(klines) >= 60:
            try:
                from modules.factors.alpha158_calculator import get_alpha158_calculator
                calc = get_alpha158_calculator()
                alpha_factors = calc.calculate_all(klines)
                if alpha_factors:
                    latest_values = {}
                    for k, v in alpha_factors.items():
                        # 支持标量和 numpy 数组两种返回类型
                        if isinstance(v, np.ndarray) and len(v) > 0:
                            last_val = v[-1]
                        elif np.isscalar(v):
                            last_val = v
                        else:
                            continue
                        if np.isscalar(last_val) and not np.isnan(last_val) and not np.isinf(last_val):
                            latest_values[k] = round(float(last_val), 4)
                    top_factors = sorted(latest_values.items(), key=lambda x: abs(x[1]), reverse=True)[:20]
                    sota_models['alpha158'] = {
                        'n_factors': len(latest_values),
                        'top_factors': dict(top_factors),
                        'latest_values': latest_values,
                        'available': len(latest_values) > 0,
                    }
                    logger.info(f"[AnalysisEngine] Alpha158: 计算 {sota_models['alpha158']['n_factors']} 个因子")
            except Exception as e:
                logger.warning(f"[AnalysisEngine] Alpha158 因子计算失败: {e}")

        # 6. GNN 图神经网络预测 (需要 12 维特征)
        # 2026-09-03: 仅 trained 真信号才注入 (未训练 = 无信息随机权重, 直接跳过不占投票;
        # GNNPredictor.train 门控: holdout_acc>0.4 才算 trained)
        if HAS_GNN and klines and len(klines) >= 60:
            try:
                from modules.models.gnn_predictor import get_gnn_predictor
                gnn = get_gnn_predictor()
                if gnn is not None and getattr(gnn, '_is_trained', False):
                    features = ml_predictor.prepare_features(stock_data, klines)
                    if features is not None:
                        # GNN 期望 (1, n_features) — 单只股票
                        if len(features.shape) == 1 and features.shape[0] == gnn.n_features:
                            gnn_pred = gnn.predict(features.reshape(1, -1))
                            if gnn_pred:
                                conf = gnn_pred.get('confidence', [0.5])
                                if isinstance(conf, list):
                                    conf = conf[0] if conf else 0.5
                                sota_models['gnn'] = {
                                    'direction': gnn_pred.get('predictions', ['neutral'])[0] if isinstance(gnn_pred.get('predictions'), list) else gnn_pred.get('direction', 'neutral'),
                                    'confidence': round(float(conf), 3),
                                    'is_trained': gnn_pred.get('is_trained', False),
                                }
                                logger.info(f"[AnalysisEngine] GNN 预测: {sota_models['gnn']['direction']} (conf={sota_models['gnn']['confidence']:.3f})")
            except Exception as e:
                logger.warning(f"[AnalysisEngine] GNN 预测失败: {e}")

        # 7. Cross-Market 跨市场因子
        if HAS_CROSS_MARKET:
            try:
                from modules.cross_market import get_cross_market_analyzer
                analyzer = get_cross_market_analyzer()
                market_summary = analyzer.get_summary()
                if market_summary:
                    sota_models['cross_market'] = {
                        'total_stocks': market_summary.get('total_stocks', 0),
                        'markets': market_summary.get('markets', {}),
                        'correlations_computed': market_summary.get('correlations_computed', False),
                        'available': True,
                    }
                    logger.info(f"[AnalysisEngine] Cross-Market: {sota_models['cross_market']['total_stocks']} 只股票已注册")
            except Exception as e:
                logger.warning(f"[AnalysisEngine] Cross-Market 分析失败: {e}")

        # 8. Sentiment 情感分析 (从 stock_data 获取新闻，无 news 则自动获取)
        if HAS_SENTIMENT:
            try:
                from modules.sentiment_engine import get_sentiment_engine
                engine = get_sentiment_engine()
                news_texts = []
                if stock_data.get('news'):
                    for news in stock_data.get('news', [])[:10]:
                        if isinstance(news, dict):
                            news_texts.append(news.get('title', '') + ' ' + news.get('content', ''))
                        elif isinstance(news, str):
                            news_texts.append(news)
                # 如果 stock_data 没有 news，自动获取
                if not news_texts:
                    stock_code = stock_data.get('code', '')
                    if stock_code:
                        try:
                            from modules.data_fetcher import StockDataFetcher
                            df = StockDataFetcher()
                            news_list = df.get_stock_news(stock_code)
                            if news_list:
                                for news in news_list[:10]:
                                    if isinstance(news, dict):
                                        news_texts.append(news.get('title', '') + ' ' + news.get('content', ''))
                                    elif isinstance(news, str):
                                        news_texts.append(news)
                        except Exception:
                            pass
                if news_texts:
                    sentiment_results = engine.analyze_batch(news_texts) if hasattr(engine, 'analyze_batch') else [engine.analyze(t) for t in news_texts[:5]]
                    avg_score = float(np.mean([r.get('score', 0) for r in sentiment_results]))
                    avg_confidence = float(np.mean([r.get('confidence', 0) for r in sentiment_results]))
                    positive_count = sum(1 for r in sentiment_results if r.get('label') == 'positive')
                    negative_count = sum(1 for r in sentiment_results if r.get('label') == 'negative')
                    neutral_count = len(sentiment_results) - positive_count - negative_count
                    sota_models['sentiment'] = {
                        'score': round(avg_score, 4),
                        'confidence': round(avg_confidence, 4),
                        'label': 'positive' if avg_score > 0.1 else 'negative' if avg_score < -0.1 else 'neutral',
                        'positive_ratio': round(positive_count / len(sentiment_results), 3) if sentiment_results else 0.5,
                        'negative_ratio': round(negative_count / len(sentiment_results), 3) if sentiment_results else 0.5,
                        'neutral_ratio': round(neutral_count / len(sentiment_results), 3) if sentiment_results else 0.5,
                        'n_articles': len(sentiment_results),
                        'available': True,
                    }
                    logger.info(f"[AnalysisEngine] Sentiment: {sota_models['sentiment']['label']} (score={sota_models['sentiment']['score']:.3f}, articles={sota_models['sentiment']['n_articles']})")
                elif klines and len(klines) >= 20:
                    # 降级方案：基于价格动量计算市场情绪
                    closes = np.array([float(k.get('close', 0)) for k in klines[-20:]])
                    returns = np.diff(closes) / closes[:-1]
                    avg_ret = float(np.mean(returns))
                    recent_5d = float(np.mean(returns[-5:])) if len(returns) >= 5 else avg_ret
                    score = round(np.tanh(avg_ret * 10), 4)  # 归一化到 [-1, 1]
                    sota_models['sentiment'] = {
                        'score': score,
                        'confidence': 0.3,
                        'label': 'positive' if score > 0.1 else 'negative' if score < -0.1 else 'neutral',
                        'positive_ratio': max(0.5, 0.5 + score / 2),
                        'negative_ratio': max(0.5, 0.5 - score / 2),
                        'neutral_ratio': max(0.0, 1.0 - abs(score)),
                        'n_articles': 0,
                        'available': True,
                        'method': 'price_momentum_fallback',
                    }
                    logger.info(f"[AnalysisEngine] Sentiment (价格动量降级): {sota_models['sentiment']['label']} (score={score:.3f})")
            except Exception as e:
                logger.warning(f"[AnalysisEngine] Sentiment 分析失败: {e}")

        # ── 综合 SOTA 模型投票 ─────────────────────
        sota_vote = {'up': 0, 'down': 0, 'neutral': 0}
        sota_confidences = []
        for name, pred in sota_models.items():
            if pred.get('direction') in ('up', 'down'):
                sota_vote[pred['direction']] += 1
                sota_confidences.append(pred.get('confidence', 0.5))
            elif pred.get('drl', {}).get('action') == 'buy':
                sota_vote['up'] += 1
                sota_confidences.append(pred.get('drl', {}).get('confidence', 0.5))
            elif pred.get('drl', {}).get('action') == 'sell':
                sota_vote['down'] += 1
                sota_confidences.append(pred.get('drl', {}).get('confidence', 0.5))

        sota_consensus = max(sota_vote, key=sota_vote.get) if any(sota_vote.values()) else 'neutral'
        sota_agreement = max(sota_vote.values()) / max(sum(sota_vote.values()), 1)
        sota_avg_confidence = float(np.mean(sota_confidences)) if sota_confidences else 0.5

        # ── P1: 统一决策引擎 — 使用预计算结果 (消除 HTTP 自调) ──
        try:
            from modules.unified_decision_engine import get_unified_decision_engine
            ude = get_unified_decision_engine()
            ude_result = ude.decide_from_predictions(
                stock_code, sota_models, klines
            )
            # UDE 结果覆盖简单投票结果
            # model_votes: Dict[str, str] (model→direction, 无置信度) — 取均值会 int+str;
            # conf 保留上方 SOTA 投票均值, UDE 只覆盖方向与共识度 (2026-09-19 断链修复)
            sota_consensus = ude_result.direction
            sota_agreement = ude_result.consensus
            logger.info(
                f"[AnalysisEngine] UDE 统一决策: {sota_consensus} "
                f"(conf={ude_result.confidence:.2f}, models={ude_result.n_models})"
            )
        except Exception as e:
            logger.warning(f"[AnalysisEngine] UDE 统一决策失败: {e}")

        # ── P2: 多模态融合预测 — 接入生产流水线 ──
        try:
            from modules.multi_modal_fusion import get_multi_modal_fusion
            from modules.hmm_market_detector import MarketRegimeDetector
            regime_detector = MarketRegimeDetector()
            regime_info = regime_detector.get_regime_with_confidence(stock_data or {}, klines)
            fusion_regime = regime_info.get('regime', 'sideways')
            # HMM 返回 'bull'/'bear'/'sideways' → 映射到 fusion 的 'bullish'/'bearish'/'sideways'
            regime_map = {'bull': 'bullish', 'bear': 'bearish', 'sideways': 'sideways'}
            fusion_regime = regime_map.get(fusion_regime, 'sideways')

            fusion = get_multi_modal_fusion()
            fusion_result = fusion.predict(stock_code, fusion_regime, use_cache=True)

            # 注入 sota_models → UDE 自动纳入投票
            sota_models['multi_modal_fusion'] = {
                'direction': fusion_result.get('direction'),
                'confidence': fusion_result.get('confidence', 0.5),
                'regime': fusion_regime,
                'modalities': fusion_result.get('modalities', {}),
                'consensus': fusion_result.get('consensus', 0.0),
            }
            logger.info(
                f"[MultiModalFusion] 融合预测: {stock_code} "
                f"→ {fusion_result['direction']} (conf={fusion_result['confidence']:.2f}, "
                f"regime={fusion_regime}, consensus={fusion_result.get('consensus', 0):.2f})"
            )
        except Exception as e:
            logger.warning(f"[MultiModalFusion] 融合预测失败: {e}")

        # ── P1: Alpha 衰减监控 — 记录因子值 + 实际收益 ──
        try:
            if klines and len(klines) >= 2 and 'alpha158' in sota_models:
                from modules.factors.factor_ic_monitor import factor_ic_monitor
                alpha_factors = sota_models['alpha158'].get('latest_values', {})
                if alpha_factors and len(alpha_factors) > 0:
                    closes = [float(k.get('close', 0)) for k in klines if float(k.get('close', 0)) > 0]
                    if len(closes) >= 2:
                        actual_return = (closes[-1] - closes[-2]) / closes[-2]
                        factor_ic_monitor.record_factor_return(
                            stock_code, alpha_factors, actual_return
                        )
                        logger.debug(
                            f"[AlphaDecay] 记录: {stock_code} return={actual_return:+.4f} "
                            f"factors={len(alpha_factors)}"
                        )
        except Exception as e:
            logger.debug(f"[AlphaDecay] 因子监控失败: {e}")

        # ── P2: 多模态融合 — IC 追踪记录实际收益 ──
        try:
            if klines and 'multi_modal_fusion' in sota_models:
                from modules.multi_modal_fusion import get_multi_modal_fusion
                fusion = get_multi_modal_fusion()
                modalities = sota_models['multi_modal_fusion'].get('modalities', {})
                closes = [float(k.get('close', 0)) for k in klines if float(k.get('close', 0)) > 0]
                if len(closes) >= 2:
                    actual_return = (closes[-1] - closes[-2]) / closes[-2]
                    fusion._ic_tracker.record(modalities, actual_return)
        except Exception as e:
            logger.debug(f"[MultiModalFusion] IC 追踪失败: {e}")

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
                # Time-LLM 统一预测
                'time_llm': time_llm_pred,
                'time_llm_regime': time_llm_regime,
                'time_llm_regime_weights': time_llm_regime_weights,
                'time_llm_models_used': time_llm_models_used,
                # Regime-Switching 多模型预测
                'regime_switching': regime_pred,
                'regime_detected': regime_detected,
                'regime_weights': regime_weights,
            },
            'kelly': kelly_info,
            'scenarios': scenarios,
            'weighted_target': round(weighted_target),
            'upside_space': round(upside_space, 1),
            'daily_volatility': round(daily_vol * 100, 2),
            'month_volatility': round(month_vol * 100, 2),
            'drift_status': drift_status,  # P0: 概念漂移检测状态
            # ── SOTA 模型集成 ──
            'sota_models': sota_models,
            'sota_consensus': sota_consensus,
            'sota_agreement': round(sota_agreement, 3),
            'sota_avg_confidence': round(sota_avg_confidence, 4),
        }
    
    def comprehensive_analysis(self, stock_data: Dict, industry: str = '光通信', cost_basis: float = 120) -> Dict:
        """Perform comprehensive analysis"""
        analysis = {
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

        # ── Sentiment 情感分析 ──
        if HAS_SENTIMENT:
            try:
                from modules.sentiment_engine import get_sentiment_engine
                engine = get_sentiment_engine()
                news_texts = []
                if stock_data.get('news'):
                    for news in stock_data.get('news', [])[:10]:
                        if isinstance(news, dict):
                            news_texts.append(news.get('title', '') + ' ' + news.get('content', ''))
                        elif isinstance(news, str):
                            news_texts.append(news)
                if news_texts:
                    sentiment_results = engine.analyze_batch(news_texts) if hasattr(engine, 'analyze_batch') else [engine.analyze(t) for t in news_texts[:5]]
                    avg_score = np.mean([r.get('score', 0) for r in sentiment_results])
                    avg_confidence = np.mean([r.get('confidence', 0) for r in sentiment_results])
                    positive_count = sum(1 for r in sentiment_results if r.get('label') == 'positive')
                    negative_count = sum(1 for r in sentiment_results if r.get('label') == 'negative')
                    neutral_count = len(sentiment_results) - positive_count - negative_count
                    analysis['sentiment'] = {
                        'score': round(float(avg_score), 4),
                        'confidence': round(float(avg_confidence), 4),
                        'label': 'positive' if avg_score > 0.1 else 'negative' if avg_score < -0.1 else 'neutral',
                        'positive_ratio': round(positive_count / len(sentiment_results), 3) if sentiment_results else 0.5,
                        'negative_ratio': round(negative_count / len(sentiment_results), 3) if sentiment_results else 0.5,
                        'neutral_ratio': round(neutral_count / len(sentiment_results), 3) if sentiment_results else 0.5,
                        'n_articles': len(sentiment_results),
                    }
                    logger.info(f"[AnalysisEngine] Sentiment: {analysis['sentiment']['label']} (score={analysis['sentiment']['score']:.3f})")
            except Exception as e:
                logger.warning(f"[AnalysisEngine] Sentiment 分析失败: {e}")
                analysis['sentiment'] = {'label': 'neutral', 'score': 0.0, 'confidence': 0.0, 'available': False}
        else:
            analysis['sentiment'] = {'label': 'neutral', 'score': 0.0, 'confidence': 0.0, 'available': False}

        return analysis

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


class FactorWeightScheduler:
    """
    动态因子权重调度器

    功能:
      - 跟踪因子历史值和收益率序列
      - 定期基于滚动 IC/ICIR 更新因子权重
      - 后台线程执行，不阻塞主线程

    用法:
      scheduler = FactorWeightScheduler(multi_factor_model_v2, interval_hours=24)
      scheduler.start()
      scheduler.record(stock_code, factor_values, return_value)  # 记录每日数据
      scheduler.stop()
    """

    def __init__(self, factor_model, interval_hours: int = 24, window: int = 60):
        """
        Args:
            factor_model: MultiFactorModelV2 实例
            interval_hours: 权重更新间隔（小时）
            window: 滚动 IC 窗口大小（天）
        """
        self.factor_model = factor_model
        self.interval_hours = interval_hours
        self.window = window
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        # 因子历史数据: {factor_name: [values...], ...}
        self._factor_history: Dict[str, List[float]] = {}
        # 收益率序列: [returns...]
        self._return_history: List[float] = []
        # 股票列表（用于去重）
        self._tracked_stocks: set = set()

    def start(self):
        """启动后台调度线程"""
        if self._thread and self._thread.is_alive():
            logger.warning("[FactorWeightScheduler] 调度器已在运行")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._scheduler_loop,
            name="factor-weight-scheduler",
            daemon=True,
        )
        self._thread.start()
        logger.info(f"[FactorWeightScheduler] 调度器已启动，更新间隔: {self.interval_hours}h, window: {self.window}天")

    def stop(self):
        """停止后台调度"""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        logger.info("[FactorWeightScheduler] 调度器已停止")

    def record(self, stock_code: str, factor_values: Dict[str, float], return_value: float):
        """
        记录单只股票的因子值和收益率

        Args:
            stock_code: 股票代码
            factor_values: {factor_name: value, ...}
            return_value: 当日收益率（百分比或小数均可）
        """
        with self._lock:
            self._tracked_stocks.add(stock_code)

            # 初始化因子历史（如果需要）
            for factor_name in factor_values.keys():
                if factor_name not in self._factor_history:
                    self._factor_history[factor_name] = []

            # 追加因子值
            for factor_name, value in factor_values.items():
                if factor_name in self._factor_history:
                    self._factor_history[factor_name].append(value)

            # 追加收益率
            self._return_history.append(return_value)

            # 保持历史长度一致（按最新长度截断）
            min_len = min(len(self._return_history), min(len(h) for h in self._factor_history.values()) if self._factor_history else 0)
            if min_len > 0:
                self._return_history = self._return_history[-min_len:]
                for factor_name in self._factor_history:
                    self._factor_history[factor_name] = self._factor_history[factor_name][-min_len:]

    def force_update(self):
        """强制触发一次权重更新"""
        with self._lock:
            if len(self._return_history) < self.window:
                logger.warning(f"[FactorWeightScheduler] 数据不足 ({len(self._return_history)} < {self.window})，跳过更新")
                return False

            try:
                self.factor_model.update_weights_rolling_ic(
                    factor_history=dict(self._factor_history),
                    returns=list(self._return_history),
                    window=self.window,
                    min_icir=0.03,  # 稍微降低阈值以保留更多因子
                )
                logger.info("[FactorWeightScheduler] 权重更新完成")
                return True
            except Exception as e:
                logger.error(f"[FactorWeightScheduler] 权重更新失败: {e}")
                return False

    def get_status(self) -> Dict:
        """获取调度器状态"""
        with self._lock:
            return {
                'is_running': self._thread is not None and self._thread.is_alive(),
                'tracked_stocks': len(self._tracked_stocks),
                'history_length': len(self._return_history),
                'factor_count': len(self._factor_history),
                'current_weights': dict(self.factor_model.factor_weights) if hasattr(self.factor_model, 'factor_weights') else {},
            }

    def _scheduler_loop(self):
        """后台调度循环"""
        import time
        interval_seconds = self.interval_hours * 3600
        last_update = time.time()

        while not self._stop_event.is_set():
            try:
                current_time = time.time()
                if current_time - last_update >= interval_seconds:
                    logger.info("[FactorWeightScheduler] 触发定期权重更新...")
                    self.force_update()
                    last_update = current_time
            except Exception as e:
                logger.error(f"[FactorWeightScheduler] 调度循环异常: {e}")

            time.sleep(60)  # 每分钟检查一次

    # ── 扩展方法: ICIR 动态权重调度 ─────────────────────────────

    def update_ic(self, factor_name: str, ic_value: float):
        """
        更新因子 IC 历史 (扩展方法)

        Args:
            factor_name: 因子名称
            ic_value: IC 值
        """
        with self._lock:
            if not hasattr(self, '_ic_history'):
                self._ic_history: Dict[str, List[float]] = {}
            if factor_name not in self._ic_history:
                self._ic_history[factor_name] = []
            self._ic_history[factor_name].append(ic_value)
            # 保持最近 60 条
            self._ic_history[factor_name] = self._ic_history[factor_name][-60:]

    def compute_icir(self, factor_name: str, window: int = 20) -> float:
        """
        计算因子 ICIR (IC Information Ratio)

        Args:
            factor_name: 因子名称
            window: 滚动窗口

        Returns:
            ICIR 值
        """
        history = getattr(self, '_ic_history', {}).get(factor_name, [])
        if len(history) < window:
            return 0.0
        recent = history[-window:]
        arr = np.array(recent)
        mean_ic = np.mean(arr)
        std_ic = np.std(arr)
        if std_ic < 1e-10:
            return 0.0
        return float(mean_ic / std_ic)

    def rebalance(self) -> Dict[str, float]:
        """
        基于 ICIR 重新平衡因子权重 (扩展方法)

        Returns:
            新的因子权重
        """
        icirs = self.get_icir_ranking()
        abs_icirs = [abs(icir) for _, icir in icirs]
        total_abs = sum(abs_icirs)

        if total_abs > 0:
            # 按 ICIR 绝对值分配权重
            new_weights = {}
            for factor_name, icir in icirs:
                idx = icirs.index((factor_name, icir))
                weight = abs_icirs[idx] / total_abs
                new_weights[factor_name] = weight
        else:
            # 无 IC 数据时回退到当前权重
            new_weights = dict(getattr(self.factor_model, 'factor_weights', {}))
            total = sum(new_weights.values())
            if total > 0:
                new_weights = {k: v / total for k, v in new_weights.items()}

        # 更新因子模型的权重
        if hasattr(self.factor_model, 'factor_weights'):
            self.factor_model.factor_weights = new_weights

        # 记录重平衡
        if not hasattr(self, '_rebalance_count'):
            self._rebalance_count = 0
        self._rebalance_count += 1
        logger.info(f"[FactorWeightScheduler] 因子权重已重新平衡 (第{self._rebalance_count}次)")

        return new_weights

    def get_icir_ranking(self) -> List[Tuple[str, float]]:
        """
        获取因子 ICIR 排名

        Returns:
            [(factor_name, icir), ...] 按 ICIR 绝对值降序
        """
        icirs = []
        factor_history = getattr(self, '_factor_history', {})
        for factor_name in factor_history:
            icir = self.compute_icir(factor_name)
            icirs.append((factor_name, icir))
        icirs.sort(key=lambda x: abs(x[1]), reverse=True)
        return icirs

    def get_smoothed_weights(self) -> Dict[str, float]:
        """
        获取平滑后的权重 (扩展方法)

        Returns:
            平滑后的因子权重
        """
        current = getattr(self.factor_model, 'factor_weights', {})
        if not hasattr(self, '_previous_weights'):
            self._previous_weights = dict(current)
            self._smoothing_factor = 0.3

        smoothed = {}
        for fname in current:
            prev = self._previous_weights.get(fname, 0)
            cur = current.get(fname, 0)
            smoothed[fname] = (1 - self._smoothing_factor) * prev + self._smoothing_factor * cur
            smoothed[fname] = round(smoothed[fname], 6)

        # 归一化
        total = sum(smoothed.values())
        if total > 0:
            smoothed = {k: v / total for k, v in smoothed.items()}

        # 更新之前的权重
        self._previous_weights = dict(current)

        return smoothed

    def get_status(self) -> Dict:
        """获取调度器状态 (重写，包含扩展信息)"""
        base_status = super().get_status() if hasattr(super(), 'get_status') else {}
        icirs = self.get_icir_ranking()
        return {
            **base_status,
            'current_weights': dict(getattr(self.factor_model, 'factor_weights', {})),
            'smoothed_weights': self.get_smoothed_weights(),
            'icir_ranking': [{'factor': f, 'icir': round(i, 4)} for f, i in icirs[:10]],
            'rebalance_count': getattr(self, '_rebalance_count', 0),
            'drift_detected': getattr(self, '_drift_detected', False),
            'should_rebalance': getattr(self, '_drift_detected', False),
            'smoothing_factor': getattr(self, '_smoothing_factor', 0.3),
        }

    def on_drift_detected(self):
        """当检测到概念漂移时调用 (扩展方法)"""
        self._drift_detected = True
        logger.warning("[FactorWeightScheduler] 检测到概念漂移，将加速权重调整")

    def on_drift_reset(self):
        """当漂移检测重置时调用 (扩展方法)"""
        self._drift_detected = False
