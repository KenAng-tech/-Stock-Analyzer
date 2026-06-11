"""
Strategy Engine V2 - Enhanced Version
Integrated with all optimization modules:
- Kelly Optimizer (real win rate)
- ATR Trend Filter (trend-aware stop loss)
- RSI Multi-Cycle (multi-timeframe RSI)
- Volatility Target (vol-adjusted position sizing)
- MACD + Bollinger Bands
- Sector Rotation
"""

import math
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from modules.kelly_optimizer import KellyOptimizer, kelly_optimizer
from modules.atr_trend_filter import ATR_TrendFilter, atr_trend_filter
from modules.rsi_multicycle import RSI_MultiCycle, rsi_multicycle
from modules.volatility_target import VolatilityTarget, volatility_target
from modules.macd_bollinger import MACD_Bollinger, macd_bollinger
from modules.sector_rotation import SectorRotation, sector_rotation
from modules.hmm_market_detector import MarketRegimeDetector
from modules.factor_orthogonalizer import FactorOrthogonalizer
from modules.transaction_cost_model import TransactionCostModel
from config import config


class StrategyEngineV2:
    """Enhanced Trading strategy generation engine V2"""
    
    def __init__(self):
        # Initialize modules
        self.kelly = kelly_optimizer
        self.atr_filter = atr_trend_filter
        self.rsi = rsi_multicycle
        self.vol_target = volatility_target
        self.macd_bb = macd_bollinger
        self.sector = sector_rotation
        
        # Load strategy settings from config
        strategy_config = config.get('strategy', {})
        self.default_settings = {
            'max_position': strategy_config.get('max_position', 100),
            'min_stop_loss': strategy_config.get('min_stop_loss', 5),
            'max_stop_loss': strategy_config.get('max_stop_loss', 20),
            'take_profit_ratio': strategy_config.get('take_profit_ratio', 2),
            'trailing_stop_pct': strategy_config.get('trailing_stop_pct', 5),
            'atr_multiplier_stop': strategy_config.get('atr_multiplier_stop', 2.0),
            'atr_multiplier_profit': strategy_config.get('atr_multiplier_profit', 3.0),
            'time_stop_days': strategy_config.get('time_stop_days', 60),
            'kelly_max_position': strategy_config.get('kelly_max_position', 0.25),
        }
        
        # Dynamic signal weights
        self.signal_weights = config.get('signal_weights', {
            'support_bounce': 1.2,
            'volume_breakout': 1.0,
            'oversold_recovery': 0.8,
            'profit_taking': 1.3,
            'stop_loss': 1.5,
            'resistance_rejection': 0.9,
            'multi_cycle_resonance': 1.4,
            'trend_following': 1.2,
            'momentum_continuation': 1.1,
        })
        
        # Historical signal performance
        self._signal_history = {}
        
        # Time stop configuration
        self.time_stop_config = {
            'max_hold_days': 90,
            'time_decay_rate': 0.02,
            'profit_time_bonus': 0.01,
            'loss_time_penalty': 0.03
        }
    
    def update_signal_weight(self, signal_type: str, is_win: bool):
        """Bayesian update signal weight"""
        if signal_type not in self._signal_history:
            self._signal_history[signal_type] = {'wins': 0, 'total': 0}
        self._signal_history[signal_type]['total'] += 1
        if is_win:
            self._signal_history[signal_type]['wins'] += 1
        history = self._signal_history[signal_type]
        win_rate = history['wins'] / history['total'] if history['total'] > 0 else 0.5
        base_weight = self.signal_weights.get(signal_type, 1.0)
        self.signal_weights[signal_type] = base_weight * (0.8 + 0.4 * win_rate)
    
    def get_market_state(self, stock_data: Dict) -> Dict:
        """Get market state with enhanced analysis"""
        turnover = stock_data.get('turnover', 100)
        change_pct = stock_data.get('change_pct', 0)
        price = stock_data.get('price', 0)
        year_high = stock_data.get('year_high', 0)
        year_low = stock_data.get('year_low', 0)
        
        # ── 已实现波动率（修复：不再用换手率判断波动率） ────────────
        klines = stock_data.get('klines', [])
        if klines and len(klines) >= 20:
            closes = np.array([k['close'] for k in klines[-20:]], dtype=float)
            log_returns = np.diff(np.log(closes))
            realized_vol = float(np.std(log_returns) * np.sqrt(252) * 100)  # 年化%
            if realized_vol > 40:
                volatility = 'high'
                volatility_factor = 0.7
            elif realized_vol > 25:
                volatility = 'medium'
                volatility_factor = 0.85
            else:
                volatility = 'low'
                volatility_factor = 1.0
        else:
            # 无 K 线数据时的回退
            if turnover > 300:
                volatility = 'high'
                volatility_factor = 0.7
            elif turnover > 200:
                volatility = 'medium'
                volatility_factor = 0.85
            else:
                volatility = 'low'
                volatility_factor = 1.0
        
        # Trend assessment
        if year_high > 0 and year_low > 0:
            price_position = (price - year_low) / (year_high - year_low)
        else:
            price_position = 0.5
        
        if change_pct > 5:
            trend = 'strong_up'
        elif change_pct > 0:
            trend = 'up'
        elif change_pct > -5:
            trend = 'down'
        else:
            trend = 'strong_down'
        
        # Momentum
        if turnover > 250 and change_pct > 3:
            momentum = 'strong'
        elif turnover > 150:
            momentum = 'moderate'
        else:
            momentum = 'weak'
        
        # ADX-based trend strength
        atr = stock_data.get('atr', 0)
        if atr > 0:
            atr_pct = atr / price * 100 if price > 0 else 0
            if atr_pct > 3:
                trend_strength = 'strong'
            elif atr_pct > 2:
                trend_strength = 'moderate'
            else:
                trend_strength = 'weak'
        else:
            trend_strength = 'moderate'
        
        return {
            'volatility': volatility,
            'volatility_factor': volatility_factor,
            'trend': trend,
            'price_position': round(price_position, 3),
            'momentum': momentum,
            'trend_strength': trend_strength,
            'turnover': turnover,
            'change_pct': change_pct
        }
    
    def generate_entry_signals(self, stock_data: Dict, analysis: Dict) -> List[Dict]:
        """Generate entry signals with enhanced analysis"""
        signals = []
        price = stock_data.get('price', 0)
        change_pct = stock_data.get('change_pct', 0)
        turnover = stock_data.get('turnover', 100)
        year_high = stock_data.get('year_high', 0)
        year_low = stock_data.get('year_low', 0)
        
        # ATR-based entry
        atr = stock_data.get('atr', 0)
        if atr > 0:
            atr_stop = price - atr * 2.0
            atr_profit = price + atr * 3.0
            signals.append({
                'type': 'BUY',
                'condition': 'ATR支撑位买入',
                'price_range': f'{atr_stop:.0f}-{price:.0f}',
                'strength': '强',
                'weight': self.signal_weights['volume_breakout'],
                'confidence': 0.75,
                'description': f'ATR={atr:.2f}，支撑位{atr_stop:.0f}元'
            })
        
        # RSI-based entry
        rsi_14 = stock_data.get('rsi_14', 50)
        if rsi_14 < 35:
            signals.append({
                'type': 'BUY',
                'condition': 'RSI超卖买入',
                'price_range': f'<{price:.0f}',
                'strength': '强',
                'weight': self.signal_weights['oversold_recovery'],
                'confidence': 0.8,
                'description': f'RSI={rsi_14:.1f}，超卖区域'
            })
        
        # Volume breakout
        if turnover > 250 and change_pct > 3:
            signals.append({
                'type': 'BUY',
                'condition': '放量突破',
                'price_range': f'>={price:.0f}',
                'strength': '强',
                'weight': self.signal_weights['volume_breakout'],
                'confidence': 0.75,
                'description': f'换手率{turnover:.0f}%，放量上涨'
            })
        
        # Support bounce
        if year_low > 0 and price < year_low * 1.05:
            signals.append({
                'type': 'BUY',
                'condition': '支撑位反弹',
                'price_range': f'{year_low:.0f}-{price:.0f}',
                'strength': '中',
                'weight': self.signal_weights['support_bounce'],
                'confidence': 0.7,
                'description': f'接近年低{year_low:.0f}元'
            })
        
        # Kelly-based entry
        kelly_position = self.kelly.calculate_kelly_fraction()
        if kelly_position > 0.15:
            signals.append({
                'type': 'BUY',
                'condition': 'Kelly仓位建议买入',
                'price_range': f'<{price:.0f}',
                'strength': '中',
                'weight': self.signal_weights['trend_following'],
                'confidence': 0.7,
                'description': f'Kelly建议仓位{kelly_position:.0%}'
            })
        
        # MACD golden cross
        macd_hist = stock_data.get('macd_histogram', 0)
        if macd_hist > 0:
            signals.append({
                'type': 'BUY',
                'condition': 'MACD多头',
                'price_range': f'>={price:.0f}',
                'strength': '中',
                'weight': self.signal_weights['momentum_continuation'],
                'confidence': 0.65,
                'description': 'MACD柱状图为正'
            })
        
        return signals
    
    def generate_exit_signals(self, stock_data: Dict, analysis: Dict,
                               cost_basis: float = 120,
                               entry_date: Optional[str] = None) -> List[Dict]:
        """Generate exit signals with enhanced analysis"""
        signals = []
        price = stock_data.get('price', 0)
        change_pct = stock_data.get('change_pct', 0)
        profit_pct = ((price - cost_basis) / cost_basis * 100) if cost_basis > 0 else 0
        
        # ATR-based exit
        atr = stock_data.get('atr', 0)
        if atr > 0:
            atr_stop = price - atr * 2.0
            signals.append({
                'type': 'SELL',
                'condition': 'ATR止损',
                'price_range': f'<{atr_stop:.0f}',
                'strength': '强',
                'weight': self.signal_weights['stop_loss'],
                'confidence': 0.8,
                'description': f'ATR止损位{atr_stop:.0f}元'
            })
        
        # Time stop
        time_stop = self.calculate_time_stop(stock_data, entry_date)
        if time_stop['time_signal'] == 'time_stop_triggered':
            signals.append({
                'type': 'SELL',
                'condition': '时间止损',
                'price_range': f'<{time_stop["time_stop_price"]:.0f}',
                'strength': '强',
                'weight': self.signal_weights['stop_loss'],
                'confidence': 0.7,
                'description': f'持仓{time_stop["days_held"]}天超过时间止损线'
            })
        
        # Profit taking
        if profit_pct > 100 and change_pct < -5:
            signals.append({
                'type': 'SELL',
                'condition': '获利了结',
                'price_range': f'<{price:.0f}',
                'strength': '强',
                'weight': self.signal_weights['profit_taking'],
                'confidence': 0.8,
                'description': '盈利超100%且回调明显，建议减仓'
            })
        
        # Resistance rejection
        position = analysis.get('position', {})
        if position.get('position') == '接近年高' and change_pct > 0:
            signals.append({
                'type': 'SELL',
                'condition': '阻力位回落',
                'price_range': f'<{price:.0f}',
                'strength': '中',
                'weight': self.signal_weights['resistance_rejection'],
                'confidence': 0.65,
                'description': '接近年高阻力位，可能回落'
            })
        
        # RSI overbought
        rsi_14 = stock_data.get('rsi_14', 50)
        if rsi_14 > 75:
            signals.append({
                'type': 'SELL',
                'condition': 'RSI超买',
                'price_range': f'<{price:.0f}',
                'strength': '中',
                'weight': self.signal_weights['oversold_recovery'],
                'confidence': 0.7,
                'description': f'RSI={rsi_14:.1f}，超买区域'
            })
        
        return signals
    
    def calculate_time_stop(self, stock_data: Dict, entry_date: Optional[str] = None) -> Dict:
        """Calculate time stop loss"""
        if entry_date is None:
            entry_date = datetime.now().strftime('%Y-%m-%d')
        
        try:
            entry_dt = datetime.strptime(entry_date, '%Y-%m-%d')
            days_held = (datetime.now() - entry_dt).days
        except:
            days_held = 0
        
        max_hold = self.time_stop_config['max_hold_days']
        profit_pct = stock_data.get('change_pct', 0)
        
        if days_held > max_hold:
            time_signal = 'time_stop_triggered'
            time_stop_price = stock_data.get('price', 0) * 0.95
        elif days_held > max_hold * 0.8:
            time_signal = 'approaching_time_stop'
            time_stop_price = stock_data.get('price', 0)
        else:
            time_signal = 'no_time_stop'
            time_stop_price = stock_data.get('price', 0)
        
        return {
            'days_held': days_held,
            'max_hold_days': max_hold,
            'time_signal': time_signal,
            'time_stop_price': time_stop_price
        }
    
    def calculate_dynamic_kelly_position(self, stock_data: Dict,
                                          win_rate: float = None,
                                          win_loss_ratio: float = None,
                                          total_capital: float = 100000) -> Dict:
        """Calculate dynamic Kelly position"""
        kelly_result = self.kelly.get_position_size(stock_data, total_capital)
        return kelly_result
    
    def generate_strategy_recommendation(self, stock_data: Dict, analysis: Dict,
                                          cost_basis: float = 120,
                                          total_capital: float = 100000,
                                          entry_date: Optional[str] = None) -> Dict:
        """Generate comprehensive strategy recommendation"""
        entry_signals = self.generate_entry_signals(stock_data, analysis)
        exit_signals = self.generate_exit_signals(stock_data, analysis, cost_basis, entry_date)
        
        # Calculate position size
        kelly_position = self.calculate_dynamic_kelly_position(
            stock_data, total_capital=total_capital
        )
        
        # Market state
        market_state = self.get_market_state(stock_data)
        
        # Overall recommendation
        profit_pct = ((stock_data.get('price', 0) - cost_basis) / cost_basis * 100) if cost_basis > 0 else 0
        change_pct = stock_data.get('change_pct', 0)
        
        if profit_pct > 100 and change_pct < -5:
            recommendation = '持有减仓'
            action = '建议减仓30-50%，保留底仓观望'
        elif profit_pct > 100 and change_pct > 0:
            recommendation = '持有加仓'
            action = '趋势向上，可考虑加仓20%'
        elif change_pct < -8:
            recommendation = '逢低买入'
            action = '超跌机会，可分批建仓'
        else:
            recommendation = '持有观望'
            action = '保持现有仓位，等待方向明确'
        
        # Signal strength
        total_weight = sum(s.get('weight', 1.0) for s in entry_signals)
        avg_confidence = sum(s.get('confidence', 0.5) for s in entry_signals) / len(entry_signals) if entry_signals else 0.5
        
        # Time stop
        time_stop_info = self.calculate_time_stop(stock_data, entry_date)
        
        # CVaR risk — 使用真实历史收益率（修复：不再用硬编码假数据）
        klines = stock_data.get('klines', [])
        if klines and len(klines) >= 30:
            closes = np.array([k['close'] for k in klines], dtype=float)
            daily_returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
            cvar_risk = self.calculate_cvar_risk(daily_returns[-60:])  # 最近 60 天
        else:
            cvar_risk = self.calculate_cvar_risk([])
        
        return {
            'recommendation': recommendation,
            'action': action,
            'entry_signals': entry_signals,
            'exit_signals': exit_signals,
            'position_size': kelly_position,
            'risk_level': '中高' if stock_data.get('turnover', 0) > 200 else '中',
            'time_horizon': '1-3个月',
            'market_state': market_state,
            'signal_strength': round(avg_confidence, 2),
            'total_signal_weight': round(total_weight, 2),
            'time_stop': time_stop_info,
            'cvar_risk': cvar_risk
        }
    
    def calculate_cvar_risk(self, returns: List[float], confidence: float = 0.95) -> Dict:
        """Calculate CVaR risk"""
        if not returns:
            return {'cvar': 0.05, 'var': 0.03}
        sorted_returns = sorted(returns)
        cutoff = int(len(sorted_returns) * (1 - confidence))
        cvar_returns = sorted_returns[:cutoff]
        cvar = abs(np.mean(cvar_returns)) if cvar_returns else 0.05
        var = abs(sorted_returns[0]) if sorted_returns else 0.03
        return {'cvar': round(cvar, 4), 'var': round(var, 4)}


    def _default_strategy(self, bar: Dict, position: int, capital: float,
                          params: Optional[Dict] = None) -> str:
        """Default strategy function for backtesting"""
        if params is None:
            params = self.default_settings
        price = bar['close']
        rsi = bar.get('rsi', 50)
        macd = bar.get('macd', 0)
        if position == 0 and rsi < 35 and macd > 0:
            return 'buy'
        elif position > 0 and rsi > 65 and macd < 0:
            return 'sell'
        return 'hold'


# Global instance
strategy_engine_v2 = StrategyEngineV2()
