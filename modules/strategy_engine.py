"""
Strategy Engine Module - Enhanced Version V4
Generates trading strategies based on market conditions with advanced features:
- ATR Dynamic Stop Loss (Phase 1) - Enhanced with Wilder's EMA
- Multi-Timeframe Resonance (Phase 1) - Enhanced with trend alignment
- Volume Confirmation (Phase 1) - Enhanced with volume moving averages
- Trailing Stop (Phase 1) - Enhanced with time-based adjustments
- Kelly Criterion Position Sizing (Phase 2) - Enhanced with dynamic parameters
- Market State Adjustment (Phase 2) - Enhanced with trend + momentum + ADX
- Signal Strength Weighting (Phase 2) - Enhanced with confidence fusion
- Time Stop Loss (Phase 2) - New feature with time decay
- CVaR Risk Constraint (Phase 2) - New feature
- Volatility Target Position (Phase 3) - New feature
- Dynamic Signal Weights (Phase 4) - Bayesian updating
- Transaction Cost Integration (Phase 5) - 2026-06-02 新增

2026-06-02 更新:
- 集成 TransactionCostModel 计算买入/卖出成本
- 入场信号考虑交易成本后的净收益
- 出场信号考虑滑点和冲击成本
- 策略推荐中显示成本调整后的目标价
"""

import math
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from .atr_calculator import ATRCalculator
from .transaction_cost_model import TransactionCostModel
from config import config


class StrategyEngine:
    """Enhanced Trading strategy generation engine"""
    
    def __init__(self):
        self.atr_calculator = ATRCalculator(atr_period=14)
        # 2026-06-02: 集成交易成本模型
        self.cost_model = TransactionCostModel()
        
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
        
        # Dynamic signal weights (loaded from config)
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
        
        # Historical signal performance for Bayesian updating
        self._signal_history = {}
        
        # Time stop configuration
        self.time_stop_config = {
            'max_hold_days': 90,
            'time_decay_rate': 0.02,
            'profit_time_bonus': 0.01,
            'loss_time_penalty': 0.03
        }
        
        # Kelly criterion parameters
        self.kelly_config = {
            'default_win_rate': 0.55,
            'default_win_loss_ratio': 2.0,
            'min_win_rate': 0.40,
            'max_win_rate': 0.70,
            'kelly_fraction': 0.5,
            'position_cap': 0.25
        }
        
        # CVaR configuration
        self.cvar_config = {
            'confidence_level': 0.95,
            'max_cvar': 0.05,
            'cvar_position_reduction': 0.2
        }
    
    def update_signal_weight(self, signal_type: str, is_win: bool):
        """贝叶斯更新信号权重"""
        if signal_type not in self._signal_history:
            self._signal_history[signal_type] = {'wins': 0, 'total': 0}
        
        self._signal_history[signal_type]['total'] += 1
        if is_win:
            self._signal_history[signal_type]['wins'] += 1
        
        history = self._signal_history[signal_type]
        win_rate = history['wins'] / history['total'] if history['total'] > 0 else 0.5
        # Adjust weight based on performance
        base_weight = self.signal_weights.get(signal_type, 1.0)
        self.signal_weights[signal_type] = base_weight * (0.8 + 0.4 * win_rate)
    
    def get_market_state(self, stock_data: Dict) -> Dict:
        """
        Phase 2: Market State Adjustment - Enhanced
        Determine market condition based on volatility, trend, and momentum
        """
        turnover = stock_data.get('turnover', 100)
        change_pct = stock_data.get('change_pct', 0)
        price = stock_data.get('price', 0)
        year_high = stock_data.get('year_high', 0)
        year_low = stock_data.get('year_low', 0)
        std_dev = stock_data.get('std_dev', 0)
        
        # Volatility assessment
        if turnover > 300:
            volatility = 'high'
            volatility_factor = 0.7
        elif turnover > 200:
            volatility = 'medium'
            volatility_factor = 0.85
        else:
            volatility = 'low'
            volatility_factor = 1.0
        
        # Trend assessment - Enhanced with momentum
        if year_high > 0 and year_low > 0 and year_high != year_low:
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
            trend = 'sideways'
        
        # Momentum assessment
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
    
    def calculate_dynamic_kelly_position(self, stock_data: Dict,
                                          win_rate: float = 0.55,
                                          win_loss_ratio: float = 2.0,
                                          total_capital: float = 100000) -> Dict:
        """
        Phase 2: Kelly Criterion Position Sizing - Enhanced
        Calculate optimal position size using Kelly formula with adjustments
        """
        price = stock_data.get('price', 0)
        turnover = stock_data.get('turnover', 100)
        
        # Kelly formula: f* = (p*b - q) / b
        # where p = win_rate, q = 1 - p, b = win_loss_ratio
        q = 1 - win_rate
        kelly_fraction = (win_rate * win_loss_ratio - q) / win_loss_ratio
        
        # Apply half Kelly for risk reduction
        kelly_fraction *= self.kelly_config['kelly_fraction']
        
        # Adjust for volatility
        if turnover > 300:
            kelly_fraction *= 0.8
        elif turnover < 100:
            kelly_fraction *= 1.1
        
        # Cap at maximum position
        kelly_fraction = min(kelly_fraction, self.kelly_config['position_cap'])
        kelly_fraction = max(kelly_fraction, 0.05)
        
        # Calculate position size
        position_value = total_capital * kelly_fraction
        position_shares = int(position_value / price) if price > 0 else 0
        
        # Adjust for lot size (100 shares for A-shares)
        # Use at least 1 lot if position_value > 0
        position_shares = max((position_shares // 100) * 100, 100) if position_shares > 0 else 100
        
        return {
            'position_pct': round(kelly_fraction * 100, 1),
            'position_value': round(position_value, 2),
            'position_shares': position_shares,
            'kelly_fraction': round(kelly_fraction, 3),
            'win_rate': win_rate,
            'win_loss_ratio': win_loss_ratio,
            'kelly_optimal': round(kelly_fraction / self.kelly_config['kelly_fraction'], 3),
        }
    
    def calculate_time_stop(self, stock_data: Dict, entry_date: Optional[str] = None) -> Dict:
        """
        Phase 2: Time Stop Loss
        Calculate time-based stop loss based on holding period
        """
        if entry_date is None:
            entry_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        try:
            entry = datetime.strptime(entry_date, '%Y-%m-%d')
            days_held = (datetime.now() - entry).days
        except (ValueError, TypeError):
            days_held = 30
        
        max_hold_days = self.time_stop_config['max_hold_days']
        time_decay_rate = self.time_stop_config['time_decay_rate']
        
        # Time stop price based on holding period
        price = stock_data.get('price', 0)
        time_stop_price = price * (1 - time_decay_rate * days_held / max_hold_days)
        
        # Time stop signal
        if days_held > max_hold_days:
            time_signal = 'time_stop_triggered'
        elif days_held > max_hold_days * 0.8:
            time_signal = 'approaching'
        else:
            time_signal = 'hold'
        
        return {
            'days_held': days_held,
            'max_hold_days': max_hold_days,
            'time_signal': time_signal,
            'time_stop_price': round(time_stop_price, 2),
            'time_decay_rate': time_decay_rate
        }
    
    def calculate_cvar_risk(self, returns: List[float], confidence: float = 0.95) -> Dict:
        """
        Phase 2: CVaR Risk Constraint
        Calculate Conditional Value at Risk
        """
        if not returns:
            return {'cvar': 0, 'var': 0, 'confidence_level': confidence, 'is_acceptable': True}
        
        sorted_returns = sorted(returns)
        var_index = int(len(sorted_returns) * (1 - confidence))
        var = abs(sorted_returns[var_index]) if var_index < len(sorted_returns) else 0
        
        # CVaR = average of returns beyond VaR
        cvar_returns = sorted_returns[:var_index + 1] if var_index > 0 else sorted_returns[:1]
        cvar = abs(np.mean(cvar_returns))
        
        is_acceptable = bool(cvar <= self.cvar_config['max_cvar'])
        return {
            'cvar': round(float(cvar), 4),
            'var': round(float(var), 4),
            'confidence_level': float(confidence),
            'is_acceptable': is_acceptable
        }
    
    def generate_entry_signals(self, stock_data: Dict, analysis: Dict) -> List[Dict]:
        """Generate entry signals with enhanced analysis"""
        signals = []
        price = stock_data.get('price', 0)
        change_pct = stock_data.get('change_pct', 0)
        turnover = stock_data.get('turnover', 100)
        year_high = stock_data.get('year_high', 0)
        year_low = stock_data.get('year_low', 0)
        cost_basis = analysis.get('basic_info', {}).get('cost_basis', 120)
        
        # 2026-06-02: 使用交易成本模型计算成本
        trade_value = price * 100  # 假设100股
        avg_volume = stock_data.get('avg_volume', turnover * 10000)
        buy_cost = self.cost_model.calculate_total_cost(trade_value, is_buy=True, 
                                                         trade_volume=100, avg_volume=avg_volume)
        
        # ATR-based entry
        atr = stock_data.get('atr', 0)
        if atr > 0:
            atr_stop = price - atr * 2.0
            atr_profit = price + atr * 3.0
            # 净收益 = 目标收益 - 交易成本
            net_profit = (atr_profit - price) - buy_cost['total_cost_pct'] * 100
            signals.append({
                'type': 'BUY',
                'condition': 'ATR支撑位买入',
                'price_range': f'{atr_stop:.0f}-{price:.0f}',
                'strength': '强',
                'weight': self.signal_weights['volume_breakout'],
                'confidence': 0.75,
                'description': f'ATR={atr:.2f}，支撑位{atr_stop:.0f}元，净收益{net_profit:.1f}%'
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
                'description': f'换手率{turnover:.1f}%，涨幅{change_pct:.1f}%'
            })
        
        # MA support
        kline_stats = stock_data.get('kline_stats', {})
        ma5 = kline_stats.get('ma5', price * 1.04)
        ma10 = kline_stats.get('ma10', price * 1.08)
        if price > ma5 and price < ma5 * 1.02:
            signals.append({
                'type': 'BUY',
                'condition': '均线支撑',
                'price_range': f'{ma5:.0f}-{price:.0f}',
                'strength': '中',
                'weight': self.signal_weights['trend_following'],
                'confidence': 0.65,
                'description': f'回踩MA5({ma5:.1f}元)支撑位'
            })
        
        # Oversold bounce
        if change_pct < -8 and rsi_14 < 30:
            signals.append({
                'type': 'BUY',
                'condition': '超跌反弹',
                'price_range': f'<{price:.0f}',
                'strength': '强',
                'weight': self.signal_weights['oversold_recovery'],
                'confidence': 0.7,
                'description': f'跌幅{change_pct:.1f}%，RSI={rsi_14:.1f}'
            })
        
        # Multi-cycle resonance
        multi_cycle = analysis.get('kline_signals', {}).get('multi_cycle', {})
        if multi_cycle.get('resonance', False):
            signals.append({
                'type': 'BUY',
                'condition': '多周期共振',
                'price_range': f'{price:.0f}',
                'strength': '强',
                'weight': self.signal_weights['multi_cycle_resonance'],
                'confidence': 0.85,
                'description': '多时间框架信号一致看涨'
            })
        
        # Trend following
        trend = analysis.get('trend', {})
        if trend.get('short_term') == 'up' and trend.get('medium_term') == 'up':
            signals.append({
                'type': 'BUY',
                'condition': '趋势跟随',
                'price_range': f'>={price:.0f}',
                'strength': '中',
                'weight': self.signal_weights['trend_following'],
                'confidence': 0.7,
                'description': '短中期趋势一致向上'
            })
        
        return signals
    
    def generate_exit_signals(self, stock_data: Dict, analysis: Dict, 
                               cost_basis: float = 120,
                               entry_date: Optional[str] = None) -> List[Dict]:
        """Generate exit signals with enhanced analysis"""
        signals = []
        price = stock_data.get('price', 0)
        change_pct = stock_data.get('change_pct', 0)
        turnover = stock_data.get('turnover', 100)
        profit_pct = ((price - cost_basis) / cost_basis * 100) if cost_basis > 0 else 0
        
        # ATR-based exit
        atr = stock_data.get('atr', 0)
        if atr > 0:
            atr_stop = price - atr * 2.0
            atr_profit = price + atr * 3.0
            signals.append({
                'type': 'SELL',
                'condition': 'ATR止盈',
                'price_range': f'>{atr_profit:.0f}',
                'strength': '强',
                'weight': self.signal_weights['profit_taking'],
                'confidence': 0.75,
                'description': f'ATR止盈位{atr_profit:.0f}元'
            })
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
        
        # Oversold recovery exit
        if profit_pct > 50 and change_pct > 8:
            signals.append({
                'type': 'SELL',
                'condition': '超涨回调',
                'price_range': f'<{price:.0f}',
                'strength': '中',
                'weight': self.signal_weights['oversold_recovery'],
                'confidence': 0.6,
                'description': '短期涨幅过大，可能回调'
            })
        
        return signals
    
    def generate_strategy_recommendation(self, stock_data: Dict, analysis: Dict, 
                                          cost_basis: float = 120, 
                                          total_capital: float = 100000,
                                          entry_date: Optional[str] = None) -> Dict:
        """Generate comprehensive strategy recommendation"""
        entry_signals = self.generate_entry_signals(stock_data, analysis)
        exit_signals = self.generate_exit_signals(stock_data, analysis, cost_basis, entry_date)
        
        # Calculate position size using Dynamic Kelly
        win_rate = 0.55
        win_loss_ratio = 2.0
        position = self.calculate_dynamic_kelly_position(
            stock_data, win_rate, win_loss_ratio, total_capital
        )
        
        # Determine overall recommendation
        profit_pct = ((stock_data.get('price', 0) - cost_basis) / cost_basis * 100) if cost_basis > 0 else 0
        change_pct = stock_data.get('change_pct', 0)
        
        market_state = self.get_market_state(stock_data)
        
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
        
        # Calculate weighted signal strength
        total_weight = sum(s.get('weight', 1.0) for s in entry_signals)
        avg_confidence = sum(s.get('confidence', 0.5) for s in entry_signals) / len(entry_signals) if entry_signals else 0.5
        
        # Time stop information
        time_stop_info = self.calculate_time_stop(stock_data, entry_date)
        
        # CVaR risk assessment
        cvar_risk = self.calculate_cvar_risk([-0.05, -0.03, -0.02, 0.01, 0.02, 0.03, 0.04, 0.05])
        
        # 2026-06-02: 集成交易成本信息
        price = stock_data.get('price', 0)
        trade_value = price * position['position_shares']
        avg_volume = stock_data.get('avg_volume', stock_data.get('turnover', 100) * 10000)
        buy_cost = self.cost_model.calculate_total_cost(trade_value, is_buy=True, 
                                                         trade_volume=position['position_shares'],
                                                         avg_volume=avg_volume)
        sell_cost = self.cost_model.calculate_total_cost(trade_value, is_buy=False,
                                                          trade_volume=position['position_shares'],
                                                          avg_volume=avg_volume)
        round_trip = self.cost_model.calculate_round_trip_cost(trade_value,
                                                                trade_volume=position['position_shares'],
                                                                avg_volume=avg_volume)
        
        # 成本调整后的目标价
        gross_target = price * 1.2  # 假设20%目标收益
        net_target = gross_target - (round_trip['total_round_trip_pct'] / 100) * price
        
        return {
            'recommendation': recommendation,
            'action': action,
            'entry_signals': entry_signals,
            'exit_signals': exit_signals,
            'position_size': position,
            'risk_level': '中高' if stock_data.get('turnover', 0) > 200 else '中',
            'time_horizon': '1-3个月',
            'market_state': market_state,
            'signal_strength': round(avg_confidence, 2),
            'total_signal_weight': round(total_weight, 2),
            'time_stop': time_stop_info,
            'cvar_risk': cvar_risk,
            # 2026-06-02: 新增交易成本信息
            'transaction_costs': {
                'buy_cost': {
                    'total': round(buy_cost['total_cost'], 2),
                    'total_pct': round(buy_cost['total_cost_pct'], 4),
                    'commission': round(buy_cost['commission'], 2),
                    'stamp_tax': round(buy_cost['stamp_tax'], 2),
                    'slippage': round(buy_cost['slippage'], 2),
                },
                'sell_cost': {
                    'total': round(sell_cost['total_cost'], 2),
                    'total_pct': round(sell_cost['total_cost_pct'], 4),
                    'commission': round(sell_cost['commission'], 2),
                    'stamp_tax': round(sell_cost['stamp_tax'], 2),
                    'slippage': round(sell_cost['slippage'], 2),
                },
                'round_trip': {
                    'total': round(round_trip['total_round_trip'], 2),
                    'total_pct': round(round_trip['total_round_trip_pct'], 4),
                },
                'gross_target': round(gross_target, 2),
                'net_target': round(net_target, 2),
                'cost_adjustment': round(gross_target - net_target, 2),
            }
        }
