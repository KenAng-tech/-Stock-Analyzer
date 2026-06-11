"""
Transaction Cost Model Module - Advanced Optimization
Models trading costs including commission, stamp tax, slippage,
and market impact
"""

import math
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta


class TransactionCostModel:
    """Comprehensive transaction cost model for Chinese A-shares"""
    
    def __init__(self):
        # Commission rate (broker-dependent, typically 0.025%-0.03%)
        self.commission_rate = 0.0003
        
        # Stamp tax (only on sell, 0.1%)
        self.stamp_tax = 0.001
        
        # Slippage (basis points)
        self.slippage_bps = 2
        
        # Market impact coefficient
        self.market_impact_coeff = 0.0001
        
        # Minimum commission (5 RMB)
        self.min_commission = 5
        
        # Price limit (10% for most stocks, 20% for ChiNext/STAR)
        self.price_limit = 0.10
        
        # Trading hour fractions for liquidity adjustment
        self.trading_hours = {
            'morning_open': {'start': 9.30, 'end': 10.00, 'liquidity_factor': 1.2},
            'morning_close': {'start': 11.00, 'end': 11.30, 'liquidity_factor': 1.1},
            'afternoon_open': {'start': 13.00, 'end': 13.30, 'liquidity_factor': 1.3},
            'afternoon_close': {'start': 14.30, 'end': 15.00, 'liquidity_factor': 1.4}
        }
    
    def calculate_commission(self, trade_value: float, is_buy: bool = True) -> float:
        """
        Calculate trading commission
        
        Args:
            trade_value: Total value of the trade
            is_buy: True for buy, False for sell
        
        Returns:
            Commission amount in RMB
        """
        commission = trade_value * self.commission_rate
        return max(commission, self.min_commission)
    
    def calculate_stamp_tax(self, trade_value: float, is_buy: bool = True) -> float:
        """
        Calculate stamp tax (only on sell)
        
        Args:
            trade_value: Total value of the trade
            is_buy: True for buy, False for sell
        
        Returns:
            Stamp tax amount in RMB
        """
        if is_buy:
            return 0
        return trade_value * self.stamp_tax
    
    def calculate_slippage(self, trade_value: float, 
                           trade_volume: float,
                           avg_volume: float = 1000000) -> float:
        """
        Calculate slippage cost
        
        Slippage increases with trade size relative to average volume
        
        Args:
            trade_value: Total value of the trade
            trade_volume: Number of shares traded
            avg_volume: Average daily volume
        
        Returns:
            Slippage cost in RMB
        """
        # Volume ratio affects slippage
        volume_ratio = trade_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Slippage increases with volume ratio (diminishing returns)
        slippage_factor = 1 + math.log(1 + volume_ratio) / math.log(2)
        
        slippage = trade_value * self.slippage_bps / 10000 * slippage_factor
        
        return slippage
    
    def calculate_market_impact(self, trade_value: float,
                                 trade_volume: float,
                                avg_volume: float = 1000000,
                                volatility: float = 0.02) -> float:
        """
        Calculate market impact cost
        
        Market impact depends on trade size and market volatility
        
        Args:
            trade_value: Total value of the trade
            trade_volume: Number of shares traded
            avg_volume: Average daily volume
            volatility: Market volatility
        
        Returns:
            Market impact cost in RMB
        """
        # Square root law of market impact
        volume_ratio = trade_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Market impact = coefficient * sqrt(volume_ratio) * volatility * trade_value
        market_impact = self.market_impact_coeff * \
                       math.sqrt(volume_ratio) * \
                       volatility * \
                       trade_value
        
        return market_impact
    
    def calculate_total_cost(self, trade_value: float, 
                              is_buy: bool = True,
                              trade_volume: float = 0,
                              avg_volume: float = 1000000,
                              volatility: float = 0.02,
                              current_time: Optional[datetime] = None) -> Dict:
        """
        Calculate total transaction cost breakdown
        
        Args:
            trade_value: Total value of the trade
            is_buy: True for buy, False for sell
            trade_volume: Number of shares traded
            avg_volume: Average daily volume
            volatility: Market volatility
            current_time: Current time for liquidity adjustment
        
        Returns:
            Dictionary with cost breakdown
        """
        # Calculate individual costs
        commission = self.calculate_commission(trade_value, is_buy)
        stamp_tax = self.calculate_stamp_tax(trade_value, is_buy)
        slippage = self.calculate_slippage(trade_value, trade_volume, avg_volume)
        market_impact = self.calculate_market_impact(
            trade_value, trade_volume, avg_volume, volatility
        )
        
        # Liquidity adjustment based on trading time
        liquidity_factor = 1.0
        if current_time:
            hour = current_time.hour + current_time.minute / 60.0
            for period, config in self.trading_hours.items():
                if config['start'] <= hour <= config['end']:
                    liquidity_factor = config['liquidity_factor']
                    break
        
        # Apply liquidity factor to variable costs
        slippage *= liquidity_factor
        market_impact *= liquidity_factor
        
        # Total cost
        total_cost = commission + stamp_tax + slippage + market_impact
        
        # Cost as percentage of trade value
        total_cost_pct = total_cost / trade_value * 100 if trade_value > 0 else 0
        
        return {
            'commission': round(commission, 2),
            'stamp_tax': round(stamp_tax, 2),
            'slippage': round(slippage, 2),
            'market_impact': round(market_impact, 2),
            'total_cost': round(total_cost, 2),
            'total_cost_pct': round(total_cost_pct, 4),
            'liquidity_factor': round(liquidity_factor, 2),
            'is_buy': is_buy
        }
    
    def calculate_round_trip_cost(self, trade_value: float,
                                   trade_volume: float = 0,
                                   avg_volume: float = 1000000,
                                   volatility: float = 0.02) -> Dict:
        """
        Calculate total cost for buy and sell round trip
        
        Args:
            trade_value: Total value of the trade
            trade_volume: Number of shares traded
            avg_volume: Average daily volume
            volatility: Market volatility
        
        Returns:
            Dictionary with round trip cost breakdown
        """
        buy_cost = self.calculate_total_cost(
            trade_value, is_buy=True,
            trade_volume=trade_volume,
            avg_volume=avg_volume,
            volatility=volatility
        )
        
        sell_cost = self.calculate_total_cost(
            trade_value, is_buy=False,
            trade_volume=trade_volume,
            avg_volume=avg_volume,
            volatility=volatility
        )
        
        return {
            'buy_cost': buy_cost,
            'sell_cost': sell_cost,
            'total_round_trip': round(buy_cost['total_cost'] + sell_cost['total_cost'], 2),
            'total_round_trip_pct': round(
                (buy_cost['total_cost_pct'] + sell_cost['total_cost_pct']), 4
            )
        }
    
    def calculate_net_return_after_costs(self, gross_return: float,
                                          trade_value: float,
                                          is_buy: bool = True,
                                          trade_volume: float = 0,
                                          avg_volume: float = 1000000) -> float:
        """
        Calculate net return after transaction costs
        
        Args:
            gross_return: Gross return (as decimal)
            trade_value: Total value of the trade
            is_buy: True for buy (costs reduce return), False for sell
            trade_volume: Number of shares traded
            avg_volume: Average daily volume
        
        Returns:
            Net return after costs (as decimal)
        """
        total_cost = self.calculate_total_cost(
            trade_value, is_buy=is_buy,
            trade_volume=trade_volume,
            avg_volume=avg_volume
        )
        
        # Cost as fraction of trade value
        cost_fraction = total_cost['total_cost_pct'] / 100
        
        if is_buy:
            # Buy costs reduce return
            net_return = gross_return - cost_fraction
        else:
            # Sell costs reduce return
            net_return = gross_return - cost_fraction
        
        return net_return
    
    def estimate_optimal_trade_size(self, max_cost_pct: float = 0.1,
                                     avg_volume: float = 1000000,
                                     price: float = 100) -> float:
        """
        Estimate optimal trade size to minimize market impact
        
        Args:
            max_cost_pct: Maximum acceptable cost as percentage
            avg_volume: Average daily volume
            price: Current stock price
        
        Returns:
            Optimal trade volume in shares
        """
        # Market impact = coeff * sqrt(V/avg_V) * P * V
        # Where V is trade volume, P is price
        
        # Solve for V where market impact = max_cost_pct
        # market_impact = coeff * sqrt(V/avg_V) * P * V = max_cost_pct * P * V
        # coeff * sqrt(V/avg_V) = max_cost_pct
        # V = avg_V * (max_cost_pct / coeff)^2
        
        optimal_volume = avg_volume * (max_cost_pct / self.market_impact_coeff) ** 2
        
        return min(optimal_volume, avg_volume * 0.1)  # Cap at 10% of daily volume
    
    def get_cost_breakdown_summary(self, trade_value: float,
                                    is_buy: bool = True) -> str:
        """
        Get human-readable cost breakdown summary
        
        Args:
            trade_value: Total value of the trade
            is_buy: True for buy, False for sell
        
        Returns:
            Formatted cost summary string
        """
        cost = self.calculate_total_cost(trade_value, is_buy=is_buy)
        
        summary = f"""
Transaction Cost Breakdown:
├── Commission: ¥{cost['commission']:,.2f} ({cost['commission']/trade_value*100:.3f}%)
├── Stamp Tax: ¥{cost['stamp_tax']:,.2f} ({cost['stamp_tax']/trade_value*100:.3f}%)
├── Slippage: ¥{cost['slippage']:,.2f} ({cost['slippage']/trade_value*100:.3f}%)
├── Market Impact: ¥{cost['market_impact']:,.2f} ({cost['market_impact']/trade_value*100:.3f}%)
└── Total: ¥{cost['total_cost']:,.2f} ({cost['total_cost_pct']:.3f}%)
        """
        
        return summary
