#!/usr/bin/env python3
"""
Real-time Monitor Module
Provides price alerts and signal pushing functionality
"""

import sys
import os
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.strategy_engine import StrategyEngine
from modules.kline_signal_analyzer import KlineSignalAnalyzer
from modules.atr_calculator import ATRCalculator, ADXCalculator
from modules.hmm_market_detector import MarketRegimeDetector
from modules.transaction_cost_model import TransactionCostModel
from modules.data_fetcher import StockDataFetcher


class RealtimeMonitor:
    """Real-time stock monitoring with alerts"""
    
    def __init__(self, stock_code: str = 'sz300620', cost_basis: float = 120):
        self.stock_code = stock_code
        self.cost_basis = cost_basis
        self.is_running = False
        
        # Initialize modules
        self.strategy_engine = StrategyEngine()
        self.kline_analyzer = KlineSignalAnalyzer()
        self.atr_calculator = ATRCalculator()
        self.adx_calculator = ADXCalculator()
        self.hmm_detector = MarketRegimeDetector()
        self.cost_model = TransactionCostModel()
        self.data_fetcher = StockDataFetcher()
        
        # Alert configuration
        self.alerts = {
            'price_alerts': [],
            'signal_alerts': [],
            'volume_alerts': [],
            'trend_alerts': []
        }
        
        # Alert callbacks
        self.alert_callbacks = []
        
        # Last analysis result
        self.last_analysis = None
        self.last_timestamp = None
    
    def add_price_alert(self, price: float, direction: str = 'above', 
                        message: str = None) -> Dict:
        """Add price alert"""
        alert = {
            'price': price,
            'direction': direction,  # 'above' or 'below'
            'triggered': False,
            'message': message or f'价格{direction}¥{price}',
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        self.alerts['price_alerts'].append(alert)
        return alert
    
    def add_signal_alert(self, signal_type: str, strength: str = '强',
                         message: str = None) -> Dict:
        """Add signal alert"""
        alert = {
            'signal_type': signal_type,
            'strength': strength,
            'triggered': False,
            'message': message or f'{signal_type}信号({strength})',
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        self.alerts['signal_alerts'].append(alert)
        return alert
    
    def check_price_alerts(self, current_price: float) -> List[Dict]:
        """Check if price alerts are triggered"""
        triggered = []
        for alert in self.alerts['price_alerts']:
            if not alert['triggered']:
                if alert['direction'] == 'above' and current_price >= alert['price']:
                    alert['triggered'] = True
                    triggered.append(alert)
                elif alert['direction'] == 'below' and current_price <= alert['price']:
                    alert['triggered'] = True
                    triggered.append(alert)
        return triggered
    
    def analyze_stock(self, stock_data: Optional[Dict] = None) -> Dict:
        """Perform comprehensive analysis"""
        if stock_data is None:
            stock_data = self.data_fetcher.get_stock_info(self.stock_code)
        
        if not stock_data:
            return {'error': 'Failed to fetch stock data'}
        
        # Update HMM detector with new data
        try:
            self.hmm_detector.update_with_new_data(stock_data)
        except:
            pass
        
        # Generate K-line signals
        kline_signals = self.kline_analyzer.generate_kline_signals(stock_data)
        
        # Calculate ADX
        adx_data = self.adx_calculator.calculate_adx_from_data(stock_data)
        
        # Get HMM regime
        regime = self.hmm_detector.predict_regime(stock_data)
        regime_adj = self.hmm_detector.get_regime_adjustment(stock_data)
        
        # Generate strategy recommendation
        strategy = self.strategy_engine.generate_strategy_recommendation(
            stock_data, kline_signals, self.cost_basis
        )
        
        # Calculate transaction costs
        position_value = 100000 * 0.25
        round_trip_cost = self.cost_model.calculate_round_trip_cost(position_value)
        
        # Check alerts
        price_alerts = self.check_price_alerts(stock_data.get('price', 0))
        
        # Compile results
        result = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'stock_data': stock_data,
            'kline_signals': kline_signals,
            'adx': adx_data,
            'hmm_regime': regime,
            'regime_adjustment': regime_adj,
            'strategy': strategy,
            'transaction_costs': round_trip_cost,
            'triggered_alerts': price_alerts,
            'analysis_complete': True
        }
        
        self.last_analysis = result
        self.last_timestamp = result['timestamp']
        
        return result
    
    def format_alert_message(self, alert: Dict) -> str:
        """Format alert message"""
        return f"[{alert['created_at']}] {alert['message']}"
    
    def run_monitoring_loop(self, interval: int = 60, max_iterations: int = 10):
        """Run monitoring loop"""
        self.is_running = True
        iteration = 0
        
        print("\n" + "=" * 70)
        print("  实时监控启动")
        print("=" * 70)
        print(f"股票: {self.stock_code}")
        print(f"监控间隔: {interval}秒")
        print("按 Ctrl+C 停止监控\n")
        
        try:
            while self.is_running and iteration < max_iterations:
                iteration += 1
                
                # Analyze stock
                result = self.analyze_stock()
                
                if 'error' not in result:
                    # Display analysis
                    price = result['stock_data'].get('price', 0)
                    change_pct = result['stock_data'].get('change_pct', 0)
                    regime = result['hmm_regime']
                    strategy = result['strategy']['recommendation']
                    
                    print(f"\n[{result['timestamp']}] 第{iteration}次分析")
                    print(f"  价格: ¥{price:.2f} ({change_pct:+.2f}%)")
                    print(f"  HMM状态: {regime}")
                    print(f"  策略建议: {strategy}")
                    
                    # Check triggered alerts
                    if result['triggered_alerts']:
                        print(f"\n  ⚠️  触发警报:")
                        for alert in result['triggered_alerts']:
                            print(f"    - {self.format_alert_message(alert)}")
                    
                    # Call callbacks
                    for callback in self.alert_callbacks:
                        callback(result)
                
                # Wait for next iteration
                if iteration < max_iterations:
                    time.sleep(interval)
        
        except KeyboardInterrupt:
            print("\n\n监控已停止")
        
        self.is_running = False
    
    def get_monitoring_summary(self) -> Dict:
        """Get monitoring summary"""
        if not self.last_analysis:
            return {'error': 'No analysis available'}
        
        result = self.last_analysis
        stock_data = result['stock_data']
        strategy = result['strategy']
        
        return {
            'stock': self.stock_code,
            'price': stock_data.get('price'),
            'change_pct': stock_data.get('change_pct'),
            'turnover': stock_data.get('turnover'),
            'hmm_regime': result['hmm_regime'],
            'recommendation': strategy['recommendation'],
            'action': strategy['action'],
            'signal_strength': strategy['signal_strength'],
            'total_signals': len(strategy['entry_signals']) + len(strategy['exit_signals']),
            'round_trip_cost': result['transaction_costs']['total_round_trip'],
            'analysis_time': result['timestamp']
        }


def main():
    """Main function to run monitor"""
    monitor = RealtimeMonitor('sz300620', cost_basis=120)
    
    # Add some sample alerts
    monitor.add_price_alert(280, 'above', '突破¥280阻力位')
    monitor.add_price_alert(270, 'below', '跌破¥270支撑位')
    monitor.add_signal_alert('多周期共振', '强', '多周期看涨共振信号')
    
    # Run monitoring for 3 iterations
    monitor.run_monitoring_loop(interval=2, max_iterations=3)
    
    # Print summary
    summary = monitor.get_monitoring_summary()
    print("\n" + "=" * 70)
    print("  监控总结")
    print("=" * 70)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
