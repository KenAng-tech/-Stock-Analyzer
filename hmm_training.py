#!/usr/bin/env python3
"""
HMM Market Regime Training Script
Trains HMM model on simulated historical data
"""

import sys
import os
import json
import numpy as np
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.hmm_market_detector import MarketRegimeDetector
from modules.atr_calculator import ATRCalculator


def generate_historical_data(n_days: int = 500) -> list:
    """Generate historical data for HMM training"""
    import random
    random.seed(42)
    
    data = []
    price = 100
    
    for day in range(n_days):
        # Market regime simulation
        cycle = day % 120
        if cycle < 40:
            regime = 'bull'
            return_mean = 0.002
        elif cycle < 80:
            regime = 'bear'
            return_mean = -0.002
        else:
            regime = 'sideways'
            return_mean = 0
        
        daily_return = np.random.normal(return_mean, 0.015)
        price = price * (1 + daily_return)
        turnover = 150 + np.random.normal(0, 30)
        
        record = {
            'day': day,
            'date': (datetime.now() - timedelta(days=n_days - day)).strftime('%Y-%m-%d'),
            'price': round(price, 2),
            'change_pct': round(daily_return * 100, 2),
            'turnover': round(max(50, turnover), 1),
            'high': round(price * 1.01, 2),
            'low': round(price * 0.99, 2),
            'open': round(price * (1 + np.random.normal(0, 0.005)), 2),
            'close': round(price, 2),
            'prev_close': round(price / (1 + daily_return), 2),
            'outer_disk': int(50000 + np.random.normal(0, 5000)),
            'inner_disk': int(45000 + np.random.normal(0, 5000)),
            'year_high': 150,
            'year_low': 80,
            'volume': int(100000 + np.random.normal(0, 10000)),
            'amount': round(price * 100000, 2),
            'sector': '科技',
            'regime': regime
        }
        
        data.append(record)
    
    return data


def train_hmm_model(historical_data: list):
    """Train HMM model on historical data"""
    print("\n" + "=" * 70)
    print("  HMM市场状态识别模型训练")
    print("=" * 70)
    
    # Initialize detector
    detector = MarketRegimeDetector(n_states=3)
    
    print(f"\n训练数据: {len(historical_data)} 个交易日")
    print("训练参数: 3个市场状态 (牛市/熊市/震荡)")
    
    # Train model
    print("\n正在训练HMM模型...")
    detector.fit(historical_data)
    
    print("✓ 模型训练完成")
    
    # Test on sample data
    print("\n--- 模型测试 ---")
    sample_data = historical_data[-1]
    regime = detector.predict_regime(sample_data)
    probabilities = detector.get_regime_probability(sample_data)
    adjustment = detector.get_regime_adjustment(sample_data)
    
    print(f"\n当前市场状态: {regime}")
    print(f"状态概率分布:")
    for state, prob in probabilities.items():
        bar = "█" * int(prob * 20)
        print(f"  {state:<10}: {prob:.3f} {bar}")
    
    print(f"\n市场调整因子:")
    for key, value in adjustment['adjustments'].items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.3f}")
    
    # Save model parameters
    model_params = {
        'n_states': detector.n_states,
        'state_names': detector.state_names,
        'feature_means': detector.feature_means.tolist() if detector.feature_means is not None else [],
        'feature_stds': detector.feature_stds.tolist() if detector.feature_stds is not None else [],
        'transition_matrix': detector.transition_matrix.tolist(),
        'is_fitted': detector.is_fitted,
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'n_training_samples': len(historical_data)
    }
    
    output_file = '/Users/claw/stock_analyzer/models/hmm_model_params.json'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(model_params, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ 模型参数已保存至: {output_file}")
    
    return detector


def main():
    print("生成历史训练数据...")
    historical_data = generate_historical_data(500)
    
    detector = train_hmm_model(historical_data)
    
    # Test regime predictions over time
    print("\n--- 市场状态时间序列测试 ---")
    predictions = []
    for i in range(0, len(historical_data), 30):
        sample = historical_data[i]
        regime = detector.predict_regime(sample)
        prob = detector.get_regime_probability(sample)[regime]
        predictions.append({
            'day': sample['day'],
            'regime': regime,
            'probability': round(prob, 3)
        })
    
    print("\n市场状态变化趋势:")
    for pred in predictions:
        regime_icon = {'bull': '📈', 'bear': '📉', 'sideways': '➡️'}
        icon = regime_icon.get(pred['regime'], '●')
        print(f"  第{pred['day']:3d}天 {icon} {pred['regime']:<10} (概率: {pred['probability']:.3f})")
    
    print("\n" + "=" * 70)
    print("  HMM模型训练完成！")
    print("=" * 70)


if __name__ == '__main__':
    main()
