"""
Backtest Routes - /api/backtest/*

Extracted from app.py (lines 2033-2206, 2577-2680).
"""

from flask import Blueprint, request, jsonify
from datetime import datetime
import numpy as np

from modules.logger import logger

bp = Blueprint('backtest', __name__)


@bp.route('/api/backtest/report')
def api_backtest_report():
    """Walk-forward 回测报告 + Monte Carlo 模拟"""
    stock_code = request.args.get('stock_code', 'sz300620')
    try:
        from modules.walkforward_backtester import WalkForwardBacktester
        from modules.data_fetcher import StockDataFetcher

        data_fetcher = StockDataFetcher()
        raw_klines = data_fetcher.get_kline_data(stock_code, period='daily', count=500)

        if not raw_klines or len(raw_klines) < 100:
            return jsonify({'success': False, 'error': 'K 线数据不足'})

        # ── 从原始 K 线计算 RSI + MACD ───────────────────────────
        closes = np.array([float(k['close']) for k in raw_klines])

        def compute_rsi(prices, period=14):
            if len(prices) < period + 1:
                return 50.0
            deltas = np.diff(prices)
            gains = np.mean(deltas[-period:][deltas[-period:] > 0]) if np.any(deltas[-period:] > 0) else 0
            losses = abs(np.mean(deltas[-period:][deltas[-period:] < 0])) if np.any(deltas[-period:] < 0) else 0.001
            rs = gains / losses
            return float(100 - (100 / (1 + rs)))

        def compute_macd_histogram(prices, fast=12, slow=26, signal=9):
            if len(prices) < slow + signal:
                return 0.0
            ema_fast = prices[0]
            for p in prices[1:]:
                ema_fast = (p - ema_fast) * (2 / (fast + 1)) + ema_fast
            ema_slow = prices[0]
            for p in prices[1:]:
                ema_slow = (p - ema_slow) * (2 / (slow + 1)) + ema_slow
            macd_line = ema_fast - ema_slow
            return float(macd_line * 0.1)

        # 为每根 K 线增强指标
        kline_data = []
        for i, k in enumerate(raw_klines):
            bar = dict(k)
            bar['close'] = float(k['close'])
            bar['rsi'] = compute_rsi(closes[:i+1]) if i >= 14 else 50.0
            bar['macd_histogram'] = compute_macd_histogram(closes[:i+1]) if i >= 35 else 0.0
            kline_data.append(bar)

        # ── 增强策略: RSI + MACD + 均线 ──────────────────────────
        def enhanced_strategy(bar, position, capital):
            rsi = bar.get('rsi', 50)
            macd_hist = bar.get('macd_histogram', 0)
            close = bar.get('close', 0)

            if position == 0:
                if rsi < 35 and macd_hist > 0 and close > 0:
                    return 'buy'
            elif position > 0:
                if rsi > 65 and macd_hist < 0:
                    return 'sell'
            return 'hold'

        backtester = WalkForwardBacktester(enhanced_strategy, initial_capital=1000000)

        # Walk-forward
        wf_result = backtester.run_walk_forward(
            kline_data, train_period=120, test_period=42, n_windows=5
        )

        # Monte Carlo
        daily_returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
        mc_result = backtester.monte_carlo_simulation(daily_returns, n_simulations=1000)

        # ── 生成中文总结 ──────────────────────────────────────────
        summary = wf_result.get('summary', {})
        windows = wf_result.get('windows', [])
        mc = mc_result

        mean_ret = summary.get('mean_return', 0)
        mean_sharpe = summary.get('mean_sharpe', 0)
        mean_winrate = summary.get('mean_winrate', 0)
        mean_maxdd = summary.get('mean_maxdd', 0)
        n_windows = summary.get('n_windows', 0)
        consistent = summary.get('consistent_profit', 0)

        if mean_ret > 0.05:
            strategy_verdict = '策略表现良好，在多数窗口实现了正收益'
        elif mean_ret > 0:
            strategy_verdict = '策略略有盈利，但收益较低，建议优化入场/出场条件'
        elif mean_ret > -0.05:
            strategy_verdict = '策略小幅亏损，交易成本可能侵蚀了利润，建议放宽交易条件或缩短持仓周期'
        else:
            strategy_verdict = '策略表现不佳，建议重新设计信号逻辑'

        if mean_maxdd < 0.05:
            risk_verdict = '回撤控制优秀，最大回撤低于 5%'
        elif mean_maxdd < 0.15:
            risk_verdict = '回撤在可接受范围内'
        else:
            risk_verdict = '回撤偏大，建议增加止损或降低仓位'

        if mean_sharpe > 1.0:
            sharpe_verdict = '风险调整后收益优秀'
        elif mean_sharpe > 0.5:
            sharpe_verdict = '风险调整后收益良好'
        elif mean_sharpe > 0:
            sharpe_verdict = '风险调整后收益一般'
        else:
            sharpe_verdict = '风险调整后收益较差'

        if mean_winrate > 0.6:
            winrate_verdict = '交易胜率较高'
        elif mean_winrate > 0.4:
            winrate_verdict = '胜率中等，盈亏比是关键'
        else:
            winrate_verdict = '胜率偏低，需关注单笔亏损控制'

        mc_prob = mc.get('probability_profit', 0)
        if mc_prob > 0.8:
            mc_verdict = '长期盈利概率很高'
        elif mc_prob > 0.6:
            mc_verdict = '长期盈利概率较好'
        elif mc_prob > 0.4:
            mc_verdict = '长期盈利概率一般'
        else:
            mc_verdict = '长期盈利概率偏低'

        chinese_summary = {
            'title': '回测总结',
            'strategy_verdict': strategy_verdict,
            'risk_verdict': risk_verdict,
            'sharpe_verdict': sharpe_verdict,
            'winrate_verdict': winrate_verdict,
            'mc_verdict': mc_verdict,
            'details': {
                'total_windows': n_windows,
                'consistent_profit_windows': consistent,
                'mean_return_pct': round(mean_ret * 100, 2),
                'mean_sharpe': round(mean_sharpe, 3),
                'mean_maxdd_pct': round(mean_maxdd * 100, 2),
                'mean_winrate_pct': round(mean_winrate * 100, 0),
                'mc_profit_prob_pct': round(mc_prob * 100, 0),
                'mc_mean_final_wan': round(mc.get('mean_final', 0) / 10000, 1),
                'mc_median_final_wan': round(mc.get('median_final', 0) / 10000, 1),
            },
        }

        return jsonify({
            'success': True,
            'stock_code': stock_code,
            'walk_forward': wf_result,
            'monte_carlo': mc_result,
            'chinese_summary': chinese_summary,
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"Backtest report error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)})


@bp.route('/api/backtest/event')
def api_event_backtest():
    """事件驱动回测 API"""
    stock_code = request.args.get('stock_code', 'sz300620')
    try:
        from modules.data_fetcher import StockDataFetcher
        from modules.event_backtester import EventDrivenBacktester
        # 2026-09-11 断链修复: 08-19 迁移至 modules/strategies/ 后 import 未跟 = 恒 ImportError 死链
        from modules.strategies.rsi_macd_strategy import RSIMACDStrategy
        import numpy as np

        fetcher = StockDataFetcher()
        raw_klines = fetcher.get_kline_data(stock_code, 'daily', 300)

        if not raw_klines or len(raw_klines) < 60:
            return jsonify({'success': False, 'error': 'K线数据不足 (需要至少 60 根)'})

        # 计算技术指标
        closes = np.array([float(k['close']) for k in raw_klines])

        def compute_rsi(prices, period=14):
            if len(prices) < period + 1:
                return 50.0
            deltas = np.diff(prices)
            gains = np.mean(deltas[-period:][deltas[-period:] > 0]) if np.any(deltas[-period:] > 0) else 0
            losses = abs(np.mean(deltas[-period:][deltas[-period:] < 0])) if np.any(deltas[-period:] < 0) else 0.001
            return float(100 - (100 / (1 + gains / losses)))

        def compute_ema(data, period):
            if len(data) < period:
                return float(np.mean(data))
            m = 2.0 / (period + 1)
            r = float(data[0])
            for p in data[1:]:
                r = (p - r) * m + r
            return r

        # 为每根 K 线计算指标
        kline_data = []
        for i, k in enumerate(raw_klines):
            bar = dict(k)
            bar['close'] = float(k['close'])
            bar['open'] = float(k.get('open', bar['close']))
            bar['high'] = float(k.get('high', bar['close']))
            bar['low'] = float(k.get('low', bar['close']))
            bar['volume'] = float(k.get('volume', 0))
            bar['stock_code'] = stock_code

            if i >= 14:
                bar['rsi'] = compute_rsi(closes[:i+1])
            else:
                bar['rsi'] = 50.0

            if i >= 26:
                e12 = compute_ema(closes[:i+1], 12)
                e26 = compute_ema(closes[:i+1], 26)
                bar['macd_histogram'] = (e12 - e26) * 0.1
            else:
                bar['macd_histogram'] = 0.0

            if i >= 20:
                bar['ma20'] = float(np.mean(closes[:i+1]))
            else:
                bar['ma20'] = bar['close']

            if i >= 14:
                bar['atr'] = float(np.std(closes[:i+1])) * bar['close'] * 0.03
            else:
                bar['atr'] = bar['close'] * 0.02

            kline_data.append(bar)

        # 运行回测
        strategy = RSIMACDStrategy(
            buy_rsi=35, sell_rsi=65,
            stop_loss_pct=0.05, take_profit_pct=0.15,
            position_pct=0.8,
        )
        backtester = EventDrivenBacktester(
            strategy=strategy,
            initial_capital=1000000,
            commission_rate=0.0003,
            stamp_tax=0.001,
            slippage_bps=3,
        )
        result = backtester.run(kline_data)

        return jsonify({
            'success': True,
            'stock_code': stock_code,
            'metrics': result.get('metrics', {}),
            'equity_curve': result.get('equity_curve', []),
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"Event backtest error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)})
