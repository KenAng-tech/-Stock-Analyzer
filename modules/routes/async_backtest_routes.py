"""
async_backtest_routes.py - 异步回测 API

将耗时的回测端点改为后台执行，通过 task_id 轮询进度。
"""

from flask import Blueprint, request, jsonify
from datetime import datetime
import numpy as np

from modules.logger import logger
from .shared_utils import compute_rsi, compute_macd_histogram, compute_ema
from .async_tasks import submit_task, get_task_status

bp = Blueprint('async_backtest', __name__)


@bp.route('/api/backtest/async/submit', methods=['POST'])
def api_backtest_async_submit():
    """提交异步回测任务"""
    try:
        body = request.get_json() or {}
        stock_code = body.get('stock_code', 'sz300620')
        backtest_type = body.get('type', 'walkforward')  # walkforward | event

        task_id = submit_task(
            func=_run_backtest,
            args=(stock_code, backtest_type),
            task_type='backtest',
            stock_code=stock_code,
        )

        return jsonify({
            'success': True,
            'task_id': task_id,
            'status': 'pending',
            'message': '回测任务已提交',
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"[AsyncBacktest] 提交失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/backtest/async/status/<task_id>')
def api_backtest_async_status(task_id):
    """查询异步回测状态"""
    try:
        status = get_task_status(task_id)
        if status is None:
            return jsonify({'success': False, 'error': 'Task not found'})
        return jsonify({
            'success': True,
            'data': status,
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"[AsyncBacktest] 状态查询失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/backtest/async/tasks')
def api_backtest_async_tasks():
    """列出所有回测任务"""
    try:
        tasks = list_tasks()
        return jsonify({
            'success': True,
            'tasks': tasks,
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"[AsyncBacktest] 列表失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


def _run_backtest(stock_code: str, backtest_type: str, _progress=None):
    """执行回测（后台线程）"""
    from modules.data_fetcher import StockDataFetcher
    from modules.walkforward_backtester import WalkForwardBacktester

    if _progress:
        _progress(10)

    fetcher = StockDataFetcher()
    raw_klines = fetcher.get_kline_data(stock_code, period='daily', count=500)

    if not raw_klines or len(raw_klines) < 100:
        raise ValueError('K 线数据不足')

    if _progress:
        _progress(30)

    closes = np.array([float(k['close']) for k in raw_klines])

    # 增强 K 线数据
    kline_data = []
    for i, k in enumerate(raw_klines):
        bar = dict(k)
        bar['close'] = float(k['close'])
        bar['rsi'] = compute_rsi(closes[:i+1]) if i >= 14 else 50.0
        bar['macd_histogram'] = compute_macd_histogram(closes[:i+1]) if i >= 35 else 0.0
        kline_data.append(bar)

    if _progress:
        _progress(50)

    # Walk-Forward 回测
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
    wf_result = backtester.run_walk_forward(
        kline_data, train_period=120, test_period=42, n_windows=5,
    )

    if _progress:
        _progress(80)

    # Monte Carlo 模拟
    daily_returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
    mc_result = backtester.monte_carlo_simulation(daily_returns, n_simulations=1000)

    if _progress:
        _progress(100)

    return {
        'walk_forward': wf_result,
        'monte_carlo': mc_result,
        'stock_code': stock_code,
        'timestamp': datetime.now().isoformat(),
    }
