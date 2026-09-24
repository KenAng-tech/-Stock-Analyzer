"""
Time-LLM Routes - /api/time-llm/*

Extracted from app.py (lines 3540-3705).
Uses predictor_factory for lazy initialization.
"""

from flask import Blueprint, request, jsonify
from datetime import datetime
import threading

from modules.logger import logger
from .shared_utils import get_stock_data, build_features, prepare_training_data
from .predictor_factory import predictor_factory

bp = Blueprint('time_llm', __name__)


def _get_time_llm():
    """Get Time-LLM instance from predictor_factory"""
    return predictor_factory.get('time_llm')


@bp.route('/api/time-llm/predict/<stock_code>')
def api_time_llm_predict(stock_code):
    """Time-LLM 统一预测 (带 25 秒超时)"""
    try:
        time_llm = _get_time_llm()
        if time_llm is None:
            logger.warning("[Time-LLM] 预测器未初始化，使用降级结果")
            return jsonify({
                'success': True,
                'data': {
                    'direction': 'neutral', 'confidence': 0.5,
                    'probabilities': {'up': 0.33, 'down': 0.33, 'neutral': 0.34},
                    'market_regime': 'sideways', 'regime_probability': 0.33,
                    'regime_weights': {'bullish': 0.3, 'bearish': 0.3, 'sideways': 0.4},
                    'uncertainty': 0.5, 'execution_time_ms': 0,
                    'models_used': ['fallback'],
                    'model_predictions': {
                        'patchtst': 0.0, 'mamba': 0.0, 'diffusion': 0.0,
                        'cross_modal': {}, 'rl': {'action': 'hold', 'confidence': 0.5},
                    },
                },
                'timestamp': datetime.now().isoformat(),
            })

        stock_data = get_stock_data(stock_code)
        if not stock_data:
            return jsonify({
                'success': True,
                'data': {
                    'direction': 'neutral', 'confidence': 0.5,
                    'probabilities': {'up': 0.33, 'down': 0.33, 'neutral': 0.34},
                    'market_regime': 'sideways', 'uncertainty': 0.5, 'execution_time_ms': 0,
                    'models_used': ['fallback'],
                    'model_predictions': {'patchtst': 0.0, 'mamba': 0.0, 'diffusion': 0.0,
                                          'cross_modal': {}, 'rl': {'action': 'hold', 'confidence': 0.5}},
                },
                'timestamp': datetime.now().isoformat(),
            })

        from modules.data_fetcher import StockDataFetcher
        daily_klines = StockDataFetcher().get_kline_data(stock_code, 'daily', 100)
        features = build_features(stock_data, daily_klines)

        result_container = [None]
        error_container = [None]

        def _predict():
            try:
                result_container[0] = time_llm.predict(features, stock_data, daily_klines)
            except Exception as e:
                error_container[0] = e

        thread = threading.Thread(target=_predict, daemon=True)
        thread.start()
        thread.join(timeout=25.0)

        if thread.is_alive():
            logger.warning("[Time-LLM] 预测超时 (25s)，使用降级结果")
            result = type('Result', (), {
                'direction': 'neutral', 'confidence': 0.5,
                'probabilities': {'up': 0.33, 'down': 0.33, 'neutral': 0.34},
                'market_regime': 'sideways', 'regime_probability': 0.33,
                'regime_weights': {'bullish': 0.3, 'bearish': 0.3, 'sideways': 0.4},
                'uncertainty': 0.5, 'execution_time_ms': 25000,
                'models_used': ['patchtst_numpy', 'mamba_numpy', 'diffusion_numpy'],
                'patchtst_prediction': 0.0, 'mamba_prediction': 0.0,
                'diffusion_prediction': 0.0, 'cross_modal_prediction': {},
                'rl_prediction': {'action': 'hold', 'confidence': 0.5},
            })()
        elif error_container[0]:
            raise error_container[0]
        else:
            result = result_container[0]

        return jsonify({
            'success': True,
            'data': {
                'direction': result.direction,
                'confidence': result.confidence,
                'probabilities': result.probabilities,
                'market_regime': result.market_regime,
                'regime_probability': result.regime_probability,
                'regime_weights': result.regime_weights,
                'uncertainty': result.uncertainty,
                'execution_time_ms': result.execution_time_ms,
                'models_used': result.models_used,
                'model_predictions': {
                    'patchtst': result.patchtst_prediction,
                    'mamba': result.mamba_prediction,
                    'diffusion': result.diffusion_prediction,
                    'cross_modal': result.cross_modal_prediction,
                    'rl': result.rl_prediction,
                },
            },
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"[Time-LLM] Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/time-llm/status')
def api_time_llm_status():
    """Time-LLM 模型状态"""
    try:
        time_llm = _get_time_llm()
        if time_llm is None:
            return jsonify({
                'success': True,
                'data': {'initialized': False, 'model': 'Time-LLM'},
                'timestamp': datetime.now().isoformat(),
            })

        return jsonify({
            'success': True,
            'data': time_llm.get_status(),
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"[Time-LLM] Error: {e}")
        return jsonify({
            'success': True,
            'data': {'initialized': False, 'error': str(e)},
            'timestamp': datetime.now().isoformat(),
        })


@bp.route('/api/time-llm/train/<stock_code>')
def api_time_llm_train(stock_code):
    """Time-LLM 训练"""
    try:
        time_llm = _get_time_llm()
        if time_llm is None:
            return jsonify({'success': False, 'error': 'Time-LLM 预测器未初始化'}), 503

        stock_data = get_stock_data(stock_code)
        if not stock_data:
            return jsonify({'success': False, 'error': 'Failed to fetch stock data'}), 500

        X, y = prepare_training_data(stock_code, 500)
        if X is None or len(X) < 50:
            return jsonify({
                'success': False,
                'error': '训练数据不足',
                'sample_count': len(X) if X is not None else 0,
            }), 400

        results = time_llm.train(X, y)

        return jsonify({
            'success': True,
            'data': {
                'results': results,
                'sample_count': len(X),
                'timestamp': datetime.now().isoformat(),
            },
        })
    except Exception as e:
        logger.error(f"[Time-LLM] Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/time-llm/save/<stock_code>')
def api_time_llm_save(stock_code):
    """保存 Time-LLM 模型"""
    try:
        time_llm = _get_time_llm()
        if time_llm is None:
            return jsonify({'success': False, 'error': 'Time-LLM 预测器未初始化'}), 503

        saved = time_llm.save()
        return jsonify({
            'success': True,
            'data': saved,
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"[Time-LLM] Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/time-llm/reset')
def api_time_llm_reset():
    """重置 Time-LLM 模型"""
    try:
        time_llm = _get_time_llm()
        if time_llm is None:
            return jsonify({'success': False, 'error': 'Time-LLM 预测器未初始化'}), 503

        time_llm.reset()
        return jsonify({
            'success': True,
            'message': 'Time-LLM 模型已重置',
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"[Time-LLM] Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
