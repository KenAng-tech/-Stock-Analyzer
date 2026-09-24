"""
Regime Routes - /api/regime-switching/*, /api/sota/enhanced-regime/*

Extracted from app.py (lines 3710-3846, 5387-5414).
Uses predictor_factory for lazy initialization.
"""

from flask import Blueprint, request, jsonify
from datetime import datetime

from modules.logger import logger
from .shared_utils import get_stock_data, build_features, prepare_training_data
from .predictor_factory import predictor_factory

bp = Blueprint('regime', __name__)


def _get_regime_switching():
    """Get regime_switching from predictor_factory"""
    return predictor_factory.get('regime')


@bp.route('/api/regime-switching/predict/<stock_code>')
def api_regime_switching_predict(stock_code):
    """Regime-Switching 多模型预测"""
    try:
        regime_switching = _get_regime_switching()
        if regime_switching is None:
            return jsonify({'success': False, 'error': 'Regime-Switching 预测器未初始化'}), 503

        stock_data = get_stock_data(stock_code)
        if not stock_data:
            return jsonify({'success': False, 'error': 'Failed to fetch stock data'}), 500

        from modules.data_fetcher import StockDataFetcher
        daily_klines = StockDataFetcher().get_kline_data(stock_code, 'daily', 100)
        features = build_features(stock_data, daily_klines)

        result = regime_switching.predict(features, stock_data, daily_klines)

        return jsonify({
            'success': True,
            'data': {
                'direction': result.direction,
                'confidence': result.confidence,
                'probabilities': result.probabilities,
                'detected_regime': result.detected_regime,
                'regime_confidence': result.regime_confidence,
                'regime_weights': result.regime_weights,
                'model_count': result.model_count,
                'execution_time_ms': result.execution_time_ms,
                'model_predictions': result.model_predictions,
            },
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"[Regime-Switching] Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/regime-switching/status')
def api_regime_switching_status():
    """Regime-Switching 模型状态"""
    try:
        regime_switching = _get_regime_switching()
        if regime_switching is None:
            return jsonify({'success': False, 'error': 'Regime-Switching 预测器未初始化'}), 503

        return jsonify({
            'success': True,
            'data': regime_switching.get_status(),
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"[Regime-Switching] Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/regime-switching/train/<stock_code>')
def api_regime_switching_train(stock_code):
    """Regime-Switching 训练"""
    try:
        regime_switching = _get_regime_switching()
        if regime_switching is None:
            return jsonify({'success': False, 'error': 'Regime-Switching 预测器未初始化'}), 503

        X, y = prepare_training_data(stock_code, 500)
        if X is None or len(X) < 50:
            return jsonify({
                'success': False,
                'error': '训练数据不足',
                'sample_count': len(X) if X is not None else 0,
            }), 400

        results = regime_switching.train(X, y)

        return jsonify({
            'success': True,
            'data': {
                'results': results,
                'sample_count': len(X),
                'timestamp': datetime.now().isoformat(),
            },
        })
    except Exception as e:
        logger.error(f"[Regime-Switching] Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/regime-switching/regime/<stock_code>')
def api_regime_switching_regime(stock_code):
    """Regime-Switching Regime 检测"""
    try:
        regime_switching = _get_regime_switching()
        if regime_switching is None:
            return jsonify({'success': False, 'error': 'Regime-Switching 预测器未初始化'}), 503

        stock_data = get_stock_data(stock_code)
        if not stock_data:
            return jsonify({'success': False, 'error': 'Failed to fetch stock data'}), 500

        from modules.data_fetcher import StockDataFetcher
        daily_klines = StockDataFetcher().get_kline_data(stock_code, 'daily', 100)
        regime = regime_switching.regime_detector.detect(stock_data, daily_klines)
        probs = regime_switching.regime_detector.get_regime_probability(stock_data, daily_klines)

        return jsonify({
            'success': True,
            'data': {
                'regime': regime,
                'regime_probability': probs.get(regime, 0.5),
                'all_probabilities': probs,
                'timestamp': datetime.now().isoformat(),
            },
        })
    except Exception as e:
        logger.error(f"[Regime-Switching] Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/regime-switching/model-ranking/<stock_code>')
def api_regime_switching_model_ranking(stock_code):
    """Regime-Switching 模型性能排名"""
    try:
        regime_switching = _get_regime_switching()
        if regime_switching is None:
            return jsonify({'success': False, 'error': 'Regime-Switching 预测器未初始化'}), 503

        rankings = regime_switching.performance_tracker.get_ranking()

        return jsonify({
            'success': True,
            'data': {
                'rankings': [{'model': m, 'accuracy': a} for m, a in rankings],
                'timestamp': datetime.now().isoformat(),
            },
        })
    except Exception as e:
        logger.error(f"[Regime-Switching] Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/sota/enhanced-regime/<stock_code>')
def api_enhanced_regime(stock_code):
    """Enhanced Regime Detection"""
    try:
        regime_switching = _get_regime_switching()
        if regime_switching is None:
            return jsonify({
                'success': True,
                'stock_code': stock_code,
                'data': {
                    'regime': 'sideways',
                    'confidence': 0.5,
                    'probabilities': {'bullish': 0.33, 'bearish': 0.33, 'sideways': 0.34},
                    'transition_prob': {'bullish': 0.33, 'bearish': 0.33, 'sideways': 0.34},
                    'timestamp': datetime.now().isoformat(),
                },
            })

        stock_data = get_stock_data(stock_code)
        if not stock_data:
            return jsonify({
                'success': True,
                'stock_code': stock_code,
                'data': {
                    'regime': 'sideways',
                    'confidence': 0.5,
                    'probabilities': {'bullish': 0.33, 'bearish': 0.33, 'sideways': 0.34},
                    'transition_prob': {'bullish': 0.33, 'bearish': 0.33, 'sideways': 0.34},
                    'timestamp': datetime.now().isoformat(),
                },
            })

        from modules.data_fetcher import StockDataFetcher
        daily_klines = StockDataFetcher().get_kline_data(stock_code, 'daily', 100)
        regime = regime_switching.regime_detector.detect(stock_data, daily_klines)
        probs = regime_switching.regime_detector.get_regime_probability(stock_data, daily_klines)

        # 计算置信度 (最大概率值)
        max_prob = max(probs.values()) if probs else 0.5
        # transition_prob 别名 probabilities 供前端使用
        return jsonify({
            'success': True,
            'stock_code': stock_code,
            'data': {
                'regime': regime,
                'confidence': round(max_prob, 4),
                'probabilities': probs,
                'transition_prob': probs,
                'timestamp': datetime.now().isoformat(),
            },
        })
    except Exception as e:
        logger.error(f"[Enhanced Regime] Error: {e}")
        return jsonify({
            'success': True,
            'stock_code': stock_code,
            'data': {
                'regime': 'sideways',
                'confidence': 0.5,
                'probabilities': {'bullish': 0.33, 'bearish': 0.33, 'sideways': 0.34},
                'transition_prob': {'bullish': 0.33, 'bearish': 0.33, 'sideways': 0.34},
                'timestamp': datetime.now().isoformat(),
            },
        })


# ============================================================================
# Regime-Aware 动态因子权重
# ============================================================================

@bp.route('/api/regime/factor-weights')
def api_regime_factor_weights():
    """获取 Regime-Aware 动态因子权重"""
    try:
        from modules.regime_aware_factor_weights import get_regime_factor_weights
        rw = get_regime_factor_weights()

        regime = request.args.get('regime', 'sideways')
        soft_probs_str = request.args.get('soft_probs')

        soft_probs = None
        if soft_probs_str:
            try:
                import json
                soft_probs = json.loads(soft_probs_str)
            except Exception:
                pass

        weights = rw.get_weights(regime, soft_probs)

        return jsonify({
            'success': True,
            'regime': regime,
            'weights': weights,
            'status': rw.get_status(),
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"Regime factor weights error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/regime/factor-weights/change')
def api_regime_weight_change():
    """获取 Regime 切换时的权重变化"""
    try:
        from modules.regime_aware_factor_weights import get_regime_factor_weights
        rw = get_regime_factor_weights()

        old_regime = request.args.get('old_regime', 'sideways')
        new_regime = request.args.get('new_regime', 'bullish')

        changes = rw.get_weight_change(old_regime, new_regime)

        return jsonify({
            'success': True,
            'old_regime': old_regime,
            'new_regime': new_regime,
            'weight_changes': changes,
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"Regime weight change error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# 漂移检测三算法共识
# ============================================================================

@bp.route('/api/drift/consensus')
def api_drift_consensus():
    """漂移检测三算法共识 API"""
    try:
        from modules.concept_drift_detector import drift_consensus

        # 模拟一些误差数据用于演示
        errors = [float(x) for x in request.args.get('errors', '0.02,0.03,0.01,0.05,0.04,0.02,0.03,0.06,0.02,0.01').split(',')]

        results = []
        for error in errors:
            result = drift_consensus.check_error(error)
            results.append(result)

        # 返回最新结果
        latest = results[-1] if results else {}

        return jsonify({
            'success': True,
            'latest': latest,
            'all_results': results,
            'status': drift_consensus.get_status(),
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"Drift consensus error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# Regime-Weighted Conformal Prediction
# ============================================================================

@bp.route('/api/conformal/rwc')
def api_conformal_rwc():
    """Regime-Weighted Conformal Prediction API"""
    try:
        from modules.models.conformal_predictor import rwc_conformal

        # 模拟校准数据
        scores_str = request.args.get('scores', '0.02,0.03,0.01,0.04,0.02,0.05,0.03,0.02,0.01,0.06')
        regimes_str = request.args.get('regimes', 'sideways,sideways,bullish,bullish,bearish,sideways,bullish,sideways,bearish,volatile')

        scores = [float(x) for x in scores_str.split(',')]
        regimes = regimes_str.split(',')

        for score, regime in zip(scores, regimes):
            rwc_conformal.update(score, regime)

        prediction = float(request.args.get('prediction', '0.02'))
        current_regime = request.args.get('regime', 'sideways')
        lower, upper = rwc_conformal.predict_interval(prediction, current_regime)

        return jsonify({
            'success': True,
            'prediction': prediction,
            'current_regime': current_regime,
            'interval': [round(lower, 6), round(upper, 6)],
            'coverage': round(1 - rwc_conformal.alpha, 2),
            'status': rwc_conformal.get_status(),
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"RWC conformal error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
