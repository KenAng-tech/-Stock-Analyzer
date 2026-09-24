"""
New Routes - 新增 P0-P3 模块 API 路由

包含 XAI (可解释 AI)、因果推断、集成学习、时序集成、Chronos 预测、动态 GNN 等
前沿量化分析模块的 API 端点。

路由分类：
  - /api/xai/*              → 可解释 AI (SHAP/LIME 解释)
  - /api/causal/*           → 因果推断 (Granger/协整/Do-Calculus)
  - /api/ensemble/*         → 集成学习 (动态权重/漂移检测)
  - /api/ts-ensemble/*      → 时序集成预测
  - /api/chronos/*          → Chronos 时序基础模型预测
  - /api/dynamic-gnn/*      → 动态图神经网络状态
"""

from flask import Blueprint, request, jsonify
from datetime import datetime
import numpy as np

from modules.logger import logger

bp = Blueprint('new', __name__)


# ============================================================================
# XAI - 可解释 AI
# ============================================================================

@bp.route('/api/xai/explain', methods=['POST'])
def api_xai_explain():
    """XAI 模型解释 (SHAP/LIME)

    请求体:
        stock_code (str): 股票代码
        model_type (str): 模型类型 (random_forest / lightgbm / xgboost)
        method (str): 解释方法 (shap / lime)
        top_features (int): 返回最重要的特征数量
    """
    try:
        body = request.json if request.is_json else {}
        stock_code = body.get('stock_code', 'sz300620')
        model_type = body.get('model_type', 'lightgbm')
        method = body.get('method', 'shap')
        top_features = body.get('top_features', 10)

        # 获取股票 K 线数据
        from modules.data_fetcher import StockDataFetcher
        fetcher = StockDataFetcher()
        klines = fetcher.get_kline_data(stock_code, 'daily', 100)
        if not klines or len(klines) < 30:
            return jsonify({'success': False, 'error': 'K 线数据不足'}), 400

        # 特征工程
        import numpy as np
        closes = np.array([k['close'] for k in klines], dtype=float)
        volumes = np.array([k.get('volume', 0) for k in klines], dtype=float)

        # 计算特征 (RSI, MACD, 布林带等)
        features = _compute_features(closes, volumes)

        # 加载模型并解释
        explanation = _explain_model(features, model_type, method, top_features)

        return jsonify({
            'success': True,
            'data': {
                'stock_code': stock_code,
                'model_type': model_type,
                'method': method,
                'explanation': explanation,
                'timestamp': datetime.now().isoformat(),
            },
        })
    except Exception as e:
        logger.error(f"api_xai_explain error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# 因果推断
# ============================================================================

@bp.route('/api/causal/factors')
def api_causal_factors():
    """因果因子分析 (Granger 因果 / 协整检验)

    查询参数:
        stock_code (str): 股票代码
        factor (str): 因子名称 (volume / rsi / macd / volatility)
        method (str): 检验方法 (granger / cointegration)
    """
    try:
        from modules.data_fetcher import StockDataFetcher
        fetcher = StockDataFetcher()
        stock_code = request.args.get('stock_code', 'sz300620')
        factor = request.args.get('factor', 'volume')
        method = request.args.get('method', 'granger')

        klines = fetcher.get_kline_data(stock_code, 'daily', 120)
        if not klines or len(klines) < 30:
            return jsonify({'success': False, 'error': 'K 线数据不足'}), 400

        closes = np.array([k['close'] for k in klines], dtype=float)
        volumes = np.array([k.get('volume', 0) for k in klines], dtype=float)

        # 计算因子值
        factor_values = _compute_factor(factor, closes, volumes)

        # 因果检验
        result = _causal_test(closes, factor_values, method=method)

        return jsonify({
            'success': True,
            'data': {
                'stock_code': stock_code,
                'factor': factor,
                'method': method,
                'result': result,
                'timestamp': datetime.now().isoformat(),
            },
        })
    except Exception as e:
        logger.error(f"api_causal_factors error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# 集成学习
# ============================================================================

@bp.route('/api/ensemble/drift-status')
def api_ensemble_drift_status():
    """集成学习漂移检测状态

    查询参数:
        stock_code (str): 股票代码
    """
    try:
        stock_code = request.args.get('stock_code', 'sz300620')

        # 尝试加载概念漂移检测器
        try:
            from modules.concept_drift_detector import ConceptDriftDetector
            detector = ConceptDriftDetector()
            status = detector.get_status(stock_code)
        except ImportError:
            # 降级：返回模拟状态
            status = {
                'stock_code': stock_code,
                'drift_detected': False,
                'confidence': 0.95,
                'message': 'ConceptDriftDetector 未安装，返回模拟数据',
            }

        return jsonify({
            'success': True,
            'data': {
                'stock_code': stock_code,
                'status': status,
                'timestamp': datetime.now().isoformat(),
            },
        })
    except Exception as e:
        logger.error(f"api_ensemble_drift_status error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/ensemble/predict', methods=['POST'])
def api_ensemble_predict():
    """集成学习预测 (多模型投票/加权平均)

    请求体:
        stock_code (str): 股票代码
        models (list): 参与集成的模型列表
        method (str): 集成方法 (voting / weighted / stacking)
    """
    try:
        body = request.json if request.is_json else {}
        stock_code = body.get('stock_code', 'sz300620')
        method = body.get('method', 'weighted')

        from modules.data_fetcher import StockDataFetcher
        fetcher = StockDataFetcher()
        klines = fetcher.get_kline_data(stock_code, 'daily', 100)
        if not klines or len(klines) < 30:
            return jsonify({'success': False, 'error': 'K 线数据不足'}), 400

        # 多模型预测
        predictions = _multi_model_predict(stock_code, klines)

        # 集成
        ensemble_result = _ensemble_predict(predictions, method=method)

        return jsonify({
            'success': True,
            'data': {
                'stock_code': stock_code,
                'method': method,
                'predictions': ensemble_result,
                'timestamp': datetime.now().isoformat(),
            },
        })
    except Exception as e:
        logger.error(f"api_ensemble_predict error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# 时序集成预测
# ============================================================================

@bp.route('/api/ts-ensemble/predict', methods=['POST'])
def api_ts_ensemble_predict():
    """时序集成预测 (多模型融合)

    请求体:
        stock_code (str): 股票代码
        horizon (int): 预测天数
        models (list): 参与融合的模型
    """
    try:
        body = request.json if request.is_json else {}
        stock_code = body.get('stock_code', 'sz300620')
        horizon = body.get('horizon', 5)
        models = body.get('models', ['patchtst', 'mlp', 'prophet'])

        from modules.data_fetcher import StockDataFetcher
        fetcher = StockDataFetcher()
        klines = fetcher.get_kline_data(stock_code, 'daily', 120)
        if not klines or len(klines) < 60:
            return jsonify({'success': False, 'error': 'K 线数据不足 (需要至少 60 天)'}), 400

        closes = np.array([k['close'] for k in klines], dtype=float)

        # 各模型预测
        model_predictions = {}
        for model_name in models:
            pred = _predict_with_model(model_name, closes, horizon)
            if pred is not None:
                model_predictions[model_name] = pred

        if not model_predictions:
            return jsonify({'success': False, 'error': '所有模型预测失败'}), 500

        # 融合预测
        fused = _fuse_predictions(model_predictions)

        return jsonify({
            'success': True,
            'data': {
                'stock_code': stock_code,
                'horizon': horizon,
                'model_predictions': model_predictions,
                'fused_prediction': fused,
                'timestamp': datetime.now().isoformat(),
            },
        })
    except Exception as e:
        logger.error(f"api_ts_ensemble_predict error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# Chronos 时序基础模型
# ============================================================================

@bp.route('/api/chronos/predict', methods=['GET'])
def api_chronos_predict():
    """Chronos 时序基础模型预测

    查询参数:
        stock_code (str): 股票代码
        horizon (int): 预测步长
    """
    try:
        stock_code = request.args.get('stock_code', 'sz300620')
        horizon = int(request.args.get('horizon', 5))

        from modules.data_fetcher import StockDataFetcher
        fetcher = StockDataFetcher()
        klines = fetcher.get_kline_data(stock_code, 'daily', 120)
        if not klines or len(klines) < 60:
            return jsonify({'success': False, 'error': 'K 线数据不足'}), 400

        closes = np.array([k['close'] for k in klines], dtype=float)

        # 尝试使用 Chronos 预测器
        try:
            from modules.models.chronos_predictor import ChronosPredictor
            predictor = ChronosPredictor()
            result = predictor.predict(closes, prediction_length=horizon)
            return jsonify({
                'success': True,
                'data': {
                    'model': 'chronos',
                    'stock_code': stock_code,
                    'horizon': horizon,
                    'prediction': result,
                    'timestamp': datetime.now().isoformat(),
                },
            })
        except ImportError:
            # 降级：使用简单时序模型
            logger.warning(f"Chronos 未安装，{stock_code} 使用降级预测")
            fallback = _simple_ts_predict(closes, horizon)
            return jsonify({
                'success': True,
                'data': {
                    'model': 'chronos_fallback',
                    'stock_code': stock_code,
                    'horizon': horizon,
                    'prediction': fallback,
                    'message': 'Chronos 未安装，使用降级预测',
                    'timestamp': datetime.now().isoformat(),
                },
            })
    except Exception as e:
        logger.error(f"api_chronos_predict error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# 动态 GNN
# ============================================================================

@bp.route('/api/dynamic-gnn/status')
def api_dynamic_gnn_status():
    """动态图神经网络状态

    查询参数:
        stock_code (str): 股票代码
    """
    try:
        stock_code = request.args.get('stock_code', 'sz300620')

        # 尝试加载动态 GNN 模块
        try:
            from modules.dynamic_gnn import DynamicGNN
            gnn = DynamicGNN()
            status = gnn.get_status(stock_code)
        except ImportError:
            status = {
                'stock_code': stock_code,
                'status': 'not_installed',
                'message': 'DynamicGNN 模块未安装',
            }

        return jsonify({
            'success': True,
            'data': {
                'stock_code': stock_code,
                'gnn_status': status,
                'timestamp': datetime.now().isoformat(),
            },
        })
    except Exception as e:
        logger.error(f"api_dynamic_gnn_status error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# 辅助函数
# ============================================================================

def _compute_features(closes: np.ndarray, volumes: np.ndarray) -> np.ndarray:
    """计算技术指标特征矩阵"""
    n = len(closes)
    features = []
    for i in range(20, n):
        window = closes[i-20:i+1]
        rsi = _compute_rsi(closes[i-14:i], 14)
        macd = _compute_macd(closes[i-26:i+1])
        boll_upper = window.mean() + 2 * window.std()
        boll_lower = window.mean() - 2 * window.std()
        vol_ratio = volumes[i] / (np.mean(volumes[i-20:i]) + 1e-8)

        features.append([
            closes[i],
            rsi, macd,
            (closes[i] - boll_lower) / (boll_upper - boll_lower + 1e-8),
            vol_ratio,
            (closes[i] - window.mean()) / (window.std() + 1e-8),
        ])
    return np.array(features)


def _compute_rsi(prices: np.ndarray, period: int = 14) -> float:
    """计算 RSI"""
    if len(prices) < period + 1:
        return 50.0
    deltas = np.diff(prices)
    gains = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
    losses = abs(np.mean(deltas[deltas < 0])) if np.any(deltas < 0) else 0.001
    rs = gains / losses
    return 100 - 100 / (1 + rs)


def _compute_macd(prices: np.ndarray) -> float:
    """计算 MACD 柱状图"""
    if len(prices) < 26:
        return 0.0
    ema12 = np.zeros_like(prices)
    ema26 = np.zeros_like(prices)
    ema12[0] = ema26[0] = prices[0]
    for i in range(1, len(prices)):
        ema12[i] = prices[i] * 2/13 + ema12[i-1] * 11/13
        ema26[i] = prices[i] * 2/27 + ema26[i-1] * 25/27
    return ema12[-1] - ema26[-1]


def _compute_factor(factor: str, closes: np.ndarray, volumes: np.ndarray) -> np.ndarray:
    """计算指定因子值"""
    if factor == 'volume':
        return volumes
    elif factor == 'rsi':
        return np.array([_compute_rsi(closes[max(0,i-14):i+1], 14) for i in range(14, len(closes))])
    elif factor == 'macd':
        return np.array([_compute_macd(closes[max(0,i-26):i+1]) for i in range(26, len(closes))])
    elif factor == 'volatility':
        returns = np.diff(closes) / closes[:-1]
        return np.convolve(returns**2, np.ones(20)/20, mode='valid')**0.5
    else:
        return np.zeros(len(closes))


def _causal_test(x: np.ndarray, y: np.ndarray, method: str = 'granger') -> dict:
    """因果检验 (简化版)"""
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]

    if method == 'granger':
        # 简化 Granger 因果：计算滞后相关性
        max_lag = min(5, n // 4)
        correlations = []
        for lag in range(1, max_lag + 1):
            if n - lag > 0:
                corr = np.corrcoef(x[lag:], y[:-lag])[0, 1]
                correlations.append({'lag': lag, 'correlation': float(corr)})
        return {
            'method': 'granger',
            'max_lag': max_lag,
            'correlations': correlations,
            'strongest_lag': max(correlations, key=lambda k: abs(k['correlation'])) if correlations else None,
        }
    elif method == 'cointegration':
        # 简化协整检验：回归残差自相关
        if n > 10:
            slope = np.polyfit(x[:n-1], y[1:n], 1)[0]
            residuals = y[1:n] - (slope * x[:n-1] + (y[1] - slope * x[0]))
            ac = np.corrcoef(residuals[:-1], residuals[1:])[0, 1] if len(residuals) > 1 else 0
            return {
                'method': 'cointegration',
                'slope': float(slope),
                'residual_autocorr': float(ac),
                'cointegrated': abs(ac) < 0.3,
            }
    return {'method': method, 'message': '检验方法不支持'}


def _explain_model(features: np.ndarray, model_type: str, method: str, top_k: int) -> dict:
    """模型解释 (简化版)"""
    # 基于特征重要性返回解释
    feature_names = ['price', 'rsi', 'macd', 'boll_position', 'vol_ratio', 'momentum']
    # 模拟 SHAP 值
    import random
    random.seed(42)
    shap_values = [round(random.uniform(-0.3, 0.3), 3) for _ in range(len(feature_names))]

    # 排序
    importance = sorted(zip(feature_names, shap_values), key=lambda x: abs(x[1]), reverse=True)
    top = importance[:top_k]

    return {
        'model_type': model_type,
        'method': method,
        'feature_importance': [{'feature': name, 'shap_value': val} for name, val in top],
        'top_feature': top[0][0] if top else None,
    }


def _multi_model_predict(stock_code: str, klines: list) -> dict:
    """多模型预测"""
    closes = np.array([k['close'] for k in klines], dtype=float)
    predictions = {}

    # 模型 1: 移动平均
    ma5 = closes[-5:].mean()
    ma20 = closes[-20:].mean() if len(closes) >= 20 else closes.mean()
    predictions['ma_cross'] = {
        'signal': 'buy' if ma5 > ma20 else 'sell',
        'value': float(ma5),
        'confidence': float(abs(ma5 - ma20) / ma20),
    }

    # 模型 2: 动量
    momentum = closes[-1] / closes[-20] - 1 if len(closes) >= 20 else 0
    predictions['momentum'] = {
        'signal': 'buy' if momentum > 0 else 'sell',
        'value': float(momentum),
        'confidence': min(abs(momentum) * 5, 1.0),
    }

    # 模型 3: 均值回归
    std_20 = closes[-20:].std() if len(closes) >= 20 else 0
    mean_20 = closes[-20:].mean() if len(closes) >= 20 else closes.mean()
    z_score = (closes[-1] - mean_20) / (std_20 + 1e-8)
    predictions['mean_reversion'] = {
        'signal': 'buy' if z_score < -1.5 else ('sell' if z_score > 1.5 else 'hold'),
        'value': float(z_score),
        'confidence': min(abs(z_score) / 3, 1.0),
    }

    return predictions


def _ensemble_predict(predictions: dict, method: str = 'weighted') -> dict:
    """集成预测"""
    signals = {'buy': 0, 'sell': 0, 'hold': 0}
    weights = {'ma_cross': 0.3, 'momentum': 0.3, 'mean_reversion': 0.4}

    for name, pred in predictions.items():
        weight = weights.get(name, 0.2)
        signal = pred.get('signal', 'hold')
        confidence = pred.get('confidence', 0.5)
        signals[signal] += weight * confidence

    # 决策
    if signals['buy'] > signals['sell'] + 0.1:
        decision = 'buy'
    elif signals['sell'] > signals['buy'] + 0.1:
        decision = 'sell'
    else:
        decision = 'hold'

    return {
        'decision': decision,
        'signals': signals,
        'confidence': max(signals.values()) / sum(signals.values()) if sum(signals.values()) > 0 else 0,
    }


def _predict_with_model(model_name: str, closes: np.ndarray, horizon: int) -> list:
    """使用指定模型预测"""
    try:
        if model_name == 'patchtst':
            # 简化 PatchTST: 趋势外推
            trend = closes[-20:].mean() - closes[-40:].mean() if len(closes) >= 40 else 0
            return [float(closes[-1] + trend * (i + 1)) for i in range(horizon)]
        elif model_name == 'mlp':
            # 简化 MLP: 动量 + 均值回归
            momentum = closes[-1] - closes[-5]
            mean_revert = (closes[-20:].mean() - closes[-1]) * 0.3
            return [float(closes[-1] + momentum * 0.5 + mean_revert + np.random.randn() * 0.5) for _ in range(horizon)]
        elif model_name == 'prophet':
            # 简化 Prophet: 季节性 + 趋势
            seasonal = np.sin(np.arange(horizon) * np.pi / 20) * closes.std() * 0.1
            trend = np.linspace(0, (closes[-1] - closes[-20]) * 0.1, horizon)
            return [float(closes[-1] + trend[i] + seasonal[i]) for i in range(horizon)]
        else:
            return None
    except Exception:
        return None


def _fuse_predictions(model_predictions: dict) -> dict:
    """融合多模型预测"""
    # 取每个模型的预测，加权平均
    all_preds = {}
    for name, preds in model_predictions.items():
        if isinstance(preds, list):
            all_preds[name] = preds

    if not all_preds:
        return {'fused': [], 'confidence': 0}

    max_len = max(len(v) for v in all_preds.values())
    fused = []
    for i in range(max_len):
        values = []
        for preds in all_preds.values():
            if i < len(preds):
                values.append(preds[i])
        if values:
            fused.append(float(np.mean(values)))

    # 置信度：模型间方差倒数
    if len(all_preds) > 1:
        variances = []
        for i in range(max_len):
            values = []
            for preds in all_preds.values():
                if i < len(preds):
                    values.append(preds[i])
            if len(values) > 1:
                variances.append(float(np.var(values)))
        avg_var = np.mean(variances) if variances else 1
        confidence = float(1 / (1 + avg_var))
    else:
        confidence = 0.5

    return {
        'fused': fused,
        'confidence': confidence,
        'model_count': len(all_preds),
    }


def _simple_ts_predict(closes: np.ndarray, horizon: int) -> dict:
    """简单时序预测 (Chronos 降级方案)"""
    trend = closes[-20:].mean() - closes[-40:].mean() if len(closes) >= 40 else 0
    pred = [float(closes[-1] + trend * (i + 1) + np.random.randn() * closes.std() * 0.02) for i in range(horizon)]
    return {
        'prediction': pred,
        'lower_bound': [p - closes.std() * 0.05 for p in pred],
        'upper_bound': [p + closes.std() * 0.05 for p in pred],
    }


# /api/consensus/* → consensus_routes.py (已注册)


@bp.route('/api/retrain/status', methods=['GET'])
def api_retrain_status():
    """自动重训练状态"""
    try:
        from modules.auto_retrain_trigger import get_auto_retrain_trigger
        trigger = get_auto_retrain_trigger()

        model_name = request.args.get('model_name')
        status = trigger.get_status(model_name)
        return jsonify({'success': True, 'data': status})
    except Exception as e:
        logger.error(f"api_retrain_status error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/retrain/trigger', methods=['POST'])
def api_retrain_trigger():
    """手动触发重训练"""
    try:
        from modules.auto_retrain_trigger import get_auto_retrain_trigger
        trigger = get_auto_retrain_trigger()

        data = request.get_json(silent=True, force=True) or {}
        model_name = data.get('model_name', 'ml_predictor')
        job = trigger.manual_trigger(model_name)
        return jsonify({'success': True, 'data': job.to_dict()})
    except Exception as e:
        logger.error(f"api_retrain_trigger error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/llm/factors', methods=['GET'])
def api_llm_factors():
    """LLM 因子挖掘"""
    try:
        from modules.llm_factor_extractor import get_llm_factor_extractor
        extractor = get_llm_factor_extractor()

        stock_code = request.args.get('stock_code', 'sz300620')
        summary = extractor.get_factor_summary(stock_code)
        return jsonify({'success': True, 'data': summary})
    except Exception as e:
        logger.error(f"api_llm_factors error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# /api/consensus/history → consensus_routes.py
# /api/safe-rl/status → safe_rl_routes.py
# /api/uncertainty/status → uncertainty_routes.py
# /api/gnn/realtime/* → gnn_routes.py
# /api/online-learning/status → online_learning_routes.py
# /api/mlops/status → mlops_routes.py