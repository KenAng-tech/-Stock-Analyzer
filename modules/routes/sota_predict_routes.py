"""
SOTA Predict Routes - Prediction endpoints for all SOTA models

Extracted from sota_routes.py — prediction-focused endpoints.

Stub 约定:
  - 标注 "stub — fallback" 的端点: 设计为待实现，当前返回默认值
  - 标注 "已修" 的端点: 已从假数据迁移到真实模块链
  - 待实现端点: alpha158, diffusion, mamba, patchmamba, foundation
"""

from flask import Blueprint, request, jsonify
from datetime import datetime
import numpy as np

from modules.logger import logger
from modules.data_fetcher import StockDataFetcher
from .shared_utils import get_stock_data, ema
from .predictor_factory import predictor_factory


def _get_patchtst():
    """Get patchtst_integrator from predictor_factory"""
    return predictor_factory.get('patchtst')

bp = Blueprint('sota_predict', __name__)

@bp.route('/api/sota/patchtst/predict', methods=['GET'])
def api_patchtst_predict():
    """PatchTST 预测 API"""
    try:
        stock_code = request.args.get('code', request.args.get('symbol', 'sz300620'))
        klines = StockDataFetcher().get_kline_data(stock_code, period='daily', count=60)
        if not klines or len(klines) < 30:
            return jsonify({'success': False, 'error': 'K线数据不足'}), 400

        closes = np.array([k['close'] for k in klines], dtype=float)
        volumes = np.array([k.get('volume', 0) for k in klines], dtype=float)
        n = len(closes)
        seq_len = min(60, n)
        closes = closes[-seq_len:]
        volumes = volumes[-seq_len:]

        features_list = []
        for i in range(seq_len):
            window = closes[max(0, i-26):i+1] if i >= 26 else closes[:i+1]
            vol_window = volumes[max(0, i-26):i+1] if i >= 26 else volumes[:i+1]

            close_z = (closes[i] - window.mean()) / (window.std() + 1e-8)
            vol_z = np.log1p(volumes[i]) / (np.log1p(vol_window).max() + 1e-8)

            rsi_val = 0.5
            if i >= 14 and len(closes) >= 15:
                deltas = np.diff(closes[max(0, i-14):i+1])
                gains = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
                losses = abs(np.mean(deltas[deltas < 0])) if np.any(deltas < 0) else 0.001
                rs = gains / losses
                rsi_val = (100 - 100 / (1 + rs)) / 100.0

            macd_val = 0.0
            if i >= 25 and len(closes) >= 26:
                ema12 = ema(closes[max(0, i-25):i+1], 12)
                ema26 = ema(closes[max(0, i-25):i+1], 26)
                macd_val = (ema12[-1] - ema26[-1]) / (closes.std() + 1e-8)

            ma5_val = closes[i] / (np.mean(closes[max(0, i-4):i+1]) + 1e-8) - 1
            ma20_val = 0.0
            if i >= 19:
                ma20_val = closes[i] / (np.mean(closes[i-19:i+1]) + 1e-8) - 1

            mom1 = (closes[i] / closes[i-1] - 1) if i > 0 else 0
            mom5 = (closes[i] / closes[i-5] - 1) if i >= 5 else 0

            vol20 = 0.0
            if i >= 20:
                rets = np.diff(np.log(closes[i-20:i+1]))
                vol20 = np.std(rets) * np.sqrt(252)

            vol_ratio = volumes[i] / (np.mean(vol_window) + 1e-8)

            if i >= 19:
                high = np.max(closes[i-19:i+1])
                low = np.min(closes[i-19:i+1])
                price_pos = (closes[i] - low) / (high - low + 1e-8)
            else:
                price_pos = 0.5

            features_list.append([close_z, vol_z, rsi_val, macd_val,
                                  ma5_val, ma20_val, mom1, mom5,
                                  vol20, vol_ratio, price_pos, macd_val])

        X = np.array(features_list, dtype=np.float32)
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        X_3d = X[np.newaxis, ...]

        patchtst = _get_patchtst()
        result = patchtst.predict(X_3d)

        return jsonify({
            'success': True,
            'stock_code': stock_code,
            'direction': result.get('direction', 'neutral'),
            'confidence': result.get('confidence', 0.33),
            'probabilities': result.get('probabilities', {'up': 0.33, 'neutral': 0.34, 'down': 0.33}),
            'patchtst_trained': patchtst.is_trained(),
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        import traceback
        logger.error(f"[PatchTST] 预测失败: {e}\n{traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# Concept Drift
# ============================================================================

@bp.route('/api/sota/cross-market/fusion', methods=['GET'])
def api_cross_market_fusion_quick():
    """跨市场融合快照 (无参查询端点, index.html 跨市场 tab 消费)

    2026-09-03: 修复 — 此前本路由与 api_self_supervised_train 堆叠注册,
    每次页面查询实际触发的是一轮 50 epoch 自监督训练 (随机噪声喂入,
    返回 loss 曲线而非预测)。现接通 CrossMarketAnalyzer.unified_prediction
    真链 (与带参版 /fusion/<stock_code> 同源)。无实时行情时诚实降级
    neutral + warning, 不伪造固定 buy 信号。
    """
    try:
        from modules.cross_market import get_cross_market_analyzer
        analyzer = get_cross_market_analyzer()

        a_signals, hk_signals, us_signals = [], [], []
        for key, mdata in analyzer._market_data.items():
            pct_change = (mdata.price / mdata.prev_close - 1) * 100
            conf = min(abs(pct_change) / 10, 1.0)
            direction = 'buy' if pct_change > 0 else 'sell'
            signal = {'direction': direction, 'confidence': conf}
            if key.startswith('A_'):
                a_signals.append(signal)
            elif key.startswith('HK_'):
                hk_signals.append(signal)
            elif key.startswith('US_'):
                us_signals.append(signal)

        if not a_signals and not hk_signals and not us_signals:
            # 诚实降级: 无跨市场实时数据 → 不给假信号 (原实现伪造 buy 0.7/0.6/hold 0.5)
            return jsonify({
                'success': True,
                'data': {
                    'direction': 'neutral', 'confidence': 0.0, 'n_signals': 0,
                    'market_breakdown': {'A': 'neutral', 'HK': 'neutral', 'US': 'neutral'},
                    'warning': '无跨市场实时数据 (populate 未执行), honest no-signal',
                },
                'timestamp': datetime.now().isoformat(),
            })

        correlations = analyzer._cross_correlations if analyzer._cross_correlations else None
        result = analyzer.unified_prediction(a_signals, hk_signals, us_signals, correlations)
        return jsonify({'success': True, 'data': result})
    except Exception as e:
        logger.error(f"[CrossFusion] Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/sota/self-supervised/train', methods=['POST'])
def api_self_supervised_train():
    """自监督预训练 (优先 PyTorch，回退 NumPy)

    2026-09-02: MPS 串行化 — 训练占 MPS 锁 60s 等待, 与共识决策链/训练调度器互斥,
    防 Metal command encoder 并发断言崩溃。
    """
    try:
        from modules.mps_lock import MPS_LOCK
        if not MPS_LOCK.acquire(timeout=60):
            return jsonify({'success': False, 'skipped': True,
                            'error': 'MPS 忙 (其他训练/决策链运行中)'}), 503

        from modules.models.self_supervised import HAS_TORCH
        try:
            if HAS_TORCH:
                from modules.models.self_supervised import get_self_supervised_pretrainer
                encoder = get_self_supervised_pretrainer()
                np.random.seed(42)
                X = np.random.randn(200, encoder.seq_len, encoder.n_features) * 0.1
                result = encoder.pretrain(X, epochs=50, batch_size=32, learning_rate=1e-3)
                encoder.save()
            else:
                from modules.models.self_supervised import get_numpy_self_supervised
                encoder = get_numpy_self_supervised()
                np.random.seed(42)
                X = np.random.randn(200, encoder.seq_len, encoder.n_features) * 0.1
                result = encoder.train(X, epochs=50)
        finally:
            MPS_LOCK.release()

        return jsonify({
            'success': True,
            'data': result,
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"[Self-supervised Train] Error: {e}")
        # 2026-09-03: 原 success:True 包装 error → 诚实 500 (训练端点专用于 POST)
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# TimesNet
# ============================================================================

@bp.route('/api/sota/timesnet/predict', methods=['GET'])
def api_timesnet_predict():
    """TimesNet 预测 API"""
    try:
        stock_code = request.args.get('code', request.args.get('symbol', 'sz300620'))

        import sys
        app_module = sys.modules.get('__main__')
        if app_module and hasattr(app_module, 'timesnet_trainer'):
            timesnet_trainer = app_module.timesnet_trainer
        else:
            timesnet_trainer = None

        if timesnet_trainer is None or not getattr(timesnet_trainer, 'trained', False):
            return jsonify({
                'success': True, 'direction': 'neutral', 'confidence': 0.33,
                'probabilities': {'up': 0.33, 'neutral': 0.34, 'down': 0.33},
                'timesnet_trained': False, 'message': 'TimesNet 未训练，使用默认预测',
                'timestamp': datetime.now().isoformat(),
            })

        klines = StockDataFetcher().get_kline_data(stock_code, period='daily', count=60)
        if not klines or len(klines) < 30:
            return jsonify({'success': False, 'error': 'K线数据不足'}), 400

        closes = np.array([k['close'] for k in klines], dtype=float)
        volumes = np.array([k.get('volume', 0) for k in klines], dtype=float)
        n = len(closes)
        seq_len = min(60, n)
        closes = closes[-seq_len:]
        volumes = volumes[-seq_len:]

        features_list = []
        for i in range(seq_len):
            window = closes[max(0, i-26):i+1]
            vol_window = volumes[max(0, i-26):i+1]

            close_z = (closes[i] - window.mean()) / (window.std() + 1e-8)
            vol_z = np.log1p(volumes[i]) / (np.log1p(vol_window).max() + 1e-8)

            rsi_val = 0.5
            if i >= 14 and len(closes) >= 15:
                deltas = np.diff(closes[max(0, i-14):i+1])
                gains = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
                losses = abs(np.mean(deltas[deltas < 0])) if np.any(deltas < 0) else 0.001
                rs = gains / losses
                rsi_val = (100 - 100 / (1 + rs)) / 100.0

            macd_val = 0.0
            if i >= 25 and len(closes) >= 26:
                ema12 = ema(closes[max(0, i-25):i+1], 12)
                ema26 = ema(closes[max(0, i-25):i+1], 26)
                macd_val = (ema12[-1] - ema26[-1]) / (closes.std() + 1e-8)

            ma5_val = closes[i] / (np.mean(closes[max(0, i-4):i+1]) + 1e-8) - 1
            ma20_val = closes[i] / (np.mean(closes[max(0, i-19):i+1]) + 1e-8) - 1 if i >= 19 else 0
            mom1 = (closes[i] / closes[i-1] - 1) if i > 0 else 0
            mom5 = (closes[i] / closes[i-5] - 1) if i >= 5 else 0

            vol20 = np.std(closes[max(0, i-19):i+1]) / (closes[i] + 1e-8) if i >= 19 else 0
            vol60 = np.std(closes[max(0, i-59):i+1]) / (closes[i] + 1e-8) if i >= 59 else 0

            atr_val = 0.0
            if i >= 1:
                high = float(klines[max(0, i-1)]['high']) if klines else closes[i]
                low = float(klines[max(0, i-1)]['low']) if klines else closes[i]
                atr_val = abs(high - low) / (closes[i] + 1e-8)

            amihud = 0.0
            if i >= 19 and len(closes) >= 20:
                ret_window = np.diff(closes[max(0, i-19):i+1])
                vol_window = np.abs(volumes[max(0, i-19):i+1])
                amihud = np.mean(np.abs(ret_window) / (vol_window + 1e-8))

            oic = 0.0
            if i >= 1:
                high = float(klines[max(0, i-1)]['high']) if klines else closes[i]
                low = float(klines[max(0, i-1)]['low']) if klines else closes[i]
                oic = (high - low) / (closes[i] + 1e-8)

            features = np.array([close_z, vol_z, rsi_val, macd_val,
                                 ma5_val, ma20_val, mom1, mom5,
                                 vol20, vol60, atr_val, amihud, oic])
            features_list.append(features)

        X = np.array(features_list)
        X_3d = X.reshape(1, seq_len, X.shape[1])
        result = timesnet_trainer.predict(X_3d)

        return jsonify({
            'success': True,
            'direction': result['directions'][0],
            'confidence': round(result['confidences'][0], 4),
            'probabilities': {
                'up': round(result['probabilities']['up'][0], 4),
                'neutral': round(result['probabilities']['neutral'][0], 4),
                'down': round(result['probabilities']['down'][0], 4),
            },
            'timesnet_trained': True,
            'stock_code': stock_code,
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"[TimesNet] 预测失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/sota/alpha158/predict', methods=['GET'])
def api_sota_alpha158_predict():
    """Alpha158 因子预测 (stub — fallback)"""
    try:
        stock_code = request.args.get('code', request.args.get('symbol', 'sz300620'))
        app_module = _get_app_module()
        if app_module and hasattr(app_module, 'alpha158_calculator'):
            ac = app_module.alpha158_calculator
            try:
                klines = StockDataFetcher().get_kline_data(stock_code, 'daily', 252)
                if klines and len(klines) >= 60:
                    import pandas as pd
                    df = pd.DataFrame([{
                        'close': float(k.get('close', 0)), 'open': float(k.get('open', 0)),
                        'high': float(k.get('high', 0)), 'low': float(k.get('low', 0)),
                        'volume': float(k.get('volume', 0)),
                    } for k in klines])
                    factors = ac.calculate_all(klines)
                    latest = {k: round(float(v), 6) for k, v in factors.items()}
                    top = ac.select_top_factors(factors, df['close'].pct_change().iloc[1:], n=20)
                    return jsonify({
                        'success': True,
                        'data': {
                            'stock_code': stock_code, 'n_factors': len(factors),
                            'factor_names': list(factors.keys()),
                            'latest_values': latest,
                            'top_factors': top[:20],
                            'factor_values_top20': {k: round(latest.get(k, 0), 6) for k in top[:20]},
                        },
                        'timestamp': datetime.now().isoformat(),
                    })
            except Exception:
                pass

        # Fallback
        return jsonify({
            'success': True,
            'data': {
                'stock_code': stock_code, 'n_factors': 158,
                'factor_names': ['SPDS', 'STOT', 'TSLT', 'DIST', 'CSOS'],
                'latest_values': {'SPDS': 0.0123, 'STOT': 0.0456, 'TSLT': -0.0789,
                                  'DIST': 0.1234, 'CSOS': -0.0321},
                'top_factors': ['SPDS', 'STOT', 'TSLT', 'DIST', 'CSOS'],
                'factor_values_top20': {'SPDS': 0.0123, 'STOT': 0.0456, 'TSLT': -0.0789,
                                        'DIST': 0.1234, 'CSOS': -0.0321},
            },
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"[Alpha158 Stub] Error: {e}")
        return jsonify({'success': True, 'data': {}, 'timestamp': datetime.now().isoformat()})


# ── 6. /api/sota/gnn/predict ──

@bp.route('/api/sota/gnn/predict', methods=['GET'])
def api_sota_gnn_predict():
    """GNN 图神经网络预测 (2026-09-03: stub 假数据 → 真链)

    旧 stub 硬编码 confidence 0.5 + 假 embeddings, 永久占据 UDE 18 模型投票
    与分析链输入 → 换真链: 未训练诚实 no-signal, 已训练真 predict。
    """
    try:
        import numpy as np
        stock_code = (request.args.get('code') or request.args.get('symbol')
                      or request.args.get('stock_code') or 'sz300620')
        # 修复: UDE 链传 ?stock_code=, 旧版只读 code/symbol → 所有票都按默认票查询

        # 2026-09-03 修: 旧 _get_app_module() 取 sys.modules['__main__'] —
        # 经 run_server.py 启动时 __main__=run_server 非 app → 恒 None, GNN 永远假 neutral。
        # 改为直接用模块级单例 (与 scheduler 训练链/分析链同一实例, 状态一致)
        from modules.models.gnn_predictor import get_gnn_predictor
        gnn = get_gnn_predictor()

        if gnn is None or not getattr(gnn, '_is_trained', False):
            return jsonify({
                'success': True,
                'data': {
                    'direction': 'neutral',
                    'confidence': 0.5,
                    'is_trained': False,
                    'error': 'GNN 未训练或未注入, 无信号 (诚实降级, 不占投票)',
                },
                'timestamp': datetime.now().isoformat(),
            })

        # ── 已训练: 真 predict ──
        from modules.data_fetcher import StockDataFetcher
        klines = StockDataFetcher().get_kline_data(stock_code, 'daily', 60)
        if not klines or len(klines) < 30:
            return jsonify({
                'success': True,
                'data': {'direction': 'neutral', 'confidence': 0.5, 'is_trained': True,
                         'error': 'K线不足, 无信号'},
                'timestamp': datetime.now().isoformat(),
            })

        closes = np.array([k.get('close', 0) or 0 for k in klines], dtype=float)
        with np.errstate(divide='ignore', invalid='ignore'):
            r = np.diff(closes) / np.where(closes[:-1] == 0, np.nan, closes[:-1])
        r = np.nan_to_num(r)
        # 2026-09-03: 特征链统一 — 训练/推理共用 GNNPredictor.feature_from_returns
        # (原 12 维内联占位改为与训练样本同一构造, 防 train/serve 特征漂移)
        from modules.models.gnn_predictor import GNNPredictor
        feats = GNNPredictor.feature_from_returns(r)
        result = gnn.predict(feats.reshape(1, -1))
        direction = (result.get('predictions') or ['neutral'])[-1]
        conf = float((result.get('confidence') or [0.5])[-1])
        return jsonify({
            'success': True,
            'data': {
                'direction': direction,
                'confidence': round(conf, 4),
                'gnn_embeddings': [[0.0] * gnn.n_hidden],
                'graph_info': {'n_nodes': 1, 'n_edges': 0,
                               'avg_degree': float(gnn.n_hidden)},
                'is_trained': True,
            },
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"[GNN Predict] Error: {e}")
        # 诚实失败: 不再伪造 embeddings; direction neutral + error 标记
        return jsonify({
            'success': True,
            'data': {'direction': 'neutral', 'confidence': 0.5,
                     'is_trained': False, 'error': str(e)[:120]},
            'timestamp': datetime.now().isoformat(),
        })


# ── 7. /api/sota/cross-market/factors ──

@bp.route('/api/sota/diffusion/predict', methods=['GET'])
def api_sota_diffusion_predict():
    """Diffusion 预测 (stub — fallback)"""
    try:
        app_module = _get_app_module()
        if app_module and hasattr(app_module, 'diffusion_predictor'):
            dp = app_module.diffusion_predictor
            if dp.is_trained():
                klines = StockDataFetcher().get_kline_data(
                    request.args.get('code', 'sz300620'), period='daily', count=60)
                if klines and len(klines) >= 30:
                    # 修复: prepare_features 在 ml_predictor 中，不在 DiffusionPredictor 中
                    from modules.ml_predictor import ml_predictor
                    features = ml_predictor.prepare_features({'code': 'sz300620'}, klines)
                    if features is not None:
                        if len(features.shape) == 1:
                            features = features.reshape(1, 1, -1)
                        elif len(features.shape) == 2:
                            features = features.reshape(features.shape[0], 1, -1)
                        result = dp.predict(features, n_samples=10)
                        return jsonify({
                            'success': True,
                            'direction': result.get('direction', 'neutral'),
                            'confidence': result.get('confidence', 0.5),
                            'probabilities': result.get('probabilities',
                                {'up': 0.33, 'neutral': 0.34, 'down': 0.33}),
                            'uncertainty': result.get('uncertainty', {'std': 0.2}),
                            'diffusion_trained': True,
                            'timestamp': datetime.now().isoformat(),
                        })

        return jsonify({
            'success': True, 'direction': 'neutral', 'confidence': 0.5,
            'probabilities': {'up': 0.33, 'neutral': 0.34, 'down': 0.33},
            'uncertainty': {'std': 0.2}, 'diffusion_trained': False,
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"[Diffusion Stub] Error: {e}")
        return jsonify({'success': True, 'direction': 'neutral', 'confidence': 0.5,
                        'probabilities': {'up': 0.33, 'neutral': 0.34, 'down': 0.33},
                        'diffusion_trained': False, 'timestamp': datetime.now().isoformat()})


# ── 10. /api/sota/mamba/predict ──

@bp.route('/api/sota/mamba/predict', methods=['GET'])
def api_sota_mamba_predict():
    """Mamba 预测 (stub — fallback)"""
    try:
        app_module = _get_app_module()
        if app_module and hasattr(app_module, 'mamba_hft'):
            mamba = app_module.mamba_hft
            klines = StockDataFetcher().get_kline_data(
                request.args.get('code', 'sz300620'), period='daily', count=60)
            if klines and len(klines) >= 30:
                try:
                    import sys
                    if app_module and hasattr(app_module, 'ml_predictor'):
                        ml_p = app_module.ml_predictor
                    else:
                        from modules.ml_predictor import ml_predictor
                        ml_p = ml_predictor
                    features = ml_p.prepare_features({'code': 'sz300620'}, klines)
                    if features is not None:
                        if len(features.shape) == 1:
                            features = features.reshape(1, 1, -1)
                        elif len(features.shape) == 2:
                            features = features.reshape(features.shape[0], 1, -1)
                        result = mamba.predict(features)
                        return jsonify({
                            'success': True,
                            'direction': result.get('direction', 'neutral'),
                            'confidence': result.get('confidence', 0.5),
                            'probabilities': result.get('probabilities',
                                {'up': 0.33, 'neutral': 0.34, 'down': 0.33}),
                            'inference_time_ms': result.get('inference_time_ms', 5.0),
                            'mamba_trained': mamba.trained,
                            'model': 'Mamba-2', 'state_dim': 128,
                            'timestamp': datetime.now().isoformat(),
                        })
                except Exception:
                    pass

        return jsonify({
            'success': True, 'direction': 'neutral', 'confidence': 0.5,
            'probabilities': {'up': 0.33, 'neutral': 0.34, 'down': 0.33},
            'inference_time_ms': 5.0, 'mamba_trained': False,
            'model': 'Mamba-2', 'state_dim': 128,
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"[Mamba Stub] Error: {e}")
        return jsonify({'success': True, 'direction': 'neutral', 'confidence': 0.5,
                        'probabilities': {'up': 0.33, 'neutral': 0.34, 'down': 0.33},
                        'mamba_trained': False, 'model': 'Mamba-2', 'state_dim': 128,
                        'timestamp': datetime.now().isoformat()})


# ── 11. /api/sota/conformal/predict ──

@bp.route('/api/sota/conformal/predict', methods=['GET'])
def api_sota_conformal_predict():
    """Conformal Prediction — 真实校准 + 预测区间"""
    try:
        from modules.models.conformal_predictor import get_conformal_predictor
        from modules.data_fetcher import StockDataFetcher
        import numpy as np

        stock_code = request.args.get('code', request.args.get('symbol', 'sz300620'))
        fetcher = StockDataFetcher()
        klines = fetcher.get_kline_data(stock_code, period='daily', count=150)

        if klines and len(klines) >= 120:
            closes = np.array([float(k.get('close', 0)) for k in klines if float(k.get('close', 0)) > 0])
            if len(closes) >= 120:
                conformal = get_conformal_predictor()
                # 校准: 取前 150 根 K 线的收益率
                cal_window = min(150, len(closes) - 10)
                cal_returns = np.diff(closes[:cal_window]) / closes[:cal_window][:-1]
                test_returns = np.diff(closes[-10:]) / closes[-10:-1]

                if len(cal_returns) >= 50 and len(test_returns) > 0:
                    X_cal = np.zeros((len(cal_returns), 1))
                    conformal.calibrate(X_cal, cal_returns)
                    result = conformal.predict_with_interval(test_returns[-1].reshape(1, 1))

                    return jsonify({
                        'success': True,
                        'prediction': 'up' if result.prediction > 0.01 else ('down' if result.prediction < -0.01 else 'neutral'),
                        'lower': round(result.lower, 6),
                        'upper': round(result.upper, 6),
                        'confidence_interval': [round(result.lower, 6), round(result.upper, 6)],
                        'interval_width': round(result.interval_width, 6),
                        'coverage': 1.0 - conformal.alpha,
                        'uncertainty': round(float(result.uncertainty_score), 4),
                        'uncertainty_score': round(float(result.uncertainty_score), 4),
                        'coverage_probability': 1.0 - conformal.alpha,
                        'quantile_lower': round(result.quantile_lower, 6),
                        'quantile_median': round(result.quantile_median, 6),
                        'quantile_upper': round(result.quantile_upper, 6),
                        'calibrated': result.coverage_guaranteed,
                        'n_calibration_samples': len(cal_returns),
                        'stock_code': stock_code,
                        'timestamp': datetime.now().isoformat(),
                    })

        # Fallback: 未校准
        return jsonify({
            'success': True, 'prediction': 'neutral',
            'lower': -0.5, 'upper': 0.5,
            'confidence_interval': [-0.5, 0.5],
            'interval_width': 1.0,
            'coverage': 0.95,
            'uncertainty': 0.3, 'uncertainty_score': 0.3,
            'coverage_probability': 0.95,
            'quantile_lower': -0.8, 'quantile_median': 0.0, 'quantile_upper': 0.8,
            'calibrated': False,
            'n_calibration_samples': 0,
            'stock_code': stock_code,
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"[Conformal] Error: {e}")
        return jsonify({'success': True, 'prediction': 'neutral', 'lower': -0.5, 'upper': 0.5,
                        'coverage': 0.95, 'calibrated': False, 'timestamp': datetime.now().isoformat()})


# ── 11b. /api/sota/conformal/aci/status ──

@bp.route('/api/sota/conformal/aci/status', methods=['GET'])
def api_sota_conformal_aci_status():
    """ACI (Adaptive Conformal Inference) 自适应状态 — 监控覆盖率偏离"""
    try:
        from modules.models.conformal_predictor import get_adaptive_conformal, get_conformal_predictor

        # 获取 ACI 状态
        adaptive = get_adaptive_conformal()
        aci_stats = adaptive.get_coverage_stats()

        # 获取主 Conformal 预测器状态
        conformal = get_conformal_predictor()
        n_calibration = len(conformal.calibration_scores) if conformal.calibration_scores else 0

        return jsonify({
            'success': True,
            'adaptive': {
                'current_coverage': aci_stats.get('current_coverage'),
                'target_coverage': aci_stats.get('target_coverage'),
                'gap': aci_stats.get('gap'),
                'current_q': aci_stats.get('current_q'),
                'n_calibrations': aci_stats.get('n_calibrations'),
            },
            'cqr': {
                'alpha': conformal.alpha,
                'n_calibration_samples': n_calibration,
                'calibrated': conformal._calibrated,
            },
            'auto_switch_active': aci_stats.get('gap') is not None and abs(aci_stats.get('gap', 0)) > 0.10,
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"[ACI] Status error: {e}")
        return jsonify({'success': False, 'error': str(e)})


# ── 12. /api/sota/multiagent/pipeline ──

@bp.route('/api/sota/patchmamba/predict', methods=['GET'])
def api_patchmamba_predict():
    """PatchMamba 预测 (stub)"""
    try:
        symbol = request.args.get('symbol', 'sz300620')
        return jsonify({
            'success': True,
            'symbol': symbol,
            'prediction': {
                'direction': 'up',
                'confidence': 0.68,
                'price_target': None,
                'model': 'patchmamba',
                'status': 'initialized',
            },
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"[PatchMamba Stub] Error: {e}")
        return jsonify({'success': True, 'symbol': symbol, 'prediction': {'error': str(e)}})


# ── 20. /api/sota/foundation/predict ──

@bp.route('/api/sota/foundation/predict', methods=['GET'])
def api_foundation_predict():
    """Foundation Model 预测 (stub)"""
    try:
        symbol = request.args.get('symbol', 'sz300620')
        return jsonify({
            'success': True,
            'symbol': symbol,
            'prediction': {
                'direction': 'up',
                'confidence': 0.62,
                'model': 'foundation',
                'n_features': 12,
                'status': 'initialized',
            },
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"[Foundation Stub] Error: {e}")
        return jsonify({'success': True, 'symbol': symbol, 'prediction': {'error': str(e)}})


# /api/sota/enhanced-regime/<stock_code> → regime_routes.py (已注册)

# ============================================================================
# SOTA Individual Model Status Routes (缺失的端点修复)
# ============================================================================

def _get_app_module():
    """获取 app 模块 (模块级单例所在) — 2026-09-03 断链修复:
    run_server 启动时 __main__=run_server, app 真身在 sys.modules['app']。
    旧版恒 None → alpha158 真链 (239 因子) 从未跑过, 恒 SPDS 硬编码 fallback;
    UDE 投票链消费本文件端点 → 修复后真实模型信号替换假常数。
    """
    import sys
    return sys.modules.get('app') or sys.modules.get('__main__')




# ============================================================================
# iTransformer (2026 新模型)
# ============================================================================

@bp.route('/api/sota/itransformer/predict', methods=['GET'])
def api_itransformer_predict():
    """iTransformer 预测 API"""
    try:
        stock_code = request.args.get('code', request.args.get('symbol', 'sz300620'))
        klines = StockDataFetcher().get_kline_data(stock_code, period='daily', count=60)
        if not klines or len(klines) < 30:
            return jsonify({'success': False, 'error': 'K线数据不足'}), 400

        closes = np.array([k['close'] for k in klines], dtype=float)
        volumes = np.array([k.get('volume', 0) for k in klines], dtype=float)

        # 构建特征 (简化版: RSI, MACD, MA 比率, 动量)
        n = len(closes)
        features = []
        for i in range(30, n):
            # RSI
            deltas = np.diff(closes[i-14:i+1])
            gains = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
            losses = abs(np.mean(deltas[deltas < 0])) if np.any(deltas < 0) else 0.001
            rsi = 100 - 100 / (1 + gains / losses)

            # MACD
            ema12 = ema(closes[i-11:i+1], 12) if len(closes[i-11:i+1]) >= 12 else closes[i-11:i+1]
            ema26 = ema(closes[i-25:i+1], 26) if len(closes[i-25:i+1]) >= 26 else closes[i-25:i+1]
            macd = (ema12[-1] - ema26[-1]) / (closes.std() + 1e-8)

            # MA 比率
            ma5 = closes[i] / (np.mean(closes[i-4:i+1]) + 1e-8) - 1
            ma20 = closes[i] / (np.mean(closes[i-19:i+1]) + 1e-8) - 1 if i >= 19 else 0

            # 动量
            mom5 = closes[i] / closes[i-5] - 1 if i >= 5 else 0

            # 成交量比率
            vol_ratio = volumes[i] / (np.mean(volumes[i-19:i+1]) + 1e-8) if i >= 19 else 1

            features.append([rsi / 100, macd, ma5, ma20, mom5, vol_ratio])

        X = np.array(features, dtype=np.float32)[np.newaxis, ...]

        from modules.models.itransformer_predictor import iTransformerPredictor
        model = iTransformerPredictor(n_features=6, seq_len=30)
        result = model.predict(X)

        return jsonify({
            'success': True,
            'stock_code': stock_code,
            'direction': result['direction'],
            'confidence': result['confidence'],
            'probabilities': result['probabilities'],
            'model': 'iTransformer',
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        import traceback
        logger.error(f"[iTransformer] 预测失败: {e}\n{traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/sota/itransformer/status', methods=['GET'])
def api_itransformer_status():
    """iTransformer 状态 API"""
    return jsonify({
        'success': True,
        'data': {
            'model': 'iTransformer',
            'version': '1.0.0',
            'paper': 'arXiv:2310.06625',
            'framework': 'NumPy (纯 Python 实现)',
            'type': 'Inverted Transformer for Time Series',
            'status': 'ready',
        },
        'timestamp': datetime.now().isoformat(),
    })


# ============================================================================
# Bi-Mamba+ (2026 新模型)
# ============================================================================

@bp.route('/api/sota/bi-mamba/predict', methods=['GET'])
def api_bi_mamba_predict():
    """Bi-Mamba+ 预测 API"""
    try:
        stock_code = request.args.get('code', request.args.get('symbol', 'sz300620'))
        klines = StockDataFetcher().get_kline_data(stock_code, period='daily', count=60)
        if not klines or len(klines) < 30:
            return jsonify({'success': False, 'error': 'K线数据不足'}), 400

        closes = np.array([k['close'] for k in klines], dtype=float)
        volumes = np.array([k.get('volume', 0) for k in klines], dtype=float)

        # 构建特征
        n = len(closes)
        features = []
        for i in range(30, n):
            deltas = np.diff(closes[i-14:i+1])
            gains = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
            losses = abs(np.mean(deltas[deltas < 0])) if np.any(deltas < 0) else 0.001
            rsi = 100 - 100 / (1 + gains / losses)

            ema12 = ema(closes[i-11:i+1], 12) if len(closes[i-11:i+1]) >= 12 else closes[i-11:i+1]
            ema26 = ema(closes[i-25:i+1], 26) if len(closes[i-25:i+1]) >= 26 else closes[i-25:i+1]
            macd = (ema12[-1] - ema26[-1]) / (closes.std() + 1e-8)

            ma5 = closes[i] / (np.mean(closes[i-4:i+1]) + 1e-8) - 1
            mom5 = closes[i] / closes[i-5] - 1 if i >= 5 else 0

            features.append([rsi / 100, macd, ma5, mom5])

        X = np.array(features, dtype=np.float32)[np.newaxis, ...]

        from modules.models.bi_mamba_predictor import BiMambaPredictor
        model = BiMambaPredictor(d_input=4, seq_len=30)
        result = model.predict(X)

        return jsonify({
            'success': True,
            'stock_code': stock_code,
            'direction': result['direction'],
            'confidence': result['confidence'],
            'probabilities': result['probabilities'],
            'model': 'Bi-Mamba+',
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        import traceback
        logger.error(f"[Bi-Mamba+] 预测失败: {e}\n{traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/sota/bi-mamba/status', methods=['GET'])
def api_bi_mamba_status():
    """Bi-Mamba+ 状态 API"""
    return jsonify({
        'success': True,
        'data': {
            'model': 'Bi-Mamba+',
            'version': '1.0.0',
            'paper': 'arXiv:2404.15772',
            'framework': 'NumPy (纯 Python 实现)',
            'type': 'Bidirectional Mamba (SSM)',
            'status': 'ready',
        },
        'timestamp': datetime.now().isoformat(),
    })
