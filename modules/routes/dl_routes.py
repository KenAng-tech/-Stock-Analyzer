"""
DL/RL Routes - /api/dl/*, /api/rl/*

Extracted from app.py (lines 2805-3095).
Uses predictor_factory for lazy initialization.
"""

from flask import Blueprint, request, jsonify
from datetime import datetime
import numpy as np

from modules.data_fetcher import StockDataFetcher
from modules.logger import logger
from .predictor_factory import predictor_factory

bp = Blueprint('dl', __name__)


@bp.route('/api/dl/predict/<stock_code>')
def api_dl_predict(stock_code):
    """深度学习模型预测 (PatchTST — 替换原 Transformer-LSTM)"""
    try:
        patchtst_integrator = predictor_factory.get('patchtst')
        if patchtst_integrator is None:
            return jsonify({'success': False, 'error': 'PatchTST 未初始化'}), 503

        fetcher = StockDataFetcher()
        klines = fetcher.get_kline_data(stock_code, 'daily', 100)

        if not klines or len(klines) < 20:
            return jsonify({'success': False, 'error': 'K 线数据不足'})

        closes = np.array([k['close'] for k in klines], dtype=float)
        volumes = np.array([k['volume'] for k in klines], dtype=float)

        def compute_features(i):
            if i < 20:
                return np.zeros(12)
            window = closes[:i+1]
            vol_window = volumes[:i+1]
            mom_1d = (window[-1] / window[-2] - 1) * 100 if len(window) >= 2 else 0
            mom_3d = (window[-1] / window[-4] - 1) * 100 if len(window) >= 4 else 0
            mom_5d = (window[-1] / window[-6] - 1) * 100 if len(window) >= 6 else 0
            mom_10d = (window[-1] / window[-11] - 1) * 100 if len(window) >= 11 else 0
            avg_vol = np.mean(vol_window[-20:]) if len(vol_window) >= 20 else np.mean(vol_window)
            vol_ratio = vol_window[-1] / avg_vol if avg_vol > 0 else 1.0
            if len(window) >= 20:
                rets = np.diff(np.log(window[-20:]))
                vol = float(np.std(rets) * np.sqrt(252) * 100)
            else:
                vol = 5.0
            if len(window) >= 15:
                deltas = np.diff(window[-15:])
                gains = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
                losses = abs(np.mean(deltas[deltas < 0])) if np.any(deltas < 0) else 0.001
                rsi = float(100 - (100 / (1 + gains / losses)))
            else:
                rsi = 50.0
            ema12 = window[0]
            ema26 = window[0]
            for p in window[1:]:
                ema12 = (p - ema12) * (2/13) + ema12
                ema26 = (p - ema26) * (2/27) + ema26
            macd_hist = (ema12 - ema26) * 0.1
            ma5 = np.mean(window[-5:]) if len(window) >= 5 else window[-1]
            ma20 = np.mean(window[-20:]) if len(window) >= 20 else window[-1]
            ma_ratio = float(ma5 / ma20) if ma20 > 0 else 1.0
            year_high = np.max(window)
            year_low = np.min(window)
            price_pos = float((window[-1] - year_low) / (year_high - year_low + 1e-10))
            return np.array([mom_1d, mom_3d, mom_5d, mom_10d, vol_ratio, vol, rsi, macd_hist, ma_ratio, price_pos, 0.5, 1.0])

        seq_len = 20
        sequences = []
        for i in range(seq_len - 1, min(len(klines), 50)):
            seq = np.stack([compute_features(i - seq_len + 1 + j) for j in range(seq_len)])
            sequences.append(seq)

        if not sequences:
            return jsonify({'success': False, 'error': '无法构建序列'})

        sequences = np.stack(sequences)

        # 使用 PatchTST 替代旧的 dl_model_v2
        result = patchtst_integrator.predict(sequences[:1])

        # 兼容单样本 (direction/confidence) 和多样本 (directions/confidences) 返回格式
        if 'directions' in result:
            direction = result['directions'][0]
            confidence = result['confidences'][0]
            probabilities = {
                'up': result['probabilities']['up'][0] if isinstance(result['probabilities']['up'], (list, np.ndarray)) else result['probabilities']['up'],
                'neutral': result['probabilities']['neutral'][0] if isinstance(result['probabilities']['neutral'], (list, np.ndarray)) else result['probabilities']['neutral'],
                'down': result['probabilities']['down'][0] if isinstance(result['probabilities']['down'], (list, np.ndarray)) else result['probabilities']['down'],
            }
        else:
            direction = result['direction']
            confidence = result['confidence']
            probabilities = {
                'up': result['probabilities']['up'],
                'neutral': result['probabilities']['neutral'],
                'down': result['probabilities']['down'],
            }

        return jsonify({
            'success': True,
            'stock_code': stock_code,
            'prediction': {
                'direction': direction,
                'confidence': confidence,
                'probabilities': probabilities,
            },
            'model': 'PatchTST',
            'timestamp': datetime.now().isoformat(),
        })

    except Exception as e:
        logger.error(f"[DL Predict] 错误：{e}")
        return jsonify({'success': False, 'error': str(e)})


@bp.route('/api/rl/trader/status')
def api_rl_trader_status():
    """强化学习交易器状态"""
    try:
        rl_trader_v2 = predictor_factory.get('rl')
        if rl_trader_v2 is None:
            return jsonify({'success': False, 'error': 'RL Trader 未初始化'}), 503

        return jsonify({
            'success': True,
            'status': {
                'trained': rl_trader_v2._trained,
                'market_regime': rl_trader_v2._market_regime,
                'ppo_available': rl_trader_v2.ppo_agent is not None,
                'sac_available': rl_trader_v2.sac_agent is not None,
            },
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"[RL Trader Status] 错误：{e}")
        return jsonify({'success': False, 'error': str(e)})


@bp.route('/api/rl/train/<stock_code>', methods=['POST'])
def api_rl_train(stock_code):
    """训练 RL Trader 模型"""
    try:
        data = request.get_json() or {}
        n_episodes = data.get('n_episodes', 30)

        # 获取 K 线数据
        fetcher = StockDataFetcher()
        klines = fetcher.get_kline_data(stock_code, period='daily', count=250)

        if not klines or len(klines) < 60:
            return jsonify({'success': False, 'error': 'K 线数据不足 (至少 60 条)'})

        closes = np.array([k['close'] for k in klines], dtype=np.float64)
        volumes = np.array([k['volume'] for k in klines], dtype=np.float64)

        # 计算 12 维特征
        def compute_features(i):
            if i < 20:
                return np.zeros(12)
            window = closes[:i + 1]
            vol_window = volumes[:i + 1]
            mom_1d = (window[-1] / window[-2] - 1) * 100 if len(window) >= 2 else 0
            mom_3d = (window[-1] / window[-4] - 1) * 100 if len(window) >= 4 else 0
            mom_5d = (window[-1] / window[-6] - 1) * 100 if len(window) >= 6 else 0
            mom_10d = (window[-1] / window[-11] - 1) * 100 if len(window) >= 11 else 0
            avg_vol = np.mean(vol_window[-20:]) if len(vol_window) >= 20 else np.mean(vol_window)
            vol_ratio = vol_window[-1] / avg_vol if avg_vol > 0 else 1.0
            if len(window) >= 20:
                rets = np.diff(np.log(window[-20:]))
                vol = float(np.std(rets) * np.sqrt(252) * 100)
            else:
                vol = 5.0
            if len(window) >= 15:
                deltas = np.diff(window[-15:])
                gains = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
                losses = abs(np.mean(deltas[deltas < 0])) if np.any(deltas < 0) else 0.001
                rsi = float(100 - (100 / (1 + gains / losses)))
            else:
                rsi = 50.0
            ema12 = window[0]
            ema26 = window[0]
            for p in window[1:]:
                ema12 = (p - ema12) * (2/13) + ema12
                ema26 = (p - ema26) * (2/27) + ema26
            macd_hist = (ema12 - ema26) * 0.1
            ma5 = np.mean(window[-5:]) if len(window) >= 5 else window[-1]
            ma20 = np.mean(window[-20:]) if len(window) >= 20 else window[-1]
            ma_ratio = float(ma5 / ma20) if ma20 > 0 else 1.0
            year_high = np.max(window)
            year_low = np.min(window)
            price_pos = float((window[-1] - year_low) / (year_high - year_low + 1e-10))
            return np.array([mom_1d, mom_3d, mom_5d, mom_10d, vol_ratio, vol, rsi, macd_hist, ma_ratio, price_pos, 0.5, 1.0])

        # 构建训练数据
        X = []
        y = []
        for i in range(20, len(klines)):
            feat = compute_features(i)
            X.append(feat)
            # 标签: 1=涨, 0=跌
            if i + 5 < len(klines):
                y.append(1 if klines[i + 5]['close'] > klines[i]['close'] else 0)
            else:
                y.append(1 if klines[-1]['close'] > klines[i]['close'] else 0)

        X = np.array(X)
        y = np.array(y)

        # 获取 RL Trader
        rl_trader_v2 = predictor_factory.get('rl')
        if rl_trader_v2 is None:
            return jsonify({'success': False, 'error': 'RL Trader 未初始化'}), 503

        # 训练 (2026-09-10 死链修复: 此前调 .train(X,y) 方法不存在 = 恒 AttributeError 死链)
        result = rl_trader_v2.train(X, y, closes=closes, n_episodes=n_episodes)

        return jsonify({
            'success': True,
            'stock_code': stock_code,
            'data': result,
            'training_result': result,  # dl_dashboard.html trainRLModel 读此键 (toFixed 链)
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"[RL Train] 错误：{e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)})


@bp.route('/api/dl/ensemble/report')
def api_dl_ensemble_report():
    """深度学习模型报告 (PatchTST — 替换原 Transformer-LSTM)"""
    try:
        patchtst_integrator = predictor_factory.get('patchtst')
        if patchtst_integrator is None:
            return jsonify({'success': False, 'error': 'PatchTST 未初始化'}), 503

        return jsonify({
            'success': True,
            'report': {
                'model': {
                    'name': 'PatchTST',
                    'architecture': 'Patch + Transformer Encoder + RoPE',
                    'description': '时序预测 SOTA 架构 (2024)',
                },
                'params': {
                    'd_model': patchtst_integrator.d_model,
                    'n_heads': patchtst_integrator.n_heads,
                    'n_layers': patchtst_integrator.n_layers,
                    'patch_len': patchtst_integrator.patch_len,
                    'seq_len': patchtst_integrator.seq_len,
                    'n_features': patchtst_integrator.n_features,
                    'n_classes': patchtst_integrator.n_classes,
                    'dropout': patchtst_integrator.dropout,
                },
                'trained': patchtst_integrator.is_trained(),
                'device': str(patchtst_integrator.device) if patchtst_integrator.device else 'N/A',
            },
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"[DL Report] 错误：{e}")
        return jsonify({'success': False, 'error': str(e)})
