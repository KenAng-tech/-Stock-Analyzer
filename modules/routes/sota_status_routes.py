"""
SOTA Status Routes - Status/health endpoints for all SOTA models

Extracted from sota_routes.py — status-focused endpoints.

Stub 约定:
  - 标注 "stub — fallback" 的端点: 设计为待实现，当前返回默认值
  - 待实现端点: cache/status, ensemble/status, factor-weights/status, drift/advanced/status
"""

from flask import Blueprint, request, jsonify
from datetime import datetime
import numpy as np

from modules.logger import logger
from modules.data_fetcher import StockDataFetcher
from .shared_utils import get_stock_data, ema
from .predictor_factory import predictor_factory

bp = Blueprint('sota_status', __name__)


def _get_patchtst():
    """Get patchtst_integrator from predictor_factory"""
    return predictor_factory.get('patchtst')


def _get_app_module():
    """获取 app 模块 (模块级单例所在).

    2026-09-03 断链修复: 旧版只查 sys.modules['__main__'] —
    run_server.py 启动时 __main__=run_server (无任何单例), 且
    `import app as app_module` 使真 app 在 sys.modules['app'] →
    本文件 8 个 status 端点恒走 fallback (假 neutral/0.5 或恒 None)。
    修复同 predictor_factory: 'app' 优先, '__main__' 兜底 (直跑 app.py 场景)。
    """
    import sys
    return sys.modules.get('app') or sys.modules.get('__main__')


@bp.route('/api/sota/drift/status', methods=['GET'])
def api_drift_status():
    """概念漂移检测状态 API"""
    try:
        app_module = _get_app_module()  # 2026-09-08 修: 原内联 '__main__' 在 run_server 启动下取不到 app 模块
        if app_module and hasattr(app_module, 'drift_monitor'):
            drift_monitor = app_module.drift_monitor
        else:
            from modules.drift_monitor import DriftMonitor
            drift_monitor = DriftMonitor()

        status = drift_monitor.get_status()
        return jsonify({
            'success': True,
            **status,
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"[DriftMonitor] 状态获取失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/sota/drift/reset', methods=['POST'])
def api_drift_reset():
    """重置概念漂移检测器"""
    try:
        app_module = _get_app_module()  # 2026-09-08 修: 原内联 '__main__' 在 run_server 启动下取不到 app 模块
        if app_module and hasattr(app_module, 'drift_monitor'):
            drift_monitor = app_module.drift_monitor
        else:
            from modules.drift_monitor import DriftMonitor
            drift_monitor = DriftMonitor()

        drift_monitor.reset()
        return jsonify({
            'success': True,
            'message': 'Drift monitor reset',
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"[DriftMonitor] 重置失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# SOTA Overall Status
# ============================================================================

@bp.route('/api/sota/status', methods=['GET'])
def api_sota_status():
    """SOTA 优化模块总体状态"""
    try:
        app_module = _get_app_module()  # 2026-09-08 修: 原内联 '__main__' 在 run_server 启动下取不到 app 模块

        patchtst = _get_patchtst()
        patchtst_trained = patchtst.is_trained() if patchtst else False
        patchtst_device = str(patchtst.device) if patchtst and hasattr(patchtst, 'device') and patchtst.device else 'N/A'
        patchtst_history = getattr(patchtst, 'training_history', []) if patchtst else []

        drift_status = {}
        if app_module and hasattr(app_module, 'drift_monitor'):
            drift_status = app_module.drift_monitor.get_status()

        sentiment_info = {}
        if app_module and hasattr(app_module, 'sentiment_engine'):
            se = app_module.sentiment_engine
            sentiment_info = {
                'method': 'finbert' if (se._finbert and se._finbert._use_hf) else 'dictionary',
                'finbert_loaded': se._finbert._initialized if se._finbert else False,
            }

        diffusion_trained = False
        diffusion_device = 'N/A'
        if app_module and hasattr(app_module, 'diffusion_predictor'):
            dp = app_module.diffusion_predictor
            diffusion_trained = dp.is_trained()
            diffusion_device = str(dp.device) if hasattr(dp, 'device') and dp.device else 'N/A'

        multiagent_info = {}
        if app_module and hasattr(app_module, 'multi_agent_coordinator'):
            multiagent_info = app_module.multi_agent_coordinator.get_status()

        mamba_trained = False
        mamba_device = 'N/A'
        if app_module and hasattr(app_module, 'mamba_hft'):
            mamba_trained = app_module.mamba_hft.trained
            mamba_device = str(app_module.mamba_hft.device) if hasattr(app_module.mamba_hft, 'device') and app_module.mamba_hft.device else 'N/A'

        self_supervised_trained = False
        self_supervised_device = 'N/A'
        if app_module and hasattr(app_module, 'self_supervised_pretrainer'):
            ss = app_module.self_supervised_pretrainer
            self_supervised_trained = getattr(ss, 'trained', False)
            self_supervised_device = str(ss.device) if hasattr(ss, 'device') and ss.device else 'N/A'

        alpha158_info = {}
        if app_module and hasattr(app_module, 'alpha158_calculator'):
            ac = app_module.alpha158_calculator
            alpha158_info = {
                'num_factors': len(ac.factors) if ac else 0,
                'factor_names': list(ac.factors.keys()) if ac else [],
            }

        return jsonify({
            'success': True,
            'patchtst': {'trained': patchtst_trained, 'device': patchtst_device, 'training_history': patchtst_history},
            'drift_monitor': drift_status,
            'sentiment_engine': sentiment_info,
            'diffusion': {'trained': diffusion_trained, 'device': diffusion_device},
            'multiagent': multiagent_info,
            'mamba': {'trained': mamba_trained, 'device': mamba_device},
            'self_supervised': {'trained': self_supervised_trained, 'device': self_supervised_device},
            'alpha158': alpha158_info,
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"[SOTA] 状态获取失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# Moirai
# ============================================================================

@bp.route('/api/sota/moirai/status', methods=['GET'])
def api_moirai_status():
    """Moirai 预测器状态"""
    try:
        from modules.models.moirai_predictor import get_moirai_predictor
        predictor = get_moirai_predictor()
        return jsonify({
            'data': predictor.to_dict(),
            'status': 'ready',
        })
    except Exception as e:
        logger.error(f"[API] Moirai status error: {e}")
        return jsonify({'error': str(e)}), 500


@bp.route('/api/sota/drl/status', methods=['GET'])
def api_drl_status():
    """DRL 代理状态"""
    try:
        from modules.models.drl_agent import get_drl_agent
        agent = get_drl_agent()
        # 2026-09-04: to_dict() 全量权重 dump 69KB → get_status() 摘要 (前端仅消费标量字段)
        return jsonify({
            'data': agent.get_status(),
            'state_dim': agent.STATE_DIM,
            'action_dim': agent.ACTION_DIM,
            'status': 'ready',
        })
    except Exception as e:
        logger.error(f"[API] DRL status error: {e}")
        return jsonify({'error': str(e)}), 500


@bp.route('/api/sota/timesfm/status', methods=['GET'])
def api_timesfm_status():
    """TimesFM 状态"""
    try:
        app_module = _get_app_module()
        if app_module and hasattr(app_module, 'timesfm_predictor'):
            timesfm_predictor = app_module.timesfm_predictor
        else:
            timesfm_predictor = None

        if timesfm_predictor is None:
            return jsonify({'success': True, 'data': {'initialized': False}})
        status = timesfm_predictor.get_status()
        return jsonify({'success': True, 'data': status})
    except Exception as e:
        logger.error(f"[TimesFM Status] Error: {e}")
        return jsonify({'success': True, 'data': {'initialized': False, 'error': str(e)}})


@bp.route('/api/sota/self-supervised/status', methods=['GET'])
def api_self_supervised_status():
    """自监督预训练状态 (优先 PyTorch，回退 NumPy)"""
    try:
        from modules.models.self_supervised import HAS_TORCH
        if HAS_TORCH:
            from modules.models.self_supervised import get_self_supervised_pretrainer
            encoder = get_self_supervised_pretrainer()
            status = encoder.get_status()
            status['model'] = 'PyTorch MaskedTimeSeries'
        else:
            from modules.models.self_supervised import get_numpy_self_supervised
            encoder = get_numpy_self_supervised()
            status = encoder.get_status()
        return jsonify({
            'success': True,
            'data': status,
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"[Self-supervised Status] Error: {e}")
        return jsonify({'success': True, 'data': {'error': str(e)}})


@bp.route('/api/sota/timesnet/status', methods=['GET'])
def api_timesnet_status():
    """TimesNet 模型状态 API"""
    try:
        app_module = _get_app_module()  # 2026-09-08 修: 原内联 '__main__' 在 run_server 启动下取不到 app 模块
        if app_module and hasattr(app_module, 'timesnet_trainer'):
            timesnet_trainer = app_module.timesnet_trainer
        else:
            timesnet_trainer = None

        if timesnet_trainer is None:
            return jsonify({
                'success': True,
                'data': {'model_type': 'TimesNet', 'trained': False},
                'timesnet_trained': False,
                'timestamp': datetime.now().isoformat(),
            })
        return jsonify({
            'success': True,
            'data': timesnet_trainer.get_status(),
            'timesnet_trained': timesnet_trainer.trained,
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"[TimesNet] 状态查询失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# Portfolio Optimization (SOTA)
# ============================================================================

@bp.route('/api/sota/cache/status', methods=['GET'])
def api_sota_cache_status():
    """SOTA 缓存状态 (stub — fallback)"""
    try:
        app_module = _get_app_module()
        if app_module and hasattr(app_module, 'sota_engine'):
            engine = app_module.sota_engine
            cache_info = None
            latest_ts = 0
            for key, info in getattr(engine, '_decision_cache', {}).items():
                if key.startswith('decision_') and info.get('timestamp', 0) > latest_ts:
                    latest_ts = info['timestamp']
                    cache_info = info
            if cache_info:
                age = __import__('time').time() - cache_info['timestamp']
                ttl = getattr(engine, '_cache_ttl', 300)
                return jsonify({
                    'success': True, 'cached': True, 'age': round(age, 1),
                    'ttl': round(ttl, 1), 'is_valid': age < ttl,
                    'timestamp': datetime.now().isoformat(),
                })
        return jsonify({
            'success': True, 'cached': False,
            'message': 'No cached decision available',
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"[SOTA Cache Stub] Error: {e}")
        return jsonify({'success': True, 'cached': False, 'timestamp': datetime.now().isoformat()})


# ── 2. /api/sota/ensemble/status ──

@bp.route('/api/sota/ensemble/status', methods=['GET'])
def api_sota_ensemble_status():
    """集成模型状态 (stub — fallback)"""
    try:
        app_module = _get_app_module()
        patchtst = _get_patchtst()
        patchtst_trained = patchtst.is_trained() if patchtst else False

        regime_status = {}
        if app_module and hasattr(app_module, 'regime_switching'):
            try:
                regime_status = app_module.regime_switching.get_status()
            except Exception:
                regime_status = {'initialized': True}

        time_llm_status = {}
        if app_module and hasattr(app_module, 'time_llm'):
            try:
                time_llm_status = app_module.time_llm.get_status()
            except Exception:
                time_llm_status = {'initialized': True}

        alpha158_info = {}
        if app_module and hasattr(app_module, 'alpha158_calculator'):
            ac = app_module.alpha158_calculator
            alpha158_info = {'n_factors': len(ac.factors) if ac else 0, 'initialized': True}

        return jsonify({
            'success': True,
            'data': {
                'patchtst': {'is_trained': patchtst_trained},
                'regime_switching': {'initialized': bool(regime_status)},
                'time_llm': {'initialized': bool(time_llm_status)},
                'alpha158': alpha158_info,
                'factor_weight_scheduler': None,
                'gnn': None,
                'cvar': None,
                'drift_advanced': None,
                'drift_monitor': None,
                'cross_market': None,
                'factor_ic': None,
            },
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"[SOTA Ensemble Stub] Error: {e}")
        return jsonify({'success': True, 'data': {}, 'timestamp': datetime.now().isoformat()})


# ── 3. /api/sota/factor-weights/status ──

@bp.route('/api/sota/factor-weights/status', methods=['GET'])
def api_sota_factor_weights_status():
    """动态因子权重状态 (stub — fallback)"""
    try:
        return jsonify({
            'success': True,
            'data': {
                'strategy': 'hybrid',
                'decay_model': 'exponential',
                'ic_decay_rate': 0.05,
                'rebalance_period': 'weekly',
            },
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"[Factor Weights Stub] Error: {e}")
        return jsonify({'success': True, 'data': {}, 'timestamp': datetime.now().isoformat()})


# ── 4. /api/sota/drift/advanced/status ──

@bp.route('/api/sota/drift/advanced/status', methods=['GET'])
def api_sota_drift_advanced_status():
    """高级概念漂移检测 (stub — fallback)"""
    try:
        # 尝试从已有 drift/status 获取数据
        drift_status = {}
        app_module = _get_app_module()
        if app_module and hasattr(app_module, 'drift_monitor'):
            try:
                drift_status = app_module.drift_monitor.get_status()
            except Exception:
                pass
        return jsonify({
            'success': True,
            'data': {
                'algorithm': drift_status.get('algorithm', 'ADWIN'),
                'drift_count': drift_status.get('drift_count', 0),
                'current_window_size': drift_status.get('window_size', 200),
                'threshold': 0.02,
            },
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"[Drift Advanced Stub] Error: {e}")
        return jsonify({'success': True, 'data': {}, 'timestamp': datetime.now().isoformat()})


# ── 5. /api/sota/alpha158/predict ──

@bp.route('/api/sota/patchtst/status', methods=['GET'])
def api_sota_patchtst_status():
    """PatchTST 模型状态 API"""
    try:
        app_module = _get_app_module()
        patchtst_integrator = getattr(app_module, 'patchtst_integrator', None) if app_module else None
        if patchtst_integrator is None:
            return jsonify({
                'success': True,
                'data': {'model_type': 'None', 'trained': False},
                'timestamp': datetime.now().isoformat(),
            })
        status = patchtst_integrator.get_status() if hasattr(patchtst_integrator, 'get_status') else {}
        return jsonify({
            'success': True,
            'data': status,
            'patchtst_trained': patchtst_integrator.is_trained() if hasattr(patchtst_integrator, 'is_trained') else False,
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"[PatchTST] 状态查询失败: {e}")
        return jsonify({'success': False, 'error': str(e)})


@bp.route('/api/sota/mamba/status', methods=['GET'])
def api_sota_mamba_status():
    """Mamba 模型状态 API"""
    try:
        app_module = _get_app_module()
        mamba_hft = getattr(app_module, 'mamba_hft', None) if app_module else None
        if mamba_hft is None:
            return jsonify({
                'success': True,
                'data': {'model_type': 'None', 'trained': False},
                'timestamp': datetime.now().isoformat(),
            })
        return jsonify({
            'success': True,
            'data': mamba_hft.get_status() if hasattr(mamba_hft, 'get_status') else {},
            'mamba_trained': mamba_hft.is_trained() if hasattr(mamba_hft, 'is_trained') else False,
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"[MambaHFT] 状态查询失败: {e}")
        return jsonify({'success': False, 'error': str(e)})


@bp.route('/api/sota/diffusion/status', methods=['GET'])
def api_sota_diffusion_status():
    """Diffusion 模型状态 API"""
    try:
        app_module = _get_app_module()
        diffusion_predictor = getattr(app_module, 'diffusion_predictor', None) if app_module else None
        if diffusion_predictor is None:
            return jsonify({
                'success': True,
                'data': {'model_type': 'None', 'trained': False},
                'timestamp': datetime.now().isoformat(),
            })
        return jsonify({
            'success': True,
            'data': diffusion_predictor.get_status() if hasattr(diffusion_predictor, 'get_status') else {},
            'diffusion_trained': diffusion_predictor.is_trained() if hasattr(diffusion_predictor, 'is_trained') else False,
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"[Diffusion] 状态查询失败: {e}")
        return jsonify({'success': False, 'error': str(e)})


@bp.route('/api/sota/conformal/status', methods=['GET'])
def api_sota_conformal_status():
    """Conformal Prediction 状态 API"""
    try:
        from modules.models.conformal_predictor import get_conformal_predictor
        cp = get_conformal_predictor()
        if cp is None:
            return jsonify({
                'success': True,
                'data': {'model': 'Conformal', 'calibrated': False},
                'timestamp': datetime.now().isoformat(),
            })
        status = cp.get_status() if hasattr(cp, 'get_status') else {'model': 'Conformal', 'calibrated': getattr(cp, 'calibrated', False)}
        return jsonify({'success': True, 'data': status, 'timestamp': datetime.now().isoformat()})
    except Exception as e:
        logger.error(f"[Conformal Status] Error: {e}")
        return jsonify({'success': True, 'data': {'error': str(e)}, 'timestamp': datetime.now().isoformat()})


@bp.route('/api/sota/gnn/status', methods=['GET'])
def api_sota_gnn_status():
    """GNN 状态 API"""
    try:
        app_module = _get_app_module()
        gnn_predictor = getattr(app_module, 'gnn_predictor', None) if app_module else None
        if gnn_predictor is None:
            return jsonify({'success': False, 'error': 'GNN 预测器未初始化'}, 503)
        return jsonify({
            'success': True,
            'data': gnn_predictor.get_status() if hasattr(gnn_predictor, 'get_status') else {},
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"[GNN] Error: {e}")
        return jsonify({'success': False, 'error': str(e)}, 500)
