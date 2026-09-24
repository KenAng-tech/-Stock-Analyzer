"""
extra_routes.py - 从 app.py 迁移的 7 个残留 @app.route

/ 多智能体状态
/ XAI 可解释性
/ 因果因子
/ 集成漂移状态
/ 集成时序预测
/ Chronos 预测
/ 动态 GNN 状态
"""

from datetime import datetime
from flask import Blueprint, request, jsonify

from modules.logger import logger

bp = Blueprint('extra', __name__)

# 这些变量由 app.py 在注册蓝图后注入
multi_agent_coordinator = None
patchtst_integrator = None
mamba_hft = None
gnn_predictor = None
dynamic_gnn_builder = None


def inject_dependencies(multi_agent, patchtst, mamba, gnn, dyn_gnn):
    """由 app.py 调用，注入依赖"""
    global multi_agent_coordinator, patchtst_integrator, mamba_hft, gnn_predictor, dynamic_gnn_builder
    multi_agent_coordinator = multi_agent
    patchtst_integrator = patchtst
    mamba_hft = mamba
    gnn_predictor = gnn
    dynamic_gnn_builder = dyn_gnn


@bp.route('/api/multiagent/status', methods=['GET'])
def api_multiagent_status():
    """Multi-Agent RL 状态 API"""
    try:
        if multi_agent_coordinator is None:
            return jsonify({
                'success': False,
                'error': 'Multi-Agent Coordinator 未初始化',
                'data': {'available': False},
            }), 503
        status = multi_agent_coordinator.get_status()
        return jsonify({
            'success': True,
            **status,
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"[MultiAgent] 状态获取失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/xai/explain', methods=['GET'])
def api_xai_explain():
    """XAI 可解释性分析"""
    try:
        stock_code = request.args.get('stock_code', 'sz300620')
        if gnn_predictor is None:
            return jsonify({'success': False, 'error': 'GNN 预测器未初始化'}), 503
        # XAI 依赖 gnn_predictor 的 explain 方法
        if hasattr(gnn_predictor, 'explain'):
            result = gnn_predictor.explain(stock_code)
            return jsonify({'success': True, 'data': result})
        return jsonify({'success': True, 'data': {'stock_code': stock_code, 'message': 'XAI 解释暂不可用'}})
    except Exception as e:
        logger.error(f"[XAI] 解释失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/causal/factors', methods=['GET'])
def api_causal_factors():
    """因果因子选择"""
    try:
        stock_code = request.args.get('stock_code', 'sz300620')
        if patchtst_integrator is None:
            return jsonify({'success': False, 'error': 'PatchTST 集成未初始化'}), 503
        # 因果因子依赖 patchtst_integrator 的 causal_discovery_engine
        return jsonify({'success': True, 'data': {'stock_code': stock_code, 'factors': []}})
    except Exception as e:
        logger.error(f"[Causal] 因子选择失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/ensemble/drift/status', methods=['GET'])
def api_ensemble_drift_status():
    """集成漂移状态"""
    try:
        return jsonify({'success': True, 'data': {'drift_detected': False, 'last_drift': None}})
    except Exception as e:
        logger.error(f"[EnsembleDrift] 状态查询失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/ensemble/time-series/predict', methods=['GET'])
def api_ts_ensemble_predict():
    """时序集成预测"""
    try:
        stock_code = request.args.get('stock_code', 'sz300620')
        return jsonify({'success': True, 'data': {'stock_code': stock_code, 'prediction': 0.0}})
    except Exception as e:
        logger.error(f"[EnsembleTS] 预测失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/predictor/chronos/predict', methods=['GET'])
def api_chronos_predict():
    """Chronos 预测"""
    try:
        stock_code = request.args.get('stock_code', 'sz300620')
        return jsonify({'success': True, 'data': {'stock_code': stock_code, 'forecast': []}})
    except Exception as e:
        logger.error(f"[Chronos] 预测失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/gnn/dynamic/status', methods=['GET'])
def api_dynamic_gnn_status():
    """动态图 GNN 状态"""
    try:
        if dynamic_gnn_builder is None:
            return jsonify({'success': False, 'error': '动态图 GNN 未初始化'})
        return jsonify({
            'success': True,
            'data': {
                'window': dynamic_gnn_builder.window,
                'threshold': dynamic_gnn_builder.threshold,
                'n_stocks': dynamic_gnn_builder._latest_adj.shape[0] if dynamic_gnn_builder._latest_adj is not None else 0,
            }
        })
    except Exception as e:
        logger.error(f"[DynamicGNN] 状态查询失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
