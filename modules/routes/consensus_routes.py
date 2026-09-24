"""
consensus_routes.py - /api/consensus/*

多智能体共识决策路由 (真实模块实例)
"""

from flask import Blueprint, request, jsonify

from modules.logger import logger
from modules.multi_agent_consensus import get_consensus_engine

bp = Blueprint('consensus', __name__)


@bp.route('/api/consensus/weights', methods=['GET'])
def api_consensus_weights():
    """获取智能体权重配置"""
    try:
        engine = get_consensus_engine()
        summary = engine.get_summary()
        return jsonify({'success': True, 'data': summary})
    except Exception as e:
        logger.error(f"[Consensus] 权重查询失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/consensus/decide', methods=['GET'])
def api_consensus_decide():
    """多智能体共识决策"""
    try:
        stock_code = request.args.get('stock_code', 'sz300620')
        engine = get_consensus_engine()
        result = engine.decide(stock_code)
        return jsonify({'success': True, 'data': result.to_dict()})
    except Exception as e:
        logger.error(f"[Consensus] 共识决策失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/consensus/history', methods=['GET'])
def api_consensus_history():
    """共识决策历史"""
    try:
        stock_code = request.args.get('stock_code', 'sz300620')
        engine = get_consensus_engine()
        history = engine.get_history(limit=10)
        return jsonify({'success': True, 'data': {'stock_code': stock_code, 'history': history, 'count': len(history)}})
    except Exception as e:
        logger.error(f"[Consensus] 历史查询失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
