"""
xai_routes.py - /api/xai/*

XAI / 不确定性量化 / Safe RL 路由 (合并 safe_rl_routes + uncertainty_routes)
"""

from flask import Blueprint, jsonify
from modules.logger import logger

bp = Blueprint('xai', __name__)


@bp.route('/api/xai/uncertainty/status', methods=['GET'])
def api_uncertainty_status():
    """贝叶斯不确定性量化状态"""
    try:
        from modules.bayesian_uncertainty import get_bayesian_uncertainty
        bayesian = get_bayesian_uncertainty()
        return jsonify({'success': True, 'data': bayesian.get_summary()})
    except Exception as e:
        logger.error(f"[Bayesian] 状态查询失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/xai/safe-rl/status', methods=['GET'])
def api_safe_rl_status():
    """Safe RL 安全约束状态"""
    try:
        from modules.safe_rl_constraint import get_safe_rl_layer
        layer = get_safe_rl_layer()
        return jsonify({'success': True, 'data': layer.get_status()})
    except Exception as e:
        logger.error(f"[SafeRL] 状态查询失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
