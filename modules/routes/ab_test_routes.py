"""
A/B Test Routes - /api/ab-tests/*

Extracted from app.py (lines 1325-1422).
"""

from flask import Blueprint, request, jsonify

from modules.logger import logger

bp = Blueprint('ab_test', __name__)


@bp.route('/api/ab-tests')
def api_ab_tests():
    """列出所有 A/B 测试实验"""
    try:
        from modules.ab_test_manager import get_ab_test_manager
        manager = get_ab_test_manager()
        return jsonify({
            'success': True,
            'data': manager.list_tests(),
        })
    except Exception as e:
        logger.error(f"[ABTest] 错误: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/ab-tests', methods=['POST'])
def api_ab_test_create():
    """创建新的 A/B 测试实验"""
    try:
        from modules.ab_test_manager import get_ab_test_manager
        data = request.get_json(silent=True, force=True) or {}
        manager = get_ab_test_manager()
        experiment = manager.create_test(
            name=data.get('name', ''),
            stock_code=data.get('stock_code', ''),
            variant_a=data.get('variant_a', ''),
            variant_b=data.get('variant_b', ''),
        )
        return jsonify({
            'success': True,
            'data': experiment,
        })
    except Exception as e:
        logger.error(f"[ABTest] 创建失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/ab-tests/<experiment_id>/variant')
def api_ab_test_variant(experiment_id):
    """获取实验的变体信息"""
    try:
        from modules.ab_test_manager import get_ab_test_manager
        manager = get_ab_test_manager()
        return jsonify({
            'success': True,
            'data': manager.get_experiment(experiment_id),
        })
    except Exception as e:
        logger.error(f"[ABTest] 获取变体失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/ab-tests/<experiment_id>/result', methods=['POST'])
def api_ab_test_record_result(experiment_id):
    """记录 A/B 测试结果"""
    try:
        from modules.ab_test_manager import get_ab_test_manager
        data = request.get_json(silent=True, force=True) or {}
        manager = get_ab_test_manager()
        manager.record_result(
            experiment_id=experiment_id,
            variant=data.get('variant', 'A'),
            result=data.get('result', 0),
        )
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"[ABTest] 记录结果失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/ab-tests/<experiment_id>/winner')
def api_ab_test_winner(experiment_id):
    """获取 A/B 测试胜者"""
    try:
        from modules.ab_test_manager import get_ab_test_manager
        manager = get_ab_test_manager()
        winner = manager.get_winner(experiment_id)
        return jsonify({
            'success': True,
            'data': winner,
        })
    except Exception as e:
        logger.error(f"[ABTest] 获取胜者失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
