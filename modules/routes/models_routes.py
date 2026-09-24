from flask import Blueprint, request, jsonify
from datetime import datetime
from modules.logger import logger

bp = Blueprint('models', __name__)


# ============================================================================
# 模型元数据管理 API
# ============================================================================

@bp.route('/api/models/registry')
def api_model_registry():
    """获取模型注册表摘要"""
    try:
        from modules.model_registry import get_model_registry
        registry = get_model_registry()
        return jsonify({
            'success': True,
            'data': registry.get_summary(),
        })
    except Exception as e:
        logger.error(f"[ModelRegistry] 错误: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/models/registry/scan', methods=['POST'])
def api_model_registry_scan():
    """扫描并注册新发现的模型文件"""
    try:
        from modules.model_registry import get_model_registry
        registry = get_model_registry()
        registered = registry.scan_directory()
        return jsonify({
            'success': True,
            'registered': registered,
            'count': len(registered),
        })
    except Exception as e:
        logger.error(f"[ModelRegistry] 扫描错误: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/models/<name>/metrics', methods=['POST'])
# DEPRECATED (2026-09-23 方案A): 项目内零调用点 (前端 40 fetch 全扫+scripts 零);
# 非删 (外部链 sina_push/Telegram 未全核); 若确认无外部消费再摘 (教训 8 先核查后动)
def api_model_update_metrics(name):
    """更新模型指标"""
    try:
        from modules.model_registry import get_model_registry
        registry = get_model_registry()
        data = request.get_json(silent=True, force=True) or {}
        metrics = data.get('metrics', {})
        success = registry.update_metrics(name, metrics)
        return jsonify({'success': success})
    except Exception as e:
        logger.error(f"[ModelRegistry] 更新指标错误: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
