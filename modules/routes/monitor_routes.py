"""
monitor_routes.py - 监控告警 API

提供请求追踪指标、健康检查、慢请求告警等 API。
"""

from flask import Blueprint, jsonify
from datetime import datetime

from modules.logger import logger
from .monitor import get_request_tracker, get_health_checker

bp = Blueprint('monitor', __name__)


@bp.route('/api/monitor/metrics')
def api_monitor_metrics():
    """获取监控指标"""
    try:
        tracker = get_request_tracker()
        metrics = tracker.get_metrics()
        return jsonify({
            'success': True,
            'data': metrics,
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"[Monitor] Metrics error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/monitor/reset', methods=['POST'])
def api_monitor_reset():
    """重置监控统计"""
    try:
        tracker = get_request_tracker()
        tracker.reset()
        return jsonify({
            'success': True,
            'message': '监控统计已重置',
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"[Monitor] Reset error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/monitor/health')
def api_monitor_health():
    """综合健康检查"""
    try:
        checker = get_health_checker()
        result = checker.run_all()
        status_code = 200 if result['healthy'] else 503
        return jsonify({
            'success': result['healthy'],
            **result,
        }), status_code
    except Exception as e:
        logger.error(f"[Monitor] Health check error: {e}")
        return jsonify({
            'success': False,
            'healthy': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat(),
        }), 500
