"""
Memory Routes - /api/memory/*

Extracted from app.py (lines 4875-4901).
"""

from flask import Blueprint, jsonify
from datetime import datetime

from modules.logger import logger

bp = Blueprint('memory', __name__)


@bp.route('/api/memory/status')
def api_memory_status():
    """获取内存监控状态"""
    try:
        from modules.memory_manager import get_memory_monitor
        monitor = get_memory_monitor()
        return jsonify({'data': monitor.get_status()})
    except Exception as e:
        logger.error(f"[API] Memory status error: {e}")
        return jsonify({'error': str(e)}), 500


@bp.route('/api/memory/gc', methods=['POST'])
def api_memory_gc():
    """强制垃圾回收"""
    try:
        from modules.memory_manager import get_memory_monitor
        monitor = get_memory_monitor()
        result = monitor.force_gc()
        return jsonify(result)
    except Exception as e:
        logger.error(f"[API] Memory GC error: {e}")
        return jsonify({'error': str(e)}), 500
