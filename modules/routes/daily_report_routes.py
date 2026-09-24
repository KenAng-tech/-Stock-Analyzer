# -*- coding: utf-8 -*-
"""
daily_report_routes.py — 每日晨报研报 API (2026-09-09)

GET  /api/daily_report  — 最新晨报 + 生成状态 (前端首屏/轮询)
POST /api/daily_report  — 手动触发晨报链 (后台线程, 立即返回)

链体在 modules/daily_report_service (08:00 自动 + 本端点手动, 同一把
幂等锁); 无 self-HTTP 自调 (09-02 教训), 端点只转发 service 方法。
"""

import threading

from flask import Blueprint, jsonify

from modules.daily_report_service import daily_report_service
from modules.logger import logger

bp = Blueprint('daily_report', __name__)


@bp.route('/api/daily_report', methods=['GET'])
def get_daily_report():
    """最新晨报 (含 source_errors / llm_degraded 标记)"""
    try:
        latest = daily_report_service.get_latest()
        return jsonify({
            'success': True,
            'generating': daily_report_service.is_generating(),
            'report': latest,
        })
    except Exception as e:
        logger.error(f"[DailyReport] 晨报查询失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/daily_report', methods=['POST'])
def trigger_daily_report():
    """手动触发晨报生成 (后台线程执行, 重入返回 409)"""
    try:
        if daily_report_service.is_generating():
            return jsonify({'success': False, 'error': '晨报链正在生成中'}), 409
        threading.Thread(
            target=lambda: daily_report_service.generate(trigger='manual'),
            daemon=True, name='daily-report-manual'
        ).start()
        return jsonify({'success': True, 'status': 'started',
                        'msg': '晨报链已启动 (约 40-150s, 含四票情报链 25s 预算), '
                               '轮询 GET /api/daily_report'})
    except Exception as e:
        logger.error(f"[DailyReport] 手动触发失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
