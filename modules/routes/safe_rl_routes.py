#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Safe RL + 贝叶斯不确定性 API 路由 (2026-09-02 补链)

背景: 2026-07-27 路由清理时把 /api/safe-rl/status 和 /api/uncertainty/status
从 new_routes.py 移除, 注释标了 "→ safe_rl_routes.py / uncertainty_routes.py",
但新文件从未创建 → 两路由 404 两个多月, 前端 "Safe RL/贝叶斯" 页永远卡加载中。
本文件补回路由并接通真实模块单例 (safe_rl_constraint / bayesian_uncertainty)。

端点:
    GET /api/safe-rl/status      - Safe RL 约束层状态 (约束配置 + 组合状态 + 最近约束检查)
    GET /api/uncertainty/status  - 贝叶斯不确定性估计摘要
"""

from flask import Blueprint, jsonify

from modules.logger import logger

bp = Blueprint('safe_rl_routes', __name__)


@bp.route('/api/safe-rl/status', methods=['GET'])
def api_safe_rl_status():
    """Safe RL 约束层状态"""
    try:
        from modules.safe_rl_constraint import get_safe_rl_layer
        layer = get_safe_rl_layer()
        return jsonify({'success': True, 'data': layer.get_status()})
    except Exception as e:
        logger.error(f"[SafeRL] status 失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/uncertainty/status', methods=['GET'])
def api_uncertainty_status():
    """贝叶斯不确定性估计摘要"""
    try:
        from modules.bayesian_uncertainty import get_bayesian_uncertainty
        engine = get_bayesian_uncertainty()
        return jsonify({'success': True, 'data': engine.get_summary()})
    except Exception as e:
        logger.error(f"[Uncertainty] status 失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
