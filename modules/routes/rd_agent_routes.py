#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
RD-Agent 因子挖掘 API 路由 (2026-08-13 新增)

端点:
    GET  /api/rdagent/status          - 挖掘器状态
    POST /api/rdagent/mine            - 触发一轮挖掘
    GET  /api/rdagent/factors         - 已挖掘因子列表
    POST /api/rdagent/validate/<name> - 手动验证因子
"""

from datetime import datetime
from flask import Blueprint, jsonify, request
from modules.logger import logger

bp = Blueprint('rd_agent_routes', __name__)


@bp.route('/api/rdagent/status', methods=['GET'])
def rdagent_status():
    """获取挖掘器状态"""
    try:
        from modules.rd_agent_miner import get_rd_agent_miner
        miner = get_rd_agent_miner()
        return jsonify(miner.get_status())
    except Exception as e:
        logger.error(f"[RDAgent] status 失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/rdagent/mine', methods=['POST'])
def rdagent_mine():
    """触发一轮挖掘"""
    try:
        from modules.rd_agent_miner import get_rd_agent_miner
        miner = get_rd_agent_miner()

        # 解析参数
        data = request.get_json(silent=True) or {}
        categories = data.get('categories')  # None = 全部

        result = miner.mine_round(categories=categories)
        return jsonify(result)
    except Exception as e:
        logger.error(f"[RDAgent] mine 失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/rdagent/factors', methods=['GET'])
def rdagent_factors():
    """获取已挖掘因子"""
    try:
        from modules.rd_agent_miner import get_rd_agent_miner
        miner = get_rd_agent_miner()

        status_filter = request.args.get('status')  # validated/rejected/pending
        return jsonify(miner.get_factors(status=status_filter))
    except Exception as e:
        logger.error(f"[RDAgent] factors 失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/rdagent/validate/<name>', methods=['POST'])
def rdagent_validate(name):
    """手动验证因子"""
    try:
        from modules.rd_agent_miner import get_rd_agent_miner
        miner = get_rd_agent_miner()

        with miner._lock:
            h = miner._hypotheses.get(name)
        if h is None:
            return jsonify({'success': False, 'error': f'因子 {name} 不存在'}), 404

        ic, icir, err = miner.validate_factor(h)
        h.ic = ic
        h.icir = icir
        if err is not None:
            h.status = 'error'  # exec 失败/面板不足 ≠ 真无预测力, 保留重试验证资格
        else:
            h.status = 'validated' if abs(ic) >= miner._status.ic_threshold else 'rejected'
        h.validated_at = datetime.now().isoformat()
        miner._save_to_db(h)

        return jsonify({
            'success': True,
            'data': {
                'name': h.name,
                'ic': ic,
                'icir': icir,
                'status': h.status,
                'error': err,
            }
        })
    except Exception as e:
        logger.error(f"[RDAgent] validate 失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500