"""
Online Learning Routes - /api/online-learning/*

在线学习与 EWC 防遗忘路由 (真实模块实例)
"""

from flask import Blueprint, request, jsonify

from modules.logger import logger
from modules.online_learning_ewc import get_online_learning_ewc

bp = Blueprint('online_learning', __name__)


@bp.route('/api/online-learning/status')
def api_online_learning_status():
    """获取在线学习 + EWC 状态"""
    try:
        manager = get_online_learning_ewc()
        return jsonify({'success': True, 'data': manager.get_status()})
    except Exception as e:
        logger.error(f"[OnlineLearning] 错误: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/online-learning/accuracy/<stock_code>')
def api_online_learning_accuracy(stock_code):
    """获取指定股票的预测准确率 — 从 OnlineLearningManager 读取真实数据"""
    try:
        from modules.online_learning import get_online_learner
        manager = get_online_learner()
        accuracy_data = manager.get_accuracy(stock_code, window=100)
        drift_alerts = manager.get_alerts(limit=10)
        # 过滤出该股票的告警
        stock_alerts = [a for a in drift_alerts if a.get('stock_code') == stock_code]
        return jsonify({
            'success': True,
            'data': {
                **accuracy_data,
                'drift_alerts': stock_alerts,
                'n_stock_alerts': len(stock_alerts),
            },
        })
    except Exception as e:
        logger.error(f"[OnlineLearning] 错误: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/online-learning/alerts')
def api_online_learning_alerts():
    """获取漂移告警 — 从 OnlineLearningManager 读取真实告警"""
    try:
        from modules.online_learning import get_online_learner
        manager = get_online_learner()
        alerts = manager.get_alerts(limit=50)
        ewc = get_online_learning_ewc()
        return jsonify({
            'success': True,
            'data': {
                'alerts': alerts,
                'n_alerts': len(alerts),
                'ewc_buffer_size': len(ewc._data_buffer) if hasattr(ewc, '_data_buffer') else 0,
            },
        })
    except Exception as e:
        logger.error(f"[OnlineLearning] 错误: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/online-learning/record', methods=['POST'])
def api_online_learning_record():
    """记录预测结果 (用于在线学习)"""
    try:
        data = request.get_json(silent=True, force=True) or {}
        manager = get_online_learning_ewc()
        if data.get('prediction') is not None and data.get('actual') is not None:
            # 2026-09-02: Web UI 单票观测短格式 {stock_code, prediction, actual}
            # → 转成 add_data 期望的 features/labels 向量。
            # 旧版只读 features/labels 键, 页面传来的体永远为空 → 恒 "已记录" 但 0 入库 (谎报成功)。
            features = [float(data['prediction'])]
            labels = [float(data['actual'])]
            metadata = {'source': 'web_ui', 'stock_code': data.get('stock_code', '')}
        else:
            features = data.get('features', [])
            labels = data.get('labels', [])
            metadata = data.get('metadata', {})
        manager.add_data(features=features, labels=labels, metadata=metadata)
        return jsonify({'success': True, 'buffer_size': len(manager._data_buffer)})
    except Exception as e:
        logger.error(f"[OnlineLearning] 记录错误: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
