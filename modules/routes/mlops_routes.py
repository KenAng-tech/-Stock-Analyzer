"""
mlops_routes.py — MLOps 管道 API 路由

提供 MLOps 编排器的 HTTP API 接口:
- GET /api/mlops/status — 编排器状态
- GET /api/mlops/check-retrain/<stock_code> — 检查是否需要重训练
- POST /api/mlops/trigger-retrain — 触发重训练
- POST /api/mlops/deploy — 部署模型
- GET /api/mlops/models — 列出已注册模型
- GET /api/mlops/drift-status — 概念漂移状态
"""

from flask import Blueprint, jsonify, request

from modules.logger import logger

bp = Blueprint('mlops', __name__)


def _get_orchestrator():
    """获取 MLOps 编排器单例"""
    from modules.mlops_orchestrator import mlops_orchestrator
    return mlops_orchestrator


@bp.route('/api/mlops/status')
def mlops_status():
    """获取编排器整体状态"""
    orch = _get_orchestrator()
    return jsonify(orch.get_status())


@bp.route('/api/mlops/check-retrain/<stock_code>')
def check_retrain(stock_code):
    """检查指定股票是否需要重训练"""
    orch = _get_orchestrator()
    result = orch.check_retrain_trigger(stock_code)
    return jsonify(result)


@bp.route('/api/mlops/trigger-retrain', methods=['POST'])
def trigger_retrain():
    """触发重训练"""
    data = request.get_json() or {}
    stock_code = data.get('stock_code', '')
    model_name = data.get('model_name')

    if not stock_code:
        return jsonify({'error': '缺少 stock_code'}), 400

    orch = _get_orchestrator()
    result = orch.trigger_retrain(stock_code, model_name)

    if result.get('success'):
        return jsonify(result)
    else:
        return jsonify(result), 500


@bp.route('/api/mlops/deploy', methods=['POST'])
def deploy_model():
    """部署指定模型"""
    data = request.get_json() or {}
    stock_code = data.get('stock_code', '')
    version = data.get('version')

    if not stock_code:
        return jsonify({'error': '缺少 stock_code'}), 400

    orch = _get_orchestrator()
    result = orch.deploy_model(stock_code, version)
    return jsonify(result)


@bp.route('/api/mlops/models')
def list_models():
    """列出已注册的模型 (ModelMetadata → dict 序列化)"""
    try:
        orch = _get_orchestrator()
        orch._ensure_modules()

        if not orch._model_registry:
            return jsonify({'error': '模型注册表未加载'}), 503

        models = orch._model_registry.list_models()
        return jsonify({'models': [m.to_dict() for m in models]})
    except Exception as e:
        logger.error(f"[MLOps] 模型列表查询失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/mlops/drift-status')
def drift_status():
    """获取概念漂移检测状态"""
    orch = _get_orchestrator()
    orch._ensure_modules()

    if not orch._drift_detector:
        # 漂移检测器需要 model 参数才能初始化，返回降级状态而非 503
        return jsonify({
            'drift_detector': False,
            'drift_detected': False,
            'drift_count': 0,
            'adwin_window_size': 0,
            'adwin_n_splits': 0,
            'should_retrain': False,
            'note': '漂移检测器未初始化（需要训练后的模型）',
        })

    status = orch._drift_detector.get_status()
    return jsonify(status)


# ── 金丝雀部署 ──────────────────────────────────────────────

@bp.route('/api/mlops/canary/deploy', methods=['POST'])
def canary_deploy():
    """启动金丝雀部署"""
    try:
        data = request.get_json() or {}
        stock_code = data.get('stock_code', '')
        new_version = data.get('new_version', '')
        old_version = data.get('old_version', 'current')

        if not stock_code or not new_version:
            return jsonify({'success': False, 'error': '缺少 stock_code 或 new_version'}), 400

        from modules.mlops_orchestrator import get_canary_deploy
        canary = get_canary_deploy()

        result = canary.deploy_canary(stock_code, new_version, old_version)

        if result.get('success'):
            return jsonify(result)
        else:
            return jsonify(result), 409

    except Exception as e:
        logger.error(f"[CanaryDeploy] 启动失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/mlops/canary/record', methods=['POST'])
def canary_record():
    """记录预测结果 (用于性能评估)"""
    try:
        data = request.get_json() or {}
        stock_code = data.get('stock_code', '')
        model_version = data.get('model_version', '')
        predicted = data.get('predicted', 0)
        actual = data.get('actual', 0)

        if not stock_code or not model_version:
            return jsonify({'success': False, 'error': '缺少 stock_code 或 model_version'}), 400

        from modules.mlops_orchestrator import get_canary_deploy
        canary = get_canary_deploy()

        result = canary.record_prediction(stock_code, model_version, predicted, actual)

        if result.get('success'):
            return jsonify(result)
        else:
            return jsonify(result), 400

    except Exception as e:
        logger.error(f"[CanaryDeploy] 记录失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/mlops/canary/status')
def canary_status():
    """获取金丝雀部署状态"""
    try:
        stock_code = request.args.get('stock_code')

        from modules.mlops_orchestrator import get_canary_deploy
        canary = get_canary_deploy()

        result = canary.get_status(stock_code)
        return jsonify(result)

    except Exception as e:
        logger.error(f"[CanaryDeploy] 状态查询失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/mlops/canary/end', methods=['POST'])
def canary_end():
    """手动结束金丝雀部署"""
    try:
        data = request.get_json() or {}
        stock_code = data.get('stock_code', '')
        reason = data.get('reason', 'manual')

        if not stock_code:
            return jsonify({'success': False, 'error': '缺少 stock_code'}), 400

        from modules.mlops_orchestrator import get_canary_deploy
        canary = get_canary_deploy()

        result = canary.end_deployment(stock_code, reason)

        if result.get('success'):
            return jsonify(result)
        else:
            return jsonify(result), 400

    except Exception as e:
        logger.error(f"[CanaryDeploy] 结束失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
