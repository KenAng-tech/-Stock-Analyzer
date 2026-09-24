"""
Training Routes - /api/training/*, /api/ml/*, /api/sota/train/*

Extracted from app.py (lines 2681-2775, 5557-5641).
Dependencies: model_training_scheduler, sota_scheduler, ml_predictor (from app.py module level)
"""

from flask import Blueprint, request, jsonify
from datetime import datetime
import numpy as np

from modules.logger import logger

bp = Blueprint('training', __name__)

# These are imported from app.py module level
# model_training_scheduler, sota_scheduler, ml_predictor


@bp.route('/api/ml/train', methods=['POST'])
def api_ml_train():
    """手动触发 ML 模型训练"""
    try:
        import sys
        app_module = sys.modules.get('__main__')
        if app_module and hasattr(app_module, 'model_training_scheduler'):
            scheduler = app_module.model_training_scheduler
        else:
            from modules.ml_predictor import model_training_scheduler
            scheduler = model_training_scheduler

        stock_code = 'sz300620'
        if request.is_json and request.json:
            stock_code = request.json.get('stock_code', 'sz300620')

        result = scheduler.force_train(stock_code)
        return jsonify({
            'success': result.get('success', False),
            'message': result.get('message', result.get('error', '')),
            'data': {k: v for k, v in result.items() if k not in ('message', 'error')},
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"ML train error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)})


@bp.route('/api/ml/train/status')
def api_ml_train_status():
    """获取 ML 模型训练状态"""
    try:
        import sys
        app_module = sys.modules.get('__main__')
        if app_module and hasattr(app_module, 'model_training_scheduler'):
            scheduler = app_module.model_training_scheduler
        else:
            from modules.ml_predictor import model_training_scheduler
            scheduler = model_training_scheduler

        status = scheduler.get_status()
        return jsonify({
            'success': True,
            'data': status,
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"ML train status error: {e}")
        return jsonify({'success': False, 'error': str(e)})


@bp.route('/api/sota/train/trigger', methods=['POST'])
def api_sota_train_trigger():
    """手动触发 SOTA 模型训练"""
    try:
        import sys
        app_module = sys.modules.get('__main__')
        if app_module and hasattr(app_module, 'sota_scheduler'):
            scheduler = app_module.sota_scheduler
        else:
            from modules.sota_training_scheduler import sota_scheduler
            scheduler = sota_scheduler

        stock_code = 'sz300620'
        if request.is_json and request.json:
            stock_code = request.json.get('stock_code', 'sz300620')

        # 2026-09-10 断链修复: 旧版在 Flask worker 里同步跑整链 (8 段+GNN,
        # 10min+ 挂起; 且长期占 scheduler._lock 会挡 23:00 定时链误报"训练中")
        # → 后台 daemon 线程执行; 幂等锁在 force_train 内 (重入返回"训练中")
        import threading

        def _bg_sota_train():
            try:
                scheduler.force_train(stock_code)
            except Exception as e:
                logger.error(f"[SOTATrain] 后台训练链异常: {e}")

        threading.Thread(target=_bg_sota_train, daemon=True,
                         name='sota-train-manual').start()
        return jsonify({
            'success': True, 'status': 'started',
            'message': f'SOTA 训练链已启动 (后台, 约 10-25min): {stock_code}',
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"SOTA train error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)})


@bp.route('/api/sota/train/status')
def api_sota_train_status():
    """获取 SOTA 模型训练状态"""
    try:
        import sys
        app_module = sys.modules.get('__main__')
        if app_module and hasattr(app_module, 'sota_scheduler'):
            scheduler = app_module.sota_scheduler
        else:
            from modules.sota_training_scheduler import sota_scheduler
            scheduler = sota_scheduler

        status = scheduler.get_status()
        return jsonify({
            'success': True,
            'data': status,
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"SOTA train status error: {e}")
        return jsonify({'success': False, 'error': str(e)})


@bp.route('/api/ml/model/report')
def api_ml_model_report():
    """获取 ML 模型报告（含新鲜度检查）"""
    try:
        import sys
        app_module = sys.modules.get('__main__')
        if app_module and hasattr(app_module, 'ml_predictor'):
            predictor = app_module.ml_predictor
        else:
            from modules.ml_predictor import ml_predictor
            predictor = ml_predictor

        report = predictor.get_model_report()
        return jsonify({
            'success': True,
            'data': report,
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"ML model report error: {e}")
        return jsonify({'success': False, 'error': str(e)})


@bp.route('/api/training/pipeline/status', methods=['GET'])
def api_training_pipeline_status():
    """训练管线状态"""
    try:
        from modules.model_training_pipeline import get_training_pipeline
        pipeline = get_training_pipeline()
        report = pipeline.get_report()
        return jsonify({
            'success': True,
            'data': {
                'has_report': report is not None,
                'report': report.to_dict() if report else None,
            },
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"[Training Pipeline] Error: {e}")
        return jsonify({'success': True, 'data': {'error': str(e)}})


@bp.route('/api/training/pipeline/run', methods=['POST'])
def api_training_pipeline_run():
    """运行训练评估管线"""
    try:
        from modules.model_training_pipeline import TrainingPipeline
        body = request.get_json() or {}
        model_name = body.get('model_name', 'web_trained_model')
        train_ratio = body.get('train_ratio', 0.6)
        val_ratio = body.get('val_ratio', 0.2)
        test_ratio = body.get('test_ratio', 0.2)

        # 生成模拟数据 (实际使用时应替换为真实数据)
        np.random.seed(body.get('seed', 42))
        n_samples = body.get('n_samples', 500)
        n_features = body.get('n_features', 12)
        X = np.random.randn(n_samples, n_features) * 0.1
        y = np.cumsum(np.random.randn(n_samples) * 0.01) + 100
        dates = [f'2024-{((i // 21) % 12) + 1:02d}-{((i % 21) + 1):02d}' for i in range(n_samples)]

        pipeline = TrainingPipeline(
            model_name=model_name,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )
        report = pipeline.run(X, y, dates)

        return jsonify({
            'success': True,
            'data': report.to_dict(),
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"[Training Pipeline Run] Error: {e}")
        return jsonify({'success': True, 'data': {'error': str(e)}})


@bp.route('/api/training/metrics', methods=['GET'])
def api_training_metrics():
    """评估指标说明"""
    from modules.model_training_pipeline import (
        compute_rmse, compute_mae, compute_mape, compute_direction_accuracy,
        compute_sharpe_ratio, compute_max_drawdown, compute_r_squared,
    )
    return jsonify({
        'success': True,
        'data': {
            'metrics': [
                {'name': 'rmse', 'desc': '均方根误差', 'formula': 'sqrt(mean((y_true - y_pred)^2))'},
                {'name': 'mae', 'desc': '平均绝对误差', 'formula': 'mean(|y_true - y_pred|)'},
                {'name': 'mape', 'desc': '平均绝对百分比误差', 'formula': 'mean(|(y_true - y_pred) / y_true|) * 100'},
                {'name': 'direction_accuracy', 'desc': '方向准确率', 'formula': 'mean(sign(diff(y_true)) == sign(diff(y_pred)))'},
                {'name': 'sharpe_ratio', 'desc': '夏普比率', 'formula': 'mean(excess_returns) / std(returns) * sqrt(252)'},
                {'name': 'max_drawdown', 'desc': '最大回撤', 'formula': 'max((running_max - equity) / running_max)'},
                {'name': 'r_squared', 'desc': 'R² 决定系数', 'formula': '1 - ss_res / ss_tot'},
                {'name': 'win_rate', 'desc': '胜率', 'formula': 'count(returns > 0) / len(returns)'},
                {'name': 'profit_factor', 'desc': '盈利因子', 'formula': 'sum(gains) / sum(|losses|)'},
            ],
        },
        'timestamp': datetime.now().isoformat(),
    })
