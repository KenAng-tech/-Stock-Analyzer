#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Kronos 金融基础模型 API 路由 (2026-08-13 新增)

参考: Kronos (Aug 2025, 52 upvotes on HuggingFace)
      "Foundation Model for Financial Markets using a unique K-line tokenizer"

端点:
    GET  /api/kronos/status           - 状态
    POST /api/kronos/tokenize         - K-line 编码
    POST /api/kronos/predict          - 零样本预测
    POST /api/kronos/pretrain         - 触发预训练
    GET  /api/kronos/tokens/<code>    - 获取 token 序列
"""

import time
from flask import Blueprint, jsonify, request
from modules.logger import logger

bp = Blueprint('kronos_routes', __name__)


@bp.route('/api/kronos/status', methods=['GET'])
def kronos_status():
    """获取 Kronos 状态"""
    try:
        from modules.models.kronos_predictor import get_kronos_pretrainer
        pretrainer = get_kronos_pretrainer()
        return jsonify(pretrainer.get_status())
    except Exception as e:
        logger.error(f"[Kronos] status 失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/kronos/tokenize', methods=['POST'])
def kronos_tokenize():
    """K-line 编码"""
    try:
        from modules.models.kronos_predictor import get_kronos_pretrainer
        pretrainer = get_kronos_pretrainer()

        data = request.get_json(silent=True) or {}
        stock_code = data.get('stock_code', 'sz300620')

        # 获取 K-line 数据
        klines = pretrainer._fetch_klines(stock_code)
        if klines is None or len(klines) == 0:
            return jsonify({'success': False, 'error': f'无法获取 {stock_code} K-line 数据'}), 404

        result = pretrainer.tokenize_and_save(stock_code, klines)
        return jsonify({'success': True, 'data': result})
    except Exception as e:
        logger.error(f"[Kronos] tokenize 失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/kronos/predict', methods=['POST', 'GET'])
def kronos_predict():
    """零样本预测"""
    try:
        from modules.models.kronos_predictor import get_kronos_pretrainer
        pretrainer = get_kronos_pretrainer()

        # 支持 GET (查询参数) 和 POST (JSON body)
        if request.method == 'GET':
            stock_code = request.args.get('stock_code', 'sz300620')
            horizon = int(request.args.get('horizon', 5))
        else:
            data = request.get_json(silent=True) or {}
            stock_code = data.get('stock_code', 'sz300620')
            horizon = int(data.get('horizon', 5))

        result = pretrainer.predict(stock_code, horizon=horizon)
        return jsonify(result)
    except Exception as e:
        logger.error(f"[Kronos] predict 失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/kronos/pretrain', methods=['POST'])
def kronos_pretrain():
    """触发预训练"""
    try:
        from modules.models.kronos_predictor import get_kronos_pretrainer, PretrainConfig
        pretrainer = get_kronos_pretrainer()

        data = request.get_json(silent=True) or {}
        config = PretrainConfig(
            epochs=int(data.get('epochs', 3)),
            batch_size=int(data.get('batch_size', 32)),
            seq_length=int(data.get('seq_length', 60)),
            stock_pool=data.get('stock_pool', ['sz300620', 'sh688981', 'sz000001']),
        )

        # 异步执行预训练 (避免阻塞)
        import threading
        result_holder = [None]

        def _run():
            try:
                result_holder[0] = pretrainer.pretrain(config)
            except Exception as e:
                result_holder[0] = {'success': False, 'error': str(e)}

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

        return jsonify({
            'success': True,
            'message': '预训练已启动 (后台运行)',
            'config': {
                'epochs': config.epochs,
                'batch_size': config.batch_size,
                'seq_length': config.seq_length,
                'stock_pool': config.stock_pool,
            },
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        })
    except Exception as e:
        logger.error(f"[Kronos] pretrain 失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/kronos/tokens/<stock_code>', methods=['GET'])
def kronos_tokens(stock_code):
    """获取某股票的 token 序列"""
    try:
        from modules.models.kronos_predictor import get_kronos_pretrainer
        pretrainer = get_kronos_pretrainer()

        import sqlite3
        import json
        with sqlite3.connect(pretrainer.db_path) as conn:
            row = conn.execute(
                'SELECT tokens, token_ids FROM token_sequences WHERE stock_code = ? ORDER BY date DESC LIMIT 1',
                (stock_code,)
            ).fetchone()

        if row is None:
            return jsonify({
                'success': False,
                'error': f'{stock_code} 暂无 token 序列，请先调用 /api/kronos/tokenize',
            }), 404

        tokens = json.loads(row[0])
        token_ids = json.loads(row[1])

        return jsonify({
            'success': True,
            'data': {
                'stock_code': stock_code,
                'n_tokens': len(tokens),
                'vocab_size': pretrainer.tokenizer.VOCAB_SIZE,
                'tokens': tokens,
                'token_ids': token_ids,
                'unique_tokens': len(set(token_ids)),
            }
        })
    except Exception as e:
        logger.error(f"[Kronos] tokens 失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500