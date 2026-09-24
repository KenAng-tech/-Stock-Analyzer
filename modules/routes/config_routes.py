"""
Config Routes - /api/config/*

Extracted from app.py (lines 1684-1733).
"""

from flask import Blueprint, request, jsonify
from datetime import datetime

from modules.logger import logger
from config import config

bp = Blueprint('config', __name__)

# 允许通过 API 修改的配置白名单 (只读非敏感项)
ALLOWED_CONFIG_KEYS = {
    'strategy.max_position',
    'strategy.min_stop_loss',
    'strategy.max_stop_loss',
    'strategy.take_profit_ratio',
    'strategy.trailing_stop_pct',
    'strategy.atr_multiplier_stop',
    'strategy.atr_multiplier_profit',
    'strategy.kelly_fraction',
    'strategy.kelly_max_position',
    'monitor.interval',
    'monitor.alert_interval',
    'cache.realtime',
    'cache.technical',
    'cache.fundamental',
    'cache.industry',
    'cache.kline',
    'cache.strategy',
}


@bp.route('/api/config')
@bp.route('/api/config/get')
def api_get_config():
    """Get current configuration"""
    return jsonify({
        'success': True,
        'config': config.get('strategy', {}),
        'timestamp': datetime.now().isoformat()
    })


@bp.route('/api/config/<key>', methods=['POST'])
def api_update_config(key):
    """Update configuration (白名单校验)"""
    if key not in ALLOWED_CONFIG_KEYS:
        return jsonify({
            'success': False,
            'error': f'不允许修改此配置: {key}',
            'allowed_keys': sorted(ALLOWED_CONFIG_KEYS),
        }), 403
    if request.is_json:
        config.set(key, request.json.get('value'))
        logger.info(f"[Config] 更新配置: {key} = {request.json.get('value')}")
    return jsonify({'success': True, 'key': key})


@bp.route('/api/config/update', methods=['POST'])
def api_update_config_body():
    """Update configuration via JSON body"""
    if not request.is_json:
        return jsonify({'success': False, 'error': '需要 JSON body'}), 400
    data = request.json
    key = data.get('key')
    if not key:
        return jsonify({'success': False, 'error': '缺少 key 字段'}), 400
    if key not in ALLOWED_CONFIG_KEYS:
        return jsonify({
            'success': False,
            'error': f'不允许修改此配置: {key}',
            'allowed_keys': sorted(ALLOWED_CONFIG_KEYS),
        }), 403
    config.set(key, data.get('value'))
    logger.info(f"[Config] 更新配置: {key} = {data.get('value')}")
    return jsonify({'success': True, 'key': key})
