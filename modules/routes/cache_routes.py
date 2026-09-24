"""
Cache Management Routes - /api/cache/*

Extracted from app.py (lines 792-859).
Dependencies: cache (modules.dynamic_cache), logger (modules.logger)
"""

from flask import Blueprint, request, jsonify
from datetime import datetime

from modules.dynamic_cache import cache
from modules.logger import logger

bp = Blueprint('cache', __name__)


@bp.route('/api/cache/stats')
def api_cache_stats():
    """Get cache statistics with hit/miss rates"""
    return jsonify({
        'success': True,
        'stats': cache.get_stats(),
        'timestamp': datetime.now().isoformat()
    })


@bp.route('/api/cache/clear', methods=['POST'])
def api_cache_clear():
    """Clear cache by category, prefix, or stock code"""
    body = request.json if request.is_json else {}
    category = body.get('category')
    stock_code = body.get('stock_code')
    prefix = body.get('prefix')

    if stock_code:
        cache.invalidate_stock(stock_code)
        msg = f'All cache for {stock_code} invalidated'
    elif prefix:
        cache.invalidate_key_prefix(prefix)
        msg = f'Cache with prefix "{prefix}" cleared'
    elif category:
        cache.invalidate_category(category)
        msg = f'Category "{category}" cache cleared'
    else:
        cache.cleanup()
        msg = 'Expired cache entries cleaned'
    return jsonify({'success': True, 'message': msg})


@bp.route('/api/cache/cleanup', methods=['POST'])
def api_cache_cleanup():
    """清理过期缓存"""
    try:
        cleaned = cache.cleanup()
        return jsonify({'success': True, 'deleted': cleaned})
    except Exception as e:
        logger.error(f"Cache cleanup failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/cache/invalidate/<stock_code>', methods=['POST'])
def api_invalidate_stock(stock_code):
    """Invalidate all cache for a specific stock"""
    cache.invalidate_stock(stock_code)
    return jsonify({
        'success': True,
        'message': f'All cache invalidated for {stock_code}',
        'timestamp': datetime.now().isoformat()
    })


@bp.route('/api/cache/reset-stats', methods=['POST'])
def api_reset_cache_stats():
    """Reset cache hit/miss statistics"""
    cache.reset_stats()
    return jsonify({
        'success': True,
        'message': 'Cache statistics reset',
        'timestamp': datetime.now().isoformat()
    })
