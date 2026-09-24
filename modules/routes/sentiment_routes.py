"""
Sentiment Routes - /api/sentiment/*, /api/sota/sentiment/*

Extracted from app.py (lines 1039-1080, 2267-2291, 3034-3059).
"""

from flask import Blueprint, request, jsonify
from datetime import datetime

from modules.data_fetcher import StockDataFetcher
from modules.sentiment_engine import get_sentiment_engine, SentimentEngine, DictionaryAnalyzer
from modules.logger import logger

bp = Blueprint('sentiment', __name__)
sentiment_engine = get_sentiment_engine()


def get_stock_data(stock_code: str = None) -> dict:
    """Helper: get stock data with caching"""
    if stock_code is None:
        stock_code = 'sz300620'
    from modules.dynamic_cache import cache
    cache_key = f"stock_{stock_code}"
    cached = cache.get(cache_key, category='realtime')
    if cached:
        return cached
    fetcher = StockDataFetcher()
    stock_data = fetcher.get_stock_info(stock_code)
    if stock_data:
        cache.set(cache_key, stock_data, category='realtime')
    return stock_data


@bp.route('/api/sota/sentiment/analyze', methods=['POST'])
def api_sota_sentiment_analyze():
    """
    FinBERT 情感分析 API

    请求体:
        {
            "texts": ["利好消息", "业绩超预期"],
            "weights": [1.0, 1.0]  // 可选
        }

    返回:
        {
            'success': True,
            'aggregate_score': -1.0 ~ +1.0,
            'aggregate_label': 'positive' | 'neutral' | 'negative',
            'aggregate_confidence': 0.0-1.0,
            'text_count': int,
            'method': 'finbert' | 'dictionary',
            'timestamp': str,
        }
    """
    try:
        body = request.get_json(silent=True, force=True) or {}
        if not body or 'texts' not in body:
            return jsonify({'success': False, 'error': 'Missing texts'}), 400

        texts = body['texts']
        weights = body.get('weights')

        result = sentiment_engine.aggregate(texts, weights)

        return jsonify({
            'success': True,
            **result,
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"[SentimentEngine] 分析失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/sentiment/<stock_code>')
def api_sentiment(stock_code):
    """情绪分析"""
    try:
        # 2026-09-10 修复: 原每次 new SentimentEngine() + 调不存在的
        # get_sentiment_score → 永久 500; 改单例 + engine 真链 (news+股吧抓取)
        # 获取股票名称
        stock_data = get_stock_data(stock_code)
        stock_name = stock_data.get('name', '') if stock_data else ''

        result = sentiment_engine.get_sentiment_score(stock_code, stock_name)
        return jsonify({
            'success': True,
            'stock_code': stock_code,
            'sentiment': result,
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"Sentiment error: {e}")
        return jsonify({'success': False, 'error': str(e)})


@bp.route('/api/sentiment/bert/<stock_code>')
def api_sentiment_bert(stock_code):
    """FinBERT 情感分析"""
    try:
        # 2026-09-10 真链化: 原「词典法硬喂股票名称」= 恒 neutral 假数据 (200≠已呈现)。
        # 现走 engine 真链: 抓取该股票新闻+股吧 → FinBERT/词典批量分析 → 聚合。
        from modules.data_fetcher import StockDataFetcher

        fetcher = StockDataFetcher()
        stock_data = fetcher.get_stock_info(stock_code)
        stock_name = stock_data.get('name', '') if stock_data else ''

        result = sentiment_engine.get_sentiment_score(stock_code, stock_name)

        return jsonify({
            'success': True,
            'stock_code': stock_code,
            'sentiment': result,
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"[FinBERT Sentiment] 错误：{e}")
        return jsonify({'success': False, 'error': str(e)})


@bp.route('/api/sentiment/history/<stock_code>')
def api_sentiment_history(stock_code):
    """获取股票历史情感数据"""
    try:
        from modules.dynamic_cache import cache
        history = cache.get(f"sentiment_history_{stock_code}", category='sentiment')
        if history is None:
            history = []
        return jsonify({
            'success': True,
            'stock_code': stock_code,
            'history': history,
            'count': len(history),
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"Sentiment history error: {e}")
        return jsonify({'success': False, 'error': str(e)})
