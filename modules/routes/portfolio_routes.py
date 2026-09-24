"""
Portfolio Routes - /api/portfolio/*, /api/brinson/*

Extracted from app.py (lines 2207-2264, 4902-4936).
"""

from flask import Blueprint, request, jsonify
from datetime import datetime
import json
import os
import numpy as np

from modules.data_fetcher import StockDataFetcher
from modules.logger import logger

bp = Blueprint('portfolio', __name__)


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


def build_bl_views(pool: list, seed_path: str = None):
    """model_seed.jsonl 每股最新决策轮 → BL views (2026-09-23 待拍板B+①).

    链形实锤: 23:10 scan 链 decide 串行 (~15s/票) → 每轮 5 票 ts 错峰 →
    「最新日期」= **per-code 取各自最新轮** (非全局 max_ts, 那只会读到
    最后 1 票)。每股聚合: 方向多数票 (2v2 平票/无反对多数 = 不注入,
    六维链同形) → {asset, return: ±conf均×0.25, confidence: 同向数/5}。
    seed 空/缺/坏 → None = 端点保现形 (链坏≠链死)。
    """
    try:
        root = os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))
        p = seed_path or os.path.join(root, 'runs', 'consensus_ab',
                                      'model_seed.jsonl')
        if not os.path.exists(p):
            return None
        with open(p, encoding='utf-8') as f:
            rows = [json.loads(l) for l in f if l.strip()]
        if not rows:
            return None
        # 最新日期纪律 (09-13 审计同源): 真链 = 每轮 5 票 ts 错峰 (23:10:19~
        # 23:11:25) → 全局 max_ts 只读到最末 1 票 = 读票偏。正确 = 最新日
        # 全部票 × 每票各自最新轮 (跨日混合轮不进)
        last_day = max(r['ts'][:10] for r in rows)
        latest: dict = {}          # code → (最新轮 ts, [(dir, conf), …])
        for r in rows:
            if r['ts'][:10] == last_day and r['code'] in pool:
                key = r['code']
                if key not in latest or r['ts'] > latest[key][0]:
                    latest[key] = (r['ts'], [(r['dir'], r['conf'])])
                elif r['ts'] == latest[key][0]:
                    latest[key][1].append((r['dir'], r['conf']))
        if not latest:
            return None
        group = {c: p for c, (_t, p) in latest.items()}
        views = []
        for code, preds in group.items():
            dirs = [d for d, c in preds if d in ('buy', 'sell')]
            if not dirs:
                continue
            top = max(set(dirs), key=dirs.count)
            n_same = dirs.count(top)
            if n_same <= len(dirs) / 2:
                continue                       # 平票/无多数 = 不注入 (六维同形)
            confs = [c for d, c in preds if d == top]
            views.append({
                'asset': pool.index(code),
                'return': round((0.25 if top == 'buy' else -0.25)
                                * (sum(confs) / len(confs)), 4),
                'confidence': round(n_same / 5.0, 4)})
        return views or None
    except (OSError, ValueError, KeyError, TypeError) as e:
        logger.error(f"[Portfolio] BL views seed 聚合失败 (用现 views): {e}")
        return None


def build_bl_pool(price_source=None):
    """监控池 5 票真实行情 → BL/RP 输入 (2026-09-23 残留①, flag=1 消费)。

    路径: get_stock_data 缓存 (23:10 warm 同源) / price_source 注入 (测试)。
    任一票缺/挂 → 整组 None = 端点保 demo 形 (禁混血: 5 票须同源同轮)。
    pe/market_cap 真链恒 0 (09-23 实测) → 50/100 uniform 先验 (非假数)。
    """
    try:
        if price_source is not None:
            src = price_source

            def _fetch(c):
                return src.get_stock_info(c) or {}

            def _fetch_hot(c):
                return src.get_stock_info(c) or {}
        else:
            def _fetch(c):
                return get_stock_data(c) or {}

            _fetch_hot = _fetch
        pool = []
        for code in ('sz300620', 'sh688981', 'sz300502', 'sh688800',
                     'sh603929'):
            sd = _fetch(code)
            if not sd:
                logger.warning(f"[Portfolio] 真池断源 {code} → demo 形保持 "
                               f"(链坏≠链死)")
                return None
            pool.append({'name': sd.get('name') or code, 'code': code,
                         'market_cap': sd.get('market_cap') or 50,
                         'expected_return': 0.15,
                         'price': sd.get('price') or 100,
                         'pe': sd.get('pe') or 100})
        return pool
    except Exception as e:
        logger.error(f"[Portfolio] 真池失败 (demo 形保持): {e}")
        return None


@bp.route('/api/portfolio/optimize')
def api_portfolio_optimize():
    """组合优化（Black-Litterman + 风险平价）"""
    stock_code = request.args.get('stock_code', 'sz300620')
    try:
        from modules.portfolio_optimizer import PortfolioOptimizer
        from modules.factors.multi_factor_model_v2 import multi_factor_model_v2

        optimizer = PortfolioOptimizer()

        # 默认 demo 形 (5 假价 + 2 主观 views); flag UDE_BL_VIEWS=1 → 监控池
        # 5 票真行情 (get_stock_data 缓存链) + seed 每股最新轮 views (5 票全
        # 覆盖)。任一源断/缺 → 整组保 demo (链坏≠链死, 禁混血 09-23 实锤:
        # 真 BL 池与监控池 seed 仅 300620 相交)。默认 0 = 零影响 (回归锁)
        _sd = get_stock_data(stock_code) or {}
        stocks = [
            {'name': '光库科技', 'code': 'sz300620', 'market_cap': 200,
             'expected_return': 0.15, 'price': _sd.get('price', 100), 'pe': _sd.get('pe', 100)},
            {'name': '仕佳光子', 'code': 'sh688313', 'market_cap': 150,
             'expected_return': 0.12, 'price': 80, 'pe': 120},
            {'name': '中际旭创', 'code': 'sz300308', 'market_cap': 500,
             'expected_return': 0.18, 'price': 150, 'pe': 60},
            {'name': '江特电机', 'code': 'sz002176', 'market_cap': 100,
             'expected_return': 0.10, 'price': 30, 'pe': 80},
            {'name': '东百集团', 'code': 'sh600693', 'market_cap': 50,
             'expected_return': 0.08, 'price': 8, 'pe': 20},
        ]
        market_caps = [s['market_cap'] for s in stocks]
        views = [
            {'asset': 0, 'return': 0.20, 'confidence': 0.6},
            {'asset': 2, 'return': 0.25, 'confidence': 0.5},
        ]
        if os.environ.get('UDE_BL_VIEWS', '0') == '1':
            _root = os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))))
            _seed_p = os.path.join(_root, 'runs', 'consensus_ab',
                                   'model_seed.jsonl')
            pool = build_bl_pool()
            if pool:
                views = build_bl_views([s['code'] for s in pool],
                                       seed_path=_seed_p) or []
                stocks, market_caps = pool, [s['market_cap'] for s in pool]
                logger.info(f"[Portfolio] 真实数据形: {len(stocks)} 票 "
                            f"{len(views)} views (flag UDE_BL_VIEWS=1)")

        # Black-Litterman
        bl_result = optimizer.black_litterman_summary(stocks, views)

        # 风险平价（简化: 假设相关系数矩阵）
        n = len(stocks)
        corr_matrix = [[1.0 if i == j else 0.3 for j in range(n)] for i in range(n)]
        rp_weights = optimizer.calculate_risk_parity_weights(corr_matrix)

        return jsonify({
            'success': True,
            'black_litterman': bl_result,
            'risk_parity_weights': {stocks[i]['name']: rp_weights[i] for i in range(n)},
            'correlation_matrix': corr_matrix,
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"Portfolio optimize error: {e}")
        return jsonify({'success': False, 'error': str(e)})


@bp.route('/api/brinson/analyze', methods=['GET', 'POST'])
def api_brinson_analyze():
    """Brinson-Fachler 绩效归因分析"""
    try:
        from modules.brinson_attribution import BrinsonAttribution

        if request.method == 'POST':
            data = request.get_json(silent=True, force=True) or {}
            portfolio_weights = data.get('portfolio_weights', {})
            benchmark_weights = data.get('benchmark_weights', {})
            portfolio_returns = data.get('portfolio_returns', {})
            benchmark_returns = data.get('benchmark_returns', {})
            period = data.get('period', 'daily')
        else:
            # Default demo data for GET
            portfolio_weights = {'科技': 0.4, '金融': 0.3, '消费': 0.3}
            benchmark_weights = {'科技': 0.3, '金融': 0.35, '消费': 0.35}
            portfolio_returns = {'科技': 0.15, '金融': 0.08, '消费': 0.12}
            benchmark_returns = {'科技': 0.12, '金融': 0.06, '消费': 0.10}
            period = 'daily'

        attribution = BrinsonAttribution()
        result = attribution.analyze(
            portfolio_weights=portfolio_weights,
            benchmark_weights=benchmark_weights,
            portfolio_returns=portfolio_returns,
            benchmark_returns=benchmark_returns,
            period=period,
        )

        return jsonify({
            'success': True,
            'data': result,
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"Brinson attribution error: {e}")
        return jsonify({'success': False, 'error': str(e)})
