"""
Factor Routes - /api/factors/*

Extracted from app.py (lines 2294-2576).
"""

import threading

from flask import Blueprint, request, jsonify
from datetime import datetime
import numpy as np

from modules.data_fetcher import StockDataFetcher
from modules.factors.multi_factor_model_v2 import multi_factor_model_v2
from modules.logger import logger

bp = Blueprint('factor', __name__)


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


@bp.route('/api/factors/norm')
def api_factors_norm():
    """因子标准化对比（P2: 使用 V2 15因子模型 + 真实股票池）"""
    stock_code = request.args.get('stock_code', 'sz300620')
    try:
        fetcher = StockDataFetcher()
        stock_data = get_stock_data(stock_code) or {}
        klines = fetcher.get_kline_data(stock_code, 'daily', 100) if stock_code else None

        # 使用 V2 模型计算所有因子 (21 因子走 dynamic_cache 内存缓存, P1-5)
        all_factors = multi_factor_model_v2.calculate_all_factors_cached(
            stock_code, stock_data, klines=klines)
        weighted = multi_factor_model_v2.weighted_score(all_factors)
        rating = multi_factor_model_v2.get_rating(weighted)

        # 构建真实股票池（从 AKShare 获取同行业股票）
        universe_codes = ['300620', '688313', '300308', '002176', '600693']
        universe: list = []
        universe_factors_map = {}
        for code in universe_codes:
            try:
                full_code = f"sh{code}" if code.startswith('6') else f"sz{code}"
                u_data = fetcher.get_stock_info(full_code)
                u_klines = fetcher.get_kline_data(full_code, 'daily', 100)
                if u_data:
                    uf = multi_factor_model_v2.calculate_all_factors(u_data, u_klines)
                    universe_factors_map[code] = uf
                    stock_entry = dict(u_data)
                    stock_entry.update(uf)
                    stock_entry['market_cap'] = u_data.get('total_market_value', 0)
                    universe.append(stock_entry)
            except Exception:
                pass

        # 增强版横截面标准化: Winsorize → Rank → Industry Neutralize → Market Cap Neutralize → Orthogonalize
        factor_scores = dict(all_factors)
        factor_scores['market_cap_value'] = stock_data.get('total_market_value', 0)
        normalized = multi_factor_model_v2.cross_sectional_normalize(factor_scores, universe)

        # 标准化后的加权分数
        norm_weighted = sum(
            multi_factor_model_v2.factor_weights.get(f, 0.05) * normalized.get(f, 0)
            for f in normalized
        ) / sum(multi_factor_model_v2.factor_weights.get(f, 0.05) for f in normalized) if normalized else 0.0

        # 最强/最弱因子
        sorted_factors = sorted(all_factors.items(), key=lambda x: x[1], reverse=True)
        top_factors = [{'name': k, 'score': round(v, 2)} for k, v in sorted_factors[:3]]
        bottom_factors = [{'name': k, 'score': round(v, 2)} for k, v in sorted_factors[-3:]]

        return jsonify({
            'success': True,
            'stock_code': stock_code,
            'raw_scores': {k: round(v, 2) for k, v in all_factors.items()},
            'all_factors': {k: round(v, 2) for k, v in all_factors.items()},
            'normalized_scores': normalized,
            'weighted_score_raw': round(weighted, 4),
            'weighted_score_normalized': round(norm_weighted, 4),
            'rating': rating,
            'normalized': True,
            'top_factors': top_factors,
            'bottom_factors': bottom_factors,
            'universe_size': len(universe),
            'factor_count': len(all_factors),
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"Factor norm error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)})


@bp.route('/api/factors/comprehensive')
def api_factors_comprehensive():
    """综合因子分析 — 基础因子 + Alpha158 + Alpha360 + Barra CNE6"""
    stock_code = request.args.get('stock_code', 'sz300620')
    try:
        from modules.data_fetcher import StockDataFetcher
        from modules.factors.alpha360_calculator import alpha360_calculator
        from modules.factors.barra_cne6_calculator import barra_cne6_calculator
        from modules.factors.alpha158_calculator import alpha158_calculator

        fetcher = StockDataFetcher()
        stock_data = get_stock_data(stock_code) or {}
        klines = fetcher.get_kline_data(stock_code, 'daily', 120) if stock_code else None

        result = {
            'success': True,
            'stock_code': stock_code,
            'timestamp': datetime.now().isoformat(),
        }

        # 1. 基础因子
        try:
            base_factors = multi_factor_model_v2.calculate_all_factors(stock_data, klines)
            result['base_factors'] = {k: round(v, 4) for k, v in base_factors.items()}
            result['base_factor_count'] = len(base_factors)
        except Exception as e:
            logger.error(f"Base factors error: {e}")
            result['base_factors'] = {}
            result['base_factor_count'] = 0

        # 2. Alpha158 因子
        alpha158_factors = {}
        if klines and len(klines) >= 60:
            try:
                alpha158_factors = alpha158_calculator.calculate_all(klines)
                alpha158_normalized = {k: round(float(np.clip(v * 5 + 5, 0, 10)), 4)
                                        for k, v in alpha158_factors.items()}
                result['alpha158_factors'] = alpha158_normalized
                result['alpha158_count'] = len(alpha158_factors)
            except Exception as e:
                logger.error(f"Alpha158 error: {e}")
                result['alpha158_factors'] = {}
                result['alpha158_count'] = 0

        # 3. Alpha360 因子
        alpha360_factors = {}
        if klines and len(klines) >= 30:
            try:
                alpha360_factors = alpha360_calculator.calculate_all(klines)
                alpha360_normalized = {k: round(v, 4) for k, v in alpha360_factors.items()}
                result['alpha360_factors'] = alpha360_normalized
                result['alpha360_count'] = len(alpha360_factors)
                result['alpha360_categories'] = alpha360_calculator.get_factor_categories()
            except Exception as e:
                logger.error(f"Alpha360 error: {e}")
                result['alpha360_factors'] = {}
                result['alpha360_count'] = 0

        # 4. Barra CNE6 风格因子
        barra_factors = {}
        try:
            barra_factors = barra_cne6_calculator.calculate_all(stock_data, klines)
            result['barra_factors'] = {k: round(v, 4) for k, v in barra_factors.items()}
            result['barra_count'] = len(barra_factors)
            result['barra_categories'] = barra_cne6_calculator.get_factor_categories()
        except Exception as e:
            logger.error(f"Barra error: {e}")
            result['barra_factors'] = {}
            result['barra_count'] = 0

        # 5. 综合得分
        all_factors = {}
        all_factors.update(base_factors)
        all_factors.update({f'alpha158_{k}': v for k, v in alpha158_factors.items()})
        all_factors.update({f'alpha360_{k}': v for k, v in alpha360_factors.items()})
        all_factors.update({f'barra_{k}': v for k, v in barra_factors.items()})

        weighted = multi_factor_model_v2.weighted_score(all_factors)
        rating = multi_factor_model_v2.get_rating(weighted)
        result['weighted_score'] = round(weighted, 4)
        result['rating'] = rating
        result['total_factor_count'] = len(all_factors)

        # 6. 因子分类统计
        category_stats = {}
        for cat_name, cat_factors in multi_factor_model_v2.FACTOR_CATEGORIES.items():
            cat_values = [all_factors.get(f, 0) for f in cat_factors if f in all_factors]
            if cat_values:
                category_stats[cat_name] = {
                    'mean': round(float(np.mean(cat_values)), 4),
                    'max': round(float(np.max(cat_values)), 4),
                    'min': round(float(np.min(cat_values)), 4),
                    'count': len(cat_values),
                }
        result['category_stats'] = category_stats

        return jsonify(result)
    except Exception as e:
        logger.error(f"Comprehensive factors error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)})


@bp.route('/api/factors/monitor')
def api_factor_monitor():
    """因子监控 API — IC Decay, 相关性, 警告"""
    try:
        stock_code = request.args.get('code', 'sz300620')

        # 加载因子监控器
        try:
            from modules.factors.factor_monitor import factor_monitor
            monitor = factor_monitor
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'因子监控器加载失败: {e}',
                'timestamp': datetime.now().isoformat(),
            }), 503

        # 获取 K 线数据
        from modules.kline_signal_analyzer import KlineSignalAnalyzer
        analyzer = KlineSignalAnalyzer()
        klines = analyzer.get_klines(stock_code, period='daily', count=120)
        if not klines or len(klines) < 30:
            return jsonify({
                'success': False,
                'error': 'K 线数据不足',
                'timestamp': datetime.now().isoformat(),
            }), 400

        # 计算因子值 (使用 Alpha158)
        try:
            from modules.factors.alpha158_calculator import Alpha158Calculator
            alpha158 = Alpha158Calculator()
            factor_dict = alpha158.calculate_factors(klines)

            # 构建 {date: {stock_code: {factor_name: value}}}
            factor_data = {}
            future_returns = {}
            if factor_dict and 'factors' in factor_dict:
                dates = sorted(factor_dict['factors'].keys(), reverse=True)
                for i, date in enumerate(dates[:50]):  # 最多 50 天
                    factors = factor_dict['factors'][date]
                    factor_data[date] = {stock_code: factors}
                    # 未来收益
                    if i + 1 < len(dates):
                        next_date = dates[i + 1]
                        next_close = None
                        for k in klines:
                            if str(k.get('date', '')) == next_date:
                                next_close = k.get('close', 0)
                                break
                        cur_close = None
                        for k in klines:
                            if str(k.get('date', '')) == date:
                                cur_close = k.get('close', 0)
                                break
                        if cur_close and next_close and cur_close > 0:
                            future_returns[date] = {stock_code: (next_close - cur_close) / cur_close}
        except Exception as e:
            logger.warning(f"因子计算失败: {e}")
            factor_data = {}
            future_returns = {}

        # 获取监控结果
        result = monitor.get_monitoring_result(stock_code, factor_data, future_returns)

        response = {
            'success': result.success,
            'stock_code': stock_code,
            'timestamp': result.timestamp,
            'ic_decay': result.ic_decay,
            'multi_scale_ic': result.multi_scale_ic,  # 60D/120D/252D
            'factor_ranking': result.factor_ranking,
            'warnings': result.warnings,
            'half_life': result.half_life,
            'decay_interpretation': result.decay_interpretation,
            'drift_consensus': result.drift_consensus,  # ADWIN/Page-Hinkley/DDM
            'correlation': {
                'names': result.factor_names,
                'matrix': result.correlation_matrix,
            },
        }

        if not result.success:
            response['error'] = result.error

        return jsonify(response)

    except Exception as e:
        logger.error(f"因子监控 API 错误: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)})


@bp.route('/api/factors/shap-importance')
def api_factors_shap_importance():
    """SHAP 因子重要性解释 API"""
    try:
        stock_code = request.args.get('stock_code', 'sz300620')

        from modules.data_fetcher import StockDataFetcher
        from modules.factors.alpha158_calculator import alpha158_calculator

        fetcher = StockDataFetcher()
        klines = fetcher.get_kline_data(stock_code, 'daily', 120)

        if not klines or len(klines) < 60:
            return jsonify({'success': False, 'error': 'K 线数据不足 (至少 60 天)'}), 400

        # 使用基础因子 + Alpha158，取前 15 个最稳定的因子
        base_factors = multi_factor_model_v2.calculate_all_factors({}, klines[:120])
        alpha158_factors = alpha158_calculator.calculate_all(klines[:120])

        # 合并并过滤
        all_f = {**base_factors, **alpha158_factors}
        numeric = {k: v for k, v in all_f.items()
                  if isinstance(v, (int, float, np.floating, np.integer))
                  and not np.isnan(float(v))}

        # 取前 15 个 (按绝对值排序，避免因子过多)
        top_factors = dict(sorted(numeric.items(), key=lambda x: abs(x[1]), reverse=True)[:15])

        # 构建时间序列数据
        factor_matrix = {}
        returns = {}
        for i, kline in enumerate(klines[:120]):
            date_key = str(i)
            factor_matrix[date_key] = {k: float(v) for k, v in top_factors.items()}

            if i > 0:
                prev_close = klines[i - 1].get('close', 0)
                curr_close = kline.get('close', 0)
                if prev_close > 0:
                    returns[date_key] = (curr_close - prev_close) / prev_close

        # 计算 SHAP 重要性
        try:
            from modules.factors.factor_ic_monitor import FactorICMonitor
            monitor = FactorICMonitor()
            shap_importance = monitor.compute_shap_importance(factor_matrix, returns)
        except Exception as e:
            logger.warning(f"SHAP 计算失败: {e}")
            shap_importance = {}

        # 排序
        sorted_factors = sorted(shap_importance.items(), key=lambda x: x[1], reverse=True)

        return jsonify({
            'success': True,
            'stock_code': stock_code,
            'shap_importance': {k: round(v, 6) for k, v in sorted_factors},
            'top_5_factors': [{'name': n, 'shap': round(s, 6)} for n, s in sorted_factors[:5]],
            'bottom_5_factors': [{'name': n, 'shap': round(s, 6)} for n, s in sorted_factors[-5:]],
            'factor_count': len(shap_importance),
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"SHAP 因子重要性 API 错误: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)})


@bp.route('/api/factors/alpha/decay/status', methods=['GET'])
def api_alpha_decay_status():
    """Alpha 衰减监控状态 — 查看因子 IC 衰减和权重调整"""
    try:
        from modules.factors.factor_ic_monitor import factor_ic_monitor
        from modules.factor_weight_scheduler import get_factor_weight_scheduler

        stock_code = request.args.get('stock_code', 'sz300620')

        # 获取调度器状态
        scheduler = get_factor_weight_scheduler()
        scheduler_status = scheduler.get_status()

        # 获取衰减告警
        alerts = factor_ic_monitor.get_ic_decay_alerts_from_returns(stock_code)

        # 获取记录数
        n_records = len(factor_ic_monitor._factor_returns.get(stock_code, {}))

        return jsonify({
            'success': True,
            'stock_code': stock_code,
            'scheduler': scheduler_status,
            'alerts': alerts,
            'n_records': n_records,
            'n_factors_monitored': len(factor_ic_monitor._factor_returns.get(stock_code, {})),
            'alert_count': len(alerts),
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"Alpha 衰减状态查询失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)})


# ── 2026-09-22 因子衰减监控接通 (用户拍板): 23:10 replay 链尾喂料 ──────────
# 三因: ①_factor_returns 内存 dict 无喂料链时重启清零 ②喂料只挂 analyze 链
# (手动/偶发, 9 天 24 条实锤 5002.log) ③面板 JS 读 recent_ic/n_factors/warnings
# 键 = 端点从不返回 (恒显 0/空, 键名错位自 09-20 上线)。本段修 ②③, ①形不动。
# 配对形 (防前视): (t 日因子 = K[:t+1] 喂算, t→t+1 收益) 逐对喂 — 非 analyze 段
# 同形 (当日因子+当日收益 = 自相关噪声); 单票 guard 链坏≠链死。

_EVAL_LOCK = threading.Lock()   # 单飞 (23:10 链 + 手动防并发)
_DECAY_POOL_DEFAULT = 'sz300620,sh688981,sz300502,sh688800,sh603929'


def _build_decay_records(monitor, calc, fetcher, pool: str,
                         pairs: int = 2, window: int = 90) -> dict:
    """监控池逐票喂因子历史对 (23:10 链形, 核心函数 = 测试缝)。

    每票: 90d K → 最末 pairs 对 (factor_t, ret t→t+1) → record_factor_return。
    单票 K 不足/单对失败 → errors 诚实记, 不拖其余 (链坏≠链死)。
    """
    n_stocks, n_pairs, errors = 0, 0, []
    for code in [c.strip() for c in pool.split(',') if c.strip()]:
        try:
            kl = fetcher.get_kline_data(code, 'daily', window)
            if not kl or len(kl) < 54:   # <60% 窗 = 观察跳过 (patchtst 守卫同形)
                errors.append({'code': code, 'error': f'K线不足 {0 if not kl else len(kl)} 根'})
                continue
            touched = 0
            for i in range(max(2, len(kl) - 1 - pairs), len(kl) - 1):
                facts = calc.calculate_all(kl[:i + 1])   # 历史切片喂算 (无前视)
                if not facts:
                    continue
                c0, c1 = kl[i].get('close'), kl[i + 1].get('close')
                if not c0 or not c1 or c0 <= 0:
                    continue
                ret = (c1 - c0) / c0
                if abs(ret) < 0.005:
                    continue  # 无方向对不喂 (consensus 噪声区同形, 防稀释 IC)
                fac = {}
                for k2, v2 in facts.items():
                    if isinstance(v2, np.ndarray):
                        if v2.size > 0:
                            fac[k2] = float(v2[-1])
                    elif np.isscalar(v2):
                        f = float(v2)
                        if f == f and f != float('inf') and f != float('-inf'):
                            fac[k2] = f
                if fac:
                    date_i = str(kl[i].get('date', ''))[:10] or 'unknown'
                    monitor.record_factor_return(code, fac, ret, date=date_i)
                    touched += 1
            n_pairs += touched
            n_stocks += 1
        except Exception as e:
            errors.append({'code': code, 'error': f"{type(e).__name__}: {str(e)[:120]}"})
    return {'stocks': n_stocks, 'pairs': n_pairs, 'errors': errors}


@bp.route('/api/factors/alpha/decay/record', methods=['POST'])
def api_decay_record():
    """23:10 replay 链尾: 喂监控池因子历史对 → IC 衰减监控 (09-22 接通)

    env: DECAY_RECORD=0 总开关 / DECISION_SCAN_POOL 监控池 / FACTOR_DECAY_PAIRS
    每票喂对数 (默认 2 = 日增量; 首跑可手动 25 = 历史回填使告警链即时可判)。
    """
    import os
    if os.environ.get('DECAY_RECORD', '1') == '0':
        return jsonify({'success': True, 'skipped': 'DECAY_RECORD=0'})
    if not _EVAL_LOCK.acquire(blocking=False):
        return jsonify({'success': False,
                        'error': 'decay record 已在运行 (单飞防并发)'}), 429
    try:
        import time as _time
        from modules.factors.factor_ic_monitor import factor_ic_monitor
        from modules.factors.alpha158_calculator import get_alpha158_calculator
        pool = os.environ.get('DECISION_SCAN_POOL', _DECAY_POOL_DEFAULT)
        # ?pairs=25 = 手动回填窗口 (历史对一次喂, 告警链 20 样本起判即时可跑)
        pairs = int(request.args.get('pairs')
                    or os.environ.get('FACTOR_DECAY_PAIRS', '2'))
        fetcher = StockDataFetcher()
        calc = get_alpha158_calculator()
        t0 = _time.time()
        r = _build_decay_records(factor_ic_monitor, calc, fetcher, pool, pairs=pairs)
        alerts = sum(len(factor_ic_monitor.get_ic_decay_alerts_from_returns(c))
                     for c in pool.split(','))
        logger.info(f"[FactorDecay] 喂料链尾: stocks={r['stocks']} pairs={r['pairs']} "
                    f"alerts={alerts} errors={len(r['errors'])} "
                    f"elapsed={_time.time() - t0:.1f}s")
        return jsonify({'success': not r['errors'], 'data': {
            **r, 'alerts_total': alerts,
            'elapsed': round(_time.time() - t0, 1)}})
    except Exception as e:
        logger.error(f"[FactorRoutes] decay record 链坏: {e}")
        return jsonify({'success': False, 'error': str(e)[:200]}), 500
    finally:
        _EVAL_LOCK.release()
