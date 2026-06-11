#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
仪表盘 API — Dashboard API Blueprint

统一聚合层: 为前端"量化模型"标签页提供所有 API 端点。

端点:
  GET /api/dashboard/factors          — 因子计算结果
  GET /api/dashboard/ml-prediction    — ML 预测 + 市场状态
  GET /api/dashboard/factor-ic        — 因子 IC/ICIR 历史
  GET /api/dashboard/risk-report      — 风险报告
  GET /api/dashboard/backtest-result  — 回测结果
  GET /api/dashboard/sentiment        — 情感分析
  GET /api/dashboard/model-health     — 模型健康状态
  POST /api/dashboard/hyperparams     — 启动超参优化
"""

from flask import Blueprint, request, jsonify
from typing import Dict, List
from modules.logger import logger
import numpy as np
import time

bp = Blueprint('dashboard', __name__, url_prefix='/api/dashboard')


def _get_stock_code():
    """从请求中获取股票代码"""
    return request.args.get('code', request.args.get('stock_code', 'sz300620'))


def _get_klines(code, days=60):
    """获取 K 线数据"""
    try:
        from app import data_fetcher
        return data_fetcher.get_kline_data(code, period='daily', count=days)
    except Exception as e:
        logger.error(f"[DashboardAPI] K线数据获取失败: {e}")
        return []


# ── 因子计算 ──────────────────────────────────────────────

@bp.route('/factors')
def factors():
    """获取所有因子计算结果"""
    try:
        code = _get_stock_code()
        from app import analysis_engine, data_fetcher
        stock_data = data_fetcher.get_stock_info(code)
        klines = _get_klines(code)

        # 多因子模型 V2
        from modules.multi_factor_model_v2 import multi_factor_model_v2 as mfm
        factor_scores = mfm.calculate_all_factors_cached(code, stock_data, klines=klines)
        weighted = mfm.weighted_score(factor_scores)
        rating = mfm.get_rating(weighted)

        # 增强特征
        try:
            from modules.enhanced_features import enhanced_features
            enhanced = enhanced_features.calculate_all(stock_data, klines=klines)
        except Exception:
            enhanced = {}

        return jsonify({
            'code': code,
            'factor_scores': factor_scores,
            'enhanced_features': enhanced,
            'weighted_score': round(weighted, 4),
            'rating': rating,
            'dominant_factors': mfm.get_dominant_factors(factor_scores, n=5),
        })
    except Exception as e:
        logger.error(f"[DashboardAPI] 因子计算失败: {e}")
        return jsonify({'error': str(e)}), 500


# ── ML 预测 ──────────────────────────────────────────────

@bp.route('/ml-prediction')
def ml_prediction():
    """获取 ML 预测结果"""
    try:
        code = _get_stock_code()
        from app import data_fetcher, ml_predictor
        stock_data = data_fetcher.get_stock_info(code)
        klines = _get_klines(code)

        # 动态集成
        try:
            from modules.dynamic_ensemble import dynamic_ensemble
            regime = dynamic_ensemble.regime_detector.detect_regime(stock_data, klines)
        except Exception:
            regime = 'sideways'

        # ML 预测 -- 始终加载缓存模型，从不在此处训练
        # 模型训练由 ModelTrainingScheduler 后台调度（启动时 + 每 24h + API 手动触发）
        if not ml_predictor.is_trained:
            ml_predictor.load_latest_model()

        if ml_predictor.is_trained:
            # 使用 prepare_features() 保证与训练特征一致（修复前视偏差 + 共线性问题）
            features = ml_predictor.prepare_features(stock_data, klines)
            if features is None:
                # 数据不足回退到默认
                features = ml_predictor._make_default_features()
            result = ml_predictor.predict_direction(features)
        else:
            result = {
                'direction': 'neutral',
                'confidence': 0.5,
                'probabilities': {'up': 0.33, 'down': 0.33, 'neutral': 0.34},
            }

        # 市场状态
        try:
            from modules.dynamic_ensemble import dynamic_ensemble
            pred = dynamic_ensemble.predict(np.zeros(12), stock_data, klines)
            regime = pred.get('regime', 'sideways')
            weights = pred.get('weights', {})
        except Exception:
            regime = 'sideways'
            weights = {}

        return jsonify({
            'code': code,
            'prediction': result,
            'regime': regime,
            'model_weights': weights,
            'is_trained': ml_predictor.is_trained,
            'cv_score': ml_predictor.cv_score,
            'feature_importances': ml_predictor.feature_importances,
        })
    except Exception as e:
        logger.error(f"[DashboardAPI] ML 预测失败: {e}")
        return jsonify({'error': str(e)}), 500


# ── 因子 IC ──────────────────────────────────────────────

@bp.route('/factor-ic')
def factor_ic():
    """获取因子 IC/ICIR 历史"""
    try:
        from modules.factor_ic_monitor import factor_ic_monitor, FactorICMonitor

        monitor = FactorICMonitor()

        # 精简股票池 (5 只活跃 A 股, 确保响应 < 30s)
        stock_pool = [
            'sh600519', 'sz000858', 'sz300750',
            'sh600036', 'sz000333',
        ]

        # 轻量因子计算: 直接从 klines 提取, 避免调用 21 因子模型
        # 使用 5 个核心因子: momentum_1d, volume_ratio, realized_vol, rsi, turnover
        factor_names = ['momentum_1d', 'volume_ratio', 'realized_vol', 'rsi_technical', 'turnover_level']
        n_dates = 5

        # factor_cross[date_key][fname] = {stock_code: factor_value}
        factor_cross: Dict[str, Dict[str, Dict[str, float]]] = {}
        return_cross: Dict[str, Dict[str, float]] = {}
        dates_used = []
        ic_data: Dict[str, Dict[str, float]] = {}

        for code in stock_pool:
            try:
                klines = _get_klines(code)
                if not klines or len(klines) < 30:
                    continue

                closes = np.array([k['close'] for k in klines], dtype=float)
                volumes = np.array([k['volume'] for k in klines], dtype=float)
                turnovers = [k.get('turnover', 0) for k in klines]

                # 从最近 n_dates 天采样
                step = max(3, len(klines) // (n_dates * 5))
                for i in range(max(29, n_dates - 1), len(klines) - 1, step):
                    date_key = klines[i].get('date', f'd{i}')
                    if date_key not in dates_used:
                        dates_used.append(date_key)

                    if date_key not in factor_cross:
                        factor_cross[date_key] = {fname: {} for fname in factor_names}
                        return_cross[date_key] = {}

                    # momentum_1d
                    mom = (closes[i] / closes[i - 1] - 1) * 100 if i > 0 else 0

                    # volume_ratio
                    avg_vol = np.mean(volumes[max(0, i - 20):i]) if i >= 20 else np.mean(volumes)
                    vol_ratio = closes[i] / (avg_vol + 1e-10)

                    # realized_vol (20d)
                    if i >= 20:
                        ret_20 = np.diff(np.log(closes[i - 20:i]))
                        vol = float(np.std(ret_20) * np.sqrt(252) * 100)
                    else:
                        vol = 5.0

                    # rsi (14d)
                    if i >= 15:
                        deltas = np.diff(closes[i - 14:i])
                        gains = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
                        losses = abs(np.mean(deltas[deltas < 0])) if np.any(deltas < 0) else 0.001
                        rsi = float(100 - (100 / (1 + gains / losses)))
                    else:
                        rsi = 50.0

                    # turnover
                    turnover = turnovers[i] if i < len(turnovers) else 0

                    for fname in factor_names:
                        factor_cross[date_key][fname][code] = {
                            'momentum_1d': mom,
                            'volume_ratio': vol_ratio,
                            'realized_vol': vol,
                            'rsi_technical': rsi,
                            'turnover_level': turnover,
                        }[fname]

                    # 未来收益率
                    if i + 1 < len(klines):
                        ret = (klines[i + 1]['close'] - klines[i]['close']) / (klines[i]['close'] * 100)
                        return_cross[date_key][code] = ret

            except Exception:
                continue

        # 如果没有足够截面数据, 返回空结果 (不抛异常)
        if not dates_used or len(dates_used) < 3:
            resp = jsonify({
                'factors': {},
                'ranking': [],
                'stock_pool_size': len(stock_pool),
                'trading_days': 0,
                'note': '数据不足, 无法计算截面 IC',
            })
            resp.headers['Cache-Control'] = 'no-cache'
            return resp

        # 计算每日 IC (直接计算 Pearson 相关系数)
        for fname in factor_names:
            ic_series = []
            for date_key in dates_used:
                if date_key not in factor_cross or fname not in factor_cross.get(date_key, {}):
                    continue
                try:
                    # 收集该日期所有股票的因子值和收益率
                    factors = []
                    returns = []
                    for code, fval in factor_cross[date_key][fname].items():
                        rval = return_cross.get(date_key, {}).get(code)
                        if rval is not None and not np.isnan(fval) and not np.isnan(rval):
                            factors.append(fval)
                            returns.append(rval)

                    if len(factors) < 3:
                        continue

                    f_arr = np.array(factors)
                    r_arr = np.array(returns)

                    f_std = np.std(f_arr)
                    r_std = np.std(r_arr)
                    if f_std < 1e-10 or r_std < 1e-10:
                        continue

                    ic = float(np.corrcoef(f_arr, r_arr)[0, 1])
                    if not np.isnan(ic):
                        ic_series.append(ic)
                except Exception:
                    continue

            if len(ic_series) < 5:
                # 数据不足, 记录空值
                ic_data_single = {
                    'ic_mean': 0.0, 'ic_std': 0.0, 'icir': 0.0,
                    't_stat': 0.0, 'ic_series': [], 'valid_days': 0,
                }
            else:
                ic_arr = np.array(ic_series)
                ic_mean = float(np.mean(ic_arr))
                ic_std = float(np.std(ic_arr))
                icir = ic_mean / (ic_std + 1e-10)
                t_stat = ic_mean / (ic_std / np.sqrt(len(ic_arr)) + 1e-10)

                ic_data_single = {
                    'ic_mean': round(ic_mean, 4),
                    'ic_std': round(ic_std, 4),
                    'icir': round(icir, 4),
                    't_stat': round(t_stat, 4),
                    'ic_series': [round(x, 4) for x in ic_series[-30:]],
                    'valid_days': len(ic_series),
                }

            ic_data[fname] = ic_data_single

        # 使用 get_factor_ranking 排序
        ranking = monitor.get_factor_ranking(ic_data)

        resp = jsonify({
            'factors': ic_data,
            'ranking': [{'name': n, 'icir': d, 'ic_mean': m} for n, m, d in ranking],
            'stock_pool_size': len(stock_pool),
            'trading_days': len(dates_used),
        })
        resp.headers['Cache-Control'] = 'max-age=300'  # 缓存 5 分钟
        return resp
    except Exception as e:
        logger.error(f"[DashboardAPI] IC 计算失败: {e}")
        return jsonify({'error': str(e)}), 500


# ── 风险报告 ──────────────────────────────────────────────

@bp.route('/risk-report')
def risk_report():
    """获取风险报告"""
    try:
        from modules.barra_risk_model import RiskOptimizer, RiskReportGenerator
        import numpy as np

        # 简化: 生成示例报告
        n_assets = 5
        expected_returns = np.array([0.001, 0.0008, 0.0012, 0.0005, 0.0009])
        cov_matrix = np.eye(n_assets) * 0.0004
        cov_matrix[0, 1] = cov_matrix[1, 0] = 0.0001

        weights = RiskOptimizer.risk_parity(cov_matrix)
        report = RiskReportGenerator.generate_report(weights, expected_returns, cov_matrix)

        return jsonify({
            'report': report,
            'weights': [round(float(w), 4) for w in weights],
        })
    except Exception as e:
        logger.error(f"[DashboardAPI] 风险报告失败: {e}")
        return jsonify({'error': str(e)}), 500


# ── 回测结果 ──────────────────────────────────────────────

@bp.route('/backtest-result')
def backtest_result():
    """获取回测结果 — 基于真实因子信号 + T+1 约束"""
    try:
        code = _get_stock_code()
        from modules.advanced_backtester import BacktestEngine, BacktestResult, TransactionCostModel
        from modules.multi_factor_model_v2 import multi_factor_model_v2 as mfm

        # 1. 获取真实 K 线数据 (120 交易日以支持滚动因子计算)
        klines = _get_klines(code, days=120)
        if not klines:
            return jsonify({'error': '无法获取 K 线数据'}), 400

        # K 线可能按时间倒序排列，需要正序以便滚动计算
        klines_sorted = list(reversed(klines))
        dates = [k['date'] for k in klines_sorted if 'date' in k]
        if len(dates) < 30:
            return jsonify({'error': 'K 线数据不足，至少需要 30 个交易日'}), 400

        # 2. 提取价格序列
        prices_map = {}
        for k in klines_sorted:
            d = k['date']
            p = float(k.get('close', 0))
            if p > 0:
                prices_map[d] = {code: p}

        # 3. 滚动因子信号生成 (每日基于当日及之前数据计算因子)
        buy_threshold = 6.5
        sell_threshold = 3.5

        signals = {}
        for i in range(10, len(klines_sorted)):
            d = dates[i]
            window = klines_sorted[:i + 1]
            latest = window[-1]

            # 计算当日因子
            factor_scores = mfm.calculate_all_factors(latest, klines=window)
            score = mfm.weighted_score(factor_scores)

            if score >= buy_threshold:
                signals[d] = {code: {'direction': 'buy', 'confidence': score / 10.0}}
            elif score <= sell_threshold:
                signals[d] = {code: {'direction': 'sell', 'confidence': (10.0 - score) / 10.0}}
            else:
                signals[d] = {code: {'direction': 'hold', 'confidence': 0.5}}

        # 4. 执行回测 (带 T+1 + 100 股整数倍约束)
        engine = BacktestEngine(initial_capital=1000000, cost_model=TransactionCostModel())
        result = BacktestResult()
        cash = engine.initial_capital
        positions = {}
        bought_today = set()  # T+1: 今日买入不可今日卖出
        trade_date_list = []

        for date in dates:
            day_signals = signals.get(date, {})
            day_prices = prices_map.get(date, {})

            for stock, sig in day_signals.items():
                price = day_prices.get(stock, 0)
                if price <= 0:
                    continue

                direction = sig.get('direction', 'hold')
                position = positions.get(stock, 0)

                if direction == 'buy':
                    # 10% 仓位限制，100 股整数倍 (A 股最小交易单位)
                    target_value = cash * 0.1
                    shares = int(target_value / price)
                    shares = (shares // 100) * 100  # 100 股整数倍
                    if shares < 100:
                        continue  # 不足 1 手不交易

                    cost = engine.cost_model.calculate_buy(price, shares)
                    if cost['total'] > cash:
                        continue  # 资金不足

                    cash -= cost['total']
                    positions[stock] = position + shares
                    bought_today.add(stock)

                    result.trades.append({
                        'date': date, 'stock': stock,
                        'direction': 'buy', 'volume': shares,
                        'price': price, 'cost': cost['total_cost'],
                    })

                elif direction == 'sell':
                    if stock not in bought_today and position > 0:
                        shares = position
                        revenue = engine.cost_model.calculate_sell(price, shares)
                        cash += revenue['total']

                        result.trades.append({
                            'date': date, 'stock': stock,
                            'direction': 'sell', 'volume': shares,
                            'price': price, 'cost': revenue['total_cost'],
                        })
                        positions[stock] = 0

            bought_today.clear()

            # 当日净值
            portfolio_value = cash
            for stock, shares in positions.items():
                p = day_prices.get(stock, 0)
                portfolio_value += shares * p

            result.equity_curve.append((date, portfolio_value))
            result.positions[date] = dict(positions)
            trade_date_list.append(date)

        metrics = result.calculate_metrics()

        return jsonify({
            'code': code,
            'metrics': metrics,
            'equity_curve': [{'date': d, 'nav': round(v, 2)} for d, v in result.equity_curve],
            'trades': result.trades,
            'signal_summary': {
                'n_buy': sum(1 for s in signals.values() if s.get(code, {}).get('direction') == 'buy'),
                'n_sell': sum(1 for s in signals.values() if s.get(code, {}).get('direction') == 'sell'),
                'n_hold': sum(1 for s in signals.values() if s.get(code, {}).get('direction') == 'hold'),
                'buy_threshold': buy_threshold,
                'sell_threshold': sell_threshold,
            },
        })
    except Exception as e:
        logger.error(f"[DashboardAPI] 回测失败: {e}")
        return jsonify({'error': str(e)}), 500


# ── 情感分析 ──────────────────────────────────────────────

@bp.route('/sentiment')
def sentiment():
    """获取情感分析结果"""
    try:
        code = _get_stock_code()
        try:
            from modules.finbert_sentiment import sentiment_analyzer

            # 模拟新闻数据
            news_list = [
                {'title': '光库科技业绩预增,机构看好未来增长', 'days_ago': 0, 'source': '新闻'},
                {'title': 'AI算力需求持续爆发,光通信板块受益', 'days_ago': 1, 'source': '新闻'},
                {'title': '短期涨幅较大,注意回调风险', 'days_ago': 2, 'source': '股吧'},
                {'title': '主力净流入超亿元,资金持续加仓', 'days_ago': 0, 'source': '新闻'},
                {'title': '技术面突破压力位,有望继续上行', 'days_ago': 1, 'source': '股吧'},
            ]

            result = sentiment_analyzer.analyze_stock_news(news_list)
        except Exception:
            result = {'score': 0.3, 'label': 'positive', 'n_news': 5}

        return jsonify({
            'code': code,
            'sentiment': result,
        })
    except Exception as e:
        logger.error(f"[DashboardAPI] 情感分析失败: {e}")
        return jsonify({'error': str(e)}), 500


# ── 模型健康 ──────────────────────────────────────────────

@bp.route('/model-health')
def model_health():
    """获取模型健康状态"""
    try:
        from app import ml_predictor
        from modules.concept_drift import health_monitor

        report = ml_predictor.get_model_report() if ml_predictor.is_trained else {
            'is_trained': False, 'cv_score': 0, 'models': [],
        }

        # 健康状态
        health = health_monitor.health_report() if health_monitor else {'status': 'unknown'}

        return jsonify({
            'ml_report': report,
            'health': health,
        })
    except Exception as e:
        logger.error(f"[DashboardAPI] 模型健康查询失败: {e}")
        return jsonify({'error': str(e)}), 500


# ── 超参优化 ──────────────────────────────────────────────

@bp.route('/hyperparams', methods=['GET', 'POST'])
def hyperparams():
    """GET: 返回当前最佳参数 | POST: 启动新的优化"""
    try:
        if request.method == 'POST':
            data = request.get_json() or {}
            n_trials = data.get('n_trials', 30)

            try:
                from modules.hyperparam_optimizer import HyperParamOrchestrator
                orchestrator = HyperParamOrchestrator(n_trials=n_trials)

                # 生成模拟特征数据
                np.random.seed(42)
                X = np.random.randn(200, 12)
                y = np.random.choice([-1, 0, 1], size=200)

                best_params = orchestrator.optimize_all(X, y)

                return jsonify({
                    'status': 'completed',
                    'n_trials': n_trials,
                    'best_params': best_params,
                })
            except Exception as e:
                logger.error(f"[DashboardAPI] 超参优化失败: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 500

        else:
            # GET: 返回默认参数
            return jsonify({
                'lgb': {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.05},
                'xgb': {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.05},
                'rf': {'n_estimators': 100, 'max_depth': 5, 'min_samples_split': 5},
            })
    except Exception as e:
        logger.error(f"[DashboardAPI] 超参优化 API 失败: {e}")
        return jsonify({'error': str(e)}), 500


# ── 数据质量 ──────────────────────────────────────────────

@bp.route('/data-quality')
def data_quality():
    """获取数据质量评估"""
    try:
        code = _get_stock_code()
        from app import data_fetcher

        # 获取 K 线数据
        klines = _get_klines(code, days=500)
        if not klines:
            return jsonify({'error': '无法获取 K 线数据'}), 400

        # 基础统计
        n_days = len(klines)
        dates = [k.get('date', '') for k in klines]
        start_date = min(dates) if dates else '--'
        end_date = max(dates) if dates else '--'

        # 缺失率：计算空值/NaN 比例
        total_values = 0
        missing_values = 0
        for k in klines:
            for field in ['open', 'high', 'low', 'close', 'volume']:
                total_values += 1
                val = k.get(field)
                if val is None or val == 0 or val == '':
                    missing_values += 1

        missing_rate = missing_values / total_values if total_values > 0 else 0

        # 复权状态：检查是否有 adjusted 标记或复权因子字段
        adjusted = any(k.get('adjusted') or k.get('adj_factor') or k.get('qfq_factor') for k in klines[:10])

        # 综合质量分 (0-100)
        score = 100
        if missing_rate > 0.05:
            score -= 20
        if missing_rate > 0.1:
            score -= 20
        if not adjusted:
            score -= 10
        if n_days < 200:
            score -= 20
        elif n_days < 400:
            score -= 10
        score = max(0, min(100, score))

        # 数据源指示器
        indicators = [
            {'type': 'real', 'label': 'K 线行情', 'value': f'{n_days} 天'},
            {'type': 'real' if adjusted else 'partial', 'label': '复权状态', 'value': '已复权' if adjusted else '未复权'},
            {'type': 'info', 'label': '缺失率', 'value': f'{missing_rate * 100:.1f}%'},
            {'type': 'info', 'label': '数据范围', 'value': f'{start_date} ~ {end_date}'},
        ]

        # 尝试获取基本面数据质量
        try:
            from app import analysis_engine
            stock_info = data_fetcher.get_stock_info(code)
            if stock_info:
                # Tencent 返回 pe, EastMoney 可能返回 pe_ratio/pb
                has_financial = bool(stock_info.get('pe') or stock_info.get('pe_ratio') or stock_info.get('pb') or stock_info.get('pb_ratio'))
                indicators.append({
                    'type': 'real' if has_financial else 'fake',
                    'label': '财务数据',
                    'value': '有' if has_financial else 'fallback',
                })
        except Exception:
            indicators.append({'type': 'partial', 'label': '财务数据', 'value': '未知'})

        return jsonify({
            'code': code,
            'quality': {
                'overall_score': score,
                'missing_rate': missing_rate,
                'adjusted': adjusted,
                'n_days': n_days,
            },
            'indicators': indicators,
            'start_date': start_date,
            'end_date': end_date,
        })
    except Exception as e:
        logger.error(f"[DashboardAPI] 数据质量查询失败: {e}")
        return jsonify({'error': str(e)}), 500


# ── 手动交易回测 ──────────────────────────────────────────────

@bp.route('/backtest-manual', methods=['POST'])
def backtest_manual():
    """手动交易记录回测 — 用户填入交易笔数，系统计算回测指标

    请求体:
        stock_code: 股票代码 (如 sz300620)
        initial_capital: 初始资金 (默认 1000000)
        trades: 交易记录列表
            - date: 日期 YYYY-MM-DD
            - direction: buy / sell
            - price: 成交价
            - volume: 成交数量 (股)
        risk_free: 无风险利率 (默认 0.02)
    """
    try:
        data = request.get_json() or {}
        stock_code = data.get('stock_code', 'sz300620')
        initial_capital = float(data.get('initial_capital', 1000000))
        risk_free = float(data.get('risk_free', 0.02))
        trades_input = data.get('trades', [])

        if not trades_input:
            return jsonify({'error': '请提供交易记录 (trades)'}), 400

        # 解析交易记录
        trades = []
        for t in trades_input:
            trades.append({
                'date': t.get('date', ''),
                'direction': t.get('direction', '').lower(),
                'price': float(t.get('price', 0)),
                'volume': int(t.get('volume', 0)),
                'stock': stock_code,
            })

        # 构建价格映射 (从交易记录中提取价格作为当日收盘价)
        prices_map = {}
        dates_set = set()
        for t in trades:
            d = t['date']
            dates_set.add(d)
            if d not in prices_map:
                prices_map[d] = {}
            prices_map[d][stock_code] = t['price']
        dates_list = sorted(dates_set)

        # 交易成本模型
        from modules.advanced_backtester import TransactionCostModel
        cost_model = TransactionCostModel()

        # 构建权益曲线 (按交易顺序处理)
        capital = initial_capital
        position = 0  # 持仓数量
        equity_curve = []

        # 先添加初始权益
        first_date = dates_list[0]
        equity_curve.append((first_date, initial_capital))

        # 按日期顺序处理交易
        trade_index = 0
        for date in dates_list:
            # 处理该日期的所有交易
            while trade_index < len(trades) and trades[trade_index]['date'] == date:
                trade = trades[trade_index]
                price = trade['price']
                vol = trade['volume']
                direction = trade['direction']

                if direction == 'buy':
                    cost_dict = cost_model.calculate_buy(price, vol)
                    total_cost = cost_dict['total']
                    capital -= total_cost
                    position += vol
                elif direction == 'sell':
                    cost_dict = cost_model.calculate_sell(price, vol)
                    # calculate_sell 返回的 total = 净收入 (base_cost - 所有费用)
                    capital += cost_dict['total']
                    position -= vol

                trade_index += 1

            # 当日权益 = 现金 + 持仓市值
            current_price = prices_map[date].get(stock_code, 0)
            nav = capital + position * current_price
            equity_curve.append((date, nav))

        # 计算回测指标
        from modules.advanced_backtester import BacktestResult
        result = BacktestResult()
        result.equity_curve = equity_curve
        result.trades = trades
        result.positions = {}
        metrics = result.calculate_metrics(risk_free=risk_free)

        # 构建返回的交易记录 (含成本)
        output_trades = []
        capital = initial_capital
        position = 0
        buy_prices = []  # 记录买入价用于计算卖出盈亏
        for t in trades:
            price = t['price']
            vol = t['volume']
            direction = t['direction']
            if direction == 'buy':
                cost_dict = cost_model.calculate_buy(price, vol)
                total_cost = cost_dict['total']
                output_trades.append({
                    'date': t['date'],
                    'direction': 'buy',
                    'price': price,
                    'volume': vol,
                    'cost': round(cost_dict['total'] - price * vol, 2),
                    'total_cost': round(total_cost, 2),
                })
                capital -= total_cost
                position += vol
                buy_prices.append(price)
            elif direction == 'sell':
                cost_dict = cost_model.calculate_sell(price, vol)
                # total = 净收入, total_cost = 费用
                net_proceeds = cost_dict['total']
                actual_cost = cost_dict['total_cost']
                gross = price * vol
                # 计算盈亏 (使用最近一次买入价作为成本)
                avg_buy = sum(buy_prices) / len(buy_prices) if buy_prices else price
                pnl = (price - avg_buy) * vol
                output_trades.append({
                    'date': t['date'],
                    'direction': 'sell',
                    'price': price,
                    'volume': vol,
                    'cost': round(actual_cost, 2),
                    'gross_proceeds': round(gross, 2),
                    'net_proceeds': round(net_proceeds, 2),
                    'pnl': round(pnl, 2),
                })
                capital += net_proceeds
                position -= vol
                if buy_prices:
                    buy_prices.pop()

        # 最终权益
        final_nav = capital + position * (trades[-1]['price'] if trades else 0)
        total_return = (final_nav - initial_capital) / initial_capital

        return jsonify({
            'code': stock_code,
            'initial_capital': initial_capital,
            'final_nav': round(final_nav, 2),
            'total_return': round(total_return, 4),
            'n_trades': len(trades),
            'metrics': metrics,
            'trades': output_trades,
            'equity_curve': [{'date': d, 'nav': round(n, 2)} for d, n in equity_curve],
        })
    except Exception as e:
        logger.error(f"[DashboardAPI] 手动回测失败: {e}")
        return jsonify({'error': str(e)}), 500


# ── 测试代码 ──────────────────────────────────────────────

if __name__ == '__main__':
    from flask import Flask
    test_app = Flask(__name__)
    test_app.register_blueprint(bp)

    with test_app.test_client() as client:
        print("=== Dashboard API 测试 ===")

        resp = client.get('/api/dashboard/factors?code=sz300620')
        print(f"GET /factors: {resp.status_code}")

        resp = client.get('/api/dashboard/ml-prediction?code=sz300620')
        print(f"GET /ml-prediction: {resp.status_code}")

        resp = client.get('/api/dashboard/factor-ic')
        print(f"GET /factor-ic: {resp.status_code}")

        resp = client.get('/api/dashboard/risk-report')
        print(f"GET /risk-report: {resp.status_code}")

        resp = client.get('/api/dashboard/sentiment?code=sz300620')
        print(f"GET /sentiment: {resp.status_code}")

        resp = client.get('/api/dashboard/model-health')
        print(f"GET /model-health: {resp.status_code}")

        print("\n测试完成!")
