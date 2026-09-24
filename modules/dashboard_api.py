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
import threading
import time
from datetime import datetime

bp = Blueprint('dashboard', __name__, url_prefix='/api/dashboard')


def _get_stock_code():
    """从请求中获取股票代码"""
    return request.args.get('code', request.args.get('stock_code', 'sz300620'))


def _get_klines(code, days=60):
    """获取 K 线数据"""
    try:
        from modules.dependencies import get_data_fetcher
        return get_data_fetcher().get_kline_data(code, period='daily', count=days)
    except Exception as e:
        logger.error(f"[DashboardAPI] K线数据获取失败: {e}")
        return []


# ── 因子计算 ──────────────────────────────────────────────

@bp.route('/factors')
def factors():
    """获取所有因子计算结果"""
    try:
        code = _get_stock_code()
        analysis_engine = None; from modules.dependencies import get_analysis_engine, get_data_fetcher
        stock_data = get_data_fetcher().get_stock_info(code)
        klines = _get_klines(code)

        # 多因子模型 V2
        from modules.factors.multi_factor_model_v2 import multi_factor_model_v2 as mfm
        factor_scores = mfm.calculate_all_factors_cached(code, stock_data, klines=klines)
        weighted = mfm.weighted_score(factor_scores)
        rating = mfm.get_rating(weighted)

        # 增强特征
        try:
            from modules.factors.enhanced_features import enhanced_features
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
        from modules.dependencies import get_data_fetcher, get_ml_predictor
        stock_data = get_data_fetcher().get_stock_info(code)
        klines = _get_klines(code, days=300)  # ML 预测需要至少 257 天数据

        # 动态集成
        try:
            from modules.dynamic_ensemble import dynamic_ensemble
            regime = dynamic_ensemble.regime_detector.detect_regime(stock_data, klines)
        except Exception:
            regime = 'sideways'

        # ML 预测 -- 不再调用 load_latest_model()（C 扩展内存冲突导致 SIGSEGV）
        # 模型训练由 ModelTrainingScheduler 后台调度（每天 23:00 + API 手动触发）
        # 如果模型未训练，返回中性预测

        if get_ml_predictor().is_trained:
            # 使用 prepare_features() 保证与训练特征一致（修复前视偏差 + 共线性问题）
            features = get_ml_predictor().prepare_features(stock_data, klines)
            if features is None:
                # 数据不足回退到默认
                features = get_ml_predictor()._make_default_features()
            result = get_ml_predictor().predict_direction(features, klines=klines)
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
            'is_trained': get_ml_predictor().is_trained,
            'cv_score': get_ml_predictor().cv_score,
            'models': list(get_ml_predictor().models.keys()),
            'feature_importances': get_ml_predictor()._flatten_feature_importances(get_ml_predictor().feature_importances),
            'model_version': 'v1.2.0',
            'last_retrain': get_ml_predictor()._trained_at or '未训练',
            'trained_at': get_ml_predictor()._trained_at,
            'is_fresh': get_ml_predictor().is_model_fresh(),
        })
    except Exception as e:
        logger.error(f"[DashboardAPI] ML 预测失败: {e}")
        return jsonify({'error': str(e)}), 500


# ── 因子 IC ──────────────────────────────────────────────

@bp.route('/factor-ic')
def factor_ic():
    """获取因子 IC/ICIR 历史"""
    try:
        from modules.factors.factor_ic_monitor import factor_ic_monitor

        monitor = factor_ic_monitor

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
        from modules.factors.barra_risk_model import RiskOptimizer, RiskReportGenerator
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
        from modules.factors.multi_factor_model_v2 import multi_factor_model_v2 as mfm

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
            # 2026-09-10 真链化: 原 5 条硬编码假新闻 (对任何股票显示同一"偏多",
            # 且 neutral→negative 误判) = 假数据冒充真实情绪, 已全部拆除。
            # 现走真实抓取该股票新闻+股吧 → 批量情感聚合; 无舆情诚实 no_data。
            from modules.sentiment_engine import get_sentiment_engine

            r = get_sentiment_engine().get_sentiment_score(code)
            result = {'score': r['score'], 'label': r['label'],
                      'n_news': r['n_articles'], 'method': r['method']}
        except Exception as e:
            logger.error(f"[DashboardAPI] 情感分析失败: {e}")
            result = {'score': 0, 'label': 'no_data', 'n_news': 0, 'method': 'error'}

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
        from modules.dependencies import get_ml_predictor
        from modules.concept_drift import health_monitor

        report = get_ml_predictor().get_model_report() if get_ml_predictor().is_trained else {
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

# 模块级单例 — 优化结果跨请求持久 (GET 读最近一次真实调参结果)
_hyperparam_orchestrator = None
_hyperparam_tuning = {'status': 'idle', 'started_at': None, 'finished_at': None,
                     'stock_code': None, 'error': None}
_hyperparam_tuning_lock = threading.Lock()


def _run_hyperparam_tuning(code: str, n_trials: int):
    """后台调参线程 — 数据准备 → 子进程调参 → 结果回写 (daemon 线程, 不阻塞请求)

    2026-09-08: 主进程 (Flask 线程) 直跑 LGBM/XGB 训练曾触发 SIGSEGV 杀死
    整个服务, 且同步阻塞 15 分钟不可用 → 数据准备留在本线程 (纯 numpy 安全),
    模型训练全部进子进程 (optimize_all_isolated, 同 ml_training_worker 模式)。
    """
    try:
        from modules.hyperparam_optimizer import segment_regimes, optimize_all_isolated
        from modules.dependencies import get_ml_predictor

        klines = _get_klines(code, days=500)
        if not klines or len(klines) < 320:
            raise ValueError(f'K线数据不足 ({len(klines or [])} 条, 需 ≥320 条)')

        # 真实市场数据: 3 类方向标签 (扣交易成本 + 波动率自适应阈值) ↔ 12 维时序特征
        ml = get_ml_predictor()
        labels = ml.create_labels(klines, horizon=5)
        features = ml.prepare_features_batch(klines, labels)
        if features is None or len(features) == 0 or len(labels) == 0:
            raise ValueError('特征/标签构建失败')

        # features[i] ↔ labels[i] 按时间索引对齐 (labels 尾部裁掉 horizon 天)
        n_common = min(len(features), len(labels))
        # per-sample regime 分段 (仅用过去 20 日窗口, 无前视) → 分段调参
        regimes = segment_regimes(klines[:n_common])
        tuning = optimize_all_isolated(features[:n_common], labels[:n_common],
                                       regimes=regimes, n_trials=n_trials)
        if not tuning:
            with _hyperparam_tuning_lock:
                _hyperparam_tuning.update(
                    status='failed', error='调参子进程失败 (详见 server 日志)',
                    finished_at=datetime.now().isoformat())
            return

        orchestrator = _hyperparam_orchestrator
        if orchestrator is not None:
            orchestrator._best_params = tuning['best_params']
            orchestrator._regime_params = tuning.get('regime_params') or {}
            # 调参结果 → MLPredictor.custom_params (模型工厂 + 子进程 trainer 重训消费)。
            # 尾部 regime 优先用该段调出的参数, 无则回退全局
            orchestrator.apply_to_predictor(ml, current_regime=tuning.get('current_regime'))

        with _hyperparam_tuning_lock:
            _hyperparam_tuning.update(status='completed', error=None,
                                      finished_at=datetime.now().isoformat())
        logger.info(f"[DashboardAPI] 超参调参完成: {code} (regime={tuning.get('current_regime')})")
    except Exception as e:
        logger.error(f"[DashboardAPI] 超参调参线程失败: {e}")
        with _hyperparam_tuning_lock:
            _hyperparam_tuning.update(status='failed', error=str(e),
                                      finished_at=datetime.now().isoformat())


@bp.route('/hyperparams', methods=['GET', 'POST'])
def hyperparams():
    """GET: 调参状态/参数 | POST: 后台启动超参优化 (异步, 立即返回)

    2026-09-08 修复: 旧实现用 np.random.randn(200,12) 纯噪声数据调参 (假训练 —
    噪声上"最优"参数无迁移价值) 且 GET 恒返回硬编码默认值。改为:
    ① POST 用真实 K 线构建 特征-标签 对 (create_labels 含成本扣除+自适应阈值)
    ② 调参在后台线程 + 子进程运行 (主进程直跑 C 扩展曾 SIGSEGV 崩服务)
    ③ GET 优先返回真实调参结果 (前端面板据此呈现), running 时返回进度
    """
    global _hyperparam_orchestrator
    try:
        from modules.hyperparam_optimizer import HyperParamOrchestrator

        if _hyperparam_orchestrator is None:
            _hyperparam_orchestrator = HyperParamOrchestrator(n_trials=30)
        orchestrator = _hyperparam_orchestrator

        if request.method == 'POST':
            data = request.get_json() or {}
            n_trials = int(data.get('n_trials', 30))
            code = _get_stock_code()

            with _hyperparam_tuning_lock:
                # 2026-09-08 僵死守卫: 后台线程若卡死在无超时阻塞点
                # (socket 读/锁死锁), status 会永远停在 running → 端点永久 409。
                # 超过 3600s 子进程预算 + 60s 缓冲即视为僵死, 解锁允许重触发。
                stale = (_hyperparam_tuning['status'] == 'running' and
                         time.time() - (_hyperparam_tuning['started_at'] or 0.0) > 3660)
                if _hyperparam_tuning['status'] == 'running' and not stale:
                    return jsonify({'status': 'running',
                                    'message': '已有调参任务在运行, 请稍后再试'}), 409
                _hyperparam_tuning.update(status='running', started_at=time.time(),
                                          stock_code=code, error=None)

            threading.Thread(target=_run_hyperparam_tuning,
                             args=(code, n_trials),
                             name='HyperparamTuning', daemon=True).start()
            return jsonify({
                'status': 'started',
                'message': '超参调参已在后台启动 (regime 分段调参, 约 25-35 分钟), '
                           'GET /api/dashboard/hyperparams 可查询进度',
            })

        # GET: running → 进度; 有真实调参结果 → 返回之; 否则默认参数
        # (保持前端扁平 {lgb,xgb,rf} 结构兼容)
        with _hyperparam_tuning_lock:
            status = _hyperparam_tuning['status']
            elapsed = time.time() - (_hyperparam_tuning['started_at'] or 0.0)
            if status == 'running' and elapsed > 3660:
                # 僵死自愈: 线程卡死在无超时阻塞点时不永久占住 running
                # (子进程侧 subprocess timeout=3600 已会先触发, 此为双保险)
                _hyperparam_tuning.update(
                    status='failed', error='调参超时 (>3660s, 疑似僵死, 已自愈解锁)',
                    finished_at=datetime.now().isoformat())
                status = 'failed'

        if status == 'running':
            return jsonify({'status': 'running', 'elapsed_seconds': round(elapsed, 1)})

        best = orchestrator.get_best_params()
        if not (best and all(v for v in best.values())):
            # 内存无结果 → 读 02:00 夜间调参快照 (2026-09-08 凌晨链)
            from modules.hyperparam_optimizer import load_param_snapshot
            best = load_param_snapshot()
        if best and all(v for v in best.values()):
            return jsonify(best)
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
        from modules.dependencies import get_data_fetcher

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
            from modules.dependencies import get_analysis_engine
            stock_info = get_data_fetcher().get_stock_info(code)
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


# ── 综合因子分析 ──────────────────────────────────────────────

@bp.route('/factors-comprehensive')
def factors_comprehensive():
    """综合因子分析 — 基础因子 + Alpha158 + Alpha360 + Barra CNE6"""
    try:
        code = _get_stock_code()
        from modules.dependencies import get_data_fetcher
        from modules.factors.multi_factor_model_v2 import multi_factor_model_v2 as mfm
        from modules.factors.alpha360_calculator import alpha360_calculator
        from modules.factors.barra_cne6_calculator import barra_cne6_calculator
        from modules.factors.alpha158_calculator import alpha158_calculator
        import numpy as np

        stock_data = get_data_fetcher().get_stock_info(code)
        klines = _get_klines(code, days=120)

        result = {'code': code, 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')}

        # 基础因子 (21 因子走 dynamic_cache 内存缓存, 5min TTL 依赖链失效)
        base_factors = mfm.calculate_all_factors_cached(code, stock_data, klines=klines)
        result['base_factors'] = {k: round(v, 4) for k, v in base_factors.items()}
        result['base_factor_count'] = len(base_factors)

        # Alpha360 因子 (P1-5: factor_cache 语义缓存旁路, 默认开 killswitch=FACTOR_SEMANTIC_CACHE=0)
        alpha360_factors = {}
        if klines and len(klines) >= 30:
            alpha360_factors = alpha360_calculator.calculate_all(
                klines, stock_code=code, use_cache=True)
            result['alpha360_factors'] = {k: round(v, 4) for k, v in alpha360_factors.items()}
            result['alpha360_count'] = len(alpha360_factors)

        # Barra CNE6 风格因子
        barra_factors = barra_cne6_calculator.calculate_all(stock_data, klines)
        result['barra_factors'] = {k: round(v, 4) for k, v in barra_factors.items()}
        result['barra_count'] = len(barra_factors)

        # 综合得分
        all_factors = {**base_factors, **alpha360_factors, **{f'barra_{k}': v for k, v in barra_factors.items()}}
        weighted = mfm.weighted_score(all_factors)
        rating = mfm.get_rating(weighted)
        result['weighted_score'] = round(weighted, 4)
        result['rating'] = rating
        result['total_factor_count'] = len(all_factors)

        return jsonify(result)
    except Exception as e:
        logger.error(f"[DashboardAPI] 综合因子分析失败: {e}")
        return jsonify({'error': str(e)}), 500


# ── IC 增强分析 ──────────────────────────────────────────────

@bp.route('/factor-ic-enhanced')
def factor_ic_enhanced():
    """增强版因子 IC 分析 — Alphalens 风格"""
    try:
        from modules.factors.factor_ic_monitor import (
            factor_ic_monitor, factor_turnover,
            long_short_analyzer, factor_group_ic, ICQQPlot, ICDecay,
        )
        from modules.dependencies import get_data_fetcher

        monitor = factor_ic_monitor
        # 2026-09-15 修: 原 fetcher = data_fetcher 为未定义名 (死链 NameError 恒 500), 对齐 :940 取数器
        fetcher = get_data_fetcher()
        from modules.factors.multi_factor_model_v2 import multi_factor_model_v2 as mfm

        # 股票池
        stock_pool = [
            'sh600519', 'sz000858', 'sz300750', 'sh600036', 'sz000333',
            'sh601318', 'sz002415', 'sh600276', 'sz300124', 'sh601166',
        ]

        # 收集截面数据
        factor_cross = {}
        return_cross = {}
        for code in stock_pool:
            try:
                s_data = fetcher.get_stock_info(code)
                klines = fetcher.get_kline_data(code, 'daily', 60)
                if not s_data or not klines:
                    continue

                # 21 因子走 dynamic_cache 内存缓存 (同数据 5min 内重复刷新免重算)
                factors = mfm.calculate_all_factors_cached(code, s_data, klines=klines)
                # 未来收益用当日涨跌幅 proxy
                future_ret = s_data.get('change_pct', 0) / 100.0

                factor_cross[code] = factors
                return_cross[code] = future_ret
            except Exception:
                continue

        # 截面 IC
        pearson_ic = monitor.compute_cross_sectional_ic(
            {'today': factor_cross},
            {'today': return_cross}
        )
        rank_ic = monitor.compute_rank_ic(
            {'today': factor_cross},
            {'today': return_cross}
        )

        # 因子 IC 统计
        ic_stats = {}
        for fname, fval in factor_cross.get('today', {}).items():
            rval = return_cross.get('today', 0)
            # 单点 IC 无法计算，用历史数据
            ic_stats[fname] = {'ic_pearson': pearson_ic, 'ic_rank': rank_ic}

        # IC QQ-图数据
        qq_data = ICQQPlot.compute_qq_data([pearson_ic, rank_ic])

        # 多空分析
        ls_result = long_short_analyzer.compute_long_short(
            factor_cross.get('today', {}),
            return_cross,
            n_groups=5,
        )

        # 因子组 IC
        factor_groups = mfm.FACTOR_CATEGORIES
        group_ic = factor_group_ic.compute_group_ic(
            {'today': factor_cross},
            {'today': return_cross},
            factor_groups,
        )

        return jsonify({
            'success': True,
            'pearson_ic': round(pearson_ic, 6),
            'rank_ic': round(rank_ic, 6),
            'factor_ic_stats': ic_stats,
            'qq_plot': qq_data,
            'long_short': ls_result,
            'group_ic': group_ic,
            'stock_count': len(factor_cross),
        })
    except Exception as e:
        logger.error(f"[DashboardAPI] IC 增强分析失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


# ── 单元测试 ──────────────────────────────────────────────

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
