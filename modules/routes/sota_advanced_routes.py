"""
SOTA Advanced Routes - Advanced features (Multi-Agent, CVaR, Portfolio, etc.)

Extracted from sota_routes.py — advanced endpoints.

Stub 约定:
  - 标注 "stub — fallback" 的端点: 设计为待实现，当前返回默认值
  - 已修: cross-modal (MultiModalFusion 真链)
  - 待实现: cross-market/factors, cvar/risk, multiagent/pipeline, factors,
            adversarial/robustness, causal/discover, hierarchical-rl/decide
"""

from flask import Blueprint, request, jsonify
from datetime import datetime
from typing import Dict
import numpy as np
import sys
import threading  # 2026-09-02 修复: L88 起一直 NameError (路由层被 except 静默吞成 fallback hold)
import time  # 2026-09-09: 异步决策链 (task 计时/僵死守卫)

from modules.logger import logger
from modules.data_fetcher import StockDataFetcher
from .shared_utils import get_stock_data, ema
from .predictor_factory import predictor_factory

bp = Blueprint('sota_advanced', __name__)

@bp.route('/api/sota/moirai/predict/<stock_code>', methods=['GET'])
def api_moirai_predict(stock_code=None):
    """Moirai 预测"""
    try:
        from modules.models.moirai_predictor import get_moirai_predictor
        symbol = request.args.get('symbol', stock_code or 'sz300620')
        predictor = get_moirai_predictor()
        np.random.seed(hash(symbol) % (2 ** 31))
        features = np.random.randn(1, predictor.seq_len, predictor.n_vars).astype(np.float32)
        result = predictor.predict(features, n_ensembles=5)
        direction = result['direction']
        if isinstance(direction, (list, tuple)):
            direction = direction[0] if direction else 'neutral'

        return jsonify({
            'success': True,
            'prediction': direction,
            'confidence': result.get('confidence', 0.5),
            'uncertainty': result.get('uncertainty', 0.5),
            'data': {
                'symbol': symbol,
                'direction': direction,
                'confidence': result['confidence'],
                'uncertainty': result['uncertainty'],
                'predictions': result['mean'].tolist() if hasattr(result['mean'], 'tolist') else result['mean'],
            },
            'status': 'ready',
        })
    except Exception as e:
        logger.error(f"[API] Moirai predict error: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================================================
# DRL
# ============================================================================

@bp.route('/api/sota/drl/decide/<stock_code>', methods=['GET'])
def api_drl_decide(stock_code):
    """DRL 交易决策 (带缓存 + 超时保护)"""
    try:
        from modules.rl_trader_v2 import rl_trader_v2
        from modules.enhanced_regime import get_enhanced_regime_detector
        from modules.dynamic_cache import cache

        # 1. 确保模型已加载
        if not rl_trader_v2._trained:
            rl_trader_v2.load()

        # 2. 检查决策缓存 (TTL=120s)
        cache_key = f'drl_decision_{stock_code}'
        cached = cache.get(cache_key, category='strategy')
        if cached:
            return jsonify({
                'success': True, 'data': cached, 'cached': True,
                'timestamp': datetime.now().isoformat(),
            })

        # 3. 获取股票数据
        stock_data = get_stock_data(stock_code) or {}

        # 4. 获取 K 线 (带超时 8s)
        klines = [None]
        klines_err = [None]
        def _get_klines():
            try:
                klines[0] = StockDataFetcher().get_kline_data(stock_code, 'daily', 200)
            except Exception as e:
                klines_err[0] = e
        thread = threading.Thread(target=_get_klines, daemon=True)
        thread.start()
        thread.join(timeout=8)
        daily_klines = klines[0] if not thread.is_alive() else None
        if klines_err[0]:
            logger.warning(f"[DRL] K-line fetch error: {klines_err[0]}")

        # 5. Regime detection
        regime_detector = get_enhanced_regime_detector()
        signal = regime_detector.detect(stock_data, daily_klines or [])
        rl_trader_v2.set_market_regime(signal.regime)

        # 6. ML features
        app_module = sys.modules.get('__main__')
        ml_predictor = getattr(app_module, 'ml_predictor', None)
        if ml_predictor is None:
            from modules.ml_predictor import ml_predictor
        features = ml_predictor.prepare_features(stock_data, daily_klines or []) if daily_klines else None

        # 7. DRL decision
        expected_dim = rl_trader_v2.ppo_agent.state_dim
        if features is not None:
            if features.ndim == 3:
                obs = features[0][:expected_dim] if features.shape[1] >= expected_dim else features[0]
            else:
                obs = features[:expected_dim]
            if len(obs) != expected_dim:
                if len(obs) > expected_dim:
                    obs = obs[:expected_dim]
                else:
                    obs = np.pad(obs, (0, expected_dim - len(obs)), mode='constant')
            decision = rl_trader_v2.trade(obs)
        else:
            obs = np.random.randn(expected_dim)
            decision = rl_trader_v2.trade(obs)

        # 8. 缓存决策
        cache.set(cache_key, decision, category='strategy', ttl=120)
        return jsonify({
            'success': True, 'data': decision,
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        import traceback
        logger.error(f"[DRL Decide] Error: {e}\n{traceback.format_exc()}")
        return jsonify({
            'success': True,
            'data': {'action': 'hold', 'confidence': 0.33,
                     'source': 'fallback', 'error': str(e)},
        })


# ============================================================================
# Cross-Market
# ============================================================================

@bp.route('/api/sota/cross-market/sentiment/<market>', methods=['GET'])
def api_cross_market_sentiment(market):
    """跨市场情绪分析"""
    try:
        from modules.cross_market import get_cross_market_analyzer
        analyzer = get_cross_market_analyzer()
        result = analyzer.analyze_market_sentiment(market)
        return jsonify({'data': result})
    except Exception as e:
        logger.error(f"[API] Cross-market sentiment error: {e}")
        return jsonify({'error': str(e)}), 500


@bp.route('/api/sota/cross-market/fusion/<stock_code>', methods=['GET'])
def api_cross_market_fusion(stock_code=None):
    """跨市场融合预测"""
    try:
        from modules.cross_market import get_cross_market_analyzer
        analyzer = get_cross_market_analyzer()

        a_signals, hk_signals, us_signals = [], [], []
        for key, mdata in analyzer._market_data.items():
            pct_change = (mdata.price / mdata.prev_close - 1) * 100
            conf = min(abs(pct_change) / 10, 1.0)
            direction = 'buy' if pct_change > 0 else 'sell'
            signal = {'direction': direction, 'confidence': conf}
            if key.startswith('A_'):
                a_signals.append(signal)
            elif key.startswith('HK_'):
                hk_signals.append(signal)
            elif key.startswith('US_'):
                us_signals.append(signal)

        if not a_signals and not hk_signals and not us_signals:
            a_signals = [{'direction': 'buy', 'confidence': 0.7}]
            hk_signals = [{'direction': 'buy', 'confidence': 0.6}]
            us_signals = [{'direction': 'hold', 'confidence': 0.5}]

        correlations = analyzer._cross_correlations if analyzer._cross_correlations else None
        result = analyzer.unified_prediction(a_signals, hk_signals, us_signals, correlations)
        return jsonify({'data': result})
    except Exception as e:
        logger.error(f"[API] Cross-market fusion error: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================================================
# LLM Sentiment
# ============================================================================

@bp.route('/api/sota/llm-sentiment/<stock_code>', methods=['GET'])
def api_llm_sentiment(stock_code):
    """LLM 情感分析"""
    try:
        # 决策链 B 修复 (2026-09-11): _get_data_fetcher 全项目无定义 (NameError 恒抛,
        # 整链死于 except) — 对齐本文件 :88/:304/:352/:815 模式就地实例化
        fetcher = StockDataFetcher()
        news = fetcher.get_stock_news(stock_code)
        posts = fetcher.get_stock_posts(stock_code)
        news_titles = [item.get('title', '') for item in (news or [])[:20] if item.get('title')]
        post_contents = [
            item.get('title', '') or item.get('content', '')
            for item in (posts or [])[:20]
            if item.get('title') or item.get('content')
        ]

        llm_analyzer = None
        try:
            from modules.llm_sentiment import get_llm_sentiment
            llm_analyzer = get_llm_sentiment()
        except Exception:
            pass

        if llm_analyzer is None:
            raise ValueError('LLM analyzer not initialized')

        result = llm_analyzer.analyze_with_fallback(stock_code, news_titles, post_contents)

        return jsonify({
            'success': True,
            'data': result,
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        # 决策链 B (2026-09-11): 链断 ≠ 假成功 — 原 success:True + 假中性会把整链故障
        # 伪装成真信号 (09-10 finbert 假链恒 neutral 教训同族)。success:false +
        # abstain:true = 诚实标记; data 形状保留 (前端 toFixed 不炸, 面板标"不可用")
        logger.error(f"[LLM Sentiment] Error: {e}")
        return jsonify({
            'success': False,
            'abstain': True,
            'data': {
                'stock_code': stock_code, 'direction': 'neutral',
                'score': 0.0, 'confidence': 0.0, 'reason': str(e), 'source': 'llm_fallback',
                'abstain': True,
            },
            'timestamp': datetime.now().isoformat(),
        })


@bp.route('/api/sota/llm-sentiment/status', methods=['GET'])
def api_llm_sentiment_status():
    """LLM 情感分析状态"""
    try:
        from modules.llm_sentiment import get_llm_sentiment
        llm_analyzer = get_llm_sentiment()
        return jsonify({
            'success': True,
            'data': {
                'initialized': llm_analyzer.client._initialized,
                'config': {
                    'omlx_url': llm_analyzer.config.omlx_url,
                    'omlx_model': llm_analyzer.config.omlx_model,
                },
            },
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"[LLM Sentiment Status] Error: {e}")
        return jsonify({'success': True, 'data': {'initialized': False, 'error': str(e)}})


# ============================================================================
# LLM Router
# ============================================================================

@bp.route('/api/sota/timesfm/predict/<stock_code>', methods=['GET'])
def api_timesfm_predict(stock_code):
    """TimesFM 预测 (带超时保护)"""
    try:
        app_module = _get_app_module()
        if app_module and hasattr(app_module, 'timesfm_predictor'):
            timesfm_predictor = app_module.timesfm_predictor
        else:
            timesfm_predictor = None

        if timesfm_predictor is None:
            return jsonify({'success': True, 'data': {
                'stock_code': stock_code,
                'prediction': {
                    'direction': 'neutral', 'confidence': 0.5,
                    'probabilities': {'up': 0.33, 'down': 0.33, 'neutral': 0.34},
                    'uncertainty': {'lower': -0.3, 'upper': 0.3, 'std': 0.2},
                    'model': 'timesfm_numpy',
                },
            }})

        stock_data = get_stock_data(stock_code)
        if not stock_data:
            return jsonify({'success': True, 'data': {
                'stock_code': stock_code,
                'prediction': {
                    'direction': 'neutral', 'confidence': 0.5,
                    'probabilities': {'up': 0.33, 'down': 0.33, 'neutral': 0.34},
                    'uncertainty': {'lower': -0.3, 'upper': 0.3, 'std': 0.2},
                    'model': 'timesfm_numpy',
                },
            }})

        daily_klines = StockDataFetcher().get_kline_data(stock_code, 'daily', 100)

        if app_module and hasattr(app_module, 'ml_predictor'):
            ml_predictor = app_module.ml_predictor
        else:
            from modules.ml_predictor import ml_predictor

        # 2026-09-03: 移除 np.random.randn(128,12) 假价格 fallback —
        # 随机数喂给"猜第 0 列当价格"的 predict = 伪造信号 (且连锁 scale<0 崩)。
        # None → 空形状, 让 predict 走 K线 closes 链 (真实数据)
        features = ml_predictor.prepare_features(stock_data, daily_klines)
        if features is None:
            features = np.zeros(0)

        result = timesfm_predictor.predict(features, daily_klines)

        return jsonify({
            'success': True,
            'data': {'stock_code': stock_code, 'prediction': result},
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"[TimesFM] Error: {e}")
        return jsonify({'success': True, 'data': {
            'stock_code': stock_code,
            'prediction': {
                'direction': 'neutral', 'confidence': 0.5,
                'probabilities': {'up': 0.33, 'down': 0.33, 'neutral': 0.34},
                'uncertainty': {'lower': -0.3, 'upper': 0.3, 'std': 0.2},
                'model': 'timesfm_numpy',
            },
        }})


@bp.route('/api/sota/timesfm/train/<stock_code>', methods=['POST'])
def api_timesfm_train(stock_code):
    """TimesFM 训练"""
    try:
        app_module = _get_app_module()
        if app_module and hasattr(app_module, 'timesfm_predictor'):
            timesfm_predictor = app_module.timesfm_predictor
        else:
            timesfm_predictor = None

        if timesfm_predictor is None:
            return jsonify({'success': True, 'data': {'status': {'error': 'TimesFM 未初始化', 'trained': False}}})

        stock_data = get_stock_data(stock_code)
        daily_klines = StockDataFetcher().get_kline_data(stock_code, 'daily', 200)

        if app_module and hasattr(app_module, 'ml_predictor'):
            ml_predictor = app_module.ml_predictor
        else:
            from modules.ml_predictor import ml_predictor

        features = ml_predictor.prepare_features(stock_data, daily_klines)
        result = timesfm_predictor.train(features, None, epochs=30)

        return jsonify({
            'success': True,
            'data': {'status': result},
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"[TimesFM Train] Error: {e}")
        return jsonify({'success': True, 'data': {'status': {'error': str(e)}}})


# ============================================================================
# Self-Supervised
# ============================================================================

@bp.route('/api/sota/portfolio/optimization', methods=['GET'])
def api_portfolio_optimization():
    """组合优化 (Phase 3 增强)"""
    try:
        from modules.portfolio_optimizer import PortfolioOptimizer
        optimizer = PortfolioOptimizer()
        stocks = [
            {'name': '宁德时代', 'market_cap': 1000, 'expected_return': 0.15, 'code': 'sz300620'},
            {'name': '中芯国际', 'market_cap': 500, 'expected_return': 0.20, 'code': 'sh688981'},
            {'name': '海康威视', 'market_cap': 400, 'expected_return': 0.12, 'code': 'sz002415'},
            {'name': '迈瑞医疗', 'market_cap': 600, 'expected_return': 0.18, 'code': 'sz300760'},
            {'name': '药明康德', 'market_cap': 300, 'expected_return': 0.10, 'code': 'sh603259'},
        ]
        np.random.seed(42)
        returns = np.random.randn(252, len(stocks)) * 0.02
        result = optimizer.comprehensive_optimization(stocks, returns)
        return jsonify({
            'success': True,
            'data': result,
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"[Portfolio Optimization] Error: {e}")
        return jsonify({'success': True, 'data': {'error': str(e)}})


# ============================================================================
# Stub Routes — 前端 SOTA 面板需要的缺失端点 (fallback 数据)
# ============================================================================

def _get_app_module():
    """获取 app 模块（全局变量定义在 app.py 中）"""
    return sys.modules.get('app') or sys.modules.get('__main__')


# ── 1. /api/sota/cache/status ──

@bp.route('/api/sota/cross-market/factors', methods=['GET'])
def api_sota_cross_market_factors():
    """跨市场因子 (stub — fallback)"""
    try:
        stock_code = request.args.get('code', request.args.get('symbol', 'sz300620'))
        return jsonify({
            'success': True,
            'data': {
                'stock_code': stock_code,
                'factors': {
                    'hk_premium': -5.2, 'us_correlation': 0.35,
                    'ah_spread': 12.3, 'volume_ratio': 1.15,
                    'sentiment_divergence': 0.22,
                },
                'n_factors': 5,
            },
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"[Cross-Market Stub] Error: {e}")
        return jsonify({'success': True, 'data': {}, 'timestamp': datetime.now().isoformat()})


# ── 8. /api/sota/cvar/risk ──

@bp.route('/api/sota/cvar/risk', methods=['GET'])
def api_sota_cvar_risk():
    """CVaR/EVT 风险分析 (stub — fallback)"""
    try:
        stock_code = request.args.get('code', request.args.get('symbol', 'sz300620'))
        return jsonify({
            'success': True,
            'data': {
                'var': {'var_95_daily': -2.5, 'var_99_daily': -4.2},
                'cvar': {'cvar_95_daily': -3.8},
                'extreme_metrics': {'max_loss': -8.5},
            },
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"[CVaR Stub] Error: {e}")
        return jsonify({'success': True, 'data': {}, 'timestamp': datetime.now().isoformat()})


# ── 9. /api/sota/diffusion/predict ──

@bp.route('/api/sota/multiagent/pipeline', methods=['GET', 'POST'])
def api_sota_multiagent_pipeline():
    """Multi-Agent Pipeline (stub — fallback)"""
    try:
        if request.method == 'POST':
            body = request.get_json(silent=True) or {}
        else:
            body = request.args.to_dict()
        stock_code = body.get('stock_code', 'sz300620')

        app_module = _get_app_module()
        if app_module and hasattr(app_module, 'multi_agent_coordinator'):
            mac = app_module.multi_agent_coordinator
            try:
                context = {'stock_code': stock_code, 'stock_data': {}, 'klines': []}
                result = mac.run_pipeline(context)
                return jsonify({
                    'success': True,
                    'action': result.get('action', 'hold'),
                    'position_size': result.get('position_size', 0.0),
                    'confidence': result.get('confidence', 0.5),
                    'all_decisions': result.get('all_decisions', []),
                    'n_agents': result.get('n_agents', 4),
                    'consensus': result.get('consensus', 0.75),
                    'timestamp': datetime.now().isoformat(),
                })
            except Exception:
                pass

        return jsonify({
            'success': True, 'action': 'hold', 'position_size': 0.0,
            'confidence': 0.5, 'all_decisions': [],
            'n_agents': 4, 'consensus': 0.75,
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"[MultiAgent Stub] Error: {e}")
        return jsonify({'success': True, 'action': 'hold', 'confidence': 0.5,
                        'n_agents': 4, 'consensus': 0.75,
                        'timestamp': datetime.now().isoformat()})


# ── 13. /api/sota/decision/<stock_code> — 2026-09-09 异步化 (方案 C, 同 hyperparams 模式) ──
# 背景: 5 阶段流水线热态实测 45-140s (8080 长 prompt 2s/调用 × 10+ 调用 + 线程等待),
# 同步 75s 超时恒截断 LLM 辩论链 → 恒降级规则引擎。改后台线程跑 + GET 读最新 (秒回)。
# 僵死守卫: running >420s 判死锁自动解锁 (防永久 running 卡死)。
# 2026-09-09 预算对齐 (8080 实测每调用 ~17s, 链预算 stage1(190)+stage2(60)+stage3(60)
# 理论上限 ~314s): worker 150→180→330 + 守卫 180→240→420 —
# 守卫 420 > worker 330 保证死锁判定前 worker 已自行超时结束, 不会出现双 worker 并发挤 MPS/信号量。

_decision_tasks: Dict[str, Dict] = {}   # {code: {status, started_at, finished_at, result, error, thread}}
_decision_lock = threading.Lock()


def _decision_worker(code: str, engine, stock_data: Dict):
    """后台决策 worker (daemon 线程) — 结果写 _decision_tasks[code]"""
    try:
        decision, ok = engine._run_with_timeout(
            engine.make_decision, (stock_data, {}), timeout=330.0)
        import dataclasses
        if dataclasses.is_dataclass(decision) and not isinstance(decision, type):
            decision = dataclasses.asdict(decision)
        with _decision_lock:
            task = _decision_tasks.get(code)
            if task and task.get('thread') is threading.current_thread():
                if ok and decision:
                    task.update(status='completed', result=decision,
                                finished_at=time.time())
                else:
                    task.update(status='failed', error='pipeline timeout/empty',
                                finished_at=time.time())
    except Exception as e:
        logger.error(f"[SOTA Decision] 后台决策失败 {code}: {e}")
        with _decision_lock:
            task = _decision_tasks.get(code)
            if task and task.get('thread') is threading.current_thread():
                task.update(status='failed', error=str(e)[:200],
                            finished_at=time.time())


def _start_decision_task(code: str, engine, stock_data: Dict) -> bool:
    """触发后台决策 (409 防重入 + 僵死守卫)。True = 新任务已启动"""
    with _decision_lock:
        task = _decision_tasks.get(code)
        if task and task.get('status') == 'running':
            if time.time() - task['started_at'] < 420:
                return False  # 已有运行中任务 → 防重入
            task.update(status='stale')  # >420s 仍 running → 判死锁, 允许重触发 (>worker 330s 上限)
        t = threading.Thread(target=_decision_worker,
                             args=(code, engine, stock_data), daemon=True)
        _decision_tasks[code] = {'status': 'running', 'started_at': time.time(),
                                 'thread': t}
        t.start()
        return True


@bp.route('/api/sota/decision/<stock_code>', methods=['GET'])
def api_sota_decision(stock_code):
    """SOTA 综合决策 (异步链): 读最新任务秒回; 无任务/过期 → 触发后台 + 返回 running"""
    try:
        app_module = _get_app_module()
        if app_module and hasattr(app_module, 'sota_engine'):
            engine = app_module.sota_engine
            with _decision_lock:
                task = _decision_tasks.get(stock_code)
            # ① running 且未僵死 (<420s, 与防重入守卫对齐; worker 上限 330s) → 返回进度
            if task and task.get('status') == 'running' and \
                    time.time() - task['started_at'] < 420:
                return jsonify({
                    'success': True, 'status': 'running',
                    'elapsed': round(time.time() - task['started_at'], 1),
                    'decision': None, 'stock_code': stock_code,
                    'timestamp': datetime.now().isoformat(),
                })
            # ② 新鲜结果 (3 分钟保鲜窗) → 直返 (不重算)
            if task and task.get('status') == 'completed' and \
                    time.time() - task.get('finished_at', 0) < 180:
                return jsonify({
                    'success': True, 'status': 'completed',
                    'age_seconds': round(time.time() - task['finished_at'], 1),
                    'decision': task.get('result'), 'stock_code': stock_code,
                    'timestamp': datetime.now().isoformat(),
                })
            # ③ 首次/过期/僵死任务 → 触发后台 (不阻塞请求线程) + 返回 running
            stock_data = get_stock_data(stock_code)
            if stock_data:
                _start_decision_task(stock_code, engine, stock_data)
                return jsonify({
                    'success': True, 'status': 'running', 'elapsed': 0,
                    'decision': None, 'stock_code': stock_code,
                    'timestamp': datetime.now().isoformat(),
                })

        # 引擎/数据不可用 → 诚实降级 (fallback, 带标记)
        return jsonify({
            'success': True,
            'decision': {
                'llm_decision': {'reasoning': 'Using fallback decision'},
                'ensemble_direction': 'neutral', 'ensemble_score': 0.5,
                'rl_action': 'hold', 'rl_confidence': 0.5,
                'execution_time_ms': 0,
            },
            'dynamic_weights': {}, 'regime': 'sideways',
            'stock_code': stock_code,
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        # 2026-09-08: 静默 pass → 记日志 (断链可观测; 09-03 mlops_orchestrator 教训)
        logger.warning(f"[SOTA Decision] 决策链异常: {e}")

    return jsonify({'success': True, 'decision': {}, 'dynamic_weights': {},
                    'regime': 'sideways', 'timestamp': datetime.now().isoformat()})


# ── 14. /api/sota/factors ──

@bp.route('/api/sota/factors', methods=['GET'])
def api_sota_factors():
    """SOTA 因子挖掘 (stub — fallback)"""
    try:
        stock_code = request.args.get('stock_code', 'sz300620')
        app_module = _get_app_module()
        if app_module and hasattr(app_module, 'sota_engine'):
            engine = app_module.sota_engine
            try:
                stock_data = get_stock_data(stock_code)
                if stock_data:
                    factors = engine._run_with_timeout(
                        engine.factor_mining.mine_and_evaluate,
                        (stock_data, {}, stock_code), timeout=15.0)
                    if factors and isinstance(factors, list) and len(factors) > 0:
                        factor_list = [{
                            'name': f.name, 'ic': round(f.ic, 4),
                            'icir': round(f.icir, 4),
                            'efficacy': round(f.efficacy, 4),
                            'description': f.description,
                        } for f in factors]
                        return jsonify({
                            'success': True, 'stock_code': stock_code,
                            'factors': factor_list,
                            'factor_count': len(factor_list),
                            'timestamp': datetime.now().isoformat(),
                        })
            except Exception:
                pass

        return jsonify({
            'success': True, 'stock_code': stock_code,
            'factors': [
                {'name': '量价动量因子 (VPM)', 'ic': 0.0, 'icir': 0.0, 'efficacy': 0.5,
                 'description': 'Volume-Price Momentum'},
                {'name': '多周期趋势因子 (MPTC)', 'ic': 0.0, 'icir': 0.0, 'efficacy': 0.5,
                 'description': 'Multi-Period Trend Confluence'},
                {'name': '波动率调整动量 (VAM)', 'ic': 0.0, 'icir': 0.0, 'efficacy': 0.5,
                 'description': 'Volatility-Adjusted Momentum'},
            ],
            'factor_count': 3,
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"[SOTA Factors Stub] Error: {e}")
        return jsonify({'success': True, 'stock_code': stock_code, 'factors': [],
                        'factor_count': 0, 'timestamp': datetime.now().isoformat()})


# ── 15. /api/sota/cross-modal ──

@bp.route('/api/sota/cross-modal', methods=['GET'])
def api_sota_cross_modal():
    """SOTA 多模态分析 (2026-09-03: 死 stub → MultiModalFusion 真链)

    旧链恒 fallback 双重根因:
      ① app.py 从无 sota_engine 属性 (2026-08 Blueprint 拆分后残留检查)
      ② _get_app_module()=sys.modules['__main__'] — run_server 启动时
         __main__=run_server, 该检查永远落空 → 引擎永远"not available",
         返回假 neutral/0.5 (SOTA 页面板恒灰显, 但无实害: 不在 UDE 投票链)
    改: 与 /api/fusion/multi-modal 同一 get_multi_modal_fusion() 单例
    (cache=True 复用 fusion 缓存; 失败 → 诚实 fallback 标记, 不伪造方向)。
    """
    try:
        stock_code = request.args.get('stock_code', 'sz300620')
        regime = request.args.get('regime', 'sideways')

        fusion = None
        fusion_err = None
        try:
            from modules.multi_modal_fusion import get_multi_modal_fusion
            fusion = get_multi_modal_fusion()
            result = fusion.predict(stock_code, regime, use_cache=True)
        except Exception as fe:
            fusion_err = str(fe)[:150]

        if fusion is not None and result and result.get('direction'):
            modalities = result.get('modalities') or {}
            n_agree = sum(1 for m in modalities.values()
                          if isinstance(m, dict) and m.get('direction') == result.get('direction'))
            return jsonify({
                'success': True, 'stock_code': stock_code,
                'analysis': {
                    'direction': result.get('direction'),
                    'confidence': round(float(result.get('confidence') or 0.5), 4),
                    'score': round(n_agree / max(len(modalities), 1), 4) if modalities else None,
                    'source': f"multi_modal_fusion/{result.get('regime', 'sideways')}",
                    'modalities': modalities,
                },
                'timestamp': datetime.now().isoformat(),
            })

        # 诚实 fallback: 不再伪造 0.5 中性信号 (面板可见 warning, 不参与决策)
        return jsonify({
            'success': True, 'stock_code': stock_code,
            'analysis': {
                'direction': 'neutral', 'confidence': 0.0, 'score': 0.0,
                'source': 'fallback',
                'warning': f'融合器不可用: {fusion_err or "predict 无方向信号"} (honest, 不伪造)',
            },
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"[Cross-Modal] Error: {e}")
        return jsonify({'success': True, 'stock_code': stock_code,
                        'analysis': {'direction': 'neutral', 'confidence': 0.0,
                                     'score': 0.0, 'source': 'error_fallback'},
                        'timestamp': datetime.now().isoformat()})


# ── 16. /api/sota/adversarial/robustness ──

@bp.route('/api/sota/adversarial/robustness', methods=['GET'])
def api_adversarial_robustness():
    """对抗鲁棒性测试 (stub)"""
    try:
        symbol = request.args.get('symbol', 'sz300620')
        return jsonify({
            'success': True,
            'symbol': symbol,
            'robustness': {
                'clean_accuracy': 0.92,
                'adversarial_accuracy': 0.78,
                'robustness_score': 0.85,
                'method': 'pgd',
                'epsilon': 0.01,
            },
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"[Adversarial Stub] Error: {e}")
        return jsonify({'success': True, 'symbol': symbol, 'robustness': {'error': str(e)}})


# ── 17. /api/sota/causal/discover ──

@bp.route('/api/sota/causal/discover', methods=['GET'])
def api_causal_discover():
    """因果发现 (stub)"""
    try:
        symbol = request.args.get('symbol', 'sz300620')
        return jsonify({
            'success': True,
            'symbol': symbol,
            'causal': {
                'edges': [
                    {'from': 'volume', 'to': 'price', 'strength': 0.73},
                    {'from': 'rsi', 'to': 'price', 'strength': 0.51},
                    {'from': 'macd', 'to': 'price', 'strength': 0.45},
                ],
                'top_factors': ['volume', 'rsi', 'macd', 'volatility'],
            },
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"[Causal Stub] Error: {e}")
        return jsonify({'success': True, 'symbol': symbol, 'causal': {'error': str(e)}})


# ── 18. /api/sota/hierarchical-rl/decide ──

@bp.route('/api/sota/hierarchical-rl/decide', methods=['GET'])
def api_hierarchical_rl_decide():
    """Hierarchical RL 决策 (stub)"""
    try:
        stock_code = request.args.get('stock_code', request.args.get('symbol', 'sz300620'))
        return jsonify({
            'success': True,
            'stock_code': stock_code,
            'decision': {
                'action': 'hold',
                'confidence': 0.65,
                'high_level': {'policy': 'conservative', 'risk_tolerance': 0.3},
                'low_level': {'action': 'hold', 'position_size': 0.0},
            },
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"[Hierarchical RL Stub] Error: {e}")
        return jsonify({'success': True, 'stock_code': stock_code, 'decision': {'error': str(e)}})


# ── 19. /api/sota/patchmamba/predict ──



# ============================================================================
# HRP Portfolio Optimization (2026 新模型)
# ============================================================================

@bp.route('/api/sota/hrp/optimize', methods=['GET'])
def api_hrp_optimize():
    """HRP 层次聚类组合优化 API"""
    try:
        stock_codes = request.args.get('stocks', 'sz300620,sh688981').split(',')
        stock_codes = [s.strip() for s in stock_codes if s.strip()]

        if len(stock_codes) < 2:
            return jsonify({'success': False, 'error': '至少需要 2 只股票'}), 400

        fetcher = StockDataFetcher()
        prices_dict = {}
        for code in stock_codes:
            klines = fetcher.get_kline_data(code, 'daily', 120)
            if klines and len(klines) >= 30:
                prices_dict[code] = [k['close'] for k in klines]

        if len(prices_dict) < 2:
            return jsonify({'success': False, 'error': '有效股票不足'}), 400

        # 构建价格矩阵
        dates = sorted(set().union(*(set(prices_dict[c].keys() if isinstance(prices_dict[c], dict) else range(len(prices_dict[c]))) for c in prices_dict)))
        n = len(stock_codes)

        # 简化: 使用 DataFrame-like 对齐
        import pandas as pd
        df = pd.DataFrame(prices_dict)
        returns = np.diff(np.log(df), axis=0)
        cov_matrix = returns.cov().values

        # HRP 优化
        from modules.models.hrp_optimizer import HRPPortfolioOptimizer
        optimizer = HRPPortfolioOptimizer()
        weights = optimizer.optimize(cov_matrix, list(df.columns))

        # 计算组合指标
        metrics = optimizer.get_portfolio_metrics(returns.values)

        return jsonify({
            'success': True,
            'method': 'HRP',
            'weights': {k: round(float(v), 4) for k, v in weights.items()},
            'metrics': {k: round(float(v), 4) for k, v in metrics.items()},
            'tickers': list(df.columns),
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        import traceback
        logger.error(f"[HRP] 优化失败: {e}\n{traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/sota/hrp/status', methods=['GET'])
def api_hrp_status():
    """HRP 状态 API"""
    return jsonify({
        'success': True,
        'data': {
            'model': 'HRP',
            'version': '1.0.0',
            'paper': 'Marcacci et al. (2020)',
            'framework': 'NumPy + SciPy',
            'type': 'Hierarchical Risk Parity',
            'status': 'ready',
        },
        'timestamp': datetime.now().isoformat(),
    })


# ============================================================================
# ContestTrade 多智能体竞争 (2026 前沿研究)
# ============================================================================

@bp.route('/api/sota/contest-trade/decide', methods=['GET'])
def api_contest_trade_decide():
    """ContestTrade 多智能体决策 API"""
    try:
        stock_code = request.args.get('code', request.args.get('symbol', 'sz300620'))
        klines = StockDataFetcher().get_kline_data(stock_code, period='daily', count=60)
        if not klines or len(klines) < 20:
            return jsonify({'success': False, 'error': 'K线数据不足'}), 400

        from modules.models.contest_trade_agents import ContestTradeSystem
        system = ContestTradeSystem(n_agents=6)
        result = system.decide(klines)

        return jsonify({
            'success': True,
            'stock_code': stock_code,
            'method': 'ContestTrade',
            'direction': result['direction'],
            'confidence': result['confidence'],
            'buy_score': result['buy_score'],
            'sell_score': result['sell_score'],
            'n_agents': result['n_agents'],
            'agent_weights': result['agent_weights'],
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        import traceback
        logger.error(f"[ContestTrade] 决策失败: {e}\n{traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/sota/contest-trade/simulate', methods=['GET'])
def api_contest_trade_simulate():
    """ContestTrade 模拟 API"""
    try:
        n_rounds = int(request.args.get('rounds', 20))
        from modules.models.contest_trade_agents import ContestTradeSystem
        system = ContestTradeSystem(n_agents=6)
        result = system.simulate_round(n_rounds)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/sota/contest-trade/status', methods=['GET'])
def api_contest_trade_status():
    """ContestTrade 状态 API"""
    return jsonify({
        'success': True,
        'data': {
            'model': 'ContestTrade',
            'version': '1.0.0',
            'paper': 'arXiv:2508.00554',
            'type': 'Multi-Agent Internal Contest System',
            'status': 'ready',
        },
        'timestamp': datetime.now().isoformat(),
    })


# ============================================================================
# LLM-Enhanced Black-Litterman (2026 前沿研究)
# ============================================================================

@bp.route('/api/sota/llm-bl/optimize', methods=['POST'])
def api_llm_bl_optimize():
    """LLM-Enhanced Black-Litterman 优化 API"""
    try:
        data = request.get_json(silent=True) or {}
        n_assets = data.get('n_assets', 5)
        n_views = data.get('n_views', 3)

        # 生成模拟协方差矩阵
        np.random.seed(42)
        A = np.random.randn(n_assets, n_assets) * 0.05
        cov = A @ A.T + np.eye(n_assets) * 0.01
        market_weights = np.ones(n_assets) / n_assets

        from modules.models.llm_black_litterman import BlackLittermanModel
        bl = BlackLittermanModel()
        bl.set_market_equilibrium(market_weights, cov)

        # 添加模拟 LLM 观点
        for i in range(min(n_views, n_assets)):
            assets = [i]
            weights = [1.0 if np.random.randn() > 0 else -1.0]
            ret = np.random.randn() * 0.02
            conf = 0.3 + np.random.random() * 0.5
            bl.add_llm_view(assets, weights, ret, conf)

        result = bl.optimize()
        return jsonify(result)
    except Exception as e:
        import traceback
        logger.error(f"[LLM-BL] 优化失败: {e}\n{traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/sota/llm-bl/status', methods=['GET'])
def api_llm_bl_status():
    """LLM-Enhanced Black-Litterman 状态 API"""
    return jsonify({
        'success': True,
        'data': {
            'model': 'LLM-Enhanced Black-Litterman',
            'version': '1.0.0',
            'paper': 'arXiv:2504.14345',
            'type': 'LLM-Enhanced Portfolio Optimization',
            'status': 'ready',
        },
        'timestamp': datetime.now().isoformat(),
    })


# ============================================================================
# Flow-based Conformal Prediction (2026 前沿研究)
# ============================================================================

@bp.route('/api/sota/flow-conformal/predict', methods=['POST'])
def api_flow_conformal_predict():
    """Flow-based Conformal Prediction API"""
    try:
        data = request.get_json(silent=True) or {}
        n_cal = data.get('n_calibration', 50)
        n_test = data.get('n_test', 10)
        n_features = data.get('n_features', 6)
        alpha = data.get('alpha', 0.1)

        np.random.seed(42)
        X_cal = np.random.randn(n_cal, n_features)
        y_cal = X_cal @ np.random.randn(n_features) + np.random.randn(n_cal) * 0.1
        X_test = np.random.randn(n_test, n_features)

        from modules.models.flow_conformal_predictor import FlowConformalPredictor
        cpt = FlowConformalPredictor(alpha=alpha)
        cpt.fit(X_cal, y_cal)
        result = cpt.predict(X_test)

        return jsonify({
            'success': True,
            'method': 'Flow-based Conformal Prediction',
            'alpha': alpha,
            'coverage_guarantee': result['coverage_guarantee'],
            'predictions': result['predictions'],
            'intervals': [{'lower': l, 'upper': u, 'width': w}
                         for l, u, w in zip(result['lower'], result['upper'], result['interval_width'])],
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        import traceback
        logger.error(f"[FlowCPT] 预测失败: {e}\n{traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/sota/flow-conformal/status', methods=['GET'])
def api_flow_conformal_status():
    """Flow-based Conformal Prediction 状态 API"""
    return jsonify({
        'success': True,
        'data': {
            'model': 'Flow-based Conformal Prediction',
            'version': '1.0.0',
            'paper': 'arXiv:2502.05709',
            'type': 'Flow-based Multi-dimensional Conformal Prediction',
            'status': 'ready',
        },
        'timestamp': datetime.now().isoformat(),
    })
