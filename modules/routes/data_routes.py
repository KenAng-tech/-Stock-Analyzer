"""
Data Routes - Stock data, analyze, ATR, heatmap, alerts

Extracted from app.py (lines 657-2032).
Dependencies imported from modules.* at file level.
"""

from flask import Blueprint, request, jsonify, send_file
from datetime import datetime
import os
import re
import time
import numpy as np

from modules.data_fetcher import StockDataFetcher
from modules.analysis_engine import AnalysisEngine
from modules.report_generator import ReportGenerator, get_report_generator
from modules.markdown_report_generator import MarkdownReportGenerator
from modules.strategy_engine import StrategyEngine
from modules.kline_signal_analyzer import KlineSignalAnalyzer
from modules.atr_calculator import ATRCalculator, ADXCalculator
from modules.heatmap_generator import HeatmapGenerator
from modules.hmm_market_detector import MarketRegimeDetector as HMMRegimeDetector
from modules.alert_engine import AlertEngine
from modules.logger import logger
from modules.input_validator import InputValidator

bp = Blueprint('data', __name__)

# Module-level instances (mirroring app.py)
data_fetcher = StockDataFetcher()
analysis_engine = AnalysisEngine()
report_generator = ReportGenerator()
_md_report_gen = MarkdownReportGenerator()  # md 报告链 (2026-09-19 复活, 见 api_analyze_stock)
strategy_engine = StrategyEngine()
kline_analyzer = KlineSignalAnalyzer()
atr_calculator = ATRCalculator()
adx_calculator = ADXCalculator()
hmm_detector = HMMRegimeDetector()
heatmap_generator = HeatmapGenerator()
alert_engine = AlertEngine()

DEFAULT_STOCK = {
    'code': 'sz300620',
    'name': '光库科技',
    'industry': '光通信',
    'cost_basis': 120.0,  # 与 app.py / config.json 保持一致
}


# ============================================================================
# Stock Data API
# ============================================================================

@bp.route('/api/stock/<stock_code>')
def api_get_stock(stock_code):
    """Get stock data API"""
    if not InputValidator.validate_stock_code(stock_code):
        return jsonify({'success': False, 'error': 'Invalid stock code format (e.g. sz300620)'}), 400
    stock_data = data_fetcher.get_stock_info(stock_code)
    if stock_data:
        return jsonify({
            'success': True,
            'data': stock_data,
            'timestamp': datetime.now().isoformat()
        })
    return jsonify({'success': False, 'error': 'Failed to fetch stock data'}), 500


@bp.route('/api/stock/enhanced/<stock_code>')
def api_get_enhanced_stock(stock_code):
    """Get enhanced stock data with K-line stats"""
    if not InputValidator.validate_stock_code(stock_code):
        return jsonify({'success': False, 'error': 'Invalid stock code format (e.g. sz300620)'}), 400
    stock_data = data_fetcher.get_enhanced_stock_info(stock_code)
    if stock_data:
        return jsonify({
            'success': True,
            'data': stock_data,
            'timestamp': datetime.now().isoformat()
        })
    return jsonify({'success': False, 'error': 'Failed to fetch stock data'}), 500


@bp.route('/api/analyze/<stock_code>', methods=['GET', 'POST'])
def api_analyze_stock(stock_code):
    """Get comprehensive analysis"""
    if not InputValidator.validate_stock_code(stock_code):
        return jsonify({'success': False, 'error': 'Invalid stock code format (e.g. sz300620)'}), 400
    stock_data = data_fetcher.get_stock_info(stock_code)
    if not stock_data:
        return jsonify({'success': False, 'error': 'Failed to fetch stock data'}), 500

    industry = DEFAULT_STOCK.get('industry', '光通信')
    # Support both query param (GET) and JSON body (POST)
    cost_basis = request.args.get('cost_basis', DEFAULT_STOCK['cost_basis'], type=float)
    if request.is_json:
        body = request.get_json(silent=True) or {}
        cost_basis = body.get('cost_basis', cost_basis)

    # ── 一次性预取 K 线数据，注入 stock_data 供后续分析复用（避免重复网络调用）─
    kline_data_dict = {}
    try:
        kline_data_dict['daily'] = data_fetcher.get_kline_data(stock_code, 'daily', 100)
        kline_data_dict['weekly'] = data_fetcher.get_kline_data(stock_code, 'weekly', 50)
        kline_data_dict['monthly'] = data_fetcher.get_kline_data(stock_code, 'monthly', 20)
    except Exception as e:
        logger.warning(f"K 线数据预取失败 (stock={stock_code}): {e}")
    # 注入日 K 线到 stock_data（供 quantitative_prediction 复用）
    stock_data['_klines_daily'] = kline_data_dict.get('daily', [])

    # Perform analysis (with caching)
    analysis = analysis_engine.comprehensive_analysis_cached(
        stock_code, stock_data, industry, cost_basis
    )

    # Generate strategies
    strategies = strategy_engine.generate_strategy_recommendation(
        stock_data, analysis, cost_basis
    )

    # K-line signal analysis
    kline_signals = kline_analyzer.generate_kline_signals(stock_data, kline_data_dict)

    # HMM regime prediction — 复用预取的日 K 线数据
    hmm_regime = 'sideways'
    hmm_probabilities = {}
    hmm_adjustment = {}
    daily_klines = kline_data_dict.get('daily', [])
    if daily_klines:
        try:
            hmm_regime = hmm_detector.predict_regime(stock_data, daily_klines)
            hmm_probabilities = hmm_detector.get_regime_probability(stock_data, daily_klines)
            hmm_adjustment = hmm_detector.get_regime_adjustment(stock_data, daily_klines)
        except Exception as e:
            logger.warning(f"HMM regime 预测失败 (stock={stock_code}): {e}")

    # ADX Indicator
    adx_data = {}
    try:
        adx_data = adx_calculator.calculate_adx_from_data(stock_data)
    except Exception as e:
        logger.warning(f"ADX 计算失败 (stock={stock_code}): {e}")

    # Kelly Position Sizing
    kelly_info = analysis_engine._calculate_dynamic_kelly(stock_data)

    # CVaR Risk
    cvar_risk = strategy_engine.calculate_cvar_risk([-0.05, -0.03, -0.02, 0.01, 0.02, 0.03, 0.04, 0.05])

    # ATR Dynamic Stop Loss / Take Profit
    try:
        atr_stop = atr_calculator.calculate_atr_stop_loss(stock_data)
        atr_profit = atr_calculator.calculate_atr_stop_gain(stock_data)
        atr_sr = atr_calculator.calculate_dynamic_support_resistance(stock_data)
    except Exception as e:
        logger.error(f"ATR calculation error: {e}")
        atr_stop = atr_profit = atr_sr = {}

    # ── 市场情报 + 执行审计 (2026-09-13 #17 链 B): 情报雷达/AI 洞察/审计 ──
    # 与晨报链 A 同形 (09-10 并行先例, 两链并行非嵌套; 09-02 非 self-HTTP)。
    # 情报链 12s 硬预算 + 审计 5s join 超时; 缺输入/超时 = 诚实跳过,
    # 绝不让报告链拖到前端 45s (09-09 教训: 无上下文比坏上下文诚实)。
    try:
        from concurrent.futures import ThreadPoolExecutor
        from modules.intel_engine import intel_summary
        from modules.insight_engine import get_insight_engine
        from modules.exec_audit import audit_for_stock

        def _intel_job():
            intel = intel_summary(stock_code) or {}
            feed = (intel.get('_insight_feed') or {}).get('news') or []
            dt = (intel.get('dragon_tiger') or {}).get('items') or []
            # 洞察 fail-closed (同晨报 A): 无料 abstain 不空转 8080
            insight = get_insight_engine().get_insight(
                stock_code, news=feed, dragon_tiger=dt)
            return intel, insight

        def _audit_job():
            return audit_for_stock(stock_code, 'neutral', timeout=5.0)

        from modules.log_context import with_trace
        with ThreadPoolExecutor(max_workers=3,
                                thread_name_prefix='intel') as _ex:
            f_intel = _ex.submit(with_trace(_intel_job))
            f_audit = _ex.submit(with_trace(_audit_job))
            try:
                intel_pack, insight_pack = f_intel.result(timeout=12.0)
                audit_pack = f_audit.result(timeout=12.0)
                analysis['intel'] = intel_pack
                if insight_pack.get('valid'):
                    analysis['insight'] = insight_pack
                if audit_pack and not audit_pack.get('skipped'):
                    analysis['exec_audit'] = audit_pack
            except Exception as e:
                logger.warning(f"市场情报链降级 (报告走精简版, 非断链): "
                               f"{stock_code}: {type(e).__name__}: {str(e)[:80]}")
    except Exception as e:
        logger.error(f"市场情报链启动失败: {stock_code}: {e}")

    # ── 将 HMM/ADX/CVaR/ATR/K线信号 注入 analysis 对象（供报告生成器使用）─
    analysis['hmm'] = {
        'regime': hmm_regime,
        'probabilities': hmm_probabilities,
        'adjustment': hmm_adjustment
    }
    analysis['adx'] = adx_data
    analysis['atr'] = {
        'stop_loss': atr_stop,
        'take_profit': atr_profit,
        'support_resistance': atr_sr
    }
    analysis['cvar'] = cvar_risk
    analysis['kline_signals'] = kline_signals

    # ── Markdown 深度报告 (08-13 md 链复活 2026-09-19) ──
    # 09-17 Archify 整合把 ReportGenerator 重写为 HTML 生成器, 但本调用点未跟进 →
    # 每请求 AttributeError (被 errors.py handler 二次放大为 500); 旧 md 类已从 backups 恢复为
    # MarkdownReportGenerator, 前端 app.js/webgui 消费 report + report_filename 契约恢复
    try:
        report = _md_report_gen.generate_report(analysis)
        # Save report to file (安全: 只允许字母、数字、下划线，防止路径穿越)
        safe_code = re.sub(r'[^a-zA-Z0-9_]', '', stock_code)
        report_filename = f"report_{safe_code}_{int(time.time())}.md"
        report_path = os.path.join('data', report_filename)
        os.makedirs('data', exist_ok=True)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
    except Exception as e:
        logger.warning(f"[DataRoutes] Markdown 报告生成失败 (非断链): {stock_code}: {e}")
        report = ""
        report_filename = ""

    # ── 可选: 生成 HTML 报告 (Archify 确定性交付模式) ──
    html_report = None
    try:
        if request.args.get('report') == '1':
            receipt = get_report_generator().generate(analysis)
            html_report = receipt.to_dict()
            logger.info(
                f"[DataRoutes] HTML 报告: {receipt.output_path} "
                f"({receipt.artifact_bytes}B, SHA={receipt.artifact_sha256[:8]})"
            )
    except Exception as e:
        logger.warning(f"[DataRoutes] HTML 报告生成失败: {e}")

    return jsonify({
        'success': True,
        'analysis': analysis,
        'strategies': strategies,
        'kline_signals': kline_signals,
        'hmm': {
            'regime': hmm_regime,
            'probabilities': hmm_probabilities,
            'adjustment': hmm_adjustment
        },
        'adx': adx_data,
        'kelly': kelly_info,
        'cvar': cvar_risk,
        'atr': {
            'stop_loss': atr_stop,
            'take_profit': atr_profit,
            'support_resistance': atr_sr
        },
        'report': report,
        'report_filename': report_filename,
        'html_report': html_report,
        'timestamp': datetime.now().isoformat()
    })


# ============================================================================
# ATR API
# ============================================================================

@bp.route('/api/atr/<stock_code>')
def api_get_atr(stock_code):
    """Get ATR analysis for a stock"""
    try:
        stock_data = data_fetcher.get_stock_info(stock_code)
        if not stock_data:
            return jsonify({'success': False, 'error': 'Failed to fetch stock data'}), 404

        atr_data = atr_calculator.calculate_atr_from_data(stock_data)
        return jsonify({
            'success': True,
            'data': atr_data,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"ATR calculation error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# Heatmap API
# ============================================================================

@bp.route('/api/heatmap')
def api_get_heatmap():
    """Get sector heatmap data"""
    try:
        heatmap = heatmap_generator.generate_heatmap()
        return jsonify({
            'success': True,
            'data': heatmap,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Heatmap generation error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# Alerts API
# ============================================================================

@bp.route('/api/alerts/<stock_code>')
def api_get_alerts(stock_code):
    """Get alerts for a stock"""
    try:
        alerts = alert_engine.get_alerts(stock_code)
        return jsonify({
            'success': True,
            'data': alerts,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Get alerts error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/alerts/config', methods=['GET'])
def api_get_alert_config():
    """Get alert configuration"""
    try:
        config = alert_engine.get_config()
        return jsonify({
            'success': True,
            'data': config,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Get alert config error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/alerts/config', methods=['POST'])
def api_update_alert_config():
    """Update alert configuration"""
    try:
        body = request.json if request.is_json else {}
        alert_engine.update_config(body)
        return jsonify({
            'success': True,
            'message': 'Alert configuration updated',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Update alert config error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
