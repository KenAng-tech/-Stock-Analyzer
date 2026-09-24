"""
Fusion Routes - /api/fusion/*

多模态融合预测 API 端点。
"""

import os
import threading

from flask import Blueprint, request, jsonify
from datetime import datetime

from modules.logger import logger

bp = Blueprint('fusion', __name__)

_eval_state = {'busy': False}    # 派发单飞标 (09-22 实修: 请求线程持锁长链/挂死 = 恒 429 形态)
_SCAN_POOL_DEFAULT = 'sz300620,sh688981,sz300502,sh688800,sh603929'


def _resolve_eval_seeds(fusion, fetcher, ledger, min_age_days: int = 3,
                        limit: int = 50) -> dict:
    """≥3 交易日老种子对账: 90d K 线真实收益 → record_evaluation → 落盘。

    consensus resolution 同形制 (窗口/|ret|<0.005 无方向只标记不写);
    单种 guard — 一种坏 ≠ 链死 (诚实记 errors 非静默)。
    """
    pend = ledger.list_pending_seeds(min_age_days=min_age_days, limit=limit)
    resolved, noise, errors = 0, 0, []
    for seed in pend:
        try:
            kl = fetcher.get_kline_data(seed['code'], 'daily', 90)
            if not kl or len(kl) < 10:
                errors.append({'code': seed['code'], 'error': 'K线<10 (对账跳过)'})
                continue
            ts = datetime.fromisoformat(seed['ts'])
            p0 = None
            for k in reversed(kl):  # 种子时刻前一交易日收盘 (同 replay 形)
                d = str(k.get('date', ''))[:10]
                if d and d <= ts.strftime('%Y-%m-%d'):
                    p0 = k.get('close')
                    break
            p1 = kl[-1].get('close')
            if not p0 or not p1 or p0 <= 0:
                errors.append({'code': seed['code'], 'error': '收盘缺失 (对账跳过)'})
                continue
            ret = (p1 - p0) / p0
            if abs(ret) < 0.005:
                ledger.mark_resolved(seed['id'])  # 无方向样本, 防重扫堆积
                noise += 1
                continue
            ev = fusion.record_evaluation(dict(seed), ret)
            ledger.append_eval(ev, seed['code'])
            ledger.mark_resolved(seed['id'])
            resolved += 1
        except Exception as e:
            errors.append({'code': seed.get('code'),
                           'error': f"{type(e).__name__}: {str(e)[:120]}"})
    return {'resolved': resolved, 'noise': noise, 'errors': errors}


def _spawn_eval_seeds(fusion, ledger, pool: str) -> dict:
    """监控池 predict 新种子 (23:10 链链尾, 串行, K线含缓存)."""
    spawned, errors = [], []
    for code in [c.strip() for c in pool.split(',') if c.strip()]:
        try:
            r = fusion.predict(code)
            if r.get('direction'):
                ledger.append_seed(r)
                spawned.append(code)
            else:
                errors.append({'code': code, 'error': 'predict 无 direction'})
        except Exception as e:
            errors.append({'code': code,
                           'error': f"{type(e).__name__}: {str(e)[:120]}"})
    return {'spawned': spawned, 'errors': errors}


@bp.route('/api/fusion/multi-modal')
def api_fusion_multi_modal():
    """多模态融合预测"""
    try:
        stock_code = request.args.get('stock_code', 'sz300620')
        regime = request.args.get('regime', 'sideways')
        use_cache = request.args.get('cache', 'true').lower() == 'true'

        from modules.multi_modal_fusion import get_multi_modal_fusion
        fusion = get_multi_modal_fusion()

        result = fusion.predict(stock_code, regime, use_cache)

        return jsonify({
            'success': True,
            **result,
        })
    except Exception as e:
        logger.error(f"[Fusion] 多模态融合预测错误: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/fusion/multi-modal/status')
def api_fusion_status():
    """融合器状态"""
    try:
        from modules.multi_modal_fusion import get_multi_modal_fusion
        fusion = get_multi_modal_fusion()

        return jsonify({
            'success': True,
            **fusion.get_status(),
        })
    except Exception as e:
        logger.error(f"[Fusion] 状态查询错误: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/fusion/compare')
def api_fusion_compare():
    """各模态独立信号对比"""
    try:
        stock_code = request.args.get('stock_code', 'sz300620')

        from modules.multi_modal_fusion import get_multi_modal_fusion
        fusion = get_multi_modal_fusion()

        result = fusion.compare(stock_code)

        return jsonify({
            'success': True,
            **result,
        })
    except Exception as e:
        logger.error(f"[Fusion] 信号对比错误: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/fusion/multi-modal/accuracy')
def api_fusion_accuracy():
    """多模态融合预测准确率 — 评估闭环"""
    try:
        import numpy as np
        from modules.multi_modal_fusion import get_multi_modal_fusion
        fusion = get_multi_modal_fusion()
        history = fusion._eval_history[-100:] if hasattr(fusion, '_eval_history') else []

        if not history:
            return jsonify({
                'success': True,
                'n_records': 0,
                'note': '尚无评估数据 (需通过 record_evaluation() 传入实际收益反馈)',
            })

        n = len(history)
        accuracy = sum(1 for e in history if e['correct']) / n
        avg_brier = float(np.mean([e['brier'] for e in history]))

        # 趋势
        trend = 'insufficient_data'
        drift_alert = False
        if n >= 20:
            recent = history[-20:]
            older = history[-40:-20] if n >= 40 else history[:len(history)//2]
            recent_acc = sum(1 for e in recent if e['correct']) / len(recent)
            older_acc = sum(1 for e in older if e['correct']) / len(older)
            trend = 'improving' if recent_acc > older_acc + 0.05 else \
                    ('declining' if recent_acc < older_acc - 0.05 else 'stable')
            # 漂移告警
            if n >= 50 and accuracy < 0.45:
                drift_alert = True

        return jsonify({
            'success': True,
            'n_records': n,
            'accuracy': round(accuracy, 4),
            'avg_brier': round(avg_brier, 4),
            'trend': trend,
            'recent_20_accuracy': round(
                sum(1 for e in history[-20:] if e['correct']) / 20, 4
            ) if n >= 20 else None,
            'drift_alert': drift_alert,
        })
    except Exception as e:
        logger.error(f"[FusionAccuracy] 错误: {e}")
        return jsonify({'success': False, 'error': str(e)})


@bp.route('/api/fusion/multi-modal/ic-tracker')
def api_fusion_ic_tracker():
    """IC 追踪器状态 — 查看各模态 IC 和自适应权重"""
    try:
        from modules.multi_modal_fusion import get_multi_modal_fusion
        fusion = get_multi_modal_fusion()
        ic_summary = fusion._ic_tracker.get_ic_summary() if hasattr(fusion, '_ic_tracker') else {}

        # 获取自适应权重
        adaptive_weights = None
        if hasattr(fusion, '_ic_tracker') and fusion._ic_tracker:
            adaptive_weights = fusion._ic_tracker.get_adaptive_weights()

        return jsonify({
            'success': True,
            'ic_tracker': {
                **ic_summary,
                'adaptive_weights': adaptive_weights,
            },
        })
    except Exception as e:
        logger.error(f"[FusionIC] 错误: {e}")
        return jsonify({'success': False, 'error': str(e)})


@bp.route('/api/fusion/evaluate', methods=['POST'])
def api_fusion_evaluate():
    """POST /api/fusion/evaluate — 23:10 replay 链尾: 派发评估链 (对账+产种子)。

    派发形 (09-22 实修): 原「请求线程同步跑完」形 — 重启后冷链 >240s +
    锁恒占实锤 (连续 429 0.01s) = 链坏形态。改 daemon 线程派发 (consensus
    queue/worker 同宗): 端点即回, busy 标防并发, 链尾结果看日志 [FusionRoutes]
    evaluate 链尾 行 + 次日 replay 链尾标。flag FUSION_EVAL=0 可关。
    """
    if os.environ.get('FUSION_EVAL', '1') == '0':
        return jsonify({'success': True, 'skipped': 'FUSION_EVAL=0'})
    if _eval_state['busy']:
        return jsonify({'success': True, 'skipped': 'evaluate 派发中 (单飞)'}), 429
    from modules.fusion_ledger import get_fusion_ledger
    if get_fusion_ledger() is None:
        return jsonify({'success': False, 'error': 'SQLite 台账不可用'}), 503
    _eval_state['busy'] = True

    def _bg():
        try:
            from modules.multi_modal_fusion import get_multi_modal_fusion
            from modules.data_fetcher import StockDataFetcher
            ledger = get_fusion_ledger()
            fusion = get_multi_modal_fusion()
            fetcher = StockDataFetcher()
            t0 = datetime.now()
            pool = os.environ.get('DECISION_SCAN_POOL', _SCAN_POOL_DEFAULT)
            r1 = _resolve_eval_seeds(fusion, fetcher, ledger)
            r2 = _spawn_eval_seeds(fusion, ledger, pool)
            logger.info(f"[FusionRoutes] evaluate 链尾: resolved={r1['resolved']} "
                        f"noise={r1['noise']} spawn={len(r2['spawned'])} "
                        f"errors={len(r1['errors']) + len(r2['errors'])} "
                        f"elapsed={(datetime.now() - t0).total_seconds():.1f}s")
        except Exception as e:
            logger.error(f"[FusionRoutes] evaluate 链坏: {e}")
        finally:
            _eval_state['busy'] = False

    threading.Thread(target=_bg, daemon=True, name='fusion-eval').start()
    return jsonify({'success': True,
                    'data': {'dispatched': True,
                             'note': '后台派发中, 链尾结果见 5002.log'}})
