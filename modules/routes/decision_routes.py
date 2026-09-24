#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""决策链观察端点 (2026-09-12 整合 P0-a/b/c + P1)

把四块新引擎接到 /api/* (前端面板消费, 均 GET 除 recalibrate):
  /api/decision/calibration   校准观察 (Brier/ECE + 角色门控状态, P0-a)
  /api/decision/recalibrate   强制重算 (23:00 decision_replay 链触发)
  /api/decision/exec_audit    可成交性审计 (触板/流动性/成本, P0-c)
  /api/intel/<code>           情报雷达 (龙虎榜+新闻情感+解禁, P1)
  /api/insight                AI 投资洞察 (新闻+龙虎榜→LLM, P0-b)

纪律: 端点只读/只触发, 不改投票; 链坏一律 success:false + 诚实错误,
不伪造 200 (09-11 决策链 B 契约教训)。
"""

import os
import os
import threading
import time

from flask import Blueprint, request, jsonify

from modules.logger import logger

bp = Blueprint('decision', __name__)

_recalc_lock = threading.Lock()      # 单飞 (23:00 链 + 手动触发防并发)
_feed_lock = threading.Lock()        # P1-a accuracy_feed 单飞 (同 seed 集防并发双喂)
_bps_lock = threading.Lock()         # P2-A bps_calibrate 单飞 (23:10 段7 + 手动防并发)


@bp.route('/api/decision/calibration', methods=['GET'])
def api_calibration():
    """校准观察面板数据: Brier/ECE (consensus 级) + 角色 Brier 门控状态"""
    try:
        from modules.multi_agent_consensus import get_consensus_engine
        eng = get_consensus_engine()
        cal = dict(eng._last_calibration or {})
        return jsonify({
            'success': True,
            'data': {
                'consensus_calibration': cal,           # brier/ece/role_gate
                'current_weights': eng.weights,         # 门控/IC 闭环后的实时权重
                'total_decisions': len(eng._history),
                'recalibrate_every': 50,                # 每 50 决策自动重算
                'ts': time.strftime('%Y-%m-%dT%H:%M:%S'),
            },
        })
    except Exception as e:
        logger.error(f"[DecisionRoutes] 校准查询失败: {e}")
        return jsonify({'success': False, 'error': str(e)[:200]}), 500


@bp.route('/api/decision/recalibrate', methods=['POST'])
def api_recalibrate():
    """强制重算 (23:00 决策复盘链 + 手动): 不等每 50 次决策触发"""
    if not _recalc_lock.acquire(blocking=False):
        return jsonify({'success': False,
                        'error': 'recalibrate 已在运行 (单飞防并发)'}), 429
    try:
        from modules.multi_agent_consensus import get_consensus_engine
        eng = get_consensus_engine()
        t0 = time.time()
        eng._recalibrate_weights()
        cal = dict(eng._last_calibration or {})
        # 2026-09-21 观测面: 样本数直通响应 (skipped≠健康, 日志可区分空转/健康 —
        # 教训: 自检恒假=失明; brier=- 单看无法区分未跑够样本 vs 闭环正常)
        cal['n_history'] = len(eng._history)
        return jsonify({
            'success': True,
            'data': {'duration': round(time.time() - t0, 2),
                     'calibration': cal,
                     'current_weights': eng.weights},
        })
    except Exception as e:
        logger.error(f"[DecisionRoutes] 强制重算失败: {e}")
        return jsonify({'success': False, 'error': str(e)[:200]}), 500
    finally:
        _recalc_lock.release()


# ---- 2026-09-22 决策闭环提效 (用户拍板): 监控池自动决策产生链 ----
# 闭环实锤空转 9 天: _history 15/20 样本不足 → recalibrate 恒 skipped
# (09-12 上线以来无自动 decide 产生源, 15 条全手动/analyze 偶发触发)。
# 本链 = 23:10 replay 链尾把监控池逐票跑 consensus.decide → 台账累积
# (dsa agent_trajectory 同构产生器, 它重启丢进度而本链 SQLite 持久)。

_scan_lock = threading.Lock()   # 单飞独立于 recalibrate (产决策 vs 算权重, 语义分开)
_SCAN_POOL_DEFAULT = 'sz300620,sh688981,sz300502,sh688800,sh603929'


def _run_decision_scan(pool: str) -> dict:
    """串行跑监控池 consensus.decide (23:10 链形, 无 Flask 上下文依赖)。

    单票 guard: 一票异常/队列满/MPS 训练中 → 跳过记错, 不拖其余票 (链坏≠链死)。
    """
    from modules.multi_agent_consensus import get_consensus_engine
    eng = get_consensus_engine()
    codes = [c.strip() for c in pool.split(',') if c.strip()]
    t0 = time.time()
    done, errors = [], []
    for code in codes:
        try:
            r = eng.decide(code)
            done.append(f"{code}:{getattr(r, 'consensus', None) or 'abstain'}")
        except Exception as e:
            errors.append({'code': code,
                           'error': f"{type(e).__name__}: {str(e)[:120]}"})
    return {'success': not errors, 'scanned': len(done), 'errors': errors,
            'n_history': len(eng._history),
            'elapsed': round(time.time() - t0, 1)}


@bp.route('/api/decision/scan', methods=['POST'])
def api_decision_scan():
    """监控池自动决策 (23:10 replay 链尾 + 手动): flag 总开关 DECISION_SCAN=0 可关"""
    if os.environ.get('DECISION_SCAN', '1') == '0':
        return jsonify({'success': True, 'skipped': 'DECISION_SCAN=0'})
    if not _scan_lock.acquire(blocking=False):
        return jsonify({'success': False,
                        'error': 'scan 已在运行 (单飞防并发)'}), 429
    try:
        pool = os.environ.get('DECISION_SCAN_POOL', _SCAN_POOL_DEFAULT)
        out = _run_decision_scan(pool)
        logger.info(f"[DecisionRoutes] scan 链尾: scanned={out['scanned']} "
                    f"errors={len(out['errors'])} n_hist={out.get('n_history')} "
                    f"elapsed={out['elapsed']}s")
        return jsonify(out)
    except Exception as e:
        logger.error(f"[DecisionRoutes] scan 链坏: {e}")
        return jsonify({'success': False, 'error': str(e)[:200]}), 500
    finally:
        _scan_lock.release()


@bp.route('/api/decision/exec_audit', methods=['GET'])
def api_exec_audit():
    """可成交性审计 (观察层): ?stock_code=sz300620&decision=buy"""
    try:
        code = request.args.get('stock_code', 'sz300620')
        decision = request.args.get('decision', 'neutral')
        from modules.exec_audit import audit_for_stock
        audit = audit_for_stock(code, decision, timeout=8.0)
        return jsonify({'success': not audit.get('skipped'),
                        'data': audit})
    except Exception as e:
        logger.error(f"[DecisionRoutes] 执行审计失败: {e}")
        return jsonify({'success': False, 'error': str(e)[:200]}), 500


@bp.route('/api/intel/<stock_code>', methods=['GET'])
def api_intel(stock_code: str):
    """情报雷达: 龙虎榜 (近5次上榜) + 新闻/情感 + 解禁日历 (单票过滤)"""
    try:
        from modules.intel_engine import intel_summary
        return jsonify({'success': True, 'data': intel_summary(stock_code)})
    except Exception as e:
        logger.error(f"[DecisionRoutes] 情报聚合失败: {e}")
        return jsonify({'success': False, 'error': str(e)[:200]}), 500


@bp.route('/api/insight', methods=['GET'])
def api_insight():
    """AI 投资洞察 (P0-b 链): ?stock_code=sz300620

    链形态: 无新闻且无龙虎榜 → abstain 不空转 LLM; LLM 降级/解析坏 →
    valid:false 诚实空态 — 面板永远分得清「真洞察」与「无洞察」。
    """
    try:
        code = request.args.get('stock_code', 'sz300620')
        from modules.insight_engine import get_insight_engine
        from modules.data_fetcher import StockDataFetcher
        from modules.dragon_tiger_fetcher import fetch_dragon_tiger_history

        eng = get_insight_engine()
        fetcher = StockDataFetcher()
        news = fetcher.get_stock_news(code) or []
        posts = fetcher.get_stock_posts(code) or []
        dt = fetch_dragon_tiger_history(code, limit=3) or []
        snap = fetcher.get_stock_info(code) or {}
        price_note = ''
        try:
            if snap:
                price_note = (f"现价 {snap.get('price')} (日涨跌 "
                              f"{snap.get('change_pct')}%), 52周 "
                              f"{snap.get('year_low')}~{snap.get('year_high')}")
        except Exception:
            price_note = ''
        out = eng.get_insight(code, news=news, posts=posts,
                              dragon_tiger=dt, price_note=price_note)
        return jsonify({'success': True, 'data': out})
    except Exception as e:
        logger.error(f"[DecisionRoutes] 洞察链失败: {e}")
        return jsonify({'success': False, 'error': str(e)[:200]}), 500


@bp.route('/api/decision/accuracy', methods=['POST'])
def api_accuracy_feed():
    """P1-a performative 反馈链 (23:10 replay 段 7 + 手动): 成熟 seed →
    K线真收益 → UDE.update_accuracy 喂入 (死入口复活, 单飞锁同 09-12 端点族)"""
    if not _feed_lock.acquire(blocking=False):
        return jsonify({'success': False,
                        'error': 'accuracy_feed 已在运行 (单飞防并发)'}), 429
    try:
        from modules.unified_decision_engine import get_decision_engine
        r = get_decision_engine().accuracy_feed()
        return jsonify({'success': True, 'data': r})
    except Exception as e:
        logger.error(f"[DecisionRoutes] accuracy 喂链失败: {e}")
        return jsonify({'success': False, 'error': str(e)[:200]}), 500
    finally:
        _feed_lock.release()


@bp.route('/api/decision/bps', methods=['POST'])
def api_bps_synthesis():
    """P2-A BPS 合成层 (23:10 replay 段 7, arXiv 2510.07180): 成熟 seed →
    η EWMA (式3 log q) → softmax(η) 权重 → bps_weights.jsonl 观察档。
    UDE_BPS_WEIGHT 三档 (2026-09-23 S5): '1'=每次即喂 / '2'=auto 门控 (观察档
    ≥3 记录后 auto feed, 证据先行门控后动) / 默认 '0'=纯观察不触活链;
    consumed_bps 独立第二锁 (不与 accuracy_feed 双锁共享, 同 seed 双观察)"""
    if not _bps_lock.acquire(blocking=False):
        return jsonify({'success': False,
                        'error': 'bps_calibrate 已在运行 (单飞防并发)'}), 429
    try:
        from modules.unified_decision_engine import get_decision_engine
        from modules.bps_synthesis import bps_calibrate
        _flag = os.environ.get('UDE_BPS_WEIGHT', '0')
        feed = True if _flag == '1' else ('auto' if _flag == '2' else False)
        r = bps_calibrate(ude=get_decision_engine(), feed_weights=feed)
        return jsonify({'success': True, 'data': r})
    except Exception as e:
        logger.error(f"[DecisionRoutes] bps 合成链失败: {e}")
        return jsonify({'success': False, 'error': str(e)[:200]}), 500
    finally:
        _bps_lock.release()
