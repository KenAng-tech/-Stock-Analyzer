#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
integration2026_routes.py — 2026 整合层统一 API (P0-P2, 2026-09-14)

把 qlib 研读 + 2026 调研落地的模块收敛为一个面板 API:
  /api/integration2026/overview        — 整合清单状态总览 (qlib 批 17 项 + PanWatch 监控 9 项, 2026-09-20)
  /api/integration2026/vol-control     — P2-11a 闭环波动率目标建议
  /api/integration2026/honest-eval     — P2-11b 五门诚实评估 (买持演示)
  /api/integration2026/search-ledger   — P0-3 搜索真实 trial 记账
  /api/integration2026/model-versions  — P1-6 模型版本 + online tag
  /api/integration2026/miner-memory    — P1-8 挖掘链三级反思记忆
  /api/integration2026/tradability     — P0-2 涨跌停/停牌可成交性
  /api/integration2026/routine         — P2-12c 夜间链编排 ledger
  /api/integration2026/consensus-ab    — P1-7 z-score 等权 A/B 观察日志

约定: 每端点 guarded import — 依赖模块未就位返回 status='pending'
(200, 非 500), 面板据此渲染"待接入"态; 绝不把异常抛给 Flask。
"""

import json
import logging
import os
from datetime import datetime

from flask import Blueprint, jsonify, request

bp = Blueprint('integration2026', __name__)

logger = logging.getLogger('stock_analyzer.integration2026')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROUTINE_LEDGER = os.path.join(PROJECT_ROOT, 'logs', 'routine_ledger.jsonl')
CONSENSUS_AB_DIR = os.path.join(PROJECT_ROOT, 'runs', 'consensus_ab')


def _load_klines(code: str, count: int = 250):
    """取日 K (List[Dict]); 失败抛异常由调用方 guarded 捕获"""
    from modules.data_fetcher import StockDataFetcher
    klines = StockDataFetcher().get_kline_data(code, count=count)
    if not klines:
        raise RuntimeError(f'{code} K 线为空')
    return klines


def _returns_series(code: str, count: int = 250):
    """日收益序列 (pd.Series, index=日期字符串), 供波动控制/诚实评估共用"""
    import pandas as pd
    klines = _load_klines(code, count)
    dates, closes = [], []
    for k in klines:
        c = k.get('close')
        d = k.get('date') or k.get('time') or str(k.get('datetime', ''))[:10]
        if c is None:
            continue
        try:
            closes.append(float(c))
            dates.append(str(d))
        except (TypeError, ValueError):
            continue
    if len(closes) < 30:
        raise RuntimeError(f'{code} 有效收盘不足 ({len(closes)})')
    s = pd.Series(closes, index=dates)
    return s.pct_change().dropna()


# ── 总览 ──────────────────────────────────────────────────────────────

@bp.route('/api/integration2026/overview', methods=['GET'])
def overview():
    """16 项整合清单状态: live(已接线)/pending(模块未就位) + 观察 flag"""
    items = []

    def _probe(key, name, prio, module, flag=None, note=''):
        try:
            __import__(module)
            status = 'live'
        except Exception:
            status = 'pending'
        entry = {'key': key, 'name': name, 'priority': prio,
                 'status': status, 'note': note}
        if flag:
            entry['flag'] = {'name': flag[0],
                             'value': os.environ.get(flag[0], flag[1]),
                             'default': flag[1]}
        items.append(entry)

    _probe('pit', 'PIT as-of 反前视数据层', 'P0', 'modules.pit_store',
           note='ann_date 可见性 + revision 链')
    _probe('tradability', '涨跌停/停牌拒单 + 三层成本', 'P0', 'modules.tradability',
           note='limit_threshold 板块自适应')
    _probe('search_ledger', '搜索全记账 (真实 trial N)', 'P0', 'modules.search_ledger',
           note='DSR 收缩用真实 N')
    _probe('run_artifact', 'runs/{ts}/ 工件目录', 'P0', 'modules.run_artifact',
           note='conf/metrics/model 落盘')
    _probe('startup_fixes', '启动链四小修', 'P0', 'boot.pid_manager',
           note='health 盲区/fail-closed host/PID 拒启/WatchPaths')
    _probe('factor_cache', '因子语义缓存 (表达式键)', 'P1', 'modules.factors.factor_cache',
           note='qlib 表达式串作键')
    _probe('rolling_retrain', '滚动重训 + online tag', 'P1', 'modules.model_registry',
           note='ModelVersionRegistry', flag=('ROLLING_MIN_INTERVAL_DAYS', '1'))
    _probe('consensus_ab', '共识 z-score 等权 A/B', 'P1', 'modules.multi_agent_consensus',
           flag=('CONSENSUS_AB_MODE', 'observe'))
    _probe('miner_reviewer', '挖掘对抗审查员 + 反思记忆', 'P1', 'modules.miner_memory',
           flag=('MINER_ADVERSARIAL_MODE', 'observe'))
    _probe('learn_infer_guard', 'learn/infer 处理器守卫', 'P1', 'modules.ml_processors',
           note='is_for_infer=False 进 infer 即 raise')
    _probe('dropout', 'TopkDropout 防抖', 'P1', 'modules.unified_decision_engine',
           flag=('DECISION_DROPOUT_MODE', 'observe'))
    _probe('vol_control', '闭环波动率目标控制', 'P2', 'modules.vol_control',
           note='arXiv 2603.01298 加性比例反馈')
    _probe('honest_eval', '五门诚实评估 (DSR/PBO/SPA/MTRL/REGIME)', 'P2', 'modules.honest_eval',
           note='审计层, 非盈利预测')
    _probe('pnl_corr', 'IC 相关 vs PnL 相关对比', 'P2', 'modules.pnl_corr',
           note='arXiv 2609.09588')
    _probe('regulatory_tags', '监管标签 (龙虎榜/解禁)', 'P2', 'modules.regulatory_tags',
           note='')
    _probe('turnover_scoring', '换手评分', 'P2', 'modules.turnover_scoring',
           note='')
    _probe('routine', '夜间链单入口编排', 'P2', 'scripts.routine',
           note='观察层, 不替换 launchd/crontab')

    # ── PanWatch 整合监控 (2026-09-20 第六批: 8 点闭环 + 上游 2 新提交) ──
    _probe('outbound_guard', 'PanWatch: 出站守门 (节流+负缓存+singleflight)', 'P0',
           'utils.outbound_guard', note='植入 5 条取数链, 降级非断链')
    _probe('push_ledger', 'PanWatch: 推送台账 (发送成功才标记+去重)', 'P0',
           'modules.push_ledger', note='daemon rm=送达闭环; 30min滞留=链断降级')
    _probe('llm_budget', 'PanWatch: token 台账 + 预算门', 'P2', 'modules.llm_budget',
           note='limit=0 纯观测', flag=('LLM_DAILY_TOKEN_LIMIT', '0'))
    _probe('trace_chain', 'PanWatch: trace_id 全链 (含跨线程 4 submit 段)', 'P2',
           'modules.log_context', note='X-Trace-Id 响应头; 一 grep 串全链')
    _probe('pw_batch_multisource', '上游候选: 跨市场混合 batch 取数 (5-worker)', 'P2',
           'modules.batch_source_check', note='1a41652 补位, 同 host 需守门器约束')
    _probe('pw_progress_recovery', '上游候选: 长任务进度恢复 (0.5.0)', 'P2',
           'modules.progress_recovery', note='晨报 233s 后台链 重启即丢 = 待评估')
    _probe('pw_cancel_stale', '上游候选: 失败任务残留取数取消', 'P2',
           'modules.cancel_stale', note='长跑链 cancel_futures 已有基础')
    _probe('patchtst_predict', 'PatchTST 预测链 (09-22 断链修复)', 'P0',
           'modules.models.patchtst_integrator',
           note='截面 (8,1) 恒抛→窗口 (60,12) 形制; 链尾实锤 up/0.537 + 集成 17 票')
    _probe('llm_budget_health', 'LLM 预算观测 (09-22 探针校准)', 'P2',
           'modules.llm_budget', note='stats() 三键在 health llm.budget 实形; 23:10 replay n_hist 直通')
    # 2026-09-21 深研 dsa (ZhuLinsen 65k★) 对照: opinion-outcome 闭环 5002 已有同构
    _probe('opinion_loop', '决策意见闭环 (SQLite 台账+23:10 回放+角色门控)', 'P1',
           'modules.decision_history_store',
           note='Brier/ECE 观察+权重重映射; 链尾日志 n_hist 可辨空转/健康')

    return jsonify({'status': 'ok', 'items': items,
                    'generated_at': datetime.now().isoformat(timespec='seconds')})


# ── P2-11a 闭环波动率 ─────────────────────────────────────────────────

@bp.route('/api/integration2026/vol-control', methods=['GET'])
def vol_control():
    """闭环波动率目标建议 (建议层, 不自动执行)"""
    code = request.args.get('code', 'sz300620')
    try:
        target_vol = float(request.args.get('target_vol', 0.15))
        kp = float(request.args.get('kp', 0.10))
    except ValueError:
        return jsonify({'status': 'error', 'error': 'target_vol/kp 须为数字'}), 400
    try:
        from modules.vol_control import closed_loop_vol_target
        r = _returns_series(code)
        res = closed_loop_vol_target(r, target_vol=target_vol, kp=kp)
        if 'error' in res:
            return jsonify({'status': 'error', 'error': res['error']}), 422
        exp = res['exposure']
        step = max(1, len(exp) // 120)  # 降采样绘图上限 ~120 点
        return jsonify({
            'status': 'ok', 'code': code,
            'suggested_exposure': round(float(exp.iloc[-1]), 4),
            'realized_vol': res['realized_vol'],
            'tracking_mae': res['tracking_mae'],
            'open_loop': res['open_loop'],
            'turnover_mean': res['turnover_mean'],
            'params': res['params'],
            'note': '闭环反馈控制 (arXiv 2603.01298); 建议层, 不自动执行',
            'series': {'dates': list(exp.index[::step]),
                       'exposure': [round(float(v), 4) for v in exp.iloc[::step]]},
        })
    except Exception as e:
        logger.error(f"[integration2026] vol-control 失败: {e}", exc_info=True)
        return jsonify({'status': 'error', 'error': str(e)}), 500


# ── P2-11b 五门诚实评估 ───────────────────────────────────────────────

@bp.route('/api/integration2026/honest-eval', methods=['GET'])
def honest_eval():
    """五门审计 (演示口径: 标的买入持有日收益; trial 数取 search_ledger 真实值)"""
    code = request.args.get('code', 'sz300620')
    try:
        from modules.honest_eval import five_gate_audit
        r = _returns_series(code)
        out = five_gate_audit(r, source='backtest', n_trials_fallback=None)
        if 'error' in out:
            return jsonify({'status': 'error', 'error': out['error']}), 422
        return jsonify({'status': 'ok', 'code': code, 'sample_days': int(len(r)),
                        'demo_note': '演示口径=买入持有收益, 非策略收益; 结论不代表策略稳健性',
                        'audit': out})
    except Exception as e:
        logger.error(f"[integration2026] honest-eval 失败: {e}", exc_info=True)
        return jsonify({'status': 'error', 'error': str(e)}), 500


# ── P0-3 搜索记账 ─────────────────────────────────────────────────────

@bp.route('/api/integration2026/search-ledger', methods=['GET'])
def search_ledger():
    """真实 trial 数 + 来源汇总 + 最近记录"""
    try:
        from modules.search_ledger import trial_count, recent_trials, source_summary
        return jsonify({
            'status': 'ok',
            'total_trials_90d': trial_count(since_days=90),
            'sources': source_summary(),
            'recent': recent_trials(limit=20),
        })
    except ImportError:
        return jsonify({'status': 'pending', 'note': 'search_ledger 未就位'})
    except Exception as e:
        logger.error(f"[integration2026] search-ledger 失败: {e}", exc_info=True)
        return jsonify({'status': 'error', 'error': str(e)}), 500


# ── P1-6 模型版本 ─────────────────────────────────────────────────────

@bp.route('/api/integration2026/model-versions', methods=['GET'])
def model_versions():
    """各模型 online 版本 + 距上次训练天数 (今晚 23:00 链首次注册)"""
    try:
        from modules.model_registry import get_version_registry
        reg = get_version_registry()
        models = {}
        for name in ('patchtst', 'mamba', 'diffusion', 'self_supervised', 'drl'):
            online = reg.get_online(name)
            vers = reg.list_versions(name)
            models[name] = {
                'online': online,
                'n_versions': len(vers),
                'days_since_last_train': reg.days_since_last_train(name),
            }
        return jsonify({'status': 'ok', 'models': models})
    except Exception as e:
        logger.error(f"[integration2026] model-versions 失败: {e}", exc_info=True)
        return jsonify({'status': 'error', 'error': str(e)}), 500


# ── P1-8 挖掘记忆 ─────────────────────────────────────────────────────

@bp.route('/api/integration2026/miner-memory', methods=['GET'])
def miner_memory():
    """三级反思记忆快照 (generation/cycle/archetype)"""
    try:
        from modules.miner_memory import get_miner_memory
        return jsonify({'status': 'ok', 'snapshot': get_miner_memory().snapshot()})
    except ImportError:
        return jsonify({'status': 'pending', 'note': 'miner_memory 未就位'})
    except Exception as e:
        logger.error(f"[integration2026] miner-memory 失败: {e}", exc_info=True)
        return jsonify({'status': 'error', 'error': str(e)}), 500


# ── P0-2 可成交性 ─────────────────────────────────────────────────────

@bp.route('/api/integration2026/tradability', methods=['GET'])
def tradability():
    """用最新 K 线判当前可成交性 (涨跌停幅度板块自适应 + 停牌无量)"""
    code = request.args.get('code', 'sz300620')
    try:
        from modules.tradability import limit_threshold_for, is_tradable
        klines = _load_klines(code, count=5)
        last, prev = klines[-1], klines[-2]
        c_now, c_prev = float(last['close']), float(prev['close'])
        pct = (c_now - c_prev) / c_prev if c_prev else 0.0
        vol = last.get('volume')
        threshold = limit_threshold_for(code)
        buy_ok, buy_reason = is_tradable(code, pct, 'buy', volume=vol)
        sell_ok, sell_reason = is_tradable(code, pct, 'sell', volume=vol)
        return jsonify({
            'status': 'ok', 'code': code,
            'date': str(last.get('date', '')),
            'close': c_now, 'prev_close': c_prev,
            'pct_change': round(pct, 4),
            'limit_threshold': threshold,
            'buy': {'tradable': buy_ok, 'reason': buy_reason},
            'sell': {'tradable': sell_ok, 'reason': sell_reason},
        })
    except Exception as e:
        logger.error(f"[integration2026] tradability 失败: {e}", exc_info=True)
        return jsonify({'status': 'error', 'error': str(e)}), 500


# ── P2-12c 编排 ledger ────────────────────────────────────────────────

@bp.route('/api/integration2026/routine', methods=['GET'])
def routine_ledger():
    """夜间链编排 OK/FAIL 台账 (最近 N 条, 只读)"""
    try:
        limit = min(int(request.args.get('limit', 7)), 50)
    except ValueError:
        limit = 7
    try:
        if not os.path.exists(ROUTINE_LEDGER):
            return jsonify({'status': 'ok', 'entries': [],
                            'note': 'ledger 尚未生成 (python scripts/routine.py --date … 干跑)'})
        with open(ROUTINE_LEDGER, 'r', encoding='utf-8', errors='replace') as f:
            lines = [l for l in f.read().splitlines() if l.strip()]
        entries = []
        for l in lines[-limit:]:
            try:
                entries.append(json.loads(l))
            except json.JSONDecodeError:
                continue
        return jsonify({'status': 'ok', 'entries': list(reversed(entries))})
    except Exception as e:
        logger.error(f"[integration2026] routine ledger 失败: {e}", exc_info=True)
        return jsonify({'status': 'error', 'error': str(e)}), 500


# ── P1-7 共识 A/B 观察日志 ────────────────────────────────────────────

@bp.route('/api/integration2026/consensus-ab', methods=['GET'])
def consensus_ab():
    """z-score 等权旁路观察日志 runs/consensus_ab/{date}.jsonl (只读)"""
    date = request.args.get('date', datetime.now().strftime('%Y-%m-%d'))
    try:
        path = os.path.join(CONSENSUS_AB_DIR, f'{date}.jsonl')
        if not os.path.exists(path):
            if not os.path.isdir(CONSENSUS_AB_DIR):
                return jsonify({'status': 'pending',
                                'note': 'A/B 日志目录未生成 (共识链跑过且 mode=observe 后出现)'})
            files = sorted(os.listdir(CONSENSUS_AB_DIR), reverse=True)
            if not files:
                return jsonify({'status': 'ok', 'date': date, 'entries': [],
                                'note': '当日无 A/B 记录'})
            path = os.path.join(CONSENSUS_AB_DIR, files[0])
            date = files[0].replace('.jsonl', '')
        entries = []
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            for l in f.read().splitlines()[-100:]:
                try:
                    entries.append(json.loads(l))
                except json.JSONDecodeError:
                    continue
        return jsonify({'status': 'ok', 'date': date, 'entries': list(reversed(entries))})
    except Exception as e:
        logger.error(f"[integration2026] consensus-ab 失败: {e}", exc_info=True)
        return jsonify({'status': 'error', 'error': str(e)}), 500
