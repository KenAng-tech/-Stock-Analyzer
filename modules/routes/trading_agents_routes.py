#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
TradingAgents 多智能体交易 API 路由 (2026-08-13 新增)

参考: TradingAgents (Tauric Research, 2024-12, 122 upvotes)
        "A Multi-Agent LLM Framework for Stock Trading"

架构 (5 Agent + 1 Coordinator):
    AnalystTeam (技术/基本面/情绪/政策) → ResearchTeam (研究员辩论)
        → TraderAgent (交易员) → RiskManager (风控) → PortfolioManager (组合)

端点:
    GET  /api/trading-agents/status        - 协调器状态
    POST /api/trading-agents/decide        - 触发决策
    GET  /api/trading-agents/agents        - 所有 Agent 状态
    GET  /api/trading-agents/history       - 决策历史
"""

import time
import json
import os
import sqlite3
import threading
from typing import Dict
from flask import Blueprint, jsonify, request
from modules.logger import logger

bp = Blueprint('trading_agents_routes', __name__)

# ── 决策历史存储 ────────────────────────────────────────────────
_HISTORY_DB = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'data', 'trading_agents_history.db'
)


def _init_history_db():
    """初始化历史数据库"""
    os.makedirs(os.path.dirname(_HISTORY_DB), exist_ok=True)
    with sqlite3.connect(_HISTORY_DB) as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stock_code TEXT,
                action TEXT,
                confidence REAL,
                analyst_consensus TEXT,
                research_recommendation TEXT,
                risk_level TEXT,
                overall_score REAL,
                elapsed_ms REAL,
                created_at TEXT,
                raw_result TEXT
            )
        ''')
        conn.commit()


def _save_decision(stock_code: str, result: Dict, elapsed_ms: float):
    """保存决策到历史"""
    try:
        with sqlite3.connect(_HISTORY_DB) as conn:
            conn.execute('''
                INSERT INTO decisions
                (stock_code, action, confidence, analyst_consensus,
                 research_recommendation, risk_level, overall_score,
                 elapsed_ms, created_at, raw_result)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                stock_code,
                result.get('trade_action', 'unknown'),
                result.get('trade_confidence', 0.0),
                result.get('analyst_consensus', ''),
                result.get('research_recommendation', ''),
                result.get('risk_level', ''),
                result.get('overall_score', 0.0),
                elapsed_ms,
                time.strftime('%Y-%m-%dT%H:%M:%S'),
                json.dumps(result, ensure_ascii=False),
            ))
            conn.commit()
    except Exception as e:
        logger.warning(f"[TradingAgents] 保存历史失败: {e}")


def _get_coordinator():
    """获取或创建 AgentCoordinator 单例"""
    from modules.llm_agents.agent_coordinator import AgentCoordinator
    from modules.llm_agents.llm_client import LLMClient

    # 模块级单例
    if not hasattr(bp, '_coordinator'):
        try:
            llm_client = LLMClient()
            bp._coordinator = AgentCoordinator(llm_client)
            logger.info("[TradingAgents] AgentCoordinator 单例已创建")
        except Exception as e:
            logger.warning(f"[TradingAgents] AgentCoordinator 创建失败: {e}")
            bp._coordinator = None

    return bp._coordinator


@bp.route('/api/trading-agents/status', methods=['GET'])
def trading_agents_status():
    """获取协调器状态"""
    try:
        coord = _get_coordinator()
        if coord is None:
            return jsonify({
                'success': False,
                'error': 'AgentCoordinator 未初始化',
                'data': {
                    'available': False,
                    'agents': [],
                }
            }), 503

        # 获取 Agent 列表
        agents = []
        for name in ('analyst_team', 'research_team', 'trader_agent', 'risk_manager', 'portfolio_manager'):
            obj = getattr(coord, name, None)
            if obj is not None:
                agents.append({
                    'name': name,
                    'class': obj.__class__.__name__,
                    'available': True,
                })

        return jsonify({
            'success': True,
            'data': {
                'available': True,
                'agents': agents,
                'agent_count': len(agents),
                'global_timeout': coord.GLOBAL_TIMEOUT,
                'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            }
        })
    except Exception as e:
        logger.error(f"[TradingAgents] status 失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/trading-agents/decide', methods=['POST'])
def trading_agents_decide():
    """触发决策"""
    try:
        coord = _get_coordinator()
        if coord is None:
            return jsonify({
                'success': False,
                'error': 'AgentCoordinator 未初始化',
            }), 503

        data = request.get_json(silent=True) or {}
        stock_code = data.get('stock_code', 'sz300620')

        # 构造 stock_data
        stock_data = data.get('stock_data') or _fetch_stock_data(stock_code)
        portfolio_state = data.get('portfolio_state', {
            'cash': 100000,
            'positions': {},
            'total_value': 100000,
        })

        # 执行决策 (带超时保护)
        start = time.time()
        try:
            decision = coord.make_decision(stock_data, portfolio_state)
            elapsed_ms = (time.time() - start) * 1000

            result = {
                'stock_code': stock_code,
                'analyst_consensus': decision.analyst_consensus,
                'analyst_confidence': decision.analyst_confidence,
                'research_direction': decision.research_direction,
                'research_recommendation': decision.research_recommendation,
                'research_confidence': decision.research_confidence,
                'trade_action': decision.trade_action,
                'trade_quantity': decision.trade_quantity,
                'trade_confidence': decision.trade_confidence,
                'risk_level': decision.risk_level,
                'risk_adjustment': decision.risk_adjustment,
                'portfolio_allocation': decision.portfolio_allocation,
                'rebalance_needed': decision.rebalance_needed,
                'overall_score': decision.overall_score,
                'elapsed_ms': elapsed_ms,
            }

            # 保存历史
            _save_decision(stock_code, result, elapsed_ms)

            return jsonify({
                'success': True,
                'data': result,
            })
        except Exception as e:
            logger.warning(f"[TradingAgents] 决策失败: {e}")
            return jsonify({
                'success': False,
                'error': str(e),
                'data': _fallback_decision(stock_code),
            }), 200  # 200 + fallback 数据，不阻塞前端
    except Exception as e:
        logger.error(f"[TradingAgents] decide 失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


def _fetch_stock_data(stock_code: str) -> Dict:
    """获取股票数据 (从 app.data_fetcher)"""
    try:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        from modules.dependencies import get_data_fetcher
        kline = get_data_fetcher().get_kline_data(stock_code, period='daily', count=60)
        if not kline or len(kline) == 0:
            return _mock_stock_data(stock_code)
        return {
            'code': stock_code,
            'kline': kline,
            'source': 'live',
        }
    except Exception as e:
        logger.warning(f"[TradingAgents] 获取 {stock_code} 数据失败: {e}")
        return _mock_stock_data(stock_code)


def _mock_stock_data(stock_code: str) -> Dict:
    """模拟股票数据 (fallback)"""
    return {
        'code': stock_code,
        'kline': [],
        'source': 'mock',
        'note': '使用模拟数据 (真实数据获取失败)',
    }


def _fallback_decision(stock_code: str) -> Dict:
    """降级决策 (LLM 不可用时)"""
    return {
        'stock_code': stock_code,
        'analyst_consensus': 'neutral',
        'analyst_confidence': 0.5,
        'trade_action': 'hold',
        'trade_quantity': 0.0,
        'trade_confidence': 0.5,
        'risk_level': 'medium',
        'overall_score': 0.5,
        'source': 'fallback',
    }


@bp.route('/api/trading-agents/agents', methods=['GET'])
def trading_agents_agents():
    """获取所有 Agent 状态"""
    try:
        coord = _get_coordinator()
        if coord is None:
            return jsonify({
                'success': False,
                'error': 'AgentCoordinator 未初始化',
            }), 503

        agents_info = []
        for name in ('analyst_team', 'research_team', 'trader_agent', 'risk_manager', 'portfolio_manager'):
            obj = getattr(coord, name, None)
            if obj is None:
                continue

            # 尝试调用 get_status()
            status = {}
            if hasattr(obj, 'get_status'):
                try:
                    status = obj.get_status()
                except Exception:
                    pass

            agents_info.append({
                'name': name,
                'class': obj.__class__.__name__,
                'status': status,
            })

        return jsonify({
            'success': True,
            'data': {
                'agents': agents_info,
                'count': len(agents_info),
            }
        })
    except Exception as e:
        logger.error(f"[TradingAgents] agents 失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/trading-agents/history', methods=['GET'])
def trading_agents_history():
    """获取决策历史"""
    try:
        _init_history_db()
        limit = min(int(request.args.get('limit', 20)), 100)
        stock_code = request.args.get('stock_code')

        with sqlite3.connect(_HISTORY_DB) as conn:
            conn.row_factory = sqlite3.Row
            if stock_code:
                rows = conn.execute(
                    'SELECT * FROM decisions WHERE stock_code = ? ORDER BY id DESC LIMIT ?',
                    (stock_code, limit)
                ).fetchall()
            else:
                rows = conn.execute(
                    'SELECT * FROM decisions ORDER BY id DESC LIMIT ?',
                    (limit,)
                ).fetchall()

            history = [dict(row) for row in rows]

        return jsonify({
            'success': True,
            'data': {
                'history': history,
                'count': len(history),
            }
        })
    except Exception as e:
        logger.error(f"[TradingAgents] history 失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# 初始化历史数据库
_init_history_db()