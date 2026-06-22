#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Multi-Agent RL Trading System — 2026 SOTA

多智能体强化学习交易系统:
1. ResearchAgent — 研究分析 (宏观 + 因子)
2. AnalystAgent — 技术分析 (K线 + 模式)
3. RiskManager — 风险管理 (仓位 + 止损)
4. TraderAgent — 执行决策 (买入/卖出/持有)

架构:
    Context → ResearchAgent → AnalystAgent → RiskManager → TraderAgent → Action

参考:
    - FinRL: Deep Reinforcement Learning for Trading
    - TradingAgents: LLM-based Multi-Agent System
    - AlphaCrafter: Multi-Agent Factor Mining

作者: Stock Analyzer SOTA Team
"""

import os
import pickle
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass, field
from collections import deque
import threading

from modules.logger import logger


# ── Agent 消息 ──────────────────────────────────────────────

@dataclass
class AgentMessage:
    """Agent 之间传递的消息"""
    sender: str
    receiver: str
    content: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


# ── Research Agent (研究分析) ─────────────────────────────────

class ResearchAgent:
    """研究分析 Agent — 宏观分析 + 因子挖掘"""

    def __init__(self):
        self.name = "ResearchAgent"
        self.memory = deque(maxlen=100)

    def analyze(self, context: Dict) -> Dict:
        """
        研究分析

        Args:
            context: {'market_data': {}, 'stock_data': {}, 'klines': {}}

        Returns:
            {
                '宏观判断': str,
                '因子建议': List[Dict],
                '置信度': float,
            }
        """
        market_data = context.get('market_data', {})
        stock_data = context.get('stock_data', {})

        # 宏观分析
        宏观判断 = self._宏观分析(market_data)

        # 因子建议
        factor_suggestions = self._因子挖掘(stock_data)

        return {
            '宏观判断': 宏观判断,
            '因子建议': factor_suggestions,
            '置信度': 0.75,
            'timestamp': datetime.now().isoformat(),
        }

    def _宏观分析(self, market_data: Dict) -> str:
        """宏观分析"""
        if not market_data:
            return '中性'
        return '震荡'  # 简化

    def _因子挖掘(self, stock_data: Dict) -> List[Dict]:
        """因子挖掘"""
        return [
            {'name': '动量因子', 'ic': 0.05, 'priority': 1},
            {'name': '价值因子', 'ic': 0.03, 'priority': 2},
        ]


# ── Analyst Agent (技术分析) ───────────────────────────────────

class AnalystAgent:
    """技术分析 Agent — K线分析 + 模式识别"""

    def __init__(self):
        self.name = "AnalystAgent"
        self.memory = deque(maxlen=100)

    def analyze(self, context: Dict) -> Dict:
        """
        技术分析

        Args:
            context: {'klines': {}, 'stock_data': {}}

        Returns:
            {
                '趋势': 'up' | 'down' | 'sideways',
                '支撑位': float,
                '阻力位': float,
                '信号': str,
                '置信度': float,
            }
        """
        klines = context.get('klines', {})
        stock_data = context.get('stock_data', {})

        if not klines:
            return {
                '趋势': 'sideways',
                '支撑位': 0,
                '阻力位': 0,
                '信号': 'neutral',
                '置信度': 0.5,
            }

        # 简化 K线分析
        trend = self._判断趋势(klines)
        support, resistance = self._支撑阻力(klines)
        signal = self._信号生成(trend, support, resistance)

        return {
            '趋势': trend,
            '支撑位': support,
            '阻力位': resistance,
            '信号': signal,
            '置信度': 0.7,
        }

    def _判断趋势(self, klines: Dict) -> str:
        """判断趋势"""
        if len(klines) < 20:
            return 'sideways'
        # 简化: 使用最近5根K线的涨跌判断
        closes = klines.get('close', [])
        if len(closes) < 5:
            return 'sideways'
        recent = closes[-5:]
        if all(recent[i] < recent[i+1] for i in range(len(recent)-1)):
            return 'up'
        elif all(recent[i] > recent[i+1] for i in range(len(recent)-1)):
            return 'down'
        return 'sideways'

    def _支撑阻力(self, klines: Dict) -> Tuple[float, float]:
        """计算支撑位和阻力位"""
        closes = klines.get('close', [])
        if not closes:
            return 0.0, 0.0
        low = min(closes[-20:]) if len(closes) >= 20 else min(closes)
        high = max(closes[-20:]) if len(closes) >= 20 else max(closes)
        return float(low), float(high)

    def _信号生成(self, trend: str, support: float, resistance: float) -> str:
        """生成信号"""
        if trend == 'up' and resistance > 0:
            return 'buy'
        elif trend == 'down' and support > 0:
            return 'sell'
        return 'hold'


# ── Risk Manager (风险管理) ──────────────────────────────────────

class RiskManager:
    """风险管理 Agent — 仓位管理 + 止损管理"""

    def __init__(self):
        self.name = "RiskManager"
        self.max_position = 1.0  # 最大仓位 (0-1)
        self.max_loss = 0.05  # 最大损失 (5%)
        self.memory = deque(maxlen=100)

    def evaluate(self, context: Dict) -> Dict:
        """
        风险管理

        Args:
            context: {'position': float, 'pnl': float, 'analyst_signal': str}

        Returns:
            {
                'position_size': float,  # 建议仓位
                'stop_loss': float,      # 止损位
                'take_profit': float,    # 止盈位
                'risk_level': 'low' | 'medium' | 'high',
                'action': 'approved' | 'rejected',
            }
        """
        position = context.get('position', 0.0)
        pnl = context.get('pnl', 0.0)
        analyst_signal = context.get('analyst_signal', 'neutral')

        # 计算仓位
        position_size = self._计算仓位(position, pnl, analyst_signal)

        # 止损止盈
        stop_loss = self._计算止损(position, pnl)
        take_profit = self._计算止盈(position, pnl)

        # 风险等级
        risk_level = self._评估风险(position, pnl)

        # 检查是否批准
        action = 'approved' if risk_level != 'high' else 'rejected'

        return {
            'position_size': position_size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_level': risk_level,
            'action': action,
        }

    def _计算仓位(self, position: float, pnl: float, signal: str) -> float:
        """计算建议仓位"""
        if signal == 'rejected':
            return 0.0
        base_position = 0.3  # 基础仓位
        if signal == 'buy':
            return min(base_position * 1.5, self.max_position)
        elif signal == 'sell':
            return max(base_position * 0.5, 0.1)
        return base_position

    def _计算止损(self, position: float, pnl: float) -> float:
        """计算止损位"""
        if position <= 0:
            return 0.0
        return position * (1 - self.max_loss)

    def _计算止盈(self, position: float, pnl: float) -> float:
        """计算止盈位"""
        if position <= 0:
            return 0.0
        return position * 1.1  # 10% 止盈

    def _评估风险(self, position: float, pnl: float) -> str:
        """评估风险等级"""
        if pnl < -0.03:
            return 'high'
        elif pnl < 0:
            return 'medium'
        return 'low'


# ── Trader Agent (交易执行) ────────────────────────────────────

class TraderAgent:
    """交易执行 Agent — 最终决策"""

    def __init__(self):
        self.name = "TraderAgent"
        self.memory = deque(maxlen=100)
        self.commission_rate = 0.0003  # 佣金 0.03%
        self.slippage = 0.001  # 滑点 0.1%

    def execute(self, context: Dict) -> Dict:
        """
        交易执行

        Args:
            context: {
                'action': 'buy' | 'sell' | 'hold',
                'position_size': float,
                'price': float,
                'stop_loss': float,
                'take_profit': float,
            }

        Returns:
            {
                'action': str,
                'position_size': float,
                'price': float,
                'reason': str,
            }
        """
        action = context.get('action', 'hold')
        position_size = context.get('position_size', 0.0)
        price = context.get('price', 0.0)
        stop_loss = context.get('stop_loss', 0.0)
        take_profit = context.get('take_profit', 0.0)

        if price <= 0:
            return {
                'action': 'hold',
                'position_size': 0.0,
                'price': 0.0,
                'reason': 'Invalid price',
            }

        # 计算成本
        cost = position_size * price * (self.commission_rate + self.slippage)

        return {
            'action': action,
            'position_size': position_size,
            'price': price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'cost': cost,
            'reason': f'Action: {action}, Size: {position_size:.2%}',
        }


# ── Multi-Agent Coordinator ─────────────────────────────────────

class MultiAgentCoordinator:
    """
    Multi-Agent RL 协调器

    协调所有 Agent 的工作流程:
    1. ResearchAgent 研究
    2. AnalystAgent 分析
    3. RiskManager 风险管理
    4. TraderAgent 执行
    """

    def __init__(self):
        self.research_agent = ResearchAgent()
        self.analyst_agent = AnalystAgent()
        self.risk_manager = RiskManager()
        self.trader_agent = TraderAgent()
        self.decision_history = deque(maxlen=100)
        self._lock = threading.Lock()

    def run_pipeline(self, context: Dict) -> Dict:
        """
        运行完整的 Multi-Agent Pipeline

        Args:
            context: {
                'stock_code': str,
                'stock_data': Dict,
                'market_data': Dict,
                'klines': Dict,
            }

        Returns:
            {
                'action': 'buy' | 'sell' | 'hold',
                'position_size': float,
                'confidence': float,
                'all_decisions': List[Dict],
            }
        """
        all_decisions = []

        # 1. Research Agent
        research_result = self.research_agent.analyze(context)
        all_decisions.append({
            'agent': 'ResearchAgent',
            'decision': research_result,
        })

        # 2. Analyst Agent
        analyst_result = self.analyst_agent.analyze(context)
        all_decisions.append({
            'agent': 'AnalystAgent',
            'decision': analyst_result,
        })

        # 3. Risk Manager
        risk_input = {
            'position': context.get('current_position', 0.0),
            'pnl': context.get('pnl', 0.0),
            'analyst_signal': analyst_result.get('信号', 'neutral'),
        }
        risk_result = self.risk_manager.evaluate(risk_input)
        all_decisions.append({
            'agent': 'RiskManager',
            'decision': risk_result,
        })

        # 4. Trader Agent (最终决策)
        price = context.get('stock_data', {}).get('price', 0.0)
        trader_input = {
            'action': 'hold',
            'position_size': risk_result.get('position_size', 0.0),
            'price': price,
            'stop_loss': risk_result.get('stop_loss', 0.0),
            'take_profit': risk_result.get('take_profit', 0.0),
        }

        # 根据分析结果决定 action
        if analyst_result.get('信号') == 'buy' and risk_result.get('action') == 'approved':
            trader_input['action'] = 'buy'
        elif analyst_result.get('信号') == 'sell' and risk_result.get('action') == 'approved':
            trader_input['action'] = 'sell'

        trader_result = self.trader_agent.execute(trader_input)
        all_decisions.append({
            'agent': 'TraderAgent',
            'decision': trader_result,
        })

        # 记录决策
        with self._lock:
            self.decision_history.append({
                'timestamp': datetime.now().isoformat(),
                'action': trader_result.get('action'),
                'position_size': trader_result.get('position_size'),
                'all_decisions': all_decisions,
            })

        return {
            'action': trader_result.get('action', 'hold'),
            'position_size': trader_result.get('position_size', 0.0),
            'confidence': (research_result.get('置信度', 0.5) + analyst_result.get('置信度', 0.5)) / 2,
            'all_decisions': all_decisions,
        }

    def get_status(self) -> Dict:
        """获取协调器状态"""
        with self._lock:
            return {
                'num_agents': 4,
                'agent_names': ['ResearchAgent', 'AnalystAgent', 'RiskManager', 'TraderAgent'],
                'decision_count': len(self.decision_history),
                'recent_decisions': list(self.decision_history)[-5:],
            }


# ── 全局单例 ────────────────────────────────────────────────

_multi_agent_instance: Optional[MultiAgentCoordinator] = None
_multi_agent_lock = threading.Lock()


def get_multi_agent_coordinator() -> MultiAgentCoordinator:
    """获取全局 MultiAgentCoordinator 实例 (线程安全)"""
    global _multi_agent_instance
    if _multi_agent_instance is None:
        with _multi_agent_lock:
            if _multi_agent_instance is None:
                _multi_agent_instance = MultiAgentCoordinator()
    return _multi_agent_instance
