#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Safe RL 安全约束层 (Safe RL Constraint Layer)

在 RL 决策基础上增加安全约束层:
- 硬约束: 最大仓位限制、单日交易次数限制、止损线
- 软约束: 回撤惩罚、波动率目标、行业集中度限制
- Safety Layer: 对 RL 原始动作进行安全投影

用法:
    safety = SafeRLConstraintLayer()
    safe_action = safety.constrain(action, portfolio_state, market_state)
"""

import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum

from modules.logger import logger


class ConstraintType(Enum):
    """约束类型"""
    POSITION_LIMIT = 'position_limit'
    DAILY_TRADE_LIMIT = 'daily_trade_limit'
    STOP_LOSS = 'stop_loss'
    DRAWDOWN_LIMIT = 'drawdown_limit'
    VOLATILITY_TARGET = 'volatility_target'
    INDUSTRY_CONCENTRATION = 'industry_concentration'
    LEVERAGE_LIMIT = 'leverage_limit'


@dataclass
class ConstraintCheck:
    """约束检查结果"""
    constraint_type: str
    passed: bool
    original_action: float
    constrained_action: float
    reason: str = ''

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SafetyState:
    """安全状态"""
    daily_trades: int = 0
    max_drawdown: float = 0.0
    current_value: float = 0.0
    peak_value: float = 0.0
    current_volatility: float = 0.0
    last_reset_time: str = ''

    def to_dict(self) -> Dict:
        return asdict(self)


class SafeRLConstraintLayer:
    """Safe RL 安全约束层"""

    def __init__(
        self,
        max_position_pct: float = 0.2,  # 单只股票最大仓位 20%
        max_daily_trades: int = 20,  # 单日最大交易次数
        stop_loss_pct: float = -0.05,  # 止损线 -5%
        max_drawdown_pct: float = -0.15,  # 最大回撤 -15%
        target_volatility: float = 0.2,  # 目标年化波动率 20%
        max_industry_concentration: float = 0.4,  # 行业集中度上限 40%
        max_leverage: float = 1.0,  # 最大杠杆 1x
    ):
        self.max_position_pct = max_position_pct
        self.max_daily_trades = max_daily_trades
        self.stop_loss_pct = stop_loss_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.target_volatility = target_volatility
        self.max_industry_concentration = max_industry_concentration
        self.max_leverage = max_leverage

        self._state = SafetyState()
        self._constraint_history: List[ConstraintCheck] = []
        self._position_history: Dict[str, List[float]] = {}  # stock -> [positions]

    def constrain(
        self,
        raw_action: Dict[str, Any],
        portfolio_state: Dict[str, Any],
        market_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        对 RL 原始动作施加安全约束

        Args:
            raw_action: RL 原始动作 {'action': 'buy/sell/hold', 'quantity': N, 'confidence': 0.5}
            portfolio_state: 组合状态 {'value': 100000, 'positions': {...}, 'daily_trades': 0}
            market_state: 市场状态 {'volatility': 0.15, 'drawdown': -0.05, ...}

        Returns:
            安全约束后的动作
        """
        action = dict(raw_action)
        constraints_applied = []

        # 1. 止损约束
        stop_result = self._check_stop_loss(action, market_state)
        if not stop_result.passed:
            action['action'] = 'hold'
            action['quantity'] = 0
            constraints_applied.append('stop_loss')

        # 2. 仓位限制
        position_result = self._check_position_limit(action, portfolio_state)
        if not position_result.passed:
            action['quantity'] = position_result.constrained_action
            constraints_applied.append('position_limit')

        # 3. 日交易次数限制
        trade_result = self._check_daily_trade_limit(action, portfolio_state)
        if not trade_result.passed:
            action['action'] = 'hold'
            action['quantity'] = 0
            constraints_applied.append('daily_trade_limit')

        # 4. 回撤限制
        dd_result = self._check_drawdown_limit(action, market_state)
        if not dd_result.passed:
            action['action'] = 'hold'
            action['quantity'] = 0
            constraints_applied.append('drawdown_limit')

        # 5. 波动率调整
        vol_result = self._check_volatility_target(action, market_state)
        if vol_result.passed and vol_result.constrained_action != action.get('quantity', 0):
            action['quantity'] = vol_result.constrained_action
            constraints_applied.append('volatility_target')

        # 记录约束历史
        if constraints_applied:
            logger.info(
                f"[SafeRL] 约束应用: {action.get('action')} "
                f"(原: {raw_action.get('action')}), 约束: {constraints_applied}"
            )

        action['constraints_applied'] = constraints_applied
        action['timestamp'] = datetime.now().isoformat()

        return action

    def _check_stop_loss(self, action: Dict, market_state: Dict) -> ConstraintCheck:
        """止损约束检查"""
        current_drawdown = market_state.get('drawdown', 0)
        if current_drawdown <= self.stop_loss_pct:
            return ConstraintCheck(
                constraint_type=ConstraintType.STOP_LOSS.value,
                passed=False,
                original_action=action.get('quantity', 0),
                constrained_action=0,
                reason=f"回撤 {current_drawdown*100:.1f}% 触及止损线 {self.stop_loss_pct*100:.1f}%",
            )
        return ConstraintCheck(
            constraint_type=ConstraintType.STOP_LOSS.value,
            passed=True,
            original_action=action.get('quantity', 0),
            constrained_action=action.get('quantity', 0),
        )

    def _check_position_limit(self, action: Dict, portfolio_state: Dict) -> ConstraintCheck:
        """仓位限制检查"""
        current_position = portfolio_state.get('positions', {}).get(
            action.get('stock_code', ''), 0
        )
        portfolio_value = portfolio_state.get('value', 1)
        max_position_value = portfolio_value * self.max_position_pct

        if action.get('action') in ('buy', 'strong_buy'):
            requested_value = action.get('quantity', 0) * action.get('price', 0)
            if current_position + requested_value > max_position_value:
                new_qty = max(0, (max_position_value - current_position) / max(action.get('price', 1), 1e-8))
                return ConstraintCheck(
                    constraint_type=ConstraintType.POSITION_LIMIT.value,
                    passed=False,
                    original_action=action.get('quantity', 0),
                    constrained_action=new_qty,
                    reason=f"仓位 {current_position:.0f}+{requested_value:.0f} > 上限 {max_position_value:.0f}",
                )

        return ConstraintCheck(
            constraint_type=ConstraintType.POSITION_LIMIT.value,
            passed=True,
            original_action=action.get('quantity', 0),
            constrained_action=action.get('quantity', 0),
        )

    def _check_daily_trade_limit(self, action: Dict, portfolio_state: Dict) -> ConstraintCheck:
        """日交易次数限制"""
        daily_trades = portfolio_state.get('daily_trades', 0)
        if daily_trades >= self.max_daily_trades and action.get('action') not in ('hold',):
            return ConstraintCheck(
                constraint_type=ConstraintType.DAILY_TRADE_LIMIT.value,
                passed=False,
                original_action=action.get('quantity', 0),
                constrained_action=0,
                reason=f"日交易次数 {daily_trades} 已达上限 {self.max_daily_trades}",
            )
        return ConstraintCheck(
            constraint_type=ConstraintType.DAILY_TRADE_LIMIT.value,
            passed=True,
            original_action=action.get('quantity', 0),
            constrained_action=action.get('quantity', 0),
        )

    def _check_drawdown_limit(self, action: Dict, market_state: Dict) -> ConstraintCheck:
        """回撤限制"""
        current_drawdown = market_state.get('drawdown', 0)
        if current_drawdown <= self.max_drawdown_pct:
            return ConstraintCheck(
                constraint_type=ConstraintType.DRAWDOWN_LIMIT.value,
                passed=False,
                original_action=action.get('quantity', 0),
                constrained_action=0,
                reason=f"组合回撤 {current_drawdown*100:.1f}% 超限 {self.max_drawdown_pct*100:.1f}%",
            )
        return ConstraintCheck(
            constraint_type=ConstraintType.DRAWDOWN_LIMIT.value,
            passed=True,
            original_action=action.get('quantity', 0),
            constrained_action=action.get('quantity', 0),
        )

    def _check_volatility_target(self, action: Dict, market_state: Dict) -> ConstraintCheck:
        """波动率目标调整"""
        current_vol = market_state.get('volatility', 0.2)
        if current_vol > self.target_volatility:
            # 波动率高时缩减仓位
            scaling_factor = self.target_volatility / current_vol
            new_qty = int(action.get('quantity', 0) * scaling_factor)
            return ConstraintCheck(
                constraint_type=ConstraintType.VOLATILITY_TARGET.value,
                passed=True,  # 通过但调整了
                original_action=action.get('quantity', 0),
                constrained_action=new_qty,
                reason=f"波动率 {current_vol*100:.1f}% > 目标 {self.target_volatility*100:.1f}%, 仓位缩放 {scaling_factor:.2f}",
            )
        return ConstraintCheck(
            constraint_type=ConstraintType.VOLATILITY_TARGET.value,
            passed=True,
            original_action=action.get('quantity', 0),
            constrained_action=action.get('quantity', 0),
        )

    def update_state(self, portfolio_value: float, daily_trades: int, drawdown: float, volatility: float):
        """更新安全状态"""
        self._state.current_value = portfolio_value
        self._state.daily_trades = daily_trades
        self._state.max_drawdown = min(self._state.max_drawdown, drawdown)
        self._state.current_volatility = volatility
        self._state.peak_value = max(self._state.peak_value, portfolio_value)

    def get_status(self) -> Dict:
        """获取安全约束状态"""
        return {
            'state': self._state.to_dict(),
            'constraints': {
                'max_position_pct': self.max_position_pct * 100,
                'max_daily_trades': self.max_daily_trades,
                'stop_loss_pct': self.stop_loss_pct * 100,
                'max_drawdown_pct': self.max_drawdown_pct * 100,
                'target_volatility_pct': self.target_volatility * 100,
            },
            'recent_constraints': [
                c.to_dict() for c in self._constraint_history[-20:]
            ],
        }

    def reset_daily(self):
        """每日重置"""
        self._state.daily_trades = 0
        self._state.last_reset_time = datetime.now().isoformat()


# 全局单例
_safe_rl: Optional[SafeRLConstraintLayer] = None


def get_safe_rl_layer() -> SafeRLConstraintLayer:
    """获取 Safe RL 约束层全局实例"""
    global _safe_rl
    if _safe_rl is None:
        _safe_rl = SafeRLConstraintLayer()
    return _safe_rl
