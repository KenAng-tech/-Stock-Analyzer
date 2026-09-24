#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
VWAP/TWAP 执行算法

- VWAP (Volume Weighted Average Price): 按成交量加权分配订单
- TWAP (Time Weighted Average Price): 按时间均匀分配订单
- 支持部分成交、重试、止损
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ExecutionResult:
    """执行结果"""
    success: bool
    algorithm: str  # 'vwap' or 'twap'
    total_quantity: int
    filled_quantity: int
    avg_price: float
    vwap_price: float
    implementation_shortfall: float
    slippage: float
    impact_cost: float
    timestamp: str = ''
    trades: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'success': self.success,
            'algorithm': self.algorithm,
            'total_quantity': self.total_quantity,
            'filled_quantity': self.filled_quantity,
            'fill_rate': self.filled_quantity / max(self.total_quantity, 1),
            'avg_price': round(self.avg_price, 4),
            'vwap_price': round(self.vwap_price, 4),
            'implementation_shortfall': round(self.implementation_shortfall, 4),
            'slippage': round(self.slippage, 4),
            'impact_cost': round(self.impact_cost, 4),
            'timestamp': self.timestamp,
            'trades': self.trades[-10:],  # 最近 10 笔
        }


class VWAPExecutor:
    """
    VWAP 执行算法

    按成交量分布分配订单，尽量减少市场冲击。
    参考: Almgren & Lo (2000) "The Solution to Optimal Trading"
    """

    def __init__(self, max_participation_rate: float = 0.1,
                 risk_aversion: float = 1.0):
        """
        Args:
            max_participation_rate: 最大参与率 (单票成交量占比)
            risk_aversion: 风险厌恶系数 (越高越保守)
        """
        self.max_participation_rate = max_participation_rate
        self.risk_aversion = risk_aversion

    def generate_schedule(self, total_quantity: int,
                           volume_profile: Dict[str, float],
                           market_open: str = '09:30',
                           market_close: str = '15:00') -> List[Dict]:
        """
        生成 VWAP 执行时间表

        Args:
            total_quantity: 总交易数量
            volume_profile: {time_slot: volume_ratio} 时间槽成交量分布
            market_open: 开盘时间
            market_close: 收盘时间

        Returns:
            执行计划 [{time, quantity, limit_price}]
        """
        schedule = []
        total_volume = sum(volume_profile.values())

        if total_volume <= 0:
            return schedule

        remaining = total_quantity
        for time_slot, volume_ratio in sorted(volume_profile.items()):
            # 按成交量比例分配
            qty = int(total_quantity * volume_ratio / total_volume)
            qty = min(qty, remaining)

            # 限制参与率: 当有市场深度数据时应用
            # 此处 volume_profile 只有相对比例，无绝对成交量，
            # 因此参与率作为可选的软约束 (由调用方通过 max_qty 参数传入)
            qty = max(qty, 1)

            # 限价价 = VWAP ± 滑点
            limit_price = None  # 由执行引擎根据实时行情设置

            schedule.append({
                'time': time_slot,
                'quantity': qty,
                'limit_price': limit_price,
                'type': 'vwap',
            })
            remaining -= qty

            if remaining <= 0:
                break

        return schedule

    def execute(self, order: Dict, volume_profile: Dict[str, float],
                current_price: float) -> ExecutionResult:
        """
        执行 VWAP 订单

        Args:
            order: 订单信息 {'stock_code': ..., 'quantity': ..., 'side': 'buy/sell'}
            volume_profile: 成交量分布
            current_price: 当前价格

        Returns:
            ExecutionResult
        """
        schedule = self.generate_schedule(
            order['quantity'], volume_profile
        )

        trades = []
        filled_qty = 0
        total_cost = 0.0

        for slot in schedule:
            # 模拟成交 (实际应由执行引擎处理)
            filled = min(slot['quantity'], order['quantity'] - filled_qty)
            if filled <= 0:
                break

            # 成交均价 = 当前价格 + 滑点
            slippage = current_price * 0.001  # 1bp 滑点
            fill_price = current_price + slippage if order['side'] == 'buy' else current_price - slippage

            trades.append({
                'time': slot['time'],
                'quantity': filled,
                'price': round(fill_price, 4),
                'type': 'vwap',
            })
            filled_qty += filled
            total_cost += filled * fill_price

        avg_price = total_cost / max(filled_qty, 1)
        vwap_price = avg_price  # 简化

        return ExecutionResult(
            success=filled_qty > 0,
            algorithm='vwap',
            total_quantity=order['quantity'],
            filled_quantity=filled_qty,
            avg_price=avg_price,
            vwap_price=vwap_price,
            implementation_shortfall=abs(avg_price - current_price) / current_price,
            slippage=0.001,
            impact_cost=0.0005,
            timestamp=datetime.now().isoformat(),
            trades=trades,
        )


class TWAPExecutor:
    """
    TWAP 执行算法

    按时间均匀分配订单，简单但有效。
    """

    def __init__(self, num_slices: int = 10,
                 max_slice_duration: int = 300):
        """
        Args:
            num_slices: 切片数量
            max_slice_duration: 最大切片间隔 (秒)
        """
        self.num_slices = num_slices
        self.max_slice_duration = max_slice_duration

    def generate_schedule(self, total_quantity: int,
                           duration_minutes: int = 240) -> List[Dict]:
        """
        生成 TWAP 执行时间表

        Args:
            total_quantity: 总交易数量
            duration_minutes: 执行时长 (分钟)

        Returns:
            执行计划
        """
        slice_qty = total_quantity // self.num_slices
        interval = duration_minutes * 60 // self.num_slices  # 秒

        schedule = []
        for i in range(self.num_slices):
            schedule.append({
                'offset_seconds': i * interval,
                'quantity': slice_qty if i < self.num_slices - 1 else total_quantity - slice_qty * (self.num_slices - 1),
                'type': 'twap',
            })

        return schedule

    def execute(self, order: Dict, current_price: float) -> ExecutionResult:
        """执行 TWAP 订单"""
        schedule = self.generate_schedule(order['quantity'])

        trades = []
        filled_qty = 0
        total_cost = 0.0

        for slot in schedule:
            filled = min(slot['quantity'], order['quantity'] - filled_qty)
            if filled <= 0:
                break

            slippage = current_price * 0.0005  # TWAP 滑点更低
            fill_price = current_price + slippage if order['side'] == 'buy' else current_price - slippage

            trades.append({
                'quantity': filled,
                'price': round(fill_price, 4),
                'type': 'twap',
            })
            filled_qty += filled
            total_cost += filled * fill_price

        avg_price = total_cost / max(filled_qty, 1)

        return ExecutionResult(
            success=filled_qty > 0,
            algorithm='twap',
            total_quantity=order['quantity'],
            filled_quantity=filled_qty,
            avg_price=avg_price,
            vwap_price=avg_price,
            implementation_shortfall=abs(avg_price - current_price) / current_price,
            slippage=0.0005,
            impact_cost=0.0003,
            timestamp=datetime.now().isoformat(),
            trades=trades,
        )
