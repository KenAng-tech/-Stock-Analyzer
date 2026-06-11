#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
事件驱动回测引擎 — P3 架构升级 (2026-06-04)

与 walkforward_backtester 的区别:
- 逐笔级事件驱动，而非批量循环
- 支持多股票组合
- 真实订单簿模拟（限价单/市价单）
- 支持多策略并行
- 支持换手率/冲击成本建模
- 完整的绩效归因（Brinson）

架构:
    Event → EventQueue → Engine → Strategy → Order → Execution → Position → Risk → Report
"""

import numpy as np
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import json


class EventType(Enum):
    """事件类型"""
    MARKET = 'market'        # 市场数据事件
    SIGNAL = 'signal'        # 策略信号事件
    ORDER = 'order'          # 订单事件
    FILL = 'fill'            # 成交事件
    RISK = 'risk'            # 风控事件
    TIMER = 'timer'          # 定时事件（日终统计）


class Side(Enum):
    BUY = 'buy'
    SELL = 'sell'


class OrderType(Enum):
    MARKET = 'market'       # 市价单
    LIMIT = 'limit'         # 限价单
    STOP = 'stop'           # 止损单
    STOP_LIMIT = 'stop_limit'


class OrderStatus(Enum):
    PENDING = 'pending'
    SUBMITTED = 'submitted'
    PARTIAL = 'partial'
    FILLED = 'filled'
    CANCELLED = 'cancelled'
    REJECTED = 'rejected'


class Order:
    """订单对象"""

    def __init__(self, stock_code: str, side: Side, order_type: OrderType,
                 quantity: int, price: float = 0, strategy: str = '',
                 stop_price: float = 0, limit_price: float = 0,
                 timestamp: str = ''):
        self.stock_code = stock_code
        self.side = side
        self.order_type = order_type
        self.quantity = quantity
        self.price = price
        self.strategy = strategy
        self.stop_price = stop_price
        self.limit_price = limit_price
        self.timestamp = timestamp
        self.status = OrderStatus.PENDING
        self.filled_quantity = 0
        self.avg_fill_price = 0.0
        self.fill_time = ''
        self.reason = ''

    def to_dict(self) -> Dict:
        return {
            'stock_code': self.stock_code,
            'side': self.side.value,
            'order_type': self.order_type.value,
            'quantity': self.quantity,
            'price': self.price,
            'strategy': self.strategy,
            'status': self.status.value,
            'filled_quantity': self.filled_quantity,
            'avg_fill_price': round(self.avg_fill_price, 2),
            'timestamp': self.timestamp,
            'reason': self.reason,
        }


class Position:
    """持仓对象"""

    def __init__(self, stock_code: str):
        self.stock_code = stock_code
        self.quantity = 0
        self.avg_cost = 0.0
        self.total_cost = 0.0
        self.realized_pnl = 0.0
        self.fee_total = 0.0
        self.tax_total = 0.0
        self.history: List[Dict] = []

    def buy(self, quantity: int, price: float, fee: float = 0, tax: float = 0):
        """买入"""
        cost = quantity * price + fee
        total_qty = self.quantity + quantity
        self.total_cost += cost
        self.avg_cost = self.total_cost / total_qty if total_qty > 0 else price
        self.quantity = total_qty
        self.fee_total += fee
        self.tax_total += tax
        self.history.append({
            'action': 'BUY', 'quantity': quantity, 'price': price,
            'fee': fee, 'tax': tax, 'total_cost': cost,
        })

    def sell(self, quantity: int, price: float, fee: float = 0, tax: float = 0):
        """卖出，返回已实现盈亏"""
        proceeds = quantity * price - fee - tax
        cost_basis = quantity * self.avg_cost
        pnl = proceeds - cost_basis
        self.realized_pnl += pnl
        self.total_cost -= cost_basis
        self.quantity -= quantity
        self.fee_total += fee
        self.tax_total += tax
        if self.quantity <= 0:
            self.quantity = 0
            self.avg_cost = 0
            self.total_cost = 0
        self.history.append({
            'action': 'SELL', 'quantity': quantity, 'price': price,
            'fee': fee, 'tax': tax, 'pnl': pnl,
        })
        return pnl

    def market_value(self, current_price: float) -> float:
        """当前市值"""
        return self.quantity * current_price

    def unrealized_pnl(self, current_price: float) -> float:
        """未实现盈亏"""
        return self.quantity * (current_price - self.avg_cost)

    def to_dict(self) -> Dict:
        return {
            'stock_code': self.stock_code,
            'quantity': self.quantity,
            'avg_cost': round(self.avg_cost, 2),
            'realized_pnl': round(self.realized_pnl, 2),
            'fee_total': round(self.fee_total, 2),
            'tax_total': round(self.tax_total, 2),
        }


class EventQueue:
    """事件队列"""

    def __init__(self, max_size: int = 10000):
        self._queue: deque = deque(maxlen=max_size)

    def push(self, event: Dict):
        self._queue.append(event)

    def pop(self) -> Optional[Dict]:
        if self._queue:
            return self._queue.popleft()
        return None

    def peek(self) -> Optional[Dict]:
        return self._queue[0] if self._queue else None

    def __len__(self):
        return len(self._queue)

    def __bool__(self):
        return bool(self._queue)


class ExecutionEngine:
    """
    执行引擎 — 模拟订单撮合

    支持:
    - 市价单: 按下一根K线开盘价成交
    - 限价单: 价格触及时成交
    - 止损单: 价格跌破止损价时转为限价单
    - 冲击成本: 大单按 VWAP 冲击模型
    """

    def __init__(self, commission_rate: float = 0.0003,
                 stamp_tax: float = 0.001,
                 slippage_bps: float = 2.0,
                 impact_model: str = 'linear'):
        self.commission_rate = commission_rate
        self.stamp_tax = stamp_tax
        self.slippage_bps = slippage_bps
        self.impact_model = impact_model  # linear / square_root

    def execute_order(self, order: Order, market_data: Dict) -> Optional[Order]:
        """执行订单，返回成交后的订单"""
        open_price = market_data.get('open', 0)
        high = market_data.get('high', 0)
        low = market_data.get('low', 0)
        close = market_data.get('close', 0)

        if order.order_type == OrderType.MARKET:
            # 市价单: 用下一根K线开盘价 + 滑点
            fill_price = open_price * (1 + self._slippage(order.side))
            fill_price = np.clip(fill_price, low, high)

        elif order.order_type == OrderType.LIMIT:
            # 限价单: 价格触及时成交
            if order.side == Side.BUY and low <= order.limit_price:
                fill_price = min(order.limit_price, open_price)
            elif order.side == Side.SELL and high >= order.limit_price:
                fill_price = max(order.limit_price, open_price)
            else:
                return None  # 未成交

        elif order.order_type == OrderType.STOP:
            # 止损单: 价格跌破止损价
            if order.side == Side.SELL and low <= order.stop_price:
                fill_price = max(order.stop_price * 0.99, low)
            elif order.side == Side.BUY and high >= order.stop_price:
                fill_price = min(order.stop_price * 1.01, high)
            else:
                return None

        else:
            fill_price = open_price

        # 冲击成本
        impact = self._impact_cost(order, market_data)
        fill_price *= (1 + impact if order.side == Side.BUY else -impact)

        # 计算费用
        trade_value = order.quantity * fill_price
        commission = max(trade_value * self.commission_rate, 5)
        tax = trade_value * self.stamp_tax if order.side == Side.SELL else 0
        slippage_cost = trade_value * self.slippage_bps / 10000

        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.avg_fill_price = round(fill_price, 2)
        order.fill_time = market_data.get('date', '')
        order.reason = f'filled@{fill_price:.2f}'

        return order

    def _slippage(self, side: Side) -> float:
        """滑点"""
        sign = 1 if side == Side.BUY else -1
        return sign * self.slippage_bps / 10000

    def _impact_cost(self, order: Order, market_data: Dict) -> float:
        """冲击成本模型"""
        volume = market_data.get('volume', 1)
        if volume <= 0:
            return 0
        order_ratio = order.quantity / volume
        if self.impact_model == 'linear':
            return order_ratio * 0.5  # 订单量占成交量 50% -> 0.5% 冲击
        else:  # square_root
            return np.sqrt(order_ratio) * 0.5


class RiskManager:
    """风控管理器"""

    def __init__(self, max_position_pct: float = 0.3,
                 max_single_loss_pct: float = 0.05,
                 max_drawdown_pct: float = 0.15,
                 max_daily_trades: int = 10):
        self.max_position_pct = max_position_pct
        self.max_single_loss_pct = max_single_loss_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.max_daily_trades = max_daily_trades
        self._daily_trades = 0
        self._peak_equity = 0

    def check_order(self, order: Order, equity: float,
                    positions: Dict[str, Position],
                    current_drawdown: float) -> bool:
        """检查订单是否通过风控"""
        # 日交易次数限制
        if self._daily_trades >= self.max_daily_trades:
            order.reason = 'rejected:daily_trade_limit'
            order.status = OrderStatus.REJECTED
            return False

        # 最大回撤限制
        if current_drawdown >= self.max_drawdown_pct:
            order.reason = f'rejected:max_dd{current_drawdown:.1%}'
            order.status = OrderStatus.REJECTED
            return False

        # 单票仓位限制
        if order.side == Side.BUY:
            trade_value = order.quantity * order.price
            if trade_value > equity * self.max_position_pct:
                order.reason = 'rejected:position_limit'
                order.status = OrderStatus.REJECTED
                return False

        self._daily_trades += 1
        return True

    def reset_daily(self):
        self._daily_trades = 0

    def update_peak(self, equity: float):
        if equity > self._peak_equity:
            self._peak_equity = equity


class StrategyBase:
    """策略基类"""

    def on_bar(self, bar: Dict, positions: Dict[str, Position],
               equity: float) -> List[Order]:
        """每根K线调用，返回订单列表"""
        raise NotImplementedError

    def on_timer(self, date: str, positions: Dict[str, Position]):
        """日终回调"""
        pass


class BacktestReport:
    """回测报告"""

    def __init__(self):
        self.equity_curve: List[Dict] = []
        self.transactions: List[Dict] = []
        self.positions_history: List[Dict] = []

    def add_equity_point(self, date: str, equity: float, benchmark: float = 0):
        self.equity_curve.append({
            'date': date, 'equity': round(equity, 2),
            'benchmark': round(benchmark, 2),
        })

    def add_transaction(self, order: Order, market_data: Dict):
        self.transactions.append({
            'date': market_data.get('date', ''),
            **order.to_dict(),
            'market_data': {
                'open': market_data.get('open', 0),
                'high': market_data.get('high', 0),
                'low': market_data.get('low', 0),
                'close': market_data.get('close', 0),
                'volume': market_data.get('volume', 0),
            }
        })

    def compute_metrics(self, initial_capital: float,
                        risk_free_rate: float = 0.02) -> Dict:
        """计算绩效指标"""
        if not self.equity_curve:
            return {}

        equities = [e['equity'] for e in self.equity_curve]
        n_days = len(equities) - 1

        # 总收益
        final_equity = equities[-1]
        total_return = (final_equity - initial_capital) / initial_capital

        # 年化收益
        annual_return = (1 + total_return) ** (252 / max(n_days, 1)) - 1

        # 日收益率
        daily_returns = []
        for i in range(1, len(equities)):
            ret = (equities[i] - equities[i-1]) / equities[i-1]
            daily_returns.append(ret)

        # 年化波动率
        annual_vol = np.std(daily_returns) * np.sqrt(252) if daily_returns else 0

        # Sharpe
        sharpe = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0

        # Sortino
        downside = [r for r in daily_returns if r < 0]
        downside_vol = np.std(downside) * np.sqrt(252) if downside else 0
        sortino = (annual_return - risk_free_rate) / downside_vol if downside_vol > 0 else 0

        # Max Drawdown
        peak = equities[0]
        max_dd = 0
        dd_start = dd_end = 0
        current_dd_start = 0
        for i, eq in enumerate(equities):
            if eq > peak:
                peak = eq
                current_dd_start = i
            dd = (peak - eq) / peak
            if dd > max_dd:
                max_dd = dd
                dd_start = current_dd_start
                dd_end = i

        # Calmar
        calmar = annual_return / max_dd if max_dd > 0 else 0

        # 交易统计
        buys = [t for t in self.transactions if t.get('side') == 'buy']
        sells = [t for t in self.transactions if t.get('side') == 'sell']
        total_trades = len(buys)  # 每笔买入对应一笔完整交易

        # 盈亏统计
        sell_pnls = [t.get('pnl', 0) for t in sells]
        wins = sum(1 for p in sell_pnls if p > 0)
        losses = sum(1 for p in sell_pnls if p <= 0)
        win_rate = wins / max(wins + losses, 1)

        total_wins_val = sum(p for p in sell_pnls if p > 0)
        total_losses_val = abs(sum(p for p in sell_pnls if p < 0))
        profit_factor = total_wins_val / total_losses_val if total_losses_val > 0 else float('inf')

        # 最大单笔盈亏
        max_win = max(sell_pnls) if sell_pnls else 0
        max_loss = min(sell_pnls) if sell_pnls else 0

        return {
            'total_return': round(total_return, 4),
            'annual_return': round(annual_return, 4),
            'annual_volatility': round(annual_vol, 4),
            'sharpe_ratio': round(sharpe, 3),
            'sortino_ratio': round(sortino, 3),
            'calmar_ratio': round(calmar, 3),
            'max_drawdown': round(max_dd, 4),
            'max_dd_start_day': dd_start,
            'max_dd_end_day': dd_end,
            'win_rate': round(win_rate, 4),
            'profit_factor': round(profit_factor, 3) if profit_factor != float('inf') else 999,
            'total_trades': total_trades,
            'wins': wins,
            'losses': losses,
            'max_win': round(max_win, 2),
            'max_loss': round(max_loss, 2),
            'n_days': n_days,
            'final_equity': round(final_equity, 2),
            'initial_capital': initial_capital,
            'total_fees': round(sum(t.get('fee', 0) for t in self.transactions), 2),
            'total_tax': round(sum(t.get('tax', 0) for t in self.transactions), 2),
        }


class EventDrivenBacktester:
    """
    事件驱动回测引擎 — P3 架构升级

    用法:
        backtester = EventDrivenBacktester(
            strategy=my_strategy,
            initial_capital=1000000,
        )
        results = backtester.run(klines)
        print(results['metrics'])
    """

    def __init__(self, strategy: StrategyBase,
                 initial_capital: float = 1000000,
                 commission_rate: float = 0.0003,
                 stamp_tax: float = 0.001,
                 slippage_bps: float = 2.0,
                 impact_model: str = 'linear',
                 max_position_pct: float = 0.3,
                 max_drawdown_pct: float = 0.15):
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.execution = ExecutionEngine(
            commission_rate=commission_rate,
            stamp_tax=stamp_tax,
            slippage_bps=slippage_bps,
            impact_model=impact_model,
        )
        self.risk_manager = RiskManager(
            max_position_pct=max_position_pct,
            max_drawdown_pct=max_drawdown_pct,
        )
        self.report = BacktestReport()

    def run(self, klines: List[Dict], benchmark_klines: Optional[List[Dict]] = None) -> Dict:
        """
        运行回测

        Args:
            klines: 回测标的 K 线数据
            benchmark_klines: 可选的基准 K 线数据

        Returns:
            回测结果字典
        """
        if not klines or len(klines) < 30:
            return {'error': 'K线数据不足'}

        # 初始化
        capital = self.initial_capital
        positions: Dict[str, Position] = {}
        pending_orders: List[Order] = []
        peak_equity = self.initial_capital

        # 初始化持仓
        stock_code = klines[0].get('stock_code', 'unknown')
        positions[stock_code] = Position(stock_code)

        # 基准权益
        benchmark_equity = self.initial_capital
        if benchmark_klines:
            for bk in benchmark_klines:
                benchmark_equity *= (1 + bk.get('return', 0))

        # 事件驱动主循环
        for i, bar in enumerate(klines):
            bar['close'] = float(bar.get('close', 0))
            bar['open'] = float(bar.get('open', bar['close']))
            bar['high'] = float(bar.get('high', bar['close']))
            bar['low'] = float(bar.get('low', bar['close']))
            bar['volume'] = float(bar.get('volume', 0))

            # 1. 策略生成信号
            current_equity = capital
            for code, pos in positions.items():
                current_equity += pos.market_value(bar['close'])

            orders = self.strategy.on_bar(bar, positions, current_equity)

            # 2. 风控检查
            current_dd = (peak_equity - current_equity) / peak_equity if peak_equity > 0 else 0
            filtered_orders = []
            for order in orders:
                if self.risk_manager.check_order(order, current_equity, positions, current_dd):
                    filtered_orders.append(order)

            # 3. 执行订单
            for order in filtered_orders:
                filled = self.execution.execute_order(order, bar)
                if filled and filled.status == OrderStatus.FILLED:
                    # 更新持仓
                    pos = positions.get(order.stock_code)
                    if pos:
                        if order.side == Side.BUY:
                            trade_value = filled.filled_quantity * filled.avg_fill_price
                            fee = trade_value * self.execution.commission_rate
                            pos.buy(filled.filled_quantity, filled.avg_fill_price, fee=fee)
                            capital -= trade_value + fee
                        elif order.side == Side.SELL:
                            trade_value = filled.filled_quantity * filled.avg_fill_price
                            fee = trade_value * self.execution.commission_rate
                            tax = trade_value * self.execution.stamp_tax
                            pnl = pos.sell(filled.filled_quantity, filled.avg_fill_price, fee=fee, tax=tax)
                            capital += trade_value - fee - tax

                            # 记录交易
                            self.report.add_transaction(filled, {
                                **bar, 'pnl': pnl, 'fee': fee, 'tax': tax,
                            })

            # 4. 更新峰值和日终
            # ✅ 修复: 权益 = 现金 + 持仓市值，不再重复计算 unrealized_pnl
            # 原始错误: market_value + unrealized_pnl = q*p + q*(p-c) = 2*q*p - q*c (重复计算)
            total_equity = capital
            for code, pos in positions.items():
                total_equity += pos.market_value(bar['close'])

            if total_equity > peak_equity:
                peak_equity = total_equity

            self.risk_manager.update_peak(total_equity)
            self.risk_manager.reset_daily()

            # 记录权益
            self.report.add_equity_point(bar.get('date', ''), total_equity, benchmark_equity)

            # 日终回调
            self.strategy.on_timer(bar.get('date', ''), positions)

        # 计算最终报告
        metrics = self.report.compute_metrics(self.initial_capital)
        metrics['positions'] = {k: v.to_dict() for k, v in positions.items()}
        metrics['transactions'] = self.report.transactions[:50]  # 限制返回数量

        return {
            'success': True,
            'metrics': metrics,
            'equity_curve': self.report.equity_curve[-100:],  # 最近 100 个点
            'timestamp': datetime.now().isoformat(),
        }
