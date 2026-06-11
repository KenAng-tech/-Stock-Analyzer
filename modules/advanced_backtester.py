#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
高级回测框架 — Advanced Backtester (A-share 增强版)

功能:
- A股交易约束: T+1 / 100股整数倍 / 涨跌停 / 停牌
- 精确交易成本模型 (佣金 + 印花税 + 滑点 + 市场冲击)
- Walk-Forward 滚动窗口回测
- Bootstrap 置信区间
- 完整绩效指标报告 (含盈利因子 / 收益痛苦比 / 持仓周期胜率)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from modules.logger import logger


class TransactionCostModel:
    """A股交易成本模型 (含市场冲击成本)"""

    def __init__(self, commission: float = 0.0003,
                 stamp_tax: float = 0.001,
                 min_commission: float = 5.0,
                 slippage: float = 0.001,
                 market_impact_base: float = 0.0001,
                 market_impact_coeff: float = 0.5,
                 volatility_factor: float = 0.3):
        """
        Args:
            commission: 佣金费率 (万三)
            stamp_tax: 印花税 (千一, 仅卖出)
            min_commission: 最低佣金 (元), A股规定最低 5 元
            slippage: 基础滑点
            market_impact_base: 基础市场冲击系数
            market_impact_coeff: 成交量占比系数 (Amihud 简化模型)
            volatility_factor: 波动率放大系数 (高波动日冲击更大)
        """
        self.commission = commission
        self.stamp_tax = stamp_tax
        self.min_commission = min_commission
        self.slippage = slippage
        self.market_impact_base = market_impact_base
        self.market_impact_coeff = market_impact_coeff
        self.volatility_factor = volatility_factor

    def calculate_buy(self, price: float, volume: float,
                      avg_daily_volume: float = 1e9,
                      volatility: float = 1.0) -> Dict:
        """
        买入总成本 (含市场冲击)

        Args:
            price: 成交价
            volume: 股数
            avg_daily_volume: 日均成交量 (股)
            volatility: 当日波动率因子 (1.0 = 正常, >1.0 = 高波动)
        """
        base_cost = price * volume
        commission = base_cost * self.commission
        commission = max(commission, self.min_commission)  # 最低 5 元
        slippage_cost = base_cost * self.slippage
        market_impact = self._calc_market_impact(
            volume, avg_daily_volume, volatility
        ) * base_cost
        return {
            'base_cost': base_cost,
            'commission': commission,
            'slippage_cost': slippage_cost,
            'market_impact_cost': market_impact,
            'total_cost': commission + slippage_cost + market_impact,
            'total': base_cost + commission + slippage_cost + market_impact,
        }

    def calculate_sell(self, price: float, volume: float,
                       avg_daily_volume: float = 1e9,
                       volatility: float = 1.0) -> Dict:
        """
        卖出总成本 (含印花税 + 市场冲击)

        Args:
            price: 成交价
            volume: 股数
            avg_daily_volume: 日均成交量 (股)
            volatility: 当日波动率因子 (1.0 = 正常, >1.0 = 高波动)
        """
        base_cost = price * volume
        commission = base_cost * self.commission
        commission = max(commission, self.min_commission)  # 最低 5 元
        stamp = base_cost * self.stamp_tax
        slippage_cost = base_cost * self.slippage
        market_impact = self._calc_market_impact(
            volume, avg_daily_volume, volatility
        ) * base_cost
        return {
            'base_cost': base_cost,
            'commission': commission,
            'stamp_tax': stamp,
            'slippage_cost': slippage_cost,
            'market_impact_cost': market_impact,
            'total_cost': commission + stamp + slippage_cost + market_impact,
            'total': base_cost - (commission + stamp + slippage_cost + market_impact),
        }

    def _calc_market_impact(self, volume: float, avg_daily_volume: float,
                            volatility: float = 1.0) -> float:
        """
        计算市场冲击成本 (Amihud 简化模型 + 波动率放大)
        冲击 = (base_coeff + coeff * volume_ratio) * volatility_factor
        高波动日 + 大单 = 显著更高的冲击成本
        """
        ratio = volume / (avg_daily_volume + 1)
        base_impact = self.market_impact_base + self.market_impact_coeff * ratio
        # 波动率放大: 涨跌停日流动性枯竭, 冲击成本倍增
        return base_impact * volatility

    def adaptive_slippage(self, volume: float, avg_daily_volume: float,
                          volatility: float = 1.0) -> float:
        """
        自适应滑点: 成交量越大 + 波动越高, 滑点越高
        涨跌停板时滑点趋近于零 (无法成交)
        """
        ratio = volume / (avg_daily_volume + 1)
        return self.slippage * (1 + ratio) * volatility


class AShareConstraints:
    """A股交易约束"""

    # 板块前缀 -> 涨跌停限制
    BOARD_LIMITS = {
        'sh68': 0.20,   # 科创板 (STAR)
        'sz30': 0.20,   # 创业板 (ChiNext)
        'sz00': 0.10,   # 深市主板
        'sh60': 0.10,   # 沪市主板
        'sh688': 0.20,  # 科创板 (完整前缀)
    }
    DEFAULT_LIMIT = 0.10
    LOT_SIZE = 100  # 最小交易单位 (手)
    T_PLUS_1 = True  # T+1 交割制度

    @classmethod
    def get_board_limit(cls, stock_code: str) -> float:
        """根据股票代码判断涨跌停限制"""
        code = stock_code.lower().lstrip('sh').lstrip('sz')
        prefix = stock_code[:4]
        for p, limit in cls.BOARD_LIMITS.items():
            if stock_code.startswith(p):
                return limit
        return cls.DEFAULT_LIMIT

    @classmethod
    def is_st_stock(cls, stock_code: str, name: str = '') -> bool:
        """判断是否为 ST/*ST 股票 (涨跌停 5%)"""
        if 'st' in name.lower() or 'ST' in name:
            return True
        if stock_code.startswith(('st', 'ST')):
            return True
        return False

    @classmethod
    def get_effective_limit(cls, stock_code: str, name: str = '') -> float:
        """获取实际涨跌停限制 (ST 股为 5%)"""
        if cls.is_st_stock(stock_code, name):
            return 0.05
        return cls.get_board_limit(stock_code)

    @classmethod
    def enforce_lot_size(cls, shares: int) -> int:
        """将股数向下取整到 100 的整数倍"""
        return (shares // cls.LOT_SIZE) * cls.LOT_SIZE

    @classmethod
    def check_price_limit(cls, current_price: float, prev_close: float,
                          stock_code: str, name: str = '') -> Dict:
        """
        检查价格是否触及涨跌停限制

        Returns:
            {'valid': bool, 'reason': str, 'limit_type': str}
            valid=True 表示价格有效, 否则说明原因
        """
        if prev_close <= 0:
            return {'valid': True, 'reason': '', 'limit_type': ''}

        limit_pct = cls.get_effective_limit(stock_code, name)
        upper_limit = prev_close * (1 + limit_pct)
        lower_limit = prev_close * (1 - limit_pct)
        # 浮点精度容忍 (1 分钱)
        eps = 0.01

        if current_price >= upper_limit - eps:
            return {
                'valid': False,
                'reason': f'涨停 {upper_limit:.3f}',
                'limit_type': 'upper',
            }
        if current_price <= lower_limit + eps:
            return {
                'valid': False,
                'reason': f'跌停 {lower_limit:.3f}',
                'limit_type': 'lower',
            }
        return {'valid': True, 'reason': '', 'limit_type': ''}

    @classmethod
    def can_sell_t1(cls, stock_code: str, current_date: str,
                    today_bought: Dict[str, int],
                    positions: Dict[str, int]) -> bool:
        """
        T+1 检查: 今日买入的份额不可卖出

        Args:
            today_bought: 今日买入的 {stock: shares} 字典
            positions: 当前持仓 {stock: shares}

        Returns:
            True 如果可以卖出
        """
        if not cls.T_PLUS_1:
            return True
        today_shares = today_bought.get(stock_code, 0)
        total = positions.get(stock_code, 0)
        # 可卖 = 总持仓 - 今日买入
        return (total - today_shares) > 0

    @classmethod
    def get_sellable_shares(cls, stock_code: str, positions: Dict[str, int],
                            today_bought: Dict[str, int]) -> int:
        """获取 T+1 后可卖的股数"""
        if not cls.T_PLUS_1:
            return positions.get(stock_code, 0)
        today_shares = today_bought.get(stock_code, 0)
        return max(0, positions.get(stock_code, 0) - today_shares)


class BacktestResult:
    """回测结果"""

    def __init__(self):
        self.equity_curve: List[Tuple[str, float]] = []  # [(date, nav), ...]
        self.positions: Dict[str, Dict] = {}  # {date: {stock: shares}}
        self.trades: List[Dict] = []  # 交易记录
        self.metrics: Dict = {}

    def calculate_metrics(self, risk_free: float = 0.02) -> Dict:
        """计算绩效指标"""
        if not self.equity_curve:
            return {}

        navs = np.array([nav for _, nav in self.equity_curve])
        initial_cap = navs[0]
        final_cap = navs[-1]

        # 总收益
        total_return = (final_cap - initial_cap) / initial_cap

        # 日收益率
        daily_returns = np.diff(navs) / (navs[:-1] + 1e-10)
        daily_returns = daily_returns[daily_returns != 0]

        n_days = max(len(daily_returns), 1)

        # 年化收益
        ann_return = (1 + total_return) ** (252 / max(n_days, 1)) - 1

        # 年化波动率
        ann_vol = float(np.std(daily_returns) * np.sqrt(252)) if len(daily_returns) > 1 else 0

        # 夏普比率
        sharpe = (ann_return - risk_free) / (ann_vol + 1e-10)

        # Sortino 比率 (只考虑下行波动)
        downside = daily_returns[daily_returns < 0]
        downside_vol = float(np.std(downside) * np.sqrt(252)) if len(downside) > 0 else 0
        sortino = (ann_return - risk_free) / (downside_vol + 1e-10)

        # 最大回撤
        running_max = np.maximum.accumulate(navs)
        drawdowns = (navs - running_max) / (running_max + 1e-10)
        max_dd = float(np.min(drawdowns))

        # 最大回撤持续天数
        dd_duration = 0
        max_dd_duration = 0
        for dd in drawdowns:
            if dd < 0:
                dd_duration += 1
                max_dd_duration = max(max_dd_duration, dd_duration)
            else:
                dd_duration = 0

        # 胜率
        win_days = np.sum(daily_returns > 0)
        win_rate = float(win_days / max(len(daily_returns), 1))

        # 盈亏比
        avg_win = float(np.mean(daily_returns[daily_returns > 0])) if np.any(daily_returns > 0) else 0
        avg_loss = float(np.mean(daily_returns[daily_returns < 0])) if np.any(daily_returns < 0) else 0
        profit_loss_ratio = abs(avg_win / (avg_loss + 1e-10))

        # 月度收益
        monthly_returns = {}
        for date, nav in self.equity_curve:
            if date and len(date) >= 7:
                month = date[:7]  # YYYY-MM
                if month not in monthly_returns:
                    monthly_returns[month] = []
                monthly_returns[month].append(nav)

        # Calmar 比率
        calmar = ann_return / (abs(max_dd) + 1e-10)

        # --- 新增指标 ---

        # 盈利因子 (Profit Factor) = 总盈利 / 总亏损 (绝对值)
        gross_profit = float(np.sum(daily_returns[daily_returns > 0]))
        gross_loss = float(np.sum(daily_returns[daily_returns < 0]))
        profit_factor = abs(gross_profit / (gross_loss + 1e-10))

        # 收益痛苦比率 (Gain-to-Pain Ratio)
        # 年化收益 / 累计回撤面积 (每日回撤的累加)
        cumulative_dd = np.sum(np.abs(drawdowns))
        gain_to_pain = ann_return / (cumulative_dd + 1e-10)

        # 按持仓周期分类的胜率
        holding_win_rate = self._calc_holding_win_rate()

        # 总交易成本
        total_commission = sum(t.get('cost', 0) for t in self.trades)

        self.metrics = {
            'total_return': round(total_return, 4),
            'annual_return': round(ann_return, 4),
            'volatility': round(ann_vol, 4),
            'sharpe_ratio': round(sharpe, 4),
            'sortino_ratio': round(sortino, 4),
            'calmar_ratio': round(calmar, 4),
            'max_drawdown': round(max_dd, 4),
            'max_drawdown_duration': max_dd_duration,
            'win_rate': round(win_rate, 4),
            'profit_loss_ratio': round(profit_loss_ratio, 4),
            'profit_factor': round(profit_factor, 4),
            'gain_to_pain_ratio': round(gain_to_pain, 4),
            'holding_win_rate': holding_win_rate,
            'n_days': n_days,
            'n_trades': len(self.trades),
            'total_commission': round(total_commission, 2),
            'monthly_returns': {k: round((v[-1] - v[0]) / v[0], 4) for k, v in monthly_returns.items()},
        }

        return self.metrics

    def _calc_holding_win_rate(self) -> Dict:
        """
        按持仓周期分类计算胜率
        将交易按持仓天数分为: 超短线(<5天), 短线(5-20天), 中线(20-60天), 长线(>60天)
        """
        if not self.trades:
            return {}

        # 配对买卖交易
        opens = {}  # {stock: (buy_date, volume)}
        completed = []  # [(holding_days, profit_pct), ...]

        for trade in self.trades:
            stock = trade['stock']
            direction = trade['direction']
            date = trade['date']

            if direction == 'buy':
                opens[stock] = (date, trade['volume'])
            elif direction == 'sell' and stock in opens:
                buy_date, _ = opens.pop(stock)
                buy_price = trade.get('_buy_price', 0)
                sell_price = trade['price']
                if buy_price > 0:
                    holding_days = self._days_between(buy_date, date)
                    profit_pct = (sell_price - buy_price) / buy_price
                    completed.append((holding_days, profit_pct))

        if not completed:
            return {}

        buckets = {
            '超短线(<5天)': [],
            '短线(5-20天)': [],
            '中线(20-60天)': [],
            '长线(>60天)': [],
        }

        for days, pnl in completed:
            if days < 5:
                bucket = '超短线(<5天)'
            elif days < 20:
                bucket = '短线(5-20天)'
            elif days < 60:
                bucket = '中线(20-60天)'
            else:
                bucket = '长线(>60天)'
            buckets[bucket].append(pnl)

        result = {}
        for name, pnls in buckets.items():
            if not pnls:
                continue
            wins = sum(1 for p in pnls if p > 0)
            result[name] = {
                'count': len(pnls),
                'win_rate': round(wins / len(pnls), 4),
                'avg_pnl': round(float(np.mean(pnls)), 4),
                'best_pnl': round(float(np.max(pnls)), 4),
                'worst_pnl': round(float(np.min(pnls)), 4),
            }

        return result

    @staticmethod
    def _days_between(date1: str, date2: str) -> int:
        """计算两个日期字符串之间的交易日差值 (简化: 日历日)"""
        fmt = '%Y-%m-%d'
        try:
            d1 = datetime.strptime(date1, fmt)
            d2 = datetime.strptime(date2, fmt)
            return max(abs((d2 - d1).days, 1))
        except (ValueError, TypeError):
            return 1


class BacktestEngine:
    """A股回测引擎 (含 T+1 / 涨跌停 / 停牌 约束)"""

    def __init__(self, initial_capital: float = 1000000,
                 cost_model: Optional[TransactionCostModel] = None,
                 constraints: Optional[AShareConstraints] = None):
        self.initial_capital = initial_capital
        self.cost_model = cost_model or TransactionCostModel()
        self.constraints = constraints or AShareConstraints()

    def run(self, signals: Dict[str, Dict],
            prices: Dict[str, Dict],
            dates: Optional[List[str]] = None,
            volumes: Optional[Dict[str, Dict[str, float]]] = None,
            prev_closes: Optional[Dict[str, Dict[str, float]]] = None,
            halt_flags: Optional[Dict[str, Dict[str, bool]]] = None,
            stock_names: Optional[Dict[str, str]] = None) -> BacktestResult:
        """
        执行回测 (A-share 增强版)

        Args:
            signals: {date: {stock_code: {direction, confidence}}}
            prices: {date: {stock_code: close_price}}
            dates: 日期列表 (可选)
            volumes: {date: {stock_code: volume}} 成交量数据 (可选)
            prev_closes: {date: {stock_code: prev_close_price}} 前一日收盘价
            halt_flags: {date: {stock_code: True}} 停牌标记 (可选)
            stock_names: {stock_code: name} 股票名称 (用于 ST 判断)

        Returns:
            BacktestResult
        """
        result = BacktestResult()
        cash = self.initial_capital
        positions = {}        # {stock_code: total_shares}
        buy_prices = {}       # {stock_code: avg_buy_price} (for win-rate tracking)
        today_bought = {}     # {stock_code: shares_bought_today} (T+1)
        stock_names = stock_names or {}

        if dates is None:
            dates = sorted(signals.keys())

        for date in dates:
            day_signals = signals.get(date, {})
            day_prices = prices.get(date, {})
            day_volumes = volumes.get(date, {}) if volumes else {}
            day_halt = halt_flags.get(date, {}) if halt_flags else {}
            day_prev_close = prev_closes.get(date, {}) if prev_closes else {}

            # 重置今日买入计数
            today_bought = {}

            # 执行交易
            for stock, sig in day_signals.items():
                price = day_prices.get(stock, 0)
                if price <= 0:
                    continue

                direction = sig.get('direction', 'hold')
                if direction == 'hold':
                    continue

                # --- 1. 停牌检查 ---
                if day_halt.get(stock, False):
                    logger.debug(f"{date} {stock} 停牌, 跳过")
                    continue

                # --- 2. 涨跌停检查 ---
                prev_close = day_prev_close.get(stock, 0)
                price_check = self.constraints.check_price_limit(
                    price, prev_close, stock, stock_names.get(stock, '')
                )
                if not price_check['valid']:
                    logger.debug(
                        f"{date} {stock} {price_check['reason']}, 跳过"
                    )
                    continue

                # --- 3. 计算仓位大小 (最多用 10% 现金) ---
                target_value = cash * 0.1

                if direction == 'buy' and target_value > price:
                    shares = int(target_value / price)
                    # 4. 100 股整数倍约束
                    shares = self.constraints.enforce_lot_size(shares)
                    if shares < self.constraints.LOT_SIZE:
                        continue  # 不足一手, 跳过

                    # 获取成交量用于成本计算
                    avg_vol = day_volumes.get(stock, 1e9)
                    # 波动率因子: 价格相对前收的涨跌幅绝对值
                    vol_factor = 1.0
                    if prev_close > 0:
                        vol_factor = max(1.0, abs(price - prev_close) / prev_close * 20)

                    cost = self.cost_model.calculate_buy(
                        price, shares, avg_vol, vol_factor
                    )
                    if cost['total'] <= cash:
                        cash -= cost['total']
                        old_shares = positions.get(stock, 0)
                        old_price = buy_prices.get(stock, price)
                        positions[stock] = old_shares + shares
                        # 更新平均买入价
                        buy_prices[stock] = (
                            (old_price * old_shares + price * shares)
                            / (old_shares + shares)
                        )
                        today_bought[stock] = today_bought.get(stock, 0) + shares
                        result.trades.append({
                            'date': date, 'stock': stock,
                            'direction': 'buy', 'volume': shares,
                            'price': price, 'cost': cost['total_cost'],
                        })

                elif direction == 'sell' and stock in positions:
                    # 5. T+1 约束: 只能卖出非今日买入的份额
                    sellable = self.constraints.get_sellable_shares(
                        stock, positions, today_bought
                    )
                    if sellable <= 0:
                        continue  # 今日买入, 不可卖出

                    shares = min(positions[stock], sellable)
                    avg_vol = day_volumes.get(stock, 1e9)
                    vol_factor = 1.0
                    if prev_close > 0:
                        vol_factor = max(1.0, abs(price - prev_close) / prev_close * 20)

                    revenue = self.cost_model.calculate_sell(
                        price, shares, avg_vol, vol_factor
                    )
                    cash += revenue['total']
                    result.trades.append({
                        'date': date, 'stock': stock,
                        'direction': 'sell', 'volume': shares,
                        'price': price, 'cost': revenue['total_cost'],
                        '_buy_price': buy_prices.get(stock, price),
                    })
                    positions[stock] -= shares
                    if positions[stock] <= 0:
                        del positions[stock]
                        buy_prices.pop(stock, None)

            # 计算当日净值
            portfolio_value = cash
            for stock, shares in positions.items():
                p = day_prices.get(stock, 0)
                portfolio_value += shares * p

            result.equity_curve.append((date, portfolio_value))
            result.positions[date] = dict(positions)

        result.calculate_metrics()
        return result


class WalkForwardBacktester:
    """Walk-Forward 回测"""

    def __init__(self, train_window: int = 120,
                 test_window: int = 20, step: int = 20):
        self.train_window = train_window
        self.test_window = test_window
        self.step = step
        self.results: List[BacktestResult] = []

    def run(self, all_dates: List[str],
            signals: Dict, prices: Dict,
            volumes: Optional[Dict] = None,
            prev_closes: Optional[Dict] = None,
            halt_flags: Optional[Dict] = None,
            stock_names: Optional[Dict] = None) -> List[BacktestResult]:
        """
        滚动窗口回测 (A-share 增强版)

        Args:
            all_dates: 所有日期
            signals: 信号数据
            prices: 价格数据
            volumes: 成交量数据 (可选)
            prev_closes: 前收盘价 (可选)
            halt_flags: 停牌标记 (可选)
            stock_names: 股票名称 (可选)

        Returns:
            各窗口回测结果列表
        """
        self.results = []

        for i in range(0, len(all_dates) - self.train_window - self.test_window, self.step):
            train_end = i + self.train_window
            test_end = train_end + self.test_window

            train_dates = all_dates[i:train_end]
            test_dates = all_dates[train_end:test_end]

            # 在训练窗口上"训练" (这里简化为直接回测测试窗口)
            train_signals = {d: signals.get(d, {}) for d in train_dates}
            test_signals = {d: signals.get(d, {}) for d in test_dates}
            test_prices = {d: prices.get(d, {}) for d in test_dates}
            test_volumes = {d: volumes.get(d, {}) for d in test_dates} if volumes else None
            test_prev_closes = {d: prev_closes.get(d, {}) for d in test_dates} if prev_closes else None
            test_halt_flags = {d: halt_flags.get(d, {}) for d in test_dates} if halt_flags else None

            engine = BacktestEngine()
            result = engine.run(
                test_signals, test_prices, test_dates,
                volumes=test_volumes,
                prev_closes=test_prev_closes,
                halt_flags=test_halt_flags,
                stock_names=stock_names,
            )
            self.results.append(result)

        return self.results


class BootstrapAnalyzer:
    """Bootstrap 置信区间分析"""

    def __init__(self, returns: np.ndarray, n_simulations: int = 1000):
        self.returns = returns
        self.n_simulations = n_simulations

    def confidence_intervals(self, confidence: float = 0.95) -> Dict:
        """计算关键指标的 Bootstrap 置信区间"""
        bootstrap_annual_returns = []
        bootstrap_sharpes = []

        for _ in range(self.n_simulations):
            # 有放回抽样
            sample = np.random.choice(self.returns, size=len(self.returns), replace=True)
            ann_ret = (1 + np.mean(sample)) ** 252 - 1
            ann_vol = np.std(sample) * np.sqrt(252)
            sharpe = ann_ret / (ann_vol + 1e-10)

            bootstrap_annual_returns.append(ann_ret)
            bootstrap_sharpes.append(sharpe)

        alpha = (1 - confidence) / 2
        lower = int(alpha * self.n_simulations)
        upper = int((1 - alpha) * self.n_simulations)

        return {
            'annual_return': {
                'mean': round(float(np.mean(bootstrap_annual_returns)), 4),
                'ci_low': round(float(np.sort(bootstrap_annual_returns)[lower]), 4),
                'ci_high': round(float(np.sort(bootstrap_annual_returns)[upper]), 4),
            },
            'sharpe_ratio': {
                'mean': round(float(np.mean(bootstrap_sharpes)), 4),
                'ci_low': round(float(np.sort(bootstrap_sharpes)[lower]), 4),
                'ci_high': round(float(np.sort(bootstrap_sharpes)[upper]), 4),
            },
        }

    @staticmethod
    def kelly_fraction(sharpe: float, win_rate: float) -> float:
        """计算最优 Kelly 仓位比例"""
        if sharpe <= 0:
            return 0.0
        # Kelly = (p*b - q) / b, 简化版: Kelly = win_rate - (1-win_rate)/sharpe
        b = sharpe
        p = win_rate
        q = 1 - p
        kelly = (p * b - q) / b
        # 半 Kelly (更保守)
        return max(0, kelly / 2)


# 全局实例
backtest_engine = BacktestEngine()
