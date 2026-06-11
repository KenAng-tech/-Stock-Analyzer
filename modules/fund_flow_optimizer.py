"""
资金流入深度优化模块
"""

import time
from typing import Dict


class FundFlowOptimizer:
    """资金流入优化器"""
    
    def __init__(self):
        # 交易等级定义
        self.trade_levels = {
            '超大单': {'min_amount': 1000000, 'label': '>100万'},
            '大单': {'min_amount': 500000, 'label': '50-100万'},
            '中单': {'min_amount': 100000, 'label': '10-50万'},
            '小单': {'min_amount': 0, 'label': '<10万'}
        }
        
        # 时间戳（用于计算流入速度）
        self.last_update_time = time.time()
    
    def estimate_trade_count(self, volume: int, amount: float, price: float = 0) -> Dict:
        """
        估算交易笔数

        基于成交额和均价估算。A 股成交量以"手"为单位（1手=100股）。
        估算逻辑: 交易笔数 ≈ 成交额 / 平均每笔成交额

        Args:
            volume: 成交量（手）
            amount: 成交额（万元）
            price: 最新价（用于估算均价，可选）

        Returns:
            估算结果
        """
        if volume == 0 or amount == 0:
            return {'trade_count': 0, 'avg_amount': 0, 'level': '无数据'}

        # 1手 = 100股，成交额单位是万元
        # 平均每手成交额（万元/手）
        avg_amount_per_lot = amount / volume if volume > 0 else 0

        # 估算交易笔数: 按每手约 1-3 笔交易估算
        # 高价股每手金额大 → 笔数少，低价股每手金额小 → 笔数多
        if price > 0 and price > 50:
            # 高价股: 平均每笔 2-3 手
            trades_per_lot = 0.4
        elif price > 0 and price > 20:
            # 中价股: 平均每笔 1.5-2 手
            trades_per_lot = 0.6
        else:
            # 低价股: 平均每笔 1-1.5 手
            trades_per_lot = 0.8

        estimated_trades = int(volume * trades_per_lot)

        # 根据 avg_amount_per_lot 判断单子类型
        # amount 是万元, volume 是手 → avg_amount_per_lot 单位: 万元/手
        # 即每手的成交额（万元）= 每股均价 × 100股 / 10000
        # 例如: 股价20元 → 每手2000元 → 0.2万元/手
        if avg_amount_per_lot > 5:  # 每手 > 5万元 → 高价股
            level = '高价股'
        elif avg_amount_per_lot > 1:  # 每手 > 1万元
            level = '中价股'
        else:
            level = '低价股/小单为主'

        return {
            'trade_count': estimated_trades,
            'avg_amount_per_lot': round(avg_amount_per_lot, 2),
            'level': level,
            'volume': volume,
            'amount': amount,
        }
    
    def analyze_trade_distribution(self, volume: int, amount: float, 
                                   outer: int, inner: int) -> Dict:
        """
        分析大中小单分布
        """
        if volume == 0:
            return {'distribution': {}, 'analysis': ''}
        
        avg_amount = amount / volume if volume > 0 else 0
        
        # 估算各等级占比
        if avg_amount > 10:  # 大单为主
            large_ratio = 60
            medium_ratio = 30
            small_ratio = 10
        elif avg_amount > 1:  # 中单为主
            large_ratio = 30
            medium_ratio = 50
            small_ratio = 20
        else:  # 小单为主
            large_ratio = 15
            medium_ratio = 35
            small_ratio = 50
        
        # 计算各等级成交量
        large_volume = int(volume * large_ratio / 100)
        medium_volume = int(volume * medium_ratio / 100)
        small_volume = volume - large_volume - medium_volume
        
        # 计算各等级成交额
        large_amount = amount * large_ratio / 100
        medium_amount = amount * medium_ratio / 100
        small_amount = amount * small_ratio / 100
        
        return {
            'distribution': {
                'large': {'volume': large_volume, 'amount': large_amount, 'ratio': large_ratio},
                'medium': {'volume': medium_volume, 'amount': medium_amount, 'ratio': medium_ratio},
                'small': {'volume': small_volume, 'amount': small_amount, 'ratio': small_ratio}
            },
            'avg_amount': round(avg_amount, 2),
            'analysis': '大单为主' if large_ratio > 40 else '中单为主' if medium_ratio > 40 else '小单为主'
        }
    
    def calculate_flow_speed(self, amount: float, minutes: int = 240) -> Dict:
        """
        计算资金流入速度
        amount: 成交额（万元）
        minutes: 交易时间（默认240分钟）
        """
        speed = amount / minutes if minutes > 0 else 0
        level = '高' if speed > 2000 else '中' if speed > 1000 else '低'
        
        return {
            'speed': round(speed, 2),
            'unit': '万元/分钟',
            'level': level,
            'total_amount': amount
        }
    
    def calculate_pressure_index(self, outer: int, inner: int, turnover: float) -> Dict:
        """
        计算买卖压力指数
        外内比 × 换手率 = 压力指数
        """
        ratio = outer / inner if inner > 0 else 1
        pressure_index = ratio * (turnover / 100)
        
        # 压力等级
        if pressure_index > 3:
            pressure_level = '强买压'
        elif pressure_index > 2:
            pressure_level = '中买压'
        elif pressure_index > 1:
            pressure_level = '弱买压'
        elif pressure_index > 0.7:
            pressure_level = '弱卖压'
        elif pressure_index > 0.5:
            pressure_level = '中卖压'
        else:
            pressure_level = '强卖压'
        
        return {
            'index': round(pressure_index, 2),
            'level': pressure_level,
            'ratio': round(ratio, 2),
            'turnover': turnover
        }
    
    def deep_fund_flow_analysis(self, stock_data: Dict) -> Dict:
        """
        深度资金流入分析
        """
        outer = stock_data.get('outer_disk', 0)
        inner = stock_data.get('inner_disk', 0)
        volume = stock_data.get('volume', 0)
        amount = stock_data.get('amount', 0)
        turnover = stock_data.get('turnover', 0)
        
        # 基础分析
        outer_inner_ratio = outer / inner if inner > 0 else 1
        main_flow = '主力净流入' if outer_inner_ratio > 1.1 else '主力净流出'
        
        # 估算交易笔数
        trade_info = self.estimate_trade_count(volume, amount)
        
        # 分析交易分布
        trade_distribution = self.analyze_trade_distribution(volume, amount, outer, inner)
        
        # 计算流入速度
        flow_speed = self.calculate_flow_speed(amount)
        
        # 计算压力指数
        pressure = self.calculate_pressure_index(outer, inner, turnover)
        
        # 筹码分布 - 基于价格位置的智能计算
        price = stock_data.get('price', 0)
        year_high = stock_data.get('year_high', 0)
        year_low = stock_data.get('year_low', 0)
        
        # 计算价格位置 (0-1)
        price_position = (price - year_low) / (year_high - year_low) if (year_high - year_low) > 0 else 0.5
        
        # 基于价格位置的筹码分布
        if price_position > 0.8:
            # 接近年高 - 获利盘少，套牢盘多
            profit_ratio = 30 + (turnover / 20)
            trapped_ratio = 100 - profit_ratio
        elif price_position < 0.2:
            # 接近年低 - 获利盘多，套牢盘少
            profit_ratio = 70 + (turnover / 10)
            trapped_ratio = 100 - profit_ratio
        else:
            # 中间位置
            profit_ratio = 50 + (turnover / 15)
            trapped_ratio = 100 - profit_ratio
        
        # 确保比例在合理范围内
        profit_ratio = max(10, min(90, profit_ratio))
        trapped_ratio = 100 - profit_ratio
        
        # 筹码分布解释
        if profit_ratio > 70:
            chip_interpretation = '获利盘充足，上涨动力强'
        elif profit_ratio > 50:
            chip_interpretation = '获利盘适中，走势平稳'
        else:
            chip_interpretation = '套牢盘较多，需关注解套压力'
        
        # 计算买/卖比率
        buy_sell_ratio = {
            'buy_percentage': round((outer / (outer + inner)) * 100, 1) if (outer + inner) > 0 else 50,
            'sell_percentage': round((inner / (outer + inner)) * 100, 1) if (outer + inner) > 0 else 50,
            'ratio': round(outer / inner, 2) if inner > 0 else 1.0,
            'dominant': '买盘主导' if outer > inner else '卖盘主导',
            'strength': '强' if abs(outer - inner) > (outer + inner) * 0.3 else '中' if abs(outer - inner) > (outer + inner) * 0.1 else '弱'
        }
        
        return {
            'main_flow': {
                'outer_disk': outer,
                'inner_disk': inner,
                'ratio': round(outer_inner_ratio, 2),
                'direction': main_flow,
                'buy_sell_ratio': buy_sell_ratio
            },
            'volume_analysis': {
                'volume': volume,
                'amount': amount,
                'turnover': turnover,
                'interpretation': '高换手率表明多空分歧大' if turnover > 200 else '正常换手'
            },
            'trade_count': trade_info,
            'trade_distribution': trade_distribution,
            'flow_speed': flow_speed,
            'pressure_index': pressure,
            # P2 修复: 合并重复的 chip_distribution key
            'chip_distribution': {
                'profit_ratio': round(profit_ratio, 1),
                'trapped_ratio': round(trapped_ratio, 1),
                'price_position': round(price_position, 2),
                'dense_zone': f'{price * 0.8:.0f}-{price * 1.1:.0f}元',
                'interpretation': chip_interpretation,
            }
        }


if __name__ == '__main__':
    optimizer = FundFlowOptimizer()
    print("✓ 资金流入优化器初始化完成")
    print("\n新增功能:")
    print("  1. 交易笔数估算")
    print("  2. 大中小单分布分析")
    print("  3. 资金流入速度计算")
    print("  4. 买卖压力指数")
    print("  5. 筹码分布优化")
