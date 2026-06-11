"""
K线信号分析器 - 高级增强版 V2
包含：K线形态、成交量、价格位置、RSI、多周期共振、形态匹配度、信号强度加权
"""

from typing import Dict, List
import math
import numpy as np


class KlineSignalAnalyzer:
    """K线信号分析器 - 高级增强版 V2"""
    
    def __init__(self):
        # 经典K线形态定义
        self.candlestick_patterns = {
            '锤子线': {'desc': '底部反转信号', 'condition': '下影线>2*实体，上影线很短', 'weight': 1.2},
            '倒锤子线': {'desc': '顶部反转信号', 'condition': '上影线>2*实体，下影线很短', 'weight': 1.1},
            '十字星': {'desc': '变盘信号', 'condition': '开盘价≈收盘价，实体很小', 'weight': 1.0},
            '孕线': {'desc': '趋势减弱信号', 'condition': '当前K线被前一根K线包裹', 'weight': 0.9},
            '吞没形态': {'desc': '强烈反转信号', 'condition': '当前K线完全吞没前一根', 'weight': 1.5},
            '启明星': {'desc': '底部反转', 'condition': '三K线组合，中间十字星', 'weight': 1.6},
            '暮星': {'desc': '顶部反转', 'condition': '三K线组合，中间十字星', 'weight': 1.6},
            '三只乌鸦': {'desc': '强烈看跌', 'condition': '连续三根阴线', 'weight': 1.5},
            '三白兵': {'desc': '强烈看涨', 'condition': '连续三根阳线', 'weight': 1.5},
        }
        
        # 成交量信号定义
        self.volume_signals = {
            '放量上涨': {'desc': '多头强势', 'condition': '量比>1.5且价格上涨', 'weight': 1.3},
            '放量下跌': {'desc': '空头强势', 'condition': '量比>1.5且价格下跌', 'weight': 1.3},
            '缩量上涨': {'desc': '上涨乏力', 'condition': '量比<0.8且价格上涨', 'weight': 0.8},
            '缩量下跌': {'desc': '下跌动能减弱', 'condition': '量比<0.8且价格下跌', 'weight': 0.8},
            '天量天价': {'desc': '可能见顶', 'condition': '成交量创近期新高', 'weight': 1.4},
            '地量地价': {'desc': '可能见底', 'condition': '成交量创近期新低', 'weight': 1.4},
        }
        
        # 价格位置信号
        self.price_position_signals = {
            '接近年高': {'desc': '突破在即', 'condition': '价格>年高*0.95', 'weight': 1.1},
            '接近年低': {'desc': '支撑强劲', 'condition': '价格<年低*1.05', 'weight': 1.1},
            '突破平台': {'desc': '趋势启动', 'condition': '价格突破近期高点', 'weight': 1.3},
            '回踩支撑': {'desc': '买入机会', 'condition': '价格回踩支撑位后反弹', 'weight': 1.2},
        }
        
        # 多周期参数 - 增强版
        self.periods = {
            '日线': {'weight': 1.0, 'rsi_period': 14, 'trend_threshold': 0.02},
            '周线': {'weight': 0.8, 'rsi_period': 14, 'trend_threshold': 0.05},
            '月线': {'weight': 0.6, 'rsi_period': 14, 'trend_threshold': 0.10},
        }
        
        # 信号强度权重
        self.signal_strength_weights = {
            'strong': 1.3,
            'medium': 1.0,
            'weak': 0.7
        }
    
    def analyze_candlestick_patterns(self, stock_data: Dict) -> Dict:
        """分析K线形态"""
        price = stock_data.get('price', 0)
        open_price = stock_data.get('open', 0)
        high = stock_data.get('high', 0)
        low = stock_data.get('low', 0)
        close = stock_data.get('close', 0)
        change_pct = stock_data.get('change_pct', 0)
        
        # 实体大小
        body_size = abs(close - open_price)
        # 上下影线
        upper_shadow = high - max(open_price, close)
        lower_shadow = min(open_price, close) - low
        # 总振幅
        total_range = high - low
        
        # 判断K线形态
        patterns = []
        
        # 锤子线
        if lower_shadow > 2 * body_size and upper_shadow < body_size * 0.5:
            patterns.append('锤子线')
        
        # 倒锤子线
        if upper_shadow > 2 * body_size and lower_shadow < body_size * 0.5:
            patterns.append('倒锤子线')
        
        # 十字星
        if body_size < total_range * 0.1:
            patterns.append('十字星')
        
        # 大阳线
        if change_pct > 5 and body_size > total_range * 0.7:
            patterns.append('大阳线')
        
        # 大阴线
        if change_pct < -5 and body_size > total_range * 0.7:
            patterns.append('大阴线')
        
        # 长上影线
        if upper_shadow > body_size * 2:
            patterns.append('长上影线')
        
        # 长下影线
        if lower_shadow > body_size * 2:
            patterns.append('长下影线')
        
        # 实体占比
        body_ratio = (body_size / total_range * 100) if total_range > 0 else 0
        
        # 形态匹配度评分 (0-100)
        pattern_score = self._calculate_pattern_score(patterns, body_ratio)
        
        return {
            'patterns': patterns,
            'body_size': body_size,
            'upper_shadow': upper_shadow,
            'lower_shadow': lower_shadow,
            'total_range': total_range,
            'body_ratio': body_ratio,
            'pattern_score': pattern_score,
            'pattern_strength': '强' if pattern_score > 70 else '中' if pattern_score > 40 else '弱'
        }
    
    def _calculate_pattern_score(self, patterns: List[str], body_ratio: float) -> float:
        """计算形态匹配度评分"""
        base_score = 50
        
        # 形态加分
        pattern_bonus = {
            '锤子线': 15, '倒锤子线': 15, '十字星': 10,
            '大阳线': 20, '大阴线': 20,
            '长上影线': 10, '长下影线': 10
        }
        
        for pattern in patterns:
            base_score += pattern_bonus.get(pattern, 5)
        
        # 实体大小加分
        if body_ratio > 70:
            base_score += 10
        elif body_ratio < 20:
            base_score += 5
        
        # 限制在0-100
        return min(100, max(0, base_score))
    
    def analyze_volume_signals(self, stock_data: Dict) -> Dict:
        """分析成交量信号 - 增强版"""
        volume = stock_data.get('volume', 0)
        turnover = stock_data.get('turnover', 0)
        change_pct = stock_data.get('change_pct', 0)
        amount = stock_data.get('amount', 0)
        
        signals = []
        
        # 高换手率
        if turnover > 200:
            signals.append('高换手率')
        elif turnover > 100:
            signals.append('中等换手率')
        else:
            signals.append('低换手率')
        
        # 放量/缩量判断
        if change_pct > 0:
            if turnover > 150:
                signals.append('放量上涨')
            elif turnover < 80:
                signals.append('缩量上涨')
        else:
            if turnover > 150:
                signals.append('放量下跌')
            elif turnover < 80:
                signals.append('缩量下跌')
        
        # 外盘内盘分析
        outer = stock_data.get('outer_disk', 0)
        inner = stock_data.get('inner_disk', 0)
        ratio = outer / inner if inner > 0 else 1
        
        if ratio > 1.3:
            signals.append('买盘强劲')
        elif ratio < 0.7:
            signals.append('卖盘强劲')
        else:
            signals.append('买卖均衡')
        
        # 成交量信号评分
        volume_score = self._calculate_volume_score(signals, turnover)
        
        return {
            'signals': signals,
            'turnover': turnover,
            'volume': volume,
            'amount': amount,
            'outer_inner_ratio': ratio,
            'buy_pressure': '强' if ratio > 1.1 else '弱' if ratio < 0.9 else '均衡',
            'volume_score': volume_score,
            'volume_strength': '强' if volume_score > 70 else '中' if volume_score > 40 else '弱'
        }
    
    def _calculate_volume_score(self, signals: List[str], turnover: float) -> float:
        """计算成交量信号评分"""
        base_score = 50
        
        signal_bonus = {
            '放量上涨': 15, '放量下跌': 15,
            '缩量上涨': 10, '缩量下跌': 10,
            '买盘强劲': 15, '卖盘强劲': 15,
            '高换手率': 10, '低换手率': 5
        }
        
        for signal in signals:
            base_score += signal_bonus.get(signal, 5)
        
        return min(100, max(0, base_score))
    
    def analyze_price_position(self, stock_data: Dict) -> Dict:
        """分析价格位置"""
        price = stock_data.get('price', 0)
        year_high = stock_data.get('year_high', 0)
        year_low = stock_data.get('year_low', 0)
        open_price = stock_data.get('open', 0)
        
        # 距年高低百分比
        dist_to_high = ((price - year_high) / year_high * 100) if year_high > 0 else 0
        dist_to_low = ((price - year_low) / year_low * 100) if year_low > 0 else 0
        
        # 位置判断
        if dist_to_high > -5:
            position = '接近年高'
        elif dist_to_low < 5:
            position = '接近年低'
        else:
            position = '中间位置'
        
        # 开盘位置
        if open_price > price * 1.02:
            open_position = '高开'
        elif open_price < price * 0.98:
            open_position = '低开'
        else:
            open_position = '平开'
        
        # 位置评分
        position_score = self._calculate_position_score(dist_to_high, dist_to_low, open_position)
        
        return {
            'position': position,
            'dist_to_high': dist_to_high,
            'dist_to_low': dist_to_low,
            'open_position': open_position,
            'in_upper_20_percent': dist_to_high > -20,
            'in_lower_20_percent': dist_to_low < 20,
            'position_score': position_score,
            'position_strength': '强' if position_score > 70 else '中' if position_score > 40 else '弱'
        }
    
    def _calculate_position_score(self, dist_to_high: float, dist_to_low: float, open_position: str) -> float:
        """计算价格位置评分"""
        base_score = 50
        
        # 接近年高加分（可能突破）
        if dist_to_high > -5:
            base_score += 15
        # 接近年低加分（支撑强劲）
        elif dist_to_low < 5:
            base_score += 15
        
        # 高开加分
        if open_position == '高开':
            base_score += 10
        
        return min(100, max(0, base_score))
    
    def calculate_rsi(self, closes: List[float], period: int = 14) -> float:
        """
        真实 RSI 计算 — P0 修复 (替代原来的估算方法)

        Args:
            closes: 收盘价列表（至少 period+1 个）
            period: RSI 周期

        Returns:
            RSI 值 (0-100)
        """
        if len(closes) < period + 1:
            return 50.0  # 数据不足时返回中性值

        deltas = np.diff(closes[-(period + 1):])
        gains = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0.0
        losses = abs(np.mean(deltas[deltas < 0])) if np.any(deltas < 0) else 0.001

        rs = gains / losses
        return round(100 - (100 / (1 + rs)), 1)

    def calculate_rsi_from_stock_data(self, stock_data: Dict, klines: List[Dict] = None,
                                       period: int = 14) -> float:
        """
        从 stock_data + klines 计算真实 RSI — P0 修复

        Args:
            stock_data: 股票实时数据
            klines: K线数据列表（用于计算历史 RSI）
            period: RSI 周期

        Returns:
            RSI 值
        """
        if klines and len(klines) >= period + 1:
            closes = [float(k['close']) for k in klines if float(k.get('close', 0)) > 0]
            return self.calculate_rsi(closes, period)

        # 如果没有 K 线数据，从 stock_data 的 change_pct 估算（降级方案）
        change_pct = stock_data.get('change_pct', 0)
        return 50 + change_pct * 2  # 降级估算
    
    def get_sentiment_signal(self, rsi: float) -> Dict:
        """获取情绪信号"""
        if rsi > 80:
            return {'signal': '极度贪婪', 'action': '考虑减仓', 'strength': '强'}
        elif rsi > 70:
            return {'signal': '贪婪', 'action': '持有观望', 'strength': '中'}
        elif rsi > 60:
            return {'signal': '偏多', 'action': '持有', 'strength': '中'}
        elif rsi > 40:
            return {'signal': '中性', 'action': '观望', 'strength': '弱'}
        elif rsi > 30:
            return {'signal': '偏空', 'action': '减仓', 'strength': '中'}
        elif rsi > 20:
            return {'signal': '恐惧', 'action': '考虑买入', 'strength': '中'}
        else:
            return {'signal': '极度恐惧', 'action': '强烈买入信号', 'strength': '强'}
    
    def analyze_multi_cycle_resonance(self, stock_data: Dict, klines: Dict = None) -> Dict:
        """
        多周期共振分析 — P0 修复: 使用真实 RSI 计算
        """
        # 获取 K 线数据
        daily_klines = None
        weekly_klines = None
        monthly_klines = None

        if klines:
            daily_klines = klines.get('daily', [])
            weekly_klines = klines.get('weekly', [])
            monthly_klines = klines.get('monthly', [])

        # 计算各周期真实 RSI
        daily_rsi = self.calculate_rsi_from_stock_data(stock_data, daily_klines, period=14)
        weekly_rsi = self.calculate_rsi_from_stock_data(stock_data, weekly_klines, period=14) if weekly_klines else daily_rsi
        monthly_rsi = self.calculate_rsi_from_stock_data(stock_data, monthly_klines, period=14) if monthly_klines else daily_rsi

        # 判断各周期趋势
        daily_trend = '上涨' if daily_rsi > 55 else '下跌' if daily_rsi < 45 else '震荡'
        weekly_trend = '上涨' if weekly_rsi > 55 else '下跌' if weekly_rsi < 45 else '震荡'
        monthly_trend = '上涨' if monthly_rsi > 55 else '下跌' if monthly_rsi < 45 else '震荡'

        # 共振判断
        trends = [daily_trend, weekly_trend, monthly_trend]
        bullish_count = trends.count('上涨')
        bearish_count = trends.count('下跌')

        if bullish_count >= 2 and bearish_count == 0:
            resonance = '三周期看涨共振'
            resonance_strength = '强'
            resonance_direction = '看涨共振'
        elif bearish_count >= 2 and bullish_count == 0:
            resonance = '三周期看跌共振'
            resonance_strength = '强'
            resonance_direction = '看跌共振'
        elif bullish_count >= 2:
            resonance = '双周期看涨'
            resonance_strength = '中'
            resonance_direction = '看涨共振'
        elif bearish_count >= 2:
            resonance = '双周期看跌'
            resonance_strength = '中'
            resonance_direction = '看跌共振'
        else:
            resonance = '无共振'
            resonance_strength = '弱'
            resonance_direction = '震荡'

        resonance_score = bullish_count * 10 + (1 - bearish_count / 3) * 10
        if daily_trend == weekly_trend:
            resonance_score += 10
        if daily_trend == monthly_trend:
            resonance_score += 5

        return {
            'daily_rsi': round(daily_rsi, 1),
            'weekly_rsi': round(weekly_rsi, 1),
            'monthly_rsi': round(monthly_rsi, 1),
            'daily_trend': daily_trend,
            'weekly_trend': weekly_trend,
            'monthly_trend': monthly_trend,
            'resonance': resonance,
            'resonance_strength': resonance_strength,
            'resonance_direction': resonance_direction,
            'resonance_score': round(resonance_score, 1)
        }
    
    def generate_kline_signals(self, stock_data: Dict, klines: Dict = None) -> Dict:
        """生成完整的K线信号 — P0 修复: 使用真实 RSI"""
        # 获取真实 RSI
        rsi = self.calculate_rsi_from_stock_data(stock_data, klines) if klines else self.calculate_rsi_from_stock_data(stock_data)
        sentiment = self.get_sentiment_signal(rsi)

        # 分析各类信号
        candlestick = self.analyze_candlestick_patterns(stock_data)
        volume = self.analyze_volume_signals(stock_data)
        position = self.analyze_price_position(stock_data)
        multi_cycle = self.analyze_multi_cycle_resonance(stock_data, klines)
        
        # 综合信号
        bullish_signals = []
        bearish_signals = []
        
        # 看涨信号
        if '锤子线' in candlestick['patterns']:
            bullish_signals.append('锤子线-底部反转')
        if '大阳线' in candlestick['patterns']:
            bullish_signals.append('大阳线-强势上涨')
        if '放量上涨' in volume['signals']:
            bullish_signals.append('放量上涨-资金流入')
        if '买盘强劲' in volume['signals']:
            bullish_signals.append('买盘强劲')
        if '接近年低' in position['position']:
            bullish_signals.append('接近年低-支撑位')
        if sentiment['action'] == '强烈买入信号':
            bullish_signals.append('极度恐惧-反弹信号')
        if '看涨共振' in multi_cycle['resonance_direction']:
            bullish_signals.append('看涨共振-多周期支持')
        
        # 看跌信号
        if '倒锤子线' in candlestick['patterns']:
            bearish_signals.append('倒锤子线-顶部反转')
        if '大阴线' in candlestick['patterns']:
            bearish_signals.append('大阴线-强势下跌')
        if '放量下跌' in volume['signals']:
            bearish_signals.append('放量下跌-资金流出')
        if '卖盘强劲' in volume['signals']:
            bearish_signals.append('卖盘强劲')
        if '接近年高' in position['position']:
            bearish_signals.append('接近年高-压力位')
        if sentiment['action'] == '考虑减仓':
            bearish_signals.append('极度贪婪-回调风险')
        if '看跌共振' in multi_cycle['resonance_direction']:
            bearish_signals.append('看跌共振-多周期压力')
        
        # 综合评分
        total_score = self._calculate_total_score(
            candlestick['pattern_score'],
            volume['volume_score'],
            position['position_score'],
            rsi
        )
        
        return {
            'rsi': rsi,
            'sentiment': sentiment,
            'candlestick': {
                'patterns': candlestick['patterns'],
                'pattern_score': candlestick['pattern_score']
            },
            'volume': {
                'signals': volume['signals'],
                'volume_score': volume['volume_score']
            },
            'position': {
                'position': position['position'],
                'position_score': position['position_score']
            },
            'price_position': position['position'],
            'bullish_signals': bullish_signals,
            'bearish_signals': bearish_signals,
            'overall_trend': '看涨' if len(bullish_signals) > len(bearish_signals) else '看跌' if len(bearish_signals) > len(bullish_signals) else '震荡',
            'multi_cycle': multi_cycle,
            'total_score': total_score,
            'trend_strength': '强' if total_score > 70 else '中' if total_score > 40 else '弱'
        }
    
    def _calculate_total_score(self, pattern_score: float, volume_score: float, 
                               position_score: float, rsi: float) -> float:
        """计算综合评分"""
        # 权重分配
        weights = {
            'pattern': 0.25,
            'volume': 0.25,
            'position': 0.20,
            'rsi': 0.30
        }
        
        # RSI转换分数 (RSI>50加分，<50减分)
        rsi_score = rsi if rsi >= 50 else 100 - rsi
        
        total = (
            pattern_score * weights['pattern'] +
            volume_score * weights['volume'] +
            position_score * weights['position'] +
            rsi_score * weights['rsi']
        )
        
        return round(total, 1)


if __name__ == '__main__':
    analyzer = KlineSignalAnalyzer()
    print("✓ K线信号分析器初始化完成")
    print("\n支持的分析类型:")
    print("  1. K线形态识别 (锤子线、十字星、大阳线等)")
    print("  2. 成交量信号 (放量、缩量、买卖盘)")
    print("  3. 价格位置分析 (年高低、开盘位置)")
    print("  4. RSI情绪指标")
    print("  5. 多周期共振分析 (日/周/月)")
    print("  6. 形态匹配度评分 (0-100)")
    print("  7. 综合多空信号")
    print("  8. 综合趋势评分")
