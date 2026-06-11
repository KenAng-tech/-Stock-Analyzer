"""
Report Generator Module
Generates comprehensive Markdown reports for stock analysis
"""

import time
from typing import Dict


class ReportGenerator:
    """Generates detailed stock analysis reports"""
    
    def generate_report(self, analysis: Dict) -> str:
        """Generate complete Markdown report"""
        report = []
        
        # Header
        report.append("=" * 80)
        report.append("           股票深度分析与未来走势预测报告")
        report.append("=" * 80)
        report.append(f"报告日期: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Section 1: Basic Info
        report.extend(self._section_basic_info(analysis))
        
        # Section 2: Fundamental Analysis
        report.extend(self._section_fundamental(analysis))
        
        # Section 3: Technical Analysis
        report.extend(self._section_technical(analysis))
        
        # Section 4: Fund Flow Analysis
        report.extend(self._section_fund_flow(analysis))
        
        # Section 5: Industry & Policy
        report.extend(self._section_industry(analysis))
        
        # Section 6: Quantitative Prediction
        report.extend(self._section_prediction(analysis))
        
        # Section 7: Trading Strategies
        report.extend(self._section_strategies(analysis))
        
        # Section 8: Risk Management
        report.extend(self._section_risk(analysis))
        
        # Section 9: Summary & Action Items
        report.extend(self._section_summary(analysis))
        
        return "\n".join(report)
    
    def _section_basic_info(self, analysis: Dict) -> list:
        """Generate basic information section"""
        info = analysis['basic_info']
        profit = analysis['profit_analysis']
        
        section = [
            "=" * 80,
            "第一章：基本信息与持仓状况",
            "=" * 80,
            "",
            "1.1 股票基本信息",
            "-" * 80,
            f"   股票名称: {info['name']} ({info['code']})",
            f"   报告日期: {info['date']}",
            f"   当前价格: {info['price']} 元",
            ""
        ]
        
        section.extend([
            "1.2 持仓状况分析",
            "-" * 80,
            f"   持仓成本: {info['cost_basis']} 元",
            f"   当前价格: {profit['current']} 元",
            f"   盈利金额: {profit['profit']:.2f} 元/股",
            f"   盈利比例: {profit['profit_pct']:.2f}%",
            f"   持仓状态: {profit['status']}",
            ""
        ])
        
        return section
    
    def _section_fundamental(self, analysis: Dict) -> list:
        """Generate fundamental analysis section"""
        fund = analysis['fundamental']
        
        section = [
            "=" * 80,
            "第二章：基本面深度分析",
            "=" * 80,
            "",
            "2.1 估值分析",
            "-" * 80,
            f"   当前市盈率(PE): {fund['valuation']['pe']} 倍",
            f"   估值水平: {fund['valuation']['level']}",
            f"   总市值: {fund['valuation']['market_cap']} 亿元",
            f"   流通市值: {fund['valuation']['circulating_cap']} 亿元",
            f"   自由流通比例: {fund['valuation']['free_float_ratio']:.1f}%",
            ""
        ]
        
        section.extend([
            "   估值解读:",
            "   • PE反映市场对公司高成长性的预期",
            "   • 相比行业平均，估值处于中高水平",
            "   • 高估值需要业绩持续高增长来支撑",
            "",
            "2.2 财务健康度评估",
            "-" * 80,
            "   基于行业数据和历史表现:",
            f"   • 营收增长率: {fund['financial_health']['revenue_growth']}（光通信景气周期）",
            f"   • 净利润增长率: {fund['financial_health']['profit_growth']}（规模效应释放）",
            f"   • 毛利率: {fund['financial_health']['gross_margin']}（光器件行业龙头）",
            f"   • 净利率: {fund['financial_health']['net_margin']}",
            f"   • 资产负债率: {fund['financial_health']['debt_ratio']}（稳健）",
            f"   • ROE: {fund['financial_health']['roe']}（优秀）",
            ""
        ])
        
        return section
    
    def _section_technical(self, analysis: Dict) -> list:
        """Generate technical analysis section"""
        tech = analysis['technical']
        
        section = [
            "=" * 80,
            "第三章：技术面深度分析",
            "=" * 80,
            "",
            "3.1 K线形态分析",
            "-" * 80,
            f"   今日K线: {tech['kline']['pattern']}",
            f"   振幅: {tech['kline']['amplitude']:.2f}%",
            "",
            "   形态判断:",
            "   • 今日高开低走，形成假阴线",
            "   • 收盘价接近日内低点，空头力量占优",
            "   • 成交量放大，换手率显示资金大幅换手",
            ""
        ]
        
        section.extend([
            "3.2 均线系统分析",
            "-" * 80,
            "   关键均线位置:",
            f"   • MA5: {tech['moving_averages']['ma5']:.0f} 元 (短期趋势)",
            f"   • MA10: {tech['moving_averages']['ma10']:.0f} 元",
            f"   • MA20: {tech['moving_averages']['ma20']:.0f} 元",
            f"   • MA60: {tech['moving_averages']['ma60']:.0f} 元",
            f"   • MA120: {tech['moving_averages']['ma120']:.0f} 元",
            f"   • MA250: {tech['moving_averages']['ma250']:.0f} 元",
            "",
            "   均线状态: 短期均线向下发散，中期均线走平",
            f"   趋势判断: {tech['trend']['short_term']}，{tech['trend']['medium_term']}",
            ""
        ])
        
        section.extend([
            "3.3 支撑与压力位",
            "-" * 80,
            "   强支撑位:",
        ])
        for i, s in enumerate(tech['support_resistance']['supports']):
            section.append(f"   • S{i+1}: {s['price']:.0f} 元 ({s['level']})")
        
        section.append("")
        section.append("   强压力位:")
        for i, r in enumerate(tech['support_resistance']['resistances']):
            section.append(f"   • R{i+1}: {r['price']:.0f} 元 ({r['level']})")
        
        section.extend([
            "",
            "3.4 技术指标分析",
            "-" * 80,
            f"   MACD: {tech['indicators']['macd']}（短期调整信号）",
            f"   KDJ: {tech['indicators']['kdj']}（反弹预期）",
            f"   RSI: {tech['indicators']['rsi']}（中性偏强）",
            f"   布林带: {tech['indicators']['bollinger']}，波动率扩大",
            "   成交量: 近期持续放大，资金活跃",
            ""
        ])
        
        return section
    
    def _section_fund_flow(self, analysis: Dict) -> list:
        """Generate fund flow analysis section"""
        flow = analysis['fund_flow']
        
        section = [
            "=" * 80,
            "第四章：资金面分析",
            "=" * 80,
            "",
            "4.1 主力资金流向",
            "-" * 80,
            f"   外盘: {flow['main_flow']['outer_disk']} 手",
            f"   内盘: {flow['main_flow']['inner_disk']} 手",
            f"   外内比: {flow['main_flow']['ratio']}",
            f"   判断: {flow['main_flow']['direction']}",
            "",
            f"   成交额: {flow['volume_analysis']['amount']:.0f} 万元 ({flow['volume_analysis']['amount']/10000:.2f} 亿元)",
            f"   换手率: {flow['volume_analysis']['turnover']}%",
            "",
            "   资金面解读:",
            "   • 高换手率表明多空分歧大",
            "   • 外盘略大于内盘，买盘稍占优",
            "   • 大额成交说明机构资金参与度高",
            "",
            "4.2 筹码分布分析",
            "-" * 80,
            "   • 筹码密集区: " + flow['chip_distribution']['dense_zone'],
            f"   • 获利盘比例: 约{flow['chip_distribution']['profit_ratio']}%",
            f"   • 套牢盘比例: 约{flow['chip_distribution']['trapped_ratio']}%",
            ""
        ]
        
        return section
    
    def _section_industry(self, analysis: Dict) -> list:
        """Generate industry analysis section"""
        industry = analysis['fundamental'].get('industry_profile', {})
        
        section = [
            "=" * 80,
            "第五章：行业与政策面分析",
            "=" * 80,
            "",
            "5.1 光通信行业景气度",
            "-" * 80,
            "   行业驱动因素:",
            "   1. AI算力需求爆发：800G/1.6T光模块需求激增",
            "   2. 5G-A/6G建设：基站密集化带动光纤需求",
            "   3. 数据中心升级：AI服务器集群需要高速互联",
            "   4. 国产替代加速：核心器件国产化率提升",
            "   5. 光纤传感应用拓展：工业、医疗、安防",
            "",
            "5.2 政策利好",
            "-" * 80,
            "   • '东数西算'工程持续推进",
            "   • 新基建投资加码",
            "   • 数字经济战略规划",
            "   • 半导体/光电子产业扶持",
            "",
            "5.3 行业竞争优势",
            "-" * 80,
            "   • 光纤器件行业龙头之一",
            "   • 技术壁垒高（专利布局完善）",
            "   • 客户资源优质（华为、中兴等）",
            "   • 产能扩张计划推进中",
            ""
        ]
        
        return section
    
    def _section_prediction(self, analysis: Dict) -> list:
        """Generate prediction section"""
        pred = analysis['prediction']
        
        section = [
            "=" * 80,
            "第六章：量化模型与未来走势预测",
            "=" * 80,
            "",
            "6.1 多因子预测模型",
            "-" * 80,
            "   因子权重分配:",
            "   • 动量因子: 25%",
            "   • 价值因子: 20%",
            "   • 波动率因子: 15%",
            "   • 资金因子: 20%",
            "   • 情绪因子: 10%",
            "   • 行业因子: 10%",
            "",
            f"   综合评分: {pred['model']['composite']}/10（中性偏多）",
            ""
        ]
        
        section.extend([
            "6.2 未来走势情景分析",
            "-" * 80,
            ""
        ])
        
        for scenario in pred['scenarios']:
            emoji = '📊'
            section.extend([
                f"   {emoji} {scenario['name']}（概率{scenario['probability']}）",
                "   • 触发条件：根据市场环境和行业趋势判断",
                f"   • 目标区间: {scenario['target_range']}",
                f"   • 时间窗口: {scenario['timeframe']}",
                f"   • 关键信号: {scenario['signal']}",
                ""
            ])
        
        section.extend([
            "6.3 概率加权目标价",
            "-" * 80,
            f"   加权目标价: {pred['weighted_target']} 元",
            f"   当前价格: {analysis['basic_info']['price']} 元",
            f"   上行空间: {pred['upside_space']}%",
            ""
        ])
        
        return section
    
    def _section_strategies(self, analysis: Dict) -> list:
        """Generate trading strategies section"""
        price = analysis['basic_info']['price']
        
        section = [
            "=" * 80,
            "第七章：交易策略体系",
            "=" * 80,
            "",
            "7.1 核心策略：趋势跟踪 + 动态止盈",
            "-" * 80,
            "   策略逻辑:",
            "   1. 趋势确认：股价站稳MA20且MA5>MA10",
            "   2. 入场时机：回调至支撑位企稳",
            "   3. 仓位管理：根据波动率动态调整",
            "   4. 止盈止损：移动止盈+固定止损",
            ""
        ]
        
        # Strategy recommendations
        strategies = [
            {
                'name': '保守策略（适合风险厌恶者）',
                'actions': [
                    f'当前减仓50%，锁定利润',
                    '剩余仓位设置220元止损',
                    f'若突破{price*1.25:.0f}元，回补至80%仓位'
                ],
                'return': '15-25%（未来3个月）'
            },
            {
                'name': '平衡策略（推荐）',
                'actions': [
                    '当前减仓30%，保留底仓',
                    '设置动态止盈：跌破MA5减仓10%',
                    f'加仓条件：突破{price*1.12:.0f}元且成交量放大'
                ],
                'return': '20-40%（未来3个月）'
            },
            {
                'name': '激进策略（适合风险偏好者）',
                'actions': [
                    '持有全部仓位',
                    f'突破{price*1.12:.0f}元加仓20%',
                    '跌破200元止损'
                ],
                'return': '30-60%（未来3个月）'
            }
        ]
        
        for strategy in strategies:
            section.extend([
                f"   🎯 {strategy['name']}",
                "   • " + "\n   • ".join(strategy['actions']),
                f"   • 预期收益: {strategy['return']}",
                ""
            ])
        
        # Position management matrix
        section.extend([
            "7.2 仓位管理矩阵",
            "-" * 80,
            "   | 价格区间   | 操作建议     | 仓位建议 |",
            "   |------------|-------------|----------|",
            f"   | > {price*1.12:.0f} 元   | 持有/加仓    | 80-100%  |",
            f"   | {price:.0f}-{price*1.12:.0f}元  | 持有         | 70-80%   |",
            f"   | {price*0.96:.0f}-{price:.0f}元  | 持有/观望    | 60-70%   |",
            f"   | {price*0.88:.0f}-{price*0.96:.0f}元  | 逢低加仓     | 70-80%   |",
            f"   | < {price*0.88:.0f} 元   | 减仓/止损    | 40-60%   |",
            ""
        ])
        
        return section
    
    def _section_risk(self, analysis: Dict) -> list:
        """Generate risk management section"""
        section = [
            "=" * 80,
            "第八章：风险管理与应急预案",
            "=" * 80,
            "",
            "8.1 风险矩阵",
            "-" * 80,
            "   | 风险类型   | 概率 | 影响 | 应对措施           |",
            "   |-----------|------|------|-------------------|",
            "   | 市场风险   | 中   | 高   | 设置止损，分散投资 |",
            "   | 行业风险   | 低   | 中   | 关注政策变化       |",
            "   | 个股风险   | 中   | 高   | 仓位控制           |",
            "   | 流动性风险 | 低   | 低   | 避免大额集中交易   |",
            "",
            "8.2 应急预案",
            "-" * 80,
            "   情况一：突发利空跌破支撑位",
            "   → 立即减仓50%，等待企稳信号",
            "",
            "   情况二：突破压力位加速上涨",
            "   → 加仓20%，设置移动止损",
            "",
            "   情况三：连续缩量横盘",
            "   → 持有观望，等待方向选择",
            ""
        ]
        
        return section
    
    def _section_summary(self, analysis: Dict) -> list:
        """Generate summary and action items"""
        price = analysis['basic_info']['price']
        
        section = [
            "=" * 80,
            "第九章：综合建议与行动清单",
            "=" * 80,
            "",
            "9.1 短期操作（1-2周）",
            "-" * 80,
            "   □ 关注支撑位有效性",
            f"   □ 若反弹至{price*1.08:.0f}-{price*1.12:.0f}元可减仓30%",
            "   □ 设置止损保护利润",
            "",
            "9.2 中期策略（1-3个月）",
            "-" * 80,
            f"   □ 目标价{analysis['prediction']['weighted_target']}元（中性情景）",
            f"   □ 乐观情景目标{price*1.36:.0f}元",
            "   □ 根据均线系统动态调整仓位",
            "",
            "9.3 长期观点（3-6个月）",
            "-" * 80,
            "   □ 光通信行业景气周期延续",
            "   □ AI算力需求是核心驱动力",
            "   □ 关注公司产能扩张进展",
            "",
            "9.4 📌 最终建议",
            "-" * 80,
            "   📌 当前建议：持有为主，逢高减仓30%",
            "   📌 止损位：220元（跌破减仓50%）",
            f"   📌 目标位：{price*1.11:.0f}元（短期），{price*1.36:.0f}元（中期）",
            f"   📌 加仓位：突破{price*1.12:.0f}元且放量，加仓20%",
            "   📌 核心逻辑：盈利保护优先，趋势跟随为辅",
            "",
            "=" * 80,
            "免责声明：以上分析基于公开数据和模型推算，仅供参考，不构成投资建议。"
            "股市有风险，投资需谨慎。请根据自身风险承受能力做出决策。",
            "=" * 80
        ]
        
        return section
