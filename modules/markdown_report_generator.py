"""
Markdown Report Generator (2026-09-19 复活: 08-13 md 报告链; Archify 整合后调用点误删致 analyze 断链 2 天, 从 backups 恢复)
Generates comprehensive Markdown reports for stock analysis
"""

import time
from typing import Dict


class MarkdownReportGenerator:
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

        # ADX 指标分析
        adx = analysis.get('adx', {})
        if adx:
            section.extend([
                "3.5 ADX 趋势强度分析",
                "-" * 80,
                f"   ADX: {adx.get('adx', 0):.2f}",
                f"   DI+ (多头): {adx.get('di_plus', 0):.2f}",
                f"   DI- (空头): {adx.get('di_minus', 0):.2f}",
                f"   ADX 趋势判断: {adx.get('trend_strength', '中性')}",
                f"   趋势方向: {adx.get('trend_direction', '无趋势')}",
                ""
            ])

        # K线信号综合
        kline_signals = analysis.get('kline_signals', {})
        if kline_signals:
            section.extend([
                "3.6 K线信号综合",
                "-" * 80,
                "   信号汇总:",
            ])
            for sig in kline_signals.get('signals', []):
                direction = sig.get('direction', 'neutral')
                indicator = sig.get('indicator', '')
                strength = sig.get('strength', '中')
                emoji = '🟢' if direction == 'bullish' else '🔴' if direction == 'bearish' else '🟡'
                section.append(f"   {emoji} {indicator}: {strength}{'看涨' if direction == 'bullish' else '看跌' if direction == 'bearish' else '中性'}")
            section.append("")

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
        """Generate prediction section with real model data"""
        pred = analysis['prediction']
        model = pred.get('model', {})
        price = analysis['basic_info']['price']

        section = [
            "=" * 80,
            "第六章：量化模型与未来走势预测",
            "=" * 80,
            "",
            "6.1 多因子预测模型",
            "-" * 80,
            f"   综合评分: {model.get('composite', 0)}/10",
            f"   因子数量: {model.get('factor_count', len(model.get('factors', [])))}",
        ]

        # 显示因子评分
        if model.get('factors') and model.get('scores'):
            section.append("   因子评分详情:")
            for name, score in zip(model.get('factors', []), model.get('scores', [])):
                section.append(f"   • {name}: {score:.2f}")
            section.append("")

        # ML 模型预测
        section.extend([
            "6.2 ML 集成模型预测 (RandomForest + LightGBM Stacking)",
            "-" * 80,
            f"   预测方向: {model.get('ml_direction', 'neutral')}",
            f"   置信度: {model.get('ml_confidence', 0) * 100:.1f}%",
            f"   模型已训练: {'是' if model.get('ml_trained') else '否'}",
            "   概率分布:",
        ])
        probs = model.get('ml_probabilities', {})
        section.extend([
            f"   • 上涨概率: {probs.get('up', 0) * 100:.1f}%",
            f"   • 中性概率: {probs.get('neutral', 0) * 100:.1f}%",
            f"   • 下跌概率: {probs.get('down', 0) * 100:.1f}%",
            ""
        ])

        # Time-LLM 预测
        if model.get('time_llm'):
            tllm = model['time_llm']
            section.extend([
                "6.3 Time-LLM 统一预测 (Transformer 架构)",
                "-" * 80,
                f"   预测方向: {tllm.get('direction', 'N/A')}",
                f"   置信度: {tllm.get('confidence', 0) * 100:.1f}%",
                f"   市场状态: {model.get('time_llm_regime', 'N/A')}",
                f"   使用模型: {', '.join(model.get('time_llm_models_used', []))}",
                ""
            ])

        # Regime-Switching 预测
        if model.get('regime_switching'):
            rs = model['regime_switching']
            section.extend([
                "6.4 Regime-Switching 多模型预测",
                "-" * 80,
                f"   预测方向: {rs.get('direction', 'N/A')}",
                f"   置信度: {rs.get('confidence', 0) * 100:.1f}%",
                f"   检测状态: {model.get('regime_detected', 'N/A')}",
                ""
            ])

        # SOTA 模型预测
        sota_models = pred.get('sota_models', {})
        if sota_models:
            section.extend([
                "6.5 SOTA 深度学习模型预测",
                "=" * 80,
                "",
            ])

            # PatchTST
            if 'patchtst' in sota_models:
                pt = sota_models['patchtst']
                section.extend([
                    "   • PatchTST (Patch-based Transformer)",
                    f"     方向: {pt.get('direction', 'N/A')} | 置信度: {pt.get('confidence', 0) * 100:.1f}% | 训练: {'是' if pt.get('trained') else '否'}",
                    ""
                ])

            # Mamba
            if 'mamba' in sota_models:
                mb = sota_models['mamba']
                section.extend([
                    "   • Mamba HFT (状态空间模型)",
                    f"     方向: {mb.get('direction', 'N/A')} | 置信度: {mb.get('confidence', 0) * 100:.1f}% | 训练: {'是' if mb.get('trained') else '否'}",
                    ""
                ])

            # Diffusion
            if 'diffusion' in sota_models:
                df = sota_models['diffusion']
                section.extend([
                    "   • Diffusion 扩散模型",
                    f"     方向: {df.get('direction', 'N/A')} | 置信度: {df.get('confidence', 0) * 100:.1f}% | 不确定性: {df.get('uncertainty', 0):.4f}",
                    ""
                ])

            # DRL
            if 'drl' in sota_models:
                dr = sota_models['drl']
                section.extend([
                    "   • DRL 深度强化学习",
                    f"     决策: {dr.get('action', 'N/A')} | 置信度: {dr.get('confidence', 0) * 100:.1f}% | 仓位: {dr.get('position_size', 0) * 100:.1f}%",
                    ""
                ])

            # Conformal
            if 'conformal' in sota_models:
                cp = sota_models['conformal']
                section.extend([
                    "   • Conformal Prediction (共形预测区间)",
                    f"     点预测: {cp.get('point_prediction', 0):.6f}",
                    f"     90% 置信区间: [{cp.get('lower_bound', 0):.6f}, {cp.get('upper_bound', 0):.6f}]",
                    f"     覆盖率保证: {cp.get('coverage_probability', 0) * 100:.0f}%",
                    ""
                ])

            # Alpha158
            if 'alpha158' in sota_models and sota_models['alpha158'].get('available'):
                a158 = sota_models['alpha158']
                section.extend([
                    "   • Alpha158 因子 (158 个量化因子)",
                    f"     可用因子数: {a158.get('n_factors', 0)}",
                    "     Top 10 因子值:",
                ])
                for fname, fval in list(a158.get('top_factors', {}).items())[:10]:
                    section.append(f"     - {fname}: {fval:.4f}")
                section.append("")

            # GNN
            if 'gnn' in sota_models:
                gn = sota_models['gnn']
                section.extend([
                    "   • GNN 图神经网络",
                    f"     方向: {gn.get('direction', 'N/A')} | 置信度: {gn.get('confidence', 0) * 100:.1f}%",
                    ""
                ])

            # Cross-Market
            if 'cross_market' in sota_models:
                cm = sota_models['cross_market']
                section.extend([
                    "   • Cross-Market 跨市场因子",
                    f"     A股情绪: {cm.get('a_share_sentiment', 'N/A')}",
                    f"     港股情绪: {cm.get('hk_sentiment', 'N/A')}",
                    f"     美股情绪: {cm.get('us_sentiment', 'N/A')}",
                    ""
                ])

            # Sentiment
            if 'sentiment' in sota_models:
                sm = sota_models['sentiment']
                section.extend([
                    "   • Sentiment 情感分析",
                    f"     整体情绪: {sm.get('label', 'N/A')} | 分数: {sm.get('score', 0):.3f} | 文章数: {sm.get('n_articles', 0)}",
                    f"     正面: {sm.get('positive_ratio', 0) * 100:.1f}% | 中性: {sm.get('neutral_ratio', 0) * 100:.1f}% | 负面: {sm.get('negative_ratio', 0) * 100:.1f}%",
                    ""
                ])

            # SOTA 综合投票
            sota_consensus = pred.get('sota_consensus', 'neutral')
            sota_agreement = pred.get('sota_agreement', 0)
            sota_avg_conf = pred.get('sota_avg_confidence', 0)
            section.extend([
                "6.6 SOTA 模型综合投票",
                "-" * 80,
                f"   综合方向: {sota_consensus}",
                f"   模型一致性: {sota_agreement * 100:.1f}%",
                f"   平均置信度: {sota_avg_conf * 100:.1f}%",
                f"   参与模型数: {len(sota_models)}",
                "",
            ])

        # 情景分析
        section.extend([
            "6.7 未来走势情景分析",
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
            "6.8 概率加权目标价",
            "-" * 80,
            f"   加权目标价: {pred['weighted_target']} 元",
            f"   当前价格: {analysis['basic_info']['price']} 元",
            f"   上行空间: {pred['upside_space']}%",
            f"   日波动率: {pred.get('daily_volatility', 0)}%",
            f"   月波动率: {pred.get('month_volatility', 0)}%",
            ""
        ])

        return section
    
    def _section_strategies(self, analysis: Dict) -> list:
        """Generate trading strategies with real model data"""
        price = analysis['basic_info']['price']
        pred = analysis.get('prediction', {})
        model = pred.get('model', {})
        kelly = pred.get('kelly', {})
        tech = analysis.get('technical', {})
        daily_vol = pred.get('daily_volatility', 3.0)

        ml_direction = model.get('ml_direction', 'neutral')
        ml_confidence = model.get('ml_confidence', 0.5)

        # 动态止损/止盈
        atr_stop_pct = daily_vol * 2
        atr_take_pct = daily_vol * 3
        stop_loss_price = price * (1 - atr_stop_pct / 100)
        take_profit_price = price * (1 + atr_take_pct / 100)

        # 根据 ML 方向调整策略
        if ml_direction == 'up' and ml_confidence > 0.6:
            trend_bias = "偏多"
            conservative_action = "持有，逢回调加仓"
            balanced_action = "持有底仓，突破加仓"
            aggressive_action = "积极持有，可适度杠杆"
        elif ml_direction == 'down' and ml_confidence > 0.6:
            trend_bias = "偏空"
            conservative_action = "减仓50%，等待企稳"
            balanced_action = "减仓30%，保留底仓观望"
            aggressive_action = "持有空仓，等待反转信号"
        else:
            trend_bias = "中性"
            conservative_action = "持有观望，等待方向明确"
            balanced_action = "持有底仓，区间操作"
            aggressive_action = "区间高抛低吸"

        section = [
            "=" * 80,
            "第七章：交易策略体系",
            "=" * 80,
            "",
            "7.1 核心策略：趋势跟踪 + 动态止盈",
            "-" * 80,
            f"   趋势判断: {trend_bias} (ML 置信度 {ml_confidence * 100:.1f}%)",
            "   策略逻辑:",
            "   1. 趋势确认：股价站稳MA20且MA5>MA10",
            "   2. 入场时机：回调至支撑位企稳",
            f"   3. 仓位管理：Kelly 最优 {kelly.get('position_pct', 0):.1f}%",
            f"   4. 止损位: {stop_loss_price:.2f} 元 (-{atr_stop_pct:.1f}%)",
            f"   5. 止盈位: {take_profit_price:.2f} 元 (+{atr_take_pct:.1f}%)",
            ""
        ]

        # 策略建议
        section.extend([
            "7.2 分层策略建议",
            "-" * 80,
            f"   🎯 保守策略（适合风险厌恶者）",
            f"   • {conservative_action}",
            f"   • 止损位: {stop_loss_price:.2f} 元",
            f"   • 预期收益: -{atr_stop_pct:.1f}% ~ +{atr_take_pct * 0.5:.1f}%",
            "",
            f"   🎯 平衡策略（推荐）",
            f"   • {balanced_action}",
            f"   • 动态止盈: 跌破 MA5 减仓 10%",
            f"   • 预期收益: -{atr_stop_pct:.1f}% ~ +{atr_take_pct:.1f}%",
            "",
            f"   🎯 激进策略（适合风险偏好者）",
            f"   • {aggressive_action}",
            f"   • 止损位: {stop_loss_price:.2f} 元",
            f"   • 预期收益: -{atr_stop_pct:.1f}% ~ +{atr_take_pct * 1.5:.1f}%",
            ""
        ])

        # Position management matrix
        section.extend([
            "7.3 仓位管理矩阵",
            "-" * 80,
            "   | 价格区间         | 操作建议       | 仓位建议 |",
            "   |------------------|---------------|----------|",
            f"   | > {price * 1.1:.0f} 元        | 持有/加仓      | 70-100%  |",
            f"   | {price:.0f}-{price * 1.1:.0f} 元      | 持有          | 60-70%   |",
            f"   | {price * 0.95:.0f}-{price:.0f} 元      | 持有/观望     | 50-60%   |",
            f"   | {stop_loss_price:.0f}-{price * 0.95:.0f} 元     | 逢低加仓     | 60-80%   |",
            f"   | < {stop_loss_price:.0f} 元        | 减仓/止损     | 0-40%    |",
            ""
        ])

        return section
    
    def _section_risk(self, analysis: Dict) -> list:
        """Generate risk management with real model data"""
        price = analysis['basic_info']['price']
        pred = analysis.get('prediction', {})
        daily_vol = pred.get('daily_volatility', 3.0)
        month_vol = pred.get('month_volatility', 15.0)
        drift = pred.get('drift_status', {})
        kelly = pred.get('kelly', {})

        section = [
            "=" * 80,
            "第八章：风险管理与应急预案",
            "=" * 80,
            "",
            "8.1 风险矩阵",
            "-" * 80,
            f"   日波动率: {daily_vol:.1f}% | 月波动率: {month_vol:.1f}%",
            "   | 风险类型   | 概率 | 影响 | 应对措施           |",
            "   |-----------|------|------|-------------------|",
            "   | 市场风险   | 中   | 高   | 设置止损，分散投资 |",
            "   | 行业风险   | 低   | 中   | 关注政策变化       |",
            "   | 个股风险   | 中   | 高   | 仓位控制           |",
            "   | 流动性风险 | 低   | 低   | 避免大额集中交易   |",
            "   | 模型风险   | 中   | 中   | 概念漂移监控       |",
            ""
        ]

        # 概念漂移警告
        if drift.get('drift_detected'):
            section.extend([
                "   ⚠️ 概念漂移警告: 模型预测性能可能已退化，建议重新训练",
                ""
            ])

        # Kelly 仓位风险
        section.extend([
            "8.2 Kelly 仓位风险控制",
            "-" * 80,
            f"   Kelly 最优分数: {kelly.get('kelly_fraction', 0):.2f}",
            f"   推荐仓位: {kelly.get('position_pct', 0):.1f}%",
            f"   胜率: {kelly.get('win_rate', 0) * 100:.1f}%",
            f"   盈亏比: {kelly.get('win_loss_ratio', 0):.2f}",
            f"   操作建议: {kelly.get('recommendation', '保守')}",
            ""
        ])

        # 应急预案
        stop_loss_pct = daily_vol * 2
        section.extend([
            "8.3 应急预案",
            "-" * 80,
            f"   情况一：突发利空跌破止损位 ({price * (1 - stop_loss_pct / 100):.2f} 元)",
            f"   → 立即减仓至 {kelly.get('position_pct', 0) * 0.3:.0f}%，等待企稳信号",
            "",
            f"   情况二：突破压力位加速上涨 (+{daily_vol * 3:.1f}%)",
            "   → 加仓 15%，设置移动止损",
            "",
            "   情况三：连续缩量横盘",
            "   → 持有观望，等待方向选择",
            "",
            "   情况四：概念漂移检测触发",
            "   → 重新训练 ML 模型，更新预测参数",
            ""
        ])

        # CVaR 风险度量
        cvar = analysis.get('cvar', {})
        if cvar:
            section.extend([
                "8.4 CVaR 风险度量 (条件风险价值)",
                "-" * 80,
            ])
            for level, val in cvar.items():
                if isinstance(val, (int, float)):
                    section.append(f"   {level} CVaR: {val:.2f}%")
            section.append("")

        # ATR 动态止损/止盈
        atr = analysis.get('atr', {})
        if atr:
            section.extend([
                "8.5 ATR 动态止损/止盈",
                "-" * 80,
            ])
            stop_loss = atr.get('stop_loss', {})
            if stop_loss:
                section.append(f"   动态止损位: {stop_loss.get('stop_loss_price', 'N/A')} 元 (-{stop_loss.get('stop_loss_pct', 0):.1f}%)")
                section.append(f"   ATR: {stop_loss.get('atr_value', 'N/A')}")
            take_profit = atr.get('take_profit', {})
            if take_profit:
                section.append(f"   动态止盈位: {take_profit.get('take_profit_price', 'N/A')} 元 (+{take_profit.get('take_profit_pct', 0):.1f}%)")
            sr = atr.get('support_resistance', {})
            if sr:
                section.append("   动态支撑/压力:")
                for s in sr.get('supports', []):
                    section.append(f"   • 支撑: {s.get('price', 'N/A')} 元 ({s.get('level', '')})")
                for r in sr.get('resistances', []):
                    section.append(f"   • 压力: {r.get('price', 'N/A')} 元 ({r.get('level', '')})")
            section.append("")

        return section
    
    def _section_summary(self, analysis: Dict) -> list:
        """Generate summary and action items with real model data"""
        price = analysis['basic_info']['price']
        pred = analysis.get('prediction', {})
        model = pred.get('model', {})
        sentiment = analysis.get('sentiment', {})
        kelly = pred.get('kelly', {})
        drift = pred.get('drift_status', {})

        # 根据 ML 方向生成动态建议
        ml_direction = model.get('ml_direction', 'neutral')
        ml_confidence = model.get('ml_confidence', 0.5)
        sota_consensus = pred.get('sota_consensus', 'neutral')
        weighted_target = pred.get('weighted_target', price)
        upside = pred.get('upside_space', 0)

        # 生成综合建议
        if ml_direction == 'up' and ml_confidence > 0.6:
            primary_action = "持有/加仓"
            primary_logic = "多模型一致看涨，趋势向上"
        elif ml_direction == 'down' and ml_confidence > 0.6:
            primary_action = "减仓/止损"
            primary_logic = "多模型一致看跌，注意风险"
        else:
            primary_action = "持有观望"
            primary_logic = "多空分歧，等待方向明确"

        # 动态止损位（基于波动率）
        daily_vol = pred.get('daily_volatility', 3.0)
        stop_loss_pct = min(daily_vol * 2, 10)  # 2x 日波动率或最多 10%
        stop_loss_price = price * (1 - stop_loss_pct / 100)

        # 动态加仓位
        add_position_pct = daily_vol * 1.5
        add_position_price = price * (1 + add_position_pct / 100)

        section = [
            "=" * 80,
            "第九章：综合建议与行动清单",
            "=" * 80,
            "",
            "9.1 多模型综合判断",
            "-" * 80,
            f"   ML 集成模型: {ml_direction} (置信度 {ml_confidence * 100:.1f}%)",
            f"   SOTA 综合投票: {sota_consensus}",
            f"   加权目标价: {weighted_target} 元 (上行空间 {upside:.1f}%)",
            f"   Kelly 最优仓位: {kelly.get('position_pct', 0):.1f}%",
            f"   概念漂移检测: {'⚠️ 已检测到' if drift.get('drift_detected') else '✅ 正常'}",
            ""
        ]

        # HMM 状态
        hmm = analysis.get('hmm', {})
        if hmm:
            regime = hmm.get('regime', 'unknown')
            regime_emoji = {'uptrend': '📈', 'downtrend': '📉', 'sideways': '➡️'}.get(regime, '❓')
            section.append(f"   HMM 市场状态: {regime_emoji} {regime}")
            probs = hmm.get('probabilities', {})
            if probs:
                section.append(f"   HMM 概率: 上涨={probs.get('uptrend', 0):.1%} | 下跌={probs.get('downtrend', 0):.1%} | 震荡={probs.get('sideways', 0):.1%}")
            section.append("")

        # K线信号汇总
        kline_sigs = analysis.get('kline_signals', {})
        if kline_sigs:
            bullish_count = sum(1 for s in kline_sigs.get('signals', []) if s.get('direction') == 'bullish')
            bearish_count = sum(1 for s in kline_sigs.get('signals', []) if s.get('direction') == 'bearish')
            total_signals = len(kline_sigs.get('signals', []))
            section.append(f"   K线信号: {total_signals} 个指标 — 看涨 {bullish_count} | 看跌 {bearish_count}")
            section.append("")

        # 情感分析
        if sentiment.get('available') and sentiment.get('n_articles', 0) > 0:
            section.extend([
                f"   市场情绪: {sentiment.get('label', 'neutral')} (分数 {sentiment.get('score', 0):.3f})",
                f"   正面 {sentiment.get('positive_ratio', 0) * 100:.1f}% | 中性 {sentiment.get('neutral_ratio', 0) * 100:.1f}% | 负面 {sentiment.get('negative_ratio', 0) * 100:.1f}%",
                ""
            ])

        section.extend([
            "9.2 短期操作（1-2周）",
            "-" * 80,
            f"   □ 当前策略: {primary_action}",
            f"   □ 核心逻辑: {primary_logic}",
            f"   □ 若反弹至 {price * 1.05:.0f}-{price * 1.08:.0f} 元可减仓 20%",
            f"   □ 设置止损位: {stop_loss_price:.2f} 元 (-{stop_loss_pct:.1f}%)",
            "",
            "9.3 中期策略（1-3个月）",
            "-" * 80,
            f"   □ 目标价: {weighted_target} 元（中性情景）",
            f"   □ 乐观情景目标: {price * 1.2:.0f} 元",
            f"   □ 加仓位: 突破 {add_position_price:.0f} 元且放量，加仓 15%",
            f"   □ 根据 Kelly 公式动态调整仓位至 {kelly.get('position_pct', 0):.0f}%",
            "",
            "9.4 长期观点（3-6个月）",
            "-" * 80,
            "   □ 关注行业景气周期变化",
            "   □ AI算力/数字经济是核心驱动力",
            "   □ 定期跟踪模型预测准确率",
            "   □ 注意概念漂移信号，及时重新训练模型",
            "",
            "9.5 📌 最终建议",
            "-" * 80,
            f"   📌 当前建议: {primary_action}",
            f"   📌 止损位: {stop_loss_price:.2f} 元 (-{stop_loss_pct:.1f}%)",
            f"   📌 目标位: {weighted_target} 元（加权平均）",
            f"   📌 加仓位: 突破 {add_position_price:.0f} 元且放量",
            f"   📌 核心逻辑: {primary_logic}",
            "",
            "=" * 80,
            "免责声明：以上分析基于公开数据和模型推算，仅供参考，不构成投资建议。"
            "股市有风险，投资需谨慎。请根据自身风险承受能力做出决策。",
            "=" * 80
        ])

        return section
