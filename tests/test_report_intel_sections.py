#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""#17 链 B 分析报告×四模块 锁链测试 (2026-09-13)

锁链: ReportGenerator 新 section (5.4 市场情报 / 8.6 可成交性审计)
的输入形 (intel_summary/get_insight/audit_for_stock 三真形) + 缺数据
诚实跳过 (09-09: 无上下文比坏上下文诚实)。in-process 直调, 零网络。

运行: python -m unittest tests.test_report_intel_sections
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.report_generator import ReportGenerator


def _analysis(intel=None, insight=None, audit=None):
    """最小完整 analysis (九章消费键齐; 四模块键仅 intel 三件套)"""
    kl = [{'date': f'2026-09-{d:02d}', 'close': 100.0, 'volume': 1e6}
          for d in range(1, 6)]
    a = {
        'basic_info': {'name': '测试', 'code': 'sz300620', 'date': '2026-09-13',
                       'price': 100.0, 'cost_basis': 90.0},
        'profit_analysis': {'current': 100.0, 'profit': 10.0,
                            'profit_pct': 11.1, 'status': '盈利'},
        'fundamental': {'valuation': {'pe': 50, 'level': '中高',
                                      'market_cap': 500, 'circulating_cap': 400,
                                      'free_float_ratio': 80.0},
                        'financial_health': {'revenue_growth': '20%',
                                             'profit_growth': '30%',
                                             'gross_margin': '35%',
                                             'net_margin': '15%',
                                             'debt_ratio': '40%', 'roe': '18%'}},
        'technical': {'kline': {'pattern': '阳线', 'amplitude': 5.0},
                      'moving_averages': {'ma5': 99, 'ma10': 98, 'ma20': 97,
                                          'ma60': 95, 'ma120': 90, 'ma250': 85},
                      'trend': {'short_term': '上', 'medium_term': '震荡'},
                      'support_resistance': {'supports': [], 'resistances': []},
                      'indicators': {'macd': '金叉', 'kdj': '中性',
                                     'rsi': 55, 'bollinger': '中轨'}},
        'fund_flow': {'main_flow': {'outer_disk': 1e4, 'inner_disk': 9e3,
                                    'ratio': 1.1, 'direction': '流入'},
                      'volume_analysis': {'amount': 5e4, 'turnover': 8.0},
                      'chip_distribution': {'dense_zone': '95-105',
                                            'profit_ratio': 60,
                                            'trapped_ratio': 20}},
        'prediction': {'model': {'composite': 6.0, 'factor_count': 5,
                                 'ml_direction': 'up', 'ml_confidence': 0.55},
                       'scenarios': [], 'weighted_target': 105,
                       'upside_space': 5, 'daily_volatility': 3.0,
                       'month_volatility': 15.0, 'drift_status': {},
                       'kelly': {}},
        'sentiment': {'available': False},
        '_klines_daily': kl,
    }
    if intel is not None:
        a['intel'] = intel
    if insight is not None:
        a['insight'] = insight
    if audit is not None:
        a['exec_audit'] = audit
    return a


# 09-13 晨实测真形 (intel_summary 三源 + 300620 真洞察 + 审计真形)
REAL_INTEL = {
    'dragon_tiger': {'on_board': True, 'count': 2,
                     'latest': {'date': '2026-08-04', 'turnover_pct': 18.0,
                                'd20_pct': -26.7, 'net_buy': 1.3e8}},
    'sentiment': {'score': -0.12, 'label': 'negative', 'n_articles': 8,
                  'method': 'finbert'},
    'news': {'titles': ['光库科技2026年中报'], 'count': 6},
    'unlock': {'next': {'unlock_date': '2026-10-20', 'type': '首发原股东',
                       'qty': 1e6, 'mv': 5e8}, 'recent_past': [],
               'calendar_count': 42},
}
REAL_INSIGHT = {'valid': True, 'direction': 'bearish', 'conviction': 0.6,
                'evidence': ['现价逼近52周高点413.3元', '两次上榜净买'],
                'invalidation': '若缩量企稳则失效', 'source': 'llm_insight'}
REAL_AUDIT = {'version': 'exec_audit_v1',
              'limit': {'status': None, 'threshold_pct': 20.0,
                       'last_pct': -2.1, 'source': 'kline'},
              'liquidity': {'ok': False, 'amount_yi_avg5': 0.8,
                            'note': '5日均成交额 0.80亿 < 1.0亿 — 滑点可能吃掉目标收益'},
              'cost': {'round_trip_bp': 26.0, 'note': '往返成本≈26.0bp'},
              'board': {'on_board': True, 'date': '2026-08-04',
                        'turnover_pct': 18.0, 'd20_pct': -26.7},
              'block_hints': ['5日均成交额 0.80亿 < 1.0亿 — 滑点可能吃掉目标收益'],
              'note': '审计=执行假设观察层, 不构成任何方向的投票或降权'}


class TestIntelSection54(unittest.TestCase):
    def test_full_renders(self):
        """真形 intel+insight → 5.4 段含龙虎榜/解禁/情绪/洞察 + 不投票注记"""
        r = ReportGenerator().generate_report(
            _analysis(REAL_INTEL, REAL_INSIGHT, REAL_AUDIT))
        for anchor in ('市场情报', '龙虎榜', 'D20收益-26.7', '2026-10-20',
                       'AI 投资洞察', 'bearish', '缩量企稳', '不构成方向投票'):
            self.assertIn(anchor, r, f'5.4 缺锚点: {anchor!r}')

    def test_abstain_honest(self):
        """insight abstain (无料/LLM 降级) → 诚实「证据不足, 弃权」非空转"""
        abstain = {'valid': False, 'abstain': True, 'source': 'empty',
                   'reason': '无新闻/无龙虎榜证据 — 不空转 LLM',
                   'direction': 'neutral', 'conviction': 0.0}
        r = ReportGenerator().generate_report(_analysis(REAL_INTEL, abstain))
        self.assertIn('证据不足, 弃权', r)
        self.assertNotIn('缩量企稳', r)              # 空态不得带判断

    def test_degraded_sources_honest(self):
        """三源全降级 (一源坏≠整卡坏) → 5.4 出现且逐源标「降级」非 0"""
        degraded = {'dragon_tiger': {'on_board': False, 'count': 0,
                                     'latest': None, 'degraded': True},
                    'sentiment': {'score': None, 'label': None,
                                  'degraded': True},
                    'unlock': {'next': None, 'recent_past': [],
                               'degraded': True}}
        r = ReportGenerator().generate_report(_analysis(degraded, None))
        self.assertIn('降级', r)
        self.assertIn('市场情报', r)

    def test_no_intel_keeps_5_4_omitted(self):
        """无 intel (页面直开/预热坏) → 报告照常 (9 章), 缺段非空段"""
        r = ReportGenerator().generate_report(_analysis())
        self.assertIn('股票深度分析', r)
        self.assertNotIn('市场情报', r)
        self.assertNotIn('可成交性审计', r)          # 缺 = 整段不出


class TestAuditSection86(unittest.TestCase):
    def test_audit_renders_observational(self):
        """审计真形 → 8.6 含流动性预警 + 26bp + 不投票语义保持"""
        r = ReportGenerator().generate_report(_analysis(audit=REAL_AUDIT))
        for anchor in ('可成交性审计', '流动性预警', '26.0bp',
                       '不构成任何方向的投票或降权'):
            self.assertIn(anchor, r, f'8.6 缺锚点: {anchor!r}')

    def test_audit_skipped_not_fake(self):
        """audit skipped (5s 超时诚实跳过) → 8.6 不出 (缺输入≠坏数据)"""
        r = ReportGenerator().generate_report(
            _analysis(audit={'skipped': True, 'reason': 'timeout'}))
        self.assertNotIn('可成交性审计', r)


if __name__ == '__main__':
    unittest.main(verbosity=2)
