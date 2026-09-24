#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""#16 链 A 晨报×情报链 锁链测试 (2026-09-13)

锁链: 09-10 并行预取先例 + 09-09 诚实空态契约 — 情报链坏/超时不得拖垮
晨报链; prompt 无情报时量化壳完整 (缺输入不降智); 降级模板仍带情报段
(降级≠降智)。mock 全三源, 零网络零 LLM。

运行: python -m unittest tests.test_daily_report_intel
"""

import os
import sys
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.daily_report_service import (DailyReportService, DailyReportScheduler,
                                          DailyReportService as _DS)


def _ok_senti():
    return {'score': 0.12, 'label': 'neutral', 'n_articles': 8,
            'method': 'finbert'}


class TestGatherChain(unittest.TestCase):
    """_gather_stock_intel 取数链形 (三源+洞察+审计 全 mock)"""

    def setUp(self):
        self.svc = DailyReportService()

    def test_full_chain_three_sources(self):
        """intel 有三源 + 洞察喂 LLM + 审计正常 → 三键齐"""
        intel_ret = {'dragon_tiger': {'on_board': True, 'count': 2, 'items': [
                         {'date': '2026-08-04', 'turnover_pct': 18.0,
                          'd20_pct': -26.7, 'net_buy': 1.3e8}]},
                     'sentiment': _ok_senti(),
                     '_insight_feed': {'news': [{'title': 'x', 'date': '2026-09-10'}]}}
        insight_ret = {'valid': True, 'direction': 'bearish', 'conviction': 0.6,
                       'evidence': ['逼近52周高点'], 'invalidation': '缩量企稳'}
        audit_ret = {'block_hints': ['流动性预警'], 'skipped': False}
        eng = mock.Mock()
        eng.get_insight.return_value = insight_ret
        with mock.patch('modules.intel_engine.intel_summary',
                        side_effect=lambda c: intel_ret), \
             mock.patch('modules.insight_engine.get_insight_engine',
                        return_value=eng), \
             mock.patch('modules.exec_audit.audit_for_stock',
                        side_effect=lambda *a, **k: audit_ret):
            out = self.svc._gather_stock_intel('sz300620')
        self.assertEqual(set(out), {'intel', 'insight', 'audit'})
        # 同源喂料: intel.news → get_insight (不二次取数)
        eng.get_insight.assert_called_once()
        kwargs = eng.get_insight.call_args.kwargs
        self.assertTrue(kwargs['news'] and kwargs['dragon_tiger'])
        self.assertEqual(out['insight']['direction'], 'bearish')
        self.assertIn('流动性', out['audit']['block_hints'][0])

    def test_no_feed_not_empty_spin(self):
        """intel 空 (无新闻无榜) → 仍喂 get_insight (其内部 abstain) —
        链形锁: 晨报侧不预判缺料, fail-closed 契约在 insight_engine"""
        eng = mock.Mock()
        eng.get_insight.return_value = {'valid': False, 'abstain': True,
                                        'reason': 'empty'}
        with mock.patch('modules.intel_engine.intel_summary',
                        side_effect=lambda c: {}), \
             mock.patch('modules.insight_engine.get_insight_engine',
                        return_value=eng), \
             mock.patch('modules.exec_audit.audit_for_stock',
                        side_effect=lambda *a, **k: {'skipped': True}):
            out = self.svc._gather_stock_intel('sz300620')
        self.assertTrue(eng.get_insight.called)
        self.assertFalse(out['insight']['valid'])       # abstain 不伪造
        self.assertTrue(out['audit']['skipped'])         # 审计缺→跳过非假数据

    def test_source_error_is_not_crash(self):
        """三源全坏 (09-10 复现形) → 不崩, 由 bundle 链尾捕获"""
        with mock.patch('modules.intel_engine.intel_summary',
                        side_effect=ConnectionError('data down')), \
             mock.patch('modules.insight_engine.get_insight_engine',
                        side_effect=RuntimeError('engine down')), \
             mock.patch('modules.exec_audit.audit_for_stock',
                        side_effect=TimeoutError('t')):
            with self.assertRaises(ConnectionError):       # raise 给调用方
                self.svc._gather_stock_intel('sz300620')


class TestIntelBundle(unittest.TestCase):
    """_intel_bundle 并行+超时+坏链捕获 (09-10 并行形)"""

    def setUp(self):
        self.svc = _DS()

    def test_per_stock_chain_exception_skipped(self):
        """单票链炸 → 该票 skipped (不拖整链), 其余票正常"""
        svc = _DS()

        def fake(code):
            if code == 'sz300620':
                raise ConnectionError('down')
            return {'intel': {}, 'insight': None, 'audit': None}
        svc._gather_stock_intel = fake
        out = svc._intel_bundle(['sz300620', 'sh688981', 'sz301000'], timeout=5)
        self.assertEqual(len(out), 3)
        self.assertTrue(out['sz300620']['skipped'])      # 坏链→标记
        self.assertIn('ConnectionError', out['sz300620']['reason'])
        self.assertTrue(all('intel' in out[c] for c in ('sh688981', 'sz301000')))

    def test_bundle_timeout(self):
        """慢链 (mock sleep 占满预算) → 超时票标 skipped, 快票正常回收"""
        def fake(code):
            if code == 'sz300620':
                time.sleep(1.0)
                return {'intel': {}, 'insight': None, 'audit': None}
            return {'intel': {'x': 1}, 'insight': None, 'audit': None}
        svc = _DS()
        svc._gather_stock_intel = fake
        t0 = time.time()
        out = svc._intel_bundle(['sz300620', 'sh688981'], timeout=0.2)
        el = time.time() - t0
        self.assertLess(el, 2.0)                         # 并行非串行
        self.assertTrue(out['sz300620']['skipped'])      # 慢票超时标 skipped
        self.assertFalse(out['sh688981'].get('skipped')) # 快票正常回收

    def test_build_holdings_degrade_not_crash(self):
        """情报并行链整条炸 + K线空 (量化段诚实跳过) → build_holdings
        不崩不拖 (09-10: 主链 decide 不依赖它), 降级=不填 intel 非假数据"""
        fetcher = mock.Mock()
        fetcher.get_kline_data.return_value = []       # K线空 = 量化段诚实跳过
        with mock.patch('modules.dependencies.get_data_fetcher',
                        return_value=fetcher), \
             mock.patch('modules.portfolio_store.portfolio_store') as ps:
            ps.get_total_value.return_value = 1000000.0
            ps.get_snapshot.return_value = {'positions': [
                {'symbol': 'sz300620', 'name': '光库', 'qty': 100, 'cost': 50}]}
            svc = _DS()
            with mock.patch.object(svc, '_intel_bundle',
                                   side_effect=ConnectionError('intel down')):
                out = svc.build_holdings()
        self.assertEqual(len(out), 1)                  # 行照常产出
        self.assertNotIn('intel', out[0])              # 降级 = 不填非伪造


class TestPromptInject(unittest.TestCase):
    """assemble 输入形: 情报进 prompt 与降级模板 (非 LLM 不伪造)"""

    def setUp(self):
        self.svc = _DS()
        self.hold = [{
            'name': '光库', 'symbol': 'sz300620', 'price': 289.5,
            'pl_pct': -3.2, 'weight_pct': 12.0, 'budget_pct': 15.0,
            'cvar_95': 2.1, 'hint': '观察区',
            'intel': {'dragon_tiger': {'on_board': True, 'count': 2,
                                       'latest': {'date': '2026-08-04',
                                                  'turnover_pct': 18.0,
                                                  'd20_pct': -26.7}},
                      'unlock': {'next': {'unlock_date': '2026-10-20',
                                           'type': '首发原股东'}},
                      'sentiment': {'score': 0.12, 'label': 'neutral',
                                    'n_articles': 8, 'method': 'finbert',
                                    'degraded': False}},
            'insight': {'valid': True, 'direction': 'bearish',
                        'conviction': 0.6, 'evidence': ['逼近52周高点'],
                        'invalidation': '缩量企稳'},
            'audit': {'block_hints': ['5日均成交额 0.8亿 < 1亿']},
        }]
        self.us = [{'name': '纳指', 'close': 21000.0, 'chg_pct': -0.5}]

    def test_prompt_has_market_evidence(self):
        """真链形: prompt 含龙虎榜/解禁/洞察失效/审计 (4 票段原料升级)"""
        seen = {}

        def fake_route(p, timeout=120):
            seen['p'] = p
            return {'success': True, 'content': '# 晨报', 'provider': 'omlx'}

        with mock.patch('modules.llm_router.llm_router.route',
                        side_effect=fake_route):
            md = self.svc.assemble({'us': self.us, 'news': ['n'],
                                    'holdings': self.hold})
        p = seen['p']
        self.assertEqual(md, '# 晨报')
        for anchor in ('龙虎榜2次上榜', 'D20收益-26.7', '2026-10-20解禁',
                       'AI洞察 bearish0.6', '缩量企稳', '执行审计'):
            self.assertIn(anchor, p, f'prompt 缺情报锚点: {anchor}')
        # 纪律语义不被情报段污染: 输出要求仍是不预测点位
        self.assertIn('观察-触发-复核', p)

    def test_degraded_template_keeps_intel(self):
        """LLM 降级 (8080 down) → 降级模板仍带情报段 (降级≠降智)"""
        with mock.patch('modules.llm_router.llm_router.route',
                        side_effect=ConnectionError('8080 down')):
            md = self.svc.assemble({'us': self.us, 'news': [],
                                    'holdings': self.hold})
        self.assertIn('LLM 降级', md)
        for anchor in ('龙虎榜2次上榜', 'AI洞察 bearish0.6', '执行审计'):
            self.assertIn(anchor, md, f'降级模板缺情报: {anchor}')

    def test_empty_intel_quant_only(self):
        """无 intel (全降级) → 只出量化壳行, 诚实非假数据 (09-09 契约)"""
        h = {'name': '光库', 'symbol': 'sz300620', 'price': 289.5,
             'pl_pct': -3.2, 'weight_pct': 12.0, 'budget_pct': 15.0,
             'cvar_95': 2.1, 'hint': '观察区'}
        with mock.patch('modules.llm_router.llm_router.route',
                        side_effect=ConnectionError('8080 down')):
            md = self.svc.assemble({'us': [], 'news': [], 'holdings': [h]})
        self.assertNotIn('情报:', md)                    # 缺=不编造
        self.assertIn('观察区', md)


if __name__ == '__main__':
    unittest.main(verbosity=2)
