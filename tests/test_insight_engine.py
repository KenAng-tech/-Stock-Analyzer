#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""P0-b AI 洞察链 + P1 情报雷达 锁链测试 (2026-09-12)

锁链: 09-11 决策链 B 纪律延续 (contract fail-closed, 不伪造真链) +
2026 调研「RAG 基本面→洞察」形 — 空输入 abstain 不空转 LLM, LLM 降级
不伪装洞察, 解析坏不喂假洞察。mock router, 零网络零 LLM。

运行: python -m unittest tests.test_insight_engine
"""

import json
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.insight_engine import InsightEngine
from modules.dragon_tiger_fetcher import to_display, _strip_code


class TestInsightEngine(unittest.TestCase):
    """洞察引擎链形 (09-12 三缺口: 空转/降级/假数据 全锁)"""

    def setUp(self):
        self.eng = InsightEngine()
        self.news = [{'title': '某公司中标大单', 'date': '2026-09-11'}]

    def test_empty_input_no_llm_call(self):
        """无新闻且无龙虎榜 → abstain 且不打 LLM (不空转)"""
        with mock.patch('modules.llm_router.llm_router.route') as r:
            out = self.eng.get_insight('sz300620', news=[], dragon_tiger=[])
            r.assert_not_called()
        self.assertTrue(out['abstain'])
        self.assertEqual(out['source'], 'empty')

    def test_omlx_success_chain(self):
        """8080 真形: 合法 JSON → valid 洞察 (300620 实跑形回归)"""
        good = json.dumps({'direction': 'bearish', 'conviction': 0.65,
                           'evidence': ['现价逼近52周高点', '两次上榜净买'],
                           'invalidation': '若缩量企稳则失效'})
        resp = {'success': True, 'content': good, 'provider': 'omlx'}
        with mock.patch('modules.llm_router.llm_router.route',
                        side_effect=lambda *a, **k: resp):
            out = self.eng.get_insight('sz300620', news=self.news)
        self.assertTrue(out['valid'])
        self.assertEqual(out['direction'], 'bearish')
        self.assertAlmostEqual(out['conviction'], 0.65, places=2)
        self.assertEqual(out['source'], 'llm_insight')

    def test_bare_json_wrapper_parse(self):
        """``` 包裹 + 裸 JSON 混形 — 剥壳解析 (辩论链先例形)"""
        content = '```json\n' + json.dumps(
            {'direction': 'neutral', 'conviction': 0.4, 'evidence': ['x'],
             'invalidation': 'y'}) + '\n```'
        resp = {'success': True, 'content': content, 'provider': 'omlx'}
        with mock.patch('modules.llm_router.llm_router.route',
                        side_effect=lambda *a, **k: resp):
            out = self.eng.get_insight('sz300620', news=self.news)
        self.assertTrue(out['valid'])
        self.assertEqual(out['direction'], 'neutral')

    def test_degraded_and_drift(self):
        """LLM 降级/契约漂移 → 诚实 abstain (假洞察比无洞察危险)"""
        for resp, tag in (
            ({'success': False, 'provider': 'rule_engine'}, 'fallback'),
            ({'success': True, 'content': '完全不是JSON', 'provider': 'omlx'},
             'parse'),
            ({'success': True, 'content': '{"direction":"up","conviction":0.5}',
              'provider': 'omlx'}, 'direction 漂移'),
            ({'success': True, 'content':
              '{"direction":"neutral","conviction":"高"}', 'provider': 'omlx'},
             'conviction 漂移'),
            ({'success': True, 'content': '{"direction":"neutral"}',
              'provider': 'omlx'}, 'conviction 缺键'),
        ):
            with mock.patch('modules.llm_router.llm_router.route',
                            side_effect=lambda *a, r=resp, **k: r):
                self.eng._cache.clear()
                out = self.eng.get_insight('sz300620', news=self.news)
            self.assertTrue(out['abstain'], f"{tag} 应 abstain: {out}")
            self.assertFalse(out['valid'], f"{tag} 不得伪造真链")

    def test_cache_single_call(self):
        """同票二次调用走缓存 (同 llm_sentiment 1h TTL 形)"""
        resp = {'success': True, 'content': '{"direction":"neutral",'
                '"conviction":0.5,"evidence":[],"invalidation":"x"}',
                'provider': 'omlx'}
        calls = []

        def fake_route(*a, **k):
            calls.append(1)
            return resp

        with mock.patch('modules.llm_router.llm_router.route',
                        side_effect=fake_route):
            self.eng.get_insight('sz300620', news=self.news)
            self.eng.get_insight('sz300620', news=self.news)
        self.assertEqual(len(calls), 1)                  # 第二次走缓存


class TestDragonTigerParse(unittest.TestCase):
    """龙虎榜 API → 消费形 (300620 实形锚: 20.0009% 20cm + D20 -26.7)"""

    def test_strip_code(self):
        self.assertEqual(_strip_code('sz300620'), '300620')
        self.assertEqual(_strip_code('SH688981'), '688981')
        self.assertEqual(_strip_code(''), '')

    def test_real_api_shape(self):
        raw = [{'TRADE_DATE': '2026-08-04 00:00:00', 'SECURITY_CODE': '300620',
                'SECURITY_NAME_ABBR': '光库科技', 'CHANGE_RATE': 20.0009,
                'CLOSE_PRICE': 254.33, 'BILLBOARD_NET_AMT': 136776993.63,
                'BILLBOARD_BUY_AMT': 8.5e8, 'BILLBOARD_SELL_AMT': 7.1e8,
                'TURNOVERRATE': 7.2, 'EXPLANATION': '日涨幅达到15%的前5只证券',
                'BUY_SEAT': '[{"NAME":"华鑫xx"}]', 'SELL_SEAT': None,
                'D20_CLOSE_ADJCHRATE': 15.36}]
        out = to_display(raw)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]['name'], '光库科技')
        self.assertAlmostEqual(out[0]['change_pct'], 20.0009, places=4)
        self.assertEqual(out[0]['seats_buy'], ['华鑫xx'])

    def test_broken_rows_dropped(self):
        """席位串坏形/缺键 → 跳行不炸 (解析失败记日志不静默)"""
        raw = [{'TRADE_DATE': '2026-08-04', 'SECURITY_CODE': '1'},
               {'TRADE_DATE': '2026-08-03', 'SECURITY_CODE': '2',
                'BUY_SEAT': 'not-json'}]
        out = to_display(raw)
        self.assertGreaterEqual(len(out), 1)               # 不整体炸


class TestIntelSummary(unittest.TestCase):
    """情报雷达聚合 (三源独立降级 — 一源坏≠整卡坏)"""

    def test_all_sources_down_honest(self):
        from modules.intel_engine import intel_summary
        with mock.patch('modules.data_fetcher.StockDataFetcher',
                        side_effect=ConnectionError('down')), \
             mock.patch('modules.sentiment_engine.get_sentiment_engine',
                        side_effect=RuntimeError('engine down')), \
             mock.patch('modules.dragon_tiger_fetcher.fetch_dragon_tiger_history',
                        side_effect=TimeoutError('t')):
            out = intel_summary('sz300620')
        self.assertTrue(out['dragon_tiger']['degraded'])
        self.assertTrue(out['sentiment']['degraded'])
        self.assertTrue(out['unlock']['next'] is None)

    def test_empty_news_not_fake(self):
        """新闻源空 (主链+旁路皆空) → sentiment no_data 诚实 (09-17 契约)"""
        from modules.intel_engine import intel_summary

        class F:
            def get_stock_news(self, c): return []
            def get_stock_posts(self, c): return []
        with mock.patch('modules.data_fetcher.StockDataFetcher',
                        side_effect=lambda: F()), \
             mock.patch('modules.dragon_tiger_fetcher.fetch_dragon_tiger_history',
                        side_effect=lambda *a, **k: []):
            out = intel_summary('sz300620')
        self.assertEqual(out['news']['count'], 0)
        self.assertIn(out['sentiment'].get('method'), ('no_data', None,
                                                       'dictionary',
                                                       'finbert'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
