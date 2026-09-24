#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""东财新闻旁路锁链测试 (2026-09-11 决策链 B 后续)

锁链: 主链 get_stock_news 旧参数双死 (200 空 body + data.list 键漂移) 的
修复形 — 旁路解析数学形 (JSONP 剥壳/<em> 剥离/坏形→[]) + data_fetcher
主链空→旁路触发。mock urlopen, 零网络零 LLM。

运行: python -m unittest tests.test_eastmoney_news
"""

import json
import sys
import os
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.eastmoney_news_fetcher import fetch_news_eastmoney


def _jsonp(articles):
    """构造东财新版 JSONP 响应形 (剥壳实测样本形, <em> 高亮包裹)"""
    return "jQuery(" + json.dumps(
        {"resStatus": {}, "resBody": {"cmsArticleCode": "ok"},
         "result": {"cmsArticleWebOld": articles}},
        ensure_ascii=False) + ")"


class TestNewsBypass(unittest.TestCase):
    """fetch_news_eastmoney 解析数学形 + guard"""

    def _fetch(self, body, code='sz300620'):
        class R:
            def __init__(self, b): self._b = b
            def read(self): return self._b.encode('utf-8') if isinstance(self._b, str) else self._b
            def __enter__(self): return self
            def __exit__(self, *a): return False
        with mock.patch('urllib.request.urlopen',
                        side_effect=lambda *a, **k: R(body) if body is not None else ConnectionError('down')):
            return fetch_news_eastmoney(code)

    def test_good_jsonp_parsed(self):
        """合法 JSONP → 剥壳 + 剥 <em> + 全字段入 dict"""
        body = _jsonp([
            {"title": "<em>宁德时代</em> 辟谣减持传闻", "date": "2026-09-11 08:00:00", "url": "http://a"},
            {"title": "小摩维持<em>宁德时代</em>增持", "date": "2026-09-10 17:00:00", "url": "http://b"},
        ])
        out = self._fetch(body)
        self.assertEqual(len(out), 2)
        self.assertNotIn('<em>', out[0]['title'])
        self.assertEqual(out[0]['title'], '宁德时代 辟谣减持传闻')
        self.assertEqual(out[1]['url'], 'http://b')

    def test_empty_and_broken_shapes_rejected(self):
        """空 body/非 JSONP/坏 JSON/全空条 → 恒 [] 不抛 (旁路降级形态)"""
        for bad in ('', '{}', 'not json', _jsonp([]), _jsonp([{"title": ""}])):
            self.assertEqual(self._fetch(bad or ''), [])
        self.assertEqual(self._fetch(None), [])          # ConnectionError → []

    def test_pure_code_keyword_and_empty_input(self):
        """代码前缀剥离 (sz300620→300620); 空输入 → [] 不发请求"""
        body = _jsonp([{"title": "x", "date": "d", "url": "u"}])
        captured = {}
        class R:
            def read(self): return body.encode()
            def __enter__(self): return self
            def __exit__(self, *a): return False
        def fake_open(req, **kw):
            captured['url'] = req.full_url
            return R()
        with mock.patch('urllib.request.urlopen',
                        side_effect=fake_open):
            out = fetch_news_eastmoney('sz300620')
        self.assertEqual(len(out), 1)
        self.assertIn('300620', captured['url'])
        self.assertNotIn('sz300620', captured['url'])
        self.assertEqual(fetch_news_eastmoney(''), [])


class TestMainChainFallback(unittest.TestCase):
    """get_stock_news: 主链空/异常 → 旁路二次取数 (data_fetcher 挂点)"""

    def _call_fetcher(self, fake_urlopen):
        from modules.data_fetcher import StockDataFetcher
        f = StockDataFetcher.__new__(StockDataFetcher)
        f.cache = {}
        with mock.patch('urllib.request.urlopen',
                        side_effect=fake_urlopen):
            return f, f.get_stock_news('sz300620')

    def test_main_empty_triggers_bypass(self):
        """主链 200 空 body (死链现行形) → 旁路被调 → 真新闻返回"""
        bypass_body = _jsonp([{"title": "<em>宁德时代</em>利好", "date": "d", "url": "u"}])
        calls = {'n': 0}

        def fake_open(req, **kw):
            class R:
                def read(self):
                    calls['n'] += 1
                    # 第 1 次 = 主链旧参数 (现网恒空 body); 第 2 次 = 旁路
                    # (bytes 形 = urllib 真实传输形; str 会在 .decode() 处炸)
                    return b'' if calls['n'] == 1 else bypass_body.encode('utf-8')
                def __enter__(self): return self
                def __exit__(self, *a): return False
            return R()
        f, out = self._call_fetcher(fake_open)
        self.assertEqual(calls['n'], 2, '主链空未触发旁路 — 挂点已断')
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]['title'], '宁德时代利好')

    def test_bypass_failure_returns_empty(self):
        """主链坏 + 旁路坏 → [] (决策链/端点只看到诚实空态, 不假数据)"""
        def boom(req, **kw):
            raise ConnectionError('network down')
        _, out = self._call_fetcher(boom)
        self.assertEqual(out, [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
