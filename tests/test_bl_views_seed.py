# -*- coding:utf-8 -*-
"""BL views 动态化测试 (2026-09-23 待拍板B, flag UDE_BL_VIEWS 门控)

背景: /api/portfolio/optimize 曾恒硬编码 2 views (主观 0.20@0.6/0.25@0.5,
09-22 实锤 = 面板手动触发链, 无真数据源)。本切片 = flag=1 时 views 从
model_seed.jsonl 末组 (最新 ts 组) 真实聚合: dir 多数票 + conf 均值 +
共识度 = 同向数/5 (2/5 弱共识形照实显, 非 1.0 满共识 = 收缩保护)。

expected = 手算 dict (独立事实源, 非 tautological)。
"""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.routes.portfolio_routes import build_bl_views  # noqa: E402

_POOL = ['sz300620', 'sh688981', 'sz300502', 'sh688800', 'sh603929']
_TS = '2026-09-22T23:10:19.123456'


def _seed_file(tmp, lines):
    p = os.path.join(tmp, 'model_seed.jsonl')
    with open(p, 'w', encoding='utf-8') as f:
        for l in lines:
            f.write(json.dumps(l) + '\n')
    return p


class TestBlViews(unittest.TestCase):
    """build_bl_views: seed 末组 → (views, pool) — flag=1 消费端"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tmp.cleanup()

    def test_views_math_hand_computed(self):
        """① 手算锁: 688981 空 0.49/共识1.0, 300620 多 0.5233/共识0.6
        (3/5 同向; 2 张 neutral 照实计 n=5 = 非满共识非 tautological 锁)"""
        lines = [
            {'ts': _TS, 'code': 'sz300620', 'model': 'm1', 'dir': 'buy', 'conf': 0.5},
            {'ts': _TS, 'code': 'sz300620', 'model': 'm2', 'dir': 'buy', 'conf': 0.5},
            {'ts': _TS, 'code': 'sz300620', 'model': 'm3', 'dir': 'buy', 'conf': 0.7},
            {'ts': _TS, 'code': 'sz300620', 'model': 'm4', 'dir': 'neutral', 'conf': 0.5},
            {'ts': _TS, 'code': 'sz300620', 'model': 'm5', 'dir': 'neutral', 'conf': 0.5},
            {'ts': _TS, 'code': 'sh688981', 'model': 'm1', 'dir': 'sell', 'conf': 0.4},
            {'ts': _TS, 'code': 'sh688981', 'model': 'm2', 'dir': 'sell', 'conf': 0.55},
            {'ts': _TS, 'code': 'sh688981', 'model': 'm3', 'dir': 'neutral', 'conf': 0.5},
            {'ts': _TS, 'code': 'sh688981', 'model': 'm4', 'dir': 'neutral', 'conf': 0.5},
            {'ts': _TS, 'code': 'sh688981', 'model': 'm5', 'dir': 'buy', 'conf': 0.5},
        ]
        p = _seed_file(self._tmp.name, lines)
        views = build_bl_views(_POOL, seed_path=p)
        self.assertIsNotNone(views, 'seed 有票 = 动态 views 产出')
        by_asset = {v['asset']: v for v in views}
        # 300620 = pool idx0: 3 买 (0.5+0.5+0.7)/3=0.5667 → 这里手算精确:
        # return = 0.5666666666666667 * 0.25 = 0.14166666666666666 → round 4 = 0.1417
        self.assertAlmostEqual(by_asset[0]['return'], 0.1417, places=4)
        self.assertAlmostEqual(by_asset[0]['confidence'], 0.6, places=4,
                               msg='共识 = 同向数/5 (2 neutral 照实计)')
        # 688981 = pool idx1: 2 空 (0.4+0.55)/2=0.475 → -0.11875 → round 4
        self.assertAlmostEqual(by_asset[1]['return'], -0.1188, places=4,
                               msg='空票负 return (0.475×0.25)')
        self.assertAlmostEqual(by_asset[1]['confidence'], 0.4, places=4)
        self.assertEqual(len(views), 2, '无票票不注入 view (诚实空态)')

    def test_views_latest_ts_group_only(self):
        """② 最新日期纪律: 只取 max(ts) 组, 旧组不混入 (09-13/22 同宗)"""
        old = {'ts': '2026-09-20T23:10:00.000000', 'code': 'sz300502',
               'model': 'm1', 'dir': 'buy', 'conf': 0.9}
        new = {'ts': _TS, 'code': 'sz300620', 'model': 'm1',
               'dir': 'sell', 'conf': 0.5}
        p = _seed_file(self._tmp.name, [old, new])
        views = build_bl_views(_POOL, seed_path=p)
        self.assertEqual(len(views), 1)
        self.assertEqual(views[0]['asset'], 0, '只取最新 ts 组 (300620)')
        self.assertLess(views[0]['return'], 0)

    def test_empty_seed_returns_none(self):
        """③ seed 缺/空 → None = 端点 fallback 现硬编码 (链坏≠链死)"""
        self.assertIsNone(build_bl_views(_POOL, seed_path='/nonexistent/x.jsonl'))
        p = _seed_file(self._tmp.name, [])
        self.assertIsNone(build_bl_views(_POOL, seed_path=p))


class TestBlPool(unittest.TestCase):
    """残留① (2026-09-23): 真数据模式 — 监控 5 票池 + 真行情 + seed 全 5 票 views"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tmp.cleanup()

    def test_views_per_stock_latest_round(self):
        """真实链形 (394 行实锤): 5 票 ts 错峰 (decide 串行 ~15s/票) →
        聚合必须 per-code 取各自最新轮, 非全局 max(ts) — 链形锁 (5 票全出)"""
        lines = []
        # 真形 = 09-23 实测: 5 票 ts 以秒级错峰 + 每票 15 模型票 (末组 16 实锤)
        for i, code in enumerate(_POOL):
            ts = f'2026-09-23T11:59:{14 + i:02d}.411392'
            for m in ('m1', 'm2', 'm3'):
                lines.append({'ts': ts, 'code': code, 'model': m,
                              'dir': 'buy', 'conf': 0.5})
        p = _seed_file(self._tmp.name, lines)
        views = build_bl_views(_POOL, seed_path=p)
        self.assertEqual(len(views), 5,
                         '5 票各自最新轮全聚合 (错峰 ts 链形锁, 非只 1 票)')
        self.assertEqual({v['asset'] for v in views}, {0, 1, 2, 3, 4})

    def test_pool_from_real_source(self):
        """build_bl_pool: 真 get_stock_info dict (09-23 实形) → 5 票真价 +
        mcap/pe=0 缺失源 → 50/100 uniform fallback (非硬编码假 200/500)"""
        from modules.routes.portfolio_routes import build_bl_pool
        snap = {'source': 'ths_data_primary', 'code': 'x', 'name': '688981',
                'price': 121.33, 'change': -1.08, 'pe': 0, 'market_cap': 0}

        class Src:
            def get_stock_info(self, code):
                return dict(snap)

        pool = build_bl_pool(price_source=Src())
        self.assertEqual(len(pool), 5)
        self.assertEqual(pool[1]['price'], 121.33, '真价直通 (expected=实测 dict)')
        self.assertEqual(pool[1]['market_cap'], 50, 'mcap 0 缺失→50 uniform 先验')

    def test_pool_fail_none(self):
        """一源全挂 → None = 端点保现形 (链坏≠链死, 5 票串行 15s 超时同护)"""
        from modules.routes.portfolio_routes import build_bl_pool

        class Bad:
            def get_stock_info(self, code):
                raise TimeoutError('t')

        self.assertIsNone(build_bl_pool(price_source=Bad()))


if __name__ == '__main__':
    unittest.main(verbosity=2)
