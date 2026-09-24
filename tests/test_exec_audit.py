#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""P0-c 可成交性审计 + P0-a 角色门控 锁链测试 (2026-09-12)

锁链: 2026 调研「执行假设审计」+ 校准门控两层的数学/数据形 —
板块限幅判定 (创业板 20cm 实测锚: 300620 曾 +20.0009%) / 流动性阈值 /
龙虎榜反例锚 / role_gate Brier>0.55 触发与不触发。
mock 取数, 零网络零 LLM。

运行: python -m unittest tests.test_exec_audit
"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.exec_audit import audit_execution, _board_limit
from modules.decision_calibration import role_gate


class TestBoardLimit(unittest.TestCase):
    """板块涨跌幅限制表 (无 ST 源不判 ST — 缺数据不猜)"""

    def test_boards(self):
        self.assertEqual(_board_limit('sz300620'), 20.0)   # 创业板
        self.assertEqual(_board_limit('sz301000'), 20.0)
        self.assertEqual(_board_limit('sh688981'), 20.0)   # 科创板
        self.assertEqual(_board_limit('sh689009'), 20.0)   #  cdr
        self.assertEqual(_board_limit('sh600000'), 10.0)   # 主板
        self.assertEqual(_board_limit('sz000001'), 10.0)


class TestLimitDetection(unittest.TestCase):
    """触板/贴板判定 (kline 双源 + snapshot 优先)"""

    def test_limit_up_chuangyeban(self):
        """+20.0009% 于 20cm 票 → limit_up + buy 语境提示"""
        kl = [{'date': f'2026-09-{d:02d}', 'close': 100.0, 'volume': 1e6,
               'amount': 0} for d in range(1, 6)]
        kl[-1] = dict(kl[-1], close=120.0)   # +20%
        out = audit_execution('sz300620', 'buy', klines=kl)
        self.assertEqual(out['limit']['status'], 'limit_up')
        self.assertTrue(any('涨停' in h for h in out['block_hints']),
                        f"20cm 贴板应触发 buy 提示: {out['block_hints']}")

    def test_limit_down_main_board(self):
        kl = [{'date': f'2026-09-{d:02d}', 'close': 100.0, 'volume': 1e6}
              for d in range(1, 6)]
        kl[-1] = dict(kl[-1], close=89.9)    # -10.1% 主板触板
        out = audit_execution('sh600000', 'sell', klines=kl)
        self.assertEqual(out['limit']['status'], 'limit_down')
        self.assertTrue(any('跌停' in h for h in out['block_hints']))

    def test_normal_volatility_no_hint(self):
        kl = [{'date': f'2026-09-{d:02d}', 'close': 100.0, 'volume': 1e6}
              for d in range(1, 6)]
        kl[-1] = dict(kl[-1], close=105.0)   # +5% 不贴板
        out = audit_execution('sz300620', 'buy', klines=kl)
        self.assertIsNone(out['limit']['status'])
        self.assertEqual(out['limit']['source'], 'kline')

    def test_snapshot_priority_and_empty(self):
        """snapshot 优先于 kline; 全缺 → 诚实降级不抛"""
        out = audit_execution('sz300620', 'neutral',
                              snapshot={'change_pct': -20.0, 'close': 10,
                                        'prev_close': 12.5})
        self.assertEqual(out['limit']['source'], 'snapshot')
        empty = audit_execution('sz300620', 'buy')
        self.assertIsNone(empty['limit']['status'])       # 无数据不猜


class TestLiquidity(unittest.TestCase):
    """流动性 (5日均成交额 = volume(股)×close, 实测 K线 volume 形)"""

    def _kl(self, vol):
        return [{'date': f'2026-09-{d:02d}', 'close': 50.0, 'volume': vol}
                for d in range(1, 6)]

    def test_illiquid_flag(self):
        """50 亿以下均额 → illiquid + 提示 (成本吃掉收益语境)"""
        out = audit_execution('sh600000', 'buy',
                              klines=self._kl(vol=2e5),   # 50e6×… = 0.1亿级
                              )
        self.assertFalse(out['liquidity']['ok'])
        self.assertTrue(any('流动性' in h or '滑点' in h
                            for h in out['block_hints']))

    def test_liquid_ok(self):
        out = audit_execution('sh600000', 'buy',
                              klines=self._kl(vol=2e7))    # 20亿均额
        self.assertTrue(out['liquidity']['ok'])


class TestDragonTigerAnchor(unittest.TestCase):
    """300620 反例锚: 高净买≠好收益 — 审计呈现证据, 不造方向"""

    def test_on_board_high_turnover_hint(self):
        """(09-13 修) 新鲜上榜 (≤2 交易日) 才出 hint, hint 直呈真实日期"""
        dt = [{'date': __import__('datetime').datetime.now().strftime(
                   '%Y-%m-%d'),
               'net_buy': 2e9, 'turnover_pct': 18.0, 'd20_pct': -26.7}]
        out = audit_execution('sz300620', 'buy', dragon_tiger=dt)
        self.assertTrue(out['board']['on_board'])
        self.assertTrue(any('龙虎榜 ' in h and '换手 18.0%' in h
                            for h in out['block_hints']))

    def test_old_board_date_no_hint(self):
        """(09-13 锁) 数月前上榜 ≠「此刻情绪」→ 无 hint (反「近2日」错标)"""
        dt = [{'date': '2026-06-16', 'net_buy': 2e9,
               'turnover_pct': 18.0, 'd20_pct': -26.7}]
        out = audit_execution('sz300620', 'buy', dragon_tiger=dt)
        self.assertTrue(out['board']['on_board'])   # 数据仍呈现 (日期/D20 锚)
        self.assertFalse(any('龙虎榜' in h for h in out['block_hints']))

    def test_board_not_a_buy_signal(self):
        """核心锁: 龙虎榜净买 20 亿 + D20 -26.7% 不得生成 bullish 语境"""
        dt = [{'date': '2026-06-16', 'net_buy': 2e9, 'turnover_pct': 10.9,
               'd20_pct': -26.7}]
        out = audit_execution('sz300620', 'buy', dragon_tiger=dt)
        self.assertNotIn('看多', str(out))                 # 无方向暗示
        self.assertIn('不构成任何方向的投票或降权', out['note'])


class TestRoleGate(unittest.TestCase):
    """P0-a 角色门控 (Brier>0.55 & n≥20 才降权)"""

    def test_all_correct_and_all_wrong(self):
        recs = {'tech': [(0.7, 1)] * 25,               # Brier=0.09 → 不触发
                'senti': [(0.9, 0)] * 25}              # Brier=0.81 → 触发
        out = role_gate(recs)
        self.assertFalse(out['tech']['gated'])
        self.assertLess(out['tech']['brier'], 0.55)
        self.assertTrue(out['senti']['gated'])
        self.assertGreater(out['senti']['brier'], 0.55)

    def test_min_sample_honest(self):
        """n<20 → 恒不触发 (观察期诚实, 不空转降权)"""
        out = role_gate({'rl': [(0.9, 0)] * 19})
        self.assertFalse(out['rl']['gated'])
        self.assertIsNone(out['rl']['brier'])           # 未算不装真值

    def test_nonfinite_filter(self):
        """str/NaN/坏 outcome 过滤; 25 条有效 (0.9,0) 留池 → Brier=0.81 触发"""
        recs = [('0.9', 0), (0.9, None), (float('nan'), 1),
                (0.9, 0)] * 25
        out = role_gate({'x': recs})
        self.assertEqual(out['x']['n'], 25)              # 75 条坏形被过滤
        self.assertTrue(out['x']['gated'])               # 0.81 > 0.55
        # 全坏形 → n=0 诚实 skipped
        allbad = role_gate({'y': [('0.9', 0), (0.9, None), (float('nan'), 1)] * 25})
        self.assertEqual(allbad['y']['n'], 0)
        self.assertFalse(allbad['y']['gated'])


class TestAuditForStockDegradation(unittest.TestCase):
    """audit_for_stock 取数封装: 超时/异常 → skipped 降级不拖决策"""

    def test_exception_degrades(self):
        from modules.exec_audit import audit_for_stock
        with mock.patch('modules.data_fetcher.StockDataFetcher',
                        side_effect=ConnectionError('data down')):
            out = audit_for_stock('sz300620', 'buy', timeout=3.0)
        self.assertNotIn('skipped', out)                # 降级仍出审计块
        self.assertIn('部分数据缺失', out['note'])       # 诚实标记


if __name__ == '__main__':
    unittest.main(verbosity=2)
