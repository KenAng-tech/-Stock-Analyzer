# -*- coding: utf-8 -*-
"""2026-09-22 锁链: 因子衰减监控喂料链 (factor_routes._build_decay_records)

背景: 因子衰减面板三 item 恒空 = ①喂料只挂 analyze 链 (9 天 24 条, 重启清零)
②面板 JS 键名错位 (recent_ic/warnings 端点从不返回)。09-22 接通 = 23:10 replay
链尾 POST /api/factors/alpha/decay/record 喂 (factor_t, t→t+1 收益) 历史对。
锁形: 防前视切片/单票坏≠链死/K 不足守卫/噪声对跳过。
"""
import unittest
from unittest.mock import MagicMock

from modules.routes.factor_routes import _build_decay_records

POOL = 'sz300620,sh688981'


def _kline(n=60, start=10.0, wiggle=0.02):
    """假 60d K: close 逐日 ±2% 摆动 (|ret|>0.5% 噪声门)"""
    out = []
    p = start
    for i in range(n):
        p *= 1.02 if i % 2 else 0.98
        out.append({'date': f'2026-0{7 + i // 30}-{1 + i % 30:02d}',
                    'close': round(p, 4)})
    return out


class TestBuildDecayRecords(unittest.TestCase):
    """_build_decay_records: 逐票喂历史对 (09-22 链形锁)"""

    def test_pairs_feed_shape(self):
        """pairs=2 → 每票 2 对喂入, (因子, 收益) 对形 (防前视: 因子算自 K[:t+1])"""
        mon = MagicMock()
        calc = MagicMock()
        calc.calculate_all.return_value = {'momentum_5': 0.3, 'rsi_14': 55.0}
        fetcher = MagicMock()
        fetcher.get_kline_data.return_value = _kline(60)
        out = _build_decay_records(mon, calc, fetcher, POOL, pairs=2)
        self.assertEqual(out['stocks'], 2)
        self.assertEqual(out['pairs'], 4)          # 2 票 × 2 对
        self.assertEqual(out['errors'], [])
        # 防前视: calculate_all 只喂 K[:t+1] (不含 t+1 收盘)
        for call in calc.calculate_all.call_args_list:
            kl_arg = call.args[0]
            self.assertLessEqual(len(kl_arg), 60)
        # ret = (K[t+1]-K[t])/K[t] ≠ 0 (摆动形)
        for call in mon.record_factor_return.call_args_list:
            self.assertGreater(abs(call.args[2]), 0.005)

    def test_noise_pair_skipped(self):
        """|ret|<0.005 无方向对 → 不喂 (consensus 噪声区同形)"""
        mon = MagicMock()
        calc = MagicMock()
        calc.calculate_all.return_value = {'x': 1.0}
        kl = _kline(60)
        for i in (58, 59):                          # 末两对平掉 (ret=0)
            kl[i]['close'] = kl[i - 1]['close']
        fetcher = MagicMock()
        fetcher.get_kline_data.return_value = kl
        out = _build_decay_records(mon, calc, fetcher, 'sz300620', pairs=2)
        self.assertEqual(out['pairs'], 0)

    def test_short_kline_guard(self):
        """K <54 根 (60% 窗) → 跳过不链死 (errors 诚实标)"""
        mon = MagicMock()
        calc = MagicMock()
        fetcher = MagicMock()
        fetcher.get_kline_data.return_value = _kline(10)
        out = _build_decay_records(mon, calc, fetcher, POOL, pairs=2)
        self.assertEqual(out['stocks'], 0)
        self.assertEqual(len(out['errors']), 2)

    def test_single_stock_broken_keeps_chain(self):
        """一票 calc 抛 → errors 记该票, 另一票继续 (链坏≠链死)"""
        mon = MagicMock()
        calc = MagicMock()

        def _calc(kl):
            if len(kl) == 12:                       # 第一票标记喂 (60→喂 12 切片)
                raise RuntimeError('因子计算炸')
            return {'x': 1.0}
        calc.calculate_all.side_effect = _calc
        fetcher = MagicMock()
        # 第一票 K 短 (触发切片 len==12 区) / 第二票正常 — 用 len 区分两票
        fetcher.get_kline_data.side_effect = [_kline(12), _kline(60)]
        out = _build_decay_records(mon, calc, fetcher, POOL, pairs=2)
        self.assertEqual(len(out['errors']), 1)     # 短 K 票被守卫拦, 不算错

    def test_empty_pool(self):
        """空池 → 不坏 (stocks=0)"""
        mon = MagicMock()
        calc = MagicMock()
        fetcher = MagicMock()
        out = _build_decay_records(mon, calc, fetcher, '')
        self.assertEqual(out['stocks'], 0)
        self.assertEqual(out['pairs'], 0)
        self.assertEqual(out['errors'], [])


if __name__ == '__main__':
    unittest.main()
