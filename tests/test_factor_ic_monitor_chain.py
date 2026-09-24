# -*- coding:utf-8 -*-
"""factor_ic_monitor 闭环锁链测试 (2026-09-22 拍板遗留 ①③)

形制: unittest + MagicMock + 依赖注入 (record_factor_return 全链为缝);
锁 4 组行为:
  ① 台账往返 (append/load 形制 + UNIQUE 防重)
  ② __init__ 恢复链 (重启不清零; get_fusion_ledger patch 注入)
  ③ 落盘链 (record → 台账行数; 重复调用防重不堆积)
  ④ reduce 闭环 (具体名→组名映射 + 组聚合≥2 门控 + type 判级 + flag 开关)
"""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.fusion_ledger import FusionLedger  # noqa: E402
from modules.factors.factor_ic_monitor import (  # noqa: E402
    FactorICMonitor, _factor_to_group)


class TestLedgerRoundTrip(unittest.TestCase):
    """① 台账往返 + 防重 (UNIQUE(code,factor,ts))"""

    def setUp(self):
        import tempfile
        self.tmp = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.tmp.close()
        self.led = FusionLedger(self.tmp.name)

    def tearDown(self):
        os.unlink(self.tmp.name)
        for suf in ('-wal', '-shm'):
            p = self.tmp.name + suf
            if os.path.exists(p):
                os.unlink(p)

    def test_append_load_roundtrip(self):
        """append_factor_records → load_factor_records = _factor_returns 形"""
        recs = {'rsi_7': (0.61, 0.012), 'macd': (-0.2, -0.012)}
        self.assertTrue(self.led.append_factor_records('sz300620', '2026-09-22', recs))
        out = self.led.load_factor_records()
        self.assertEqual(out['sz300620']['rsi_7'], [('2026-09-22', 0.61, 0.012)])
        self.assertEqual(len(out['sz300620']['macd']), 1)

    def test_same_day_dedupe(self):
        """同日同票同因子重入 = 1 行 (23:10 链 + analyze 链双喂防堆积)"""
        recs = {'rsi_7': (0.61, 0.012)}
        self.led.append_factor_records('sz300620', '2026-09-22', recs)
        self.led.append_factor_records('sz300620', '2026-09-22', recs)
        out = self.led.load_factor_records()
        self.assertEqual(len(out['sz300620']['rsi_7']), 1)


class TestMonitorRestoreChain(unittest.TestCase):
    """② __init__ 台账恢复 (重启不清零)"""

    def setUp(self):
        import tempfile
        self.tmp = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.tmp.close()
        self.led = FusionLedger(self.tmp.name)

    def tearDown(self):
        os.unlink(self.tmp.name)
        for suf in ('-wal', '-shm'):
            p = self.tmp.name + suf
            if os.path.exists(p):
                os.unlink(p)

    def test_init_restores_from_ledger(self):
        self.led.append_factor_records(
            'sh688981', '2026-09-22', {'rsi_7': (0.5, 0.01), 'macd': (0.1, 0.02)})
        with patch('modules.fusion_ledger.get_fusion_ledger', return_value=self.led):
            m = FactorICMonitor()
        self.assertIn('sh688981', m._factor_returns)
        self.assertEqual(len(m._factor_returns['sh688981']['rsi_7']), 1)

    def test_init_without_ledger_degrades(self):
        """台账 None = 内存降级不炸 (链坏≠链死)"""
        with patch('modules.fusion_ledger.get_fusion_ledger', return_value=None):
            m = FactorICMonitor()
        self.assertEqual(m._factor_returns, {})


class TestRecordPersist(unittest.TestCase):
    """③ 落盘链: record_factor_return → 台账"""

    def setUp(self):
        import tempfile
        self.tmp = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.tmp.close()
        self.led = FusionLedger(self.tmp.name)
        with patch('modules.fusion_ledger.get_fusion_ledger', return_value=self.led):
            self.m = FactorICMonitor()

    def tearDown(self):
        os.unlink(self.tmp.name)
        for suf in ('-wal', '-shm'):
            p = self.tmp.name + suf
            if os.path.exists(p):
                os.unlink(p)

    def test_record_persists(self):
        """record → factor_records 有行 (链尾形: 23:10 链 2 对/票/日)"""
        with patch('modules.fusion_ledger.get_fusion_ledger', return_value=self.led):
            self.m.record_factor_return('sz300620', {'rsi_7': 0.6, 'macd': -0.2},
                                        0.011, date='2026-09-22')
        out = self.led.load_factor_records()
        self.assertIn('rsi_7', out.get('sz300620', {}))
        # 二次同日重入 = 防重 (行数不涨)
        with patch('modules.fusion_ledger.get_fusion_ledger', return_value=self.led):
            self.m.record_factor_return('sz300620', {'rsi_7': 0.6, 'macd': -0.2},
                                        0.011, date='2026-09-22')
        out2 = self.led.load_factor_records()
        self.assertEqual(len(out2['sz300620']['rsi_7']),
                         len(out['sz300620']['rsi_7']))

    def test_duplicate_pair_not_double_counted(self):
        """同 (factor, date) 内存防重: replay 首跑 10 对 + 回填 88 对 交叠不双计
        (09-22 实锤: 双喂曾致 IC 失真 35→0; 台账 OR IGNORE + 内存尾部窗双防)"""
        with patch('modules.fusion_ledger.get_fusion_ledger', return_value=None):
            self.m.record_factor_return('sz300620', {'rsi_7': 0.6, 'macd': -0.2},
                                        0.011, date='2026-09-22')
            self.m.record_factor_return('sz300620', {'rsi_7': 0.6, 'macd': -0.2},
                                        0.011, date='2026-09-22')
            self.assertEqual(len(self.m._factor_returns['sz300620']['rsi_7']), 1)
            # 合法新对 (不同 fval) 正常入
            self.m.record_factor_return('sz300620', {'rsi_7': 0.9}, 0.03,
                                        date='2026-09-22')
            self.assertEqual(len(self.m._factor_returns['sz300620']['rsi_7']), 2)


class TestWeightReduceLoop(unittest.TestCase):
    """④ 衰减→降权闭环 (09-22 实修: 键名错位空转 → 映射+组聚合)"""

    ALERT_KEYS = ('type', 'factor', 'ic_mean', 'n_samples', 'message')

    def _monitor_with_alerts(self, alerts):
        """构造: 5+ 因子缓冲触发告警 + mock scheduler (记录 reduce 调用)"""
        m = FactorICMonitor()
        m._weight_scheduler = MagicMock()
        m.get_ic_decay_alerts_from_returns = MagicMock(return_value=alerts)
        return m

    def _mk_alerts(self, *pairs):
        return [{'type': t, 'factor': f, 'ic_mean': -0.1, 'n_samples': 25,
                 'message': 'x'} for t, f in pairs]

    def _record_trigger(self, m):
        """>5 因子喂入触发告警段 (绕过统计 = 锁判定逻辑)"""
        fac = {f'f{i}': 0.5 + i / 100 for i in range(6)}
        with patch('modules.fusion_ledger.get_fusion_ledger', return_value=None):
            m.record_factor_return('sz300620', fac, 0.02)

    def test_group_trigger(self):
        """technical 组 2 告警 → reduce('technical') 被调 (映射生效)"""
        m = self._monitor_with_alerts(
            self._mk_alerts(('warning', 'rsi_7'), ('critical', 'macd_hist')))
        self._record_trigger(m)
        calls = m._weight_scheduler.reduce_factor_weight.call_args_list
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].args[0], 'technical')
        self.assertEqual(calls[0].args[1], 0.5)  # 含 critical → 取最狠 0.5

    def test_single_alert_no_reduce(self):
        """组内 1 告警 = 不触发 (单因子衰减≠整组, 防误降权)"""
        m = self._monitor_with_alerts(self._mk_alerts(('critical', 'rsi_7')))
        self._record_trigger(m)
        m._weight_scheduler.reduce_factor_weight.assert_not_called()

    def test_unmapped_name_skipped(self):
        """无映射名 = 诚实跳过不炸 (不硬映射)"""
        m = self._monitor_with_alerts(
            self._mk_alerts(('warning', 'zzz_factor'), ('warning', 'qqq_x')))
        self._record_trigger(m)
        m._weight_scheduler.reduce_factor_weight.assert_not_called()

    def test_flag_off(self):
        """DECAY_AUTO_WEIGHT=0 = 闭环停 (可回滚)"""
        m = self._monitor_with_alerts(
            self._mk_alerts(('critical', 'rsi_7'), ('critical', 'macd')))
        os.environ['DECAY_AUTO_WEIGHT'] = '0'
        try:
            self._record_trigger(m)
        finally:
            del os.environ['DECAY_AUTO_WEIGHT']
        m._weight_scheduler.reduce_factor_weight.assert_not_called()

    def test_scheduler_none_chain_alive(self):
        """scheduler None = 链不炸 (record 正常返回)"""
        m = FactorICMonitor()
        m._weight_scheduler = None
        m.get_ic_decay_alerts_from_returns = MagicMock(
            return_value=self._mk_alerts(('critical', 'rsi_7'), ('critical', 'macd')))
        self._record_trigger(m)  # 不抛 = 过

    def test_mapping_table_sanity(self):
        """映射表对 69 因子真键空间抽检 (实锤自 alpha158_calculator)"""
        cases = {'rsi_7': 'technical', 'macd_hist': 'technical',
                 'boll_position': 'technical', 'atr_14': 'technical',
                 'momentum_5': 'momentum', 'reversal_3': 'momentum',
                 'volatility_20': 'volatility', 'amihud_10': 'volatility',
                 'return_skewness': 'volatility', 'up_down_vol_ratio': 'volatility',
                 'volume_ratio_5': 'volume', 'volume_price_corr_10': 'volume',
                 'volume_position_20': 'volume', 'price_position_60': 'technical',
                 'zzz_unknown': None}
        for k, v in cases.items():
            self.assertEqual(_factor_to_group(k), v, f'映射错: {k}')


if __name__ == '__main__':
    unittest.main(verbosity=2)
