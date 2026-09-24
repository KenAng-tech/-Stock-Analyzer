# -*- coding:utf-8 -*-
"""P1-a performative 反馈链测试 (2026-09-22, arXiv 2412.10545 CB-PDD 落地)

链背景: UDE.decide 的 18 模型票 (model_votes) 原零落盘零 replay (09-22 段 5
死链摘除 = 诚实标记; spy 实验实锤 drift/accuracy 喂料 0 入 = 自检失明第 4 例)。
P1-a = 补两段链: decide 尾模型票 seed 落盘 (model_seed.jsonl, 23:10 scan 链每晚
自产 5 票×~15 模型) + 23:10 replay 链 accuracy_feed (≥3 交易日成熟 seed → 真收益
对账 → update_accuracy 喂入 = 死入口复活, 每模型 EWMA acc→softmax 自动降权)。

锁形 (expected 全独立事实来源):
  ① seed 落盘: 双写双落 (文件层无重, 防重在 consumed) + 五键全 (ts/code/model/dir/conf)
  ② 对账链: 成熟 seed 配对喂入 → 期望 acc 手算 (300620: buy↑=1.0 sell↑=0.0;
     688981: sell↓=1.0 buy↓=0.0) + 0-acc 模型权重降档 (softmax 实锤)
  ③ 防重喂: 连续两次 feed = consumed.jsonl 第二锁跳过 (无重喂 = acc 不稀释)
  ④ flag 回滚: UDE_ACC_FEED=0 → 全链跳过 (落链可关, 09-22 极化门同形)
  ⑤ 未成熟 seed 诚实跳: 空跑不崩 (honest ≠ 断链)
"""
import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.unified_decision_engine import UnifiedDecisionEngine  # noqa: E402

_NOW = datetime.now()


class TestSeedLog(unittest.TestCase):
    """① seed 落盘链 (_seed_log)"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.ude = UnifiedDecisionEngine()
        self.ude._seed_path = os.path.join(self._tmp.name, 'model_seed.jsonl')
        self.ude._feed_path = os.path.join(self._tmp.name, 'seed_consumed.jsonl')
        self.ude._metrics_path = os.path.join(self._tmp.name, 'm.json')  # 不污染真 metrics

    def tearDown(self):
        self._tmp.cleanup()

    def test_seed_log_roundtrip(self):
        """① 双写双落 + 五键全 + 值完整"""
        self.ude._seed_log('sz300620', '2026-09-15T14:00:00',
                           {'moirai': ('buy', 0.8), 'gnn': ('sell', 0.6)})
        self.ude._seed_log('sz300620', '2026-09-15T14:00:00',
                           {'moirai': ('buy', 0.8), 'gnn': ('sell', 0.6)})
        with open(self.ude._seed_path, encoding='utf-8') as f:
            lines = [json.loads(x) for x in f if x.strip()]
        self.assertEqual(len(lines), 4, 'seed 文件层不去重 (防重喂在消费层)')
        self.assertEqual(set(lines[0]), {'ts', 'code', 'model', 'dir', 'conf'})
        self.assertEqual(lines[0]['dir'], 'buy')
        self.assertAlmostEqual(lines[0]['conf'], 0.8)


class TestAccuracyFeed(unittest.TestCase):
    """②③④⑤ accuracy_feed (成熟 seed 对账喂入)"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.ude = UnifiedDecisionEngine()
        self.ude._seed_path = os.path.join(self._tmp.name, 'model_seed.jsonl')
        self.ude._feed_path = os.path.join(self._tmp.name, 'seed_consumed.jsonl')
        self.ude._metrics_path = os.path.join(self._tmp.name, 'm.json')  # 不污染真 metrics
        # 成熟 seed (7d 前): 300620 moirai=buy/gnn=sell; 688981 moirai=sell/gnn=buy
        old = (_NOW - timedelta(days=7)).strftime('%Y-%m-%dT14:00:00')
        with open(self.ude._seed_path, 'w', encoding='utf-8') as f:
            for code, specs in (('sz300620', (('moirai', 'buy', .8),
                                               ('gnn', 'sell', .6))),
                                ('sh688981', (('moirai', 'sell', .7),
                                               ('gnn', 'buy', .5)))):
                for m, d, c in specs:
                    f.write(json.dumps({'ts': old, 'code': code,
                                        'model': m, 'dir': d,
                                        'conf': c}) + '\n')

    def tearDown(self):
        self._tmp.cleanup()

    def _kl(self, up):
        """K线: 决策日 (7d 前) 收盘 10 → 末根 up=11 / down=9 (±10%)"""
        dates = [(_NOW - timedelta(days=d)).strftime('%Y-%m-%d')
                 for d in range(9, -1, -1)]
        dates += [(_NOW + timedelta(days=d)).strftime('%Y-%m-%d')
                  for d in range(1, 13)]
        return ([{'date': d, 'close': 10.0} for d in dates[:10]]
                + [{'date': d, 'close': 11.0 if up else 9.0} for d in dates[10:]])

    def test_feed_pairing_and_weights(self):
        """② 对账链全跑: acc 期望手算 + 0-acc 模型 softmax 降权实锤"""
        kl = {'sz300620': self._kl(True), 'sh688981': self._kl(False)}
        with patch('modules.data_fetcher.StockDataFetcher') as F:
            F.return_value.get_kline_data.side_effect = \
                lambda code, *a, **k: kl.get(code)
            r = self.ude.accuracy_feed()
        self.assertEqual(r.get('fed_pairs'), 4)
        acc = self.ude._accuracy_history
        self.assertAlmostEqual(acc['sz300620']['moirai'], 1.0)  # buy ∧ up ✓
        self.assertAlmostEqual(acc['sz300620']['gnn'], 0.0)      # sell ∧ up ✗
        self.assertAlmostEqual(acc['sh688981']['moirai'], 1.0)  # sell ∧ down ✓
        self.assertAlmostEqual(acc['sh688981']['gnn'], 0.0)      # buy ∧ down ✗
        self.assertLess(self.ude.weights['gnn'], self.ude.weights['moirai'],
                        '0-acc 模型权重降档 (softmax acc×10, 22026:1)')

    def test_feed_dedup(self):
        """③ 连续两次 feed = consumed 第二锁跳过 (acc 不稀释)"""
        kl = {'sz300620': self._kl(True), 'sh688981': self._kl(False)}
        with patch('modules.data_fetcher.StockDataFetcher') as F:
            F.return_value.get_kline_data.side_effect = \
                lambda code, *a, **k: kl.get(code)
            r1 = self.ude.accuracy_feed()
            snap = {c: dict(m) for c, m in self.ude._accuracy_history.items()}
            r2 = self.ude.accuracy_feed()
        self.assertEqual(r1['fed_pairs'], 4)
        self.assertIn('skipped', r2, '二次喂=consumed 键全跳 (第二锁)')
        self.assertEqual(self.ude._accuracy_history, snap,
                         'acc 不变 = 无稀释')
        self.assertTrue(os.path.exists(self.ude._feed_path),
                        'consumed 持久 (跨重启防重)')

    def test_flag_off_rollback(self):
        """④ UDE_ACC_FEED=0 全链跳过 (落链可关回滚形)"""
        kl = {'sz300620': self._kl(True), 'sh688981': self._kl(False)}
        with patch.dict(os.environ, {'UDE_ACC_FEED': '0'}):
            with patch('modules.data_fetcher.StockDataFetcher') as F:
                F.return_value.get_kline_data.side_effect = \
                    lambda code, *a, **k: kl.get(code)
                r = self.ude.accuracy_feed()
        self.assertIn('skipped', r)
        self.assertFalse(os.path.exists(self.ude._feed_path))

    def test_unripe_seed_skip(self):
        """⑤ 未成熟 seed (<3d) 诚实跳: 空跑不崩 (honest 非断链)"""
        today = _NOW.strftime('%Y-%m-%dT14:00:00')
        with open(self.ude._seed_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps({'ts': today, 'code': 'sz300620', 'model': 'gnn',
                               'dir': 'buy', 'conf': 0.6}) + '\n')
        with patch('modules.data_fetcher.StockDataFetcher') as F:
            F.return_value.get_kline_data.side_effect = \
                lambda code, *a, **k: self._kl(True)
            r = self.ude.accuracy_feed()
        self.assertIn('skipped', r, '未成熟 = 诚实 skipped (非崩非假喂)')


if __name__ == '__main__':
    unittest.main(verbosity=2)
