# -*- coding:utf-8 -*-
"""P2-A BPS 合成层测试 (2026-09-22, arXiv 2510.05739→2510.07180 落地)

链背景: 2510.07180 真身 = Bayesian Portfolio Optimization by Predictive
Synthesis (Kato et al., 非 belief polarization — 09-22 侦察更正)。5002 化 =
model_seed.jsonl 成熟 (dir,conf)×真收益 → per-model η EWMA (log q,
δ=0.05, q=conf 对/1-conf 错) → softmax(η) 模型权重 → UDE.decide 聚合 +
BL views (待 data 成熟 09-26 拍板) — 与 accuracy_feed 独立 (第二锁
consumed_bps 不共享, 观察档双跑对比, 翻转待数据成熟)。

锁形 (expected = 独立事实来源, 非 tautological):
  ① 权重分化: moirai 2/2 对 (conf0.8/0.7) vs gnn 0/2 (conf0.6/0.5) —
     η/softmax 按 paper 式 3 独立手算 + 方向断言 (acc 2/2 vs 0/2)
  ② 便宜 conf 不奖励: 同对但 0.6 < 0.9 → 低 conf 低权 (acc binary 同分,
     BPS density 区分 = 非 tautological 断言核心)
  ③ consumed 第二锁: 二次 calibrate = consumed 全跳 (不稀释)
  ④ min_groups 门 (默认 20, Brier n<20 同形): <20 组 skipped + 不写观察档
  ⑤ 默认喂权重不污染 UDE (feed_weights 默认 True → 显式 False 测)
  ⑥ 未成熟 seed 诚实跳 (honest ≠ 断链)
  ⑦ 端到端 smoke: flag 开链跑通 (降级链不崩 + 权重形锁)
"""
import json
import math
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.unified_decision_engine import UnifiedDecisionEngine  # noqa: E402
from modules.bps_synthesis import bps_calibrate  # noqa: E402

_NOW = datetime.now()


def _seed(ude, specs, ts=None):
    """seed.jsonl 构造 (形同真 model_seed.jsonl: ts/code/model/dir/conf)"""
    ts = ts or (_NOW - timedelta(days=7)).strftime('%Y-%m-%dT14:00:00')
    with open(ude._seed_path, 'w', encoding='utf-8') as f:
        for code, models in specs.items():
            for m, d, c in models:
                f.write(json.dumps({'ts': ts, 'code': code, 'model': m,
                                    'dir': d, 'conf': c}) + '\n')


def _kl(up):
    """K 线: seed ts 日 (7d 前) close=10 → 成熟后 up=11/down=9 (P1-a ② 同形)"""
    dates = [(_NOW - timedelta(days=d)).strftime('%Y-%m-%d')
             for d in range(9, -1, -1)]
    dates += [(_NOW + timedelta(days=d)).strftime('%Y-%m-%d')
              for d in range(1, 13)]
    return ([{'date': d, 'close': 10.0} for d in dates[:10]]
            + [{'date': d, 'close': 11.0 if up else 9.0} for d in dates[10:]])


class TestBpsWeights(unittest.TestCase):
    """bps_calibrate: seed 成熟对 → η → softmax 权重 → 观察档"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.ude = UnifiedDecisionEngine()
        self.ude._seed_path = os.path.join(self._tmp.name, 'model_seed.jsonl')
        self.ude._bps_consumed = os.path.join(
            self._tmp.name, 'consumed_bps.jsonl')
        self.ude._bps_weights_path = os.path.join(
            self._tmp.name, 'bps_weights.jsonl')
        self.ude._metrics_path = os.path.join(self._tmp.name, 'm.json')

    def tearDown(self):
        self._tmp.cleanup()

    def test_seed_absent_skipped(self):
        """⑥ seed 未建 → 诚实 skipped (链坏≠链死, 今晚 23:10 起积累)"""
        r = bps_calibrate(self.ude)
        self.assertIn('skipped', r)
        self.assertFalse(os.path.exists(self.ude._bps_weights_path))

    def test_unripe_seed_skip(self):
        """⑥ 未成熟 (<3d) 诚实跳: 空跑不崩"""
        today = _NOW.strftime('%Y-%m-%dT14:00:00')
        _seed(self.ude, {'sz300620': [('moirai', 'buy', .8)]}, ts=today)
        r = bps_calibrate(self.ude, min_groups=1)
        self.assertIn('skipped', r)

    def test_weights_diff_and_formula(self):
        """① 分化 + 公式: moirai 2/2 对 vs gnn 0/2 — 权重按 paper 式 3 独立手算"""
        _seed(self.ude, {'sz300620': [('moirai', 'buy', .8), ('gnn', 'sell', .6)],
                         'sh688981': [('moirai', 'sell', .7), ('gnn', 'buy', .5)]})
        kl = {'sz300620': _kl(True), 'sh688981': _kl(False)}
        with patch('modules.data_fetcher.StockDataFetcher') as F:
            F.return_value.get_kline_data.side_effect = \
                lambda code, *a, **k: kl.get(code)
            r = bps_calibrate(self.ude, min_groups=1)
        self.assertEqual(r['fed_pairs'], 4, '4 模型对全成熟')
        self.assertEqual(r['groups'], 2, '2 决策组')
        # 式 3 独立链 (δ=0.05): η=(1-δ)·η_prev+δ·log q; 两对先 300620 后 688981
        eta_m = 0.95 * (0.05 * math.log(0.8)) + 0.05 * math.log(0.7)
        eta_g = 0.95 * (0.05 * math.log(0.4)) + 0.05 * math.log(0.5)
        e1, e2 = math.exp(eta_m), math.exp(eta_g)
        w = r['weights']
        self.assertAlmostEqual(w['moirai'], e1 / (e1 + e2), places=6,
                               msg='softmax(η) 式 3 实现锁')
        self.assertAlmostEqual(sum(w.values()), 1.0, places=6,
                               msg='权重归一')
        self.assertGreater(w['moirai'], w['gnn'],
                           'acc 2/2 vs 0/2 = 独立事实分化')
        self.assertTrue(os.path.exists(self.ude._bps_consumed),
                        'consumed 第二锁持久')
        self.assertTrue(os.path.exists(self.ude._bps_weights_path),
                        '观察档落盘')

    def test_second_run_all_consumed(self):
        """③ consumed 第二锁: 二次 calibrate 全跳 (seed 不重喂 = η 不稀释)"""
        _seed(self.ude, {'sz300620': [('moirai', 'buy', .8), ('gnn', 'sell', .6)],
                         'sh688981': [('moirai', 'sell', .7), ('gnn', 'buy', .5)]})
        kl = {'sz300620': _kl(True), 'sh688981': _kl(False)}
        with patch('modules.data_fetcher.StockDataFetcher') as F:
            F.return_value.get_kline_data.side_effect = \
                lambda code, *a, **k: kl.get(code)
            r1 = bps_calibrate(self.ude, min_groups=1)
            r2 = bps_calibrate(self.ude, min_groups=1)
        self.assertIn('skipped', r2, 'consumed 全跳 = 不稀释')
        with open(self.ude._bps_weights_path, encoding='utf-8') as f:
            self.assertEqual(len(f.readlines()), 1,
                             '观察档 append 形 (seed 文件层不重 = 链坏≠链死)')

    def test_cheap_conf_not_rewarded(self):
        """② BPS≠acc binary: 同对 0.6 conf 低权 (acc 形两者同分 = density 区分锁)"""
        _seed(self.ude, {'sz300620': [('techA', 'buy', .9), ('techB', 'buy', .6)],
                         'sh688981': [('techA', 'buy', .9), ('techB', 'buy', .6)]})
        kl = {'sz300620': _kl(True), 'sh688981': _kl(True)}
        with patch('modules.data_fetcher.StockDataFetcher') as F:
            F.return_value.get_kline_data.side_effect = \
                lambda code, *a, **k: kl.get(code)
            r = bps_calibrate(self.ude, min_groups=1)
        self.assertGreater(r['weights']['techA'], r['weights']['techB'],
                           'conf0.6 对 = log(0.6) 重罚, 非 binary 同分')

    def test_min_groups_gate(self):
        """④ min_groups 门默认 20: 2 组 skipped + 不写档 (Brier n<20 同形)"""
        _seed(self.ude, {'sz300620': [('moirai', 'buy', .8)],
                         'sh688981': [('gnn', 'sell', .6)]})
        kl = {'sz300620': _kl(True), 'sh688981': _kl(False)}
        with patch('modules.data_fetcher.StockDataFetcher') as F:
            F.return_value.get_kline_data.side_effect = \
                lambda code, *a, **k: kl.get(code)
            r = bps_calibrate(self.ude)  # 默认 min_groups=20
        self.assertIn('skipped', r, '成熟对<20 组诚实 skipped')
        self.assertFalse(os.path.exists(self.ude._bps_consumed),
                         '门跳 = 不消费 (下次仍可喂)')

    def test_feed_off_default_no_pollution(self):
        """⑤ 默认喂权重 (23:10 设计) + 显式 False = UDE.weights 不污染 (回滚锁)"""
        before = dict(self.ude.weights)
        _seed(self.ude, {'sz300620': [('moirai', 'buy', .8), ('gnn', 'sell', .6)],
                         'sh688981': [('moirai', 'sell', .7), ('gnn', 'buy', .5)]})
        kl = {'sz300620': _kl(True), 'sh688981': _kl(False)}
        with patch('modules.data_fetcher.StockDataFetcher') as F:
            F.return_value.get_kline_data.side_effect = \
                lambda code, *a, **k: kl.get(code)
            r = bps_calibrate(self.ude, min_groups=1, feed_weights=False)
        self.assertEqual(self.ude.weights, before,
                         'feed_weights=False = 只观察不喂链 (回滚形)')
        self.assertIn('weights', r, '观察输出仍带 weights')

    def test_obs_record_carries_acc_snapshot(self):
        """⑧ 分叉证据自积累 (S1): 观察档记录 = {acc: 现链快照, bps: 新权重}
        双链对比 09-27 可回读; acc 快照 expected = 手工注入 dict (非 tautological)"""
        custom = {m: 1.0 for m in self.ude.AVAILABLE_MODELS}
        custom['gnn'] = 2.0
        custom['moirai'] = 0.5
        self.ude.weights = custom   # 手工注入 acc 链现值 (独立事实源)
        _seed(self.ude, {'sz300620': [('moirai', 'buy', .8), ('gnn', 'sell', .6)],
                         'sh688981': [('moirai', 'sell', .7), ('gnn', 'buy', .5)]})
        kl = {'sz300620': _kl(True), 'sh688981': _kl(False)}
        with patch('modules.data_fetcher.StockDataFetcher') as F:
            F.return_value.get_kline_data.side_effect = \
                lambda code, *a, **k: kl.get(code)
            r = bps_calibrate(self.ude, min_groups=1, feed_weights=False)
        with open(self.ude._bps_weights_path, encoding='utf-8') as f:
            rec = json.loads(f.readlines()[-1])
        self.assertEqual(rec['acc'], {m: round(v, 6) for m, v in custom.items()},
                         'acc 快照 = 喂入时 ude.weights 原样 (双链对比原料)')
        self.assertEqual(set(rec['bps']), set(r['weights']),
                         'bps 段键集 = 本组出票模型 (非全模型)')

    def test_end_to_end_smoke(self):
        """⑦ 端到端: 喂权重→decide 降级链跑通 + 权重形锁 (非 tautological)"""
        _seed(self.ude, {'sz300620': [('moirai', 'buy', .8), ('gnn', 'sell', .6)],
                         'sh688981': [('moirai', 'sell', .7), ('gnn', 'buy', .5)]})
        kl = {'sz300620': _kl(True), 'sh688981': _kl(False)}
        with patch('modules.data_fetcher.StockDataFetcher') as F:
            F.return_value.get_kline_data.side_effect = \
                lambda code, *a, **k: kl.get(code)
            r = bps_calibrate(self.ude, min_groups=1, feed_weights=True)
        self.assertGreater(r['weights']['moirai'], r['weights']['gnn'])
        # 喂后 decide: 降级链跑通 (dev 无 8080/数据源 = 诚实降级形非断链)
        d = self.ude.decide('sz300620')
        self.assertIsNotNone(d.direction)
        self.assertGreater(sum(self.ude.weights.values()), 0.5,
                           'softmax 喂后权重形在 (和≤1, 无 NaN/清零)')


class TestBpsWeightConsumer(unittest.TestCase):
    """S2 (2026-09-23): 持久消费端 — bps 权重重启不丢 (23:10 段7 观察→喂料闭环)"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.ude = UnifiedDecisionEngine()
        self.ude._seed_path = os.path.join(self._tmp.name, 'model_seed.jsonl')
        self.ude._bps_consumed = os.path.join(
            self._tmp.name, 'consumed_bps.jsonl')
        self.ude._bps_weights_path = os.path.join(
            self._tmp.name, 'bps_weights.jsonl')
        self.ude._metrics_path = os.path.join(self._tmp.name, 'm.json')

    def tearDown(self):
        self._tmp.cleanup()

    def test_load_bps_merge_math(self):
        """merge 语义: 出票模型取 bps 值, 未出票保 acc 原值, 归一 (独立手算)"""
        from modules.bps_synthesis import load_bps_weights
        base = {'gnn': 1.0, 'moirai': 1.5, 'drl': 1.0}
        bps = {'gnn': 0.7, 'moirai': 0.2}
        w, info = load_bps_weights(base, bps)
        self.assertAlmostEqual(w['gnn'], 0.7 / 1.9, places=6)
        self.assertAlmostEqual(w['moirai'], 0.2 / 1.9, places=6)
        self.assertAlmostEqual(w['drl'], 1.0 / 1.9, places=6,
                               msg='未出票模型保留原权 (非全替换=decide 萎缩锁)')

    def test_init_smoke_bps_file(self):
        """flag=1 + 观察档存在 → 新实例自动载 (重启持久链, 09-27 无需人工盯)"""
        from modules.unified_decision_engine import UnifiedDecisionEngine
        seed_dir = os.path.join(self._tmp.name, 'runs', 'consensus_ab')
        os.makedirs(seed_dir)
        rec = {'ts': (_NOW - timedelta(days=1)).strftime('%Y-%m-%dT23:11:11'),
               'pairs': 2, 'groups': 2,
               'acc': {m: 1.0 for m in ('gnn', 'moirai')},
               'bps': {'gnn': 0.7, 'moirai': 0.2}}
        with open(os.path.join(seed_dir, 'bps_weights.jsonl'), 'w') as f:
            f.write(json.dumps(rec) + '\n')
        import modules.unified_decision_engine as ud_mod
        with patch.dict(os.environ, {'UDE_BPS_WEIGHT': '1'}), \
                patch.object(ud_mod, 'PROJECT_ROOT', self._tmp.name):
            ude = UnifiedDecisionEngine()
        # smoke 形锁 (test⑦ 同宗): flag=1 链跑通 ≠ 断链 (09-27 拍板观察档形)
        ude_off = UnifiedDecisionEngine()   # env 已退 = 默认链基线 (无 auto 载)
        self.assertNotEqual(ude.weights, ude_off.weights,
                            'flag=1 链跑通: 自动载 ≠ 默认权重')
        self.assertAlmostEqual(sum(ude.weights.values()), 1.0, places=3,
                               msg='merge 后和=1 非清零 (decide 萎缩锁)')

    def test_feed_path_carries_bps_over_acc(self):
        """calibrate feed=True → merge 入 UDE (同语义单源, 非 dict 全替换)"""
        from modules.bps_synthesis import bps_calibrate as bc
        _seed(self.ude, {'sz300620': [('moirai', 'buy', .8), ('gnn', 'sell', .6)],
                         'sh688981': [('moirai', 'sell', .7), ('gnn', 'buy', .5)]})
        kl = {'sz300620': _kl(True), 'sh688981': _kl(False)}
        before = dict(self.ude.weights)
        with patch('modules.data_fetcher.StockDataFetcher') as F:
            F.return_value.get_kline_data.side_effect = \
                lambda code, *a, **k: kl.get(code)
            bc(self.ude, min_groups=1, feed_weights=True)
        self.assertEqual(len(self.ude.AVAILABLE_MODELS), len(self.ude.weights),
                         'feed 后模型数不缩 (merge 非全替换 = decide 萎缩锁)')
        self.assertAlmostEqual(sum(self.ude.weights.values()), 1.0, places=3)
        self.assertLess(self.ude.weights['moirai'] / self.ude.weights['gnn'],
                        before['moirai'] / before['gnn'],
                        'gnn 0/2 被 bps 拉近 moirai (密度≠binary = feed 真生效)')


class TestAutoFlip(unittest.TestCase):
    """S5 auto_flip 门控 (2026-09-23 残留②): UDE_BPS_WEIGHT 三档 0/1/'2' —
    '2'=auto = 先观察 ≥3 条观察记录 → 第 4 次触发 auto feed。门 = 证据先行
    门控后动: 记录攒不满 = 永不 auto feed (安全向, 非提前 flip)。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.ude = UnifiedDecisionEngine()
        self.ude._seed_path = os.path.join(self._tmp.name, 'model_seed.jsonl')
        self.ude._bps_consumed = os.path.join(
            self._tmp.name, 'consumed_bps.jsonl')
        self.ude._bps_weights_path = os.path.join(
            self._tmp.name, 'bps_weights.jsonl')
        self.ude._metrics_path = os.path.join(self._tmp.name, 'm.json')

    def tearDown(self):
        self._tmp.cleanup()

    def _seed_obs(self, n):
        """观察档预填 n 条最小记录 (含 bps 键 = 消费端可解析形)"""
        with open(self.ude._bps_weights_path, 'w', encoding='utf-8') as f:
            for _ in range(n):
                f.write(json.dumps({'ts': _NOW.strftime('%Y-%m-%dT23:11:11'),
                                    'pairs': 2, 'groups': 2,
                                    'acc': {}, 'bps': {'gnn': 0.7, 'moirai': 0.2}})
                                    + '\n')

    def test_obs_count_pure(self):
        """① bps_obs_count 纯函数: 缺=0 / 空=0 / 3 行=3 (expected 手写常量)"""
        from modules.bps_synthesis import bps_obs_count
        self.assertEqual(bps_obs_count(self.ude._bps_weights_path), 0,
                         '文件不存在 = 0 (链未建非断链)')
        open(self.ude._bps_weights_path, 'w').close()
        self.assertEqual(bps_obs_count(self.ude._bps_weights_path), 0,
                         '空文件 = 0')
        self._seed_obs(3)
        self.assertEqual(bps_obs_count(self.ude._bps_weights_path), 3)

    def test_auto_gate_below_no_feed(self):
        """② 观察记录 2 条 <3 门 → 'auto' = 只观察不喂 (weights 原样 + 档 append 到 3)"""
        self._seed_obs(2)
        before = dict(self.ude.weights)
        _seed(self.ude, {'sz300620': [('moirai', 'buy', .8), ('gnn', 'sell', .6)],
                         'sh688981': [('moirai', 'sell', .7), ('gnn', 'buy', .5)]})
        kl = {'sz300620': _kl(True), 'sh688981': _kl(False)}
        with patch('modules.data_fetcher.StockDataFetcher') as F:
            F.return_value.get_kline_data.side_effect = \
                lambda code, *a, **k: kl.get(code)
            r = bps_calibrate(self.ude, min_groups=1, feed_weights='auto')
        self.assertIn('weights', r, '观察段照常产出 (门只挡喂入不挡观察)')
        self.assertEqual(self.ude.weights, before,
                         '2 记录 <3 门 = 只观察 (证据不足不喂活链)')
        with open(self.ude._bps_weights_path, encoding='utf-8') as f:
            self.assertEqual(len(f.readlines()), 3, '本轮观察仍 append (2→3)')

    def test_auto_gate_at_feed(self):
        """③ 观察记录 3 条 = 第 4 次触发 → 'auto' = auto feed 生效 (merge 形)"""
        self._seed_obs(3)
        before = dict(self.ude.weights)
        _seed(self.ude, {'sz300620': [('moirai', 'buy', .8), ('gnn', 'sell', .6)],
                         'sh688981': [('moirai', 'sell', .7), ('gnn', 'buy', .5)]})
        kl = {'sz300620': _kl(True), 'sh688981': _kl(False)}
        with patch('modules.data_fetcher.StockDataFetcher') as F:
            F.return_value.get_kline_data.side_effect = \
                lambda code, *a, **k: kl.get(code)
            r = bps_calibrate(self.ude, min_groups=1, feed_weights='auto')
        self.assertIn('weights', r)
        self.assertNotEqual(self.ude.weights, before,
                           '3 记录 ≥3 门 = 第 4 次 auto feed 生效')
        self.assertLess(self.ude.weights['moirai'] / self.ude.weights['gnn'],
                        before['moirai'] / before['gnn'],
                        '喂后 = merge 形非清零 (gnn 0/2 被拉近, 同 ① 形)')
        self.assertAlmostEqual(sum(self.ude.weights.values()), 1.0, places=3,
                               msg='merge 归一锁')

    def test_init_flag2_autoload(self):
        """④ flag='2' → UDE.__init__ 也自动载观察档 (重启持久 = 与 '1' 同形)"""
        import modules.unified_decision_engine as ud_mod
        seed_dir = os.path.join(self._tmp.name, 'runs', 'consensus_ab')
        os.makedirs(seed_dir)
        rec = {'ts': (_NOW - timedelta(days=1)).strftime('%Y-%m-%dT23:11:11'),
               'pairs': 2, 'groups': 2,
               'acc': {m: 1.0 for m in ('gnn', 'moirai')},
               'bps': {'gnn': 0.7, 'moirai': 0.2}}
        with open(os.path.join(seed_dir, 'bps_weights.jsonl'), 'w') as f:
            f.write(json.dumps(rec) + '\n')
        with patch.dict(os.environ, {'UDE_BPS_WEIGHT': '2'}), \
                patch.object(ud_mod, 'PROJECT_ROOT', self._tmp.name):
            ude = UnifiedDecisionEngine()
        ude_off = UnifiedDecisionEngine()   # env 退 = 默认链基线
        self.assertNotEqual(ude.weights, ude_off.weights,
                            "flag='2' 重启自动载 ≠ 默认权重")
        self.assertAlmostEqual(sum(ude.weights.values()), 1.0, places=3)


if __name__ == '__main__':
    unittest.main(verbosity=2)
