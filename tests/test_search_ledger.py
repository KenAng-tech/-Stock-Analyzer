#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""search_ledger + run_artifact 单元测试 (P0-3, 2026-09-14)

覆盖:
1. search_ledger — record/count/recent/summary 基本语义 + since_days 过滤
2. prune 保留策略 (旧行删除, 新行保留)
3. 并发 10 线程 record 不丢不炸 (WAL + busy_timeout 多写者纪律)
4. record_trial 内部异常绝不外抛 (坏路径 / 不可序列化 metric)
5. run_artifact — start_run 工件落盘 / check 完整性判定 (qlib 思想) /
   save_pred sha256+shape / finish ok|fail / list_runs

测试隔离: ledger 用临时 db_path; run_artifact monkeypatch RUNS_ROOT 到临时目录。
"""

import json
import os
import pickle
import shutil
import sqlite3
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules import search_ledger as sl
from modules import run_artifact as ra


class TestSearchLedger(unittest.TestCase):
    """账本基本语义 + 防御式降级"""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = os.path.join(self.tmp, 'ledger.db')

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _age(self, ts_iso):
        """把最新一行 ts 改旧 (模拟历史 trial)"""
        conn = sqlite3.connect(self.db)
        try:
            conn.execute('UPDATE trials SET ts = ? WHERE id = '
                         '(SELECT MAX(id) FROM trials)', (ts_iso,))
            conn.commit()
        finally:
            conn.close()

    def test_record_and_count(self):
        """record_trial 返回 True, trial_count 按 source 过滤"""
        self.assertTrue(sl.record_trial('hyperparam', {'lr': 0.05}, 'cv_ic', 0.12,
                                        run_id='r1', note='lgb/global', db_path=self.db))
        self.assertTrue(sl.record_trial('other', {'x': 1}, 'auc', 0.7, db_path=self.db))
        self.assertEqual(sl.trial_count(db_path=self.db), 2)
        self.assertEqual(sl.trial_count(source='hyperparam', db_path=self.db), 1)
        self.assertEqual(sl.trial_count(source='missing', db_path=self.db), 0)

    def test_since_days_filter(self):
        """since_days 按 ts 过滤 (旧行不计入近窗, 仍计入全历史)"""
        sl.record_trial('s', {}, 'm', 1.0, db_path=self.db)
        self._age('2020-01-01T00:00:00')
        self.assertEqual(sl.trial_count(source='s', since_days=90, db_path=self.db), 0)
        self.assertEqual(sl.trial_count(source='s', db_path=self.db), 1)

    def test_recent_trials_order_and_parse(self):
        """recent_trials 新→旧 + limit + params_json 解析回 dict"""
        for i in range(3):
            sl.record_trial('s', {'i': i}, 'cv_ic', float(i), db_path=self.db)
        rows = sl.recent_trials(source='s', limit=2, db_path=self.db)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]['params']['i'], 2)   # 最新在前
        self.assertEqual(rows[0]['metric_value'], 2.0)
        self.assertEqual(rows[1]['params']['i'], 1)

    def test_nonfinite_metric_stored_null(self):
        """NaN/inf 指标存 NULL (不污染 MAX/收缩统计), 仍返回 True"""
        self.assertTrue(sl.record_trial('s', {}, 'm', float('nan'), db_path=self.db))
        self.assertTrue(sl.record_trial('s', {}, 'm', float('inf'), db_path=self.db))
        rows = sl.recent_trials(source='s', db_path=self.db)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(r['metric_value'] is None for r in rows))

    def test_source_summary(self):
        """每 source 的 trials 数/最优值/最近时间"""
        sl.record_trial('hyperparam', {}, 'cv_ic', 0.1, db_path=self.db)
        sl.record_trial('hyperparam', {}, 'cv_ic', 0.3, db_path=self.db)
        sl.record_trial('rdagent', {}, 'ic', 0.05, db_path=self.db)
        summary = {s['source']: s for s in sl.source_summary(db_path=self.db)}
        self.assertEqual(summary['hyperparam']['trials'], 2)
        self.assertEqual(summary['hyperparam']['best_value'], 0.3)
        self.assertEqual(summary['hyperparam']['best_metric_name'], 'cv_ic')
        self.assertIsNotNone(summary['hyperparam']['last_ts'])
        self.assertEqual(summary['rdagent']['trials'], 1)

    def test_prune(self):
        """prune(keep_days) 删旧留新, 返回删除数"""
        sl.record_trial('s', {}, 'm', 1.0, db_path=self.db)   # 旧
        self._age('2020-01-01T00:00:00')
        sl.record_trial('s', {}, 'm', 2.0, db_path=self.db)   # 新
        deleted = sl.prune(keep_days=90, db_path=self.db)
        self.assertEqual(deleted, 1)
        self.assertEqual(sl.trial_count(db_path=self.db), 1)
        remaining = sl.recent_trials(source='s', db_path=self.db)
        self.assertEqual(remaining[0]['metric_value'], 2.0)

    def test_concurrent_records_no_loss(self):
        """10 线程 × 20 条并发 record: 不丢 (count=200) 不炸 (无异常逃逸)"""
        errors = []

        def worker(tid):
            try:
                for i in range(20):
                    ok = sl.record_trial('conc', {'t': tid, 'i': i}, 'cv_ic',
                                         tid + i / 100.0, db_path=self.db)
                    if not ok:
                        errors.append(f'write failed t{tid} i{i}')
            except Exception as e:  # record_trial 承诺不外抛
                errors.append(f'{type(e).__name__}: {e}')

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        self.assertEqual(sl.trial_count(source='conc', db_path=self.db), 200)

    def test_record_trial_never_raises_bad_path(self):
        """坏路径 (父路径是普通文件) → False, 不外抛"""
        blocker = os.path.join(self.tmp, 'blocker')
        with open(blocker, 'w') as f:
            f.write('x')
        bad = os.path.join(blocker, 'sub', 'ledger.db')
        self.assertFalse(sl.record_trial('s', {'a': 1}, 'm', 0.5, db_path=bad))
        # 查询 API 同样防御式降级
        self.assertEqual(sl.trial_count(db_path=bad), 0)
        self.assertEqual(sl.recent_trials(db_path=bad), [])
        self.assertEqual(sl.source_summary(db_path=bad), [])
        self.assertEqual(sl.prune(db_path=bad), 0)

    def test_record_trial_never_raises_bad_metric(self):
        """metric_value 无法 float 化 → False, 不外抛"""
        self.assertFalse(sl.record_trial('s', {}, 'm', object(), db_path=self.db))
        self.assertEqual(sl.trial_count(db_path=self.db), 0)


class TestRunArtifact(unittest.TestCase):
    """qlib trainer 工件纪律: conf/metrics/pred/model/status + check/list"""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._orig_root = ra.RUNS_ROOT
        ra.RUNS_ROOT = os.path.join(self.tmp, 'runs')

    def tearDown(self):
        ra.RUNS_ROOT = self._orig_root
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_start_run_writes_conf(self):
        """start_run 建目录 + conf.json 冻结配置"""
        rc = ra.start_run('hyperparam', {'n_trials': 12, 'stock': 'sz300620'})
        self.assertTrue(os.path.isdir(rc.path))
        with open(os.path.join(rc.path, 'conf.json'), encoding='utf-8') as f:
            conf = json.load(f)
        self.assertEqual(conf['conf']['n_trials'], 12)
        self.assertEqual(conf['name'], 'hyperparam')
        self.assertEqual(conf['run_id'], rc.run_id)

    def test_check_completeness(self):
        """check(need=['conf','metrics']): metrics 缺失 → None; 补齐 → 最近完整 run"""
        rc1 = ra.start_run('hyperparam', {'run': 1})
        self.assertIsNone(ra.check('hyperparam', need=('conf', 'metrics')))
        self.assertIsNotNone(ra.check('hyperparam', need=('conf',)))
        rc1.save_metrics({'best_params': {'lgb': {'num_leaves': 32}}})
        found = ra.check('hyperparam', need=('conf', 'metrics'))
        self.assertIsNotNone(found)
        self.assertEqual(found['run_id'], rc1.run_id)
        # 更新的完整 run 优先返回
        rc2 = ra.start_run('hyperparam', {'run': 2})
        rc2.save_metrics({'best_params': {}})
        found2 = ra.check('hyperparam', need=('conf', 'metrics'))
        self.assertEqual(found2['run_id'], rc2.run_id)

    def test_check_unknown_need_is_none(self):
        """未知工件名 → 视为缺失 → None (防御式, 不抛)"""
        ra.start_run('hyperparam', {'run': 1})
        self.assertIsNone(ra.check('hyperparam', need=('nonexistent',)))

    def test_save_pred_sha_and_shape(self):
        """save_pred: pickle 落盘 + meta.json 记 sha256(64hex)+shape, 可反序列化"""
        import numpy as np
        rc = ra.start_run('pred_chain', {})
        arr = np.arange(12, dtype=np.float64).reshape(3, 4)
        self.assertTrue(rc.save_pred(arr))
        meta = json.load(open(os.path.join(rc.path, 'meta.json'), encoding='utf-8'))
        self.assertEqual(meta['pred']['shape'], [3, 4])
        self.assertEqual(len(meta['pred']['sha256']), 64)
        with open(os.path.join(rc.path, 'pred.pkl'), 'rb') as f:
            loaded = pickle.load(f)
        self.assertTrue((loaded == arr).all())

    def test_save_pred_from_path_no_load(self):
        """save_pred(路径): 流式拷贝, sha256 与源文件一致 (不反序列化)"""
        import hashlib
        src = os.path.join(self.tmp, 'src_pred.pkl')
        with open(src, 'wb') as f:
            f.write(b'\x80\x05 large-payload-bytes' * 100)
        src_sha = hashlib.sha256(open(src, 'rb').read()).hexdigest()
        rc = ra.start_run('pred_chain', {})
        self.assertTrue(rc.save_pred(src))
        meta = json.load(open(os.path.join(rc.path, 'meta.json'), encoding='utf-8'))
        self.assertEqual(meta['pred']['sha256'], src_sha)
        self.assertIsNone(meta['pred']['shape'])

    def test_save_pred_missing_path_false(self):
        """save_pred 不存在的路径 → False 不外抛"""
        rc = ra.start_run('pred_chain', {})
        self.assertFalse(rc.save_pred(os.path.join(self.tmp, 'nope.pkl')))

    def test_save_model_and_check(self):
        """save_model 拷贝到 model/ + check(need=['model']) 判定"""
        model_file = os.path.join(self.tmp, 'model.lgb')
        with open(model_file, 'wb') as f:
            f.write(b'model-bytes')
        rc = ra.start_run('train', {})
        self.assertTrue(rc.save_model(model_file))
        self.assertTrue(os.path.isfile(os.path.join(rc.path, 'model', 'model.lgb')))
        self.assertIsNone(ra.check('other_name', need=('model',)))
        found = ra.check('train', need=('conf', 'model'))
        self.assertIsNotNone(found)

    def test_finish_ok_and_fail(self):
        """finish 落盘 status.json (ok / fail+error) 并并入 meta.json"""
        rc_ok = ra.start_run('chain', {})
        self.assertTrue(rc_ok.finish('ok'))
        st = json.load(open(os.path.join(rc_ok.path, 'status.json'), encoding='utf-8'))
        self.assertEqual(st['status'], 'ok')

        rc_fail = ra.start_run('chain', {})
        self.assertTrue(rc_fail.finish('fail', error='ValueError: boom'))
        st2 = json.load(open(os.path.join(rc_fail.path, 'status.json'), encoding='utf-8'))
        self.assertEqual(st2['status'], 'fail')
        self.assertIn('boom', st2['error'])
        meta = json.load(open(os.path.join(rc_fail.path, 'meta.json'), encoding='utf-8'))
        self.assertEqual(meta['status']['status'], 'fail')

    def test_list_runs_order_limit_status(self):
        """list_runs 新→旧 + limit + 未 finish 显示 running"""
        rc1 = ra.start_run('chain', {'i': 1})
        rc1.finish('ok')
        rc2 = ra.start_run('chain', {'i': 2})   # 未 finish
        runs = ra.list_runs('chain', limit=10)
        self.assertEqual(len(runs), 2)
        self.assertEqual(runs[0]['run_id'], rc2.run_id)      # 新→旧
        self.assertEqual(runs[0]['status'], 'running')
        self.assertEqual(runs[1]['status'], 'ok')
        self.assertEqual(len(ra.list_runs('chain', limit=1)), 1)
        self.assertEqual(ra.list_runs('nonexistent'), [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
