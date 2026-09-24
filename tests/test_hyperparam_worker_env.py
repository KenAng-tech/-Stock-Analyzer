#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""hyperparam 02:00 链死锁修复回归 (2026-09-11)

锁形态: 连续 3 夜 2400s 全杀根因 = libomp 多线程 barrier 死锁 (sample 栈:
XGQuantileDMatrixCreateFromCallback → __kmpc_fork_call → barrier 悬挂;
5.4 核满转空转). 修复 = 调参子进程 env 注入 OMP_NUM_THREADS=1 +
worker 脚本顶部 setdefault 双防御。

教训 (09-11 四度重演): unittest 主进程 in-process Optuna×LGBM/XGB = 07-31/09-08
SIGSEGV — 本套测试只 mock subprocess 验 env 注入形态, 不真跑 Optuna。
"""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


class TestWorkerEnvInjection(unittest.TestCase):
    """optimize_all_isolated 必须向子进程注入单线程 env (死锁绕开)"""

    def test_env_injects_single_thread(self):
        """子进程 env 收到 OMP/MKL 单线程键 = 死锁防御在位"""
        from modules import hyperparam_optimizer as ho
        captured = {}

        def fake_run(cmd, **kw):
            captured['cmd'] = cmd
            captured.update(kw)

            class R:
                returncode = 1
                stdout = ''
                stderr = ''
            return R()

        X = np.random.RandomState(42).randn(40, 3)
        y = np.random.RandomState(1).choice([0, 1, 2], 40)
        with mock.patch.object(ho.subprocess, 'run', side_effect=fake_run):
            out = ho.optimize_all_isolated(X, y, regimes=None, n_trials=1,
                                           timeout=5)
        self.assertIsNone(out)  # rc=1 → None (降级, 不重试)
        self.assertIn('env', captured, "subprocess.run 缺 env kwarg — "
                                       "OMP 死锁防御已断")
        env = captured['env']
        self.assertEqual(env.get('OMP_NUM_THREADS'), '1')
        self.assertEqual(env.get('MKL_NUM_THREADS'), '1')
        self.assertEqual(env.get('OMP_WAIT_POLICY'), 'passive')

    def test_worker_script_sets_defense(self):
        """worker 脚本在 import numpy/sklearn 之前设 OMP env (第二道防御)"""
        path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), 'modules', '_hyperparam_worker.py')
        src = open(path, encoding='utf-8').read()
        i_def = src.find("setdefault('OMP_NUM_THREADS'")
        i_np = src.find('import numpy')
        self.assertGreater(i_def, -1, 'worker 顶部 OMP 防御已删')
        self.assertGreater(i_np, i_def, 'OMP env 必须在 numpy import 之前')


if __name__ == '__main__':
    unittest.main(verbosity=2)
