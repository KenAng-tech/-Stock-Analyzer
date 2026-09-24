#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
超参优化模块 — Hyperparameter Optimizer

使用 Optuna (贝叶斯优化) 自动搜索最优模型参数。
如果 Optuna 未安装, 回退到网格搜索。

支持的模型:
- LightGBM
- XGBoost
- RandomForest

优化目标: IC (Information Coefficient) 最大化
交叉验证: TimeSeriesSplit (防止前视偏差)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import os
import json
import random
import math
import subprocess
import sys
from sklearn.model_selection import TimeSeriesSplit
from modules.logger import logger

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


# ── 调参快照持久化 (2026-09-08: 02:00 夜间调参链 → 22:00 重训/GET 展示消费) ──
# 凌晨链 (scripts/daily_hyperparam_tune.py) 独立进程跑完调参后写快照;
# 消费方: ① 22:00 ML 重训 (ml_predictor scheduler → ml_training_worker custom_params)
#        ② GET /api/dashboard/hyperparams (前端展示)。
# 7 天时效: 市场状态迁移快于模型迭代, 过期参数不如默认参数 (诚实边界)。
SNAPSHOT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             'data', 'hyperparam_snapshot.json')
SNAPSHOT_MAX_AGE_DAYS = 7


def save_param_snapshot(snapshot: Dict) -> bool:
    """原子写快照 (tmp+rename, 防读写撕裂)"""
    try:
        os.makedirs(os.path.dirname(SNAPSHOT_PATH), exist_ok=True)
        tmp = SNAPSHOT_PATH + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(snapshot, f, ensure_ascii=False, indent=1)
        os.replace(tmp, SNAPSHOT_PATH)
        logger.info(f"[HyperParamOrchestrator] 调参快照已写入: {SNAPSHOT_PATH}")
        return True
    except Exception as e:
        logger.error(f"[HyperParamOrchestrator] 快照写入失败: {e}")
        return False


def load_param_snapshot() -> Optional[Dict]:
    """读快照 → flat {lgb,xgb,rf} 参数 (缺失/损坏/超 7 天 → None)"""
    try:
        if not os.path.exists(SNAPSHOT_PATH):
            return None
        with open(SNAPSHOT_PATH, encoding='utf-8') as f:
            snap = json.load(f)
        gen = datetime.fromisoformat(snap.get('generated_at', ''))
        from datetime import timedelta
        if datetime.now() - gen > timedelta(days=SNAPSHOT_MAX_AGE_DAYS):
            logger.info(f"[HyperParamOrchestrator] 快照已过期 "
                        f"(>{SNAPSHOT_MAX_AGE_DAYS}d, 生成于 {snap.get('generated_at')}), 弃用")
            return None
        best = snap.get('best_params') or {}
        if best and all(v for v in best.values()):
            logger.info(f"[HyperParamOrchestrator] 已加载调参快照 "
                        f"(generated_at={snap.get('generated_at')})")
            return best
        return None
    except Exception as e:
        logger.debug(f"[HyperParamOrchestrator] 快照读取失败: {e}")
        return None


def segment_regimes(klines: List[Dict], lookback: int = 20,
                    trend_threshold: float = 0.06) -> List[str]:
    """按样本切分市场状态 (2026-09-08, per-regime 调参的输入)

    仅用过去 lookback 窗口判断 (无前视偏差, 符合 ml.md 时间分割规则):
    - bull: 窗口收益率 > +trend_threshold (20 日 +6% ≈ 强趋势)
    - bear: 窗口收益率 < -trend_threshold
    - sideways: 其余 (含窗口不足 lookback 的前导样本 — 保守归类)

    Args:
        klines: K 线 dict 列表 (需含 'close')
        lookback: 切分参考窗口天数
        trend_threshold: 趋势阈值

    Returns:
        与 klines 等长的 regime 标签列表 ('bull'/'bear'/'sideways')
    """
    labels: List[str] = []
    for i in range(len(klines)):
        if i < lookback:
            labels.append('sideways')
            continue
        c0 = klines[i - lookback].get('close', 0) or 0
        c1 = klines[i].get('close', 0) or 0
        if c0 <= 0:
            labels.append('sideways')
            continue
        ret = c1 / c0 - 1
        if ret > trend_threshold:
            labels.append('bull')
        elif ret < -trend_threshold:
            labels.append('bear')
        else:
            labels.append('sideways')
    return labels


def optimize_all_isolated(X: np.ndarray, y: np.ndarray,
                          regimes: Optional[List[str]] = None,
                          n_trials: int = 5, embargo_periods: int = 0,
                          timeout: int = 2400) -> Optional[Dict]:
    """在子进程中运行全链调参 (2026-09-08: 主进程 Optuna → SIGSEGV 修复)

    主进程 (Flask 请求线程) 直跑 LGBM/XGB 训练于 2026-09-08 实测 segfault
    杀死整个 API 服务 — 与 07-31 ml_training_worker SIGSEGV 同源 (C 扩展
    在主进程崩溃)。照同一模式: subprocess 隔离 + 子进程内 SIGSEGV handler。

    Returns:
        成功 → {'best_params': {...}, 'regime_params': {...}, 'current_regime': str|None}
        失败 → None (调用方降级, 不在主进程重试)
    """
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '_hyperparam_worker.py')
    input_data = {
        'X': X.tolist(),
        'y': y.tolist(),
        'regimes': regimes,
        'n_trials': n_trials,
        'embargo_periods': embargo_periods,
    }
    # 2026-09-11 根因修复 (连续 3 夜 2400s 全杀实锤): libomp 多线程 barrier 死锁。
    # 链上 worker 三次精确 2400s 被 subprocess 墙钟杀, sample 栈 =
    # XGQuantileDMatrixCreateFromCallback → __kmpc_fork_call → barrier 悬挂
    # (import modules → torch/libgomp 先加载, xgboost 的 libomp 后被污染;
    # 5.4 核满转空转 ≠ 计算慢)。同数据同码在单线程 env 下全链仅 ~7s。
    # 修复 = 子进程 env 注入单线程 (worker 脚本内另有一道 setdefault 防御)。
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OMP_WAIT_POLICY='passive')
    try:
        proc = subprocess.run(
            [sys.executable, script],
            input=json.dumps(input_data),
            capture_output=True, text=True, timeout=timeout,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            env=env,
        )
        if proc.returncode != 0:
            logger.error(f"[HyperParamOrchestrator] 调参子进程失败 (rc={proc.returncode}): "
                         f"{(proc.stderr or '')[:300]}")
            return None
        result = json.loads(proc.stdout.strip().splitlines()[-1])
        if not result.get('success'):
            logger.error(f"[HyperParamOrchestrator] 调参子进程无结果: {result.get('message')}")
            return None
        return result
    except Exception as e:
        logger.error(f"[HyperParamOrchestrator] 调参子进程异常: {e}")
        return None


def selection_bias_check(values: List[float]) -> Dict:
    """selection-bias (winner's curse) 诊断 — Optuna 多选 max 的膨胀校验 (2026-09-11 #30)

    Bailey & López de Prado (2014) deflation: γ = Φ⁻¹(1-1/N)·√(2/N);
    deflated = mean − γ·std。deflated<0 = 跨 trial 方差主导, best_value ≈ 噪声放大
    → selection_suspect。n<2 → skipped (跨 trial 方差无定义, 同 statistical_overfit_check
    low_activity 门先例)。

    观察模式 (09-11 决策): 只入 snapshot + logger, 不接 22:00 消费门控 (先观测后决策)。
    标准 CSCV (IS/OOS 候选矩阵) 缓议: n_trials=3/cv_splits=3 统计功效不足 (论文 N≥10)。
    """
    try:
        from scipy import stats as scipy_stats
    except ImportError:
        return {'skipped': 'scipy_unavailable'}
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n < 2:
        return {'skipped': f'trials {n}/2 (跨 trial 方差无定义)'}
    sr_mean = float(arr.mean())
    sr_std = float(arr.std(ddof=0))
    gamma = float(scipy_stats.norm.ppf(1.0 - 1.0 / n)) * math.sqrt(2.0 / n)
    deflated = sr_mean - gamma * sr_std
    return {
        'n_trials': n,
        'mean_ic': round(sr_mean, 4), 'std_ic': round(sr_std, 4),
        'deflated_ic': round(deflated, 4),
        'selection_suspect': bool(deflated < 0 or sr_mean <= 0),
        'method': 'selection_bias_check_v1 (Bailey-LdP deflation, IC proxy)',
    }


def pbo_cscv_check(matrix: List[List[float]]) -> Dict:
    """CSCV 多候选对比 — 2026-09-11 (09-11 #30 标准形升级)

    matrix = trial × block IC 矩阵 (行=Optuna 候选, 列=TimeSeriesSplit 时间折)。
    PBO = 「IS 最优候选在 OOS 退居后半区」的 IS/OOS 组合占比 (Bailey &
    López de Prado CSCV)。与 selection_bias_check (trial 值分布版) 互补:
    本函数是候选间跨期 rank 对比 = 真·多候选选择偏校验。

    统计功效门 (论文 N≥10): 候选 <8 或 block <4 (combos<10) → skipped
    (同 #30 low_activity 门先例: 无功效不装 PBO 真值, 诚实标记)。
    """
    try:
        import itertools
        M = np.asarray(matrix, dtype=float)
        M = M[np.isfinite(M).all(axis=1)]
        n, b = M.shape
        if n < 8 or b < 4:
            return {'skipped': f'candidates {n}/8 blocks {b}/4 (功效不足, N≥10)'}
        half = b // 2
        combos = list(itertools.combinations(range(b), half))
        if len(combos) < 10:
            return {'skipped': f'combos {len(combos)}/10'}
        failures = 0
        for is_blocks in combos:
            oos_blocks = [j for j in range(b) if j not in is_blocks]
            is_mean = M[:, list(is_blocks)].mean(axis=1)
            oos_mean = M[:, oos_blocks].mean(axis=1)
            best_is = int(np.argmax(is_mean))
            if float(np.mean(oos_mean > oos_mean[best_is])) > 0.5:
                failures += 1   # IS 最优在 OOS 后半区 = 选择偏
        pbo = failures / len(combos)
        return {
            'n_candidates': n, 'n_blocks': b, 'combos': len(combos),
            'pbo': round(pbo, 4),
            'selection_suspect': bool(pbo > 0.5),
            'method': 'pbo_cscv_v1 (CSCV IS/OOS combo rank, LdP)',
        }
    except Exception as e:
        return {'skipped': f'{type(e).__name__}: {e}'}


class LGBMOptimizer:
    """LightGBM 超参优化"""

    def __init__(self, n_trials: int = 30, cv_splits: int = 8, embargo_periods: int = 0):
        self.n_trials = n_trials
        self.cv_splits = cv_splits
        self.embargo_periods = embargo_periods
        self.best_params = None
        self.best_score = -1.0
        self.study = None

    def _split_data(self, X: np.ndarray, y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        使用时间序列分割生成训练/测试折, 支持 embargo 防止边界数据泄露.

        Returns:
            List of (train_idx, test_idx) tuples
        """
        n = len(X)
        tscv = TimeSeriesSplit(n_splits=self.cv_splits)
        folds = []
        for train_idx, test_idx in tscv.split(X):
            # Apply embargo: remove last embargo_periods samples from train
            if self.embargo_periods > 0 and len(train_idx) > self.embargo_periods:
                train_idx = train_idx[:-self.embargo_periods]
            folds.append((train_idx, test_idx))
        return folds

    def _define_params(self, trial) -> Dict:
        """定义 LightGBM 搜索空间"""
        params = {
            # 2026-09-08 预算重设计: n_estimators 上限 1000→300 (500 样本上深树
            # 必过拟合且单 trial 耗时 ×3 — 2400s 预算三轮实测不足的根因)
            'n_estimators': trial.suggest_int('n_estimators', 100, 300) if OPTUNA_AVAILABLE else random.choice([200, 300]),
            'max_depth': trial.suggest_int('max_depth', 3, 12) if OPTUNA_AVAILABLE else random.choice([3, 4, 5, 6, 7, 8]),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True) if OPTUNA_AVAILABLE else random.uniform(0.001, 0.1),
            # 2026-09-08 修: LGBM 真实参数名是 num_leaves — 原 'n_leaves' 被
            # LGBMClassifier 静默忽略 (调了个不存在的参数, 空转搜索维度);
            # 256 leaves on 500 样本 = 深度失控 → 4-64
            'num_leaves': trial.suggest_int('num_leaves', 4, 64) if OPTUNA_AVAILABLE else random.choice([16, 32, 64]),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 200) if OPTUNA_AVAILABLE else random.choice([10, 20, 30, 50, 100]),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0) if OPTUNA_AVAILABLE else random.uniform(0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0) if OPTUNA_AVAILABLE else random.uniform(0.3, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True) if OPTUNA_AVAILABLE else random.uniform(0.001, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True) if OPTUNA_AVAILABLE else random.uniform(0.001, 10),
        }
        return params

    def _objective(self, params: Dict, X: np.ndarray, y: np.ndarray) -> float:
        """优化目标: TimeSeriesSplit CV 的 IC 分数"""
        try:
            import lightgbm as lgb

            folds = self._split_data(X, y)
            ics = []
            self._last_fold_ics = None   # 防 stale: 失败 trial 不得携带上一次矩阵
            for train_idx, test_idx in folds:
                if len(train_idx) < 20 or len(test_idx) < 5:
                    continue

                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                model = lgb.LGBMClassifier(**params, verbose=-1, random_state=42, n_jobs=1)
                model.fit(X_train, y_train)
                proba = model.predict_proba(X_test)

                # 计算 IC: 预测概率与真实标签的排名相关
                # 2026-09-08 修: 3 类标签 {-1,0,1} 时 proba[:,1] 取的是中性类概率 (IC 语义错误),
                # 统一取最后一列 = up 类概率 (2 类时为 +1 类, 3 类时为索引 2 = up, 均正确)
                pred_scores = proba[:, -1] if proba.shape[1] > 1 else proba[:, 0]
                if np.std(pred_scores) > 0 and np.std(y_test) > 0:
                    ic = np.corrcoef(pred_scores, y_test)[0, 1]
                    if not np.isnan(ic):
                        ics.append(ic)

            self._last_fold_ics = ics   # #30+ 多候选: trial×block IC 矩阵喂 PBO
            return float(np.mean(ics)) if ics else 0.0
        except ImportError:
            return 0.0
        except Exception:
            return 0.0

    def optimize(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        优化 LightGBM 参数

        Args:
            X: 特征矩阵 (n_samples, n_features)
            y: 标签 (n_samples,)

        Returns:
            最佳参数字典
        """
        logger.info(f"[LGBMOptimizer] 开始优化, {self.n_trials} trials, {X.shape[0]} 样本")

        if OPTUNA_AVAILABLE:
            self.study = optuna.create_study(
                direction='maximize',
                pruner=optuna.pruners.MedianPruner(n_startup_trials=5)
            )
            # 定义优化目标闭包
            def objective(trial):
                params = self._define_params(trial)
                val = self._objective(params, X, y)
                trial.set_user_attr(
                    'fold_ics', list(getattr(self, '_last_fold_ics', None) or []))
                return val

            try:
                # 2026-09-08 预算重设计: timeout 240→120 (study 总时长预算;
                # 495 样本上 375 次训练实测超 3600s 预算 → 全局+分段 12 study 总时长 ≤1200s)
                self.study.optimize(objective, n_trials=self.n_trials, timeout=120)
                self.best_params = self.study.best_params
                self.best_score = self.study.best_value
                logger.info(f"[LGBMOptimizer] Optuna 优化完成, 最佳 IC: {self.best_score:.4f}")
            except Exception as e:
                logger.error(f"[LGBMOptimizer] Optuna 优化失败: {e}, 回退到网格搜索")
                self._grid_search(X, y)
        else:
            logger.info("[LGBMOptimizer] Optuna 未安装, 使用网格搜索")
            self._grid_search(X, y)

        return self.best_params or self._default_params()

    def _grid_search(self, X: np.ndarray, y: np.ndarray):
        """网格搜索回退"""
        best_score = -1.0
        best_params = {}

        param_grid = {
            'n_estimators': [200, 500],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.01, 0.05],
            'subsample': [0.7, 1.0],
            'colsample_bytree': [0.7, 1.0],
        }

        keys = list(param_grid.keys())
        values = [param_grid[k] for k in keys]

        from itertools import product
        for combo in product(*values):
            params = dict(zip(keys, combo))
            params.update({
                'num_leaves': 31,
                'min_child_samples': 20,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
            })
            score = self._objective(params, X, y)
            if score > best_score:
                best_score = score
                best_params = params

        self.best_params = best_params
        self.best_score = best_score
        logger.info(f"[LGBMOptimizer] 网格搜索完成, 最佳 IC: {best_score:.4f}")

    def _default_params(self) -> Dict:
        return {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
        }


class XGBOptimizer:
    """XGBoost 超参优化"""

    def __init__(self, n_trials: int = 30, cv_splits: int = 2, embargo_periods: int = 0):
        self.n_trials = n_trials
        self.cv_splits = cv_splits
        self.embargo_periods = embargo_periods
        self.best_params = None
        self.best_score = -1.0

    def _split_data(self, X: np.ndarray, y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """使用时间序列分割生成训练/测试折, 支持 embargo 防止边界数据泄露."""
        n = len(X)
        tscv = TimeSeriesSplit(n_splits=self.cv_splits)
        folds = []
        for train_idx, test_idx in tscv.split(X):
            if self.embargo_periods > 0 and len(train_idx) > self.embargo_periods:
                train_idx = train_idx[:-self.embargo_periods]
            folds.append((train_idx, test_idx))
        return folds

    def _define_params(self, trial) -> Dict:
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 300) if OPTUNA_AVAILABLE else random.choice([200, 300]),  # 2026-09-08 预算重设计: 1000→300
            'max_depth': trial.suggest_int('max_depth', 3, 12) if OPTUNA_AVAILABLE else random.choice([3, 4, 5, 6, 7, 8]),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True) if OPTUNA_AVAILABLE else random.uniform(0.001, 0.1),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0) if OPTUNA_AVAILABLE else random.uniform(0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0) if OPTUNA_AVAILABLE else random.uniform(0.3, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10) if OPTUNA_AVAILABLE else random.choice([1, 3, 5, 7]),
            'gamma': trial.suggest_float('gamma', 0, 5) if OPTUNA_AVAILABLE else random.uniform(0, 5),
            # 2026-09-08 修: log=True 要求 low>0, 原 low=0 使全部 trial 抛
            # ValueError → XGB 调参恒失败 (与 n_leaves 空转同批核查发现)
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10, log=True) if OPTUNA_AVAILABLE else random.uniform(0.001, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10, log=True) if OPTUNA_AVAILABLE else random.uniform(0.001, 10),
        }
        return params

    def _objective(self, params: Dict, X: np.ndarray, y: np.ndarray) -> float:
        try:
            import xgboost as xgb
            folds = self._split_data(X, y)
            ics = []
            self._last_fold_ics = None   # 防 stale (同 LGBMOptimizer)
            for train_idx, test_idx in folds:
                if len(train_idx) < 20 or len(test_idx) < 5:
                    continue
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                model = xgb.XGBClassifier(**params, verbosity=0, random_state=42, n_jobs=1)
                model.fit(X_train, y_train)
                proba = model.predict_proba(X_test)
                # 2026-09-08 修: 3 类标签 {-1,0,1} 时 proba[:,1] 取的是中性类概率 (IC 语义错误),
                # 统一取最后一列 = up 类概率 (2 类时为 +1 类, 3 类时为索引 2 = up, 均正确)
                pred_scores = proba[:, -1] if proba.shape[1] > 1 else proba[:, 0]
                if np.std(pred_scores) > 0 and np.std(y_test) > 0:
                    ic = np.corrcoef(pred_scores, y_test)[0, 1]
                    if not np.isnan(ic):
                        ics.append(ic)
            self._last_fold_ics = ics
            return float(np.mean(ics)) if ics else 0.0
        except ImportError:
            return 0.0
        except Exception:
            return 0.0

    def optimize(self, X: np.ndarray, y: np.ndarray) -> Dict:
        logger.info(f"[XGBOptimizer] 开始优化, {self.n_trials} trials")
        if OPTUNA_AVAILABLE:
            self.study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner(n_startup_trials=5))
            def objective(trial):
                params = self._define_params(trial)
                val = self._objective(params, X, y)
                trial.set_user_attr(
                    'fold_ics', list(getattr(self, '_last_fold_ics', None) or []))
                return val
            try:
                self.study.optimize(objective, n_trials=self.n_trials, timeout=120)
                self.best_params = self.study.best_params
                self.best_score = self.study.best_value
            except Exception as e:
                logger.error(f"[XGBOptimizer] Optuna 失败: {e}")
                self._grid_search(X, y)
        else:
            self._grid_search(X, y)
        return self.best_params or self._default_params()

    def _grid_search(self, X, y):
        best_score, best_params = -1.0, {}
        param_grid = {'n_estimators': [200, 500], 'max_depth': [4, 6], 'learning_rate': [0.01, 0.05], 'subsample': [0.8, 1.0]}
        keys = list(param_grid.keys())
        values = [param_grid[k] for k in keys]
        from itertools import product
        for combo in product(*values):
            params = dict(zip(keys, combo))
            params.update({'min_child_weight': 3, 'gamma': 0, 'reg_alpha': 0.1, 'reg_lambda': 1.0})
            score = self._objective(params, X, y)
            if score > best_score:
                best_score, best_params = score, params
        self.best_params, self.best_score = best_params, best_score

    def _default_params(self) -> Dict:
        return {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 3, 'gamma': 0, 'reg_alpha': 0.1, 'reg_lambda': 1.0}


class RFOptimizer:
    """RandomForest 超参优化"""

    def __init__(self, n_trials: int = 20, cv_splits: int = 2, embargo_periods: int = 0):
        self.n_trials = n_trials
        self.cv_splits = cv_splits
        self.embargo_periods = embargo_periods
        self.best_params = None
        self.best_score = -1.0

    def _split_data(self, X: np.ndarray, y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """使用时间序列分割生成训练/测试折, 支持 embargo 防止边界数据泄露."""
        n = len(X)
        tscv = TimeSeriesSplit(n_splits=self.cv_splits)
        folds = []
        for train_idx, test_idx in tscv.split(X):
            if self.embargo_periods > 0 and len(train_idx) > self.embargo_periods:
                train_idx = train_idx[:-self.embargo_periods]
            folds.append((train_idx, test_idx))
        return folds

    def _define_params(self, trial) -> Dict:
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300) if OPTUNA_AVAILABLE else random.choice([100, 200, 300]),  # 2026-09-08 预算重设计: 500→300
            'max_depth': trial.suggest_int('max_depth', 3, 20) if OPTUNA_AVAILABLE else random.choice([3, 5, 8, 10, 15, 20]),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 50) if OPTUNA_AVAILABLE else random.choice([2, 5, 10, 20]),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20) if OPTUNA_AVAILABLE else random.choice([1, 2, 5, 10]),
        }
        max_feat = trial.suggest_categorical('max_features', ['sqrt', 'log2']) if OPTUNA_AVAILABLE else random.choice(['sqrt', 'log2', None])
        params['max_features'] = max_feat
        return params

    def _objective(self, params: Dict, X: np.ndarray, y: np.ndarray) -> float:
        try:
            from sklearn.ensemble import RandomForestClassifier
            folds = self._split_data(X, y)
            ics = []
            self._last_fold_ics = None   # 防 stale (同 LGBMOptimizer)
            for train_idx, test_idx in folds:
                if len(train_idx) < 20 or len(test_idx) < 5:
                    continue
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                model = RandomForestClassifier(**params, random_state=42, n_jobs=1)
                model.fit(X_train, y_train)
                proba = model.predict_proba(X_test)
                # 2026-09-08 修: 3 类标签 {-1,0,1} 时 proba[:,1] 取的是中性类概率 (IC 语义错误),
                # 统一取最后一列 = up 类概率 (2 类时为 +1 类, 3 类时为索引 2 = up, 均正确)
                pred_scores = proba[:, -1] if proba.shape[1] > 1 else proba[:, 0]
                if np.std(pred_scores) > 0 and np.std(y_test) > 0:
                    ic = np.corrcoef(pred_scores, y_test)[0, 1]
                    if not np.isnan(ic):
                        ics.append(ic)
            self._last_fold_ics = ics
            return float(np.mean(ics)) if ics else 0.0
        except Exception:
            return 0.0

    def optimize(self, X: np.ndarray, y: np.ndarray) -> Dict:
        logger.info(f"[RFOptimizer] 开始优化, {self.n_trials} trials")
        if OPTUNA_AVAILABLE:
            self.study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner(n_startup_trials=5))
            def objective(trial):
                params = self._define_params(trial)
                val = self._objective(params, X, y)
                trial.set_user_attr(
                    'fold_ics', list(getattr(self, '_last_fold_ics', None) or []))
                return val
            try:
                self.study.optimize(objective, n_trials=self.n_trials, timeout=60)
                self.best_params = self.study.best_params
                self.best_score = self.study.best_value
            except Exception as e:
                logger.error(f"[RFOptimizer] Optuna 失败: {e}")
                self._grid_search(X, y)
        else:
            self._grid_search(X, y)
        return self.best_params or self._default_params()

    def _grid_search(self, X, y):
        best_score, best_params = -1.0, {}
        param_grid = {'n_estimators': [100, 300], 'max_depth': [5, 10], 'min_samples_split': [5, 10]}
        keys = list(param_grid.keys())
        values = [param_grid[k] for k in keys]
        from itertools import product
        for combo in product(*values):
            params = dict(zip(keys, combo))
            params.update({'min_samples_leaf': 2, 'max_features': 'sqrt'})
            score = self._objective(params, X, y)
            if score > best_score:
                best_score, best_params = score, params
        self.best_params, self.best_score = best_params, best_score

    def _default_params(self) -> Dict:
        return {'n_estimators': 100, 'max_depth': 5, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'sqrt'}


class HyperParamOrchestrator:
    """超参优化编排器"""

    def __init__(self, n_trials: int = 12, embargo_periods: int = 0,
                 min_regime_samples: int = 120, cv_splits: int = 8):
        self.n_trials = n_trials
        self.embargo_periods = embargo_periods
        self.min_regime_samples = min_regime_samples
        self.lgb_optimizer = LGBMOptimizer(n_trials, cv_splits=cv_splits,
                                           embargo_periods=embargo_periods)
        self.xgb_optimizer = XGBOptimizer(n_trials, cv_splits=cv_splits,
                                          embargo_periods=embargo_periods)
        self.rf_optimizer = RFOptimizer(n_trials, cv_splits=cv_splits,
                                        embargo_periods=embargo_periods)
        self._best_params = {}
        self._regime_params = {}   # {regime: {model: params}} — per-regime 调参结果
        self._selection_reports: List[Dict] = []   # #30: 每 study 的 selection-bias 诊断
        self._pbo_reports: List[Dict] = []         # #30+: 每 study 的 CSCV 多候选 PBO

    def _selection_check(self, opt):
        """Optuna study trials → selection_bias_check (grid 回退无 study = 静默跳过)"""
        try:
            study = getattr(opt, 'study', None)
            if study is None:
                return
            vals = [t.value for t in study.trials
                    if t.state.name == 'COMPLETE' and t.value is not None]
            if len(vals) >= 2:
                self._selection_reports.append(selection_bias_check(vals))
        except Exception as e:
            logger.debug(f"[selection-bias] study trials 采集失败: {e}")
        try:
            # 多候选 PBO (CSCV): trial×block IC 矩阵 (行数≥8 且块长≥4 才喂)
            matrix = [list(t.user_attrs.get('fold_ics') or [])
                      for t in study.trials
                      if t.state.name == 'COMPLETE'
                      and len(t.user_attrs.get('fold_ics') or []) >= 4]
            if len(matrix) >= 8:
                rep = pbo_cscv_check(matrix)
                if 'skipped' not in rep:
                    self._pbo_reports.append(rep)
                    if rep.get('selection_suspect'):
                        logger.warning(
                            f"[PBO] {opt.__class__.__name__} PBO={rep['pbo']} "
                            f"(N={rep['n_candidates']}×{rep['n_blocks']}块, "
                            f"IS 最优在 OOS 后半区占比>50%) — 选择偏疑似 (观察模式)")
        except Exception as e:
            logger.debug(f"[PBO] trial 矩阵采集失败: {e}")

    def selection_summary(self) -> Dict:
        """#30 观察汇总 (02:00 snapshot 键 + logger 行)"""
        evaluated = [r for r in self._selection_reports if 'skipped' not in r]
        pbo_eval = [r for r in self._pbo_reports if 'skipped' not in r]
        return {
            'studies': len(self._selection_reports),
            'evaluated': len(evaluated),
            'suspect_count': sum(1 for r in evaluated if r.get('selection_suspect')),
            'deflated_ic_min': min((r['deflated_ic'] for r in evaluated), default=None),
            'reports': self._selection_reports[:6],   # 全局3 + per-regime 前3 (全量在 log)
            'method': 'selection_bias_check_v1',
            'pbo': {
                'studies': len(self._pbo_reports),
                'evaluated': len(pbo_eval),
                'suspect': sum(1 for r in pbo_eval if r.get('selection_suspect')),
                'pbo_max': max((r['pbo'] for r in pbo_eval), default=None),
                'method': 'pbo_cscv_v1',
            },
        }

    def optimize_all(self, X: np.ndarray, y: np.ndarray,
                     regimes: Optional[List[str]] = None) -> Dict:
        """优化所有模型 (全局 + 可选 per-regime 分段)

        Args:
            X, y: 训练特征/标签 (时间对齐)
            regimes: 可选 per-sample regime 标签 (segment_regimes 产物)。
                提供时额外做 per-regime 调参, 双守卫防"噪声调参":
                ① 分段样本 < min_regime_samples → 跳过 (CV 折太小, 调参=拟合噪声)
                ② 分段 OOS IC ≤ 0 → 不存 (参数无泛化力, 不配占用 trials 预算)
        """
        logger.info("[HyperParamOrchestrator] 开始全模型优化 (全局)")
        for name in ('lgb', 'xgb', 'rf'):
            opt = {'lgb': self.lgb_optimizer, 'xgb': self.xgb_optimizer,
                   'rf': self.rf_optimizer}[name]
            self._best_params[name] = opt.optimize(X, y)
            self._selection_check(opt)   # #30: 每 study 收 trial 矩阵 (观察链)
        logger.info(f"[HyperParamOrchestrator] 全模型优化完成: {list(self._best_params.keys())}")

        if regimes is not None and len(regimes) == len(X):
            half_trials = max(3, self.n_trials // 2)
            opts = {'lgb': self.lgb_optimizer, 'xgb': self.xgb_optimizer,
                    'rf': self.rf_optimizer}
            for regime in ('bull', 'bear', 'sideways'):
                mask = [i for i, r in enumerate(regimes) if r == regime]
                if len(mask) < self.min_regime_samples:
                    logger.info(
                        f"[HyperParamOrchestrator] {regime} 段样本 {len(mask)} < "
                        f"{self.min_regime_samples}, 跳过 per-regime (CV 折过小, 调参=拟合噪声)")
                    continue
                X_r, y_r = X[mask], y[mask]
                result = {}
                for name, opt in opts.items():
                    saved = opt.n_trials
                    opt.n_trials = half_trials
                    try:
                        params = opt.optimize(X_r, y_r)
                        self._selection_check(opt)   # #30: per-regime study 同采集
                        if params and getattr(opt, 'best_score', -1.0) > 0:
                            result[name] = params
                        else:
                            logger.info(
                                f"[HyperParamOrchestrator] {regime}.{name} "
                                f"OOS IC≤0 或空参数, 不存储")
                    finally:
                        opt.n_trials = saved
                if result:
                    self._regime_params[regime] = result
                    logger.info(
                        f"[HyperParamOrchestrator] {regime} 段 per-regime 调参完成: "
                        f"{sorted(result.keys())}")

        return self._best_params

    def get_best_params(self, model_name: str = None,
                        regime: str = None) -> Dict:
        """获取最佳参数 (regime 优先, 回退全局)

        get_best_params('lgb') → 全局参数 (旧行为不变)
        get_best_params('lgb', regime='bull') → bull 段参数, 无则回退全局
        """
        if model_name:
            if regime and regime in self._regime_params:
                p = self._regime_params[regime].get(model_name)
                if p:
                    return p
            return self._best_params.get(model_name, {})
        return self._best_params

    def apply_to_predictor(self, predictor, current_regime: str = None) -> bool:
        """将调参结果应用到 MLPredictor (2026-09-08: pass 桩 → 真实链)

        写入 predictor.custom_params — 由模型工厂 (_make_lightgbm/_make_xgboost/
        _make_random_forest) 与子进程 trainer 脚本 (ml_training_worker) 合并消费,
        下次模型重训时生效。主进程内不做模型重建 (07-31 SIGSEGV 边界:
        C 扩展模型只在子进程训练)。

        current_regime 指定且 per-regime 参数存在 → 优先该 regime 段调出的参数,
        否则回退全局。无调参结果时返回 False + warning (不再谎报"已应用")。
        """
        try:
            source = None
            if current_regime and current_regime in self._regime_params:
                source = self._regime_params[current_regime]
                logger.info(
                    f"[HyperParamOrchestrator] 使用 {current_regime} 段参数 "
                    f"(per-regime 调参)")
            if not source:
                source = self._best_params
            params = {k: dict(v) for k, v in (source or {}).items() if v}
            if not params:
                logger.warning("[HyperParamOrchestrator] 无调参结果可用, 跳过应用")
                return False
            if not hasattr(predictor, 'custom_params'):
                predictor.custom_params = {}
            predictor.custom_params.update(params)
            logger.info(
                f"[HyperParamOrchestrator] 参数已应用到 predictor: "
                f"{sorted(params.keys())} (regime={current_regime})")
            return True
        except Exception as e:
            logger.error(f"[HyperParamOrchestrator] 应用参数失败: {e}")
            return False

    def save_study(self, filepath: str):
        """保存优化结果到 JSON"""
        try:
            data = {
                'best_params': self._best_params,
                'regime_params': self._regime_params,
                'timestamp': datetime.now().isoformat(),
            }
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"[HyperParamOrchestrator] 优化结果已保存: {filepath}")
        except Exception as e:
            logger.error(f"[HyperParamOrchestrator] 保存失败: {e}")


# 全局实例
hyperparam_optimizer = HyperParamOrchestrator(n_trials=30, embargo_periods=0)
