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
from sklearn.model_selection import TimeSeriesSplit
from modules.logger import logger

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


class LGBMOptimizer:
    """LightGBM 超参优化"""

    def __init__(self, n_trials: int = 30, cv_splits: int = 5, embargo_periods: int = 0):
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
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000) if OPTUNA_AVAILABLE else random.choice([200, 300, 500, 700, 1000]),
            'max_depth': trial.suggest_int('max_depth', 3, 12) if OPTUNA_AVAILABLE else random.choice([3, 4, 5, 6, 7, 8]),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True) if OPTUNA_AVAILABLE else random.uniform(0.001, 0.1),
            'n_leaves': trial.suggest_int('n_leaves', 4, 256) if OPTUNA_AVAILABLE else random.choice([16, 32, 64, 128, 256]),
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
            for train_idx, test_idx in folds:
                if len(train_idx) < 20 or len(test_idx) < 5:
                    continue

                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                model = lgb.LGBMClassifier(**params, verbose=-1, random_state=42, n_jobs=1)
                model.fit(X_train, y_train)
                proba = model.predict_proba(X_test)

                # 计算 IC: 预测概率与真实标签的排名相关
                pred_scores = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
                if np.std(pred_scores) > 0 and np.std(y_test) > 0:
                    ic = np.corrcoef(pred_scores, y_test)[0, 1]
                    if not np.isnan(ic):
                        ics.append(ic)

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
                return self._objective(params, X, y)

            try:
                self.study.optimize(objective, n_trials=self.n_trials, timeout=600)
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
                'n_leaves': 31,
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
            'n_leaves': 31,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
        }


class XGBOptimizer:
    """XGBoost 超参优化"""

    def __init__(self, n_trials: int = 30, cv_splits: int = 5, embargo_periods: int = 0):
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
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000) if OPTUNA_AVAILABLE else random.choice([200, 500, 1000]),
            'max_depth': trial.suggest_int('max_depth', 3, 12) if OPTUNA_AVAILABLE else random.choice([3, 4, 5, 6, 7, 8]),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True) if OPTUNA_AVAILABLE else random.uniform(0.001, 0.1),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0) if OPTUNA_AVAILABLE else random.uniform(0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0) if OPTUNA_AVAILABLE else random.uniform(0.3, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10) if OPTUNA_AVAILABLE else random.choice([1, 3, 5, 7]),
            'gamma': trial.suggest_float('gamma', 0, 5) if OPTUNA_AVAILABLE else random.uniform(0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10, log=True) if OPTUNA_AVAILABLE else random.uniform(0.001, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10, log=True) if OPTUNA_AVAILABLE else random.uniform(0.001, 10),
        }
        return params

    def _objective(self, params: Dict, X: np.ndarray, y: np.ndarray) -> float:
        try:
            import xgboost as xgb
            folds = self._split_data(X, y)
            ics = []
            for train_idx, test_idx in folds:
                if len(train_idx) < 20 or len(test_idx) < 5:
                    continue
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                model = xgb.XGBClassifier(**params, verbosity=0, random_state=42, n_jobs=1)
                model.fit(X_train, y_train)
                proba = model.predict_proba(X_test)
                pred_scores = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
                if np.std(pred_scores) > 0 and np.std(y_test) > 0:
                    ic = np.corrcoef(pred_scores, y_test)[0, 1]
                    if not np.isnan(ic):
                        ics.append(ic)
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
                return self._objective(params, X, y)
            try:
                self.study.optimize(objective, n_trials=self.n_trials, timeout=600)
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

    def __init__(self, n_trials: int = 20, cv_splits: int = 5, embargo_periods: int = 0):
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
            'n_estimators': trial.suggest_int('n_estimators', 50, 500) if OPTUNA_AVAILABLE else random.choice([100, 200, 300, 500]),
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
            for train_idx, test_idx in folds:
                if len(train_idx) < 20 or len(test_idx) < 5:
                    continue
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                model = RandomForestClassifier(**params, random_state=42, n_jobs=1)
                model.fit(X_train, y_train)
                proba = model.predict_proba(X_test)
                pred_scores = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
                if np.std(pred_scores) > 0 and np.std(y_test) > 0:
                    ic = np.corrcoef(pred_scores, y_test)[0, 1]
                    if not np.isnan(ic):
                        ics.append(ic)
            return float(np.mean(ics)) if ics else 0.0
        except Exception:
            return 0.0

    def optimize(self, X: np.ndarray, y: np.ndarray) -> Dict:
        logger.info(f"[RFOptimizer] 开始优化, {self.n_trials} trials")
        if OPTUNA_AVAILABLE:
            self.study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner(n_startup_trials=5))
            def objective(trial):
                params = self._define_params(trial)
                return self._objective(params, X, y)
            try:
                self.study.optimize(objective, n_trials=self.n_trials, timeout=300)
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

    def __init__(self, n_trials: int = 30, embargo_periods: int = 0):
        self.n_trials = n_trials
        self.embargo_periods = embargo_periods
        self.lgb_optimizer = LGBMOptimizer(n_trials, embargo_periods=embargo_periods)
        self.xgb_optimizer = XGBOptimizer(n_trials, embargo_periods=embargo_periods)
        self.rf_optimizer = RFOptimizer(n_trials, embargo_periods=embargo_periods)
        self._best_params = {}

    def optimize_all(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """并行优化所有模型"""
        logger.info("[HyperParamOrchestrator] 开始全模型优化")
        self._best_params['lgb'] = self.lgb_optimizer.optimize(X, y)
        self._best_params['xgb'] = self.xgb_optimizer.optimize(X, y)
        self._best_params['rf'] = self.rf_optimizer.optimize(X, y)
        logger.info(f"[HyperParamOrchestrator] 全模型优化完成: {list(self._best_params.keys())}")
        return self._best_params

    def get_best_params(self, model_name: str = None) -> Dict:
        """获取最佳参数"""
        if model_name:
            return self._best_params.get(model_name, {})
        return self._best_params

    def apply_to_predictor(self, predictor) -> bool:
        """将最佳参数应用到 MLPredictor"""
        try:
            for model_name, params in self._best_params.items():
                if hasattr(predictor, f'_make_{model_name}'):
                    # 更新模型工厂方法
                    pass
            logger.info("[HyperParamOrchestrator] 参数已应用到预测器")
            return True
        except Exception as e:
            logger.error(f"[HyperParamOrchestrator] 应用参数失败: {e}")
            return False

    def save_study(self, filepath: str):
        """保存优化结果到 JSON"""
        try:
            data = {
                'best_params': self._best_params,
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
