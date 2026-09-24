#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
ML 训练 Worker — 使用 subprocess 实现进程隔离

在独立子进程中运行模型训练，完全隔离主进程的 C 扩展内存空间，
避免 SIGSEGV 崩溃。

用法:
    result = run_training(X, y, feature_names, model_dir)
"""

import subprocess
import os
import sys
import json
import tempfile
import numpy as np
from datetime import datetime
from typing import Dict, List, Any


# ── 训练脚本（独立文件，避免 -c 参数长度/转义问题） ──
_TRAINER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_trainer_process.py')


def _ensure_trainer_script():
    """确保训练脚本文件存在（避免内联脚本的转义/编码问题）"""
    if os.path.exists(_TRAINER_SCRIPT):
        return
    content = r'''#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""ML 训练进程 — 由 ml_training_worker 在独立子进程中启动"""
import os, sys, json, pickle, warnings, traceback, signal
# 确保项目根目录在 sys.path 中（脚本位于 modules/ 下时 sys.path[0] 是 modules/）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from datetime import datetime

warnings.filterwarnings('ignore')

# ── 轻量 logger（避免 modules.logger 触发完整导入链） ──
_logger = None
def get_logger():
    global _logger
    if _logger is None:
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            handlers=[logging.StreamHandler(sys.stdout)],
        )
        _logger = logging.getLogger('TrainingWorker')
    return _logger

# ── SIGSEGV 保护 ──
def _sigsegv_handler(signum, frame):
    print(json.dumps({
        'success': False,
        'message': '训练进程 SIGSEGV (C 扩展崩溃)',
        'error': 'segmentation violation at signal %d' % signum,
    }))
    sys.exit(129)

signal.signal(signal.SIGSEGV, _sigsegv_handler)
signal.signal(signal.SIGABRT, _sigsegv_handler)

warnings.filterwarnings('ignore')

# ── SIGSEGV 保护: 子进程内 C 扩展崩溃时捕获信号 ──
def _sigsegv_handler(signum, frame):
    print(json.dumps({
        'success': False,
        'message': '训练进程 SIGSEGV (C 扩展崩溃)',
        'error': 'segmentation violation at signal %d' % signum,
    }))
    sys.exit(129)

signal.signal(signal.SIGSEGV, _sigsegv_handler)
signal.signal(signal.SIGABRT, _sigsegv_handler)


def main():
    # 从 stdin 读取训练数据
    input_data = json.loads(sys.stdin.read())
    X = np.array(input_data['X'])
    y = np.array(input_data['y'])
    feature_names = input_data['feature_names']
    model_dir = input_data['model_dir']

    logger = get_logger()

    n = len(X)
    # 确保 y 长度与 X 对齐（prepare_features_batch 可能比 labels 少几行）
    if len(y) > n:
        y = y[:n]

    # ── 标签映射: 原始 [-1, 0, 1] → [0, 1, 2] (sklearn 要求连续 class) ──
    def map_label(val):
        return 0 if val == -1 else (2 if val == 1 else 1)

    y_mapped = np.array([map_label(v) for v in y])

    # ── Purged K-Fold CV ──
    n_splits = min(5, max(3, n // 50))
    fold_size = n // n_splits
    emb_size = max(1, int(fold_size * 0.05))
    folds = []
    for fold in range(n_splits):
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < n_splits - 1 else n
        test_idx = list(range(test_start, test_end))
        # Embargo: 排除测试集前方 emb_size 样本（后方无数据可不排除）
        train_idx = [i for i in range(n)
                     if i not in test_idx
                     and i >= test_start - emb_size]
        folds.append((train_idx, test_idx))

    # ── 训练 Level-0 模型 ──
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

    # 2026-09-08: 合并调参结果 (主进程 HyperParamOrchestrator 注入, 空=默认参数)
    custom_params = input_data.get('custom_params') or {}

    level0_models = {
        'rf': RandomForestClassifier(**{'n_estimators': 200, 'max_depth': 10,
                                        'class_weight': 'balanced', 'random_state': 42,
                                        'n_jobs': -1, **custom_params.get('rf', {})}),
        # GradientBoosting = xgboost 未安装时的占位 (不合并调参参数, kwargs 不兼容)
        'xgb': GradientBoostingClassifier(n_estimators=200, max_depth=5,
                                           learning_rate=0.1, random_state=42),
    }

    # 尝试 LightGBM
    try:
        import lightgbm as lgb
        level0_models['lgb'] = lgb.LGBMClassifier(
            **{'n_estimators': 200, 'max_depth': 8, 'learning_rate': 0.1,
               'class_weight': 'balanced', 'random_state': 42, 'verbose': -1,
               'n_jobs': -1, 'force_col_wise': True, **custom_params.get('lgb', {})})
    except ImportError:
        pass

    # 尝试 XGBoost（覆盖 GradientBoosting）
    try:
        import xgboost as xgb
        level0_models['xgb'] = xgb.XGBClassifier(
            **{'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.1,
               'use_label_encoder': False, 'eval_metric': 'logloss',
               'random_state': 42, 'n_jobs': -1, 'verbosity': 0,
               **custom_params.get('xgb', {})})
    except ImportError:
        pass

    logger.info(f"[TrainingWorker] 可用模型: {list(level0_models.keys())}")

    oof_preds = np.zeros((n, len(level0_models)))
    model_ic = {}
    fold_metrics = []
    model_names = list(level0_models.keys())

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        if len(train_idx) < 30 or len(test_idx) < 5:
            continue

        # 检查训练集是否包含至少 2 个标签类别
        train_labels_unique = set(y_mapped[train_idx])
        if len(train_labels_unique) < 2:
            logger.warning(f"[TrainingWorker] Fold {fold_idx+1}: 训练集只有 {len(train_labels_unique)} 个类别，跳过")
            fold_metrics.append({
                'fold': fold_idx + 1, 'status': 'skipped_insufficient_classes',
                'train_size': len(train_idx), 'test_size': len(test_idx),
            })
            continue

        test_labels_unique = set(y_mapped[test_idx])

        fold_metrics.append({
            'fold': fold_idx + 1,
            'train_size': len(train_idx),
            'test_size': len(test_idx),
            'train_classes': sorted(train_labels_unique),
            'test_classes': sorted(test_labels_unique),
        })

        X_train, X_test = X[train_idx], X[test_idx]
        y_train_m = y_mapped[train_idx]
        y_test_orig = y[test_idx]

        for col_idx, (name, model) in enumerate(level0_models.items()):
            try:
                model.fit(X_train, y_train_m)
                preds = model.predict(X_test)
                # 映射回原始标签: [0,1,2] → [-1,0,1]
                preds_orig = np.array([0 if p == 1 else (-1 if p == 0 else 1) for p in preds])
                oof_preds[test_idx, col_idx] = preds_orig

                if len(y_test_orig) >= 5 and len(test_labels_unique) >= 2:
                    pred_corr = np.corrcoef(preds_orig, y_test_orig)[0, 1]
                    if not np.isnan(pred_corr):
                        if name not in model_ic:
                            model_ic[name] = []
                        model_ic[name].append(float(pred_corr))
            except ValueError as e:
                logger.warning(f"[TrainingWorker] Fold {fold_idx+1} {name} ValueError: {e}")
            except Exception as e:
                logger.warning(f"[TrainingWorker] Fold {fold_idx+1} {name} 失败: {type(e).__name__}: {e}")

    # ── 评估 ──
    model_scores = {}
    for name, ics in model_ic.items():
        if ics:
            mean_ic = float(np.mean(ics))
            std_ic = float(np.std(ics)) if len(ics) > 1 else 0.0
            icir = mean_ic / std_ic if std_ic > 0 else 0.0
            model_scores[name] = {'mean_ic': mean_ic, 'icir': icir, 'n_folds': len(ics)}
            logger.info(f"[TrainingWorker]   {name:6s} IC={mean_ic:+.4f}  ICIR={icir:.3f}")

    # ── Ridge 元学习器 ──
    from sklearn.linear_model import Ridge
    from sklearn.metrics import accuracy_score, f1_score

    valid_models = [name for name, info in model_scores.items() if info['mean_ic'] > 0]
    meta_learner = None
    if valid_models:
        valid_cols = [model_names.index(n) for n in valid_models]
        oof_valid = oof_preds[:, valid_cols]
        meta_learner = Ridge(alpha=1.0)
        meta_learner.fit(oof_valid, y)
        meta_coefs = dict(zip(valid_models, meta_learner.coef_))
        logger.info(f"[TrainingWorker] Ridge 权重: {', '.join(f'{k}={v:.3f}' for k, v in meta_coefs.items())}")

    # ── 全量训练最终模型 ──
    final_models = {}
    for name, model in level0_models.items():
        try:
            if hasattr(model, 'set_params'):
                model.set_params(random_state=42)
            model.fit(X, y_mapped)
            final_models[name] = model
        except Exception as e:
            logger.warning(f"[TrainingWorker] 全量训练 {name} 失败: {type(e).__name__}: {e}")

    # ── 保存模型为原生 JSON 格式（避免主进程 pickle.load SIGSEGV） ──
    os.makedirs(model_dir, exist_ok=True)
    native_model_files = {}  # name -> json file path

    for name, model in final_models.items():
        try:
            if name == 'lgb':
                # LightGBM: 使用 booster_ 保存为 JSON
                if hasattr(model, 'booster_'):
                    json_path = os.path.join(model_dir, f'model_{name}.json')
                    model.booster_.save_model(json_path)
                    native_model_files[name] = json_path
                    logger.info(f"[TrainingWorker] LightGBM 已保存: {json_path}")
                else:
                    logger.warning(f"[TrainingWorker] LightGBM 无 booster_ 属性")

            elif name == 'xgb':
                # XGBoost: 原生 JSON 格式
                json_path = os.path.join(model_dir, f'model_{name}.json')
                model.get_booster().save_model(json_path)
                native_model_files[name] = json_path
                logger.info(f"[TrainingWorker] XGBoost 已保存: {json_path}")

            else:
                # sklearn 模型: 保存为单独 pickle（主进程跳过加载）
                pkl_path = os.path.join(model_dir, f'model_{name}.pkl')
                with open(pkl_path, 'wb') as f:
                    pickle.dump(model, f)
                logger.info(f"[TrainingWorker] {name} 已保存 (pickle, 主进程跳过): {pkl_path}")
        except Exception as e:
            logger.warning(f"[TrainingWorker] 保存 {name} 失败: {e}")

    # ── 评分 ──
    oof_flat = oof_preds.argmax(axis=1) if oof_preds.shape[1] > 1 else oof_preds[:, 0]
    cv_accuracy = float(accuracy_score(y, oof_flat))
    cv_f1 = float(f1_score(y, oof_flat, average='macro', zero_division=0))

    # ── 特征重要性 ──
    feature_importances = {}
    for name, model in final_models.items():
        if hasattr(model, 'feature_importances_'):
            imp = model.feature_importances_
            feature_importances[name] = dict(zip(feature_names, [float(v) for v in imp]))

    # ── 保存模型 ──
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_filename = f'model_{timestamp}.pkl'
    model_path = os.path.join(model_dir, model_filename)
    os.makedirs(model_dir, exist_ok=True)

    model_data = {
        'models': final_models,
        'meta_learner': meta_learner,
        'feature_names': feature_names,
        'feature_importances': feature_importances,
        'factor_ic': {k: v for k, v in model_scores.items()},
        'factor_ic_history': model_ic,
        'cv_score': cv_accuracy,
        'is_trained': True,
        'trained_at': datetime.now().isoformat(),
        'fold_metrics': fold_metrics,
    }

    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    logger.info(f"[TrainingWorker] 模型已保存: {model_path} (CV={cv_accuracy:.3f})")

    # ── 保存元数据到 JSON（主进程不读取 pickle） ──
    meta_path = os.path.join(model_dir, f'model_meta_{timestamp}.json')
    meta_data = {
        'model_path': model_path,
        'cv_score': cv_accuracy,
        'cv_f1': cv_f1,
        'n_samples': n,
        'n_features': len(feature_names),
        'models_trained': list(final_models.keys()),
        'model_scores': {k: {kk: float(vv) if isinstance(vv, (int, float, np.number)) else vv
                             for kk, vv in v.items()} for k, v in model_scores.items()},
        'feature_importances': {k: {kk: float(vv) for kk, vv in v.items()}
                                for k, v in feature_importances.items()},
        'factor_ic': {k: [float(vv) if isinstance(vv, (int, float, np.number)) else vv
                          for vv in v] for k, v in model_ic.items()},
        'fold_metrics': fold_metrics,
        'native_model_files': native_model_files,
        'meta_learner_params': None,
        'is_trained': True,
        'trained_at': datetime.now().isoformat(),
    }
    # 保存 meta_learner 参数
    if meta_learner is not None:
        meta_data['meta_learner_params'] = {
            'alpha': float(meta_learner.alpha),
            'coef_': meta_learner.coef_.tolist(),
            'intercept_': meta_learner.intercept_.tolist(),
            'max_iter': meta_learner.max_iter,
        }
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, default=str)
    logger.info(f"[TrainingWorker] 元数据已保存: {meta_path}")

    # 输出结果 JSON（包含原生模型文件路径）
    result = {
        'success': True,
        'message': f'训练完成，模型已保存: {model_filename}',
        'model_path': model_path,
        'meta_path': meta_path,
        'cv_score': cv_accuracy,
        'cv_f1': cv_f1,
        'n_samples': n,
        'n_features': len(feature_names),
        'models_trained': list(final_models.keys()),
        'native_model_files': native_model_files,
        'model_scores': model_scores,
        'feature_importances': feature_importances,
        'fold_metrics': fold_metrics,
        'feature_names': feature_names,
    }
    print(json.dumps(result, default=str))


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        traceback.print_exc()
        print(json.dumps({'success': False, 'message': f'训练失败: {e}', 'error': str(e)}))
        sys.exit(1)
'''
    with open(_TRAINER_SCRIPT, 'w') as f:
        f.write(content)


def run_training(X: np.ndarray, y: np.ndarray, feature_names: List[str],
                 model_dir: str, timeout: int = 300,
                 custom_params=None) -> Dict[str, Any]:
    """
    在子进程中运行模型训练

    Args:
        X: 特征矩阵 (n_samples, n_features)
        y: 标签数组 (n_samples,)
        feature_names: 特征名称列表
        model_dir: 模型保存目录
        timeout: 超时秒数（默认 5 分钟）
        custom_params: 调参结果 {model_name: params} (2026-09-08) —
            由 HyperParamOrchestrator 经 MLPredictor.custom_params 注入,
            子进程 trainer 脚本将其合并进模型构造参数 (None/空 = 默认参数)

    Returns:
        训练结果字典
    """
    _ensure_trainer_script()

    X_list = X.tolist() if isinstance(X, np.ndarray) else X
    y_list = y.tolist() if isinstance(y, np.ndarray) else y

    input_data = {
        'X': X_list,
        'y': y_list,
        'feature_names': feature_names,
        'model_dir': model_dir,
        'custom_params': custom_params or {},
    }

    try:
        proc = subprocess.run(
            [sys.executable, _TRAINER_SCRIPT],
            input=json.dumps(input_data),
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )

        if proc.returncode != 0:
            error_msg = proc.stderr[:500] if proc.stderr else '未知错误'
            return {
                'success': False,
                'message': f'子进程失败 (rc={proc.returncode}): {error_msg}',
                'stderr': error_msg,
            }

        # 解析子进程输出（日志 + JSON 混在 stdout 中）
        output_lines = proc.stdout.strip().split('\n')
        for line in reversed(output_lines):
            line = line.strip()
            if line.startswith('{'):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue

        return {
            'success': False,
            'message': '子进程无有效 JSON 输出',
            'stdout_tail': proc.stdout[-500:] if proc.stdout else '',
        }

    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'message': f'训练超时 ({timeout}s)',
        }
    except Exception as e:
        return {
            'success': False,
            'message': f'训练进程异常: {e}',
            'error': str(e),
        }
