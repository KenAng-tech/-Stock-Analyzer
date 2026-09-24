#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
预测 Worker — 在子进程中加载 C 扩展模型（LightGBM/XGBoost）并返回概率

通过进程隔离避免 C 扩展 SIGSEGV 影响主进程。

用法:
    echo '{"model": "/path/to/model_lgb.json", "features": [[...]]}' | python _predictor_process.py
    输出: {"up": 0.65, "down": 0.15, "neutral": 0.20}
"""

import os, sys, json, traceback
import numpy as np

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    try:
        input_data = json.loads(sys.stdin.read())
        model_path = input_data['model']
        features = np.array(input_data['features'], dtype=np.float32)

        model_name = os.path.basename(model_path)

        if 'lgb' in model_name:
            # LightGBM: 从 JSON 加载
            import lightgbm as lgb
            booster = lgb.Booster(model_file=model_path)
            probs = booster.predict(features)

            # 多分类输出 (n_samples, n_classes)，n_classes=3
            if probs.ndim == 2 and probs.shape[1] >= 3:
                # prob[0]=down, prob[1]=neutral, prob[2]=up
                result = {
                    'up': float(probs[0, 2]) if probs.ndim > 1 else float(probs[2]),
                    'down': float(probs[0, 0]) if probs.ndim > 1 else float(probs[0]),
                    'neutral': float(probs[0, 1]) if probs.ndim > 1 else float(probs[1]),
                }
            else:
                # 单值输出: P(up)
                p_up = float(probs[0]) if probs.ndim > 0 else float(probs)
                result = {
                    'up': p_up,
                    'down': 0.0,
                    'neutral': 1.0 - p_up,
                }

        elif 'xgb' in model_name:
            # XGBoost: 从 JSON 加载
            import xgboost as xgb
            booster = xgb.Booster()
            booster.load_model(model_path)
            dmatrix = xgb.DMatrix(features)
            probs = booster.predict(dmatrix)

            # 多分类输出 (n_samples, n_classes)
            if probs.ndim == 2 and probs.shape[1] >= 3:
                result = {
                    'up': float(probs[0, 2]),
                    'down': float(probs[0, 0]),
                    'neutral': float(probs[0, 1]),
                }
            else:
                p_up = float(probs[0]) if probs.ndim > 0 else float(probs)
                result = {
                    'up': p_up,
                    'down': 0.0,
                    'neutral': 1.0 - p_up,
                }
        else:
            result = {'up': 0.33, 'down': 0.33, 'neutral': 0.34}

        print(json.dumps(result))

    except Exception as e:
        traceback.print_exc(file=sys.stdout)
        print(json.dumps({'success': False, 'error': str(e)}))
        sys.exit(1)


if __name__ == '__main__':
    main()
