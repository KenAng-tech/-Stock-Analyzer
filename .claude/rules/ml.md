# ML Model Rules

当修改 `ml_predictor.py`, `analysis_engine.py`, 或 ML 相关模块时：

## 必须遵守
1. **数据分割** — 训练/测试集按时间分割，不能随机分割
2. **特征泄露** — 确保训练特征不包含测试期信息
3. **交叉验证** — 使用 TimeSeriesSplit，不能 ShuffleSplit
4. **模型评估** — 报告准确率、精确率、召回率、F1、AUC

## 模型选择
- 默认：RandomForest + LightGBM 集成
- 特征数量 > 50 时优先 LightGBM
- 需要可解释性时使用 RandomForest (feature_importances_)

## 禁止
- 不要用未来数据训练模型
- 不要忽略类别不平衡（使用 class_weight='balanced'）
- 不要在测试集上调参
