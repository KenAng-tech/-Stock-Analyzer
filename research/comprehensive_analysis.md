# stock_analyzer 量化模型 — 全面审查与优化方案

> 日期: 2026-06-09
> 范围: 70+ 模块深度代码审查 + 业界 SOTA 对比 (microsoft/qlib, vnpy, Alpha191 等)

---

## 执行摘要

**综合评分: 5.8/10** — 架构设计有良好基础，但存在 5 个关键问题阻碍模型有效性

### 核心发现

| 问题 | 严重度 | 影响 |
|------|--------|------|
| 因子层与 ML 层割裂 | 🔴 P0 | ML 只用 12 维特征，21 因子白算 |
| 因子缺少截面标准化 | 🔴 P0 | IC 计算无意义，因子不可比 |
| 预测特征与训练不一致 | 🔴 P0 | 预测偏差，模型失效 |
| IC/风险数据为随机生成 | 🟡 P1 | 仪表盘数据不可信 |
| 单股训练无法泛化 | 🟡 P1 | 数据利用率低 |

---

## 一、当前架构全景

```
数据层: StockDataFetcher → KlineDataFetcher → FundamentalFetcher → SentimentAnalyzerV2
因子层: MultiFactorModelV2(21因子) → BarraRiskModel(10风格) → FactorOrthogonalizer → TemporalEncoder → GraphEncoder
ML层:   MLPredictor(Stacking: LGBM+XGB+RF+GRU→Ridge) → DynamicEnsemble → HMMMarketDetector
策略层: StrategyEngine → KlineSignalAnalyzer → FundFlowOptimizer → KellyOptimizer
回测层: EventBacktester(事件驱动) → WalkForwardBacktester(Monte Carlo)
风控层: AlertEngine → BarraRiskModel → RiskReportGenerator
监控层: FactorICMonitor → ConceptDrift(ADWIN) → ModelTrainingScheduler
```

---

## 二、深度审查 — 按层级

### 2.1 数据层 (6.5/10)

**优点**: 多数据源降级链 (AKShare→东方财富→新浪)、动态缓存 5 级 TTL
**问题**:
- 单股分析无截面数据 → IC 无法真实计算
- 情感分析走字典法，FinBERT 基本未用
- 缺少行业指数、宏观数据、北向资金

### 2.2 因子层 (5.5/10)

**优点**: 21 因子覆盖 8 大类、Gram-Schmidt 正交化
**问题**:
- **因子质量参差不齐**: `volume_momentum`、`turnover_level` 用分段函数映射到 [0,10]，不是连续值
- **RSI 因子逻辑反向**: RSI 接近 50 最优 — 这是反向逻辑，RSI 应作为动量因子
- **因子权重静态**: `DEFAULT_WEIGHTS` 硬编码，没有用 IC 历史做动态权重
- **Barra 风格因子计算过于简化**: Beta 用换手率近似

### 2.3 ML 层 (5.0/10)

**优点**: Stacking 集成架构合理、Purged K-Fold CV 防止前视偏差
**问题**:
- **12 维特征与 21 因子完全割裂** — 最严重问题
- `momentum_1d/3d/5d/10d` 高度共线
- 预测时 `change_pct` 被重复 4 次 × 不同系数 — 人为制造共线性
- 标签 3 分类阈值硬编码 2%，没有波动率自适应
- GRU: `epochs=50, batch_size=32`，对 250 个交易日数据 batch 太大

### 2.4 策略层 (6.0/10)

**优点**: K线形态识别全面、多周期共振
**问题**: 缺少统一信号评分框架、信号没有时间衰减

### 2.5 回测层 (7.0/10)

**优点**: 事件驱动架构、交易成本建模、A 股约束 (T+1/涨跌停/100股)
**问题**: 回测与实盘信号源不一致、滑点固定 2bps、缺少基准对比

### 2.6 风控层 (5.0/10)

**问题**: `risk-report` API 返回硬编码示例数据、缺少 VaR/CVaR/压力测试

---

## 三、与业界 SOTA 对比

### 3.1 因子体系

| 维度 | stock_analyzer | 业界 SOTA |
|------|---------------|-----------|
| 因子数量 | 21 | 1000+ (工业级) |
| 标准化 | 个股内 [0,10] | 截面 Rank-ZScore + 行业中性化 |
| 权重 | 静态硬编码 | ICIR 加权 / Lasso / 梯度提升 |
| 特色因子 | 无 | 涨跌停、北向资金、融资融券 |

### 3.2 ML 模型

| 维度 | stock_analyzer | 业界 SOTA |
|------|---------------|-----------|
| 特征维度 | 12 | 500-5000 |
| 训练方式 | 单股票 | 横截面 (全市场) |
| 深度学习 | 基础 GRU | Transformer / TFT / TabNet / GNN |
| 超参优化 | 有框架 | Optuna / Ray Tune |

### 3.3 参考项目

1. **microsoft/qlib** (44k stars) — 完整因子→模型→回测流水线
2. **vnpy** (41k stars) — 事件驱动实盘框架
3. **Alpha191** (华泰证券) — A 股专用 191 因子体系

---

## 四、优化方案 — 按优先级

### 🔴 P0: 立即修复

#### 优化 1: 打通因子层 ↔ ML 层
```python
# 新增 FeatureFusion 模块
class FeatureFusion:
    def fuse(self, stock_data, klines, factor_scores, 
             temporal_features, graph_features):
        # 21 因子 + 8×21 时序统计 + 图嵌入 + 基本面 + 情绪
        # → 100-200 维特征向量
```
**收益**: 特征维度 12→100+，表达能力 5-10 倍提升

#### 优化 2: 修复预测特征不一致
`dashboard_api.py` 中用 `prepare_features()` 替代硬编码特征构建

#### 优化 3: 因子截面标准化
```python
def cross_sectional_rank_normalize(factor_values):
    """截面 Rank-ZScore 标准化"""
```
**收益**: 因子可比性大幅提升，IC 计算成为可能

#### 优化 4: 真实 IC 计算
替换 `np.random.normal(0.03, 0.02, 60)` 为真实截面 IC

### 🟡 P1: 短期优化 (1-2 周)

5. 扩展因子 (WorldQuant Alpha 子集、Amihud 非流动性、日内反转)
6. 改进标签设计 (波动率自适应阈值)
7. Optuna 超参优化
8. 回测与实盘统一信号源

### 🟢 P2: 中期优化 (1-2 月)

9. Transformer 替代 GRU
10. 横截面训练 (全市场)
11. 多信号融合框架
12. VaR/CVaR + 压力测试

### 🔵 P3: 长期优化 (3-6 月)

13. 另类数据 (分析师预期、机构持仓)
14. MLOps (MLflow + Airflow)
15. 强化学习策略升级

---

## 五、实施建议

**建议按 P0→P1→P2→P3 顺序逐步实施**，每完成一个级别后重启服务验证。

P0 的 4 项优化是基础中的基础，不修复这些问题，后续所有优化都建立在沙地上。

**需要我现在开始实施 P0 级别的修复吗？**
