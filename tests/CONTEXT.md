# 测试上下文 — stock_analyzer

## 领域术语

- **标的 / stock**: A 股股票代码，格式如 `sz300620` (深交所), `sh688981` (上交所), `bj830799` (北交所)
- **thscode**: 同花顺统一标识符，格式如 `600519.SH`, `300620.SZ`, `830799.BJ`
- **K 线 / kline**: 蜡烛图数据，包含 open/high/low/close/volume
- **因子 / factor**: 量化因子，如 Alpha158 (158 个), Alpha360 (360 个)
- **SOTA 模型**: State-of-the-art 预测模型，包括 PatchTST, Mamba, Diffusion, Conformal, TimesFM 等
- **融合 / fusion**: 多模态融合预测，整合价格/情感/基本面信号
- **regime**: 市场状态 (bull/bear/sideways)，由 HMM 检测
- **UDE**: Unified Decision Engine，统一决策引擎
- **IC**: Information Coefficient，因子信息与收益的相关系数
- **Brier score**: 概率预测的校准度评估指标

## 测试命名约定

`test_<module>_<feature>_<condition>_<expected>.py`

示例:
- `test_ths_provider_realtime_success.py` — 实时行情正常返回
- `test_fusion_predict_regime_weights.py` — 融合预测按 regime 加权
- `test_session_manager_cleanup_idle.py` — 空闲 session 清理

## 测试数据约定

- 使用**真实格式 + 模拟值**，不用 tautological 断言
- 外部 API 调用必须 mock (CLI, HTTP, JAX, torch)
- 测试用股票代码: `sz300620` (宁德时代), `sh688981` (中芯国际)
- 测试用 thscode: `600519.SH`, `300620.SZ`

## 关键 Seams (待测试的公共接口)

### ThsDataProvider
- `get_realtime(thscode) -> Optional[Dict]` — CLI `market quote` 调用
- `get_klines(thscode, start, end) -> Optional[List[Dict]]` — CLI `market history` 调用
- `get_valuation(thscode) -> Optional[Dict]` — CLI `market valuation` 调用
- `resolve_symbol(query, limit) -> Optional[List[Dict]]` — CLI `market search` 调用
- `sync_data(start, end) -> bool` — CLI `data sync` 调用
- `get_dragon_tiger(limit) -> Optional[List[Dict]]` — CLI `list dragon-tiger` 调用

### ScraplingFallback
- `eastmoney_quote(stock_code) -> Optional[Dict]` — 东方财富行情抓取
- `eastmoney_guba_news(stock_code, limit) -> Optional[List[Dict]]` — 东方财富股吧抓取

### SessionManager
- `create_session(session_type, session_id) -> Optional[str]` — 创建 browser session
- `close_session(session_id) -> bool` — 关闭 session
- `cleanup_idle() -> int` — 清理空闲 session
- `get_stats() -> Dict` — 统计信息

### MultiModalFusion
- `predict(stock_code, regime, use_cache) -> Dict` — 融合预测主入口
- `_evaluate_prediction(result, actual_return)` — 评估预测准确性
- `ModalityICTracker.get_adaptive_weights() -> Optional[Dict]` — IC 自适应权重

### analysis_engine
- `quantitative_prediction(...) -> Dict` — 量化预测主入口 (集成 UDE + Fusion)
- `decide_from_predictions(predictions) -> Dict` — UDE 统一决策

## 外部依赖 Mock 策略

- **hithink-finance CLI**: mock `subprocess.run` 返回模拟输出
- **Scrapling**: mock `scrapling.open_session` 和 `session.fetch`
- **JAX/Torch**: 已有条件导入 fallback，测试时 mock 实际推理
- **AKShare**: mock `akshare.*` 函数
- **时间相关**: 使用 `unittest.mock.patch('datetime.datetime')` 或传入固定时间

## 禁止的测试模式

- **Mock 内部实现**: 不要 mock 私有方法或内部类
- **Tautological 断言**: 不要用代码自身计算的结果作为 expected
- **Horizontal slicing**: 不要先写所有测试再写实现，逐个 slice
- **测试路由函数**: 不要直接测试 Flask route 内部的 try/except 块
