# Stock Analyzer — 项目指南

## 快速启动
- 启动: `source venv/bin/activate && python run_server.py`
- 服务: http://127.0.0.1:5002
- 分析页: http://127.0.0.1:5002/webgui.html
- API: http://127.0.0.1:5002/api/stock/<code> (如 sz300620)

## 架构
```
app.py                  — Flask + Flask-SocketIO 主服务 (run_server.py 启动)
modules/                — 核心模块
  multi_factor_model_v2.py  — 21 因子模型 (动量/价值/波动率/成交量/流动性/质量/技术/情绪)
  event_backtester.py       — 事件驱动回测引擎 (Order→Execution→Position→Risk→Report)
  walkforward_backtester.py — Walk-Forward + Bootstrap Monte Carlo
  ml_predictor.py           — RandomForest + LightGBM 双模型集成
  analysis_engine.py        — 基本面分析 + ML 预测
  fundamental_fetcher.py    — AKShare 真实财报数据
  kline_signal_analyzer.py  — K 线信号分析 (RSI/MACD/布林带/ATR)
  sentiment_analyzer_v2.py  — 中文情感分析 (东方财富新闻/股吧)
  portfolio_optimizer.py    — Black-Litterman + 风险平价组合优化
  heatmap_generator.py      — 热力图生成
  alert_engine.py           — 告警引擎
  websocket_handler.py      — WebSocket 实时推送
  # P0: 真实技术指标计算
  technical_indicators.py   — RSI/MACD/KDJ/布林带/ATR 真实计算
  # P1: iTransformer + 情绪因子
  itransformer_predictor.py — ICLR 2024 Spotlight 倒置 Transformer
  # P2: 因子衰减 + 缓存 + Brinson
  alpha158_calculator.py    — 158 个 Alpha 因子
  brinson_attribution.py    — Brinson-Fachler 绩效归因
  dynamic_cache.py          — 分级缓存 + 依赖链自动失效
  concept_drift_detector.py — ADWIN 概念漂移检测
  atr_calculator.py         — ATR 计算
  rl_trader_v2.py           — 强化学习交易器
  # P3: 内存 + SQLite
  memory_manager.py         — 内存监控 + GC + 泄漏检测
  # 已删除 (勿当活链恢复): sqlite_cache.py→dynamic_cache 继承 / moirai_predictor + drl_agent → PatchTST 池 (modules/models/) 继承
  cross_market.py           — A股+港股+美股跨市场集成 (活)
  search_ledger.py          — 搜索全记账账本 (09-14 自研, DSR/PBO 真实 trial 数源)
templates/index.html      — 主页面 (Flask render_template)
webgui.html               — 分析页面 (4 个 Tab: Dashboard/Analysis/量化模型/系统监控)
static/css/style.css      — 主样式 (绿色→蓝色已替换为 #3b82f6)
static/js/                — 前端 JS
static/img/               — 静态图片
tests/                    — 单元测试
  test_technical_indicators.py  — 技术指标 14 个测试
  test_backtester.py            — 回测引擎 9 个测试
  test_memory_and_cache.py      — 内存 + SQLite 17 个测试
```

## 数据源
- AKShare — 真实财务数据 (营收、利润、毛利率、ROE 等)，未安装时 fallback
- 东方财富 — 新闻标题 + 股吧帖子 (API: search-api-web.eastmoney.com, guba.eastmoney.com)
- 腾讯财经 — 实时行情 (通过 app.py API 端点)

## 回测
- `modules/event_backtester.py` — 事件驱动回测引擎
  - 支持市价单/限价单/止损单 + 滑点 + 冲击成本
  - RiskManager: 仓位限制/回撤限制/日交易次数限制
  - BacktestReport: Sharpe/Sortino/Calmar/MaxDD/WinRate/ProfitFactor
- `modules/walkforward_backtester.py` — Walk-Forward 滚动窗口 + Bootstrap Monte Carlo
  - ATR 动态止损/止盈
  - TransactionCostModel: 佣金 0.03% + 印花税 0.1% + 滑点
- `modules/strategies/rsi_macd_strategy.py` — RSI+MACD+均线示例策略

## 组合优化
- `modules/portfolio_optimizer.py` — Black-Litterman 观点注入 + 风险平价
- `modules/kelly_optimizer.py` — Kelly 公式仓位优化
- `modules/adaptive_kelly.py` — 自适应 Kelly (动态调整 fraction)

## 编码规范
- 中文注释，Python 3.11+ (项目用 3.14)
- 使用 `modules/logger.py` 的 logger 而非 print
- 所有 API 端点需要 try/except + logger.error
- 向量操作优先用 numpy/pandas，避免显式循环
- 新增模块放在 modules/ 下，__init__.py 保持干净
- 前端颜色主题: 蓝色系 (#3b82f6 / #60a5fa / #2563eb)，已替换掉绿色

## API 端点 (新增)
- `GET  /api/memory/status`      — 内存监控状态
- `POST /api/memory/gc`          — 强制垃圾回收
- `GET  /api/cache/stats`        — SQLite/动态缓存统计
- `POST /api/cache/cleanup`      — 清理过期缓存
- `POST /api/cache/clear`        — 清空缓存
- `GET/POST /api/brinson/analyze` — Brinson-Fachler 绩效归因
- `GET  /api/health`             — 健康检查 (含内存状态)
- `GET  /api/sota/moirai/status` — Moirai 预测器状态
- `GET  /api/sota/moirai/predict` — Moirai 预测
- `GET  /api/sota/drl/status`    — DRL 代理状态
- `GET  /api/sota/drl/decide`    — DRL 交易决策
- `GET  /api/sota/cross-market/summary` — 跨市场摘要
- `GET  /api/sota/cross-market/ah-premium` — AH 溢价
- `GET  /api/sota/cross-market/sentiment/<market>` — 市场情绪
- `GET  /api/sota/cross-market/fusion` — 跨市场融合预测
- `GET  /api/fusion/multi-modal` — 多模态融合预测
- `GET  /api/fusion/multi-modal/status` — 融合器状态
- `GET  /api/fusion/compare` — 各模态独立对比
- `POST /api/mlops/canary/deploy` — 启动金丝雀部署
- `POST /api/mlops/canary/record` — 记录预测结果
- `GET  /api/mlops/canary/status` — 金丝雀部署状态
- `POST /api/mlops/canary/end` — 结束金丝雀部署

## 前端页面
- `webgui.html` — 4 个 Tab: Dashboard / Analysis / 量化模型 / 系统监控
  - 系统监控 Tab: 内存监控图表 + SQLite 缓存统计 + Brinson 归因交互面板

## TDD 工作流
- 新增功能或修复 bug 前，先调用 `tdd` skill (已部署在 `~/.claude/skills/tdd/`)
- 遵循 red → green 循环: 先写失败测试 → 最小实现 → 通过 → 重构
- 测试只针对公共接口 (seams)，不 mock 内部实现
- 测试放在 `tests/` 下，命名: `test_<模块>_<功能>_<条件>.py`
- 项目领域术语见 `tests/CONTEXT.md`
- 外部依赖 (CLI, HTTP, JAX, torch) 必须 mock
- 不写 tautological 断言 (expected 值必须来自独立事实来源)

## 注意事项
- 不要修改 backup_* 目录下的文件
- index.html 有多个备份 (.bak.*)，修改时注意是最新文件
- app.py 通过 run_server.py 启动 (Werkzeug threading 模式，非 eventlet)
- WebSocket 使用 threading 模式
- 端口 5002，host 0.0.0.0
