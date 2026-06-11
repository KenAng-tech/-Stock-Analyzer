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
templates/index.html      — 主页面 (Flask render_template)
webgui.html               — 分析页面 (send_file)
static/css/style.css      — 主样式 (绿色→蓝色已替换为 #3b82f6)
static/js/                — 前端 JS
static/img/               — 静态图片
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

## 注意事项
- 不要修改 backup_* 目录下的文件
- index.html 有多个备份 (.bak.*)，修改时注意是最新文件
- app.py 通过 run_server.py 启动 (Werkzeug threading 模式，非 eventlet)
- WebSocket 使用 threading 模式
- 端口 5002，host 0.0.0.0
