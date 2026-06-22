# Frontend Rules

当修改 `webgui.html`, `webgui_new.html`, `templates/index.html`, `static/css/style.css`, `static/js/` 时：

## 必须遵守
1. **蓝色主题** — 主色调 #3b82f6，禁止使用绿色 (#10b981, #34d399)
2. **CSS 变量** — 优先使用 `--accent`, `--accent-light` 等变量
3. **图表配色** — 上涨 #3b82f6 (蓝), 下跌 #f85149 (红), 中性 #d29922 (黄)
4. **玻璃拟态** — dashboard 使用 backdrop-filter: blur() 效果
5. **深色背景配白字** — 所有使用深色背景 (`rgba(15, 23, 42, ...)`, `#0f172a`, `#1e293b` 等) 的面板 (`.qm-panel`, `.section` 深色背景) 内，所有文本必须使用浅色值：
   - 主文本: `#ffffff` 或 `#e2e8f0`
   - 次要文本: `#94a3b8` 或 `#cbd5e1`
   - 强调值: `#60a5fa` (蓝色)
   - 成功/涨: `#10b981` (绿色) 或 `#34d399`
   - 警告/跌: `#ef4444` (红色) 或 `#f87171`
   - 绝对禁止在深色背景上使用深色文字 (如 `#0f172a` on dark bg)

## WebSocket
- 连接断开时必须自动重连
- 重连间隔：1s → 2s → 4s → 8s (指数退避)
- 重连失败超过 30s 后显示提示

## 禁止
- 不要在 CSS 中硬编码颜色值（优先用 CSS 变量）
- 不要在 WebSocket handler 中阻塞主线程
- 不要在内层循环中创建 Chart.js 实例
- 不要在深色背景面板内使用未定义样式的 class (所有 JS 动态生成的 HTML 必须有对应 CSS)
