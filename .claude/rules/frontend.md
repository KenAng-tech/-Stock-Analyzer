# Frontend Rules

当修改 `webgui.html`, `webgui_new.html`, `templates/index.html`, `static/css/style.css`, `static/js/` 时：

## 必须遵守
1. **蓝色主题** — 主色调 #3b82f6，禁止使用绿色 (#10b981, #34d399)
2. **CSS 变量** — 优先使用 `--accent`, `--accent-light` 等变量
3. **图表配色** — 上涨 #3b82f6 (蓝), 下跌 #f85149 (红), 中性 #d29922 (黄)
4. **玻璃拟态** — dashboard 使用 backdrop-filter: blur() 效果

## WebSocket
- 连接断开时必须自动重连
- 重连间隔：1s → 2s → 4s → 8s (指数退避)
- 重连失败超过 30s 后显示提示

## 禁止
- 不要在 CSS 中硬编码颜色值（优先用 CSS 变量）
- 不要在 WebSocket handler 中阻塞主线程
- 不要在内层循环中创建 Chart.js 实例
