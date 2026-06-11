# Stock Analyzer System - Agent Development Guidelines

## 核心开发原则

### 1. 前后端数据契约验证（Critical）

**问题历史：** 2026-05-29 形态/持仓/成交量评分不显示
**根因：** API 返回结构与 app.js 期望结构不匹配

**检查清单：**
- [ ] 修改后端 API 返回结构时，同步检查前端 app.js 的解析逻辑
- [ ] 修改前端 app.js 解析逻辑时，确认后端 API 返回结构
- [ ] 在 app.js 中添加数据验证日志（`console.log('klineSignals:', klineSignals)`）
- [ ] 使用 TypeScript 类型定义或 JSDoc 注释定义数据契约

**验证命令：**
```bash
# 检查 API 返回结构
curl -s http://127.0.0.1:5002/api/analyze/sz300620 | python3 -c "import sys, json; print(json.dumps(json.load(sys.stdin)['kline_signals'], indent=2, ensure_ascii=False))"

# 检查 app.js 期望结构
grep -n "klineSignals\." /Users/claw/stock_analyzer/static/js/app.js | head -20
```

### 2. 浏览器缓存管理

**问题历史：** 修改后浏览器仍显示旧版本
**根因：** 浏览器缓存了旧的 app.js 和 webgui.html

**检查清单：**
- [ ] 修改静态文件后，添加版本号参数（`?v=2`）
- [ ] 在开发环境中添加 `Cache-Control: no-cache` 头
- [ ] 指导用户强制刷新（Cmd+Shift+R / Ctrl+Shift+R）
- [ ] 使用文件名哈希（如 `app.js?v=1.2.3`）

**验证命令：**
```bash
# 检查文件修改时间
ls -la /Users/claw/stock_analyzer/static/js/app.js
ls -la /Users/claw/stock_analyzer/webgui.html

# 检查缓存头
curl -I http://127.0.0.1:5002/static/js/app.js
```

### 3. JavaScript 空引用安全

**问题历史：** `TypeError: null is not an object (evaluating 'historyEl.innerHTML = ''')`
**根因：** `renderAlerts()` 未检查 `alertHistory` 元素是否存在

**检查清单：**
- [ ] 所有 `document.getElementById()` 后添加 null 检查
- [ ] 使用可选链操作符（`?.`）
- [ ] 在 DOM 操作前验证元素存在性
- [ ] 添加全局错误捕获器

**代码模式：**
```javascript
// 推荐模式
const element = document.getElementById('myElement');
if (element) {
    element.innerHTML = 'value';
}

// 或使用可选链
document.getElementById('myElement')?.innerHTML = 'value';
```

### 4. API 端点完整性验证

**问题历史：** webgui.html 期望的 `/api/quant/` 端点不存在
**根因：** 前端调用不存在的 API 端点

**检查清单：**
- [ ] 前端调用 API 前，确认端点存在
- [ ] 在 app.py 中添加所有前端需要的端点
- [ ] 添加 `/api/health` 端点用于健康检查
- [ ] 在 app.js 中添加 API 响应日志

**验证命令：**
```bash
# 检查所有 API 端点
curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:5002/api/quant/signals
curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:5002/api/quant/positions
curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:5002/api/quant/performance
curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:5002/api/quant/stocks
curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:5002/api/quant/kline-scores
```

### 5. 服务器重启验证

**问题历史：** 修改 app.py 后未重启服务器
**根因：** Flask 开发服务器需要重启才能加载新代码

**检查清单：**
- [ ] 修改 app.py 后，重启服务器
- [ ] 修改 kline_signal_analyzer.py 后，重启服务器
- [ ] 使用 `ps aux | grep "start_threading"` 确认进程
- [ ] 使用 `kill <PID>` 停止，然后重新启动

**验证命令：**
```bash
# 检查服务器进程
ps aux | grep "start_threading" | grep -v grep

# 重启服务器
kill $(pgrep -f "start_threading.py")
cd /Users/claw/stock_analyzer && source venv/bin/activate && python start_threading.py
```

### 6. 页面路由确认

**问题历史：** 用户访问 `/`（index.html）而非 `/webgui.html`
**根因：** 两个页面使用不同的前端代码

**检查清单：**
- [ ] 确认用户访问的页面（`/` vs `/webgui.html`）
- [ ] `/` 路由使用 `templates/index.html` + `static/js/app.js`
- [ ] `/webgui.html` 路由使用 `webgui.html`（内嵌 JavaScript）
- [ ] 在文档中明确说明不同页面的功能差异

**验证命令：**
```bash
# 检查路由配置
grep -A2 "@app.route('/')" /Users/claw/stock_analyzer/app.py
grep -A2 '"/webgui.html"' /Users/claw/stock_analyzer/app.py
```

---

## 开发工作流

### 修改后端代码（app.py, modules/*.py）
1. 修改代码
2. 重启服务器
3. 验证 API 响应
4. 检查前端兼容性

### 修改前端代码（app.js, webgui.html, index.html）
1. 修改代码
2. 添加版本号参数（如 `?v=2`）
3. 指导用户强制刷新浏览器
4. 检查浏览器控制台错误

### 修改数据结构
1. 更新后端返回结构
2. 更新前端解析逻辑
3. 添加数据验证日志
4. 验证端到端数据流

---

## 审核检查清单

### 代码修改前
- [ ] 确认修改范围（后端/前端/两者）
- [ ] 确认依赖关系
- [ ] 确认影响范围

### 代码修改后
- [ ] 重启服务器（后端修改）
- [ ] 添加缓存刷新（前端修改）
- [ ] 验证 API 响应
- [ ] 检查浏览器控制台
- [ ] 验证数据流

### 部署前
- [ ] 运行所有 API 端点测试
- [ ] 检查浏览器控制台错误
- [ ] 验证所有评分/数据显示
- [ ] 确认无 TypeError/ReferenceError

---

## 常见问题快速参考

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| 数据显示为 `--` | 数据结构不匹配 | 检查 API 返回 vs app.js 期望 |
| 页面无变化 | 浏览器缓存 | 强制刷新或添加 `?v=2` |
| TypeError: null | 元素不存在 | 添加 null 检查 |
| 404 Not Found | 端点不存在 | 检查 API 端点定义 |
| 数据不更新 | 服务器未重启 | 重启服务器 |
| 页面空白 | 路由错误 | 确认访问 `/` 或 `/webgui.html` |
