---
name: webgui-review
description: "Review frontend changes for dashboard, charts, and real-time WebSocket UI"
whenToUse: "When reviewing changes to webgui.html, webgui_new.html, templates/index.html, or static/css/style.css"
---

# WebGUI Review Skill

## Overview
Review frontend changes for the stock analyzer dashboard:
- `webgui.html` / `webgui_new.html` — main dashboard
- `templates/index.html` — homepage
- `static/css/style.css` — styling (blue theme #3b82f6)
- `static/js/` — JavaScript modules

## Checklist

### Visual
- [ ] Blue theme (#3b82f6) consistent across all components
- [ ] Glassmorphism effects intact
- [ ] Responsive layout works on mobile
- [ ] Chart colors consistent (blue/red/yellow)

### Functionality
- [ ] WebSocket updates render correctly
- [ ] Chart.js charts display and update
- [ ] Real-time data streaming works
- [ ] Error states handled (empty data, connection lost)

### Performance
- [ ] No memory leaks in WebSocket handlers
- [ ] Chart updates use diff (not full re-render)
- [ ] CSS animations use GPU acceleration
- [ ] No excessive reflows

### Data Display
- [ ] Factor scores displayed correctly
- [ ] Backtest results shown with proper formatting
- [ ] ML predictions displayed with confidence intervals
- [ ] Sentiment scores visualized correctly

## Common Pitfalls
- Breaking the blue color theme with hardcoded green values
- WebSocket reconnection not handled
- Chart data not cleared before update (causes overlay)
- CSS specificity conflicts
