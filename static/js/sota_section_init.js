/* DEAD CODE MARK (2026-09-23 全链 review, 方案A): DEAD CODE (2026-09-23 review): 零加载 (index/tab 页零引用, 09-20 快照). 内部含废弃路径 (sentiment_bert/multiagent/pipeline 无前缀版/portfolio/optimization = 09-23 实测 404/死码区). 若重启用: 先核路径再接. */
// sota_section_init.js — showSection 覆盖 + 初始化 (依赖 app.js)
// (2026-08-16 从 templates/index.html 内联 <script> 外置, 架构清理)

    // 保存 app.js 中原始的 showSection
    const _originalShowSection = window.showSection;
    window.showSection = function(section) {
        // 始终执行 section 切换
        document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
        const target = document.getElementById(section);
        if (target) target.classList.add('active');
        // 同时调用原始函数（如果存在）
        if (_originalShowSection) {
            _originalShowSection(section);
        }
        // 加载数据
        if (section === 'sota') loadSOTADashboard();
        if (section === 'monitor') loadMonitorTab();
        if (section === 'cross-market') loadCrossMarketTab();
        if (section === 'consensus') loadConsensusTab();
        if (section === 'safe-rl') loadSafeRLTab();
        if (section === 'mlops') loadMLOPSTab();
    };
    // ========================================================================
    // 系统监控 Functions
    // ========================================================================
    async function loadMonitorTab() {
        console.log('[loadMonitorTab] START');
        loadMonitorMemory();
        loadMonitorCache();
        console.log('[loadMonitorTab] ALL fetches dispatched');
    }

    async function loadMonitorMemory() {
        try {
            const resp = await fetch('/api/memory/status');
            const raw = await resp.json();
            console.log('[loadMonitorMemory] raw keys:', Object.keys(raw));
            const m = raw.data || raw;
            console.log('[loadMonitorMemory] m keys:', Object.keys(m));
            document.getElementById('mem-rss').textContent = (m.rss_mb || 0).toFixed(1);
            document.getElementById('mem-heap').textContent = (m.python_heap_mb || 0).toFixed(1);
            document.getElementById('mem-objects').textContent = (m.object_count || 0).toLocaleString();
            const healthEl = document.getElementById('mem-health');
            healthEl.textContent = m.health || '--';
            healthEl.style.color = m.health === 'healthy' ? '#10b981' : '#ef4444';
            document.getElementById('mem-leak').textContent = (m.leak_detection && m.leak_detection.potential_leak) ? '⚠ 泄漏检测' : '✓ 正常';
            console.log('[loadMonitorMemory] Done, rss=', document.getElementById('mem-rss').textContent);
        } catch (e) {
            document.getElementById('monitor-memory-stats').innerHTML = `<div style="color:#ef4444">加载失败: ${e.message}</div>`;
        }
    }

    async function forceMemoryGC() {
        try {
            const resp = await fetch('/api/memory/gc', {method: 'POST'});
            const data = await resp.json();
            alert(`GC 完成: 回收 ${data.collected || 0} 个对象, 释放 ${(data.freed_mb || 0).toFixed(2)} MB`);
            loadMonitorMemory();
        } catch (e) {
            alert(`GC 失败: ${e.message}`);
        }
    }

    async function loadMonitorCache() {
        try {
            const resp = await fetch('/api/cache/stats');
            const raw = await resp.json();
            const stats = raw.stats || raw;
            document.getElementById('sqlite-total').textContent = stats.total_entries || 0;
            document.getElementById('sqlite-active').textContent = stats.active_entries || 0;
            document.getElementById('sqlite-expired').textContent = stats.expired_entries || 0;
            document.getElementById('sqlite-size').textContent = stats.db_size_mb ? (stats.db_size_mb).toFixed(2) + ' MB' : 'N/A';
        } catch (e) {
            document.getElementById('monitor-cache-stats').innerHTML = `<div style="color:#ef4444">加载失败: ${e.message}</div>`;
        }
    }

    async function cleanupSQLiteCache() {
        try {
            const resp = await fetch('/api/cache/cleanup', {method: 'POST'});
            const data = await resp.json();
            if (data.success) {
                alert(`清理完成: 删除 ${data.deleted || 0} 条过期记录`);
            } else {
                alert(`清理失败: ${data.error || 'unknown'}`);
            }
            loadMonitorCache();
        } catch (e) {
            alert(`清理失败: ${e.message}`);
        }
    }

    async function clearSQLiteCache() {
        if (!confirm('确定要清空所有缓存吗？')) return;
        try {
            const resp = await fetch('/api/cache/clear', {method: 'POST'});
            const data = await resp.json();
            alert(`已清空: ${data.message || 'done'}`);
            loadMonitorCache();
        } catch (e) {
            alert(`清空失败: ${e.message}`);
        }
    }

    async function runBrinson() {
        try {
            const pw = JSON.parse(document.getElementById('brinson-pw').value);
            const bw = JSON.parse(document.getElementById('brinson-bw').value);
            const parts = document.getElementById('brinson-pr').value.split('|');
            const pr = JSON.parse(parts[0]);
            const br = JSON.parse(parts[1]);

            const resp = await fetch('/api/brinson/analyze', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({portfolio_weights: pw, benchmark_weights: bw, portfolio_returns: pr, benchmark_returns: br})
            });
            const raw = await resp.json();
            const r = raw.data || raw;
            let html = '<table style="width:100%;border-collapse:collapse;margin-top:12px">';
            html += '<tr style="border-bottom:1px solid var(--border-color)"><th style="text-align:left;padding:8px">行业</th><th style="text-align:right;padding:8px">配置效应</th><th style="text-align:right;padding:8px">选择效应</th><th style="text-align:right;padding:8px">交互效应</th><th style="text-align:right;padding:8px">合计</th></tr>';
            // industry_breakdown 或 sector_attribution
            const breakdown = r.industry_breakdown || r.sector_attribution || [];
            breakdown.forEach(ind => {
                const total = (ind.allocation || ind.total_effect || 0) + (ind.selection || 0) + (ind.interaction || 0);
                const color = total >= 0 ? '#10b981' : '#ef4444';
                const alloc = ind.allocation || 0;
                const sel = ind.selection || 0;
                const inter = ind.interaction || 0;
                const industry = ind.industry || ind.sector || '';
                html += `<tr style="border-bottom:1px solid rgba(255,255,255,0.05)">
                    <td style="padding:8px">${industry}</td>
                    <td style="text-align:right;padding:8px;color:${alloc>=0?'#10b981':'#ef4444'}">${(alloc*100).toFixed(2)}%</td>
                    <td style="text-align:right;padding:8px;color:${sel>=0?'#10b981':'#ef4444'}">${(sel*100).toFixed(2)}%</td>
                    <td style="text-align:right;padding:8px;color:${inter>=0?'#10b981':'#ef4444'}">${(inter*100).toFixed(2)}%</td>
                    <td style="text-align:right;padding:8px;color:${color};font-weight:700">${(total*100).toFixed(2)}%</td>
                </tr>`;
            });
            html += '</table>';
            // 字段名可能不同，兼容处理
            const excess = r.excess_return || r.total_excess_return || 0;
            const allocEffect = r.allocation_effect || 0;
            const selEffect = r.selection_effect || 0;
            const interEffect = r.interaction_effect || 0;
            html += `<div style="margin-top:12px;padding:12px;background:rgba(59,130,246,0.1);border-radius:8px">
                <div style="font-weight:700;margin-bottom:4px">总超额收益: ${(excess*100).toFixed(2)}%</div>
                <div style="font-size:12px;color:var(--text-secondary)">配置效应: ${(allocEffect*100).toFixed(2)}% | 选择效应: ${(selEffect*100).toFixed(2)}% | 交互效应: ${(interEffect*100).toFixed(2)}%</div>
                <div style="font-size:12px;color:var(--text-secondary);margin-top:4px">${r.conclusion || ''}</div>
            </div>`;
            document.getElementById('brinson-result').innerHTML = html;
        } catch (e) {
            document.getElementById('brinson-result').innerHTML = `<div style="color:#ef4444">错误: ${e.message}</div>`;
        }
    }

    // ========================================================================
    // 跨市场 Functions
    // ========================================================================
    async function loadCrossMarketTab() {
        console.log('[loadCrossMarketTab] START');
        loadMoiraiStatus();
        loadDRLStatus();
        loadCrossFusion();
        loadCrossMarketSummary();
        loadCrossSentiment();
        loadCrossAHPremium();
        console.log('[loadCrossMarketTab] ALL fetches dispatched, cross-moirai-status:', document.getElementById('cross-moirai-status'));
    }

    async function loadMoiraiStatus() {
        try {
            const resp = await fetch('/api/sota/moirai/status');
            const raw = await resp.json();
            const data = raw.data || raw;
            document.getElementById('cross-moirai-status').innerHTML = `
                <div style="display:grid; grid-template-columns:repeat(2,1fr); gap:8px">
                    <div><span style="color:var(--text-secondary)">序列长度:</span> ${data.seq_len || '--'}</div>
                    <div><span style="color:var(--text-secondary)">预测长度:</span> ${data.pred_len || '--'}</div>
                    <div><span style="color:var(--text-secondary)">已训练:</span> ${data.trained ? '✓' : '✗'}</div>
                    <div><span style="color:var(--text-secondary)">模型类型:</span> ${data.model_type || 'Moirai'}</div>
                </div>`;
        } catch (e) {
            document.getElementById('cross-moirai-status').innerHTML = `<div style="color:#ef4444">加载失败: ${e.message}</div>`;
        }
    }

    async function loadDRLStatus() {
        try {
            const resp = await fetch('/api/sota/drl/status');
            const raw = await resp.json();
            const data = raw.data || raw;
            document.getElementById('cross-drl-status').innerHTML = `
                <div style="display:grid; grid-template-columns:repeat(2,1fr); gap:8px">
                    <div><span style="color:var(--text-secondary)">状态维度:</span> ${data.state_dim || 24}</div>
                    <div><span style="color:var(--text-secondary)">动作维度:</span> ${data.action_dim || 2}</div>
                    <div><span style="color:var(--text-secondary)">已训练:</span> ${data.trained ? '✓' : '✗'}</div>
                    <div><span style="color:var(--text-secondary)">最大仓位:</span> ${(data.max_weight*100 || 80).toFixed(0)}%</div>
                </div>`;
        } catch (e) {
            document.getElementById('cross-drl-status').innerHTML = `<div style="color:#ef4444">加载失败: ${e.message}</div>`;
        }
    }

    async function loadCrossMarketSummary() {
        try {
            const resp = await fetch('/api/sota/cross-market/summary');
            const raw = await resp.json();
            const data = raw.data || raw;
            let html = '<div style="display:grid; grid-template-columns:repeat(3,1fr); gap:12px">';
            const markets = data.markets || {};
            const entries = Object.entries(markets);
            if (entries.length) {
                entries.forEach(([market, info]) => {
                    const n_stocks = typeof info === 'object' ? (info.n_stocks || info.count || 0) : (info || 0);
                    const n_days = typeof info === 'object' ? (info.n_historical_days || info.days || 0) : 0;
                    html += `<div style="padding:12px;background:rgba(59,130,246,0.05);border-radius:8px">
                        <div style="font-weight:700">${market}</div>
                        <div style="font-size:12px;color:var(--text-secondary);margin-top:4px">
                            股票: ${n_stocks}<br>
                            历史天数: ${n_days}
                        </div>
                    </div>`;
                });
            } else {
                html = '<div style="color:var(--text-secondary);padding:12px">暂无跨市场数据（需先分析股票）</div>';
            }
            html += `<div style="padding:12px;background:rgba(16,185,129,0.05);border-radius:8px">
                <div style="font-weight:700">总计</div>
                <div style="font-size:12px;color:var(--text-secondary);margin-top:4px">
                    股票: ${data.total_stocks || entries.reduce((s,v) => s + (typeof v[1]==='object'?v[1].n_stocks||v[1].count||0:v[1]||0), 0)}<br>
                    套利机会: ${data.arbitrage_opportunities || 0}
                </div>
            </div>`;
            html += '</div>';
            document.getElementById('cross-fusion-result').innerHTML = html;
        } catch (e) {
            document.getElementById('cross-fusion-result').innerHTML = `<div style="color:#ef4444">加载失败: ${e.message}</div>`;
        }
    }

    async function loadCrossSentiment() {
        ['A', 'HK', 'US'].forEach(async market => {
            try {
                const resp = await fetch(`/api/sota/cross-market/sentiment/${market}`);
                const raw = await resp.json();
                const data = raw.data || raw;
                const el = document.getElementById(`cm-sentiment-${market}`);
                if (el) {
                    const score = Math.max(0, Math.min(1, data.score || 0));
                    el.textContent = (score * 100).toFixed(0) + '%';
                    el.style.color = score > 0.5 ? '#10b981' : score < 0.5 ? '#ef4444' : '#d29922';
                }
            } catch (e) {
                const el = document.getElementById(`cm-sentiment-${market}`);
                if (el) el.textContent = 'N/A';
            }
        });
    }

    async function loadCrossFusion() {
        try {
            const resp = await fetch('/api/sota/cross-market/fusion');
            const raw = await resp.json();
            const d = raw.data || raw;
            const el = document.getElementById('cross-drl-result');
            if (el) {
                const dirColors = { bullish: '#10b981', bearish: '#ef4444', neutral: '#d29922' };
                const dir = d.direction || 'neutral';
                const conf = (d.confidence * 100).toFixed(1);
                const breakdown = d.market_breakdown || {};
                el.innerHTML =
                    `<div style="display:grid; grid-template-columns:repeat(2,1fr); gap:8px">
                        <div style="grid-column:1/-1;text-align:center">
                            <span style="font-size:24px;font-weight:700;color:${dirColors[dir] || '#d29922'}">${dir.toUpperCase()}</span>
                            <span style="font-size:14px;color:var(--text-secondary);margin-left:8px">置信度 ${conf}%</span>
                        </div>
                        <div><span style="color:var(--text-secondary)">A 股:</span> <strong>${breakdown.A || '-'}</strong></div>
                        <div><span style="color:var(--text-secondary)">港股:</span> <strong>${breakdown.HK || '-'}</strong></div>
                        <div><span style="color:var(--text-secondary)">美股:</span> <strong>${breakdown.US || '-'}</strong></div>
                        <div><span style="color:var(--text-secondary)">信号数:</span> ${d.n_signals || 0}</div>
                    </div>`;
            }
        } catch (e) {
            const el = document.getElementById('cross-drl-result');
            if (el) el.innerHTML = `<div style="color:#ef4444">加载失败: ${e.message}</div>`;
        }
    }

    async function loadCrossAHPremium() {
        try {
            const resp = await fetch('/api/sota/cross-market/ah-premium');
            const raw = await resp.json();
            const d = raw.data || raw;
            const el = document.getElementById('cross-ah-result');
            if (el) {
                const color = d.premium_pct > 15 ? '#ef4444' : d.premium_pct < -15 ? '#10b981' : '#d29922';
                el.innerHTML =
                    `<div style="display:grid; grid-template-columns:repeat(2,1fr); gap:8px">
                        <div><span style="color:var(--text-secondary)">A股价格:</span> ¥${d.a_price || '-'}</div>
                        <div><span style="color:var(--text-secondary)">H股(CNY):</span> ¥${d.h_price_cny || '-'}</div>
                        <div><span style="color:var(--text-secondary)">溢价率:</span> <strong style="color:${color}">${d.premium_pct?.toFixed(2) || 0}%</strong></div>
                        <div><span style="color:var(--text-secondary)">信号:</span> ${d.signal || '无数据'}</div>
                    </div>`;
            }
        } catch (e) {
            const el = document.getElementById('cross-ah-result');
            if (el) el.innerHTML = `<div style="color:#ef4444">加载失败: ${e.message}</div>`;
        }
    }

    // 页面加载完成后自动加载跨市场和监控数据
    function initCrossMarketAndMonitor() {
        console.log('[INDEX] DOM ready, loading cross-market and monitor...');
        setTimeout(() => {
            console.log('[INDEX] Calling loadCrossMarketTab()...');
            loadCrossMarketTab().then(() => console.log('[INDEX] loadCrossMarketTab done'));
            console.log('[INDEX] Calling loadMonitorTab()...');
            loadMonitorTab().then(() => console.log('[INDEX] loadMonitorTab done'));
        }, 200);
    }
    if (document.readyState === 'loading') {
        window.addEventListener('DOMContentLoaded', initCrossMarketAndMonitor);
    } else {
        initCrossMarketAndMonitor();
    }
    });

    // ============================================================================
    // Phase 2: 多智能体共识
    // ============================================================================

    function loadConsensusTab() {
        loadConsensusWeights();
        loadConsensusHistory();
    }

    async function runConsensus() {
        const stockCode = document.getElementById('consensus-stock').value || 'sz300620';
        const container = document.getElementById('consensus-result');
        container.innerHTML = '<p style="color:var(--text-secondary);text-align:center;">共识决策中...</p>';

        try {
            const resp = await fetch(`/api/consensus/decide?stock_code=${stockCode}`);
            const data = await resp.json();
            if (data.success) {
                // 兼容 data.data 嵌套和直接返回格式
                const d = data.consensus || data.data || data;
                let html = '';
                html += '<div style="text-align:center;margin-bottom:16px;">';
                const color = d.consensus.includes('buy') ? '#10b981' : d.consensus.includes('sell') ? '#ef4444' : 'var(--text-primary)';
                html += `<div style="font-size:24px;font-weight:bold;color:${color};">${d.consensus.replace('_', ' ').toUpperCase()}</div>`;
                html += `<div style="color:var(--text-secondary);margin-top:4px;">置信度: ${(d.confidence * 100).toFixed(1)}%</div>`;
                html += '</div>';

                html += '<div style="margin-top:12px;">';
                html += '<div style="color:var(--text-secondary);margin-bottom:8px;">各智能体投票:</div>';
                for (const vote of d.agent_votes) {
                    const vcolor = vote.decision.includes('buy') ? '#10b981' : vote.decision.includes('sell') ? '#ef4444' : 'var(--text-secondary)';
                    html += `<div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.05);">`;
                    html += `<span style="color:var(--text-secondary);">${vote.role}</span>`;
                    html += `<span style="color:${vcolor};">${vote.decision} (${(vote.confidence * 100).toFixed(0)}%)</span>`;
                    html += '</div>';
                }
                html += '</div>';

                if (d.conflict) {
                    html += `<div style="margin-top:12px;padding:8px;background:rgba(239,68,68,0.1);border-radius:6px;color:#ef4444;">⚠️ 检测到冲突 (多空比例: ${d.conflict_ratio.toFixed(2)})</div>`;
                }
                container.innerHTML = html;
            } else {
                container.innerHTML = `<p style="color:#ef4444;">错误: ${data.error}</p>`;
            }
        } catch (e) {
            container.innerHTML = `<p style="color:#ef4444;">请求失败: ${e.message}</p>`;
        }
    }

    async function loadConsensusWeights() {
        const container = document.getElementById('consensus-weights');
        try {
            const resp = await fetch('/api/consensus/weights');
            const data = await resp.json();
            if (data.success) {
                // 兼容 data.data.current_weights 和直接返回 data.weights
                const w = data.weights || data.data?.current_weights || {};
                let html = '';
                for (const [role, weight] of Object.entries(w)) {
                    const pct = (weight * 100).toFixed(0);
                    html += `<div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.05);">`;
                    html += `<span style="color:var(--text-secondary);">${role}</span>`;
                    html += `<span style="color:var(--accent-blue);">${pct}%</span>`;
                    html += '</div>';
                }
                container.innerHTML = html || '<p style="color:var(--text-secondary);">无权重数据</p>';
            }
        } catch (e) {
            container.innerHTML = `<p style="color:#ef4444;">${e.message}</p>`;
        }
    }

    async function loadConsensusHistory() {
        const container = document.getElementById('consensus-history');
        try {
            const resp = await fetch('/api/consensus/history');
            const data = await resp.json();
            if (data.success) {
                // 兼容 data.data 和直接返回 data.history
                const history = data.history || data.data || [];
                if (history.length === 0) {
                    container.innerHTML = '<p style="color:var(--text-secondary);text-align:center;padding:20px 0;">暂无历史记录</p>';
                    return;
                }
                let html = '';
                const recent = history.slice(-10).reverse();
                for (const h of recent) {
                    const color = h.consensus.includes('buy') ? '#10b981' : h.consensus.includes('sell') ? '#ef4444' : 'var(--text-secondary)';
                    html += `<div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.05);">`;
                    html += `<span style="color:var(--text-secondary);font-size:11px;">${h.timestamp || '-'}</span>`;
                    html += `<span style="color:${color};font-weight:bold;">${h.consensus}</span>`;
                    html += `<span style="color:var(--text-secondary);">${h.stock_code || '-'}</span>`;
                    html += '</div>';
                }
                container.innerHTML = html;
            }
        } catch (e) {
            container.innerHTML = `<p style="color:#ef4444;">${e.message}</p>`;
        }
    }

    // ============================================================================
    // Phase 3: Safe RL + 贝叶斯 + 在线学习
    // ============================================================================

    function loadSafeRLTab() {
        loadSafeRLStatus();
        loadUncertaintyStatus();
        loadOnlineLearningStatus();
    }

    async function loadSafeRLStatus() {
        const container = document.getElementById('safe-rl-status');
        try {
            const resp = await fetch('/api/safe-rl/status');
            const data = await resp.json();
            if (data.success) {
                // 兼容 data.data.constraints 和直接返回 data.constraints
                const c = data.constraints || data.data?.constraints || {};
                let html = '<div style="font-size:13px;">';
                html += `<div style="padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.05);">状态: ${data.status || 'initialized'}</div>`;
                html += `<div style="padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.05);">算法: ${data.algorithm || 'PPO'}</div>`;
                html += `<div style="padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.05);">回撤上限: ${(c.max_drawdown || 0.15) * 100}%</div>`;
                html += `<div style="padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.05);">VaR 限制: ${(c.var_limit || 0.05) * 100}%</div>`;
                html += `<div style="padding:4px 0;">仓位限制: ${(c.position_limit || 0.30) * 100}%</div>`;
                html += '</div>';
                container.innerHTML = html;
            }
        } catch (e) {
            container.innerHTML = `<p style="color:#ef4444;">${e.message}</p>`;
        }
    }

    async function loadUncertaintyStatus() {
        const container = document.getElementById('uncertainty-status');
        try {
            const resp = await fetch('/api/uncertainty/status');
            const data = await resp.json();
            if (data.success) {
                // 兼容 data.data 和直接返回
                const d = data.conformal || data.data?.conformal || {};
                const methods = data.methods || [];
                let html = '<div style="font-size:13px;">';
                html += `<div style="padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.05);">状态: ${data.status || 'initialized'}</div>`;
                html += `<div style="padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.05);">方法: ${methods.join(', ') || 'conformal'}</div>`;
                html += `<div style="padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.05);">目标覆盖率: ${(d.coverage_target || 0.90) * 100}%</div>`;
                html += `<div style="padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.05);">实际覆盖率: ${(d.current_coverage || 0.88) * 100}%</div>`;
                html += `<div style="padding:4px 0;">区间宽度: ${d.interval_width || 0.045}</div>`;
                html += '</div>';
                container.innerHTML = html;
            }
        } catch (e) {
            container.innerHTML = `<p style="color:#ef4444;">${e.message}</p>`;
        }
    }

    async function loadOnlineLearningStatus() {
        const container = document.getElementById('online-learning-status');
        try {
            const resp = await fetch('/api/online-learning/status');
            const data = await resp.json();
            if (data.success) {
                // 兼容 data.data 和直接返回
                const d = data.data || data;
                let html = '<div style="font-size:13px;">';
                html += `<div style="padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.05);">状态: ${d.status || 'stopped'}</div>`;
                html += `<div style="padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.05);">监控股票: ${d.n_stocks_monitored || 0}</div>`;
                html += `<div style="padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.05);">告警数: ${d.n_alerts || 0}</div>`;
                html += `<div style="padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.05);">准确率阈值: ${(d.accuracy_threshold || 0.5) * 100}%</div>`;
                html += `<div style="padding:4px 0;">最小样本: ${d.min_samples || 30}</div>`;
                html += '</div>';
                container.innerHTML = html;
            }
        } catch (e) {
            container.innerHTML = `<p style="color:#ef4444;">${e.message}</p>`;
        }
    }

    // ============================================================================
    // Phase 4: MLOps + 实时 GNN
    // ============================================================================

    function loadMLOPSTab() {
        loadMLOpsStatus();
        loadGNNRealtimeStatus();
    }

    async function loadMLOpsStatus() {
        const container = document.getElementById('mlops-status');
        try {
            const resp = await fetch('/api/mlops/status');
            const data = await resp.json();
            if (data.success) {
                // 兼容 data.data 和直接返回
                const d = data.data || data;
                let html = '<div style="font-size:13px;">';
                html += `<div style="padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.05);">状态: ${d.status || 'idle'}</div>`;
                html += `<div style="padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.05);">重训练次数: ${d.retrain_count || 0}</div>`;
                html += `<div style="padding:4px 0;">部署次数: ${d.deployment_count || 0}</div>`;
                html += '</div>';
                container.innerHTML = html;
            }
        } catch (e) {
            container.innerHTML = `<p style="color:#ef4444;">${e.message}</p>`;
        }
    }

    async function loadGNNRealtimeStatus() {
        const container = document.getElementById('gnn-realtime-status');
        try {
            const resp = await fetch('/api/gnn/realtime/status');
            const data = await resp.json();
            if (data.success) {
                // 兼容 data.data 和直接返回
                const d = data.data || data;
                let html = '<div style="font-size:13px;">';
                html += `<div style="padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.05);">节点: ${d.n_nodes || 0}</div>`;
                html += `<div style="padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.05);">边: ${d.n_edges || 0}</div>`;
                html += `<div style="padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.05);">市场: ${(d.markets || []).join(', ') || '无'}</div>`;
                html += `<div style="padding:4px 0;">跨市场信号: ${d.cross_signals?.length || 0}</div>`;
                html += '</div>';
                container.innerHTML = html;
            }
        } catch (e) {
            container.innerHTML = `<p style="color:#ef4444;">${e.message}</p>`;
        }
    }

    // ============================================================================
    // 新增 API 集成函数
    // ============================================================================

    // --- 系统健康 & 告警 ---
    async function loadHealthStatus() {
        const container = document.getElementById('health-status');
        try {
            const resp = await fetch('/api/health');
            const data = await resp.json();
            if (data && (data.success || data.status)) {
                const d = data.data || data;
                let html = '<div style="font-size:13px;">';
                html += `<div style="padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.05);">状态: ${d.status || d.state || 'unknown'}</div>`;
                html += `<div style="padding:4px 0;">Python: ${d.python || '--'}</div>`;
                html += `<div style="padding:4px 0;">Uptime: ${d.uptime || '--'}</div>`;
                html += '</div>';
                container.innerHTML = html;
            } else {
                container.innerHTML = '<p style="color:var(--text-secondary);text-align:center;padding:10px;">无数据</p>';
            }
        } catch (e) {
            container.innerHTML = `<p style="color:#ef4444;">${e.message}</p>`;
        }
    }

    async function loadAlerts() {
        const container = document.getElementById('alerts-list');
        try {
            const resp = await fetch('/api/alerts/sz300620');
            const data = await resp.json();
            if (data.success && data.data) {
                const alerts = data.data;
                if (alerts.length === 0) {
                    container.innerHTML = '<p style="color:var(--text-secondary);text-align:center;padding:10px;">暂无告警</p>';
                    return;
                }
                let html = '';
                for (const a of alerts.slice(-10).reverse()) {
                    html += `<div style="padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.05);font-size:12px;">`;
                    html += `<span style="color:${a.level === 'warning' ? '#f59e0b' : '#ef4444'};">[${a.level || 'info'}]</span> `;
                    html += `<span style="color:var(--text-secondary);">${a.message || a.type || '-'}</span>`;
                    html += `</div>`;
                }
                container.innerHTML = html;
            } else {
                container.innerHTML = '<p style="color:var(--text-secondary);text-align:center;padding:10px;">暂无告警</p>';
            }
        } catch (e) {
            container.innerHTML = `<p style="color:#ef4444;">${e.message}</p>`;
        }
    }

    async function loadSystemMetrics() {
        const container = document.getElementById('system-metrics');
        try {
            const resp = await fetch('/api/monitor/metrics');
            const data = await resp.json();
            if (data.success && data.data) {
                const d = data.data;
                let html = '<div style="font-size:13px;">';
                html += `<div style="padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.05);">API 调用: ${d.api_calls || 0}</div>`;
                html += `<div style="padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.05);">错误率: ${(d.error_rate || 0).toFixed(1)}%</div>`;
                html += `<div style="padding:4px 0;">平均延迟: ${d.avg_latency_ms || '--'}ms</div>`;
                html += '</div>';
                container.innerHTML = html;
            }
        } catch (e) {
            container.innerHTML = `<p style="color:#ef4444;">${e.message}</p>`;
        }
    }

    // --- 自动重训练 ---
    async function loadRetrainStatus() {
        const container = document.getElementById('retrain-status');
        try {
            const resp = await fetch('/api/retrain/status');
            const data = await resp.json();
            if (data.success) {
                const d = data.data;
                let html = '<div style="font-size:13px;">';
                html += `<div style="padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.05);">状态: ${d.status || 'idle'}</div>`;
                html += `<div style="padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.05);">最后触发: ${d.last_triggered || '无'}</div>`;
                html += `<div style="padding:4px 0;">冷却中: ${d.cooling_down ? '是' : '否'}</div>`;
                html += '</div>';
                container.innerHTML = html;
            }
        } catch (e) {
            container.innerHTML = `<p style="color:#ef4444;">${e.message}</p>`;
        }
    }

    async function triggerRetrain() {
        const container = document.getElementById('retrain-status');
        container.innerHTML = '<p style="color:var(--accent-blue);text-align:center;padding:10px;">触发重训练...</p>';
        try {
            const resp = await fetch('/api/retrain/trigger', { method: 'POST', headers: {'Content-Type': 'application/json'}});
            const data = await resp.json();
            if (data.success) {
                container.innerHTML = `<p style="color:#10b981;text-align:center;padding:10px;">✓ ${data.message || '已触发'}</p>`;
                loadRetrainStatus();
            } else {
                container.innerHTML = `<p style="color:#ef4444;">${data.error || '失败'}</p>`;
            }
        } catch (e) {
            container.innerHTML = `<p style="color:#ef4444;">${e.message}</p>`;
        }
    }

    // --- LLM 因子 ---
    async function extractLLMFactors() {
        const container = document.getElementById('llm-factors-result');
        container.innerHTML = '<p style="color:var(--accent-blue);text-align:center;padding:10px;">LLM 因子提取中...</p>';
        try {
            const resp = await fetch('/api/llm/factors?stock_code=sz300620');
            const data = await resp.json();
            if (data.success && data.data && Array.isArray(data.data)) {
                const factors = data.data.slice(0, 20);
                let html = '';
                for (const f of factors) {
                    html += `<div style="padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.05);font-size:12px;">`;
                    html += `<span style="color:var(--accent-blue);font-weight:bold;">${f.name || f.factor || '-'}</span> `;
                    html += `<span style="color:var(--text-secondary);">(${(f.value || 0).toFixed(2)})</span>`;
                    html += `</div>`;
                }
                container.innerHTML = html || '<p style="color:var(--text-secondary);text-align:center;padding:10px;">无因子</p>';
            }
        } catch (e) {
            container.innerHTML = `<p style="color:#ef4444;">${e.message}</p>`;
        }
    }

    // --- RL 交易器 ---
    async function loadRLTraderStatus() {
        const container = document.getElementById('rl-trader-status');
        const stockCode = document.getElementById('rl-stock-code').value || 'sz300620';
        try {
            const resp = await fetch(`/api/rl/trader/status`);
            const data = await resp.json();
            if (data.success) {
                // 兼容 data.data 和 data.status 和直接返回
                const d = data.status || data.data || data;
                let html = '<div style="font-size:13px;">';
                html += `<div style="padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.05);">状态: ${d.trained ? '已训练' : '未训练'}</div>`;
                html += `<div style="padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.05);">PPO: ${d.ppo_available ? '可用' : '未安装'}</div>`;
                html += `<div style="padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.05);">SAC: ${d.sac_available ? '可用' : '未安装'}</div>`;
                html += `<div style="padding:4px 0;">市场状态: ${d.market_regime || 'N/A'}</div>`;
                html += '</div>';
                container.innerHTML = html;
            }
        } catch (e) {
            container.innerHTML = `<p style="color:#ef4444;">${e.message}</p>`;
        }
    }

    // --- 在线学习记录 ---
    async function recordOnlineLearning() {
        const container = document.getElementById('online-learning-record');
        const stockCode = document.getElementById('ol-stock-code').value || 'sz300620';
        container.innerHTML = '<p style="color:var(--accent-blue);text-align:center;padding:10px;">记录中...</p>';
        try {
            const resp = await fetch('/api/online-learning/record', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({stock_code: stockCode, prediction: 0.6, actual: 0.7})
            });
            const data = await resp.json();
            if (data.success) {
                container.innerHTML = '<p style="color:#10b981;text-align:center;padding:10px;">✓ 已记录</p>';
                loadOnlineLearningStatus();
            } else {
                container.innerHTML = `<p style="color:#ef4444;">${data.error || '失败'}</p>`;
            }
        } catch (e) {
            container.innerHTML = `<p style="color:#ef4444;">${e.message}</p>`;
        }
    }

    // --- 动态 GNN 更新 ---
    async function updateGNNRealtime() {
        const container = document.getElementById('gnn-realtime-status');
        container.innerHTML = '<p style="color:var(--accent-blue);text-align:center;padding:10px;">更新图中...</p>';
        try {
            const resp = await fetch('/api/gnn/realtime/update', { method: 'POST', headers: {'Content-Type': 'application/json'}});
            const data = await resp.json();
            if (data.success) {
                container.innerHTML = '<p style="color:#10b981;text-align:center;padding:10px;">✓ 图已更新</p>';
                loadGNNRealtimeStatus();
            }
        } catch (e) {
            container.innerHTML = `<p style="color:#ef4444;">${e.message}</p>`;
        }
    }

    // --- ML 训练 ---
    async function trainMLModel() {
        const container = document.getElementById('ml-training-status');
        const modelName = document.getElementById('ml-model-name').value || 'patchtst';
        container.innerHTML = '<p style="color:var(--accent-blue);text-align:center;padding:10px;">训练中...</p>';
        try {
            const resp = await fetch('/api/ml/train', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({model_name: modelName})
            });
            const data = await resp.json();
            if (data.success) {
                container.innerHTML = `<p style="color:#10b981;text-align:center;padding:10px;">✓ 训练任务已提交 (${data.data?.task_id || modelName})</p>`;
            } else {
                container.innerHTML = `<p style="color:#ef4444;">${data.error || '失败'}</p>`;
            }
        } catch (e) {
            container.innerHTML = `<p style="color:#ef4444;">${e.message}</p>`;
        }
    }

    // --- 训练流水线 ---
    async function runTrainingPipeline() {
        const container = document.getElementById('training-pipeline-status');
        container.innerHTML = '<p style="color:var(--accent-blue);text-align:center;padding:10px;">运行流水线...</p>';
        try {
            const resp = await fetch('/api/training/pipeline/run', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({model_name: 'patchtst'})
            });
            const data = await resp.json();
            if (data.success) {
                container.innerHTML = `<p style="color:#10b981;text-align:center;padding:10px;">✓ 流水线已启动</p>`;
            } else {
                container.innerHTML = `<p style="color:#ef4444;">${data.error || '失败'}</p>`;
            }
        } catch (e) {
            container.innerHTML = `<p style="color:#ef4444;">${e.message}</p>`;
        }
    }

    async function loadTrainingMetrics() {
        const container = document.getElementById('training-pipeline-status');
        try {
            const resp = await fetch('/api/training/metrics');
            const data = await resp.json();
            if (data.success && data.data) {
                const d = data.data;
                let html = '<div style="font-size:13px;">';
                html += `<div style="padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.05);">总训练: ${d.total_trained || 0}</div>`;
                html += `<div style="padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.05);">最新: ${d.last_trained || '无'}</div>`;
                html += `<div style="padding:4px 0;">模型: ${d.models?.join(', ') || '无'}</div>`;
                html += '</div>';
                container.innerHTML = html;
            }
        } catch (e) {
            container.innerHTML = `<p style="color:#ef4444;">${e.message}</p>`;
        }
    }

    // ============================================================================
    // P0-5: 多模态融合预测面板
    // ============================================================================

    async function loadFusionPanel() {
        const container = document.getElementById('fusion-panel');
        try {
            const resp = await fetch('/api/fusion/multi-modal?stock_code=sz300620&cache=false');
            const data = await resp.json();
            if (data.success) {
                const d = data;
                const dirColor = d.direction === 'up' ? '#10b981' : d.direction === 'down' ? '#ef4444' : '#d29922';
                const dirIcon = d.direction === 'up' ? '📈' : d.direction === 'down' ? '📉' : '➡️';
                const dirText = d.direction === 'up' ? '看涨' : d.direction === 'down' ? '看跌' : '震荡';

                let html = '<div style="font-size:13px;">';
                // 主预测
                html += `<div style="text-align:center;padding:12px 0;border-bottom:1px solid rgba(255,255,255,0.05);">`;
                html += `<div style="font-size:24px;">${dirIcon}</div>`;
                html += `<div style="font-size:18px;font-weight:bold;color:${dirColor};">${dirText}</div>`;
                html += `<div style="font-size:12px;color:#94a3b8;">置信度: ${(d.confidence * 100).toFixed(1)}%</div>`;
                html += `<div style="font-size:12px;color:#94a3b8;">Regime: ${d.regime} | 一致度: ${(d.consensus * 100).toFixed(0)}%</div>`;
                html += `</div>`;
                // 模态权重
                html += '<div style="margin-top:8px;">';
                html += '<div style="font-size:12px;color:#94a3b8;margin-bottom:4px;">模态权重</div>';
                const weights = d.modality_weights;
                html += `<div style="display:flex;justify-content:space-between;padding:2px 0;">`;
                html += `<span>📊 价格</span><span style="color:#60a5fa;">${(weights.price * 100).toFixed(0)}%</span>`;
                html += `</div>`;
                html += `<div style="display:flex;justify-content:space-between;padding:2px 0;">`;
                html += `<span>💬 情感</span><span style="color:#60a5fa;">${(weights.sentiment * 100).toFixed(0)}%</span>`;
                html += `</div>`;
                html += `<div style="display:flex;justify-content:space-between;padding:2px 0;">`;
                html += `<span>📋 基本面</span><span style="color:#60a5fa;">${(weights.fundamental * 100).toFixed(0)}%</span>`;
                html += `</div>`;
                html += '</div>';
                // 各模态得分
                html += '<div style="margin-top:8px;">';
                html += '<div style="font-size:12px;color:#94a3b8;margin-bottom:4px;">方向得分</div>';
                const scores = d.scores;
                html += `<div style="display:flex;justify-content:space-between;padding:2px 0;">`;
                html += `<span>📈 看涨</span><span style="color:#10b981;">${(scores.up * 100).toFixed(1)}%</span>`;
                html += `</div>`;
                html += `<div style="display:flex;justify-content:space-between;padding:2px 0;">`;
                html += `<span>➡️ 震荡</span><span style="color:#d29922;">${(scores.neutral * 100).toFixed(1)}%</span>`;
                html += `</div>`;
                html += `<div style="display:flex;justify-content:space-between;padding:2px 0;">`;
                html += `<span>📉 看跌</span><span style="color:#ef4444;">${(scores.down * 100).toFixed(1)}%</span>`;
                html += `</div>`;
                html += '</div>';
                // 各模态独立信号
                html += '<div style="margin-top:8px;">';
                html += '<div style="font-size:12px;color:#94a3b8;margin-bottom:4px;">各模态信号</div>';
                const mods = d.modalities;
                const modLabels = {price: '📊 价格', sentiment: '💬 情感', fundamental: '📋 基本面'};
                for (const [key, mod] of Object.entries(mods)) {
                    const dir = mod.direction === 'up' ? '📈' : mod.direction === 'down' ? '📉' : '➡️';
                    html += `<div style="display:flex;justify-content:space-between;padding:2px 0;">`;
                    html += `<span>${modLabels[key] || key}</span><span>${dir} ${(mod.confidence * 100).toFixed(0)}%</span>`;
                    html += `</div>`;
                }
                html += '</div>';
                html += '</div>';
                container.innerHTML = html;
            } else {
                container.innerHTML = `<p style="color:#ef4444;">加载失败: ${data.error || '未知错误'}</p>`;
            }
        } catch (e) {
            container.innerHTML = `<p style="color:#ef4444;">${e.message}</p>`;
        }
    }

    // ============================================================================
    // P0-6: 金丝雀部署面板
    // ============================================================================

    async function loadCanaryPanel() {
        const container = document.getElementById('canary-panel');
        try {
            const resp = await fetch('/api/mlops/canary/status');
            const data = await resp.json();
            if (data.success) {
                const active = data.active_count || 0;
                const deployments = data.deployments || {};
                const keys = Object.keys(deployments);

                let html = '<div style="font-size:13px;">';
                html += `<div style="padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.05);">活跃部署: <span style="color:${active > 0 ? '#10b981' : '#94a3b8'}">${active}</span></div>`;

                for (const [code, dep] of Object.entries(deployments)) {
                    const statusColor = dep.status === 'promoted' ? '#10b981' : dep.status === 'rolled_back' ? '#ef4444' : dep.status === 'scaling' ? '#d29922' : '#3b82f6';
                    const statusText = dep.status === 'promoted' ? '已提升' : dep.status === 'rolled_back' ? '已回滚' : dep.status === 'scaling' ? '流量提升中' : dep.status === 'ended' ? '已结束' : '活跃';

                    html += '<div style="margin-top:8px;padding:8px;background:rgba(15,23,42,0.5);border-radius:6px;">';
                    html += `<div style="font-weight:bold;color:${statusColor};margin-bottom:4px;">${code} — ${statusText}</div>`;
                    html += `<div style="font-size:11px;color:#94a3b8;">${dep.old_version} → ${dep.new_version}</div>`;
                    html += `<div style="font-size:11px;color:#94a3b8;">流量: ${(dep.traffic_ratio * 100).toFixed(0)}% | 评估: ${dep.canary_total} 次</div>`;
                    if (dep.canary_total > 0) {
                        const canaryAcc = (dep.canary_correct / dep.canary_total * 100).toFixed(1);
                        html += `<div style="font-size:11px;color:#94a3b8;">Canary 准确率: ${canaryAcc}%</div>`;
                    }
                    if (dep.end_reason) {
                        html += `<div style="font-size:11px;color:#94a3b8;">原因: ${dep.end_reason}</div>`;
                    }
                    html += '</div>';
                }

                if (keys.length === 0) {
                    html += '<div style="color:#94a3b8;font-size:12px;margin-top:8px;">暂无活跃部署</div>';
                }

                html += '</div>';
                container.innerHTML = html;
            } else {
                container.innerHTML = `<p style="color:#ef4444;">加载失败: ${data.error || '未知错误'}</p>`;
            }
        } catch (e) {
            container.innerHTML = `<p style="color:#ef4444;">${e.message}</p>`;
        }
    }

    // ============================================================================
    // 页面加载：初始化所有新面板
    // ============================================================================

    function initAllPanels() {
        // 初始化所有面板
        loadHealthStatus();
        loadAlerts();
        loadSystemMetrics();
        loadRetrainStatus();
        extractLLMFactors();
        loadRLTraderStatus();
        loadTrainingMetrics();
        // P0-5 / P0-6
        loadFusionPanel();
        loadCanaryPanel();
    }
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initAllPanels);
    } else {
        initAllPanels();
    }
