/* DEAD CODE MARK (2026-09-23 全链 review, 方案A): DEAD CODE (2026-09-23 review): 零加载 (tab 页零引用). 内部 sota/timemoe/predict 09-23 实测 404 = 端点从未注册 (非断链, 死码区). 若重启用: 先核路径再接. */
// sota_dashboard_tabs.js — SOTA 子 Tab 切换 + Dashboard 渲染
// (2026-08-16 从 templates/index.html 内联 <script> 外置, 架构清理)

    // SOTA 子 Tab 切换
    function switchSOTATab(group) {
        document.querySelectorAll('#sotaSubTabs .quant-sub-tab').forEach(tab => {
            tab.classList.toggle('active', tab.dataset.group === group);
        });
        // 修复: data-group="sota-consensus" → ID="sota-group-consensus" (去掉 sota- 前缀)
        const baseGroup = group.replace(/^sota-/, '');
        document.querySelectorAll('[id^="sota-group-"]').forEach(g => {
            g.classList.toggle('active', g.id === `sota-group-${baseGroup}`);
        });
    }

    // SOTA 模型相关函数
    // 安全更新面板：失败时显示错误提示而非永久"加载中..."
    function safeUpdatePanel(elId, fetchPromise) {
        fetchPromise
            .then(r => r.json())
            .catch(e => { console.error(`[${elId}] fetch error:`, e); return {success: false, error: e.message}; })
            .then(data => {
                const el = document.getElementById(elId);
                if (!el) return;
                if (!data.success) {
                    el.innerHTML = `<div style="color:var(--text-secondary);font-size:12px;padding:8px">⚠ ${data.error || '数据暂不可用'}</div>`;
                }
            });
    }

    function loadSOTADashboard() {
        // 添加加载进度条
        const loadingOverlay = document.getElementById('loadingOverlay');
        if (loadingOverlay) {
            loadingOverlay.style.display = 'flex';
        }

        // 检查缓存状态
        fetch('/api/sota/cache/status')
            .then(r => r.json())
            .then(data => {
                if (data.success && data.cached) {
                    const cacheMsg = document.getElementById('sota-cache-msg');
                    if (cacheMsg) {
                        const isValid = data.is_valid ? '有效' : '已过期';
                        cacheMsg.textContent = `使用缓存决策 (年龄: ${data.age.toFixed(1)}s, ${isValid})`;
                        cacheMsg.style.display = 'block';
                    }
                }
            })
            .catch(e => console.error('Cache status error:', e));

        // 进度更新函数
        const updateProgress = (stage, progress) => {
            const progressText = document.getElementById('sota-progress-text');
            if (progressText) {
                progressText.textContent = `正在加载: ${stage} (${progress}%)`;
            }
        };

        // 获取股票代码
        const stockCode = document.getElementById('stockCode')?.value || 'sz300620';

        // 设置超时消息
        const timeoutMsg = document.getElementById('sota-timeout-msg');
        if (timeoutMsg) {
            timeoutMsg.style.display = 'none';
        }

        // ========== 第一阶段：快速面板 (10 秒超时) ==========
        const fastController = new AbortController();
        const fastTimeout = setTimeout(() => fastController.abort(), 10000);

        Promise.allSettled([
            // 引擎状态 (无股票依赖)
            fetch('/api/sota/ensemble/status', { signal: fastController.signal })
                .then(r => r.json())
                .then(data => {
                    if (data.success) {
                        const s = data.data;
                        const items = [];
                        if (s.patchtst) items.push({name: 'PatchTST', value: s.patchtst.is_trained ? '已训练' : '未训练'});
                        if (s.regime_switching) items.push({name: 'Regime-Switching', value: '已加载'});
                        if (s.time_llm) items.push({name: 'Time-LLM', value: '已加载'});
                        if (s.alpha158) items.push({name: 'Alpha158', value: s.alpha158.n_factors + ' factors'});
                        if (s.factor_weight_scheduler) items.push({name: '因子权重', value: '动态调度'});
                        if (s.gnn) items.push({name: 'GNN', value: s.gnn.is_trained ? '已训练' : '未训练'});
                        if (s.cvar) items.push({name: 'CVaR/EVT', value: '已加载'});
                        if (s.drift_advanced) items.push({name: '漂移检测', value: '高级'});
                        if (s.drift_monitor) items.push({name: 'Drift Monitor', value: (s.drift_monitor.drift_count || 0) + ' drifts'});
                        if (s.cross_market) items.push({name: '跨市场因子', value: '已加载'});
                        if (s.factor_ic) items.push({name: '因子 IC', value: '已加载'});
                        document.getElementById('sota-status').innerHTML = `
                            <div class="status-grid">
                                ${items.map(item => `<div class="status-item"><span>${item.name}</span><span class="status-ok">${item.value}</span></div>`).join('')}
                            </div>
                        `;
                    }
                    return data;
                }).catch(e => { if (e.name !== 'AbortError') console.error('SOTA status error:', e); }),

            // 动态因子权重
            fetch('/api/sota/factor-weights/status', { signal: fastController.signal })
                .then(r => r.json())
                .then(data => {
                    if (data.success && data.data) {
                        const w = data.data;
                        const html = `
                            <div class="metric"><span>调度策略</span><span>${w.strategy || 'N/A'}</span></div>
                            <div class="metric"><span>衰减模型</span><span>${w.decay_model || 'N/A'}</span></div>
                            <div class="metric"><span>IC 衰减率</span><span>${(w.ic_decay_rate * 100 || 0).toFixed(1)}%</span></div>
                            <div class="metric"><span>重平衡周期</span><span>${w.rebalance_period || 'N/A'}</span></div>
                        `;
                        document.getElementById('sota-dynamic-weights').innerHTML = html;
                    }
                }).catch(e => { if (e.name !== 'AbortError') console.error('SOTA factor-weights error:', e); }),

            // 高级漂移检测
            fetch('/api/sota/drift/advanced/status', { signal: fastController.signal })
                .then(r => r.json())
                .then(data => {
                    if (data.success && data.data) {
                        const d = data.data;
                        const html = `
                            <div class="metric"><span>检测算法</span><span>${d?.algorithm || 'ADWIN'}</span></div>
                            <div class="metric"><span>漂移计数</span><span>${d?.drift_count || 0}</span></div>
                            <div class="metric"><span>当前窗口</span><span>${d?.current_window_size || 'N/A'}</span></div>
                            <div class="metric"><span>漂移阈值</span><span>${(d?.threshold || 0.5).toFixed(3)}</span></div>
                        `;
                        document.getElementById('sota-drift').innerHTML = html;
                    }
                }).catch(e => { if (e.name !== 'AbortError') console.error('SOTA drift error:', e); }),

            // Alpha158 因子
            fetch(`/api/sota/alpha158/predict?code=${stockCode}`, { signal: fastController.signal })
                .then(r => r.json())
                .then(data => {
                    if (data.success && data.data) {
                        const d = data.data;
                        const topVals = d.factor_values_top20 ?
                            Object.entries(d.factor_values_top20).slice(0, 5).map(([k, v]) =>
                                `<div class="metric"><span>${k}</span><span>${Number(v).toFixed(4)}</span></div>`
                            ).join('') : '';
                        const html = `
                            <div class="metric"><span>因子总数</span><span>${d.n_factors || 'N/A'}</span></div>
                            ${topVals}
                            <div class="factor-list" style="max-height:120px;overflow-y:auto;margin-top:8px;font-size:12px;color:var(--text-secondary)">
                                ${Object.entries(d.latest_values || {}).slice(0, 10).map(([k, v]) =>
                                    `<div style="display:flex;justify-content:space-between"><span>${k}</span><span>${Number(v).toFixed(4)}</span></div>`
                                ).join('')}
                            </div>
                        `;
                        document.getElementById('sota-alpha158').innerHTML = html;
                    }
                }).catch(e => { if (e.name !== 'AbortError') console.error('SOTA alpha158 error:', e); }),

            // GNN 图神经网络
            fetch(`/api/sota/gnn/predict?code=${stockCode}`, { signal: fastController.signal })
                .then(r => r.json())
                .then(data => {
                    if (data.success && data.data) {
                        const d = data.data;
                        const conf = Array.isArray(d.confidence) ? d.confidence[0] : (d.confidence || 0);
                        const emb = d.gnn_embeddings;
                        const dim = Array.isArray(emb) && emb.length ? emb[0].length : 0;
                        const html = `
                            <div class="metric"><span>置信度</span><span>${(conf * 100 || 0).toFixed(1)}%</span></div>
                            <div class="metric"><span>嵌入维度</span><span>${dim}</span></div>
                            <div class="metric"><span>状态</span><span>已预测</span></div>
                        `;
                        document.getElementById('sota-gnn').innerHTML = html;
                    }
                }).catch(e => { if (e.name !== 'AbortError') console.error('SOTA gnn error:', e); }),

            // 跨市场因子
            fetch(`/api/sota/cross-market/factors?code=${stockCode}`, { signal: fastController.signal })
                .then(r => r.json())
                .then(data => {
                    if (data.success && data.data && data.data.factors) {
                        const f = data.data.factors;
                        const items = typeof f === 'object' && !Array.isArray(f)
                            ? Object.entries(f).map(([name, val]) =>
                                `<div class="metric"><span>${name}</span><span>${Number(val).toFixed(2)}</span></div>`
                            ).join('')
                            : f.map(x => `<div class="metric"><span>${x.name}</span><span>${(x.weight * 100 || 0).toFixed(1)}%</span></div>`).join('');
                        document.getElementById('sota-cross-market').innerHTML = items || '<p class="factor-count">暂无跨市场因子</p>';
                    }
                }).catch(e => { if (e.name !== 'AbortError') console.error('SOTA cross-market error:', e); }),

            // Regime-Switching 市场状态
            fetch('/api/sota/enhanced-regime/sz300620', { signal: fastController.signal })
                .then(r => r.json())
                .then(data => {
                    try {
                        if (data.success && data.data) {
                            const p = data.data;
                            const regime = p.regime || 'N/A';
                            const dirMap = { 'bull': '看涨', 'bear': '看跌', 'sideways': '震荡', 'volatile': '高波动' };
                            const dirText = dirMap[regime] || regime;
                            const html = `
                                <div class="metric"><span>市场状态</span><span>${regime}</span></div>
                                <div class="metric"><span>隐含方向</span><span>${dirText}</span></div>
                                <div class="metric"><span>置信度</span><span>${(Number(p.confidence) * 100 || 0).toFixed(1)}%</span></div>
                                <div class="metric"><span>Bear 概率</span><span>${((p.transition_prob || {})['bear'] * 100 || 0).toFixed(1)}%</span></div>
                            `;
                            document.getElementById('sota-regime').innerHTML = html;
                        }
                    } catch (e) {
                        console.error('SOTA regime render error:', e);
                    }
                    return data;
                }).catch(e => { if (e.name !== 'AbortError') console.error('SOTA regime error:', e); }),

            // CVaR 风险度量
            fetch(`/api/sota/cvar/risk?code=${stockCode}`, { signal: fastController.signal })
                .then(r => r.json())
                .then(data => {
                    if (data.success && data.data) {
                        const r = data.data;
                        const html = `
                            <div class="metric"><span>VaR (95% 日)</span><span>${(r.var?.var_95_daily || 0).toFixed(2)}%</span></div>
                            <div class="metric"><span>CVaR (95% 日)</span><span>${(r.cvar?.cvar_95_daily || 0).toFixed(2)}%</span></div>
                            <div class="metric"><span>VaR (99% 日)</span><span>${(r.var?.var_99_daily || 0).toFixed(2)}%</span></div>
                            <div class="metric"><span>最大损失</span><span>${(r.extreme_metrics?.max_loss || 0).toFixed(2)}%</span></div>
                        `;
                        document.getElementById('sota-risk').innerHTML = html;
                    }
                }).catch(e => { if (e.name !== 'AbortError') console.error('SOTA CVaR error:', e); }),

            // Self-Supervised 模型状态
            fetch('/api/sota/self-supervised/status', { signal: fastController.signal })
                .then(r => r.json())
                .then(data => {
                    if (data.success && data.data) {
                        const p = data.data;
                        const html = `
                            <div class="metric"><span>模型</span><span>${p.model || 'N/A'}</span></div>
                            <div class="metric"><span>已训练</span><span>${p.trained ? '✅ 是' : '❌ 否'}</span></div>
                            <div class="metric"><span>序列长度</span><span>${p.seq_len || 'N/A'}</span></div>
                            <div class="metric"><span>特征数</span><span>${p.n_features || 'N/A'}</span></div>
                        `;
                        document.getElementById('sota-self-supervised').innerHTML = html;
                    }
                }).catch(e => { if (e.name !== 'AbortError') console.error('SOTA self-supervised error:', e); }),

            // Diffusion 预测
            fetch(`/api/sota/diffusion/predict?code=${stockCode}`, { signal: fastController.signal })
                .then(r => r.json())
                .then(data => {
                    if (data.success && data.direction !== undefined) {
                        const probs = data.probabilities || {};
                        const html = `
                            <div class="metric"><span>方向</span><span>${data.direction === 'up' ? '上涨' : data.direction === 'down' ? '下跌' : data.direction}</span></div>
                            <div class="metric"><span>置信度</span><span>${(data.confidence * 100 || 0).toFixed(1)}%</span></div>
                            <div class="metric"><span>上涨概率</span><span>${((probs['up'] || 0) * 100).toFixed(1)}%</span></div>
                            <div class="metric"><span>不确定性</span><span>${(data.uncertainty?.std || 0).toFixed(4)}</span></div>
                        `;
                        document.getElementById('sota-diffusion').innerHTML = html;
                    }
                }).catch(e => { if (e.name !== 'AbortError') console.error('SOTA diffusion error:', e); }),

            // Mamba 预测
            fetch(`/api/sota/mamba/predict?stock_code=${stockCode}`, { signal: fastController.signal })
                .then(r => r.json())
                .then(data => {
                    if (data.success && data.direction !== undefined) {
                        const html = `
                            <div class="metric"><span>方向</span><span>${data.direction || 'N/A'}</span></div>
                            <div class="metric"><span>置信度</span><span>${(data.confidence * 100 || 0).toFixed(1)}%</span></div>
                            <div class="metric"><span>模型</span><span>${data.model || 'Mamba-2'}</span></div>
                            <div class="metric"><span>状态维度</span><span>${data.state_dim || 'N/A'}</span></div>
                        `;
                        document.getElementById('sota-mamba').innerHTML = html;
                    }
                }).catch(e => { if (e.name !== 'AbortError') console.error('SOTA mamba error:', e); }),

            // Conformal Prediction
            fetch(`/api/sota/conformal/predict?stock_code=${stockCode}`, { signal: fastController.signal })
                .then(r => r.json())
                .then(data => {
                    if (data.success && data.prediction !== undefined) {
                        const html = `
                            <div class="metric"><span>预测</span><span>${data.prediction || 'N/A'}</span></div>
                            <div class="metric"><span>下界</span><span>${(data.lower || 0).toFixed(4)}</span></div>
                            <div class="metric"><span>上界</span><span>${(data.upper || 0).toFixed(4)}</span></div>
                            <div class="metric"><span>覆盖率</span><span>${(data.coverage || 0).toFixed(4)}</span></div>
                        `;
                        document.getElementById('sota-conformal').innerHTML = html;
                    }
                }).catch(e => { if (e.name !== 'AbortError') console.error('SOTA conformal error:', e); }),

            // Time-MoE 零样本预测 (2026-08-17 新增)
            fetch(`/api/sota/timemoe/predict?code=${stockCode}`, { signal: fastController.signal })
                .then(r => r.json())
                .then(data => {
                    if (data.success && data.prediction !== undefined) {
                        const html = `
                            <div class="metric"><span>预测</span><span>${Number(data.prediction).toFixed(4) || 'N/A'}</span></div>
                            <div class="metric"><span>方向</span><span>${data.direction || 'N/A'}</span></div>
                            <div class="metric"><span>置信度</span><span>${(Number(data.confidence) * 100 || 0).toFixed(1)}%</span></div>
                            <div class="metric"><span>下界</span><span>${data.lower != null ? Number(data.lower).toFixed(4) : 'N/A'}</span></div>
                            <div class="metric"><span>上界</span><span>${data.upper != null ? Number(data.upper).toFixed(4) : 'N/A'}</span></div>
                            <div class="metric"><span>模型</span><span>${data.model || 'timemoe'}</span></div>
                        `;
                        document.getElementById('sota-timemoe').innerHTML = html;
                    }
                }).catch(e => { if (e.name !== 'AbortError') console.error('SOTA timemoe error:', e); }),

            // Multi-Agent Pipeline
            fetch('/api/sota/multiagent/pipeline', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({stock_code: stockCode}),
                signal: fastController.signal
            })
                .then(r => r.json())
                .then(data => {
                    if (data.action) {
                        const html = `
                            <div class="metric"><span>最终决策</span><span>${data.action || 'N/A'}</span></div>
                            <div class="metric"><span>置信度</span><span>${(data.confidence * 100 || 0).toFixed(1)}%</span></div>
                            <div class="metric"><span>Agent 数</span><span>${data.n_agents || 'N/A'}</span></div>
                            <div class="metric"><span>共识度</span><span>${data.consensus ? (data.consensus * 100).toFixed(0) + '%' : 'N/A'}</span></div>
                        `;
                        document.getElementById('sota-multiagent').innerHTML = html;
                    }
                }).catch(e => { if (e.name !== 'AbortError') console.error('SOTA multiagent error:', e); }),

            // TimesFM 预测
            fetch(`/api/sota/timesfm/predict/${stockCode}`, { signal: fastController.signal })
                .then(r => r.json())
                .then(data => {
                    if (data.success && data.data && data.data.prediction) {
                        const pred = data.data.prediction;
                        let html = '<div class="metric"><span>方向</span><span>' + (pred.direction || 'N/A') + '</span></div>';
                        html += '<div class="metric"><span>置信度</span><span>' + (pred.confidence * 100 || 0).toFixed(1) + '%</span></div>';
                        html += '<div class="metric"><span>模型</span><span>' + (pred.model || 'timesfm_numpy') + '</span></div>';
                        html += '<div class="metric"><span>不确定性</span><span>[' + (pred.uncertainty?.lower || 0).toFixed(4) + ', ' + (pred.uncertainty?.upper || 0).toFixed(4) + ']</span></div>';
                        document.getElementById('sota-timesfm').innerHTML = html;
                    }
                }).catch(e => { if (e.name !== 'AbortError') console.error('SOTA timesfm error:', e); }),

        ]).finally(() => {
            clearTimeout(fastTimeout);
            updateProgress('快速面板完成', 60);
            // 快速阶段完成后隐藏加载遮罩
            if (loadingOverlay) loadingOverlay.style.display = 'none';
            // 2026-08-15: 引擎状态面板追加 5 个 per-model 实时状态
            if (typeof loadSOTAModelStatusPanel === 'function') loadSOTAModelStatusPanel();
        });

        // ========== 第二阶段：慢速面板 (135 秒超时，延迟 2 秒后启动) ==========
        setTimeout(() => {
            const slowController = new AbortController();
            const slowTimeout = setTimeout(() => slowController.abort(), 240000);

            Promise.allSettled([
                // SOTA 综合决策 (2026-09-09 异步链: 后端后台线程算, GET 秒回;
                // status=running → 每 5s 轮询 ×45 (225s 窗口, 240s abort 内) —
                // 8080 实测每调用 ~17s (5 模型驻留, TTFT 15s), 完整 LLM 链 ~159s)
                new Promise((resolve) => {
                    const poll = (n) => {
                        if (n > 45 || slowController.signal.aborted) {
                            resolve({ success: false, error: 'decision computing' });
                            return;
                        }
                        fetch(`/api/sota/decision/${stockCode}`, { signal: slowController.signal })
                            .then(r => r.json())
                            .then(d => {
                                if (d.status === 'running' && !slowController.signal.aborted) {
                                    setTimeout(() => poll(n + 1), 5000);
                                } else {
                                    resolve(d);
                                }
                            })
                            .catch(() => resolve({ success: false, error: 'poll error' }));
                    };
                    poll(0);
                })
                    .then(data => {
                        try {
                            updateProgress('SOTA 决策', 80);
                            if (data.success && data.decision) {
                                const d = data.decision;
                                const dir = d.ensemble_direction || 'neutral';
                                const score = Number(d.ensemble_score) || 0;
                                const rlConf = Number(d.rl_confidence) || 0;
                                const execTime = Number(d.execution_time_ms) || 0;
                                const llmDir = d.llm_decision?.reasoning
                                    || (d.llm_decision?.research_direction
                                        ? `${d.llm_decision.research_direction.toUpperCase()} · 分析置信 ${((d.llm_decision.analyst_confidence||0)*100).toFixed(0)}% · 风险 ${d.llm_decision.risk_level || '-'}`
                                        : (d.llm?.research_direction || 'N/A'));
                                document.getElementById('sota-decision').innerHTML = `
                                    <div class="sota-decision-card">
                                        <div class="sota-direction ${dir}">
                                            <i class="fas fa-arrow-${dir === 'bullish' ? 'up' : dir === 'bearish' ? 'down' : 'right'}"></i>
                                            <h4>${dir === 'bullish' ? '看涨' : dir === 'bearish' ? '看跌' : '中性'}</h4>
                                        </div>
                                        <div class="sota-metrics">
                                            <div class="metric"><span>综合得分</span><span>${score.toFixed(3)}</span></div>
                                            <div class="metric"><span>LLM 决策</span><span>${llmDir}</span></div>
                                            <div class="metric"><span>RL 动作</span><span>${d.rl_action || 'N/A'}</span></div>
                                            <div class="metric"><span>RL 置信度</span><span>${(rlConf * 100).toFixed(1)}%</span></div>
                                            <div class="metric"><span>执行时间</span><span>${execTime.toFixed(1)}ms</span></div>
                                        </div>
                                    </div>
                                `;
                            }
                        } catch (e) {
                            console.error('SOTA decision render error:', e);
                        }
                        return data;
                    })
                    .catch(e => {
                        if (e.name === 'AbortError') {
                            console.error('SOTA decision timeout');
                        } else {
                            console.error('SOTA decision error:', e);
                        }
                        return { success: false, error: e.message };
                    }),

                // 因子挖掘 (LLM 调用，约 30-60 秒)
                fetch(`/api/sota/factors?stock_code=${stockCode}`, { signal: slowController.signal })
                    .then(r => r.json())
                    .then(data => {
                        try {
                            updateProgress('因子挖掘', 90);
                            if (data.success && data.factors) {
                                const factors = data.factors.map(f =>
                                    `<div class="factor-item"><span>${f.name}</span><span>${Number(f.efficacy).toFixed(4)}</span></div>`
                                ).join('');
                                document.getElementById('sota-factors').innerHTML = `
                                    <div class="factors-list">${factors}</div>
                                    <p class="factor-count">共 ${data.factor_count || data.factors.length} 个因子</p>
                                `;
                            }
                        } catch (e) {
                            console.error('SOTA factors render error:', e);
                        }
                        return data;
                    })
                    .catch(e => {
                        if (e.name !== 'AbortError') {
                            console.error('SOTA factors error:', e);
                        }
                        return { success: false, error: e.message };
                    }),

                // 多模态分析 (LLM 调用，约 10-20 秒)
                fetch(`/api/sota/cross-modal?stock_code=${stockCode}`, { signal: slowController.signal })
                    .then(r => r.json())
                    .then(data => {
                        try {
                            updateProgress('多模态分析', 95);
                            if (data.success) {
                                const cm = data.analysis || {};
                                document.getElementById('sota-cross-modal').innerHTML = `
                                    <div class="cross-modal-card">
                                        <div class="cm-layer"><span>方向</span><span>${cm.direction || 'N/A'}</span></div>
                                        <div class="cm-layer"><span>置信度</span><span>${cm.confidence ? (Number(cm.confidence) * 100).toFixed(0) + '%' : 'N/A'}</span></div>
                                        <div class="cm-layer"><span>一致性</span><span>${cm.score ? (Number(cm.score) * 100).toFixed(0) + '%' : 'N/A'}</span></div>
                                        <div class="cm-layer"><span>来源</span><span>${cm.source || 'N/A'}</span></div>
                                        ${cm.warning ? `<div class="cm-warning">${cm.warning}</div>` : ''}
                                    </div>
                                `;
                            }
                        } catch (e) {
                            console.error('SOTA cross-modal render error:', e);
                        }
                        return data;
                    })
                    .catch(e => {
                        if (e.name !== 'AbortError') {
                            console.error('SOTA cross-modal error:', e);
                        }
                        return { success: false, error: e.message };
                    }),

                // Time-LLM 预测 (约 30 秒，含 cross_modal)
                fetch('/api/time-llm/predict/sz300620', { signal: slowController.signal })
                    .then(r => r.json())
                    .then(data => {
                        try {
                            console.log('[Time-LLM] Response:', JSON.stringify(data).substring(0, 200));
                            if (data.success && data.data && data.data.direction !== undefined) {
                                const p = data.data;
                                const conf = (Number(p.confidence) * 100 || 0).toFixed(1);
                                const execTime = p.execution_time_ms != null ? Math.round(p.execution_time_ms) : null;
                                const html = `
                                    <div class="metric"><span>方向</span><span>${p.direction || 'N/A'}</span></div>
                                    <div class="metric"><span>置信度</span><span>${conf}%</span></div>
                                    <div class="metric"><span>Regime</span><span>${p.market_regime || 'N/A'}</span></div>
                                    <div class="metric"><span>执行时间</span><span>${execTime ? execTime + 'ms' : 'N/A'}</span></div>
                                `;
                                document.getElementById('sota-time-llm').innerHTML = html;
                                console.log('[Time-LLM] Panel updated');
                            } else {
                                console.error('[Time-LLM] Condition failed: success=' + data.success + ' hasData=' + !!data.data + ' dir=' + data.data?.direction);
                            }
                        } catch (e) {
                            console.error('[Time-LLM] Render error:', e);
                        }
                        return data;
                    }).catch(e => {
                        if (e.name === 'AbortError') {
                            console.log('[Time-LLM] Aborted (timeout)');
                        } else {
                            console.error('[Time-LLM] Error:', e);
                        }
                    }),

            ]).finally(() => {
                clearTimeout(slowTimeout);
                updateProgress('全部完成', 100);
                // 隐藏加载遮罩
                if (loadingOverlay) {
                    setTimeout(() => { loadingOverlay.style.display = 'none'; }, 500);
                }
            });
        }, 2000);
    }

    // 重试函数
    function retrySOTADashboard() {
        const timeoutMsg = document.getElementById('sota-timeout-msg');
        if (timeoutMsg) timeoutMsg.style.display = 'none';
        loadSOTADashboard();
    }
