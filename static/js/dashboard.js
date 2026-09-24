// Stock Analyzer - Dashboard JavaScript (量化模型面板)
// 为"量化模型"标签页提供所有前端交互逻辑
// API_BASE 和 _activeApiRequests 已由 core.js 在全局作用域定义

// ═══════════════════════════════════════════════════
// 工具函数
// ═══════════════════════════════════════════════════

// _activeApiRequests 和 _globalLoadingTimer 已由 core.js 定义，此处复用

function _incApiRequests() {
    _activeApiRequests++;
    clearTimeout(_globalLoadingTimer);
    _globalLoadingTimer = setTimeout(() => {
        const overlay = document.getElementById('global-loading-overlay');
        if (overlay) overlay.classList.add('active');
    }, 300);  // 300ms 延迟才显示全局 loading，避免闪烁
}

function _decApiRequests() {
    _activeApiRequests = Math.max(0, _activeApiRequests - 1);
    if (_activeApiRequests <= 0) {
        clearTimeout(_globalLoadingTimer);
        const overlay = document.getElementById('global-loading-overlay');
        if (overlay) overlay.classList.remove('active');
    }
}

/**
 * Dashboard API 请求包装器 (Phase 2: 集成 API Key + 超时保护)
 * 替代原生 fetch，自动附加 X-API-Key 头
 */
async function apiGet(url, timeoutMs = 10000) {
    _incApiRequests();
    try {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), timeoutMs);
        const headers = {};
        const key = typeof getApiKey === 'function' ? getApiKey() : '';
        if (key) headers['X-API-Key'] = key;
        const resp = await fetch(url, {
            signal: controller.signal,
            headers,
        });
        clearTimeout(timeout);
        if (!resp.ok) {
            const text = await resp.text();
            throw new Error(`HTTP ${resp.status}: ${text.slice(0, 100)}`);
        }
        return await resp.json();
    } catch (e) {
        console.error(`[Dashboard API] ${url} 请求失败:`, e.message);
        return null;
    } finally {
        _decApiRequests();
    }
}

// 添加全局 loading overlay 到 body
(function initGlobalLoading() {
    const overlay = document.createElement('div');
    overlay.id = 'global-loading-overlay';
    overlay.className = 'loading-overlay';
    overlay.innerHTML = `
        <div class="spinner"></div>
        <div class="loading-text">正在加载数据...</div>
    `;
    document.body.appendChild(overlay);
})();

function showLoading(containerId, text = '加载中...') {
    const el = document.getElementById(containerId);
    if (el) {
        el.innerHTML = `
            <div class="panel-loading">
                <div class="spinner-icon"></div>
                <div class="loading-text">${text}</div>
                <div class="loading-progress">
                    <div class="progress-bar"></div>
                </div>
            </div>`;
    }
}

function showError(containerId, message, retryFn = null) {
    const el = document.getElementById(containerId);
    if (!el) return;
    let retryHtml = '';
    if (retryFn) {
        retryHtml = `<button class="retry-btn" onclick="(${retryFn.toString()})()">重试</button>`;
    }
    el.innerHTML = `
        <div class="error">
            <i class="fas fa-exclamation-triangle"></i>
            <div>${message}</div>
            ${retryHtml}
        </div>`;
}

function showEmpty(containerId, message = '暂无数据') {
    const el = document.getElementById(containerId);
    if (el) {
        el.innerHTML = `<div class="loading"><div class="spinner-icon"></div>${message}</div>`;
    }
}

// ═══════════════════════════════════════════════════
// 因子仪表盘
// ═══════════════════════════════════════════════════

async function loadFactorDashboard(code) {
    showLoading('factor-dashboard');
    const data = await apiGet(`${API_BASE}/api/dashboard/factors?code=${code}`);
    const container = document.getElementById('factor-dashboard');

    if (!data || !container) return;

    // 原始因子评分
    const rawScores = data.factor_scores || {};
    const enhanced = data.enhanced_features || {};
    const allScores = { ...rawScores, ...enhanced };

    let html = `
        <div class="dashboard-card">
            <h3><i class="fas fa-chart-bar"></i> 因子评分</h3>
            <div class="factor-table">
                <table>
                    <thead><tr><th>因子</th><th>得分</th><th>评级</th></tr></thead>
                    <tbody>
    `;

    const sorted = Object.entries(allScores).sort((a, b) => b[1] - a[1]);
    for (const [name, score] of sorted.slice(0, 20)) {
        const pct = Math.max(0, Math.min(100, score * 10));
        const color = score >= 7 ? '#10b981' : score >= 5 ? '#3b82f6' : '#f59e0b';
        html += `<tr>
            <td>${name}</td>
            <td>
                <div class="factor-bar" style="background: #1e293b; border-radius: 4px; height: 20px; overflow: hidden;">
                    <div style="width: ${pct}%; height: 100%; background: ${color}; transition: width 0.5s;"></div>
                </div>
            </td>
            <td>${score >= 7 ? '强' : score >= 5 ? '中' : '弱'}</td>
        </tr>`;
    }

    html += `</tbody></table></div>
            <div class="metric-cards">
                <div class="metric-card">
                    <div class="metric-label">综合得分</div>
                    <div class="metric-value" style="color: ${data.weighted_score >= 6 ? '#10b981' : '#3b82f6'}">${data.weighted_score}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">评级</div>
                    <div class="metric-value">${data.rating || 'N/A'}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">因子数量</div>
                    <div class="metric-value">${sorted.length}</div>
                </div>
            </div>
        </div>
    `;

    container.innerHTML = html;
}

// ═══════════════════════════════════════════════════
// ML 预测面板
// ═══════════════════════════════════════════════════

async function loadMLPrediction(code) {
    showLoading('ml-prediction-panel');
    const data = await apiGet(`${API_BASE}/api/dashboard/ml-prediction?code=${code}`);
    const container = document.getElementById('ml-prediction-panel');

    if (!data || !container) return;

    const pred = data.prediction || {};
    const probs = pred.probabilities || {};
    const direction = pred.direction || 'neutral';
    const confidence = pred.confidence || 0.5;

    // 方向颜色
    const dirColor = direction === 'up' ? '#ef4444' : direction === 'down' ? '#10b981' : '#6b7280';
    const dirIcon = direction === 'up' ? 'fa-arrow-trend-up' : direction === 'down' ? 'fa-arrow-trend-down' : 'fa-minus';
    const dirText = direction === 'up' ? '看涨' : direction === 'down' ? '看跌' : '中性';

    // 市场状态
    const regimeColor = {
        'bull': '#ef4444', 'bear': '#10b981',
        'sideways': '#6b7280', 'volatile': '#f59e0b',
    }[data.regime] || '#6b7280';

    // 模型版本信息
    const modelVersion = data.model_version || 'v0.1.0';
    const lastRetrain = data.last_retrain || '未知';
    const isTrained = data.is_trained !== undefined ? data.is_trained : true;
    const statusClass = isTrained ? 'trained' : 'untrained';
    const statusText = isTrained ? '已训练' : '未训练';
    const statusIcon = isTrained ? 'fa-check-circle' : 'fa-circle-xmark';

    let html = `
        <div class="dashboard-card">
            <h3><i class="fas fa-brain"></i> ML 预测</h3>

            <div class="model-info-row">
                <span class="info-label">模型版本</span>
                <span class="info-value">${modelVersion}</span>
            </div>
            <div class="model-info-row">
                <span class="info-label">上次训练</span>
                <span class="info-value">${lastRetrain}</span>
            </div>
            <div class="model-info-row">
                <span class="info-label">训练状态</span>
                <span class="model-status-badge ${statusClass}"><i class="fas ${statusIcon}"></i> ${statusText}</span>
            </div>

            <div class="prediction-summary">
                <div class="direction-badge" style="background: ${dirColor}20; color: ${dirColor}; border: 2px solid ${dirColor};">
                    <i class="fas ${dirIcon}"></i> ${dirText}
                </div>
                <div class="confidence-meter">
                    <div class="confidence-label">置信度</div>
                    <div class="confidence-bar-bg">
                        <div class="confidence-bar-fill" style="width: ${confidence * 100}%; background: ${dirColor};"></div>
                    </div>
                    <div class="confidence-value">${(confidence * 100).toFixed(1)}%</div>
                </div>
            </div>

            <div class="metric-cards">
                <div class="metric-card">
                    <div class="metric-label">市场状态</div>
                    <div class="metric-value" style="color: ${regimeColor}">${data.regime || 'N/A'}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">CV 准确率</div>
                    <div class="metric-value">${(data.cv_score * 100 || 0).toFixed(1)}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">模型数量</div>
                    <div class="metric-value">${(data.models || []).length}</div>
                </div>
            </div>

            <div class="probabilities-chart">
                <h4>概率分布</h4>
                <div class="prob-bar">
                    <span>上涨</span>
                    <div class="prob-fill up" style="width: ${(probs.up || 0) * 100}%;"></div>
                    <span>${((probs.up || 0) * 100).toFixed(1)}%</span>
                </div>
                <div class="prob-bar">
                    <span>震荡</span>
                    <div class="prob-fill neutral" style="width: ${(probs.neutral || 0) * 100}%;"></div>
                    <span>${((probs.neutral || 0) * 100).toFixed(1)}%</span>
                </div>
                <div class="prob-bar">
                    <span>下跌</span>
                    <div class="prob-fill down" style="width: ${(probs.down || 0) * 100}%;"></div>
                    <span>${((probs.down || 0) * 100).toFixed(1)}%</span>
                </div>
            </div>

            ${data.feature_importances ? `<div class="feature-importance">
                <h4>特征重要性</h4>
                <div class="importance-bars">
    ` + (function() {
        const fi = data.feature_importances || {};
        const sortedFI = Object.entries(fi).sort((a, b) => b[1] - a[1]).slice(0, 8);
        const maxFI = sortedFI.length > 0 ? sortedFI[0][1] : 1;
        let s = '';
        for (const [name, imp] of sortedFI) {
            const pct = (imp / maxFI) * 100;
            s += `<div class="importance-bar">
                <span>${name}</span>
                <div class="imp-bar-bg"><div class="imp-bar-fill" style="width: ${pct}%"></div></div>
                <span>${imp.toFixed(4)}</span>
            </div>`;
        }
        return s + '</div></div>';
    })() : ''}
        </div>`;
    container.innerHTML = html;
}

// ═══════════════════════════════════════════════════
// 因子 IC 面板
// ═══════════════════════════════════════════════════

async function loadFactorICPanel() {
    showLoading('factor-ic-panel');
    const data = await apiGet(`${API_BASE}/api/dashboard/factor-ic`, 30000);
    const container = document.getElementById('factor-ic-panel');

    if (!data || !container) return;

    const ranking = data.ranking || [];
    const decay = data.ic_decay || [];
    const turnover = data.turnover || {};

    // IC 衰减图表
    const icDecayHTML = (function() {
        if (!decay || decay.length === 0) return '';
        const maxVal = Math.max(...decay.map(d => Math.abs(d.ic || 0)), 0.01);
        const bars = decay.slice(0, 20).map(d => {
            const h = (Math.abs(d.ic) / maxVal) * 70;
            const cls = d.ic >= 0 ? 'positive' : 'negative';
            return `<div class="ic-decay-bar ${cls}" style="height:${h}px" title="${d.lag || '?'}期: ${d.ic?.toFixed(4) || '?'}"></div>`;
        }).join('');
        return `<div style="margin-top:14px;">
            <h4>IC 衰减 (滞后期间)</h4>
            <div class="ic-decay-chart">${bars}</div>
            <div class="ic-decay-labels"><span>1期</span><span>滞后 →</span></div>
        </div>`;
    })();

    // 换手率指标
    const turnoverHTML = (function() {
        if (!turnover || Object.keys(turnover).length === 0) return '';
        const avgTurnover = turnover.avg_turnover || 0;
        const turnoverColor = avgTurnover < 0.3 ? '#10b981' : avgTurnover < 0.6 ? '#3b82f6' : '#f59e0b';
        const turnoverDesc = avgTurnover < 0.3 ? '低 (稳定)' : avgTurnover < 0.6 ? '中 (正常)' : '高 (需关注)';
        return `<div class="risk-metric-row" style="margin-top:14px;">
            <span class="rm-label">平均换手率</span>
            <span class="rm-value" style="color:${turnoverColor}">${(avgTurnover * 100).toFixed(1)}% (${turnoverDesc})</span>
        </div>`;
    })();

    let html = `
        <div class="dashboard-card">
            <h3><i class="fas fa-chart-line"></i> 因子 IC/ICIR 分析</h3>
            <div class="ic-ranking-table">
                <table>
                    <thead><tr><th>排名</th><th>因子</th><th>IC Mean</th><th>ICIR</th><th>t-Stat</th><th>状态</th></tr></thead>
                    <tbody>
    `;

    for (let i = 0; i < Math.min(ranking.length, 15); i++) {
        const r = ranking[i];
        const icirColor = Math.abs(r.icir) > 1 ? '#10b981' : Math.abs(r.icir) > 0.5 ? '#3b82f6' : '#f59e0b';
        const status = Math.abs(r.icir) > 1 ? '优秀' : Math.abs(r.icir) > 0.5 ? '良好' : '一般';
        html += `<tr>
            <td>${i + 1}</td>
            <td>${r.name}</td>
            <td>${r.ic_mean}</td>
            <td style="color: ${icirColor}; font-weight: 700;">${r.icir}</td>
            <td>${r.t_stat || 'N/A'}</td>
            <td>${status}</td>
        </tr>`;
    }

    html += `</tbody></table></div>${icDecayHTML}${turnoverHTML}</div>`;
    container.innerHTML = html;
}

// ═══════════════════════════════════════════════════
// 因子质量面板
// ═══════════════════════════════════════════════════

async function loadFactorQualityPanel() {
    showLoading('factor-quality-panel');
    const data = await apiGet(`${API_BASE}/api/dashboard/factor-ic`, 30000);
    const container = document.getElementById('factor-quality-panel');

    if (!data || !container) return;

    const factors = data.factors || {};
    const ranking = data.ranking || [];

    // 计算统计指标
    const icMeanValues = Object.values(factors).map(f => f.ic_mean || 0);
    const icirValues = Object.values(factors).map(f => f.icir || 0);
    const tStatValues = Object.values(factors).map(f => f.t_stat || 0);

    const avgIC = icMeanValues.length > 0 ? (icMeanValues.reduce((a, b) => a + b, 0) / icMeanValues.length) : 0;
    const avgICIR = icirValues.length > 0 ? (icirValues.reduce((a, b) => a + b, 0) / icirValues.length) : 0;
    const avgTStat = tStatValues.length > 0 ? (tStatValues.reduce((a, b) => a + b, 0) / tStatValues.length) : 0;

    // 显著因子数量 (|ICIR| > 1)
    const significantCount = icirValues.filter(v => Math.abs(v) > 1).length;
    const totalFactors = Math.max(icirValues.length, 1);

    // 因子质量卡片
    const qualityCards = `
        <div class="factor-quality-grid">
            <div class="fq-card">
                <div class="fq-label">因子数量</div>
                <div class="fq-value">${totalFactors}</div>
                <div class="fq-desc">已计算因子</div>
            </div>
            <div class="fq-card">
                <div class="fq-label">平均 IC</div>
                <div class="fq-value" style="color:${avgIC > 0.03 ? '#10b981' : avgIC > 0 ? '#3b82f6' : '#ef4444'}">${avgIC.toFixed(4)}</div>
                <div class="fq-desc">预测能力</div>
            </div>
            <div class="fq-card">
                <div class="fq-label">平均 ICIR</div>
                <div class="fq-value" style="color:${avgICIR > 1.5 ? '#10b981' : avgICIR > 0.5 ? '#3b82f6' : '#f59e0b'}">${avgICIR.toFixed(3)}</div>
                <div class="fq-desc">稳定性指标</div>
            </div>
            <div class="fq-card">
                <div class="fq-label">显著因子</div>
                <div class="fq-value" style="color:${significantCount >= totalFactors * 0.7 ? '#10b981' : '#f59e0b'}">${significantCount}/${totalFactors}</div>
                <div class="fq-desc">ICIR > 1.0</div>
            </div>
        </div>
    `;

    // 因子质量排名表
    let tableRows = '';
    for (let i = 0; i < Math.min(ranking.length, 10); i++) {
        const r = ranking[i];
        const icirColor = Math.abs(r.icir) > 1.5 ? '#10b981' : Math.abs(r.icir) > 1 ? '#3b82f6' : '#f59e0b';
        const quality = Math.abs(r.icir) > 1.5 ? '优秀' : Math.abs(r.icir) > 1 ? '良好' : '一般';
        tableRows += `<tr>
            <td>${i + 1}</td>
            <td>${r.name}</td>
            <td>${r.ic_mean?.toFixed(4) || '0.0000'}</td>
            <td style="color:${icirColor};font-weight:700">${r.icir?.toFixed(3) || '0.000'}</td>
            <td>${quality}</td>
        </tr>`;
    }

    const tableHTML = `
        <div style="margin-top:16px;">
            <h4 style="font-size:14px;color:#cbd5e1;margin:0 0 8px;font-weight:700;">因子质量排名 (前10)</h4>
            <table style="width:100%;border-collapse:collapse;font-size:13px;">
                <thead>
                    <tr style="border-bottom:1px solid rgba(59,130,246,0.2);color:#94a3b8;">
                        <th style="padding:8px;text-align:left;width:40px">#</th>
                        <th style="padding:8px;text-align:left">因子</th>
                        <th style="padding:8px;text-align:right">IC Mean</th>
                        <th style="padding:8px;text-align:right">ICIR</th>
                        <th style="padding:8px;text-align:right">质量</th>
                    </tr>
                </thead>
                <tbody>${tableRows}</tbody>
            </table>
        </div>
    `;

    container.innerHTML = qualityCards + tableHTML;
}

// ═══════════════════════════════════════════════════
// 风险报告面板
// ═══════════════════════════════════════════════════

async function loadRiskReport() {
    showLoading('risk-report-panel');
    const data = await apiGet(`${API_BASE}/api/dashboard/risk-report`);
    const container = document.getElementById('risk-report-panel');

    if (!data || !container) return;

    const report = data.report || {};

    let html = `
        <div class="dashboard-card">
            <h3><i class="fas fa-shield-halved"></i> 风险报告</h3>
            <div class="metric-cards">
                <div class="metric-card">
                    <div class="metric-label">年化收益</div>
                    <div class="metric-value" style="color: ${(report.annual_return || 0) > 0 ? '#10b981' : '#ef4444'}">${((report.annual_return || 0) * 100).toFixed(1)}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">年化波动率</div>
                    <div class="metric-value">${((report.volatility || 0) * Math.sqrt(252) * 100).toFixed(1)}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">夏普比率</div>
                    <div class="metric-value" style="color: ${(report.sharpe_ratio || 0) > 1 ? '#10b981' : '#3b82f6'}">${report.sharpe_ratio?.toFixed(2) || 'N/A'}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">最大回撤</div>
                    <div class="metric-value" style="color: #ef4444">${((report.max_drawdown || 0) * 100).toFixed(1)}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Sortino</div>
                    <div class="metric-value">${report.sortino_ratio?.toFixed(2) || 'N/A'}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Calmar</div>
                    <div class="metric-value">${report.calmar_ratio?.toFixed(2) || 'N/A'}</div>
                </div>
            </div>
            <div class="risk-decomposition">
                <h4>权重分配</h4>
                <div class="weight-bars">
    `;

    const weights = data.weights || [];
    for (let i = 0; i < weights.length; i++) {
        const pct = weights[i] * 100;
        html += `<div class="weight-bar">
            <span>资产 ${i + 1}</span>
            <div class="w-bar-bg"><div class="w-bar-fill" style="width: ${pct}%"></div></div>
            <span>${pct.toFixed(1)}%</span>
        </div>`;
    }

    html += `</div></div></div>`;
    container.innerHTML = html;
}

// ═══════════════════════════════════════════════════
// 回测面板
// ═══════════════════════════════════════════════════

async function loadBacktestPanel(code) {
    showLoading('backtest-panel');
    const data = await apiGet(`${API_BASE}/api/dashboard/backtest-result?code=${code}`);
    const container = document.getElementById('backtest-panel');

    if (!data || !container) return;

    const metrics = data.metrics || {};
    const constraints = data.constraints || {};
    const risk = data.risk_metrics || {};

    // 约束标签
    const constraintTags = (function() {
        const tags = [];
        if (constraints.t_plus_1 !== undefined) {
            const cls = constraints.t_plus_1 ? 'enforced' : 'warning';
            const icon = constraints.t_plus_1 ? 'fa-check' : 'fa-exclamation-triangle';
            tags.push(`<span class="constraint-tag ${cls}"><i class="fas ${icon}"></i> T+1 ${constraints.t_plus_1 ? '已执行' : '未执行'}</span>`);
        }
        if (constraints.lot_size !== undefined) {
            const cls = constraints.lot_size ? 'enforced' : 'warning';
            const icon = constraints.lot_size ? 'fa-check' : 'fa-exclamation-triangle';
            tags.push(`<span class="constraint-tag ${cls}"><i class="fas ${icon}"></i> 整手交易 ${constraints.lot_size ? '已执行' : '未执行'}</span>`);
        }
        if (constraints.limit_up_down !== undefined) {
            const cls = constraints.limit_up_down ? 'enforced' : 'warning';
            const icon = constraints.limit_up_down ? 'fa-check' : 'fa-exclamation-triangle';
            tags.push(`<span class="constraint-tag ${cls}"><i class="fas ${icon}"></i> 涨跌停 ${constraints.limit_up_down ? '已处理' : '未处理'}</span>`);
        }
        if (constraints.transaction_cost !== undefined) {
            const cls = constraints.transaction_cost ? 'enforced' : 'warning';
            const icon = constraints.transaction_cost ? 'fa-check' : 'fa-exclamation-triangle';
            tags.push(`<span class="constraint-tag ${cls}"><i class="fas ${icon}"></i> 交易成本 ${constraints.transaction_cost ? '已包含' : '未包含'}</span>`);
        }
        if (constraints.stopping !== undefined) {
            const cls = constraints.stopping ? 'enforced' : 'warning';
            const icon = constraints.stopping ? 'fa-check' : 'fa-exclamation-triangle';
            tags.push(`<span class="constraint-tag ${cls}"><i class="fas ${icon}"></i> 止损机制 ${constraints.stopping ? '已启用' : '未启用'}</span>`);
        }
        return tags.join('');
    })();

    // 风险指标增强行
    const riskRows = (function() {
        const rows = [];
        if (risk.var_95 !== undefined) {
            rows.push(`<div class="risk-metric-row"><span class="rm-label">VaR (95%)</span><span class="rm-value negative">${(risk.var_95 * 100).toFixed(2)}%</span></div>`);
        }
        if (risk.cvar_95 !== undefined) {
            rows.push(`<div class="risk-metric-row"><span class="rm-label">CVaR (95%)</span><span class="rm-value negative">${(risk.cvar_95 * 100).toFixed(2)}%</span></div>`);
        }
        if (risk.sortino_ratio !== undefined) {
            const color = risk.sortino_ratio > 1 ? 'positive' : risk.sortino_ratio > 0 ? 'warning' : 'negative';
            rows.push(`<div class="risk-metric-row"><span class="rm-label">Sortino</span><span class="rm-value ${color}">${risk.sortino_ratio.toFixed(2)}</span></div>`);
        }
        if (risk.calmar_ratio !== undefined) {
            const color = risk.calmar_ratio > 1 ? 'positive' : risk.calmar_ratio > 0 ? 'warning' : 'negative';
            rows.push(`<div class="risk-metric-row"><span class="rm-label">Calmar</span><span class="rm-value ${color}">${risk.calmar_ratio.toFixed(2)}</span></div>`);
        }
        if (risk.profit_factor !== undefined) {
            const color = risk.profit_factor > 1.5 ? 'positive' : risk.profit_factor > 1 ? 'warning' : 'negative';
            rows.push(`<div class="risk-metric-row"><span class="rm-label">盈亏比</span><span class="rm-value ${color}">${risk.profit_factor.toFixed(2)}</span></div>`);
        }
        if (risk.max_consecutive_losses !== undefined) {
            rows.push(`<div class="risk-metric-row"><span class="rm-label">最大连亏</span><span class="rm-value warning">${risk.max_consecutive_losses} 笔</span></div>`);
        }
        return rows.join('');
    })();

    let html = `
        <div class="dashboard-card">
            <h3><i class="fas fa-chart-area"></i> 回测结果</h3>

            <div class="constraint-tags-row" style="margin-bottom: 14px;">
                ${constraintTags || '<span style="color:#64748b;font-size:12px;">无约束数据</span>'}
            </div>

            <div class="metric-cards">
                <div class="metric-card">
                    <div class="metric-label">总收益</div>
                    <div class="metric-value" style="color: ${(metrics.total_return || 0) > 0 ? '#10b981' : '#ef4444'}">${((metrics.total_return || 0) * 100).toFixed(1)}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">年化收益</div>
                    <div class="metric-value" style="color: ${(metrics.annual_return || 0) > 0 ? '#10b981' : '#ef4444'}">${((metrics.annual_return || 0) * 100).toFixed(1)}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">夏普比率</div>
                    <div class="metric-value">${metrics.sharpe_ratio?.toFixed(2) || 'N/A'}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">最大回撤</div>
                    <div class="metric-value" style="color: #ef4444">${((metrics.max_drawdown || 0) * 100).toFixed(1)}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">胜率</div>
                    <div class="metric-value">${((metrics.win_rate || 0) * 100).toFixed(1)}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">交易次数</div>
                    <div class="metric-value">${metrics.n_trades || 0}</div>
                </div>
            </div>

            ${riskRows ? `<div style="margin-top: 14px;">
                <h4 style="font-size:14px;color:#cbd5e1;margin:0 0 8px;font-weight:700;">增强风险指标</h4>
                ${riskRows}
            </div>` : ''}
        </div>
    `;

    container.innerHTML = html;
}

// ═══════════════════════════════════════════════════
// ═══════════════════════════════════════════════════
// 系统运维 - 模型健康面板 (health-panel)
// ═══════════════════════════════════════════════════

async function loadHealthPanel() {
    const container = document.getElementById('health-panel');
    if (!container) return;

    try {
        const resp = await apiGet(`${API_BASE}/api/health`);
        if (!resp) {
            container.innerHTML = '<div style="color:#94a3b8;font-size:13px;">健康检查不可用</div>';
            return;
        }

        const d = resp.data || resp;
        let html = '<div style="font-size:13px;">';
        const statusColor = d.status === 'healthy' ? '#10b981' : '#f59e0b';
        html += `<div style="text-align:center;padding:8px;background:rgba(15,23,42,0.5);border-radius:6px;margin-bottom:8px;">`;
        html += `<div style="font-size:12px;color:#94a3b8;">服务器状态</div>`;
        html += `<div style="font-size:18px;font-weight:700;color:${statusColor};">${d.status || 'unknown'}</div>`;
        html += `</div>`;
        html += `<div style="padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.05);">Python: ${d.python || '--'}</div>`;
        html += `<div style="padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.05);">平台: ${d.server?.platform || d.platform || '--'}</div>`;
        html += `<div style="padding:4px 0;">GPU: ${d.gpu?.mps_available ? 'MPS ✅' : d.gpu?.cuda_available ? 'CUDA ✅' : '无'}</div>`;
        html += '</div>';
        container.innerHTML = html;

    } catch (e) {
        console.error('loadHealthPanel error:', e);
        container.innerHTML = `<div style="color:#ef4444;font-size:13px;">加载失败: ${e.message}</div>`;
    }
}

// 模型健康面板
// ═══════════════════════════════════════════════════

async function loadModelHealthPanel() {
    showLoading('model-health-panel', '正在加载模型健康状态...');
    try {
        const data = await apiGet(`${API_BASE}/api/dashboard/model-health`);
        const container = document.getElementById('model-health-panel');

        if (!data || !container) {
            showError('model-health-panel', '数据加载失败，请检查网络连接', 'loadModelHealthPanel');
            return;
        }

    const ml = data.ml_report || {};
    const health = data.health || {};

    const statusColor = { healthy: '#10b981', warning: '#f59e0b', critical: '#ef4444' };
    const statusIcon = { healthy: 'fa-check-circle', warning: 'fa-exclamation-triangle', critical: 'fa-times-circle' };
    const hc = statusColor[health.status] || '#6b7280';
    const hi = statusIcon[health.status] || 'fa-question-circle';

    let html = `
        <div class="dashboard-card">
            <h3><i class="fas fa-heartbeat"></i> 模型健康</h3>
            <div class="health-status" style="display: flex; align-items: center; gap: 12px; padding: 16px; background: ${hc}10; border-radius: 8px; border-left: 4px solid ${hc};">
                <i class="fas ${hi}" style="color: ${hc}; font-size: 24px;"></i>
                <div>
                    <div style="font-size: 18px; font-weight: 700; color: ${hc};">${health.status || 'unknown'}</div>
                    <div style="font-size: 12px; color: #9ca3af;">${(health.alerts || []).join('; ') || '无告警'}</div>
                </div>
            </div>
            <div class="metric-cards" style="margin-top: 16px;">
                <div class="metric-card">
                    <div class="metric-label">模型已训练</div>
                    <div class="metric-value">${ml.is_trained ? '是' : '否'}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">CV 准确率</div>
                    <div class="metric-value">${(ml.cv_score * 100 || 0).toFixed(1)}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">模型数量</div>
                    <div class="metric-value">${(ml.models || []).length}</div>
                </div>
            </div>
            ${ml.feature_importances ? `<div style="margin-top: 16px;">
                <h4>特征重要性 Top 5</h4>
                <div class="importance-bars">
    ` + (function() {
        const fi = ml.feature_importances || {};
        const sortedFI = Object.entries(fi).sort((a, b) => b[1] - a[1]).slice(0, 5);
        let s = '';
        for (const [name, imp] of sortedFI) {
            s += `<div class="importance-bar">
                <span>${name}</span>
                <div class="imp-bar-bg"><div class="imp-bar-fill" style="width: ${Math.min(100, imp * 100)}%"></div></div>
                <span>${imp.toFixed(4)}</span>
            </div>`;
        }
        return s + '</div></div>';
    })() : ''}
        </div>`;
    container.innerHTML = html;
    } catch (e) {
        const container = document.getElementById('model-health-panel');
        if (container) {
            showError('model-health-panel', '模型健康加载失败: ' + e.message, 'loadModelHealthPanel');
        }
    }
}

// ═══════════════════════════════════════════════════
// 超参优化面板
// ═══════════════════════════════════════════════════

async function loadHyperparamsPanel() {
    showLoading('hyperparams-panel');
    const data = await apiGet(`${API_BASE}/api/dashboard/hyperparams`);
    const container = document.getElementById('hyperparams-panel');

    if (!data || !container) return;

    let html = `
        <div class="dashboard-card">
            <h3><i class="fas fa-sliders"></i> 超参优化</h3>
            <div class="metric-cards">
                <div class="metric-card">
                    <div class="metric-label">LightGBM</div>
                    <div class="metric-value" style="font-size: 13px;">n_est=${(data.lgb?.n_estimators || 200)}, depth=${(data.lgb?.max_depth || 6)}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">XGBoost</div>
                    <div class="metric-value" style="font-size: 13px;">n_est=${(data.xgb?.n_estimators || 200)}, depth=${(data.xgb?.max_depth || 5)}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">RandomForest</div>
                    <div class="metric-value" style="font-size: 13px;">n_est=${(data.rf?.n_estimators || 100)}, depth=${(data.rf?.max_depth || 5)}</div>
                </div>
            </div>
            <button class="btn btn-primary" onclick="runHyperparamOptimization()" style="margin-top: 16px;">
                <i class="fas fa-cogs"></i> 运行优化
            </button>
        </div>
    `;

    container.innerHTML = html;
}

async function runHyperparamOptimization() {
    const btn = event.target;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> 启动中...';
    btn.disabled = true;

    const resp = await fetch(`${API_BASE}/api/dashboard/hyperparams`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ n_trials: 30 }),
    });

    const data = await resp.json();

    if (data.status === 'started') {
        btn.innerHTML = '<i class="fas fa-hourglass-half"></i> 优化已启动 (约 15-35 分钟)，稍后刷新查看';
    } else if (data.status === 'running') {
        btn.innerHTML = '<i class="fas fa-hourglass-half"></i> 已有调参任务在运行';
    } else if (data.best_params) {
        btn.innerHTML = '<i class="fas fa-check"></i> 完成';
        loadHyperparamsPanel();
    } else {
        btn.innerHTML = '<i class="fas fa-exclamation-triangle"></i> 启动失败，点击重试';
    }
    btn.disabled = false;
}

// ═══════════════════════════════════════════════════
// 情感分析面板
// ═══════════════════════════════════════════════════

async function loadSentimentPanel(code) {
    showLoading('sentiment-panel');
    const data = await apiGet(`${API_BASE}/api/dashboard/sentiment?code=${code}`, 25000);
    const container = document.getElementById('sentiment-panel');

    if (!data || !container) return;

    const sent = data.sentiment || {};
    const score = sent.score || 0;
    const label = sent.label || 'neutral';
    const labelColor = score > 0.1 ? '#ef4444' : score < -0.1 ? '#10b981' : '#6b7280';
    const labelIcon = score > 0.1 ? 'fa-face-smile' : score < -0.1 ? 'fa-face-frown' : 'fa-face-meh';

    let html = `
        <div class="dashboard-card">
            <h3><i class="fas fa-comment-dots"></i> 情感分析</h3>
            <div class="sentiment-display" style="text-align: center; padding: 24px;">
                <i class="fas ${labelIcon}" style="font-size: 48px; color: ${labelColor};"></i>
                <div style="font-size: 24px; font-weight: 700; color: ${labelColor}; margin: 8px 0;">
                    ${label === 'positive' ? '偏多' : label === 'negative' ? '偏空' : '中性'}
                </div>
                <div class="confidence-bar-bg" style="max-width: 300px; margin: 0 auto;">
                    <div class="confidence-bar-fill" style="width: ${Math.abs(score) * 100}%; background: ${labelColor};"></div>
                </div>
                <div style="color: #cbd5e1; margin-top: 8px; font-size: 14px;">
                    情感分数: ${score.toFixed(3)} | 新闻数: ${sent.n_news || 0}
                </div>
            </div>
        </div>
    `;

    container.innerHTML = html;
}

// ═══════════════════════════════════════════════════
// 量化模型标签页加载器
// ═══════════════════════════════════════════════════

function switchQuantTab(group) {
    // 切换子 Tab 样式
    document.querySelectorAll('#quantSubTabs .quant-sub-tab').forEach(tab => {
        tab.classList.toggle('active', tab.dataset.group === group);
    });
    // 切换面板组可见性
    document.querySelectorAll('.quant-panel-group').forEach(g => {
        g.classList.toggle('active', g.id === `group-${group}`);
    });
}

function loadQuantModelTab(code) {
    // 自动获取主页股票代码输入框，未找到则默认 sz300620
    if (!code) {
        const codeInput = document.getElementById('stockCode');
        code = codeInput ? (codeInput.value.trim() || 'sz300620') : 'sz300620';
    }
    loadFactorDashboard(code);
    loadMLPrediction(code);
    loadFactorICPanel();
    loadFactorQualityPanel();
    loadRiskReport();
    loadBacktestPanel(code);
    loadModelHealthPanel();
    loadSentimentPanel(code);
    loadHyperparamsPanel();
    loadQualityPanel(code);
    loadDLV2Panel(code);  // 深度学习 V2
    loadTimeLLMPanel(code);  // Time-LLM 统一预测
    loadRegimeSwitchingPanel(code);  // Regime-Switching 多模型
    loadCVaRPositionPanel(code);  // CVaR 仓位管理
    loadHealthPanel(code);   // 模型健康 & 版本
    loadMLOpsPanel(code);  // MLOps 管道
    loadFusionPanel(code);  // P0-5 多模态融合预测
    loadCanaryPanel();       // P0-6 金丝雀部署
}

// ═══════════════════════════════════════════════════
// 数据质量面板
// ═══════════════════════════════════════════════════

async function loadQualityPanel(code) {
    showLoading('quality-panel');
    const data = await apiGet(`${API_BASE}/api/dashboard/data-quality?code=${code}`);
    const container = document.getElementById('quality-panel');

    if (!data || !container) return;

    const quality = data.quality || {};
    const indicators = data.indicators || [];

    // 数据源指示器
    const indicatorHTML = (function() {
        if (!indicators || indicators.length === 0) return '';
        return indicators.map(ind => {
            const dotClass = ind.type === 'real' ? 'real' : ind.type === 'fake' ? 'fake' : ind.type === 'partial' ? 'partial' : 'info';
            const label = ind.label || ind.type;
            const value = ind.value != null ? ind.value : '--';
            return `<div class="data-quality-bar">
                <div class="dq-indicator">
                    <span class="dq-dot ${dotClass}"></span>
                    <span class="dq-label">${label}</span>
                </div>
                <span class="dq-value">${value}</span>
            </div>`;
        }).join('');
    })();

    // 综合质量分
    const overallScore = quality.overall_score || 0;
    const scoreColor = overallScore >= 80 ? '#10b981' : overallScore >= 60 ? '#3b82f6' : '#f59e0b';
    const scoreDesc = overallScore >= 80 ? '优秀' : overallScore >= 60 ? '良好' : overallScore >= 40 ? '一般' : '较差';

    // 缺失率
    const missingRate = quality.missing_rate != null ? (quality.missing_rate * 100).toFixed(1) : '--';
    const missingColor = quality.missing_rate < 0.05 ? '#10b981' : quality.missing_rate < 0.15 ? '#3b82f6' : '#ef4444';

    // 时间范围
    const startDate = data.start_date || '--';
    const endDate = data.end_date || '--';

    // 复权状态
    const adjusted = quality.adjusted || false;
    const adjustedColor = adjusted ? '#10b981' : '#ef4444';
    const adjustedIcon = adjusted ? 'fa-check' : 'fa-xmark';

    let html = `
        <div class="data-quality-bar">
            <div class="dq-indicator">
                <span class="dq-dot info"></span>
                <span class="dq-label">综合质量</span>
            </div>
            <div style="display:flex;align-items:center;gap:12px;">
                <span class="dq-value" style="color:${scoreColor};font-size:24px;">${overallScore}</span>
                <span style="font-size:13px;color:#94a3b8;">${scoreDesc}</span>
            </div>
        </div>

        <div class="metric-cards">
            <div class="metric-card">
                <div class="metric-label">缺失率</div>
                <div class="metric-value" style="color:${missingColor}">${missingRate}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">复权状态</div>
                <div class="metric-value" style="color:${adjustedColor};font-size:18px;">
                    <i class="fas ${adjustedIcon}" style="margin-right:4px;"></i> ${adjusted ? '已复权' : '未复权'}
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-label">数据天数</div>
                <div class="metric-value">${quality.n_days || '--'}</div>
            </div>
        </div>

        <div class="model-info-row" style="margin-top:12px;">
            <span class="info-label">数据范围</span>
            <span class="info-value">${startDate} ~ ${endDate}</span>
        </div>

        ${indicatorHTML ? `<div style="margin-top:14px;">
            <h4 style="font-size:14px;color:#cbd5e1;margin:0 0 8px;font-weight:700;">数据源明细</h4>
            ${indicatorHTML}
        </div>` : ''}
    `;

    container.innerHTML = html;
}

// 别名：与 index.html 中 onclick="refreshQuantModel()" 兼容
function refreshQuantModel() {
    loadQuantModelTab(getCurrentStockCode());
}

/**
 * 获取当前股票代码 (Phase 2: thscode 消歧)
 * 自动标准化: "300620" → "sz300620", "300620.SZ" → "sz300620"
 */
function getCurrentStockCode() {
    const codeInput = document.getElementById('stockCode');
    const raw = codeInput ? codeInput.value.trim() : '';
    const code = raw || 'sz300620';
    // Phase 2: thscode 消歧
    return typeof normalizeStockCode === 'function' ? normalizeStockCode(code) : code;
}

// ═══════════════════════════════════════════════════
// 深度学习 V2 面板 (Transformer-LSTM + RL + FinBERT)
// ═══════════════════════════════════════════════════

async function loadDLV2Panel(code) {
    const container = document.getElementById('dl-v2-panel');
    if (!container) return;

    container.innerHTML = '<div class="loading"><i class="fas fa-spinner fa-spin"></i> 加载深度学习模型...</div>';

    try {
        // 并行加载多个 API (使用 allSettled 防止单个失败导致整体失败)
        const results = await Promise.allSettled([
            apiGet(`${API_BASE}/api/dl/predict/${code}`),
            apiGet(`${API_BASE}/api/rl/trader/status`),
            apiGet(`${API_BASE}/api/sentiment/bert/${code}`, 25000),
            apiGet(`${API_BASE}/api/dl/ensemble/report`)
        ]);
        const [predRes, rlRes, sentimentRes, reportRes] = results.map(r => r.status === 'fulfilled' ? r.value : null);

        // 深度学习预测
        let dlHtml = '<div class="dl-v2-grid">';

        // 预测卡片
        dlHtml += `
            <div class="dl-card">
                <div class="dl-card-header">
                    <i class="fas fa-brain"></i> Transformer-LSTM 预测
                </div>
                <div class="dl-card-body">
        `;

        // 兼容 data.data 嵌套和直接返回格式
        const pred = predRes?.prediction || predRes?.data?.prediction;
        if (pred) {
            const p = predRes.prediction;
            const dirClass = p.direction === 'up' ? 'bullish' : p.direction === 'down' ? 'bearish' : 'neutral';
            const dirText = p.direction === 'up' ? '看涨' : p.direction === 'down' ? '看跌' : '中性';

            dlHtml += `
                <div class="dl-prediction-main ${dirClass}">
                    <div class="dl-direction">${dirText}</div>
                    <div class="dl-confidence">置信度 ${(p.confidence * 100).toFixed(1)}%</div>
                </div>
                <div class="dl-prob-bars">
                    <div class="dl-prob-row">
                        <span>上涨</span>
                        <div class="dl-prob-fill" style="width:${p.probabilities.up * 100}%"></div>
                        <span>${(p.probabilities.up * 100).toFixed(1)}%</span>
                    </div>
                    <div class="dl-prob-row">
                        <span>中性</span>
                        <div class="dl-prob-fill neutral" style="width:${p.probabilities.neutral * 100}%"></div>
                        <span>${(p.probabilities.neutral * 100).toFixed(1)}%</span>
                    </div>
                    <div class="dl-prob-row">
                        <span>下跌</span>
                        <div class="dl-prob-fill bearish" style="width:${p.probabilities.down * 100}%"></div>
                        <span>${(p.probabilities.down * 100).toFixed(1)}%</span>
                    </div>
                </div>
            `;
        } else {
            dlHtml += '<div class="dl-no-data">预测数据不可用</div>';
        }

        dlHtml += `
                </div>
            </div>
        `;

        // RL 状态卡片
        dlHtml += `
            <div class="dl-card">
                <div class="dl-card-header">
                    <i class="fas fa-robot"></i> 强化学习 RL Trader
                </div>
                <div class="dl-card-body">
        `;

        if (rlRes && rlRes.status) {
            const s = rlRes.status;
            const trainedClass = s.trained ? 'trained' : 'untrained';
            const trainedText = s.trained ? '已训练' : '未训练';

            dlHtml += `
                <div class="dl-rl-status ${trainedClass}">
                    <div class="dl-status-dot"></div>
                    <span>${trainedText}</span>
                </div>
                <div class="dl-rl-info">
                    <div class="dl-info-row">
                        <span>市场状态:</span>
                        <span>${s.market_regime || '未知'}</span>
                    </div>
                    <div class="dl-info-row">
                        <span>PPO Agent:</span>
                        <span>${s.ppo_available ? '可用' : '不可用'}</span>
                    </div>
                    <div class="dl-info-row">
                        <span>SAC Agent:</span>
                        <span>${s.sac_available ? '可用' : '不可用'}</span>
                    </div>
                </div>
            `;
        } else {
            dlHtml += '<div class="dl-no-data">RL 状态不可用</div>';
        }

        dlHtml += `
                </div>
            </div>
        `;

        // 情感分析卡片
        dlHtml += `
            <div class="dl-card">
                <div class="dl-card-header">
                    <i class="fas fa-comment-dots"></i> FinBERT 情感
                </div>
                <div class="dl-card-body">
        `;

        // 2026-09-10 形状断修复: 原读 s.positive/neutral/negative (后端真实
        // 契约已无这些扁平键) → NaN% 渲染; 现对齐 engine.get_sentiment_score
        // 契约 {score, label, confidence, n_articles, method}; 无舆情诚实空态
        const s = sentimentRes && sentimentRes.sentiment;
        if (s && (s.n_articles ?? 0) > 0 && typeof s.score === 'number') {
            const score = s.score || 0;
            const sentimentClass = score > 0.15 ? 'positive' : score < -0.15 ? 'negative' : 'neutral';
            const sentimentText = score > 0.15 ? '正面' : score < -0.15 ? '负面' : '中性';

            dlHtml += `
                <div class="dl-sentiment-main ${sentimentClass}">
                    <div class="dl-sentiment-score">${score.toFixed(3)}</div>
                    <div class="dl-sentiment-label">${sentimentText}</div>
                </div>
                <div class="dl-sentiment-breakdown">
                    <div class="dl-sb-item neutral">
                        <div class="dl-sb-value" style="color:#d29922">${s.n_articles}</div>
                        <div class="dl-sb-label">舆情 ${s.method || 'finbert'}</div>
                    </div>
                    <div class="dl-sb-item positive">
                        <div class="dl-sb-value" style="color:#10b981">${(((s.confidence ?? 0) * 100) || 0).toFixed(0)}%</div>
                        <div class="dl-sb-label">置信</div>
                    </div>
                </div>
            `;
        } else {
            const empty = (s && (s.n_articles ?? 0) === 0) ? '情感数据不可用 (新闻0/股吧0)' : '情感数据不可用';
            dlHtml += `<div class="dl-no-data">${empty}</div>`;
        }

        dlHtml += `
                </div>
            </div>
        `;

        // 模型架构卡片
        dlHtml += `
            <div class="dl-card">
                <div class="dl-card-header">
                    <i class="fas fa-sitemap"></i> 模型架构
                </div>
                <div class="dl-card-body dl-architecture">
        `;

        if (reportRes && reportRes.report) {
            const r = reportRes.report;
            const p = r.params || {};
            dlHtml += `
                <div class="dl-arch-layer">
                    <span class="dl-arch-icon">📥</span>
                    <span>Input: ${p.n_features || 12} 维 × ${p.seq_len || 60} 步</span>
                </div>
                <div class="dl-arch-arrow">↓</div>
                <div class="dl-arch-layer">
                    <span class="dl-arch-icon">🧩</span>
                    <span>Patch Embedding</span>
                    <span class="dl-arch-detail">patch_len=${p.patch_len || 8}</span>
                </div>
                <div class="dl-arch-arrow">↓</div>
                <div class="dl-arch-layer">
                    <span class="dl-arch-icon">🔀</span>
                    <span>Transformer Encoder (RoPE)</span>
                    <span class="dl-arch-detail">${p.d_model || 128}d, ${p.n_heads || 8} heads, ${p.n_layers || 4} layers</span>
                </div>
                <div class="dl-arch-arrow">↓</div>
                <div class="dl-arch-layer">
                    <span class="dl-arch-icon">🎯</span>
                    <span>Self-Attention Pooling</span>
                </div>
                <div class="dl-arch-arrow">↓</div>
                <div class="dl-arch-layer dl-arch-output">
                    <span class="dl-arch-icon">📤</span>
                    <span>Output: Direction + Confidence</span>
                </div>
            `;
        } else {
            dlHtml += '<div class="dl-no-data">模型报告不可用</div>';
        }

        dlHtml += `
                </div>
            </div>
        `;

        dlHtml += '</div>'; // dl-v2-grid
        container.innerHTML = dlHtml;

    } catch (e) {
        console.error('loadDLV2Panel error:', e);
        container.innerHTML = '<div class="dl-error">加载失败：' + e.message + '</div>';
    }
}

// ═══════════════════════════════════════════════════
// Time-LLM 统一预测面板
// ═══════════════════════════════════════════════════

async function loadTimeLLMPanel(code) {
    const container = document.getElementById('time-llm-panel');
    if (!container) return;

    container.innerHTML = '<div class="loading"><i class="fas fa-spinner fa-spin"></i> 加载 Time-LLM...</div>';

    try {
        const [predRes, statusRes] = await Promise.all([
            apiGet(`${API_BASE}/api/time-llm/predict/${code}`),
            apiGet(`${API_BASE}/api/time-llm/status`)
        ]);

        let html = '<div class="dl-v2-grid">';

        // 预测卡片
        if (predRes && predRes.success && predRes.data) {
            const p = predRes.data;
            const dirClass = p.direction === 'bullish' ? 'bullish' : p.direction === 'bearish' ? 'bearish' : 'neutral';
            const dirText = p.direction === 'up' ? '看涨' : p.direction === 'down' ? '看跌' : p.direction === 'bearish' ? '看跌' : '中性';

            html += `
                <div class="dl-card">
                    <div class="dl-card-header">
                        <i class="fas fa-clock"></i> Time-LLM 预测
                    </div>
                    <div class="dl-card-body">
                        <div class="dl-prediction-main ${dirClass}">
                            <div class="dl-direction">${dirText}</div>
                            <div class="dl-confidence">置信度 ${(p.confidence * 100).toFixed(1)}%</div>
                        </div>
                        <div class="dl-prob-bars">
                            <div class="dl-prob-row">
                                <span>上涨</span>
                                <div class="dl-prob-fill" style="width:${(p.probabilities?.up || 0) * 100}%"></div>
                                <span>${((p.probabilities?.up || 0) * 100).toFixed(1)}%</span>
                            </div>
                            <div class="dl-prob-row">
                                <span>中性</span>
                                <div class="dl-prob-fill neutral" style="width:${(p.probabilities?.neutral || 0) * 100}%"></div>
                                <span>${((p.probabilities?.neutral || 0) * 100).toFixed(1)}%</span>
                            </div>
                            <div class="dl-prob-row">
                                <span>下跌</span>
                                <div class="dl-prob-fill bearish" style="width:${(p.probabilities?.down || 0) * 100}%"></div>
                                <span>${((p.probabilities?.down || 0) * 100).toFixed(1)}%</span>
                            </div>
                        </div>
                        <div class="dl-rl-info" style="margin-top: 12px;">
                            <div class="dl-info-row">
                                <span>市场状态:</span>
                                <span>${p.market_regime || '未知'}</span>
                            </div>
                            <div class="dl-info-row">
                                <span>执行时间:</span>
                                <span>${(p.execution_time_ms || 0).toFixed(0)}ms</span>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        } else {
            html += `
                <div class="dl-card">
                    <div class="dl-card-header">
                        <i class="fas fa-clock"></i> Time-LLM 预测
                    </div>
                    <div class="dl-card-body">
                        <div class="dl-no-data">预测数据不可用</div>
                    </div>
                </div>
            `;
        }

        // 模型状态卡片
        if (statusRes && statusRes.success && statusRes.data) {
            const s = statusRes.data;
            html += `
                <div class="dl-card">
                    <div class="dl-card-header">
                        <i class="fas fa-info-circle"></i> Time-LLM 状态
                    </div>
                    <div class="dl-card-body">
                        <div class="dl-rl-info">
                            <div class="dl-info-row">
                                <span>已训练:</span>
                                <span>${s.is_trained ? '是' : '否'}</span>
                            </div>
                            <div class="dl-info-row">
                                <span>模型数量:</span>
                                <span>${s.model_count || 0}</span>
                            </div>
                            ${s.patchtst_trained !== undefined ? `
                            <div class="dl-info-row">
                                <span>PatchTST:</span>
                                <span>${s.patchtst_trained ? '已训练' : '未训练'}</span>
                            </div>
                            ` : ''}
                            ${s.mamba_trained !== undefined ? `
                            <div class="dl-info-row">
                                <span>Mamba:</span>
                                <span>${s.mamba_trained ? '已训练' : '未训练'}</span>
                            </div>
                            ` : ''}
                            ${s.diffusion_trained !== undefined ? `
                            <div class="dl-info-row">
                                <span>Diffusion:</span>
                                <span>${s.diffusion_trained ? '已训练' : '未训练'}</span>
                            </div>
                            ` : ''}
                        </div>
                    </div>
                </div>
            `;
        }

        html += '</div>';
        container.innerHTML = html;

    } catch (e) {
        console.error('loadTimeLLMPanel error:', e);
        container.innerHTML = '<div class="dl-error">加载失败：' + e.message + '</div>';
    }
}

// ═══════════════════════════════════════════════════
// Regime-Switching 多模型预测面板
// ═══════════════════════════════════════════════════

async function loadRegimeSwitchingPanel(code) {
    const container = document.getElementById('regime-switching-panel');
    if (!container) return;

    container.innerHTML = '<div class="loading"><i class="fas fa-spinner fa-spin"></i> 加载 Regime-Switching...</div>';

    try {
        const [predRes, statusRes, regimeRes] = await Promise.all([
            apiGet(`${API_BASE}/api/regime-switching/predict/${code}`),
            apiGet(`${API_BASE}/api/regime-switching/status`),
            apiGet(`${API_BASE}/api/regime-switching/regime/${code}`)
        ]);

        let html = '<div class="dl-v2-grid">';

        // 预测卡片
        if (predRes && predRes.success && predRes.data) {
            const p = predRes.data;
            const dirClass = p.direction === 'bullish' ? 'bullish' : p.direction === 'bearish' ? 'bearish' : 'neutral';
            const dirText = p.direction === 'up' ? '看涨' : p.direction === 'down' ? '看跌' : p.direction === 'bearish' ? '看跌' : '中性';

            html += `
                <div class="dl-card">
                    <div class="dl-card-header">
                        <i class="fas fa-random"></i> Regime-Switching 预测
                    </div>
                    <div class="dl-card-body">
                        <div class="dl-prediction-main ${dirClass}">
                            <div class="dl-direction">${dirText}</div>
                            <div class="dl-confidence">置信度 ${(p.confidence * 100).toFixed(1)}%</div>
                        </div>
                        <div class="dl-prob-bars">
                            <div class="dl-prob-row">
                                <span>上涨</span>
                                <div class="dl-prob-fill" style="width:${(p.probabilities?.up || 0) * 100}%"></div>
                                <span>${((p.probabilities?.up || 0) * 100).toFixed(1)}%</span>
                            </div>
                            <div class="dl-prob-row">
                                <span>中性</span>
                                <div class="dl-prob-fill neutral" style="width:${(p.probabilities?.neutral || 0) * 100}%"></div>
                                <span>${((p.probabilities?.neutral || 0) * 100).toFixed(1)}%</span>
                            </div>
                            <div class="dl-prob-row">
                                <span>下跌</span>
                                <div class="dl-prob-fill bearish" style="width:${(p.probabilities?.down || 0) * 100}%"></div>
                                <span>${((p.probabilities?.down || 0) * 100).toFixed(1)}%</span>
                            </div>
                        </div>
                        <div class="dl-rl-info" style="margin-top: 12px;">
                            <div class="dl-info-row">
                                <span>检测到的 Regime:</span>
                                <span>${p.detected_regime || '未知'}</span>
                            </div>
                            <div class="dl-info-row">
                                <span>Regime 置信度:</span>
                                <span>${(p.regime_confidence * 100).toFixed(1)}%</span>
                            </div>
                            <div class="dl-info-row">
                                <span>模型数量:</span>
                                <span>${p.model_count || 0}</span>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        } else {
            html += `
                <div class="dl-card">
                    <div class="dl-card-header">
                        <i class="fas fa-random"></i> Regime-Switching 预测
                    </div>
                    <div class="dl-card-body">
                        <div class="dl-no-data">预测数据不可用</div>
                    </div>
                </div>
            `;
        }

        // 模型排名卡片
        if (statusRes && statusRes.success && statusRes.data) {
            const s = statusRes.data;
            const models = s.models || [];
            const ranking = s.ranking || [];

            html += `
                <div class="dl-card">
                    <div class="dl-card-header">
                        <i class="fas fa-trophy"></i> 模型性能排名
                    </div>
                    <div class="dl-card-body">
                        ${ranking.length > 0 ? `
                        <div class="dl-rl-info">
                            ${ranking.slice(0, 5).map((r, i) => `
                                <div class="dl-info-row">
                                    <span>#${i + 1} ${r.model || 'Model'}:</span>
                                    <span>${(r.accuracy * 100).toFixed(1)}%</span>
                                </div>
                            `).join('')}
                        </div>
                        ` : `
                        <div class="dl-no-data">暂无排名数据</div>
                        `}
                    </div>
                </div>
            `;
        }

        // Regime 检测卡片
        if (regimeRes && regimeRes.success && regimeRes.data) {
            const r = regimeRes.data;
            html += `
                <div class="dl-card">
                    <div class="dl-card-header">
                        <i class="fas fa-chart-pie"></i> Regime 检测
                    </div>
                    <div class="dl-card-body">
                        <div class="dl-rl-info">
                            <div class="dl-info-row">
                                <span>当前 Regime:</span>
                                <span style="font-weight: 700; color: #3b82f6;">${r.regime || '未知'}</span>
                            </div>
                            <div class="dl-info-row">
                                <span>Regime 概率:</span>
                                <span>${(r.regime_probability * 100).toFixed(1)}%</span>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }

        html += '</div>';
        container.innerHTML = html;

    } catch (e) {
        console.error('loadRegimeSwitchingPanel error:', e);
        container.innerHTML = '<div class="dl-error">加载失败：' + e.message + '</div>';
    }
}

// ═══════════════════════════════════════════════════
// CVaR 仓位管理面板
// ═══════════════════════════════════════════════════

async function loadCVaRPositionPanel(code) {
    const container = document.getElementById('cvar-position-panel');
    if (!container) return;

    try {
        const data = await apiGet(`${API_BASE}/api/position/cvar/${code}`);
        if (!data || !data.success) {
            container.innerHTML = `<div style="color:#94a3b8;font-size:13px;">${data?.error || '加载失败'}</div>`;
            return;
        }

        const pos = data.position || {};
        const risk = data.risk || {};
        const signal = data.signal || {};

        const posColor = pos.vol_adjusted > 0.3 ? '#10b981' : pos.vol_adjusted > 0.1 ? '#3b82f6' : '#94a3b8';
        const cvarColor = parseFloat(risk.cvar_95) > 3 ? '#ef4444' : parseFloat(risk.cvar_95) > 1.5 ? '#f59e0b' : '#10b981';

        // 2026-09-09: 真实持仓对照区 (vs_position 来自 PortfolioStore, 无持仓时不渲染)
        let vsHtml = '';
        const v = data.vs_position;
        if (v && v.has_position) {
            const hintColor = (v.hint || '').includes('减') || (v.hint || '').includes('警戒') ? '#ef4444'
                : (v.hint || '').includes('加仓') ? '#10b981' : '#94a3b8';
            vsHtml = `
                <div style="margin-top:12px;padding:10px;background:rgba(15,23,42,0.6);border:1px solid rgba(59,130,246,0.25);border-radius:6px;">
                    <div style="font-size:12px;color:#94a3b8;margin-bottom:6px;"><i class="fas fa-briefcase"></i> 我的真实持仓 · ${v.name || code}</div>
                    <div style="display:flex;justify-content:space-between;font-size:12px;color:#cbd5e1;"><span>成本 ¥${v.cost} × ${Number(v.qty).toLocaleString()} 股</span><span>市值 ¥${Number(v.market_value).toLocaleString()}</span></div>
                    <div style="display:flex;justify-content:space-between;font-size:12px;color:#cbd5e1;margin-top:4px;"><span>实际仓位 <b style="color:#60a5fa;">${v.weight_actual_pct}%</b> <span style="color:#64748b;">vs 风险预算 ${v.budget_pct}%</span></span><span style="color:${v.pl_pct >= 0 ? '#10b981' : '#ef4444'};">浮${v.pl_pct >= 0 ? '盈' : '亏'} ${v.pl_pct}%</span></div>
                    <div style="display:flex;justify-content:space-between;font-size:12px;margin-top:4px;"><span style="color:#94a3b8;">止损警戒 ¥${v.stop_line}</span><span style="color:${hintColor};font-weight:700;">${v.hint}</span></div>
                </div>`;
        }

        let html = `
            <div class="metric-cards">
                <div class="metric-card">
                    <div class="metric-label">Kelly 仓位</div>
                    <div class="metric-value">${(pos.kelly_fraction * 100).toFixed(1)}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">CVaR 约束</div>
                    <div class="metric-value">${(pos.cvar_constraint * 100).toFixed(1)}%</div>
                </div>
            </div>

            <div style="margin-top:12px;padding:10px;background:rgba(59,130,246,0.1);border-radius:6px;text-align:center;">
                <div style="font-size:12px;color:#94a3b8;">建议仓位</div>
                <div style="font-size:28px;font-weight:700;color:${posColor};">${(pos.vol_adjusted * 100).toFixed(1)}%</div>
                <div style="font-size:13px;color:#cbd5e1;margin-top:4px;">
                    价值: ¥${(pos.position_value || 0).toLocaleString()}
                </div>
            </div>

            <div class="metric-cards" style="margin-top:12px;">
                <div class="metric-card">
                    <div class="metric-label">CVaR (95%)</div>
                    <div class="metric-value" style="color:${cvarColor};font-size:16px;">${risk.cvar_95}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">年化波动</div>
                    <div class="metric-value" style="font-size:16px;">${risk.annualized_vol}%</div>
                </div>
            </div>

            <div style="margin-top:10px;font-size:12px;color:#94a3b8;">
                <div>胜率: ${risk.win_rate}% | 平均盈利: ${risk.avg_win}% | 平均亏损: ${risk.avg_loss}%</div>
                <div style="margin-top:4px;">
                    信号: <span style="color:${signal.direction === 'up' ? '#10b981' : signal.direction === 'down' ? '#ef4444' : '#94a3b8'};font-weight:700;">
                        ${signal.direction === 'up' ? '↑ 看涨' : signal.direction === 'down' ? '↓ 看跌' : '→ 中性'}
                    </span>
                    | 置信度: ${(signal.confidence * 100).toFixed(0)}%
                </div>
            </div>
        `;

        container.innerHTML = html + vsHtml;

    } catch (e) {
        console.error('loadCVaRPositionPanel error:', e);
        container.innerHTML = '<div style="color:#94a3b8;font-size:13px;">加载失败</div>';
    }
}

// ═══════════════════════════════════════════════════
// MLOps 管道面板
// ═══════════════════════════════════════════════════

async function loadMLOpsPanel(code) {
    const container = document.getElementById('mlops-panel');
    if (!container) return;

    try {
        // 并行加载 MLOps 状态和漂移状态
        const [statusRes, driftRes, checkRes] = await Promise.allSettled([
            apiGet(`${API_BASE}/api/mlops/status`),
            apiGet(`${API_BASE}/api/mlops/drift-status`),
            apiGet(`${API_BASE}/api/mlops/check-retrain/${code}`),
        ]);

        const status = statusRes.status === 'fulfilled' ? statusRes.value : null;
        const drift = driftRes.status === 'fulfilled' ? driftRes.value : null;
        const check = checkRes.status === 'fulfilled' ? checkRes.value : null;

        if (!status) {
            container.innerHTML = '<div style="color:#94a3b8;font-size:13px;">MLOps 服务不可用</div>';
            return;
        }

        const statusVal = status.status || 'idle';
        const statusColors = {
            idle: '#94a3b8', monitoring: '#3b82f6', training: '#f59e0b',
            a_b_testing: '#8b5cf6', deploying: '#06b6d4', error: '#ef4444',
        };
        const statusLabels = {
            idle: '⏸ 空闲', monitoring: '👁 监控中', training: '🔄 训练中',
            a_b_testing: '🧪 A/B 测试', deploying: '🚀 部署中', error: '❌ 错误',
        };

        const modules = status.modules || {};
        const moduleList = [
            { name: '漂移检测', key: 'drift_detector', icon: 'fa-exclamation-triangle' },
            { name: '重训练触发', key: 'retrain_trigger', icon: 'fa-rotate' },
            { name: 'A/B 测试', key: 'ab_test', icon: 'fa-flask' },
            { name: '模型注册', key: 'model_registry', icon: 'fa-database' },
            { name: '训练管道', key: 'training_pipeline', icon: 'fa-gears' },
        ];

        let modulesHTML = moduleList.map(m => `
            <div style="display:flex;align-items:center;gap:6px;font-size:12px;color:#cbd5e1;">
                <i class="fas ${modules[m.key] ? 'fa-circle-check' : 'fa-circle-xmark'}"
                   style="color:${modules[m.key] ? '#10b981' : '#ef4444'};"></i>
                ${m.name}
            </div>
        `).join('');

        let retrainHTML = '';
        if (check && check.need_retrain) {
            retrainHTML = `
                <div style="margin-top:8px;padding:8px;background:rgba(245,158,246,0.1);border-radius:4px;font-size:12px;color:#f59e0b;">
                    <i class="fas fa-exclamation-triangle"></i> 需要重训练: ${check.reasons.join(', ')}
                </div>
            `;
        }

        let driftHTML = '';
        if (drift && drift.drift_detected) {
            driftHTML = `
                <div style="margin-top:8px;padding:8px;background:rgba(239,68,68,0.1);border-radius:4px;font-size:12px;color:#ef4444;">
                    <i class="fas fa-exclamation-triangle"></i> 检测到概念漂移!
                </div>
            `;
        }

        let html = `
            <div style="text-align:center;padding:8px;background:rgba(15,23,42,0.5);border-radius:6px;margin-bottom:10px;">
                <div style="font-size:12px;color:#94a3b8;">管道状态</div>
                <div style="font-size:16px;font-weight:700;color:${statusColors[statusVal] || '#94a3b8'};">
                    ${statusLabels[statusVal] || statusVal}
                </div>
            </div>

            <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:10px;">
                <div style="text-align:center;padding:8px;background:rgba(59,130,246,0.1);border-radius:6px;">
                    <div style="font-size:11px;color:#94a3b8;">重训练次数</div>
                    <div style="font-size:20px;font-weight:700;color:#3b82f6;">${status.retrain_count || 0}</div>
                </div>
                <div style="text-align:center;padding:8px;background:rgba(16,185,129,0.1);border-radius:6px;">
                    <div style="font-size:11px;color:#94a3b8;">部署次数</div>
                    <div style="font-size:20px;font-weight:700;color:#10b981;">${status.deployment_count || 0}</div>
                </div>
            </div>

            <div style="margin-bottom:10px;">
                <div style="font-size:12px;color:#94a3b8;margin-bottom:4px;">模块状态</div>
                ${modulesHTML}
            </div>

            ${retrainHTML}
            ${driftHTML}

            <div style="margin-top:8px;font-size:11px;color:#64748b;">
                上次漂移: ${status.last_drift_time ? new Date(status.last_drift_time).toLocaleString('zh-CN') : '无'}
                <br>
                上次部署: ${status.last_deployment_time ? new Date(status.last_deployment_time).toLocaleString('zh-CN') : '无'}
            </div>
        `;

        container.innerHTML = html;

    } catch (e) {
        console.error('loadMLOpsPanel error:', e);
        container.innerHTML = '<div style="color:#94a3b8;font-size:13px;">加载失败</div>';
    }
}

// ============================================================================
// P0-5: 多模态融合预测面板
// ============================================================================

async function loadFusionPanel(code) {
    const container = document.getElementById('fusion-panel');
    if (!container) return;

    try {
        const resp = await apiGet(`${API_BASE}/api/fusion/multi-modal?stock_code=${code}&cache=false`);
        if (!resp || !resp.success) {
            container.innerHTML = '<div style="color:#94a3b8;font-size:13px;">融合预测服务不可用</div>';
            return;
        }

        const d = resp;
        const dirColor = d.direction === 'up' ? '#10b981' : d.direction === 'down' ? '#ef4444' : '#d29922';
        const dirIcon = d.direction === 'up' ? '📈' : d.direction === 'down' ? '📉' : '➡️';
        const dirText = d.direction === 'up' ? '看涨' : d.direction === 'down' ? '看跌' : '震荡';

        const weights = d.modality_weights || {};
        const scores = d.scores || {};
        const mods = d.modalities || {};
        const modLabels = {price: '📊 价格', sentiment: '💬 情感', fundamental: '📋 基本面'};

        let html = `
            <div style="text-align:center;padding:10px;background:rgba(15,23,42,0.5);border-radius:6px;margin-bottom:10px;">
                <div style="font-size:20px;">${dirIcon}</div>
                <div style="font-size:16px;font-weight:700;color:${dirColor};">${dirText}</div>
                <div style="font-size:12px;color:#94a3b8;">置信度 ${(d.confidence * 100).toFixed(1)}% | Regime: ${d.regime} | 一致度 ${(d.consensus * 100).toFixed(0)}%</div>
            </div>

            <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:10px;">
                <div style="padding:6px;background:rgba(59,130,246,0.1);border-radius:4px;font-size:12px;">
                    <div style="color:#94a3b8;">📊 价格权重</div>
                    <div style="color:#60a5fa;font-weight:700;">${((weights.price || 0) * 100).toFixed(0)}%</div>
                </div>
                <div style="padding:6px;background:rgba(59,130,246,0.1);border-radius:4px;font-size:12px;">
                    <div style="color:#94a3b8;">💬 情感权重</div>
                    <div style="color:#60a5fa;font-weight:700;">${((weights.sentiment || 0) * 100).toFixed(0)}%</div>
                </div>
                <div style="padding:6px;background:rgba(59,130,246,0.1);border-radius:4px;font-size:12px;">
                    <div style="color:#94a3b8;">📋 基本面权重</div>
                    <div style="color:#60a5fa;font-weight:700;">${((weights.fundamental || 0) * 100).toFixed(0)}%</div>
                </div>
                <div style="padding:6px;background:rgba(59,130,246,0.1);border-radius:4px;font-size:12px;">
                    <div style="color:#94a3b8;">模态一致度</div>
                    <div style="color:#60a5fa;font-weight:700;">${(d.consensus * 100).toFixed(0)}%</div>
                </div>
            </div>

            <div style="font-size:12px;">
                <div style="color:#94a3b8;margin-bottom:4px;">方向得分</div>
                <div style="display:flex;justify-content:space-between;padding:2px 0;">
                    <span>📈 看涨</span><span style="color:#10b981;">${((scores.up || 0) * 100).toFixed(1)}%</span>
                </div>
                <div style="display:flex;justify-content:space-between;padding:2px 0;">
                    <span>➡️ 震荡</span><span style="color:#d29922;">${((scores.neutral || 0) * 100).toFixed(1)}%</span>
                </div>
                <div style="display:flex;justify-content:space-between;padding:2px 0;">
                    <span>📉 看跌</span><span style="color:#ef4444;">${((scores.down || 0) * 100).toFixed(1)}%</span>
                </div>
            </div>

            <div style="margin-top:8px;font-size:12px;">
                <div style="color:#94a3b8;margin-bottom:4px;">各模态信号</div>
                ${Object.entries(mods).map(([key, mod]) => {
                    const dir = mod.direction === 'up' ? '📈' : mod.direction === 'down' ? '📉' : '➡️';
                    return `<div style="display:flex;justify-content:space-between;padding:2px 0;">
                        <span>${modLabels[key] || key}</span><span>${dir} ${(mod.confidence * 100).toFixed(0)}%</span>
                    </div>`;
                }).join('')}
            </div>
        `;

        container.innerHTML = html;

    } catch (e) {
        console.error('loadFusionPanel error:', e);
        container.innerHTML = '<div style="color:#94a3b8;font-size:13px;">加载失败</div>';
    }
}

// ============================================================================
// P0-6: 金丝雀部署面板
// ============================================================================

async function loadCanaryPanel() {
    const container = document.getElementById('canary-panel');
    if (!container) return;

    try {
        const resp = await apiGet(`${API_BASE}/api/mlops/canary/status`);
        if (!resp || !resp.success) {
            container.innerHTML = '<div style="color:#94a3b8;font-size:13px;">金丝雀部署服务不可用</div>';
            return;
        }

        const active = resp.active_count || 0;
        const deployments = resp.deployments || {};
        const keys = Object.keys(deployments);

        const statusColors = {
            promoted: '#10b981', rolled_back: '#ef4444', scaling: '#d29922',
            active: '#3b82f6', ended: '#94a3b8',
        };
        const statusTexts = {
            promoted: '已提升', rolled_back: '已回滚', scaling: '流量提升中',
            active: '活跃', ended: '已结束',
        };

        // 流量阶梯可视化
        const steps = [5, 25, 50, 100];

        let html = `
            <div style="padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.05);">
                活跃部署: <span style="color:${active > 0 ? '#10b981' : '#94a3b8'};font-weight:700;">${active}</span>
            </div>

            <!-- 启动金丝雀按钮 -->
            <button onclick="startCanaryDeploy()" style="
                width: 100%; margin-top: 8px; padding: 8px;
                background: linear-gradient(135deg, #3b82f6, #2563eb);
                color: white; border: none; border-radius: 6px;
                cursor: pointer; font-size: 13px; font-weight: 600;
            ">
                🚀 启动金丝雀部署
            </button>

            <!-- 流量阶梯 -->
            <div style="margin-top:10px;padding:8px;background:rgba(15,23,42,0.5);border-radius:6px;">
                <div style="font-size:11px;color:#94a3b8;margin-bottom:6px;">流量阶梯</div>
                <div style="display:flex;gap:4px;">
                    ${steps.map((s, i) => `
                        <div style="flex:1;text-align:center;padding:4px 0;background:rgba(59,130,246,0.1);border-radius:4px;font-size:11px;color:#60a5fa;">
                            ${s}%
                        </div>
                    `).join('')}
                </div>
            </div>

            <!-- 回滚条件 -->
            <div style="margin-top:8px;font-size:11px;color:#94a3b8;">
                <div style="color:#ef4444;margin-bottom:2px;">⚠️ 自动回滚条件:</div>
                <div>• Canary 准确率 &lt; 55%</div>
                <div>• 比稳定模型差 &gt; 2%</div>
            </div>
        `;

        for (const [code, dep] of Object.entries(deployments)) {
            const sc = statusColors[dep.status] || '#3b82f6';
            const st = statusTexts[dep.status] || dep.status;
            const canaryAcc = dep.canary_total > 0
                ? (dep.canary_correct / dep.canary_total * 100).toFixed(1)
                : '-';
            const currentStep = dep.traffic_step || 0;

            // 流量进度条
            const progressWidth = (currentStep / (steps.length - 1)) * 100;

            html += `
                <div style="margin-top:8px;padding:8px;background:rgba(15,23,42,0.5);border-radius:6px;">
                    <div style="font-weight:bold;color:${sc};margin-bottom:4px;">${code} — ${st}</div>
                    <div style="font-size:11px;color:#94a3b8;">${dep.old_version} → ${dep.new_version}</div>
                    <div style="font-size:11px;color:#94a3b8;margin-top:2px;">
                        流量: ${(dep.traffic_ratio * 100).toFixed(0)}%
                        <span style="display:inline-block;width:60px;height:4px;background:rgba(255,255,255,0.1);border-radius:2px;vertical-align:middle;margin-left:4px;">
                            <span style="display:block;width:${progressWidth}%;height:100%;background:#3b82f6;border-radius:2px;"></span>
                        </span>
                    </div>
                    <div style="font-size:11px;color:#94a3b8;">评估: ${dep.canary_total} 次 | Canary 准确率: ${canaryAcc}%</div>
                    ${dep.end_reason ? `<div style="font-size:11px;color:#94a3b8;margin-top:2px;">原因: ${dep.end_reason}</div>` : ''}
                </div>
            `;
        }

        if (keys.length === 0) {
            html += '<div style="color:#94a3b8;font-size:12px;margin-top:8px;text-align:center;">点击上方按钮启动金丝雀部署</div>';
        }

        container.innerHTML = html;

    } catch (e) {
        console.error('loadCanaryPanel error:', e);
        container.innerHTML = '<div style="color:#94a3b8;font-size:13px;">加载失败</div>';
    }
}

// 启动金丝雀部署
async function startCanaryDeploy() {
    // 获取当前选中的股票代码
    const stockSelect = document.getElementById('analysisStockSelect');
    const stockCode = stockSelect ? stockSelect.value : 'sz300620';

    try {
        const resp = await fetch(`${API_BASE}/api/mlops/canary/deploy`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                stock_code: stockCode,
                new_version: 'v2.1',
                old_version: 'v2.0',
            }),
        });
        const data = await resp.json();
        if (data.success) {
            alert(`金丝雀部署已启动!\n${stockCode}: v2.0 → v2.1 (流量 5%)`);
            loadCanaryPanel();
        } else {
            alert('启动失败: ' + (data.error || '未知错误'));
        }
    } catch (e) {
        alert('请求失败: ' + e.message);
    }
}

// ═══════════════════════════════════════════════════
// 每日晨报研报 (2026-09-09): 实时 Tab 呈现 + 08:00 链 + 手动触发 + 轮询
// ═══════════════════════════════════════════════════
let _drPollTimer = null;

function _renderDailyReport(rep) {
    const box = document.getElementById('dailyReportContent');
    const dateEl = document.getElementById('drDate');
    if (!box) return;
    if (!rep || !rep.report_md) {
        if (dateEl) dateEl.textContent = '';
        box.innerHTML = '<div style="color:#94a3b8;font-size:13px;">暂无晨报 (每日 08:00 自动生成, 或点「立即生成」)</div>';
        return;
    }
    if (dateEl) {
        dateEl.textContent = `${rep.date || ''} · ${rep.trigger === 'manual' ? '手动' : '自动'} · ${rep.elapsed_s || '?'}s`
            + (rep.llm_degraded ? ' · ⚠ LLM 降级 (规则模板)' : '');
    }
    const esc = s => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    let html = esc(rep.report_md)
        .replace(/^#{1,4} (.+)$/gm, '<b style="color:#60a5fa;">$1</b>')
        .replace(/\*\*(.+?)\*\*/g, '<b>$1</b>');
    if (rep.source_errors && rep.source_errors.length) {
        html += `\n\n<b style="color:#f59e0b;">⚠ 数据源: ${esc(rep.source_errors.join('; '))}</b>`;
    }
    box.innerHTML = `<div style="white-space:pre-wrap;font-size:13px;line-height:1.7;color:#e2e8f0;">${html}</div>`;
}

function _setDrBtn(busy) {
    const btn = document.getElementById('drTriggerBtn');
    if (!btn) return;
    btn.disabled = !busy;
    btn.innerHTML = busy
        ? '<i class="fas fa-spinner fa-spin"></i> 生成中…'
        : '<i class="fas fa-bolt"></i> 立即生成';
}

async function loadDailyReport() {
    const box = document.getElementById('dailyReportContent');
    if (!box) return;
    try {
        const r = await fetch(`${API_BASE}/api/daily_report`);
        const data = await r.json();
        _renderDailyReport(data.report);
        if (data.generating) { _setDrBtn(true); _drStartPolling(); }
    } catch (e) {
        console.error('loadDailyReport error:', e);
    }
}

function _drStartPolling() {
    if (_drPollTimer) return;
    let n = 0;
    _drPollTimer = setInterval(async () => {
        if (++n > 24) {  // 24×5s = 120s 上限, 超时不吞错 (控制台可见)
            clearInterval(_drPollTimer); _drPollTimer = null;
            console.warn('[DailyReport] 生成超时 (120s), 停止轮询');
            _setDrBtn(false);
            return;
        }
        try {
            const r = await fetch(`${API_BASE}/api/daily_report`);
            const d = await r.json();
            if (!d.generating) {
                clearInterval(_drPollTimer); _drPollTimer = null;
                _setDrBtn(false);
                _renderDailyReport(d.report);
            }
        } catch (e) {
            console.error('[DailyReport] 轮询失败 (继续重试):', e);
        }
    }, 5000);
}

window.triggerDailyReport = async function () {
    try {
        const r = await fetch(`${API_BASE}/api/daily_report`, { method: 'POST' });
        const d = await r.json();
        if (d.success) {
            _setDrBtn(true);
            const box = document.getElementById('dailyReportContent');
            if (box) box.innerHTML =
                '<div class="loading">晨报链启动: 美股行情 → 国际资讯 → 持仓量化 → LLM 汇编 (约 40-120s)…</div>';
            _drStartPolling();
        } else {
            alert(d.error || '触发失败');
        }
    } catch (e) {
        alert('触发失败: ' + e.message);
    }
};

// 加载时机: 进入「实时」Tab 时触发 (index.html nav onclick 调 loadDailyReport);
// 08:00 自动链/手动触发后轮询也走 _drStartPolling → 同一渲染函数
