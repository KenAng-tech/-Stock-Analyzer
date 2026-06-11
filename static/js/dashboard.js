// Stock Analyzer - Dashboard JavaScript (量化模型面板)
// 为"量化模型"标签页提供所有前端交互逻辑
// API_BASE 已由 app.js 在全局作用域定义，此处直接使用

// ═══════════════════════════════════════════════════
// 工具函数
// ═══════════════════════════════════════════════════

async function apiGet(url, timeoutMs = 10000) {
    try {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), timeoutMs);
        const resp = await fetch(url, { signal: controller.signal });
        clearTimeout(timeout);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        return await resp.json();
    } catch (e) {
        console.error(`[Dashboard API] ${url} 请求失败:`, e.message);
        return null;
    }
}

function showLoading(containerId) {
    const el = document.getElementById(containerId);
    if (el) el.innerHTML = '<div class="loading"><i class="fas fa-spinner fa-spin"></i> 加载中...</div>';
}

function showError(containerId, message) {
    const el = document.getElementById(containerId);
    if (el) el.innerHTML = `<div class="error"><i class="fas fa-exclamation-triangle"></i> ${message}</div>`;
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
    showLoading('risk-panel');
    const data = await apiGet(`${API_BASE}/api/dashboard/risk-report`);
    const container = document.getElementById('risk-panel');

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
// 模型健康面板
// ═══════════════════════════════════════════════════

async function loadModelHealthPanel() {
    showLoading('health-panel');
    const data = await apiGet(`${API_BASE}/api/dashboard/model-health`);
    const container = document.getElementById('health-panel');

    if (!data || !container) return;

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
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> 优化中...';
    btn.disabled = true;

    const resp = await fetch(`${API_BASE}/api/dashboard/hyperparams`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ n_trials: 30 }),
    });

    const data = await resp.json();
    btn.innerHTML = '<i class="fas fa-check"></i> 完成';

    if (data.best_params) {
        loadHyperparamsPanel();
    }
}

// ═══════════════════════════════════════════════════
// 情感分析面板
// ═══════════════════════════════════════════════════

async function loadSentimentPanel(code) {
    showLoading('sentiment-panel');
    const data = await apiGet(`${API_BASE}/api/dashboard/sentiment?code=${code}`);
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

function getCurrentStockCode() {
    const codeInput = document.getElementById('stockCode');
    if (codeInput) return codeInput.value.trim() || 'sz300620';
    return 'sz300620';
}

// ═══════════════════════════════════════════════════
// 深度学习 V2 面板 (Transformer-LSTM + RL + FinBERT)
// ═══════════════════════════════════════════════════

async function loadDLV2Panel(code) {
    const container = document.getElementById('dl-v2-panel');
    if (!container) return;

    container.innerHTML = '<div class="loading"><i class="fas fa-spinner fa-spin"></i> 加载深度学习模型...</div>';

    try {
        // 并行加载多个 API
        const [predRes, rlRes, sentimentRes, reportRes] = await Promise.all([
            apiGet(`${API_BASE}/api/dl/predict/${code}`),
            apiGet(`${API_BASE}/api/rl/trader/status`),
            apiGet(`${API_BASE}/api/sentiment/bert/${code}`),
            apiGet(`${API_BASE}/api/dl/ensemble/report`)
        ]);

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

        if (predRes && predRes.success && predRes.prediction) {
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

        if (sentimentRes && sentimentRes.sentiment) {
            const s = sentimentRes.sentiment;
            const score = s.score || 0;
            const sentimentClass = score > 0.15 ? 'positive' : score < -0.15 ? 'negative' : 'neutral';
            const sentimentText = score > 0.15 ? '正面' : score < -0.15 ? '负面' : '中性';

            dlHtml += `
                <div class="dl-sentiment-main ${sentimentClass}">
                    <div class="dl-sentiment-score">${score.toFixed(3)}</div>
                    <div class="dl-sentiment-label">${sentimentText}</div>
                </div>
                <div class="dl-sentiment-breakdown">
                    <div class="dl-sb-item positive">
                        <div class="dl-sb-value" style="color:#10b981">${(s.positive * 100).toFixed(0)}%</div>
                        <div class="dl-sb-label">正面</div>
                    </div>
                    <div class="dl-sb-item neutral">
                        <div class="dl-sb-value" style="color:#d29922">${(s.neutral * 100).toFixed(0)}%</div>
                        <div class="dl-sb-label">中性</div>
                    </div>
                    <div class="dl-sb-item negative">
                        <div class="dl-sb-value" style="color:#f85149">${(s.negative * 100).toFixed(0)}%</div>
                        <div class="dl-sb-label">负面</div>
                    </div>
                </div>
            `;
        } else {
            dlHtml += '<div class="dl-no-data">情感数据不可用</div>';
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
            dlHtml += `
                <div class="dl-arch-layer">
                    <span class="dl-arch-icon">📥</span>
                    <span>Input: Features (12 维)</span>
                </div>
                <div class="dl-arch-arrow">↓</div>
                <div class="dl-arch-layer">
                    <span class="dl-arch-icon">🔀</span>
                    <span>Transformer Encoder</span>
                    <span class="dl-arch-detail">${r.models.transformer_lstm.d_model}d, ${r.models.transformer_lstm.num_heads} heads</span>
                </div>
                <div class="dl-arch-arrow">↓</div>
                <div class="dl-arch-layer">
                    <span class="dl-arch-icon">🔁</span>
                    <span>LSTM Layer</span>
                    <span class="dl-arch-detail">hidden=${r.models.transformer_lstm.lstm_hidden}</span>
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
