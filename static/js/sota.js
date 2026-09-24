// Stock Analyzer - SOTA Model Panels
// SOTA 模型面板: 加载、更新、状态检查

// ── SOTA 面板加载 ───────────────────────────────────────
async function loadSOTAPanel(panelId, apiUrl, renderFn) {
    showPanelLoading(panelId, 'SOTA 模型加载中...');
    try {
        const data = await apiGet(apiUrl);
        if (data && data.success) {
            if (typeof renderFn === 'function') {
                renderFn(panelId, data.data || data);
            } else {
                renderSOTADefault(panelId, data.data || data);
            }
        } else {
            showPanelError(panelId, 'SOTA 模型数据不可用');
        }
    } catch (e) {
        showPanelError(panelId, `加载失败: ${e.message}`);
    }
}

// ── SOTA 默认渲染 ───────────────────────────────────────
function renderSOTADefault(containerId, data) {
    const el = document.getElementById(containerId);
    if (!el) return;

    if (!data || Object.keys(data).length === 0) {
        el.innerHTML = '<div class="text-secondary">暂无数据</div>';
        return;
    }

    let html = '<div class="sota-panel">';
    for (const [key, value] of Object.entries(data)) {
        const displayVal = typeof value === 'object' ? JSON.stringify(value) : value;
        html += `
            <div class="sota-row">
                <span class="sota-label">${key}</span>
                <span class="sota-value">${displayVal}</span>
            </div>`;
    }
    html += '</div>';
    el.innerHTML = html;
}

// ── SOTA 状态检查 ───────────────────────────────────────
async function checkAllSOTAStatus() {
    const endpoints = [
        { id: 'patchtst-status', url: `${API_BASE}/api/sota/patchtst/status` },
        { id: 'mamba-status', url: `${API_BASE}/api/sota/mamba/status` },
        { id: 'diffusion-status', url: `${API_BASE}/api/sota/diffusion/status` },
        { id: 'moirai-status', url: `${API_BASE}/api/sota/moirai/status` },
        { id: 'conformal-status', url: `${API_BASE}/api/sota/conformal/status` },
        { id: 'gnn-status', url: `${API_BASE}/api/sota/gnn/status` },
    ];

    const results = await Promise.allSettled(
        endpoints.map(ep =>
            apiGet(ep.url).then(data => ({ id: ep.id, data }))
        )
    );

    results.forEach(result => {
        if (result.status === 'fulfilled' && result.value) {
            updateSOTAStatusCard(result.value.id, result.value.data);
        }
    });
}

function updateSOTAStatusCard(containerId, data) {
    const el = document.getElementById(containerId);
    if (!el) return;

    const status = data?.data?.status || data?.status || 'unknown';
    const isReady = status === 'ready' || status === 'trained';

    el.innerHTML = `
        <div class="sota-status-card ${isReady ? 'ready' : 'loading'}">
            <div class="sota-status-indicator ${isReady ? 'green' : 'yellow'}"></div>
            <div class="sota-status-text">${isReady ? '就绪' : '加载中'}</div>
        </div>`;
}

// ── SOTA 训练控制 ───────────────────────────────────────
async function triggerSOTATraining(modelType) {
    const result = await apiPost(`${API_BASE}/api/sota/${modelType}/train`, {});
    if (result && result.success) {
        alert(`${modelType} 训练已启动`);
    } else {
        alert(`${modelType} 训练启动失败`);
    }
}

// ── 初始化所有 SOTA 面板 ────────────────────────────────
function initAllSOTAPanels() {
    const sotaPanels = document.querySelectorAll('[data-sota-panel]');
    sotaPanels.forEach(panel => {
        const panelId = panel.id;
        const apiUrl = panel.dataset.apiUrl;
        if (panelId && apiUrl) {
            loadSOTAPanel(panelId, apiUrl);
        }
    });
}

// ── 自动初始化 ───────────────────────────────────────────
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        setTimeout(initAllSOTAPanels, 500);
        setTimeout(checkAllSOTAStatus, 1000);
    });
} else {
    setTimeout(initAllSOTAPanels, 500);
    setTimeout(checkAllSOTAStatus, 1000);
}