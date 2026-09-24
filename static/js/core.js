// Stock Analyzer - Core Shared Utilities
// 共享工具函数: API 请求、Loading、错误处理、格式化

const API_BASE = '';
// 使用相对路径避免 127.0.0.1 vs localhost 的跨域问题

// ── Loading Overlay ──────────────────────────────────────
let _activeApiRequests = 0;
let _globalLoadingTimer = null;

function showLoading(show) {
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) {
        overlay.classList.toggle('active', show);
    }
}

function _incApiRequests() {
    _activeApiRequests++;
    clearTimeout(_globalLoadingTimer);
    _globalLoadingTimer = setTimeout(() => {
        const overlay = document.getElementById('global-loading-overlay');
        if (overlay) overlay.classList.add('active');
    }, 300);
}

function _decApiRequests() {
    _activeApiRequests = Math.max(0, _activeApiRequests - 1);
    if (_activeApiRequests <= 0) {
        clearTimeout(_globalLoadingTimer);
        const overlay = document.getElementById('global-loading-overlay');
        if (overlay) overlay.classList.remove('active');
    }
}

// ── API 请求 ─────────────────────────────────────────────
async function apiGet(url, timeoutMs = 10000) {
    _incApiRequests();
    try {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), timeoutMs);
        const resp = await fetch(url, { signal: controller.signal });
        clearTimeout(timeout);
        if (!resp.ok) {
            const text = await resp.text();
            throw new Error(`HTTP ${resp.status}: ${text.slice(0, 100)}`);
        }
        return await resp.json();
    } catch (e) {
        console.error(`[API] ${url} 请求失败:`, e.message);
        return null;
    } finally {
        _decApiRequests();
    }
}

async function apiPost(url, body, timeoutMs = 10000) {
    _incApiRequests();
    try {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), timeoutMs);
        const resp = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
            signal: controller.signal,
        });
        clearTimeout(timeout);
        if (!resp.ok) {
            const text = await resp.text();
            throw new Error(`HTTP ${resp.status}: ${text.slice(0, 100)}`);
        }
        return await resp.json();
    } catch (e) {
        console.error(`[API] POST ${url} 失败:`, e.message);
        return null;
    } finally {
        _decApiRequests();
    }
}

// ── UI 工具 ──────────────────────────────────────────────
function showPanelLoading(containerId, text = '加载中...') {
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

function showPanelError(containerId, message, retryFn = null) {
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

function showPanelEmpty(containerId, message = '暂无数据') {
    const el = document.getElementById(containerId);
    if (el) {
        el.innerHTML = `<div class="loading"><div class="spinner-icon"></div>${message}</div>`;
    }
}

// ── 格式化工具 ────────────────────────────────────────────
function formatPercent(val) {
    if (val === null || val === undefined || isNaN(val)) return '--';
    return (val * 100).toFixed(2) + '%';
}

function formatPrice(val) {
    if (val === null || val === undefined || isNaN(val)) return '--';
    return val.toFixed(2);
}

function formatLargeNum(val) {
    if (val === null || val === undefined || isNaN(val)) return '--';
    if (val >= 1e12) return (val / 1e12).toFixed(2) + '万亿';
    if (val >= 1e8) return (val / 1e8).toFixed(2) + '亿';
    if (val >= 1e4) return (val / 1e4).toFixed(2) + '万';
    return val.toFixed(2);
}

function formatDate(ts) {
    if (!ts) return '--';
    const d = new Date(ts);
    return d.toLocaleDateString('zh-CN');
}

function formatDateTime(ts) {
    if (!ts) return '--';
    const d = new Date(ts);
    return d.toLocaleString('zh-CN');
}

// ── 颜色工具 ─────────────────────────────────────────────
function getMarketColor(val) {
    if (val === null || val === undefined || isNaN(val)) return 'var(--text-secondary)';
    return val >= 0 ? 'var(--bull)' : 'var(--bear)';
}

function getSignalColor(val) {
    if (val === null || val === undefined || isNaN(val)) return 'var(--text-secondary)';
    if (val > 0.6) return 'var(--bull)';
    if (val > 0.4) return 'var(--neutral)';
    return 'var(--bear)';
}

// ── 全局 Loading Overlay 初始化 ──────────────────────────
(function initGlobalLoading() {
    if (!document.getElementById('global-loading-overlay')) {
        const overlay = document.createElement('div');
        overlay.id = 'global-loading-overlay';
        overlay.className = 'loading-overlay';
        overlay.innerHTML = `
            <div class="spinner"></div>
            <div class="loading-text">正在加载数据...</div>
        `;
        document.body.appendChild(overlay);
    }
})();