/**
 * utils.js — 统一工具层
 *
 * 基于 fuyao 最佳实践:
 * - API Key 零持久化 (cookie 临时存储，不写 localStorage)
 * - 统一 fetch 包装器 (API Key + 超时 + 错误降级)
 * - 空状态渲染 (诚实呈现，不伪造数据)
 * - thscode 消歧 (模糊输入 → 标准代码)
 */

// ── API Key 管理 ──────────────────────────────────────────────

/** 当前内存中的 API Key */
let _apiKey = '';

/** 从 cookie 读取 API Key */
function _getApiKeyFromCookie() {
    const m = document.cookie.match(/stock_api_key=([^;]+)/);
    return m ? decodeURIComponent(m[1]) : '';
}

/** 设置 API Key (写入 cookie，7 天过期) */
function setApiKey(key) {
    _apiKey = key;
    const input = document.getElementById('apiKeyInput');
    if (input) input.value = key;
    if (key) {
        document.cookie = `stock_api_key=${encodeURIComponent(key)};max-age=604800;path=/;SameSite=Lax`;
    } else {
        document.cookie = 'stock_api_key=;max-age=0;path=/';
    }
}

/** 获取当前 API Key */
function getApiKey() {
    if (!_apiKey) {
        _apiKey = _getApiKeyFromCookie() || '';
    }
    return _apiKey;
}

/** 测试 API Key 连接 */
async function testApiKey() {
    const key = document.getElementById('apiKeyInput')?.value?.trim() || '';
    const status = document.getElementById('apiStatus');
    if (!status) return;

    status.textContent = '测试中...';
    status.style.color = '#64748b';

    try {
        const resp = await fetch('/api/health');
        const data = await resp.json();
        if (data.status) {
            setApiKey(key);
            status.textContent = '✓ 连接成功';
            status.style.color = '#10b981';
        } else {
            status.textContent = '✗ 服务异常';
            status.style.color = '#ef4444';
        }
    } catch (e) {
        status.textContent = `✗ ${e.message}`;
        status.style.color = '#ef4444';
    }
}

/** 清除 API Key */
function clearApiKey() {
    setApiKey('');
    const status = document.getElementById('apiStatus');
    if (status) {
        status.textContent = '已清除';
        status.style.color = '#64748b';
    }
}

// ── 统一 fetch 包装器 ──────────────────────────────────────────

/**
 * 统一 API 请求包装器
 *
 * @param {string} url - 请求 URL
 * @param {object} options - fetch options
 * @param {number} timeout - 超时毫秒数 (默认 15000)
 * @returns {Promise<object>} { ok, data?, error? }
 */
async function apiFetch(url, options = {}, timeout = 15000) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeout);

    const headers = { ...options.headers };
    const key = getApiKey();
    if (key) headers['X-API-Key'] = key;

    try {
        const resp = await fetch(url, {
            ...options,
            headers,
            signal: controller.signal,
        });
        clearTimeout(timer);

        if (!resp.ok) {
            // 尝试读取错误体
            let errorMsg = `HTTP ${resp.status}: ${resp.statusText}`;
            try {
                const errData = await resp.json();
                if (errData.error?.hint) errorMsg = errData.error.hint;
            } catch { /* ignore */ }
            return { ok: false, error: { code: 'http_error', status: resp.status, hint: errorMsg } };
        }

        // 处理非 JSON 响应 (如 204 No Content)
        const contentType = resp.headers.get('content-type') || '';
        if (contentType.includes('application/json')) {
            const data = await resp.json();
            return { ok: true, data };
        } else {
            const text = await resp.text();
            return { ok: true, data: text };
        }
    } catch (e) {
        clearTimeout(timer);
        const message = e.name === 'AbortError'
            ? `请求超时 (${timeout / 1000}s)`
            : e.message;
        return { ok: false, error: { code: e.name === 'AbortError' ? 'timeout' : 'fetch_error', hint: message } };
    }
}

// ── 空状态渲染 ────────────────────────────────────────────────

/**
 * 渲染空/错误状态
 *
 * @param {string} id - 元素 ID
 * @param {string} message - 主消息
 * @param {string} hint - 提示消息 (可选)
 */
function renderEmpty(id, message, hint) {
    const el = document.getElementById(id);
    if (!el) return;
    el.innerHTML = `
        <div style="text-align:center;padding:24px 16px;color:#64748b;">
            <div style="font-size:28px;margin-bottom:8px;opacity:0.5;">⊘</div>
            <p style="margin:0 0 4px;font-size:13px;color:#94a3b8;">${message}</p>
            ${hint ? `<p style="margin:0;font-size:11px;">${hint}</p>` : ''}
        </div>`;
}

/** 渲染加载中状态 */
function renderLoading(id) {
    const el = document.getElementById(id);
    if (!el) return;
    el.innerHTML = `
        <div style="text-align:center;padding:20px;color:#64748b;">
            <div style="display:inline-block;width:16px;height:16px;border:2px solid #334155;
                        border-top-color:#3b82f6;border-radius:50%;animation:spin 0.8s linear infinite;"></div>
            <span style="margin-left:8px;font-size:12px;">加载中...</span>
        </div>`;
}

// ── 安全设置元素文本 ──────────────────────────────────────────

/**
 * 安全地设置元素文本内容 (防 XSS)
 *
 * @param {string} id - 元素 ID
 * @param {string} text - 文本内容
 */
function i26Set(id, text) {
    const el = document.getElementById(id);
    if (el) el.textContent = text ?? '';
}

/**
 * 安全地转义 HTML (防 XSS)
 *
 * @param {string} str - 原始字符串
 * @returns {string} 转义后的字符串
 */
function i26Esc(str) {
    if (str == null) return '';
    const div = document.createElement('div');
    div.appendChild(document.createTextNode(str));
    return div.innerHTML;
}

// ── thscode 消歧 ──────────────────────────────────────────────

/**
 * 将模糊股票代码转换为标准 thscode 格式
 *
 * 规则:
 * - sh688981 → sh688981 (已有前缀)
 * - 688981 → 自动判断: 688→sh, 30→sz, 00→sz, 83/43/87→bj
 * - 300620.SZ → sz300620 (标准化)
 *
 * @param {string} input - 输入代码
 * @returns {string} 标准化代码
 */
function normalizeStockCode(input) {
    if (!input) return '';

    // 清理空格和前后缀
    let code = input.trim().toLowerCase();

    // 已有前缀: sh688981, sz300620, bj830799
    if (/^(sh|sz|bj)\d{6}$/.test(code)) {
        return code;
    }

    // 带交易所后缀: 688981.SH, 300620.SZ, 830799.BJ
    const suffixMatch = code.match(/^(\d{6})\.(sh|sz|bj)$/i);
    if (suffixMatch) {
        return `${suffixMatch[2]}${suffixMatch[1]}`;
    }

    // 纯数字: 688981
    if (/^\d{6}$/.test(code)) {
        const first = parseInt(code[0]);
        let prefix;
        if (first >= 6) prefix = 'sh';       // 6xxxxx → 上交所
        else if (first >= 3) prefix = 'sz';   // 3xxxxx, 0xxxxx → 深交所
        else if (first >= 8 || first === 4) prefix = 'bj'; // 8xxxxx, 4xxxxx → 北交所
        else prefix = 'sz';                   // 默认深交所
        return `${prefix}${code}`;
    }

    // 无前缀 6 位数字，直接加默认前缀
    if (/\d{6}/.test(code)) {
        const match = code.match(/\d{6}/);
        return normalizeStockCode(match[0]);
    }

    // 无法解析，返回原始输入
    return code;
}

/**
 * 通过 API 解析 thscode (如果后端支持)
 *
 * @param {string} input - 输入代码
 * @returns {Promise<string>} 标准化代码
 */
async function resolveStockCode(input) {
    const normalized = normalizeStockCode(input);

    // 尝试通过后端 API 解析 (如果实现)
    try {
        const resp = await apiFetch(`/api/stock/resolve?q=${encodeURIComponent(input)}`);
        if (resp.ok && resp.data?.code) {
            return resp.data.code;
        }
    } catch { /* fallback to client-side */ }

    return normalized;
}

// ── 数据溯源 ──────────────────────────────────────────────────

/**
 * 渲染数据溯源 footer
 *
 * @param {string} containerId - 容器 ID
 * @param {object} meta - { source, timestamp, adjusted }
 */
function renderDataFooter(containerId, meta) {
    const el = document.getElementById(containerId);
    if (!el) return;

    const footer = document.createElement('div');
    footer.className = 'data-footer';
    footer.style.cssText = 'display:flex;gap:12px;font-size:10px;color:#64748b;margin-top:8px;padding-top:8px;border-top:1px solid rgba(100,116,139,0.2);flex-wrap:wrap;';

    const source = meta?.source || 'AKShare';
    const timestamp = meta?.timestamp ? new Date(meta.timestamp).toLocaleString('zh-CN') : '--';
    const adjusted = meta?.adjusted || '后复权';

    footer.innerHTML = `
        <span><i class="fas fa-database" style="margin-right:2px;"></i> 来源: ${i26Esc(source)}</span>
        <span><i class="fas fa-clock" style="margin-right:2px;"></i> 更新: ${i26Esc(timestamp)}</span>
        <span><i class="fas fa-chart-bar" style="margin-right:2px;"></i> 复权: ${i26Esc(adjusted)}</span>
    `;

    // 避免重复添加
    const existing = el.querySelector('.data-footer');
    if (!existing) el.appendChild(footer);
}

// ── 数据新鲜度指示器 ──────────────────────────────────────────

/**
 * 更新全局数据新鲜度指示器
 *
 * @param {object} freshness - { status, latestDate, issues }
 */
function updateFreshnessIndicator(freshness) {
    const el = document.getElementById('freshness-indicator');
    if (!el) return;

    const dot = el.querySelector('.freshness-dot');
    const text = el.querySelector('.freshness-text');

    if (!freshness || freshness.status === 'unknown') {
        dot.style.background = '#64748b';
        text.textContent = '数据: 未获取';
        return;
    }

    if (freshness.status === 'fresh') {
        dot.style.background = '#10b981';
        text.textContent = `数据: 最新 ${freshness.latest_date || '今天'}`;
    } else if (freshness.status === 'stale') {
        dot.style.background = '#d29922';
        text.textContent = `数据: ${freshness.issues?.[0] || '部分陈旧'}`;
    } else {
        dot.style.background = '#ef4444';
        text.textContent = '数据: 异常';
    }
}

// ── 免责声明 ──────────────────────────────────────────────────

/**
 * 添加页面底部免责声明
 */
function addDisclaimer() {
    // 避免重复添加
    if (document.getElementById('site-disclaimer')) return;

    const footer = document.createElement('footer');
    footer.id = 'site-disclaimer';
    footer.style.cssText = `
        text-align:center;padding:16px;font-size:11px;color:#64748b;
        border-top:1px solid rgba(59,130,246,0.1);margin-top:24px;
        line-height:1.6;
    `;
    footer.innerHTML = `
        <div>
            <i class="fas fa-exclamation-triangle" style="color:#d29922;margin-right:4px;"></i>
            <strong>免责声明</strong>
        </div>
        <div style="margin-top:4px;">
            本系统仅供研究学习使用，不构成任何投资建议。
            模型预测存在误差，投资有风险，决策需谨慎。
        </div>
        <div style="margin-top:2px;font-size:10px;color:#475569;">
            Stock Analyzer v1.0.0 · Built with AKShare + ML Models
        </div>
    `;

    const main = document.querySelector('main');
    if (main) main.appendChild(footer);
}

// ── 健康检查 + 数据新鲜度 ─────────────────────────────────────

/**
 * 加载健康数据并更新新鲜度指示器
 *
 * @param {number} intervalMs - 刷新间隔 (默认 60000ms = 1 分钟)
 */
async function loadHealthData(intervalMs) {
    intervalMs = intervalMs || 60000;

    async function _refresh() {
        try {
            const data = await apiFetch('/api/health', {}, 10000);
            if (!data.ok || !data.data) return;

            const freshness = data.data?.data_freshness;
            if (freshness) {
                updateFreshnessIndicator(freshness);
            }

            // 更新在线状态
            const statusDot = document.querySelector('.status-dot');
            const statusText = document.querySelector('.status-text');
            if (statusDot && statusText) {
                const status = data.data?.status || 'unknown';
                statusDot.className = `status-dot ${status === 'healthy' ? 'online' : status === 'degraded' ? 'warning' : 'error'}`;
                statusText.textContent = status === 'healthy' ? '在线' : status === 'degraded' ? '部分降级' : '异常';
            }
        } catch (e) {
            console.warn('[Health] 健康检查失败:', e.message);
        }
    }

    // 立即执行一次
    _refresh();

    // 定时刷新
    if (intervalMs > 0) {
        setInterval(_refresh, intervalMs);
    }
}

// ── 初始化 ────────────────────────────────────────────────────

/** 页面加载时自动恢复 API Key */
(function initUtils() {
    const cookieKey = _getApiKeyFromCookie();
    if (cookieKey) {
        _apiKey = cookieKey;
        const input = document.getElementById('apiKeyInput');
        if (input) input.value = cookieKey;
    }

    // Phase 3: 自动加载健康数据
    if (typeof loadHealthData === 'function') {
        loadHealthData(60000);  // 每 60 秒刷新
    }
})();

// ── 预测评估面板 (Phase 3) ────────────────────────────────────

/**
 * 加载预测评估面板数据
 */
async function loadPredictionEval() {
    // 准确率
    const accEl = document.getElementById('pred-eval-accuracy');
    if (accEl) renderLoading('pred-eval-accuracy');

    const accResp = await apiFetch('/api/fusion/multi-modal/accuracy');
    if (accResp.ok && accResp.data) {
        const d = accResp.data;
        if (accEl) {
            if (d.n_records === 0) {
                renderEmpty('pred-eval-accuracy', '尚无评估数据', '需等待实际收益反馈后出现');
            } else {
                const acc = (d.accuracy * 100).toFixed(1);
                const trend = d.trend === 'improving' ? '↑ 改善中' : d.trend === 'declining' ? '↓ 下降中' : '→ 稳定';
                const trendColor = d.trend === 'improving' ? '#10b981' : d.trend === 'declining' ? '#ef4444' : '#d29922';
                accEl.innerHTML = `
                    <div style="text-align:center;padding:12px;">
                        <div style="font-size:32px;font-weight:700;color:${d.accuracy > 0.55 ? '#10b981' : d.accuracy < 0.45 ? '#ef4444' : '#d29922'};">${acc}%</div>
                        <div style="font-size:11px;color:#64748b;margin-top:4px;">方向准确率 (${d.n_records} 条记录)</div>
                        <div style="font-size:11px;color:${trendColor};margin-top:4px;">${trend}</div>
                        ${d.recent_20_accuracy !== null ? `<div style="font-size:10px;color:#94a3b8;margin-top:4px;">近 20 条: ${(d.recent_20_accuracy * 100).toFixed(1)}%</div>` : ''}
                    </div>`;
            }
        }
    } else {
        if (accEl) renderEmpty('pred-eval-accuracy', accResp.error?.hint || '加载失败');
    }

    // Brier Score
    const brierEl = document.getElementById('pred-eval-brier');
    if (brierEl) renderLoading('pred-eval-brier');
    if (accResp.ok && accResp.data) {
        const d = accResp.data;
        if (brierEl) {
            if (d.n_records === 0) {
                renderEmpty('pred-eval-brier', '尚无评估数据');
            } else {
                const brier = d.avg_brier || 0;
                const brierColor = brier < 0.15 ? '#10b981' : brier < 0.25 ? '#d29922' : '#ef4444';
                brierEl.innerHTML = `
                    <div style="text-align:center;padding:12px;">
                        <div style="font-size:28px;font-weight:700;color:${brierColor};">${brier.toFixed(4)}</div>
                        <div style="font-size:11px;color:#64748b;margin-top:4px;">平均 Brier Score (越低越好)</div>
                        <div style="font-size:10px;color:#94a3b8;margin-top:4px;">0=完美, 1=最差</div>
                    </div>`;
            }
        }
    } else {
        if (brierEl) renderEmpty('pred-eval-brier', '加载失败');
    }

    // 漂移告警
    const driftEl = document.getElementById('pred-eval-drift');
    if (driftEl) renderLoading('pred-eval-drift');
    if (accResp.ok && accResp.data) {
        const d = accResp.data;
        if (driftEl) {
            if (d.drift_alert) {
                driftEl.innerHTML = `
                    <div style="text-align:center;padding:12px;">
                        <div style="font-size:24px;">⚠️</div>
                        <div style="font-size:13px;color:#ef4444;margin-top:4px;font-weight:600;">检测到预测准确率漂移!</div>
                        <div style="font-size:11px;color:#94a3b8;margin-top:4px;">准确率 < 45%, 建议检查模型或重新训练</div>
                    </div>`;
            } else if (d.n_records >= 50) {
                driftEl.innerHTML = `
                    <div style="text-align:center;padding:12px;">
                        <div style="font-size:24px;">✓</div>
                        <div style="font-size:13px;color:#10b981;margin-top:4px;font-weight:600;">无异常漂移</div>
                        <div style="font-size:11px;color:#94a3b8;margin-top:4px;">准确率在可接受范围内</div>
                    </div>`;
            } else {
                driftEl.innerHTML = `
                    <div style="text-align:center;padding:12px;">
                        <div style="font-size:24px;">⏳</div>
                        <div style="font-size:13px;color:#64748b;margin-top:4px;">数据不足</div>
                        <div style="font-size:11px;color:#94a3b8;margin-top:4px;">需要至少 50 条记录进行漂移检测</div>
                    </div>`;
            }
        }
    } else {
        if (driftEl) renderEmpty('pred-eval-drift', '加载失败');
    }
}

// ── 因子衰减面板 (Phase 3) ────────────────────────────────────

/**
 * 加载因子衰减状态
 */
async function loadFactorDecayStatus() {
    // IC 趋势
    const icEl = document.getElementById('factor-decay-ic-trend');
    if (icEl) renderLoading('factor-decay-ic-trend');

    const icResp = await apiFetch('/api/factors/alpha/decay/status');
    // 09-22 对齐: 端点真实键 = scheduler.icir_ranking / alerts / n_records
    // (旧版读 recent_ic/avg_ic/n_factors/warnings = 端点从不返回 → 面板恒空 9 天)
    const d = (icResp.ok && icResp.data) ? icResp.data : null;
    const rank = (d && d.scheduler && d.scheduler.icir_ranking) || [];
    const alerts = (d && d.alerts) || [];

    const icMean = rank.length ? rank.reduce((s, r) => s + (r.icir || 0), 0) / rank.length : null;
    if (icEl) {
        if (d && d.success && icMean !== null) {
            icEl.innerHTML = `
                <div style="text-align:center;padding:12px;">
                    <div style="font-size:32px;font-weight:700;color:${icMean > 0.05 ? '#10b981' : icMean < 0 ? '#ef4444' : '#d29922'};">${icMean.toFixed(4)}</div>
                    <div style="font-size:11px;color:#64748b;margin-top:4px;">7 因子 ICIR 均值 (20d 滚动)</div>
                    <div style="font-size:11px;color:#94a3b8;margin-top:2px;">${icMean > 0.05 ? '因子有效' : icMean < 0 ? '因子链负相关' : '因子中性'}</div>
                </div>`;
        } else {
            icEl.innerHTML = `<div style="text-align:center;padding:12px;color:#94a3b8;font-size:12px;">${d && d.success ? '无 IC 样本 (&lt;20 条/20d 窗)' : (d && d.error ? i26Esc(String(d.error).slice(0, 60)) : '加载失败')}</div>`;
        }
    }

    // 衰减警告 (alerts = 因子级 IC 衰减判定, n≥20 起判)
    const warnEl = document.getElementById('factor-decay-warnings');
    if (warnEl) renderLoading('factor-decay-warnings');
    if (warnEl) {
        if (!d) warnEl.innerHTML = '<div style="padding:12px;color:#94a3b8;font-size:12px;">加载失败</div>';
        else if (!alerts.length) warnEl.innerHTML = `<div style="text-align:center;padding:12px;color:#10b981;font-size:13px;">✓ 无衰减警告 (${d.n_records || 0} 条监控样本)</div>`;
        else warnEl.innerHTML = alerts.slice(0, 5).map(a =>
            `<div style="padding:4px 0;border-bottom:1px solid rgba(100,116,139,0.2);font-size:11px;color:#f87171;">
                        ⚠ ${i26Esc(a.factor || '?')} IC=${(a.ic_mean ?? 0).toFixed(3)}
                    </div>`
        ).join('');
    }

    // 因子健康度
    const healthEl = document.getElementById('factor-decay-health');
    if (healthEl) renderLoading('factor-decay-health');
    if (icResp.ok && icResp.data) {
        // 09-22 对齐: 健康度 = 7 因子中未衰减数 (alerts 判据同 ICIR 门)
        const d = icResp.data;
        const total = ((d && d.scheduler && d.scheduler.icir_ranking) || []).length || 7;
        const healthy = total - ((d && d.alerts) || []).length;
        const pct = total > 0 ? (healthy / total * 100) : 0;
        const color = pct > 80 ? '#10b981' : pct > 50 ? '#d29922' : '#ef4444';
        if (healthEl) {
            healthEl.innerHTML = `
                <div style="text-align:center;padding:12px;">
                    <div style="font-size:28px;font-weight:700;color:${color};">${pct.toFixed(0)}%</div>
                    <div style="font-size:11px;color:#64748b;margin-top:4px;">${healthy}/${total} 因子健康</div>
                    <div style="margin-top:8px;background:rgba(30,41,59,0.8);border-radius:4px;height:6px;overflow:hidden;">
                        <div style="background:${color};height:100%;width:${pct}%;transition:width 0.3s;"></div>
                    </div>
                </div>`;
        }
    } else if (healthEl) {
        renderEmpty('factor-decay-health', '加载失败');
    }
}
