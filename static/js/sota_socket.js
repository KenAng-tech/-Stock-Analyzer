// sota_socket.js — WebSocket SOTA 预测实时推送
// (2026-08-16 从 templates/index.html 内联 <script> 外置, 架构清理)

    // ═══════════════════════════════════════════════════
    // WebSocket SOTA 预测实时推送
    // ═══════════════════════════════════════════════════
    let sotaSocket = null;
    let sotaConnected = false;

    function initSOTASocket() {
        if (sotaSocket) return;  // 已初始化

        sotaSocket = io();  // 连接到 Flask-SocketIO

        sotaSocket.on('connect', function() {
            console.log('[SOTA WS] Connected');
            sotaConnected = true;
            updateSotaWsStatus('已连接');
        });

        sotaSocket.on('disconnect', function() {
            console.log('[SOTA WS] Disconnected');
            sotaConnected = false;
            updateSotaWsStatus('已断开');
        });

        sotaSocket.on('sota_prediction_update', function(data) {
            handleSOTAPredictionUpdate(data);
        });
    }

    function updateSotaWsStatus(text) {
        const el = document.getElementById('sota-ws-status');
        if (el) {
            el.textContent = 'SOTA WebSocket: ' + text;
            el.style.color = text === '已连接' ? '#10b981' : '#f59e0b';
        }
    }

    function handleSOTAPredictionUpdate(data) {
        console.log('[SOTA WS] Prediction update:', data);

        // 更新 SOTA 决策面板
        const decisionEl = document.getElementById('sota-decision');
        if (decisionEl) {
            const dir = data.ensemble_direction;
            const score = data.ensemble_score;
            const dirText = dir === 'bullish' ? '看涨' : dir === 'bearish' ? '看跌' : '中性';
            const dirClass = dir;

            // 方向突变高亮
            const changeBadge = data.direction_change
                ? '<span style="color:#f59e0b;font-size:12px;margin-left:8px;">⚠ 方向变化</span>'
                : '';

            decisionEl.innerHTML = `
                <div class="sota-decision-card">
                    <div class="sota-direction ${dirClass}">
                        <i class="fas fa-arrow-${dir === 'bullish' ? 'up' : dir === 'bearish' ? 'down' : 'right'}"></i>
                        <h4>${dirText}${changeBadge}</h4>
                    </div>
                    <div class="sota-metrics">
                        <div class="metric"><span>综合得分</span><span>${score.toFixed(4)}</span></div>
                        <div class="metric"><span>更新时间</span><span>${data.timestamp}</span></div>
                    </div>
                    <div style="margin-top:12px;font-size:12px;color:#94a3b8;">
                        <i class="fas fa-bolt" style="color:#3b82f6;"></i> 实时推送
                    </div>
                </div>`;
        }

        // 更新 SOTA 面板状态
        const statusEl = document.getElementById('sota-panel-status');
        if (statusEl) {
            statusEl.textContent = `最后更新: ${data.timestamp}`;
        }
    }

    function subscribeSOTAPredict(stockCode) {
        if (!sotaSocket || !sotaConnected) {
            initSOTASocket();
            setTimeout(() => subscribeSOTAPredict(stockCode), 1000);
            return;
        }
        sotaSocket.emit('subscribe_sota', { stock_code: stockCode });
        console.log('[SOTA WS] Subscribed to:', stockCode);
    }

    function unsubscribeSOTAPredict(stockCode) {
        if (sotaSocket && sotaConnected) {
            sotaSocket.emit('unsubscribe_sota', { stock_code: stockCode });
        }
    }

    // 页面加载时初始化 WebSocket
    function initSOTASocketOnReady() {
        initSOTASocket();
    }
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initSOTASocketOnReady);
    } else {
        initSOTASocketOnReady();
    }
