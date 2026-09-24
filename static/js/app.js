// Stock Analyzer - Main JavaScript

// API_BASE 已由 core.js 定义，此处复用
let currentReportFile = '';

// Phase 2: API Key 请求头注入
function _apiHeaders(extra = {}) {
    const headers = { ...extra };
    const key = typeof getApiKey === 'function' ? getApiKey() : '';
    if (key) headers['X-API-Key'] = key;
    return headers;
}

// Section Navigation
function showSection(sectionId) {
    document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
    document.querySelectorAll('.nav-link').forEach(n => n.classList.remove('active'));

    const section = document.getElementById(sectionId);
    if (section) {
        section.classList.add('active');
    }

    // Find and activate the correct nav link
    const navLinks = document.querySelectorAll('.nav-link');
    navLinks.forEach(link => {
        const onclick = link.getAttribute('onclick') || '';
        if (onclick.includes(sectionId)) {
            link.classList.add('active');
        }
    });
}

// Tab Navigation
function showTab(tabId) {
    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    
    document.getElementById('tab-' + tabId).classList.add('active');
    event.target.classList.add('active');
}

// DOM 安全访问工具
function $(id) {
    const el = document.getElementById(id);
    if (!el) console.warn(`[DOM] Element not found: #${id}`);
    return el;
}

// Show Toast Notification
function showToast(message, type = 'info') {
    const colors = {
        success: '#10b981',
        error: '#ef4444',
        warning: '#f59e0b',
        info: '#3b82f6',
    };
    const color = colors[type] || colors.info;

    // 移除已有 toast
    const existing = document.getElementById('toastNotification');
    if (existing) existing.remove();

    const toast = document.createElement('div');
    toast.id = 'toastNotification';
    toast.style.cssText = `
        position: fixed; top: 20px; right: 20px; z-index: 10000;
        background: ${color}; color: #fff; padding: 14px 24px;
        border-radius: 8px; font-size: 14px; font-weight: 600;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        transform: translateX(120%); transition: transform 0.3s ease;
        max-width: 400px;
    `;
    toast.textContent = message;
    document.body.appendChild(toast);

    // 滑入动画
    requestAnimationFrame(() => {
        toast.style.transform = 'translateX(0)';
    });

    // 3 秒后自动消失
    setTimeout(() => {
        toast.style.transform = 'translateX(120%)';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// Show/Hide Loading
function showLoading(show) {
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) {
        overlay.classList.toggle('active', show);
        // 清除可能残留的 inline display 样式 (loadSOTADashboard 设置的 display: flex/none)
        overlay.style.display = show ? 'flex' : 'none';
    }
}

// Refresh Data
function refreshData() {
    const stockCode = document.getElementById('stockCode').value;
    fetchStockData(stockCode);
}

// Fetch Stock Data
async function fetchStockData(stockCode) {
    showLoading(true);
    try {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 20000); // 20s 超时
        const response = await fetch(`${API_BASE}/api/stock/${stockCode}`, {
            signal: controller.signal,
            headers: _apiHeaders(),
        });
        clearTimeout(timeout);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        const result = await response.json();

        if (result.success && result.data) {
            updateStockDisplay(result.data);
        } else {
            console.error('[fetchStockData] API 返回失败:', result.error || '无 data 字段');
        }
    } catch (error) {
        console.error('[fetchStockData] 请求失败:', error.name === 'AbortError' ? '请求超时 (20s)' : error.message);
    } finally {
        showLoading(false);
    }
}

// 仪表盘数据加载 — 在 app.js 中定义确保 initOnReady 可用
async function loadDashboardData() {
    const stockCode = document.getElementById('stockCode')?.value || 'sz300620';
    console.log('[loadDashboardData] 开始加载，股票代码:', stockCode);
    async function _load(timeoutMs = 20000) {
        try {
            const url = `${API_BASE}/api/stock/${stockCode}`;
            console.log('[loadDashboardData] 请求 URL:', url);
            const controller = new AbortController();
            const timeout = setTimeout(() => controller.abort(), timeoutMs);
            const response = await fetch(url, {
                signal: controller.signal,
                headers: _apiHeaders(),
            });
            clearTimeout(timeout);
            console.log('[loadDashboardData] 响应状态:', response.status);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const result = await response.json();
            console.log('[loadDashboardData] 响应数据:', result.success ? 'success' : 'failed');
            if (result.success && result.data) {
                console.log('[loadDashboardData] 调用 updateStockDisplay');
                updateStockDisplay(result.data);
                console.log('[loadDashboardData] ✅ 更新完成');
                return true;
            }
            return false;
        } catch (e) {
            console.error('[loadDashboardData] _load 失败:', e.message);
            return false;
        }
    }
    // 首次加载
    if (await _load(20000)) return;
    console.error('[loadDashboardData] 首次加载失败，1s 后重试...');
    // 首次请求可能因服务器冷启动超时，1s 后重试（缓存已预热）
    await new Promise(r => setTimeout(r, 1000));
    if (await _load(15000)) return;
    console.error('[loadDashboardData] 重试也失败');
}

// Analyze Stock
async function analyzeStock() {
    console.log('analyzeStock called');
    const stockCode = document.getElementById('stockCode').value;
    const costBasis = document.getElementById('costBasis').value;
    const industry = document.getElementById('industry').value;
    
    console.log('Stock code:', stockCode, 'Cost basis:', costBasis);
    
    showLoading(true);
    try {
        const url = `${API_BASE}/api/analyze/${stockCode}?cost_basis=${costBasis}`;
        console.log('Fetching:', url);
        
        const response = await fetch(url, { headers: _apiHeaders() });
        console.log('Response status:', response.status);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        console.log('Response data:', result);
        
        if (result.success) {
            console.log('Analysis success, updating UI...');

            // Update hero stock info
            const basicInfo = result.analysis.basic_info;
            const heroNameEl = document.getElementById('heroStockName');
            const heroCodeEl = document.getElementById('heroStockCode');
            if (heroNameEl) heroNameEl.textContent = basicInfo.name || '光库科技';
            if (heroCodeEl) heroCodeEl.textContent = `${basicInfo.code} · ${industry}`;

            // Add profit_analysis if missing
            if (!result.analysis.profit_analysis) {
                const price = parseFloat(basicInfo.price) || 0;
                const cost = parseFloat(basicInfo.cost_basis) || 0;
                result.analysis.profit_analysis = {
                    cost: cost,
                    current: price,
                    profit: price - cost,
                    profit_pct: ((price - cost) / cost * 100),
                    status: price > cost ? '✅ 大幅盈利' : '❌ 亏损'
                };
            }
            
            try {
                updateDashboard(result.analysis);
                console.log('Dashboard updated');
            } catch (e) {
                console.error('Error updating dashboard:', e);
            }
            
            try {
                console.log('ATR data before update:', JSON.stringify(result.atr));
                updateAnalysis(result.analysis, result.kline_signals, result.atr);
                console.log('Analysis updated');
            } catch (e) {
                console.error('Error updating analysis:', e);
            }
            
            try {
                updateReport(result.report);
                console.log('Report updated');
            } catch (e) {
                console.error('Error updating report:', e);
            }
            
            currentReportFile = result.report_filename || result.report_file;
            console.log('Analysis completed successfully');
            console.log('Report length:', result.report ? result.report.length : 0);
            
            // Switch to analysis section (show analysis, then scroll to report)
            try {
                document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
                const analysisSection = document.getElementById('analysis');
                if (analysisSection) {
                    analysisSection.classList.add('active');
                    console.log('Switched to analysis section');
                } else {
                    console.error('Analysis section not found!');
                }
            } catch (e) {
                console.error('Error switching section:', e);
            }

            // 显示完成通知
            showToast('✅ 分析完成！已切换到分析面板', 'success');

            // Scroll to report section after brief delay
            setTimeout(() => {
                const reportSection = document.getElementById('report');
                if (reportSection) {
                    reportSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }
            }, 500);
        } else {
            console.error('Analysis failed:', result);
            alert('分析失败: ' + (result.error || 'Unknown error'));
        }
    } catch (error) {
        console.error('Error analyzing stock:', error);
        console.error('Response:', response);
        alert('分析失败: ' + error.message);
    } finally {
        showLoading(false);
        console.log('Loading hidden');
    }
}

// Update Dashboard Display
function updateStockDisplay(data) {
    console.log('[updateStockDisplay] 开始更新，data:', data);
    try {
        if (!data) {
            console.error('[updateStockDisplay] Invalid stock data:', data);
            return;
        }

        const price = parseFloat(data.price) || 0;
        const el1 = document.getElementById('currentPrice');
        const el2 = document.getElementById('heroPrice');
        console.log('[updateStockDisplay] currentPrice el:', el1 ? 'found' : 'NULL');
        if (el1) el1.textContent = price.toFixed(2);
        if (el2) el2.textContent = price.toFixed(2);

        const changePct = parseFloat(data.change_pct) || 0;
        const changeClass = changePct >= 0 ? 'positive' : 'negative';
        const changeText = changePct >= 0 ? `+${changePct.toFixed(2)}%` : `${changePct.toFixed(2)}%`;
        const priceChangeEl = document.getElementById('priceChange');
        if (priceChangeEl) {
            priceChangeEl.textContent = changeText;
            priceChangeEl.className = 'card-change ' + changeClass;
        }

        const heroChangeEl = document.getElementById('heroChange');
        if (heroChangeEl) {
            heroChangeEl.textContent = changeText;
            heroChangeEl.className = 'hero-change ' + changeClass;
        }

        const volEl = document.getElementById('volume');
        if (volEl) volEl.textContent = (parseFloat(data.volume) || 0).toLocaleString();
        const turnEl = document.getElementById('turnover');
        if (turnEl) turnEl.textContent = (parseFloat(data.turnover) || 0).toFixed(2);
        const openEl = document.getElementById('openPrice');
        if (openEl) openEl.textContent = (parseFloat(data.open) || 0).toFixed(2);
        const highEl = document.getElementById('highPrice');
        if (highEl) highEl.textContent = (parseFloat(data.high) || 0).toFixed(2);
        const lowEl = document.getElementById('lowPrice');
        if (lowEl) lowEl.textContent = (parseFloat(data.low) || 0).toFixed(2);
        const peEl = document.getElementById('peRatio');
        if (peEl) peEl.textContent = (parseFloat(data.pe) || 0).toFixed(2) + ' 倍';
        const mcEl = document.getElementById('marketCap');
        if (mcEl) mcEl.textContent = (parseFloat(data.market_cap) || 0).toFixed(2) + ' 亿元';
        const yrEl = document.getElementById('yearRange');
        if (yrEl) yrEl.textContent = `${(parseFloat(data.year_high) || 0).toFixed(2)} / ${(parseFloat(data.year_low) || 0).toFixed(2)}`;

        console.log('[updateStockDisplay] ✅ 更新完成');
    } catch (error) {
        console.error('[updateStockDisplay] 更新失败:', error);
    }
}

// Update Dashboard with Analysis
function updateDashboard(analysis) {
    try {
        if (!analysis || !analysis.profit_analysis) {
            console.error('Invalid analysis data:', analysis);
            return;
        }

        const profit = analysis.profit_analysis;
        const current = parseFloat(profit.current) || 0;
        const cost = parseFloat(profit.cost) || 0;
        const profitPct = parseFloat(profit.profit_pct) || 0;
        const profitAmt = parseFloat(profit.profit) || 0;

        // Update hero section
        document.getElementById('heroPrice').textContent = current.toFixed(2);
        const changeClass = profitPct >= 0 ? 'positive' : 'negative';
        const changeText = profitPct >= 0 ? `+${profitPct.toFixed(2)}%` : `${profitPct.toFixed(2)}%`;
        document.getElementById('heroChange').textContent = changeText;
        document.getElementById('heroChange').className = 'hero-change ' + changeClass;

        // Update overview cards
        document.getElementById('currentPrice').textContent = current.toFixed(2);
        const priceChangeEl = document.getElementById('priceChange');
        priceChangeEl.textContent = changeText;
        priceChangeEl.className = 'card-change ' + changeClass;

        document.getElementById('profitAmount').textContent = profitAmt.toFixed(2) + ' 元';
        document.getElementById('profitPercent').textContent = profitPct.toFixed(2) + '%';
        document.getElementById('profitPercent').className = 'card-change ' + changeClass;

        // Update stats group
        document.getElementById('costBasisDisplay').textContent = cost.toFixed(2) + ' 元';
        document.getElementById('profitPctDisplay').textContent = profitPct.toFixed(2) + '%';
        document.getElementById('profitPctDisplay').className = 'stat-value ' + (profitPct >= 0 ? 'positive' : 'negative');

        if (analysis.fund_flow && analysis.fund_flow.volume_analysis) {
            document.getElementById('volume').textContent = (parseFloat(analysis.fund_flow.volume_analysis.volume) || 0).toLocaleString();
            document.getElementById('turnover').textContent = (parseFloat(analysis.fund_flow.volume_analysis.turnover) || 0).toFixed(2);
        }

        // 估值指标: 使用基本面数据 (如果可用), 否则保留 updateStockDisplay 设置的实时行情数据
        if (analysis.fundamental && analysis.fundamental.valuation) {
            const pe = parseFloat(analysis.fundamental.valuation.pe);
            const mc = parseFloat(analysis.fundamental.valuation.market_cap);
            if (!isNaN(pe) && pe > 0) {
                document.getElementById('peRatio').textContent = pe.toFixed(2) + ' 倍';
            }
            if (!isNaN(mc) && mc > 0) {
                document.getElementById('marketCap').textContent = mc.toFixed(2) + ' 亿元';
            }
        }
        // 注意: openPrice/highPrice/lowPrice/yearRange 由 updateStockDisplay() 从腾讯实时行情正确设置
        // 此处不再覆盖, 避免用 MA5/支撑阻力位错误替换真实价格数据
    } catch (error) {
        console.error('Error updating dashboard:', error);
        alert('更新仪表盘失败: ' + error.message);
    }

    // Phase 2: 数据溯源 footer (添加到 dashboard 容器底部)
    try {
        const dashboardEl = document.getElementById('dashboard');
        if (dashboardEl && typeof renderDataFooter === 'function') {
            renderDataFooter('dashboard', {
                source: analysis?.basic_info?.data_source || 'AKShare',
                timestamp: new Date().toISOString(),
                adjusted: '后复权',
            });
        }
    } catch (e) { /* non-critical */ }
}

// Update Analysis Section
function updateAnalysis(analysis, klineSignals, atrData) {
    console.log('updateAnalysis called with:', {analysis: !!analysis, klineSignals: !!klineSignals});
    try {
        if (!analysis) {
            console.error('No analysis data');
            return;
        }

        // Fundamental - with null checks
        if (analysis.fundamental && analysis.fundamental.valuation) {
            const val = analysis.fundamental.valuation;
            const fundPe = $('fundPe'); if (fundPe) fundPe.textContent = (parseFloat(val.pe) || 0).toFixed(2) + ' 倍';
            const fundPeStd = $('fundPeStandard'); if (fundPeStd) fundPeStd.textContent = (parseFloat(val.pe_standard) || 0).toFixed(2) + ' 倍';
            const fundPeLvl = $('fundPeLevel'); if (fundPeLvl) {
                fundPeLvl.textContent = val.level || '--';
                const levelClass = val.level === '极高' ? 'high' : val.level === '偏高' ? 'medium' : 'low';
                fundPeLvl.className = 'fund-level ' + levelClass;  // 保留 fund-level 确保文字颜色正确
            }
            const fundMktCap = $('fundMarketCap'); if (fundMktCap) fundMktCap.textContent = (parseFloat(val.market_cap) || 0).toFixed(2) + ' 亿元';
            const fundCircCap = $('fundCirculatingCap'); if (fundCircCap) fundCircCap.textContent = (parseFloat(val.circulating_cap) || 0).toFixed(2) + ' 亿元';
        }

        if (analysis.fundamental && analysis.fundamental.financial_health) {
            const fh = analysis.fundamental.financial_health;
            const fundRevGrowth = $('fundRevenueGrowth'); if (fundRevGrowth) fundRevGrowth.textContent = fh.revenue_growth || '--';
            const fundProfitGrowth = $('fundProfitGrowth'); if (fundProfitGrowth) fundProfitGrowth.textContent = fh.profit_growth || '--';
            const fundGrossMargin = $('fundGrossMargin'); if (fundGrossMargin) fundGrossMargin.textContent = fh.gross_margin || '--';
        }

        // Technical - with null checks
        if (analysis.technical) {
            if (analysis.technical.kline) {
                const tkp = $('techKlinePattern'); if (tkp) tkp.textContent = analysis.technical.kline.pattern || '--';
                const ta = $('techAmplitude'); if (ta) ta.textContent = '振幅: ' + (parseFloat(analysis.technical.kline.amplitude) || 0).toFixed(2) + '%';
            }

            if (analysis.technical.moving_averages) {
                const ma = analysis.technical.moving_averages;
                const ids = ['ma5','ma10','ma20','ma60','ma120','ma250'];
                ids.forEach((id, i) => {
                    const el = $(id);
                    if (el && ma[id]) el.textContent = (parseFloat(ma[id]) || 0).toFixed(2);
                });
            }

            // Support/Resistance
            const supportEl = document.getElementById('supportLevels');
            const resistanceEl = document.getElementById('resistanceLevels');
            if (supportEl) supportEl.innerHTML = '';
            if (resistanceEl) resistanceEl.innerHTML = '';

            if (analysis.technical.support_resistance) {
                if (analysis.technical.support_resistance.supports) {
                    analysis.technical.support_resistance.supports.forEach(s => {
                        const price = parseFloat(s.price) || parseFloat(s.level) || 0;
                        const desc = s.desc || s.level || '';
                        supportEl.innerHTML += `<div class="sr-item"><span class="sr-level">${price.toFixed(0)}元</span><span class="sr-desc">${desc}</span></div>`;
                    });
                }
                if (analysis.technical.support_resistance.resistances) {
                    analysis.technical.support_resistance.resistances.forEach(r => {
                        const price = parseFloat(r.price) || parseFloat(r.level) || 0;
                        const desc = r.desc || r.level || '';
                        resistanceEl.innerHTML += `<div class="sr-item"><span class="sr-level">${price.toFixed(0)}元</span><span class="sr-desc">${desc}</span></div>`;
                    });
                }
            }
        }

        // Fund Flow - with null checks
        if (analysis.fund_flow) {
            if (analysis.fund_flow.main_flow) {
                const mf = analysis.fund_flow.main_flow;
                document.getElementById('flowOuter').textContent = (mf.outer_disk || 0).toLocaleString();
                document.getElementById('flowInner').textContent = (mf.inner_disk || 0).toLocaleString();
                document.getElementById('flowRatio').textContent = mf.ratio || '--';
                document.getElementById('flowDirection').textContent = mf.direction || '--';
            }

            // Trade count
            if (analysis.fund_flow.trade_count) {
                const tc = analysis.fund_flow.trade_count;
                document.getElementById('tradeCount').textContent = (parseFloat(tc.trade_count) || 0).toLocaleString();
                document.getElementById('avgAmount').textContent = tc.avg_amount_per_lot || '--';
                const tradeLevelEl = document.getElementById('tradeLevel');
                if (tradeLevelEl) tradeLevelEl.textContent = tc.level || '--';
            }

            // Flow speed
            if (analysis.fund_flow.flow_speed) {
                const flowSpeedEl = document.getElementById('flowSpeed');
                if (flowSpeedEl) flowSpeedEl.textContent = parseFloat(analysis.fund_flow.flow_speed.speed) || '--';
            }

            // Pressure index
            if (analysis.fund_flow.pressure_index) {
                const pi = analysis.fund_flow.pressure_index;
                const pressureIndex = parseFloat(pi.index) || 1;
                const pressureLevel = pi.level || '--';
                
                // 更新 Gauge 文本
                const pressureValueText = document.getElementById('pressureValueText');
                if (pressureValueText) pressureValueText.textContent = pressureIndex.toFixed(2);
                
                const pressureLevelText = document.getElementById('pressureLevelText');
                if (pressureLevelText) pressureLevelText.textContent = pressureLevel;
                
                // 计算压力百分比 (0.5-5.0 范围映射到 0-100%)
                const pressurePercent = Math.min(100, Math.max(0, (pressureIndex - 0.5) / 4.5 * 100));
                
                // 更新 Gauge 弧
                const pressureArc = document.getElementById('pressureArc');
                if (pressureArc) {
                    const maxOffset = 251.2; // 弧长
                    const offset = maxOffset - (maxOffset * pressurePercent / 100);
                    pressureArc.style.strokeDashoffset = offset;
                    
                    // 设置方向类
                    const direction = pressureIndex > 1 ? 'buy' : 'sell';
                    pressureArc.className.baseVal = 'gauge-active ' + direction;
                    
                    // 设置颜色
                    if (direction === 'buy') {
                        pressureArc.style.stroke = '#10b981';
                    } else {
                        pressureArc.style.stroke = '#f87171';
                    }
                }
                
                // 添加脉冲动画效果
                if (pressureArc) {
                    pressureArc.style.animation = 'pulse 2s ease-in-out infinite';
                }
            }
            
            // 渲染买/卖比率
            if (analysis.fund_flow.main_flow && analysis.fund_flow.main_flow.buy_sell_ratio) {
                const bsr = analysis.fund_flow.main_flow.buy_sell_ratio;
                const buyPercent = bsr.buy_percentage || 50;
                const sellPercent = bsr.sell_percentage || 50;
                
                // 更新比率条
                const buySegment = document.getElementById('buySegment');
                const sellSegment = document.getElementById('sellSegment');
                if (buySegment) buySegment.style.width = buyPercent + '%';
                if (sellSegment) sellSegment.style.width = sellPercent + '%';
                
                // 更新比率值
                const buyValue = document.getElementById('buyValue');
                const sellValue = document.getElementById('sellValue');
                if (buyValue) buyValue.textContent = buyPercent + '%';
                if (sellValue) sellValue.textContent = sellPercent + '%';
                
                // 更新主导方向
                const ratioDominant = document.getElementById('ratioDominant');
                if (ratioDominant) ratioDominant.textContent = bsr.dominant || '平衡';
                
                // 更新强度
                const ratioStrength = document.getElementById('ratioStrength');
                if (ratioStrength) {
                    ratioStrength.textContent = bsr.strength || '中';
                    ratioStrength.className = 'ratio-strength ' + (bsr.strength === '强' ? 'strong' : bsr.strength === '中' ? 'medium' : 'weak');
                }
            }

            // Trade distribution
            if (analysis.fund_flow.trade_distribution && analysis.fund_flow.trade_distribution.distribution) {
                const dist = analysis.fund_flow.trade_distribution.distribution;
                const largeRatio = dist.large ? dist.large.ratio || 0 : 0;
                const mediumRatio = dist.medium ? dist.medium.ratio || 0 : 0;
                const smallRatio = dist.small ? dist.small.ratio || 0 : 0;

                // 大单
                const largeDistEl = $('largeDist');
                if (largeDistEl) {
                    largeDistEl.style.width = largeRatio + '%';
                    const largeFill = largeDistEl.querySelector('.progress-fill');
                    if (largeFill) largeFill.style.width = largeRatio + '%';
                }
                const largeValue = $('largeValue'); if (largeValue) largeValue.textContent = largeRatio + '%';

                // 中单
                const mediumDistEl = $('mediumDist');
                if (mediumDistEl) {
                    mediumDistEl.style.width = mediumRatio + '%';
                    const mediumFill = mediumDistEl.querySelector('.progress-fill');
                    if (mediumFill) mediumFill.style.width = mediumRatio + '%';
                }
                const mediumValue = $('mediumValue'); if (mediumValue) mediumValue.textContent = mediumRatio + '%';

                // 小单
                const smallDistEl = $('smallDist');
                if (smallDistEl) {
                    smallDistEl.style.width = smallRatio + '%';
                    const smallFill = smallDistEl.querySelector('.progress-fill');
                    if (smallFill) smallFill.style.width = smallRatio + '%';
                }
                const smallValue = $('smallValue'); if (smallValue) smallValue.textContent = smallRatio + '%';
                
                // 添加动画效果
                const allBars = document.querySelectorAll('.progress-fill');
                allBars.forEach(bar => {
                    bar.style.transition = 'width 0.8s cubic-bezier(0.4, 0, 0.2, 1)';
                });
            }

            // Chip Distribution
            if (analysis.fund_flow.chip_distribution) {
                const chipProfit = analysis.fund_flow.chip_distribution.profit_ratio || 0;
                const chipTrapped = analysis.fund_flow.chip_distribution.trapped_ratio || 0;
                const cp = $('chipProfit'); if (cp) cp.style.width = chipProfit + '%';
                const ct = $('chipTrapped'); if (ct) ct.style.width = chipTrapped + '%';
                const cpv = $('chipProfitValue'); if (cpv) cpv.textContent = chipProfit + '%';
                const ctv = $('chipTrappedValue'); if (ctv) ctv.textContent = chipTrapped + '%';
            }
        }

        // Prediction
        if (analysis.prediction) {
            if (analysis.prediction.model) {
                const cs = $('compositeScore'); if (cs) cs.textContent = parseFloat(analysis.prediction.model.composite) || '--';
            }
            const wt = $('weightedTarget'); if (wt) wt.textContent = (parseFloat(analysis.prediction.weighted_target) || 0) + ' 元';
            const wu = $('wtUpside'); if (wu) wu.textContent = '+' + (parseFloat(analysis.prediction.upside_space) || 0) + '%';

            // Scenarios
            const scenarios = analysis.prediction.scenarios;
            if (scenarios && scenarios[0]) { const ot = $('optimisticTarget'); if (ot) ot.textContent = scenarios[0].target_range || '--'; }
            if (scenarios && scenarios[1]) { const nt = $('neutralTarget'); if (nt) nt.textContent = scenarios[1].target_range || '--'; }
            if (scenarios && scenarios[2]) { const pt = $('pessimisticTarget'); if (pt) pt.textContent = scenarios[2].target_range || '--'; }
        }

        // ML Model Details (Stacking Ensemble)
        updateMLDetails(analysis);

        // SOTA 模型详情
        updateSOTADetails(analysis);

        // K线信号分析
        updateKlineSignals(klineSignals);

        // ATR 动态止损/止盈
        updateATR(atrData);

    } catch (error) {
        console.error('Error updating analysis:', error);
        alert('更新分析失败: ' + error.message);
    }
}

// Update ML Model Details
function updateMLDetails(analysis) {
    const model = analysis?.prediction?.model;
    if (!model) return;

    // ML 状态
    const mlTrained = model.ml_trained;
    const statusEl = document.getElementById('mlStatus');
    if (statusEl) {
        statusEl.textContent = mlTrained ? '✅ 已训练' : '⏳ 未训练';
        statusEl.className = 'ml-value ' + (mlTrained ? 'bullish' : 'neutral');
    }

    // 预测方向
    const direction = model.ml_direction;
    const dirEl = document.getElementById('mlDirection');
    if (dirEl) {
        const dirText = direction === 'up' ? '📈 看涨' : direction === 'down' ? '📉 看跌' : '➡️ 中性';
        dirEl.textContent = dirText;
        dirEl.className = 'ml-value ' + (direction === 'up' ? 'bullish' : direction === 'down' ? 'bearish' : 'neutral');
    }

    // 置信度
    const confidence = model.ml_confidence;
    const confEl = document.getElementById('mlConfidence');
    if (confEl) {
        confEl.textContent = confidence ? (confidence * 100).toFixed(1) + '%' : '--';
    }

    // CV 准确率
    const report = model.model_report;
    const cvEl = document.getElementById('mlCVScore');
    if (cvEl && report?.cv_score) {
        cvEl.textContent = (report.cv_score * 100).toFixed(1) + '%';
        cvEl.className = 'ml-value ' + (report.cv_score > 0.55 ? 'bullish' : report.cv_score > 0.5 ? 'neutral' : 'bearish');
    }

    // 概率分布
    const probs = model.ml_probabilities || {};
    const up = probs.up || 0.33;
    const neutral = probs.neutral || 0.34;
    const down = probs.down || 0.33;

    const barUp = document.getElementById('mlProbUp');
    const barNeutral = document.getElementById('mlProbNeutral');
    const barDown = document.getElementById('mlProbDown');
    if (barUp) barUp.style.width = (up * 100) + '%';
    if (barNeutral) barNeutral.style.width = (neutral * 100) + '%';
    if (barDown) barDown.style.width = (down * 100) + '%';

    const labelUp = document.getElementById('mlProbUpLabel');
    const labelNeutral = document.getElementById('mlProbNeutralLabel');
    const labelDown = document.getElementById('mlProbDownLabel');
    if (labelUp) labelUp.textContent = '涨 ' + (up * 100).toFixed(0) + '%';
    if (labelNeutral) labelNeutral.textContent = '中 ' + (neutral * 100).toFixed(0) + '%';
    if (labelDown) labelDown.textContent = '跌 ' + (down * 100).toFixed(0) + '%';

    // 特征重要性
    const featuresEl = document.getElementById('mlFeatureImportance');
    if (featuresEl && report?.feature_importances) {
        const fi = report.feature_importances;
        const topFeatures = Object.entries(fi)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 6);
        if (topFeatures.length > 0) {
            featuresEl.innerHTML = topFeatures.map(([name, val]) =>
                `<span class="ml-feature-item">${name}: ${(val * 100).toFixed(1)}%</span>`
            ).join('');
        }
    }
}

// Update SOTA Model Details
function updateSOTADetails(analysis) {
    const pred = analysis?.prediction;
    if (!pred) return;

    const sotaModels = pred.sota_models;
    if (!sotaModels || Object.keys(sotaModels).length === 0) {
        console.log('[SOTA] No SOTA model data available');
        return;
    }

    // SOTA 综合投票
    const consensus = pred.sota_consensus;
    const agreement = pred.sota_agreement;
    const avgConf = pred.sota_avg_confidence;

    const consensusEl = document.getElementById('sotaConsensus');
    if (consensusEl) {
        const text = consensus === 'up' ? '📈 看涨' : consensus === 'down' ? '📉 看跌' : '➡️ 中性';
        consensusEl.textContent = text;
        consensusEl.className = 'ml-value ' + (consensus === 'up' ? 'bullish' : consensus === 'down' ? 'bearish' : 'neutral');
    }

    const agreementEl = document.getElementById('sotaAgreement');
    if (agreementEl) {
        agreementEl.textContent = (agreement * 100).toFixed(1) + '%';
    }

    const avgConfEl = document.getElementById('sotaAvgConfidence');
    if (avgConfEl) {
        avgConfEl.textContent = (avgConf * 100).toFixed(1) + '%';
    }

    // 逐个 SOTA 模型详情
    const modelDetailsEl = document.getElementById('sotaModelDetails');
    if (modelDetailsEl) {
        let html = '';

        // PatchTST
        if (sotaModels.patchtst) {
            const pt = sotaModels.patchtst;
            const dir = pt.direction === 'up' ? '📈涨' : '📉跌';
            html += `<span class="ml-feature-item">PatchTST: ${dir} (${(pt.confidence*100).toFixed(0)}%)</span> `;
        }

        // Mamba
        if (sotaModels.mamba) {
            const mb = sotaModels.mamba;
            const dir = mb.direction === 'up' ? '📈涨' : '📉跌';
            html += `<span class="ml-feature-item">Mamba: ${dir} (${(mb.confidence*100).toFixed(0)}%)</span> `;
        }

        // Diffusion
        if (sotaModels.diffusion) {
            const df = sotaModels.diffusion;
            const dir = df.direction === 'up' ? '📈涨' : '📉跌';
            html += `<span class="ml-feature-item">Diffusion: ${dir} (${(df.confidence*100).toFixed(0)}%)</span> `;
        }

        // DRL
        if (sotaModels.drl) {
            const dr = sotaModels.drl;
            html += `<span class="ml-feature-item">DRL: ${dr.action} (${(dr.confidence*100).toFixed(0)}%)</span> `;
        }

        // Conformal
        if (sotaModels.conformal) {
            const cp = sotaModels.conformal;
            html += `<span class="ml-feature-item">Conformal: [${cp.lower_bound.toFixed(4)}, ${cp.upper_bound.toFixed(4)}]</span> `;
        }

        // Alpha158
        if (sotaModels.alpha158 && sotaModels.alpha158.available) {
            html += `<span class="ml-feature-item">Alpha158: ${sotaModels.alpha158.n_factors} 因子</span> `;
        }

        // GNN
        if (sotaModels.gnn) {
            const gn = sotaModels.gnn;
            const dir = gn.direction === 'up' ? '📈涨' : '📉跌';
            html += `<span class="ml-feature-item">GNN: ${dir} (${(gn.confidence*100).toFixed(0)}%)</span> `;
        }

        // Cross-Market
        if (sotaModels.cross_market) {
            const cm = sotaModels.cross_market;
            html += `<span class="ml-feature-item">Cross-Market: A=${cm.a_share_sentiment} HK=${cm.hk_sentiment} US=${cm.us_sentiment}</span> `;
        }

        // Sentiment
        if (sotaModels.sentiment) {
            const sm = sotaModels.sentiment;
            html += `<span class="ml-feature-item">Sentiment: ${sm.label} (${(sm.score*100).toFixed(0)}‰)</span> `;
        }

        modelDetailsEl.innerHTML = html;
    }
}

// Update ATR Section
function updateATR(atrData) {
    if (!atrData) {
        console.log('No ATR data');
        return;
    }

    console.log('ATR data:', atrData);

    // Stop Loss
    const sl = atrData.stop_loss || {};
    const slPrice = parseFloat(sl.stop_loss_price) || 0;
    const slPct = parseFloat(sl.stop_loss_pct) || 0;
    const slEl = document.getElementById('atrStopLoss');
    const slPctEl = document.getElementById('atrStopLossPct');
    if (slEl) slEl.textContent = slPrice.toFixed(2) + ' 元';
    if (slPctEl) slPctEl.textContent = slPct.toFixed(2) + '%';

    // Take Profit
    const tp = atrData.take_profit || {};
    const tpPrice = parseFloat(tp.take_profit_price) || 0;
    const tpPct = parseFloat(tp.take_profit_pct) || 0;
    const tpEl = document.getElementById('atrTakeProfit');
    const tpPctEl = document.getElementById('atrTakeProfitPct');
    if (tpEl) tpEl.textContent = tpPrice.toFixed(2) + ' 元';
    if (tpPctEl) tpPctEl.textContent = tpPct.toFixed(2) + '%';

    // Support / Resistance — use ID-based selectors for reliability
    const sr = atrData.support_resistance || {};
    const atrSupportValueEl = document.getElementById('atrSupportValue');
    const atrResistanceValueEl = document.getElementById('atrResistanceValue');
    if (atrSupportValueEl) atrSupportValueEl.textContent = (parseFloat(sr.support) || 0).toFixed(2) + ' 元';
    if (atrResistanceValueEl) atrResistanceValueEl.textContent = (parseFloat(sr.resistance) || 0).toFixed(2) + ' 元';

    console.log('ATR section updated, support:', sr.support, 'resistance:', sr.resistance);
}

// Convert Markdown to HTML
function markdownToHtml(md) {
    if (!md) return '<p>暂无报告内容</p>';

    const lines = md.split('\n');
    let html = '';
    let inList = false;
    let inTable = false;
    let tableRows = [];

    function closeList() {
        if (inList) { html += '</ul>'; inList = false; }
    }
    function closeTable() {
        if (inTable) {
            html += '<table class="report-table">';
            tableRows.forEach((row, i) => {
                const tag = i === 0 ? 'th' : 'td';
                html += '<tr>' + row.map(cell => `<${tag}>${cell}</${tag}>`).join('') + '</tr>';
            });
            html += '</table>';
            inTable = false;
            tableRows = [];
        }
    }

    lines.forEach(line => {
        // Horizontal rule
        if (/^={3,}$/.test(line.trim())) {
            closeList();
            closeTable();
            html += '<hr class="report-divider">';
            return;
        }

        // Section title (e.g., "第一章：基本信息与持仓状况")
        if (/^.{1,20}：/.test(line.trim()) && !line.trim().startsWith('•') && !line.trim().startsWith('→') && !line.trim().startsWith('□') && !line.trim().startsWith('📌')) {
            closeList();
            closeTable();
            const trimmed = line.trim();
            // Check if it's a section header (has Chinese number or is short)
            if (/^(第.{1,2}章|声明)/.test(trimmed) || trimmed.length < 20) {
                html += `<h2>${trimmed}</h2>`;
            } else {
                html += `<h3>${trimmed}</h3>`;
            }
            return;
        }

        // Subsection (e.g., "1.1 股票基本信息")
        if (/^\d+\.\d+\s/.test(line.trim())) {
            closeList();
            closeTable();
            html += `<h4>${line.trim()}</h4>`;
            return;
        }

        // Table rows (pipe-separated)
        if (line.includes('|') && line.trim().startsWith('|')) {
            closeList();
            const cells = line.trim().split('|').filter(c => c.trim() !== '');
            if (!inTable) {
                inTable = true;
            }
            tableRows.push(cells);
            return;
        }

        // List items
        if (/^\s*(•|–|→|□|📌|🎯|📊)\s/.test(line) || /^\s*[\d.]+\.\s/.test(line)) {
            if (!inList) { html += '<ul>'; inList = true; }
            const content = line.trim().replace(/^[\s•–→□📌🎯📊\d.]+\s*/, '');
            html += `<li>${content}</li>`;
            return;
        }

        // Empty line
        if (line.trim() === '') {
            closeList();
            closeTable();
            return;
        }

        // Regular text
        closeList();
        closeTable();
        const formatted = line
            .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.+?)\*/g, '<em>$1</em>');
        html += `<p>${formatted}</p>`;
    });

    closeList();
    closeTable();
    return html;
}

// Update Report
function updateReport(report) {
    const reportEl = document.getElementById('reportContent');
    if (!reportEl) {
        console.error('reportContent element not found');
        return;
    }

    if (!report || typeof report !== 'string') {
        reportEl.innerHTML = '<p>暂无报告内容</p>';
        return;
    }

    reportEl.innerHTML = markdownToHtml(report);
    console.log('Report rendered as HTML, length:', reportEl.innerHTML.length);
}

// Download Report
function downloadReport() {
    if (currentReportFile) {
        window.open(`${API_BASE}/api/download_report/${currentReportFile}`, '_blank');
    }
}

// Initialize on page load
// K线信号分析更新 (新增)
function updateKlineSignals(klineSignals) {
    if (!klineSignals) return;
    
    console.log('K线信号分析数据:', klineSignals);
    
    // 更新看涨/看跌信号
    const bullishEl = document.getElementById('bullishSignals');
    const bearishEl = document.getElementById('bearishSignals');
    
    bullishEl.innerHTML = '';
    bearishEl.innerHTML = '';
    
    if (klineSignals.bullish_signals) {
        klineSignals.bullish_signals.forEach(signal => {
            bullishEl.innerHTML += `<div class="signal-item bullish">✓ ${signal}</div>`;
        });
    }
    
    if (klineSignals.bearish_signals) {
        klineSignals.bearish_signals.forEach(signal => {
            bearishEl.innerHTML += `<div class="signal-item bearish">✗ ${signal}</div>`;
        });
    }
    
    // 更新趋势强度
    document.getElementById('overallTrend').textContent = klineSignals.overall_trend || '--';
    const trendBadge = document.getElementById('trendBadge');
    trendBadge.textContent = klineSignals.trend_strength || '--';
    
    // 更新 RSI
    if (klineSignals.rsi !== undefined) {
        const rsiVal = parseFloat(klineSignals.rsi) || 0;
        document.getElementById('rsiValue').textContent = rsiVal;
        document.getElementById('rsiSignal').textContent = klineSignals.sentiment?.signal || '--';
        document.getElementById('rsiAction').textContent = klineSignals.sentiment?.action || '--';

        // 更新 RSI 进度条
        const rsiFill = document.getElementById('rsiFill');
        rsiFill.style.width = Math.min(100, Math.max(0, rsiVal)) + '%';

        // 根据 RSI 值设置颜色
        if (rsiVal > 70) {
            rsiFill.className = 'rsi-fill high';
        } else if (rsiVal < 30) {
            rsiFill.className = 'rsi-fill low';
        } else {
            rsiFill.className = 'rsi-fill medium';
        }
    }
    
    // 更新多周期共振
    if (klineSignals.multi_cycle) {
        const mc = klineSignals.multi_cycle;
        document.getElementById('dailyRsi').textContent = mc.daily_rsi || '--';
        document.getElementById('weeklyRsi').textContent = mc.weekly_rsi || '--';
        document.getElementById('monthlyRsi').textContent = mc.monthly_rsi || '--';
        document.getElementById('dailyTrend').textContent = mc.daily_trend || '--';
        document.getElementById('weeklyTrend').textContent = mc.weekly_trend || '--';
        document.getElementById('monthlyTrend').textContent = mc.monthly_trend || '--';
        document.getElementById('resonanceType').textContent = mc.resonance || '--';
        document.getElementById('resonanceDirection').textContent = mc.resonance_direction || '--';
    }
    
    // 更新形态匹配度 - 添加 null 检查
    if (klineSignals.candlestick && klineSignals.candlestick.pattern_score) {
        document.getElementById('patternScore').textContent = parseFloat(klineSignals.candlestick.pattern_score) || '--';
    } else if (klineSignals.candlestick) {
        document.getElementById('patternScore').textContent = parseFloat(klineSignals.candlestick.pattern_score) || '--';
    }

    // 更新成交量评分 - 添加 null 检查
    if (klineSignals.volume && klineSignals.volume.volume_score) {
        document.getElementById('volumeScore').textContent = parseFloat(klineSignals.volume.volume_score) || '--';
    } else if (klineSignals.volume) {
        document.getElementById('volumeScore').textContent = parseFloat(klineSignals.volume.volume_score) || '--';
    }

    // 更新持仓评分 - 添加 null 检查
    if (klineSignals.position && klineSignals.position.position_score) {
        document.getElementById('positionScore').textContent = parseFloat(klineSignals.position.position_score) || '--';
    } else if (klineSignals.position) {
        document.getElementById('positionScore').textContent = parseFloat(klineSignals.position.position_score) || '--';
    }

    // 更新价格位置
    if (klineSignals.price_position) {
        const ppEl = document.getElementById('pricePosition');
        if (ppEl) ppEl.textContent = klineSignals.price_position;
    }

    // 更新总分圆环
    if (klineSignals.total_score) {
        const scoreCircle = document.getElementById('scoreCircle');
        const scoreText = document.getElementById('scoreText');
        const totalScore = parseFloat(klineSignals.total_score) || 0;
        const offset = 314 - (314 * totalScore / 100);
        scoreCircle.style.strokeDashoffset = offset;
        scoreText.textContent = totalScore;
    }
}

// WebSocket 实时推送功能
let socket = null;
let wsConnected = false;

function connectWebSocket() {
    // 加载 Socket.IO 客户端库
    if (!window.io) {
        const script = document.createElement('script');
        script.src = 'https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.2/socket.io.js';
        script.onload = () => initializeSocket();
        document.head.appendChild(script);
    } else {
        initializeSocket();
    }
}

function initializeSocket() {
    const stockCode = document.getElementById('stockCode').value || 'sz300620';
    socket = io('', {
        reconnection: true,
        reconnectionDelay: 1000,
        reconnectionDelayMax: 8000,
        reconnectionAttempts: 10,
    });

    socket.on('connect', () => {
        wsConnected = true;
        const dot = document.getElementById('wsDot');
        const text = document.getElementById('wsText');
        const status = document.getElementById('wsStatus');
        if (dot) dot.classList.add('connected');
        if (text) text.textContent = '已连接';
        if (status) status.textContent = '停止推送';
        socket.emit('subscribe_stock', { stock_code: stockCode });
    });

    socket.on('disconnect', () => {
        wsConnected = false;
        const dot = document.getElementById('wsDot');
        const text = document.getElementById('wsText');
        const status = document.getElementById('wsStatus');
        if (dot) dot.classList.remove('connected');
        if (text) text.textContent = '已断开';
        if (status) status.textContent = '启动推送';
    });

    socket.on('fund_flow_update', (data) => {
        updateWebSocketData(data);
    });

    socket.on('new_alert', (data) => {
        // 实时推送新告警 — 更新 WebSocket 告警列表
        if (data.alerts && data.alerts.length > 0) {
            updateWebSocketAlerts(data.alerts);
        }
    });
}

function toggleWebSocket() {
    if (wsConnected) {
        socket.emit('unsubscribe_stock', {});
        socket.disconnect();
    } else {
        connectWebSocket();
    }
}

function updateWebSocketData(data) {
    // 更新价格
    if (data.data.price) {
        const price = parseFloat(data.data.price) || 0;
        document.getElementById('wsPrice').textContent = price.toFixed(2);
        const changeEl = document.getElementById('wsPriceChange');
        const changePct = parseFloat(data.data.price_change) || 0;
        changeEl.textContent = (changePct >= 0 ? '+' : '') + changePct.toFixed(2) + '%';
        changeEl.style.color = changePct >= 0 ? '#ef4444' : '#10b981';
    }

    // 更新速度
    if (data.data.speed) {
        document.getElementById('wsSpeed').textContent = (parseFloat(data.data.speed) || 0).toFixed(2);
    }

    // 更新成交量
    if (data.data.volume) {
        document.getElementById('wsVolume').textContent = (parseFloat(data.data.volume) || 0).toLocaleString();
    }

    // 更新提醒
    if (data.anomalies && data.anomalies.length > 0) {
        updateWebSocketAlerts(data.anomalies);
    }
}

function updateWebSocketAlerts(anomalies) {
    const alertList = document.getElementById('wsAlertList');
    anomalies.forEach(anomaly => {
        const alertItem = document.createElement('div');
        alertItem.className = 'alert-item ' + (anomaly.severity || 'medium');
        alertItem.innerHTML = `<span>⚠️</span><span>${anomaly.message}</span><span style="margin-left:auto;font-size:11px;color:#94a3b8">${anomaly.timestamp}</span>`;
        alertList.insertBefore(alertItem, alertList.firstChild);
        
        // 限制显示数量
        if (alertList.children.length > 20) {
            alertList.removeChild(alertList.lastChild);
        }
    });
}

// 页面加载时初始化
// 修复: 由于 script 标签在 </body> 之前，DOMContentLoaded 可能在脚本加载前已触发。
// 使用 document.readyState 检查：如果 'complete' 则立即执行，否则等待事件。
function initOnReady() {
    console.log('[initOnReady] DOM ready, initializing...');

    // 先加载仪表盘数据
    loadDashboardData();

    // Auto-refresh every 60 seconds
    setInterval(() => {
        const stockCode = document.getElementById('stockCode')?.value || 'sz300620';
        fetchStockData(stockCode);
    }, 60000);

    // 自动连接 WebSocket
    connectWebSocket();

    console.log('[initOnReady] Initialization complete');
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initOnReady);
} else {
    // DOM 已加载完成（script 在 </body> 前），立即执行
    initOnReady();
}

// 添加 Pulse 动画样式到文档 head
(function addPulseStyle() {
    const pulseStyle = document.createElement('style');
    pulseStyle.textContent = `
        @keyframes pulse {
            0%, 100% { filter: drop-shadow(0 0 8px rgba(16, 185, 129, 0.4)); }
            50% { filter: drop-shadow(0 0 16px rgba(16, 185, 129, 0.6)); }
        }
        @keyframes pulseSell {
            0%, 100% { filter: drop-shadow(0 0 8px rgba(248, 113, 113, 0.4)); }
            50% { filter: drop-shadow(0 0 16px rgba(248, 113, 113, 0.6)); }
        }
        .gauge-active.buy {
            animation: pulse 2s ease-in-out infinite;
        }
        .gauge-active.sell {
            animation: pulseSell 2s ease-in-out infinite;
        }
    `;
    document.head.appendChild(pulseStyle);
})();

// ── 资金流向显示更新（WebSocket 实时数据） ──────────────────

function updateFundFlowDisplay(data) {
    if (!data) return;
    const outer = data.outer || 0;
    const inner = data.inner || 0;
    const ratio = data.ratio || 1.0;
    const total = outer + inner || 1;
    const buyPct = ((outer / total) * 100).toFixed(1);
    const sellPct = ((inner / total) * 100).toFixed(1);

    // 买/卖比率条
    const buySegment = document.getElementById('buySegment');
    const sellSegment = document.getElementById('sellSegment');
    if (buySegment) buySegment.style.width = buyPct + '%';
    if (sellSegment) sellSegment.style.width = sellPct + '%';

    // 比率值
    const buyValue = document.getElementById('buyValue');
    const sellValue = document.getElementById('sellValue');
    if (buyValue) buyValue.textContent = buyPct + '%';
    if (sellValue) sellValue.textContent = sellPct + '%';

    // 主导方向
    const ratioDominant = document.getElementById('ratioDominant');
    if (ratioDominant) {
        ratioDominant.textContent = ratio > 1 ? '买盘主导' : '卖盘主导';
    }

    // 强度
    const ratioStrength = document.getElementById('ratioStrength');
    if (ratioStrength) {
        const strength = ratio > 1.2 ? '强' : ratio > 0.8 ? '中' : '弱';
        const dominant = ratio > 1 ? 'buy' : 'sell';
        ratioStrength.textContent = strength;
        ratioStrength.className = 'ratio-strength ' + dominant;
    }

    // 刷新指示器动画
    const indicator = document.getElementById('flowRefreshIndicator');
    if (indicator) {
        indicator.style.transform = 'scale(1.3)';
        indicator.style.opacity = '1';
        setTimeout(() => {
            indicator.style.transform = 'scale(1)';
            indicator.style.opacity = '0.6';
        }, 300);
    }
}

// 资金流向数据刷新
function refreshFundFlowData(stockCode) {
    fetch(`/api/stock/${stockCode}`, { headers: _apiHeaders() })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                updateFundFlowDisplay(data.data);
            }
        })
        .catch(error => console.error('Error refreshing fund flow:', error));
}

// ═══════════ 回测报告 ═══════════

function runBacktest() {
    const stockCode = document.getElementById('stockCode').value || 'sz300620';
    showLoading(true);
    fetch(`/api/backtest/report?stock_code=${stockCode}`, { headers: _apiHeaders() })
        .then(r => r.json())
        .then(data => {
            if (data.success) {
                updateBacktestReport(data);
            } else {
                alert('回测失败: ' + (data.error || '未知错误'));
            }
        })
        .catch(e => { console.error(e); alert('回测请求失败: ' + e.message); })
        .finally(() => showLoading(false));
}

function updateBacktestReport(data) {
    const wf = data.walk_forward;
    const mc = data.monte_carlo;
    const cs = data.chinese_summary;
    const summary = wf?.summary || {};

    // 汇总指标
    document.getElementById('bt_total_return').textContent = (summary.mean_return * 100).toFixed(1) + '%';
    document.getElementById('bt_sharpe').textContent = summary.mean_sharpe?.toFixed(2) || '--';
    document.getElementById('bt_maxdd').textContent = (summary.mean_maxdd * 100).toFixed(1) + '%';
    document.getElementById('bt_winrate').textContent = (summary.mean_winrate * 100).toFixed(0) + '%';
    const consistent = summary.consistent_profit_windows ?? cs?.details?.consistent_profit_windows ?? 0;
    document.getElementById('bt_consistent').textContent = `${consistent}/${summary.n_windows || 0}`;
    document.getElementById('bt_mc_prob').textContent = (mc?.probability_profit * 100 || 0).toFixed(0) + '%';

    // 窗口结果
    const windowsEl = document.getElementById('bt_windows');
    if (wf?.windows?.length > 0) {
        windowsEl.innerHTML = wf.windows.map((w, i) => `
            <div class="wf-window">
                <span class="wf-label">窗口 ${i+1}</span>
                <span class="wf-metric">收益: ${(w.total_return * 100).toFixed(1)}%</span>
                <span class="wf-metric">Sharpe: ${w.sharpe_ratio?.toFixed(2) || '--'}</span>
                <span class="wf-metric">回撤: ${(w.max_drawdown * 100).toFixed(1)}%</span>
                <span class="wf-metric">胜率: ${(w.win_rate * 100).toFixed(0)}%</span>
            </div>
        `).join('');
    }

    // Monte Carlo
    if (mc) {
        document.getElementById('mc_mean').textContent = '¥' + (mc.mean_final / 10000).toFixed(1) + '万';
        document.getElementById('mc_median').textContent = '¥' + (mc.median_final / 10000).toFixed(1) + '万';
        document.getElementById('mc_p5').textContent = '¥' + (mc.percentile_5 / 10000).toFixed(1) + '万';
        document.getElementById('mc_p95').textContent = '¥' + (mc.percentile_95 / 10000).toFixed(1) + '万';
        document.getElementById('mc_prob').textContent = (mc.probability_profit * 100).toFixed(0) + '%';

        // 简易柱状图
        const histEl = document.getElementById('mc_histogram');
        const bins = mc.histogram?.values || [];
        if (bins.length > 0) {
            const maxVal = Math.max(...bins);
            histEl.innerHTML = bins.map((v, i) =>
                `<div class="mc-bar" style="width:${(v/maxVal*100).toFixed(0)}%">
                    <span class="mc-bar-label">${(v/10000).toFixed(0)}万</span>
                </div>`
            ).join('');
        }
    }

    // ── 中文总结 ──────────────────────────────────────────────
    if (cs) {
        const card = document.getElementById('bt_chinese_summary_card');
        const el = document.getElementById('bt_chinese_summary');
        card.style.display = 'block';

        const details = cs.details || {};
        el.innerHTML = `
            <div class="summary-item ${cs.strategy_verdict.includes('良好') ? 'positive' : cs.strategy_verdict.includes('不佳') ? 'negative' : 'neutral'}">
                <span class="summary-icon">📊</span>
                <div>
                    <div class="summary-label">策略评价</div>
                    <div class="summary-text">${cs.strategy_verdict}</div>
                </div>
            </div>
            <div class="summary-item ${cs.risk_verdict.includes('优秀') ? 'positive' : cs.risk_verdict.includes('偏大') ? 'negative' : 'neutral'}">
                <span class="summary-icon">🛡️</span>
                <div>
                    <div class="summary-label">风险评估</div>
                    <div class="summary-text">${cs.risk_verdict}</div>
                </div>
            </div>
            <div class="summary-item ${cs.sharpe_verdict.includes('优秀') ? 'positive' : cs.sharpe_verdict.includes('较差') ? 'negative' : 'neutral'}">
                <span class="summary-icon">⚖️</span>
                <div>
                    <div class="summary-label">Sharpe 评价</div>
                    <div class="summary-text">${cs.sharpe_verdict}</div>
                </div>
            </div>
            <div class="summary-item ${cs.winrate_verdict.includes('较高') ? 'positive' : cs.winrate_verdict.includes('偏低') ? 'negative' : 'neutral'}">
                <span class="summary-icon">🎯</span>
                <div>
                    <div class="summary-label">胜率评价</div>
                    <div class="summary-text">${cs.winrate_verdict}</div>
                </div>
            </div>
            <div class="summary-item ${cs.mc_verdict.includes('很高') || cs.mc_verdict.includes('较好') ? 'positive' : cs.mc_verdict.includes('偏低') ? 'negative' : 'neutral'}">
                <span class="summary-icon">🎲</span>
                <div>
                    <div class="summary-label">蒙特卡洛结论</div>
                    <div class="summary-text">${cs.mc_verdict}</div>
                </div>
            </div>
            <div class="summary-details">
                <div class="summary-detail-item">
                    <span class="summary-detail-value">${details.mean_return_pct || 0}%</span>
                    <span class="summary-detail-label">平均收益</span>
                </div>
                <div class="summary-detail-item">
                    <span class="summary-detail-value">${details.mean_sharpe || 0}</span>
                    <span class="summary-detail-label">Sharpe 比率</span>
                </div>
                <div class="summary-detail-item">
                    <span class="summary-detail-value">${details.mean_maxdd_pct || 0}%</span>
                    <span class="summary-detail-label">最大回撤</span>
                </div>
                <div class="summary-detail-item">
                    <span class="summary-detail-value">${details.mean_winrate_pct || 0}%</span>
                    <span class="summary-detail-label">平均胜率</span>
                </div>
                <div class="summary-detail-item">
                    <span class="summary-detail-value">${details.mc_profit_prob_pct || 0}%</span>
                    <span class="summary-detail-label">盈利概率</span>
                </div>
                <div class="summary-detail-item">
                    <span class="summary-detail-value">¥${details.mc_median_final_wan || 0}万</span>
                    <span class="summary-detail-label">蒙特卡洛中位数</span>
                </div>
            </div>
        `;
    }
}

// ═══════════ 组合优化 ═══════════

function runPortfolioOptimize() {
    const stockCode = document.getElementById('stockCode').value || 'sz300620';
    showLoading(true);
    fetch(`/api/portfolio/optimize?stock_code=${stockCode}`, { headers: _apiHeaders() })
        .then(r => r.json())
        .then(data => {
            if (data.success) {
                updatePortfolioOptimization(data);
            } else {
                alert('优化失败: ' + (data.error || '未知错误'));
            }
        })
        .catch(e => { console.error(e); alert('优化请求失败: ' + e.message); })
        .finally(() => showLoading(false));
}

function updatePortfolioOptimization(data) {
    const bl = data.black_litterman;
    const rp = data.risk_parity_weights;

    // Black-Litterman 权重
    const blEl = document.getElementById('bl_weights');
    if (bl?.weights) {
        const entries = Object.entries(bl.weights);
        const maxW = Math.max(...entries.map(e => e[1]));
        blEl.innerHTML = entries.map(([name, w]) => `
            <div class="bl-item">
                <span class="bl-name">${name}</span>
                <div class="bl-bar-bg"><div class="bl-bar" style="width:${(w/maxW*100).toFixed(0)}%"></div></div>
                <span class="bl-value">${(w*100).toFixed(1)}%</span>
            </div>
        `).join('');
    }

    // 风险平价权重
    const rpEl = document.getElementById('rp_weights');
    if (rp) {
        const entries = Object.entries(rp);
        const maxW = Math.max(...entries.map(e => e[1]));
        rpEl.innerHTML = entries.map(([name, w]) => `
            <div class="bl-item">
                <span class="bl-name">${name}</span>
                <div class="bl-bar-bg"><div class="bl-bar" style="width:${(w/maxW*100).toFixed(0)}%;background:#f59e0b"></div></div>
                <span class="bl-value">${(w*100).toFixed(1)}%</span>
            </div>
        `).join('');
    }

    // 对比
    const compEl = document.getElementById('weight_comparison');
    if (bl?.weights && rp) {
        const names = Object.keys(bl.weights);
        compEl.innerHTML = `
            <table class="weight-table">
                <thead><tr><th>股票</th><th>Black-Litterman</th><th>风险平价</th><th>差异</th></tr></thead>
                <tbody>
                    ${names.map(n => {
                        const blw = bl.weights[n] || 0;
                        const rpw = rp[n] || 0;
                        return `<tr>
                            <td>${n}</td>
                            <td>${(blw*100).toFixed(1)}%</td>
                            <td>${(rpw*100).toFixed(1)}%</td>
                            <td style="color:${(blw-rpw)>0?'#ef4444':'#10b981'}">${((blw-rpw)*100).toFixed(1)}pp</td>
                        </tr>`;
                    }).join('')}
                </tbody>
            </table>
        `;
    }
}

// ═══════════ 因子分析 ═══════════

function runFactorAnalysis() {
    const stockCode = document.getElementById('stockCode').value || 'sz300620';
    showLoading(true);
    fetch(`/api/factors/norm?stock_code=${stockCode}`, { headers: _apiHeaders() })
        .then(r => r.json())
        .then(data => {
            if (data.success) {
                updateFactorAnalysis(data);
            } else {
                alert('因子分析失败: ' + (data.error || '未知错误'));
            }
        })
        .catch(e => { console.error(e); alert('因子分析请求失败: ' + e.message); })
        .finally(() => showLoading(false));
}

function updateFactorAnalysis(data) {
    const raw = data.raw_scores || {};
    const norm = data.normalized_scores || {};

    // 原始评分
    const rawEl = document.getElementById('raw_scores');
    const rawEntries = Object.entries(raw);
    const rawMax = Math.max(...rawEntries.map(e => e[1]), 1);
    rawEl.innerHTML = rawEntries.map(([name, score]) => `
        <div class="bl-item">
            <span class="bl-name">${name}</span>
            <div class="bl-bar-bg"><div class="bl-bar" style="width:${(score/rawMax*100).toFixed(0)}%;background:#3b82f6"></div></div>
            <span class="bl-value">${score.toFixed(1)}</span>
        </div>
    `).join('');

    // 标准化评分
    const normEl = document.getElementById('norm_scores');
    const normEntries = Object.entries(norm);
    const normAbsMax = Math.max(...normEntries.map(e => Math.abs(e[1])), 0.01);
    normEl.innerHTML = normEntries.map(([name, zscore]) => `
        <div class="bl-item">
            <span class="bl-name">${name}</span>
            <div class="bl-bar-bg">
                <div class="bl-bar ${zscore >= 0 ? 'bl-bar-pos' : 'bl-bar-neg'}"
                     style="width:${(Math.abs(zscore)/normAbsMax*100).toFixed(0)}%;
                            margin-left:${zscore >= 0 ? '0' : (50 - Math.abs(zscore)/normAbsMax*50).toFixed(0)}%">
                </div>
            </div>
            <span class="bl-value ${zscore >= 0 ? 'positive' : 'negative'}">${zscore.toFixed(2)}</span>
        </div>
    `).join('');

    // 因子暴露
    const expEl = document.getElementById('factor_exposure');
    const absMax = Math.max(...normEntries.map(e => Math.abs(e[1])), 0.01);
    expEl.innerHTML = normEntries.map(([name, zscore]) => {
        const deviation = zscore;  // Z-Score 本身就是偏离均值的标准差数
        return `
            <div class="bl-item">
                <span class="bl-name">${name}</span>
                <div class="bl-bar-bg">
                    <div class="bl-bar ${deviation >= 0 ? 'bl-bar-pos' : 'bl-bar-neg'}"
                         style="width:${(Math.abs(deviation)/absMax*100).toFixed(0)}%;
                                margin-left:${deviation >= 0 ? '0' : (50 - Math.abs(deviation)/absMax*50).toFixed(0)}%">
                    </div>
                </div>
                <span class="bl-value ${deviation >= 0 ? 'positive' : 'negative'}">${deviation >= 0 ? '+' : ''}${deviation.toFixed(2)}σ</span>
            </div>
        `;
    }).join('');

    // 综合评级
    const ratingEl = document.getElementById('factor_rating');
    const rating = data.rating || '中性';
    const ratingColor = rating === '强烈推荐' ? '#ef4444' : rating === '推荐' ? '#f87171' : rating === '观望' ? '#f59e0b' : '#10b981';
    ratingEl.innerHTML = `
        <div class="rating-display">
            <div class="rating-badge" style="background:${ratingColor}">${rating}</div>
            <div class="rating-detail">
                <p>原始加权评分: <strong>${data.weighted_score_raw?.toFixed(2) || '--'}</strong></p>
                <p>标准化加权评分: <strong>${data.weighted_score_normalized?.toFixed(4) || '--'}</strong></p>
                <p>标准化: <strong style="color:${data.normalized ? '#10b981' : '#94a3b8'}">${data.normalized ? '已启用 ✓' : '未启用'}</strong></p>
            </div>
        </div>
    `;
}
