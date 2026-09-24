// Stock Analyzer - Tab & Section Navigation
// Tab 切换、Section 导航、懒加载

// ── Section Navigation ──────────────────────────────────
function showSection(sectionId) {
    document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
    document.querySelectorAll('.nav-link').forEach(n => n.classList.remove('active'));

    const section = document.getElementById(sectionId);
    if (section) {
        section.classList.add('active');
    }

    const navLinks = document.querySelectorAll('.nav-link');
    navLinks.forEach(link => {
        const onclick = link.getAttribute('onclick') || '';
        if (onclick.includes(sectionId)) {
            link.classList.add('active');
        }
    });
}

// ── Tab Navigation ──────────────────────────────────────
function showTab(tabId) {
    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));

    const tabContent = document.getElementById('tab-' + tabId);
    if (tabContent) {
        tabContent.classList.add('active');
    }

    // 高亮当前按钮
    const activeBtn = document.querySelector(`.tab-btn[onclick*="'${tabId}'"]`) ||
                      document.querySelector(`.tab-btn[onclick*='"${tabId}"']`);
    if (activeBtn) {
        activeBtn.classList.add('active');
    }
}

// ── 懒加载 Tab 内容 ─────────────────────────────────────
function lazyLoadTab(tabId, loadFn) {
    const tabContent = document.getElementById('tab-' + tabId);
    if (!tabContent) return;

    // 检查是否已加载
    if (tabContent.dataset.loaded === 'true') return;

    // 使用 IntersectionObserver 延迟加载
    if ('IntersectionObserver' in window) {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    tabContent.dataset.loaded = 'true';
                    if (typeof loadFn === 'function') loadFn();
                    observer.disconnect();
                }
            });
        }, { rootMargin: '200px' });
        observer.observe(tabContent);
    } else {
        // 降级: 直接加载
        tabContent.dataset.loaded = 'true';
        if (typeof loadFn === 'function') loadFn();
    }
}

// ── Tab 分组导航 ────────────────────────────────────────
function initTabGroups() {
    const tabNav = document.querySelector('.tab-nav');
    if (!tabNav) return;

    const tabs = tabNav.querySelectorAll('.tab-btn');
    let currentGroup = '';

    tabs.forEach(tab => {
        const group = tab.dataset.tabGroup;
        if (group && group !== currentGroup) {
            currentGroup = group;
            const label = document.createElement('span');
            label.className = 'tab-group-label';
            label.textContent = group;
            tabNav.insertBefore(label, tab);
        }
    });
}

// ── 初始化 ──────────────────────────────────────────────
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        initTabGroups();
    });
} else {
    initTabGroups();
}