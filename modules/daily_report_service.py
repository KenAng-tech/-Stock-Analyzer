# -*- coding: utf-8 -*-
"""
daily_report_service.py — 每日晨报研报服务 (2026-09-09)

08:00 自动生成「隔夜美股 + 国际资讯 + 真实持仓」综合晨报:
  ① 美股隔夜: 腾讯 us 行情 (三大指数 + 纳指科技/光链股, GBK→UTF8)
  ② 国际资讯: 东财 search-api 新闻标题 (美股/算力/锂价关键词)
  ③ 持仓量化: portfolio_store.snapshot + cvar_position_manager (K线/CVaR/信号)
  ④ LLM 汇编: llm_router.route (8080, 失败降级规则模板, 不吞错)

持久化: ~/.stock_analyzer_daily_report.db (SQLite, 最近 30 日);
线程: DailyReportScheduler daemon 线程, 由 boot/monitor_starter 统一启动
(复用 23:00 链同款 daemon 模式, 单实例非双影);
任一数据源失败标 source_errors, 链不中断 (09-02 self-HTTP 教训:
全程模块级直调, 无 self-HTTP 自调)。
"""

import json
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import requests

from modules.logger import logger
from utils.outbound_guard import host_throttle

US_WATCHLIST = [
    'usDJI', 'usIXIC', 'usINX',
    'usNVDA', 'usAMD', 'usMRVL', 'usCOHR', 'usAAOI', 'usANET', 'usAVGO', 'usTSM', 'usMU',
]
US_NAMES = {
    'usDJI': '道指', 'usIXIC': '纳指', 'usINX': '标普500', 'usNVDA': '英伟达',
    'usAMD': 'AMD', 'usMRVL': '迈威尔', 'usCOHR': 'Coherent', 'usAAOI': 'AAOI',
    'usANET': 'Arista', 'usAVGO': '博通', 'usTSM': '台积电ADR', 'usMU': '美光',
}
NEWS_KEYWORDS = ['隔夜美股 收盘', 'AI算力 光通信 1.6T', '碳酸锂 价格', '英伟达 光模块 订单']
_DB_PATH = '~/.stock_analyzer_daily_report.db'
_HTTP_HEADERS = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'}


def _db():
    import os
    path = os.path.expanduser(_DB_PATH)
    conn = sqlite3.connect(path, timeout=5)
    conn.execute('CREATE TABLE IF NOT EXISTS daily_report '
                 '(date TEXT PRIMARY KEY, json TEXT)')
    return conn


def _intel_lines(h: Dict) -> List[str]:
    """持仓行 → 市场情报段 (2026-09-13 #16 链 A; prompt 与降级模板共用).

    intel/insight/audit 缺 = 只出量化壳行 (诚实降级, 缺输入不降智 —
    09-09 教训: 无上下文比坏上下文诚实); 段内只呈现事实,
    不构成方向投票 (审计=观察层, 2026 调研纪律)。
    """
    line = (f"- {h['name']}({h['symbol']}) 现价{h.get('price')} 浮{h.get('pl_pct')}% "
            f"仓位{h.get('weight_pct')}% 预算{h.get('budget_pct')}% "
            f"CVaR{h.get('cvar_95')}% {h.get('hint', '')}")
    it = h.get('intel') or {}
    seg: List[str] = []
    dt = it.get('dragon_tiger') or {}
    if dt.get('on_board'):
        lt = dt.get('latest') or {}
        seg.append(f"龙虎榜{dt.get('count', '?')}次上榜 (最新 {lt.get('date', '')}: "
                   f"换手{lt.get('turnover_pct', '?')}% "
                   f"D20收益{lt.get('d20_pct', '无')})")
    nx = (it.get('unlock') or {}).get('next')
    if nx:
        seg.append(f"{nx.get('unlock_date', '?')}解禁 {nx.get('type', '')[:12]}")
    sm = it.get('sentiment') or {}
    if sm.get('label') is not None and not sm.get('degraded'):
        seg.append(f"舆情{sm.get('label')}{sm.get('score')} ({sm.get('n_articles')}篇)")
    ins = h.get('insight') or {}
    if ins.get('valid'):
        seg.append(f"AI洞察 {ins.get('direction')}{ins.get('conviction')}: "
                   + ' / '.join(ins.get('evidence') or [])
                   + f" — 失效: {ins.get('invalidation', '')}")
    au = h.get('audit') or {}
    if au.get('block_hints'):
        seg.append('执行审计: ' + ' | '.join(au['block_hints']))
    return [line, '  情报: ' + '; '.join(seg)] if seg else [line]


class DailyReportService:
    """晨报数据链 + LLM 汇编 + 存储"""

    def __init__(self):
        self._lock = threading.Lock()
        self._generating = False
        self._scheduler: Optional['DailyReportScheduler'] = None
        # 2026-09-20 整合①: 推送台账 — 同内容当日去重 + 送达确认 (PanWatch 发送成功才标记式)
        from modules.push_ledger import PushLedger
        self._push_ledger = PushLedger()

    # ── 数据源 ①: 美股隔夜 ──────────────────────────────

    def fetch_us_overnight(self) -> List[Dict]:
        """腾讯美股行情 (GBK): 指数 + 科技/光链股 收盘/涨跌%"""
        url = 'http://qt.gtimg.cn/q=' + ','.join(US_WATCHLIST)
        try:
            host_throttle.acquire('tencent', host_key='qt.gtimg.cn')  # P0-3 (2026-09-20)
            r = requests.get(url, timeout=12, headers=_HTTP_HEADERS)
            text = r.content.decode('gbk', errors='ignore')
            out = []
            for block in text.split('v_us')[1:]:
                body = block.split('=', 1)[1].strip(';"') if '=' in block else ''
                f = body.split('~')
                if len(f) < 5 or not f[3]:
                    continue
                try:
                    price = float(f[3]); prev = float(f[4]) if f[4] else price
                    chg = (price - prev) / prev * 100 if prev else 0.0
                except ValueError:
                    continue
                code = 'us' + f[0]
                out.append({'name': US_NAMES.get(code, f[1]),
                            'close': price, 'chg_pct': round(chg, 2)})
            return out
        except Exception as e:
            logger.error(f"[DailyReport] 美股行情抓取失败: {e}")
            return []

    # ── 数据源 ②: 国际资讯 (东财 search-api) ─────────────

    def fetch_global_news(self, keywords: Optional[List[str]] = None) -> List[str]:
        """东财新闻标题 (美股/算力/锂价/订单), 跨关键词去重"""
        url = 'https://search-api-web.eastmoney.com/search/jsonp'
        news: List[str] = []
        seen = set()
        for kw in (keywords or NEWS_KEYWORDS):
            try:
                host_throttle.acquire('search-api-web', host_key='search-api-web.eastmoney.com')  # P0-3 (2026-09-20)
                param = {'uid': '', 'keyword': kw, 'type': ['cmsArticleWebOld'],
                         'client': 'web', 'keyword': kw,
                         'param': {'cmsArticleWebOld': {
                             'searchScope': 'default', 'sort': 'default',
                             'pageIndex': 1, 'pageSize': 6, 'preTag': '', 'postTag': ''}}}
                r = requests.get(url, params={'cb': 'x', 'param': json.dumps(param, ensure_ascii=False)},
                                 timeout=10, headers=_HTTP_HEADERS)
                body = r.text[r.text.index('(') + 1:r.text.rindex(')')]
                for a in (json.loads(body).get('result', {}) or {}).get('cmsArticleWebOld', []):
                    title = a.get('title', '').replace('<em>', '').replace('</em>', '')
                    if title and title not in seen:
                        seen.add(title)
                        news.append(f"[{a.get('date', '')[:10]}] {title}")
            except Exception as e:
                logger.error(f"[DailyReport] 新闻抓取失败 {kw}: {e}")
        return news

    # ── 数据源 ③: 真实持仓量化 ──────────────────────────

    def build_holdings(self) -> List[Dict]:
        """4 票: 现价/浮盈/权重/CVaR/风险预算/方向信号 (模块级直调, 非 HTTP)"""
        rows: List[Dict] = []
        try:
            from modules.portfolio_store import portfolio_store
            from modules.dependencies import get_data_fetcher
            from modules.cvar_position_manager import cvar_position_manager

            total = portfolio_store.get_total_value() or 1000000.0
            snap = portfolio_store.get_snapshot()
            if not snap:
                return rows
            fetcher = get_data_fetcher()
            import numpy as np

            for p in snap['positions']:
                code, name = p['symbol'].lower(), p['name']
                row = dict(p)
                try:
                    klines = fetcher.get_kline_data(code, 'daily', 120) or []
                    closes = np.array([float(k.get('close', 0)) for k in klines
                                       if float(k.get('close', 0) or 0) > 0])
                    if len(closes) >= 60:
                        r = np.diff(np.log(closes))
                        cvar95 = -float(np.sort(r)[:6].mean()) * 100
                        last = float(closes[-1])
                        row['price'] = round(last, 3)
                        row['pl_pct'] = round((last - p['cost']) / p['cost'] * 100, 2)
                        row['cvar_95'] = round(cvar95, 3)
                        # 风险预算 (走已验证的 CVaR 仓位链)
                        res = cvar_position_manager.calculate_position(
                            code, klines,
                            {'direction': 'up', 'confidence': 0.65, 'target_price': None},
                            total, 0.95)
                        budget = (res.get('position') or {}).get('vol_adjusted', 0) or 0
                        weight = p['qty'] * last / total if total > 0 else 0
                        row['budget_pct'] = round(budget * 100, 2)
                        row['weight_pct'] = round(weight * 100, 2)
                        if budget <= 0:
                            row['hint'] = '数据不足 (未算出预算)'
                        elif weight > budget:
                            row['hint'] = '减仓区 (实际仓位>风险预算)'
                        elif row['pl_pct'] <= -2:
                            row['hint'] = '警戒 (浮亏已穿-2%容忍线)'
                        else:
                            row['hint'] = '观察区'
                except Exception as e:
                    logger.error(f"[DailyReport] {name} 量化数据失败: {e}")
                    row['source_error'] = str(e)[:120]
                rows.append(row)

            # ② 市场情报链 (2026-09-13 #16 链 A): 四票 intel→insight→audit 并行
            # (09-10 并行预取 K线先例, 四链并行; 25s 预算耗尽=诚实缺情报)
            bundle = self._intel_bundle([p['symbol'].lower() for p in snap['positions']])
            for row in rows:
                intel = bundle.get(row['symbol'].lower()) or {}
                if intel.get('skipped'):
                    continue
                row['intel'] = intel.get('intel') or {}
                if intel.get('insight') and intel['insight'].get('valid'):
                    row['insight'] = intel['insight']
                if intel.get('audit') and not intel['audit'].get('skipped'):
                    row['audit'] = intel['audit']
        except Exception as e:
            logger.error(f"[DailyReport] 持仓快照构建失败: {e}")
        return rows

    # ── 情报链 ⑤: 市场情报→AI洞察→执行审计 (2026-09-13 #16 链 A) ──

    def _gather_stock_intel(self, code: str) -> Dict:
        """单票市场情报: intel_summary → AI 洞察 (喂料非空转) → 执行审计.

        链纪律 (09-10 教训: 降级形交调用方, 此处无 try/except —
        由 _intel_bundle 链尾捕获+记日志 = 链尾检查, 无坏链):
        - intel_summary 三源独立降级 (一源坏不拖整链);
        - AI 洞察 fail-closed: 以 intel 的 news/龙虎榜作 RAG 输入喂
          get_insight (同源不二次取数), 缺料 abstain 不空转 8080;
        - 执行审计 audit_for_stock (自带 5s join 超时跳过, 09-12 审计原形)。
        """
        from modules.intel_engine import intel_summary
        from modules.insight_engine import get_insight_engine
        from modules.exec_audit import audit_for_stock

        intel = intel_summary(code) or {}
        feed = (intel.get('_insight_feed') or {}).get('news') or []
        dt = (intel.get('dragon_tiger') or {}).get('items') or []
        insight = get_insight_engine().get_insight(
            code, news=feed, dragon_tiger=dt)
        audit = audit_for_stock(code, 'neutral', timeout=5.0)
        return {'intel': intel, 'insight': insight, 'audit': audit}

    def _intel_bundle(self, codes: List[str], timeout: float = 25.0) -> Dict:
        """四票情报链并行 (09-10 并行预取 K线先例).

        as_completed(timeout): 某票链坏 → 捕获记日志 (链尾检查形);
        超时未归 → skipped 标记 (缺输入不降智, 09-09 教训: 无情报比
        坏情报诚实); 两层线程池不嵌套 (09-03 教训: 无 self-HTTP 自调)。
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed, \
            TimeoutError as FutTimeout
        out: Dict[str, Dict] = {}
        if not codes:
            return out
        try:
            with ThreadPoolExecutor(max_workers=4,
                                    thread_name_prefix='intel') as ex:
                futs = {ex.submit(self._gather_stock_intel, c): c
                        for c in codes}          # as_completed 吃 Futures 非 keys
                try:
                    for fut in as_completed(futs, timeout=timeout):
                        c = futs[fut]
                        try:
                            out[c] = fut.result()
                        except Exception as e:
                            logger.error(f"[DailyReport] {c} 情报链坏: "
                                         f"{type(e).__name__}: {str(e)[:100]}")
                            out[c] = {'skipped': True,
                                       'reason': f'{type(e).__name__}'}
                except FutTimeout:
                    logger.warning(f"[DailyReport] 情报链 {timeout}s 预算耗尽, "
                                   f"未归票按缺情报处理")
        except Exception as e:
            logger.error(f"[DailyReport] 情报并行启动失败: {e}")
        for c in codes:
            out.setdefault(c, {'skipped': True, 'reason': '未归 (并行预算耗尽)'})
        return out

    # ── 汇编 ④: LLM 晨报 (失败降级规则模板) ─────────────

    def assemble(self, material: Dict) -> str:
        from modules.llm_router import llm_router

        us = material.get('us') or []
        news = material.get('news') or []
        hold = material.get('holdings') or []
        prompt = (
            "你是A股盘前晨报分析师。基于以下实时数据写今日晨报 (≤900字, Markdown):\n"
            "## 隔夜美股\n" + '\n'.join(
                f"- {u['name']}: 收盘 {u['close']} ({u['chg_pct']:+.2f}%)" for u in us) + "\n"
            "## 国际资讯标题\n" + '\n'.join(f"- {n}" for n in news[:14]) + "\n"
            "## 我的真实持仓 (行情+市场情报; 情报段=呈现证据非结论)\n" + '\n'.join(
                l for h in hold for l in _intel_lines(h)) + "\n\n"
            "输出: ①隔夜美科技与光链观察(对A股光通信/算力/锂映射) ②四票逐一:趋势/利空利多/"
            "触发观察价 ③今日纪律 (不预测点位, 只给触发条件; 不说买入卖出, 给'观察-触发-复核'清单)。"
        )
        try:
            res = llm_router.route(prompt, timeout=120)
            if res.get('success') and res.get('content'):
                c = res['content']
                # route() 内部 8080 不可用时降级返回 rule_engine JSON
                # (09-03 教训: 它带 success=True; provider/fallback 字段可靠, 再兜底查内容)
                if (res.get('provider') != 'rule_engine' and not res.get('fallback')
                        and 'fallback decision' not in c and 'Using fallback' not in c):
                    return c
                logger.warning("[DailyReport] LLM 链返回规则引擎降级 (8080 不可用), 走晨报模板")
        except Exception as e:
            logger.error(f"[DailyReport] LLM 汇编失败: {e}")
        # 规则降级模板 (数据仍完整, 标明 LLM 降级)
        lines = ['### 隔夜美股 (LLM 降级, 规则模板)', '']
        lines += [f"- {u['name']}: {u['close']} ({u['chg_pct']:+.2f}%)" for u in us]
        lines += ['', '### 国际资讯', ''] + [f"- {n}" for n in news[:10]]
        lines += ['', '### 我的真实持仓 (触发观察见各条 hint)', '']
        lines += [l for h in hold for l in _intel_lines(h)]
        lines += ['', '> 08:00 链: LLM 服务不可用, 本表为纯数据降级版 (系统规则模板)。']
        return '\n'.join(lines)

    # ── Telegram 推送 ────────────────────────────────────────

    def _push_to_telegram(self, report_md: str, date_str: str, trigger: str = 'auto'):
        """将晨报写入 Telegram 文件队列 (bot 轮询发送) + 推送台账去重 (2026-09-20 整合①)。

        去重 = 同内容当日一次 (auto 触发); manual/tg 触发绕过但照常记账。
        写队列 = pending (daemon 5s 内取走并 Telegram 接收 = 删文件 = confirmed);
        滞留 30min 未确认 = failed (降级非断链, 允许补推); 连续 3 次 failed = 推送链断升级人工。
        """
        import os
        import hashlib
        key = f"report:{date_str}:{hashlib.sha1(report_md.encode()).hexdigest()[:12]}"
        allowed, reason, state = self._push_ledger.check(key, force=(trigger != 'auto'))
        if not allowed:
            logger.info(f"[DailyReport] Telegram 去重跳过 ({trigger}): {state} — {reason}")
            return
        queue_dir = os.path.expanduser('~/.claude/channels/telegram/reply_queue')
        if not os.path.isdir(queue_dir):
            logger.info("[DailyReport] Telegram 回复队列不存在, 跳过推送")
            return
        try:
            # Telegram 单条消息上限 4096 字符, 截断保留头部
            if len(report_md) > 3800:
                report_md = report_md[:3800] + '\n\n--- (已截断, 完整报告见 /api/daily_report) ---'
            payload = {
                'chat_id': '5438240773',
                'text': f'📊 每日晨报 {date_str}\n\n' + report_md,
            }
            import time
            fname = f'telegram_report_{int(time.time()*1000)}.json'
            fpath = os.path.join(queue_dir, fname)
            with open(fpath, 'w', encoding='utf-8') as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            self._push_ledger.register(key, fname)
            logger.info(f"[DailyReport] Telegram 队列已写: {fname} (trigger={trigger}, {reason})")
        except Exception as e:
            logger.error(f"[DailyReport] Telegram 推送失败: {e}")

    # ── 全链编排 ────────────────────────────────────────

    def _price_lookup(self, code: str):
        """持仓现价注入 (行情→收盘近似)"""
        try:
            from modules.dependencies import get_data_fetcher
            si = get_data_fetcher().get_stock_info(code) or {}
            return si.get('price') or 0
        except Exception:
            return 0

    def generate(self, trigger: str = 'auto') -> Dict:
        """生成今日晨报 (幂等锁: 并发重入直接返回 409 语义)"""
        with self._lock:
            if self._generating:
                return {'success': False, 'error': '晨报链正在生成中', 'status': 'running'}
            # 2026-09-20 整合②: 链首绑 trace_id → 本链全部日志 (取数→LLM→持久化→
            # 推送→送达) 同 trace, grep 一即串全链 (PanWatch 链尾可观测式)
            from modules.log_context import clear_trace, set_trace, gen_trace_id
            clear_trace()
            set_trace(gen_trace_id('report', trigger))
            self._generating = True
        errors: List[str] = []
        try:
            t0 = time.time()
            us = self.fetch_us_overnight()
            if not us:
                errors.append('美股行情源无响应')
            # 联网采集关键词: 通用 4 + 产业链 2 + 持仓股名 (真实持仓动态注入)
            keywords = NEWS_KEYWORDS + ['液冷服务器', '铜缆高速连接']
            try:
                from modules.portfolio_store import portfolio_store
                snap = portfolio_store.get_snapshot()
                if snap:
                    keywords += [p['name'] for p in snap['positions']]
            except Exception as e:
                logger.error(f"[DailyReport] 持仓关键词注入失败: {e}")
            news = self.fetch_global_news(keywords=keywords)
            if not news:
                errors.append('国际资讯源无响应')
            hold = self.build_holdings()
            if not hold:
                errors.append('持仓快照不可得 (portfolio_store)')
            material = {'us': us, 'news': news, 'holdings': hold}
            report_md = self.assemble(material)
            llm_degraded = 'LLM 降级' in report_md[:200]
            data = {'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
                    'trigger': trigger, 'report_md': report_md,
                    'source_errors': errors, 'llm_degraded': llm_degraded,
                    'elapsed_s': round(time.time() - t0, 1)}
            try:
                conn = _db()
                conn.execute('INSERT OR REPLACE INTO daily_report VALUES (?,?)',
                             (data['date'], json.dumps(data, ensure_ascii=False)))
                conn.execute('DELETE FROM daily_report WHERE date < '
                             'date("now","-30 day")')
                conn.commit(); conn.close()
            except Exception as e:
                errors.append(f'持久化失败: {e}')
            logger.info(f"[DailyReport] 晨报生成完成 {data['date']} "
                        f"耗时{data['elapsed_s']}s 降级={llm_degraded} 错误={errors}")
            # Telegram 推送 (非阻塞, 失败不影响主链)
            try:
                self._push_to_telegram(report_md, data['date'][:10], trigger=trigger)
            except Exception as e:
                logger.error(f"[DailyReport] Telegram 推送异常: {e}")
            return {'success': True, 'data': data, 'status': 'completed'}
        except Exception as e:
            logger.error(f"[DailyReport] 晨报链整体失败: {e}")
            return {'success': False, 'error': str(e), 'status': 'failed'}
        finally:
            with self._lock:
                self._generating = False

    def is_generating(self) -> bool:
        """当前是否正在生成 (前端轮询/手动触发去重用)"""
        with self._lock:
            return self._generating

    def get_latest(self) -> Optional[Dict]:
        """每次从 DB 读 (单行主键查询 <1ms, 无缓存: 防止 08:00 自动链/
        手动推送的新研报因进程级缓存而不可见 — 2026-09-10 修复)"""
        try:
            conn = _db()
            row = conn.execute('SELECT json FROM daily_report '
                               'ORDER BY date DESC LIMIT 1').fetchone()
            conn.close()
            if row:
                return json.loads(row[0])
        except Exception as e:
            logger.error(f"[DailyReport] 读历史失败: {e}")
        return None


class DailyReportScheduler(threading.Thread):
    """08:00 daemon 调度 (每日; 周六/日美股数据源照常可用, 周五夜盘数据)"""

    def __init__(self, service: DailyReportService, target_hour: int = 8):
        super().__init__(name='daily-report-scheduler', daemon=True)
        self.service = service
        self.target_hour = target_hour
        self._stop_event = threading.Event()
        self._last_run_date: Optional[str] = None

    def stop(self):
        self._stop_event.set()

    def run(self):
        logger.info(f"[DailyReport] 调度器已启动 (每日 {self.target_hour:02d}:00, "
                    f"daemon, 跳过当日已生成)")
        while not self._stop_event.wait(60):
            try:
                if self._stop_event.is_set():
                    break
                now = datetime.now()
                today = now.strftime('%Y-%m-%d')
                if (now.hour == self.target_hour and now.minute < 55
                        and self._last_run_date != today):
                    self._last_run_date = today
                    logger.info("[DailyReport] 08:00 定时触发晨报链")
                    self.service.generate(trigger='auto')
            except Exception as e:
                logger.error(f"[DailyReport] 调度循环异常: {e}")


daily_report_service = DailyReportService()
daily_report_scheduler = DailyReportScheduler(daily_report_service, target_hour=8)
