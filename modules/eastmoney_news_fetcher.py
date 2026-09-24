#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""eastmoney_news_fetcher.py — 东财个股新闻旁路 (2026-09-11 决策链 B 后续)

背景 (链尸检): 主链 data_fetcher.get_stock_news(:420) 的 search-api-web 旧版
参数 (type=JSON 标量, 无 param.cmsArticleWebOld 嵌套层) 已恒返 200 空 body =
新闻取数断链; llm-sentiment 面板 n:0 / 共识 sentiment 票 0.3 均是其产物
(09-10 "bert 假链恒 neutral" 教训同族 — 200+空 body 假无数据)。

本模块 = 主链取空后的二次取数 (旁路不替换主链):
  - 参数 = 新版 JSONP (akshare 上游同款形; 2026-09-11 实测 10 条真新闻,
    keyword=股票名 + 纯代码双试)
  - <em> 标签自剥 (akshare.stock_news_em 在本项目 Python 3.14 抛
    "Invalid escape sequence" — 其内部正则踩 Py3.14 收紧, 故不复调它,
    自实现解析)
  - 纯 urllib (同项目主链形态, 零新依赖; Scrapling FetcherSession 作为
    后续备选传输层 — 本次探针已用其证伪"反爬拦截"假说)
  - 全路径吞异常 → [] (数据源故障 ≠ 决策链故障, llm-sentiment/共识票不受拖累)
"""

import json
import re
import time
import urllib.parse
import urllib.request
from typing import Dict, List

from modules.logger import logger
from utils.outbound_guard import host_throttle

_JSONP_RE = re.compile(r'^[a-zA-Z_$][\w$]*\((.*)\)\s*$', re.S)
_TAG_RE = re.compile(r'<[^>]+>')
_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                  'Apple/Safari/537.36 Chrome/141.0.0.0 Safari/537.36',
    'Referer': 'https://so.eastmoney.com/',
    'Accept': '*/*',
}


def fetch_news_eastmoney(stock_code: str, timeout: int = 15,
                         page_size: int = 10) -> List[Dict]:
    """东财 search-api-web 新闻 JSONP (新版参数) → [{'title','time','url'}].

    Args:
        stock_code: 'sz300620'/'603777' 等; keyword 先试纯代码, 空再试股票名?
                    (名映射无依赖来源, 本版用纯代码 — akshare 上游同形)
        timeout: 单请求超时 (同项目数据链 25s 纪律的一半, 旁路宁短勿长)
        page_size: 1-10

    Returns:
        [{'title','time','url'}] 或 [] (任何异常/空/坏形 → [], 恒不抛)
    """
    try:
        keyword = re.sub(r'^(sz|sh|bj|SZ|SH|BJ)', '', (stock_code or '').strip())
        if not keyword:
            return []
        inner = {
            "uid": "", "keyword": keyword,
            "type": ["cmsArticleWebOld"],          # list = 新版 (主链旧版标量已死)
            "client": "web", "clientType": "web", "clientVersion": "curr",
            "param": {"cmsArticleWebOld": {        # 嵌套层 = 旧版整链死的正主
                "searchScope": "default", "sort": "default",
                "pageIndex": 1, "pageSize": page_size,
                "preTag": "<em>", "postTag": "</em>",
            }},
        }
        query = urllib.parse.urlencode({
            'cb': 'jQuery', 'param': json.dumps(inner, ensure_ascii=False),
            '_': str(int(time.time() * 1000)),
        })
        # P0-3 (2026-09-20): 同 host 节流 — intel 链 4-worker 并行同 host burst 串行化, 防 search-api 限频
        host_throttle.acquire('search-api-web.eastmoney.com', host_key='search-api-web.eastmoney.com')
        req = urllib.request.Request(
            f"https://search-api-web.eastmoney.com/search/jsonp?{query}",
            headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode('utf-8', errors='replace')
        if not body:
            return []
        m = _JSONP_RE.match(body.strip())
        payload = m.group(1) if m else body.strip()
        data = json.loads(payload)
        arts = (data.get('result') or {}).get('cmsArticleWebOld') or []
        out = []
        for a in arts:
            title = _TAG_RE.sub('', str(a.get('title', ''))).strip()
            if title:
                out.append({'title': title,
                            'time': str(a.get('date', ''))[:19],
                            'url': str(a.get('url', ''))})
        logger.info(f"[NewsBypass] {stock_code}: 旁路取到 {len(out)} 条新闻")
        return out
    except Exception as e:
        logger.debug(f"[NewsBypass] {stock_code} 旁路失败 (已吞): {e}")
        return []
