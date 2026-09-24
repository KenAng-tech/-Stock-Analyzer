"""
utils/scrapling_fallback.py — Scrapling 备用数据源 + Session 管理

当 AKShare / 同花顺 API 不可用时，用 Scrapling 浏览器抓取作为 fallback:
  - 东方财富股吧新闻
  - 东方财富实时行情
  - 新浪财经数据

Session 管理:
  - 启动时创建 browser session
  - 定期清理空闲 session
  - 关闭时释放所有资源

使用方式:
    from utils.scrapling_fallback import ScraplingFallback, get_scrapling_session

    fallback = ScraplingFallback()

    # 抓取股吧新闻
    news = fallback.eastmoney_guba_news("300620")

    # 获取实时行情
    quote = fallback.eastmoney_quote("300620")
"""

import logging
import time
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta

log = logging.getLogger(__name__)

# Session 超时: 30 分钟无活动则清理
SESSION_IDLE_TIMEOUT = 1800  # 30 分钟


class ScraplingFallback:
    """Scrapling 备用数据源"""

    def __init__(self):
        self._session = None
        self._session_created_at = None
        self._last_activity = None

    def _get_session(self):
        """获取或创建 browser session"""
        if self._session is None:
            try:
                from scrapling import open_session
                self._session = open_session(session_type='dynamic')
                self._session_created_at = datetime.now()
                self._last_activity = datetime.now()
                log.info("[ScraplingFallback] Browser session created")
            except ImportError:
                log.error("[ScraplingFallback] scrapling 未安装")
                return None
            except Exception as e:
                log.error(f"[ScraplingFallback] 创建 session 失败: {e}")
                return None
        else:
            # 检查是否超时
            if (datetime.now() - self._last_activity).seconds > SESSION_IDLE_TIMEOUT:
                self._close_session()
                return self._get_session()
        self._last_activity = datetime.now()
        return self._session

    def _close_session(self):
        """关闭 browser session"""
        if self._session is not None:
            try:
                self._session.close()
                log.info("[ScraplingFallback] Browser session closed")
            except Exception as e:
                log.warning(f"[ScraplingFallback] 关闭 session 失败: {e}")
            finally:
                self._session = None
                self._session_created_at = None
                self._last_activity = None

    def eastmoney_quote(self, stock_code: str) -> Optional[Dict[str, Any]]:
        """
        从东方财富获取实时行情 (fallback)。

        :param stock_code: 股票代码 (如 "300620")
        :return: {name, price, change, change_pct, ...} 或 None
        """
        session = self._get_session()
        if session is None:
            return None

        try:
            # 东方财富行情 API
            market = 1 if stock_code.startswith('6') else 0
            url = f"https://push2.eastmoney.com/api/qt/stock/get?secid={market}.{stock_code}&fields=f43,f44,f45,f46,f47,f48,f170"

            result = session.fetch(
                url,
                extraction_type='json',
                main_content_only=True,
                timeout=10,
            )

            if result and result.get('data'):
                d = result['data']
                return {
                    'name': d.get('name', ''),
                    'code': stock_code,
                    'price': d.get('f43', 0) / 100 if d.get('f43') else 0,
                    'change': d.get('f44', 0) / 100 if d.get('f44') else 0,
                    'change_pct': d.get('f45', 0) / 100 if d.get('f45') else 0,
                    'high': d.get('f46', 0) / 100 if d.get('f46') else 0,
                    'low': d.get('f47', 0) / 100 if d.get('f47') else 0,
                    'open': d.get('f48', 0) / 100 if d.get('f48') else 0,
                    'volume': d.get('f170', 0),
                    'timestamp': datetime.now().isoformat(),
                }
            return None

        except Exception as e:
            log.error(f"[ScraplingFallback] 东方财富行情抓取失败: {e}")
            return None

    def eastmoney_guba_news(self, stock_code: str, limit: int = 10) -> Optional[List[Dict]]:
        """
        从东方财富股吧获取新闻/帖子 (fallback)。

        :param stock_code: 股票代码
        :param limit: 返回数量
        :return: [{title, url, time, author}, ...] 或 None
        """
        session = self._get_session()
        if session is None:
            return None

        try:
            # 东方财富股吧列表页
            url = f"https://guba.eastmoney.com/list,{stock_code}.html"

            result = session.fetch(
                url,
                extraction_type='markdown',
                main_content_only=True,
                timeout=15,
            )

            if result and result.get('content'):
                # 简单解析 markdown 中的帖子列表
                content = result['content']
                posts = self._parse_guba_posts(content, limit)
                return posts

            return None

        except Exception as e:
            log.error(f"[ScraplingFallback] 股吧新闻抓取失败: {e}")
            return None

    def _parse_guba_posts(self, markdown: str, limit: int) -> List[Dict]:
        """简单解析股吧 markdown 中的帖子列表"""
        posts = []
        lines = markdown.split('\n')
        for line in lines:
            # 匹配帖子标题模式: **标题** 或 [标题](url)
            if '**' in line and 'http' in line:
                # 提取标题和链接
                import re
                title_match = re.search(r'\*\*(.+?)\*\*', line)
                url_match = re.search(r'\]\((https?://[^)]+)\)', line)
                if title_match and url_match:
                    posts.append({
                        'title': title_match.group(1),
                        'url': url_match.group(1),
                        'time': '',
                        'author': '',
                    })
                    if len(posts) >= limit:
                        break
        return posts

    def cleanup(self):
        """清理资源"""
        self._close_session()
        log.info("[ScraplingFallback] 清理完成")


# ── 模块级单例 ──────────────────────────────────────────────────

_fallback = None


def get_scrapling_fallback() -> ScraplingFallback:
    """获取 ScraplingFallback 单例"""
    global _fallback
    if _fallback is None:
        _fallback = ScraplingFallback()
    return _fallback
