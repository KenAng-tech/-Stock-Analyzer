"""
utils/session_manager.py — Scrapling Session 管理器

管理所有 Scrapling browser sessions 的生命周期:
  - 创建/关闭 session
  - 自动清理空闲 session
  - 会话统计

使用方式:
    from utils.session_manager import SessionManager, get_session_manager

    manager = SessionManager()

    # 创建 session
    session = manager.create_session()

    # 获取所有 session 统计
    stats = manager.get_stats()

    # 清理空闲 session
    manager.cleanup_idle()

    # 关闭所有 session (应用关闭时)
    manager.close_all()
"""

import logging
import threading
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta

log = logging.getLogger(__name__)

# Session 超时: 30 分钟无活动
IDLE_TIMEOUT = timedelta(minutes=30)


class SessionManager:
    """Scrapling browser session 管理器"""

    def __init__(self):
        self._sessions: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def create_session(self, session_type: str = 'dynamic', session_id: str = None) -> Optional[str]:
        """
        创建新的 browser session。

        :param session_type: 'dynamic' 或 'stealthy'
        :param session_id: 自定义 session ID (可选)
        :return: session_id 或 None
        """
        try:
            from scrapling import open_session
            sid = session_id or f"session_{int(datetime.now().timestamp())}"

            session = open_session(
                session_type=session_type,
                session_id=sid,
                headless=True,
            )

            with self._lock:
                self._sessions[sid] = {
                    'session': session,
                    'type': session_type,
                    'created_at': datetime.now(),
                    'last_active': datetime.now(),
                    'requests': 0,
                }

            log.info(f"[SessionManager] Created session {sid} (type={session_type})")
            return sid

        except ImportError:
            log.error("[SessionManager] scrapling 未安装")
            return None
        except Exception as e:
            log.error(f"[SessionManager] 创建 session 失败: {e}")
            return None

    def get_session(self, session_id: str) -> Optional[Any]:
        """获取 session 对象"""
        with self._lock:
            info = self._sessions.get(session_id)
            if info:
                info['last_active'] = datetime.now()
                info['requests'] += 1
                return info['session']
        return None

    def close_session(self, session_id: str) -> bool:
        """关闭指定 session"""
        with self._lock:
            info = self._sessions.pop(session_id, None)
            if info:
                try:
                    info['session'].close()
                    log.info(f"[SessionManager] Closed session {session_id}")
                    return True
                except Exception as e:
                    log.warning(f"[SessionManager] 关闭 session 失败: {e}")
                    return False
        return False

    def cleanup_idle(self) -> int:
        """清理空闲 session，返回清理数量"""
        now = datetime.now()
        to_remove = []

        with self._lock:
            for sid, info in self._sessions.items():
                if (now - info['last_active']) > IDLE_TIMEOUT:
                    to_remove.append(sid)

        for sid in to_remove:
            self.close_session(sid)

        if to_remove:
            log.info(f"[SessionManager] 清理了 {len(to_remove)} 个空闲 session")

        return len(to_remove)

    def get_stats(self) -> Dict[str, Any]:
        """获取 session 统计"""
        with self._lock:
            total = len(self._sessions)
            idle = 0
            for info in self._sessions.values():
                if (datetime.now() - info['last_active']) > IDLE_TIMEOUT:
                    idle += 1

            return {
                'total_sessions': total,
                'idle_sessions': idle,
                'active_sessions': total - idle,
                'sessions': [
                    {
                        'id': sid,
                        'type': info['type'],
                        'created_at': info['created_at'].isoformat(),
                        'last_active': info['last_active'].isoformat(),
                        'requests': info['requests'],
                        'idle': (datetime.now() - info['last_active']) > IDLE_TIMEOUT,
                    }
                    for sid, info in self._sessions.items()
                ],
            }

    def close_all(self):
        """关闭所有 session"""
        with self._lock:
            session_ids = list(self._sessions.keys())

        for sid in session_ids:
            self.close_session(sid)

        log.info("[SessionManager] 所有 session 已关闭")


# ── 模块级单例 ──────────────────────────────────────────────────

_manager = None


def get_session_manager() -> SessionManager:
    """获取 SessionManager 单例"""
    global _manager
    if _manager is None:
        _manager = SessionManager()
    return _manager
