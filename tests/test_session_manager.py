"""
test_session_manager.py — SessionManager 测试

验证:
- 创建/关闭 session
- 空闲 session 自动清理
- 统计信息准确
- 单例返回同一实例
"""

import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
import sys


class TestSessionManager(unittest.TestCase):
    """SessionManager — mock scrapling session"""

    def setUp(self):
        """清除模块级单例"""
        import utils.session_manager as mod
        mod._manager = None
        # 清除 scrapling 缓存
        for key in list(sys.modules.keys()):
            if key.startswith("scrapling"):
                del sys.modules[key]

    def tearDown(self):
        import utils.session_manager as mod
        mod._manager = None

    def test_create_session(self):
        """create_session 应创建 session 并返回 session_id"""
        mock_scrapling = MagicMock()
        mock_session_obj = MagicMock()
        mock_scrapling.open_session.return_value = mock_session_obj

        with patch.dict(sys.modules, {"scrapling": mock_scrapling}):
            from utils.session_manager import SessionManager

            manager = SessionManager()
            sid = manager.create_session(session_type="dynamic")

            self.assertIsNotNone(sid)
            self.assertTrue(sid.startswith("session_"))
            mock_scrapling.open_session.assert_called_once()

    def test_create_session_custom_id(self):
        """create_session 应支持自定义 session_id"""
        mock_scrapling = MagicMock()
        mock_session_obj = MagicMock()
        mock_scrapling.open_session.return_value = mock_session_obj

        with patch.dict(sys.modules, {"scrapling": mock_scrapling}):
            from utils.session_manager import SessionManager

            manager = SessionManager()
            sid = manager.create_session(session_type="stealthy", session_id="my_custom_session")

            self.assertEqual(sid, "my_custom_session")
            mock_scrapling.open_session.assert_called_once()
            kwargs = mock_scrapling.open_session.call_args[1]
            self.assertEqual(kwargs["session_type"], "stealthy")
            self.assertEqual(kwargs["session_id"], "my_custom_session")

    def test_create_session_returns_none_on_import_error(self):
        """scrapling 未安装时 create_session 应返回 None"""
        # 确保 scrapling 不在 sys.modules 中
        with patch.dict(sys.modules, {"scrapling": None}):
            from utils.session_manager import SessionManager

            manager = SessionManager()
            result = manager.create_session()
            self.assertIsNone(result)

    def test_get_session_updates_activity(self):
        """get_session 应更新 last_active 和 requests 计数"""
        mock_scrapling = MagicMock()
        mock_session_obj = MagicMock()
        mock_scrapling.open_session.return_value = mock_session_obj

        with patch.dict(sys.modules, {"scrapling": mock_scrapling}):
            from utils.session_manager import SessionManager

            manager = SessionManager()
            sid = manager.create_session()

            before = manager._sessions[sid]
            before_requests = before["requests"]
            before_active = before["last_active"]

            # 等待 1ms 确保时间变化
            import time
            time.sleep(0.01)

            session = manager.get_session(sid)

            self.assertIs(session, mock_session_obj)
            self.assertEqual(manager._sessions[sid]["requests"], before_requests + 1)
            self.assertGreater(manager._sessions[sid]["last_active"], before_active)

    def test_get_session_returns_none_for_unknown_id(self):
        """get_session 对不存在的 session_id 应返回 None"""
        from utils.session_manager import SessionManager

        manager = SessionManager()
        result = manager.get_session("nonexistent_session")
        self.assertIsNone(result)

    def test_close_session(self):
        """close_session 应关闭 session 并从字典中移除"""
        mock_scrapling = MagicMock()
        mock_session_obj = MagicMock()
        mock_scrapling.open_session.return_value = mock_session_obj

        with patch.dict(sys.modules, {"scrapling": mock_scrapling}):
            from utils.session_manager import SessionManager

            manager = SessionManager()
            sid = manager.create_session()

            self.assertIn(sid, manager._sessions)
            result = manager.close_session(sid)

            self.assertTrue(result)
            self.assertNotIn(sid, manager._sessions)
            mock_session_obj.close.assert_called_once()

    def test_close_session_returns_false_for_unknown(self):
        """close_session 对不存在的 session 应返回 False"""
        from utils.session_manager import SessionManager

        manager = SessionManager()
        result = manager.close_session("nonexistent")
        self.assertFalse(result)

    def test_cleanup_idle_removes_idle_sessions(self):
        """cleanup_idle 应清理超过超时时间的 session"""
        mock_scrapling = MagicMock()
        mock_session_obj = MagicMock()
        mock_scrapling.open_session.return_value = mock_session_obj

        with patch.dict(sys.modules, {"scrapling": mock_scrapling}):
            from utils.session_manager import SessionManager, IDLE_TIMEOUT

            manager = SessionManager()
            sid = manager.create_session()

            # 模拟 session 空闲超过 30 分钟
            manager._sessions[sid]["last_active"] = datetime.now() - IDLE_TIMEOUT - timedelta(minutes=1)

            cleaned = manager.cleanup_idle()

            self.assertEqual(cleaned, 1)
            self.assertNotIn(sid, manager._sessions)

    def test_cleanup_idle_keeps_active_sessions(self):
        """cleanup_idle 不应清理活跃的 session"""
        mock_scrapling = MagicMock()
        mock_session_obj = MagicMock()
        mock_scrapling.open_session.return_value = mock_session_obj

        with patch.dict(sys.modules, {"scrapling": mock_scrapling}):
            from utils.session_manager import SessionManager

            manager = SessionManager()
            sid = manager.create_session()

            # session 保持活跃
            manager._sessions[sid]["last_active"] = datetime.now()

            cleaned = manager.cleanup_idle()

            self.assertEqual(cleaned, 0)
            self.assertIn(sid, manager._sessions)

    def test_get_stats(self):
        """get_stats 应返回准确的统计信息"""
        mock_scrapling = MagicMock()
        mock_scrapling.open_session.side_effect = [MagicMock(), MagicMock()]

        # 确保两次 create_session 获得不同 session_id
        call_count = [0]
        base_time = datetime(2026, 9, 17, 12, 0, 0)

        def mock_now():
            t = base_time + timedelta(seconds=call_count[0])
            call_count[0] += 1
            return t

        with patch.dict(sys.modules, {"scrapling": mock_scrapling}):
            with patch("utils.session_manager.datetime") as mock_dt:
                mock_dt.now.side_effect = mock_now
                mock_dt.timedelta = timedelta

                from utils.session_manager import SessionManager, IDLE_TIMEOUT

                manager = SessionManager()
                sid1 = manager.create_session(session_type="dynamic")
                sid2 = manager.create_session(session_type="stealthy")

                # 标记 sid1 为空闲
                manager._sessions[sid1]["last_active"] = mock_now() - IDLE_TIMEOUT - timedelta(minutes=1)

                stats = manager.get_stats()

                self.assertEqual(stats["total_sessions"], 2)
                self.assertEqual(stats["active_sessions"], 1)
                self.assertEqual(stats["idle_sessions"], 1)
                self.assertEqual(len(stats["sessions"]), 2)

    def test_close_all(self):
        """close_all 应关闭所有 session"""
        mock_scrapling = MagicMock()
        mock_session_obj = MagicMock()
        mock_scrapling.open_session.return_value = mock_session_obj

        with patch.dict(sys.modules, {"scrapling": mock_scrapling}):
            from utils.session_manager import SessionManager

            manager = SessionManager()
            sid1 = manager.create_session()
            sid2 = manager.create_session()

            manager.close_all()

            self.assertEqual(len(manager._sessions), 0)
            mock_session_obj.close.call_count  # 两次调用 (但只保留最后一次)

    def test_singleton(self):
        """get_session_manager 应返回同一单例"""
        from utils.session_manager import get_session_manager

        a = get_session_manager()
        b = get_session_manager()

        self.assertIs(a, b)

    def test_thread_safety(self):
        """创建/关闭 session 应在多线程下安全"""
        import threading
        from unittest.mock import MagicMock as MockMag

        mock_scrapling = MagicMock()
        # 每次调用返回不同 mock session
        mock_scrapling.open_session.side_effect = [MagicMock() for _ in range(50)]

        # Mock datetime 使每个 session 获得不同 ID
        base_time = datetime(2026, 9, 17, 12, 0, 0)
        current_time = [base_time]

        def mock_now():
            t = current_time[0]
            current_time[0] += timedelta(seconds=1)
            return t

        with patch.dict(sys.modules, {"scrapling": mock_scrapling}):
            with patch("utils.session_manager.datetime") as mock_dt:
                mock_dt.now.side_effect = mock_now
                mock_dt.timedelta = timedelta

                from utils.session_manager import SessionManager

                manager = SessionManager()
                sids = []
                errors = []

                def create_sessions(n):
                    try:
                        for _ in range(n):
                            sid = manager.create_session()
                            sids.append(sid)
                    except Exception as e:
                        errors.append(e)

                threads = [threading.Thread(target=create_sessions, args=(10,)) for _ in range(4)]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()

                self.assertEqual(len(errors), 0, f"线程中出现错误: {errors}")
                self.assertEqual(len(manager._sessions), 40)


if __name__ == "__main__":
    unittest.main()
