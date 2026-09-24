#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Stock Analyzer - Robust Server Startup (2026-08-14 增强版)

boot/ 模块:
- boot.dependency_check: 依赖版本检查 + 配置校验
- boot.pid_manager: PID 文件管理 (含启动锁)
- boot.module_importer: 模块导入 (解耦 app.py + 重试)
- boot.optimization_bootstrap: Phase 1 优化模块注入 (完整 27 模块)
- boot.monitor_starter: 监控线程启动 (统一管理)
- boot.health_check: 启动后健康检查 (异步 + 等待 ready)
- boot.shutdown_handler: 优雅关闭 (带超时保护)
- boot.model_cleanup: 模型文件清理

启动流程:
    1. 依赖检查 + 配置校验
    2. 注册信号处理器 (统一由 shutdown_handler 管理)
    3. 导入 app.py 模块 (带重试)
    4. 注入 Phase 1 优化模块 (完整 27 个)
    5. 启动监控线程
    6. 写入 PID (含启动锁检测)
    7. 异步健康检查
    8. 启动 SocketIO 服务器
"""

import sys
import os
import time

# 行缓冲 stdout (2026-09-04): nohup 重定向到文件时 stdout 默认块缓冲 (8KB),
# 故障现场日志会延迟落盘; 行缓冲保证每行即时 flush, 等效 PYTHONUNBUFFERED=1
try:
    sys.stdout.reconfigure(line_buffering=True)
except (AttributeError, ValueError):
    pass  # stdout 非标准流 (如已重定向) 时跳过

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.logger import logger

# ── 启动流程 ──────────────────────────────────────────────────
from boot import (
    check_dependencies,
    validate_config,
    import_app_modules,
    bootstrap_optimization,
    start_monitors,
    post_startup_health_check,
    register_shutdown_handlers,
)
from boot.pid_manager import write_pid, cleanup_pid
from boot.model_cleanup import clean_old_models, get_model_stats


def _load_server_config() -> dict:
    """
    从 config.py 加载服务器配置

    Returns:
        {'host': str, 'port': int}
    """
    # 2026-09-14 安全修复: 兜底 host 由 0.0.0.0 改为 127.0.0.1 (fail-closed)。
    # config.py 损坏/被覆盖时服务不得静默回到全网卡监听 (09-14 启动链审查结论)。
    try:
        from config import config as app_config
        server = getattr(app_config, 'get', lambda k, d=None: d)('server', {})
        host = server.get('host', '127.0.0.1')
        port = int(server.get('port', 5002))
        return {'host': host, 'port': port}
    except Exception as e:
        logger.warning(f"[Main] 读取 config.py 失败: {e}，使用安全默认配置 (127.0.0.1)")
        return {'host': '127.0.0.1', 'port': 5002}


def main():
    """主启动流程"""
    start_time = time.time()

    # 0. 凭据检查 (2026-09-16: 从 hithink-finance 借鉴)
    try:
        from config.credentials import get_credentials_status
        cred_status = get_credentials_status()
        if not cred_status.get('configured', False):
            logger.warning("[Main] HITHINK_FINANCE_API_KEY 未配置 (金融数据服务不可用)")
        else:
            logger.info(f"[Main] API Key 已配置 (env={cred_status['environment_configured']}, file={cred_status['file_configured']})")
    except Exception as e:
        logger.warning(f"[Main] 凭据检查失败: {e}")

    # 1. 依赖检查 + 配置校验
    check_dependencies()
    config_warnings = validate_config()
    if config_warnings:
        logger.info(f"[Main] 配置检查完成: {len(config_warnings)} 个警告")

    # 2. 注册信号处理器 (PID 清理 + 优雅关闭)
    register_shutdown_handlers(extra_callbacks=[cleanup_pid])

    # 3. 导入 app.py 模块 (带重试)
    modules, error = import_app_modules(max_retries=3, retry_delay=1.0)
    if error:
        logger.error(f"[Main] 模块导入失败: {error}")
        sys.exit(1)

    app = modules['app']
    data_fetcher = modules.get('data_fetcher')
    analysis_engine = modules.get('analysis_engine')
    socketio = modules['socketio']
    alert_engine = modules.get('alert_engine')
    websocket_handler = modules.get('websocket_handler')

    # 4. 注入 Phase 1 优化模块 (完整 27 个 SOTA 模块)
    import app as app_module
    bootstrap_result = bootstrap_optimization(
        app,
        data_fetcher=data_fetcher,
        analysis_engine=analysis_engine,
        app_module=app_module,
    )

    # 5. 启动监控线程 (2026-09-09: + 每日晨报调度器, 08:00 daemon)
    try:
        from modules.daily_report_service import daily_report_scheduler as _drs
    except Exception as e:
        logger.warning(f"[Main] 晨报调度器导入失败: {e}")
        _drs = None
    monitor_status = start_monitors(
        websocket_handler=websocket_handler,
        alert_engine=alert_engine,
        data_fetcher=data_fetcher,
        ws_interval=5,
        alert_interval=30,
        daily_report_scheduler=_drs,
    )

    # 6. 写入 PID (含启动锁检测)
    write_pid()

    # 7. 日志健康检查 (2026-09-16: 防止长期运行占满磁盘)
    try:
        from utils.log_cleaner import check_log_health
        log_health = check_log_health()
        if not log_health.get('healthy', True):
            logger.warning(f"[Main] 日志健康: {log_health.get('issues', [])}")
        else:
            logger.info(f"[Main] 日志健康: {log_health['stats']['total_size_mb']:.1f}MB, {log_health['stats']['file_count']} 文件")
    except Exception as e:
        logger.warning(f"[Main] 日志健康检查失败: {e}")

    # 8. 启动前模型清理 (dry_run 模式)
    model_stats = get_model_stats()
    if model_stats['old_files_count'] > 0:
        deleted, errors = clean_old_models(dry_run=True)
        logger.info(
            f"[Main] 模型目录: {model_stats['total_files']} 文件, "
            f"{model_stats['total_size_mb']:.1f}MB, "
            f"其中 {model_stats['old_files_count']} 个可清理"
        )

    # 8. 启动 banner
    elapsed = time.time() - start_time
    _print_startup_banner(modules, bootstrap_result, monitor_status, model_stats, elapsed)

    # 9. 异步健康检查
    post_startup_health_check(max_wait=30)

    # 10. 启动 SocketIO 服务器
    server_config = _load_server_config()
    logger.info(
        f"[Main] 启动 SocketIO 服务器 "
        f"({server_config['host']}:{server_config['port']}, threading mode)..."
    )
    try:
        socketio.run(
            app,
            host=server_config['host'],
            port=server_config['port'],
            debug=False,
            allow_unsafe_werkzeug=True,
            use_reloader=False,
        )
    except KeyboardInterrupt:
        logger.info("[Main] 收到 Ctrl+C，正在关闭...")
    except Exception as e:
        logger.error(f"[Main] 服务器启动失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def _print_startup_banner(modules, bootstrap_result, monitor_status, model_stats, elapsed):
    """打印启动 banner"""
    module_count = len([f for f in os.listdir('modules') if f.endswith('.py') and f != '__init__.py'])

    print("=" * 70)
    print("  Stock Analyzer System - Enhanced Startup (2026-08-14)")
    print("=" * 70)
    print(f"  Startup Time: {elapsed:.2f}s")
    print(f"  Server: http://127.0.0.1:5002")
    print(f"  API: http://127.0.0.1:5002/api/stock/sz300620")
    print(f"  WebSocket: Enabled (threading mode)")
    print(f"  Async Mode: threading (NOT eventlet)")
    print(f"  WebSocket Polling: {'5s (active)' if monitor_status['websocket'] else '5s (FAILED)'}")
    print(f"  Alert Monitoring: {'30s (active)' if monitor_status['alert'] else '30s (FAILED)'}")
    print(f"  Phase 1 Optimization: {'Active' if bootstrap_result.get('success') else 'FAILED'}")
    print(f"  SOTA Modules Registered: {bootstrap_result.get('registered_resources', 0)}/27")
    print(f"  Health Check: http://127.0.0.1:5002/api/health")
    print(f"  Optimization Status: http://127.0.0.1:5002/api/optimization/status")
    print(f"  Modules loaded: {module_count} files")
    print(f"  Model files: {model_stats['total_files']} ({model_stats['total_size_mb']:.1f}MB)")
    print(f"  SOTA models: PatchTST, Mamba, Diffusion, DRL, Moirai, GNN, Conformal")
    print(f"  Signal handlers: Unified (shutdown_handler)")
    print(f"  Graceful shutdown: Timeout-protected")
    print("=" * 70)
    print("\nStarting server...")


if __name__ == '__main__':
    main()