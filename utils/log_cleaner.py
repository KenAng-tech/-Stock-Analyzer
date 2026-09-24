"""
utils/log_cleaner.py — 日志清理工具

从 hithink-finance 借鉴的日志管理最佳实践:
  - 限制单文件 10MB, 保留 5 个备份
  - 自动清理超过 30 天的旧日志
  - 启动时检查日志大小, 超过阈值自动清理

使用方式:
    from utils.log_cleaner import clean_old_logs, get_log_stats

    # 启动时调用
    stats = get_log_stats()
    if stats['total_size_mb'] > 100:
        clean_old_logs()
        logger.info(f"[log_cleaner] 已清理旧日志, 剩余 {stats['total_size_mb']:.1f}MB")
"""

import os
import logging
from pathlib import Path
from datetime import datetime, timedelta

log = logging.getLogger(__name__)

# 日志目录
LOGS_DIR = Path("/Users/claw/stock_analyzer/logs")

# 清理规则
MAX_AGE_DAYS = 30       # 超过 30 天的日志自动删除
MAX_TOTAL_SIZE_MB = 200 # 总日志大小超过此值时触发清理
MAX_SINGLE_FILE_MB = 50 # 单文件超过此值时警告


def get_log_stats() -> dict:
    """获取日志统计信息"""
    if not LOGS_DIR.exists():
        return {"exists": False, "total_size_mb": 0, "file_count": 0}

    files = list(LOGS_DIR.glob("*.log*"))
    total_size = sum(f.stat().st_size for f in files if f.is_file())
    old_files = [f for f in files if (datetime.now() - datetime.fromtimestamp(f.stat().st_mtime)).days > MAX_AGE_DAYS]

    return {
        "exists": True,
        "total_size_mb": total_size / (1024 * 1024),
        "file_count": len(files),
        "old_file_count": len(old_files),
        "largest_file": max((f.stat().st_size for f in files if f.is_file()), default=0) / (1024 * 1024),
    }


def clean_old_logs(keep_days: int = MAX_AGE_DAYS, max_files: int = 20) -> dict:
    """
    清理旧日志。

    :param keep_days: 保留天数
    :param max_files: 每个日志前缀最多保留的文件数
    :return: {"deleted": int, "freed_mb": float}
    """
    if not LOGS_DIR.exists():
        return {"deleted": 0, "freed_mb": 0}

    deleted = 0
    freed = 0
    cutoff = datetime.now() - timedelta(days=keep_days)

    for f in LOGS_DIR.glob("*.log*"):
        if not f.is_file():
            continue

        mtime = datetime.fromtimestamp(f.stat().st_mtime)
        if mtime < cutoff:
            try:
                freed += f.stat().st_size
                f.unlink()
                deleted += 1
                log.info(f"[log_cleaner] 已删除: {f.name} (创建于 {mtime:%Y-%m-%d})")
            except OSError as e:
                log.warning(f"[log_cleaner] 删除失败 {f.name}: {e}")

    # 清理每个前缀的旧备份 (保留最新的 max_files 个)
    from collections import defaultdict
    prefixes = defaultdict(list)
    for f in LOGS_DIR.glob("*.log*"):
        if f.is_file():
            # 提取前缀: stock_analyzer.log.1 → stock_analyzer
            name = f.stem
            if "." in name:
                name = name.rsplit(".", 1)[0]
            prefixes[name].append(f)

    for prefix, files in prefixes.items():
        if len(files) > max_files:
            files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            for f in files[max_files:]:
                try:
                    freed += f.stat().st_size
                    f.unlink()
                    deleted += 1
                    log.info(f"[log_cleaner] 清理备份: {f.name}")
                except OSError as e:
                    log.warning(f"[log_cleaner] 清理失败 {f.name}: {e}")

    return {"deleted": deleted, "freed_mb": freed / (1024 * 1024)}


def check_log_health() -> dict:
    """检查日志健康状态"""
    stats = get_log_stats()

    issues = []
    if stats["total_size_mb"] > MAX_TOTAL_SIZE_MB:
        issues.append(f"总日志大小 {stats['total_size_mb']:.1f}MB > 阈值 {MAX_TOTAL_SIZE_MB}MB")
    if stats["largest_file"] > MAX_SINGLE_FILE_MB:
        issues.append(f"最大日志文件 {stats['largest_file']:.1f}MB > 阈值 {MAX_SINGLE_FILE_MB}MB")
    if stats["old_file_count"] > 10:
        issues.append(f"{stats['old_file_count']} 个旧日志文件 (> {MAX_AGE_DAYS} 天)")

    return {
        "healthy": len(issues) == 0,
        "stats": stats,
        "issues": issues,
    }
