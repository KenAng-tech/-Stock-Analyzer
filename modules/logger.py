"""
日志系统
结构化日志，支持文件和控制台输出
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from datetime import datetime
from typing import Optional, Dict
import json

from config import config
from modules.log_context import get_trace


class StructuredFormatter(logging.Formatter):
    """结构化日志格式化器"""

    def format(self, record):
        log_data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'level': record.levelname,
            'module': record.name,
            'message': record.getMessage(),
        }
        # 2026-09-20 整合②: 链式追踪 — 活跃 trace 注入每行 (PanWatch log_context 式),
        # 一 grep trace 串全链 (生成→持久化→推送→送达); 无 trace 零污染
        _tr = get_trace()
        if _tr:
            log_data['trace'] = _tr
        if hasattr(record, 'extra_data'):
            log_data['extra'] = record.extra_data
        if record.exc_info and record.exc_info[0]:
            log_data['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_data, ensure_ascii=False)


class Logger:
    """日志管理器"""
    
    _instance = None
    _loggers = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._setup()
    
    def _setup(self):
        """初始化日志系统"""
        log_config = config.get('logging', {})
        log_level = getattr(logging, log_config.get('level', 'INFO'))
        log_file = log_config.get('file', 'logs/stock_analyzer.log')
        
        # 确保日志目录存在
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # 根日志器
        self.logger = logging.getLogger('stock_analyzer')
        self.logger.setLevel(log_level)
        
        # 清除已有handler
        self.logger.handlers.clear()
        
        # 2026-09-14: 移除 StreamHandler(sys.stdout) 去重 —
        # 原双写: 每条 log 同时写 stdout (launchd/重定向到 5002.log, 无 rotation,
        # 实测 6.8KB/min 无限膨胀) 和 stock_analyzer.log (本文件 10MB×5 轮转链)。
        # 去重后 stdout 仅剩 banner/print/traceback, 全量日志单一来源 = 轮转文件。
        
        # 文件handler
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=log_config.get('max_bytes', 10 * 1024 * 1024),
            backupCount=log_config.get('backup_count', 5),
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(StructuredFormatter())
        self.logger.addHandler(file_handler)
    
    def get_logger(self, name: str) -> logging.Logger:
        """获取子日志器"""
        if name not in self._loggers:
            self._loggers[name] = logging.getLogger(f'stock_analyzer.{name}')
        return self._loggers[name]
    
    def log_extra(self, logger, level, message, extra: Optional[Dict] = None):
        """记录带额外数据的日志"""
        record = logger.makeRecord(
            logger.name, level, '(unknown)', 0, message, (), None
        )
        if extra:
            record.extra_data = extra
        logger.handle(record)
    
    def info(self, message: str, extra: Optional[Dict] = None, **kwargs):
        self.log_extra(self.logger, logging.INFO, message, extra)

    def warning(self, message: str, extra: Optional[Dict] = None, **kwargs):
        self.log_extra(self.logger, logging.WARNING, message, extra)

    def error(self, message: str, extra: Optional[Dict] = None, **kwargs):
        # exc_info=True (标准 logging 习惯, errors.py 等 10 处使用) → 真因 traceback 并入消息, 不静默吞掉
        if kwargs.get('exc_info'):
            import traceback
            message = f"{message} | {traceback.format_exc(limit=2).strip()}"
        self.log_extra(self.logger, logging.ERROR, message, extra)

    def debug(self, message: str, extra: Optional[Dict] = None, **kwargs):
        self.log_extra(self.logger, logging.DEBUG, message, extra)


# 全局日志实例
logger = Logger()
