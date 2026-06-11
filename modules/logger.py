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


class StructuredFormatter(logging.Formatter):
    """结构化日志格式化器"""
    
    def format(self, record):
        log_data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'level': record.levelname,
            'module': record.name,
            'message': record.getMessage(),
        }
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
        
        # 控制台handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_format = logging.Formatter(
            log_config.get('format', '%(asctime)s [%(levelname)s] %(name)s: %(message)s')
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        
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
    
    def info(self, message: str, extra: Optional[Dict] = None):
        self.log_extra(self.logger, logging.INFO, message, extra)
    
    def warning(self, message: str, extra: Optional[Dict] = None):
        self.log_extra(self.logger, logging.WARNING, message, extra)
    
    def error(self, message: str, extra: Optional[Dict] = None):
        self.log_extra(self.logger, logging.ERROR, message, extra)
    
    def debug(self, message: str, extra: Optional[Dict] = None):
        self.log_extra(self.logger, logging.DEBUG, message, extra)


# 全局日志实例
logger = Logger()
