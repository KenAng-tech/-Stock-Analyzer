#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Stock Analyzer 标准化错误处理
"""

from flask import jsonify
from werkzeug.exceptions import HTTPException
from modules.logger import logger


class StockAnalyzerError(Exception):
    """Stock Analyzer 基础异常"""

    def __init__(self, message: str, code: int = 500, details: dict = None):
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}


class StockNotFoundError(StockAnalyzerError):
    """股票未找到异常"""

    def __init__(self, stock_code: str):
        super().__init__(f"股票 {stock_code} 未找到", 404)


class AnalysisError(StockAnalyzerError):
    """分析异常"""

    def __init__(self, stock_code: str, reason: str):
        super().__init__(f"分析失败: {reason}", 500, {'stock_code': stock_code})


class ModelError(StockAnalyzerError):
    """模型异常"""

    def __init__(self, model_name: str, reason: str):
        super().__init__(f"模型 {model_name} 错误: {reason}", 503)


class ValidationError(StockAnalyzerError):
    """验证异常"""

    def __init__(self, field: str, message: str):
        super().__init__(f"验证失败: {field} - {message}", 400)


class RateLimitError(StockAnalyzerError):
    """限流异常"""

    def __init__(self, retry_after: int):
        super().__init__(f"请求频率超限，请 {retry_after} 秒后重试", 429, {'retry_after': retry_after})


# 全局错误处理
def register_error_handlers(app):
    """注册错误处理器"""

    @app.errorhandler(StockAnalyzerError)
    def handle_stock_analyzer_error(error: StockAnalyzerError):
        """处理Stock Analyzer异常"""
        logger.error(f"[Error] {error.message} - {error.details}")
        response = {
            'success': False,
            'error': error.message,
            'code': error.code,
            'details': error.details
        }
        return jsonify(response), error.code

    @app.errorhandler(HTTPException)
    def handle_http_error(error: HTTPException):
        """处理HTTP异常"""
        logger.error(f"[Error] {error.name} - {error.code}")
        response = {
            'success': False,
            'error': error.name,
            'code': error.code
        }
        return jsonify(response), error.code

    @app.errorhandler(Exception)
    def handle_generic_error(error: Exception):
        """处理通用异常"""
        logger.error(f"[Error] 未知错误: {str(error)}", exc_info=True)
        response = {
            'success': False,
            'error': 'Internal server error',
            'code': 500
        }
        return jsonify(response), 500