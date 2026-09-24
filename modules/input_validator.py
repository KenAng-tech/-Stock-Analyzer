#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
输入验证器
"""

import re
from functools import wraps
from flask import request, jsonify
from modules.logger import logger


class InputValidator:
    """输入验证器"""

    @staticmethod
    def validate_stock_code(code: str) -> bool:
        """验证股票代码格式"""
        pattern = r'^(sh|sz)\d{6}$'
        return bool(re.match(pattern, code))

    @staticmethod
    def validate_price(price: float) -> bool:
        """验证价格范围"""
        return 0 < price < 100000

    @staticmethod
    def validate_volume(volume: float) -> bool:
        """验证成交量"""
        return volume >= 0

    @staticmethod
    def validate_date_format(date_str: str) -> bool:
        """验证日期格式"""
        from datetime import datetime
        try:
            datetime.strptime(date_str, '%Y-%m-%d')
            return True
        except ValueError:
            return False


# 装饰器
def validate_stock_code(f):
    """股票代码验证装饰器"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        stock_code = kwargs.get('stock_code')
        if not InputValidator.validate_stock_code(stock_code):
            return jsonify({'error': 'Invalid stock code format'}), 400
        return f(*args, **kwargs)
    return wrapper


def validate_price(f):
    """价格验证装饰器"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        price = kwargs.get('price', 0)
        if not InputValidator.validate_price(price):
            return jsonify({'error': 'Invalid price range'}), 400
        return f(*args, **kwargs)
    return wrapper


def validate_volume(f):
    """成交量验证装饰器"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        volume = kwargs.get('volume', 0)
        if not InputValidator.validate_volume(volume):
            return jsonify({'error': 'Invalid volume'}), 400
        return f(*args, **kwargs)
    return wrapper