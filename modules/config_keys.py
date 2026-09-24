#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
统一 API Key / 敏感配置管理

所有敏感凭据从此模块读取，禁止在任何业务代码中硬编码。
环境变量优先，未设置时返回 None（调用方自行决定降级策略）。
"""

import os
from typing import Optional


def get_omlx_api_key() -> Optional[str]:
    """获取 OMLX API Key"""
    return os.environ.get("OMLX_API_KEY")


def get_eastmoney_ut() -> Optional[str]:
    """获取东方财富 API Token"""
    return os.environ.get("EASTMONEY_UT")


def get_openai_api_key() -> Optional[str]:
    """获取 OpenAI API Key"""
    return os.environ.get("OPENAI_API_KEY")


def get_llm_base_url() -> str:
    """获取 LLM 基础 URL"""
    return os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8080")


def get_llm_model() -> str:
    """获取默认 LLM 模型名"""
    return os.environ.get("LLM_MODEL", "GLM-4.7-Flash-MLX-8bit")


def get_database_url() -> str:
    """获取数据库连接 URL"""
    return os.environ.get("DATABASE_URL", "sqlite:///stock_analyzer.db")


# ── 便捷函数: 带日志的降级 ──────────────────────────────────

def get_api_key(service: str, env_name: str, fallback: Optional[str] = None) -> Optional[str]:
    """
    获取 API Key，按以下优先级:
    1. 环境变量
    2. fallback 参数 (仅用于测试/开发环境)
    3. None (生产环境推荐)

    注意: 生产环境应通过环境变量设置，不要使用 fallback。
    """
    value = os.environ.get(env_name)
    if value:
        return value
    if fallback:
        import logging
        logging.warning(f"[Config] {service} API Key 未通过环境变量设置，使用 fallback 值")
    return fallback
