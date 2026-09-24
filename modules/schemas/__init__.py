"""
modules/schemas/__init__.py — 结构化数据 Schema 注册中心

提供运行时验证器导入:
    from modules.schemas import validate_analysis, validate_prediction

Schema 定义文件 (JSON) 位于同目录下:
    common.schema.json
    analysis_result.schema.json
    prediction.schema.json
    model_metadata.schema.json
"""

from .validator import (
    validate_analysis,
    validate_prediction,
    validate_model_metadata,
    validate_and_raise,
)

__all__ = [
    "validate_analysis",
    "validate_prediction",
    "validate_model_metadata",
    "validate_and_raise",
]
