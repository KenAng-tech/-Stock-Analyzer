"""
modules/schemas/validator.py — 运行时数据验证

借鉴 Archify 的编译时验证器模式:
- 使用 jsonschema Draft202012Validator 加载 JSON Schema
- 模块级预编译验证器 (单例)
- 提供 validate_* 函数和 validate_and_raise 快捷函数

使用方式:
    from modules.schemas import validate_analysis, validate_prediction

    errors = validate_analysis(analysis_data)
    if errors:
        raise ValueError(f"Invalid analysis: {errors}")
"""

import json
import os
from jsonschema import Draft202012Validator
from typing import Dict, Any, List

_SCHEMA_DIR = os.path.dirname(__file__)


def _load_schema(name: str) -> Dict[str, Any]:
    """加载 JSON Schema 文件"""
    path = os.path.join(_SCHEMA_DIR, f"{name}.schema.json")
    with open(path) as f:
        return json.load(f)


def _make_validator(name: str) -> Draft202012Validator:
    """创建 Draft202012Validator 实例"""
    schema = _load_schema(name)
    return Draft202012Validator(schema)


# ── 预编译验证器 (模块级单例) ──────────────────────────────────

_analysis_validator = _make_validator("analysis_result")
_prediction_validator = _make_validator("prediction")
_model_validator = _make_validator("model_metadata")


def validate_analysis(data: Dict[str, Any]) -> List[str]:
    """
    验证分析报告数据。

    :param data: 分析报告 dict
    :return: 错误消息列表 (空列表 = 通过)
    """
    errors = list(_analysis_validator.iter_errors(data))
    return [f"{e.path[-1] if e.path else '$'}: {e.message}" for e in errors]


def validate_prediction(data: Dict[str, Any]) -> List[str]:
    """验证预测数据"""
    errors = list(_prediction_validator.iter_errors(data))
    return [f"{e.path[-1] if e.path else '$'}: {e.message}" for e in errors]


def validate_model_metadata(data: Dict[str, Any]) -> List[str]:
    """验证模型元数据"""
    errors = list(_model_validator.iter_errors(data))
    return [f"{e.path[-1] if e.path else '$'}: {e.message}" for e in errors]


def validate_and_raise(data: Dict[str, Any], name: str) -> None:
    """
    验证失败时抛出 ValueError。

    :param data: 待验证数据
    :param name: 数据类型名称 (用于错误消息)
    """
    if name == "analysis":
        errors = validate_analysis(data)
    elif name == "prediction":
        errors = validate_prediction(data)
    elif name == "model":
        errors = validate_model_metadata(data)
    else:
        errors = [f"Unknown type: {name}"]

    if errors:
        raise ValueError(f"Schema validation failed for {name}: {'; '.join(errors)}")
