"""
utils/api_errors.py — 统一 API 错误格式

从 hithink-finance 借鉴的输出契约:
  ok: true/false + error.code + error.category + error.hint

比 HTTP status code 更可靠, 因为 HTTP 200 不代表业务成功。

使用方式:
    from utils.api_errors import api_error, api_ok

    # 成功响应
    return api_ok(data={"direction": "up", "confidence": 0.75})

    # 错误响应
    return api_error(
        code="MODEL_NOT_FOUND",
        category="validation",
        hint="模型 'xxx' 不存在, 可用模型: ['patchtst', 'mamba', 'diffusion']",
        status=400,
    )

    # 上游错误 (自动重试)
    return api_error(
        code="UPSTREAM_TIMEOUT",
        category="upstream",
        hint="数据源暂时不可用, 请重试",
        retryable=True,
        status=502,
    )
"""

from typing import Any, Optional
import logging

log = logging.getLogger(__name__)


def api_ok(
    data: Any = None,
    meta: dict | None = None,
    request_id: str | None = None,
) -> tuple[dict, int]:
    """
    成功响应: ok=true + data 字段。

    :param data: 业务数据
    :param meta: 元数据 (source, count, truncated, request_id 等)
    :param request_id: 链路追踪 ID
    :return: (响应字典, HTTP 状态码)
    """
    response = {"ok": True}
    if data is not None:
        response["data"] = data
    if meta:
        response["meta"] = meta
    if request_id:
        response["meta"] = response.get("meta", {})
        response["meta"]["request_id"] = request_id
    return response, 200


def api_error(
    code: str,
    category: str,
    hint: str,
    *,
    debug: dict | None = None,
    retryable: bool = False,
    status: int = 400,
) -> tuple[dict, int]:
    """
    错误响应: ok=false + error.code + error.category + error.hint。

    :param code: 错误码 (如 MODEL_NOT_FOUND, UPSTREAM_TIMEOUT)
    :param category: 错误分类 (validation / upstream / auth / internal)
    :param hint: 对用户/Agent 的友好提示
    :param debug: 调试信息 (脱敏, 不暴露 Key)
    :param retryable: 是否可重试 (429/502/503/504 优先于业务错误信封进行有界重试)
    :param status: HTTP 状态码
    :return: (响应字典, HTTP 状态码)
    """
    error = {
        "code": code,
        "category": category,
        "hint": hint,
    }
    if retryable:
        error["retryable"] = True
    if debug:
        error["diagnostics"] = debug

    response = {
        "ok": False,
        "error": error,
    }
    return response, status


def api_not_found(hint: str = "资源未找到") -> tuple[dict, int]:
    """快捷方法: 404 响应"""
    return api_error(
        code="NOT_FOUND",
        category="validation",
        hint=hint,
        status=404,
    )


def api_internal_error(hint: str = "内部错误", debug: dict | None = None) -> tuple[dict, int]:
    """快捷方法: 500 响应"""
    return api_error(
        code="INTERNAL_ERROR",
        category="internal",
        hint=hint,
        debug=debug,
        status=500,
    )


def api_upstream_error(hint: str = "上游服务不可用", retryable: bool = True) -> tuple[dict, int]:
    """快捷方法: 502/503 上游错误"""
    return api_error(
        code="UPSTREAM_ERROR",
        category="upstream",
        hint=hint,
        retryable=retryable,
        status=502,
    )


def api_rate_limited(hint: str = "请求过于频繁, 请稍后重试") -> tuple[dict, int]:
    """快捷方法: 429 限流"""
    return api_error(
        code="RATE_LIMITED",
        category="upstream",
        hint=hint,
        retryable=True,
        status=429,
    )


def parse_response(result: dict) -> tuple[bool, Any, str | None]:
    """
    解析 hithink-finance 风格的响应信封。

    :param result: 响应字典
    :return: (success, data, error_hint)
    """
    if result.get("ok") is True:
        return True, result.get("data"), None
    else:
        err = result.get("error", {})
        hint = err.get("hint", "未知错误")
        return False, None, hint
