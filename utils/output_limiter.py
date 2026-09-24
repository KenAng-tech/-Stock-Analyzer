"""
utils/output_limiter.py — 大结果落盘工具

从 hithink-finance 借鉴的大结果落盘纪律:
  - 全市场/分页/长区间数据必须通过 --output 落盘
  - stdout 只返回路径 + count + 摘要

使用方式:
    from utils.output_limiter import limit_output, export_to_file

    # 在 API 端点中使用
    @bp.route('/api/factors/panel')
    def factors_panel():
        data = factor_engine.compute_panel()  # 可能很大
        return limit_output(data, prefix='factors_panel')
"""

import os
import json
import uuid
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# 输出目录
OUTPUT_DIR = Path("/tmp/stock_analyzer_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 阈值: 超过此大小的数据自动落盘
SIZE_THRESHOLD = 1000  # 行数
KEY_THRESHOLD = 50     # 字典键数
CHAR_THRESHOLD = 50000  # 字符数


def _estimate_size(data: Any) -> int:
    """估算数据大小 (行数或键数)"""
    if isinstance(data, (list, tuple)):
        return len(data)
    elif isinstance(data, dict):
        return len(data.get("data", data)) if "data" in data else len(data)
    elif isinstance(data, str):
        return len(data)
    return 0


def _estimate_char_size(data: Any) -> int:
    """估算数据的字符数"""
    try:
        return len(json.dumps(data, ensure_ascii=False))
    except (TypeError, ValueError):
        return 0


def limit_output(
    data: Any,
    prefix: str = "output",
    max_rows: int | None = None,
) -> dict:
    """
    智能限制输出: 大数据自动落盘, 小数据直接返回。

    :param data: 响应数据
    :param prefix: 输出文件前缀
    :param max_rows: 最大返回行数 (超过则截断)
    :return: {"ok": True, "data": 摘要或完整数据, "meta": {...}}
    """
    size = _estimate_size(data)
    char_size = _estimate_char_size(data)

    # 小数据: 直接返回
    if size <= (max_rows or SIZE_THRESHOLD) and char_size <= CHAR_THRESHOLD:
        return {
            "ok": True,
            "data": data,
            "meta": {"source": "inline", "truncated": False},
        }

    # 大数据: 落盘
    file_path = _export_to_file(data, prefix)
    row_count = size if isinstance(data, (list, tuple)) else (
        len(data.get("data", data)) if isinstance(data, dict) else 0
    )

    return {
        "ok": True,
        "data": {
            "path": str(file_path),
            "count": row_count,
            "size_chars": char_size,
            "note": "大结果已落盘, 完整数据请读取文件",
        },
        "meta": {"source": "file", "truncated": True},
    }


def export_to_file(
    data: Any,
    prefix: str = "output",
    file_format: str = "json",
) -> str:
    """
    导出数据到文件。

    :param data: 要导出的数据
    :param prefix: 文件名前缀
    :param file_format: 文件格式 (json/parquet/csv)
    :return: 文件路径
    """
    file_path = _export_to_file(data, prefix, file_format)
    return str(file_path)


def _export_to_file(
    data: Any,
    prefix: str = "output",
    file_format: str = "json",
) -> Path:
    """内部: 导出数据到文件"""
    filename = f"{prefix}_{uuid.uuid4().hex[:8]}.{file_format}"
    file_path = OUTPUT_DIR / filename

    try:
        if file_format == "json":
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        elif file_format == "csv":
            import pandas as pd
            if isinstance(data, dict) and "data" in data:
                df = pd.DataFrame(data["data"])
            elif isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = pd.DataFrame([data])
            df.to_csv(file_path, index=False, encoding="utf-8-sig")
        elif file_format == "parquet":
            import pandas as pd
            if isinstance(data, dict) and "data" in data:
                df = pd.DataFrame(data["data"])
            elif isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = pd.DataFrame([data])
            df.to_parquet(file_path, index=False)
        else:
            raise ValueError(f"不支持的文件格式: {file_format}")

        log.info(f"[output_limiter] 数据已落盘: {file_path}")
        return file_path
    except Exception as e:
        log.error(f"[output_limiter] 导出失败: {e}")
        # 回退到 JSON
        fallback = OUTPUT_DIR / f"{prefix}_fallback_{uuid.uuid4().hex[:8]}.json"
        with open(fallback, "w", encoding="utf-8") as f:
            json.dump(str(data), f, ensure_ascii=False, default=str)
        return fallback
