"""
utils/http_timeout.py — 统一 HTTP 超时保护

从 hithink-finance 借鉴的超时模式:
  - 所有外部 API 调用必须带 timeout
  - 默认 timeout=10s, 分析引擎 total_timeout=120s
  - 重试: HTTP 429/502/503/504 有界重试 (最多 3 次)

使用方式:
    from utils.http_timeout import fetch_with_timeout, retryable_get

    # 简单 GET
    data = fetch_with_timeout("http://api.example.com/data", timeout=10)

    # 带重试的 GET
    data, success = retryable_get("http://api.example.com/data", max_retries=3)
"""

import requests
import urllib.request
import urllib.error
import logging
import time
import random
from typing import Any, Optional, Tuple

log = logging.getLogger(__name__)

# 默认超时 (秒)
DEFAULT_TIMEOUT = 10
MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0  # 指数退避基数

# 可重试的 HTTP 状态码
RETRYABLE_CODES = {429, 502, 503, 504}


def fetch_with_timeout(url: str, timeout: int = DEFAULT_TIMEOUT, **kwargs) -> Optional[Any]:
    """
    带超时的 HTTP GET。

    :param url: 目标 URL
    :param timeout: 超时秒数
    :param kwargs: 传递给 requests.get 的其他参数
    :return: response.json() 或 None
    """
    try:
        resp = requests.get(url, timeout=timeout, **kwargs)
        return resp.json()
    except requests.Timeout:
        log.warning(f"[http_timeout] 超时: {url} (>{timeout}s)")
        return None
    except requests.RequestException as e:
        log.warning(f"[http_timeout] 请求失败: {url} - {e}")
        return None
    except Exception as e:
        log.warning(f"[http_timeout] 未知错误: {url} - {e}")
        return None


def retryable_get(
    url: str,
    max_retries: int = MAX_RETRIES,
    timeout: int = DEFAULT_TIMEOUT,
    **kwargs,
) -> Tuple[Optional[Any], bool]:
    """
    带重试的 HTTP GET。

    对 HTTP 429/502/503/504 进行有界重试, 其他错误不重试。

    :param url: 目标 URL
    :param max_retries: 最大重试次数
    :param timeout: 超时秒数
    :return: (response.json(), success)
    """
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, timeout=timeout, **kwargs)

            # 可重试的状态码
            if resp.status_code in RETRYABLE_CODES and attempt < max_retries:
                delay = RETRY_BASE_DELAY * attempt + random.uniform(0, 0.5)
                log.info(f"[http_timeout] HTTP {resp.status_code}, {delay:.1f}s 后重试 ({attempt}/{max_retries})")
                time.sleep(delay)
                continue

            return resp.json(), True

        except requests.Timeout:
            last_error = f"Timeout (>{timeout}s)"
            if attempt < max_retries:
                delay = RETRY_BASE_DELAY * attempt
                time.sleep(delay)
            continue

        except requests.RequestException as e:
            last_error = str(e)
            if attempt < max_retries:
                time.sleep(RETRY_BASE_DELAY)
            continue

        except Exception as e:
            last_error = str(e)
            break  # 非网络错误, 不重试

    log.warning(f"[http_timeout] 重试耗尽: {url} (最后错误: {last_error})")
    return None, False


def safe_urlopen(url: str, timeout: int = DEFAULT_TIMEOUT) -> Optional[str]:
    """
    带超时的 urllib.urlopen。

    :param url: 目标 URL
    :param timeout: 超时秒数
    :return: response.read().decode() 或 None
    """
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode('utf-8')
    except urllib.error.URLError as e:
        log.warning(f"[http_timeout] urllib 错误: {url} - {e}")
        return None
    except Exception as e:
        log.warning(f"[http_timeout] urllib 未知错误: {url} - {e}")
        return None
