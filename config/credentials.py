"""
config/credentials.py — 统一凭据管理

从 Financial-API (同花顺金融数据服务) 借鉴的凭证三级回退模式：
  1. 环境变量 (HITHINK_FINANCE_API_KEY / OPENAI_API_KEY 等)
  2. 用户级凭证文件 (~/.appdata/stock_analyzer/credentials.env)
  3. 旧版兼容 (旧环境变量名)

安全规则：
  - Key 不得写入代码、日志、Git、Prompt
  - 凭证文件权限 600
  - 通过 stdin / 进程环境传递，不写入命令参数
"""

import os
import stat
import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# 凭证文件目录
CREDENTIALS_DIR = Path.home() / ".appdata" / "stock_analyzer"
CREDENTIALS_FILE = CREDENTIALS_DIR / "credentials.env"


def _parse_credentials_file(path: Path) -> dict[str, str]:
    """解析 credentials.env 文件 (KEY=VALUE 格式)"""
    result: dict[str, str] = {}
    if not path.exists():
        return result
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, _, value = line.partition("=")
                    result[key.strip()] = value.strip()
    except (OSError, UnicodeDecodeError) as e:
        log.warning(f"[credentials] 读取凭证文件失败 {path}: {e}")
    return result


def resolve_api_key(
    env_var: str = "OPENAI_API_KEY",
    fallback_env_vars: list[str] | None = None,
    *,
    require_length: int = 10,
) -> Optional[str]:
    """
    三级回退解析 API Key。

    优先级：
      1. 主环境变量 (env_var)
      2. 备用环境变量 (fallback_env_vars)
      3. 凭证文件

    :param env_var: 主环境变量名
    :param fallback_env_vars: 备用环境变量名列表
    :return: API Key 或 None
    """
    # 1. 主环境变量
    key = os.environ.get(env_var)
    if key and len(key) >= require_length:
        return key

    # 2. 备用环境变量
    if fallback_env_vars:
        for fallback in fallback_env_vars:
            key = os.environ.get(fallback)
            if key and len(key) >= require_length:
                return key

    # 3. 凭证文件
    creds = _parse_credentials_file(CREDENTIALS_FILE)
    file_key = creds.get(env_var)
    if file_key and len(file_key) >= require_length:
        return file_key
    if fallback_env_vars:
        file_key = creds.get(fallback_env_vars[0])
        if file_key and len(file_key) >= require_length:
            return file_key

    return None


def save_api_key(api_key: str, env_var: str = "OPENAI_API_KEY", *, require_length: int = 10) -> bool:
    """
    安全保存 API Key 到凭证文件。

    :param api_key: API Key (不得为 None 或空)
    :param env_var: 环境变量名
    :return: 是否成功
    """
    if not api_key:
        log.error("[credentials] API Key 不能为空")
        return False

    if len(api_key) < require_length:
        log.warning(f"[credentials] API Key 长度不足 {require_length} (实际 {len(api_key)}), 跳过保存")
        return False

    try:
        CREDENTIALS_DIR.mkdir(parents=True, exist_ok=True)
        # 读取现有凭证
        existing = _parse_credentials_file(CREDENTIALS_FILE) if CREDENTIALS_FILE.exists() else {}
        existing[env_var] = api_key
        # 原子写入
        _write_credentials_file(existing)
        log.info(f"[credentials] API Key 已保存到 {CREDENTIALS_FILE}")
        return True
    except OSError as e:
        log.error(f"[credentials] 保存 API Key 失败: {e}")
        return False


def _write_credentials_file(creds: dict[str, str]) -> None:
    """写入凭证文件，权限 600"""
    # 原子写入：先写临时文件再替换
    tmp_file = CREDENTIALS_FILE.with_suffix(".tmp")
    lines = []
    for key in sorted(creds.keys()):
        lines.append(f"{key}={creds[key]}")
    content = "\n".join(lines) + "\n"

    fd = os.open(str(tmp_file), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(str(tmp_file), str(CREDENTIALS_FILE))
    except BaseException:
        try:
            os.unlink(str(tmp_file))
        except OSError:
            pass
        raise


def is_api_key_configured(env_var: str = "OPENAI_API_KEY") -> bool:
    """检查 API Key 是否已配置"""
    return resolve_api_key(env_var) is not None


def get_credentials_status(env_var: str = "OPENAI_API_KEY") -> dict:
    """获取凭据状态（不暴露 Key 值）"""
    env_key = os.environ.get(env_var)
    file_creds = _parse_credentials_file(CREDENTIALS_FILE)
    file_key = file_creds.get(env_var)

    return {
        "env_var": env_var,
        "environment_configured": bool(env_key),
        "file_configured": bool(file_key),
        "file_path": str(CREDENTIALS_FILE),
        "file_exists": CREDENTIALS_FILE.exists(),
        "configured": bool(env_key or file_key),
    }
