"""
utils/io.py — 原子文件写入工具

从 Serena MCP 借鉴的原子写入模式：
- 先写入同目录下的临时文件
- 用 os.replace() 原子替换（POSIX 同文件系统保证原子性）
- 避免 open(path, "w") 截断后崩溃导致旧内容丢失

无 git 项目特别重要：没有回滚能力，编辑中断不能恢复。
"""

import os
import stat
import tempfile
import logging

log = logging.getLogger(__name__)


def _new_file_mode() -> int:
    """新文件的默认权限：0o666 掩码 umask"""
    current_umask = os.umask(0o022)
    os.umask(current_umask)
    return 0o666 & ~current_umask


def write_file_atomic(path: str, content: str, *, encoding: str = "utf-8") -> None:
    """
    原子写入文件：先写临时文件，再 os.replace() 原子替换。

    与普通 open(path, "w") 的区别：
    - 普通写入：先截断文件 → 写入新内容。如果中途崩溃，文件既无旧内容也无新内容。
    - 原子写入：临时文件写满 → os.replace() 原子替换。崩溃时旧内容完好无损。

    :param path: 目标文件路径
    :param content: 要写入的内容
    :param encoding: 文件编码，默认 utf-8
    """
    path = os.path.realpath(path)
    target_dir = os.path.dirname(path) or "."

    try:
        existing_mode = stat.S_IMODE(os.stat(path).st_mode)
    except FileNotFoundError:
        existing_mode = None

    # 在同目录下创建临时文件（保证同文件系统，os.replace 原子性）
    fd, tmp_path = tempfile.mkstemp(
        dir=target_dir,
        prefix=os.path.basename(path) + ".",
        suffix=".tmp",
    )

    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(content)

        # 恢复原有权限（mkstemp 创建的是 0600，避免悄悄收紧权限）
        os.chmod(tmp_path, existing_mode if existing_mode is not None else _new_file_mode())

        # 原子替换
        os.replace(tmp_path, path)
    except BaseException:
        # 清理临时文件
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def read_file_safe(path: str, *, encoding: str = "utf-8") -> str | None:
    """
    安全读取文件：文件不存在或读取出错时返回 None 而非抛出异常。

    :param path: 文件路径
    :param encoding: 文件编码
    :return: 文件内容，或 None
    """
    try:
        with open(path, "r", encoding=encoding) as f:
            return f.read()
    except (FileNotFoundError, OSError, UnicodeDecodeError) as e:
        log.warning(f"[io] 读取失败 {path}: {e}")
        return None
