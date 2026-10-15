"""项目级工具模块"""

import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path


def find_project_root(marker: str = "pyproject.toml") -> Path:
    """从当前文件向上查找包含 marker 的目录，作为项目根。

    Args:
        marker:
            用于识别项目根的文件名，默认 "pyproject.toml"。

    Returns:
        项目根目录。

    Raises:
        FileNotFoundError:
            向上找不到 marker 时抛出。
    """

    for parent in Path(__file__).resolve().parents:
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f"未找到项目根（缺少 {marker}）")


def utc_now_iso() -> str:
    """返回当前 UTC 时间。

    Returns:
        当前 UTC 时间的 ISO 8601 字符串，例如
        2026-09-22T12:34:56+00:00。
    """
    return datetime.now(UTC).isoformat(timespec="seconds")


def atomic_write_json(
    path: Path,
    data,
    *,
    indent: int = 2,
) -> None:
    """以原子方式将 JSON 数据写入文件。

    先写入同目录下的临时文件，fsync 后用 os.replace 替换目标文件，避免
    写入中断时目标文件只写了一半。

    Args:
        path:
            JSON 目标文件路径。
        data:
            可被 JSON 序列化的数据。
        indent:
            JSON 缩进空格数，默认 2。

    Raises:
        OSError:
            临时文件创建、写入或替换失败时抛出。
        TypeError:
            data 无法被 JSON 序列化时抛出。
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(
                data,
                f,
                ensure_ascii=False,
                indent=indent,
            )
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_name, path)

    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def load_json(path: Path):
    """读取 JSON 文件。

    Args:
        path:
            JSON 文件路径。

    Returns:
        解析后的 JSON 数据。

    Raises:
        FileNotFoundError:
            文件不存在时抛出。
        json.JSONDecodeError:
            文件内容不是合法 JSON 时抛出。
    """
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def file_sha256(path: Path) -> str:
    """计算文件的 SHA-256。

    按块读取文件内容，避免一次性载入大文件。

    Args:
        path:
            文件路径。

    Returns:
        文件的 SHA-256 十六进制摘要。
    """
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
