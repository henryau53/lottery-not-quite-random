import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path


def utc_now_iso() -> str:
    """返回当前 UTC 时间。

    Returns:
        str: ISO 8601 格式的当前 UTC 时间，例如
            ``2026-09-22T12:34:56+00:00``。
    """
    return datetime.now(UTC).isoformat(timespec="seconds")


def atomic_write_json(
    path: Path,
    data,
    *,
    indent: int = 2,
) -> None:
    """以原子方式将 JSON 数据写入文件。

    写入时先在目标目录创建临时文件，完成写入并执行 ``fsync`` 后，
    再通过 ``os.replace()`` 替换目标文件。

    这样可以尽量避免程序在写入过程中异常退出，导致目标 JSON
    文件只写入了一部分内容。

    Args:
        path: JSON 目标文件路径。
        data: 要写入的、可被 JSON 序列化的数据。
        indent: JSON 缩进空格数，默认为 2。

    Raises:
        OSError: 临时文件创建、写入或替换失败时抛出。
        TypeError: ``data`` 无法被 JSON 序列化时抛出。
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
        path: JSON 文件路径。

    Returns:
        Any: 解析后的 JSON 数据。

    Raises:
        FileNotFoundError: 文件不存在时抛出。
        json.JSONDecodeError: 文件内容不是合法 JSON 时抛出。
    """
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
