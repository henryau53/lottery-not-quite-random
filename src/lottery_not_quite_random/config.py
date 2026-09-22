from pathlib import Path


def _find_project_root(marker: str = "pyproject.toml") -> Path:
    """从当前文件向上查找包含 marker 的目录，作为项目根

    Args:
        marker (str, optional): 用于识别项目根的文件名。.默认值 "pyproject.toml"

    Raises:
        FileNotFoundError: 向上找不到 marker 时抛出

    Returns:
        Path: 项目根目录
    """

    for parent in Path(__file__).resolve().parents:
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f"未找到项目根（缺少 {marker}）")


# 项目根目录
PROJECT_ROOT = _find_project_root()

# 大乐透数据目录
DLT_DATA_DIR = PROJECT_ROOT / "data" / "dlt"

# 原始数据目录
DLT_RAW_DIR = DLT_DATA_DIR / "raw"
DLT_RAW_DRAWS_FILE = DLT_RAW_DIR / "draws.json"
DLT_RAW_META_FILE = DLT_RAW_DIR / "meta.json"
