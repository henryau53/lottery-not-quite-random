"""大乐透级别的工具模块"""

import numpy as np
import pandas as pd

from ..config import RULE_VERSIONS


def normalize_integer_series(
    series: pd.Series,
    label: str,
    field_name: str,
) -> pd.Series:
    """验证并将 Series 规范化为 int64 整数列。

    接受整数、整数形式的浮点数和可转换为整数的数值字符串。
    拒绝空值、非数值、非有限数值、小数以及超出 int64 范围的值。

    Args:
        series: 待规范化的 Series。
        label: 用于构造错误信息的数据集名称或前缀。
        field_name: 字段名称，用于定位错误。

    Returns:
        规范化后的 int64 Series，保留原索引和字段名称。

    Raises:
        ValueError: 存在无效值或无法转换为 int64 的值。
    """
    try:
        numeric = pd.to_numeric(series, errors="raise")
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{label}：{field_name} 存在非数值。") from exc

    if numeric.isna().any():
        raise ValueError(f"{label}：{field_name} 存在空值。")

    values = numeric.to_numpy(dtype=np.float64)

    if not np.isfinite(values).all():
        raise ValueError(f"{label}：{field_name} 存在非有限数值。")

    if not np.equal(values, np.floor(values)).all():
        raise ValueError(f"{label}：{field_name} 必须是整数。")

    int64_info = np.iinfo(np.int64)

    if (values < int64_info.min).any() or (values >= 2**63).any():
        raise ValueError(f"{label}：{field_name} 超出 int64 的合法范围。")

    return numeric.astype("int64")


def get_rule_version(issue: str) -> str:
    """根据期号获取奖级规则版本。

    Args:
        issue (str): 期号

    Raises:
        ValueError: 无法确定奖级规则版本

    Returns:
        str: 奖级规则版本
    """
    issue_number = int(issue)

    for rule in RULE_VERSIONS:
        if rule.matches(issue_number):
            return rule.name

    raise ValueError(f"无法确定奖级规则版本: {issue}")


def validate_required_columns(
    columns: tuple[str, ...] | list[str],
    required: tuple[str, ...],
    label: str,
) -> None:
    """校验字段集合是否包含必需字段。

    将调用方提供的现有字段与 required 逐一比对，若存在缺失则抛错，
    并在错误信息中给出具体缺失的字段名与对应的说明前缀，便于定位问题。

    Args:
        columns:
            当前已有的字段名集合，可由 df.columns 转换得到。
        required:
            必须存在的字段名集合。
        label:
            用于构造错误信息的名称或前缀说明，例如表名、数据集名或统一前缀。

    Raises:
        ValueError:
            当 required 中存在缺失字段时。
    """
    missing = [column for column in required if column not in columns]
    if missing:
        raise ValueError(f"{label}：缺少必需字段: {missing}")
