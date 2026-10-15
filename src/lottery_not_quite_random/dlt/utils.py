"""大乐透级别的工具模块"""

import numpy as np
import pandas as pd

from ..config import RULE_VERSIONS


def normalize_issue_series(
    series: pd.Series,
    table_name: str,
) -> pd.Series:
    """将期号列规范化为整数，并拒绝无效期号。

    Args:
        series:
            待规范化的期号列。
        table_name:
            调用方表名，用于构造更清晰的错误信息。

    Returns:
        规范化后的期号列，类型为 int64，且不含空值与小数。

    Raises:
        ValueError:
            当期号列存在空值，或存在非整数期号时。
    """
    issue = pd.to_numeric(series, errors="raise")

    if issue.isna().any():
        raise ValueError(f"{table_name}.issue 存在空值。")

    if not np.equal(issue, np.floor(issue)).all():
        raise ValueError(f"{table_name}.issue 必须是整数期号。")

    return issue.astype("int64")


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
    name: str,
) -> None:
    """校验字段集合是否包含必需字段。

    将调用方提供的现有字段与 required 逐一比对，若存在缺失则抛错，
    并在错误信息中给出具体缺失的字段名与所属表名，便于定位问题。

    Args:
        columns:
            当前已有的字段名集合，可由 df.columns 转换得到。
        required:
            必须存在的字段名集合。
        name:
            名称，用于构造更清晰的错误信息。

    Raises:
        ValueError:
            当 required 中存在缺失字段时。
    """
    missing = [column for column in required if column not in columns]
    if missing:
        raise ValueError(f"{name} 缺少必需字段: {missing}")
