"""大乐透级别的工具模块"""

import pandas as pd
import numpy as np


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
