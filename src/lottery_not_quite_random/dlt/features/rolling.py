"""开奖级特征滚动统计特征（Draw-level Rolling Features）。

用于体彩大乐透 processed draws 的期级特征提取。该模块针对
ROLLING_SOURCES 中配置的期级汇总字段，按指定历史窗口生成滚动均值与
滚动标准差。

所有滚动统计均先按对源字段做 shift(1)，
再在历史窗口上计算，确保第 t 期特征只依赖第 t 期之前的数据，避免当前期
开奖结果泄漏到特征中。
"""

from __future__ import annotations

import pandas as pd

from ...config import FEATURE_ROLLING_WINDOWS

# rolling 所取用的字段与号码区
ROLLING_SOURCES: tuple[tuple[str, str], ...] = (
    ("red_sum", "red"),
    ("blue_sum", "blue"),
    ("red_span", "red"),
    ("blue_span", "blue"),
    ("red_odd_count", "red"),
    ("blue_odd_count", "blue"),
    ("red_consecutive_group_count", "red"),
    ("red_max_consecutive_length", "red"),
)


def build_rolling_features(
    draws_df: pd.DataFrame,
) -> pd.DataFrame:
    """生成 Draw-level 滚动均值与滚动标准差特征。

    对 ROLLING_SOURCES 中的每个源字段，先进行 shift(1)，再在 windows 指定
    的各历史窗口上计算滚动均值和滚动标准差。
    输出表保留 draw_index、issue、draw_date 作为标识列，不包含 processed 中
    已有的基础事实字段。

    Args:
        draws_df:
            按 draw_index 升序排列的 processed draws，需包含 draw_index、
            issue、draw_date 以及 ROLLING_SOURCES 中列出的源字段。

    Returns:
        以 draw_index、issue、draw_date 为标识列的 Draw-level 滚动特征表。
        每行对应一期，列包括各源字段在指定窗口下的滚动均值和滚动标准差；
        当历史数据不足以填满窗口时，对应值为 NA。
        不包含 processed 中已有的基础事实字段，只输出滚动统计结果。

    Note:
        先 shift 再 rolling，确保第 t 期特征只使用第 t 期之前的历史数据，
        避免当前期开奖结果泄漏到特征中。
    """
    output = draws_df.loc[:, ["draw_index", "issue", "draw_date"]].copy()

    for source_column, _ in ROLLING_SOURCES:
        source = draws_df[source_column].astype("float64")
        shifted = source.shift(1)

        for window in FEATURE_ROLLING_WINDOWS:
            rolling = shifted.rolling(window=window, min_periods=window)
            output[f"{source_column}_mean_{window}"] = rolling.mean().to_numpy()
            output[f"{source_column}_std_{window}"] = rolling.std(ddof=1).to_numpy()

    return output
