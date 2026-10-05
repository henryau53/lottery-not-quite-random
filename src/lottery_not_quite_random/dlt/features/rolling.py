"""Draw-level Rolling Features。"""

from __future__ import annotations

import pandas as pd

from ...config import FEATURE_DEFAULT_PREDICTIVE_SHIFT

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
    windows: tuple[int, ...],
) -> pd.DataFrame:
    """生成 Draw-level rolling mean/std。

    Args:
        draws_df:
            按 draw_index 升序排列的 processed draws。
        windows:
            Rolling 历史窗口。

    Returns:
        Draw-level rolling features，不包含 processed 中已有的基础事实字段。
    """
    output = draws_df.loc[:, ["draw_index", "issue", "draw_date"]].copy()

    for source_column, _ in ROLLING_SOURCES:
        source = draws_df[source_column].astype("float64")
        shifted = source.shift(FEATURE_DEFAULT_PREDICTIVE_SHIFT)

        for window in windows:
            rolling = shifted.rolling(window=window, min_periods=window)
            output[f"{source_column}_mean_{window}"] = rolling.mean().to_numpy()
            output[f"{source_column}_std_{window}"] = rolling.std(ddof=1).to_numpy()

    return output
