"""Number-level Frequency Feature。"""

from __future__ import annotations

import pandas as pd

from ...config import BLUE_NUMBERS, RED_NUMBERS


def _build_occurrence_matrix(
    draws_df: pd.DataFrame,
    zone: str,
    numbers: tuple[int, ...],
) -> pd.DataFrame:
    """构造指定号码区的历史出现矩阵。"""
    columns = {}
    for number in numbers:
        source_columns = [
            f"{zone}_{index}" for index in range(1, 6 if zone == "red" else 3)
        ]
        columns[number] = draws_df[source_columns].eq(number).any(axis=1).astype("int8")
    return pd.DataFrame(columns, index=draws_df.index)


def build_frequency_features(
    draws_df: pd.DataFrame,
    windows: tuple[int, ...],
) -> pd.DataFrame:
    """生成 Number-level frequency_10/20/50/100。

    Args:
        draws_df:
            按 draw_index 升序排列的 processed draws。
        windows:
            Frequency 历史窗口。

    Returns:
        每个 draw × number entity 的 frequency features。
    """
    rows: list[dict[str, object]] = []

    for zone, numbers in (("red", RED_NUMBERS), ("blue", BLUE_NUMBERS)):
        occurrence_df = _build_occurrence_matrix(draws_df, zone, numbers)

        rolling_cache = {
            window: occurrence_df.rolling(window=window, min_periods=window)
            .sum()
            .shift(1)
            for window in windows
        }

        for position, (_, draw_row) in enumerate(draws_df.iterrows()):
            for number in numbers:
                row: dict[str, object] = {
                    "draw_index": int(draw_row["draw_index"]),
                    "issue": str(draw_row["issue"]),
                    "draw_date": draw_row["draw_date"],
                    "number_zone": zone,
                    "number": number,
                }
                for window in windows:
                    value = rolling_cache[window].iloc[position][number]
                    row[f"frequency_{window}"] = (
                        int(value) if pd.notna(value) else pd.NA
                    )
                rows.append(row)

    return pd.DataFrame(rows)
