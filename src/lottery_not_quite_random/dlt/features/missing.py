"""Number-level Missing Feature。"""

from __future__ import annotations

import pandas as pd

from ...config import BLUE_NUMBERS, RED_NUMBERS


def build_missing_features(
    draws_df: pd.DataFrame,
) -> pd.DataFrame:
    """生成每个号码当前观察点的 missing_current。

    Args:
        draws_df:
            按 draw_index 升序排列的 processed draws。

    Returns:
        每个 draw × number entity 的 missing_current。
    """
    rows: list[dict[str, object]] = []

    for zone, numbers in (("red", RED_NUMBERS), ("blue", BLUE_NUMBERS)):
        source_columns = [
            f"{zone}_{index}" for index in range(1, 6 if zone == "red" else 3)
        ]
        seen_positions: dict[int, int] = {}

        for position, (_, draw_row) in enumerate(draws_df.iterrows()):
            for number in numbers:
                last_position = seen_positions.get(number)
                missing = (
                    position - last_position - 1 if last_position is not None else pd.NA
                )
                rows.append(
                    {
                        "draw_index": int(draw_row["draw_index"]),
                        "issue": str(draw_row["issue"]),
                        "draw_date": draw_row["draw_date"],
                        "number_zone": zone,
                        "number": number,
                        "missing_current": missing,
                    }
                )

            for number in numbers:
                if draw_row[source_columns].eq(number).any():
                    seen_positions[number] = position

    return pd.DataFrame(rows)
