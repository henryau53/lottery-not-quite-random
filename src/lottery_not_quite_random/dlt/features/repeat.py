"""Repeat Features。"""

from __future__ import annotations

import pandas as pd

from ...config import BLUE_NUMBERS, FEATURE_DEFAULT_PREDICTIVE_SHIFT, RED_NUMBERS


def _number_set(row: pd.Series, zone: str) -> set[int]:
    """提取单期开奖指定号码区的集合。

    Args:
        row:
            单期开奖数据行。
        zone:
            号码区间，red 或 blue。

    Returns:
        所属号码区间的号码集合。
    """
    columns = [f"{zone}_{index}" for index in range(1, 6 if zone == "red" else 3)]
    return {int(value) for value in row[columns]}


def build_prev_draw_hit(draws_df: pd.DataFrame) -> pd.DataFrame:
    """生成 prev_draw_hit。

    Args:
        draws_df:
            按 draw_index 升序排列的 processed draws。

    Returns:
        每个 draw x number entity 的上一期开奖命中状态。
    """
    rows: list[dict[str, object]] = []

    previous_sets: dict[str, set[int] | None] = {"red": None, "blue": None}

    for _, draw_row in draws_df.iterrows():
        for zone, numbers in (("red", RED_NUMBERS), ("blue", BLUE_NUMBERS)):
            previous = previous_sets[zone]
            for number in numbers:
                rows.append(
                    {
                        "draw_index": int(draw_row["draw_index"]),
                        "issue": str(draw_row["issue"]),
                        "draw_date": draw_row["draw_date"],
                        "number_zone": zone,
                        "number": number,
                        "prev_draw_hit": (
                            pd.NA if previous is None else number in previous
                        ),
                    }
                )
            previous_sets[zone] = _number_set(draw_row, zone)

    return pd.DataFrame(rows)


def calculate_repeat_count(a_row: pd.Series, b_row: pd.Series) -> int:
    """计算相邻两期开奖的全区（红 + 蓝）号码重复数量。

    Args:
        a_row:
            对比的单期开奖数据行。
        b_row:
            对比的单期开奖数据行。

    Returns:
        重复数量。
    """
    a = _number_set(a_row, "red") | _number_set(a_row, "blue")
    b = _number_set(b_row, "red") | _number_set(b_row, "blue")
    return len(a & b)


def build_repeat_history_features(
    draws_df: pd.DataFrame,
    windows: tuple[int, ...],
) -> pd.DataFrame:
    """生成历史相邻开奖重复数量的 rolling mean。

    Args:
        draws_df:
            按 draw_index 升序排列的 processed draws。
        windows:
            历史相邻开奖关系窗口。

    Returns:
        Draw-level repeat_count_mean_*。
    """
    repeat_counts: list[float] = [float("nan")]

    for position in range(1, len(draws_df)):
        previous_row = draws_df.iloc[position - 1]
        current_row = draws_df.iloc[position]
        repeat_counts.append(float(calculate_repeat_count(previous_row, current_row)))

    # TODO 这里仅添加了历史相邻开奖的 rolling mean 字段，是否将 repeat_series 也就是当前与上期重复号码也作为特征加入
    repeat_series = pd.Series(repeat_counts, index=draws_df.index, dtype="float64")
    shifted = repeat_series.shift(FEATURE_DEFAULT_PREDICTIVE_SHIFT)
    output = draws_df.loc[:, ["draw_index", "issue", "draw_date"]].copy()

    for window in windows:
        output[f"repeat_count_mean_{window}"] = (
            shifted.rolling(window=window, min_periods=window).mean().to_numpy()
        )

    return output
