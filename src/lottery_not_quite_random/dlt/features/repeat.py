"""开奖级重复特征（Draw-level Repeat Features）。

用于体彩大乐透 processed draws 的期级重复特征提取。该模块提供：
- 每个号码相对上一期同区开奖的命中状态（prev_draw_hit）；
- 相邻两期开奖的全区重复号码数量；
- 基于历史相邻重复数量的滚动均值特征。

所有历史滚动特征均先 shift(1) 再计算，确保第 t 期特征只依赖第 t 期
之前的数据，避免当前期开奖结果泄漏到特征中。
"""

from __future__ import annotations

import pandas as pd

from ...config import BLUE_NUMBERS, FEATURE_REPEAT_WINDOWS, RED_NUMBERS


def _number_set(row: pd.Series, zone: str) -> set[int]:
    """提取单期开奖指定号码区的号码集合。

    根据 zone 从 row 中读取对应号码列，红球读取 red_1..red_5，蓝球读取
    blue_1..blue_2，并返回这些号码组成的集合。

    Args:
        row:
            单期开奖数据行。
        zone:
            号码区，取值为 "red" 或 "blue"。

    Returns:
        该期指定号码区所包含的号码集合。
    """
    columns = [f"{zone}_{index}" for index in range(1, 6 if zone == "red" else 3)]
    return {int(value) for value in row[columns]}


def calculate_repeat_count(a_row: pd.Series, b_row: pd.Series) -> int:
    """计算两期开奖的全区（红 + 蓝）重复号码数量。

    分别提取两期的红球与蓝球号码，合并为全区号码集合后，计算两个集合的
    交集大小，即两期开奖中重复出现的号码个数。

    Args:
        a_row:
            用于对比的单期开奖数据行。
        b_row:
            用于对比的单期开奖数据行。

    Returns:
        两期开奖在全区号码上的重复号码数量。
    """
    a = _number_set(a_row, "red") | _number_set(a_row, "blue")
    b = _number_set(b_row, "red") | _number_set(b_row, "blue")
    return len(a & b)


def build_repeat_history_features(
    draws_df: pd.DataFrame,
) -> pd.DataFrame:
    """生成历史相邻开奖重复数量的滚动均值特征。

    先逐期计算当前期与上一期的全区重复号码数量，得到历史相邻重复数量序列；
    再对该序列 shift(1) 后，按 windows 指定的各历史窗口计算滚动均值。输出
    表保留 draw_index、issue、draw_date 作为标识列。

    Args:
        draws_df:
            按 draw_index 升序排列的 processed draws。

    Returns:
        以 draw_index、issue、draw_date 为标识列的 Draw-level 特征表。
        每行对应一期，列包括各窗口下的 repeat_count_mean_*，表示截至上一期
        的历史相邻重复数量滚动均值；历史数据不足窗口时对应值为 NA。
    """
    repeat_counts: list[float] = [float("nan")]

    for position in range(1, len(draws_df)):
        previous_row = draws_df.iloc[position - 1]
        current_row = draws_df.iloc[position]
        repeat_counts.append(float(calculate_repeat_count(previous_row, current_row)))

    # TODO 这里仅添加了历史相邻开奖的 rolling mean 字段，是否将 repeat_series 也就是当前与上期重复号码也作为特征加入
    repeat_series = pd.Series(repeat_counts, index=draws_df.index, dtype="float64")
    shifted = repeat_series.shift(1)
    output = draws_df.loc[:, ["draw_index", "issue", "draw_date"]].copy()

    for window in FEATURE_REPEAT_WINDOWS:
        output[f"repeat_count_mean_{window}"] = (
            shifted.rolling(window=window, min_periods=window).mean().to_numpy()
        )

    return output


def build_prev_draw_hit(draws_df: pd.DataFrame) -> pd.DataFrame:
    """生成每个号码相对上一期同区开奖的命中状态。

    对每期、每个号码，判断该号码是否出现在上一期相同号码区（红球或蓝球）
    的开奖结果中。输出为 draw x number 的长表，首期因无上一期数据，其
    prev_draw_hit 为 pd.NA。

    Args:
        draws_df:
            按 draw_index 升序排列的 processed draws。

    Returns:
        每行对应一个 draw x number 实体的长表，包含 draw_index、issue、
        draw_date、number_zone、number 以及 prev_draw_hit。
        prev_draw_hit 为 True 表示该号码出现在上一期同区开奖中，False 表示
        未出现，首期为 pd.NA。
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
