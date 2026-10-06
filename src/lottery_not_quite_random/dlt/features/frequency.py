"""号码级频率特征（Number-level Frequency Feature）。

该模块基于 processed draws 生成 draw x number 粒度的 frequency_* 特征：
先构造红球/蓝球各号码的逐期出现指示矩阵，再对出现矩阵 shift(1) 后
按配置窗口滚动求和，从而统计截至上一期、最近若干期内的出现次数。
"""

from __future__ import annotations

import pandas as pd

from ...config import (
    BLUE_NUMBERS,
    RED_NUMBERS,
)


def _build_occurrence_matrix(
    draws_df: pd.DataFrame,
    zone: str,
    numbers: tuple[int, ...],
) -> pd.DataFrame:
    """构造指定号码区的逐期出现指示矩阵。

    根据 draws_df 中对应号码区的开奖号码列，判断 numbers 中的每个号码
    在每期是否出现，返回一个与 draws_df 行索引对齐的 0/1 矩阵。

    Args:
        draws_df:
            开奖数据集。红球区需包含 red_1..red_5，蓝球区需包含
            blue_1..blue_2 等号码列。
        zone:
            号码区，取值为 "red" 或 "blue"。
        numbers:
            该号码区需要统计的号码集合。红球为 1-35，蓝球为 1-12。

    Returns:
        列为 numbers 中的号码，索引与 draws_df 一致。
        列值为 1 表示该期包含该号码，0 表示该期不包含该号码。
    """
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
    """生成号码级频率特征 frequency_*。

    对红球和蓝球分别构造各号码的逐期出现矩阵，先 shift(1) 排除当前期
    开奖结果，再按 FEATURE_FREQUENCY_WINDOWS 中的窗口计算滚动出现次数。
    最终输出每期 × 每个号码的长表。

    Args:
        draws_df:
            按 draw_index 升序排列的 processed draws。需包含 draw_index、
            issue、draw_date 以及红球/蓝球号码列。
        windows:
            滚动历史窗口长度。

    Returns:
        每行表示一个 draw × number 实体，包含以下列：
        - draw_index: 期次索引；
        - issue: 期号；
        - draw_date: 开奖日期；
        - number_zone: 号码区，red 或 blue；
        - number: 号码；
        - frequency_{window}: 截至上一期，最近 window 期内该号码的
            出现次数；历史数据不足 window 期时为 pd.NA。

    Note:
        使用 shift(1) 后再滚动统计，确保第 t 期的频率特征只依赖第 t-1 期
        及更早的数据，避免当前期开奖结果泄漏到特征中。
    """
    rows: list[dict[str, object]] = []

    for zone, numbers in (("red", RED_NUMBERS), ("blue", BLUE_NUMBERS)):
        occurrence_df = _build_occurrence_matrix(draws_df, zone, numbers)
        shifted = occurrence_df.shift(1)
        rolling_cache = {
            window: shifted.rolling(window=window, min_periods=window).sum()
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
