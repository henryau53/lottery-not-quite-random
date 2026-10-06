"""号码级遗漏特征（Number-level Missing Feature）。

用于体彩大乐透 processed draws 的号码级遗漏特征提取。该模块对红球、蓝球
中的每个号码，逐期计算其在当前观察点的 missing_current，即截至当前期之前，
该号码连续未出现的期数。

计算时先为当前期生成 missing_current，再根据当前期开奖结果更新号码的出现
位置，确保第 t 期的 missing_current 只依赖第 t 期之前的数据，避免当前期
开奖结果泄漏到特征中。
"""

from __future__ import annotations

import pandas as pd

from ...config import BLUE_NUMBERS, RED_NUMBERS


def build_missing_features(
    draws_df: pd.DataFrame,
) -> pd.DataFrame:
    """生成每个号码在当前观察点的遗漏值 missing_current。

    对红球和蓝球分别遍历每一期，并为该区每个号码生成一行记录。
    missing_current 表示截至当前期之前，该号码连续未出现的期数：
    - 若该号码此前已经出现过，则等于当前期位置与上一次出现位置之间的间隔
      期数；
    - 若该号码此前从未出现，则为 pd.NA。

    每期先为所有号码生成 missing_current，再根据当前期开奖结果更新各号码的
    最近出现位置，从而保证特征只使用当前期之前的历史数据。

    Args:
        draws_df:
            按 draw_index 升序排列的 processed draws，需包含 draw_index、
            issue、draw_date 以及红球/蓝球号码列。

    Returns:
        每行对应一个 draw x number 实体的长表，包含 draw_index、issue、
        draw_date、number_zone、number 以及 missing_current。
        missing_current 表示截至当前期之前该号码连续未出现的期数；若该号码
        此前从未出现，则为 pd.NA。
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
