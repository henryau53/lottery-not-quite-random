"""前区与后区号码出现频率描述统计。"""

from __future__ import annotations

import pandas as pd

from ....config import BLUE_COLUMNS, RED_COLUMNS

ZONE_SPEC = {
    "red": {"columns": RED_COLUMNS, "number_max": 35, "draw_count": 5},
    "blue": {"columns": BLUE_COLUMNS, "number_max": 12, "draw_count": 2},
}


def build_number_frequency_table(draws_df: pd.DataFrame) -> pd.DataFrame:
    """计算各号码历史出现次数、观测比例和均匀抽样参考值。

    Args:
        draws_df: 已通过 ``validate_draws`` 的 processed draws。

    Returns:
        每个号码一行的描述性统计表，不包含 p-value 或预测排序。
    """
    n_draws = len(draws_df)
    rows: list[dict[str, object]] = []
    for zone, spec in ZONE_SPEC.items():
        values = draws_df.loc[:, spec["columns"]].to_numpy(dtype=int)
        theoretical_probability = spec["draw_count"] / spec["number_max"]
        for number in range(1, spec["number_max"] + 1):
            observed_count = int((values == number).any(axis=1).sum())
            observed_rate = observed_count / n_draws if n_draws else float("nan")
            rows.append(
                {
                    "number_zone": zone,
                    "number": number,
                    "draws_n": n_draws,
                    "observed_count": observed_count,
                    "observed_rate": observed_rate,
                    "theoretical_probability": theoretical_probability,
                    "expected_count": n_draws * theoretical_probability,
                    "rate_difference": observed_rate - theoretical_probability,
                    "rate_difference_percentage_points": 100
                    * (observed_rate - theoretical_probability),
                    "relative_deviation": (observed_rate - theoretical_probability)
                    / theoretical_probability,
                }
            )
    return (
        pd.DataFrame(rows)
        .sort_values(["number_zone", "number"], kind="mergesort")
        .reset_index(drop=True)
    )
