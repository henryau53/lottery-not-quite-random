"""开奖结构变量的描述性统计与经验分布。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..validation import DRAW_STRUCTURE_COLUMNS


def build_draw_distribution_summary(draws_df: pd.DataFrame) -> pd.DataFrame:
    """计算主要开奖结构变量的样本量、缺失数、分位数及范围。"""
    rows: list[dict[str, object]] = []
    for column in DRAW_STRUCTURE_COLUMNS:
        if column not in draws_df.columns:
            continue
        series = pd.to_numeric(draws_df[column], errors="coerce")
        valid = series.dropna()
        rows.append(
            {
                "variable": column,
                "sample_size": int(valid.size),
                "null_count": int(series.isna().sum()),
                "mean": float(valid.mean()) if not valid.empty else np.nan,
                "std": float(valid.std(ddof=1)) if valid.size >= 2 else np.nan,
                "min": float(valid.min()) if not valid.empty else np.nan,
                "q25": float(valid.quantile(0.25)) if not valid.empty else np.nan,
                "median": float(valid.median()) if not valid.empty else np.nan,
                "q75": float(valid.quantile(0.75)) if not valid.empty else np.nan,
                "max": float(valid.max()) if not valid.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_draw_value_distribution(draws_df: pd.DataFrame) -> pd.DataFrame:
    """生成整数型开奖结构变量的经验频数表，保留变量定义和计数。"""
    rows: list[dict[str, object]] = []
    for column in DRAW_STRUCTURE_COLUMNS:
        if column not in draws_df.columns:
            continue
        series = pd.to_numeric(draws_df[column], errors="coerce")
        valid_count = int(series.notna().sum())
        counts = series.dropna().value_counts().sort_index()
        for value, count in counts.items():
            rows.append(
                {
                    "variable": column,
                    "value": float(value),
                    "count": int(count),
                    "valid_sample_size": valid_count,
                    "observed_proportion": float(count / valid_count) if valid_count else np.nan,
                }
            )
    return pd.DataFrame(rows, columns=[
        "variable", "value", "count", "valid_sample_size", "observed_proportion"
    ])
