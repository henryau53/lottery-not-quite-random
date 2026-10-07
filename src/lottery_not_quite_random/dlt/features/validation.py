"""Feature 输入、输出与时间边界验证。

该模块集中提供特征构建过程中的校验函数，覆盖：
- processed draws / prizes 输入数据的基本质量校验；
- Feature 输出的字段与标识列空值校验；
- 特征的时间边界（as-of）不变性校验，确保未来数据变化不会影响 cutoff
  及之前已生成的特征。
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import pandas as pd

from ...config import PRIZE_EVENT_TYPES, RULE_VERSION_NAMES

LOGGER = logging.getLogger(__name__)

# 开奖数据文件必须存在的字段
DRAW_REQUIRED_INPUT_COLUMNS = (
    "draw_index",
    "issue",
    "draw_date",
    "red_sum",
    "blue_sum",
    "red_span",
    "blue_span",
    "red_odd_count",
    "blue_odd_count",
    "red_consecutive_group_count",
    "red_max_consecutive_length",
    "red_1",
    "red_2",
    "red_3",
    "red_4",
    "red_5",
    "blue_1",
    "blue_2",
)

# 奖金数据文件必须存在的字段
PRIZE_REQUIRED_INPUT_COLUMNS = (
    "issue",
    "draw_date",
    "rule_version",
    "prize_rank",
    "prize_event_type",
)


def validate_draw_input(draws_df: pd.DataFrame) -> None:
    """验证 processed draws 数据。

    校验 required columns、draw_index 连续性、issue 唯一性、draw_date 非空，
    以及红球/蓝球的取值范围与单期互不重复。

    Args:
        draws_df:
            processed/draws.parquet。

    Raises:
        ValueError:
            当输入不满足项目数据质量契约时。
    """
    missing = [
        column
        for column in DRAW_REQUIRED_INPUT_COLUMNS
        if column not in draws_df.columns
    ]
    if missing:
        raise ValueError(f"draws 缺少 required columns: {missing}")

    expected_index = list(range(len(draws_df)))
    actual_index = draws_df["draw_index"].astype(int).tolist()

    if actual_index != expected_index:
        raise ValueError("draw_index 必须从 0 开始连续递增。")

    if draws_df["issue"].duplicated().any():
        raise ValueError("issue 必须唯一。")

    if draws_df["draw_date"].isna().any():
        raise ValueError("draw_date 不允许为空。")

    for column in [
        *(f"red_{i}" for i in range(1, 6)),
        *(f"blue_{i}" for i in range(1, 3)),
    ]:
        if draws_df[column].isna().any():
            raise ValueError(f"{column} 不允许为空。")

    red_columns = [f"red_{i}" for i in range(1, 6)]
    blue_columns = [f"blue_{i}" for i in range(1, 3)]

    if not draws_df[red_columns].apply(lambda col: col.between(1, 35)).all().all():
        raise ValueError("red 号码存在非法值。")

    if not draws_df[blue_columns].apply(lambda col: col.between(1, 12)).all().all():
        raise ValueError("blue 号码存在非法值。")

    if draws_df[red_columns].apply(lambda row: row.nunique() != 5, axis=1).any():
        raise ValueError("单期开奖 red 号码必须互不重复。")

    if draws_df[blue_columns].apply(lambda row: row.nunique() != 2, axis=1).any():
        raise ValueError("单期开奖 blue 号码必须互不重复。")


def validate_prize_input(prizes_df: pd.DataFrame) -> None:
    """验证 processed prizes 数据。

    校验 required columns 是否齐全，prizes.draw_index 是否存在空值，以及
    rule_version 与 prize_event_type 是否在已知取值范围内。

    Args:
        prizes_df:
            processed/prizes.parquet。

    Raises:
        ValueError:
            当 required columns 缺失、字段为空或字段值非法时。
    """

    missing = [
        column
        for column in PRIZE_REQUIRED_INPUT_COLUMNS
        if column not in prizes_df.columns
    ]

    if missing:
        raise ValueError(f"prizes 缺少 required columns: {missing}")

    if "draw_index" in prizes_df.columns and prizes_df["draw_index"].isna().any():
        raise ValueError("prizes.draw_index 不允许为空。")

    unknown_rules = sorted(
        set(prizes_df["rule_version"].dropna().astype(str)) - set(RULE_VERSION_NAMES)
    )

    if unknown_rules:
        raise ValueError(f"未知奖级规则版本 rule_version: {unknown_rules}")

    unknown_events = sorted(
        set(prizes_df["prize_event_type"].dropna().astype(str)) - set(PRIZE_EVENT_TYPES)
    )

    if unknown_events:
        raise ValueError(f"未知奖金事件类型 prize_event_type: {unknown_events}")


def validate_feature_output(
    df: pd.DataFrame,
    required_columns: Iterable[str],
    name: str,
) -> None:
    """验证 Feature 输出字段与标识列空值。

    先检查 required_columns 是否全部存在，再对 draw_index、issue、draw_date、
    number_zone、number 等标识列检查是否存在空值。

    Args:
        df:
            Feature DataFrame。
        required_columns:
            必须存在的字段。
        name:
            输出数据集名称，用于错误信息。

    Raises:
        ValueError:
            当字段缺失或标识字段存在空值时。
    """
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"{name} 缺少 required columns: {missing}")

    identifier_columns = [
        column
        for column in ("draw_index", "issue", "draw_date", "number_zone", "number")
        if column in df.columns
    ]
    nulls = {column: int(df[column].isna().sum()) for column in identifier_columns}
    invalid = {column: count for column, count in nulls.items() if count}
    if invalid:
        raise ValueError(f"{name} 标识字段存在 null: {invalid}")


def assert_as_of_invariance(
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    cutoff_draw_index: int,
) -> None:
    """验证未来数据变化不会影响 cutoff 及之前的 Feature。

    取 cutoff_draw_index 及之前的记录，按标识列排序后比较 before_df 与
    after_df，确保修改未来数据后，历史部分的特征没有变化。

    Args:
        before_df:
            原始 Feature 输出。
        after_df:
            修改未来数据后的 Feature 输出。
        cutoff_draw_index:
            未来数据起点之前的最后一个观察点。

    Raises:
        AssertionError:
            当 cutoff 及之前的 Feature 发生变化时。
    """
    keys = [
        column
        for column in ("draw_index", "number_zone", "number")
        if column in before_df.columns
    ]
    feature_columns = [
        column
        for column in before_df.columns
        if column not in keys and column not in {"issue", "draw_date"}
    ]

    left = before_df.loc[
        before_df["draw_index"] <= cutoff_draw_index, [*keys, *feature_columns]
    ]
    right = after_df.loc[
        after_df["draw_index"] <= cutoff_draw_index, [*keys, *feature_columns]
    ]

    left = left.sort_values(keys, kind="mergesort").reset_index(drop=True)
    right = right.sort_values(keys, kind="mergesort").reset_index(drop=True)

    pd.testing.assert_frame_equal(left, right, check_dtype=False, check_exact=True)
