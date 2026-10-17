"""Feature 输入、输出与时间边界验证。

该模块集中提供特征构建过程中的校验函数，覆盖：
- processed draws / prizes 输入数据的基本质量校验；
- Feature 输出的字段与标识列空值校验；
- 特征的时间边界（as-of）不变性校验，确保未来数据变化不会影响 cutoff
  及之前已生成的特征。
"""

from __future__ import annotations

import logging
from typing import cast

import numpy as np
import pandas as pd

from ...config import (
    BLUE_COLUMNS,
    MAX_ADDITIONAL_RANK_BY_RULE_MAPPING,
    MAX_RANK_BY_RULE_MAPPING,
    PRIZE_EVENT_TYPES,
    RED_COLUMNS,
    RULE_VERSION_NAMES,
    RULE_VERSIONS,
)
from ..utils import normalize_integer_series, validate_required_columns

logger = logging.getLogger(__name__)

# 开奖数据文件必须存在的字段
DRAW_REQUIRED_INPUT_COLUMNS = (
    "draw_index",
    "issue",
    "draw_date",
    "red_1",
    "red_2",
    "red_3",
    "red_4",
    "red_5",
    "blue_1",
    "blue_2",
    "red_sum",
    "blue_sum",
    "red_span",
    "blue_span",
    "red_odd_count",
    "blue_odd_count",
    "red_consecutive_group_count",
    "red_max_consecutive_length",
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
    """验证 Feature 层使用的开奖数据。

    验证必需字段、期号、开奖序号、开奖日期、红蓝球取值范围、
    单期开奖号码唯一性及统计字段。

    不修改输入数据，也不对数据进行排序。

    Args:
        draws_df: 已处理的开奖数据。

    Raises:
        TypeError: 输入不是 pandas DataFrame。
        ValueError: 数据未满足预期的数据契约。
    """
    if not isinstance(draws_df, pd.DataFrame):
        raise TypeError("draws：输入必须是 pandas DataFrame。")

    validate_required_columns(
        columns=tuple(draws_df.columns),
        required=DRAW_REQUIRED_INPUT_COLUMNS,
        label="draws",
    )

    if draws_df.empty:
        raise ValueError("draws：输入数据为空。")

    # -------------------------------------------------------------------------
    # 开奖索引校验
    # -------------------------------------------------------------------------
    draw_index = normalize_integer_series(
        series=draws_df["draw_index"],
        label="draws",
        field_name="draw_index",
    )

    if draw_index.duplicated().any():
        duplicate_indices = (
            draw_index[draw_index.duplicated(keep=False)].drop_duplicates().tolist()
        )
        raise ValueError(f"draws：draw_index 存在重复值 {duplicate_indices}。")

    expected_indices = np.arange(len(draws_df), dtype=np.int64)

    if not np.array_equal(
        draw_index.to_numpy(dtype=np.int64),
        expected_indices,
    ):
        raise ValueError("draws：draw_index 必须按照当前行顺序从 0 开始连续递增。")

    # -------------------------------------------------------------------------
    # 期号校验
    # -------------------------------------------------------------------------
    issues = normalize_integer_series(
        series=draws_df["issue"],
        label="draws",
        field_name="issue",
    )

    if issues.duplicated().any():
        raise ValueError("draws：issue 存在重复值。")

    # -------------------------------------------------------------------------
    # 开奖日期校验
    # -------------------------------------------------------------------------
    dates = pd.to_datetime(
        draws_df["draw_date"],
        errors="coerce",
        utc=True,
    )

    if dates.isna().any():
        raise ValueError("draws：draw_date 存在空值或无法解析的日期。")

    if not dates.is_monotonic_increasing:
        raise ValueError("draws：draw_date 必须按照 draw_index 的顺序单调递增。")

    # -------------------------------------------------------------------------
    # 红蓝号码校验
    # -------------------------------------------------------------------------
    number_columns = {
        **{column: (1, 35) for column in RED_COLUMNS},
        **{column: (1, 12) for column in BLUE_COLUMNS},
    }

    number_values: dict[str, np.ndarray] = {}

    for column, (minimum, maximum) in number_columns.items():
        values = normalize_integer_series(
            series=draws_df[column],
            label="draws",
            field_name=column,
        )

        numeric_values = values.to_numpy(dtype=np.int64)

        invalid_mask = (numeric_values < minimum) | (numeric_values > maximum)

        if invalid_mask.any():
            invalid_values = sorted(set(numeric_values[invalid_mask].tolist()))
            raise ValueError(
                f"draws：{column} 存在超出合法范围 "
                f"[{minimum}, {maximum}] 的值 {invalid_values}。"
            )

        number_values[column] = numeric_values

    # 每期红球不得重复。
    red_values = np.column_stack([number_values[column] for column in RED_COLUMNS])

    duplicate_red_mask = np.array(
        [len(set(row)) != len(RED_COLUMNS) for row in red_values],
        dtype=bool,
    )

    if duplicate_red_mask.any():
        duplicate_indices = draws_df.loc[
            duplicate_red_mask,
            "draw_index",
        ].tolist()

        raise ValueError(
            f"draws：同一期存在重复红球，对应 draw_index 为 {duplicate_indices}。"
        )

    # 每期蓝球不得重复。
    blue_values = np.column_stack([number_values[column] for column in BLUE_COLUMNS])

    duplicate_blue_mask = np.array(
        [len(set(row)) != len(BLUE_COLUMNS) for row in blue_values],
        dtype=bool,
    )

    if duplicate_blue_mask.any():
        duplicate_indices = draws_df.loc[
            duplicate_blue_mask,
            "draw_index",
        ].tolist()

        raise ValueError(
            f"draws：同一期存在重复蓝球，对应 draw_index 为 {duplicate_indices}。"
        )

    # -------------------------------------------------------------------------
    # 统计字段校验
    # -------------------------------------------------------------------------
    statistic_columns = (
        "red_sum",
        "red_span",
        "blue_sum",
        "blue_span",
        "red_odd_count",
        "red_even_count",
        "blue_odd_count",
        "blue_even_count",
        "red_zone_1_count",
        "red_zone_2_count",
        "red_zone_3_count",
        "red_consecutive_group_count",
        "red_max_consecutive_length",
    )

    for column in statistic_columns:
        if column not in draws_df.columns:
            continue

        normalize_integer_series(
            series=draws_df[column],
            label="draws",
            field_name=column,
        )


ADDITIONAL_EVENT_TYPES = {
    "additional_prize",
    "additional_bonus",
}

PRIZE_EVENT_KEY_COLUMNS = (
    "issue",
    "prize_event_type",
    "prize_rank",
    "stake_amount",
)


def validate_prize_input(
    prizes_df: pd.DataFrame,
    draws_df: pd.DataFrame,
) -> None:
    """验证 Feature 层奖金数据，不修改输入数据。

    Args:
        prizes_df: 待验证的奖金数据。
        draws_df: 已通过 validate_draw_input() 校验的开奖数据。

    Raises:
        ValueError: 数据缺少必需字段或不符合业务规则。
    """
    if not isinstance(prizes_df, pd.DataFrame):
        raise TypeError("prizes：prizes_df 必须是 pandas.DataFrame。")

    if not isinstance(draws_df, pd.DataFrame):
        raise TypeError("prizes：draws_df 必须是 pandas.DataFrame。")

    validate_required_columns(
        list(prizes_df.columns),
        PRIZE_REQUIRED_INPUT_COLUMNS,
        "prizes",
    )

    if prizes_df.empty:
        logger.warning("prizes：输入数据为空。")
        return

    # 校验非空列
    for column in (
        "issue",
        "draw_date",
        "rule_version",
        "prize_event_type",
        "prize_rank",
    ):
        if prizes_df[column].isna().any():
            raise ValueError(f"prizes：字段 {column} 存在空值。")

    # 校验整数列（仅检查）
    for column in (
        "issue",
        "prize_rank",
        "stake_count",
    ):
        normalize_integer_series(
            series=prizes_df[column],
            label="prizes",
            field_name=column,
        )

    # 校验规则版本
    known_rule_versions = set(RULE_VERSION_NAMES)
    actual_rule_versions = set(prizes_df["rule_version"].unique())
    unknown_rule_versions = actual_rule_versions - known_rule_versions

    if unknown_rule_versions:
        raise ValueError(f"prizes：存在未知规则版本：{sorted(unknown_rule_versions)}。")

    # 校验开奖事件类型
    known_event_types = set(PRIZE_EVENT_TYPES)
    actual_event_types = set(prizes_df["prize_event_type"].unique())
    unknown_event_types = actual_event_types - known_event_types

    if unknown_event_types:
        raise ValueError(
            f"prizes：存在未知派奖事件类型：{sorted(unknown_event_types)}。"
        )

    # 验证奖级为正整数，并检查对应规则的奖级上限。
    if (prizes_df["prize_rank"] <= 0).any():
        raise ValueError("prizes：prize_rank 必须为正整数。")

    for rule_version, group in prizes_df.groupby("rule_version"):
        rule_version = cast("str", rule_version)
        max_rank = MAX_RANK_BY_RULE_MAPPING.get(rule_version)
        if max_rank is None:
            raise ValueError(f"prizes：规则版本 {rule_version} 缺少奖级上限配置。")

        if (group["prize_rank"] > max_rank).any():
            invalid_ranks = sorted(
                group.loc[
                    group["prize_rank"] > max_rank,
                    "prize_rank",
                ]
                .unique()
                .tolist()
            )
            raise ValueError(
                f"prizes：规则版本 {rule_version} 的奖级超出范围，"
                f"允许的最大奖级为 {max_rank}，异常值：{invalid_ranks}。"
            )

    # 验证追加投注事件的奖级范围。
    additional_mask = prizes_df["prize_event_type"].isin(ADDITIONAL_EVENT_TYPES)
    additional_df = prizes_df.loc[additional_mask]

    for rule_version, group in additional_df.groupby("rule_version"):
        rule_version = cast("str", rule_version)
        max_rank = MAX_ADDITIONAL_RANK_BY_RULE_MAPPING.get(rule_version)
        if max_rank is None:
            raise ValueError(
                f"prizes：规则版本 {rule_version} 缺少追加投注奖级上限配置。"
            )

        if (group["prize_rank"] > max_rank).any():
            invalid_ranks = sorted(
                group.loc[
                    group["prize_rank"] > max_rank,
                    "prize_rank",
                ]
                .unique()
                .tolist()
            )
            raise ValueError(
                f"prizes：规则版本 {rule_version} 的追加投注奖级超出范围，"
                f"允许的最大追加投注奖级为 {max_rank}，"
                f"异常值：{invalid_ranks}。"
            )

    # 验证期号与规则版本是否匹配。
    for rule_version in RULE_VERSIONS:
        name = rule_version.name
        start_issue = rule_version.start_issue
        end_issue = rule_version.end_issue

        rule_mask = prizes_df["rule_version"].eq(name)
        rule_issues = prizes_df.loc[rule_mask, "issue"]

        invalid_mask = (rule_issues < start_issue) | (rule_issues > end_issue)
        if invalid_mask.any():
            invalid_issues = sorted(rule_issues.loc[invalid_mask].unique().tolist())
            raise ValueError(
                f"prizes：规则版本 {rule_version} 的期号超出适用范围 "
                f"[{start_issue}, {end_issue}]，"
                f"异常期号：{invalid_issues}。"
            )

    # 校验期号与开奖数据的关联。
    draw_issues = set(draws_df["issue"].tolist())
    prize_issues = set(prizes_df["issue"].tolist())
    unmatched_issues = prize_issues - draw_issues

    if unmatched_issues:
        raise ValueError(
            f"prizes：以下期号在开奖数据中不存在：{sorted(unmatched_issues)}。"
        )

    # 校验 draw_date 与开奖数据一致。
    prize_dates = pd.to_datetime(
        prizes_df["draw_date"],
        errors="coerce",
        utc=True,
    ).dt.normalize()

    draw_dates = pd.to_datetime(
        draws_df["draw_date"],
        errors="coerce",
        utc=True,
    ).dt.normalize()

    if prize_dates.isna().any():
        raise ValueError("prizes：draw_date 存在无法解析的日期。")

    if draw_dates.isna().any():
        raise ValueError("prizes：开奖数据的 draw_date 存在无法解析的日期。")

    draw_date_by_issue = pd.Series(
        draw_dates.to_numpy(),
        index=draws_df["issue"].to_numpy(),
    )
    expected_dates = prizes_df["issue"].map(draw_date_by_issue)

    mismatch_mask = prize_dates.ne(
        pd.to_datetime(expected_dates, utc=True).dt.normalize()
    )

    if mismatch_mask.any():
        mismatch_issues = sorted(
            prizes_df.loc[mismatch_mask, "issue"].unique().tolist()
        )
        raise ValueError(
            f"prizes：以下期号的 draw_date 与开奖数据不一致：{mismatch_issues}。"
        )

    # 校验事件记录是否重复。
    duplicate_mask = prizes_df.duplicated(
        subset=list(PRIZE_EVENT_KEY_COLUMNS),
        keep=False,
    )
    if duplicate_mask.any():
        duplicate_records = prizes_df.loc[
            duplicate_mask,
            list(PRIZE_EVENT_KEY_COLUMNS),
        ].to_dict(orient="records")
        raise ValueError(f"prizes：存在重复的派奖事件记录：{duplicate_records}。")

    logger.debug(
        "prizes：Feature 层奖金数据验证通过，共 %d 条记录。",
        len(prizes_df),
    )


def validate_feature_output(
    df: pd.DataFrame,
    name: str,
) -> None:
    """验证 Feature 输出字段与标识列空值。

    对 draw_index、issue、draw_date、number_zone、
    number 等标识列检查是否存在空值。

    Args:
        df:
            Feature DataFrame。
        name:
            输出数据集名称，用于错误信息。

    Raises:
        ValueError:
            当字段缺失或标识字段存在空值时。
    """
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
