"""Analysis 输入与粒度验证。"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from ...config import (
    BLUE_COLUMNS,
    BONUS_CAMPAIGNS,
    PRIZE_EVENT_TYPES,
    RED_COLUMNS,
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

# FIXME 这里与 feature 阶段 validation 中的 PRIZE_REQUIRED_INPUT_COLUMNS 字段不相同
# 奖金数据文件必须存在的字段
PRIZE_REQUIRED_INPUT_COLUMNS = (
    "issue",
    "draw_date",
    "rule_version",
    "prize_rank",
    "prize_event_type",
    "stake_amount",
    "stake_count",
    "total_prize_amount",
)


def validate_draw_input(draws_df: pd.DataFrame) -> pd.DataFrame:
    """验证并规范化 Analysis 层使用的开奖数据。

    验证必需字段、开奖序号、期号、开奖日期、红蓝球取值范围、
    单期开奖号码唯一性及可选统计字段，并返回按 draw_index
    排序的副本。

    Args:
        draws_df: 已处理的开奖数据。

    Returns:
        pd.DataFrame: 通过校验、期号和开奖序号规范化，
            并按 draw_index 排序的 DataFrame 副本。

    Raises:
        TypeError: 输入不是 pandas DataFrame。
        ValueError: 数据缺少必需字段，或未满足数据契约。
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

    result = draws_df.copy()

    # -------------------------------------------------------------------------
    # 开奖索引校验
    # -------------------------------------------------------------------------
    result["draw_index"] = normalize_integer_series(
        series=result["draw_index"],
        label="draws",
        field_name="draw_index",
    )

    if result["draw_index"].duplicated().any():
        duplicate_indices = (
            result.loc[
                result["draw_index"].duplicated(keep=False),
                "draw_index",
            ]
            .drop_duplicates()
            .tolist()
        )
        raise ValueError(f"draws：draw_index 存在重复值 {duplicate_indices}。")

    expected_indices = np.arange(len(result), dtype=np.int64)
    actual_indices = np.sort(result["draw_index"].to_numpy(dtype=np.int64))

    if not np.array_equal(actual_indices, expected_indices):
        raise ValueError("draws：draw_index 必须按照当前行顺序从 0 开始连续递增。")

    # -------------------------------------------------------------------------
    # 期号校验
    # -------------------------------------------------------------------------
    result["issue"] = normalize_integer_series(
        series=result["issue"],
        label="draws",
        field_name="issue",
    )

    if result["issue"].duplicated().any():
        raise ValueError("draws：issue 存在重复值。")

    # -------------------------------------------------------------------------
    # draw_index 排序
    # -------------------------------------------------------------------------
    result = result.sort_values(
        "draw_index",
        kind="stable",
    ).reset_index(drop=True)

    # -------------------------------------------------------------------------
    # 开奖日期校验
    # -------------------------------------------------------------------------
    dates = pd.to_datetime(
        result["draw_date"],
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
            series=result[column],
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
        result[column] = values

    # 每期红球不得重复。
    red_values = np.column_stack([number_values[column] for column in RED_COLUMNS])

    duplicate_red_mask = np.array(
        [len(set(row)) != len(RED_COLUMNS) for row in red_values],
        dtype=bool,
    )

    if duplicate_red_mask.any():
        duplicate_indices = result.loc[
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
        duplicate_indices = result.loc[
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
        if column not in result.columns:
            continue

        result[column] = normalize_integer_series(
            series=result[column],
            label="draws",
            field_name=column,
        )

    return result


def validate_prizes(prizes_df: pd.DataFrame, draws_df: pd.DataFrame) -> pd.DataFrame:
    """验证奖级记录、规则版本和事件类型。"""
    if not isinstance(prizes_df, pd.DataFrame):
        raise TypeError("prizes_df 必须是 pandas.DataFrame。")
    validate_required_columns(
        tuple(prizes_df.columns), PRIZE_REQUIRED_INPUT_COLUMNS, "prizes"
    )
    result = prizes_df.copy()
    if result.empty:
        logger.warning("processed prizes 为空；本次运行只会输出空奖金摘要。")
        result["issue"] = result["issue"].astype("string")
        return result

    result["issue"] = normalize_integer_series(
        result["issue"],
        label="prizes_processed",
        field_name="issue",
    )
    draw_issues = set(draws_df["issue"].astype(str))
    unknown_issues = sorted(set(result["issue"].astype(str)) - draw_issues)
    if unknown_issues:
        raise ValueError(
            f"prizes 中存在 processed draws 未包含的期号: {unknown_issues[:10]}"
        )

    invalid_events = sorted(
        set(result["prize_event_type"].dropna().astype(str)) - set(PRIZE_EVENT_TYPES)
    )
    if invalid_events or result["prize_event_type"].isna().any():
        raise ValueError(
            "prizes.prize_event_type 必须属于 "
            f"{sorted(PRIZE_EVENT_TYPES)}；非法值: {invalid_events}"
        )

    # 奖金记录的 draw_date 必须与开奖事实对应，避免按 issue 关联时混入错误日期。
    date_map = draws_df.assign(
        __analysis_date=pd.to_datetime(draws_df["draw_date"])
    ).set_index("issue")["__analysis_date"]
    prize_dates = pd.to_datetime(result["draw_date"], errors="coerce")
    expected_dates = result["issue"].map(date_map)
    if (
        prize_dates.isna().any()
        or expected_dates.isna().any()
        or not prize_dates.dt.normalize().equals(expected_dates.dt.normalize())
    ):
        raise ValueError(
            "prizes.draw_date 与 processed draws 中相同期号的开奖日期不一致。"
        )

    if result["rule_version"].isna().any():
        raise ValueError("prizes.rule_version 存在空值。")
    invalid_versions = sorted(
        set(result["rule_version"].astype(str)) - {"v1", "v2", "v3", "v4"}
    )
    if invalid_versions:
        raise ValueError(f"prizes.rule_version 存在未知值: {invalid_versions}")

    prize_rank = pd.to_numeric(result["prize_rank"], errors="coerce")
    if (
        prize_rank.isna().any()
        or not np.equal(prize_rank, np.floor(prize_rank)).all()
        or prize_rank.lt(1).any()
    ):
        raise ValueError("prizes.prize_rank 必须是正整数。")
    result["prize_rank"] = prize_rank.astype("int64")
    max_rank_by_rule = {"v1": 8, "v2": 6, "v3": 9, "v4": 7}
    max_rank = result["rule_version"].astype(str).map(max_rank_by_rule)
    invalid_rank_mask = result["prize_rank"] > max_rank
    if invalid_rank_mask.any():
        examples = (
            result.loc[invalid_rank_mask, ["issue", "rule_version", "prize_rank"]]
            .head(5)
            .to_dict("records")
        )
        raise ValueError(
            f"prizes.prize_rank 超出对应 rule_version 的奖级范围，示例: {examples}"
        )

    additional_mask = (
        result["prize_event_type"]
        .astype(str)
        .isin({"additional_prize", "additional_bonus"})
    )
    max_additional_rank_by_rule = {"v1": 7, "v2": 5, "v3": 2, "v4": 2}
    max_additional_rank = (
        result["rule_version"].astype(str).map(max_additional_rank_by_rule)
    )
    invalid_additional_mask = additional_mask & result["prize_rank"].gt(
        max_additional_rank
    )
    if invalid_additional_mask.any():
        examples = (
            result.loc[
                invalid_additional_mask,
                ["issue", "rule_version", "prize_rank", "prize_event_type"],
            ]
            .head(5)
            .to_dict("records")
        )
        raise ValueError(
            f"追加投注记录的 prize_rank 超出对应 rule_version 的追加范围，示例: {examples}"
        )

    issue_numbers = pd.to_numeric(result["issue"], errors="coerce")
    if (
        issue_numbers.isna().any()
        or not np.equal(issue_numbers, np.floor(issue_numbers)).all()
    ):
        raise ValueError("prizes.issue 必须是可识别的整数期号。")

    expected_versions = pd.Series(
        # TODO 此处更改逻辑，对 expected_versions 的结果输出查看验证是否存在问题，查看 default=np.nan 的默认值问题
        np.select(
            [
                issue_numbers.between(rule.start_issue, rule.end_issue)
                for rule in RULE_VERSIONS
            ],
            [rule.name for rule in RULE_VERSIONS],
            default="unknow",
        ),
        index=result.index,
    ).astype(str)

    mismatch_mask = result["rule_version"].astype(str).ne(expected_versions)
    if mismatch_mask.any():
        mismatch = result.loc[mismatch_mask, ["issue", "rule_version"]].head(5).copy()
        mismatch["expected"] = expected_versions.loc[mismatch.index]
        raise ValueError(
            "prizes.rule_version 与项目规则边界不一致，示例: "
            f"{mismatch.to_dict('records')}"
        )

    bonus_mask = (
        result["prize_event_type"].astype(str).isin({"basic_bonus", "additional_bonus"})
    )
    if bonus_mask.any():
        known_bonus_mask = pd.Series(False, index=result.index)
        # TODO 此处更改逻辑，对 known_bonus_mask 的结果输出查看是否存在问题
        for campaign in BONUS_CAMPAIGNS:
            known_bonus_mask |= issue_numbers.between(
                campaign.start_issue, campaign.end_issue
            )
        outside_ranges = (
            result.loc[bonus_mask & ~known_bonus_mask, "issue"].astype(str).tolist()
        )
        if outside_ranges:
            raise ValueError(
                "prizes 中派奖事件出现在 PROJECT_CONTEXT 已定义的派奖期号范围之外；"
                f"期号示例: {outside_ranges[:10]}。如新增活动，应先更新业务规则文档。"
            )

    key_columns = ["issue", "rule_version", "prize_rank", "prize_event_type"]
    if result.duplicated(key_columns).any():
        examples = (
            result.loc[result.duplicated(key_columns, keep=False), key_columns]
            .head(5)
            .to_dict("records")
        )
        raise ValueError(f"prizes 存在重复的奖级事件键 {key_columns}，示例: {examples}")

    for column in ("stake_amount", "stake_count", "total_prize_amount"):
        numeric = pd.to_numeric(result[column], errors="coerce")
        invalid = result[column].notna() & numeric.isna()
        if invalid.any():
            raise ValueError(f"prizes.{column} 包含非数值内容。")
        result[column] = numeric

    return result


def validate_draw_features(
    feature_df: pd.DataFrame, draws_df: pd.DataFrame
) -> pd.DataFrame:
    """验证 Draw-level Feature 键和覆盖范围。"""
    validate_required_columns(
        tuple(feature_df.columns), ("draw_index", "issue", "draw_date"), "draw_features"
    )
    result = feature_df.copy()
    result["issue"] = normalize_integer_series(
        result["issue"],
        label="draw_features",
        field_name="issue",
    )
    if result["draw_index"].duplicated().any() or result["issue"].duplicated().any():
        raise ValueError(
            "draw_features 应为一行一个开奖期；draw_index 和 issue 必须唯一。"
        )
    expected = draws_df.loc[:, ["draw_index", "issue", "draw_date"]].copy()
    expected["issue"] = expected["issue"].astype(str)
    expected["__date_key"] = (
        pd.to_datetime(expected["draw_date"], errors="coerce")
        .dt.normalize()
        .astype(str)
    )
    actual = result.loc[:, ["draw_index", "issue", "draw_date"]].copy()
    actual["draw_index"] = pd.to_numeric(actual["draw_index"], errors="coerce")
    actual["issue"] = actual["issue"].astype(str)
    actual["__date_key"] = (
        pd.to_datetime(actual["draw_date"], errors="coerce").dt.normalize().astype(str)
    )
    key_columns = ["draw_index", "issue", "__date_key"]
    if (
        actual["__date_key"].eq("NaT").any()
        or len(actual) != len(expected)
        or not expected.merge(actual, on=key_columns, how="outer", indicator=True)[
            "_merge"
        ]
        .eq("both")
        .all()
    ):
        raise ValueError(
            "draw_features 与 processed draws 的开奖键、日期或覆盖范围不一致。"
        )
    return result.sort_values("draw_index", kind="mergesort").reset_index(drop=True)


def validate_number_features(
    feature_df: pd.DataFrame, draws_df: pd.DataFrame
) -> pd.DataFrame:
    """验证 Number-level Feature 长表键、号码范围与完整实体覆盖。"""
    required = ("draw_index", "issue", "draw_date", "number_zone", "number")
    validate_required_columns(tuple(feature_df.columns), required, "number_features")
    result = feature_df.copy()
    result["issue"] = normalize_integer_series(
        result["issue"],
        label="number_features",
        field_name="issue",
    )
    result["number"] = pd.to_numeric(result["number"], errors="coerce")
    if (
        result["number"].isna().any()
        or not np.equal(result["number"], np.floor(result["number"])).all()
    ):
        raise ValueError("number_features.number 必须全部为整数。")
    result["number"] = result["number"].astype("int64")
    if not set(result["number_zone"].dropna().astype(str)).issubset({"red", "blue"}):
        raise ValueError("number_features.number_zone 仅允许 red 或 blue。")
    if result["number_zone"].isna().any():
        raise ValueError("number_features.number_zone 存在空值。")
    red = result["number_zone"].eq("red")
    if ((result.loc[red, "number"] < 1) | (result.loc[red, "number"] > 35)).any():
        raise ValueError("number_features 前区号码必须位于 1..35。")
    if ((result.loc[~red, "number"] < 1) | (result.loc[~red, "number"] > 12)).any():
        raise ValueError("number_features 后区号码必须位于 1..12。")

    key = ["draw_index", "number_zone", "number"]
    if result.duplicated(key).any():
        examples = (
            result.loc[result.duplicated(key, keep=False), key]
            .head(5)
            .to_dict("records")
        )
        raise ValueError(f"number_features 存在重复实体键，示例: {examples}")
    expected_rows = len(draws_df) * (35 + 12)
    if len(result) != expected_rows:
        raise ValueError(
            "number_features 必须覆盖每期开奖的全部 35 个前区实体和 12 个后区实体；"
            f"预期 {expected_rows} 行，实际 {len(result)} 行。"
        )
    counts = result.groupby(["draw_index", "number_zone"], observed=True)[
        "number"
    ].nunique()
    expected_counts = {"red": 35, "blue": 12}
    for zone, expected_count in expected_counts.items():
        zone_counts = (
            counts.xs(zone, level="number_zone")
            if zone in counts.index.get_level_values("number_zone")
            else pd.Series(dtype=int)
        )
        if (
            len(zone_counts) != len(draws_df)
            or not zone_counts.eq(expected_count).all()
        ):
            raise ValueError(
                f"number_features 每期 {zone} 必须恰好包含 {expected_count} 个号码实体。"
            )
    expected_keys = draws_df.loc[:, ["draw_index", "issue", "draw_date"]].copy()
    expected_keys["issue"] = expected_keys["issue"].astype(str)
    expected_keys["__date_key"] = (
        pd.to_datetime(expected_keys["draw_date"], errors="coerce")
        .dt.normalize()
        .astype(str)
    )
    actual_keys = (
        result.loc[:, ["draw_index", "issue", "draw_date"]].drop_duplicates().copy()
    )
    actual_keys["issue"] = actual_keys["issue"].astype(str)
    actual_keys["__date_key"] = (
        pd.to_datetime(actual_keys["draw_date"], errors="coerce")
        .dt.normalize()
        .astype(str)
    )
    key_columns = ["draw_index", "issue", "__date_key"]
    if (
        actual_keys["__date_key"].eq("NaT").any()
        or len(expected_keys) != len(actual_keys)
        or not expected_keys.merge(
            actual_keys, on=key_columns, how="outer", indicator=True
        )["_merge"]
        .eq("both")
        .all()
    ):
        raise ValueError(
            "number_features 的开奖键、日期或覆盖范围与 processed draws 不一致。"
        )
    return result.sort_values(
        ["draw_index", "number_zone", "number"], kind="mergesort"
    ).reset_index(drop=True)


def validate_prize_features(
    feature_df: pd.DataFrame, prizes_df: pd.DataFrame, draws_df: pd.DataFrame
) -> pd.DataFrame:
    """验证 Prize-level Feature 一行一条 prize record 的粒度。"""
    required = (
        "draw_index",
        "issue",
        "draw_date",
        "rule_version",
        "prize_rank",
        "prize_event_type",
    )
    validate_required_columns(tuple(feature_df.columns), required, "prize_features")
    result = feature_df.copy()
    result["issue"] = normalize_integer_series(
        result["issue"],
        label="prize_features",
        field_name="issue",
    )
    if len(result) != len(prizes_df):
        raise ValueError(
            "prize_features 必须保持 prize record 粒度；"
            f"processed prizes={len(prizes_df)} 行，prize_features={len(result)} 行。"
        )
    keys = ["issue", "rule_version", "prize_rank", "prize_event_type"]
    if result.duplicated(keys).any():
        raise ValueError(f"prize_features 存在重复记录键: {keys}")
    normalized_prize_keys = prizes_df.loc[:, keys].copy()
    normalized_feature_keys = result.loc[:, keys].copy()
    for frame in (normalized_prize_keys, normalized_feature_keys):
        frame["issue"] = frame["issue"].astype(str)
        frame["rule_version"] = frame["rule_version"].astype(str)
        frame["prize_event_type"] = frame["prize_event_type"].astype(str)
        frame["prize_rank"] = pd.to_numeric(frame["prize_rank"], errors="coerce")
    if (
        normalized_prize_keys["prize_rank"].isna().any()
        or normalized_feature_keys["prize_rank"].isna().any()
    ):
        raise ValueError("processed prizes 或 prize_features.prize_rank 包含无效值。")
    comparison = normalized_prize_keys.merge(
        normalized_feature_keys, on=keys, how="outer", indicator=True
    )
    if not comparison["_merge"].eq("both").all():
        raise ValueError(
            "prize_features 的 issue/rule_version/prize_rank/prize_event_type 与 processed prizes 不一致。"
        )
    draw_index_by_issue = draws_df.set_index("issue")["draw_index"].to_dict()
    expected_indices = result["issue"].map(draw_index_by_issue)
    actual_indices = pd.to_numeric(result["draw_index"], errors="coerce")
    if (
        expected_indices.isna().any()
        or actual_indices.isna().any()
        or not actual_indices.astype(int).equals(expected_indices.astype(int))
    ):
        raise ValueError("prize_features.draw_index 与期号映射不一致。")
    date_by_issue = pd.to_datetime(
        draws_df.set_index("issue")["draw_date"], errors="coerce"
    )
    expected_dates = result["issue"].map(date_by_issue)
    actual_dates = pd.to_datetime(result["draw_date"], errors="coerce")
    if (
        expected_dates.isna().any()
        or actual_dates.isna().any()
        or not actual_dates.dt.normalize().equals(expected_dates.dt.normalize())
    ):
        raise ValueError("prize_features.draw_date 与相同期号的开奖日期不一致。")
    return result.sort_values(
        ["draw_index", "prize_rank", "prize_event_type"], kind="mergesort"
    ).reset_index(drop=True)


def validate_feature_metadata(document: Mapping[str, Any]) -> dict[str, Any]:
    """验证 Feature Metadata 的基本顶层结构。"""
    if not isinstance(document, Mapping):
        raise ValueError("feature_metadata.json 顶层必须是 JSON 对象。")
    features = document.get("features")
    if not isinstance(features, Mapping):
        raise ValueError("feature_metadata.json 必须包含对象类型的 features 字段。")
    return dict(document)


def validate_feature_columns_against_metadata(
    draw_features: pd.DataFrame,
    number_features: pd.DataFrame,
    prize_features: pd.DataFrame,
    metadata_document: Mapping[str, Any],
) -> None:
    """检查实际输出的 Feature 字段是否均有 Feature Metadata 定义。

    该检查不复制或重新定义 Feature Schema；字段语义与时间规则仍以
    FEATURE_DESIGN.md 及 feature_metadata.json 为准。
    """
    definitions = metadata_document.get("features", {})
    metadata_names = set(definitions) if isinstance(definitions, Mapping) else set()
    identifiers = {
        "draw_features": {"draw_index", "issue", "draw_date"},
        "number_features": {
            "draw_index",
            "issue",
            "draw_date",
            "number_zone",
            "number",
        },
        "prize_features": {
            "draw_index",
            "issue",
            "draw_date",
            "rule_version",
            "prize_rank",
            "prize_event_type",
        },
    }
    frames = {
        "draw_features": draw_features,
        "number_features": number_features,
        "prize_features": prize_features,
    }
    missing_by_table = {
        table_name: sorted(
            set(frame.columns) - identifiers[table_name] - metadata_names
        )
        for table_name, frame in frames.items()
    }
    missing_by_table = {
        name: fields for name, fields in missing_by_table.items() if fields
    }
    if missing_by_table:
        raise ValueError(
            "Feature 输出中存在未在 feature_metadata.json 中定义的字段: "
            f"{missing_by_table}。请先核对 FEATURE_DESIGN.md、Builder 输出和 Metadata。"
        )
