"""Feature 输出 Schema 与输入契约。"""

from __future__ import annotations

from dataclasses import dataclass

from ...config import (
    FEATURE_FREQUENCY_WINDOWS,
    FEATURE_REPEAT_WINDOWS,
    FEATURE_ROLLING_WINDOWS,
)
from .rolling import ROLLING_SOURCES

# 开奖级（Draw-level）数据集的标识字段
DRAW_ID_COLUMNS = ("draw_index", "issue", "draw_date")

# 号码级（Number-level）数据集的标识字段
NUMBER_ID_COLUMNS = (
    "draw_index",
    "issue",
    "draw_date",
    "number_zone",
    "number",
)

# 奖金级（Prize-level）数据集的标识字段
PRIZE_ID_COLUMNS = (
    "draw_index",
    "issue",
    "draw_date",
    "rule_version",
    "prize_rank",
    "prize_event_type",
)


@dataclass(frozen=True)
class FeatureSchema:
    """描述一个 Feature 数据集的字段契约。"""

    required_columns: tuple[str, ...]
    feature_columns: tuple[str, ...]
    nullable_columns: tuple[str, ...] = ()

    def validate_columns(self, columns: list[str]) -> None:
        """验证必需字段存在。

        Args:
            columns:
                DataFrame 当前字段名。

        Raises:
            ValueError:
                当必需字段缺失时。
        """
        missing = [column for column in self.required_columns if column not in columns]
        if missing:
            raise ValueError(f"缺少 required columns: {missing}")


def build_draw_schema() -> FeatureSchema:
    """创建 Draw-level Schema。"""
    rolling_feature_columns = tuple(
        column
        for source_column, _ in ROLLING_SOURCES
        for window in FEATURE_ROLLING_WINDOWS
        for column in (
            f"{source_column}_mean_{window}",
            f"{source_column}_std_{window}",
        )
    )

    repeat_feature_columns = tuple(
        f"repeat_count_mean_{window}" for window in FEATURE_REPEAT_WINDOWS
    )

    feature_columns = (
        *rolling_feature_columns,
        *repeat_feature_columns,
    )

    return FeatureSchema(
        required_columns=(
            *DRAW_ID_COLUMNS,
            *feature_columns,
        ),
        feature_columns=feature_columns,
    )


def build_number_schema() -> FeatureSchema:
    """创建 Number-level Schema。"""
    frequency_feature_columns = tuple(
        f"frequency_{window}" for window in FEATURE_FREQUENCY_WINDOWS
    )

    repeat_feature_columns = ("prev_draw_hit",)

    missing_feature_columns = ("missing_current",)

    feature_columns = (
        *frequency_feature_columns,
        *repeat_feature_columns,
        *missing_feature_columns,
    )

    return FeatureSchema(
        required_columns=(
            *NUMBER_ID_COLUMNS,
            *feature_columns,
        ),
        feature_columns=feature_columns,
        nullable_columns=feature_columns,
    )


def build_prize_schema() -> FeatureSchema:
    """创建 Prize-level Schema。"""

    prize_context_columns = (
        "is_bonus_period",
        "bonus_campaign_id",
    )

    return FeatureSchema(
        required_columns=(*PRIZE_ID_COLUMNS, *prize_context_columns),
        feature_columns=prize_context_columns,
    )
