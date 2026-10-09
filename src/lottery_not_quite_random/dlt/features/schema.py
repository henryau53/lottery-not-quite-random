"""Feature 输出 Schema 与输入契约。

该模块定义各 Feature 数据集（Draw-level、Number-level、Prize-level）的
标识列、字段契约以及相应的 Schema 构造函数，供特征输出校验使用。
"""

from __future__ import annotations

from dataclasses import dataclass

from ...config import (
    FEATURE_FREQUENCY_WINDOWS,
    FEATURE_REPEAT_WINDOWS,
    FEATURE_ROLLING_WINDOWS,
)
from .rolling import ROLLING_SOURCES


@dataclass(frozen=True)
class FeatureSchema:
    """描述一个 Feature 数据集的字段契约。

    包含必需字段、特征字段以及允许为空的字段。构造后不可变，供各数据集
    的 Schema 构造函数返回使用。

    Attributes:
        required_columns:
            数据集中必须存在的字段，通常由标识字段与特征字段组成。
        feature_columns:
            由特征工程生成的特征字段。
        nullable_columns:
            允许为空的字段，默认无。
    """

    required_columns: tuple[str, ...]
    feature_columns: tuple[str, ...]
    nullable_columns: tuple[str, ...] = ()

    def validate_columns(self, columns: list[str]) -> None:
        """验证必需字段是否全部存在。

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
    """创建开奖级（Draw-level）Schema。

    Returns:
        返回开奖级数据集的字段契约。
    """
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
            *("draw_index", "issue", "draw_date"),
            *feature_columns,
        ),
        feature_columns=feature_columns,
    )


def build_number_schema() -> FeatureSchema:
    """创建号码级（Number-level）Schema。

    Returns:
        返回号码级数据集的字段契约。
    """
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
            *(
                "draw_index",
                "issue",
                "draw_date",
                "number_zone",
                "number",
            ),
            *feature_columns,
        ),
        feature_columns=feature_columns,
        nullable_columns=feature_columns,
    )


def build_prize_schema() -> FeatureSchema:
    """创建奖金级（Prize-level）Schema。

    Returns:
        返回奖金级数据集的字段契约。
    """

    prize_context_columns = (
        "is_bonus_period",
        "bonus_campaign_id",
    )

    return FeatureSchema(
        required_columns=(
            *(
                "draw_index",
                "issue",
                "draw_date",
                "rule_version",
                "prize_rank",
                "prize_event_type",
            ),
            *prize_context_columns,
        ),
        feature_columns=prize_context_columns,
    )
