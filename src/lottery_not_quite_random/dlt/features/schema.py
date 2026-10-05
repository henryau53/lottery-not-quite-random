"""Feature 输出 Schema 与输入契约。"""

from __future__ import annotations

from dataclasses import dataclass

DRAW_ID_COLUMNS = ("draw_index", "issue", "draw_date")

NUMBER_ID_COLUMNS = (
    "draw_index",
    "issue",
    "draw_date",
    "number_zone",
    "number",
)

PRIZE_ID_COLUMNS = (
    "draw_index",
    "issue",
    "draw_date",
    "rule_version",
    "prize_rank",
    "prize_event_type",
)

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

PRIZE_REQUIRED_INPUT_COLUMNS = (
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


def build_draw_schema(feature_columns: list[str]) -> FeatureSchema:
    """创建 Draw-level Schema。"""
    return FeatureSchema(
        required_columns=(*DRAW_ID_COLUMNS, *feature_columns),
        feature_columns=tuple(feature_columns),
    )


def build_number_schema(feature_columns: list[str]) -> FeatureSchema:
    """创建 Number-level Schema。"""
    return FeatureSchema(
        required_columns=(*NUMBER_ID_COLUMNS, *feature_columns),
        feature_columns=tuple(feature_columns),
        nullable_columns=(
            "missing_current",
            *[name for name in feature_columns if name.startswith("frequency_")],
        ),
    )


def build_prize_schema(feature_columns: list[str]) -> FeatureSchema:
    """创建 Prize-level Schema。"""
    return FeatureSchema(
        required_columns=(*PRIZE_ID_COLUMNS, *feature_columns),
        feature_columns=tuple(feature_columns),
    )
