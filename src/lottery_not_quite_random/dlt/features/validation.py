"""Feature 输入、输出与时间边界验证。"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import pandas as pd

from .schema import DRAW_REQUIRED_INPUT_COLUMNS, PRIZE_REQUIRED_INPUT_COLUMNS

LOGGER = logging.getLogger(__name__)


def validate_draw_input(draws_df: pd.DataFrame) -> pd.DataFrame:
    """验证并标准化 processed draws 输入。

    Args:
        draws_df:
            processed/draws.parquet。

    Returns:
        按 draw_index 升序、索引重置后的 DataFrame。

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

    result = draws_df.copy()
    result = result.sort_values("draw_index", kind="mergesort", ignore_index=True)

    expected_index = list(range(len(result)))
    actual_index = result["draw_index"].astype(int).tolist()
    if actual_index != expected_index:
        raise ValueError("draw_index 必须从 0 开始连续递增。")

    if result["issue"].duplicated().any():
        raise ValueError("issue 必须唯一。")

    if result["draw_date"].isna().any():
        raise ValueError("draw_date 不允许为空。")

    for column in [
        *(f"red_{i}" for i in range(1, 6)),
        *(f"blue_{i}" for i in range(1, 3)),
    ]:
        if result[column].isna().any():
            raise ValueError(f"{column} 不允许为空。")

    red_columns = [f"red_{i}" for i in range(1, 6)]
    blue_columns = [f"blue_{i}" for i in range(1, 3)]

    if not result[red_columns].apply(lambda col: col.between(1, 35)).all().all():
        raise ValueError("red 号码存在非法值。")
    if not result[blue_columns].apply(lambda col: col.between(1, 12)).all().all():
        raise ValueError("blue 号码存在非法值。")

    if result[red_columns].apply(lambda row: row.nunique() != 5, axis=1).any():
        raise ValueError("单期开奖 red 号码必须互不重复。")
    if result[blue_columns].apply(lambda row: row.nunique() != 2, axis=1).any():
        raise ValueError("单期开奖 blue 号码必须互不重复。")

    return result


def validate_prize_input(prizes_df: pd.DataFrame) -> pd.DataFrame:
    """验证 processed prizes 输入。

    Args:
        prizes_df:
            processed/prizes.parquet。

    Returns:
        复制后的 prizes DataFrame。

    Raises:
        ValueError:
            当 required columns 缺失时。
    """
    missing = [
        column
        for column in PRIZE_REQUIRED_INPUT_COLUMNS
        if column not in prizes_df.columns
    ]
    if missing:
        raise ValueError(f"prizes 缺少 required columns: {missing}")

    result = prizes_df.copy()
    if "draw_index" in result.columns and result["draw_index"].isna().any():
        raise ValueError("prizes.draw_index 不允许为空。")
    return result


def validate_feature_output(
    df: pd.DataFrame,
    required_columns: Iterable[str],
    name: str,
) -> None:
    """验证 Feature 输出字段与基本空值契约。

    Args:
        df:
            Feature DataFrame。
        required_columns:
            必须存在的字段。
        name:
            输出数据集名称。

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
