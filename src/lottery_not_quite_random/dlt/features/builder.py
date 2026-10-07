"""Feature Builder 统一入口。"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from ...config import (
    DLT_FEATURE_DIR,
    DLT_FEATURE_DRAW_FILE,
    DLT_FEATURE_METADATA_FILE,
    DLT_FEATURE_NUMBER_FILE,
    DLT_FEATURE_PRIZE_FILE,
    DLT_PROCESSED_DRAWS_FILE,
    DLT_PROCESSED_PRIZES_FILE,
    FEATURE_FREQUENCY_WINDOWS,
    FEATURE_REPEAT_WINDOWS,
    FEATURE_ROLLING_WINDOWS,
)
from .context import build_draw_context
from .frequency import build_frequency_features
from .metadata import build_metadata_document, write_metadata
from .missing import build_missing_features
from .prize import build_prize_features
from .repeat import build_prev_draw_hit, build_repeat_history_features
from .rolling import build_rolling_features
from .schema import (
    build_draw_schema,
    build_number_schema,
    build_prize_schema,
)
from .validation import (
    validate_draw_input,
    validate_feature_output,
    validate_prize_input,
)

LOGGER = logging.getLogger(__name__)


def build_features() -> dict[str, Path]:
    """从 processed 数据构建第一阶段 Feature 输出。

    Returns:
        dict[str, Path]:
            生成的输出文件路径，包含：

            - ``draw``: draw_features.parquet
            - ``number``: number_features.parquet
            - ``prize``: prize_features.parquet
            - ``metadata``: feature_metadata.json

    Raises:
        FileNotFoundError:
            当 processed 输入文件不存在时。

        ValueError:
            当输入或输出不符合 Feature 契约时。
    """

    draw_path = DLT_PROCESSED_DRAWS_FILE
    prize_path = DLT_PROCESSED_PRIZES_FILE

    if not draw_path.exists():
        raise FileNotFoundError(f"缺少 processed draws: {draw_path}")

    if not prize_path.exists():
        raise FileNotFoundError(f"缺少 processed prizes: {prize_path}")

    LOGGER.info(
        "读取 processed: draws=%s prizes=%s",
        draw_path,
        prize_path,
    )

    draws_df = (
        pd.read_parquet(draw_path)
        .copy()
        .sort_values("draw_index", kind="mergesort", ignore_index=True)
    )
    validate_draw_input(draws_df)

    prizes_df = pd.read_parquet(prize_path).copy()
    validate_prize_input(prizes_df)

    LOGGER.info(
        "开奖数据输入: rows=%d",
        len(draws_df),
    )

    LOGGER.info(
        "奖金数据输入: rows=%d",
        len(prizes_df),
    )

    # -------------------------------------------------------------------------
    # 构建开奖级上下文
    # -------------------------------------------------------------------------

    context_df = build_draw_context(draws_df)

    # -------------------------------------------------------------------------
    # 开奖级（Draw-level）特征值
    # -------------------------------------------------------------------------

    rolling_df = build_rolling_features(draws_df)

    repeat_history_df = build_repeat_history_features(draws_df)

    draw_features = context_df.merge(
        rolling_df,
        on=["draw_index", "issue", "draw_date"],
        how="left",
        validate="one_to_one",
    ).merge(
        repeat_history_df,
        on=["draw_index", "issue", "draw_date"],
        how="left",
        validate="one_to_one",
    )

    draw_features = draw_features.sort_values(
        "draw_index",
        kind="mergesort",
    ).reset_index(drop=True)

    # -------------------------------------------------------------------------
    # 号码级（Number-level）特征值
    # -------------------------------------------------------------------------

    frequency_df = build_frequency_features(draws_df)

    missing_df = build_missing_features(draws_df)

    previous_df = build_prev_draw_hit(draws_df)

    number_features = frequency_df.merge(
        missing_df,
        on=[
            "draw_index",
            "issue",
            "draw_date",
            "number_zone",
            "number",
        ],
        how="left",
        validate="one_to_one",
    ).merge(
        previous_df,
        on=[
            "draw_index",
            "issue",
            "draw_date",
            "number_zone",
            "number",
        ],
        how="left",
        validate="one_to_one",
    )

    number_features = number_features.sort_values(
        [
            "draw_index",
            "number_zone",
            "number",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    # -------------------------------------------------------------------------
    # 奖金级（Prize-level）特征值
    #
    # 第一阶段不新增具体 Prize 派生统计，只稳定保留 prize record 粒度、
    # 关联字段、原始奖金事实以及业务上下文。
    # -------------------------------------------------------------------------

    prize_features = build_prize_features(
        prizes_df,
        context_df,
    )

    # -------------------------------------------------------------------------
    # Schema validation
    # -------------------------------------------------------------------------

    build_draw_schema().validate_columns(draw_features.columns.tolist())

    build_number_schema().validate_columns(number_features.columns.tolist())

    build_prize_schema().validate_columns(prize_features.columns.tolist())

    # -------------------------------------------------------------------------
    # Feature output validation
    # -------------------------------------------------------------------------

    # validate_feature_output(
    #     draw_features,
    #     [
    #         "draw_index",
    #         "issue",
    #         "draw_date",
    #     ],
    #     "draw_features",
    # )

    # validate_feature_output(
    #     number_features,
    #     [
    #         "draw_index",
    #         "issue",
    #         "draw_date",
    #         "number_zone",
    #         "number",
    #     ],
    #     "number_features",
    # )

    # validate_feature_output(
    #     prize_features,
    #     [
    #         "draw_index",
    #         "issue",
    #         "draw_date",
    #         "rule_version",
    #         "prize_rank",
    #         "prize_event_type",
    #     ],
    #     "prize_features",
    # )

    # -------------------------------------------------------------------------
    # Output paths
    # -------------------------------------------------------------------------

    # DLT_FEATURE_DIR.mkdir(
    #     parents=True,
    #     exist_ok=True,
    # )

    # -------------------------------------------------------------------------
    # Write Feature outputs
    # -------------------------------------------------------------------------

    # draw_features.to_parquet(
    #     DLT_FEATURE_DRAW_FILE,
    #     index=False,
    # )

    # number_features.to_parquet(
    #     DLT_FEATURE_NUMBER_FILE,
    #     index=False,
    # )

    # prize_features.to_parquet(
    #     DLT_FEATURE_PRIZE_FILE,
    #     index=False,
    # )

    # # -------------------------------------------------------------------------
    # # Metadata
    # # -------------------------------------------------------------------------

    # metadata = build_metadata_document(
    #     output_paths=[
    #         DLT_FEATURE_DRAW_FILE,
    #         DLT_FEATURE_NUMBER_FILE,
    #         DLT_FEATURE_PRIZE_FILE,
    #     ],
    #     processed_paths=[
    #         draw_path,
    #         prize_path,
    #     ],
    #     feature_columns={
    #         "draw": draw_feature_columns,
    #         "number": number_feature_columns,
    #         "prize": [
    #             *prize_feature_columns,
    #             "prize_event_type",
    #         ],
    #     },
    # )

    # write_metadata(
    #     metadata,
    #     DLT_FEATURE_METADATA_FILE,
    # )

    # LOGGER.info(
    #     "Feature 构建完成: draw_rows=%d number_rows=%d prize_rows=%d",
    #     len(draw_features),
    #     len(number_features),
    #     len(prize_features),
    # )

    return {
        "draw": DLT_FEATURE_DRAW_FILE,
        "number": DLT_FEATURE_NUMBER_FILE,
        "prize": DLT_FEATURE_PRIZE_FILE,
        "metadata": DLT_FEATURE_METADATA_FILE,
    }
