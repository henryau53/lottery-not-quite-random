"""Prize-level Feature 构建。"""

from __future__ import annotations

import pandas as pd

from .context import validate_prize_context
from .schema import PRIZE_ID_COLUMNS


def build_prize_features(
    prizes_df: pd.DataFrame,
    draw_context_df: pd.DataFrame,
) -> pd.DataFrame:
    """构建第一阶段 Prize-level 输出。

    第一阶段不新增具体 Prize 派生统计，只稳定保留 prize record 粒度、
    关联字段、原始奖金事实以及 Business Context。

    Args:
        prizes_df:
            processed/prizes.parquet。
        draw_context_df:
            Draw-level Business Context。

    Returns:
        prize_features.parquet 对应的 DataFrame。
    """
    validate_prize_context(prizes_df)

    context_columns = ["draw_index", "rule_version", "is_bonus_period", "bonus_campaign_id"]
    context = draw_context_df.loc[:, context_columns].copy()

    output = prizes_df.copy()
    if "draw_index" not in output.columns:
        issue_to_index = (
            draw_context_df.loc[:, ["issue", "draw_index"]]
            .drop_duplicates("issue")
        )
        output = output.merge(issue_to_index, on="issue", how="left", validate="many_to_one")

    output = output.merge(
        context,
        on=["draw_index", "rule_version"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_context"),
    )

    if "is_bonus_period_context" in output.columns:
        output["is_bonus_period"] = output["is_bonus_period_context"]
        output = output.drop(columns=["is_bonus_period_context"])

    if "bonus_campaign_id_context" in output.columns:
        output["bonus_campaign_id"] = output["bonus_campaign_id_context"]
        output = output.drop(columns=["bonus_campaign_id_context"])

    if "rule_version_context" in output.columns:
        output = output.drop(columns=["rule_version_context"])

    ordered = [column for column in PRIZE_ID_COLUMNS if column in output.columns]
    remaining = [column for column in output.columns if column not in ordered]
    output = output.loc[:, [*ordered, *remaining]]

    return output.sort_values(
        ["draw_index", "prize_rank", "prize_event_type"],
        kind="mergesort",
    ).reset_index(drop=True)
