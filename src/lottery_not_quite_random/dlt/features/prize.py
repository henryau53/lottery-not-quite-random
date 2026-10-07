"""奖金级特征构建（Prize-level Feature 构建）。

用于体彩大乐透 processed 数据的奖级级特征构建。第一阶段以 prizes 数据中的
奖级记录为粒度，稳定保留奖级记录标识、开奖关联字段、原始奖金事实以及
Draw-level 业务上下文，不新增具体奖级派生统计。
"""

from __future__ import annotations

import pandas as pd

from .schema import PRIZE_ID_COLUMNS


def build_prize_features(
    prizes_df: pd.DataFrame,
    draw_context_df: pd.DataFrame,
) -> pd.DataFrame:
    """构建 Prize-level 特征。

    以 prizes_df 的奖级记录为粒度，将其与 draw_context_df
    提供的开奖级（Draw-level）业务上下文关联。若 prizes_df 缺少 draw_index，则通过
    issue 从 draw_context_df 映射补全。关联字段包括 draw_index、rule_version、
    is_bonus_period 和 bonus_campaign_id；合并过程中会处理同名列冲突，确保最终
    采用业务上下文中的 is_bonus_period、bonus_campaign_id 和 rule_version。

    Args:
        prizes_df:
            processed/prizes.parquet 对应的奖级记录数据。
        draw_context_df:
            开奖级（Draw-level）业务上下文数据。

    Returns:
        prize_features.parquet 对应的结果表。每行对应一条奖级记录，保留原始
        奖金事实，并补充 draw_index、rule_version、is_bonus_period、
        bonus_campaign_id 等业务上下文字段。列顺序优先使用 PRIZE_ID_COLUMNS，
        其余列保持原有顺序；行按 draw_index、prize_rank、prize_event_type
        稳定排序。
    """

    context_columns = [
        "draw_index",
        "rule_version",
        "is_bonus_period",
        "bonus_campaign_id",
    ]
    context = draw_context_df.loc[:, context_columns].copy()

    output = prizes_df.copy()
    if "draw_index" not in output.columns:
        issue_to_index = draw_context_df.loc[
            :, ["issue", "draw_index"]
        ].drop_duplicates("issue")
        output = output.merge(
            issue_to_index, on="issue", how="left", validate="many_to_one"
        )

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
