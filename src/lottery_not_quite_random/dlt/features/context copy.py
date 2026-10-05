"""Business Context Feature。"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from ...config import PRIZE_EVENT_TYPES, RULE_VERSIONS

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class BonusCampaign:
    """一个历史派奖活动的确定性上下文定义。"""

    campaign_id: str
    start_issue: int
    end_issue: int


BONUS_CAMPAIGNS: tuple[BonusCampaign, ...] = (
    BonusCampaign("bonus_2021_21039_21059", 21039, 21059),
    BonusCampaign("bonus_2022_22040_22068", 22040, 22068),
    BonusCampaign("bonus_2023_23039_23059", 23039, 23059),
    BonusCampaign("bonus_2024_24030_24062", 24030, 24062),
    BonusCampaign("bonus_2025_25038_25057", 25038, 25057),
    BonusCampaign("bonus_2026_26050_26066", 26050, 26066),
)


def infer_rule_version(issue: int) -> str:
    """根据期号确定开奖时适用的奖级规则版本。

    Args:
        issue:
            大乐透开奖期号。

    Returns:
        对应的 rule_version。

    Raises:
        ValueError:
            当期号不属于已定义的规则范围时。
    """
    if issue <= 14051:
        return "v1"
    if issue <= 19018:
        return "v2"
    if issue <= 26013:
        return "v3"
    return "v4"


def build_draw_context(draws_df: pd.DataFrame) -> pd.DataFrame:
    """生成 Draw-level Business Context。

    Args:
        draws_df:
            已按 draw_index 升序排列的开奖数据。

    Returns:
        包含 rule_version、is_bonus_period 和 bonus_campaign_id 的 DataFrame。
    """
    context_df = draws_df.loc[:, ["draw_index", "issue"]].copy()
    issue_numbers = pd.to_numeric(context_df["issue"], errors="raise")
    context_df["rule_version"] = issue_numbers.map(infer_rule_version)
    context_df["is_bonus_period"] = False
    context_df["bonus_campaign_id"] = pd.Series(
        pd.NA, index=context_df.index, dtype="string"
    )

    for campaign in BONUS_CAMPAIGNS:
        issue_numbers = pd.to_numeric(context_df["issue"], errors="raise")
        mask = issue_numbers.between(campaign.start_issue, campaign.end_issue)
        context_df.loc[mask, "is_bonus_period"] = True
        context_df.loc[mask, "bonus_campaign_id"] = campaign.campaign_id

    return context_df


def validate_prize_context(prizes_df: pd.DataFrame) -> None:
    """验证 Prize-level Business Context 的允许集合与规则版本。

    Args:
        prizes_df:
            processed/prizes.parquet 数据。

    Raises:
        ValueError:
            当 rule_version 或 prize_event_type 非法时。
    """
    unknown_rules = sorted(
        set(prizes_df["rule_version"].dropna().astype(str)) - set(RULE_VERSIONS)
    )
    if unknown_rules:
        raise ValueError(f"未知 rule_version: {unknown_rules}")

    unknown_events = sorted(
        set(prizes_df["prize_event_type"].dropna().astype(str)) - set(PRIZE_EVENT_TYPES)
    )
    if unknown_events:
        raise ValueError(f"未知 prize_event_type: {unknown_events}")
