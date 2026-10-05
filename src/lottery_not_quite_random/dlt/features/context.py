"""开奖上下文与奖级规则版本查询。

本模块负责根据开奖期号确定：
- 奖级规则版本
- 派奖活动
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ...config import (
    BONUS_CAMPAIGNS,
    PRIZE_EVENT_TYPES,
    RULE_VERSION_NAMES,
    RULE_VERSIONS,
    BonusCampaign,
    RuleVersion,
)


@dataclass(frozen=True)
class DrawContext:
    """某一期开奖对应的奖级规则版本上下文。"""

    issue: int
    rule_version: str
    bonus_campaign: str | None


def get_rule_version_config(issue: int) -> RuleVersion:
    """根据开奖期号获取对应的奖级规则版本配置。

    Args:
        issue: 开奖期号。

    Returns:
        匹配到的奖级规则版本。

    Raises:
        ValueError: 没有任何规则版本匹配当前期号。
    """

    for rule in RULE_VERSIONS:
        if rule.matches(issue):
            return rule

    raise ValueError(f"无法确定 issue={issue} 对应的奖级规则版本 rule_version")


def get_rule_version(issue: int) -> str:
    """根据开奖期号获取规则版本名称。

    Args:
        issue: 开奖期号。

    Returns:
        匹配到的奖级规则版本名称。
    """
    return get_rule_version_config(issue).name


def get_bonus_campaign_config(
    issue: int,
) -> BonusCampaign | None:
    """根据开奖期号获取对应的派奖活动。

    Args:
        issue: 开奖期号。

    Returns:
        匹配到的派奖活动，如果该期不属于任何派奖活动，则返回 None。
    """

    for campaign in BONUS_CAMPAIGNS:
        if campaign.matches(issue):
            return campaign

    return None


def get_bonus_campaign(issue: int) -> str | None:
    """根据开奖期号获取对应的派奖活动名称。

    Args:
        issue: 开奖期号。

    Returns:
        匹配到的派奖活动名称，如果该期不属于任何派奖活动，则返回 None。
    """

    campaign = get_bonus_campaign_config(issue)

    if campaign is None:
        return None

    return campaign.name


def validate_prize_context(prizes_df: pd.DataFrame) -> None:
    """校验奖金级（Prize-level）业务上下文数据。

    Args:
        prizes_df:
            processed/prizes.parquet 数据。

    Raises:
        ValueError:
            当 rule_version 或 prize_event_type 非法时。
    """

    unknown_rules = sorted(
        set(prizes_df["rule_version"].dropna().astype(str)) - set(RULE_VERSION_NAMES)
    )

    if unknown_rules:
        raise ValueError(f"未知奖级规则版本 rule_version: {unknown_rules}")

    unknown_events = sorted(
        set(prizes_df["prize_event_type"].dropna().astype(str)) - set(PRIZE_EVENT_TYPES)
    )

    if unknown_events:
        raise ValueError(f"未知奖金事件类型 prize_event_type: {unknown_events}")


def get_draw_context(issue: int) -> DrawContext:
    """根据开奖期号构建完整的开奖上下文环境。

    Args:
        issue: 开奖期号。

    Returns:
        开奖上下文环境
    """

    return DrawContext(
        issue=issue,
        rule_version=get_rule_version(issue),
        bonus_campaign=get_bonus_campaign(issue),
    )


def build_draw_context(draws_df: pd.DataFrame) -> pd.DataFrame:
    """生成开奖级（Draw-level）业务上下文数据。

    Args:
        draws_df:
            已按 draw_index 升序排列的开奖数据。

    Returns:
        包含 rule_version、is_bonus_period 和 bonus_campaign_id 的 DataFrame。
    """
    context_df = draws_df.loc[:, ["draw_index", "issue", "draw_date"]].copy()

    issue_numbers = pd.to_numeric(
        context_df["issue"],
        errors="raise",
    )

    context_df["rule_version"] = issue_numbers.map(get_rule_version)

    context_df["is_bonus_period"] = False

    context_df["bonus_campaign_id"] = pd.Series(
        pd.NA,
        index=context_df.index,
        dtype="string",
    )

    for campaign in BONUS_CAMPAIGNS:
        mask = issue_numbers.between(
            campaign.start_issue,
            campaign.end_issue,
        )

        context_df.loc[mask, "is_bonus_period"] = True
        context_df.loc[mask, "bonus_campaign_id"] = campaign.name

    return context_df
