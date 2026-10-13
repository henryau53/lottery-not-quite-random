"""输入数据覆盖、粒度与缺失概况。"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd


def _table_overview(table_name: str, df: pd.DataFrame, path: Path | None = None) -> dict[str, Any]:
    """为一张表构建总体覆盖摘要。"""
    result: dict[str, Any] = {
        "table": table_name,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "null_cells": int(df.isna().to_numpy().sum()),
        "cell_count": int(df.shape[0] * df.shape[1]),
        "null_rate": float(df.isna().to_numpy().sum() / (df.shape[0] * df.shape[1]))
        if df.size else 0.0,
    }
    if path is not None:
        result["path"] = str(path)
    if "draw_index" in df.columns and not df.empty:
        index = pd.to_numeric(df["draw_index"], errors="coerce")
        result["draw_index_min"] = int(index.min()) if index.notna().any() else None
        result["draw_index_max"] = int(index.max()) if index.notna().any() else None
    if "issue" in df.columns and not df.empty:
        result["issue_min"] = str(df["issue"].astype(str).min())
        result["issue_max"] = str(df["issue"].astype(str).max())
    if "draw_date" in df.columns and not df.empty:
        dates = pd.to_datetime(df["draw_date"], errors="coerce").dropna()
        result["draw_date_min"] = dates.min().date().isoformat() if not dates.empty else None
        result["draw_date_max"] = dates.max().date().isoformat() if not dates.empty else None
    return result


def build_data_overview(
    tables: Mapping[str, tuple[pd.DataFrame, Path | None]],
    feature_metadata: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """生成表级概况与字段级缺失统计。"""
    table_rows: list[dict[str, Any]] = []
    column_rows: list[dict[str, Any]] = []
    for table_name, (df, path) in tables.items():
        table_rows.append(_table_overview(table_name, df, path))
        for column in df.columns:
            null_count = int(df[column].isna().sum())
            column_rows.append(
                {
                    "table": table_name,
                    "column": str(column),
                    "dtype": str(df[column].dtype),
                    "row_count": int(len(df)),
                    "null_count": null_count,
                    "null_rate": float(null_count / len(df)) if len(df) else 0.0,
                }
            )
    overview_df = pd.DataFrame(table_rows)
    column_nulls_df = pd.DataFrame(column_rows)
    if feature_metadata is not None:
        features = feature_metadata.get("features", {})
        overview_df.attrs["feature_definition_count"] = len(features) if isinstance(features, Mapping) else 0
    return overview_df, column_nulls_df


def build_prize_summary(prizes_df: pd.DataFrame, draws_df: pd.DataFrame) -> pd.DataFrame:
    """按规则版本、派奖事件类型和奖级分层汇总奖金数据。"""
    columns = [
        "rule_version", "prize_event_type", "prize_rank", "record_count",
        "stake_amount_non_null_count", "stake_count_non_null_count", "stake_count_sum",
        "total_prize_amount_non_null_count", "total_prize_amount_sum",
    ]
    if prizes_df.empty:
        return pd.DataFrame(columns=columns)
    value = prizes_df.copy()
    value["prize_rank"] = pd.to_numeric(value["prize_rank"], errors="coerce")
    grouped = value.groupby(
        ["rule_version", "prize_event_type", "prize_rank"], dropna=False, sort=True
    )
    rows: list[dict[str, Any]] = []
    for (rule_version, event_type, prize_rank), group in grouped:
        rows.append(
            {
                "rule_version": str(rule_version),
                "prize_event_type": str(event_type),
                "prize_rank": int(prize_rank) if pd.notna(prize_rank) else None,
                "record_count": int(len(group)),
                "stake_amount_non_null_count": int(group["stake_amount"].notna().sum()),
                "stake_count_non_null_count": int(group["stake_count"].notna().sum()),
                "stake_count_sum": float(group["stake_count"].sum(min_count=1)) if group["stake_count"].notna().any() else None,
                "total_prize_amount_non_null_count": int(group["total_prize_amount"].notna().sum()),
                "total_prize_amount_sum": float(group["total_prize_amount"].sum(min_count=1)) if group["total_prize_amount"].notna().any() else None,
            }
        )
    return pd.DataFrame(rows, columns=columns)
