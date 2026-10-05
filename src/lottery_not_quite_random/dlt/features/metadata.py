"""Feature Metadata 生成。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ...config import (
    FEATURE_DEFAULT_PREDICTIVE_SHIFT,
    FEATURE_CODE_VERSION,
    FEATURE_DEFINITION_VERSION,
    FEATURE_METADATA_VERSION,
    FEATURE_FREQUENCY_WINDOWS,
    FEATURE_REPEAT_WINDOWS,
    FEATURE_ROLLING_WINDOWS,
)


def file_sha256(path: Path) -> str:
    """计算文件 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _base_metadata(
    name: str,
    category: str,
    level: str,
    entity: str,
    definition: str,
    source: list[str],
    formula: str,
    dtype: str,
    window: int | None,
    research_role: list[str],
    availability: str = "pre_draw",
    includes_current_draw: bool = False,
    shift: int = FEATURE_DEFAULT_PREDICTIVE_SHIFT,
    leakage_rule: str = "仅能使用当前 draw_index 之前已经发生的数据。",
) -> dict[str, Any]:
    """创建单个 Feature Metadata。"""
    return {
        "name": name,
        "category": category,
        "level": level,
        "entity": entity,
        "definition": definition,
        "source": source,
        "formula": formula,
        "dtype": dtype,
        "window": window,
        "observation_point": "before_current_draw",
        "includes_current_draw": includes_current_draw,
        "shift": shift,
        "availability": availability,
        "leakage_rule": leakage_rule,
        "research_role": research_role,
        "version": FEATURE_DEFINITION_VERSION,
    }


def build_feature_definitions() -> dict[str, dict[str, Any]]:
    """根据第一阶段正式 Feature 集合生成 Metadata 定义。"""
    definitions: dict[str, dict[str, Any]] = {}

    for window in FEATURE_FREQUENCY_WINDOWS:
        definitions[f"frequency_{window}"] = _base_metadata(
            f"frequency_{window}",
            "frequency",
            "number",
            "number",
            f"当前期之前最近 {window} 期开奖中该号码出现的次数。",
            ["processed.draws.red_1..red_5", "processed.draws.blue_1..blue_2"],
            f"sum(hit[number, t-{window}:t-1])",
            "nullable[int]",
            window,
            ["historical_state", "predictive", "scoring"],
        )

    definitions["missing_current"] = _base_metadata(
        "missing_current",
        "missing",
        "number",
        "number",
        "当前期之前，号码距离最近一次出现经过的期数；历史范围内从未出现时为 null。",
        ["processed.draws.red_1..red_5", "processed.draws.blue_1..blue_2"],
        "t - last_hit_position - 1",
        "nullable[int]",
        None,
        ["historical_state", "predictive", "scoring"],
    )

    definitions["prev_draw_hit"] = _base_metadata(
        "prev_draw_hit",
        "repeat",
        "number",
        "number",
        "某号码是否出现在上一期开奖。",
        ["processed.draws.red_1..red_5", "processed.draws.blue_1..blue_2"],
        "number in draw[t-1]",
        "nullable[bool]",
        1,
        ["historical_state", "predictive", "scoring"],
    )

    rolling_sources = (
        ("red_sum", "red", "processed.draws.red_sum"),
        ("blue_sum", "blue", "processed.draws.blue_sum"),
        ("red_span", "red", "processed.draws.red_span"),
        ("blue_span", "blue", "processed.draws.blue_span"),
        ("red_odd_count", "red", "processed.draws.red_odd_count"),
        ("blue_odd_count", "blue", "processed.draws.blue_odd_count"),
        (
            "red_consecutive_group_count",
            "red",
            "processed.draws.red_consecutive_group_count",
        ),
        (
            "red_max_consecutive_length",
            "red",
            "processed.draws.red_max_consecutive_length",
        ),
    )
    for object_name, entity, source in rolling_sources:
        for window in FEATURE_ROLLING_WINDOWS:
            for aggregation in ("mean", "std"):
                name = f"{object_name}_{aggregation}_{window}"
                definitions[name] = _base_metadata(
                    name,
                    "rolling" if "consecutive" not in object_name else "structure",
                    "draw",
                    entity,
                    f"当前期之前最近 {window} 期开奖的 {object_name} {aggregation}。",
                    [source],
                    f"{aggregation}({object_name}[t-{window}:t-1])",
                    "float64",
                    window,
                    ["historical_state", "predictive", "scoring"],
                )

    for window in FEATURE_REPEAT_WINDOWS:
        name = f"repeat_count_mean_{window}"
        definitions[name] = _base_metadata(
            name,
            "repeat",
            "draw",
            "draw",
            f"当前期之前最近 {window} 次相邻开奖关系中重复号码数量的平均值。",
            ["processed.draws.red_1..red_5", "processed.draws.blue_1..blue_2"],
            f"mean(repeat_count[t-{window}:t-1])",
            "float64",
            window,
            ["historical_state", "predictive", "scoring"],
        )

    context_defs = {
        "rule_version": (
            "当前开奖适用的彩票规则版本。",
            "issue -> v1/v2/v3/v4",
            "string",
            None,
            "context",
            "draw",
        ),
        "is_bonus_period": (
            "当前开奖是否处于历史派奖活动期间。",
            "issue in configured bonus campaign ranges",
            "bool",
            None,
            "context",
            "draw",
        ),
        "bonus_campaign_id": (
            "当前开奖对应的确定性派奖活动编号。",
            "configured bonus campaign range -> campaign_id",
            "string",
            None,
            "context",
            "bonus",
        ),
    }
    for name, (
        definition,
        formula,
        dtype,
        window,
        category,
        entity,
    ) in context_defs.items():
        definitions[name] = _base_metadata(
            name,
            category,
            entity,
            entity,
            definition,
            ["processed.draws.issue"],
            formula,
            dtype,
            window,
            ["historical_state", "predictive", "scoring"],
            leakage_rule="仅使用开奖前已确定的规则/派奖活动信息。",
        )

    prize_fact_defs = {
        "prize_level_raw": (
            "官方原始奖级字段。",
            "processed.prizes.prize_level_raw",
            "copy(processed.prizes.prize_level_raw)",
            "string",
        ),
        "stake_amount": (
            "单注奖金。",
            "processed.prizes.stake_amount",
            "copy(processed.prizes.stake_amount)",
            "float64",
        ),
        "stake_count": (
            "中奖注数。",
            "processed.prizes.stake_count",
            "copy(processed.prizes.stake_count)",
            "nullable[int]",
        ),
        "total_prize_amount": (
            "总奖金金额。",
            "processed.prizes.total_prize_amount",
            "copy(processed.prizes.total_prize_amount)",
            "float64",
        ),
    }
    for name, (definition, source, formula, dtype) in prize_fact_defs.items():
        definitions[name] = _base_metadata(
            name,
            "context",
            "prize",
            "prize",
            definition,
            [source],
            formula,
            dtype,
            None,
            ["descriptive", "historical_state"],
            availability="post_draw",
            leakage_rule="该字段来自开奖后的 Prize record；用于预测时不得视为 pre_draw 信息。",
        )

    for name in ("prize_event_type",):
        definitions[name] = _base_metadata(
            name,
            "context",
            "prize",
            "prize",
            "奖金事件类型，用于区分普通奖金与特殊派奖事件。",
            ["processed.prizes.prize_event_type"],
            "copy(processed.prizes.prize_event_type)",
            "string",
            None,
            ["descriptive", "historical_state"],
            availability="post_draw",
            leakage_rule="该字段描述开奖后的奖金事件；用于预测时不得视为 pre_draw 信息。",
        )

    return definitions


def build_metadata_document(
    output_paths: list[Path],
    processed_paths: list[Path],
    feature_columns: dict[str, list[str]],
) -> dict[str, Any]:
    """构建完整 feature_metadata.json 文档。"""
    definitions = build_feature_definitions()
    actual_names = {name for columns in feature_columns.values() for name in columns}

    missing_metadata = sorted(actual_names - set(definitions))
    if missing_metadata:
        raise ValueError(f"存在未定义 Metadata 的 Feature: {missing_metadata}")

    return {
        "metadata_version": FEATURE_METADATA_VERSION,
        "feature_definition_version": FEATURE_DEFINITION_VERSION,
        "feature_code_version": FEATURE_CODE_VERSION,
        "configuration": {
            "rolling_windows": list(FEATURE_ROLLING_WINDOWS),
            "frequency_windows": list(FEATURE_FREQUENCY_WINDOWS),
            "repeat_windows": list(FEATURE_REPEAT_WINDOWS),
            "predictive_shift": FEATURE_DEFAULT_PREDICTIVE_SHIFT,
        },
        "inputs": {str(path): file_sha256(path) for path in processed_paths},
        "outputs": {str(path): str(path) for path in output_paths},
        "features": {name: definitions[name] for name in sorted(actual_names)},
    }


def write_metadata(document: dict[str, Any], path: Path) -> None:
    """以稳定 JSON 格式写出 Metadata。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
