"""Feature Metadata 生成。

该模块为第一阶段正式 Feature 集合生成元数据定义，并组装为完整的
feature_metadata.json 文档。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ...config import (
    FEATURE_CODE_VERSION,
    FEATURE_DEFINITION_VERSION,
    FEATURE_FREQUENCY_WINDOWS,
    FEATURE_METADATA_VERSION,
    FEATURE_REPEAT_WINDOWS,
    FEATURE_ROLLING_WINDOWS,
)
from ...utils import file_sha256


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
    observation_point: str = "before_current_draw",
    includes_current_draw: bool = False,
    shift: int = 1,
    leakage_rule: str = "仅能使用当前 draw_index 之前已经发生的数据。",
) -> dict[str, Any]:
    """创建单个 Feature Metadata。

    提供各 Feature 元数据的公共字段与默认值，具体 Feature 只需给出差异
    部分。默认按 pre_draw 可用、观察点在当前期开奖之前、shift=1 处理，
    并按通用泄漏约束填写。

    Args:
        name:
            Feature 的唯一名称。
        category:
            Feature 所属的功能类别。
        level:
            Feature 的分析层级/粒度。
        entity:
            Feature 所描述或分析的对象。
        definition:
            Feature 的自然语言定义/含义。
        source:
            Feature 所依赖的数据来源。
        formula:
            Feature 的计算公式或计算逻辑。
        dtype:
            Feature 的数据类型。
        window:
            Feature 计算所使用的历史数据窗口。
        research_role:
            Feature 在研究流程中的用途。
        availability:
            Feature 在开奖流程中的可用时间点，定义 Feature 在什么时候可获得，默认 "pre_draw"。包括 "pre_draw" 与 "post_draw"。
        observation_point:
            Feature 对应的时间观察点，默认 "before_current_draw"。
        includes_current_draw:
            是否包含当前开奖期的数据，默认 False。
        shift:
            计算 Feature 时使用的时间偏移量，默认 1。
        leakage_rule:
            Feature 避免未来信息泄漏的时间边界规则。

    Returns:
        单个 Feature 的元数据字典，含名称、类别、粒度、来源、公式、类型、
        窗口、观察点、可用时间、泄漏约束、研究用途与定义版本。
    """
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
        "observation_point": observation_point,
        "includes_current_draw": includes_current_draw,
        "shift": shift,
        "availability": availability,
        "leakage_rule": leakage_rule,
        "research_role": research_role,
        "version": FEATURE_DEFINITION_VERSION,
    }


def build_feature_definitions() -> dict[str, dict[str, Any]]:
    """生成第一阶段正式 Feature 集合的元数据定义。

    Returns:
        以 Feature 名称为键、以元数据字典为值的映射，覆盖第一阶段正式 Feature 集合。
    """
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
            f"当前期之前最近 {window} 个有效相邻开奖全区号码重复数量的平均值。",
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
            "当前开奖对应的派奖活动名称或标识。",
            "configured bonus campaign range -> campaign.name",
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
        "prize_name": (
            "标准奖级名称。",
            "processed.prizes.prize_name",
            "copy(processed.prizes.prize_name)",
            "string",
        ),
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
        "prize_event_type": (
            "奖金事件类型，用于区分基本奖金、追加奖金、基本派奖、追加派奖。",
            "processed.prizes.prize_event_type",
            "copy(processed.prizes.prize_event_type)",
            "string",
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
            leakage_rule=(
                "该字段来自开奖后的奖金数据记录（Prize record）；"
                "用于预测时不得视为 pre_draw 信息。"
            ),
        )

    return definitions


def build_metadata_document(
    output_paths: list[Path],
    processed_paths: list[Path],
    feature_columns: dict[str, list[str]],
) -> dict[str, Any]:
    """构建完整的 feature_metadata.json 文档。

    Args:
        output_paths:
            本次生成的 Feature 输出文件路径列表。
        processed_paths:
            作为输入的 processed 文件路径列表，用于计算哈希。
        feature_columns:
            实际生成的特征值列名集合。

    Returns:
        完整的 Metadata 定义。

    Raises:
        ValueError:
            当存在实际 Feature 字段缺少 Metadata 定义时。
    """
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
            "predictive_shift": 1,
        },
        "inputs": {str(path): file_sha256(path) for path in processed_paths},
        "outputs": {str(path): str(path) for path in output_paths},
        "features": {name: definitions[name] for name in sorted(actual_names)},
    }


def write_metadata(document: dict[str, Any], path: Path) -> None:
    """以稳定 JSON 格式写出 Metadata 文档。

    使用 UTF-8、缩进 2 与键排序写出，便于版本管理与差异比较；目录不存在时
    自动创建。

    Args:
        document:
            由 build_metadata_document 生成的 Metadata 文档字典。
        path:
            目标 JSON 文件路径。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
