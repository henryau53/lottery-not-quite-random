"""Feature 第一阶段核心测试。"""

from __future__ import annotations

import pandas as pd
import pytest

from lottery_not_quite_random.features.context import build_draw_context, infer_rule_version
from lottery_not_quite_random.features.frequency import build_frequency_features
from lottery_not_quite_random.features.missing import build_missing_features
from lottery_not_quite_random.features.repeat import build_repeat_history_features
from lottery_not_quite_random.features.rolling import build_rolling_features
from lottery_not_quite_random.features.validation import validate_draw_input


def make_draws(count: int = 25) -> pd.DataFrame:
    """构造最小确定性 processed draws 测试数据。"""
    rows = []
    for index in range(count):
        red = [1, 2, 3, 4, 5] if index % 2 == 0 else [6, 7, 8, 9, 10]
        blue = [1, 2] if index % 2 == 0 else [3, 4]
        rows.append({
            "draw_index": index,
            "issue": f"25{index + 1:03d}",
            "draw_date": f"2025-01-{(index % 28) + 1:02d}",
            "red_sum": sum(red),
            "blue_sum": sum(blue),
            "red_span": max(red) - min(red),
            "blue_span": max(blue) - min(blue),
            "red_odd_count": sum(n % 2 for n in red),
            "blue_odd_count": sum(n % 2 for n in blue),
            "red_consecutive_group_count": 1,
            "red_max_consecutive_length": 5,
            **{f"red_{i + 1}": value for i, value in enumerate(red)},
            **{f"blue_{i + 1}": value for i, value in enumerate(blue)},
        })
    return pd.DataFrame(rows)


def test_frequency_excludes_current_draw() -> None:
    """frequency 必须只使用当前期之前的窗口。"""
    draws = make_draws()
    features = build_frequency_features(draws, (10,))
    current = features.query("draw_index == 10 and number_zone == 'red' and number == 1")
    assert current["frequency_10"].iloc[0] == 5


def test_missing_is_null_before_first_hit() -> None:
    """历史范围内从未出现的号码必须为 null。"""
    draws = make_draws()
    features = build_missing_features(draws)
    current = features.query("draw_index == 0 and number_zone == 'red' and number == 35")
    assert pd.isna(current["missing_current"].iloc[0])


def test_prev_draw_hit_is_null_for_first_draw() -> None:
    """第一期不存在上一期开奖，因此状态为 null。"""
    draws = make_draws()
    from lottery_not_quite_random.features.repeat import build_prev_draw_hit
    features = build_prev_draw_hit(draws)
    current = features.query("draw_index == 0 and number_zone == 'red' and number == 1")
    assert pd.isna(current["prev_draw_hit"].iloc[0])


def test_rolling_uses_exact_previous_window() -> None:
    """Rolling 第一个完整值必须使用完整历史窗口。"""
    draws = make_draws(11)
    features = build_rolling_features(draws, (10,))
    assert pd.isna(features.loc[9, "red_sum_mean_10"])
    assert features.loc[10, "red_sum_mean_10"] == draws.loc[0:9, "red_sum"].mean()


def test_repeat_history_excludes_current_adjacent_relation() -> None:
    """repeat_count_mean_10 不得包含当前期与上一期刚产生的关系。"""
    draws = make_draws(12)
    features = build_repeat_history_features(draws, (10,))
    assert pd.isna(features.loc[10, "repeat_count_mean_10"])
    assert pd.notna(features.loc[11, "repeat_count_mean_10"])


def test_rule_version_boundaries() -> None:
    """历史规则边界必须与设计文档一致。"""
    assert infer_rule_version(14051) == "v1"
    assert infer_rule_version(14052) == "v2"
    assert infer_rule_version(19018) == "v2"
    assert infer_rule_version(19019) == "v3"
    assert infer_rule_version(26013) == "v3"
    assert infer_rule_version(26014) == "v4"


def test_context_bonus_period() -> None:
    """派奖活动范围必须正确映射。"""
    draws = make_draws(1)
    draws.loc[0, "issue"] = "26050"
    context = build_draw_context(draws)
    assert bool(context.loc[0, "is_bonus_period"]) is True
    assert context.loc[0, "bonus_campaign_id"] == "bonus_2026_26050_26066"


def test_input_draw_index_must_be_contiguous() -> None:
    """非法 draw_index 必须 fail fast。"""
    draws = make_draws(3)
    draws.loc[1, "draw_index"] = 9
    with pytest.raises(ValueError, match="draw_index"):
        validate_draw_input(draws)
