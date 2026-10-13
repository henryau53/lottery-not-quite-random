"""Analysis 第一阶段单元与编排测试。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lottery_not_quite_random.dlt.analysis.config import AnalysisConfig
from lottery_not_quite_random.dlt.analysis.descriptive.number_frequency import build_number_frequency_table
from lottery_not_quite_random.dlt.analysis.runner import AnalysisRunner
from lottery_not_quite_random.dlt.analysis.statistical_tests.frequency_tests import (
    holm_adjust_pvalues,
    run_number_frequency_tests,
)
from lottery_not_quite_random.dlt.analysis.validation import validate_draws, validate_prizes


def make_draws(count: int = 8) -> pd.DataFrame:
    """构造符合项目 processed/draws.parquet 基础契约的开奖样本。"""
    rows = []
    for index in range(count):
        red = [((index * 5 + offset) % 35) + 1 for offset in range(5)]
        blue = [((index * 2 + offset) % 12) + 1 for offset in range(2)]
        rows.append(
            {
                "draw_index": index,
                "issue": str(26014 + index),
                "draw_date": (pd.Timestamp("2026-02-02") + pd.Timedelta(days=2 * index)).date().isoformat(),
                **{f"red_{position + 1}": number for position, number in enumerate(red)},
                **{f"blue_{position + 1}": number for position, number in enumerate(blue)},
                "red_sum": sum(red),
                "red_span": max(red) - min(red),
                "blue_sum": sum(blue),
                "blue_span": max(blue) - min(blue),
                "red_odd_count": sum(number % 2 for number in red),
                "red_even_count": sum(number % 2 == 0 for number in red),
                "blue_odd_count": sum(number % 2 for number in blue),
                "blue_even_count": sum(number % 2 == 0 for number in blue),
                "red_zone_1_count": sum(1 <= number <= 12 for number in red),
                "red_zone_2_count": sum(13 <= number <= 24 for number in red),
                "red_zone_3_count": sum(25 <= number <= 35 for number in red),
                "red_consecutive_group_count": 0,
                "red_max_consecutive_length": 1,
            }
        )
    return pd.DataFrame(rows)


def make_prizes(draws: pd.DataFrame) -> pd.DataFrame:
    """构造奖级记录。"""
    return pd.DataFrame(
        {
            "issue": draws["issue"].astype(str),
            "draw_date": draws["draw_date"],
            "rule_version": "v4",
            "prize_rank": 7,
            "prize_event_type": "basic_prize",
            "stake_amount": 5.0,
            "stake_count": 100,
            "total_prize_amount": 500.0,
        }
    )


def make_feature_tables(draws: pd.DataFrame, prizes: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """构造完整粒度的三张 Feature 表。"""
    draw_features = draws.loc[:, ["draw_index", "issue", "draw_date"]].copy()
    draw_features["red_sum_mean_10"] = np.nan

    number_rows = []
    for row in draws.itertuples(index=False):
        for zone, maximum in (("red", 35), ("blue", 12)):
            for number in range(1, maximum + 1):
                number_rows.append(
                    {
                        "draw_index": row.draw_index,
                        "issue": str(row.issue),
                        "draw_date": row.draw_date,
                        "number_zone": zone,
                        "number": number,
                        "frequency_10": 0,
                    }
                )
    number_features = pd.DataFrame(number_rows)

    prize_features = prizes.copy()
    issue_to_index = dict(zip(draws["issue"].astype(str), draws["draw_index"]))
    prize_features["draw_index"] = prize_features["issue"].astype(str).map(issue_to_index)
    prize_features["prize_amount"] = prize_features["total_prize_amount"]
    return {
        "draw_features": draw_features,
        "number_features": number_features,
        "prize_features": prize_features.loc[:, [
            "draw_index", "issue", "draw_date", "rule_version", "prize_rank",
            "prize_event_type", "prize_amount",
        ]],
    }


def test_validate_draws_sorts_by_draw_index_without_mutating_source() -> None:
    """内部计算应按 draw_index 排序，但不修改调用方原始表。"""
    draws = make_draws(5)
    reversed_draws = draws.iloc[::-1].reset_index(drop=True)
    validated = validate_draws(reversed_draws)
    assert validated["draw_index"].tolist() == [0, 1, 2, 3, 4]
    assert reversed_draws["draw_index"].tolist() == [4, 3, 2, 1, 0]


def test_validate_draws_rejects_duplicate_red_numbers() -> None:
    """同一期前区不得有重复号码。"""
    draws = make_draws(3)
    draws.loc[0, "red_2"] = draws.loc[0, "red_1"]
    with pytest.raises(ValueError, match="前区必须包含 5 个互不重复号码"):
        validate_draws(draws)


def test_validate_draws_rejects_non_contiguous_draw_index() -> None:
    """draw_index 必须从 0 开始且连续。"""
    draws = make_draws(3)
    draws.loc[1, "draw_index"] = 7
    with pytest.raises(ValueError, match="draw_index 必须从 0 开始连续递增"):
        validate_draws(draws)


def test_frequency_table_has_47_entities_and_uses_correct_baselines() -> None:
    """频率统计覆盖前区 35 个和后区 12 个号码。"""
    draws = validate_draws(make_draws(8))
    frequency = build_number_frequency_table(draws)
    assert len(frequency) == 47
    assert frequency.query("number_zone == 'red'")["theoretical_probability"].eq(5 / 35).all()
    assert frequency.query("number_zone == 'blue'")["theoretical_probability"].eq(2 / 12).all()
    assert frequency["observed_count"].between(0, 8).all()
    assert frequency.query("number_zone == 'red'")["observed_count"].sum() == 8 * 5
    assert frequency.query("number_zone == 'blue'")["observed_count"].sum() == 8 * 2


def test_holm_adjustment_known_values() -> None:
    """Holm step-down 结果与手工可核对的样例一致。"""
    adjusted = holm_adjust_pvalues([0.01, 0.04, 0.03])
    assert adjusted == pytest.approx([0.03, 0.06, 0.06])


def test_holm_adjustment_preserves_nan_and_validates_range() -> None:
    """缺失 p-value 不参与 family size，非法 p-value 明确报错。"""
    adjusted = holm_adjust_pvalues([0.01, np.nan, 0.04])
    assert adjusted[0] == pytest.approx(0.02)
    assert np.isnan(adjusted[1])
    assert adjusted[2] == pytest.approx(0.04)
    with pytest.raises(ValueError, match="p-value 必须位于"):
        holm_adjust_pvalues([1.2])


def test_frequency_tests_include_hypotheses_ci_and_adjustment() -> None:
    """推断结果应报告假设、检验、区间和多重校正字段。"""
    draws = validate_draws(make_draws(30))
    frequencies = build_number_frequency_table(draws)
    tests = run_number_frequency_tests(frequencies, alpha=0.05)
    assert len(tests) == 47
    required = {
        "h0", "h1", "test_statistic", "p_value", "p_value_adjusted",
        "observed_rate_ci95_low_exact", "observed_rate_ci95_high_exact",
        "p_value_adjustment", "reject_h0_after_correction", "cross_draw_assumption",
    }
    assert required.issubset(tests.columns)
    assert tests["p_value"].between(0, 1).all()
    assert tests["p_value_adjusted"].between(0, 1).all()
    assert tests["p_value_adjusted"].ge(tests["p_value"]).all()
    assert tests["research_type"].eq("exploratory").all()


def test_validate_prizes_rejects_rule_version_conflict() -> None:
    """奖金数据必须遵守项目定义的历史规则版本边界。"""
    draws = validate_draws(make_draws(3))
    prizes = make_prizes(draws)
    prizes.loc[0, "rule_version"] = "v3"
    with pytest.raises(ValueError, match="rule_version 与项目规则边界不一致"):
        validate_prizes(prizes, draws)


def test_validate_prizes_rejects_bonus_outside_documented_period() -> None:
    """派奖事件必须落在 PROJECT_CONTEXT 已定义的期号区间内。"""
    draws = validate_draws(make_draws(3))
    prizes = make_prizes(draws)
    prizes.loc[0, "prize_event_type"] = "basic_bonus"
    with pytest.raises(ValueError, match="派奖事件出现在.*范围之外"):
        validate_prizes(prizes, draws)


def test_validate_prizes_rejects_rank_outside_rule_version() -> None:
    """奖级编号不得超出对应历史规则的奖级范围。"""
    draws = validate_draws(make_draws(3))
    prizes = make_prizes(draws)
    prizes.loc[0, "prize_rank"] = 8
    with pytest.raises(ValueError, match="超出对应 rule_version 的奖级范围"):
        validate_prizes(prizes, draws)


def test_validate_prizes_rejects_additional_rank_outside_rule_version() -> None:
    """追加投注奖级应遵守各版本定义的追加范围。"""
    draws = validate_draws(make_draws(3))
    prizes = make_prizes(draws)
    prizes.loc[0, "prize_rank"] = 3
    prizes.loc[0, "prize_event_type"] = "additional_prize"
    with pytest.raises(ValueError, match="追加范围"):
        validate_prizes(prizes, draws)


def test_issue_number_accepts_integer_like_parquet_values() -> None:
    """整数期号被 Parquet 提升为浮点字符串时仍能按规则边界解释。"""
    from lottery_not_quite_random.dlt.analysis.validation import expected_rule_version
    assert expected_rule_version("26014.0") == "v4"


def test_runner_writes_a_reproducible_run_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """编排入口应写入摘要、Markdown 报告、表格和成功运行元数据。"""
    processed_dir = tmp_path / "processed"
    features_dir = tmp_path / "features"
    output_dir = tmp_path / "analysis"
    processed_dir.mkdir()
    features_dir.mkdir()
    for path in (
        processed_dir / "draws.parquet",
        processed_dir / "prizes.parquet",
        features_dir / "draw_features.parquet",
        features_dir / "number_features.parquet",
        features_dir / "prize_features.parquet",
    ):
        path.write_bytes(b"test placeholder; read_parquet is stubbed in this test")

    draws = make_draws(6)
    prizes = make_prizes(draws)
    feature_tables = make_feature_tables(draws, prizes)
    input_frames = {
        "draws.parquet": draws,
        "prizes.parquet": prizes,
        **{f"{name}.parquet": frame for name, frame in feature_tables.items()},
    }
    monkeypatch.setattr(pd, "read_parquet", lambda path, *args, **kwargs: input_frames[Path(path).name].copy())
    (features_dir / "feature_metadata.json").write_text(
        json.dumps({"features": {"red_sum_mean_10": {}, "frequency_10": {}, "prize_amount": {}}}),
        encoding="utf-8",
    )

    config = AnalysisConfig(
        processed_dir=processed_dir,
        features_dir=features_dir,
        output_dir=output_dir,
        run_id="unit-test-run",
        include_features=True,
        make_plots=False,
    )
    run_dir = AnalysisRunner(config).run()
    assert run_dir == output_dir / "runs" / "unit-test-run"
    assert (run_dir / "summary.json").is_file()
    assert (run_dir / "report.md").is_file()
    assert (run_dir / "tables" / "number_frequency.csv").is_file()
    assert (run_dir / "tables" / "number_frequency_tests.csv").is_file()
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "completed"
    assert metadata["inputs"]["draws"]["sha256"]
    assert summary["draw_count"] == len(draws)
    assert summary["number_frequency_tests_evaluated"] == 47
    assert "不自动代表稳定规律" in (run_dir / "report.md").read_text(encoding="utf-8")


def test_runner_does_not_overwrite_an_existing_run(tmp_path: Path) -> None:
    """相同 run_id 不允许覆盖之前的研究结果。"""
    output_dir = tmp_path / "analysis"
    existing = output_dir / "runs" / "same-id"
    existing.mkdir(parents=True)
    runner = AnalysisRunner(AnalysisConfig(output_dir=output_dir, run_id="same-id", include_features=False))
    with pytest.raises(FileExistsError, match="不覆盖既有结果"):
        runner.run()
