"""单号码频率相对均匀抽样基准的精确二项检验。"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from ..descriptive.number_frequency import ZONE_SPEC


def holm_adjust_pvalues(p_values: list[float] | np.ndarray) -> np.ndarray:
    """使用 Holm step-down 方法校正一组 p-value。

    Holm 校正控制 family-wise error rate，且不要求各项检验相互独立。
    输入中的非有限值保持为 NaN，不参与检验家族大小计算。

    Args:
        p_values: 原始 p-value 序列。

    Returns:
        与输入等长的校正后 p-value 数组。
    """
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    valid_positions = np.flatnonzero(np.isfinite(values))
    if len(valid_positions) == 0:
        return adjusted
    valid_p = values[valid_positions]
    if ((valid_p < 0) | (valid_p > 1)).any():
        raise ValueError("p-value 必须位于 [0, 1]。")

    order = np.argsort(valid_p, kind="mergesort")
    sorted_p = valid_p[order]
    family_size = len(sorted_p)
    raw_adjusted = np.array(
        [(family_size - rank) * p_value for rank, p_value in enumerate(sorted_p)],
        dtype=float,
    )
    monotonic_adjusted = np.minimum(1.0, np.maximum.accumulate(raw_adjusted))
    sorted_result = np.empty_like(monotonic_adjusted)
    sorted_result[order] = monotonic_adjusted
    adjusted[valid_positions] = sorted_result
    return adjusted


def run_number_frequency_tests(
    frequency_df: pd.DataFrame,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """检验每个号码的跨期开奖出现次数是否偏离理论入选概率。

    H0：指定号码每期入选概率等于对应区域的均匀抽样概率（red=5/35，blue=2/12）。
    H1：指定号码每期入选概率不同于该理论概率。

    每个号码使用双侧精确二项检验；随后对前后区合计 47 项检验统一应用
    Holm 校正。二项模型的跨期独立同分布假设必须在解释结果时明确披露。
    本模块只对历史频率进行检验，不评估未来预测价值。

    Args:
        frequency_df: ``build_number_frequency_table`` 的输出。
        alpha: Holm 校正后的显著性阈值。

    Returns:
        每个号码一行的推断性结果表。
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha 必须位于 (0, 1) 区间内。")
    required = {
        "number_zone", "number", "draws_n", "observed_count", "observed_rate",
        "theoretical_probability", "expected_count", "rate_difference",
    }
    missing = required - set(frequency_df.columns)
    if missing:
        raise ValueError(f"frequency_df 缺少字段: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    for record in frequency_df.sort_values(["number_zone", "number"], kind="mergesort").to_dict("records"):
        zone = str(record["number_zone"])
        if zone not in ZONE_SPEC:
            raise ValueError(f"未知号码分区: {zone}")
        n_draws = int(record["draws_n"])
        observed_count = int(record["observed_count"])
        p0 = float(record["theoretical_probability"])
        if n_draws < 0 or observed_count < 0 or observed_count > n_draws:
            raise ValueError(
                f"非法计数：zone={zone}, number={record['number']}, "
                f"observed_count={observed_count}, draws_n={n_draws}。"
            )
        if not np.isclose(p0, ZONE_SPEC[zone]["draw_count"] / ZONE_SPEC[zone]["number_max"]):
            raise ValueError(f"{zone} 分区号码的理论概率与大乐透设计不一致。")

        if n_draws == 0:
            p_value = np.nan
            ci_low = np.nan
            ci_high = np.nan
            status = "insufficient_data"
        else:
            test_result = binomtest(observed_count, n_draws, p=p0, alternative="two-sided")
            p_value = float(test_result.pvalue)
            ci = test_result.proportion_ci(confidence_level=0.95, method="exact")
            ci_low = float(ci.low)
            ci_high = float(ci.high)
            status = "evaluated"

        rows.append(
            {
                "number_zone": zone,
                "number": int(record["number"]),
                "research_type": "exploratory",
                "analysis_unit": "one_number_across_draws",
                "sample_size_draws": n_draws,
                "observed_count": observed_count,
                "observed_rate": float(record["observed_rate"]),
                "theoretical_probability": p0,
                "expected_count": float(record["expected_count"]),
                "rate_difference": float(record["rate_difference"]),
                "rate_difference_percentage_points": float(record["rate_difference"] * 100),
                "observed_rate_ci95_low_exact": ci_low,
                "observed_rate_ci95_high_exact": ci_high,
                "h0": f"p = {p0:.12g}",
                "h1": f"p != {p0:.12g}",
                "test_method": "two_sided_exact_binomial_test",
                "test_statistic": float(observed_count),
                "p_value": p_value,
                "p_value_adjustment": "holm",
                "alpha": alpha,
                "status": status,
                "cross_draw_assumption": "跨期开奖独立且同分布；该假设不是本检验自动证明的事实。",
            }
        )

    result = pd.DataFrame(rows)
    result["p_value_adjusted"] = holm_adjust_pvalues(result["p_value"].to_numpy(dtype=float))
    result["reject_h0_after_correction"] = (
        result["p_value_adjusted"].notna() & result["p_value_adjusted"].lt(alpha)
    )
    result["interpretation_guardrail"] = (
        "统计偏离本身不表示存在稳定规律、未来预测能力或投注价值。"
    )
    return result
