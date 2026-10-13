"""机器可读摘要与 Markdown 报告。"""

from __future__ import annotations

import json
import math
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _json_safe(value: Any) -> Any:
    """将 pandas / NumPy 值转换为标准 JSON 可序列化类型。"""
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def write_json(path: Path, content: dict[str, Any]) -> None:
    """以 UTF-8、稳定键顺序写入 JSON 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(_json_safe(content), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(path)


def build_summary_document(
    run_id: str,
    draws_df: pd.DataFrame,
    frequency_df: pd.DataFrame,
    frequency_tests_df: pd.DataFrame,
    output_files: list[str],
    feature_definition_count: int | None = None,
) -> dict[str, Any]:
    """构建核心摘要，不输出预测或推荐结论。"""
    dates = pd.to_datetime(draws_df["draw_date"], errors="coerce")
    rejected = frequency_tests_df.loc[frequency_tests_df["reject_h0_after_correction"]]
    return {
        "analysis_name": "dlt_baseline_descriptive_and_frequency_tests",
        "analysis_version": "0.1.0",
        "run_id": run_id,
        "research_type": "descriptive_and_exploratory",
        "status": "completed",
        "draw_count": int(len(draws_df)),
        "issue_range_by_draw_index": {
            "first_issue": str(draws_df.iloc[0]["issue"]),
            "last_issue": str(draws_df.iloc[-1]["issue"]),
        },
        "draw_date_range": {
            "start": dates.min().date().isoformat() if dates.notna().any() else None,
            "end": dates.max().date().isoformat() if dates.notna().any() else None,
        },
        "number_entities_analyzed": int(len(frequency_df)),
        "number_frequency_tests_evaluated": int(frequency_tests_df["p_value"].notna().sum()),
        "holm_rejections_at_alpha": int(len(rejected)),
        "holm_rejection_entities": [
            {"number_zone": str(row.number_zone), "number": int(row.number), "p_value_adjusted": float(row.p_value_adjusted)}
            for row in rejected.itertuples(index=False)
        ],
        "multiple_testing": {
            "method": "Holm step-down family-wise error rate correction",
            "family": "all 35 red-number tests and 12 blue-number tests together",
        },
        "feature_definition_count": feature_definition_count,
        "interpretation_limits": [
            "单号码检验假定跨期开奖入选状态独立且同分布；该假设未由检验本身证明。",
            "检验结果是历史频率层面的统计证据，不证明稳定规律、预测能力或投注价值。",
            "本运行不训练模型、不构造预测、不进行回测，也不生成选号建议。",
        ],
        "output_files": sorted(output_files),
    }


def build_markdown_report(
    summary: dict[str, Any],
    data_overview_df: pd.DataFrame,
    distribution_df: pd.DataFrame,
    frequency_tests_df: pd.DataFrame,
    prize_summary_df: pd.DataFrame,
    *,
    alpha: float,
    feature_definition_count: int | None,
) -> str:
    """根据结构化结果生成中文 Markdown 报告。"""
    lines = [
        "# 大乐透历史数据 Analysis 报告",
        "",
        f"- 运行标识：`{summary['run_id']}`",
        f"- Analysis 版本：`{summary['analysis_version']}`",
        f"- 研究类型：`{summary['research_type']}`",
        f"- 开奖期数：{summary['draw_count']}",
        f"- 期号范围（按 draw_index）：{summary['issue_range_by_draw_index']['first_issue']} – {summary['issue_range_by_draw_index']['last_issue']}",
        f"- 开奖日期范围：{summary['draw_date_range']['start']} – {summary['draw_date_range']['end']}",
        "",
        "## 1. 数据概况与验证",
        "",
        "输入数据在统计前已执行字段、开奖键、时间顺序、号码范围/唯一性及相应 Feature 粒度验证。详细表级/字段级覆盖和 Null 计数见 `tables/data_overview.csv` 与 `tables/column_nulls.csv`。",
        "",
        "| 数据表 | 行数 | 列数 | Null 单元格 | Null 比例 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in data_overview_df.to_dict("records"):
        lines.append(
            f"| {row.get('table', '')} | {row.get('rows', 0)} | {row.get('columns', 0)} | "
            f"{row.get('null_cells', 0)} | {float(row.get('null_rate', 0.0)):.4%} |"
        )
    lines.extend([
        "",
        "## 2. 开奖结构描述统计",
        "",
        "下表仅描述历史样本，不附带随机性或预测性结论。完整统计和经验频数见 `tables/draw_distribution_summary.csv`、`tables/draw_value_distribution.csv`。",
        "",
        "| 变量 | 样本量 | 均值 | 标准差 | 最小值 | 中位数 | 最大值 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in distribution_df.head(15).to_dict("records"):
        def fmt(value: Any) -> str:
            return "—" if pd.isna(value) else f"{float(value):.4g}"
        lines.append(
            f"| {row.get('variable', '')} | {row.get('sample_size', 0)} | {fmt(row.get('mean'))} | "
            f"{fmt(row.get('std'))} | {fmt(row.get('min'))} | {fmt(row.get('median'))} | {fmt(row.get('max'))} |"
        )
    lines.extend([
        "",
        "## 3. 单号码频率的理论基准比较",
        "",
        "在均匀抽样参考假设下，前区单号每期入选概率为 `5/35`，后区为 `2/12`。`tables/number_frequency.csv` 保存每个号码的实际出现次数、观测比例、理论比例和偏离幅度。",
        "",
        "推断部分对 47 个号码分别进行双侧精确二项检验，并对整个检验家族统一使用 Holm 校正。原假设是单号码跨期开奖的入选概率等于其区域理论概率；该推断依赖跨期开奖独立同分布假设。",
        "",
        f"- 校正阈值：`alpha = {alpha}`",
        f"- 完成检验的号码数：{summary['number_frequency_tests_evaluated']}",
        f"- Holm 校正后拒绝原假设的号码数：{summary['holm_rejections_at_alpha']}",
        "- 完整统计量、原始/校正 p-value、逐号码精确 95% 边际比例置信区间和假设说明见 `tables/number_frequency_tests.csv`；该置信区间是逐项区间，不是同时覆盖全部 47 个号码的区间。",
        "",
    ])
    rejected = frequency_tests_df.loc[frequency_tests_df["reject_h0_after_correction"]]
    if rejected.empty:
        lines.append("在本次设定的检验家族与显著性阈值下，没有号码在 Holm 校正后拒绝原假设。这不等于证明开奖完全随机。")
    else:
        lines.extend([
            "校正后达到阈值的历史频率差异如下。它们仍属于探索性统计发现，不是未来号码推荐：",
            "",
            "| 分区 | 号码 | 出现次数 | 观测比例 | 理论概率 | 原始 p-value | Holm 校正 p-value |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for row in rejected.sort_values(["number_zone", "number"]).to_dict("records"):
            lines.append(
                f"| {row['number_zone']} | {int(row['number']):02d} | {int(row['observed_count'])} | "
                f"{float(row['observed_rate']):.4%} | {float(row['theoretical_probability']):.4%} | "
                f"{float(row['p_value']):.4g} | {float(row['p_value_adjusted']):.4g} |"
            )
    lines.extend([
        "",
        "## 4. 奖金与派奖事件摘要",
        "",
        "奖金记录按 `rule_version`、`prize_event_type` 和 `prize_rank` 分层；`basic_prize`、`additional_prize`、`basic_bonus`、`additional_bonus`、`unknown` 不会被合并成单一业务类别。完整结果见 `tables/prize_summary.csv`。该表是描述性汇总，不用于跨规则版本的直接因果比较。",
        "",
        "## 5. Feature 覆盖概况",
        "",
        f"Feature Metadata 中定义的特征数：{feature_definition_count if feature_definition_count is not None else '未读取（配置跳过 Feature 输入）'}。具体 Feature 表行数、字段数、空值情况见数据概况表。Feature 的含义及时间语义仍以 `FEATURE_DESIGN.md` 为准。",
        "",
        "## 6. 解释限制与下一步",
        "",
        "1. 频率差异与统计显著性不自动代表稳定规律或未来预测优势。",
        "2. 单号码检验的跨期独立同分布前提需要结合后续依赖研究进行审查；同期开奖内的固定数量、不放回结构已通过前区 `5/35`、后区 `2/12` 的边际基准加以区分，但本次检验不检验跨号码联合分布。",
        "3. 探索性结论如需进入确认性研究，应事先固定研究问题和样本边界，并使用独立或时间顺序正确的留出期验证。",
        "4. 本阶段不包含 Target 定义、模型训练、样本外预测、回测或购买建议。",
        "",
        "## 7. 输出清单",
        "",
    ])
    for output_file in summary.get("output_files", []):
        lines.append(f"- `{output_file}`")
    lines.append("")
    return "\n".join(lines)
