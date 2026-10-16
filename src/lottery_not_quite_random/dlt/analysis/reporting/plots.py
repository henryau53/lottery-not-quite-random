"""可选的描述性图形输出。"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def build_plots(
    frequency_df: pd.DataFrame,
    draws_df: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    """生成号码频率与核心开奖结构的 PNG 图形。

    Args:
        frequency_df: 号码频率描述统计表。
        draws_df: 已验证并按时间排序的 processed draws。
        output_dir: 图形输出目录。

    Returns:
        实际生成的图形路径列表。

    Raises:
        ModuleNotFoundError: 当环境未安装 matplotlib 时。
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    for zone, theory in (("red", 5 / 35), ("blue", 2 / 12)):
        subset = frequency_df.loc[frequency_df["number_zone"].eq(zone)].sort_values(
            "number"
        )
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(subset["number"].astype(int), subset["observed_rate"].astype(float))
        ax.axhline(
            theory, linestyle="--", linewidth=1.2, label=f"均匀抽样基准 = {theory:.4f}"
        )
        ax.set_title(f"{zone.upper()} 区各号码历史出现比例（描述性）")
        ax.set_xlabel("号码")
        ax.set_ylabel("观测比例")
        ax.legend()
        fig.tight_layout()
        path = output_dir / f"number_frequency_{zone}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        outputs.append(path)

    for column in ("red_sum", "blue_sum", "red_span", "blue_span"):
        if column not in draws_df.columns:
            continue
        values = pd.to_numeric(draws_df[column], errors="coerce").dropna()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(values, bins="auto")
        ax.set_title(f"{column} 历史经验分布（描述性）")
        ax.set_xlabel(column)
        ax.set_ylabel("期数")
        fig.tight_layout()
        path = output_dir / f"draw_distribution_{column}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        outputs.append(path)

    logger.info("已生成 %d 张描述性图形。", len(outputs))
    return outputs
