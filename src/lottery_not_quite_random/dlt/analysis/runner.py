"""Analysis 运行编排、运行记录和结果输出。"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import scipy

from . import __version__
from .config import AnalysisConfig
from .descriptive.data_overview import build_data_overview, build_prize_summary
from .descriptive.draw_distribution import build_draw_distribution_summary, build_draw_value_distribution
from .descriptive.number_frequency import build_number_frequency_table
from .reporting.plots import build_plots
from .reporting.summary import build_markdown_report, build_summary_document, write_json
from .statistical_tests.frequency_tests import run_number_frequency_tests
from .validation import (
    validate_draw_features,
    validate_draws,
    validate_feature_columns_against_metadata,
    validate_feature_metadata,
    validate_number_features,
    validate_prize_features,
    validate_prizes,
)

LOGGER = logging.getLogger(__name__)


class AnalysisRunner:
    """协调大乐透第一阶段 Analysis 的读取、验证、计算和结果保存。"""

    def __init__(self, config: AnalysisConfig | None = None) -> None:
        """初始化运行器。

        Args:
            config: Analysis 配置；未提供时使用项目默认数据目录。
        """
        self.config = config or AnalysisConfig()

    def run(self) -> Path:
        """执行基础概况、描述统计、单号码频率检验和报告输出。

        Returns:
            本次运行的结果目录 ``data/dlt/analysis/runs/<run_id>/``。

        Raises:
            FileNotFoundError: 当所需输入文件不存在时。
            ValueError: 当输入违反 Analysis 数据契约时。
            Exception: 输入读取或分析阶段的底层异常会记录到元数据后继续抛出。
        """
        run_id = self.config.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = self.config.output_dir / "runs" / run_id
        if run_dir.exists():
            raise FileExistsError(
                f"Analysis 运行目录已存在，不覆盖既有结果: {run_dir}。请指定新的 run_id。"
            )
        run_dir.mkdir(parents=True, exist_ok=False)
        tables_dir = run_dir / "tables"
        figures_dir = run_dir / "figures"
        tables_dir.mkdir(parents=True, exist_ok=True)

        started_at = datetime.now(timezone.utc)
        monotonic_start = time.perf_counter()
        metadata: dict[str, Any] = {
            "analysis_name": "dlt_baseline_descriptive_and_frequency_tests",
            "analysis_version": __version__,
            "run_id": run_id,
            "status": "running",
            "started_at_utc": started_at.isoformat(),
            "finished_at_utc": None,
            "duration_seconds": None,
            "parameters": {
                "alpha": self.config.alpha,
                "number_frequency_test": "two-sided exact binomial test",
                "multiple_testing_method": "Holm step-down family-wise error rate correction",
                "include_features": self.config.include_features,
                "make_plots": self.config.make_plots,
            },
            "software": {
                "python": platform.python_version(),
                "pandas": pd.__version__,
                "scipy": scipy.__version__,
            },
            "inputs": {},
            "outputs": [],
            "error": None,
        }
        write_json(run_dir / "run_metadata.json", metadata)

        try:
            input_paths = self._expected_input_paths()
            for label, path in input_paths.items():
                if not path.is_file():
                    raise FileNotFoundError(f"Analysis 缺少必需输入文件（{label}）: {path}")

            LOGGER.info("读取 processed 开奖数据: %s", input_paths["draws"])
            draws_df = validate_draws(pd.read_parquet(input_paths["draws"]))
            LOGGER.info("读取 processed 奖级数据: %s", input_paths["prizes"])
            prizes_df = validate_prizes(pd.read_parquet(input_paths["prizes"]), draws_df)

            table_inputs: dict[str, tuple[pd.DataFrame, Path | None]] = {
                "processed_draws": (draws_df, input_paths["draws"]),
                "processed_prizes": (prizes_df, input_paths["prizes"]),
            }
            feature_metadata_doc: dict[str, Any] | None = None
            if self.config.include_features:
                feature_tables: dict[str, pd.DataFrame] = {}
                for name in ("draw_features", "number_features", "prize_features"):
                    feature_tables[name] = pd.read_parquet(input_paths[name])

                with input_paths["feature_metadata"].open("r", encoding="utf-8") as file:
                    feature_metadata_doc = validate_feature_metadata(json.load(file))

                feature_tables["draw_features"] = validate_draw_features(feature_tables["draw_features"], draws_df)
                feature_tables["number_features"] = validate_number_features(feature_tables["number_features"], draws_df)
                feature_tables["prize_features"] = validate_prize_features(
                    feature_tables["prize_features"], prizes_df, draws_df
                )
                validate_feature_columns_against_metadata(
                    feature_tables["draw_features"],
                    feature_tables["number_features"],
                    feature_tables["prize_features"],
                    feature_metadata_doc,
                )
                for name, frame in feature_tables.items():
                    table_inputs[name] = (frame, input_paths[name])

            feature_definition_count = None
            if feature_metadata_doc is not None:
                feature_definition_count = len(feature_metadata_doc.get("features", {}))

            overview_df, column_nulls_df = build_data_overview(table_inputs, feature_metadata_doc)
            draw_summary_df = build_draw_distribution_summary(draws_df)
            draw_values_df = build_draw_value_distribution(draws_df)
            frequency_df = build_number_frequency_table(draws_df)
            frequency_tests_df = run_number_frequency_tests(frequency_df, alpha=self.config.alpha)
            prize_summary_df = build_prize_summary(prizes_df, draws_df)

            table_frames = {
                "data_overview.csv": overview_df,
                "column_nulls.csv": column_nulls_df,
                "draw_distribution_summary.csv": draw_summary_df,
                "draw_value_distribution.csv": draw_values_df,
                "number_frequency.csv": frequency_df,
                "number_frequency_tests.csv": frequency_tests_df,
                "prize_summary.csv": prize_summary_df,
            }
            for filename, frame in table_frames.items():
                output_path = tables_dir / filename
                frame.to_csv(output_path, index=False, encoding="utf-8-sig", float_format="%.12g")
                LOGGER.info("输出统计表: %s (%d 行)", output_path, len(frame))

            figure_paths: list[Path] = []
            if self.config.make_plots:
                try:
                    figure_paths = build_plots(frequency_df, draws_df, figures_dir)
                except ModuleNotFoundError as error:
                    if error.name != "matplotlib":
                        raise
                    LOGGER.warning("未安装 matplotlib，跳过 PNG 图形；统计表和报告仍会输出。")

            relative_outputs = [
                str(path.relative_to(run_dir))
                for path in [*(tables_dir / name for name in table_frames), *figure_paths]
            ]
            relative_outputs.extend(["summary.json", "report.md", "run_metadata.json"])
            summary_doc = build_summary_document(
                run_id=run_id,
                draws_df=draws_df,
                frequency_df=frequency_df,
                frequency_tests_df=frequency_tests_df,
                output_files=relative_outputs,
                feature_definition_count=feature_definition_count,
            )
            write_json(run_dir / "summary.json", summary_doc)
            report = build_markdown_report(
                summary_doc,
                overview_df,
                draw_summary_df,
                frequency_tests_df,
                prize_summary_df,
                alpha=self.config.alpha,
                feature_definition_count=feature_definition_count,
            )
            (run_dir / "report.md").write_text(report, encoding="utf-8")

            for label, path in input_paths.items():
                if path.is_file():
                    metadata["inputs"][label] = {
                        "path": str(path),
                        "size_bytes": path.stat().st_size,
                        "sha256": _sha256_file(path),
                    }
            metadata["status"] = "completed"
            metadata["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            metadata["duration_seconds"] = round(time.perf_counter() - monotonic_start, 6)
            metadata["outputs"] = sorted(relative_outputs)
            metadata["data_summary"] = {
                "draw_count": int(len(draws_df)),
                "prize_record_count": int(len(prizes_df)),
                "number_frequency_test_count": int(frequency_tests_df["p_value"].notna().sum()),
                "holm_rejections_at_alpha": int(frequency_tests_df["reject_h0_after_correction"].sum()),
            }
            write_json(run_dir / "run_metadata.json", metadata)
            LOGGER.info("Analysis 完成: %s", run_dir)
            return run_dir
        except Exception as error:
            metadata["status"] = "failed"
            metadata["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            metadata["duration_seconds"] = round(time.perf_counter() - monotonic_start, 6)
            metadata["error"] = {"type": type(error).__name__, "message": str(error)}
            metadata["outputs"] = sorted(
                str(path.relative_to(run_dir)) for path in run_dir.rglob("*") if path.is_file()
            )
            write_json(run_dir / "run_metadata.json", metadata)
            LOGGER.exception("Analysis 失败: %s", error)
            raise

    def _expected_input_paths(self) -> dict[str, Path]:
        """构建并返回本次运行要求的所有输入文件路径。"""
        paths = {
            "draws": self.config.processed_dir / "draws.parquet",
            "prizes": self.config.processed_dir / "prizes.parquet",
        }
        if self.config.include_features:
            paths.update(
                {
                    "draw_features": self.config.features_dir / "draw_features.parquet",
                    "number_features": self.config.features_dir / "number_features.parquet",
                    "prize_features": self.config.features_dir / "prize_features.parquet",
                    "feature_metadata": self.config.features_dir / "feature_metadata.json",
                }
            )
        return paths


def _sha256_file(path: Path) -> str:
    """按块读取文件并计算 SHA-256，避免一次性加载大文件。"""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
