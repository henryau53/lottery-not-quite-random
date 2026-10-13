"""Analysis 命令行入口。"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config import AnalysisConfig
from .runner import AnalysisRunner


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="运行大乐透 Analysis 第一阶段：数据概况、描述统计和号码频率检验。"
    )
    parser.add_argument("--processed-dir", type=Path, default=Path("data/dlt/processed"))
    parser.add_argument("--features-dir", type=Path, default=Path("data/dlt/features"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/dlt/analysis"))
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--without-features", action="store_true", help="只读取 processed；不检查 Feature 表。")
    parser.add_argument("--no-plots", action="store_true", help="跳过可选 PNG 图形。")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser


def main() -> None:
    """解析命令行并执行 Analysis。"""
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = AnalysisConfig(
        processed_dir=args.processed_dir,
        features_dir=args.features_dir,
        output_dir=args.output_dir,
        run_id=args.run_id,
        alpha=args.alpha,
        include_features=not args.without_features,
        make_plots=not args.no_plots,
    )
    output_path = AnalysisRunner(config).run()
    print(f"Analysis 输出目录：{output_path}")


if __name__ == "__main__":
    main()
