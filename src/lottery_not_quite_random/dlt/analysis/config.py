"""Analysis 配置。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class AnalysisConfig:
    """定义一次 Analysis 运行的输入、输出和统计参数。

    Args:
        processed_dir: processed Parquet 输入目录。
        features_dir: Feature Parquet 与 Metadata 输入目录。
        output_dir: Analysis 输出根目录；运行结果写入其 runs/<run_id>/ 子目录。
        run_id: 可选的显式运行标识；为空时由 Runner 生成 UTC 时间标识。
        alpha: 多重检验校正后的显著性阈值。
        include_features: 是否要求并读取全部第一阶段 Feature 输出。
        make_plots: 是否尝试生成 PNG 图形；需要安装 matplotlib。
    """

    processed_dir: Path = field(default_factory=lambda: Path("data/dlt/processed"))
    features_dir: Path = field(default_factory=lambda: Path("data/dlt/features"))
    output_dir: Path = field(default_factory=lambda: Path("data/dlt/analysis"))
    run_id: str | None = None
    alpha: float = 0.05
    include_features: bool = True
    make_plots: bool = True

    def __post_init__(self) -> None:
        """校验配置值，尽早暴露无效参数。"""
        if not 0 < self.alpha < 1:
            raise ValueError("alpha 必须位于 (0, 1) 区间内。")
        for name in ("processed_dir", "features_dir", "output_dir"):
            value = getattr(self, name)
            if not isinstance(value, Path):
                object.__setattr__(self, name, Path(value))
        if self.run_id is not None:
            if not self.run_id or self.run_id in {".", ".."}:
                raise ValueError("run_id 不能为空、不能是 '.' 或 '..'。")
            allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
            if any(char not in allowed for char in self.run_id):
                raise ValueError("run_id 只能包含字母、数字、点、下划线和连字符。")
