"""项目配置。

从项目根目录的 ``config.toml`` 读取可配置项，
并定义项目内部固定的数据目录、文件路径以及业务常量。
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass

from .utils import find_project_root

# ============================================================================
# 基础配置
# ============================================================================

PROJECT_ROOT = find_project_root()
CONFIG_FILE = PROJECT_ROOT / "config.toml"

if not CONFIG_FILE.exists():
    raise FileNotFoundError(f"配置文件不存在: {CONFIG_FILE}")

with CONFIG_FILE.open("rb") as f:
    _CONFIG = tomllib.load(f)


# ============================================================================
# 数据目录
# ============================================================================

# 数据根目录
DATA_DIR = PROJECT_ROOT / _CONFIG["paths"]["data_dir"]


# ============================================================================
# 大乐透数据目录
# ============================================================================

# 数据根目录
DLT_DATA_DIR = DATA_DIR / "dlt"

# 原始数据目录
DLT_RAW_DIR = DLT_DATA_DIR / "raw"

# 原始开奖数据文件
DLT_RAW_DRAWS_FILE = DLT_RAW_DIR / "draws.json"

# 原始元数据文件
DLT_RAW_META_FILE = DLT_RAW_DIR / "meta.json"

# 处理后数据目录
DLT_PROCESSED_DIR = DLT_DATA_DIR / "processed"

# 处理后开奖数据
DLT_PROCESSED_DRAWS_FILE = DLT_PROCESSED_DIR / "draws.parquet"

# 处理后奖级数据
DLT_PROCESSED_PRIZES_FILE = DLT_PROCESSED_DIR / "prizes.parquet"

# 特征值 Feature 数据目录
DLT_FEATURE_DIR = DLT_DATA_DIR / "features"

# Draw-level Feature 数据
DLT_FEATURE_DRAW_FILE = DLT_FEATURE_DIR / "draw_features.parquet"

# Number-level Feature 数据
DLT_FEATURE_NUMBER_FILE = DLT_FEATURE_DIR / "number_features.parquet"

# Prize-level Feature 数据
DLT_FEATURE_PRIZE_FILE = DLT_FEATURE_DIR / "prize_features.parquet"

# Feature 元数据
DLT_FEATURE_METADATA_FILE = DLT_FEATURE_DIR / "feature_metadata.json"


# ============================================================================
# 特征值 Feature 配置
# ============================================================================

_FEATURE_CONFIG = _CONFIG["feature"]

# Feature 定义规范版本
FEATURE_DEFINITION_VERSION = _FEATURE_CONFIG["definition_version"]

# Feature 实现代码版本
FEATURE_CODE_VERSION = _FEATURE_CONFIG["code_version"]

# Feature 元数据格式版本
FEATURE_METADATA_VERSION = _FEATURE_CONFIG["metadata_version"]

# Draw-level / Number-level 特征使用的滚动窗口
FEATURE_ROLLING_WINDOWS: tuple[int, ...] = tuple(_FEATURE_CONFIG["rolling_windows"])

# Number-level 号码频率特征使用的历史窗口
FEATURE_FREQUENCY_WINDOWS: tuple[int, ...] = tuple(_FEATURE_CONFIG["frequency_windows"])

# Number-level 重复历史特征使用的历史窗口
FEATURE_REPEAT_WINDOWS: tuple[int, ...] = tuple(_FEATURE_CONFIG["repeat_windows"])

# 默认预测性特征的时间偏移量，确保不使用当前开奖信息
FEATURE_DEFAULT_PREDICTIVE_SHIFT = 1


# ============================================================================
# 数据字典
# ============================================================================

# 大乐透前区号码范围
RED_NUMBERS = tuple(range(1, 36))

# 大乐透后区号码范围
BLUE_NUMBERS = tuple(range(1, 13))

# 号码所属区域
NUMBER_ZONES = ("red", "blue")

# 奖级事件类型
PRIZE_EVENT_TYPES = (
    "basic_prize",
    "additional_prize",
    "basic_bonus",
    "additional_bonus",
    "unknown",  # 目前并未使用
)

# 奖级编号映射
PRIZE_RANK_MAPPING = {
    "一等奖": 1,
    "二等奖": 2,
    "三等奖": 3,
    "四等奖": 4,
    "五等奖": 5,
    "六等奖": 6,
    "七等奖": 7,
    "八等奖": 8,
    "九等奖": 9,
}


# ============================================================================
# 奖级规则版本数据结构
# ============================================================================

# TODO 将 process 中定义版本 v1 等地方使用当前读取配置文件动态方式


@dataclass(frozen=True)
class RuleVersion:
    """一个奖级规则版本的有效期。"""

    name: str
    start_issue: int
    end_issue: int

    def matches(self, issue: int) -> bool:
        """判断指定期号是否属于当前规则版本。

        Args:
            issue : 指定期号

        Returns:
            bool: 是否
        """
        return self.start_issue <= issue <= self.end_issue


@dataclass(frozen=True)
class BonusCampaign:
    """一个派奖活动的有效期。"""

    name: str
    start_issue: int
    end_issue: int

    def matches(self, issue: int) -> bool:
        """判断指定期号是否属于当前派奖活动。

        Args:
            issue : 指定期号

        Returns:
            bool: 是否
        """
        return self.start_issue <= issue <= self.end_issue


# ============================================================================
# 规则配置
# ============================================================================

_RULES_CONFIG = _CONFIG.get("rules", {})


# ---------------------------------------------------------------------------
# 奖级规则版本
# ---------------------------------------------------------------------------

RULE_VERSIONS: tuple[RuleVersion, ...] = tuple(
    RuleVersion(
        name=item["name"],
        start_issue=item["start_issue"],
        end_issue=item["end_issue"],
    )
    for item in _RULES_CONFIG.get("rule_versions", [])
)

# 奖级规则版本名称
RULE_VERSION_NAMES: tuple[str, ...] = tuple(rule.name for rule in RULE_VERSIONS)


# ---------------------------------------------------------------------------
# 派奖活动
# ---------------------------------------------------------------------------

BONUS_CAMPAIGNS: tuple[BonusCampaign, ...] = tuple(
    BonusCampaign(
        name=item["name"],
        start_issue=item["start_issue"],
        end_issue=item["end_issue"],
    )
    for item in _RULES_CONFIG.get("bonus_campaigns", [])
)


# ============================================================================
# 配置校验
# ============================================================================


def _validate_rule_versions() -> None:
    """校验奖级规则版本数据。"""

    previous: RuleVersion | None = None

    for current in RULE_VERSIONS:
        if current.start_issue > current.end_issue:
            raise ValueError(
                f"奖级规则版本 {current.name!r} 的 start_issue 大于 end_issue"
            )

        if previous is not None:
            if current.start_issue <= previous.end_issue:
                raise ValueError(
                    f"奖级规则版本区间存在重叠: {previous.name!r} 与 {current.name!r}"
                )

            if current.start_issue != previous.end_issue + 1:
                raise ValueError(
                    f"奖级规则版本区间存在断档: {previous.name!r} -> {current.name!r}"
                )

        previous = current


def _validate_bonus_campaigns() -> None:
    """校验派奖活动数据。"""

    previous: BonusCampaign | None = None

    for current in BONUS_CAMPAIGNS:
        if current.start_issue > current.end_issue:
            raise ValueError(
                f"派奖活动规则 {current.name!r} 的 start_issue 大于 end_issue"
            )

        if previous is not None:
            if current.start_issue <= previous.end_issue:
                raise ValueError(
                    f"派奖活动规则区间存在重叠: {previous.name!r} 与 {current.name!r}"
                )

        previous = current


_validate_rule_versions()
_validate_bonus_campaigns()
