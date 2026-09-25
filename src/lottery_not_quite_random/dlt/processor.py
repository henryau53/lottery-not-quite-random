"""大乐透 processed 数据预处理生成模块。

该模块负责将 raw 层的大乐透开奖数据转换为
processed 层 parquet 数据。

数据流程：

    data/dlt/raw/draws.json
              |
              v
        processor.py
              |
        +-----+-----+
        |           |
        v           v
 draws.parquet  prizes.parquet
"""

import logging
from collections.abc import Callable
from datetime import date
from itertools import pairwise
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from ..config import (
    DLT_PROCESSED_DRAWS_FILE,
    DLT_PROCESSED_PRIZES_FILE,
    DLT_RAW_DRAWS_FILE,
)
from ..utils import load_json

logger = logging.getLogger(__name__)


# =============================================================================
# Parquet schema
# =============================================================================

DLT_DRAW_SCHEMA = pa.schema(
    [
        pa.field(
            "issue",
            pa.string(),
            nullable=False,
        ),
        pa.field(
            "draw_date",
            pa.date32(),
            nullable=False,
        ),
        pa.field(
            "draw_result",
            pa.string(),
            nullable=False,
        ),
        pa.field(
            "draw_result_unsorted",
            pa.string(),
            nullable=True,
        ),
        pa.field(
            "total_sale_amount",
            pa.float64(),
            nullable=True,
        ),
        pa.field(
            "pool_balance",
            pa.float64(),
            nullable=True,
        ),
        pa.field(
            "pool_balance_afterdraw",
            pa.float64(),
            nullable=True,
        ),
        pa.field("red_1", pa.int32(), nullable=False),
        pa.field("red_2", pa.int32(), nullable=False),
        pa.field("red_3", pa.int32(), nullable=False),
        pa.field("red_4", pa.int32(), nullable=False),
        pa.field("red_5", pa.int32(), nullable=False),
        pa.field("blue_1", pa.int32(), nullable=False),
        pa.field("blue_2", pa.int32(), nullable=False),
        pa.field("red_sum", pa.int32(), nullable=False),
        pa.field("red_span", pa.int32(), nullable=False),
        pa.field("blue_sum", pa.int32(), nullable=False),
        pa.field("blue_span", pa.int32(), nullable=False),
        pa.field("red_odd_count", pa.int32(), nullable=False),
        pa.field("red_even_count", pa.int32(), nullable=False),
        pa.field("blue_odd_count", pa.int32(), nullable=False),
        pa.field("blue_even_count", pa.int32(), nullable=False),
        pa.field("red_zone_1_count", pa.int32(), nullable=False),
        pa.field("red_zone_2_count", pa.int32(), nullable=False),
        pa.field("red_zone_3_count", pa.int32(), nullable=False),
        pa.field("red_consecutive_group_count", pa.int32(), nullable=False),
        pa.field("red_max_consecutive_length", pa.int32(), nullable=False),
        pa.field("year", pa.int32(), nullable=False),
        pa.field("month", pa.int32(), nullable=False),
        pa.field("day_of_week", pa.int32(), nullable=False),
        pa.field("draw_index", pa.int32(), nullable=False),
    ]
)


DLT_PRIZE_SCHEMA = pa.schema(
    [
        pa.field("issue", pa.string(), nullable=False),
        pa.field("draw_date", pa.date32(), nullable=False),
        pa.field("rule_version", pa.string(), nullable=False),
        pa.field("prize_rank", pa.int32(), nullable=False),
        pa.field("prize_name", pa.string(), nullable=False),
        pa.field("prize_event_type", pa.string(), nullable=False),
        pa.field("prize_level_raw", pa.string(), nullable=False),
        pa.field("stake_amount", pa.float64(), nullable=True),
        pa.field("stake_count", pa.int64(), nullable=False),
        pa.field("total_prize_amount", pa.float64(), nullable=False),
    ]
)


# =============================================================================
# PRIZE ENUMERATIONS
# =============================================================================


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


PRIZE_RULE_TIMELINE = [
    (None, "14051", "v1"),
    ("14052", "19018", "v2"),
    ("19019", "26013", "v3"),
    ("26014", None, "v4"),
]


VALID_PRIZE_EVENT_TYPES = {
    "basic_prize",
    "additional_prize",
    "basic_bonus",
    "additional_bonus",
    "unknown",
}


# =============================================================================
# Utility functions
# =============================================================================


def parse_numeric_value[NumberT: (int, float)](
    value: Any, type_func: Callable[[Any], NumberT] = int
) -> NumberT | None:
    """将分组数值转换为数值。

    例如：12,345 转换为 12345

    Args:
        value (Any): 需转换的值
        type_func (Any, optional): 需要转换的类型类，默认 int

    Returns:
        Any | 0: 转换后的数值；无法转换时返回 None。
    """
    if value in (None, "", "---", "-1"):
        return None

    return type_func(str(value).replace(",", ""))


# =============================================================================
# Validation
# =============================================================================


def validate_red_numbers(
    red: list[int],
) -> None:
    """验证大乐透前区号码。

    Args:
        red: 前区号码列表。

    Raises:
        ValueError: 前区号码不符合规则时抛出。
    """

    if len(red) != 5:
        raise ValueError(f"前区号码数量错误: {red}")

    if len(set(red)) != 5:
        raise ValueError(f"前区存在重复号码: {red}")

    if not all(1 <= n <= 35 for n in red):
        raise ValueError(f"前区号码超出范围: {red}")

    if red != sorted(red):
        raise ValueError(f"前区号码未排序: {red}")


def validate_blue_numbers(
    blue: list[int],
) -> None:
    """验证大乐透后区号码。

    Args:
        blue: 后区号码列表。

    Raises:
        ValueError: 后区号码不符合规则时抛出。
    """

    if len(blue) != 2:
        raise ValueError(f"后区号码数量错误: {blue}")

    if len(set(blue)) != 2:
        raise ValueError(f"后区存在重复号码: {blue}")

    if not all(1 <= n <= 12 for n in blue):
        raise ValueError(f"后区号码超出范围: {blue}")

    if blue != sorted(blue):
        raise ValueError(f"后区号码未排序: {blue}")


def validate_prize_record(
    prize: dict[str, Any],
) -> None:
    """验证单条奖金记录。

    Args:
        prize:
            已解析后的奖金记录。

    Raises:
        ValueError:
            奖金字段不符合预期规则时抛出。
    """

    prize_rank = prize.get("prize_rank")

    if not isinstance(
        prize_rank,
        int,
    ):
        raise ValueError(f"奖级编号类型错误: {prize}")

    if not 1 <= prize_rank <= 9:
        raise ValueError(f"奖级编号超出范围: {prize_rank}")

    stake_count = prize.get("stake_count")

    if not isinstance(
        stake_count,
        (float, int),
    ):
        raise ValueError(f"中奖注数类型错误: {prize}")

    if stake_count < 0:
        raise ValueError(f"中奖注数异常: {stake_count}")

    stake_amount = prize.get("stake_amount")

    if stake_amount is not None and stake_amount < 0:
        raise ValueError(f"单注奖金金额异常: {stake_amount}")

    total_amount = prize.get("total_prize_amount")

    if not isinstance(
        total_amount,
        (float, int),
    ):
        raise ValueError(f"总奖金金额类型错误: {prize}")

    if total_amount < 0:
        raise ValueError(f"总奖金金额异常: {total_amount}")


def calculate_consecutive_features(
    numbers: list[int],
) -> tuple[int, int]:
    """计算连续号码特征。

    计算两个指标：

    1. consecutive_group_count: 连续号码组数量。
    2. max_consecutive_length: 最大连续号码长度。

    Examples:
        [1, 2, 3, 4, 5] -> (1, 5)
        [1, 2, 3, 5, 6] -> (2, 3)
        [1, 3, 5, 6, 8] -> (1, 2)
        [1, 2, 4, 5] -> (2, 2)
        [1, 3, 5, 7, 9] -> (0, 1)

    Args:
        numbers: 已排序号码列表。

    Returns:
        tuple[int, int]: (连续号码组数量, 最大连续长度)
    """

    if not numbers:
        return 0, 0

    if len(numbers) == 1:
        return 0, 1

    group_count = 0
    max_length = 1
    current_length = 1
    in_consecutive_group = False

    for previous, current in pairwise(
        numbers,
    ):
        if current - previous == 1:
            current_length += 1

            if not in_consecutive_group:
                group_count += 1
                in_consecutive_group = True
        else:
            current_length = 1
            in_consecutive_group = False

        max_length = max(max_length, current_length)

    return group_count, max_length


def get_prize_rule_version(issue: str) -> str:
    """根据期号获取奖级规则版本。

    Args:
        issue (str): 期号

    Raises:
        ValueError: 无法确定奖级规则版本

    Returns:
        str: 奖级规则版本
    """
    issue_number = int(issue)

    for start, end, version in PRIZE_RULE_TIMELINE:
        if start is not None and issue_number < int(start):
            continue
        if end is not None and issue_number > int(end):
            continue
        return version

    raise ValueError(f"无法确定奖级规则版本: {issue}")


def parse_prize_name(prize_level: str) -> tuple[int, str]:
    """解析奖级编号和标准名称。

    Args:
        prize_level (str): raw 中 prizeLevel 奖级名称值

    Raises:
        ValueError: 未知奖级

    Returns:
        tuple[int, str]: (奖级序号, 标准化奖级名称)
    """
    for name, rank in PRIZE_RANK_MAPPING.items():
        if name in prize_level:
            return rank, name

    raise ValueError(f"未知奖级: {prize_level}")


def parse_prize_event_type(prize_level: str) -> str:
    """解析奖金事件类型。

    Args:
        prize_level (str): raw 中 prizeLevel 奖级名称值

    Returns:
        str: 奖金事件类型值
    """
    if "追加派奖" in prize_level:
        return "additional_bonus"
    if "派奖" in prize_level:
        return "basic_bonus"
    if "追加" in prize_level:
        return "additional_prize"
    return "basic_prize"


# =============================================================================
# Transform
# =============================================================================


def parse_draw_result(
    draw_result: str,
) -> tuple[list[int], list[int]]:
    """解析官方开奖号码。

    Args:
        draw_result: 例如 "02 05 07 14 22 04 10"

    Returns:
        tuple: (前区号码, 后区号码)
    """

    values = [int(x) for x in draw_result.split()]

    if len(values) != 7:
        raise ValueError(f"开奖格式错误: {draw_result}")

    red = values[:5]
    blue = values[5:]

    validate_red_numbers(red)
    validate_blue_numbers(blue)

    return red, blue


def transform_draw_record(
    record: dict[str, Any],
    draw_index: int,
) -> dict[str, Any]:
    """转换单期开奖号码数据。

    Args:
        record: 单期开奖记录。
        draw_index: 数据索引。

    Returns:
        dict: processed 开奖号码数据记录。
    """

    draw_result = record.get("lotteryDrawResult")

    if not draw_result:
        raise ValueError("缺少 lotteryDrawResult")

    red, blue = parse_draw_result(draw_result)

    (
        red_consecutive_group_count,
        red_max_consecutive_length,
    ) = calculate_consecutive_features(red)

    draw_date = date.fromisoformat(record["lotteryDrawTime"])

    return {
        "issue": str(record["lotteryDrawNum"]),
        "draw_date": draw_date,
        "draw_result": draw_result,
        "draw_result_unsorted": record.get("lotteryUnsortDrawresult") or None,
        "total_sale_amount": parse_numeric_value(
            record.get("totalSaleAmount"),
            float,
        )
        or None,
        "pool_balance": parse_numeric_value(
            record.get("poolBalance"),
            float,
        )
        or None,
        "pool_balance_afterdraw": parse_numeric_value(
            record.get("poolBalanceAfterdraw"),
            float,
        )
        or None,
        "red_1": red[0],
        "red_2": red[1],
        "red_3": red[2],
        "red_4": red[3],
        "red_5": red[4],
        "blue_1": blue[0],
        "blue_2": blue[1],
        "red_sum": sum(red),
        "red_span": (max(red) - min(red)),
        "blue_sum": sum(blue),
        "blue_span": (max(blue) - min(blue)),
        "red_odd_count": sum(n % 2 == 1 for n in red),
        "red_even_count": sum(n % 2 == 0 for n in red),
        "blue_odd_count": sum(n % 2 == 1 for n in blue),
        "blue_even_count": sum(n % 2 == 0 for n in blue),
        "red_zone_1_count": sum(1 <= n <= 12 for n in red),
        "red_zone_2_count": sum(13 <= n <= 24 for n in red),
        "red_zone_3_count": sum(25 <= n <= 35 for n in red),
        "red_consecutive_group_count": red_consecutive_group_count,
        "red_max_consecutive_length": red_max_consecutive_length,
        "year": draw_date.year,
        "month": draw_date.month,
        "day_of_week": draw_date.isoweekday(),
        "draw_index": draw_index,
    }


# =============================================================================
# Build parquet
# =============================================================================


def process_draw_records(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """转换全部开奖号码数据。

    Args:
        records: 开奖号码数据列表。

    Returns:
        list: processed 开奖号码数据列表。
    """

    result = []

    # 按照开奖期号升序排序，保证下方 index 的顺序
    sorted_records = sorted(records, key=lambda r: int(r["lotteryDrawNum"]))

    for index, record in enumerate(sorted_records):
        result.append(
            transform_draw_record(
                record,
                index,
            )
        )

    return result


def write_draws_parquet(
    records: list[dict[str, Any]],
) -> None:
    """写入 draws.parquet 文件。

    Args:
        records: processed 开奖号码数据。
    """

    table = pa.Table.from_pylist(
        records,
        schema=DLT_DRAW_SCHEMA,
    )

    DLT_PROCESSED_DRAWS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pq.write_table(
        table,
        DLT_PROCESSED_DRAWS_FILE,
        compression="zstd",
    )

    logger.info(
        "生成开奖号码 parquet 完成: %s",
        DLT_PROCESSED_DRAWS_FILE,
    )


def transform_prize_record(
    record: dict[str, Any],
) -> list[dict[str, Any]]:
    """转换单期开奖奖金数据。

    Args:
        record: raw 层单期开奖数据。

    Returns:
        当前开奖期所有奖级记录。

    Raises:
        ValueError:
            raw 数据缺少必要字段或奖级无法解析。
    """
    issue = str(record["lotteryDrawNum"])
    draw_date = date.fromisoformat(record["lotteryDrawTime"])
    rule_version = get_prize_rule_version(issue)

    result = []

    for prize in record.get("prizeLevelList", []):
        prize_level_raw = str(prize.get("prizeLevel", ""))

        if not prize_level_raw:
            raise ValueError(f"{issue} 存在空奖级")

        prize_rank, prize_name = parse_prize_name(prize_level_raw)

        item = {
            "issue": issue,
            "draw_date": draw_date,
            "rule_version": rule_version,
            "prize_rank": prize_rank,
            "prize_name": prize_name,
            "prize_event_type": parse_prize_event_type(prize_level_raw),
            "prize_level_raw": prize_level_raw,
            "stake_amount": parse_numeric_value(prize.get("stakeAmountFormat"), float),
            "stake_count": parse_numeric_value(prize.get("stakeCount", 0), int),
            "total_prize_amount": parse_numeric_value(
                prize.get("totalPrizeamount"), float
            ),
        }

        validate_prize_record(item)
        result.append(item)

    return result


def process_prize_records(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """转换全部开奖奖金数据。

    Args:
        records:
            raw 层开奖记录列表。

    Returns:
        processed 层奖金记录列表。

    Raises:
        RuntimeError:
            未生成任何奖金记录。
    """
    result = []

    for record in records:
        result.extend(transform_prize_record(record))

    return result


def write_prizes_parquet(
    records: list[dict[str, Any]],
) -> None:
    """写入 prizes.parquet 文件。

    Args:
        records:
            已处理后的奖金结构化数据。

    Raises:
        ValueError:
            数据无法匹配 DLT_PRIZE_SCHEMA。
    """
    table = pa.Table.from_pylist(
        records,
        schema=DLT_PRIZE_SCHEMA,
    )

    DLT_PROCESSED_PRIZES_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pq.write_table(
        table,
        DLT_PROCESSED_PRIZES_FILE,
        compression="zstd",
    )

    logger.info(
        "生成奖金 parquet 完成: %s",
        DLT_PROCESSED_PRIZES_FILE,
    )


def build_processed() -> None:
    """构建大乐透 processed 层 parquet 数据。

    同时生成：

    - draws.parquet:
        开奖事实数据。

    - prizes.parquet:
        奖金分布数据。

    Raises:
        ValueError:
            raw 数据格式错误。

        RuntimeError:
            raw 数据为空。
    """
    logger.info("开始生成大乐透预处理数据")

    records = load_json(DLT_RAW_DRAWS_FILE)

    if not records:
        raise RuntimeError("raw/draws.json 为空")

    if not isinstance(records, list):
        raise ValueError("raw/draws.json 必须是 list")

    issues = [record["lotteryDrawNum"] for record in records]

    if len(set(issues)) != len(issues):
        raise ValueError("raw/draws.json 存在重复开奖期号记录")

    processed_draws = process_draw_records(records)
    write_draws_parquet(processed_draws)

    processed_prizes = process_prize_records(records)
    write_prizes_parquet(processed_prizes)

    logger.info(
        "处理完成，共 %s 条开奖记录，共 %s 条奖金记录",
        len(processed_draws),
        len(processed_prizes),
    )
