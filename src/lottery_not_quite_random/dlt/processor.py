"""大乐透 processed 数据生成模块。

该模块负责将 raw 层的大乐透开奖数据转换为
processed 层 parquet 数据。

数据流程：

    data/dlt/raw/draws.json
              |
              v
        processor.py
              |
              v
    data/dlt/processed/draws.parquet


主要职责：

- 读取 raw/draws.json。
- 解析官方开奖字符串。
- 转换为结构化开奖事实数据。
- 生成基础统计字段。
- 执行数据质量检查。
- 输出稳定 schema 的 parquet 文件。


设计原则：

- 不修改 raw 数据。
- 不依赖数据库。
- 不依赖 DuckDB。
- processed 层只保存开奖事实数据。
- 不生成依赖历史窗口的 feature。


字段设计参考：

dlt_draws schema：

- issue
- draw_date
- draw_result

- red_1 ~ red_5
- blue_1 ~ blue_2

- red_sum
- red_span
- blue_sum
- blue_span

- red_odd_count
- red_even_count
- blue_odd_count
- blue_even_count

- red_zone_1_count
- red_zone_2_count
- red_zone_3_count

- red_consecutive_group_count
- red_max_consecutive_length

- year
- month
- day_of_week

- draw_index


不负责：

- 数据采集（fetcher）
- 特征工程（features）
- 统计分析（analysis）
- 模型训练（models）
"""

import logging
from datetime import date
from itertools import pairwise
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from ..config import DLT_PROCESSED_DRAWS_FILE, DLT_RAW_DRAWS_FILE
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


def transform_record(
    record: dict[str, Any],
    draw_index: int,
) -> dict[str, Any]:
    """转换单条 raw 数据。

    Args:
        record: raw 开奖记录。
        draw_index: 数据索引。

    Returns:
        dict: processed 数据记录。
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


def process_records(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """转换全部 raw 数据。

    Args:
        records: raw 数据列表。

    Returns:
        list: processed 数据列表。
    """

    result = []

    for index, record in enumerate(records):
        result.append(
            transform_record(
                record,
                index,
            )
        )

    return result


def write_parquet(
    records: list[dict[str, Any]],
) -> None:
    """写入 parquet 文件。

    Args:
        records: processed 数据。
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
        "生成 parquet 完成: %s",
        DLT_PROCESSED_DRAWS_FILE,
    )


def build_processed_draws() -> None:
    """构建大乐透 processed parquet。

    Raises:
        RuntimeError: raw 数据为空时抛出。
    """

    logger.info("开始生成大乐透 processed 数据")

    records = load_json(DLT_RAW_DRAWS_FILE)

    if not isinstance(records, list):
        raise ValueError("raw/draws.json 必须是 list")

    if not records:
        raise RuntimeError("raw/draws.json 为空")

    processed_records = process_records(records)

    write_parquet(processed_records)

    logger.info(
        "处理完成，共 %s 条记录",
        len(processed_records),
    )
