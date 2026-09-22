import logging
import time
from ..config import DLT_RAW_DRAWS_FILE, DLT_RAW_META_FILE

import requests

from ..utils import atomic_write_json, load_json, utc_now_iso

logger = logging.getLogger(__name__)


# =============================================================================
# 大乐透说明
# =============================================================================
#
# 【规则】
#   前区：01-35 选 5
#   后区：01-12 选 2
#   开奖时间：周一、三、六 21:10
#   九个奖级：一等奖(5+2) 到 九等奖(3+0/1+2/0+2)
#
# 【接口来源】
#   大乐透开奖列表：
#       https://www.sporttery.cn/kj/kjlb.html?dlt
#
#   大乐透走势图：
#       https://www.sporttery.cn/zst/dlt/
#
# 【历史开奖接口】
#   https://webapi.sporttery.cn/gateway/lottery/getHistoryPageListV1.qry
#
# 【接口参数】
#   gameNo=85
#   provinceId=0
#   isVerify=1
#   pageNo=<page>
#   pageSize=<size>
#
# 【重要】
#   单次请求最大 pageSize = 100
#
# =============================================================================


SPORTTERY_URL = "https://webapi.sporttery.cn/gateway/lottery/getHistoryPageListV1.qry"
SPORTTERY_REFERER = "https://www.sporttery.cn/"

GAME_NO = "85"
PROVINCE_ID = "0"
IS_VERIFY = "1"

# getHistoryPageListV1 接口查询最大分页条数
# 注意，该值不要更改，sporttery.cn 接口硬性上限
SPORTTERY_MAX_PAGE_SIZE = 100


# =============================================================================
# Utility functions
# =============================================================================


def load_raw_draws() -> list[dict]:
    """读取 raw/draws.json 中的原始开奖数据。

    raw 文件约定直接保存 API 返回的 ``value.list``，例如：

        [
            {
                "lotteryDrawNum": "...",
                ...
            },
            {
                "lotteryDrawNum": "...",
                ...
            }
        ]

    Returns:
        list[dict]: 原始开奖记录。
            如果 raw/draws.json 不存在，则返回空列表。

    Raises:
        ValueError: JSON 根节点不是 ``list`` 时抛出。
        json.JSONDecodeError: 文件内容不是合法 JSON 时抛出。
    """
    if not DLT_RAW_DRAWS_FILE.exists():
        return []

    data = load_json(DLT_RAW_DRAWS_FILE)

    if not isinstance(data, list):
        raise ValueError(f"{DLT_RAW_DRAWS_FILE} 格式错误：根节点必须是 list")

    return data


def load_meta() -> dict:
    """读取 raw/meta.json 中的数据集元信息。

    ``meta.json`` 仅作为同步状态和数据集信息的缓存，
    不作为历史开奖数据的唯一来源。

    Returns:
        dict: 元信息字典。
            如果 meta.json 不存在，则返回空字典。

    Raises:
        ValueError: JSON 根节点不是 ``dict`` 时抛出。
        json.JSONDecodeError: 文件内容不是合法 JSON 时抛出。
    """
    if not DLT_RAW_META_FILE.exists():
        return {}

    data = load_json(DLT_RAW_META_FILE)

    if not isinstance(data, dict):
        raise ValueError(f"{DLT_RAW_META_FILE} 格式错误：根节点必须是 object")

    return data


def get_issue(record: dict) -> str:
    """获取一条开奖记录的期号。

    Args:
        record: API 返回的一条原始开奖记录。

    Returns:
        str: 开奖期号。

    Raises:
        ValueError: ``lotteryDrawNum`` 不存在或为空时抛出。
    """
    issue = record.get("lotteryDrawNum")

    if issue is None or str(issue).strip() == "":
        raise ValueError(f"发现开奖记录缺少 lotteryDrawNum：{record!r}")

    return str(issue)


def get_draw_time(record: dict) -> str | None:
    """获取一条开奖记录的开奖时间。

    Args:
        record: API 返回的一条原始开奖记录。

    Returns:
        str | None: 开奖时间。如果记录中不存在
            ``lotteryDrawTime``，则返回 ``None``。
    """
    value = record.get("lotteryDrawTime")

    if value is None:
        return None

    return str(value)


def build_issue_index(records: list[dict]) -> dict[str, dict]:
    """根据开奖记录构建期号索引。

    索引结构为：

        {
            "26107": {...},
            "26106": {...},
            ...
        }

    Args:
        records: 原始开奖记录列表。

    Returns:
        dict[str, dict]: 以期号为 key、开奖记录为 value 的字典。

    Raises:
        ValueError: 某条记录缺少有效期号，或者存在重复期号时抛出。
    """
    index: dict[str, dict] = {}

    for record in records:
        issue = get_issue(record)

        if issue in index:
            raise ValueError(f"raw 数据存在重复期号：{issue}")

        index[issue] = record

    return index


# =============================================================================
# HTTP / API
# =============================================================================


def create_session() -> requests.Session:
    """创建用于访问 Sporttery API 的 HTTP Session。

    Returns:
        requests.Session: 已配置请求头的 HTTP Session。
    """
    session = requests.Session()

    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) "
                "AppleWebKit/537.36 "
                "(KHTML, like Gecko) "
                "Chrome/146.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json",
            "Referer": SPORTTERY_REFERER,
        }
    )

    return session


def fetch_page(
    session: requests.Session,
    page_no: int,
    *,
    page_size: int = SPORTTERY_MAX_PAGE_SIZE,
    timeout: int = 10,
) -> dict:
    """请求 Sporttery 历史开奖接口的一页数据。

    返回 API 中的 ``value`` 对象。
    其中可能包含 ``pages``、``pageNo``、``pageSize``、``total``、
    ``list`` 和 ``lastPoolDraw`` 等字段。

    Args:
        session: 用于发送 HTTP 请求的 Session。
        page_no: 要请求的页码。
        page_size: 每页请求数量，接口最大支持 100 条。
        timeout: HTTP 请求超时时间，单位为秒。

    Returns:
        dict: API 返回的 ``value`` 对象。

    Raises:
        ValueError: ``page_size`` 超过接口限制，或者响应内容不是合法
            JSON 时抛出。
        requests.HTTPError: HTTP 请求返回错误状态码时抛出。
        RuntimeError: API 返回业务错误，或者 ``value`` 不是字典时抛出。
    """
    if page_size > SPORTTERY_MAX_PAGE_SIZE:
        raise ValueError(f"page_size 不能超过接口上限 {SPORTTERY_MAX_PAGE_SIZE}")

    params = {
        "gameNo": GAME_NO,
        "provinceId": PROVINCE_ID,
        "isVerify": IS_VERIFY,
        "pageNo": page_no,
        "pageSize": page_size,
    }

    logger.info(
        "请求 Sporttery：page=%s, page_size=%s",
        page_no,
        page_size,
    )

    response = session.get(
        SPORTTERY_URL,
        params=params,
        timeout=timeout,
    )

    response.raise_for_status()

    try:
        data = response.json()
    except ValueError as ve:
        raise ValueError(
            "Sporttery 返回的不是合法 JSON："
            f"status={response.status_code}, "
            f"body={response.text[:500]!r}"
        ) from ve

    if data.get("errorCode") != "0":
        raise RuntimeError(data.get("errorMsg") or "Sporttery API 返回业务错误")

    value = data.get("value")

    if not isinstance(value, dict):
        raise RuntimeError("Sporttery API 返回的 value 不是 object")

    return value


# =============================================================================
# API consistency checks
# =============================================================================


def validate_page_value(value: dict) -> None:
    """检查 API 返回的单页数据结构。

    正常情况下，接口返回的：

        lastPoolDraw.lotteryDrawNum

    应当与：

        list[0].lotteryDrawNum

    一致。

    ``lastPoolDraw`` 仅用于同步辅助和一致性检查，
    不作为独立历史数据写入 raw。

    Args:
        value: Sporttery API 返回的 ``value`` 对象。

    Raises:
        RuntimeError: ``value.list`` 缺失、类型错误，或者
            ``lastPoolDraw`` 与 ``list[0]`` 的期号不一致时抛出。
    """
    records = value.get("list")

    if records is None:
        raise RuntimeError("Sporttery API 返回中缺少 value.list")

    if not isinstance(records, list):
        raise RuntimeError("Sporttery API 返回中的 value.list 不是 list")

    # -------------------------------------------------------------------------
    # lastPoolDraw 一致性检查
    # -------------------------------------------------------------------------

    last_pool_draw = value.get("lastPoolDraw")

    if last_pool_draw and records:
        last_pool_issue = last_pool_draw.get("lotteryDrawNum")

        first_issue = records[0].get("lotteryDrawNum")

        if (
            last_pool_issue is not None
            and first_issue is not None
            and str(last_pool_issue) != str(first_issue)
        ):
            raise RuntimeError(
                "API 一致性检查失败："
                f"lastPoolDraw.lotteryDrawNum="
                f"{last_pool_issue!r}，"
                f"但 list[0].lotteryDrawNum="
                f"{first_issue!r}"
            )


def validate_records(records: list[dict]) -> None:
    """检查原始开奖记录。

    进行检查的内容：

    - 期号唯一性

    Args:
        records: 待检查的原始开奖记录列表。

    Raises:
        ValueError: ``records`` 不是列表、某条记录不是字典、
            记录缺少期号，或者存在重复期号时抛出。
    """
    if not isinstance(records, list):
        raise ValueError("records 必须是 list")

    seen: set[str] = set()

    for record in records:
        if not isinstance(record, dict):
            raise ValueError("开奖记录必须是 dict")

        issue = get_issue(record)

        if issue in seen:
            raise ValueError(f"同一批数据中存在重复期号：{issue}")

        seen.add(issue)


# =============================================================================
# Full synchronization
# =============================================================================


def fetch_all_history(
    *,
    page_size: int = SPORTTERY_MAX_PAGE_SIZE,
    sleep_seconds: float = 0.3,
    timeout: int = 10,
    max_pages: int = 1000,
) -> list[dict]:
    """全量抓取 Sporttery 历史开奖数据。

    函数会从第 1 页开始依次请求所有可用页面，
    将每一页的 ``value.list`` 合并成一个完整列表。

    ``lastPoolDraw`` 不会进入返回结果。

    Args:
        page_size: 每页请求数量，接口最大支持 100 条。
        sleep_seconds: 两次 HTTP 请求之间的等待时间，单位为秒。
        timeout: 单次 HTTP 请求的超时时间，单位为秒。
        max_pages: 最大请求页数，用于防止接口异常导致无限翻页。

    Returns:
        list[dict]: 全量历史开奖记录。

    Raises:
        ValueError: ``page_size`` 超过接口限制时抛出。
        RuntimeError: 接口返回异常数据，或者最终抓取结果为空时抛出。
    """
    if page_size > SPORTTERY_MAX_PAGE_SIZE:
        raise ValueError(f"page_size 不能超过 {SPORTTERY_MAX_PAGE_SIZE}")

    session = create_session()

    result: list[dict] = []

    page_no = 1
    total_pages = None

    while page_no <= max_pages:
        value = fetch_page(
            session,
            page_no,
            page_size=page_size,
            timeout=timeout,
        )

        validate_page_value(value)

        records = value.get("list") or []

        if not records:
            logger.info(
                "Sporttery 第 %s 页为空，结束全量抓取",
                page_no,
            )
            break

        validate_records(records)

        result.extend(records)

        api_pages = value.get("pages")

        if api_pages:
            total_pages = int(api_pages)

        display_total_pages = (
            min(total_pages, max_pages) if total_pages is not None else "?"
        )

        logger.info(
            "全量抓取：%s/%s 页，累计 %s 条",
            page_no,
            display_total_pages,
            len(result),
        )

        if total_pages is not None and page_no >= total_pages:
            break

        page_no += 1

        if page_no <= max_pages and sleep_seconds > 0:
            time.sleep(sleep_seconds)

    if not result:
        raise RuntimeError("全量抓取结果为空，拒绝覆盖现有 raw 数据")

    validate_records(result)

    return result


# =============================================================================
# Incremental synchronization
# =============================================================================


def get_local_latest_issue(records: list[dict]) -> str | None:
    """获取本地 raw 数据中的最新期号。

    当前约定 raw/draws.json 保持 API 返回的倒序结构，
    即最新一期位于列表第一个元素。

    Args:
        records: 本地原始开奖记录列表。

    Returns:
        str | None: 最新期号。如果列表为空，则返回 ``None``。
    """
    if not records:
        return None

    return get_issue(records[0])


def fetch_incremental(
    local_records: list[dict],
    *,
    page_size: int = SPORTTERY_MAX_PAGE_SIZE,
    sleep_seconds: float = 0.3,
    timeout: int = 10,
    max_pages: int = 1000,
) -> tuple[list[dict], dict]:
    """增量抓取本地 raw 数据中不存在的新开奖记录。

    接口按照最新期在前的倒序方式返回数据。

    基本流程：

    1. 从 API 第 1 页开始请求。
    2. 获取远端最新期号。
    3. 如果远端最新期与本地最新期一致，则认为数据已经最新。
    4. 否则逐条检查接口返回的记录。
    5. 将本地不存在的期号加入新增记录。
    6. 遇到本地已经存在的期号后停止继续寻找。
    7. 如果第一页没有找到本地已知期号，则继续请求下一页。
    8. 如果达到 ``max_pages`` 仍然找不到本地已知期号，则中止同步，
       避免将可能不完整的数据写入 raw。

    Args:
        local_records: 当前本地 raw 数据中的开奖记录。
        page_size: 每页请求数量，接口最大支持 100 条。
        sleep_seconds: 两次 HTTP 请求之间的等待时间，单位为秒。
        timeout: 单次 HTTP 请求的超时时间，单位为秒。
        max_pages: 最大检查页数。

    Returns:
        tuple[list[dict], dict]: 返回一个二元组：

            - ``new_records``：本地不存在的新增开奖记录。
            - ``sync_info``：本次同步的状态信息，包括本地最新期号、
              远端最新期号、新增记录数量以及请求页数等。

    Raises:
        ValueError: ``local_records`` 为空，或者无法确定本地最新期号时
            抛出。
        RuntimeError: 远端数据中始终没有找到本地已知期号时抛出。
    """
    if not local_records:
        raise ValueError("fetch_incremental 要求已有本地 raw 数据")

    local_index = build_issue_index(local_records)

    local_latest_issue = get_local_latest_issue(local_records)

    if local_latest_issue is None:
        raise ValueError("无法确定本地最新期号")

    session = create_session()

    new_records: list[dict] = []

    remote_latest_issue: str | None = None
    remote_latest_draw_time: str | None = None

    known_issue_found = False

    page_no = 1

    while page_no <= max_pages:
        value = fetch_page(
            session,
            page_no,
            page_size=page_size,
            timeout=timeout,
        )

        validate_page_value(value)

        records = value.get("list") or []

        if not records:
            break

        validate_records(records)

        # ---------------------------------------------------------------------
        # 第一页：
        # 获取远端最新期号。
        # ---------------------------------------------------------------------

        if page_no == 1:
            first_record = records[0]

            remote_latest_issue = get_issue(first_record)

            remote_latest_draw_time = get_draw_time(first_record)

            logger.info(
                "本地最新期：%s",
                local_latest_issue,
            )

            logger.info(
                "远端最新期：%s",
                remote_latest_issue,
            )

            if remote_latest_issue == local_latest_issue:
                logger.info("本地数据已经是最新，无需增量更新")

                return [], {
                    "status": "up_to_date",
                    "local_latest_issue": local_latest_issue,
                    "remote_latest_issue": remote_latest_issue,
                    "remote_latest_draw_time": (remote_latest_draw_time),
                    "new_records": 0,
                    "pages_fetched": 1,
                }

        # ---------------------------------------------------------------------
        # 收集新增记录。
        #
        # API 返回顺序：
        #
        #   新
        #   新
        #   新
        #   本地已有
        #   ...
        #
        # 因此遇到本地已知期号后即可停止。
        # ---------------------------------------------------------------------

        for record in records:
            issue = get_issue(record)

            if issue in local_index:
                known_issue_found = True
                break

            new_records.append(record)

        if known_issue_found:
            break

        api_pages = value.get("pages")

        if api_pages is not None:
            api_pages = int(api_pages)

            if page_no >= api_pages:
                break

        page_no += 1

        if page_no <= max_pages and sleep_seconds > 0:
            time.sleep(sleep_seconds)

    # -------------------------------------------------------------------------
    # 如果远端明显比本地新，但始终没有找到本地已知期号，
    # 不应该悄悄把可能不完整的数据写入 raw。
    # -------------------------------------------------------------------------

    if not known_issue_found:
        raise RuntimeError(
            "增量同步未找到本地已知期号 "
            f"{local_latest_issue}。"
            "为避免产生不完整 raw 数据，本次同步已中止。"
            "建议检查 API 返回顺序或执行 --full 重建。"
        )

    validate_records(new_records)

    sync_info = {
        "status": ("updated" if new_records else "up_to_date"),
        "local_latest_issue": local_latest_issue,
        "remote_latest_issue": remote_latest_issue,
        "remote_latest_draw_time": remote_latest_draw_time,
        "new_records": len(new_records),
        "pages_fetched": page_no,
    }

    return new_records, sync_info


# =============================================================================
# Raw dataset operations
# =============================================================================


def save_raw_draws(records: list[dict]) -> None:
    """保存 raw/draws.json。

    raw/draws.json 直接保存 API ``value.list`` 中的开奖记录
    Args:
        records: 要保存的原始开奖记录列表。

    Raises:
        ValueError: 数据结构不合法或存在重复期号时抛出。
        OSError: 文件写入失败时抛出。
        TypeError: 数据无法被 JSON 序列化时抛出。
    """
    validate_records(records)

    atomic_write_json(
        DLT_RAW_DRAWS_FILE,
        records,
    )

    logger.info(
        "raw 数据已保存：%s（%s 条）",
        DLT_RAW_DRAWS_FILE,
        len(records),
    )


def merge_incremental_records(
    local_records: list[dict],
    new_records: list[dict],
) -> list[dict]:
    """将新增开奖记录合并到本地 raw 数据。

    合并后保持最新记录在前、历史记录在后的顺序。

    函数同时根据 ``lotteryDrawNum`` 去重，避免重复写入同一期数据。

    Args:
        local_records: 当前本地 raw 数据。
        new_records: 本次增量同步获取的新记录。

    Returns:
        list[dict]: 合并并去重后的完整 raw 数据。

    Raises:
        ValueError: 输入数据结构不合法或合并后存在重复期号时抛出。
    """
    local_index = build_issue_index(local_records)

    unique_new_records: list[dict] = []
    seen_new: set[str] = set()

    for record in new_records:
        issue = get_issue(record)

        if issue in local_index:
            continue

        if issue in seen_new:
            continue

        seen_new.add(issue)
        unique_new_records.append(record)

    merged = unique_new_records + local_records

    validate_records(merged)

    return merged


def rebuild_meta(
    records: list[dict],
    *,
    last_sync: dict | None = None,
) -> dict:
    """根据 raw 数据重新构建 meta.json。

    ``meta.json`` 中的核心数据均从 ``draws.json`` 重新计算，
    因此即使 meta.json 丢失，也可以根据 raw 数据重新生成。

    Args:
        records: 当前完整的原始开奖记录。
        last_sync: 最近一次同步操作的信息。如果未提供，则使用默认值。

    Returns:
        dict: 根据 raw 数据生成的数据集元信息。

    Raises:
        ValueError: raw 数据结构不合法或存在重复期号时抛出。
    """
    validate_records(records)

    if records:
        latest_record = records[0]
        oldest_record = records[-1]

        latest_issue = get_issue(latest_record)
        oldest_issue = get_issue(oldest_record)

        latest_draw_date = get_draw_time(latest_record)
        oldest_draw_date = get_draw_time(oldest_record)
    else:
        latest_issue = None
        oldest_issue = None
        latest_draw_date = None
        oldest_draw_date = None

    meta = {
        "dataset": {
            "name": "dlt",
            "game_no": GAME_NO,
        },
        "source": {
            "provider": "sporttery.cn",
            "endpoint": SPORTTERY_URL,
        },
        "cursor": {
            "latest_issue": latest_issue,
            "latest_draw_date": latest_draw_date,
        },
        "range": {
            "first_issue": oldest_issue,
            "first_draw_date": oldest_draw_date,
            "last_issue": latest_issue,
            "last_draw_date": latest_draw_date,
        },
        "stats": {
            "records": len(records),
        },
        "sync": last_sync
        or {
            "last_success_at": None,
            "last_added": 0,
            "status": None,
        },
    }

    return meta


def save_meta(meta: dict) -> None:
    """保存 raw/meta.json。

    Args:
        meta: 要保存的数据集元信息。

    Raises:
        OSError: 文件写入失败时抛出。
        TypeError: ``meta`` 无法被 JSON 序列化时抛出。
    """
    atomic_write_json(
        DLT_RAW_META_FILE,
        meta,
    )

    logger.info(
        "meta 已保存：%s",
        DLT_RAW_META_FILE,
    )


# =============================================================================
# Sync workflows
# =============================================================================


def full_sync(
    *,
    page_size: int = SPORTTERY_MAX_PAGE_SIZE,
    sleep_seconds: float = 0.3,
    timeout: int = 10,
    max_pages: int = 1000,
) -> dict:
    """执行一次完整的历史数据同步。

    只有全量数据成功抓取并通过验证后，才会覆盖现有
    ``raw/draws.json``。

    Args:
        page_size: 每页请求数量，接口最大支持 100 条。
        sleep_seconds: 两次 HTTP 请求之间的等待时间，单位为秒。
        timeout: 单次 HTTP 请求的超时时间，单位为秒。
        max_pages: 最大请求页数。

    Returns:
        dict: 本次同步结果，包括同步状态、记录总数和新增数量。

    Raises:
        ValueError: 参数或数据结构不合法时抛出。
        RuntimeError: 全量抓取失败或结果为空时抛出。
    """
    logger.info("开始全量同步")

    records = fetch_all_history(
        page_size=page_size,
        sleep_seconds=sleep_seconds,
        timeout=timeout,
        max_pages=max_pages,
    )

    # -------------------------------------------------------------------------
    # 先保存完整 raw 数据。
    # -------------------------------------------------------------------------

    save_raw_draws(records)

    # -------------------------------------------------------------------------
    # 根据最终 raw 数据生成 meta。
    # -------------------------------------------------------------------------

    sync_info = {
        "last_success_at": utc_now_iso(),
        "last_added": len(records),
        "status": "full_sync",
    }

    meta = rebuild_meta(
        records,
        last_sync=sync_info,
    )

    save_meta(meta)

    logger.info(
        "全量同步完成：%s 条",
        len(records),
    )

    return {
        "status": "full_sync",
        "records": len(records),
        "added": len(records),
    }


def incremental_sync(
    *,
    page_size: int = SPORTTERY_MAX_PAGE_SIZE,
    sleep_seconds: float = 0.3,
    timeout: int = 10,
    max_pages: int = 1000,
) -> dict:
    """执行一次增量数据同步。

    如果本地不存在 ``raw/draws.json``，则自动退化为首次全量同步。

    如果本地已经存在 raw 数据，则从 API 第 1 页开始寻找
    本地尚未保存的新期号，并将新增记录合并到现有 raw 数据中。

    Args:
        page_size: 每页请求数量，接口最大支持 100 条。
        sleep_seconds: 两次 HTTP 请求之间的等待时间，单位为秒。
        timeout: 单次 HTTP 请求的超时时间，单位为秒。
        max_pages: 最大检查页数。

    Returns:
        dict: 本次同步结果，包括同步状态、记录总数和新增数量。

    Raises:
        ValueError: 参数或本地 raw 数据结构不合法时抛出。
        RuntimeError: API 返回异常，或者增量同步无法找到本地已知
            期号时抛出。
    """
    local_records = load_raw_draws()

    # -------------------------------------------------------------------------
    # 首次运行：不存在本地 raw 数据。
    # -------------------------------------------------------------------------

    if not local_records:
        logger.info("未发现现有 raw 数据，执行首次全量同步")

        return full_sync(
            page_size=page_size,
            sleep_seconds=sleep_seconds,
            timeout=timeout,
            max_pages=max_pages,
        )

    # -------------------------------------------------------------------------
    # 已存在数据：执行增量同步。
    # -------------------------------------------------------------------------

    new_records, sync_info = fetch_incremental(
        local_records,
        page_size=page_size,
        sleep_seconds=sleep_seconds,
        timeout=timeout,
        max_pages=max_pages,
    )

    # -------------------------------------------------------------------------
    # 没有新增数据。
    #
    # 即使没有新增，也刷新 meta 的同步状态。
    # -------------------------------------------------------------------------

    if not new_records:
        meta_sync = {
            "last_success_at": utc_now_iso(),
            "last_added": 0,
            "status": sync_info["status"],
        }

        meta = rebuild_meta(
            local_records,
            last_sync=meta_sync,
        )

        save_meta(meta)

        return {
            "status": sync_info["status"],
            "records": len(local_records),
            "added": 0,
        }

    # -------------------------------------------------------------------------
    # 合并新旧数据。
    # -------------------------------------------------------------------------

    merged_records = merge_incremental_records(
        local_records,
        new_records,
    )

    # -------------------------------------------------------------------------
    # 保存 raw。
    # -------------------------------------------------------------------------

    save_raw_draws(merged_records)

    # -------------------------------------------------------------------------
    # 保存 meta。
    # -------------------------------------------------------------------------

    meta_sync = {
        "last_success_at": utc_now_iso(),
        "last_added": len(new_records),
        "status": "incremental_sync",
    }

    meta = rebuild_meta(
        merged_records,
        last_sync=meta_sync,
    )

    save_meta(meta)

    logger.info(
        "增量同步完成：新增 %s 条，总计 %s 条",
        len(new_records),
        len(merged_records),
    )

    return {
        "status": "incremental_sync",
        "records": len(merged_records),
        "added": len(new_records),
    }


def rebuild_meta_only() -> dict:
    """仅根据现有 draws.json 重建 meta.json。

    此操作不会访问网络。

    适用于 meta.json 被删除、损坏，或者需要根据当前 raw 数据
    重新计算元信息的情况。

    Returns:
        dict: 重建结果，包括状态和当前 raw 记录数量。

    Raises:
        RuntimeError: 不存在有效的 raw/draws.json 时抛出。
        ValueError: raw 数据结构不合法时抛出。
    """
    records = load_raw_draws()

    if not records:
        raise RuntimeError("不存在有效的 raw/draws.json，无法重建 meta。")

    meta = rebuild_meta(
        records,
        last_sync={
            "last_success_at": None,
            "last_added": 0,
            "status": "rebuild_meta",
        },
    )

    save_meta(meta)

    return {
        "status": "rebuild_meta",
        "records": len(records),
    }
