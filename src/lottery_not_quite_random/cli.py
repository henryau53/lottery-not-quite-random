import argparse
import logging

from .dlt import fetcher as dlt_fetcher
from .dlt import processor as dlt_processor

logger = logging.getLogger(__name__)


def _parse_args(
    argv: list[str] | None = None,
):
    """解析命令行参数。

    Args:
        argv: 命令行参数列表。

    Returns:
        argparse.Namespace: 解析后的参数。
    """

    parser = argparse.ArgumentParser(
        description=("一本正经地用量化的方法，认真研究彩票到底有没有规律"),
        add_help=False,
    )

    parser.add_argument(
        "-h",
        "--help",
        action="help",
        default=argparse.SUPPRESS,
        help="显示帮助信息并退出",
    )

    sub_parsers = parser.add_subparsers(
        dest="command",
        required=True,
        description="选择要运行的彩票品种。",
        help="彩票品种，当前支持：dlt（体彩大乐透）。",
    )

    dlt_parser = sub_parsers.add_parser(
        "dlt",
        description="体彩大乐透",
        help="体彩大乐透",
        add_help=False,
    )

    dlt_parser.add_argument(
        "-h",
        "--help",
        action="help",
        default=argparse.SUPPRESS,
        help="显示帮助信息并退出",
    )

    # -------------------------------------------------------------------------
    # raw 数据同步
    # -------------------------------------------------------------------------

    dlt_parser.add_argument(
        "--full",
        action="store_true",
        help="强制执行全量抓取并重建 draws.json",
    )

    dlt_parser.add_argument(
        "--rebuild-meta",
        action="store_true",
        help="只根据现有 draws.json 重建 meta.json，不访问网络",
    )

    dlt_parser.add_argument(
        "--page-size",
        type=int,
        default=100,
        help="API 每页请求数量，最大 100，默认 100",
    )

    dlt_parser.add_argument(
        "--sleep",
        type=float,
        default=0.3,
        help="请求之间的等待秒数，默认 0.3",
    )

    dlt_parser.add_argument(
        "--timeout",
        type=int,
        default=10,
        help="HTTP 请求超时时间，默认 10 秒",
    )

    dlt_parser.add_argument(
        "--max-pages",
        type=int,
        default=1000,
        help="最大抓取页数，默认 1000",
    )

    # -------------------------------------------------------------------------
    # processed 数据生成
    # -------------------------------------------------------------------------

    dlt_parser.add_argument(
        "--process",
        action="store_true",
        help=("数据处理，根据 raw/draws.json 生成 processed/draws.parquet"),
    )

    return parser.parse_args(argv)


def _run_dlt(
    args: argparse.Namespace,
) -> None:
    """执行 dlt 子命令。

    Args:
        args: 命令参数。

    Raises:
        ValueError: 参数错误。
        RuntimeError: 执行失败。
    """

    if args.process:
        dlt_processor.build_processed_draws()

        logger.info(
            "processed 生成完成：%s",
            dlt_processor.DLT_PROCESSED_DRAWS_FILE,
        )

        return

    # -------------------------------------------------------------------------
    # 仅重建 meta
    # -------------------------------------------------------------------------

    if args.rebuild_meta:
        result = dlt_fetcher.rebuild_meta_only()

        logger.info(
            "操作结果：%s",
            result,
        )

        return

    # -------------------------------------------------------------------------
    # 参数检查
    # -------------------------------------------------------------------------

    if args.page_size <= 0:
        raise ValueError("--page-size 必须大于 0")

    if args.page_size > (dlt_fetcher.SPORTTERY_MAX_PAGE_SIZE):
        raise ValueError(f"--page-size 不能超过 {dlt_fetcher.SPORTTERY_MAX_PAGE_SIZE}")

    if args.sleep < 0:
        raise ValueError("--sleep 不能为负数")

    if args.timeout <= 0:
        raise ValueError("--timeout 必须大于 0")

    if args.max_pages <= 0:
        raise ValueError("--max-pages 必须大于 0")

    # -------------------------------------------------------------------------
    # raw 同步
    # -------------------------------------------------------------------------

    if args.full:
        result = dlt_fetcher.full_sync(
            page_size=args.page_size,
            sleep_seconds=args.sleep,
            timeout=args.timeout,
            max_pages=args.max_pages,
        )

    else:
        result = dlt_fetcher.incremental_sync(
            page_size=args.page_size,
            sleep_seconds=args.sleep,
            timeout=args.timeout,
            max_pages=args.max_pages,
        )

    logger.info(
        "操作结果：%s",
        result,
    )


def main(
    argv: list[str] | None = None,
) -> int:
    """程序入口。

    Args:
        argv: 命令行参数。

    Returns:
        int: 退出码。
    """

    logging.basicConfig(
        level=logging.INFO,
        format=("%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
        force=True,
    )

    args = _parse_args(argv)

    try:
        match args.command:
            case "dlt":
                _run_dlt(args)

            case _:
                raise ValueError(f"未知命令: {args.command}")

    except Exception as e:
        logger.error(
            "执行失败：%s",
            e,
        )

        logger.debug(
            "详细堆栈：",
            exc_info=True,
        )

        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
