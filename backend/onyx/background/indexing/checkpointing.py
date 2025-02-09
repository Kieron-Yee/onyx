"""Experimental functionality related to splitting up indexing
into a series of checkpoints to better handle intermittent failures
/ jobs being killed by cloud providers.

实验性功能：将索引过程分解为一系列检查点，以更好地处理间歇性故障和被云提供商终止的作业。
"""

import datetime

from onyx.configs.app_configs import EXPERIMENTAL_CHECKPOINTING_ENABLED
from onyx.configs.constants import DocumentSource
from onyx.connectors.cross_connector_utils.miscellaneous_utils import datetime_to_utc


def _2010_dt() -> datetime.datetime:
    """返回2010年1月1日的UTC时间"""
    return datetime.datetime(year=2010, month=1, day=1, tzinfo=datetime.timezone.utc)


def _2020_dt() -> datetime.datetime:
    """返回2020年1月1日的UTC时间"""
    return datetime.datetime(year=2020, month=1, day=1, tzinfo=datetime.timezone.utc)


def _default_end_time(
    last_successful_run: datetime.datetime | None,
) -> datetime.datetime:
    """If year is before 2010, go to the beginning of 2010.
    If year is 2010-2020, go in 5 year increments.
    If year > 2020, then go in 180 day increments.

    如果年份在2010年之前，则从2010年初开始。
    如果年份在2010-2020年之间，则以5年为增量。
    如果年份在2020年之后，则以180天为增量。

    For connectors that don't support a `filter_by` and instead rely on `sort_by`
    for polling, then this will cause a massive duplication of fetches. For these
    connectors, you may want to override this function to return a more reasonable
    plan (e.g. extending the 2020+ windows to 6 months, 1 year, or higher).

    对于不支持`filter_by`而是依赖`sort_by`进行轮询的连接器，这将导致大量重复获取。
    对于这些连接器，你可能需要重写此函数以返回更合理的计划（例如，将2020年后的时间窗口延长到6个月、1年或更长）。

    Args:
        last_successful_run: 上次成功运行的时间，可以为None
        
    Returns:
        datetime.datetime: 计算得出的结束时间
    """
    last_successful_run = (
        datetime_to_utc(last_successful_run) if last_successful_run else None
    )
    if last_successful_run is None or last_successful_run < _2010_dt():
        return _2010_dt()

    if last_successful_run < _2020_dt():
        return min(last_successful_run + datetime.timedelta(days=365 * 5), _2020_dt())

    return last_successful_run + datetime.timedelta(days=180)


def find_end_time_for_indexing_attempt(
    last_successful_run: datetime.datetime | None,
    source_type: DocumentSource,
) -> datetime.datetime | None:
    """Is the current time unless the connector is run over a large period, in which case it is
    split up into large time segments that become smaller as it approaches the present

    除非连接器运行时间跨度较大，否则返回当前时间。对于大时间跨度，随着接近当前时间，
    时间段会被分割成越来越小的部分。

    Args:
        last_successful_run: 上次成功运行的时间
        source_type: 文档源类型，可用于为特定连接器重写默认行为（当前未使用）

    Returns:
        datetime.datetime | None: 如果返回None表示应该索引到当前时间
    """
    # NOTE: source_type can be used to override the default for certain connectors
    # 注意：source_type可用于为某些连接器重写默认行为
    end_of_window = _default_end_time(last_successful_run)
    now = datetime.datetime.now(tz=datetime.timezone.utc)
    if end_of_window < now:
        return end_of_window

    # None signals that we should index up to current time
    # None表示我们应该索引到当前时间
    return None


def get_time_windows_for_index_attempt(
    last_successful_run: datetime.datetime, source_type: DocumentSource
) -> list[tuple[datetime.datetime, datetime.datetime]]:
    """获取索引尝试的时间窗口列表

    Args:
        last_successful_run: 上次成功运行的时间
        source_type: 文档源类型

    Returns:
        list[tuple[datetime.datetime, datetime.datetime]]: 时间窗口列表，每个元素为(开始时间, 结束时间)的元组
    """
    if not EXPERIMENTAL_CHECKPOINTING_ENABLED:
        return [(last_successful_run, datetime.datetime.now(tz=datetime.timezone.utc))]

    time_windows: list[tuple[datetime.datetime, datetime.datetime]] = []
    start_of_window: datetime.datetime | None = last_successful_run
    while start_of_window:
        end_of_window = find_end_time_for_indexing_attempt(
            last_successful_run=start_of_window, source_type=source_type
        )
        time_windows.append(
            (
                start_of_window,
                end_of_window or datetime.datetime.now(tz=datetime.timezone.utc),
            )
        )
        start_of_window = end_of_window

    return time_windows
