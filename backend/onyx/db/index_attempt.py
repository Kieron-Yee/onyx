"""
这个文件主要负责索引尝试(IndexAttempt)相关的数据库操作。
包括创建、更新、查询和删除索引尝试记录，以及处理索引尝试过程中的错误记录。
主要用于跟踪文档索引的状态和进度。
"""

from collections.abc import Sequence
from datetime import datetime
from datetime import timedelta
from datetime import timezone

from sqlalchemy import and_
from sqlalchemy import delete
from sqlalchemy import desc
from sqlalchemy import func
from sqlalchemy import select
from sqlalchemy import update
from sqlalchemy.orm import joinedload
from sqlalchemy.orm import Session

from onyx.connectors.models import Document
from onyx.connectors.models import DocumentErrorSummary
from onyx.db.models import IndexAttempt
from onyx.db.models import IndexAttemptError
from onyx.db.models import IndexingStatus
from onyx.db.models import IndexModelStatus
from onyx.db.models import SearchSettings
from onyx.server.documents.models import ConnectorCredentialPair
from onyx.server.documents.models import ConnectorCredentialPairIdentifier
from onyx.utils.logger import setup_logger

logger = setup_logger()

"""
获取指定连接器凭证对的最后一次索引尝试记录
参数:
    cc_pair_id: 连接器凭证对ID
    search_settings_id: 搜索设置ID
    db_session: 数据库会话
返回:
    最后一次索引尝试记录或None
"""
def get_last_attempt_for_cc_pair(
    cc_pair_id: int,
    search_settings_id: int,
    db_session: Session,
) -> IndexAttempt | None:
    return (
        db_session.query(IndexAttempt)
        .filter(
            IndexAttempt.connector_credential_pair_id == cc_pair_id,
            IndexAttempt.search_settings_id == search_settings_id,
        )
        .order_by(IndexAttempt.time_updated.desc())
        .first()
    )

"""
根据索引尝试ID获取索引尝试记录
参数:
    db_session: 数据库会话
    index_attempt_id: 索引尝试ID
返回:
    索引尝试记录或None
"""
def get_index_attempt(
    db_session: Session, index_attempt_id: int
) -> IndexAttempt | None:
    stmt = select(IndexAttempt).where(IndexAttempt.id == index_attempt_id)
    return db_session.scalars(stmt).first()

"""
创建新的索引尝试记录
参数:
    connector_credential_pair_id: 连接器凭证对ID
    search_settings_id: 搜索设置ID
    db_session: 数据库会话
    from_beginning: 是否从头开始索引
返回:
    新创建的索引尝试ID
"""
def create_index_attempt(
    connector_credential_pair_id: int,
    search_settings_id: int,
    db_session: Session,
    from_beginning: bool = False,
) -> int:
    new_attempt = IndexAttempt(
        connector_credential_pair_id=connector_credential_pair_id,
        search_settings_id=search_settings_id,
        from_beginning=from_beginning,
        status=IndexingStatus.NOT_STARTED,
    )
    db_session.add(new_attempt)
    db_session.commit()

    return new_attempt.id

"""
删除指定的索引尝试记录
参数:
    db_session: 数据库会话
    index_attempt_id: 要删除的索引尝试ID
"""
def delete_index_attempt(db_session: Session, index_attempt_id: int) -> None:
    index_attempt = get_index_attempt(db_session, index_attempt_id)
    if index_attempt:
        db_session.delete(index_attempt)
        db_session.commit()

"""
模拟创建成功的索引尝试记录
参数:
    connector_credential_pair_id: 连接器凭证对ID
    search_settings_id: 搜索设置ID
    docs_indexed: 已索引的文档数量
    db_session: 数据库会话
返回:
    新创建的索引尝试ID
注意: 不应该在用户触发的流程中使用 / Should not be used in any user triggered flows
"""
def mock_successful_index_attempt(
    connector_credential_pair_id: int,
    search_settings_id: int,
    docs_indexed: int,
    db_session: Session,
) -> int:
    """Should not be used in any user triggered flows"""
    db_time = func.now()
    new_attempt = IndexAttempt(
        connector_credential_pair_id=connector_credential_pair_id,
        search_settings_id=search_settings_id,
        from_beginning=True,
        status=IndexingStatus.SUCCESS,
        total_docs_indexed=docs_indexed,
        new_docs_indexed=docs_indexed,
        # Need this to be some convincing random looking value and it can't be 0
        # or the indexing rate would calculate out to infinity
        time_started=db_time - timedelta(seconds=1.92),
        time_updated=db_time,
    )
    db_session.add(new_attempt)
    db_session.commit()

    return new_attempt.id

"""
获取进行中的索引尝试记录
参数:
    connector_id: 连接器ID
    db_session: 数据库会话
返回:
    进行中的索引尝试记录列表
"""
def get_in_progress_index_attempts(
    connector_id: int | None,
    db_session: Session,
) -> list[IndexAttempt]:
    stmt = select(IndexAttempt)
    if connector_id is not None:
        stmt = stmt.where(
            IndexAttempt.connector_credential_pair.has(connector_id=connector_id)
        )
    stmt = stmt.where(IndexAttempt.status == IndexingStatus.IN_PROGRESS)

    incomplete_attempts = db_session.scalars(stmt)
    return list(incomplete_attempts.all())

"""
根据状态获取所有索引尝试记录
参数:
    status: 索引状态
    db_session: 数据库会话
返回:
    指定状态的索引尝试记录列表
"""
def get_all_index_attempts_by_status(
    status: IndexingStatus, db_session: Session
) -> list[IndexAttempt]:
    """
    这个函数会急切加载连接器和凭证,这样在运行长期索引任务时可以过期db_session,
    避免内存使用量不断增加。
    
    结果按照time_created排序(从最早到最新)
    
    原文:
    This eagerly loads the connector and credential so that the db_session can be expired
    before running long-living indexing jobs, which causes increasing memory usage.
    Results are ordered by time_created (oldest to newest).
    """
    stmt = select(IndexAttempt)
    stmt = stmt.where(IndexAttempt.status == status)
    stmt = stmt.order_by(IndexAttempt.time_created)
    stmt = stmt.options(
        joinedload(IndexAttempt.connector_credential_pair).joinedload(
            ConnectorCredentialPair.connector
        ),
        joinedload(IndexAttempt.connector_credential_pair).joinedload(
            ConnectorCredentialPair.credential
        ),
    )
    new_attempts = db_session.scalars(stmt)
    return list(new_attempts.all())

"""
将索引尝试记录状态转换为进行中
参数:
    index_attempt_id: 索引尝试ID
    db_session: 数据库会话
返回:
    更新后的索引尝试记录
"""
def transition_attempt_to_in_progress(
    index_attempt_id: int,
    db_session: Session,
) -> IndexAttempt:
    """在尝试更新时锁定行 / Locks the row when we try to update"""
    try:
        attempt = db_session.execute(
            select(IndexAttempt)
            .where(IndexAttempt.id == index_attempt_id)
            .with_for_update()
        ).scalar_one()

        if attempt is None:
            raise RuntimeError(
                f"Unable to find IndexAttempt for ID '{index_attempt_id}'"
            )

        if attempt.status != IndexingStatus.NOT_STARTED:
            raise RuntimeError(
                f"Indexing attempt with ID '{index_attempt_id}' is not in NOT_STARTED status. "
                f"Current status is '{attempt.status}'."
            )

        attempt.status = IndexingStatus.IN_PROGRESS
        attempt.time_started = attempt.time_started or func.now()  # type: ignore
        db_session.commit()
        return attempt
    except Exception:
        db_session.rollback()
        logger.exception("transition_attempt_to_in_progress exceptioned.")
        raise

"""
将索引尝试记录标记为进行中
参数:
    index_attempt: 索引尝试记录
    db_session: 数据库会话
"""
def mark_attempt_in_progress(
    index_attempt: IndexAttempt,
    db_session: Session,
) -> None:
    try:
        attempt = db_session.execute(
            select(IndexAttempt)
            .where(IndexAttempt.id == index_attempt.id)
            .with_for_update()
        ).scalar_one()

        attempt.status = IndexingStatus.IN_PROGRESS
        attempt.time_started = index_attempt.time_started or func.now()  # type: ignore
        db_session.commit()
    except Exception:
        db_session.rollback()
        raise

"""
将索引尝试记录标记为成功
参数:
    index_attempt: 索引尝试记录
    db_session: 数据库会话
"""
def mark_attempt_succeeded(
    index_attempt: IndexAttempt,
    db_session: Session,
) -> None:
    try:
        attempt = db_session.execute(
            select(IndexAttempt)
            .where(IndexAttempt.id == index_attempt.id)
            .with_for_update()
        ).scalar_one()

        attempt.status = IndexingStatus.SUCCESS
        db_session.commit()
    except Exception:
        db_session.rollback()
        raise

"""
将索引尝试记录标记为部分成功
参数:
    index_attempt: 索引尝试记录
    db_session: 数据库会话
"""
def mark_attempt_partially_succeeded(
    index_attempt: IndexAttempt,
    db_session: Session,
) -> None:
    try:
        attempt = db_session.execute(
            select(IndexAttempt)
            .where(IndexAttempt.id == index_attempt.id)
            .with_for_update()
        ).scalar_one()

        attempt.status = IndexingStatus.COMPLETED_WITH_ERRORS
        db_session.commit()
    except Exception:
        db_session.rollback()
        raise

"""
将索引尝试记录标记为已取消
参数:
    index_attempt_id: 索引尝试ID
    db_session: 数据库会话
    reason: 取消原因
"""
def mark_attempt_canceled(
    index_attempt_id: int,
    db_session: Session,
    reason: str = "Unknown",
) -> None:
    try:
        attempt = db_session.execute(
            select(IndexAttempt)
            .where(IndexAttempt.id == index_attempt_id)
            .with_for_update()
        ).scalar_one()

        if not attempt.time_started:
            attempt.time_started = datetime.now(timezone.utc)
        attempt.status = IndexingStatus.CANCELED
        attempt.error_msg = reason
        db_session.commit()
    except Exception:
        db_session.rollback()
        raise

"""
将索引尝试记录标记为失败
参数:
    index_attempt_id: 索引尝试ID
    db_session: 数据库会话
    failure_reason: 失败原因
    full_exception_trace: 完整的异常追踪信息
"""
def mark_attempt_failed(
    index_attempt_id: int,
    db_session: Session,
    failure_reason: str = "Unknown",
    full_exception_trace: str | None = None,
) -> None:
    try:
        attempt = db_session.execute(
            select(IndexAttempt)
            .where(IndexAttempt.id == index_attempt_id)
            .with_for_update()
        ).scalar_one()

        if not attempt.time_started:
            attempt.time_started = datetime.now(timezone.utc)
        attempt.status = IndexingStatus.FAILED
        attempt.error_msg = failure_reason
        attempt.full_exception_trace = full_exception_trace
        db_session.commit()
    except Exception:
        db_session.rollback()
        raise

"""
更新索引尝试记录的文档索引信息
参数:
    db_session: 数据库会话
    index_attempt: 索引尝试记录
    total_docs_indexed: 总共索引的文档数量
    new_docs_indexed: 新索引的文档数量
    docs_removed_from_index: 从索引中移除的文档数量
"""
def update_docs_indexed(
    db_session: Session,
    index_attempt: IndexAttempt,
    total_docs_indexed: int,
    new_docs_indexed: int,
    docs_removed_from_index: int,
) -> None:
    index_attempt.total_docs_indexed = total_docs_indexed
    index_attempt.new_docs_indexed = new_docs_indexed
    index_attempt.docs_removed_from_index = docs_removed_from_index

    db_session.add(index_attempt)
    db_session.commit()

"""
获取指定连接器和凭证的最后一次索引尝试记录
参数:
    connector_id: 连接器ID
    credential_id: 凭证ID
    search_settings_id: 搜索设置ID
    db_session: 数据库会话
返回:
    最后一次索引尝试记录或None
"""
def get_last_attempt(
    connector_id: int,
    credential_id: int,
    search_settings_id: int | None,
    db_session: Session,
) -> IndexAttempt | None:
    stmt = (
        select(IndexAttempt)
        .join(ConnectorCredentialPair)
        .where(
            ConnectorCredentialPair.connector_id == connector_id,
            ConnectorCredentialPair.credential_id == credential_id,
            IndexAttempt.search_settings_id == search_settings_id,
        )
    )

    # Note, the below is using time_created instead of time_updated
    stmt = stmt.order_by(desc(IndexAttempt.time_created))

    return db_session.execute(stmt).scalars().first()

"""
根据状态获取最新的索引尝试记录
参数:
    secondary_index: 是否为次要索引
    db_session: 数据库会话
    status: 索引状态
返回:
    最新的索引尝试记录列表
"""
def get_latest_index_attempts_by_status(
    secondary_index: bool,
    db_session: Session,
    status: IndexingStatus,
) -> Sequence[IndexAttempt]:
    """
    获取每个connector_credential_pair指定状态的最新索引尝试。
    基于secondary_index标志过滤尝试,以获取future或present的索引尝试。
    为每个唯一的connector_credential_pair返回一个IndexAttempt对象序列。
    
    原文:
    Retrieves the most recent index attempt with the specified status for each connector_credential_pair.
    Filters attempts based on the secondary_index flag to get either future or present index attempts.
    Returns a sequence of IndexAttempt objects, one for each unique connector_credential_pair.
    """
    latest_failed_attempts = (
        select(
            IndexAttempt.connector_credential_pair_id,
            func.max(IndexAttempt.id).label("max_failed_id"),
        )
        .join(SearchSettings, IndexAttempt.search_settings_id == SearchSettings.id)
        .where(
            SearchSettings.status
            == (
                IndexModelStatus.FUTURE if secondary_index else IndexModelStatus.PRESENT
            ),
            IndexAttempt.status == status,
        )
        .group_by(IndexAttempt.connector_credential_pair_id)
        .subquery()
    )

    stmt = select(IndexAttempt).join(
        latest_failed_attempts,
        (
            IndexAttempt.connector_credential_pair_id
            == latest_failed_attempts.c.connector_credential_pair_id
        )
        & (IndexAttempt.id == latest_failed_attempts.c.max_failed_id),
    )

    return db_session.execute(stmt).scalars().all()

"""
获取最新的索引尝试记录
参数:
    secondary_index: 是否为次要索引
    db_session: 数据库会话
返回:
    最新的索引尝试记录列表
"""
def get_latest_index_attempts(
    secondary_index: bool,
    db_session: Session,
) -> Sequence[IndexAttempt]:
    ids_stmt = select(
        IndexAttempt.connector_credential_pair_id,
        func.max(IndexAttempt.id).label("max_id"),
    ).join(SearchSettings, IndexAttempt.search_settings_id == SearchSettings.id)

    if secondary_index:
        ids_stmt = ids_stmt.where(SearchSettings.status == IndexModelStatus.FUTURE)
    else:
        ids_stmt = ids_stmt.where(SearchSettings.status == IndexModelStatus.PRESENT)

    ids_stmt = ids_stmt.group_by(IndexAttempt.connector_credential_pair_id)
    ids_subquery = ids_stmt.subquery()

    stmt = (
        select(IndexAttempt)
        .join(
            ids_subquery,
            IndexAttempt.connector_credential_pair_id
            == ids_subquery.c.connector_credential_pair_id,
        )
        .where(IndexAttempt.id == ids_subquery.c.max_id)
    )

    return db_session.execute(stmt).scalars().all()

"""
统计指定连接器的索引尝试记录数量
参数:
    db_session: 数据库会话
    connector_id: 连接器ID
    only_current: 是否仅统计当前索引
    disinclude_finished: 是否排除已完成的索引
返回:
    索引尝试记录数量
"""
def count_index_attempts_for_connector(
    db_session: Session,
    connector_id: int,
    only_current: bool = True,
    disinclude_finished: bool = False,
) -> int:
    stmt = (
        select(IndexAttempt)
        .join(ConnectorCredentialPair)
        .where(ConnectorCredentialPair.connector_id == connector_id)
    )
    if disinclude_finished:
        stmt = stmt.where(
            IndexAttempt.status.in_(
                [IndexingStatus.NOT_STARTED, IndexingStatus.IN_PROGRESS]
            )
        )
    if only_current:
        stmt = stmt.join(SearchSettings).where(
            SearchSettings.status == IndexModelStatus.PRESENT
        )
    # Count total items for pagination
    count_stmt = stmt.with_only_columns(func.count()).order_by(None)
    total_count = db_session.execute(count_stmt).scalar_one()
    return total_count

"""
获取指定连接器的分页索引尝试记录
参数:
    db_session: 数据库会话
    connector_id: 连接器ID
    page: 页码
    page_size: 每页大小
    only_current: 是否仅获取当前索引
    disinclude_finished: 是否排除已完成的索引
返回:
    分页的索引尝试记录列表
"""
def get_paginated_index_attempts_for_cc_pair_id(
    db_session: Session,
    connector_id: int,
    page: int,
    page_size: int,
    only_current: bool = True,
    disinclude_finished: bool = False,
) -> list[IndexAttempt]:
    stmt = (
        select(IndexAttempt)
        .join(ConnectorCredentialPair)
        .where(ConnectorCredentialPair.connector_id == connector_id)
    )
    if disinclude_finished:
        stmt = stmt.where(
            IndexAttempt.status.in_(
                [IndexingStatus.NOT_STARTED, IndexingStatus.IN_PROGRESS]
            )
        )
    if only_current:
        stmt = stmt.join(SearchSettings).where(
            SearchSettings.status == IndexModelStatus.PRESENT
        )

    stmt = stmt.order_by(IndexAttempt.time_started.desc())

    # Apply pagination
    stmt = stmt.offset((page - 1) * page_size).limit(page_size)

    return list(db_session.execute(stmt).scalars().all())

"""
获取指定连接器凭证对的最新索引尝试记录
参数:
    db_session: 数据库会话
    connector_credential_pair_id: 连接器凭证对ID
    secondary_index: 是否为次要索引
    only_finished: 是否仅获取已完成的索引
返回:
    最新的索引尝试记录或None
"""
def get_latest_index_attempt_for_cc_pair_id(
    db_session: Session,
    connector_credential_pair_id: int,
    secondary_index: bool,
    only_finished: bool = True,
) -> IndexAttempt | None:
    stmt = select(IndexAttempt)
    stmt = stmt.where(
        IndexAttempt.connector_credential_pair_id == connector_credential_pair_id,
    )
    if only_finished:
        stmt = stmt.where(
            IndexAttempt.status.not_in(
                [IndexingStatus.NOT_STARTED, IndexingStatus.IN_PROGRESS]
            ),
        )
    if secondary_index:
        stmt = stmt.join(SearchSettings).where(
            SearchSettings.status == IndexModelStatus.FUTURE
        )
    else:
        stmt = stmt.join(SearchSettings).where(
            SearchSettings.status == IndexModelStatus.PRESENT
        )
    stmt = stmt.order_by(desc(IndexAttempt.time_created))
    stmt = stmt.limit(1)
    return db_session.execute(stmt).scalar_one_or_none()

"""
获取指定连接器凭证对的索引尝试记录
参数:
    db_session: 数据库会话
    cc_pair_identifier: 连接器凭证对标识符
    only_current: 是否仅获取当前索引
    disinclude_finished: 是否排除已完成的索引
返回:
    索引尝试记录列表
"""
def get_index_attempts_for_cc_pair(
    db_session: Session,
    cc_pair_identifier: ConnectorCredentialPairIdentifier,
    only_current: bool = True,
    disinclude_finished: bool = False,
) -> Sequence[IndexAttempt]:
    stmt = (
        select(IndexAttempt)
        .join(ConnectorCredentialPair)
        .where(
            and_(
                ConnectorCredentialPair.connector_id == cc_pair_identifier.connector_id,
                ConnectorCredentialPair.credential_id
                == cc_pair_identifier.credential_id,
            )
        )
    )
    if disinclude_finished:
        stmt = stmt.where(
            IndexAttempt.status.in_(
                [IndexingStatus.NOT_STARTED, IndexingStatus.IN_PROGRESS]
            )
        )
    if only_current:
        stmt = stmt.join(SearchSettings).where(
            SearchSettings.status == IndexModelStatus.PRESENT
        )

    stmt = stmt.order_by(IndexAttempt.time_created.desc())
    return db_session.execute(stmt).scalars().all()

"""
删除指定连接器凭证对的索引尝试记录
参数:
    cc_pair_id: 连接器凭证对ID
    db_session: 数据库会话
"""
def delete_index_attempts(
    cc_pair_id: int,
    db_session: Session,
) -> None:
    # First, delete related entries in IndexAttemptErrors
    stmt_errors = delete(IndexAttemptError).where(
        IndexAttemptError.index_attempt_id.in_(
            select(IndexAttempt.id).where(
                IndexAttempt.connector_credential_pair_id == cc_pair_id
            )
        )
    )
    db_session.execute(stmt_errors)

    stmt = delete(IndexAttempt).where(
        IndexAttempt.connector_credential_pair_id == cc_pair_id,
    )

    db_session.execute(stmt)

"""
使指定搜索设置ID的索引尝试记录失效
参数:
    search_settings_id: 搜索设置ID
    db_session: 数据库会话
"""
def expire_index_attempts(
    search_settings_id: int,
    db_session: Session,
) -> None:
    not_started_query = (
        update(IndexAttempt)
        .where(IndexAttempt.search_settings_id == search_settings_id)
        .where(IndexAttempt.status == IndexingStatus.NOT_STARTED)
        .values(
            status=IndexingStatus.CANCELED,
            error_msg="Canceled, likely due to model swap",
        )
    )
    db_session.execute(not_started_query)

    update_query = (
        update(IndexAttempt)
        .where(IndexAttempt.search_settings_id == search_settings_id)
        .where(IndexAttempt.status != IndexingStatus.SUCCESS)
        .values(
            status=IndexingStatus.FAILED,
            error_msg="Canceled due to embedding model swap",
        )
    )
    db_session.execute(update_query)

    db_session.commit()

"""
取消指定连接器凭证对的索引尝试记录
参数:
    cc_pair_id: 连接器凭证对ID
    db_session: 数据库会话
    include_secondary_index: 是否包括次要索引
"""
def cancel_indexing_attempts_for_ccpair(
    cc_pair_id: int,
    db_session: Session,
    include_secondary_index: bool = False,
) -> None:
    stmt = (
        update(IndexAttempt)
        .where(IndexAttempt.connector_credential_pair_id == cc_pair_id)
        .where(IndexAttempt.status == IndexingStatus.NOT_STARTED)
        .values(
            status=IndexingStatus.CANCELED,
            error_msg="Canceled by user",
            time_started=datetime.now(timezone.utc),
        )
    )

    if not include_secondary_index:
        subquery = select(SearchSettings.id).where(
            SearchSettings.status != IndexModelStatus.FUTURE
        )
        stmt = stmt.where(IndexAttempt.search_settings_id.in_(subquery))

    db_session.execute(stmt)

"""
取消所有过去模型的索引尝试记录
参数:
    db_session: 数据库会话
"""
def cancel_indexing_attempts_past_model(
    db_session: Session,
) -> None:
    """
    停止所有正在进行或未开始的过期嵌入模型的索引尝试
    
    原文:
    Stops all indexing attempts that are in progress or not started for
    any embedding model that not present/future
    """
    db_session.execute(
        update(IndexAttempt)
        .where(
            IndexAttempt.status.in_(
                [IndexingStatus.IN_PROGRESS, IndexingStatus.NOT_STARTED]
            ),
            IndexAttempt.search_settings_id == SearchSettings.id,
            SearchSettings.status == IndexModelStatus.PAST,
        )
        .values(status=IndexingStatus.FAILED)
    )

"""
统计具有成功索引尝试记录的唯一连接器凭证对数量
参数:
    search_settings_id: 搜索设置ID
    db_session: 数据库会话
返回:
    唯一连接器凭证对数量
"""
def count_unique_cc_pairs_with_successful_index_attempts(
    search_settings_id: int | None,
    db_session: Session,
) -> int:
    """
    收集指定嵌入模型的所有成功索引尝试
    然后按connector_id和credential_id去重(等同于cc-pair)
    最后统计具有成功尝试的唯一cc-pairs的总数
    
    原文:
    Collect all of the Index Attempts that are successful and for the specified embedding model
    Then do distinct by connector_id and credential_id which is equivalent to the cc-pair. Finally,
    do a count to get the total number of unique cc-pairs with successful attempts
    """
    unique_pairs_count = (
        db_session.query(IndexAttempt.connector_credential_pair_id)
        .join(ConnectorCredentialPair)
        .filter(
            IndexAttempt.search_settings_id == search_settings_id,
            IndexAttempt.status == IndexingStatus.SUCCESS,
        )
        .distinct()
        .count()
    )

    return unique_pairs_count

"""
创建索引尝试错误记录
参数:
    index_attempt_id: 索引尝试ID
    batch: 批次号
    docs: 文档列表
    exception_msg: 异常信息
    exception_traceback: 异常追踪信息
    db_session: 数据库会话
返回:
    新创建的索引尝试错误ID
"""
def create_index_attempt_error(
    index_attempt_id: int | None,
    batch: int | None,
    docs: list[Document],
    exception_msg: str,
    exception_traceback: str,
    db_session: Session,
) -> int:
    doc_summaries = []
    for doc in docs:
        doc_summary = DocumentErrorSummary.from_document(doc)
        doc_summaries.append(doc_summary.to_dict())

    new_error = IndexAttemptError(
        index_attempt_id=index_attempt_id,
        batch=batch,
        doc_summaries=doc_summaries,
        error_msg=exception_msg,
        traceback=exception_traceback,
    )
    db_session.add(new_error)
    db_session.commit()

    return new_error.id

"""
获取指定索引尝试的错误记录
参数:
    index_attempt_id: 索引尝试ID
    db_session: 数据库会话
返回:
    错误记录列表
"""
def get_index_attempt_errors(
    index_attempt_id: int,
    db_session: Session,
) -> list[IndexAttemptError]:
    stmt = select(IndexAttemptError).where(
        IndexAttemptError.index_attempt_id == index_attempt_id
    )

    errors = db_session.scalars(stmt)
    return list(errors.all())
