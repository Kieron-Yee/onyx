"""
这个文件主要用于处理文档相关的数据库操作，包括:
- 文档的增删改查
- 文档与连接器/凭证对的关系管理
- 文档的同步状态管理
- 文档的访问控制
"""

import contextlib
import time
from collections.abc import Generator
from collections.abc import Sequence
from datetime import datetime
from datetime import timezone

from sqlalchemy import and_
from sqlalchemy import delete
from sqlalchemy import exists
from sqlalchemy import func
from sqlalchemy import or_
from sqlalchemy import Select
from sqlalchemy import select
from sqlalchemy import tuple_
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine.util import TransactionalContext
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session
from sqlalchemy.sql.expression import null

from onyx.configs.constants import DEFAULT_BOOST
from onyx.db.connector_credential_pair import get_connector_credential_pair_from_id
from onyx.db.enums import AccessType
from onyx.db.enums import ConnectorCredentialPairStatus
from onyx.db.feedback import delete_document_feedback_for_documents__no_commit
from onyx.db.models import ConnectorCredentialPair
from onyx.db.models import Credential
from onyx.db.models import Document as DbDocument
from onyx.db.models import DocumentByConnectorCredentialPair
from onyx.db.models import User
from onyx.db.tag import delete_document_tags_for_documents__no_commit
from onyx.db.utils import model_to_dict
from onyx.document_index.interfaces import DocumentMetadata
from onyx.server.documents.models import ConnectorCredentialPairIdentifier
from onyx.utils.logger import setup_logger

logger = setup_logger()


def check_docs_exist(db_session: Session) -> bool:
    """
    检查数据库中是否存在任何文档
    
    Args:
        db_session: 数据库会话对象
    Returns:
        bool: 存在文档返回True，否则返回False
    """
    stmt = select(exists(DbDocument))
    result = db_session.execute(stmt)
    return result.scalar() or False


def count_documents_by_needs_sync(session: Session) -> int:
    """
    获取需要同步的文档数量
    
    统计满足以下条件的文档数量:
    1. last_modified比last_synced新
    2. last_synced为空(从未同步过)
    且文档与连接器/凭证对有关联关系
    
    原注释:
    Get the count of all documents where:
    1. last_modified is newer than last_synced 
    2. last_synced is null (meaning we've never synced)
    AND the document has a relationship with a connector/credential pair
    
    中文翻译:
    获取所有满足以下条件的文档数量:
    1. last_modified 比 last_synced 新
    2. last_synced 为空(表示从未同步)
    且文档与连接器/凭证对有关联关系
    
    TODO: 没有与连接器/凭证对关系的文档最终应该被清理掉
    """
    count = (
        session.query(func.count(DbDocument.id.distinct()))
        .select_from(DbDocument)
        .join(
            DocumentByConnectorCredentialPair,
            DbDocument.id == DocumentByConnectorCredentialPair.id,
        )
        .filter(
            or_(
                DbDocument.last_modified > DbDocument.last_synced,
                DbDocument.last_synced.is_(None),
            )
        )
        .scalar()
    )

    return count


def construct_document_select_for_connector_credential_pair_by_needs_sync(
    connector_id: int, credential_id: int
) -> Select:
    """
    构建SQL查询以获取指定连接器和凭证对应的需要同步的文档
    
    Args:
        connector_id: 连接器ID
        credential_id: 凭证ID
    Returns:
        Select: SQL查询对象 
    """
    initial_doc_ids_stmt = select(DocumentByConnectorCredentialPair.id).where(
        and_(
            DocumentByConnectorCredentialPair.connector_id == connector_id,
            DocumentByConnectorCredentialPair.credential_id == credential_id,
        )
    )

    stmt = (
        select(DbDocument)
        .where(
            DbDocument.id.in_(initial_doc_ids_stmt),
            or_(
                DbDocument.last_modified
                > DbDocument.last_synced,  # last_modified is newer than last_synced
                DbDocument.last_synced.is_(None),  # never synced
            ),
        )
        .distinct()
    )

    return stmt


def get_all_documents_needing_vespa_sync_for_cc_pair(
    db_session: Session, cc_pair_id: int
) -> list[DbDocument]:
    """
    获取指定连接器/凭证对中需要与Vespa同步的所有文档
    
    Args:
        db_session: 数据库会话对象
        cc_pair_id: 连接器/凭证对ID
    Returns:
        list[DbDocument]: 需要同步的文档列表
    """
    cc_pair = get_connector_credential_pair_from_id(
        cc_pair_id=cc_pair_id, db_session=db_session
    )
    if not cc_pair:
        raise ValueError(f"No CC pair found with ID: {cc_pair_id}")

    stmt = construct_document_select_for_connector_credential_pair_by_needs_sync(
        cc_pair.connector_id, cc_pair.credential_id
    )

    return list(db_session.scalars(stmt).all())


def construct_document_select_for_connector_credential_pair(
    connector_id: int, credential_id: int | None = None
) -> Select:
    """
    构建SQL查询以获取指定连接器和凭证对应的文档
    
    Args:
        connector_id: 连接器ID
        credential_id: 凭证ID
    Returns:
        Select: SQL查询对象 
    """
    initial_doc_ids_stmt = select(DocumentByConnectorCredentialPair.id).where(
        and_(
            DocumentByConnectorCredentialPair.connector_id == connector_id,
            DocumentByConnectorCredentialPair.credential_id == credential_id,
        )
    )
    stmt = select(DbDocument).where(DbDocument.id.in_(initial_doc_ids_stmt)).distinct()
    return stmt


def get_documents_for_cc_pair(
    db_session: Session,
    cc_pair_id: int,
) -> list[DbDocument]:
    """
    获取指定连接器/凭证对的所有文档
    
    Args:
        db_session: 数据库会话对象
        cc_pair_id: 连接器/凭证对ID
    Returns:
        list[DbDocument]: 文档列表
    """
    cc_pair = get_connector_credential_pair_from_id(
        cc_pair_id=cc_pair_id, db_session=db_session
    )
    if not cc_pair:
        raise ValueError(f"No CC pair found with ID: {cc_pair_id}")
    stmt = construct_document_select_for_connector_credential_pair(
        connector_id=cc_pair.connector_id, credential_id=cc_pair.credential_id
    )
    return list(db_session.scalars(stmt).all())


def get_document_ids_for_connector_credential_pair(
    db_session: Session, connector_id: int, credential_id: int, limit: int | None = None
) -> list[str]:
    """
    获取指定连接器/凭证对的文档ID列表
    
    Args:
        db_session: 数据库会话对象
        connector_id: 连接器ID
        credential_id: 凭证ID
        limit: 限制返回的文档数量
    Returns:
        list[str]: 文档ID列表
    """
    doc_ids_stmt = select(DocumentByConnectorCredentialPair.id).where(
        and_(
            DocumentByConnectorCredentialPair.connector_id == connector_id,
            DocumentByConnectorCredentialPair.credential_id == credential_id,
        )
    )
    return list(db_session.execute(doc_ids_stmt).scalars().all())


def get_documents_for_connector_credential_pair(
    db_session: Session, connector_id: int, credential_id: int, limit: int | None = None
) -> Sequence[DbDocument]:
    """
    获取指定连接器/凭证对的文档列表
    
    Args:
        db_session: 数据库会话对象
        connector_id: 连接器ID
        credential_id: 凭证ID
        limit: 限制返回的文档数量
    Returns:
        Sequence[DbDocument]: 文档列表
        
    原注释:
    TODO: The documents without a relationship with a connector/credential pair
    should be cleaned up somehow eventually.
    
    中文翻译:
    TODO: 没有与连接器/凭证对关系的文档最终应该以某种方式清理掉
    """
    initial_doc_ids_stmt = select(DocumentByConnectorCredentialPair.id).where(
        and_(
            DocumentByConnectorCredentialPair.connector_id == connector_id,
            DocumentByConnectorCredentialPair.credential_id == credential_id,
        )
    )
    stmt = select(DbDocument).where(DbDocument.id.in_(initial_doc_ids_stmt)).distinct()
    if limit:
        stmt = stmt.limit(limit)
    return db_session.scalars(stmt).all()


def get_documents_by_ids(
    db_session: Session,
    document_ids: list[str],
) -> list[DbDocument]:
    """
    根据文档ID列表获取文档
    
    Args:
        db_session: 数据库会话对象
        document_ids: 文档ID列表
    Returns:
        list[DbDocument]: 文档列表
    """
    stmt = select(DbDocument).where(DbDocument.id.in_(document_ids))
    documents = db_session.execute(stmt).scalars().all()
    return list(documents)


def get_document_connector_count(
    db_session: Session,
    document_id: str,
) -> int:
    """
    获取指定文档的连接器数量
    
    Args:
        db_session: 数据库会话对象
        document_id: 文档ID
    Returns:
        int: 连接器数量
    """
    results = get_document_connector_counts(db_session, [document_id])
    if not results or len(results) == 0:
        return 0

    return results[0][1]


def get_document_connector_counts(
    db_session: Session,
    document_ids: list[str],
) -> Sequence[tuple[str, int]]:
    """
    获取指定文档列表的连接器数量
    
    Args:
        db_session: 数据库会话对象
        document_ids: 文档ID列表
    Returns:
        Sequence[tuple[str, int]]: 每个文档的连接器数量
    """
    stmt = (
        select(
            DocumentByConnectorCredentialPair.id,
            func.count(),
        )
        .where(DocumentByConnectorCredentialPair.id.in_(document_ids))
        .group_by(DocumentByConnectorCredentialPair.id)
    )
    return db_session.execute(stmt).all()  # type: ignore


def get_document_counts_for_cc_pairs(
    db_session: Session, cc_pair_identifiers: list[ConnectorCredentialPairIdentifier]
) -> Sequence[tuple[int, int, int]]:
    """
    获取指定连接器/凭证对的文档数量
    
    Args:
        db_session: 数据库会话对象
        cc_pair_identifiers: 连接器/凭证对标识符列表
    Returns:
        Sequence[tuple[int, int, int]]: 每个连接器/凭证对的文档数量
    """
    # Prepare a list of (connector_id, credential_id) tuples
    cc_ids = [(x.connector_id, x.credential_id) for x in cc_pair_identifiers]

    stmt = (
        select(
            DocumentByConnectorCredentialPair.connector_id,
            DocumentByConnectorCredentialPair.credential_id,
            func.count(),
        )
        .where(
            tuple_(
                DocumentByConnectorCredentialPair.connector_id,
                DocumentByConnectorCredentialPair.credential_id,
            ).in_(cc_ids)
        )
        .group_by(
            DocumentByConnectorCredentialPair.connector_id,
            DocumentByConnectorCredentialPair.credential_id,
        )
    )

    return db_session.execute(stmt).all()  # type: ignore


def get_access_info_for_document(
    db_session: Session,
    document_id: str,
) -> tuple[str, list[str | None], bool] | None:
    """
    获取单个文档的访问信息
    
    通过调用get_access_info_for_documents函数并传递一个包含单个文档ID的列表来获取访问信息。
    
    Args:
        db_session: 数据库会话对象
        document_id: 文档ID
    Returns:
        Optional[Tuple[str, List[str | None], bool]]: 包含文档ID、用户邮箱列表和文档是否公开的元组，或None
    """
    results = get_access_info_for_documents(db_session, [document_id])
    if not results:
        return None

    return results[0]


def get_access_info_for_documents(
    db_session: Session,
    document_ids: list[str],
) -> Sequence[tuple[str, list[str | None], bool]]:
    """
    获取指定文档的访问信息
    
    获取给定文档的所有相关访问信息，包括:
    - 与文档关联的cc对的用户ID
    - 任何关联的cc对是否打算使文档公开
    
    Args:
        db_session: 数据库会话对象
        document_ids: 文档ID列表
    Returns:
        Sequence[tuple[str, list[str | None], bool]]: 每个文档的访问信息
    """
    stmt = select(
        DocumentByConnectorCredentialPair.id,
        func.array_agg(func.coalesce(User.email, null())).label("user_emails"),
        func.bool_or(ConnectorCredentialPair.access_type == AccessType.PUBLIC).label(
            "public_doc"
        ),
    ).where(DocumentByConnectorCredentialPair.id.in_(document_ids))

    stmt = (
        stmt.join(
            Credential,
            DocumentByConnectorCredentialPair.credential_id == Credential.id,
        )
        .join(
            ConnectorCredentialPair,
            and_(
                DocumentByConnectorCredentialPair.connector_id
                == ConnectorCredentialPair.connector_id,
                DocumentByConnectorCredentialPair.credential_id
                == ConnectorCredentialPair.credential_id,
            ),
        )
        .outerjoin(
            User,
            and_(
                Credential.user_id == User.id,
                ConnectorCredentialPair.access_type != AccessType.SYNC,
            ),
        )
        # don't include CC pairs that are being deleted
        # NOTE: CC pairs can never go from DELETING to any other state -> it's safe to ignore them
        .where(ConnectorCredentialPair.status != ConnectorCredentialPairStatus.DELETING)
        .group_by(DocumentByConnectorCredentialPair.id)
    )
    return db_session.execute(stmt).all()  # type: ignore


def upsert_documents(
    db_session: Session,
    document_metadata_batch: list[DocumentMetadata],
    initial_boost: int = DEFAULT_BOOST,
) -> None:
    """
    插入或更新文档
    
    注意: 这个函数是Postgres特定的。并非所有数据库都支持ON CONFLICT子句。
    此外，注意这个函数不应该用于更新文档，只用于创建和确保文档存在。它忽略doc_updated_at字段。
    
    Args:
        db_session: 数据库会话对象
        document_metadata_batch: 文档元数据列表
        initial_boost: 初始boost值
    """
    seen_documents: dict[str, DocumentMetadata] = {}
    for document_metadata in document_metadata_batch:
        doc_id = document_metadata.document_id
        if doc_id not in seen_documents:
            seen_documents[doc_id] = document_metadata

    if not seen_documents:
        logger.info("No documents to upsert. Skipping.")
        return

    insert_stmt = insert(DbDocument).values(
        [
            model_to_dict(
                DbDocument(
                    id=doc.document_id,
                    from_ingestion_api=doc.from_ingestion_api,
                    boost=initial_boost,
                    hidden=False,
                    semantic_id=doc.semantic_identifier,
                    link=doc.first_link,
                    doc_updated_at=None,  # this is intentional
                    last_modified=datetime.now(timezone.utc),
                    primary_owners=doc.primary_owners,
                    secondary_owners=doc.secondary_owners,
                )
            )
            for doc in seen_documents.values()
        ]
    )

    # This does not update the permissions of the document if
    # the document already exists.
    on_conflict_stmt = insert_stmt.on_conflict_do_update(
        index_elements=["id"],  # Conflict target
        set_={
            "from_ingestion_api": insert_stmt.excluded.from_ingestion_api,
            "boost": insert_stmt.excluded.boost,
            "hidden": insert_stmt.excluded.hidden,
            "semantic_id": insert_stmt.excluded.semantic_id,
            "link": insert_stmt.excluded.link,
            "primary_owners": insert_stmt.excluded.primary_owners,
            "secondary_owners": insert_stmt.excluded.secondary_owners,
        },
    )
    db_session.execute(on_conflict_stmt)
    db_session.commit()


def upsert_document_by_connector_credential_pair(
    db_session: Session, connector_id: int, credential_id: int, document_ids: list[str]
) -> None:
    """
    插入或更新文档与连接器/凭证对的关系
    
    注意: 这个函数是Postgres特定的。并非所有数据库都支持ON CONFLICT子句。
    
    原注释:
    for now, there are no columns to update. If more metadata is added, then this
    needs to change to an `on_conflict_do_update`
    
    中文翻译:
    目前没有需要更新的列。如果添加了更多元数据，那么这需要改为`on_conflict_do_update`
    
    Args:
        db_session: 数据库会话对象
        connector_id: 连接器ID
        credential_id: 凭证ID
        document_ids: 文档ID列表
    """
    if not document_ids:
        logger.info("`document_ids` is empty. Skipping.")
        return

    insert_stmt = insert(DocumentByConnectorCredentialPair).values(
        [
            model_to_dict(
                DocumentByConnectorCredentialPair(
                    id=doc_id,
                    connector_id=connector_id,
                    credential_id=credential_id,
                )
            )
            for doc_id in document_ids
        ]
    )
    # for now, there are no columns to update. If more metadata is added, then this
    # needs to change to an `on_conflict_do_update`
    on_conflict_stmt = insert_stmt.on_conflict_do_nothing()
    db_session.execute(on_conflict_stmt)
    db_session.commit()


def update_docs_updated_at__no_commit(
    ids_to_new_updated_at: dict[str, datetime],
    db_session: Session,
) -> None:
    """
    更新文档的更新时间(不提交事务)
    
    Args:
        ids_to_new_updated_at: 文档ID到新更新时间的映射
        db_session: 数据库会话对象
    """
    doc_ids = list(ids_to_new_updated_at.keys())
    documents_to_update = (
        db_session.query(DbDocument).filter(DbDocument.id.in_(doc_ids)).all()
    )

    for document in documents_to_update:
        document.doc_updated_at = ids_to_new_updated_at[document.id]


def update_docs_last_modified__no_commit(
    document_ids: list[str],
    db_session: Session,
) -> None:
    """
    更新文档的最后修改时间(不提交事务)
    
    Args:
        document_ids: 文档ID列表
        db_session: 数据库会话对象
    """
    documents_to_update = (
        db_session.query(DbDocument).filter(DbDocument.id.in_(document_ids)).all()
    )

    now = datetime.now(timezone.utc)
    for doc in documents_to_update:
        doc.last_modified = now


def update_docs_chunk_count__no_commit(
    document_ids: list[str],
    doc_id_to_chunk_count: dict[str, int],
    db_session: Session,
) -> None:
    """
    更新文档的块计数(不提交事务)
    
    Args:
        document_ids: 文档ID列表
        doc_id_to_chunk_count: 文档ID到块计数的映射
        db_session: 数据库会话对象
    """
    documents_to_update = (
        db_session.query(DbDocument).filter(DbDocument.id.in_(document_ids)).all()
    )
    for doc in documents_to_update:
        doc.chunk_count = doc_id_to_chunk_count[doc.id]


def mark_document_as_modified(
    document_id: str,
    db_session: Session,
) -> None:
    """
    标记文档为已修改
    
    Args:
        document_id: 文档ID
        db_session: 数据库会话对象
    """
    stmt = select(DbDocument).where(DbDocument.id == document_id)
    doc = db_session.scalar(stmt)
    if (doc is None):
        raise ValueError(f"No document with ID: {document_id}")

    # update last_synced
    doc.last_modified = datetime.now(timezone.utc)
    db_session.commit()


def mark_document_as_synced(document_id: str, db_session: Session) -> None:
    """
    标记文档为已同步
    
    Args:
        document_id: 文档ID
        db_session: 数据库会话对象
    """
    stmt = select(DbDocument).where(DbDocument.id == document_id)
    doc = db_session.scalar(stmt)
    if doc is None:
        raise ValueError(f"No document with ID: {document_id}")

    # update last_synced
    doc.last_synced = datetime.now(timezone.utc)
    db_session.commit()


def delete_document_by_connector_credential_pair__no_commit(
    db_session: Session,
    document_id: str,
    connector_credential_pair_identifier: ConnectorCredentialPairIdentifier | None = None,
) -> None:
    """
    删除单个文档与连接器/凭证对的关系(不提交事务)
    
    原注释:
    Foreign key rows are left in place.
    The implicit assumption is that the document itself still has other cc_pair
    references and needs to continue existing.
    
    中文翻译:
    保留外键行。
    隐含假设是该文档本身仍然有其他cc_pair引用并且需要继续存在。
    
    Args:
        db_session: 数据库会话对象
        document_id: 文档ID
        connector_credential_pair_identifier: 连接器/凭证对标识符
    """
    delete_documents_by_connector_credential_pair__no_commit(
        db_session=db_session,
        document_ids=[document_id],
        connector_credential_pair_identifier=connector_credential_pair_identifier,
    )


def delete_documents_by_connector_credential_pair__no_commit(
    db_session: Session,
    document_ids: list[str],
    connector_credential_pair_identifier: ConnectorCredentialPairIdentifier | None = None,
) -> None:
    """
    删除多个文档与连接器/凭证对的关系(不提交事务)
    
    原注释:
    This deletes just the document by cc pair entries for a particular cc pair.
    Foreign key rows are left in place.
    The implicit assumption is that the document itself still has other cc_pair
    references and needs to continue existing.
    
    中文翻译:
    这只删除特定cc对的文档与cc对的关联条目。
    保留外键行。
    隐含假设是该文档本身仍然有其他cc_pair引用并且需要继续存在。
    
    Args:
        db_session: 数据库会话对象
        document_ids: 文档ID列表
        connector_credential_pair_identifier: 连接器/凭证对标识符
    """
    stmt = delete(DocumentByConnectorCredentialPair).where(
        DocumentByConnectorCredentialPair.id.in_(document_ids)
    )
    if connector_credential_pair_identifier:
        stmt = stmt.where(
            and_(
                DocumentByConnectorCredentialPair.connector_id
                == connector_credential_pair_identifier.connector_id,
                DocumentByConnectorCredentialPair.credential_id
                == connector_credential_pair_identifier.credential_id,
            )
        )
    db_session.execute(stmt)


def delete_documents__no_commit(db_session: Session, document_ids: list[str]) -> None:
    """
    删除文档(不提交事务)
    
    Args:
        db_session: 数据库会话对象
        document_ids: 文档ID列表
    """
    db_session.execute(delete(DbDocument).where(DbDocument.id.in_(document_ids)))


def delete_documents_complete__no_commit(
    db_session: Session, document_ids: list[str]
) -> None:
    """
    完全删除文档，包括所有外键关系(不提交事务)
    
    Args:
        db_session: 数据库会话对象
        document_ids: 文档ID列表
    """
    delete_documents_by_connector_credential_pair__no_commit(db_session, document_ids)
    delete_document_feedback_for_documents__no_commit(
        document_ids=document_ids, db_session=db_session
    )
    delete_document_tags_for_documents__no_commit(
        document_ids=document_ids, db_session=db_session
    )
    delete_documents__no_commit(db_session, document_ids)


def acquire_document_locks(db_session: Session, document_ids: list[str]) -> bool:
    """
    获取指定文档的锁。理想情况下，不应使用大列表调用此函数(除非持锁时间非常短)。
    
    如果任何文档已被锁定，将引发异常。这可以防止死锁(假设调用者在单次调用中传递所有必需的文档ID)。
    
    Args:
        db_session: 数据库会话对象
        document_ids: 文档ID列表
    Returns:
        bool: 成功获取锁返回True，否则返回False
    """
    stmt = (
        select(DbDocument.id)
        .where(DbDocument.id.in_(document_ids))
        .with_for_update(nowait=True)
    )
    # will raise exception if any of the documents are already locked
    documents = db_session.scalars(stmt).all()

    # make sure we found every document
    if len(documents) != len(set(document_ids)):
        logger.warning("Didn't find row for all specified document IDs. Aborting.")
        return False

    return True


_NUM_LOCK_ATTEMPTS = 10
_LOCK_RETRY_DELAY = 10


@contextlib.contextmanager
def prepare_to_modify_documents(
    db_session: Session, document_ids: list[str], retry_delay: int = _LOCK_RETRY_DELAY
) -> Generator[TransactionalContext, None, None]:
    """
    尝试获取文档的锁，以防止其他作业同时修改它们(例如，避免竞争条件)。
    这应该在对Vespa进行任何修改之前调用。锁应在更新完成后由调用者通过完成事务释放。
    
    注意: 仅允许在此函数返回的上下文管理器内进行一次提交。
    多次提交将导致sqlalchemy.exc.InvalidRequestError。
    注意: 此函数将提交任何现有事务。
    
    Args:
        db_session: 数据库会话对象
        document_ids: 文档ID列表
        retry_delay: 重试延迟时间
    """
    db_session.commit()  # ensure that we're not in a transaction

    lock_acquired = False
    for i in range(_NUM_LOCK_ATTEMPTS):
        try:
            with db_session.begin() as transaction:
                lock_acquired = acquire_document_locks(
                    db_session=db_session, document_ids=document_ids
                )
                if lock_acquired:
                    yield transaction
                    break
        except OperationalError as e:
            logger.warning(
                f"Failed to acquire locks for documents on attempt {i}, retrying. Error: {e}"
            )

        time.sleep(retry_delay)

    if not lock_acquired:
        raise RuntimeError(
            f"Failed to acquire locks after {_NUM_LOCK_ATTEMPTS} attempts "
            f"for documents: {document_ids}"
        )


def get_ingestion_documents(
    db_session: Session,
) -> list[DbDocument]:
    """
    获取通过ingestion API导入的文档
    
    Args:
        db_session: 数据库会话对象
    Returns:
        list[DbDocument]: 文档列表
    """
    # TODO add the option to filter by DocumentSource
    stmt = select(DbDocument).where(DbDocument.from_ingestion_api.is_(True))
    documents = db_session.execute(stmt).scalars().all()
    return list(documents)


def get_documents_by_cc_pair(
    cc_pair_id: int,
    db_session: Session,
) -> list[DbDocument]:
    """
    获取指定连接器/凭证对的文档
    
    Args:
        cc_pair_id: 连接器/凭证对ID
        db_session: 数据库会话对象
    Returns:
        list[DbDocument]: 文档列表
    """
    return (
        db_session.query(DbDocument)
        .join(
            DocumentByConnectorCredentialPair,
            DbDocument.id == DocumentByConnectorCredentialPair.id,
        )
        .join(
            ConnectorCredentialPair,
            and_(
                DocumentByConnectorCredentialPair.connector_id
                == ConnectorCredentialPair.connector_id,
                DocumentByConnectorCredentialPair.credential_id
                == ConnectorCredentialPair.credential_id,
            ),
        )
        .filter(ConnectorCredentialPair.id == cc_pair_id)
        .all()
    )


def get_document(
    document_id: str,
    db_session: Session,
) -> DbDocument | None:
    """
    获取指定ID的文档
    
    Args:
        document_id: 文档ID
        db_session: 数据库会话对象
    Returns:
        Optional[DbDocument]: 文档对象或None
    """
    stmt = select(DbDocument).where(DbDocument.id == document_id)
    doc: DbDocument | None = db_session.execute(stmt).scalar_one_or_none()
    return doc


def fetch_chunk_counts_for_documents(
    document_ids: list[str],
    db_session: Session,
) -> list[tuple[str, int | None]]:
    """
    获取文档的块计数
    
    返回一个包含(document_id, chunk_count)元组的列表。
    注意: 如果在数据库中未设置chunk_count，则chunk_count可能为None，
    因此我们将其声明为Optional[int]。
    
    Args:
        document_ids: 文档ID列表
        db_session: 数据库会话对象
    Returns:
        list[tuple[str, int | None]]: 每个文档的块计数
    """
    stmt = select(DbDocument.id, DbDocument.chunk_count).where(
        DbDocument.id.in_(document_ids)
    )

    # results is a list of 'Row' objects, each containing two columns
    results = db_session.execute(stmt).all()

    # If DbDocument.id is guaranteed to be a string, you can just do row.id;
    # otherwise cast to str if you need to be sure it's a string:
    return [(str(row[0]), row[1]) for row in results]
    # or row.id, row.chunk_count if they are named attributes in your ORM model
