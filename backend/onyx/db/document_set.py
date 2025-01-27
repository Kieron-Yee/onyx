"""
这个文件主要用于处理文档集(Document Set)相关的数据库操作。
包括文档集的创建、更新、删除、查询等功能，以及文档集与连接器凭证对(Connector Credential Pair)、
用户组等之间关系的管理。
"""

from collections.abc import Sequence
from typing import cast
from uuid import UUID

from sqlalchemy import and_
from sqlalchemy import delete
from sqlalchemy import exists
from sqlalchemy import func
from sqlalchemy import or_
from sqlalchemy import Select
from sqlalchemy import select
from sqlalchemy.orm import aliased
from sqlalchemy.orm import Session

from onyx.db.connector_credential_pair import get_cc_pair_groups_for_ids
from onyx.db.connector_credential_pair import get_connector_credential_pairs
from onyx.db.enums import AccessType
from onyx.db.enums import ConnectorCredentialPairStatus
from onyx.db.models import ConnectorCredentialPair
from onyx.db.models import Document
from onyx.db.models import DocumentByConnectorCredentialPair
from onyx.db.models import DocumentSet as DocumentSetDBModel
from onyx.db.models import DocumentSet__ConnectorCredentialPair
from onyx.db.models import DocumentSet__UserGroup
from onyx.db.models import User
from onyx.db.models import User__UserGroup
from onyx.db.models import UserRole
from onyx.server.features.document_set.models import DocumentSetCreationRequest
from onyx.server.features.document_set.models import DocumentSetUpdateRequest
from onyx.utils.logger import setup_logger
from onyx.utils.variable_functionality import fetch_versioned_implementation

logger = setup_logger()


def _add_user_filters(
    stmt: Select, user: User | None, get_editable: bool = True
) -> Select:
    """
    为查询添加用户过滤条件，控制用户对文档集的访问权限
    
    Args:
        stmt: 原始查询语句
        user: 用户对象
        get_editable: 是否只获取可编辑的文档集
    
    Returns:
        添加了用户过滤条件的查询语句
    """
    # If user is None, assume the user is an admin or auth is disabled
    # 如果用户为空，则假定用户是管理员或认证被禁用
    if user is None or user.role == UserRole.ADMIN:
        return stmt

    DocumentSet__UG = aliased(DocumentSet__UserGroup)
    User__UG = aliased(User__UserGroup)
    """
    通过以下关系选择cc_pairs:
    User -> User__UserGroup -> DocumentSet__UserGroup -> DocumentSet
    """
    stmt = stmt.outerjoin(DocumentSet__UG).outerjoin(
        User__UserGroup,
        User__UserGroup.user_group_id == DocumentSet__UG.user_group_id,
    )
    """
    按以下条件过滤DocumentSets:
    - 如果用户在拥有DocumentSet的user_group中
    - 如果用户不是global_curator，他们还必须与user_group有curator关系
    - 如果在进行编辑，我们还会过滤掉用户不是curator的组所拥有的DocumentSets
    - 如果不是在编辑，我们显示用户是curator的组中的所有DocumentSets（以及公共DocumentSets）
    """
    where_clause = User__UserGroup.user_id == user.id
    if user.role == UserRole.CURATOR and get_editable:
        where_clause &= User__UserGroup.is_curator == True  # noqa: E712
    if get_editable:
        user_groups = select(User__UG.user_group_id).where(User__UG.user_id == user.id)
        if user.role == UserRole.CURATOR:
            user_groups = user_groups.where(User__UG.is_curator == True)  # noqa: E712
        where_clause &= (
            ~exists()
            .where(DocumentSet__UG.document_set_id == DocumentSetDBModel.id)
            .where(~DocumentSet__UG.user_group_id.in_(user_groups))
            .correlate(DocumentSetDBModel)
        )
    else:
        where_clause |= DocumentSetDBModel.is_public == True  # noqa: E712

    return stmt.where(where_clause)


def _delete_document_set_cc_pairs__no_commit(
    db_session: Session, document_set_id: int, is_current: bool | None = None
) -> None:
    """
    删除文档集与连接器凭证对之间的关联关系
    
    注意：不提交事务，需要调用者自行提交
    """
    stmt = delete(DocumentSet__ConnectorCredentialPair).where(
        DocumentSet__ConnectorCredentialPair.document_set_id == document_set_id
    )
    if is_current is not None:
        stmt = stmt.where(DocumentSet__ConnectorCredentialPair.is_current == is_current)
    db_session.execute(stmt)


def _mark_document_set_cc_pairs_as_outdated__no_commit(
    db_session: Session, document_set_id: int
) -> None:
    """
    将文档集的所有连接器凭证对标记为过期
    
    注意：不提交事务，需要调用者自行提交
    """
    stmt = select(DocumentSet__ConnectorCredentialPair).where(
        DocumentSet__ConnectorCredentialPair.document_set_id == document_set_id
    )
    for row in db_session.scalars(stmt):
        row.is_current = False


def delete_document_set_privacy__no_commit(
    document_set_id: int, db_session: Session
) -> None:
    """
    删除文档集的隐私设置
    
    在MIT版本中不支持私有文档集
    """


def get_document_set_by_id(
    db_session: Session,
    document_set_id: int,
    user: User | None = None,
    get_editable: bool = True,
) -> DocumentSetDBModel | None:
    """
    根据ID获取文档集
    
    Args:
        db_session: 数据库会话
        document_set_id: 文档集ID
        user: 用户对象，用于权限检查
        get_editable: 是否只获取可编辑的文档集
    
    Returns:
        文档集对象，如果不存在则返回None
    """
    stmt = select(DocumentSetDBModel).distinct()
    stmt = stmt.where(DocumentSetDBModel.id == document_set_id)
    stmt = _add_user_filters(stmt=stmt, user=user, get_editable=get_editable)
    return db_session.scalar(stmt)


def get_document_set_by_name(
    db_session: Session, document_set_name: str
) -> DocumentSetDBModel | None:
    """
    根据名称获取文档集
    
    Args:
        db_session: 数据库会话
        document_set_name: 文档集名称
    
    Returns:
        文档集对象，如果不存在则返回None
    """
    return db_session.scalar(
        select(DocumentSetDBModel).where(DocumentSetDBModel.name == document_set_name)
    )


def get_document_sets_by_ids(
    db_session: Session, document_set_ids: list[int]
) -> Sequence[DocumentSetDBModel]:
    """
    根据ID列表获取多个文档集
    
    Args:
        db_session: 数据库会话
        document_set_ids: 文档集ID列表
    
    Returns:
        文档集对象序列
    """
    if not document_set_ids:
        return []
    return db_session.scalars(
        select(DocumentSetDBModel).where(DocumentSetDBModel.id.in_(document_set_ids))
    ).all()


def make_doc_set_private(
    document_set_id: int,
    user_ids: list[UUID] | None,
    group_ids: list[int] | None,
    db_session: Session,
) -> None:
    """
    将文档集设为私有（在MIT版本中不支持此功能）
    
    Args:
        document_set_id: 文档集ID
        user_ids: 用户ID列表
        group_ids: 用户组ID列表
        db_session: 数据库会话
    
    Raises:
        NotImplementedError: MIT版本不支持私有文档集
    """
    # May cause error if someone switches down to MIT from EE
    if user_ids or group_ids:
        raise NotImplementedError("Onyx MIT does not support private Document Sets")


def _check_if_cc_pairs_are_owned_by_groups(
    db_session: Session,
    cc_pair_ids: list[int],
    group_ids: list[int],
) -> None:
    """
    检查连接器凭证对是否由指定的用户组拥有或为公共的。
    如果不是，则抛出ValueError异常。
    """
    group_cc_pair_relationships = get_cc_pair_groups_for_ids(
        db_session=db_session,
        cc_pair_ids=cc_pair_ids,
    )

    group_cc_pair_relationships_set = {
        (relationship.cc_pair_id, relationship.user_group_id)
        for relationship in group_cc_pair_relationships
    }

    missing_cc_pair_ids = []
    for cc_pair_id in cc_pair_ids:
        for group_id in group_ids:
            if (cc_pair_id, group_id) not in group_cc_pair_relationships_set:
                missing_cc_pair_ids.append(cc_pair_id)
                break

    if missing_cc_pair_ids:
        cc_pairs = get_connector_credential_pairs(
            db_session=db_session,
            ids=missing_cc_pair_ids,
        )
        for cc_pair in cc_pairs:
            if cc_pair.access_type != AccessType.PUBLIC:
                raise ValueError(
                    f"Connector Credential Pair with ID: '{cc_pair.id}'"
                    " is not owned by the specified groups"
                )


def insert_document_set(
    document_set_creation_request: DocumentSetCreationRequest,
    user_id: UUID | None,
    db_session: Session,
) -> tuple[DocumentSetDBModel, list[DocumentSet__ConnectorCredentialPair]]:
    """
    创建新的文档集
    
    Args:
        document_set_creation_request: 文档集创建请求对象
        user_id: 创建文档集的用户ID
        db_session: 数据库会话
    
    Returns:
        包含新创建的文档集和其关联的连接器凭证对关系列表的元组
        
    Raises:
        ValueError: 当没有提供连接器或其他验证失败时
    """
    # 实际上是cc-pairs，但UI显示这个错误
    if not document_set_creation_request.cc_pair_ids:
        raise ValueError("Cannot create a document set with no Connectors")

    if not document_set_creation_request.is_public:
        _check_if_cc_pairs_are_owned_by_groups(
            db_session=db_session,
            cc_pair_ids=document_set_creation_request.cc_pair_ids,
            group_ids=document_set_creation_request.groups or [],
        )

    try:
        new_document_set_row = DocumentSetDBModel(
            name=document_set_creation_request.name,
            description=document_set_creation_request.description,
            user_id=user_id,
            is_public=document_set_creation_request.is_public,
        )
        db_session.add(new_document_set_row)
        db_session.flush()  # ensure the new document set gets assigned an ID

        ds_cc_pairs = [
            DocumentSet__ConnectorCredentialPair(
                document_set_id=new_document_set_row.id,
                connector_credential_pair_id=cc_pair_id,
                is_current=True,
            )
            for cc_pair_id in document_set_creation_request.cc_pair_ids
        ]
        db_session.add_all(ds_cc_pairs)

        versioned_private_doc_set_fn = fetch_versioned_implementation(
            "onyx.db.document_set", "make_doc_set_private"
        )

        # Private Document Sets
        versioned_private_doc_set_fn(
            document_set_id=new_document_set_row.id,
            user_ids=document_set_creation_request.users,
            group_ids=document_set_creation_request.groups,
            db_session=db_session,
        )

        db_session.commit()
    except Exception as e:
        db_session.rollback()
        logger.error(f"Error creating document set: {e}")

    return new_document_set_row, ds_cc_pairs


def update_document_set(
    db_session: Session,
    document_set_update_request: DocumentSetUpdateRequest,
    user: User | None = None,
) -> tuple[DocumentSetDBModel, list[DocumentSet__ConnectorCredentialPair]]:
    """
    更新文档集信息。如果成功，将设置document_set_row.is_up_to_date = False。
    这将通过Celery在check_for_vespa_sync_task中处理，并触发长时间运行的后台同步到Vespa。
    """
    if not document_set_update_request.cc_pair_ids:
        # It's cc-pairs in actuality but the UI displays this error
        raise ValueError("Cannot create a document set with no Connectors")

    if not document_set_update_request.is_public:
        _check_if_cc_pairs_are_owned_by_groups(
            db_session=db_session,
            cc_pair_ids=document_set_update_request.cc_pair_ids,
            group_ids=document_set_update_request.groups,
        )

    try:
        # update the description
        document_set_row = get_document_set_by_id(
            db_session=db_session,
            document_set_id=document_set_update_request.id,
            user=user,
            get_editable=True,
        )
        if document_set_row is None:
            raise ValueError(
                f"No document set with ID '{document_set_update_request.id}'"
            )
        if not document_set_row.is_up_to_date:
            raise ValueError(
                "Cannot update document set while it is syncing. Please wait "
                "for it to finish syncing, and then try again."
            )

        document_set_row.description = document_set_update_request.description
        document_set_row.is_up_to_date = False
        document_set_row.is_public = document_set_update_request.is_public

        versioned_private_doc_set_fn = fetch_versioned_implementation(
            "onyx.db.document_set", "make_doc_set_private"
        )

        # Private Document Sets
        versioned_private_doc_set_fn(
            document_set_id=document_set_row.id,
            user_ids=document_set_update_request.users,
            group_ids=document_set_update_request.groups,
            db_session=db_session,
        )

        # update the attached CC pairs
        # first, mark all existing CC pairs as not current
        _mark_document_set_cc_pairs_as_outdated__no_commit(
            db_session=db_session, document_set_id=document_set_row.id
        )
        # add in rows for the new CC pairs
        ds_cc_pairs = [
            DocumentSet__ConnectorCredentialPair(
                document_set_id=document_set_update_request.id,
                connector_credential_pair_id=cc_pair_id,
                is_current=True,
            )
            for cc_pair_id in document_set_update_request.cc_pair_ids
        ]
        db_session.add_all(ds_cc_pairs)
        db_session.commit()
    except:
        db_session.rollback()
        raise

    return document_set_row, ds_cc_pairs


def mark_document_set_as_synced(document_set_id: int, db_session: Session) -> None:
    """
    将文档集标记为已同步状态
    
    Args:
        document_set_id: 文档集ID
        db_session: 数据库会话
    
    Raises:
        ValueError: 如果指定ID的文档集不存在
    """
    stmt = select(DocumentSetDBModel).where(DocumentSetDBModel.id == document_set_id)
    document_set = db_session.scalar(stmt)
    if (document_set is None):
        raise ValueError(f"No document set with ID: {document_set_id}")

    # mark as up to date
    document_set.is_up_to_date = True
    # delete outdated relationship table rows
    _delete_document_set_cc_pairs__no_commit(
        db_session=db_session, document_set_id=document_set_id, is_current=False
    )
    db_session.commit()


def delete_document_set(
    document_set_row: DocumentSetDBModel, db_session: Session
) -> None:
    """
    删除文档集及其所有相关联系
    
    Args:
        document_set_row: 要删除的文档集对象
        db_session: 数据库会话
    """
    # delete all relationships to CC pairs
    _delete_document_set_cc_pairs__no_commit(
        db_session=db_session, document_set_id=document_set_row.id
    )
    db_session.delete(document_set_row)
    db_session.commit()


def mark_document_set_as_to_be_deleted(
    db_session: Session,
    document_set_id: int,
    user: User | None = None,
) -> None:
    """
    清除所有文档集与连接器凭证对的关系，并将文档集标记为需要更新。
    实际的文档集行将由同步到Vespa的后台作业删除。
    """
    try:
        document_set_row = get_document_set_by_id(
            db_session=db_session,
            document_set_id=document_set_id,
            user=user,
            get_editable=True,
        )
        if document_set_row is None:
            error_msg = f"Document set with ID: '{document_set_id}' does not exist "
            if user is not None:
                error_msg += f"or is not editable by user with email: '{user.email}'"
            raise ValueError(error_msg)
        if not document_set_row.is_up_to_date:
            raise ValueError(
                "Cannot delete document set while it is syncing. Please wait "
                "for it to finish syncing, and then try again."
            )

        # delete all relationships to CC pairs
        _delete_document_set_cc_pairs__no_commit(
            db_session=db_session, document_set_id=document_set_id
        )

        # delete all private document set information
        versioned_delete_private_fn = fetch_versioned_implementation(
            "onyx.db.document_set", "delete_document_set_privacy__no_commit"
        )
        versioned_delete_private_fn(
            document_set_id=document_set_id, db_session=db_session
        )

        # mark the row as needing a sync, it will be deleted there since there
        # are no more relationships to cc pairs
        document_set_row.is_up_to_date = False
        db_session.commit()
    except:
        db_session.rollback()
        raise


def delete_document_set_cc_pair_relationship__no_commit(
    connector_id: int, credential_id: int, db_session: Session
) -> int:
    """
    删除DocumentSet__ConnectorCredentialPair中所有与给定cc_pair_id匹配的行。
    
    Args:
        connector_id: 连接器ID
        credential_id: 凭证ID
        db_session: 数据库会话
    
    Returns:
        删除的行数
    """
    delete_stmt = delete(DocumentSet__ConnectorCredentialPair).where(
        and_(
            ConnectorCredentialPair.connector_id == connector_id,
            ConnectorCredentialPair.credential_id == credential_id,
            DocumentSet__ConnectorCredentialPair.connector_credential_pair_id
            == ConnectorCredentialPair.id,
        )
    )
    result = db_session.execute(delete_stmt)
    return result.rowcount  # type: ignore


def fetch_document_sets(
    user_id: UUID | None, db_session: Session, include_outdated: bool = False
) -> list[tuple[DocumentSetDBModel, list[ConnectorCredentialPair]]]:
    """
    获取文档集及其关联的连接器凭证对。
    
    Args:
        user_id: 用户ID
        db_session: 数据库会话
        include_outdated: 是否包含过期的连接器凭证对
    
    Returns:
        包含文档集及其关联的连接器凭证对的列表
    """
    stmt = (
        select(DocumentSetDBModel, ConnectorCredentialPair)
        .join(
            DocumentSet__ConnectorCredentialPair,
            DocumentSetDBModel.id
            == DocumentSet__ConnectorCredentialPair.document_set_id,
            isouter=True,  # 需要外部连接以同时获取没有cc pairs的文档集
        )
        .join(
            ConnectorCredentialPair,
            ConnectorCredentialPair.id
            == DocumentSet__ConnectorCredentialPair.connector_credential_pair_id,
            isouter=True,  # 需要外部连接以同时获取没有cc pairs的文档集
        )
    )
    if not include_outdated:
        stmt = stmt.where(
            or_(
                DocumentSet__ConnectorCredentialPair.is_current == True,  # noqa: E712
                # `None` handles case where no CC Pairs exist for a Document Set
                DocumentSet__ConnectorCredentialPair.is_current.is_(None),
            )
        )

    results = cast(
        list[tuple[DocumentSetDBModel, ConnectorCredentialPair | None]],
        db_session.execute(stmt).all(),
    )

    aggregated_results: dict[
        int, tuple[DocumentSetDBModel, list[ConnectorCredentialPair]]
    ] = {}
    for document_set, cc_pair in results:
        if document_set.id not in aggregated_results:
            aggregated_results[document_set.id] = (
                document_set,
                [cc_pair] if cc_pair else [],
            )
        else:
            if cc_pair:
                aggregated_results[document_set.id][1].append(cc_pair)

    return [
        (document_set, cc_pairs)
        for document_set, cc_pairs in aggregated_results.values()
    ]


def fetch_all_document_sets_for_user(
    db_session: Session,
    user: User | None = None,
    get_editable: bool = True,
) -> Sequence[DocumentSetDBModel]:
    """
    获取用户可访问的所有文档集
    
    Args:
        db_session: 数据库会话
        user: 用户对象
        get_editable: 是否只获取可编辑的文档集
    
    Returns:
        文档集对象序列
    """
    stmt = select(DocumentSetDBModel).distinct()
    stmt = _add_user_filters(stmt, user, get_editable=get_editable)
    return db_session.scalars(stmt).all()


def fetch_documents_for_document_set_paginated(
    document_set_id: int,
    db_session: Session,
    current_only: bool = True,
    last_document_id: str | None = None,
    limit: int = 100,
) -> tuple[Sequence[Document], str | None]:
    """
    分页获取文档集中的文档
    
    Args:
        document_set_id: 文档集ID
        db_session: 数据库会话
        current_only: 是否只获取当前的文档
        last_document_id: 上一个文档的ID
        limit: 每页的文档数量
    
    Returns:
        包含文档列表和最后一个文档ID的元组
    """
    stmt = (
        select(Document)
        .join(
            DocumentByConnectorCredentialPair,
            DocumentByConnectorCredentialPair.id == Document.id,
        )
        .join(
            ConnectorCredentialPair,
            and_(
                ConnectorCredentialPair.connector_id
                == DocumentByConnectorCredentialPair.connector_id,
                ConnectorCredentialPair.credential_id
                == DocumentByConnectorCredentialPair.credential_id,
            ),
        )
        .join(
            DocumentSet__ConnectorCredentialPair,
            DocumentSet__ConnectorCredentialPair.connector_credential_pair_id
            == ConnectorCredentialPair.id,
        )
        .join(
            DocumentSetDBModel,
            DocumentSetDBModel.id
            == DocumentSet__ConnectorCredentialPair.document_set_id,
        )
        .where(DocumentSetDBModel.id == document_set_id)
        .order_by(Document.id)
        .limit(limit)
    )
    if last_document_id is not None:
        stmt = stmt.where(Document.id > last_document_id)
    if current_only:
        stmt = stmt.where(
            DocumentSet__ConnectorCredentialPair.is_current == True  # noqa: E712
        )
    stmt = stmt.distinct()

    documents = db_session.scalars(stmt).all()
    return documents, documents[-1].id if documents else None


def construct_document_select_by_docset(
    document_set_id: int,
    current_only: bool = True,
) -> Select:
    """
    构建一个查询语句，用于根据文档集ID获取文档。
    
    这个查询语句应该使用.yield_per()执行，以最小化开销。
    这个函数的主要消费者是后台处理任务生成器。
    
    Args:
        document_set_id: 文档集ID
        current_only: 是否只获取当前的文档
    
    Returns:
        查询语句
    """
    stmt = (
        select(Document)
        .join(
            DocumentByConnectorCredentialPair,
            DocumentByConnectorCredentialPair.id == Document.id,
        )
        .join(
            ConnectorCredentialPair,
            and_(
                ConnectorCredentialPair.connector_id
                == DocumentByConnectorCredentialPair.connector_id,
                ConnectorCredentialPair.credential_id
                == DocumentByConnectorCredentialPair.credential_id,
            ),
        )
        .join(
            DocumentSet__ConnectorCredentialPair,
            DocumentSet__ConnectorCredentialPair.connector_credential_pair_id
            == ConnectorCredentialPair.id,
        )
        .join(
            DocumentSetDBModel,
            DocumentSetDBModel.id
            == DocumentSet__ConnectorCredentialPair.document_set_id,
        )
        .where(DocumentSetDBModel.id == document_set_id)
        .order_by(Document.id)
    )

    if current_only:
        stmt = stmt.where(
            DocumentSet__ConnectorCredentialPair.is_current == True  # noqa: E712
        )

    stmt = stmt.distinct()
    return stmt


def fetch_document_sets_for_document(
    document_id: str,
    db_session: Session,
) -> list[str]:
    """
    获取单个文档ID的文档集名称。
    
    Args:
        document_id: 文档ID
        db_session: 数据库会话
    
    Returns:
        文档集名称列表
    """
    result = fetch_document_sets_for_documents([document_id], db_session)
    if not result:
        return []

    return result[0][1]


def fetch_document_sets_for_documents(
    document_ids: list[str],
    db_session: Session,
) -> Sequence[tuple[str, list[str]]]:
    """
    获取多个文档ID的文档集名称。
    
    Args:
        document_ids: 文档ID列表
        db_session: 数据库会话
    
    Returns:
        包含文档ID和文档集名称列表的元组序列
    """
    """构建子查询"""
    # 注意：必须首先构建这些子查询，以确保我们为每个指定的document_id获得一行返回。
    # 基本上，我们要先做过滤，然后再做外部连接。

    # 不包括正在删除的CC pairs
    # 注意：CC pairs永远不会从DELETING状态转换到任何其他状态 -> 可以安全地忽略它们，
    # 因为我们可以假设它们的文档集不再相关
    valid_cc_pairs_subquery = aliased(
        ConnectorCredentialPair,
        select(ConnectorCredentialPair)
        .where(
            ConnectorCredentialPair.status != ConnectorCredentialPairStatus.DELETING
        )  # noqa: E712
        .subquery(),
    )

    valid_document_set__cc_pairs_subquery = aliased(
        DocumentSet__ConnectorCredentialPair,
        select(DocumentSet__ConnectorCredentialPair)
        .where(DocumentSet__ConnectorCredentialPair.is_current == True)  # noqa: E712
        .subquery(),
    )
    """结束构建子查询"""

    stmt = (
        select(
            Document.id,
            func.coalesce(
                func.array_remove(func.array_agg(DocumentSetDBModel.name), None), []
            ).label("document_set_names"),
        )
        # 通过以下关系选择文档集：
        # Document -> DocumentByConnectorCredentialPair -> ConnectorCredentialPair ->
        # DocumentSet__ConnectorCredentialPair -> DocumentSet
        .outerjoin(
            DocumentByConnectorCredentialPair,
            Document.id == DocumentByConnectorCredentialPair.id,
        )
        .outerjoin(
            valid_cc_pairs_subquery,
            and_(
                DocumentByConnectorCredentialPair.connector_id
                == valid_cc_pairs_subquery.connector_id,
                DocumentByConnectorCredentialPair.credential_id
                == valid_cc_pairs_subquery.credential_id,
            ),
        )
        .outerjoin(
            valid_document_set__cc_pairs_subquery,
            valid_cc_pairs_subquery.id
            == valid_document_set__cc_pairs_subquery.connector_credential_pair_id,
        )
        .outerjoin(
            DocumentSetDBModel,
            DocumentSetDBModel.id
            == valid_document_set__cc_pairs_subquery.document_set_id,
        )
        .where(Document.id.in_(document_ids))
        .group_by(Document.id)
    )
    return db_session.execute(stmt).all()  # type: ignore


def get_or_create_document_set_by_name(
    db_session: Session,
    document_set_name: str,
    document_set_description: str = "Default Persona created Document-Set, "
    "please update description",
) -> DocumentSetDBModel:
    """
    获取或创建文档集。
    
    这个函数用于默认的角色(personas)，它们需要在服务器启动时附加到文档集。
    
    Args:
        db_session: 数据库会话
        document_set_name: 文档集名称
        document_set_description: 文档集描述
    
    Returns:
        文档集对象
    """
    doc_set = get_document_set_by_name(db_session, document_set_name)
    if doc_set is not None:
        return doc_set

    new_doc_set = DocumentSetDBModel(
        name=document_set_name,
        description=document_set_description,
        user_id=None,
        is_up_to_date=True,
    )

    db_session.add(new_doc_set)
    db_session.commit()

    return new_doc_set


def check_document_sets_are_public(
    db_session: Session,
    document_set_ids: list[int],
) -> bool:
    """
    检查文档集中的所有连接器凭证对是否为公共的。
    
    Args:
        db_session: 数据库会话
        document_set_ids: 文档集ID列表
    
    Returns:
        如果所有连接器凭证对都是公共的，则返回True，否则返回False
    """
    connector_credential_pair_ids = (
        db_session.query(
            DocumentSet__ConnectorCredentialPair.connector_credential_pair_id
        )
        .filter(
            DocumentSet__ConnectorCredentialPair.document_set_id.in_(document_set_ids)
        )
        .subquery()
    )

    not_public_exists = (
        db_session.query(ConnectorCredentialPair.id)
        .filter(
            ConnectorCredentialPair.id.in_(
                connector_credential_pair_ids  # type:ignore
            ),
            ConnectorCredentialPair.access_type != AccessType.PUBLIC,
        )
        .limit(1)
        .first()
        is not None
    )

    return not not_public_exists
