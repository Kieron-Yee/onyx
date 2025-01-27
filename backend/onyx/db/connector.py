"""
这个文件是数据库连接器(Connector)相关的操作模块。
主要包含了对连接器的创建、查询、更新、删除等数据库操作，
以及连接器与凭证配对(ConnectorCredentialPair)相关的操作功能。
"""

from datetime import datetime
from datetime import timezone
from typing import cast

from sqlalchemy import and_
from sqlalchemy import exists
from sqlalchemy import func
from sqlalchemy import select
from sqlalchemy.orm import aliased
from sqlalchemy.orm import Session

from onyx.configs.app_configs import DEFAULT_PRUNING_FREQ
from onyx.configs.constants import DocumentSource
from onyx.connectors.models import InputType
from onyx.db.enums import IndexingMode
from onyx.db.models import Connector
from onyx.db.models import ConnectorCredentialPair
from onyx.db.models import IndexAttempt
from onyx.server.documents.models import ConnectorBase
from onyx.server.documents.models import ObjectCreationIdResponse
from onyx.server.models import StatusResponse
from onyx.utils.logger import setup_logger

logger = setup_logger()


def check_connectors_exist(db_session: Session) -> bool:
    """
    检查是否存在有效的连接器。
    
    Args:
        db_session: 数据库会话对象
    
    Returns:
        bool: 如果存在有效连接器返回True，否则返回False
    """
    # Connector 0在服务器启动时创建作为摄取的默认值 / Connector 0 is created on server startup as a default for ingestion
    # 它将始终存在，我们不需要将其计入 / it will always exist and we don't need to count it for this
    stmt = select(exists(Connector).where(Connector.id > 0))
    result = db_session.execute(stmt)
    return result.scalar() or False


def fetch_connectors(
    db_session: Session,
    sources: list[DocumentSource] | None = None,
    input_types: list[InputType] | None = None,
) -> list[Connector]:
    """
    获取符合条件的连接器列表。
    
    Args:
        db_session: 数据库会话对象
        sources: 文档来源列表，用于过滤连接器
        input_types: 输入类型列表，用于过滤连接器
    
    Returns:
        list[Connector]: 符合条件的连接器列表
    """
    stmt = select(Connector)
    if sources is not None:
        stmt = stmt.where(Connector.source.in_(sources))
    if input_types is not None:
        stmt = stmt.where(Connector.input_type.in_(input_types))
    results = db_session.scalars(stmt)
    return list(results.all())


def connector_by_name_source_exists(
    connector_name: str, source: DocumentSource, db_session: Session
) -> bool:
    """
    检查指定名称和来源的连接器是否已存在。
    
    Args:
        connector_name: 连接器名称
        source: 文档来源
        db_session: 数据库会话对象
    
    Returns:
        bool: 如果连接器已存在返回True，否则返回False
    """
    stmt = select(Connector).where(
        Connector.name == connector_name, Connector.source == source
    )
    result = db_session.execute(stmt)
    connector = result.scalar_one_or_none()
    return connector is not None


def fetch_connector_by_id(connector_id: int, db_session: Session) -> Connector | None:
    """
    根据连接器ID获取连接器对象。
    
    Args:
        connector_id: 连接器ID
        db_session: 数据库会话对象
    
    Returns:
        Connector | None: 如果找到连接器则返回连接器对象，否则返回None
    """
    stmt = select(Connector).where(Connector.id == connector_id)
    result = db_session.execute(stmt)
    connector = result.scalar_one_or_none()
    return connector


def fetch_ingestion_connector_by_name(
    connector_name: str, db_session: Session
) -> Connector | None:
    """
    根据名称获取摄取API的连接器对象。
    
    Args:
        connector_name: 连接器名称
        db_session: 数据库会话对象
    
    Returns:
        Connector | None: 如果找到连接器则返回连接器对象，否则返回None
    """
    stmt = (
        select(Connector)
        .where(Connector.name == connector_name)
        .where(Connector.source == DocumentSource.INGESTION_API)
    )
    result = db_session.execute(stmt)
    connector = result.scalar_one_or_none()
    return connector


def create_connector(
    db_session: Session,
    connector_data: ConnectorBase,
) -> ObjectCreationIdResponse:
    """
    创建新的连接器。
    
    Args:
        db_session: 数据库会话对象
        connector_data: 连接器基础数据
    
    Returns:
        ObjectCreationIdResponse: 包含新创建连接器ID的响应对象
    """
    if connector_by_name_source_exists(
        connector_data.name, connector_data.source, db_session
    ):
        raise ValueError(
            "Connector by this name already exists, duplicate naming not allowed."
        )

    connector = Connector(
        name=connector_data.name,
        source=connector_data.source,
        input_type=connector_data.input_type,
        connector_specific_config=connector_data.connector_specific_config,
        refresh_freq=connector_data.refresh_freq,
        indexing_start=connector_data.indexing_start,
        prune_freq=connector_data.prune_freq,
    )
    db_session.add(connector)
    db_session.commit()

    return ObjectCreationIdResponse(id=connector.id)


def update_connector(
    connector_id: int,
    connector_data: ConnectorBase,
    db_session: Session,
) -> Connector | None:
    """
    更新指定ID的连接器。
    
    Args:
        connector_id: 连接器ID
        connector_data: 连接器基础数据
        db_session: 数据库会话对象
    
    Returns:
        Connector | None: 如果找到并更新连接器则返回连接器对象，否则返回None
    """
    connector = fetch_connector_by_id(connector_id, db_session)
    if connector is None:
        return None

    if connector_data.name != connector.name and connector_by_name_source_exists(
        connector_data.name, connector_data.source, db_session
    ):
        raise ValueError(
            "Connector by this name already exists, duplicate naming not allowed."
        )

    connector.name = connector_data.name
    connector.source = connector_data.source
    connector.input_type = connector_data.input_type
    connector.connector_specific_config = connector_data.connector_specific_config
    connector.refresh_freq = connector_data.refresh_freq
    connector.prune_freq = (
        connector_data.prune_freq
        if connector_data.prune_freq is not None
        else DEFAULT_PRUNING_FREQ
    )

    db_session.commit()
    return connector


def delete_connector(
    db_session: Session,
    connector_id: int,
) -> StatusResponse[int]:
    """
    删除指定的连接器。
    
    Args:
        db_session: 数据库会话对象
        connector_id: 要删除的连接器ID
    
    Returns:
        StatusResponse[int]: 包含操作状态和结果的响应对象
        
    注意：
    仅在特殊情况下使用（例如连接器处于错误状态需要删除时）。
    使用时要非常小心，因为如果使用不当可能导致系统状态错误。
    """
    # 仅在特殊情况下使用（例如连接器处于错误状态需要删除时）。
    # 使用时要非常小心，因为如果使用不当可能导致系统状态错误。
    # Only used in special cases (e.g. a connector is in a bad state and we need to delete it).
    # Be VERY careful using this, as it could lead to a bad state if not used correctly.
    connector = fetch_connector_by_id(connector_id, db_session)
    if connector is None:
        return StatusResponse(
            success=True, message="Connector was already deleted", data=connector_id
        )

    db_session.delete(connector)
    return StatusResponse(
        success=True, message="Connector deleted successfully", data=connector_id
    )


def get_connector_credential_ids(
    connector_id: int,
    db_session: Session,
) -> list[int]:
    """
    获取指定连接器的凭证ID列表。
    
    Args:
        connector_id: 连接器ID
        db_session: 数据库会话对象
    
    Returns:
        list[int]: 凭证ID列表
    """
    connector = fetch_connector_by_id(connector_id, db_session)
    if connector is None:
        raise ValueError(f"Connector by id {connector_id} does not exist")

    return [association.credential.id for association in connector.credentials]


def fetch_latest_index_attempt_by_connector(
    db_session: Session,
    source: DocumentSource | None = None,
) -> list[IndexAttempt]:
    """
    获取每个连接器的最新索引尝试。
    
    Args:
        db_session: 数据库会话对象
        source: 文档来源（可选）
    
    Returns:
        list[IndexAttempt]: 最新的索引尝试列表
    """
    latest_index_attempts: list[IndexAttempt] = []

    if source:
        connectors = fetch_connectors(db_session, sources=[source])
    else:
        connectors = fetch_connectors(db_session)

    if not connectors:
        return []

    for connector in connectors:
        latest_index_attempt = (
            db_session.query(IndexAttempt)
            .join(ConnectorCredentialPair)
            .filter(ConnectorCredentialPair.connector_id == connector.id)
            .order_by(IndexAttempt.time_updated.desc())
            .first()
        )

        if latest_index_attempt is not None:
            latest_index_attempts.append(latest_index_attempt)

    return latest_index_attempts


def fetch_latest_index_attempts_by_status(
    db_session: Session,
) -> list[IndexAttempt]:
    """
    获取每个连接器凭证配对的最新索引尝试，按状态分组。
    
    Args:
        db_session: 数据库会话对象
    
    Returns:
        list[IndexAttempt]: 最新的索引尝试列表
    """
    subquery = (
        db_session.query(
            IndexAttempt.connector_credential_pair_id,
            IndexAttempt.status,
            func.max(IndexAttempt.time_updated).label("time_updated"),
        )
        .group_by(IndexAttempt.connector_credential_pair_id)
        .group_by(IndexAttempt.status)
        .subquery()
    )

    alias = aliased(IndexAttempt, subquery)

    query = db_session.query(IndexAttempt).join(
        alias,
        and_(
            IndexAttempt.connector_credential_pair_id
            == alias.connector_credential_pair_id,
            IndexAttempt.status == alias.status,
            IndexAttempt.time_updated == alias.time_updated,
        ),
    )

    return cast(list[IndexAttempt], query.all())


def fetch_unique_document_sources(db_session: Session) -> list[DocumentSource]:
    """
    获取所有唯一的文档来源。
    
    Args:
        db_session: 数据库会话对象
    
    Returns:
        list[DocumentSource]: 唯一的文档来源列表
    """
    distinct_sources = db_session.query(Connector.source).distinct().all()

    sources = [
        source[0]
        for source in distinct_sources
        if source[0] != DocumentSource.INGESTION_API
    ]

    return sources


def create_initial_default_connector(db_session: Session) -> None:
    """
    创建初始的默认连接器。如果默认连接器已存在但配置不正确，则更新其配置。
    
    Args:
        db_session: 数据库会话对象
    """
    default_connector_id = 0
    default_connector = fetch_connector_by_id(default_connector_id, db_session)
    if default_connector is not None:
        if (
            default_connector.source != DocumentSource.INGESTION_API
            or default_connector.input_type != InputType.LOAD_STATE
            or default_connector.refresh_freq is not None
            or default_connector.name != "Ingestion API"
            or default_connector.connector_specific_config != {}
            or default_connector.prune_freq is not None
        ):
            logger.warning(
                "默认连接器没有预期的值。正在更新到正确状态。"  # Default connector does not have expected values. Updating to proper state.
            )
            # 确保默认连接器具有正确的值 / Ensure default connector has correct values
            default_connector.source = DocumentSource.INGESTION_API
            default_connector.input_type = InputType.LOAD_STATE
            default_connector.refresh_freq = None
            default_connector.name = "Ingestion API"
            default_connector.connector_specific_config = {}
            default_connector.prune_freq = None
            db_session.commit()
        return

    # 创建新的默认连接器（如果不存在） / Create a new default connector if it doesn't exist
    connector = Connector(
        id=default_connector_id,
        name="Ingestion API",
        source=DocumentSource.INGESTION_API,
        input_type=InputType.LOAD_STATE,
        connector_specific_config={},
        refresh_freq=None,
        prune_freq=None,
    )
    db_session.add(connector)
    db_session.commit()


def mark_ccpair_as_pruned(cc_pair_id: int, db_session: Session) -> None:
    """
    标记指定的连接器凭证配对为已修剪。
    
    Args:
        cc_pair_id: 连接器凭证配对ID
        db_session: 数据库会话对象
    """
    stmt = select(ConnectorCredentialPair).where(
        ConnectorCredentialPair.id == cc_pair_id
    )
    cc_pair = db_session.scalar(stmt)
    if cc_pair is None:
        raise ValueError(f"No cc_pair with ID: {cc_pair_id}")

    cc_pair.last_pruned = datetime.now(timezone.utc)
    db_session.commit()


def mark_cc_pair_as_permissions_synced(
    db_session: Session, cc_pair_id: int, start_time: datetime | None
) -> None:
    """
    标记指定的连接器凭证配对的权限为已同步。
    
    Args:
        db_session: 数据库会话对象
        cc_pair_id: 连接器凭证配对ID
        start_time: 同步开始时间
    """
    stmt = select(ConnectorCredentialPair).where(
        ConnectorCredentialPair.id == cc_pair_id
    )
    cc_pair = db_session.scalar(stmt)
    if cc_pair is None:
        raise ValueError(f"No cc_pair with ID: {cc_pair_id}")

    cc_pair.last_time_perm_sync = start_time
    db_session.commit()


def mark_cc_pair_as_external_group_synced(db_session: Session, cc_pair_id: int) -> None:
    """
    标记指定的连接器凭证配对的外部组为已同步。
    
    Args:
        db_session: 数据库会话对象
        cc_pair_id: 连接器凭证配对ID
    """
    stmt = select(ConnectorCredentialPair).where(
        ConnectorCredentialPair.id == cc_pair_id
    )
    cc_pair = db_session.scalar(stmt)
    if cc_pair is None:
        raise ValueError(f"No cc_pair with ID: {cc_pair_id}")

    # 同步时间可以在运行后标记，因为所有组同步都是完全运行的，不会轮询更改。
    # 如果这发生变化，我们需要更新此函数。
    # The sync time can be marked after it ran because all group syncs
    # are run in full, not polling for changes.
    # If this changes, we need to update this function.
    cc_pair.last_time_external_group_sync = datetime.now(timezone.utc)
    db_session.commit()


def mark_ccpair_with_indexing_trigger(
    cc_pair_id: int, indexing_mode: IndexingMode | None, db_session: Session
) -> None:
    """
    标记指定的连接器凭证配对以触发索引。
    
    Args:
        cc_pair_id: 连接器凭证配对ID
        indexing_mode: 索引模式（可选）
        db_session: 数据库会话对象
    
    注意：
    indexing_mode设置一个字段，该字段将由后台任务拾取以触发索引。
    设置为None以禁用触发器。
    """
    try:
        cc_pair = db_session.execute(
            select(ConnectorCredentialPair)
            .where(ConnectorCredentialPair.id == cc_pair_id)
            .with_for_update()
        ).scalar_one()

        if cc_pair is None:
            raise ValueError(f"No cc_pair with ID: {cc_pair_id}")

        cc_pair.indexing_trigger = indexing_mode
        db_session.commit()
    except Exception:
        db_session.rollback()
        raise
