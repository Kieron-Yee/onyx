"""
This module handles the relationship between connectors and credentials, managing their associations,
permissions and statuses in the database.
此模块处理连接器和凭据之间的关系，管理它们在数据库中的关联、权限和状态。
"""

from datetime import datetime

from fastapi import HTTPException
from sqlalchemy import delete
from sqlalchemy import desc
from sqlalchemy import exists
from sqlalchemy import Select
from sqlalchemy import select
from sqlalchemy.orm import aliased
from sqlalchemy.orm import joinedload
from sqlalchemy.orm import Session

from onyx.configs.constants import DocumentSource
from onyx.db.connector import fetch_connector_by_id
from onyx.db.credentials import fetch_credential_by_id
from onyx.db.enums import AccessType
from onyx.db.enums import ConnectorCredentialPairStatus
from onyx.db.models import ConnectorCredentialPair
from onyx.db.models import IndexAttempt
from onyx.db.models import IndexingStatus
from onyx.db.models import IndexModelStatus
from onyx.db.models import SearchSettings
from onyx.db.models import User
from onyx.db.models import User__UserGroup
from onyx.db.models import UserGroup__ConnectorCredentialPair
from onyx.db.models import UserRole
from onyx.server.models import StatusResponse
from onyx.utils.logger import setup_logger
from onyx.utils.variable_functionality import fetch_ee_implementation_or_noop


logger = setup_logger()


def _add_user_filters(
    stmt: Select, user: User | None, get_editable: bool = True
) -> Select:
    """
    Add filters to a query based on user permissions
    根据用户权限添加查询过滤条件
    
    Args:
        stmt: The base SQL select statement 基础SQL查询语句
        user: The user to filter by 需要过滤的用户
        get_editable: Whether to only return editable pairs 是否只返回可编辑的配对
    """
    # If user is None, assume the user is an admin or auth is disabled
    if user is None or user.role == UserRole.ADMIN:
        return stmt

    UG__CCpair = aliased(UserGroup__ConnectorCredentialPair)
    User__UG = aliased(User__UserGroup)

    """
    Here we select cc_pairs by relation:
    User -> User__UserGroup -> UserGroup__ConnectorCredentialPair ->
    ConnectorCredentialPair
    这里我们通过关系选择cc_pairs：
    用户 -> 用户__用户组 -> 用户组__连接器凭据配对 -> 连接器凭据配对
    """
    stmt = stmt.outerjoin(UG__CCpair).outerjoin(
        User__UG,
        User__UG.user_group_id == UG__CCpair.user_group_id,
    )

    """
    Filter cc_pairs by:
    - if the user is in the user_group that owns the cc_pair
    - if the user is not a global_curator, they must also have a curator relationship
    to the user_group
    - if editing is being done, we also filter out cc_pairs that are owned by groups
    that the user isn't a curator for
    - if we are not editing, we show all cc_pairs in the groups the user is a curator
    for (as well as public cc_pairs)

    通过以下条件过滤cc_pairs：
    - 如果用户属于拥有cc_pair的用户组
    - 如果用户不是全局管理员，他们必须与用户组有管理员关系
    - 如果正在进行编辑，我们还要过滤掉用户不是管理员的组所拥有的cc_pairs
    - 如果不是在编辑，我们显示用户是管理员的组中的所有cc_pairs（以及公共cc_pairs）
    """
    where_clause = User__UG.user_id == user.id
    if user.role == UserRole.CURATOR and get_editable:
        where_clause &= User__UG.is_curator == True  # noqa: E712
    if get_editable:
        user_groups = select(User__UG.user_group_id).where(User__UG.user_id == user.id)
        if user.role == UserRole.CURATOR:
            user_groups = user_groups.where(
                User__UserGroup.is_curator == True  # noqa: E712
            )
        where_clause &= (
            ~exists()
            .where(UG__CCpair.cc_pair_id == ConnectorCredentialPair.id)
            .where(~UG__CCpair.user_group_id.in_(user_groups))
            .correlate(ConnectorCredentialPair)
        )
        where_clause |= ConnectorCredentialPair.creator_id == user.id
    else:
        where_clause |= ConnectorCredentialPair.access_type == AccessType.PUBLIC
        where_clause |= ConnectorCredentialPair.access_type == AccessType.SYNC

    return stmt.where(where_clause)


def get_connector_credential_pairs(
    db_session: Session,
    include_disabled: bool = True,
    user: User | None = None,
    get_editable: bool = True,
    ids: list[int] | None = None,
    eager_load_connector: bool = False,
) -> list[ConnectorCredentialPair]:
    """
    Get all connector credential pairs based on given filters
    根据给定的过滤条件获取所有连接器凭据配对
    """
    stmt = select(ConnectorCredentialPair).distinct()

    if eager_load_connector:
        stmt = stmt.options(joinedload(ConnectorCredentialPair.connector))

    stmt = _add_user_filters(stmt, user, get_editable)

    if not include_disabled:
        stmt = stmt.where(
            ConnectorCredentialPair.status == ConnectorCredentialPairStatus.ACTIVE
        )
    if ids:
        stmt = stmt.where(ConnectorCredentialPair.id.in_(ids))

    return list(db_session.scalars(stmt).all())


def add_deletion_failure_message(
    db_session: Session,
    cc_pair_id: int,
    failure_message: str,
) -> None:
    """
    Add a failure message to a connector credential pair when deletion fails
    当删除失败时，为连接器凭据配对添加失败消息
    """
    cc_pair = get_connector_credential_pair_from_id(cc_pair_id, db_session)
    if not cc_pair:
        return
    cc_pair.deletion_failure_message = failure_message
    db_session.commit()


def get_cc_pair_groups_for_ids(
    db_session: Session,
    cc_pair_ids: list[int],
    user: User | None = None,
    get_editable: bool = True,
) -> list[UserGroup__ConnectorCredentialPair]:
    """
    Get all user groups associated with given connector credential pair IDs
    获取与给定连接器凭据配对ID关联的所有用户组
    """
    stmt = select(UserGroup__ConnectorCredentialPair).distinct()
    stmt = stmt.outerjoin(
        ConnectorCredentialPair,
        UserGroup__ConnectorCredentialPair.cc_pair_id == ConnectorCredentialPair.id,
    )
    stmt = _add_user_filters(stmt, user, get_editable)
    stmt = stmt.where(UserGroup__ConnectorCredentialPair.cc_pair_id.in_(cc_pair_ids))
    return list(db_session.scalars(stmt).all())


def get_connector_credential_pair(
    connector_id: int,
    credential_id: int,
    db_session: Session,
    user: User | None = None,
    get_editable: bool = True,
) -> ConnectorCredentialPair | None:
    """
    Get a specific connector credential pair by connector and credential IDs
    通过连接器ID和凭据ID获取特定的连接器凭据配对
    """
    stmt = select(ConnectorCredentialPair)
    stmt = _add_user_filters(stmt, user, get_editable)
    stmt = stmt.where(ConnectorCredentialPair.connector_id == connector_id)
    stmt = stmt.where(ConnectorCredentialPair.credential_id == credential_id)
    result = db_session.execute(stmt)
    return result.scalar_one_or_none()


def get_connector_credential_source_from_id(
    cc_pair_id: int,
    db_session: Session,
    user: User | None = None,
    get_editable: bool = True,
) -> DocumentSource | None:
    """
    Get the document source type from a connector credential pair ID
    从连接器凭据配对ID获取文档源类型
    """
    stmt = select(ConnectorCredentialPair)
    stmt = _add_user_filters(stmt, user, get_editable)
    stmt = stmt.where(ConnectorCredentialPair.id == cc_pair_id)
    result = db_session.execute(stmt)
    cc_pair = result.scalar_one_or_none()
    return cc_pair.connector.source if cc_pair else None


def get_connector_credential_pair_from_id(
    cc_pair_id: int,
    db_session: Session,
    user: User | None = None,
    get_editable: bool = True,
) -> ConnectorCredentialPair | None:
    """
    Get a connector credential pair by its ID
    通过ID获取连接器凭据配对
    """
    stmt = select(ConnectorCredentialPair).distinct()
    stmt = _add_user_filters(stmt, user, get_editable)
    stmt = stmt.where(ConnectorCredentialPair.id == cc_pair_id)
    result = db_session.execute(stmt)
    return result.scalar_one_or_none()


def get_last_successful_attempt_time(
    connector_id: int,
    credential_id: int,
    earliest_index: float,
    search_settings: SearchSettings,
    db_session: Session,
) -> float:
    """Gets the timestamp of the last successful index run stored in
    the CC Pair row in the database
    获取存储在数据库中CC配对行中的最后一次成功索引运行的时间戳
    """
    if search_settings.status == IndexModelStatus.PRESENT:
        connector_credential_pair = get_connector_credential_pair(
            connector_id, credential_id, db_session
        )
        if (
            connector_credential_pair is None
            or connector_credential_pair.last_successful_index_time is None
        ):
            return earliest_index

        return connector_credential_pair.last_successful_index_time.timestamp()

    # For Secondary Index we don't keep track of the latest success, so have to calculate it live
    # 对于二级索引，我们不会跟踪最新的成功，因此必须实时计算
    attempt = (
        db_session.query(IndexAttempt)
        .join(
            ConnectorCredentialPair,
            IndexAttempt.connector_credential_pair_id == ConnectorCredentialPair.id,
        )
        .filter(
            ConnectorCredentialPair.connector_id == connector_id,
            ConnectorCredentialPair.credential_id == credential_id,
            IndexAttempt.search_settings_id == search_settings.id,
            IndexAttempt.status == IndexingStatus.SUCCESS,
        )
        .order_by(IndexAttempt.time_started.desc())
        .first()
    )

    if not attempt or not attempt.time_started:
        return earliest_index

    return attempt.time_started.timestamp()


"""Updates 
更新操作
"""


def _update_connector_credential_pair(
    db_session: Session,
    cc_pair: ConnectorCredentialPair,
    status: ConnectorCredentialPairStatus | None = None,
    net_docs: int | None = None,
    run_dt: datetime | None = None,
) -> None:
    """
    Update a connector credential pair with new status, document count, and run datetime
    使用新的状态、文档计数和运行日期时间更新连接器凭据配对
    """
    # simply don't update last_successful_index_time if run_dt is not specified
    # at worst, this would result in re-indexing documents that were already indexed
    # 如果未指定run_dt，则不更新last_successful_index_time
    # 最坏的情况是，这将导致重新索引已索引的文档
    if run_dt is not None:
        cc_pair.last_successful_index_time = run_dt
    if net_docs is not None:
        cc_pair.total_docs_indexed += net_docs
    if status is not None:
        cc_pair.status = status
    db_session.commit()


def update_connector_credential_pair_from_id(
    db_session: Session,
    cc_pair_id: int,
    status: ConnectorCredentialPairStatus | None = None,
    net_docs: int | None = None,
    run_dt: datetime | None = None,
) -> None:
    """
    Update a connector credential pair by its ID with new status, document count, and run datetime
    通过其ID使用新的状态、文档计数和运行日期时间更新连接器凭据配对
    """
    cc_pair = get_connector_credential_pair_from_id(cc_pair_id, db_session)
    if not cc_pair:
        logger.warning(
            f"Attempted to update pair for Connector Credential Pair '{cc_pair_id}'"
            f" but it does not exist"
        )
        return

    _update_connector_credential_pair(
        db_session=db_session,
        cc_pair=cc_pair,
        status=status,
        net_docs=net_docs,
        run_dt=run_dt,
    )


def update_connector_credential_pair(
    db_session: Session,
    connector_id: int,
    credential_id: int,
    status: ConnectorCredentialPairStatus | None = None,
    net_docs: int | None = None,
    run_dt: datetime | None = None,
) -> None:
    """
    Update a connector credential pair by connector and credential IDs with new status, document count, and run datetime
    通过连接器和凭据ID使用新的状态、文档计数和运行日期时间更新连接器凭据配对
    """
    cc_pair = get_connector_credential_pair(connector_id, credential_id, db_session)
    if not cc_pair:
        logger.warning(
            f"Attempted to update pair for connector id {connector_id} "
            f"and credential id {credential_id}"
        )
        return

    _update_connector_credential_pair(
        db_session=db_session,
        cc_pair=cc_pair,
        status=status,
        net_docs=net_docs,
        run_dt=run_dt,
    )


def delete_connector_credential_pair__no_commit(
    db_session: Session,
    connector_id: int,
    credential_id: int,
) -> None:
    """
    Delete a connector credential pair without committing the transaction
    删除连接器凭据配对但不提交事务
    """
    stmt = delete(ConnectorCredentialPair).where(
        ConnectorCredentialPair.connector_id == connector_id,
        ConnectorCredentialPair.credential_id == credential_id,
    )
    db_session.execute(stmt)


def associate_default_cc_pair(db_session: Session) -> None:
    """
    Associate the default connector credential pair if it does not already exist
    如果默认连接器凭据配对不存在，则关联它
    """
    existing_association = (
        db_session.query(ConnectorCredentialPair)
        .filter(
            ConnectorCredentialPair.connector_id == 0,
            ConnectorCredentialPair.credential_id == 0,
        )
        .one_or_none()
    )
    if existing_association is not None:
        return

    # DefaultCCPair has id 1 since it is the first CC pair created
    # It is DEFAULT_CC_PAIR_ID, but can't set it explicitly because it messed with the
    # auto-incrementing id
    # DefaultCCPair的id为1，因为它是创建的第一个CC配对
    # 它是DEFAULT_CC_PAIR_ID，但不能显式设置它，因为这会干扰自动递增的id功能
    association = ConnectorCredentialPair(
        connector_id=0,
        credential_id=0,
        access_type=AccessType.PUBLIC,
        name="DefaultCCPair",
        status=ConnectorCredentialPairStatus.ACTIVE,
    )
    db_session.add(association)
    db_session.commit()


def _relate_groups_to_cc_pair__no_commit(
    db_session: Session,
    cc_pair_id: int,
    user_group_ids: list[int] | None = None,
) -> None:
    """
    Relate user groups to a connector credential pair without committing the transaction
    将用户组与连接器凭据配对关联但不提交事务
    """
    if not user_group_ids:
        return

    for group_id in user_group_ids:
        db_session.add(
            UserGroup__ConnectorCredentialPair(
                user_group_id=group_id, cc_pair_id=cc_pair_id
            )
        )


def add_credential_to_connector(
    db_session: Session,
    user: User | None,
    connector_id: int,
    credential_id: int,
    cc_pair_name: str | None,
    access_type: AccessType,
    groups: list[int] | None,
    auto_sync_options: dict | None = None,
    initial_status: ConnectorCredentialPairStatus = ConnectorCredentialPairStatus.ACTIVE,
    last_successful_index_time: datetime | None = None,
) -> StatusResponse:
    """
    Add a credential to a connector, creating a new connector credential pair
    将凭据添加到连接器，创建新的连接器凭据配对
    """
    connector = fetch_connector_by_id(connector_id, db_session)
    credential = fetch_credential_by_id(
        credential_id,
        user,
        db_session,
        get_editable=False,
    )

    if connector is None:
        raise HTTPException(status_code=404, detail="Connector does not exist")

    if access_type == AccessType.SYNC:
        if not fetch_ee_implementation_or_noop(
            "onyx.external_permissions.sync_params",
            "check_if_valid_sync_source",
            noop_return_value=True,
        )(connector.source):
            raise HTTPException(
                status_code=400,
                detail=f"Connector of type {connector.source} does not support SYNC access type",
            )

    if credential is None:
        error_msg = (
            f"Credential {credential_id} does not exist or does not belong to user"
        )
        logger.error(error_msg)
        raise HTTPException(
            status_code=401,
            detail=error_msg,
        )

    existing_association = (
        db_session.query(ConnectorCredentialPair)
        .filter(
            ConnectorCredentialPair.connector_id == connector_id,
            ConnectorCredentialPair.credential_id == credential_id,
        )
        .one_or_none()
    )
    if existing_association is not None:
        return StatusResponse(
            success=False,
            message=f"Connector {connector_id} already has Credential {credential_id}",
            data=connector_id,
        )

    association = ConnectorCredentialPair(
        creator_id=user.id if user else None,
        connector_id=connector_id,
        credential_id=credential_id,
        name=cc_pair_name,
        status=initial_status,
        access_type=access_type,
        auto_sync_options=auto_sync_options,
        last_successful_index_time=last_successful_index_time,
    )
    db_session.add(association)
    db_session.flush()  # make sure the association has an id
    db_session.refresh(association)

    _relate_groups_to_cc_pair__no_commit(
        db_session=db_session,
        cc_pair_id=association.id,
        user_group_ids=groups,
    )

    db_session.commit()

    return StatusResponse(
        success=True,
        message=f"Creating new association between Connector {connector_id} and Credential {credential_id}",
        data=association.id,
    )


def remove_credential_from_connector(
    connector_id: int,
    credential_id: int,
    user: User | None,
    db_session: Session,
) -> StatusResponse[int]:
    """
    Remove a credential from a connector, deleting the connector credential pair
    从连接器中删除凭据，删除连接器凭据配对
    """
    connector = fetch_connector_by_id(connector_id, db_session)
    credential = fetch_credential_by_id(
        credential_id,
        user,
        db_session,
        get_editable=False,
    )

    if connector is None:
        raise HTTPException(status_code=404, detail="Connector does not exist")

    if credential is None:
        raise HTTPException(
            status_code=404,
            detail="Credential does not exist or does not belong to user",
        )

    association = get_connector_credential_pair(
        connector_id=connector_id,
        credential_id=credential_id,
        db_session=db_session,
        user=user,
        get_editable=True,
    )

    if association is not None:
        fetch_ee_implementation_or_noop(
            "onyx.db.external_perm",
            "delete_user__ext_group_for_cc_pair__no_commit",
        )(
            db_session=db_session,
            cc_pair_id=association.id,
        )
        db_session.delete(association)
        db_session.commit()
        return StatusResponse(
            success=True,
            message=f"Credential {credential_id} removed from Connector",
            data=connector_id,
        )

    return StatusResponse(
        success=False,
        message=f"Connector already does not have Credential {credential_id}",
        data=connector_id,
    )


def fetch_connector_credential_pairs(
    db_session: Session,
) -> list[ConnectorCredentialPair]:
    """
    Fetch all connector credential pairs from the database
    从数据库中获取所有连接器凭据配对
    """
    return db_session.query(ConnectorCredentialPair).all()


def resync_cc_pair(
    cc_pair: ConnectorCredentialPair,
    db_session: Session,
) -> None:
    """
    Resynchronize a connector credential pair by updating its last successful index time
    通过更新最后成功索引时间来重新同步连接器凭据配对
    """
    def find_latest_index_attempt(
        connector_id: int,
        credential_id: int,
        only_include_success: bool,
        db_session: Session,
    ) -> IndexAttempt | None:
        """
        Find the latest index attempt for a given connector and credential
        查找给定连接器和凭据的最新索引尝试
        """
        query = (
            db_session.query(IndexAttempt)
            .join(
                ConnectorCredentialPair,
                IndexAttempt.connector_credential_pair_id == ConnectorCredentialPair.id,
            )
            .join(SearchSettings, IndexAttempt.search_settings_id == SearchSettings.id)
            .filter(
                ConnectorCredentialPair.connector_id == connector_id,
                ConnectorCredentialPair.credential_id == credential_id,
                SearchSettings.status == IndexModelStatus.PRESENT,
            )
        )

        if only_include_success:
            query = query.filter(IndexAttempt.status == IndexingStatus.SUCCESS)

        latest_index_attempt = query.order_by(desc(IndexAttempt.time_started)).first()

        return latest_index_attempt

    last_success = find_latest_index_attempt(
        connector_id=cc_pair.connector_id,
        credential_id=cc_pair.credential_id,
        only_include_success=True,
        db_session=db_session,
    )

    cc_pair.last_successful_index_time = (
        last_success.time_started if last_success else None
    )

    db_session.commit()
