"""
这个文件主要用于处理凭证(credentials)相关的数据库操作，包括凭证的创建、查询、更新和删除。
它处理用户权限验证，并管理凭证与用户组之间的关系。
"""

from typing import Any

from sqlalchemy import exists
from sqlalchemy import Select
from sqlalchemy import select
from sqlalchemy import update
from sqlalchemy.orm import Session
from sqlalchemy.sql.expression import and_
from sqlalchemy.sql.expression import or_

from onyx.auth.schemas import UserRole
from onyx.configs.constants import DocumentSource
from onyx.connectors.google_utils.shared_constants import (
    DB_CREDENTIALS_DICT_SERVICE_ACCOUNT_KEY,
)
from onyx.db.models import ConnectorCredentialPair
from onyx.db.models import Credential
from onyx.db.models import Credential__UserGroup
from onyx.db.models import DocumentByConnectorCredentialPair
from onyx.db.models import User
from onyx.db.models import User__UserGroup
from onyx.server.documents.models import CredentialBase
from onyx.utils.logger import setup_logger


logger = setup_logger()

# 这些数据源的凭证不是真实的，因此不强制执行权限
# The credentials for these sources are not real so permissions are not enforced for them
CREDENTIAL_PERMISSIONS_TO_IGNORE = {
    DocumentSource.FILE,
    DocumentSource.WEB,
    DocumentSource.NOT_APPLICABLE,
    DocumentSource.GOOGLE_SITES,
    DocumentSource.WIKIPEDIA,
    DocumentSource.MEDIAWIKI,
}

PUBLIC_CREDENTIAL_ID = 0


def _add_user_filters(
    stmt: Select,
    user: User | None,
    assume_admin: bool = False,  # 用于API密钥
    get_editable: bool = True,
) -> Select:
    """
    为查询语句添加用户过滤条件，确保用户只能访问其有权限的凭证
    
    Args:
        stmt: 原始查询语句
        user: 用户对象
        assume_admin: 是否假定为管理员权限（用于API密钥）
        get_editable: 是否获取可编辑的凭证

    Returns:
        添加了用户过滤条件的查询语句
    """
    if not user:
        if assume_admin:
            # 应用管理员过滤器，去除user_id检查
            stmt = stmt.where(
                or_(
                    Credential.user_id.is_(None),
                    Credential.admin_public == True,  # noqa: E712
                    Credential.source.in_(CREDENTIAL_PERMISSIONS_TO_IGNORE),
                )
            )
        return stmt

    if user.role == UserRole.ADMIN:
        # 管理员可以访问所有公共的、由他们拥有的或没有关联用户的凭证
        return stmt.where(
            or_(
                Credential.user_id == user.id,
                Credential.user_id.is_(None),
                Credential.admin_public == True,  # noqa: E712
                Credential.source.in_(CREDENTIAL_PERMISSIONS_TO_IGNORE),
            )
        )
    if user.role == UserRole.BASIC:
        # 基础用户只能访问他们拥有的凭证
        return stmt.where(Credential.user_id == user.id)

    """
    这部分是针对管理员和全局管理员的
    这里我们通过以下关系选择cc_pairs：
    用户 -> 用户组 -> 凭证用户组 -> 凭证
    """
    stmt = stmt.outerjoin(Credential__UserGroup).outerjoin(
        User__UserGroup,
        User__UserGroup.user_group_id == Credential__UserGroup.user_group_id,
    )
    """
    按以下条件过滤凭证：
    - 如果用户在拥有凭证的用户组中
    - 如果用户是管理员，他们还必须与用户组有管理关系
    - 如果正在进行编辑，我们还会过滤掉用户不是其管理员的组所拥有的凭证
    - 如果不是编辑，我们显示用户是其管理员的组中的所有凭证（以及公共凭证）
    - 如果不是编辑，我们返回所有直接与用户相关的凭证
    """
    where_clause = User__UserGroup.user_id == user.id
    if user.role == UserRole.CURATOR:
        where_clause &= User__UserGroup.is_curator == True  # noqa: E712

    if get_editable:
        user_groups = select(User__UserGroup.user_group_id).where(
            User__UserGroup.user_id == user.id
        )
        if user.role == UserRole.CURATOR:
            user_groups = user_groups.where(
                User__UserGroup.is_curator == True  # noqa: E712
            )
        where_clause &= (
            ~exists()
            .where(Credential__UserGroup.credential_id == Credential.id)
            .where(~Credential__UserGroup.user_group_id.in_(user_groups))
            .correlate(Credential)
        )
    else:
        where_clause |= Credential.curator_public == True  # noqa: E712
        where_clause |= Credential.user_id == user.id  # noqa: E712

    where_clause |= Credential.source.in_(CREDENTIAL_PERMISSIONS_TO_IGNORE)

    return stmt.where(where_clause)


def _relate_credential_to_user_groups__no_commit(
    db_session: Session,
    credential_id: int,
    user_group_ids: list[int],
) -> None:
    """
    建立凭证与用户组之间的关联关系（不提交事务）
    
    Args:
        db_session: 数据库会话
        credential_id: 凭证ID
        user_group_ids: 用户组ID列表
    """
    credential_user_groups = []
    for group_id in user_group_ids:
        credential_user_groups.append(
            Credential__UserGroup(
                credential_id=credential_id,
                user_group_id=group_id,
            )
        )
    db_session.add_all(credential_user_groups)


def fetch_credentials(
    db_session: Session,
    user: User | None = None,
    get_editable: bool = True,
) -> list[Credential]:
    """
    获取用户有权限访问的所有凭证列表
    
    Args:
        db_session: 数据库会话
        user: 用户对象
        get_editable: 是否只获取可编辑的凭证
    
    Returns:
        凭证对象列表
    """
    stmt = select(Credential)
    stmt = _add_user_filters(stmt, user, get_editable=get_editable)
    results = db_session.scalars(stmt)
    return list(results.all())


def fetch_credential_by_id(
    credential_id: int,
    user: User | None,
    db_session: Session,
    assume_admin: bool = False,
    get_editable: bool = True,
) -> Credential | None:
    """
    根据ID获取特定的凭证
    
    Args:
        credential_id: 要获取的凭证ID
        user: 请求的用户对象
        db_session: 数据库会话
        assume_admin: 是否假定管理员权限
        get_editable: 是否只获取可编辑的凭证
    
    Returns:
        找到的凭证对象，如果未找到则返回None
    """
    stmt = select(Credential).distinct()
    stmt = stmt.where(Credential.id == credential_id)
    stmt = _add_user_filters(
        stmt=stmt,
        user=user,
        assume_admin=assume_admin,
        get_editable=get_editable,
    )
    result = db_session.execute(stmt)
    credential = result.scalar_one_or_none()
    return credential


def fetch_credentials_by_source(
    db_session: Session,
    user: User | None,
    document_source: DocumentSource | None = None,
    get_editable: bool = True,
) -> list[Credential]:
    """
    根据数据源获取用户有权限访问的凭证列表
    
    Args:
        db_session: 数据库会话
        user: 用户对象
        document_source: 数据源
        get_editable: 是否只获取可编辑的凭证
    
    Returns:
        凭证对象列表
    """
    base_query = select(Credential).where(Credential.source == document_source)
    base_query = _add_user_filters(base_query, user, get_editable=get_editable)
    credentials = db_session.execute(base_query).scalars().all()
    return list(credentials)


def swap_credentials_connector(
    new_credential_id: int, connector_id: int, user: User | None, db_session: Session
) -> ConnectorCredentialPair:
    """
    交换连接器的凭证
    
    Args:
        new_credential_id: 新凭证ID
        connector_id: 连接器ID
        user: 用户对象
        db_session: 数据库会话
    
    Returns:
        更新后的连接器凭证对
    """
    # Check if the user has permission to use the new credential
    new_credential = fetch_credential_by_id(new_credential_id, user, db_session)
    if not new_credential:
        raise ValueError(
            f"No Credential found with id {new_credential_id} or user doesn't have permission to use it"
        )

    # Existing pair
    existing_pair = db_session.execute(
        select(ConnectorCredentialPair).where(
            ConnectorCredentialPair.connector_id == connector_id
        )
    ).scalar_one_or_none()

    if not existing_pair:
        raise ValueError(
            f"No ConnectorCredentialPair found for connector_id {connector_id}"
        )

    # Check if the new credential is compatible with the connector
    if new_credential.source != existing_pair.connector.source:
        raise ValueError(
            f"New credential source {new_credential.source} does not match connector source {existing_pair.connector.source}"
        )

    db_session.execute(
        update(DocumentByConnectorCredentialPair)
        .where(
            and_(
                DocumentByConnectorCredentialPair.connector_id == connector_id,
                DocumentByConnectorCredentialPair.credential_id
                == existing_pair.credential_id,
            )
        )
        .values(credential_id=new_credential_id)
    )

    # Update the existing pair with the new credential
    existing_pair.credential_id = new_credential_id
    existing_pair.credential = new_credential

    # Commit the changes
    db_session.commit()

    # Refresh the object to ensure all relationships are up-to-date
    db_session.refresh(existing_pair)
    return existing_pair


def create_credential(
    credential_data: CredentialBase,
    user: User | None,
    db_session: Session,
) -> Credential:
    """
    创建新的凭证
    
    Args:
        credential_data: 凭证基础数据
        user: 用户对象
        db_session: 数据库会话
    
    Returns:
        创建的凭证对象
    """
    credential = Credential(
        credential_json=credential_data.credential_json,
        user_id=user.id if user else None,
        admin_public=credential_data.admin_public,
        source=credential_data.source,
        name=credential_data.name,
        curator_public=credential_data.curator_public,
    )
    db_session.add(credential)
    db_session.flush()  # 这确保凭证获得一个ID
    _relate_credential_to_user_groups__no_commit(
        db_session=db_session,
        credential_id=credential.id,
        user_group_ids=credential_data.groups,
    )

    db_session.commit()
    return credential


def _cleanup_credential__user_group_relationships__no_commit(
    db_session: Session, credential_id: int
) -> None:
    """
    清理凭证与用户组之间的关系（不提交事务）
    
    Args:
        db_session: 数据库会话
        credential_id: 凭证ID
    """
    db_session.query(Credential__UserGroup).filter(
        Credential__UserGroup.credential_id == credential_id
    ).delete(synchronize_session=False)


def alter_credential(
    credential_id: int,
    name: str,
    credential_json: dict[str, Any],
    user: User,
    db_session: Session,
) -> Credential | None:
    """
    修改现有的凭证
    
    Args:
        credential_id: 凭证ID
        name: 新的凭证名称
        credential_json: 新的凭证JSON数据
        user: 用户对象
        db_session: 数据库会话
    
    Returns:
        修改后的凭证对象，如果未找到则返回None
    """
    # TODO: 添加用户组关系更新
    credential = fetch_credential_by_id(credential_id, user, db_session)

    if credential is None:
        return None

    credential.name = name

    # 为credential.credential_json分配一个新的字典
    credential.credential_json = {
        **credential.credential_json,
        **credential_json,
    }

    credential.user_id = user.id if user is not None else None
    db_session.commit()
    return credential


def update_credential(
    credential_id: int,
    credential_data: CredentialBase,
    user: User,
    db_session: Session,
) -> Credential | None:
    """
    更新现有的凭证
    
    Args:
        credential_id: 凭证ID
        credential_data: 凭证基础数据
        user: 用户对象
        db_session: 数据库会话
    
    Returns:
        更新后的凭证对象，如果未找到则返回None
    """
    credential = fetch_credential_by_id(credential_id, user, db_session)
    if credential is None:
        return None

    credential.credential_json = credential_data.credential_json
    credential.user_id = user.id if user is not None else None

    db_session.commit()
    return credential


def update_credential_json(
    credential_id: int,
    credential_json: dict[str, Any],
    user: User,
    db_session: Session,
) -> Credential | None:
    """
    更新凭证的JSON数据
    
    Args:
        credential_id: 凭证ID
        credential_json: 新的凭证JSON数据
        user: 用户对象
        db_session: 数据库会话
    
    Returns:
        更新后的凭证对象，如果未找到则返回None
    """
    credential = fetch_credential_by_id(credential_id, user, db_session)
    if credential is None:
        return None

    credential.credential_json = credential_json
    db_session.commit()
    return credential


def backend_update_credential_json(
    credential: Credential,
    credential_json: dict[str, Any],
    db_session: Session,
) -> None:
    """
    后端更新凭证的JSON数据（不涉及前端或用户的流程）
    
    Args:
        credential: 凭证对象
        credential_json: 新的凭证JSON数据
        db_session: 数据库会话
    """
    credential.credential_json = credential_json
    db_session.commit()


def delete_credential(
    credential_id: int,
    user: User | None,
    db_session: Session,
    force: bool = False,
) -> None:
    """
    删除凭证
    
    Args:
        credential_id: 凭证ID
        user: 用户对象
        db_session: 数据库会话
        force: 是否强制删除
    
    Raises:
        ValueError: 如果凭证不存在或不属于用户，或凭证仍与连接器或文档关联
    """
    credential = fetch_credential_by_id(credential_id, user, db_session)
    if credential is None:
        raise ValueError(
            f"Credential by provided id {credential_id} does not exist or does not belong to user"
        )

    associated_connectors = (
        db_session.query(ConnectorCredentialPair)
        .filter(ConnectorCredentialPair.credential_id == credential_id)
        .all()
    )

    associated_doc_cc_pairs = (
        db_session.query(DocumentByConnectorCredentialPair)
        .filter(DocumentByConnectorCredentialPair.credential_id == credential_id)
        .all()
    )

    if associated_connectors or associated_doc_cc_pairs:
        if force:
            logger.warning(
                f"强制删除凭证 {credential_id} 及其关联记录"
            )

            # 首先删除DocumentByConnectorCredentialPair记录
            for doc_cc_pair in associated_doc_cc_pairs:
                db_session.delete(doc_cc_pair)

            # 然后删除ConnectorCredentialPair记录
            for connector in associated_connectors:
                db_session.delete(connector)

            # 在删除凭证之前提交这些删除操作
            db_session.flush()
        else:
            raise ValueError(
                f"Cannot delete credential as it is still associated with "
                f"{len(associated_connectors)} connector(s) and {len(associated_doc_cc_pairs)} document(s). "
            )

    if force:
        logger.warning(f"Force deleting credential {credential_id}")
    else:
        logger.notice(f"Deleting credential {credential_id}")

    _cleanup_credential__user_group_relationships__no_commit(db_session, credential_id)
    db_session.delete(credential)
    db_session.commit()


def create_initial_public_credential(db_session: Session) -> None:
    """
    创建初始的公共凭证
    
    Args:
        db_session: 数据库会话
    
    Raises:
        ValueError: 如果数据库不在有效的初始状态
    """
    error_msg = (
        "DB is not in a valid initial state."
        "There must exist an empty public credential for data connectors that do not require additional Auth."
    )
    first_credential = fetch_credential_by_id(PUBLIC_CREDENTIAL_ID, None, db_session)

    if first_credential is not None:
        if first_credential.credential_json != {} or first_credential.user is not None:
            raise ValueError(error_msg)
        return

    credential = Credential(
        id=PUBLIC_CREDENTIAL_ID,
        credential_json={},
        user_id=None,
    )
    db_session.add(credential)
    db_session.commit()


def cleanup_gmail_credentials(db_session: Session) -> None:
    """
    清理Gmail凭证
    
    Args:
        db_session: 数据库会话
    """
    gmail_credentials = fetch_credentials_by_source(
        db_session=db_session, user=None, document_source=DocumentSource.GMAIL
    )
    for credential in gmail_credentials:
        db_session.delete(credential)
    db_session.commit()


def cleanup_google_drive_credentials(db_session: Session) -> None:
    """
    清理Google Drive凭证
    
    Args:
        db_session: 数据库会话
    """
    google_drive_credentials = fetch_credentials_by_source(
        db_session=db_session, user=None, document_source=DocumentSource.GOOGLE_DRIVE
    )
    for credential in google_drive_credentials:
        db_session.delete(credential)
    db_session.commit()


def delete_service_account_credentials(
    user: User | None, db_session: Session, source: DocumentSource
) -> None:
    """
    删除服务账号凭证
    
    Args:
        user: 用户对象
        db_session: 数据库会话
        source: 数据源
    """
    credentials = fetch_credentials(db_session=db_session, user=user)
    for credential in credentials:
        if (
            credential.credential_json.get(DB_CREDENTIALS_DICT_SERVICE_ACCOUNT_KEY)
            and credential.source == source
        ):
            db_session.delete(credential)

    db_session.commit()
