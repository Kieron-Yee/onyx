"""
此文件实现了凭证管理相关的API路由功能。
主要包括：
1. 凭证的创建、读取、更新和删除操作
2. 管理员专用的凭证管理接口
3. 普通用户的凭证管理接口
"""

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import Query
from sqlalchemy.orm import Session

from onyx.auth.users import current_admin_user
from onyx.auth.users import current_curator_or_admin_user
from onyx.auth.users import current_user
from onyx.db.credentials import alter_credential
from onyx.db.credentials import cleanup_gmail_credentials
from onyx.db.credentials import cleanup_google_drive_credentials
from onyx.db.credentials import create_credential
from onyx.db.credentials import CREDENTIAL_PERMISSIONS_TO_IGNORE
from onyx.db.credentials import delete_credential
from onyx.db.credentials import fetch_credential_by_id
from onyx.db.credentials import fetch_credentials
from onyx.db.credentials import fetch_credentials_by_source
from onyx.db.credentials import swap_credentials_connector
from onyx.db.credentials import update_credential
from onyx.db.engine import get_session
from onyx.db.models import DocumentSource
from onyx.db.models import User
from onyx.server.documents.models import CredentialBase
from onyx.server.documents.models import CredentialDataUpdateRequest
from onyx.server.documents.models import CredentialSnapshot
from onyx.server.documents.models import CredentialSwapRequest
from onyx.server.documents.models import ObjectCreationIdResponse
from onyx.server.models import StatusResponse
from onyx.utils.logger import setup_logger
from onyx.utils.variable_functionality import fetch_ee_implementation_or_noop

# 创建日志记录器
logger = setup_logger()

# 创建路由器，设置前缀为"/manage"
router = APIRouter(prefix="/manage")


def _ignore_credential_permissions(source: DocumentSource) -> bool:
    """
    检查是否需要忽略指定来源的凭证权限

    参数:
        source: 文档来源类型

    返回:
        bool: 如果需要忽略权限则返回True，否则返回False
    """
    return source in CREDENTIAL_PERMISSIONS_TO_IGNORE


"""Admin-only endpoints 管理员专用端点"""


@router.get("/admin/credential")
def list_credentials_admin(
    user: User | None = Depends(current_curator_or_admin_user),
    db_session: Session = Depends(get_session),
) -> list[CredentialSnapshot]:
    """
    Lists all public credentials
    列出所有公共凭证

    参数:
        user: 当前管理员或策展人用户
        db_session: 数据库会话

    返回:
        list[CredentialSnapshot]: 凭证快照列表
    """
    credentials = fetch_credentials(
        db_session=db_session,
        user=user,
        get_editable=False,
    )
    return [
        CredentialSnapshot.from_credential_db_model(credential)
        for credential in credentials
    ]


@router.get("/admin/similar-credentials/{source_type}")
def get_cc_source_full_info(
    source_type: DocumentSource,
    user: User | None = Depends(current_curator_or_admin_user),
    db_session: Session = Depends(get_session),
    get_editable: bool = Query(
        False, description="If true, return editable credentials"
    ),
) -> list[CredentialSnapshot]:
    """
    获取指定来源类型的所有凭证信息

    参数:
        source_type: 文档来源类型
        user: 当前管理员或策展人用户
        db_session: 数据库会话
        get_editable: 是否返回可编辑的凭证

    返回:
        list[CredentialSnapshot]: 凭证快照列表
    """
    credentials = fetch_credentials_by_source(
        db_session=db_session,
        user=user,
        document_source=source_type,
        get_editable=get_editable,
    )
    return [
        CredentialSnapshot.from_credential_db_model(credential)
        for credential in credentials
    ]


@router.delete("/admin/credential/{credential_id}")
def delete_credential_by_id_admin(
    credential_id: int,
    _: User = Depends(current_admin_user),
    db_session: Session = Depends(get_session),
) -> StatusResponse:
    """
    Same as the user endpoint, but can delete any credential (not just the user's own)
    与用户端点相同，但可以删除任何凭证（不仅仅是用户自己的）

    参数:
        credential_id: 要删除的凭证ID
        _: 当前管理员用户
        db_session: 数据库会话

    返回:
        StatusResponse: 删除操作的状态响应
    """
    delete_credential(db_session=db_session, credential_id=credential_id, user=None)
    return StatusResponse(
        success=True, message="Credential deleted successfully", data=credential_id
    )


@router.put("/admin/credential/swap")
def swap_credentials_for_connector(
    credential_swap_req: CredentialSwapRequest,
    user: User | None = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> StatusResponse:
    """
    交换连接器的凭证

    参数:
        credential_swap_req: 凭证交换请求
        user: 当前用户
        db_session: 数据库会话

    返回:
        StatusResponse: 交换操作的状态响应
    """
    connector_credential_pair = swap_credentials_connector(
        new_credential_id=credential_swap_req.new_credential_id,
        connector_id=credential_swap_req.connector_id,
        db_session=db_session,
        user=user,
    )

    return StatusResponse(
        success=True,
        message="Credential swapped successfully",
        data=connector_credential_pair.id,
    )


@router.post("/credential")
def create_credential_from_model(
    credential_info: CredentialBase,
    user: User | None = Depends(current_curator_or_admin_user),
    db_session: Session = Depends(get_session),
) -> ObjectCreationIdResponse:
    """
    从模型创建凭证

    参数:
        credential_info: 凭证基本信息
        user: 当前管理员或策展人用户
        db_session: 数据库会话

    返回:
        ObjectCreationIdResponse: 创建操作的响应，包含新创建的凭证ID
    """
    if not _ignore_credential_permissions(credential_info.source):
        fetch_ee_implementation_or_noop(
            "onyx.db.user_group", "validate_object_creation_for_user", None
        )(
            db_session=db_session,
            user=user,
            target_group_ids=credential_info.groups,
            object_is_public=credential_info.curator_public,
        )

    # Temporary fix for empty Google App credentials
    # 临时修复空的Google App凭证
    if credential_info.source == DocumentSource.GMAIL:
        cleanup_gmail_credentials(db_session=db_session)
    if credential_info.source == DocumentSource.GOOGLE_DRIVE:
        cleanup_google_drive_credentials(db_session=db_session)

    credential = create_credential(credential_info, user, db_session)
    return ObjectCreationIdResponse(
        id=credential.id,
        credential=CredentialSnapshot.from_credential_db_model(credential),
    )


"""Endpoints for all 所有用户的端点"""


@router.get("/credential")
def list_credentials(
    user: User | None = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> list[CredentialSnapshot]:
    """
    列出所有凭证

    参数:
        user: 当前用户
        db_session: 数据库会话

    返回:
        list[CredentialSnapshot]: 凭证快照列表
    """
    credentials = fetch_credentials(db_session=db_session, user=user)
    return [
        CredentialSnapshot.from_credential_db_model(credential)
        for credential in credentials
    ]


@router.get("/credential/{credential_id}")
def get_credential_by_id(
    credential_id: int,
    user: User = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> CredentialSnapshot | StatusResponse[int]:
    """
    根据ID获取凭证

    参数:
        credential_id: 凭证ID
        user: 当前用户
        db_session: 数据库会话

    返回:
        CredentialSnapshot | StatusResponse[int]: 凭证快照或状态响应
    """
    credential = fetch_credential_by_id(
        credential_id,
        user,
        db_session,
        get_editable=False,
    )
    if credential is None:
        raise HTTPException(
            status_code=401,
            detail=f"Credential {credential_id} does not exist or does not belong to user",
        )

    return CredentialSnapshot.from_credential_db_model(credential)


@router.put("/admin/credential/{credential_id}")
def update_credential_data(
    credential_id: int,
    credential_update: CredentialDataUpdateRequest,
    user: User = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> CredentialBase:
    """
    更新凭证数据

    参数:
        credential_id: 凭证ID
        credential_update: 凭证数据更新请求
        user: 当前用户
        db_session: 数据库会话

    返回:
        CredentialBase: 更新后的凭证基本信息
    """
    credential = alter_credential(
        credential_id,
        credential_update.name,
        credential_update.credential_json,
        user,
        db_session,
    )

    if credential is None:
        raise HTTPException(
            status_code=401,
            detail=f"Credential {credential_id} does not exist or does not belong to user",
        )

    return CredentialSnapshot.from_credential_db_model(credential)


@router.patch("/credential/{credential_id}")
def update_credential_from_model(
    credential_id: int,
    credential_data: CredentialBase,
    user: User = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> CredentialSnapshot | StatusResponse[int]:
    """
    从模型更新凭证

    参数:
        credential_id: 凭证ID
        credential_data: 凭证基本信息
        user: 当前用户
        db_session: 数据库会话

    返回:
        CredentialSnapshot | StatusResponse[int]: 更新后的凭证快照或状态响应
    """
    updated_credential = update_credential(
        credential_id, credential_data, user, db_session
    )
    if updated_credential is None:
        raise HTTPException(
            status_code=401,
            detail=f"Credential {credential_id} does not exist or does not belong to user",
        )

    return CredentialSnapshot(
        source=updated_credential.source,
        id=updated_credential.id,
        credential_json=updated_credential.credential_json,
        user_id=updated_credential.user_id,
        name=updated_credential.name,
        admin_public=updated_credential.admin_public,
        time_created=updated_credential.time_created,
        time_updated=updated_credential.time_updated,
        curator_public=updated_credential.curator_public,
    )


@router.delete("/credential/{credential_id}")
def delete_credential_by_id(
    credential_id: int,
    user: User = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> StatusResponse:
    """
    根据ID删除凭证

    参数:
        credential_id: 凭证ID
        user: 当前用户
        db_session: 数据库会话

    返回:
        StatusResponse: 删除操作的状态响应
    """
    delete_credential(
        credential_id,
        user,
        db_session,
    )

    return StatusResponse(
        success=True, message="Credential deleted successfully", data=credential_id
    )


@router.delete("/credential/force/{credential_id}")
def force_delete_credential_by_id(
    credential_id: int,
    user: User = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> StatusResponse:
    """
    强制删除指定ID的凭证

    参数:
        credential_id: 要删除的凭证ID
        user: 当前用户
        db_session: 数据库会话

    返回:
        StatusResponse: 删除操作的状态响应
    """
    delete_credential(credential_id, user, db_session, True)

    return StatusResponse(
        success=True, message="Credential deleted successfully", data=credential_id
    )
