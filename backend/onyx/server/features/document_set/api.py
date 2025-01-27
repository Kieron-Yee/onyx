"""
本模块提供了文档集(Document Set)相关的API接口
主要功能：
1. 提供文档集的创建、更新、删除等管理功能
2. 提供文档集的查询和访问控制功能
3. 处理普通用户和管理员的不同权限请求
"""

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import Query
from sqlalchemy.orm import Session

from onyx.auth.users import current_curator_or_admin_user
from onyx.auth.users import current_user
from onyx.db.document_set import check_document_sets_are_public
from onyx.db.document_set import fetch_all_document_sets_for_user
from onyx.db.document_set import insert_document_set
from onyx.db.document_set import mark_document_set_as_to_be_deleted
from onyx.db.document_set import update_document_set
from onyx.db.engine import get_session
from onyx.db.models import User
from onyx.server.features.document_set.models import CheckDocSetPublicRequest
from onyx.server.features.document_set.models import CheckDocSetPublicResponse
from onyx.server.features.document_set.models import DocumentSet
from onyx.server.features.document_set.models import DocumentSetCreationRequest
from onyx.server.features.document_set.models import DocumentSetUpdateRequest
from onyx.utils.variable_functionality import fetch_ee_implementation_or_noop


router = APIRouter(prefix="/manage")


@router.post("/admin/document-set")
def create_document_set(
    document_set_creation_request: DocumentSetCreationRequest,
    user: User = Depends(current_curator_or_admin_user),
    db_session: Session = Depends(get_session),
) -> int:
    """
    创建新的文档集
    
    参数:
        document_set_creation_request: 文档集创建请求对象
        user: 当前用户（必须是管理员或策展人）
        db_session: 数据库会话
    
    返回:
        int: 新创建的文档集ID
        
    异常:
        HTTPException: 创建失败时抛出400错误
    """
    fetch_ee_implementation_or_noop(
        "onyx.db.user_group", "validate_object_creation_for_user", None
    )(
        db_session=db_session,
        user=user,
        target_group_ids=document_set_creation_request.groups,
        object_is_public=document_set_creation_request.is_public,
    )
    try:
        document_set_db_model, _ = insert_document_set(
            document_set_creation_request=document_set_creation_request,
            user_id=user.id if user else None,
            db_session=db_session,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return document_set_db_model.id


@router.patch("/admin/document-set")
def patch_document_set(
    document_set_update_request: DocumentSetUpdateRequest,
    user: User = Depends(current_curator_or_admin_user),
    db_session: Session = Depends(get_session),
) -> None:
    """
    更新现有文档集
    
    参数:
        document_set_update_request: 文档集更新请求对象
        user: 当前用户（必须是管理员或策展人）
        db_session: 数据库会话
    
    异常:
        HTTPException: 更新失败时抛出400错误
    """
    fetch_ee_implementation_or_noop(
        "onyx.db.user_group", "validate_object_creation_for_user", None
    )(
        db_session=db_session,
        user=user,
        target_group_ids=document_set_update_request.groups,
        object_is_public=document_set_update_request.is_public,
    )
    try:
        update_document_set(
            document_set_update_request=document_set_update_request,
            db_session=db_session,
            user=user,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/admin/document-set/{document_set_id}")
def delete_document_set(
    document_set_id: int,
    user: User = Depends(current_curator_or_admin_user),
    db_session: Session = Depends(get_session),
) -> None:
    """
    删除指定的文档集
    
    参数:
        document_set_id: 要删除的文档集ID
        user: 当前用户（必须是管理员或策展人）
        db_session: 数据库会话
    
    异常:
        HTTPException: 删除失败时抛出400错误
    """
    try:
        mark_document_set_as_to_be_deleted(
            db_session=db_session,
            document_set_id=document_set_id,
            user=user,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


"""非管理员端点 Endpoints for non-admins"""


@router.get("/document-set")
def list_document_sets_for_user(
    user: User | None = Depends(current_user),
    db_session: Session = Depends(get_session),
    get_editable: bool = Query(
        False, description="If true, return editable document sets"  # 如果为true，返回可编辑的文档集
    ),
) -> list[DocumentSet]:
    """
    获取用户可访问的文档集列表
    
    参数:
        user: 当前用户
        db_session: 数据库会话
        get_editable: 是否只返回可编辑的文档集
    
    返回:
        list[DocumentSet]: 文档集列表
    """
    return [
        DocumentSet.from_model(ds)
        for ds in fetch_all_document_sets_for_user(
            db_session=db_session, user=user, get_editable=get_editable
        )
    ]


@router.get("/document-set-public")
def document_set_public(
    check_public_request: CheckDocSetPublicRequest,
    _: User = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> CheckDocSetPublicResponse:
    """
    检查文档集是否公开可访问
    
    参数:
        check_public_request: 检查公开性请求对象
        _: 当前用户（用于验证权限）
        db_session: 数据库会话
    
    返回:
        CheckDocSetPublicResponse: 包含是否公开的响应对象
    """
    is_public = check_document_sets_are_public(
        document_set_ids=check_public_request.document_set_ids, db_session=db_session
    )
    return CheckDocSetPublicResponse(is_public=is_public)
