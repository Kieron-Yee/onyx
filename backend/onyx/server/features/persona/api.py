"""
这个模块实现了AI助手(Persona)相关的API接口。
主要功能包括:
- AI助手的创建、更新、删除和查询
- AI助手类别的管理
- AI助手的共享和权限控制
- AI助手的提示语生成
"""

import uuid
from uuid import UUID

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import Query
from fastapi import UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session

from onyx.auth.users import current_admin_user
from onyx.auth.users import current_chat_accesssible_user
from onyx.auth.users import current_curator_or_admin_user
from onyx.auth.users import current_limited_user
from onyx.auth.users import current_user
from onyx.chat.prompt_builder.utils import build_dummy_prompt
from onyx.configs.constants import FileOrigin
from onyx.configs.constants import MilestoneRecordType
from onyx.configs.constants import NotificationType
from onyx.db.engine import get_current_tenant_id
from onyx.db.engine import get_session
from onyx.db.models import StarterMessageModel as StarterMessage
from onyx.db.models import User
from onyx.db.notification import create_notification
from onyx.db.persona import create_assistant_category
from onyx.db.persona import create_update_persona
from onyx.db.persona import delete_persona_category
from onyx.db.persona import get_assistant_categories
from onyx.db.persona import get_persona_by_id
from onyx.db.persona import get_personas
from onyx.db.persona import mark_persona_as_deleted
from onyx.db.persona import mark_persona_as_not_deleted
from onyx.db.persona import update_all_personas_display_priority
from onyx.db.persona import update_persona_category
from onyx.db.persona import update_persona_public_status
from onyx.db.persona import update_persona_shared_users
from onyx.db.persona import update_persona_visibility
from onyx.file_store.file_store import get_default_file_store
from onyx.file_store.models import ChatFileType
from onyx.secondary_llm_flows.starter_message_creation import (
    generate_starter_messages,
)
from onyx.server.features.persona.models import CreatePersonaRequest
from onyx.server.features.persona.models import GenerateStarterMessageRequest
from onyx.server.features.persona.models import ImageGenerationToolStatus
from onyx.server.features.persona.models import PersonaCategoryCreate
from onyx.server.features.persona.models import PersonaCategoryResponse
from onyx.server.features.persona.models import PersonaSharedNotificationData
from onyx.server.features.persona.models import PersonaSnapshot
from onyx.server.features.persona.models import PromptTemplateResponse
from onyx.server.models import DisplayPriorityRequest
from onyx.tools.utils import is_image_generation_available
from onyx.utils.logger import setup_logger
from onyx.utils.telemetry import create_milestone_and_report


logger = setup_logger()


admin_router = APIRouter(prefix="/admin/persona")
basic_router = APIRouter(prefix="/persona")


class IsVisibleRequest(BaseModel):
    """
    控制AI助手可见性的请求模型
    
    属性:
        is_visible (bool): 是否可见
    """
    is_visible: bool


class IsPublicRequest(BaseModel):
    """
    控制AI助手公开状态的请求模型
    
    属性:
        is_public (bool): 是否公开
    """
    is_public: bool


@admin_router.patch("/{persona_id}/visible")
def patch_persona_visibility(
    persona_id: int,
    is_visible_request: IsVisibleRequest,
    user: User | None = Depends(current_curator_or_admin_user),
    db_session: Session = Depends(get_session),
) -> None:
    """
    更新AI助手的可见性状态
    
    参数:
        persona_id: AI助手ID
        is_visible_request: 可见性请求对象
        user: 当前用户对象
        db_session: 数据库会话
    """
    update_persona_visibility(
        persona_id=persona_id,
        is_visible=is_visible_request.is_visible,
        db_session=db_session,
        user=user,
    )


@basic_router.patch("/{persona_id}/public")
def patch_user_presona_public_status(
    persona_id: int,
    is_public_request: IsPublicRequest,
    user: User | None = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> None:
    """
    更新AI助手的公开状态
    
    参数:
        persona_id: AI助手ID
        is_public_request: 公开状态请求对象
        user: 当前用户对象
        db_session: 数据库会话
        
    异常:
        HTTPException: 更新公开状态失败时抛出403错误
    """
    try:
        update_persona_public_status(
            persona_id=persona_id,
            is_public=is_public_request.is_public,
            db_session=db_session,
            user=user,
        )
    except ValueError as e:
        logger.exception("Failed to update persona public status")
        raise HTTPException(status_code=403, detail=str(e))


@admin_router.put("/display-priority")
def patch_persona_display_priority(
    display_priority_request: DisplayPriorityRequest,
    _: User | None = Depends(current_admin_user),
    db_session: Session = Depends(get_session),
) -> None:
    """
    更新所有AI助手的显示优先级
    
    参数:
        display_priority_request: 显示优先级请求对象
        _: 当前用户对象
        db_session: 数据库会话
    """
    update_all_personas_display_priority(
        display_priority_map=display_priority_request.display_priority_map,
        db_session=db_session,
    )


@admin_router.get("")
def list_personas_admin(
    user: User | None = Depends(current_curator_or_admin_user),
    db_session: Session = Depends(get_session),
    include_deleted: bool = False,
    get_editable: bool = Query(False, description="If true, return editable personas"),
) -> list[PersonaSnapshot]:
    """
    获取所有AI助手的列表（管理员接口）
    
    参数:
        user: 当前用户对象
        db_session: 数据库会话
        include_deleted: 是否包含已删除的AI助手
        get_editable: 是否只获取可编辑的AI助手
        
    返回:
        list[PersonaSnapshot]: AI助手快照列表
    """
    return [
        PersonaSnapshot.from_model(persona)
        for persona in get_personas(
            db_session=db_session,
            user=user,
            get_editable=get_editable,
            include_deleted=include_deleted,
            joinedload_all=True,
        )
    ]


@admin_router.patch("/{persona_id}/undelete")
def undelete_persona(
    persona_id: int,
    user: User | None = Depends(current_admin_user),
    db_session: Session = Depends(get_session),
) -> None:
    """
    恢复已删除的AI助手
    
    参数:
        persona_id: AI助手ID
        user: 当前用户对象
        db_session: 数据库会话
    """
    mark_persona_as_not_deleted(
        persona_id=persona_id,
        user=user,
        db_session=db_session,
    )


# used for assistat profile pictures
@admin_router.post("/upload-image")
def upload_file(
    file: UploadFile,
    db_session: Session = Depends(get_session),
    _: User | None = Depends(current_user),
) -> dict[str, str]:
    """
    上传AI助手的头像图片
    
    参数:
        file: 上传的文件对象
        db_session: 数据库会话
        _: 当前用户对象
        
    返回:
        dict[str, str]: 文件ID字典
    """
    file_store = get_default_file_store(db_session)
    file_type = ChatFileType.IMAGE
    file_id = str(uuid.uuid4())
    file_store.save_file(
        file_name=file_id,
        content=file.file,
        display_name=file.filename,
        file_origin=FileOrigin.CHAT_UPLOAD,
        file_type=file.content_type or file_type.value,
    )
    return {"file_id": file_id}


"""Endpoints for all"""


@basic_router.post("")
def create_persona(
    create_persona_request: CreatePersonaRequest,
    user: User | None = Depends(current_user),
    db_session: Session = Depends(get_session),
    tenant_id: str | None = Depends(get_current_tenant_id),
) -> PersonaSnapshot:
    """
    创建新的AI助手
    
    参数:
        create_persona_request: 创建AI助手请求对象
        user: 当前用户对象
        db_session: 数据库会话
        tenant_id: 当前租户ID
        
    返回:
        PersonaSnapshot: 创建的AI助手快照
    """
    persona_snapshot = create_update_persona(
        persona_id=None,
        create_persona_request=create_persona_request,
        user=user,
        db_session=db_session,
    )

    create_milestone_and_report(
        user=user,
        distinct_id=tenant_id or "N/A",
        event_type=MilestoneRecordType.CREATED_ASSISTANT,
        properties=None,
        db_session=db_session,
    )

    return persona_snapshot


# NOTE: This endpoint cannot update persona configuration options that
# are core to the persona, such as its display priority and
# whether or not the assistant is a built-in / default assistant
@basic_router.patch("/{persona_id}")
def update_persona(
    persona_id: int,
    update_persona_request: CreatePersonaRequest,
    user: User | None = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> PersonaSnapshot:
    """
    更新AI助手的信息
    
    参数:
        persona_id: AI助手ID
        update_persona_request: 更新AI助手请求对象
        user: 当前用户对象
        db_session: 数据库会话
        
    返回:
        PersonaSnapshot: 更新后的AI助手快照
    """
    return create_update_persona(
        persona_id=persona_id,
        create_persona_request=update_persona_request,
        user=user,
        db_session=db_session,
    )


class PersonaCategoryPatchRequest(BaseModel):
    """
    更新AI助手类别的请求模型
    
    属性:
        category_description (str): 类别描述
        category_name (str): 类别名称
    """
    category_description: str
    category_name: str


@basic_router.get("/categories")
def get_categories(
    db: Session = Depends(get_session),
    _: User | None = Depends(current_user),
) -> list[PersonaCategoryResponse]:
    """
    获取所有AI助手类别
    
    参数:
        db: 数据库会话
        _: 当前用户对象
        
    返回:
        list[PersonaCategoryResponse]: AI助手类别响应列表
    """
    return [
        PersonaCategoryResponse.from_model(category)
        for category in get_assistant_categories(db_session=db)
    ]


@admin_router.post("/categories")
def create_category(
    category: PersonaCategoryCreate,
    db: Session = Depends(get_session),
    _: User | None = Depends(current_admin_user),
) -> PersonaCategoryResponse:
    """
    创建新的AI助手类别
    
    参数:
        category: 创建类别请求对象
        db: 数据库会话
        _: 当前用户对象
        
    返回:
        PersonaCategoryResponse: 创建的AI助手类别响应
    """
    category_model = create_assistant_category(
        name=category.name, description=category.description, db_session=db
    )
    return PersonaCategoryResponse.from_model(category_model)


@admin_router.patch("/category/{category_id}")
def patch_persona_category(
    category_id: int,
    persona_category_patch_request: PersonaCategoryPatchRequest,
    _: User | None = Depends(current_admin_user),
    db_session: Session = Depends(get_session),
) -> None:
    """
    更新AI助手类别信息
    
    参数:
        category_id: 类别ID
        persona_category_patch_request: 更新类别请求对象
        _: 当前用户对象
        db_session: 数据库会话
    """
    update_persona_category(
        category_id=category_id,
        category_description=persona_category_patch_request.category_description,
        category_name=persona_category_patch_request.category_name,
        db_session=db_session,
    )


@admin_router.delete("/category/{category_id}")
def delete_category(
    category_id: int,
    _: User | None = Depends(current_admin_user),
    db_session: Session = Depends(get_session),
) -> None:
    """
    删除AI助手类别
    
    参数:
        category_id: 类别ID
        _: 当前用户对象
        db_session: 数据库会话
    """
    delete_persona_category(category_id=category_id, db_session=db_session)


class PersonaShareRequest(BaseModel):
    """
    共享AI助手的请求模型
    
    属性:
        user_ids (list[UUID]): 用户ID列表
    """
    user_ids: list[UUID]


# We notify each user when a user is shared with them
@basic_router.patch("/{persona_id}/share")
def share_persona(
    persona_id: int,
    persona_share_request: PersonaShareRequest,
    user: User = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> None:
    """
    共享AI助手给指定用户
    
    参数:
        persona_id: AI助手ID
        persona_share_request: 共享请求对象
        user: 当前用户对象
        db_session: 数据库会话
    """
    update_persona_shared_users(
        persona_id=persona_id,
        user_ids=persona_share_request.user_ids,
        user=user,
        db_session=db_session,
    )

    for user_id in persona_share_request.user_ids:
        # Don't notify the user that they have access to their own persona
        if user_id != user.id:
            create_notification(
                user_id=user_id,
                notif_type=NotificationType.PERSONA_SHARED,
                db_session=db_session,
                additional_data=PersonaSharedNotificationData(
                    persona_id=persona_id,
                ).model_dump(),
            )


@basic_router.delete("/{persona_id}")
def delete_persona(
    persona_id: int,
    user: User | None = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> None:
    """
    删除AI助手
    
    参数:
        persona_id: AI助手ID
        user: 当前用户对象
        db_session: 数据库会话
    """
    mark_persona_as_deleted(
        persona_id=persona_id,
        user=user,
        db_session=db_session,
    )


@basic_router.get("/image-generation-tool")
def get_image_generation_tool(
    _: User
    | None = Depends(current_user),  # User param not used but kept for consistency
    db_session: Session = Depends(get_session),
) -> ImageGenerationToolStatus:  # Use bool instead of str for boolean values
    """
    获取图像生成工具的状态
    
    参数:
        _: 当前用户对象
        db_session: 数据库会话
        
    返回:
        ImageGenerationToolStatus: 图像生成工具状态
    """
    is_available = is_image_generation_available(db_session=db_session)
    return ImageGenerationToolStatus(is_available=is_available)


@basic_router.get("")
def list_personas(
    user: User | None = Depends(current_chat_accesssible_user),
    db_session: Session = Depends(get_session),
    include_deleted: bool = False,
    persona_ids: list[int] = Query(None),
) -> list[PersonaSnapshot]:
    """
    获取所有AI助手的列表
    
    参数:
        user: 当前用户对象
        db_session: 数据库会话
        include_deleted: 是否包含已删除的AI助手
        persona_ids: 指定的AI助手ID列表
        
    返回:
        list[PersonaSnapshot]: AI助手快照列表
    """
    personas = get_personas(
        user=user,
        include_deleted=include_deleted,
        db_session=db_session,
        get_editable=False,
        joinedload_all=True,
    )

    if persona_ids:
        personas = [p for p in personas if p.id in persona_ids]

    # Filter out personas with unavailable tools
    personas = [
        p
        for p in personas
        if not (
            any(tool.in_code_tool_id == "ImageGenerationTool" for tool in p.tools)
            and not is_image_generation_available(db_session=db_session)
        )
    ]

    return [PersonaSnapshot.from_model(p) for p in personas]


@basic_router.get("/{persona_id}")
def get_persona(
    persona_id: int,
    user: User | None = Depends(current_limited_user),
    db_session: Session = Depends(get_session),
) -> PersonaSnapshot:
    """
    获取指定ID的AI助手信息
    
    参数:
        persona_id: AI助手ID
        user: 当前用户对象
        db_session: 数据库会话
        
    返回:
        PersonaSnapshot: AI助手快照
    """
    return PersonaSnapshot.from_model(
        get_persona_by_id(
            persona_id=persona_id,
            user=user,
            db_session=db_session,
            is_for_edit=False,
        )
    )


@basic_router.get("/utils/prompt-explorer")
def build_final_template_prompt(
    system_prompt: str,
    task_prompt: str,
    retrieval_disabled: bool = False,
    _: User | None = Depends(current_user),
) -> PromptTemplateResponse:
    """
    构建最终的提示模板
    
    参数:
        system_prompt: 系统提示语
        task_prompt: 任务提示语
        retrieval_disabled: 是否禁用检索
        _: 当前用户对象
        
    返回:
        PromptTemplateResponse: 提示模板响应
    """
    return PromptTemplateResponse(
        final_prompt_template=build_dummy_prompt(
            system_prompt=system_prompt,
            task_prompt=task_prompt,
            retrieval_disabled=retrieval_disabled,
        )
    )


@basic_router.post("/assistant-prompt-refresh")
def build_assistant_prompts(
    generate_persona_prompt_request: GenerateStarterMessageRequest,
    db_session: Session = Depends(get_session),
    user: User | None = Depends(current_user),
) -> list[StarterMessage]:
    """
    生成AI助手的初始对话提示
    
    参数:
        generate_persona_prompt_request: 生成提示语请求对象
        db_session: 数据库会话
        user: 当前用户对象
        
    返回:
        list[StarterMessage]: 生成的初始对话提示列表
        
    异常:
        HTTPException: 生成提示语失败时抛出500错误
    """
    try:
        logger.info(
            "Generating starter messages for user: %s", # 正在为用户生成初始消息
            user.id if user else "Anonymous"
        )
        return generate_starter_messages(
            name=generate_persona_prompt_request.name,
            description=generate_persona_prompt_request.description,
            instructions=generate_persona_prompt_request.instructions,
            document_set_ids=generate_persona_prompt_request.document_set_ids,
            db_session=db_session,
            user=user,
        )
    except Exception as e:
        logger.exception("Failed to generate starter messages") # 生成初始消息失败
        raise HTTPException(status_code=500, detail=str(e))
