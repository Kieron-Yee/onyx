"""
此文件实现了OpenAI Assistants API的后端接口。
提供了创建、获取、修改、删除和列出AI助手的功能。
这些接口遵循OpenAI的API规范，但实际使用内部的Persona系统来实现功能。
"""

from typing import Any
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from onyx.auth.users import current_user
from onyx.context.search.enums import RecencyBiasSetting
from onyx.db.engine import get_session
from onyx.db.models import Persona
from onyx.db.models import User
from onyx.db.persona import get_persona_by_id
from onyx.db.persona import get_personas
from onyx.db.persona import mark_persona_as_deleted
from onyx.db.persona import upsert_persona
from onyx.db.persona import upsert_prompt
from onyx.db.tools import get_tool_by_name
from onyx.utils.logger import setup_logger

logger = setup_logger()


router = APIRouter(prefix="/assistants")


# 基础模型 Base models
class AssistantObject(BaseModel):
    """
    AI助手对象的数据模型
    
    属性:
        id: 助手唯一标识符
        object: 对象类型，固定为"assistant"
        created_at: 创建时间戳
        name: 助手名称
        description: 助手描述
        model: 使用的语言模型
        instructions: 助手指令
        tools: 可用工具列表
        file_ids: 关联文件ID列表
        metadata: 元数据
    """
    id: int
    object: str = "assistant"
    created_at: int
    name: Optional[str] = None
    description: Optional[str] = None
    model: str
    instructions: Optional[str] = None
    tools: list[dict[str, Any]]
    file_ids: list[str]
    metadata: Optional[dict[str, Any]] = None


class CreateAssistantRequest(BaseModel):
    """
    创建助手的请求模型
    
    属性:
        model: 要使用的语言模型
        name: 助手名称（可选）
        description: 助手描述（可选）
        instructions: 助手指令（可选）
        tools: 工具配置列表（可选）
        file_ids: 关联文件ID列表（可选）
        metadata: 元数据（可选）
    """
    model: str
    name: Optional[str] = None
    description: Optional[str] = None
    instructions: Optional[str] = None
    tools: Optional[list[dict[str, Any]]] = None
    file_ids: Optional[list[str]] = None
    metadata: Optional[dict[str, Any]] = None


class ModifyAssistantRequest(BaseModel):
    model: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    instructions: Optional[str] = None
    tools: Optional[list[dict[str, Any]]] = None
    file_ids: Optional[list[str]] = None
    metadata: Optional[dict[str, Any]] = None


class DeleteAssistantResponse(BaseModel):
    id: int
    object: str = "assistant.deleted"
    deleted: bool


class ListAssistantsResponse(BaseModel):
    object: str = "list"
    data: list[AssistantObject]
    first_id: Optional[int] = None
    last_id: Optional[int] = None
    has_more: bool


def persona_to_assistant(persona: Persona) -> AssistantObject:
    """
    将Persona对象转换为AssistantObject
    
    参数:
        persona: Persona实例
        
    返回:
        AssistantObject实例，包含从Persona转换的所有相关信息
    """
    return AssistantObject(
        id=persona.id,
        created_at=0,
        name=persona.name,
        description=persona.description,
        model=persona.llm_model_version_override or "gpt-3.5-turbo",
        instructions=persona.prompts[0].system_prompt if persona.prompts else None,
        tools=[
            {
                "type": tool.display_name,
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "schema": tool.openapi_schema,
                },
            }
            for tool in persona.tools
        ],
        file_ids=[],  # 假设目前不支持文件
        metadata={},  # 假设目前没有元数据
    )


# API endpoints
@router.post("")
def create_assistant(
    request: CreateAssistantRequest,
    user: User | None = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> AssistantObject:
    """
    创建新的AI助手
    
    参数:
        request: 创建助手的请求数据
        user: 当前用户
        db_session: 数据库会话
        
    返回:
        新创建的AssistantObject实例
        
    异常:
        HTTPException: 当工具不存在时抛出404错误
    """
    prompt = None
    if request.instructions:
        prompt = upsert_prompt(
            user=user,
            name=f"Prompt for {request.name or 'New Assistant'}",
            description="Auto-generated prompt",
            system_prompt=request.instructions,
            task_prompt="",
            include_citations=True,
            datetime_aware=True,
            personas=[],
            db_session=db_session,
        )

    tool_ids = []
    for tool in request.tools or []:
        tool_type = tool.get("type")
        if not tool_type:
            continue

        try:
            tool_db = get_tool_by_name(tool_type, db_session)
            tool_ids.append(tool_db.id)
        except ValueError:
            # 跳过数据库中不存在的工具
            logger.error(f"Tool {tool_type} not found in database")
            raise HTTPException(
                status_code=404, detail=f"Tool {tool_type} not found in database"
            )

    persona = upsert_persona(
        user=user,
        name=request.name or f"Assistant-{uuid4()}",
        description=request.description or "",
        num_chunks=25,
        llm_relevance_filter=True,
        llm_filter_extraction=True,
        recency_bias=RecencyBiasSetting.AUTO,
        llm_model_provider_override=None,
        llm_model_version_override=request.model,
        starter_messages=None,
        is_public=False,
        db_session=db_session,
        prompt_ids=[prompt.id] if prompt else [0],
        document_set_ids=[],
        tool_ids=tool_ids,
        icon_color=None,
        icon_shape=None,
        is_visible=True,
    )

    if prompt:
        prompt.personas = [persona]
        db_session.commit()

    return persona_to_assistant(persona)


@router.get("/{assistant_id}")
def retrieve_assistant(
    assistant_id: int,
    user: User | None = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> AssistantObject:
    """
    获取指定ID的助手信息
    
    参数:
        assistant_id: 助手ID
        user: 当前用户
        db_session: 数据库会话
        
    返回:
        AssistantObject实例
        
    异常:
        HTTPException: 当助手不存在时抛出404错误
    """
    try:
        persona = get_persona_by_id(
            persona_id=assistant_id,
            user=user,
            db_session=db_session,
            is_for_edit=False,
        )
    except ValueError:
        persona = None

    if not persona:
        raise HTTPException(status_code=404, detail="Assistant not found")
    return persona_to_assistant(persona)


@router.post("/{assistant_id}")
def modify_assistant(
    assistant_id: int,
    request: ModifyAssistantRequest,
    user: User | None = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> AssistantObject:
    """
    修改指定ID的助手信息
    
    参数:
        assistant_id: 助手ID
        request: 修改请求数据
        user: 当前用户
        db_session: 数据库会话
        
    返回:
        更新后的AssistantObject实例
        
    异常:
        HTTPException: 当助手不存在时抛出404错误
    """
    persona = get_persona_by_id(
        persona_id=assistant_id,
        user=user,
        db_session=db_session,
        is_for_edit=True,
    )
    if not persona:
        raise HTTPException(status_code=404, detail="Assistant not found")

    update_data = request.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(persona, key, value)

    if "instructions" in update_data and persona.prompts:
        persona.prompts[0].system_prompt = update_data["instructions"]

    db_session.commit()
    return persona_to_assistant(persona)


@router.delete("/{assistant_id}")
def delete_assistant(
    assistant_id: int,
    user: User | None = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> DeleteAssistantResponse:
    """
    删除指定ID的助手
    
    参数:
        assistant_id: 助手ID
        user: 当前用户
        db_session: 数据库会话
        
    返回:
        删除操作的响应对象
        
    异常:
        HTTPException: 当助手不存在时抛出404错误
    """
    try:
        mark_persona_as_deleted(
            persona_id=int(assistant_id),
            user=user,
            db_session=db_session,
        )
        return DeleteAssistantResponse(id=assistant_id, deleted=True)
    except ValueError:
        raise HTTPException(status_code=404, detail="Assistant not found")


@router.get("")
def list_assistants(
    limit: int = Query(20, le=100),
    order: str = Query("desc", regex="^(asc|desc)$"),
    after: Optional[int] = None,
    before: Optional[int] = None,
    user: User | None = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> ListAssistantsResponse:
    """
    列出所有可用的助手
    
    参数:
        limit: 返回结果的最大数量，默认20，最大100
        order: 排序方式，"asc"升序或"desc"降序
        after: 只返回ID大于此值的助手
        before: 只返回ID小于此值的助手
        user: 当前用户
        db_session: 数据库会话
        
    返回:
        包含助手列表的响应对象
    """
    personas = list(
        get_personas(
            user=user,
            db_session=db_session,
            get_editable=False,
            joinedload_all=True,
        )
    )

    # 根据after和before进行过滤
    if after:
        personas = [p for p in personas if p.id > int(after)]
    if before:
        personas = [p for p in personas if p.id < int(before)]

    # 应用排序
    personas.sort(key=lambda p: p.id, reverse=(order == "desc"))

    # 应用限制
    personas = personas[:limit]

    assistants = [persona_to_assistant(p) for p in personas]

    return ListAssistantsResponse(
        data=assistants,
        first_id=assistants[0].id if assistants else None,
        last_id=assistants[-1].id if assistants else None,
        has_more=len(personas) == limit,
    )
