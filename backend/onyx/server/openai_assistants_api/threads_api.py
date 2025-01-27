"""
此模块实现了OpenAI Assistants API中的Threads相关功能。
主要提供了对话线程的创建、获取、修改、删除等操作接口。
"""

from typing import Optional
from uuid import UUID

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from onyx.auth.users import current_user
from onyx.db.chat import create_chat_session
from onyx.db.chat import delete_chat_session
from onyx.db.chat import get_chat_session_by_id
from onyx.db.chat import get_chat_sessions_by_user
from onyx.db.chat import update_chat_session
from onyx.db.engine import get_session
from onyx.db.models import User
from onyx.server.query_and_chat.models import ChatSessionDetails
from onyx.server.query_and_chat.models import ChatSessionsResponse

router = APIRouter(prefix="/threads")


# 模型定义
class Thread(BaseModel):
    """
    对话线程模型
    
    属性:
        id: 线程唯一标识符
        object: 对象类型，固定为"thread"
        created_at: 创建时间戳
        metadata: 可选的元数据字典
    """
    id: UUID
    object: str = "thread"
    created_at: int
    metadata: Optional[dict[str, str]] = None


class CreateThreadRequest(BaseModel):
    """
    创建线程请求模型
    
    属性:
        messages: 可选的消息列表
        metadata: 可选的元数据字典
    """
    messages: Optional[list[dict]] = None
    metadata: Optional[dict[str, str]] = None


class ModifyThreadRequest(BaseModel):
    """
    修改线程请求模型
    
    属性:
        metadata: 可选的元数据字典
    """
    metadata: Optional[dict[str, str]] = None


# API 端点实现
@router.post("")
def create_thread(
    request: CreateThreadRequest,
    user: User | None = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> Thread:
    """
    创建新的对话线程
    
    参数:
        request: 创建线程的请求数据
        user: 当前用户对象
        db_session: 数据库会话
    
    返回:
        Thread: 新创建的线程对象
    """
    user_id = user.id if user else None
    new_chat_session = create_chat_session(
        db_session=db_session,
        description="",  # Leave the naming till later to prevent delay
        user_id=user_id,
        persona_id=0,
    )

    return Thread(
        id=new_chat_session.id,
        created_at=int(new_chat_session.time_created.timestamp()),
        metadata=request.metadata,
    )


@router.get("/{thread_id}")
def retrieve_thread(
    thread_id: UUID,
    user: User | None = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> Thread:
    """
    获取指定ID的对话线程
    
    参数:
        thread_id: 线程ID
        user: 当前用户对象
        db_session: 数据库会话
    
    返回:
        Thread: 获取到的线程对象
    
    异常:
        HTTPException: 当线程未找到时抛出404错误
    """
    user_id = user.id if user else None
    try:
        chat_session = get_chat_session_by_id(
            chat_session_id=thread_id,
            user_id=user_id,
            db_session=db_session,
        )
    except ValueError:
        raise HTTPException(status_code=404, detail="Thread not found")

    return Thread(
        id=chat_session.id,
        created_at=int(chat_session.time_created.timestamp()),
        metadata=None,  # Assuming we don't store metadata in our current implementation
    )


@router.post("/{thread_id}")
def modify_thread(
    thread_id: UUID,
    request: ModifyThreadRequest,
    user: User | None = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> Thread:
    """
    修改指定ID的对话线程
    
    参数:
        thread_id: 线程ID
        request: 修改线程的请求数据
        user: 当前用户对象
        db_session: 数据库会话
    
    返回:
        Thread: 修改后的线程对象
    
    异常:
        HTTPException: 当线程未找到时抛出404错误
    """
    user_id = user.id if user else None
    try:
        chat_session = update_chat_session(
            db_session=db_session,
            user_id=user_id,
            chat_session_id=thread_id,
            description=None,  # Not updating description
            sharing_status=None,  # Not updating sharing status
        )
    except ValueError:
        raise HTTPException(status_code=404, detail="Thread not found")

    return Thread(
        id=chat_session.id,
        created_at=int(chat_session.time_created.timestamp()),
        metadata=request.metadata,
    )


@router.delete("/{thread_id}")
def delete_thread(
    thread_id: UUID,
    user: User | None = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> dict:
    """
    删除指定ID的对话线程
    
    参数:
        thread_id: 线程ID
        user: 当前用户对象
        db_session: 数据库会话
    
    返回:
        dict: 包含删除结果的字典
    
    异常:
        HTTPException: 当线程未找到时抛出404错误
    """
    user_id = user.id if user else None
    try:
        delete_chat_session(
            user_id=user_id,
            chat_session_id=thread_id,
            db_session=db_session,
        )
    except ValueError:
        raise HTTPException(status_code=404, detail="Thread not found")

    return {"id": str(thread_id), "object": "thread.deleted", "deleted": True}


@router.get("")
def list_threads(
    user: User | None = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> ChatSessionsResponse:
    """
    获取用户的所有对话线程列表
    
    参数:
        user: 当前用户对象
        db_session: 数据库会话
    
    返回:
        ChatSessionsResponse: 包含对话会话列表的响应对象
    """
    user_id = user.id if user else None
    chat_sessions = get_chat_sessions_by_user(
        user_id=user_id,
        deleted=False,
        db_session=db_session,
    )

    return ChatSessionsResponse(
        sessions=[
            ChatSessionDetails(
                id=chat.id,
                name=chat.description,
                persona_id=chat.persona_id,
                time_created=chat.time_created.isoformat(),
                shared_status=chat.shared_status,
                folder_id=chat.folder_id,
                current_alternate_model=chat.current_alternate_model,
            )
            for chat in chat_sessions
        ]
    )
