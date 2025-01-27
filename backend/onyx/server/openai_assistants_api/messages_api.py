"""
该模块实现了OpenAI Assistants API中的消息相关接口
主要功能包括：
- 创建新消息
- 获取消息列表
- 获取单个消息
- 修改消息元数据
"""

import uuid
from datetime import datetime
from typing import Any
from typing import Literal
from typing import Optional

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from pydantic import BaseModel
from pydantic import Field
from sqlalchemy.orm import Session

from onyx.auth.users import current_user
from onyx.configs.constants import MessageType
from onyx.db.chat import create_new_chat_message
from onyx.db.chat import get_chat_message
from onyx.db.chat import get_chat_messages_by_session
from onyx.db.chat import get_chat_session_by_id
from onyx.db.chat import get_or_create_root_message
from onyx.db.engine import get_session
from onyx.db.models import User
from onyx.llm.utils import check_number_of_tokens

router = APIRouter(prefix="")


Role = Literal["user", "assistant"]


class MessageContent(BaseModel):
    """
    消息内容模型
    
    属性:
        type: 消息类型，目前仅支持文本类型
        text: 消息文本内容
    """
    type: Literal["text"]
    text: str


class Message(BaseModel):
    """
    消息模型
    
    属性:
        id: 消息唯一标识符
        object: 对象类型，固定为"thread.message"
        created_at: 消息创建时间戳
        thread_id: 会话线程ID
        role: 消息角色（用户或助手）
        content: 消息内容列表
        file_ids: 关联的文件ID列表
        assistant_id: 助手ID（可选）
        run_id: 运行ID（可选）
        metadata: 元数据字典（可选）
    """
    id: str = Field(default_factory=lambda: f"msg_{uuid.uuid4()}")
    object: Literal["thread.message"] = "thread.message"
    created_at: int = Field(default_factory=lambda: int(datetime.now().timestamp()))
    thread_id: str
    role: Role
    content: list[MessageContent]
    file_ids: list[str] = []
    assistant_id: Optional[str] = None
    run_id: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None  # Change this line to use dict[str, Any]


class CreateMessageRequest(BaseModel):
    """
    创建消息请求模型
    
    属性:
        role: 消息角色
        content: 消息内容
        file_ids: 关联的文件ID列表
        metadata: 元数据字典（可选）
    """
    role: Role
    content: str
    file_ids: list[str] = []
    metadata: Optional[dict] = None


class ListMessagesResponse(BaseModel):
    """
    消息列表响应模型
    
    属性:
        object: 对象类型，固定为"list"
        data: 消息列表
        first_id: 第一条消息ID
        last_id: 最后一条消息ID
        has_more: 是否还有更多消息
    """
    object: Literal["list"] = "list"
    data: list[Message]
    first_id: str
    last_id: str
    has_more: bool


@router.post("/threads/{thread_id}/messages")
def create_message(
    thread_id: str,
    message: CreateMessageRequest,
    user: User | None = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> Message:
    """
    创建新消息
    
    参数:
        thread_id: 会话线程ID
        message: 创建消息的请求数据
        user: 当前用户（可选）
        db_session: 数据库会话
    
    返回:
        Message: 创建的消息对象
    
    异常:
        HTTPException: 当会话未找到时抛出404错误
    """
    user_id = user.id if user else None

    try:
        chat_session = get_chat_session_by_id(
            chat_session_id=uuid.UUID(thread_id),
            user_id=user_id,
            db_session=db_session,
        )
    except ValueError:
        raise HTTPException(status_code=404, detail="Chat session not found")

    chat_messages = get_chat_messages_by_session(
        chat_session_id=chat_session.id,
        user_id=user.id if user else None,
        db_session=db_session,
    )
    latest_message = (
        chat_messages[-1]
        if chat_messages
        else get_or_create_root_message(chat_session.id, db_session)
    )

    new_message = create_new_chat_message(
        chat_session_id=chat_session.id,
        parent_message=latest_message,
        message=message.content,
        prompt_id=chat_session.persona.prompts[0].id,
        token_count=check_number_of_tokens(message.content),
        message_type=(
            MessageType.USER if message.role == "user" else MessageType.ASSISTANT
        ),
        db_session=db_session,
    )

    return Message(
        id=str(new_message.id),
        thread_id=thread_id,
        role="user",
        content=[MessageContent(type="text", text=message.content)],
        file_ids=message.file_ids,
        metadata=message.metadata,
    )


@router.get("/threads/{thread_id}/messages")
def list_messages(
    thread_id: str,
    limit: int = 20,
    order: Literal["asc", "desc"] = "desc",
    after: Optional[str] = None,
    before: Optional[str] = None,
    user: User | None = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> ListMessagesResponse:
    """
    获取消息列表
    
    参数:
        thread_id: 会话线程ID
        limit: 返回消息的最大数量，默认20条
        order: 排序方式，"asc"升序或"desc"降序
        after: 获取在此ID之后的消息（可选）
        before: 获取在此ID之前的消息（可选）
        user: 当前用户（可选）
        db_session: 数据库会话
    
    返回:
        ListMessagesResponse: 消息列表响应
    
    异常:
        HTTPException: 当会话未找到时抛出404错误
    """
    user_id = user.id if user else None

    try:
        chat_session = get_chat_session_by_id(
            chat_session_id=uuid.UUID(thread_id),
            user_id=user_id,
            db_session=db_session,
        )
    except ValueError:
        raise HTTPException(status_code=404, detail="Chat session not found")

    messages = get_chat_messages_by_session(
        chat_session_id=chat_session.id,
        user_id=user_id,
        db_session=db_session,
    )

    # Apply filtering based on after and before
    if after:
        messages = [m for m in messages if str(m.id) >= after]
    if before:
        messages = [m for m in messages if str(m.id) <= before]

    # Apply ordering
    messages = sorted(messages, key=lambda m: m.id, reverse=(order == "desc"))

    # Apply limit
    messages = messages[:limit]

    data = [
        Message(
            id=str(m.id),
            thread_id=thread_id,
            role="user" if m.message_type == "user" else "assistant",
            content=[MessageContent(type="text", text=m.message)],
            created_at=int(m.time_sent.timestamp()),
        )
        for m in messages
    ]

    return ListMessagesResponse(
        data=data,
        first_id=str(data[0].id) if data else "",
        last_id=str(data[-1].id) if data else "",
        has_more=len(messages) == limit,
    )


@router.get("/threads/{thread_id}/messages/{message_id}")
def retrieve_message(
    thread_id: str,
    message_id: int,
    user: User | None = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> Message:
    """
    获取单个消息
    
    参数:
        thread_id: 会话线程ID
        message_id: 消息ID
        user: 当前用户（可选）
        db_session: 数据库会话
    
    返回:
        Message: 消息对象
    
    异常:
        HTTPException: 当消息未找到时抛出404错误
    """
    user_id = user.id if user else None

    try:
        chat_message = get_chat_message(
            chat_message_id=message_id,
            user_id=user_id,
            db_session=db_session,
        )
    except ValueError:
        raise HTTPException(status_code=404, detail="Message not found")

    return Message(
        id=str(chat_message.id),
        thread_id=thread_id,
        role="user" if chat_message.message_type == "user" else "assistant",
        content=[MessageContent(type="text", text=chat_message.message)],
        created_at=int(chat_message.time_sent.timestamp()),
    )


class ModifyMessageRequest(BaseModel):
    """
    修改消息请求模型
    
    属性:
        metadata: 更新的元数据字典
    """
    metadata: dict


@router.post("/threads/{thread_id}/messages/{message_id}")
def modify_message(
    thread_id: str,
    message_id: int,
    request: ModifyMessageRequest,
    user: User | None = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> Message:
    """
    修改消息元数据
    
    参数:
        thread_id: 会话线程ID
        message_id: 消息ID
        request: 修改消息的请求数据
        user: 当前用户（可选）
        db_session: 数据库会话
    
    返回:
        Message: 更新后的消息对象
    
    异常:
        HTTPException: 当消息未找到时抛出404错误
    """
    user_id = user.id if user else None

    try:
        chat_message = get_chat_message(
            chat_message_id=message_id,
            user_id=user_id,
            db_session=db_session,
        )
    except ValueError:
        raise HTTPException(status_code=404, detail="Message not found")

    # Update metadata
    # TODO: Uncomment this once we have metadata in the chat message
    # chat_message.metadata = request.metadata
    # db_session.commit()

    return Message(
        id=str(chat_message.id),
        thread_id=thread_id,
        role="user" if chat_message.message_type == "user" else "assistant",
        content=[MessageContent(type="text", text=chat_message.message)],
        created_at=int(chat_message.time_sent.timestamp()),
        metadata=request.metadata,
    )
