"""
This file contains the API endpoints and logic for handling chat functionality, including:
- Chat session management (create, update, delete)
- Message handling and streaming
- File upload and management for chat
- Chat feedback handling

此文件包含处理聊天功能的API端点和逻辑，包括：
- 聊天会话管理(创建、更新、删除)
- 消息处理和流式传输
- 聊天文件上传和管理
- 聊天反馈处理
"""

import asyncio
import io
import json
import os
import uuid
from collections.abc import Callable
from collections.abc import Generator
from typing import Tuple
from uuid import UUID

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import Request
from fastapi import Response
from fastapi import UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image
from pydantic import BaseModel
from sqlalchemy.orm import Session

from onyx.auth.users import current_chat_accesssible_user
from onyx.auth.users import current_limited_user
from onyx.auth.users import current_user
from onyx.chat.chat_utils import create_chat_chain
from onyx.chat.chat_utils import extract_headers
from onyx.chat.process_message import stream_chat_message
from onyx.chat.prompt_builder.citations_prompt import (
    compute_max_document_tokens_for_persona,
)
from onyx.configs.app_configs import WEB_DOMAIN
from onyx.configs.constants import FileOrigin
from onyx.configs.constants import MessageType
from onyx.configs.constants import MilestoneRecordType
from onyx.configs.model_configs import LITELLM_PASS_THROUGH_HEADERS
from onyx.db.chat import add_chats_to_session_from_slack_thread
from onyx.db.chat import create_chat_session
from onyx.db.chat import create_new_chat_message
from onyx.db.chat import delete_all_chat_sessions_for_user
from onyx.db.chat import delete_chat_session
from onyx.db.chat import duplicate_chat_session_for_user_from_slack
from onyx.db.chat import get_chat_message
from onyx.db.chat import get_chat_messages_by_session
from onyx.db.chat import get_chat_session_by_id
from onyx.db.chat import get_chat_sessions_by_user
from onyx.db.chat import get_or_create_root_message
from onyx.db.chat import set_as_latest_chat_message
from onyx.db.chat import translate_db_message_to_chat_message_detail
from onyx.db.chat import update_chat_session
from onyx.db.engine import get_current_tenant_id
from onyx.db.engine import get_session
from onyx.db.engine import get_session_with_tenant
from onyx.db.feedback import create_chat_message_feedback
from onyx.db.feedback import create_doc_retrieval_feedback
from onyx.db.models import User
from onyx.db.persona import get_persona_by_id
from onyx.document_index.document_index_utils import get_both_index_names
from onyx.document_index.factory import get_default_document_index
from onyx.file_processing.extract_file_text import docx_to_txt_filename
from onyx.file_processing.extract_file_text import extract_file_text
from onyx.file_store.file_store import get_default_file_store
from onyx.file_store.models import ChatFileType
from onyx.file_store.models import FileDescriptor
from onyx.llm.exceptions import GenAIDisabledException
from onyx.llm.factory import get_default_llms
from onyx.llm.factory import get_llms_for_persona
from onyx.natural_language_processing.utils import get_tokenizer
from onyx.secondary_llm_flows.chat_session_naming import (
    get_renamed_conversation_name,
)
from onyx.server.query_and_chat.models import ChatFeedbackRequest
from onyx.server.query_and_chat.models import ChatMessageIdentifier
from onyx.server.query_and_chat.models import ChatRenameRequest
from onyx.server.query_and_chat.models import ChatSessionCreationRequest
from onyx.server.query_and_chat.models import ChatSessionDetailResponse
from onyx.server.query_and_chat.models import ChatSessionDetails
from onyx.server.query_and_chat.models import ChatSessionsResponse
from onyx.server.query_and_chat.models import ChatSessionUpdateRequest
from onyx.server.query_and_chat.models import CreateChatMessageRequest
from onyx.server.query_and_chat.models import CreateChatSessionID
from onyx.server.query_and_chat.models import LLMOverride
from onyx.server.query_and_chat.models import PromptOverride
from onyx.server.query_and_chat.models import RenameChatSessionResponse
from onyx.server.query_and_chat.models import SearchFeedbackRequest
from onyx.server.query_and_chat.models import UpdateChatSessionThreadRequest
from onyx.server.query_and_chat.token_limit import check_token_rate_limits
from onyx.utils.headers import get_custom_tool_additional_request_headers
from onyx.utils.logger import setup_logger
from onyx.utils.telemetry import create_milestone_and_report


logger = setup_logger()

router = APIRouter(prefix="/chat")


@router.get("/get-user-chat-sessions")
def get_user_chat_sessions(
    user: User | None = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> ChatSessionsResponse:
    """
    获取指定用户的所有聊天会话列表
    
    参数:
        user: 当前用户对象,可能为None
        db_session: 数据库会话对象
        
    返回:
        ChatSessionsResponse: 包含用户所有聊天会话信息的响应对象
    """
    user_id = user.id if user is not None else None

    try:
        chat_sessions = get_chat_sessions_by_user(
            user_id=user_id, deleted=False, db_session=db_session
        )

    except ValueError:
        raise ValueError("Chat session does not exist or has been deleted") 
        # 聊天会话不存在或已被删除

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


@router.put("/update-chat-session-model")
def update_chat_session_model(
    update_thread_req: UpdateChatSessionThreadRequest,
    user: User | None = Depends(current_user), 
    db_session: Session = Depends(get_session),
) -> None:
    """
    更新聊天会话使用的模型
    
    参数:
        update_thread_req: 包含更新信息的请求对象
        user: 当前用户对象,可能为None
        db_session: 数据库会话对象
        
    返回:
        None
    """
    chat_session = get_chat_session_by_id(
        chat_session_id=update_thread_req.chat_session_id,
        user_id=user.id if user is not None else None,
        db_session=db_session,
    )
    chat_session.current_alternate_model = update_thread_req.new_alternate_model

    db_session.add(chat_session)
    db_session.commit()


@router.get("/get-chat-session/{session_id}")
def get_chat_session(
    session_id: UUID,
    is_shared: bool = False,
    user: User | None = Depends(current_chat_accesssible_user),
    db_session: Session = Depends(get_session),
) -> ChatSessionDetailResponse:
    """
    获取指定ID的聊天会话详情
    
    参数:
        session_id: 聊天会话ID
        is_shared: 是否是共享会话
        user: 当前用户对象
        db_session: 数据库会话对象
        
    返回:
        ChatSessionDetailResponse: 包含聊天会话完整信息的响应对象
        
    异常:
        ValueError: 当聊天会话不存在或已删除时抛出
    """
    user_id = user.id if user is not None else None
    try:
        chat_session = get_chat_session_by_id(
            chat_session_id=session_id,
            user_id=user_id,
            db_session=db_session,
            is_shared=is_shared,
        )
    except ValueError:
        raise ValueError("Chat session does not exist or has been deleted")

    # for chat-seeding: if the session is unassigned, assign it now. This is done here
    # to avoid another back and forth between FE -> BE before starting the first
    # message generation
    if chat_session.user_id is None and user_id is not None:
        chat_session.user_id = user_id
        db_session.commit()

    session_messages = get_chat_messages_by_session(
        chat_session_id=session_id,
        user_id=user_id,
        db_session=db_session,
        # we already did a permission check above with the call to
        # `get_chat_session_by_id`, so we can skip it here
        skip_permission_check=True,
        # we need the tool call objs anyways, so just fetch them in a single call
        prefetch_tool_calls=True,
    )

    return ChatSessionDetailResponse(
        chat_session_id=session_id,
        description=chat_session.description,
        persona_id=chat_session.persona_id,
        persona_name=chat_session.persona.name if chat_session.persona else None,
        persona_icon_color=chat_session.persona.icon_color
        if chat_session.persona
        else None,
        persona_icon_shape=chat_session.persona.icon_shape
        if chat_session.persona
        else None,
        current_alternate_model=chat_session.current_alternate_model,
        messages=[
            translate_db_message_to_chat_message_detail(msg) for msg in session_messages
        ],
        time_created=chat_session.time_created,
        shared_status=chat_session.shared_status,
    )


@router.post("/create-chat-session")
def create_new_chat_session(
    chat_session_creation_request: ChatSessionCreationRequest,
    user: User | None = Depends(current_chat_accesssible_user),
    db_session: Session = Depends(get_session),
) -> CreateChatSessionID:
    """
    创建新的聊天会话
    
    参数:
        chat_session_creation_request: 创建聊天会话的请求数据
        user: 当前用户对象
        db_session: 数据库会话对象
        
    返回:
        CreateChatSessionID: 包含新创建的聊天会话ID
        
    异常:
        HTTPException: 当提供的persona无效时抛出400错误
    """
    user_id = user.id if user is not None else None
    try:
        new_chat_session = create_chat_session(
            db_session=db_session,
            description=chat_session_creation_request.description
            or "",  # Leave the naming till later to prevent delay
            user_id=user_id,
            persona_id=chat_session_creation_request.persona_id,
        )
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=400, detail="Invalid Persona provided.")

    return CreateChatSessionID(chat_session_id=new_chat_session.id)


@router.put("/rename-chat-session")
def rename_chat_session(
    rename_req: ChatRenameRequest,
    request: Request,
    user: User | None = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> RenameChatSessionResponse:
    """
    重命名聊天会话
    
    参数:
        rename_req: 重命名请求对象,包含新的名称
        request: FastAPI请求对象
        user: 当前用户对象
        db_session: 数据库会话对象
        
    返回:
        RenameChatSessionResponse: 包含更新后的会话名称
        
    说明:
        如果请求中提供了新名称,直接使用该名称更新;
        否则使用LLM根据会话历史自动生成新的名称。
    """
    name = rename_req.name
    chat_session_id = rename_req.chat_session_id
    user_id = user.id if user is not None else None

    if name:
        update_chat_session(
            db_session=db_session,
            user_id=user_id,
            chat_session_id=chat_session_id,
            description=name,
        )
        return RenameChatSessionResponse(new_name=name)

    final_msg, history_msgs = create_chat_chain(
        chat_session_id=chat_session_id, db_session=db_session
    )
    full_history = history_msgs + [final_msg]

    try:
        llm, _ = get_default_llms(
            additional_headers=extract_headers(
                request.headers, LITELLM_PASS_THROUGH_HEADERS
            )
        )
    except GenAIDisabledException:
        # This may be longer than what the LLM tends to produce but is the most
        # clear thing we can do
        return RenameChatSessionResponse(new_name=full_history[0].message)

    new_name = get_renamed_conversation_name(full_history=full_history, llm=llm)

    update_chat_session(
        db_session=db_session,
        user_id=user_id,
        chat_session_id=chat_session_id,
        description=new_name,
    )

    return RenameChatSessionResponse(new_name=new_name)


@router.patch("/chat-session/{session_id}")
def patch_chat_session(
    session_id: UUID,
    chat_session_update_req: ChatSessionUpdateRequest,
    user: User | None = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> None:
    """
    更新聊天会话的共享状态
    
    参数:
        session_id: 要更新的聊天会话ID
        chat_session_update_req: 更新请求对象
        user: 当前用户对象
        db_session: 数据库会话对象
    """
    user_id = user.id if user is not None else None
    update_chat_session(
        db_session=db_session,
        user_id=user_id,
        chat_session_id=session_id,
        sharing_status=chat_session_update_req.sharing_status,
    )
    return None


@router.delete("/delete-all-chat-sessions")
def delete_all_chat_sessions(
    user: User | None = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> None:
    """
    删除用户的所有聊天会话
    
    参数:
        user: 当前用户对象
        db_session: 数据库会话对象
        
    异常:
        HTTPException: 删除失败时抛出400错误
    """
    try:
        delete_all_chat_sessions_for_user(user=user, db_session=db_session)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/delete-chat-session/{session_id}")
def delete_chat_session_by_id(
    session_id: UUID,
    user: User | None = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> None:
    """
    删除指定ID的聊天会话
    
    参数:
        session_id: 要删除的聊天会话ID
        user: 当前用户对象
        db_session: 数据库会话对象
        
    异常:
        HTTPException: 删除失败时抛出400错误
    """
    user_id = user.id if user is not None else None
    try:
        delete_chat_session(user_id, session_id, db_session)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


async def is_connected(request: Request) -> Callable[[], bool]:
    """
    检查客户端连接状态的异步函数
    
    参数:
        request: FastAPI请求对象
        
    返回:
        Callable[[], bool]: 返回一个可调用对象,用于检查连接状态
        
    说明:
        此函数用于检查WebSocket连接是否仍然活跃。如果连接断开或发生错误,
        将返回True以确保安全处理。
    """
    main_loop = asyncio.get_event_loop()

    def is_connected_sync() -> bool:
        future = asyncio.run_coroutine_threadsafe(request.is_disconnected(), main_loop)
        try:
            is_connected = not future.result(timeout=0.01)
            return is_connected
        except asyncio.TimeoutError:
            logger.error("Asyncio timed out")
            return True
        except Exception as e:
            error_msg = str(e)
            logger.critical(
                f"An unexpected error occured with the disconnect check coroutine: {error_msg}"
            )
            return True

    return is_connected_sync


@router.post("/send-message")
def handle_new_chat_message(
    chat_message_req: CreateChatMessageRequest,
    request: Request,
    user: User | None = Depends(current_chat_accesssible_user),
    _rate_limit_check: None = Depends(check_token_rate_limits),
    is_connected_func: Callable[[], bool] = Depends(is_connected),
    tenant_id: str = Depends(get_current_tenant_id),
) -> StreamingResponse:
    """
    此端点用于以下所有用途：
    - 在会话中发送新消息
    - 在会话中重新生成消息（再次发送相同的消息）
    - 编辑消息（类似于重新生成但发送不同的消息）
    - 启动预设聊天会话（设置 use_existing_user_message）

    假定之前的消息已设置为最新以最小化开销。

    参数:
        chat_message_req: 新聊天消息的详细信息
        request: 当前HTTP请求上下文
        user: 当前用户，通过依赖注入获得
        _rate_limit_check: 如果启用了用户/组/全局速率限制，则运行速率限制检查
        is_connected_func: 用于检查客户端断开连接的函数
        tenant_id: 租户ID

    返回:
        StreamingResponse: 流式传输对新聊天消息的响应
    """
    logger.debug(f"Received new chat message: {chat_message_req.message}")

    if (
        not chat_message_req.message
        and chat_message_req.prompt_id is not None
        and not chat_message_req.use_existing_user_message
    ):
        raise HTTPException(status_code=400, detail="Empty chat message is invalid")

    with get_session_with_tenant(tenant_id) as db_session:
        create_milestone_and_report(
            user=user,
            distinct_id=user.email if user else tenant_id or "N/A",
            event_type=MilestoneRecordType.RAN_QUERY,
            properties=None,
            db_session=db_session,
        )

    def stream_generator() -> Generator[str, None, None]:
        try:
            for packet in stream_chat_message(
                new_msg_req=chat_message_req,
                user=user,
                litellm_additional_headers=extract_headers(
                    request.headers, LITELLM_PASS_THROUGH_HEADERS
                ),
                custom_tool_additional_headers=get_custom_tool_additional_request_headers(
                    request.headers
                ),
                is_connected=is_connected_func,
            ):
                yield json.dumps(packet) if isinstance(packet, dict) else packet

        except Exception as e:
            logger.exception("Error in chat message streaming")
            yield json.dumps({"error": str(e)})

        finally:
            logger.debug("Stream generator finished")

    return StreamingResponse(stream_generator(), media_type="text/event-stream")


@router.put("/set-message-as-latest")
def set_message_as_latest(
    message_identifier: ChatMessageIdentifier,
    user: User | None = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> None:
    """
    将指定消息设置为最新消息
    
    参数:
        message_identifier: 消息标识符对象
        user: 当前用户对象
        db_session: 数据库会话对象
    """
    user_id = user.id if user is not None else None

    chat_message = get_chat_message(
        chat_message_id=message_identifier.message_id,
        user_id=user_id,
        db_session=db_session,
    )

    set_as_latest_chat_message(
        chat_message=chat_message,
        user_id=user_id,
        db_session=db_session,
    )


@router.post("/create-chat-message-feedback")
def create_chat_feedback(
    feedback: ChatFeedbackRequest,
    user: User | None = Depends(current_limited_user),
    db_session: Session = Depends(get_session),
) -> None:
    """
    创建聊天消息的反馈
    
    参数:
        feedback: 反馈请求对象
        user: 当前用户对象
        db_session: 数据库会话对象
    """
    user_id = user.id if user else None

    create_chat_message_feedback(
        is_positive=feedback.is_positive,
        feedback_text=feedback.feedback_text,
        predefined_feedback=feedback.predefined_feedback,
        chat_message_id=feedback.chat_message_id,
        user_id=user_id,
        db_session=db_session,
    )


@router.post("/document-search-feedback")
def create_search_feedback(
    feedback: SearchFeedbackRequest,
    _: User | None = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> None:
    """
    此端点未受保护 - 不检查用户是否有权访问文档
    用户可以尝试更改任意文档的权重，但这不会泄露任何数据。
    
    参数:
        feedback: 文档搜索反馈的请求对象
        _: 当前用户对象
        db_session: 数据库会话对象
    """
    curr_ind_name, sec_ind_name = get_both_index_names(db_session)
    document_index = get_default_document_index(
        primary_index_name=curr_ind_name, secondary_index_name=sec_ind_name
    )

    create_doc_retrieval_feedback(
        message_id=feedback.message_id,
        document_id=feedback.document_id,
        document_rank=feedback.document_rank,
        clicked=feedback.click,
        feedback=feedback.search_feedback,
        document_index=document_index,
        db_session=db_session,
    )


class MaxSelectedDocumentTokens(BaseModel):
    """
    表示文档可选择的最大token数量
    
    属性:
        max_tokens: 最大token数量
    """
    max_tokens: int


@router.get("/max-selected-document-tokens")
def get_max_document_tokens(
    persona_id: int,
    user: User | None = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> MaxSelectedDocumentTokens:
    """
    获取指定角色可选择的最大文档token数
    
    参数:
        persona_id: 角色ID
        user: 当前用户对象
        db_session: 数据库会话对象
        
    返回:
        MaxSelectedDocumentTokens: 包含最大token数的响应对象
        
    异常:
        HTTPException: 当角色不存在时抛出404错误
    """
    try:
        persona = get_persona_by_id(
            persona_id=persona_id,
            user=user,
            db_session=db_session,
            is_for_edit=False,
        )
    except ValueError:
        raise HTTPException(status_code=404, detail="Persona not found")

    return MaxSelectedDocumentTokens(
        max_tokens=compute_max_document_tokens_for_persona(persona),
    )


"""Endpoints for chat seeding"""  # 聊天会话预设的端点


class ChatSeedRequest(BaseModel):
    """
    预设聊天会话的请求模型
    
    属性:
        persona_id: 角色ID
        prompt_id: 提示词ID,可选
        llm_override: LLM覆盖配置,可选
        prompt_override: 提示词覆盖配置,可选
        description: 会话描述,可选
        message: 初始消息,可选
    """
    # standard chat session stuff  # 标准会话相关字段
    persona_id: int
    prompt_id: int | None = None

    # overrides / seeding  # 覆盖/预设配置
    llm_override: LLMOverride | None = None
    prompt_override: PromptOverride | None = None
    description: str | None = None
    message: str | None = None

    # TODO: support this  # TODO: 支持这个功能
    # initial_message_retrieval_options: RetrievalDetails | None = None


class ChatSeedResponse(BaseModel):
    """
    预设聊天会话的响应模型
    
    属性:
        redirect_url: 重定向URL
    """
    redirect_url: str


@router.post("/seed-chat-session")
def seed_chat(
    chat_seed_request: ChatSeedRequest,
    # NOTE: realistically, this will be an API key not an actual user
    _: User | None = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> ChatSeedResponse:
    """
    创建预设聊天会话
    
    参数:
        chat_seed_request: 预设聊天会话的请求数据
        _: 当前用户对象（通常是API密钥而不是实际用户）
        db_session: 数据库会话对象
        
    返回:
        ChatSeedResponse: 包含重定向URL的响应对象
        
    异常:
        HTTPException: 当提供的persona无效时抛出400错误
    """
    try:
        new_chat_session = create_chat_session(
            db_session=db_session,
            description=chat_seed_request.description or "",
            user_id=None,  # this chat session is "unassigned" until a user visits the web UI
            persona_id=chat_seed_request.persona_id,
            llm_override=chat_seed_request.llm_override,
            prompt_override=chat_seed_request.prompt_override,
        )
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=400, detail="Invalid Persona provided.")

    if chat_seed_request.message is not None:
        root_message = get_or_create_root_message(
            chat_session_id=new_chat_session.id, db_session=db_session
        )
        llm, fast_llm = get_llms_for_persona(persona=new_chat_session.persona)

        tokenizer = get_tokenizer(
            model_name=llm.config.model_name,
            provider_type=llm.config.model_provider,
        )
        token_count = len(tokenizer.encode(chat_seed_request.message))

        create_new_chat_message(
            chat_session_id=new_chat_session.id,
            parent_message=root_message,
            prompt_id=chat_seed_request.prompt_id
            or (
                new_chat_session.persona.prompts[0].id
                if new_chat_session.persona.prompts
                else None
            ),
            message=chat_seed_request.message,
            token_count=token_count,
            message_type=MessageType.USER,
            db_session=db_session,
        )

    return ChatSeedResponse(
        redirect_url=f"{WEB_DOMAIN}/chat?chatId={new_chat_session.id}&seeded=true"
    )


class SeedChatFromSlackRequest(BaseModel):
    """
    从Slack导入聊天会话的请求模型
    
    属性:
        chat_session_id: Slack中的会话ID
    """
    chat_session_id: UUID


class SeedChatFromSlackResponse(BaseModel):
    """
    从Slack导入聊天会话的响应模型
    
    属性:
        redirect_url: 重定向URL
    """
    redirect_url: str


@router.post("/seed-chat-session-from-slack")
def seed_chat_from_slack(
    chat_seed_request: SeedChatFromSlackRequest,
    user: User | None = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> SeedChatFromSlackResponse:
    """
    从Slack导入聊天会话并创建新的会话
    
    参数:
        chat_seed_request: 包含Slack聊天会话ID的请求数据
        user: 当前用户对象
        db_session: 数据库会话对象
        
    返回:
        SeedChatFromSlackResponse: 包含重定向URL的响应对象
    """
    slack_chat_session_id = chat_seed_request.chat_session_id
    new_chat_session = duplicate_chat_session_for_user_from_slack(
        db_session=db_session,
        user=user,
        chat_session_id=slack_chat_session_id,
    )

    add_chats_to_session_from_slack_thread(
        db_session=db_session,
        slack_chat_session_id=slack_chat_session_id,
        new_chat_session_id=new_chat_session.id,
    )

    return SeedChatFromSlackResponse(
        redirect_url=f"{WEB_DOMAIN}/chat?chatId={new_chat_session.id}"
    )


"""File upload"""  # 文件上传


def convert_to_jpeg(file: UploadFile) -> Tuple[io.BytesIO, str]:
    """
    将上传的图片文件转换为JPEG格式
    
    参数:
        file: 上传的文件对象
        
    返回:
        Tuple[io.BytesIO, str]: 包含转换后的JPEG图片数据流和文件类型
        
    异常:
        HTTPException: 当图片转换失败时抛出
    """
    try:
        with Image.open(file.file) as img:
            if (img.mode != "RGB"):
                img = img.convert("RGB")
            jpeg_io = io.BytesIO()
            img.save(jpeg_io, format="JPEG", quality=85)
            jpeg_io.seek(0)
        return jpeg_io, "image/jpeg"
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to convert image: {str(e)}"
            # 图片转换失败
        )


@router.post("/file")
def upload_files_for_chat(
    files: list[UploadFile],
    db_session: Session = Depends(get_session),
    _: User | None = Depends(current_user),
) -> dict[str, list[FileDescriptor]]:
    """
    上传聊天使用的文件
    
    参数:
        files: 上传的文件列表
        db_session: 数据库会话对象
        _: 当前用户对象
        
    返回:
        dict[str, list[FileDescriptor]]: 包含已上传文件描述符的字典
        
    异常:
        HTTPException: 当文件类型不支持或文件大小超限时抛出400错误
    """
    image_content_types = {"image/jpeg", "image/png", "image/webp"}
    csv_content_types = {"text/csv"}
    text_content_types = {
        "text/plain",
        "text/markdown",
        "text/x-markdown",
        "text/x-config",
        "text/tab-separated-values",
        "application/json",
        "application/xml",
        "text/xml",
        "application/x-yaml",
    }
    document_content_types = {
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "message/rfc822",
        "application/epub+zip",
    }

    allowed_content_types = (
        image_content_types.union(text_content_types)
        .union(document_content_types)
        .union(csv_content_types)
    )

    for file in files:
        if file.content_type not in allowed_content_types:
            if file.content_type in image_content_types:
                error_detail = "Unsupported image file type. Supported image types include .jpg, .jpeg, .png, .webp."
            elif file.content_type in text_content_types:
                error_detail = "Unsupported text file type. Supported text types include .txt, .csv, .md, .mdx, .conf, "
                ".log, .tsv."
            elif file.content_type in csv_content_types:
                error_detail = (
                    "Unsupported CSV file type. Supported CSV types include .csv."
                )
            else:
                error_detail = (
                    "Unsupported document file type. Supported document types include .pdf, .docx, .pptx, .xlsx, "
                    ".json, .xml, .yml, .yaml, .eml, .epub."
                )
            raise HTTPException(status_code=400, detail=error_detail)

        if (
            file.content_type in image_content_types
            and file.size
            and file.size > 20 * 1024 * 1024
        ):
            raise HTTPException(
                status_code=400,
                detail="File size must be less than 20MB",
            )

    file_store = get_default_file_store(db_session)

    file_info: list[tuple[str, str | None, ChatFileType]] = []
    for file in files:
        if file.content_type in image_content_types:
            file_type = ChatFileType.IMAGE
            # Convert image to JPEG
            file_content, new_content_type = convert_to_jpeg(file)
        elif file.content_type in csv_content_types:
            file_type = ChatFileType.CSV
            file_content = io.BytesIO(file.file.read())
            new_content_type = file.content_type or ""
        elif file.content_type in document_content_types:
            file_type = ChatFileType.DOC
            file_content = io.BytesIO(file.file.read())
            new_content_type = file.content_type or ""
        else:
            file_type = ChatFileType.PLAIN_TEXT
            file_content = io.BytesIO(file.file.read())
            new_content_type = file.content_type or ""

        # store the file (now JPEG for images)
        file_id = str(uuid.uuid4())
        file_store.save_file(
            file_name=file_id,
            content=file_content,
            display_name=file.filename,
            file_origin=FileOrigin.CHAT_UPLOAD,
            file_type=new_content_type or file_type.value,
        )

        # if the file is a doc, extract text and store that so we don't need
        # to re-extract it every time we send a message
        if file_type == ChatFileType.DOC:
            extracted_text = extract_file_text(
                file=file.file,
                file_name=file.filename or "",
            )
            text_file_id = str(uuid.uuid4())
            file_store.save_file(
                file_name=text_file_id,
                content=io.BytesIO(extracted_text.encode()),
                display_name=file.filename,
                file_origin=FileOrigin.CHAT_UPLOAD,
                file_type="text/plain",
            )
            # for DOC type, just return this for the FileDescriptor
            # as we would always use this as the ID to attach to the
            # message
            file_info.append((text_file_id, file.filename, ChatFileType.PLAIN_TEXT))
        else:
            file_info.append((file_id, file.filename, file_type))

    return {
        "files": [
            {"id": file_id, "type": file_type, "name": file_name}
            for file_id, file_name, file_type in file_info
        ]
    }


@router.get("/file/{file_id:path}")
def fetch_chat_file(
    file_id: str,
    db_session: Session = Depends(get_session),
    _: User | None = Depends(current_user),
) -> Response:
    """
    获取聊天文件内容
    
    参数:
        file_id: 文件ID
        db_session: 数据库会话对象
        _: 当前用户对象
        
    返回:
        Response: 文件内容的响应对象
        
    异常:
        HTTPException: 当文件不存在时抛出404错误
    """
    file_store = get_default_file_store(db_session)
    file_record = file_store.read_file_record(file_id)
    if not file_record:
        raise HTTPException(status_code=404, detail="File not found")

    original_file_name = file_record.display_name
    if file_record.file_type.startswith(
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ):
        # Check if a converted text file exists for .docx files
        txt_file_name = docx_to_txt_filename(original_file_name)
        txt_file_id = os.path.join(os.path.dirname(file_id), txt_file_name)
        txt_file_record = file_store.read_file_record(txt_file_id)
        if txt_file_record:
            file_record = txt_file_record
            file_id = txt_file_id

    media_type = file_record.file_type
    file_io = file_store.read_file(file_id, mode="b")

    return StreamingResponse(file_io, media_type=media_type)
