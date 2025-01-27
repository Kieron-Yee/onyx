"""
该文件实现了OpenAI Assistants API中的Runs相关功能，主要包括：
- 创建和管理assistant的运行实例(runs)
- 处理运行状态的更新和查询
- 实现runs的取消和列表功能
- 集成与DAnswer系统的消息处理
"""

from typing import Literal
from typing import Optional
from uuid import UUID

from fastapi import APIRouter
from fastapi import BackgroundTasks
from fastapi import Depends
from fastapi import HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from onyx.auth.users import current_user
from onyx.chat.process_message import stream_chat_message_objects
from onyx.configs.constants import MessageType
from onyx.context.search.models import RetrievalDetails
from onyx.db.chat import create_new_chat_message
from onyx.db.chat import get_chat_message
from onyx.db.chat import get_chat_messages_by_session
from onyx.db.chat import get_chat_session_by_id
from onyx.db.chat import get_or_create_root_message
from onyx.db.engine import get_session
from onyx.db.models import ChatMessage
from onyx.db.models import User
from onyx.server.query_and_chat.models import ChatMessageDetail
from onyx.server.query_and_chat.models import CreateChatMessageRequest
from onyx.tools.tool_implementations.search.search_tool import SearchTool
from onyx.utils.logger import setup_logger


logger = setup_logger()


router = APIRouter()


class RunRequest(BaseModel):
    """
    运行请求的数据模型
    
    参数说明:
    - assistant_id: assistant的ID
    - model: 可选，使用的模型名称
    - instructions: 可选，运行指令
    - additional_instructions: 可选，额外指令
    - tools: 可选，可用工具列表
    - metadata: 可选，元数据
    """
    assistant_id: int
    model: Optional[str] = None
    instructions: Optional[str] = None
    additional_instructions: Optional[str] = None
    tools: Optional[list[dict]] = None
    metadata: Optional[dict] = None


RunStatus = Literal[
    "queued",  # 排队中
    "in_progress",  # 进行中
    "requires_action",  # 需要操作
    "cancelling",  # 取消中
    "cancelled",  # 已取消
    "failed",  # 失败
    "completed",  # 完成
    "expired",  # 过期
]


class RunResponse(BaseModel):
    """
    运行响应的数据模型
    
    包含运行实例的完整状态信息
    """
    id: str
    object: Literal["thread.run"]
    created_at: int
    assistant_id: int
    thread_id: UUID
    status: RunStatus
    started_at: Optional[int] = None
    expires_at: Optional[int] = None
    cancelled_at: Optional[int] = None
    failed_at: Optional[int] = None
    completed_at: Optional[int] = None
    last_error: Optional[dict] = None
    model: str
    instructions: str
    tools: list[dict]
    file_ids: list[str]
    metadata: Optional[dict] = None


def process_run_in_background(
    message_id: int,
    parent_message_id: int,
    chat_session_id: UUID,
    assistant_id: int,
    instructions: str,
    tools: list[dict],
    user: User | None,
    db_session: Session,
) -> None:
    """
    在后台处理运行请求
    
    参数说明:
    - message_id: 消息ID
    - parent_message_id: 父消息ID
    - chat_session_id: 会话ID
    - assistant_id: assistant ID
    - instructions: 运行指令
    - tools: 工具列表
    - user: 用户对象
    - db_session: 数据库会话
    """
    # Get the latest message in the chat session
    chat_session = get_chat_session_by_id(
        chat_session_id=chat_session_id,
        user_id=user.id if user else None,
        db_session=db_session,
    )

    search_tool_retrieval_details = RetrievalDetails()
    for tool in tools:
        if tool["type"] == SearchTool.__name__ and (
            retrieval_details := tool.get("retrieval_details")
        ):
            search_tool_retrieval_details = RetrievalDetails.model_validate(
                retrieval_details
            )
            break

    new_msg_req = CreateChatMessageRequest(
        chat_session_id=chat_session_id,
        parent_message_id=int(parent_message_id) if parent_message_id else None,
        message=instructions,
        file_descriptors=[],
        prompt_id=chat_session.persona.prompts[0].id,
        search_doc_ids=None,
        retrieval_options=search_tool_retrieval_details,  # Adjust as needed
        rerank_settings=None,
        query_override=None,
        regenerate=None,
        llm_override=None,
        prompt_override=None,
        alternate_assistant_id=assistant_id,
        use_existing_user_message=True,
        existing_assistant_message_id=message_id,
    )

    run_message = get_chat_message(message_id, user.id if user else None, db_session)
    try:
        for packet in stream_chat_message_objects(
            new_msg_req=new_msg_req,
            user=user,
            db_session=db_session,
        ):
            if isinstance(packet, ChatMessageDetail):
                # Update the run status and message content
                run_message = get_chat_message(
                    message_id, user.id if user else None, db_session
                )
                if run_message:
                    # this handles cancelling
                    if run_message.error:
                        return

                    run_message.message = packet.message
                    run_message.message_type = MessageType.ASSISTANT
                    db_session.commit()
    except Exception as e:
        logger.exception("Error processing run in background")
        run_message.error = str(e)
        db_session.commit()
        return

    db_session.refresh(run_message)
    if run_message.token_count == 0:
        run_message.error = "No tokens generated"
        db_session.commit()


@router.post("/threads/{thread_id}/runs")
def create_run(
    thread_id: UUID,
    run_request: RunRequest,
    background_tasks: BackgroundTasks,
    user: User | None = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> RunResponse:
    """
    创建新的运行实例
    
    参数说明:
    - thread_id: 会话线程ID
    - run_request: 运行请求对象
    - background_tasks: 后台任务管理器
    - user: 当前用户
    - db_session: 数据库会话
    
    返回:
    - RunResponse: 运行实例的响应对象
    """
    try:
        chat_session = get_chat_session_by_id(
            chat_session_id=thread_id,
            user_id=user.id if user else None,
            db_session=db_session,
        )
    except ValueError:
        raise HTTPException(status_code=404, detail="Thread not found")

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

    # Create a new "run" (chat message) in the session
    new_message = create_new_chat_message(
        chat_session_id=chat_session.id,
        parent_message=latest_message,
        message="",
        prompt_id=chat_session.persona.prompts[0].id,
        token_count=0,
        message_type=MessageType.ASSISTANT,
        db_session=db_session,
        commit=False,
    )
    db_session.flush()
    latest_message.latest_child_message = new_message.id
    db_session.commit()

    # Schedule the background task
    background_tasks.add_task(
        process_run_in_background,
        new_message.id,
        latest_message.id,
        chat_session.id,
        run_request.assistant_id,
        run_request.instructions or "",
        run_request.tools or [],
        user,
        db_session,
    )

    return RunResponse(
        id=str(new_message.id),
        object="thread.run",
        created_at=int(new_message.time_sent.timestamp()),
        assistant_id=run_request.assistant_id,
        thread_id=chat_session.id,
        status="queued",
        model=run_request.model or "default_model",
        instructions=run_request.instructions or "",
        tools=run_request.tools or [],
        file_ids=[],
        metadata=run_request.metadata,
    )


@router.get("/threads/{thread_id}/runs/{run_id}")
def retrieve_run(
    thread_id: UUID,
    run_id: str,
    user: User | None = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> RunResponse:
    """
    获取运行实例的详细信息
    
    参数说明:
    - thread_id: 会话线程ID
    - run_id: 运行实例ID
    - user: 当前用户
    - db_session: 数据库会话
    
    返回:
    - RunResponse: 运行实例的详细信息
    """
    # Retrieve the chat message (which represents a "run" in DAnswer)
    chat_message = get_chat_message(
        chat_message_id=int(run_id),  # Convert string run_id to int
        user_id=user.id if user else None,
        db_session=db_session,
    )
    if not chat_message:
        raise HTTPException(status_code=404, detail="Run not found")

    chat_session = chat_message.chat_session

    # Map DAnswer status to OpenAI status
    run_status: RunStatus = "queued"
    if chat_message.message:
        run_status = "in_progress"
    if chat_message.token_count != 0:
        run_status = "completed"
    if chat_message.error:
        run_status = "cancelled"

    return RunResponse(
        id=run_id,
        object="thread.run",
        created_at=int(chat_message.time_sent.timestamp()),
        assistant_id=chat_session.persona_id or 0,
        thread_id=chat_session.id,
        status=run_status,
        started_at=int(chat_message.time_sent.timestamp()),
        completed_at=(
            int(chat_message.time_sent.timestamp()) if chat_message.message else None
        ),
        model=chat_session.current_alternate_model or "default_model",
        instructions="",  # DAnswer doesn't store per-message instructions
        tools=[],  # DAnswer doesn't have a direct equivalent for tools
        file_ids=(
            [file["id"] for file in chat_message.files] if chat_message.files else []
        ),
        metadata=None,  # DAnswer doesn't store metadata for individual messages
    )


@router.post("/threads/{thread_id}/runs/{run_id}/cancel")
def cancel_run(
    thread_id: UUID,
    run_id: str,
    user: User | None = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> RunResponse:
    """
    取消运行实例
    
    参数说明:
    - thread_id: 会话线程ID
    - run_id: 运行实例ID
    - user: 当前用户
    - db_session: 数据库会话
    
    返回:
    - RunResponse: 更新后的运行实例信息
    """
    # In DAnswer, we don't have a direct equivalent to cancelling a run
    # We'll simulate it by marking the message as "cancelled"
    chat_message = (
        db_session.query(ChatMessage).filter(ChatMessage.id == run_id).first()
    )
    if not chat_message:
        raise HTTPException(status_code=404, detail="Run not found")

    chat_message.error = "Cancelled"
    db_session.commit()

    return retrieve_run(thread_id, run_id, user, db_session)


@router.get("/threads/{thread_id}/runs")
def list_runs(
    thread_id: UUID,
    limit: int = 20,
    order: Literal["asc", "desc"] = "desc",
    after: Optional[str] = None,
    before: Optional[str] = None,
    user: User | None = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> list[RunResponse]:
    """
    获取会话线程中的运行实例列表
    
    参数说明:
    - thread_id: 会话线程ID
    - limit: 返回结果数量限制
    - order: 排序方式（升序/降序）
    - after: 分页参数，获取指定ID之后的结果
    - before: 分页参数，获取指定ID之前的结果
    - user: 当前用户
    - db_session: 数据库会话
    
    返回:
    - list[RunResponse]: 运行实例列表
    """
    # In DAnswer, we'll treat each message in a chat session as a "run"
    chat_messages = get_chat_messages_by_session(
        chat_session_id=thread_id,
        user_id=user.id if user else None,
        db_session=db_session,
    )

    # Apply pagination
    if after:
        chat_messages = [msg for msg in chat_messages if str(msg.id) > after]
    if before:
        chat_messages = [msg for msg in chat_messages if str(msg.id) < before]

    # Apply ordering
    chat_messages = sorted(
        chat_messages, key=lambda msg: msg.time_sent, reverse=(order == "desc")
    )

    # Apply limit
    chat_messages = chat_messages[:limit]

    return [
        retrieve_run(thread_id, str(msg.id), user, db_session) for msg in chat_messages
    ]


@router.get("/threads/{thread_id}/runs/{run_id}/steps")
def list_run_steps(
    run_id: str,
    limit: int = 20,
    order: Literal["asc", "desc"] = "desc",
    after: Optional[str] = None,
    before: Optional[str] = None,
    user: User | None = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> list[dict]:  # You may want to create a specific model for run steps
    """
    获取运行实例的步骤列表
    
    参数说明:
    - run_id: 运行实例ID
    - limit: 返回结果数量限制
    - order: 排序方式
    - after: 分页参数
    - before: 分页参数
    - user: 当前用户
    - db_session: 数据库会话
    
    返回:
    - list[dict]: 步骤列表（在DAnswer中始终返回空列表）
    """
    # DAnswer doesn't have an equivalent to run steps
    # We'll return an empty list to maintain API compatibility
    return []


# Additional helper functions can be added here if needed
