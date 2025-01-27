"""
This file contains database operations related to chat sessions and messages.
It provides functions for creating, retrieving, updating and deleting chat sessions 
and their associated messages.

该文件包含与聊天会话和消息相关的数据库操作。
提供了创建、获取、更新和删除聊天会话及其关联消息的功能。
"""

from collections.abc import Sequence
from datetime import datetime
from datetime import timedelta
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import delete
from sqlalchemy import desc
from sqlalchemy import func
from sqlalchemy import nullsfirst
from sqlalchemy import or_
from sqlalchemy import select
from sqlalchemy import update
from sqlalchemy.exc import MultipleResultsFound
from sqlalchemy.orm import joinedload
from sqlalchemy.orm import Session

from onyx.auth.schemas import UserRole
from onyx.chat.models import DocumentRelevance
from onyx.configs.chat_configs import HARD_DELETE_CHATS
from onyx.configs.constants import MessageType
from onyx.context.search.models import RetrievalDocs
from onyx.context.search.models import SavedSearchDoc
from onyx.context.search.models import SearchDoc as ServerSearchDoc
from onyx.db.models import ChatMessage
from onyx.db.models import ChatMessage__SearchDoc
from onyx.db.models import ChatSession
from onyx.db.models import ChatSessionSharedStatus
from onyx.db.models import Prompt
from onyx.db.models import SearchDoc
from onyx.db.models import SearchDoc as DBSearchDoc
from onyx.db.models import ToolCall
from onyx.db.models import User
from onyx.db.persona import get_best_persona_id_for_user
from onyx.db.pg_file_store import delete_lobj_by_name
from onyx.file_store.models import FileDescriptor
from onyx.llm.override_models import LLMOverride
from onyx.llm.override_models import PromptOverride
from onyx.server.query_and_chat.models import ChatMessageDetail
from onyx.tools.tool_runner import ToolCallFinalResult
from onyx.utils.logger import setup_logger


logger = setup_logger()


def get_chat_session_by_id(
    chat_session_id: UUID,
    user_id: UUID | None,
    db_session: Session,
    include_deleted: bool = False,
    is_shared: bool = False,
) -> ChatSession:
    """
    根据ID获取聊天会话
    
    参数:
    - chat_session_id: 聊天会话ID
    - user_id: 用户ID
    - db_session: 数据库会话
    - include_deleted: 是否包含已删除的会话
    - is_shared: 是否为共享会话
    
    返回: ChatSession对象
    """
    stmt = select(ChatSession).where(ChatSession.id == chat_session_id)

    if is_shared:
        stmt = stmt.where(ChatSession.shared_status == ChatSessionSharedStatus.PUBLIC)
    else:
        # if user_id is None, assume this is an admin who should be able
        # to view all chat sessions
        if user_id is not None:
            stmt = stmt.where(
                or_(ChatSession.user_id == user_id, ChatSession.user_id.is_(None))
            )

    result = db_session.execute(stmt)
    chat_session = result.scalar_one_or_none()

    if not chat_session:
        raise ValueError("Invalid Chat Session ID provided")

    if not include_deleted and chat_session.deleted:
        raise ValueError("Chat session has been deleted")

    return chat_session


def get_chat_sessions_by_slack_thread_id(
    slack_thread_id: str,
    user_id: UUID | None,
    db_session: Session,
) -> Sequence[ChatSession]:
    """
    根据Slack线程ID获取聊天会话列表
    """
    stmt = select(ChatSession).where(ChatSession.slack_thread_id == slack_thread_id)
    if user_id is not None:
        stmt = stmt.where(
            or_(ChatSession.user_id == user_id, ChatSession.user_id.is_(None))
        )
    return db_session.scalars(stmt).all()


def get_valid_messages_from_query_sessions(
    chat_session_ids: list[UUID],
    db_session: Session,
) -> dict[UUID, str]:
    """
    获取指定会话ID列表中的有效消息
    
    参数:
    - chat_session_ids: 聊天会话ID列表
    - db_session: 数据库会话
    
    返回: 以会话ID为键，消息内容为值的字典
    """
    user_message_subquery = (
        select(
            ChatMessage.chat_session_id, func.min(ChatMessage.id).label("user_msg_id")
        )
        .where(
            ChatMessage.chat_session_id.in_(chat_session_ids),
            ChatMessage.message_type == MessageType.USER,
        )
        .group_by(ChatMessage.chat_session_id)
        .subquery()
    )

    assistant_message_subquery = (
        select(
            ChatMessage.chat_session_id,
            func.min(ChatMessage.id).label("assistant_msg_id"),
        )
        .where(
            ChatMessage.chat_session_id.in_(chat_session_ids),
            ChatMessage.message_type == MessageType.ASSISTANT,
        )
        .group_by(ChatMessage.chat_session_id)
        .subquery()
    )

    query = (
        select(ChatMessage.chat_session_id, ChatMessage.message)
        .join(
            user_message_subquery,
            ChatMessage.chat_session_id == user_message_subquery.c.chat_session_id,
        )
        .join(
            assistant_message_subquery,
            ChatMessage.chat_session_id == assistant_message_subquery.c.chat_session_id,
        )
        .join(
            ChatMessage__SearchDoc,
            ChatMessage__SearchDoc.chat_message_id
            == assistant_message_subquery.c.assistant_msg_id,
        )
        .where(ChatMessage.id == user_message_subquery.c.user_msg_id)
    )

    first_messages = db_session.execute(query).all()
    logger.info(f"Retrieved {len(first_messages)} first messages with documents")

    return {row.chat_session_id: row.message for row in first_messages}


def get_chat_sessions_by_user(
    user_id: UUID | None,
    deleted: bool | None,
    db_session: Session,
    include_onyxbot_flows: bool = False,
    limit: int = 50,
) -> list[ChatSession]:
    """
    获取指定用户的聊天会话列表
    
    参数:
    - user_id: 用户ID
    - deleted: 是否获取已删除的会话
    - db_session: 数据库会话
    - include_onyxbot_flows: 是否包含onyxbot流程会话
    - limit: 返回结果数量限制
    
    返回: ChatSession对象列表
    """
    stmt = select(ChatSession).where(ChatSession.user_id == user_id)

    if not include_onyxbot_flows:
        stmt = stmt.where(ChatSession.onyxbot_flow.is_(False))

    stmt = stmt.order_by(desc(ChatSession.time_created))

    if deleted is not None:
        stmt = stmt.where(ChatSession.deleted == deleted)

    if limit:
        stmt = stmt.limit(limit)

    result = db_session.execute(stmt)
    chat_sessions = result.scalars().all()

    return list(chat_sessions)


def delete_search_doc_message_relationship(
    message_id: int, db_session: Session
) -> None:
    """
    删除消息与搜索文档之间的关联关系
    
    参数:
    - message_id: 消息ID
    - db_session: 数据库会话
    """
    db_session.query(ChatMessage__SearchDoc).filter(
        ChatMessage__SearchDoc.chat_message_id == message_id
    ).delete(synchronize_session=False)

    db_session.commit()


def delete_tool_call_for_message_id(message_id: int, db_session: Session) -> None:
    """
    删除与消息关联的工具调用记录
    
    参数:
    - message_id: 消息ID
    - db_session: 数据库会话
    """
    stmt = delete(ToolCall).where(ToolCall.message_id == message_id)
    db_session.execute(stmt)
    db_session.commit()


def delete_orphaned_search_docs(db_session: Session) -> None:
    """
    删除孤立的搜索文档（没有关联任何消息的文档）
    
    参数:
    - db_session: 数据库会话
    """
    orphaned_docs = (
        db_session.query(SearchDoc)
        .outerjoin(ChatMessage__SearchDoc)
        .filter(ChatMessage__SearchDoc.chat_message_id.is_(None))
        .all()
    )
    for doc in orphaned_docs:
        db_session.delete(doc)
    db_session.commit()


def delete_messages_and_files_from_chat_session(
    chat_session_id: UUID, db_session: Session
) -> None:
    """
    删除聊天会话中的所有消息和相关文件
    
    参数:
    - chat_session_id: 聊天会话ID
    - db_session: 数据库会话
    """
    # Select messages older than cutoff_time with files
    messages_with_files = db_session.execute(
        select(ChatMessage.id, ChatMessage.files).where(
            ChatMessage.chat_session_id == chat_session_id,
        )
    ).fetchall()

    for id, files in messages_with_files:
        delete_tool_call_for_message_id(message_id=id, db_session=db_session)
        delete_search_doc_message_relationship(message_id=id, db_session=db_session)
        for file_info in files or {}:
            lobj_name = file_info.get("id")
            if lobj_name:
                logger.info(f"Deleting file with name: {lobj_name}")
                delete_lobj_by_name(lobj_name, db_session)

    db_session.execute(
        delete(ChatMessage).where(ChatMessage.chat_session_id == chat_session_id)
    )
    db_session.commit()

    delete_orphaned_search_docs(db_session)


def create_chat_session(
    db_session: Session,
    description: str | None,
    user_id: UUID | None,
    persona_id: int | None,  # Can be none if temporary persona is used
    llm_override: LLMOverride | None = None,
    prompt_override: PromptOverride | None = None,
    onyxbot_flow: bool = False,
    slack_thread_id: str | None = None,
) -> ChatSession:
    """
    创建新的聊天会话
    
    参数:
    - db_session: 数据库会话
    - description: 会话描述
    - user_id: 用户ID 
    - persona_id: 角色ID
    - llm_override: LLM覆盖设置
    - prompt_override: 提示词覆盖设置
    - onyxbot_flow: 是否为onyxbot流程
    - slack_thread_id: Slack线程ID
    
    返回: 新创建的ChatSession对象
    """
    chat_session = ChatSession(
        user_id=user_id,
        persona_id=persona_id,
        description=description,
        llm_override=llm_override,
        prompt_override=prompt_override,
        onyxbot_flow=onyxbot_flow,
        slack_thread_id=slack_thread_id,
    )

    db_session.add(chat_session)
    db_session.commit()

    return chat_session


def duplicate_chat_session_for_user_from_slack(
    db_session: Session,
    user: User | None,
    chat_session_id: UUID,
) -> ChatSession:
    """
    从Slack复制聊天会话给指定用户
    
    参数:
    - db_session: 数据库会话
    - user: 用户对象
    - chat_session_id: 要复制的聊天会话ID
    
    返回: 新创建的ChatSession对象
    """
    chat_session = get_chat_session_by_id(
        chat_session_id=chat_session_id,
        user_id=None,  # Ignore user permissions for this
        db_session=db_session,
    )
    if not chat_session:
        raise HTTPException(status_code=400, detail="Invalid Chat Session ID provided")

    # This enforces permissions and sets a default
    new_persona_id = get_best_persona_id_for_user(
        db_session=db_session,
        user=user,
        persona_id=chat_session.persona_id,
    )

    return create_chat_session(
        db_session=db_session,
        user_id=user.id if user else None,
        persona_id=new_persona_id,
        # Set this to empty string so the frontend will force a rename
        description="",
        llm_override=chat_session.llm_override,
        prompt_override=chat_session.prompt_override,
        # Chat is in UI now so this is false
        onyxbot_flow=False,
        # Maybe we want this in the future to track if it was created from Slack
        slack_thread_id=None,
    )


def update_chat_session(
    db_session: Session,
    user_id: UUID | None,
    chat_session_id: UUID,
    description: str | None = None,
    sharing_status: ChatSessionSharedStatus | None = None,
) -> ChatSession:
    """
    更新聊天会话信息
    
    参数:
    - db_session: 数据库会话
    - user_id: 用户ID
    - chat_session_id: 聊天会话ID
    - description: 新的会话描述
    - sharing_status: 新的共享状态
    
    返回: 更新后的ChatSession对象
    """
    chat_session = get_chat_session_by_id(
        chat_session_id=chat_session_id, user_id=user_id, db_session=db_session
    )

    if chat_session.deleted:
        raise ValueError("Trying to rename a deleted chat session")

    if description is not None:
        chat_session.description = description
    if sharing_status is not None:
        chat_session.shared_status = sharing_status

    db_session.commit()

    return chat_session


def delete_all_chat_sessions_for_user(
    user: User | None, db_session: Session, hard_delete: bool = HARD_DELETE_CHATS
) -> None:
    """
    删除用户的所有聊天会话
    
    参数:
    - user: 用户对象
    - db_session: 数据库会话
    - hard_delete: 是否硬删除（真实删除而不是标记删除）
    """
    user_id = user.id if user is not None else None

    query = db_session.query(ChatSession).filter(
        ChatSession.user_id == user_id, ChatSession.onyxbot_flow.is_(False)
    )

    if hard_delete:
        query.delete(synchronize_session=False)
    else:
        query.update({ChatSession.deleted: True}, synchronize_session=False)

    db_session.commit()


def delete_chat_session(
    user_id: UUID | None,
    chat_session_id: UUID,
    db_session: Session,
    hard_delete: bool = HARD_DELETE_CHATS,
) -> None:
    """
    删除聊天会话
    
    参数:
    - user_id: 用户ID
    - chat_session_id: 聊天会话ID
    - db_session: 数据库会话
    - hard_delete: 是否硬删除(真实删除而不是标记删除)
    """
    chat_session = get_chat_session_by_id(
        chat_session_id=chat_session_id, user_id=user_id, db_session=db_session
    )

    if chat_session.deleted:
        raise ValueError("Cannot delete an already deleted chat session")

    if hard_delete:
        delete_messages_and_files_from_chat_session(chat_session_id, db_session)
        db_session.execute(delete(ChatSession).where(ChatSession.id == chat_session_id))
    else:
        chat_session = get_chat_session_by_id(
            chat_session_id=chat_session_id, user_id=user_id, db_session=db_session
        )
        chat_session.deleted = True

    db_session.commit()


def delete_chat_sessions_older_than(days_old: int, db_session: Session) -> None:
    """
    删除指定天数之前的聊天会话
    
    参数:
    - days_old: 天数
    - db_session: 数据库会话
    """
    cutoff_time = datetime.utcnow() - timedelta(days=days_old)
    old_sessions = db_session.execute(
        select(ChatSession.user_id, ChatSession.id).where(
            ChatSession.time_created < cutoff_time
        )
    ).fetchall()

    for user_id, session_id in old_sessions:
        delete_chat_session(user_id, session_id, db_session, hard_delete=True)


def get_chat_message(
    chat_message_id: int,
    user_id: UUID | None,
    db_session: Session,
) -> ChatMessage:
    """
    获取单条聊天消息
    
    参数:
    - chat_message_id: 聊天消息ID
    - user_id: 用户ID
    - db_session: 数据库会话
    
    返回: ChatMessage对象
    """
    stmt = select(ChatMessage).where(ChatMessage.id == chat_message_id)

    result = db_session.execute(stmt)
    chat_message = result.scalar_one_or_none()

    if not chat_message:
        raise ValueError("Invalid Chat Message specified")

    chat_user = chat_message.chat_session.user
    expected_user_id = chat_user.id if chat_user is not None else None

    if expected_user_id != user_id:
        logger.error(
            f"User {user_id} tried to fetch a chat message that does not belong to them"
        )
        raise ValueError("Chat message does not belong to user")

    return chat_message


def get_chat_session_by_message_id(
    db_session: Session,
    message_id: int,
) -> ChatSession:
    """
    Should only be used for Slack
    Get the chat session associated with a specific message ID
    Note: this ignores permission checks.
    """
    stmt = select(ChatMessage).where(ChatMessage.id == message_id)

    result = db_session.execute(stmt)
    chat_message = result.scalar_one_or_none()

    if chat_message is None:
        raise ValueError(
            f"Unable to find chat session associated with message ID: {message_id}"
        )

    return chat_message.chat_session


def get_chat_messages_by_sessions(
    chat_session_ids: list[UUID],
    user_id: UUID | None,
    db_session: Session,
    skip_permission_check: bool = False,
) -> Sequence[ChatMessage]:
    """
    获取多个会话的所有聊天消息
    
    参数:
    - chat_session_ids: 聊天会话ID列表
    - user_id: 用户ID
    - db_session: 数据库会话
    - skip_permission_check: 是否跳过权限检查
    
    返回: ChatMessage对象序列
    """
    if not skip_permission_check:
        for chat_session_id in chat_session_ids:
            get_chat_session_by_id(
                chat_session_id=chat_session_id, user_id=user_id, db_session=db_session
            )
    stmt = (
        select(ChatMessage)
        .where(ChatMessage.chat_session_id.in_(chat_session_ids))
        .order_by(nullsfirst(ChatMessage.parent_message))
    )
    return db_session.execute(stmt).scalars().all()


def add_chats_to_session_from_slack_thread(
    db_session: Session,
    slack_chat_session_id: UUID,
    new_chat_session_id: UUID,
) -> None:
    """
    从Slack线程复制聊天消息到新会话
    
    参数:
    - db_session: 数据库会话
    - slack_chat_session_id: Slack聊天会话ID
    - new_chat_session_id: 新会话ID
    """
    new_root_message = get_or_create_root_message(
        chat_session_id=new_chat_session_id,
        db_session=db_session,
    )

    for chat_message in get_chat_messages_by_sessions(
        chat_session_ids=[slack_chat_session_id],
        user_id=None,  # Ignore user permissions for this
        db_session=db_session,
        skip_permission_check=True,
    ):
        if chat_message.message_type == MessageType.SYSTEM:
            continue
        # Duplicate the message
        new_root_message = create_new_chat_message(
            db_session=db_session,
            chat_session_id=new_chat_session_id,
            parent_message=new_root_message,
            message=chat_message.message,
            files=chat_message.files,
            rephrased_query=chat_message.rephrased_query,
            error=chat_message.error,
            citations=chat_message.citations,
            reference_docs=chat_message.search_docs,
            tool_call=chat_message.tool_call,
            prompt_id=chat_message.prompt_id,
            token_count=chat_message.token_count,
            message_type=chat_message.message_type,
            alternate_assistant_id=chat_message.alternate_assistant_id,
            overridden_model=chat_message.overridden_model,
        )


def get_search_docs_for_chat_message(
    chat_message_id: int, db_session: Session
) -> list[SearchDoc]:
    """
    获取与聊天消息关联的搜索文档列表
    
    参数:
    - chat_message_id: 聊天消息ID
    - db_session: 数据库会话
    
    返回: 搜索文档列表
    """
    stmt = (
        select(SearchDoc)
        .join(
            ChatMessage__SearchDoc, ChatMessage__SearchDoc.search_doc_id == SearchDoc.id
        )
        .where(ChatMessage__SearchDoc.chat_message_id == chat_message_id)
    )

    return list(db_session.scalars(stmt).all())


def get_chat_messages_by_session(
    chat_session_id: UUID,
    user_id: UUID | None,
    db_session: Session,
    skip_permission_check: bool = False,
    prefetch_tool_calls: bool = False,
) -> list[ChatMessage]:
    """
    获取单个会话的所有聊天消息
    
    参数:
    - chat_session_id: 聊天会话ID
    - user_id: 用户ID
    - db_session: 数据库会话
    - skip_permission_check: 是否跳过权限检查
    - prefetch_tool_calls: 是否预加载工具调用信息
    
    返回: ChatMessage对象列表
    """
    if not skip_permission_check:
        get_chat_session_by_id(
            chat_session_id=chat_session_id, user_id=user_id, db_session=db_session
        )

    stmt = (
        select(ChatMessage)
        .where(ChatMessage.chat_session_id == chat_session_id)
        .order_by(nullsfirst(ChatMessage.parent_message))
    )

    if prefetch_tool_calls:
        stmt = stmt.options(joinedload(ChatMessage.tool_call))
        result = db_session.scalars(stmt).unique().all()
    else:
        result = db_session.scalars(stmt).all()

    return list(result)


def get_or_create_root_message(
    chat_session_id: UUID,
    db_session: Session,
) -> ChatMessage:
    """
    获取或创建会话的根消息
    
    参数:
    - chat_session_id: 聊天会话ID
    - db_session: 数据库会话
    
    返回: ChatMessage对象（根消息）
    """
    try:
        root_message: ChatMessage | None = (
            db_session.query(ChatMessage)
            .filter(
                ChatMessage.chat_session_id == chat_session_id,
                ChatMessage.parent_message.is_(None),
            )
            .one_or_none()
        )
    except MultipleResultsFound:
        raise Exception(
            "Multiple root messages found for chat session. Data inconsistency detected."
        )

    if root_message is not None:
        return root_message
    else:
        new_root_message = ChatMessage(
            chat_session_id=chat_session_id,
            prompt_id=None,
            parent_message=None,
            latest_child_message=None,
            message="",
            token_count=0,
            message_type=MessageType.SYSTEM,
        )
        db_session.add(new_root_message)
        db_session.commit()
        return new_root_message


def reserve_message_id(
    db_session: Session,
    chat_session_id: UUID,
    parent_message: int,
    message_type: MessageType,
) -> int:
    """
    预留消息ID
    
    参数:
    - db_session: 数据库会话
    - chat_session_id: 聊天会话ID
    - parent_message: 父消息ID
    - message_type: 消息类型
    
    返回: 预留的消息ID
    """
    # Create an empty chat message
    empty_message = ChatMessage(
        chat_session_id=chat_session_id,
        parent_message=parent_message,
        latest_child_message=None,
        message="",
        token_count=0,
        message_type=message_type,
    )

    # Add the empty message to the session
    db_session.add(empty_message)

    # Flush the session to get an ID for the new chat message
    db_session.flush()

    # Get the ID of the newly created message
    new_id = empty_message.id

    return new_id


def create_new_chat_message(
    chat_session_id: UUID,
    parent_message: ChatMessage,
    message: str,
    prompt_id: int | None,
    token_count: int,
    message_type: MessageType,
    db_session: Session,
    files: list[FileDescriptor] | None = None,
    rephrased_query: str | None = None,
    error: str | None = None,
    reference_docs: list[DBSearchDoc] | None = None,
    alternate_assistant_id: int | None = None,
    # Maps the citation number [n] to the DB SearchDoc
    citations: dict[int, int] | None = None,
    tool_call: ToolCall | None = None,
    commit: bool = True,
    reserved_message_id: int | None = None,
    overridden_model: str | None = None,
) -> ChatMessage:
    """
    创建新的聊天消息
    
    参数:
    - chat_session_id: 聊天会话ID
    - parent_message: 父消息
    - message: 消息内容
    - prompt_id: 提示词ID
    - token_count: token数量
    - message_type: 消息类型
    - db_session: 数据库会话
    - files: 相关文件列表
    - rephrased_query: 重新表述的查询
    - error: 错误信息
    - reference_docs: 参考文档
    - alternate_assistant_id: 备选助手ID
    - citations: 引用
    - tool_call: 工具调用
    - commit: 是否提交事务
    - reserved_message_id: 预留消息ID
    - overridden_model: 覆盖模型
    
    返回: 新创建的ChatMessage对象
    """
    if reserved_message_id is not None:
        # Edit existing message
        existing_message = db_session.query(ChatMessage).get(reserved_message_id)
        if existing_message is None:
            raise ValueError(f"No message found with id {reserved_message_id}")

        existing_message.chat_session_id = chat_session_id
        existing_message.parent_message = parent_message.id
        existing_message.message = message
        existing_message.rephrased_query = rephrased_query
        existing_message.prompt_id = prompt_id
        existing_message.token_count = token_count
        existing_message.message_type = message_type
        existing_message.citations = citations
        existing_message.files = files
        existing_message.tool_call = tool_call
        existing_message.error = error
        existing_message.alternate_assistant_id = alternate_assistant_id
        existing_message.overridden_model = overridden_model

        new_chat_message = existing_message
    else:
        # Create new message
        new_chat_message = ChatMessage(
            chat_session_id=chat_session_id,
            parent_message=parent_message.id,
            latest_child_message=None,
            message=message,
            rephrased_query=rephrased_query,
            prompt_id=prompt_id,
            token_count=token_count,
            message_type=message_type,
            citations=citations,
            files=files,
            tool_call=tool_call,
            error=error,
            alternate_assistant_id=alternate_assistant_id,
            overridden_model=overridden_model,
        )
        db_session.add(new_chat_message)

    # SQL Alchemy will propagate this to update the reference_docs' foreign keys
    if reference_docs:
        new_chat_message.search_docs = reference_docs

    # Flush the session to get an ID for the new chat message
    db_session.flush()

    parent_message.latest_child_message = new_chat_message.id
    if commit:
        db_session.commit()

    return new_chat_message


def set_as_latest_chat_message(
    chat_message: ChatMessage,
    user_id: UUID | None,
    db_session: Session,
) -> None:
    """
    将指定消息设置为最新消息
    
    参数:
    - chat_message: 聊天消息对象
    - user_id: 用户ID
    - db_session: 数据库会话
    """
    parent_message_id = chat_message.parent_message

    if parent_message_id is None:
        raise RuntimeError(
            f"Trying to set a latest message without parent, message id: {chat_message.id}"
        )

    parent_message = get_chat_message(
        chat_message_id=parent_message_id, user_id=user_id, db_session=db_session
    )

    parent_message.latest_child_message = chat_message.id

    db_session.commit()


def attach_files_to_chat_message(
    chat_message: ChatMessage,
    files: list[FileDescriptor],
    db_session: Session,
    commit: bool = True,
) -> None:
    """
    为聊天消息附加文件
    
    参数:
    - chat_message: 聊天消息对象
    - files: 文件描述符列表
    - db_session: 数据库会话
    - commit: 是否立即提交事务
    """
    chat_message.files = files
    if commit:
        db_session.commit()


def get_prompt_by_id(
    prompt_id: int,
    user: User | None,
    db_session: Session,
    include_deleted: bool = False,
) -> Prompt:
    """
    获取提示词
    
    参数:
    - prompt_id: 提示词ID
    - user: 用户对象
    - db_session: 数据库会话
    - include_deleted: 是否包含已删除的提示词
    
    返回: Prompt对象
    """
    stmt = select(Prompt).where(Prompt.id == prompt_id)

    # if user is not specified OR they are an admin, they should
    # have access to all prompts, so this where clause is not needed
    if user and user.role != UserRole.ADMIN:
        stmt = stmt.where(or_(Prompt.user_id == user.id, Prompt.user_id.is_(None)))

    if not include_deleted:
        stmt = stmt.where(Prompt.deleted.is_(False))

    result = db_session.execute(stmt)
    prompt = result.scalar_one_or_none()

    if prompt is None:
        raise ValueError(
            f"Prompt with ID {prompt_id} does not exist or does not belong to user"
        )

    return prompt


def get_doc_query_identifiers_from_model(
    search_doc_ids: list[int],
    chat_session: ChatSession,
    user_id: UUID | None,
    db_session: Session,
    enforce_chat_session_id_for_search_docs: bool,
) -> list[tuple[str, int]]:
    """
    根据搜索文档ID获取文档查询标识符
    
    参数:
    - search_doc_ids: 搜索文档ID列表
    - chat_session: 聊天会话对象
    - user_id: 用户ID
    - db_session: 数据库会话
    - enforce_chat_session_id_for_search_docs: 是否强制检查文档所属会话
    
    返回: 文档ID和分块索引的元组列表
    """
    """Given a list of search_doc_ids"""
    search_docs = (
        db_session.query(SearchDoc).filter(SearchDoc.id.in_(search_doc_ids)).all()
    )

    if user_id != chat_session.user_id:
        logger.error(
            f"Docs referenced are from a chat session not belonging to user {user_id}"
        )
        raise ValueError("Docs references do not belong to user")

    try:
        if any(
            [
                doc.chat_messages[0].chat_session_id != chat_session.id
                for doc in search_docs
            ]
        ):
            if enforce_chat_session_id_for_search_docs:
                raise ValueError("Invalid reference doc, not from this chat session.")
    except IndexError:
        # This happens when the doc has no chat_messages associated with it.
        # which happens as an edge case where the chat message failed to save
        # This usually happens when the LLM fails either immediately or partially through.
        raise RuntimeError("Chat session failed, please start a new session.")

    doc_query_identifiers = [(doc.document_id, doc.chunk_ind) for doc in search_docs]

    return doc_query_identifiers


def update_search_docs_table_with_relevance(
    db_session: Session,
    reference_db_search_docs: list[SearchDoc],
    relevance_summary: DocumentRelevance,
) -> None:
    """
    更新搜索文档表中的相关性信息
    
    参数:
    - db_session: 数据库会话
    - reference_db_search_docs: 搜索文档列表
    - relevance_summary: 文档相关性总结
    """
    for search_doc in reference_db_search_docs:
        relevance_data = relevance_summary.relevance_summaries.get(
            search_doc.document_id
        )
        if relevance_data is not None:
            db_session.execute(
                update(SearchDoc)
                .where(SearchDoc.id == search_doc.id)
                .values(
                    is_relevant=relevance_data.relevant,
                    relevance_explanation=relevance_data.content,
                )
            )
    db_session.commit()


def create_db_search_doc(
    server_search_doc: ServerSearchDoc,
    db_session: Session,
) -> SearchDoc:
    """
    创建数据库搜索文档
    
    参数:
    - server_search_doc: 服务器搜索文档对象
    - db_session: 数据库会话
    
    返回: SearchDoc对象
    """
    db_search_doc = SearchDoc(
        document_id=server_search_doc.document_id,
        chunk_ind=server_search_doc.chunk_ind,
        semantic_id=server_search_doc.semantic_identifier,
        link=server_search_doc.link,
        blurb=server_search_doc.blurb,
        source_type=server_search_doc.source_type,
        boost=server_search_doc.boost,
        hidden=server_search_doc.hidden,
        doc_metadata=server_search_doc.metadata,
        is_relevant=server_search_doc.is_relevant,
        relevance_explanation=server_search_doc.relevance_explanation,
        # For docs further down that aren't reranked, we can't use the retrieval score
        score=server_search_doc.score or 0.0,
        match_highlights=server_search_doc.match_highlights,
        updated_at=server_search_doc.updated_at,
        primary_owners=server_search_doc.primary_owners,
        secondary_owners=server_search_doc.secondary_owners,
        is_internet=server_search_doc.is_internet,
    )

    db_session.add(db_search_doc)
    db_session.commit()
    return db_search_doc


def get_db_search_doc_by_id(doc_id: int, db_session: Session) -> DBSearchDoc | None:
    """There are no safety checks here like user permission etc., use with caution"""
    search_doc = db_session.query(SearchDoc).filter(SearchDoc.id == doc_id).first()
    return search_doc


def translate_db_search_doc_to_server_search_doc(
    db_search_doc: SearchDoc,
    remove_doc_content: bool = False,
) -> SavedSearchDoc:
    """
    将数据库搜索文档对象转换为服务器搜索文档对象
    
    参数:
    - db_search_doc: 数据库搜索文档对象
    - remove_doc_content: 是否移除文档内容
    
    返回: SavedSearchDoc对象
    """
    return SavedSearchDoc(
        db_doc_id=db_search_doc.id,
        document_id=db_search_doc.document_id,
        chunk_ind=db_search_doc.chunk_ind,
        semantic_identifier=db_search_doc.semantic_id,
        link=db_search_doc.link,
        blurb=db_search_doc.blurb if not remove_doc_content else "",
        source_type=db_search_doc.source_type,
        boost=db_search_doc.boost,
        hidden=db_search_doc.hidden,
        metadata=db_search_doc.doc_metadata if not remove_doc_content else {},
        score=db_search_doc.score,
        match_highlights=(
            db_search_doc.match_highlights if not remove_doc_content else []
        ),
        relevance_explanation=db_search_doc.relevance_explanation,
        is_relevant=db_search_doc.is_relevant,
        updated_at=db_search_doc.updated_at if not remove_doc_content else None,
        primary_owners=db_search_doc.primary_owners if not remove_doc_content else [],
        secondary_owners=(
            db_search_doc.secondary_owners if not remove_doc_content else []
        ),
        is_internet=db_search_doc.is_internet,
    )


def get_retrieval_docs_from_chat_message(
    chat_message: ChatMessage, remove_doc_content: bool = False
) -> RetrievalDocs:
    """
    从聊天消息中获取检索文档
    
    参数:
    - chat_message: 聊天消息对象
    - remove_doc_content: 是否移除文档内容
    
    返回: RetrievalDocs对象
    """
    top_documents = [
        translate_db_search_doc_to_server_search_doc(
            db_doc, remove_doc_content=remove_doc_content
        )
        for db_doc in chat_message.search_docs
    ]
    top_documents = sorted(top_documents, key=lambda doc: doc.score, reverse=True)  # type: ignore
    return RetrievalDocs(top_documents=top_documents)


def translate_db_message_to_chat_message_detail(
    chat_message: ChatMessage,
    remove_doc_content: bool = False,
) -> ChatMessageDetail:
    """
    将数据库消息对象转换为聊天消息详情对象
    
    参数:
    - chat_message: 聊天消息对象
    - remove_doc_content: 是否移除文档内容
    
    返回: ChatMessageDetail对象
    """
    chat_msg_detail = ChatMessageDetail(
        chat_session_id=chat_message.chat_session_id,
        message_id=chat_message.id,
        parent_message=chat_message.parent_message,
        latest_child_message=chat_message.latest_child_message,
        message=chat_message.message,
        rephrased_query=chat_message.rephrased_query,
        context_docs=get_retrieval_docs_from_chat_message(
            chat_message, remove_doc_content=remove_doc_content
        ),
        message_type=chat_message.message_type,
        time_sent=chat_message.time_sent,
        citations=chat_message.citations,
        files=chat_message.files or [],
        tool_call=ToolCallFinalResult(
            tool_name=chat_message.tool_call.tool_name,
            tool_args=chat_message.tool_call.tool_arguments,
            tool_result=chat_message.tool_call.tool_result,
        )
        if chat_message.tool_call
        else None,
        alternate_assistant_id=chat_message.alternate_assistant_id,
        overridden_model=chat_message.overridden_model,
    )

    return chat_msg_detail
