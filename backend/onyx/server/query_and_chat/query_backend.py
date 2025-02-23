"""
这个模块提供了查询和聊天相关的API接口实现。
主要功能包括：
1. 管理员搜索接口
2. 标签查询接口
3. 用户搜索会话管理
4. 搜索会话详情查询
"""

from uuid import UUID

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from sqlalchemy.orm import Session

from onyx.auth.users import current_curator_or_admin_user
from onyx.auth.users import current_user
from onyx.configs.constants import DocumentSource
from onyx.configs.constants import MessageType
from onyx.context.search.models import IndexFilters
from onyx.context.search.models import SearchDoc
from onyx.context.search.preprocessing.access_filters import (
    build_access_filters_for_user,
)
from onyx.context.search.utils import chunks_or_sections_to_search_docs
from onyx.db.chat import get_chat_messages_by_session
from onyx.db.chat import get_chat_session_by_id
from onyx.db.chat import get_chat_sessions_by_user
from onyx.db.chat import get_search_docs_for_chat_message
from onyx.db.chat import get_valid_messages_from_query_sessions
from onyx.db.chat import translate_db_message_to_chat_message_detail
from onyx.db.chat import translate_db_search_doc_to_server_search_doc
from onyx.db.engine import get_current_tenant_id
from onyx.db.engine import get_session
from onyx.db.models import User
from onyx.db.search_settings import get_current_search_settings
from onyx.db.tag import find_tags
from onyx.document_index.factory import get_default_document_index
from onyx.document_index.vespa.index import VespaIndex
from onyx.server.query_and_chat.models import AdminSearchRequest
from onyx.server.query_and_chat.models import AdminSearchResponse
from onyx.server.query_and_chat.models import ChatSessionDetails
from onyx.server.query_and_chat.models import ChatSessionsResponse
from onyx.server.query_and_chat.models import SearchSessionDetailResponse
from onyx.server.query_and_chat.models import SourceTag
from onyx.server.query_and_chat.models import TagResponse
from onyx.utils.logger import setup_logger

logger = setup_logger()

admin_router = APIRouter(prefix="/admin")
basic_router = APIRouter(prefix="/query")


@admin_router.post("/search")
def admin_search(
    question: AdminSearchRequest,
    user: User | None = Depends(current_curator_or_admin_user),
    db_session: Session = Depends(get_session),
    tenant_id: str = Depends(get_current_tenant_id),
) -> AdminSearchResponse:
    """
    管理员搜索接口
    
    参数:
        question: 管理员搜索请求对象
        user: 当前用户对象，必须是管理员或策展人
        db_session: 数据库会话
        tenant_id: 租户ID
    
    返回:
        AdminSearchResponse: 包含搜索结果的响应对象
    """
    query = question.query
    logger.notice(f"Received admin search query: {query}")
    user_acl_filters = build_access_filters_for_user(user, db_session)
    final_filters = IndexFilters(
        source_type=question.filters.source_type,
        document_set=question.filters.document_set,
        time_cutoff=question.filters.time_cutoff,
        tags=question.filters.tags,
        access_control_list=user_acl_filters,
        tenant_id=tenant_id,
    )
    search_settings = get_current_search_settings(db_session)
    document_index = get_default_document_index(
        primary_index_name=search_settings.index_name, secondary_index_name=None
    )
    if not isinstance(document_index, VespaIndex):
        raise HTTPException(
            status_code=400,
            detail="Cannot use admin-search when using a non-Vespa document index",
        )
    matching_chunks = document_index.admin_retrieval(query=query, filters=final_filters)

    documents = chunks_or_sections_to_search_docs(matching_chunks)

    # Deduplicate documents by id
    deduplicated_documents: list[SearchDoc] = []
    seen_documents: set[str] = set()
    for document in documents:
        if document.document_id not in seen_documents:
            deduplicated_documents.append(document)
            seen_documents.add(document.document_id)
    return AdminSearchResponse(documents=deduplicated_documents)


@basic_router.get("/valid-tags")
def get_tags(
    match_pattern: str | None = None,
    # If this is empty or None, then tags for all sources are considered
    sources: list[DocumentSource] | None = None,
    allow_prefix: bool = True,  # This is currently the only option
    limit: int = 50,
    _: User = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> TagResponse:
    """
    获取有效标签列表
    
    参数:
        match_pattern: 匹配模式字符串
        sources: 文档来源列表
        allow_prefix: 是否允许前缀匹配
        limit: 返回结果数量限制
        _: 当前用户对象
        db_session: 数据库会话
    
    返回:
        TagResponse: 包含标签列表的响应对象
    """
    if not allow_prefix:
        raise NotImplementedError("Cannot disable prefix match for now")

    key_prefix = match_pattern
    value_prefix = match_pattern
    require_both_to_match = False

    # split on = to allow the user to type in "author=bob"
    EQUAL_PAT = "="
    if match_pattern and EQUAL_PAT in match_pattern:
        split_pattern = match_pattern.split(EQUAL_PAT)
        key_prefix = split_pattern[0]
        value_prefix = EQUAL_PAT.join(split_pattern[1:])
        require_both_to_match = True

    db_tags = find_tags(
        tag_key_prefix=key_prefix,
        tag_value_prefix=value_prefix,
        sources=sources,
        limit=limit,
        db_session=db_session,
        require_both_to_match=require_both_to_match,
    )
    server_tags = [
        SourceTag(
            tag_key=db_tag.tag_key, tag_value=db_tag.tag_value, source=db_tag.source
        )
        for db_tag in db_tags
    ]
    return TagResponse(tags=server_tags)


@basic_router.get("/user-searches")
def get_user_search_sessions(
    user: User | None = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> ChatSessionsResponse:
    """
    获取用户搜索会话列表
    
    参数:
        user: 当前用户对象
        db_session: 数据库会话
    
    返回:
        ChatSessionsResponse: 包含用户搜索会话列表的响应对象
    
    异常:
        HTTPException: 当会话不存在或已被删除时抛出
    """
    user_id = user.id if user is not None else None

    try:
        search_sessions = get_chat_sessions_by_user(
            user_id=user_id, deleted=False, db_session=db_session
        )
    except ValueError:
        raise HTTPException(
            status_code=404, detail="Chat session does not exist or has been deleted"
        )
    # Extract IDs from search sessions
    search_session_ids = [chat.id for chat in search_sessions]
    # Fetch first messages for each session, only including those with documents
    sessions_with_documents = get_valid_messages_from_query_sessions(
        search_session_ids, db_session
    )
    sessions_with_documents_dict = dict(sessions_with_documents)

    # Prepare response with detailed information for each valid search session
    response = ChatSessionsResponse(
        sessions=[
            ChatSessionDetails(
                id=search.id,
                name=sessions_with_documents_dict[search.id],
                persona_id=search.persona_id,
                time_created=search.time_created.isoformat(),
                shared_status=search.shared_status,
                folder_id=search.folder_id,
                current_alternate_model=search.current_alternate_model,
            )
            for search in search_sessions
            if search.id
            in sessions_with_documents_dict  # Only include sessions with documents
        ]
    )

    return response


@basic_router.get("/search-session/{session_id}")
def get_search_session(
    session_id: UUID,
    is_shared: bool = False,
    user: User | None = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> SearchSessionDetailResponse:
    """
    获取搜索会话详情
    
    参数:
        session_id: 会话ID
        is_shared: 是否是共享会话
        user: 当前用户对象
        db_session: 数据库会话
    
    返回:
        SearchSessionDetailResponse: 包含会话详细信息的响应对象
    
    异常:
        ValueError: 当会话不存在或已被删除时抛出
    """
    user_id = user.id if user is not None else None

    try:
        search_session = get_chat_session_by_id(
            chat_session_id=session_id,
            user_id=user_id,
            db_session=db_session,
            is_shared=is_shared,
        )
    except ValueError:
        raise ValueError("Search session does not exist or has been deleted")

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
    docs_response: list[SearchDoc] = []
    for message in session_messages:
        if (
            message.message_type == MessageType.ASSISTANT
            or message.message_type == MessageType.SYSTEM
        ):
            docs = get_search_docs_for_chat_message(
                db_session=db_session, chat_message_id=message.id
            )
            for doc in docs:
                server_doc = translate_db_search_doc_to_server_search_doc(doc)
                docs_response.append(server_doc)

    response = SearchSessionDetailResponse(
        search_session_id=session_id,
        description=search_session.description,
        documents=docs_response,
        messages=[
            translate_db_message_to_chat_message_detail(
                msg, remove_doc_content=is_shared  # if shared, don't leak doc content
            )
            for msg in session_messages
        ],
    )
    return response
