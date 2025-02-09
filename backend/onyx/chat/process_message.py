"""
这个文件是聊天消息处理的核心模块,主要功能包括:
1. 处理和流式输出聊天消息
2. 管理聊天会话和消息链
3. 处理文档搜索和引用
4. 集成 LLM 模型生成回复
5. 处理工具调用(搜索、图片生成等)
"""

import traceback
from collections.abc import Callable
from collections.abc import Iterator
from functools import partial
from typing import cast

from sqlalchemy.orm import Session

from onyx.chat.answer import Answer
from onyx.chat.chat_utils import create_chat_chain
from onyx.chat.chat_utils import create_temporary_persona
from onyx.chat.models import AllCitations
from onyx.chat.models import AnswerStyleConfig
from onyx.chat.models import ChatOnyxBotResponse
from onyx.chat.models import CitationConfig
from onyx.chat.models import CitationInfo
from onyx.chat.models import CustomToolResponse
from onyx.chat.models import DocumentPruningConfig
from onyx.chat.models import FileChatDisplay
from onyx.chat.models import FinalUsedContextDocsResponse
from onyx.chat.models import LLMRelevanceFilterResponse
from onyx.chat.models import MessageResponseIDInfo
from onyx.chat.models import MessageSpecificCitations
from onyx.chat.models import OnyxAnswerPiece
from onyx.chat.models import OnyxContexts
from onyx.chat.models import PromptConfig
from onyx.chat.models import QADocsResponse
from onyx.chat.models import StreamingError
from onyx.chat.models import StreamStopInfo
from onyx.configs.chat_configs import CHAT_TARGET_CHUNK_PERCENTAGE
from onyx.configs.chat_configs import DISABLE_LLM_CHOOSE_SEARCH
from onyx.configs.chat_configs import MAX_CHUNKS_FED_TO_CHAT
from onyx.configs.constants import MessageType
from onyx.configs.constants import MilestoneRecordType
from onyx.configs.constants import NO_AUTH_USER_ID
from onyx.context.search.enums import OptionalSearchSetting
from onyx.context.search.enums import QueryFlow
from onyx.context.search.enums import SearchType
from onyx.context.search.models import InferenceSection
from onyx.context.search.models import RetrievalDetails
from onyx.context.search.retrieval.search_runner import inference_sections_from_ids
from onyx.context.search.utils import chunks_or_sections_to_search_docs
from onyx.context.search.utils import dedupe_documents
from onyx.context.search.utils import drop_llm_indices
from onyx.context.search.utils import relevant_sections_to_indices
from onyx.db.chat import attach_files_to_chat_message
from onyx.db.chat import create_db_search_doc
from onyx.db.chat import create_new_chat_message
from onyx.db.chat import get_chat_message
from onyx.db.chat import get_chat_session_by_id
from onyx.db.chat import get_db_search_doc_by_id
from onyx.db.chat import get_doc_query_identifiers_from_model
from onyx.db.chat import get_or_create_root_message
from onyx.db.chat import reserve_message_id
from onyx.db.chat import translate_db_message_to_chat_message_detail
from onyx.db.chat import translate_db_search_doc_to_server_search_doc
from onyx.db.engine import get_session_context_manager
from onyx.db.milestone import check_multi_assistant_milestone
from onyx.db.milestone import create_milestone_if_not_exists
from onyx.db.milestone import update_user_assistant_milestone
from onyx.db.models import SearchDoc as DbSearchDoc
from onyx.db.models import ToolCall
from onyx.db.models import User
from onyx.db.persona import get_persona_by_id
from onyx.db.search_settings import get_current_search_settings
from onyx.document_index.factory import get_default_document_index
from onyx.file_store.models import ChatFileType
from onyx.file_store.models import FileDescriptor
from onyx.file_store.utils import load_all_chat_files
from onyx.file_store.utils import save_files
from onyx.llm.exceptions import GenAIDisabledException
from onyx.llm.factory import get_llms_for_persona
from onyx.llm.factory import get_main_llm_from_tuple
from onyx.llm.models import PreviousMessage
from onyx.llm.utils import litellm_exception_to_error_msg
from onyx.natural_language_processing.utils import get_tokenizer
from onyx.server.query_and_chat.models import ChatMessageDetail
from onyx.server.query_and_chat.models import CreateChatMessageRequest
from onyx.server.utils import get_json_line
from onyx.tools.force import ForceUseTool
from onyx.tools.models import ToolResponse
from onyx.tools.tool import Tool
from onyx.tools.tool_constructor import construct_tools
from onyx.tools.tool_constructor import CustomToolConfig
from onyx.tools.tool_constructor import ImageGenerationToolConfig
from onyx.tools.tool_constructor import InternetSearchToolConfig
from onyx.tools.tool_constructor import SearchToolConfig
from onyx.tools.tool_implementations.custom.custom_tool import (
    CUSTOM_TOOL_RESPONSE_ID,
)
from onyx.tools.tool_implementations.custom.custom_tool import CustomToolCallSummary
from onyx.tools.tool_implementations.images.image_generation_tool import (
    IMAGE_GENERATION_RESPONSE_ID,
)
from onyx.tools.tool_implementations.images.image_generation_tool import (
    ImageGenerationResponse,
)
from onyx.tools.tool_implementations.internet_search.internet_search_tool import (
    INTERNET_SEARCH_RESPONSE_ID,
)
from onyx.tools.tool_implementations.internet_search.internet_search_tool import (
    internet_search_response_to_search_docs,
)
from onyx.tools.tool_implementations.internet_search.internet_search_tool import (
    InternetSearchResponse,
)
from onyx.tools.tool_implementations.internet_search.internet_search_tool import (
    InternetSearchTool,
)
from onyx.tools.tool_implementations.search.search_tool import (
    FINAL_CONTEXT_DOCUMENTS_ID,
)
from onyx.tools.tool_implementations.search.search_tool import SEARCH_DOC_CONTENT_ID
from onyx.tools.tool_implementations.search.search_tool import (
    SEARCH_RESPONSE_SUMMARY_ID,
)
from onyx.tools.tool_implementations.search.search_tool import SearchResponseSummary
from onyx.tools.tool_implementations.search.search_tool import SearchTool
from onyx.tools.tool_implementations.search.search_tool import (
    SECTION_RELEVANCE_LIST_ID,
)
from onyx.tools.tool_runner import ToolCallFinalResult
from onyx.utils.logger import setup_logger
from onyx.utils.long_term_log import LongTermLogger
from onyx.utils.telemetry import mt_cloud_telemetry
from onyx.utils.timing import log_function_time
from onyx.utils.timing import log_generator_function_time
from shared_configs.contextvars import CURRENT_TENANT_ID_CONTEXTVAR


logger = setup_logger()


def _translate_citations(
    citations_list: list[CitationInfo], db_docs: list[DbSearchDoc]
) -> MessageSpecificCitations:
    """将引用信息转换为消息特定的引用格式
    
    Always cites the first instance of the document_id, assumes the db_docs 
    are sorted in the order displayed in the UI
    总是引用文档ID的第一个实例,假设db_docs按UI显示顺序排序
    
    Args:
        citations_list: 引用信息列表
        db_docs: 数据库文档列表
    
    Returns:
        MessageSpecificCitations: 转换后的消息特定引用信息
    """
    doc_id_to_saved_doc_id_map: dict[str, int] = {}
    for db_doc in db_docs:
        if db_doc.document_id not in doc_id_to_saved_doc_id_map:
            doc_id_to_saved_doc_id_map[db_doc.document_id] = db_doc.id

    citation_to_saved_doc_id_map: dict[int, int] = {}
    for citation in citations_list:
        if citation.citation_num not in citation_to_saved_doc_id_map:
            citation_to_saved_doc_id_map[
                citation.citation_num
            ] = doc_id_to_saved_doc_id_map[citation.document_id]

    return MessageSpecificCitations(citation_map=citation_to_saved_doc_id_map)


def _handle_search_tool_response_summary(
    packet: ToolResponse,
    db_session: Session,
    selected_search_docs: list[DbSearchDoc] | None,
    dedupe_docs: bool = False,
) -> tuple[QADocsResponse, list[DbSearchDoc], list[int] | None]:
    """处理搜索工具的响应结果摘要
    
    Args:
        packet: 工具响应包
        db_session: 数据库会话
        selected_search_docs: 已选择的搜索文档列表
        dedupe_docs: 是否去重文档
        
    Returns:
        tuple: 包含QA文档响应、参考DB搜索文档列表、被丢弃的索引列表
    """
    response_sumary = cast(SearchResponseSummary, packet.response)

    dropped_inds = None
    if not selected_search_docs:
        top_docs = chunks_or_sections_to_search_docs(response_sumary.top_sections)

        deduped_docs = top_docs
        # 去重在最后一步进行,以避免过早丢弃内容而影响质量
        if dedupe_docs:
            deduped_docs, dropped_inds = dedupe_documents(top_docs)

        reference_db_search_docs = [
            create_db_search_doc(server_search_doc=doc, db_session=db_session)
            for doc in deduped_docs
        ]
    else:
        reference_db_search_docs = selected_search_docs

    response_docs = [
        translate_db_search_doc_to_server_search_doc(db_search_doc)
        for db_search_doc in reference_db_search_docs
    ]
    return (
        QADocsResponse(
            rephrased_query=response_sumary.rephrased_query,
            top_documents=response_docs,
            predicted_flow=response_sumary.predicted_flow,
            predicted_search=response_sumary.predicted_search,
            applied_source_filters=response_sumary.final_filters.source_type,
            applied_time_cutoff=response_sumary.final_filters.time_cutoff,
            recency_bias_multiplier=response_sumary.recency_bias_multiplier,
        ),
        reference_db_search_docs,
        dropped_inds,
    )


def _handle_internet_search_tool_response_summary(
    packet: ToolResponse,
    db_session: Session,
) -> tuple[QADocsResponse, list[DbSearchDoc]]:
    """处理互联网搜索工具的响应结果摘要
    
    Args:
        packet: 工具响应包 
        db_session: 数据库会话
        
    Returns:
        tuple: 包含QA文档响应和参考DB搜索文档列表
    """
    internet_search_response = cast(InternetSearchResponse, packet.response)
    server_search_docs = internet_search_response_to_search_docs(
        internet_search_response
    )

    reference_db_search_docs = [
        create_db_search_doc(server_search_doc=doc, db_session=db_session)
        for doc in server_search_docs
    ]
    response_docs = [
        translate_db_search_doc_to_server_search_doc(db_search_doc)
        for db_search_doc in reference_db_search_docs
    ]
    return (
        QADocsResponse(
            rephrased_query=internet_search_response.revised_query,
            top_documents=response_docs,
            predicted_flow=QueryFlow.QUESTION_ANSWER,
            predicted_search=SearchType.SEMANTIC,
            applied_source_filters=[],
            applied_time_cutoff=None,
            recency_bias_multiplier=1.0,
        ),
        reference_db_search_docs,
    )


def _get_force_search_settings(
    new_msg_req: CreateChatMessageRequest, tools: list[Tool]
) -> ForceUseTool:
    """获取强制搜索设置
    
    Args:
        new_msg_req: 创建消息请求
        tools: 可用工具列表
        
    Returns:
        ForceUseTool: 强制使用工具的设置
    """
    internet_search_available = any(
        isinstance(tool, InternetSearchTool) for tool in tools
    )
    search_tool_available = any(isinstance(tool, SearchTool) for tool in tools)

    if not internet_search_available and not search_tool_available:
        # Does not matter much which tool is set here as force is false and neither tool is available
        # 这里设置哪个工具并不重要,因为force是false且没有工具可用
        return ForceUseTool(force_use=False, tool_name=SearchTool._NAME)

    tool_name = SearchTool._NAME if search_tool_available else InternetSearchTool._NAME
    # Currently, the internet search tool does not support query override
    # 当前,互联网搜索工具不支持查询覆盖
    args = (
        {"query": new_msg_req.query_override}
        if new_msg_req.query_override and tool_name == SearchTool._NAME
        else None
    )

    if new_msg_req.file_descriptors:
        # If user has uploaded files they're using, don't run any of the search tools
        # 如果用户上传了他们正在使用的文件,不要运行任何搜索工具
        return ForceUseTool(force_use=False, tool_name=tool_name)

    should_force_search = any(
        [
            new_msg_req.retrieval_options
            and new_msg_req.retrieval_options.run_search
            == OptionalSearchSetting.ALWAYS,
            new_msg_req.search_doc_ids,
            DISABLE_LLM_CHOOSE_SEARCH,
        ]
    )

    if should_force_search:
        # If we are using selected docs, just put something here so the Tool doesn't need to build its own args via an LLM call
        # 如果我们使用选定的文档,只需在此处放置一些内容,这样工具就不需要通过LLM调用来构建自己的参数
        args = {"query": new_msg_req.message} if new_msg_req.search_doc_ids else args
        return ForceUseTool(force_use=True, tool_name=tool_name, args=args)

    return ForceUseTool(force_use=False, tool_name=tool_name, args=args)


ChatPacket = (
    StreamingError
    | QADocsResponse
    | OnyxContexts
    | LLMRelevanceFilterResponse
    | FinalUsedContextDocsResponse
    | ChatMessageDetail
    | OnyxAnswerPiece
    | AllCitations
    | CitationInfo
    | FileChatDisplay
    | CustomToolResponse
    | MessageSpecificCitations
    | MessageResponseIDInfo
    | StreamStopInfo
)
ChatPacketStream = Iterator[ChatPacket]


def stream_chat_message_objects(
    new_msg_req: CreateChatMessageRequest,
    user: User | None,
    db_session: Session,
    # 需要将persona数量chunks转换为对应LLM的token数
    default_num_chunks: float = MAX_CHUNKS_FED_TO_CHAT,
    # 对于需要搜索的流程,不要包含太多chunks因为我们需要保留空间给聊天历史
    # 对于较小的模型,我们可能无法获得 MAX_CHUNKS_FED_TO_CHAT 数量的chunks 
    max_document_percentage: float = CHAT_TARGET_CHUNK_PERCENTAGE,
    # 如果指定了,则使用最后一条用户消息,而不是基于new_msg_req.message创建新的用户消息
    litellm_additional_headers: dict[str, str] | None = None,
    custom_tool_additional_headers: dict[str, str] | None = None,
    is_connected: Callable[[], bool] | None = None,
    enforce_chat_session_id_for_search_docs: bool = True,
    bypass_acl: bool = False,
    include_contexts: bool = False,
) -> ChatPacketStream:
    """流式输出聊天消息对象
    
    按以下顺序流式输出:
    1. [条件性] 如果需要进行搜索,则返回检索到的文档
    2. [条件性] 如果启用了LLM内容过滤,则返回由LLM选择的chunk索引
    3. [必然] 流式输出LLM生成的token,如果过程中发生错误则返回错误信息
    4. [必然] 返回最终AI响应消息的详细信息

    Args:
        new_msg_req: 创建新消息的请求
        user: 用户对象
        db_session: 数据库会话
        default_num_chunks: 默认chunk数量,用于将persona的num_chunks转换为LLM的token数
        max_document_percentage: 文档使用的最大百分比
        litellm_additional_headers: litellm的附加headers
        custom_tool_additional_headers: 自定义工具的附加headers
        is_connected: 检查连接状态的回调函数
        enforce_chat_session_id_for_search_docs: 是否强制要求搜索文档有聊天会话ID
        bypass_acl: 是否绕过访问控制
        include_contexts: 是否包含上下文
        
    Returns:
        ChatPacketStream: 聊天消息包的迭代器
    """
    tenant_id = CURRENT_TENANT_ID_CONTEXTVAR.get()
    use_existing_user_message = new_msg_req.use_existing_user_message
    existing_assistant_message_id = new_msg_req.existing_assistant_message_id

    # 当前不支持聊天的上下文
    # 聊天已经很消耗token且模型处理较困难,而且会更快地滚动历史记录
    new_msg_req.chunks_above = 0
    new_msg_req.chunks_below = 0

    try:
        user_id = user.id if user is not None else None

        chat_session = get_chat_session_by_id(
            chat_session_id=new_msg_req.chat_session_id,
            user_id=user_id,
            db_session=db_session,
        )

        message_text = new_msg_req.message
        chat_session_id = new_msg_req.chat_session_id
        parent_id = new_msg_req.parent_message_id
        reference_doc_ids = new_msg_req.search_doc_ids
        retrieval_options = new_msg_req.retrieval_options
        alternate_assistant_id = new_msg_req.alternate_assistant_id

        # 永久"日志"存储,主要用于调试
        long_term_logger = LongTermLogger(
            metadata={"user_id": str(user_id), "chat_session_id": str(chat_session_id)}
        )

        if alternate_assistant_id is not None:
            # 允许用户在聊天会话中使用临时助手
            # 这具有最高优先级,因为是用户指定的
            persona = get_persona_by_id(
                alternate_assistant_id,
                user=user,
                db_session=db_session,
                is_for_edit=False,
            )
        elif new_msg_req.persona_override_config:
            # 某些端点允许用户指定任意助手设置
            # 这不应该与alternate_assistant_id冲突
            persona = create_temporary_persona(
                db_session=db_session,
                persona_config=new_msg_req.persona_override_config,
                user=user,
            )
        else:
            persona = chat_session.persona

        if not persona:
            raise RuntimeError("No persona specified or found for chat session")

        multi_assistant_milestone, _is_new = create_milestone_if_not_exists(
            user=user,
            event_type=MilestoneRecordType.MULTIPLE_ASSISTANTS,
            db_session=db_session,
        )

        update_user_assistant_milestone(
            milestone=multi_assistant_milestone,
            user_id=str(user.id) if user else NO_AUTH_USER_ID,
            assistant_id=persona.id,
            db_session=db_session,
        )

        _, just_hit_multi_assistant_milestone = check_multi_assistant_milestone(
            milestone=multi_assistant_milestone,
            db_session=db_session,
        )

        if just_hit_multi_assistant_milestone:
            mt_cloud_telemetry(
                distinct_id=tenant_id,
                event=MilestoneRecordType.MULTIPLE_ASSISTANTS,
                properties=None,
            )

        # 如果通过API指定了提示覆盖,则使用它作为最高优先级
        # 但为了保存,我们只是将其映射到现有提示
        prompt_id = new_msg_req.prompt_id
        if prompt_id is None and persona.prompts:
            prompt_id = sorted(persona.prompts, key=lambda x: x.id)[-1].id

        if reference_doc_ids is None and retrieval_options is None:
            raise RuntimeError(
                "Must specify a set of documents for chat or specify search options"
            )

        try:
            llm, fast_llm = get_llms_for_persona(
                persona=persona,
                llm_override=new_msg_req.llm_override or chat_session.llm_override,
                additional_headers=litellm_additional_headers,
                long_term_logger=long_term_logger,
            )
        except GenAIDisabledException:
            raise RuntimeError("LLM is disabled. Can't use chat flow without LLM.")

        llm_provider = llm.config.model_provider
        llm_model_name = llm.config.model_name

        llm_tokenizer = get_tokenizer(
            model_name=llm_model_name,
            provider_type=llm_provider,
        )
        llm_tokenizer_encode_func = cast(
            Callable[[str], list[int]], llm_tokenizer.encode
        )

        search_settings = get_current_search_settings(db_session)
        document_index = get_default_document_index(
            primary_index_name=search_settings.index_name, secondary_index_name=None
        )

        # 每个聊天会话都以一个空的根消息开始
        root_message = get_or_create_root_message(
            chat_session_id=chat_session_id, db_session=db_session
        )

        if parent_id is not None:
            parent_message = get_chat_message(
                chat_message_id=parent_id,
                user_id=user_id,
                db_session=db_session,
            )
        else:
            parent_message = root_message

        user_message = None

        if new_msg_req.regenerate:
            final_msg, history_msgs = create_chat_chain(
                stop_at_message_id=parent_id,
                chat_session_id=chat_session_id,
                db_session=db_session,
            )

        elif not use_existing_user_message:
            # 在树中的正确位置创建新消息并更新父节点的子指针
            # 在验证聊天消息链之前不要提交
            user_message = create_new_chat_message(
                chat_session_id=chat_session_id,
                parent_message=parent_message,
                prompt_id=prompt_id,
                message=message_text,
                token_count=len(llm_tokenizer_encode_func(message_text)),
                message_type=MessageType.USER,
                files=None,  # 需要稍后附加以优化并行加载文件
                db_session=db_session,
                commit=False,
            )
            # 重新创建消息的线性历史记录
            final_msg, history_msgs = create_chat_chain(
                chat_session_id=chat_session_id, db_session=db_session
            )
            if final_msg.id != user_message.id:
                db_session.rollback()
                raise RuntimeError(
                    "The new message was not on the mainline. "
                    "Be sure to update the chat pointers before calling this."
                )

            # 注意:不要提交用户消息 - 它将在助手消息成功生成时提交
        else:
            # 重新创建消息的线性历史记录
            final_msg, history_msgs = create_chat_chain(
                chat_session_id=chat_session_id, db_session=db_session
            )
            if existing_assistant_message_id is None:
                if final_msg.message_type != MessageType.USER:
                    raise RuntimeError(
                        "The last message was not a user message. Cannot call "
                        "`stream_chat_message_objects` with `is_regenerate=True` "
                        "when the last message is not a user message."
                    )
            else:
                if final_msg.id != existing_assistant_message_id:
                    raise RuntimeError(
                        "The last message was not the existing assistant message. "
                        f"Final message id: {final_msg.id}, "
                        f"existing assistant message id: {existing_assistant_message_id}"
                    )

        # 禁用第一条消息的查询重述
        # 因为LLM重述问题会导致搜索质量下降,所以这样可以得到更好的首次响应
        if not history_msgs:
            new_msg_req.query_override = (
                new_msg_req.query_override or new_msg_req.message
            )

        # 在内存中加载此聊天链所需的所有文件
        files = load_all_chat_files(
            history_msgs, new_msg_req.file_descriptors, db_session
        )
        latest_query_files = [
            file
            for file in files
            if file.file_id in [f["id"] for f in new_msg_req.file_descriptors]
        ]

        if user_message:
            attach_files_to_chat_message(
                chat_message=user_message,
                files=[
                    new_file.to_file_descriptor() for new_file in latest_query_files
                ],
                db_session=db_session,
                commit=False,
            )

        selected_db_search_docs = None
        selected_sections: list[InferenceSection] | None = None
        if reference_doc_ids:
            identifier_tuples = get_doc_query_identifiers_from_model(
                search_doc_ids=reference_doc_ids,
                chat_session=chat_session,
                user_id=user_id,
                db_session=db_session,
                enforce_chat_session_id_for_search_docs=enforce_chat_session_id_for_search_docs,
            )

            # 生成当前是完整文档
            # 未来可能扩展为使用sections
            selected_sections = inference_sections_from_ids(
                doc_identifiers=identifier_tuples,
                document_index=document_index,
            )
            document_pruning_config = DocumentPruningConfig(
                is_manually_selected_docs=True
            )

            # 如果搜索文档被删除,就不包含它
            # 虽然这种情况不应该发生
            db_search_docs_or_none = [
                get_db_search_doc_by_id(doc_id=doc_id, db_session=db_session)
                for doc_id in reference_doc_ids
            ]

            selected_db_search_docs = [
                db_sd for db_sd in db_search_docs_or_none if db_sd
            ]

        else:
            document_pruning_config = DocumentPruningConfig(
                max_chunks=int(
                    persona.num_chunks
                    if persona.num_chunks is not None
                    else default_num_chunks
                ),
                max_window_percentage=max_document_percentage,
            )

        # we don't need to reserve a message id if we're using an existing assistant message
        reserved_message_id = (
            final_msg.id
            if existing_assistant_message_id is not None
            else reserve_message_id(
                db_session=db_session,
                chat_session_id=chat_session_id,
                parent_message=user_message.id
                if user_message is not None
                else parent_message.id,
                message_type=MessageType.ASSISTANT,
            )
        )
        yield MessageResponseIDInfo(
            user_message_id=user_message.id if user_message else None,
            reserved_assistant_message_id=reserved_message_id,
        )

        overridden_model = (
            new_msg_req.llm_override.model_version if new_msg_req.llm_override else None
        )

        # Cannot determine these without the LLM step or breaking out early
        # 如果我们使用现有的助手消息,这只是一个更新操作,在这种情况下父消息应该是最新消息的父消息
        # 如果我们创建新的助手消息,那么父消息应该是最新消息(最新用户消息)
        partial_response = partial(
            create_new_chat_message,
            chat_session_id=chat_session_id,
            parent_message=(
                final_msg if existing_assistant_message_id is None else parent_message
            ),
            prompt_id=prompt_id,
            overridden_model=overridden_model,
            # message=,
            # rephrased_query=,
            # token_count=,
            message_type=MessageType.ASSISTANT,
            alternate_assistant_id=new_msg_req.alternate_assistant_id,
            # error=,
            # reference_docs=,
            db_session=db_session,
            commit=False,
            reserved_message_id=reserved_message_id,
        )

        prompt_override = new_msg_req.prompt_override or chat_session.prompt_override
        if new_msg_req.persona_override_config:
            prompt_config = PromptConfig(
                system_prompt=new_msg_req.persona_override_config.prompts[
                    0
                ].system_prompt,
                task_prompt=new_msg_req.persona_override_config.prompts[0].task_prompt,
                datetime_aware=new_msg_req.persona_override_config.prompts[
                    0
                ].datetime_aware,
                include_citations=new_msg_req.persona_override_config.prompts[
                    0
                ].include_citations,
            )
        elif prompt_override:
            if not final_msg.prompt:
                raise ValueError(
                    "Prompt override cannot be applied, no base prompt found."
                )
            prompt_config = PromptConfig.from_model(
                final_msg.prompt,
                prompt_override=prompt_override,
            )
        elif final_msg.prompt:
            prompt_config = PromptConfig.from_model(final_msg.prompt)
        else:
            prompt_config = PromptConfig.from_model(persona.prompts[0])

        answer_style_config = AnswerStyleConfig(
            citation_config=CitationConfig(
                all_docs_useful=selected_db_search_docs is not None
            ),
            document_pruning_config=document_pruning_config,
            structured_response_format=new_msg_req.structured_response_format,
        )

        tool_dict = construct_tools(
            persona=persona,
            prompt_config=prompt_config,
            db_session=db_session,
            user=user,
            llm=llm,
            fast_llm=fast_llm,
            search_tool_config=SearchToolConfig(
                answer_style_config=answer_style_config,
                document_pruning_config=document_pruning_config,
                retrieval_options=retrieval_options or RetrievalDetails(),
                rerank_settings=new_msg_req.rerank_settings,
                selected_sections=selected_sections,
                chunks_above=new_msg_req.chunks_above,
                chunks_below=new_msg_req.chunks_below,
                full_doc=new_msg_req.full_doc,
                latest_query_files=latest_query_files,
                bypass_acl=bypass_acl,
            ),
            internet_search_tool_config=InternetSearchToolConfig(
                answer_style_config=answer_style_config,
            ),
            image_generation_tool_config=ImageGenerationToolConfig(
                additional_headers=litellm_additional_headers,
            ),
            custom_tool_config=CustomToolConfig(
                chat_session_id=chat_session_id,
                message_id=user_message.id if user_message else None,
                additional_headers=custom_tool_additional_headers,
            ),
        )

        tools: list[Tool] = []
        for tool_list in tool_dict.values():
            tools.extend(tool_list)

        # LLM提示词构建,响应捕获等
        answer = Answer(
            is_connected=is_connected,
            question=final_msg.message,
            latest_query_files=latest_query_files,
            answer_style_config=answer_style_config,
            prompt_config=prompt_config,
            llm=(
                llm
                or get_main_llm_from_tuple(
                    get_llms_for_persona(
                        persona=persona,
                        llm_override=(
                            new_msg_req.llm_override or chat_session.llm_override
                        ),
                        additional_headers=litellm_additional_headers,
                    )
                )
            ),
            message_history=[
                PreviousMessage.from_chat_message(msg, files) for msg in history_msgs
            ],
            tools=tools,
            force_use_tool=_get_force_search_settings(new_msg_req, tools),
        )

        reference_db_search_docs = None
        qa_docs_response = None
        # 与AI消息相关的任何文件,例如dall-e生成的图片
        ai_message_files = []
        dropped_indices = None
        tool_result = None

        for packet in answer.processed_streamed_output:
            if isinstance(packet, ToolResponse):
                if packet.id == SEARCH_RESPONSE_SUMMARY_ID:
                    (
                        qa_docs_response,
                        reference_db_search_docs,
                        dropped_indices,
                    ) = _handle_search_tool_response_summary(
                        packet=packet,
                        db_session=db_session,
                        selected_search_docs=selected_db_search_docs,
                        # Deduping happens at the last step to avoid harming quality by dropping content early on
                        dedupe_docs=(
                            retrieval_options.dedupe_docs
                            if retrieval_options
                            else False
                        ),
                    )
                    yield qa_docs_response
                elif packet.id == SECTION_RELEVANCE_LIST_ID:
                    relevance_sections = packet.response

                    if reference_db_search_docs is not None:
                        llm_indices = relevant_sections_to_indices(
                            relevance_sections=relevance_sections,
                            items=[
                                translate_db_search_doc_to_server_search_doc(doc)
                                for doc in reference_db_search_docs
                            ],
                        )

                        if dropped_indices:
                            llm_indices = drop_llm_indices(
                                llm_indices=llm_indices,
                                search_docs=reference_db_search_docs,
                                dropped_indices=dropped_indices,
                            )

                        yield LLMRelevanceFilterResponse(
                            llm_selected_doc_indices=llm_indices
                        )
                elif packet.id == FINAL_CONTEXT_DOCUMENTS_ID:
                    yield FinalUsedContextDocsResponse(
                        final_context_docs=packet.response
                    )

                elif packet.id == IMAGE_GENERATION_RESPONSE_ID:
                    img_generation_response = cast(
                        list[ImageGenerationResponse], packet.response
                    )

                    file_ids = save_files(
                        urls=[img.url for img in img_generation_response if img.url],
                        base64_files=[
                            img.image_data
                            for img in img_generation_response
                            if img.image_data
                        ],
                        tenant_id=tenant_id,
                    )
                    ai_message_files = [
                        FileDescriptor(id=str(file_id), type=ChatFileType.IMAGE)
                        for file_id in file_ids
                    ]
                    yield FileChatDisplay(
                        file_ids=[str(file_id) for file_id in file_ids]
                    )
                elif packet.id == INTERNET_SEARCH_RESPONSE_ID:
                    (
                        qa_docs_response,
                        reference_db_search_docs,
                    ) = _handle_internet_search_tool_response_summary(
                        packet=packet,
                        db_session=db_session,
                    )
                    yield qa_docs_response
                elif packet.id == CUSTOM_TOOL_RESPONSE_ID:
                    custom_tool_response = cast(CustomToolCallSummary, packet.response)

                    if (
                        custom_tool_response.response_type == "image"
                        or custom_tool_response.response_type == "csv"
                    ):
                        file_ids = custom_tool_response.tool_result.file_ids
                        ai_message_files.extend(
                            [
                                FileDescriptor(
                                    id=str(file_id),
                                    type=(
                                        ChatFileType.IMAGE
                                        if custom_tool_response.response_type == "image"
                                        else ChatFileType.CSV
                                    ),
                                )
                                for file_id in file_ids
                            ]
                        )
                        yield FileChatDisplay(
                            file_ids=[str(file_id) for file_id in file_ids]
                        )
                    else:
                        yield CustomToolResponse(
                            response=custom_tool_response.tool_result,
                            tool_name=custom_tool_response.tool_name,
                        )
                elif packet.id == SEARCH_DOC_CONTENT_ID and include_contexts:
                    yield cast(OnyxContexts, packet.response)

            elif isinstance(packet, StreamStopInfo):
                pass
            else:
                if isinstance(packet, ToolCallFinalResult):
                    tool_result = packet
                yield cast(ChatPacket, packet)
        logger.debug("已到达流的末尾")
    except ValueError as e:
        logger.exception("处理聊天消息失败.")

        error_msg = str(e)
        yield StreamingError(error=error_msg)
        db_session.rollback()
        return

    except Exception as e:
        logger.exception("处理聊天消息失败.")

        error_msg = str(e)
        stack_trace = traceback.format_exc()
        client_error_msg = litellm_exception_to_error_msg(e, llm)
        if llm.config.api_key and len(llm.config.api_key) > 2:
            error_msg = error_msg.replace(llm.config.api_key, "[REDACTED_API_KEY]")
            stack_trace = stack_trace.replace(llm.config.api_key, "[REDACTED_API_KEY]")

        yield StreamingError(error=client_error_msg, stack_trace=stack_trace)
        db_session.rollback()
        return

    # LLM回答后的处理
    try:
        logger.debug("正在进行LLM回答的后处理")
        message_specific_citations: MessageSpecificCitations | None = None
        if reference_db_search_docs:
            message_specific_citations = _translate_citations(
                citations_list=answer.citations,
                db_docs=reference_db_search_docs,
            )
            if not answer.is_cancelled():
                yield AllCitations(citations=answer.citations)

        # 保存生成AI的回答并返回消息信息
        tool_name_to_tool_id: dict[str, int] = {}
        for tool_id, tool_list in tool_dict.items():
            for tool in tool_list:
                tool_name_to_tool_id[tool.name] = tool_id

        gen_ai_response_message = partial_response(
            message=answer.llm_answer,
            rephrased_query=(
                qa_docs_response.rephrased_query if qa_docs_response else None
            ),
            reference_docs=reference_db_search_docs,
            files=ai_message_files,
            token_count=len(llm_tokenizer_encode_func(answer.llm_answer)),
            citations=(
                message_specific_citations.citation_map
                if message_specific_citations
                else None
            ),
            error=None,
            tool_call=(
                ToolCall(
                    tool_id=tool_name_to_tool_id[tool_result.tool_name],
                    tool_name=tool_result.tool_name,
                    tool_arguments=tool_result.tool_args,
                    tool_result=tool_result.tool_result,
                )
                if tool_result
                else None
            ),
        )

        logger.debug("正在提交消息")
        db_session.commit()  # 实际保存用户/助手消息

        msg_detail_response = translate_db_message_to_chat_message_detail(
            gen_ai_response_message
        )

        yield msg_detail_response
    except Exception as e:
        error_msg = str(e)
        logger.exception(error_msg)

        # 前端将擦除任何答案并显示这个错误
        yield StreamingError(error="解析LLM输出失败")


@log_generator_function_time()
def stream_chat_message(
    new_msg_req: CreateChatMessageRequest,
    user: User | None,
    litellm_additional_headers: dict[str, str] | None = None,
    custom_tool_additional_headers: dict[str, str] | None = None,
    is_connected: Callable[[], bool] | None = None,
) -> Iterator[str]:
    """流式输出聊天消息
    
    将聊天消息对象转换为JSON字符串格式流式输出
    
    Args:
        new_msg_req: 创建新消息的请求
        user: 用户对象
        litellm_additional_headers: litellm的附加headers
        custom_tool_additional_headers: 自定义工具的附加headers
        is_connected: 检查连接状态的回调函数
        
    Returns:
        Iterator[str]: JSON字符串迭代器
    """
    with get_session_context_manager() as db_session:
        objects = stream_chat_message_objects(
            new_msg_req=new_msg_req,
            user=user,
            db_session=db_session,
            litellm_additional_headers=litellm_additional_headers,
            custom_tool_additional_headers=custom_tool_additional_headers,
            is_connected=is_connected,
        )
        for obj in objects:
            yield get_json_line(obj.model_dump())


@log_function_time()
def gather_stream_for_slack(
    packets: ChatPacketStream,
) -> ChatOnyxBotResponse:
    """收集流式数据用于Slack响应
    
    将流式输出的聊天消息包整合为Slack机器人可用的响应格式
    
    Args:
        packets: 聊天消息包流
        
    Returns:
        ChatOnyxBotResponse: Slack机器人响应对象
    """
    response = ChatOnyxBotResponse()

    answer = ""
    for packet in packets:
        if isinstance(packet, OnyxAnswerPiece) and packet.answer_piece:
            answer += packet.answer_piece
        elif isinstance(packet, QADocsResponse):
            response.docs = packet
        elif isinstance(packet, StreamingError):
            response.error_msg = packet.error
        elif isinstance(packet, ChatMessageDetail):
            response.chat_message_id = packet.message_id
        elif isinstance(packet, LLMRelevanceFilterResponse):
            response.llm_selected_doc_indices = packet.llm_selected_doc_indices
        elif isinstance(packet, AllCitations):
            response.citations = packet.citations

    if answer:
        response.answer = answer

    return response
