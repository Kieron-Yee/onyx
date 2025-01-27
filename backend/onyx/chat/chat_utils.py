"""
此文件包含与聊天功能相关的实用工具函数。
主要功能包括：
- 聊天消息的准备和处理
- 聊天会话的创建和管理
- 消息线程的组合和处理
- 引用信息的重组织
- 请求头的处理
- 临时persona的创建
"""

import re
from typing import cast
from uuid import UUID

from fastapi import HTTPException
from fastapi.datastructures import Headers
from sqlalchemy.orm import Session

from onyx.auth.users import is_user_admin
from onyx.chat.models import CitationInfo
from onyx.chat.models import LlmDoc
from onyx.chat.models import PersonaOverrideConfig
from onyx.chat.models import ThreadMessage
from onyx.configs.constants import DEFAULT_PERSONA_ID
from onyx.configs.constants import MessageType
from onyx.context.search.models import InferenceSection
from onyx.context.search.models import RerankingDetails
from onyx.context.search.models import RetrievalDetails
from onyx.db.chat import create_chat_session
from onyx.db.chat import get_chat_messages_by_session
from onyx.db.llm import fetch_existing_doc_sets
from onyx.db.llm import fetch_existing_tools
from onyx.db.models import ChatMessage
from onyx.db.models import Persona
from onyx.db.models import Prompt
from onyx.db.models import Tool
from onyx.db.models import User
from onyx.db.persona import get_prompts_by_ids
from onyx.llm.models import PreviousMessage
from onyx.natural_language_processing.utils import BaseTokenizer
from onyx.server.query_and_chat.models import CreateChatMessageRequest
from onyx.tools.tool_implementations.custom.custom_tool import (
    build_custom_tools_from_openapi_schema_and_headers,
)
from onyx.utils.logger import setup_logger

logger = setup_logger()


def prepare_chat_message_request(
    message_text: str,
    user: User | None,
    persona_id: int | None,
    persona_override_config: PersonaOverrideConfig | None,
    prompt: Prompt | None,
    message_ts_to_respond_to: str | None,
    retrieval_details: RetrievalDetails | None,
    rerank_settings: RerankingDetails | None,
    db_session: Session,
) -> CreateChatMessageRequest:
    """
    准备聊天消息请求。
    
    参数:
        message_text: 消息文本内容
        user: 用户对象或None
        persona_id: persona ID或None
        persona_override_config: persona覆盖配置
        prompt: 提示对象或None
        message_ts_to_respond_to: 需要响应的消息时间戳
        retrieval_details: 检索详情
        rerank_settings: 重排序设置
        db_session: 数据库会话
    
    返回:
        CreateChatMessageRequest: 创建的聊天消息请求对象
    """
    # Typically used for one shot flows like SlackBot or non-chat API endpoint use cases
    # 通常用于SlackBot等一次性流程或非聊天API端点用例
    new_chat_session = create_chat_session(
        db_session=db_session,
        description=None,
        user_id=user.id if user else None,
        # If using an override, this id will be ignored later on
        # 如果使用覆盖配置，此ID稍后将被忽略
        persona_id=persona_id or DEFAULT_PERSONA_ID,
        onyxbot_flow=True,
        slack_thread_id=message_ts_to_respond_to,
    )

    return CreateChatMessageRequest(
        chat_session_id=new_chat_session.id,
        parent_message_id=None,  # It's a standalone chat session each time
        # 每次都是独立的聊天会话
        message=message_text,
        file_descriptors=[],  # Currently SlackBot/answer api do not support files in the context
        # 目前SlackBot/answer API不支持上下文中的文件
        prompt_id=prompt.id if prompt else None,
        # Can always override the persona for the single query, if it's a normal persona
        # then it will be treated the same
        # 始终可以为单个查询覆盖persona，如果是普通persona，则会被同样处理
        persona_override_config=persona_override_config,
        search_doc_ids=None,
        retrieval_options=retrieval_details,
        rerank_settings=rerank_settings,
    )


def llm_doc_from_inference_section(inference_section: InferenceSection) -> LlmDoc:
    """
    将推理段落转换为LLM文档。
    
    参数:
        inference_section: 推理段落对象
    
    返回:
        LlmDoc: 转换后的LLM文档对象
    """
    return LlmDoc(
        document_id=inference_section.center_chunk.document_id,
        # This one is using the combined content of all the chunks of the section
        # In default settings, this is the same as just the content of base chunk
        # 这个使用了该段落所有块的组合内容，在默认设置下，这与仅使用基础块的内容相同
        content=inference_section.combined_content,
        blurb=inference_section.center_chunk.blurb,
        semantic_identifier=inference_section.center_chunk.semantic_identifier,
        source_type=inference_section.center_chunk.source_type,
        metadata=inference_section.center_chunk.metadata,
        updated_at=inference_section.center_chunk.updated_at,
        link=inference_section.center_chunk.source_links[0]
        if inference_section.center_chunk.source_links
        else None,
        source_links=inference_section.center_chunk.source_links,
        match_highlights=inference_section.center_chunk.match_highlights,
    )


def combine_message_thread(
    messages: list[ThreadMessage],
    max_tokens: int | None,
    llm_tokenizer: BaseTokenizer,
) -> str:
    """
    将消息线程合并成单个上下文。
    
    Used to create a single combined message context from threads
    用于从线程创建单个合并的消息上下文
    
    参数:
        messages: 线程消息列表
        max_tokens: 最大令牌数或None
        llm_tokenizer: 分词器对象
    
    返回:
        str: 合并后的消息文本
    """
    if not messages:
        return ""

    message_strs: list[str] = []
    total_token_count = 0

    for message in reversed(messages):
        if message.role == MessageType.USER:
            role_str = message.role.value.upper()
            if message.sender:
                role_str += " " + message.sender
            else:
                # Since other messages might have the user identifying information
                # better to use Unknown for symmetry
                # 由于其他消息可能具有用户识别信息，为了对称，最好使用Unknown
                role_str += " Unknown"
        else:
            role_str = message.role.value.upper()

        msg_str = f"{role_str}:\n{message.message}"
        message_token_count = len(llm_tokenizer.encode(msg_str))

        if (
            max_tokens is not None
            and total_token_count + message_token_count > max_tokens
        ):
            break

        message_strs.insert(0, msg_str)
        total_token_count += message_token_count

    return "\n\n".join(message_strs)


def create_chat_chain(
    chat_session_id: UUID,
    db_session: Session,
    prefetch_tool_calls: bool = True,
    stop_at_message_id: int | None = None,
) -> tuple[ChatMessage, list[ChatMessage]]:
    """
    创建聊天链。
    
    Build the linear chain of messages without including the root message
    构建不包括根消息的线性消息链
    
    参数:
        chat_session_id: 聊天会话ID
        db_session: 数据库会话
        prefetch_tool_calls: 是否预取工具调用
        stop_at_message_id: 可选的停止处理的消息ID
    
    返回:
        tuple: 包含根消息和消息链的元组
    """
    mainline_messages: list[ChatMessage] = []

    all_chat_messages = get_chat_messages_by_session(
        chat_session_id=chat_session_id,
        user_id=None,
        db_session=db_session,
        skip_permission_check=True,
        prefetch_tool_calls=prefetch_tool_calls,
    )
    id_to_msg = {msg.id: msg for msg in all_chat_messages}

    if not all_chat_messages:
        raise RuntimeError("No messages in Chat Session")

    root_message = all_chat_messages[0]
    if root_message.parent_message is not None:
        raise RuntimeError(
            "Invalid root message, unable to fetch valid chat message sequence"
        )

    current_message: ChatMessage | None = root_message
    while current_message is not None:
        child_msg = current_message.latest_child_message

        if not child_msg or (
            stop_at_message_id and current_message.id == stop_at_message_id
        ):
            break
        current_message = id_to_msg.get(child_msg)

        if current_message is None:
            raise RuntimeError(
                "Invalid message chain,"
                "could not find next message in the same session"
            )

        mainline_messages.append(current_message)

    if not mainline_messages:
        raise RuntimeError("Could not trace chat message history")

    return mainline_messages[-1], mainline_messages[:-1]


def combine_message_chain(
    messages: list[ChatMessage] | list[PreviousMessage],
    token_limit: int,
    msg_limit: int | None = None,
) -> str:
    """
    将消息链合并成单个上下文。
    
    Used for secondary LLM flows that require the chat history
    用于需要聊天历史的二级LLM流程
    
    参数:
        messages: 消息列表
        token_limit: 令牌限制
        msg_limit: 消息限制或None
    
    返回:
        str: 合并后的消息文本
    """
    message_strs: list[str] = []
    total_token_count = 0

    if msg_limit is not None:
        messages = messages[-msg_limit:]

    for message in cast(list[ChatMessage] | list[PreviousMessage], reversed(messages)):
        message_token_count = message.token_count

        if total_token_count + message_token_count > token_limit:
            break

        role = message.message_type.value.upper()
        message_strs.insert(0, f"{role}:\n{message.message}")
        total_token_count += message_token_count

    return "\n\n".join(message_strs)


def reorganize_citations(
    answer: str, citations: list[CitationInfo]
) -> tuple[str, list[CitationInfo]]:
    """
    重新组织引用信息。
    
    For a complete, citation-aware response, we want to reorganize the citations so that
    they are in the order of the documents that were used in the response. This just looks nicer / avoids
    confusion ("Why is there [7] when only 2 documents are cited?").
    为了完整的、引用感知的响应，我们希望重新组织引用信息，使其按照响应中使用的文档顺序排列。
    这样看起来更好/避免混淆（“为什么只有2个文档被引用却有[7]？”）。
    
    参数:
        answer: 回答文本
        citations: 引用信息列表
    
    返回:
        tuple: 包含重新组织后的回答文本和引用信息的元组
    """
    # Regular expression to find all instances of [[x]](LINK)
    # 正则表达式查找所有[[x]](LINK)实例
    pattern = r"\[\[(.*?)\]\]\((.*?)\)"

    all_citation_matches = re.findall(pattern, answer)

    new_citation_info: dict[int, CitationInfo] = {}
    for citation_match in all_citation_matches:
        try:
            citation_num = int(citation_match[0])
            if citation_num in new_citation_info:
                continue

            matching_citation = next(
                iter([c for c in citations if c.citation_num == int(citation_num)]),
                None,
            )
            if matching_citation is None:
                continue

            new_citation_info[citation_num] = CitationInfo(
                citation_num=len(new_citation_info) + 1,
                document_id=matching_citation.document_id,
            )
        except Exception:
            pass

    # Function to replace citations with their new number
    # 替换引用为新编号的函数
    def slack_link_format(match: re.Match) -> str:
        link_text = match.group(1)
        try:
            citation_num = int(link_text)
            if citation_num in new_citation_info:
                link_text = new_citation_info[citation_num].citation_num
        except Exception:
            pass

        link_url = match.group(2)
        return f"[[{link_text}]]({link_url})"

    # Substitute all matches in the input text
    # 替换输入文本中的所有匹配项
    new_answer = re.sub(pattern, slack_link_format, answer)

    # if any citations weren't parsable, just add them back to be safe
    # 如果有任何引用无法解析，只需将它们添加回来以确保安全
    for citation in citations:
        if citation.citation_num not in new_citation_info:
            new_citation_info[citation.citation_num] = citation

    return new_answer, list(new_citation_info.values())


def extract_headers(
    headers: dict[str, str] | Headers, pass_through_headers: list[str] | None
) -> dict[str, str]:
    """
    提取指定的请求头。
    
    Extract headers specified in pass_through_headers from input headers.
    Handles both dict and FastAPI Headers objects, accounting for lowercase keys.
    从输入头中提取pass_through_headers中指定的头。
    处理dict和FastAPI Headers对象，考虑小写键。
    
    参数:
        headers: 输入头，dict或Headers对象
        pass_through_headers: 需要提取的头列表或None
    
    返回:
        dict: 根据pass_through_headers过滤后的头
    """
    if not pass_through_headers:
        return {}

    extracted_headers: dict[str, str] = {}
    for key in pass_through_headers:
        if key in headers:
            extracted_headers[key] = headers[key]
        else:
            # fastapi makes all header keys lowercase, handling that here
            # fastapi将所有头键转换为小写，在此处理
            lowercase_key = key.lower()
            if lowercase_key in headers:
                extracted_headers[lowercase_key] = headers[lowercase_key]
    return extracted_headers


def create_temporary_persona(
    persona_config: PersonaOverrideConfig, db_session: Session, user: User | None = None
) -> Persona:
    """
    创建临时persona。
    
    Create a temporary Persona object from the provided configuration.
    从提供的配置创建临时Persona对象。
    
    参数:
        persona_config: persona覆盖配置
        db_session: 数据库会话
        user: 用户对象或None
    
    返回:
        Persona: 创建的临时Persona对象
    """
    if not is_user_admin(user):
        raise HTTPException(
            status_code=403,
            detail="User is not authorized to create a persona in one shot queries",
        )

    persona = Persona(
        name=persona_config.name,
        description=persona_config.description,
        num_chunks=persona_config.num_chunks,
        llm_relevance_filter=persona_config.llm_relevance_filter,
        llm_filter_extraction=persona_config.llm_filter_extraction,
        recency_bias=persona_config.recency_bias,
        llm_model_provider_override=persona_config.llm_model_provider_override,
        llm_model_version_override=persona_config.llm_model_version_override,
    )

    if persona_config.prompts:
        persona.prompts = [
            Prompt(
                name=p.name,
                description=p.description,
                system_prompt=p.system_prompt,
                task_prompt=p.task_prompt,
                include_citations=p.include_citations,
                datetime_aware=p.datetime_aware,
            )
            for p in persona_config.prompts
        ]
    elif persona_config.prompt_ids:
        persona.prompts = get_prompts_by_ids(
            db_session=db_session, prompt_ids=persona_config.prompt_ids
        )

    persona.tools = []
    if persona_config.custom_tools_openapi:
        for schema in persona_config.custom_tools_openapi:
            tools = cast(
                list[Tool],
                build_custom_tools_from_openapi_schema_and_headers(schema),
            )
            persona.tools.extend(tools)

    if persona_config.tools:
        tool_ids = [tool.id for tool in persona_config.tools]
        persona.tools.extend(
            fetch_existing_tools(db_session=db_session, tool_ids=tool_ids)
        )

    if persona_config.tool_ids:
        persona.tools.extend(
            fetch_existing_tools(
                db_session=db_session, tool_ids=persona_config.tool_ids
            )
        )

    fetched_docs = fetch_existing_doc_sets(
        db_session=db_session, doc_ids=persona_config.document_set_ids
    )
    persona.document_sets = fetched_docs

    return persona
