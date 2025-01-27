"""
本模块提供了搜索类工具的实用函数。
主要用于构建和处理搜索相关的提示信息，包括引用和引用样式的处理。
"""

from typing import cast

from langchain_core.messages import HumanMessage

from onyx.chat.models import AnswerStyleConfig
from onyx.chat.models import LlmDoc
from onyx.chat.models import PromptConfig
from onyx.chat.prompt_builder.build import AnswerPromptBuilder
from onyx.chat.prompt_builder.citations_prompt import (
    build_citations_system_message,
)
from onyx.chat.prompt_builder.citations_prompt import build_citations_user_message
from onyx.chat.prompt_builder.quotes_prompt import build_quotes_user_message
from onyx.tools.message import ToolCallSummary
from onyx.tools.models import ToolResponse


# 定义常量用于标识文档
ORIGINAL_CONTEXT_DOCUMENTS_ID = "search_doc_content"  # 原始上下文文档ID
FINAL_CONTEXT_DOCUMENTS_ID = "final_context_documents"  # 最终上下文文档ID


def build_next_prompt_for_search_like_tool(
    prompt_builder: AnswerPromptBuilder,
    tool_call_summary: ToolCallSummary,
    tool_responses: list[ToolResponse],
    using_tool_calling_llm: bool,
    answer_style_config: AnswerStyleConfig,
    prompt_config: PromptConfig,
) -> AnswerPromptBuilder:
    """
    为搜索类工具构建下一个提示。

    参数:
        prompt_builder: 答案提示构建器对象
        tool_call_summary: 工具调用摘要
        tool_responses: 工具响应列表
        using_tool_calling_llm: 是否使用工具调用LLM
        answer_style_config: 答案样式配置
        prompt_config: 提示配置

    返回:
        更新后的答案提示构建器对象
    """
    
    if not using_tool_calling_llm:
        # 如果不使用工具调用LLM，从工具响应中获取最终上下文文档
        final_context_docs_response = next(
            response
            for response in tool_responses
            if response.id == FINAL_CONTEXT_DOCUMENTS_ID
        )
        final_context_documents = cast(
            list[LlmDoc], final_context_docs_response.response
        )
    else:
        # 如果使用工具调用LLM，最终上下文文档为空列表
        final_context_documents = []

    if answer_style_config.citation_config:
        # 处理引用配置
        prompt_builder.update_system_prompt(
            build_citations_system_message(prompt_config)
        )
        prompt_builder.update_user_prompt(
            build_citations_user_message(
                message=prompt_builder.user_message_and_token_cnt[0],
                prompt_config=prompt_config,
                context_docs=final_context_documents,
                all_doc_useful=(
                    answer_style_config.citation_config.all_docs_useful
                    if answer_style_config.citation_config
                    else False
                ),
                history_message=prompt_builder.single_message_history or "",
            )
        )
    elif answer_style_config.quotes_config:
        # 处理引用配置 - 对于引用，系统提示包含在用户提示中
        prompt_builder.update_system_prompt(None)

        human_message = HumanMessage(content=prompt_builder.raw_user_message)

        prompt_builder.update_user_prompt(
            build_quotes_user_message(
                message=human_message,
                context_docs=final_context_documents,
                history_str=prompt_builder.single_message_history or "",
                prompt=prompt_config,
            )
        )

    if using_tool_calling_llm:
        # 如果使用工具调用LLM，添加工具调用请求和结果
        prompt_builder.append_message(tool_call_summary.tool_call_request)
        prompt_builder.append_message(tool_call_summary.tool_call_result)

    return prompt_builder
