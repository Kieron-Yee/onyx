"""
消息处理工具模块

本模块提供了处理工具调用消息的相关功能，包括构建工具消息和计算工具调用的token数量。
主要用于处理AI工具调用过程中的消息传递和token计算。
"""

import json
from typing import Any

from langchain_core.messages.ai import AIMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.messages.tool import ToolMessage
from pydantic.v1 import BaseModel as BaseModel__v1

from onyx.natural_language_processing.utils import BaseTokenizer

# Langchain has their own version of pydantic which is version 1
# Langchain 使用的是他们自己的 pydantic 版本1


def build_tool_message(
    tool_call: ToolCall, tool_content: str | list[str | dict[str, Any]]
) -> ToolMessage:
    """
    构建工具消息对象

    Args:
        tool_call (ToolCall): 工具调用对象，包含工具调用的ID和名称
        tool_content (str | list[str | dict[str, Any]]): 工具调用的内容

    Returns:
        ToolMessage: 构建好的工具消息对象
    """
    return ToolMessage(
        tool_call_id=tool_call["id"] or "",
        name=tool_call["name"],
        content=tool_content,
    )


class ToolCallSummary(BaseModel__v1):
    """
    工具调用摘要类

    用于存储工具调用的请求和结果信息

    Attributes:
        tool_call_request (AIMessage): AI发起的工具调用请求消息
        tool_call_result (ToolMessage): 工具调用的结果消息
    """
    tool_call_request: AIMessage
    tool_call_result: ToolMessage


def tool_call_tokens(
    tool_call_summary: ToolCallSummary, llm_tokenizer: BaseTokenizer
) -> int:
    """
    计算工具调用过程中使用的token数量

    Args:
        tool_call_summary (ToolCallSummary): 工具调用的摘要信息
        llm_tokenizer (BaseTokenizer): 用于计算token的分词器

    Returns:
        int: 工具调用请求和结果消息使用的总token数
    """
    request_tokens = len(
        llm_tokenizer.encode(
            json.dumps(tool_call_summary.tool_call_request.tool_calls[0]["args"])
        )
    )
    result_tokens = len(
        llm_tokenizer.encode(json.dumps(tool_call_summary.tool_call_result.content))
    )

    return request_tokens + result_tokens
