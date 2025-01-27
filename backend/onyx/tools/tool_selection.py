"""
该文件实现了工具选择的相关功能，主要用于在多个可用工具中选择最合适的工具来回答用户查询。
主要包含工具选择的提示模板和工具选择的核心函数。
"""

import re
from typing import Any

from onyx.chat.chat_utils import combine_message_chain
from onyx.configs.model_configs import GEN_AI_HISTORY_CUTOFF
from onyx.llm.interfaces import LLM
from onyx.llm.models import PreviousMessage
from onyx.llm.utils import message_to_string
from onyx.prompts.constants import GENERAL_SEP_PAT
from onyx.tools.tool import Tool
from onyx.utils.logger import setup_logger

logger = setup_logger()

# 工具选择提示模板
# 用于指导AI选择最合适的工具来回答查询
SINGLE_TOOL_SELECTION_PROMPT = f"""
You are an expert at selecting the most useful tool to run for answering the query.
You will be given a numbered list of tools and their arguments, a message history, and a query.
You will select a single tool that will be most useful for answering the query.
Respond with only the number corresponding to the tool you want to use.

Conversation History:
{GENERAL_SEP_PAT}
{{chat_history}}
{GENERAL_SEP_PAT}

Query:
{{query}}

Tools:
{{tool_list}}

Respond with EXACTLY and ONLY the number corresponding to the tool you want to use.

Your selection:
"""


def select_single_tool_for_non_tool_calling_llm(
    tools_and_args: list[tuple[Tool, dict[str, Any]]],
    history: list[PreviousMessage],
    query: str,
    llm: LLM,
) -> tuple[Tool, dict[str, Any]] | None:
    """
    为不支持工具调用的LLM选择单个最合适的工具。
    
    参数:
        tools_and_args: 工具和其参数的列表，每个元素是(工具, 参数字典)的元组
        history: 对话历史记录列表
        query: 用户查询字符串
        llm: 语言模型接口实例
    
    返回:
        tuple[Tool, dict[str, Any]] | None: 返回选中的工具和其参数的元组，如果选择失败则返回None
    """
    # 如果只有一个工具，直接返回该工具
    if len(tools_and_args) == 1:
        return tools_and_args[0]

    # 构建工具列表字符串，包含工具编号、名称、参数和描述
    tool_list_str = "\n".join(
        f"""```{ind}: {tool.name} ({args}) - {tool.description}```"""
        for ind, (tool, args) in enumerate(tools_and_args)
    ).lstrip()

    # 获取截断后的对话历史
    history_str = combine_message_chain(
        messages=history,
        token_limit=GEN_AI_HISTORY_CUTOFF,
    )
    
    # 构建完整的提示内容
    prompt = SINGLE_TOOL_SELECTION_PROMPT.format(
        tool_list=tool_list_str, chat_history=history_str, query=query
    )
    
    # 使用语言模型获取响应
    output = message_to_string(llm.invoke(prompt))
    
    try:
        # 首先尝试匹配数字
        number_match = re.search(r"\d+", output)
        if number_match:
            tool_ind = int(number_match.group())
            return tools_and_args[tool_ind]

        # 如果匹配数字失败，尝试匹配工具名称
        for tool, args in tools_and_args:
            if tool.name.lower() in output.lower():
                return tool, args

        # 如果所有匹配都失败，返回第一个工具
        return tools_and_args[0]

    except Exception:
        logger.error(f"为不支持工具调用的LLM选择工具失败: {output}")
        return None
