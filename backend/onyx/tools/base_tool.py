"""
本文件实现了基础工具类和相关辅助函数。
主要功能：
1. 提供工具调用时的消息构建功能
2. 实现基础工具类的通用功能
"""

from typing import cast
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage

from onyx.llm.utils import message_to_prompt_and_imgs
from onyx.tools.tool import Tool

if TYPE_CHECKING:
    from onyx.chat.prompt_builder.build import AnswerPromptBuilder
    from onyx.tools.tool_implementations.custom.custom_tool import (
        CustomToolCallSummary,
    )
    from onyx.tools.message import ToolCallSummary
    from onyx.tools.models import ToolResponse


def build_user_message_for_non_tool_calling_llm(
    message: HumanMessage,
    tool_name: str,
    *args: "ToolResponse",
) -> str:
    """
    为非工具调用LLM构建用户消息。
    
    参数:
        message: 用户的原始消息
        tool_name: 工具名称
        args: 工具响应结果列表
    
    返回:
        构建好的用户消息字符串
    """
    query, _ = message_to_prompt_and_imgs(message)

    tool_run_summary = cast("CustomToolCallSummary", args[0].response).tool_result
    return f"""
Here's the result from the {tool_name} tool:

{tool_run_summary}

Now respond to the following:

{query}
""".strip()


class BaseTool(Tool):
    """
    基础工具类，提供工具的基本功能实现。
    继承自Tool类，实现了构建下一个提示的方法。
    """
    
    def build_next_prompt(
        self,
        prompt_builder: "AnswerPromptBuilder",
        tool_call_summary: "ToolCallSummary",
        tool_responses: list["ToolResponse"],
        using_tool_calling_llm: bool,
    ) -> "AnswerPromptBuilder":
        """
        构建下一个提示。

        参数:
            prompt_builder: 提示构建器实例
            tool_call_summary: 工具调用摘要
            tool_responses: 工具响应列表
            using_tool_calling_llm: 是否使用工具调用LLM的标志

        返回:
            更新后的提示构建器实例
        """
        if using_tool_calling_llm:
            prompt_builder.append_message(tool_call_summary.tool_call_request)
            prompt_builder.append_message(tool_call_summary.tool_call_result)
        else:
            prompt_builder.update_user_prompt(
                HumanMessage(
                    content=build_user_message_for_non_tool_calling_llm(
                        prompt_builder.user_message_and_token_cnt[0],
                        self.name,
                        *tool_responses,
                    )
                )
            )

        return prompt_builder
