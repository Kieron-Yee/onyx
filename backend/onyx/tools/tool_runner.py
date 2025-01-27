"""
工具运行器模块
该模块提供了用于执行工具和管理工具执行过程的类和函数。
主要包含ToolRunner类和工具运行相关的辅助函数。
"""

from collections.abc import Callable
from collections.abc import Generator
from typing import Any

from onyx.llm.interfaces import LLM
from onyx.llm.models import PreviousMessage
from onyx.tools.models import ToolCallFinalResult
from onyx.tools.models import ToolCallKickoff
from onyx.tools.models import ToolResponse
from onyx.tools.tool import Tool
from onyx.utils.threadpool_concurrency import run_functions_tuples_in_parallel


class ToolRunner:
    """
    工具运行器类
    负责执行特定工具及管理其执行状态、结果等
    """

    def __init__(self, tool: Tool, args: dict[str, Any]):
        """
        初始化工具运行器
        
        参数:
            tool: Tool - 要执行的工具实例
            args: dict[str, Any] - 工具执行所需的参数字典
        """
        self.tool = tool
        self.args = args
        self._tool_responses: list[ToolResponse] | None = None

    def kickoff(self) -> ToolCallKickoff:
        """
        启动工具执行
        
        返回:
            ToolCallKickoff - 包含工具名称和参数的工具调用启动对象
        """
        return ToolCallKickoff(tool_name=self.tool.name, tool_args=self.args)

    def tool_responses(self) -> Generator[ToolResponse, None, None]:
        """
        获取工具执行的响应结果
        
        返回:
            Generator[ToolResponse, None, None] - 工具响应结果的生成器
        """
        if self._tool_responses is not None:
            yield from self._tool_responses
            return

        tool_responses: list[ToolResponse] = []
        for tool_response in self.tool.run(**self.args):
            yield tool_response
            tool_responses.append(tool_response)

        self._tool_responses = tool_responses

    def tool_message_content(self) -> str | list[str | dict[str, Any]]:
        """
        构建工具执行的消息内容
        
        返回:
            str | list[str | dict[str, Any]] - 工具执行结果的消息内容
        """
        tool_responses = list(self.tool_responses())
        return self.tool.build_tool_message_content(*tool_responses)

    def tool_final_result(self) -> ToolCallFinalResult:
        """
        获取工具执行的最终结果
        
        返回:
            ToolCallFinalResult - 包含工具名称、参数和执行结果的最终结果对象
        """
        return ToolCallFinalResult(
            tool_name=self.tool.name,
            tool_args=self.args,
            tool_result=self.tool.final_result(*self.tool_responses()),
        )


def check_which_tools_should_run_for_non_tool_calling_llm(
    tools: list[Tool], query: str, history: list[PreviousMessage], llm: LLM
) -> list[dict[str, Any] | None]:
    """
    检查在非工具调用LLM模式下应该运行哪些工具
    
    参数:
        tools: list[Tool] - 可用工具列表
        query: str - 用户查询
        history: list[PreviousMessage] - 历史消息记录
        llm: LLM - 语言模型实例
    
    返回:
        list[dict[str, Any] | None] - 需要运行的工具及其参数列表
    """
    tool_args_list: list[tuple[Callable[..., Any], tuple[Any, ...]]] = [
        (tool.get_args_for_non_tool_calling_llm, (query, history, llm))
        for tool in tools
    ]
    return run_functions_tuples_in_parallel(tool_args_list)
