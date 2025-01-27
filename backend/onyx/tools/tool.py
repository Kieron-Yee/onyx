"""
本文件定义了Tool基类，作为所有工具类的抽象基类。
该类定义了工具所需的基本接口和方法，包括工具的名称、描述、执行逻辑等核心功能。
所有具体的工具类都需要继承这个基类并实现其抽象方法。
"""

import abc
from collections.abc import Generator
from typing import Any
from typing import TYPE_CHECKING

from onyx.llm.interfaces import LLM
from onyx.llm.models import PreviousMessage
from onyx.utils.special_types import JSON_ro


if TYPE_CHECKING:
    from onyx.chat.prompt_builder.build import AnswerPromptBuilder
    from onyx.tools.message import ToolCallSummary
    from onyx.tools.models import ToolResponse


class Tool(abc.ABC):
    """
    工具基类，定义了所有工具必须实现的接口。
    这个类作为抽象基类，规定了工具的基本行为和属性。
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """
        获取工具的名称
        
        Returns:
            str: 工具的唯一标识名称
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def description(self) -> str:
        """
        获取工具的描述信息
        
        Returns:
            str: 工具的详细描述
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def display_name(self) -> str:
        """
        获取工具的显示名称
        
        Returns:
            str: 用于显示的工具名称
        """
        raise NotImplementedError

    """For LLMs which support explicit tool calling
    用于支持显式工具调用的LLM"""

    @abc.abstractmethod
    def tool_definition(self) -> dict:
        """
        获取工具的定义信息
        
        Returns:
            dict: 工具的定义信息，包括参数、功能等
        """
        raise NotImplementedError

    @abc.abstractmethod
    def build_tool_message_content(
        self, *args: "ToolResponse"
    ) -> str | list[str | dict[str, Any]]:
        """
        构建工具的消息内容
        
        Args:
            *args: 工具响应对象
            
        Returns:
            str | list[str | dict[str, Any]]: 构建的消息内容
        """
        raise NotImplementedError

    """For LLMs which do NOT support explicit tool calling
    用于不支持显式工具调用的LLM"""

    @abc.abstractmethod
    def get_args_for_non_tool_calling_llm(
        self,
        query: str,
        history: list[PreviousMessage],
        llm: LLM,
        force_run: bool = False,
    ) -> dict[str, Any] | None:
        """
        为不支持工具调用的LLM获取参数
        
        Args:
            query: 查询字符串
            history: 历史消息列表
            llm: LLM对象
            force_run: 是否强制运行
            
        Returns:
            dict[str, Any] | None: 参数字典或None
        """
        raise NotImplementedError

    """Actual execution of the tool
    工具的实际执行"""

    @abc.abstractmethod
    def run(self, **kwargs: Any) -> Generator["ToolResponse", None, None]:
        """
        运行工具的核心方法
        
        Args:
            **kwargs: 运行所需的参数
            
        Returns:
            Generator["ToolResponse", None, None]: 工具响应生成器
        """
        raise NotImplementedError

    @abc.abstractmethod
    def final_result(self, *args: "ToolResponse") -> JSON_ro:
        """
        This is the "final summary" result of the tool.
        It is the result that will be stored in the database.
        这是工具的"最终汇总"结果。
        这个结果将被存储在数据库中。
        
        Args:
            *args: 工具响应对象列表
            
        Returns:
            JSON_ro: JSON格式的最终结果
        """
        raise NotImplementedError

    """Some tools may want to modify the prompt based on the tool call summary and tool responses.
    Default behavior is to continue with just the raw tool call request/result passed to the LLM.
    某些工具可能想要根据工具调用摘要和工具响应修改提示。
    默认行为是仅使用传递给LLM的原始工具调用请求/结果继续。"""

    @abc.abstractmethod
    def build_next_prompt(
        self,
        prompt_builder: "AnswerPromptBuilder",
        tool_call_summary: "ToolCallSummary",
        tool_responses: list["ToolResponse"],
        using_tool_calling_llm: bool,
    ) -> "AnswerPromptBuilder":
        """
        构建下一个提示
        
        Args:
            prompt_builder: 提示构建器
            tool_call_summary: 工具调用摘要
            tool_responses: 工具响应列表
            using_tool_calling_llm: 是否使用工具调用LLM
            
        Returns:
            AnswerPromptBuilder: 构建的提示对象
        """
        raise NotImplementedError
