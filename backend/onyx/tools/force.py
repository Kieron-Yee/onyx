"""
此文件用于处理工具的强制使用功能。
主要包含了强制使用工具的数据模型和相关的工具过滤函数。
"""

from typing import Any

from pydantic import BaseModel

from onyx.tools.tool import Tool


class ForceUseTool(BaseModel):
    """
    强制使用工具的配置模型类。
    用于定义工具的强制使用参数，包括是否强制使用、工具名称和参数。
    """
    # 可能不是强制使用工具，但仍然有参数，在这种情况下
    # 如果调用了工具，这些参数将替代LLM原本想要使用的参数
    force_use: bool  # 是否强制使用工具
    tool_name: str   # 工具名称
    args: dict[str, Any] | None = None  # 工具参数，可选

    def build_openai_tool_choice_dict(self) -> dict[str, Any]:
        """
        构建OpenAI期望的格式的字典，用于指示使用特定工具。

        Returns:
            dict[str, Any]: 包含工具选择信息的字典，符合OpenAI的格式要求
        """
        return {"type": "function", "function": {"name": self.tool_name}}


def filter_tools_for_force_tool_use(
    tools: list[Tool], 
    force_use_tool: ForceUseTool
) -> list[Tool]:
    """
    根据强制使用工具的配置过滤工具列表。

    Args:
        tools (list[Tool]): 待过滤的工具列表
        force_use_tool (ForceUseTool): 强制使用工具的配置

    Returns:
        list[Tool]: 过滤后的工具列表。如果强制使用，则只返回指定的工具；
                   如果不强制使用，则返回原始工具列表
    """
    if not force_use_tool.force_use:
        return tools

    return [tool for tool in tools if tool.name == force_use_tool.tool_name]
