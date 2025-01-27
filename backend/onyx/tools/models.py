"""
工具模型定义模块
本模块定义了与工具调用相关的各种数据模型，包括工具响应、工具调用启动、工具运行响应等基础模型类。
这些模型用于规范化工具调用过程中的数据结构和验证逻辑。
"""

from typing import Any
from uuid import UUID

from pydantic import BaseModel
from pydantic import model_validator


class ToolResponse(BaseModel):
    """
    工具响应模型
    用于封装工具执行后的响应结果
    
    属性:
        id: 响应标识符
        response: 工具执行的响应内容
    """
    id: str | None = None
    response: Any = None


class ToolCallKickoff(BaseModel):
    """
    工具调用启动模型
    用于定义工具调用的初始参数
    
    属性:
        tool_name: 要调用的工具名称
        tool_args: 工具调用所需的参数字典
    """
    tool_name: str
    tool_args: dict[str, Any]


class ToolRunnerResponse(BaseModel):
    """
    工具运行响应模型
    用于封装工具运行过程中的各种响应状态
    
    属性:
        tool_run_kickoff: 工具调用启动信息
        tool_response: 工具执行响应
        tool_message_content: 工具消息内容
    """
    tool_run_kickoff: ToolCallKickoff | None = None
    tool_response: ToolResponse | None = None
    tool_message_content: str | list[str | dict[str, Any]] | None = None

    @model_validator(mode="after")
    def validate_tool_runner_response(self) -> "ToolRunnerResponse":
        """
        验证工具运行响应的有效性
        
        确保tool_response、tool_message_content和tool_run_kickoff三个字段中有且仅有一个被提供
        
        返回值:
            ToolRunnerResponse: 验证后的响应对象
            
        异常:
            ValueError: 当不符合只有一个字段被提供的条件时抛出
        """
        fields = ["tool_response", "tool_message_content", "tool_run_kickoff"]
        provided = sum(1 for field in fields if getattr(self, field) is not None)

        if provided != 1:
            raise ValueError(
                "必须且只能提供'tool_response'、'tool_message_content'或'tool_run_kickoff'中的一个"
            )

        return self


class ToolCallFinalResult(ToolCallKickoff):
    """
    工具调用最终结果模型
    继承自ToolCallKickoff，用于存储工具调用的最终执行结果
    
    属性:
        tool_result: 工具执行的最终结果
    """
    tool_result: Any = None  # 由于JSON_ro的递归特性，这里不能使用它 
    # we would like to use JSON_ro, but can't due to its recursive nature 
    # 我们想使用JSON_ro，但由于其递归特性而无法使用


class DynamicSchemaInfo(BaseModel):
    """
    动态模式信息模型
    用于存储聊天会话相关的动态信息
    
    属性:
        chat_session_id: 聊天会话ID
        message_id: 消息ID
    """
    chat_session_id: UUID | None
    message_id: int | None


# 占位符常量定义
CHAT_SESSION_ID_PLACEHOLDER = "CHAT_SESSION_ID"  # 聊天会话ID占位符
MESSAGE_ID_PLACEHOLDER = "MESSAGE_ID"  # 消息ID占位符
