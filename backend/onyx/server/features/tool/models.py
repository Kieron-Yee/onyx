"""
This module contains Pydantic models for tool-related features
本模块包含工具相关功能的Pydantic模型

主要功能：
1. 定义工具快照模型用于序列化工具数据
2. 定义自定义工具的创建和更新模型
3. 提供工具相关数据的验证和转换功能
"""

from typing import Any

from pydantic import BaseModel

from onyx.db.models import Tool


class ToolSnapshot(BaseModel):
    """
    工具快照模型，用于序列化工具信息
    
    属性说明：
    - id: 工具唯一标识符
    - name: 工具名称
    - description: 工具描述
    - definition: 工具的OpenAPI定义
    - display_name: 工具显示名称
    - in_code_tool_id: 代码中的工具ID
    - custom_headers: 自定义请求头列表
    """
    id: int
    name: str
    description: str
    definition: dict[str, Any] | None
    display_name: str
    in_code_tool_id: str | None
    custom_headers: list[Any] | None

    @classmethod
    def from_model(cls, tool: Tool) -> "ToolSnapshot":
        """
        从数据库模型创建工具快照实例
        
        参数：
        - tool: Tool实例，数据库中的工具模型
        
        返回：
        - ToolSnapshot实例，包含工具的序列化数据
        """
        return cls(
            id=tool.id,
            name=tool.name,
            description=tool.description,
            definition=tool.openapi_schema,
            display_name=tool.display_name or tool.name,
            in_code_tool_id=tool.in_code_tool_id,
            custom_headers=tool.custom_headers,
        )


class Header(BaseModel):
    """
    HTTP请求头模型
    
    属性说明：
    - key: 请求头的键
    - value: 请求头的值
    """
    key: str
    value: str


class CustomToolCreate(BaseModel):
    """
    自定义工具创建模型
    
    属性说明：
    - name: 工具名称
    - description: 工具描述（可选）
    - definition: 工具的OpenAPI定义
    - custom_headers: 自定义请求头列表（可选）
    """
    name: str
    description: str | None = None
    definition: dict[str, Any]
    custom_headers: list[Header] | None = None


class CustomToolUpdate(BaseModel):
    """
    自定义工具更新模型
    
    属性说明：
    - name: 工具名称（可选）
    - description: 工具描述（可选）
    - definition: 工具的OpenAPI定义（可选）
    - custom_headers: 自定义请求头列表（可选）
    """
    name: str | None = None
    description: str | None = None
    definition: dict[str, Any] | None = None
    custom_headers: list[Header] | None = None
