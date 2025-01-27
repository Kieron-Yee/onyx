"""
此模块提供了与工具（Tool）相关的数据库操作功能。
包含了工具的创建、查询、更新和删除等基本操作函数。
"""

from typing import Any
from typing import cast
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.db.models import Tool
from onyx.server.features.tool.models import Header
from onyx.utils.headers import HeaderItemDict
from onyx.utils.logger import setup_logger

logger = setup_logger()

"""
获取数据库中所有工具的列表
参数:
    db_session: 数据库会话对象
返回:
    工具对象列表
"""
def get_tools(db_session: Session) -> list[Tool]:
    return list(db_session.scalars(select(Tool)).all())

"""
根据工具ID获取特定工具
参数:
    tool_id: 工具ID
    db_session: 数据库会话对象
返回:
    工具对象
异常:
    ValueError: 当指定ID的工具不存在时抛出
"""
def get_tool_by_id(tool_id: int, db_session: Session) -> Tool:
    tool = db_session.scalar(select(Tool).where(Tool.id == tool_id))
    if not tool:
        raise ValueError("Tool by specified id does not exist")
    return tool

"""
根据工具名称获取特定工具
参数:
    tool_name: 工具名称
    db_session: 数据库会话对象
返回:
    工具对象
异常:
    ValueError: 当指定名称的工具不存在时抛出
"""
def get_tool_by_name(tool_name: str, db_session: Session) -> Tool:
    tool = db_session.scalar(select(Tool).where(Tool.name == tool_name))
    if not tool:
        raise ValueError("Tool by specified name does not exist")
    return tool

"""
创建新的工具
参数:
    name: 工具名称
    description: 工具描述
    openapi_schema: OpenAPI架构定义
    custom_headers: 自定义请求头列表
    user_id: 用户ID
    db_session: 数据库会话对象
返回:
    新创建的工具对象
"""
def create_tool(
    name: str,
    description: str | None,
    openapi_schema: dict[str, Any] | None,
    custom_headers: list[Header] | None,
    user_id: UUID | None,
    db_session: Session,
) -> Tool:
    new_tool = Tool(
        name=name,
        description=description,
        in_code_tool_id=None,
        openapi_schema=openapi_schema,
        custom_headers=[header.model_dump() for header in custom_headers]
        if custom_headers
        else [],
        user_id=user_id,
    )
    db_session.add(new_tool)
    db_session.commit()
    return new_tool

"""
更新已存在的工具信息
参数:
    tool_id: 要更新的工具ID
    name: 新的工具名称
    description: 新的工具描述
    openapi_schema: 新的OpenAPI架构定义
    custom_headers: 新的自定义请求头列表
    user_id: 新的用户ID
    db_session: 数据库会话对象
返回:
    更新后的工具对象
异常:
    ValueError: 当指定ID的工具不存在时抛出
"""
def update_tool(
    tool_id: int,
    name: str | None,
    description: str | None,
    openapi_schema: dict[str, Any] | None,
    custom_headers: list[Header] | None,
    user_id: UUID | None,
    db_session: Session,
) -> Tool:
    tool = get_tool_by_id(tool_id, db_session)
    if tool is None:
        raise ValueError(f"Tool with ID {tool_id} does not exist")

    if name is not None:
        tool.name = name
    if description is not None:
        tool.description = description
    if openapi_schema is not None:
        tool.openapi_schema = openapi_schema
    if user_id is not None:
        tool.user_id = user_id
    if custom_headers is not None:
        tool.custom_headers = [
            cast(HeaderItemDict, header.model_dump()) for header in custom_headers
        ]
    db_session.commit()

    return tool

"""
删除指定的工具
参数:
    tool_id: 要删除的工具ID
    db_session: 数据库会话对象
异常:
    ValueError: 当指定ID的工具不存在时抛出
"""
def delete_tool(tool_id: int, db_session: Session) -> None:
    tool = get_tool_by_id(tool_id, db_session)
    if tool is None:
        raise ValueError(f"Tool with ID {tool_id} does not exist")

    db_session.delete(tool)
    db_session.commit()
