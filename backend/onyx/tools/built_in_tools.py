"""
This module manages the built-in tools system for the application.
本模块管理应用程序的内置工具系统。

主要功能：
1. 定义和管理内置工具的配置
2. 处理工具的加载、更新和删除
3. 提供工具缓存和检索功能
4. 自动化工具分配给Persona
"""

import os
from typing import Type
from typing_extensions import TypedDict  # noreorder

from sqlalchemy import not_
from sqlalchemy import or_
from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.db.models import Persona
from onyx.db.models import Tool as ToolDBModel
from onyx.tools.tool_implementations.images.image_generation_tool import (
    ImageGenerationTool,
)
from onyx.tools.tool_implementations.internet_search.internet_search_tool import (
    InternetSearchTool,
)
from onyx.tools.tool_implementations.search.search_tool import SearchTool
from onyx.tools.tool import Tool
from onyx.utils.logger import setup_logger

logger = setup_logger()


class InCodeToolInfo(TypedDict):
    """
    Represents the configuration structure for built-in tools
    表示内置工具的配置结构
    
    属性说明：
    cls: 工具类
    description: 工具描述
    in_code_tool_id: 工具在代码中的唯一标识
    display_name: 工具显示名称
    """
    cls: Type[Tool]
    description: str
    in_code_tool_id: str
    display_name: str


BUILT_IN_TOOLS: list[InCodeToolInfo] = [
    InCodeToolInfo(
        cls=SearchTool,
        description="The Search Tool allows the Assistant to search through connected knowledge to help build an answer.",
        in_code_tool_id=SearchTool.__name__,
        display_name=SearchTool._DISPLAY_NAME,
    ),
    InCodeToolInfo(
        cls=ImageGenerationTool,
        description=(
            "The Image Generation Tool allows the assistant to use DALL-E 3 to generate images. "
            "The tool will be used when the user asks the assistant to generate an image."
        ),
        in_code_tool_id=ImageGenerationTool.__name__,
        display_name=ImageGenerationTool._DISPLAY_NAME,
    ),
    # don't show the InternetSearchTool as an option if BING_API_KEY is not available
    *(
        [
            InCodeToolInfo(
                cls=InternetSearchTool,
                description=(
                    "The Internet Search Tool allows the assistant "
                    "to perform internet searches for up-to-date information."
                ),
                in_code_tool_id=InternetSearchTool.__name__,
                display_name=InternetSearchTool._DISPLAY_NAME,
            )
        ]
        if os.environ.get("BING_API_KEY")
        else []
    ),
]


def load_builtin_tools(db_session: Session) -> None:
    """
    Load and synchronize built-in tools with the database
    加载并同步内置工具到数据库
    
    参数:
        db_session: 数据库会话对象
    
    功能：
    - 更新现有工具的信息
    - 添加新的内置工具
    - 删除不再使用的工具
    """
    existing_in_code_tools = db_session.scalars(
        select(ToolDBModel).where(not_(ToolDBModel.in_code_tool_id.is_(None)))
    ).all()
    in_code_tool_id_to_tool = {
        tool.in_code_tool_id: tool for tool in existing_in_code_tools
    }

    # Add or update existing tools
    for tool_info in BUILT_IN_TOOLS:
        tool_name = tool_info["cls"].__name__
        tool = in_code_tool_id_to_tool.get(tool_info["in_code_tool_id"])
        if tool:
            # Update existing tool
            tool.name = tool_name
            tool.description = tool_info["description"]
            tool.display_name = tool_info["display_name"]
            logger.notice(f"Updated tool: {tool_name}")
        else:
            # Add new tool
            new_tool = ToolDBModel(
                name=tool_name,
                description=tool_info["description"],
                display_name=tool_info["display_name"],
                in_code_tool_id=tool_info["in_code_tool_id"],
            )
            db_session.add(new_tool)
            logger.notice(f"Added new tool: {tool_name}")

    # Remove tools that are no longer in BUILT_IN_TOOLS
    built_in_ids = {tool_info["in_code_tool_id"] for tool_info in BUILT_IN_TOOLS}
    for tool_id, tool in list(in_code_tool_id_to_tool.items()):
        if tool_id not in built_in_ids:
            db_session.delete(tool)
            logger.notice(f"Removed tool no longer in built-in list: {tool.name}")

    db_session.commit()
    logger.notice("All built-in tools are loaded/verified.")


def auto_add_search_tool_to_personas(db_session: Session) -> None:
    """
    Automatically adds the SearchTool to all Persona objects in the database that have
    `num_chunks` either unset or set to a value that isn't 0.
    自动将搜索工具添加到数据库中所有num_chunks未设置或不为0的Persona对象中。
    
    参数:
        db_session: 数据库会话对象
    
    功能：
    - 查找数据库中的SearchTool
    - 将SearchTool添加到符合条件的Persona对象中
    - 同步更新到数据库
    """
    # Fetch the SearchTool from the database based on in_code_tool_id from BUILT_IN_TOOLS
    search_tool_id = next(
        (
            tool["in_code_tool_id"]
            for tool in BUILT_IN_TOOLS
            if tool["cls"].__name__ == SearchTool.__name__
        ),
        None,
    )
    if not search_tool_id:
        raise RuntimeError("SearchTool not found in the BUILT_IN_TOOLS list.")

    search_tool = db_session.execute(
        select(ToolDBModel).where(ToolDBModel.in_code_tool_id == search_tool_id)
    ).scalar_one_or_none()

    if not search_tool:
        raise RuntimeError("SearchTool not found in the database.")

    # Fetch all Personas that need the SearchTool added
    personas_to_update = (
        db_session.execute(
            select(Persona).where(
                or_(Persona.num_chunks.is_(None), Persona.num_chunks != 0)
            )
        )
        .scalars()
        .all()
    )

    # Add the SearchTool to each relevant Persona
    for persona in personas_to_update:
        if search_tool not in persona.tools:
            persona.tools.append(search_tool)
            logger.notice(f"Added SearchTool to Persona ID: {persona.id}")

    # Commit changes to the database
    db_session.commit()
    logger.notice("Completed adding SearchTool to relevant Personas.")


_built_in_tools_cache: dict[int, Type[Tool]] | None = None


def refresh_built_in_tools_cache(db_session: Session) -> None:
    """
    刷新内置工具的缓存
    
    参数:
        db_session: 数据库会话对象
    
    功能：
    - 清空现有缓存
    - 从数据库重新加载所有内置工具
    - 更新工具缓存字典
    """
    global _built_in_tools_cache
    _built_in_tools_cache = {}
    all_tool_built_in_tools = (
        db_session.execute(
            select(ToolDBModel).where(not_(ToolDBModel.in_code_tool_id.is_(None)))
        )
        .scalars()
        .all()
    )
    for tool in all_tool_built_in_tools:
        tool_info = next(
            (
                item
                for item in BUILT_IN_TOOLS
                if item["in_code_tool_id"] == tool.in_code_tool_id
            ),
            None,
        )
        if tool_info:
            _built_in_tools_cache[tool.id] = tool_info["cls"]


def get_built_in_tool_by_id(
    tool_id: int, db_session: Session, force_refresh: bool = False
) -> Type[Tool]:
    """
    根据ID获取内置工具类
    
    参数:
        tool_id: 工具ID
        db_session: 数据库会话对象
        force_refresh: 是否强制刷新缓存
    
    返回:
        Tool类型的工具类
    
    异常:
        ValueError: 当工具ID在缓存中不存在时
        RuntimeError: 当工具缓存刷新失败时
    """
    global _built_in_tools_cache
    if _built_in_tools_cache is None or force_refresh:
        refresh_built_in_tools_cache(db_session)

    if _built_in_tools_cache is None:
        raise RuntimeError(
            "Built-in tools cache is None despite being refreshed. Should never happen."
        )

    if tool_id in _built_in_tools_cache:
        return _built_in_tools_cache[tool_id]
    else:
        raise ValueError(f"No built-in tool found in the cache with ID {tool_id}")
