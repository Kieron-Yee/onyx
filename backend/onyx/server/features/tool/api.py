"""
该文件实现了工具相关的API路由处理。
主要功能包括：
1. 自定义工具的创建、更新、删除和验证
2. 工具的查询和列表获取
3. 支持管理员和普通用户的不同权限访问
"""

from typing import Any

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from onyx.auth.users import current_admin_user
from onyx.auth.users import current_user
from onyx.db.engine import get_session
from onyx.db.models import User
from onyx.db.tools import create_tool
from onyx.db.tools import delete_tool
from onyx.db.tools import get_tool_by_id
from onyx.db.tools import get_tools
from onyx.db.tools import update_tool
from onyx.server.features.tool.models import CustomToolCreate
from onyx.server.features.tool.models import CustomToolUpdate
from onyx.server.features.tool.models import ToolSnapshot
from onyx.tools.tool_implementations.custom.openapi_parsing import MethodSpec
from onyx.tools.tool_implementations.custom.openapi_parsing import (
    openapi_to_method_specs,
)
from onyx.tools.tool_implementations.custom.openapi_parsing import (
    validate_openapi_schema,
)
from onyx.tools.tool_implementations.images.image_generation_tool import (
    ImageGenerationTool,
)
from onyx.tools.utils import is_image_generation_available

router = APIRouter(prefix="/tool")
admin_router = APIRouter(prefix="/admin/tool")


def _validate_tool_definition(definition: dict[str, Any]) -> None:
    """验证工具定义的合法性
    
    Args:
        definition: 工具定义的字典数据
        
    Raises:
        HTTPException: 当验证失败时抛出400错误
    """
    try:
        validate_openapi_schema(definition)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@admin_router.post("/custom")
def create_custom_tool(
    tool_data: CustomToolCreate,
    db_session: Session = Depends(get_session),
    user: User | None = Depends(current_admin_user),
) -> ToolSnapshot:
    """创建自定义工具
    
    Args:
        tool_data: 工具创建所需的数据
        db_session: 数据库会话
        user: 当前管理员用户
        
    Returns:
        ToolSnapshot: 创建的工具快照
    """
    _validate_tool_definition(tool_data.definition)
    tool = create_tool(
        name=tool_data.name,
        description=tool_data.description,
        openapi_schema=tool_data.definition,
        custom_headers=tool_data.custom_headers,
        user_id=user.id if user else None,
        db_session=db_session,
    )
    return ToolSnapshot.from_model(tool)


@admin_router.put("/custom/{tool_id}")
def update_custom_tool(
    tool_id: int,
    tool_data: CustomToolUpdate,
    db_session: Session = Depends(get_session),
    user: User | None = Depends(current_admin_user),
) -> ToolSnapshot:
    """更新自定义工具
    
    Args:
        tool_id: 要更新的工具ID
        tool_data: 更新的工具数据
        db_session: 数据库会话
        user: 当前管理员用户
        
    Returns:
        ToolSnapshot: 更新后的工具快照
    """
    if tool_data.definition:
        _validate_tool_definition(tool_data.definition)
    updated_tool = update_tool(
        tool_id=tool_id,
        name=tool_data.name,
        description=tool_data.description,
        openapi_schema=tool_data.definition,
        custom_headers=tool_data.custom_headers,
        user_id=user.id if user else None,
        db_session=db_session,
    )
    return ToolSnapshot.from_model(updated_tool)


@admin_router.delete("/custom/{tool_id}")
def delete_custom_tool(
    tool_id: int,
    db_session: Session = Depends(get_session),
    _: User | None = Depends(current_admin_user),
) -> None:
    """删除自定义工具
    
    Args:
        tool_id: 要删除的工具ID
        db_session: 数据库会话
        _: 当前管理员用户
        
    Raises:
        HTTPException: 当工具不存在时返回404，当工具仍在使用时返回400
    """
    try:
        delete_tool(tool_id, db_session)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        # handles case where tool is still used by an Assistant
        raise HTTPException(status_code=400, detail=str(e))


class ValidateToolRequest(BaseModel):
    """工具验证请求模型"""
    definition: dict[str, Any]


class ValidateToolResponse(BaseModel):
    """工具验证响应模型"""
    methods: list[MethodSpec]


@admin_router.post("/custom/validate")
def validate_tool(
    tool_data: ValidateToolRequest,
    _: User | None = Depends(current_admin_user),
) -> ValidateToolResponse:
    """验证工具定义
    
    Args:
        tool_data: 包含工具定义的请求数据
        _: 当前管理员用户
        
    Returns:
        ValidateToolResponse: 包含验证后方法规范的响应
    """
    _validate_tool_definition(tool_data.definition)
    method_specs = openapi_to_method_specs(tool_data.definition)
    return ValidateToolResponse(methods=method_specs)


"""Endpoints for all 所有用户可用的端点"""


@router.get("/{tool_id}")
def get_custom_tool(
    tool_id: int,
    db_session: Session = Depends(get_session),
    _: User | None = Depends(current_user),
) -> ToolSnapshot:
    """获取指定ID的工具
    
    Args:
        tool_id: 工具ID
        db_session: 数据库会话
        _: 当前用户
        
    Returns:
        ToolSnapshot: 工具快照
        
    Raises:
        HTTPException: 当工具不存在时返回404
    """
    try:
        tool = get_tool_by_id(tool_id, db_session)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return ToolSnapshot.from_model(tool)


@router.get("")
def list_tools(
    db_session: Session = Depends(get_session),
    _: User | None = Depends(current_user),
) -> list[ToolSnapshot]:
    """获取工具列表
    
    Args:
        db_session: 数据库会话
        _: 当前用户
        
    Returns:
        list[ToolSnapshot]: 工具快照列表，会根据图像生成工具的可用性过滤结果
    """
    tools = get_tools(db_session)
    return [
        ToolSnapshot.from_model(tool)
        for tool in tools
        if tool.in_code_tool_id != ImageGenerationTool.name
        or is_image_generation_available(db_session=db_session)
    ]
