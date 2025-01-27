"""
此模块定义了API密钥相关的数据模型。
主要包含API密钥创建和管理所需的数据结构。
"""

from pydantic import BaseModel
from onyx.auth.schemas import UserRole


class APIKeyArgs(BaseModel):
    """
    API密钥参数模型类。
    用于创建和更新API密钥时的参数传递。

    属性：
        name (str | None): API密钥的名称，可选参数，默认为None
        role (UserRole): API密钥的权限角色，默认为基础用户权限(BASIC)
    """
    name: str | None = None
    role: UserRole = UserRole.BASIC
