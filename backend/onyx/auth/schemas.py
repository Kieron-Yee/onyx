"""
此文件定义了用户认证相关的数据模型和schema，包括用户角色枚举、用户读取模型、用户创建模型和用户更新模型。
主要用于处理用户数据的序列化和反序列化，以及用户角色的定义和管理。
"""

import uuid
from enum import Enum

from fastapi_users import schemas


class UserRole(str, Enum):
    """
    用户角色定义
    
    User roles
    - Basic can't perform any admin actions
    - Admin can perform all admin actions
    - Curator can perform admin actions for
        groups they are curators of
    - Global Curator can perform admin actions
        for all groups they are a member of
    - Limited can access a limited set of basic api endpoints
    - Slack are users that have used onyx via slack but dont have a web login
    - External permissioned users that have been picked up during the external permissions sync process but don't have a web login

    用户角色：
    - Basic：无法执行任何管理操作
    - Admin：可以执行所有管理操作
    - Curator：可以执行其作为策展人的组的管理操作
    - Global Curator：可以执行其所属的所有组的管理操作
    - Limited：只能访问有限的基本API端点
    - Slack：通过slack使用onyx但没有web登录的用户
    - External：在外部权限同步过程中获取的但没有web登录的用户
    """

    LIMITED = "limited"
    BASIC = "basic"
    ADMIN = "admin"
    CURATOR = "curator"
    GLOBAL_CURATOR = "global_curator"
    SLACK_USER = "slack_user"
    EXT_PERM_USER = "ext_perm_user"

    def is_web_login(self) -> bool:
        """
        判断用户角色是否支持网页登录
        
        返回值：
            bool: True表示支持网页登录，False表示不支持
        """
        return self not in [
            UserRole.SLACK_USER,
            UserRole.EXT_PERM_USER,
        ]


class UserRead(schemas.BaseUser[uuid.UUID]):
    """
    用户信息读取模型
    继承自BaseUser，用于序列化用户信息供读取使用
    
    属性：
        role: UserRole - 用户角色
    """
    role: UserRole


class UserCreate(schemas.BaseUserCreate):
    """
    用户创建模型
    继承自BaseUserCreate，用于创建新用户时的数据验证和序列化
    
    属性：
        role: UserRole - 用户角色，默认为BASIC
        tenant_id: str | None - 租户ID，可选
    """
    role: UserRole = UserRole.BASIC
    tenant_id: str | None = None


class UserUpdate(schemas.BaseUserUpdate):
    """
    用户更新模型
    继承自BaseUserUpdate，用于更新用户信息时的数据验证和序列化
    
    Role updates are not allowed through the user update endpoint for security reasons
    Role changes should be handled through a separate, admin-only process

    注意：
    出于安全考虑，不允许通过用户更新接口更新角色
    角色变更应该通过单独的管理员专用流程处理
    """
