"""
数据库认证相关操作模块
主要负责用户认证、访问令牌管理、用户数据库操作等功能
"""

from collections.abc import AsyncGenerator
from collections.abc import Callable
from typing import Any
from typing import Dict

from fastapi import Depends
from fastapi_users.models import ID
from fastapi_users.models import UP
from fastapi_users_db_sqlalchemy import SQLAlchemyUserDatabase
from fastapi_users_db_sqlalchemy.access_token import SQLAlchemyAccessTokenDatabase
from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import Session

from onyx.auth.invited_users import get_invited_users
from onyx.auth.schemas import UserRole
from onyx.db.api_key import get_api_key_email_pattern
from onyx.db.engine import get_async_session
from onyx.db.engine import get_async_session_with_tenant
from onyx.db.models import AccessToken
from onyx.db.models import OAuthAccount
from onyx.db.models import User
from onyx.utils.variable_functionality import (
    fetch_versioned_implementation_with_fallback,
)


def get_default_admin_user_emails() -> list[str]:
    """
    获取默认管理员用户邮箱列表
    仅在企业版(EE)中使用,开源版(MIT)返回空列表
    
    Returns:
        list[str]: 默认管理员邮箱列表
    """
    get_default_admin_user_emails_fn: Callable[
        [], list[str]
    ] = fetch_versioned_implementation_with_fallback(
        "onyx.auth.users", "get_default_admin_user_emails_", lambda: list[str]()
    )
    return get_default_admin_user_emails_fn()


def get_total_users_count(db_session: Session) -> int:
    """
    获取系统中的总用户数量
    包含实际用户数和被邀请用户数的总和
    
    Args:
        db_session: 数据库会话对象
    
    Returns:
        int: 总用户数量
    """
    user_count = (
        db_session.query(User)
        .filter(
            ~User.email.endswith(get_api_key_email_pattern()),  # type: ignore
            User.role != UserRole.EXT_PERM_USER,
        )
        .count()
    )
    invited_users = len(get_invited_users())
    return user_count + invited_users


async def get_user_count(only_admin_users: bool = False) -> int:
    """
    异步获取用户数量
    
    Args:
        only_admin_users: 是否只统计管理员用户数量
    
    Returns:
        int: 用户数量
        
    Raises:
        RuntimeError: 获取用户数量失败时抛出
    """
    async with get_async_session_with_tenant() as session:
        stmt = select(func.count(User.id))
        if only_admin_users:
            stmt = stmt.where(User.role == UserRole.ADMIN)
        result = await session.execute(stmt)
        user_count = result.scalar()
        if user_count is None:
            raise RuntimeError("Was not able to fetch the user count.")
        return user_count


# Need to override this because FastAPI Users doesn't give flexibility for backend field creation logic in OAuth flow
class SQLAlchemyUserAdminDB(SQLAlchemyUserDatabase[UP, ID]):
    """
    自定义用户数据库管理类
    继承自FastAPI Users的SQLAlchemyUserDatabase
    重写create方法以实现自定义的用户角色分配逻辑
    """
    
    async def create(
        self,
        create_dict: Dict[str, Any],
    ) -> UP:
        """
        创建新用户
        
        Args:
            create_dict: 包含用户信息的字典
            
        Returns:
            UP: 创建的用户对象
            
        Note:
            - 首个创建的用户将被赋予管理员权限
            - 在默认管理员邮箱列表中的用户也会被赋予管理员权限
            - 其他用户将被赋予基础用户权限
        """
        user_count = await get_user_count()
        if user_count == 0 or create_dict["email"] in get_default_admin_user_emails():
            create_dict["role"] = UserRole.ADMIN
        else:
            create_dict["role"] = UserRole.BASIC
        return await super().create(create_dict)


async def get_user_db(
    session: AsyncSession = Depends(get_async_session),
) -> AsyncGenerator[SQLAlchemyUserAdminDB, None]:
    """
    获取用户数据库实例的依赖函数
    
    Args:
        session: 异步数据库会话
        
    Yields:
        SQLAlchemyUserAdminDB: 用户数据库管理实例
    """
    yield SQLAlchemyUserAdminDB(session, User, OAuthAccount)  # type: ignore


async def get_access_token_db(
    session: AsyncSession = Depends(get_async_session),
) -> AsyncGenerator[SQLAlchemyAccessTokenDatabase, None]:
    """
    获取访问令牌数据库实例的依赖函数
    
    Args:
        session: 异步数据库会话
        
    Yields:
        SQLAlchemyAccessTokenDatabase: 访问令牌数据库管理实例
    """
    yield SQLAlchemyAccessTokenDatabase(session, AccessToken)  # type: ignore
