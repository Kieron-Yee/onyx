"""
API密钥管理模块：
提供API密钥的创建、查询、更新、重新生成和删除等数据库操作功能。
包含API密钥相关的数据库交互逻辑，用于管理系统中的API密钥及其关联用户。
"""

import uuid

from fastapi_users.password import PasswordHelper
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload
from sqlalchemy.orm import Session

from onyx.auth.api_key import ApiKeyDescriptor
from onyx.auth.api_key import build_displayable_api_key
from onyx.auth.api_key import generate_api_key
from onyx.auth.api_key import hash_api_key
from onyx.configs.constants import DANSWER_API_KEY_DUMMY_EMAIL_DOMAIN
from onyx.configs.constants import DANSWER_API_KEY_PREFIX
from onyx.configs.constants import UNNAMED_KEY_PLACEHOLDER
from onyx.db.models import ApiKey
from onyx.db.models import User
from onyx.server.api_key.models import APIKeyArgs
from shared_configs.configs import MULTI_TENANT
from shared_configs.contextvars import CURRENT_TENANT_ID_CONTEXTVAR


def get_api_key_email_pattern() -> str:
    """获取API密钥关联的邮箱域名模式
    
    Returns:
        str: API密钥使用的邮箱域名后缀
    """
    return DANSWER_API_KEY_DUMMY_EMAIL_DOMAIN


def is_api_key_email_address(email: str) -> bool:
    """检查给定的邮箱是否是API密钥关联的邮箱地址
    
    Args:
        email: 需要检查的邮箱地址
    Returns:
        bool: 如果是API密钥邮箱则返回True，否则返回False
    """
    return email.endswith(get_api_key_email_pattern())


def fetch_api_keys(db_session: Session) -> list[ApiKeyDescriptor]:
    """获取所有API密钥信息
    
    Args:
        db_session: 数据库会话对象
    Returns:
        list[ApiKeyDescriptor]: API密钥描述符列表
    """
    api_keys = (
        db_session.scalars(select(ApiKey).options(joinedload(ApiKey.user)))
        .unique()
        .all()
    )
    return [
        ApiKeyDescriptor(
            api_key_id=api_key.id,
            api_key_role=api_key.user.role,
            api_key_display=api_key.api_key_display,
            api_key_name=api_key.name,
            user_id=api_key.user_id,
        )
        for api_key in api_keys
    ]


async def fetch_user_for_api_key(
    hashed_api_key: str, async_db_session: AsyncSession
) -> User | None:
    """根据哈希后的API密钥获取关联用户
    
    Args:
        hashed_api_key: 经过哈希处理的API密钥
        async_db_session: 异步数据库会话对象
    Returns:
        User | None: 关联的用户对象，如果未找到则返回None
    """
    return await async_db_session.scalar(
        select(User)
        .join(ApiKey, ApiKey.user_id == User.id)
        .where(ApiKey.hashed_api_key == hashed_api_key)
    )


def get_api_key_fake_email(
    name: str,
    unique_id: str,
) -> str:
    """生成API密钥关联的虚拟邮箱地址
    
    Args:
        name: API密钥名称
        unique_id: 唯一标识符
    Returns:
        str: 生成的虚拟邮箱地址
    """
    return f"{DANSWER_API_KEY_PREFIX}{name}@{unique_id}{DANSWER_API_KEY_DUMMY_EMAIL_DOMAIN}"


def insert_api_key(
    db_session: Session, api_key_args: APIKeyArgs, user_id: uuid.UUID | None
) -> ApiKeyDescriptor:
    """创建新的API密钥
    
    Args:
        db_session: 数据库会话对象
        api_key_args: API密钥创建参数
        user_id: 创建者的用户ID
    Returns:
        ApiKeyDescriptor: 新创建的API密钥描述符
    """
    std_password_helper = PasswordHelper()

    # Get tenant_id from context var (will be default schema for single tenant)
    tenant_id = CURRENT_TENANT_ID_CONTEXTVAR.get()

    api_key = generate_api_key(tenant_id if MULTI_TENANT else None)
    api_key_user_id = uuid.uuid4()

    display_name = api_key_args.name or UNNAMED_KEY_PLACEHOLDER
    api_key_user_row = User(
        id=api_key_user_id,
        email=get_api_key_fake_email(display_name, str(api_key_user_id)),
        # a random password for the "user"
        hashed_password=std_password_helper.hash(std_password_helper.generate()),
        is_active=True,
        is_superuser=False,
        is_verified=True,
        role=api_key_args.role,
    )
    db_session.add(api_key_user_row)

    api_key_row = ApiKey(
        name=api_key_args.name,
        hashed_api_key=hash_api_key(api_key),
        api_key_display=build_displayable_api_key(api_key),
        user_id=api_key_user_id,
        owner_id=user_id,
    )
    db_session.add(api_key_row)

    db_session.commit()
    return ApiKeyDescriptor(
        api_key_id=api_key_row.id,
        api_key_role=api_key_user_row.role,
        api_key_display=api_key_row.api_key_display,
        api_key=api_key,
        api_key_name=api_key_args.name,
        user_id=api_key_user_id,
    )


def update_api_key(
    db_session: Session, api_key_id: int, api_key_args: APIKeyArgs
) -> ApiKeyDescriptor:
    """更新现有API密钥的信息
    
    Args:
        db_session: 数据库会话对象
        api_key_id: 要更新的API密钥ID
        api_key_args: 更新参数
    Returns:
        ApiKeyDescriptor: 更新后的API密钥描述符
    Raises:
        ValueError: 当API密钥不存在时抛出
        RuntimeError: 当API密钥没有关联用户时抛出
    """
    existing_api_key = db_session.scalar(select(ApiKey).where(ApiKey.id == api_key_id))
    if existing_api_key is None:
        raise ValueError(f"API key with id {api_key_id} does not exist")

    existing_api_key.name = api_key_args.name
    api_key_user = db_session.scalar(
        select(User).where(User.id == existing_api_key.user_id)  # type: ignore
    )
    if api_key_user is None:
        raise RuntimeError("API Key does not have associated user.")

    email_name = api_key_args.name or UNNAMED_KEY_PLACEHOLDER
    api_key_user.email = get_api_key_fake_email(email_name, str(api_key_user.id))
    api_key_user.role = api_key_args.role
    db_session.commit()

    return ApiKeyDescriptor(
        api_key_id=existing_api_key.id,
        api_key_display=existing_api_key.api_key_display,
        api_key_name=api_key_args.name,
        api_key_role=api_key_user.role,
        user_id=existing_api_key.user_id,
    )


def regenerate_api_key(db_session: Session, api_key_id: int) -> ApiKeyDescriptor:
    """重新生成API密钥
    
    Args:
        db_session: 数据库会话对象
        api_key_id: 要重新生成的API密钥ID
    Returns:
        ApiKeyDescriptor: 重新生成后的API密钥描述符
    Raises:
        ValueError: 当API密钥不存在时抛出
        RuntimeError: 当API密钥没有关联用户时抛出
    """
    """NOTE: currently, any admin can regenerate any API key."""
    existing_api_key = db_session.scalar(select(ApiKey).where(ApiKey.id == api_key_id))
    if existing_api_key is None:
        raise ValueError(f"API key with id {api_key_id} does not exist")

    api_key_user = db_session.scalar(
        select(User).where(User.id == existing_api_key.user_id)  # type: ignore
    )
    if api_key_user is None:
        raise RuntimeError("API Key does not have associated user.")

    new_api_key = generate_api_key()
    existing_api_key.hashed_api_key = hash_api_key(new_api_key)
    existing_api_key.api_key_display = build_displayable_api_key(new_api_key)
    db_session.commit()

    return ApiKeyDescriptor(
        api_key_id=existing_api_key.id,
        api_key_display=existing_api_key.api_key_display,
        api_key=new_api_key,
        api_key_name=existing_api_key.name,
        api_key_role=api_key_user.role,
        user_id=existing_api_key.user_id,
    )


def remove_api_key(db_session: Session, api_key_id: int) -> None:
    """删除指定的API密钥及其关联用户
    
    Args:
        db_session: 数据库会话对象
        api_key_id: 要删除的API密钥ID
    Raises:
        ValueError: 当API密钥或关联用户不存在时抛出
    """
    existing_api_key = db_session.scalar(select(ApiKey).where(ApiKey.id == api_key_id))
    if existing_api_key is None:
        raise ValueError(f"API key with id {api_key_id} does not exist")

    user_associated_with_key = db_session.scalar(
        select(User).where(User.id == existing_api_key.user_id)  # type: ignore
    )
    if user_associated_with_key is None:
        raise ValueError(
            f"User associated with API key with id {api_key_id} does not exist. This should not happen."
        )

    db_session.delete(existing_api_key)
    db_session.delete(user_associated_with_key)
    db_session.commit()
