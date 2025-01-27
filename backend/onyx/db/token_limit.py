"""
此模块用于管理令牌使用率限制的数据库操作。
主要包含用户级别和全局级别的令牌限制的增删改查功能。
"""

from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.configs.constants import TokenRateLimitScope
from onyx.db.models import TokenRateLimit
from onyx.db.models import TokenRateLimit__UserGroup
from onyx.server.token_rate_limits.models import TokenRateLimitArgs


def fetch_all_user_token_rate_limits(
    db_session: Session,
    enabled_only: bool = False,
    ordered: bool = True,
) -> Sequence[TokenRateLimit]:
    """
    获取所有用户级别的令牌使用率限制配置。
    
    Args:
        db_session: 数据库会话对象
        enabled_only: 是否只获取已启用的限制
        ordered: 是否按创建时间倒序排序
    
    Returns:
        包含所有符合条件的用户令牌限制配置的序列
    """
    query = select(TokenRateLimit).where(
        TokenRateLimit.scope == TokenRateLimitScope.USER
    )

    if enabled_only:
        query = query.where(TokenRateLimit.enabled.is_(True))

    if ordered:
        query = query.order_by(TokenRateLimit.created_at.desc())

    return db_session.scalars(query).all()


def fetch_all_global_token_rate_limits(
    db_session: Session,
    enabled_only: bool = False,
    ordered: bool = True,
) -> Sequence[TokenRateLimit]:
    """
    获取所有全局级别的令牌使用率限制配置。
    
    Args:
        db_session: 数据库会话对象
        enabled_only: 是否只获取已启用的限制
        ordered: 是否按创建时间倒序排序
    
    Returns:
        包含所有符合条件的全局令牌限制配置的序列
    """
    query = select(TokenRateLimit).where(
        TokenRateLimit.scope == TokenRateLimitScope.GLOBAL
    )

    if enabled_only:
        query = query.where(TokenRateLimit.enabled.is_(True))

    if ordered:
        query = query.order_by(TokenRateLimit.created_at.desc())

    token_rate_limits = db_session.scalars(query).all()
    return token_rate_limits


def insert_user_token_rate_limit(
    db_session: Session,
    token_rate_limit_settings: TokenRateLimitArgs,
) -> TokenRateLimit:
    """
    创建新的用户级别令牌使用率限制配置。
    
    Args:
        db_session: 数据库会话对象
        token_rate_limit_settings: 令牌限制配置参数
    
    Returns:
        新创建的令牌限制配置对象
    """
    token_limit = TokenRateLimit(
        enabled=token_rate_limit_settings.enabled,
        token_budget=token_rate_limit_settings.token_budget,
        period_hours=token_rate_limit_settings.period_hours,
        scope=TokenRateLimitScope.USER,
    )
    db_session.add(token_limit)
    db_session.commit()

    return token_limit


def insert_global_token_rate_limit(
    db_session: Session,
    token_rate_limit_settings: TokenRateLimitArgs,
) -> TokenRateLimit:
    """
    创建新的全局级别令牌使用率限制配置。
    
    Args:
        db_session: 数据库会话对象
        token_rate_limit_settings: 令牌限制配置参数
    
    Returns:
        新创建的令牌限制配置对象
    """
    token_limit = TokenRateLimit(
        enabled=token_rate_limit_settings.enabled,
        token_budget=token_rate_limit_settings.token_budget,
        period_hours=token_rate_limit_settings.period_hours,
        scope=TokenRateLimitScope.GLOBAL,
    )
    db_session.add(token_limit)
    db_session.commit()

    return token_limit


def update_token_rate_limit(
    db_session: Session,
    token_rate_limit_id: int,
    token_rate_limit_settings: TokenRateLimitArgs,
) -> TokenRateLimit:
    """
    更新指定的令牌使用率限制配置。
    
    Args:
        db_session: 数据库会话对象
        token_rate_limit_id: 要更新的限制配置ID
        token_rate_limit_settings: 新的限制配置参数
    
    Returns:
        更新后的令牌限制配置对象
    
    Raises:
        ValueError: 当指定ID的限制配置不存在时抛出
    """
    token_limit = db_session.get(TokenRateLimit, token_rate_limit_id)
    if token_limit is None:
        raise ValueError(f"TokenRateLimit with id '{token_rate_limit_id}' not found")

    token_limit.enabled = token_rate_limit_settings.enabled
    token_limit.token_budget = token_rate_limit_settings.token_budget
    token_limit.period_hours = token_rate_limit_settings.period_hours
    db_session.commit()

    return token_limit


def delete_token_rate_limit(
    db_session: Session,
    token_rate_limit_id: int,
) -> None:
    """
    删除指定的令牌使用率限制配置及其关联的用户组关系。
    
    Args:
        db_session: 数据库会话对象
        token_rate_limit_id: 要删除的限制配置ID
    
    Raises:
        ValueError: 当指定ID的限制配置不存在时抛出
    """
    token_limit = db_session.get(TokenRateLimit, token_rate_limit_id)
    if token_limit is None:
        raise ValueError(f"TokenRateLimit with id '{token_rate_limit_id}' not found")

    db_session.query(TokenRateLimit__UserGroup).filter(
        TokenRateLimit__UserGroup.rate_limit_id == token_rate_limit_id
    ).delete()

    db_session.delete(token_limit)
    db_session.commit()
