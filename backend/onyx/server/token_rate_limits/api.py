"""
此文件实现了令牌速率限制的API接口，主要功能包括：
1. 全局令牌限制设置的管理（查询、创建）
2. 一般令牌限制设置的管理（更新、删除）
所有接口都需要管理员权限
"""

from fastapi import APIRouter
from fastapi import Depends
from sqlalchemy.orm import Session

from onyx.auth.users import current_admin_user
from onyx.db.engine import get_session
from onyx.db.models import User
from onyx.db.token_limit import delete_token_rate_limit
from onyx.db.token_limit import fetch_all_global_token_rate_limits
from onyx.db.token_limit import insert_global_token_rate_limit
from onyx.db.token_limit import update_token_rate_limit
from onyx.server.query_and_chat.token_limit import any_rate_limit_exists
from onyx.server.token_rate_limits.models import TokenRateLimitArgs
from onyx.server.token_rate_limits.models import TokenRateLimitDisplay

router = APIRouter(prefix="/admin/token-rate-limits")


"""
Global Token Limit Settings
全局令牌限制设置
"""


@router.get("/global")
def get_global_token_limit_settings(
    _: User | None = Depends(current_admin_user),
    db_session: Session = Depends(get_session),
) -> list[TokenRateLimitDisplay]:
    """
    获取所有全局令牌限制设置

    Args:
        _: 当前管理员用户，通过依赖注入获取
        db_session: 数据库会话，通过依赖注入获取

    Returns:
        list[TokenRateLimitDisplay]: 全局令牌限制设置列表
    """
    return [
        TokenRateLimitDisplay.from_db(token_rate_limit)
        for token_rate_limit in fetch_all_global_token_rate_limits(db_session)
    ]


@router.post("/global")
def create_global_token_limit_settings(
    token_limit_settings: TokenRateLimitArgs,
    _: User | None = Depends(current_admin_user),
    db_session: Session = Depends(get_session),
) -> TokenRateLimitDisplay:
    """
    创建新的全局令牌限制设置

    Args:
        token_limit_settings: 令牌限制设置参数
        _: 当前管理员用户，通过依赖注入获取
        db_session: 数据库会话，通过依赖注入获取

    Returns:
        TokenRateLimitDisplay: 新创建的令牌限制设置
    """
    rate_limit_display = TokenRateLimitDisplay.from_db(
        insert_global_token_rate_limit(db_session, token_limit_settings)
    )
    # clear cache in case this was the first rate limit created
    # 清除缓存，以防这是第一个创建的速率限制
    any_rate_limit_exists.cache_clear()
    return rate_limit_display


"""
General Token Limit Settings
一般令牌限制设置
"""


@router.put("/rate-limit/{token_rate_limit_id}")
def update_token_limit_settings(
    token_rate_limit_id: int,
    token_limit_settings: TokenRateLimitArgs,
    _: User | None = Depends(current_admin_user),
    db_session: Session = Depends(get_session),
) -> TokenRateLimitDisplay:
    """
    更新指定的令牌限制设置

    Args:
        token_rate_limit_id: 要更新的令牌限制设置ID
        token_limit_settings: 新的令牌限制设置参数
        _: 当前管理员用户，通过依赖注入获取
        db_session: 数据库会话，通过依赖注入获取

    Returns:
        TokenRateLimitDisplay: 更新后的令牌限制设置
    """
    return TokenRateLimitDisplay.from_db(
        update_token_rate_limit(
            db_session=db_session,
            token_rate_limit_id=token_rate_limit_id,
            token_rate_limit_settings=token_limit_settings,
        )
    )


@router.delete("/rate-limit/{token_rate_limit_id}")
def delete_token_limit_settings(
    token_rate_limit_id: int,
    _: User | None = Depends(current_admin_user),
    db_session: Session = Depends(get_session),
) -> None:
    """
    删除指定的令牌限制设置

    Args:
        token_rate_limit_id: 要删除的令牌限制设置ID
        _: 当前管理员用户，通过依赖注入获取
        db_session: 数据库会话，通过依赖注入获取

    Returns:
        None
    """
    return delete_token_rate_limit(
        db_session=db_session,
        token_rate_limit_id=token_rate_limit_id,
    )
