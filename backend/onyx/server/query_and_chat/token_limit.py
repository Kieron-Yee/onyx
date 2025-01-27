"""
此模块用于管理和控制聊天系统的令牌（token）使用限制。
主要功能包括：
- 全局令牌使用率限制检查
- 令牌使用情况统计
- 速率限制验证
"""

from collections.abc import Sequence
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from functools import lru_cache

from dateutil import tz
from fastapi import Depends
from fastapi import HTTPException
from sqlalchemy import func
from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.auth.users import current_chat_accesssible_user
from onyx.db.engine import get_session_context_manager
from onyx.db.engine import get_session_with_tenant
from onyx.db.models import ChatMessage
from onyx.db.models import ChatSession
from onyx.db.models import TokenRateLimit
from onyx.db.models import User
from onyx.db.token_limit import fetch_all_global_token_rate_limits
from onyx.utils.logger import setup_logger
from onyx.utils.variable_functionality import fetch_versioned_implementation
from shared_configs.contextvars import CURRENT_TENANT_ID_CONTEXTVAR


logger = setup_logger()


# 设置令牌预算单位
TOKEN_BUDGET_UNIT = 1_000


def check_token_rate_limits(
    user: User | None = Depends(current_chat_accesssible_user),
) -> None:
    """
    检查用户的令牌使用率限制。
    如果没有设置速率限制，则直接返回；否则执行相应的速率限制策略。
    
    参数:
        user: 当前用户对象，可以为None
    """
    # 如果没有设置速率限制则直接返回
    # 注意：any_rate_limit_exists 的结果已缓存，所以99%的情况下这个调用很快
    if not any_rate_limit_exists():
        return

    versioned_rate_limit_strategy = fetch_versioned_implementation(
        "onyx.server.query_and_chat.token_limit", "_check_token_rate_limits"
    )
    return versioned_rate_limit_strategy(user, CURRENT_TENANT_ID_CONTEXTVAR.get())


def _check_token_rate_limits(_: User | None, tenant_id: str | None) -> None:
    """
    执行令牌速率限制检查。
    
    参数:
        _: 用户对象（未使用）
        tenant_id: 租户ID
    """
    _user_is_rate_limited_by_global(tenant_id)


"""
Global rate limits
全局速率限制
"""


def _user_is_rate_limited_by_global(tenant_id: str | None) -> None:
    """
    检查是否超出全局速率限制。
    如果超出限制，抛出HTTP 429异常。
    
    参数:
        tenant_id: 租户ID
    """
    with get_session_with_tenant(tenant_id) as db_session:
        global_rate_limits = fetch_all_global_token_rate_limits(
            db_session=db_session, enabled_only=True, ordered=False
        )

        if global_rate_limits:
            global_cutoff_time = _get_cutoff_time(global_rate_limits)
            global_usage = _fetch_global_usage(global_cutoff_time, db_session)

            if _is_rate_limited(global_rate_limits, global_usage):
                raise HTTPException(
                    status_code=429,
                    detail="Token budget exceeded for organization. Try again later.",
                )


def _fetch_global_usage(
    cutoff_time: datetime, db_session: Session
) -> Sequence[tuple[datetime, int]]:
    """
    获取截止时间内的全局令牌使用情况，按分钟分组
    
    参数:
        cutoff_time: 截止时间
        db_session: 数据库会话
        
    返回:
        包含时间和令牌使用量的序列
    """
    result = db_session.execute(
        select(
            func.date_trunc("minute", ChatMessage.time_sent),
            func.sum(ChatMessage.token_count),
        )
        .join(ChatSession, ChatMessage.chat_session_id == ChatSession.id)
        .filter(
            ChatMessage.time_sent >= cutoff_time,
        )
        .group_by(func.date_trunc("minute", ChatMessage.time_sent))
    ).all()

    return [(row[0], row[1]) for row in result]


"""
Common functions
公共函数
"""


def _get_cutoff_time(rate_limits: Sequence[TokenRateLimit]) -> datetime:
    """
    计算所有速率限制中最大时间周期的截止时间。
    
    参数:
        rate_limits: 速率限制序列
        
    返回:
        截止时间
    """
    max_period_hours = max(rate_limit.period_hours for rate_limit in rate_limits)
    return datetime.now(tz=timezone.utc) - timedelta(hours=max_period_hours)


def _is_rate_limited(
    rate_limits: Sequence[TokenRateLimit], usage: Sequence[tuple[datetime, int]]
) -> bool:
    """
    如果至少有一个速率限制被超出，返回True
    
    参数:
        rate_limits: 速率限制序列
        usage: 使用情况序列
        
    返回:
        是否超出限制
    """
    for rate_limit in rate_limits:
        tokens_used = sum(
            u_token_count
            for u_date, u_token_count in usage
            if u_date
            >= datetime.now(tz=tz.UTC) - timedelta(hours=rate_limit.period_hours)
        )

        if tokens_used >= rate_limit.token_budget * TOKEN_BUDGET_UNIT:
            return True

    return False


@lru_cache()
def any_rate_limit_exists() -> bool:
    """
    检查数据库中是否存在任何速率限制。结果已缓存，因此如果没有设置速率限制，
    不会影响平均查询延迟。
    
    返回:
        是否存在速率限制
    """
    logger.debug("检查是否存在速率限制...")
    with get_session_context_manager() as db_session:
        return (
            db_session.scalar(
                select(TokenRateLimit.id).where(
                    TokenRateLimit.enabled == True  # noqa: E712
                )
            )
            is not None
        )
