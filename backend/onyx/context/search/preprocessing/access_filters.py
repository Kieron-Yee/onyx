"""
访问过滤器模块
此模块用于构建用户访问权限相关的过滤条件，包括：
- 构建用户的访问控制列表过滤器
- 构建仅用户相关的过滤条件
"""

from sqlalchemy.orm import Session

from onyx.access.access import get_acl_for_user
from onyx.context.search.models import IndexFilters
from onyx.db.models import User


def build_access_filters_for_user(user: User | None, session: Session) -> list[str]:
    """
    构建指定用户的访问控制过滤器列表

    参数:
        user: User | None - 用户对象或None
        session: Session - 数据库会话对象

    返回:
        list[str] - 用户的访问控制列表
    """
    user_acl = get_acl_for_user(user, session)
    return list(user_acl)


def build_user_only_filters(user: User | None, db_session: Session) -> IndexFilters:
    """
    构建仅包含用户访问控制过滤器的索引过滤条件

    参数:
        user: User | None - 用户对象或None
        db_session: Session - 数据库会话对象

    返回:
        IndexFilters - 包含用户访问控制列表的过滤器对象，其他过滤条件均设为None
    """
    user_acl_filters = build_access_filters_for_user(user, db_session)
    return IndexFilters(
        source_type=None,
        document_set=None,
        time_cutoff=None,
        tags=None,
        access_control_list=user_acl_filters,
    )
