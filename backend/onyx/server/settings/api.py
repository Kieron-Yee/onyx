"""
这个文件实现了系统设置相关的API路由处理。
主要功能包括：
- 管理员设置的更新和获取
- 用户设置的获取
- 系统通知的处理
"""

from typing import cast

from fastapi import APIRouter
from fastapi import Depends
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from onyx.auth.users import current_admin_user
from onyx.auth.users import current_user
from onyx.auth.users import is_user_admin
from onyx.configs.constants import KV_REINDEX_KEY
from onyx.configs.constants import NotificationType
from onyx.db.engine import get_session
from onyx.db.models import User
from onyx.db.notification import create_notification
from onyx.db.notification import dismiss_all_notifications
from onyx.db.notification import get_notifications
from onyx.db.notification import update_notification_last_shown
from onyx.key_value_store.factory import get_kv_store
from onyx.key_value_store.interface import KvKeyNotFoundError
from onyx.server.settings.models import Notification
from onyx.server.settings.models import Settings
from onyx.server.settings.models import UserSettings
from onyx.server.settings.store import load_settings
from onyx.server.settings.store import store_settings
from onyx.utils.logger import setup_logger


logger = setup_logger()


admin_router = APIRouter(prefix="/admin/settings")
basic_router = APIRouter(prefix="/settings")


@admin_router.put("")
def put_settings(
    settings: Settings, _: User | None = Depends(current_admin_user)
) -> None:
    """更新系统设置
    
    Args:
        settings: 要存储的设置对象
        _: 当前管理员用户(通过依赖注入)
    """
    store_settings(settings)


@basic_router.get("")
def fetch_settings(
    user: User | None = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> UserSettings:
    """Settings and notifications are stuffed into this single endpoint to reduce number of
    Postgres calls
    将设置和通知整合到这个单一端点以减少PostgreSQL调用次数
    
    Args:
        user: 当前用户(通过依赖注入)
        db_session: 数据库会话(通过依赖注入)
    
    Returns:
        UserSettings: 包含通用设置和用户通知的设置对象
    """
    general_settings = load_settings()
    settings_notifications = get_settings_notifications(user, db_session)

    try:
        kv_store = get_kv_store()
        needs_reindexing = cast(bool, kv_store.load(KV_REINDEX_KEY))
    except KvKeyNotFoundError:
        needs_reindexing = False

    return UserSettings(
        **general_settings.model_dump(),
        notifications=settings_notifications,
        needs_reindexing=needs_reindexing,
    )


def get_settings_notifications(
    user: User | None, db_session: Session
) -> list[Notification]:
    """Get notifications for settings page, including product gating and reindex notifications
    获取设置页面的通知，包括产品限制和重新索引通知
    
    Args:
        user: 当前用户
        db_session: 数据库会话
    
    Returns:
        list[Notification]: 通知列表
    
    Note:
        - 检查产品限制通知
        - 仅向管理员显示重新索引通知
        - 处理重新索引标志的状态
    """
    # Check for product gating notification
    # 检查产品限制通知
    product_notif = get_notifications(
        user=None,
        notif_type=NotificationType.TRIAL_ENDS_TWO_DAYS,
        db_session=db_session,
    )
    notifications = [Notification.from_model(product_notif[0])] if product_notif else []

    # Only show reindex notifications to admins
    # 仅向管理员显示重新索引通知
    is_admin = is_user_admin(user)
    if not is_admin:
        return notifications

    # Check if reindexing is needed
    # 检查是否需要重新索引
    kv_store = get_kv_store()
    try:
        needs_index = cast(bool, kv_store.load(KV_REINDEX_KEY))
        if not needs_index:
            dismiss_all_notifications(
                notif_type=NotificationType.REINDEX, db_session=db_session
            )
            return notifications
    except KvKeyNotFoundError:
        # If something goes wrong and the flag is gone, better to not start a reindexing
        # it's a heavyweight long running job and maybe this flag is cleaned up later
        # 如果出现错误且标志消失，最好不要开始重新索引
        # 因为这是一个重量级的长期运行任务，可能这个标志稍后会被清理
        logger.warning("Could not find reindex flag")
        return notifications

    try:
        # Need a transaction in order to prevent under-counting current notifications
        # 需要事务以防止当前通知计数不足
        reindex_notifs = get_notifications(
            user=user, notif_type=NotificationType.REINDEX, db_session=db_session
        )

        if not reindex_notifs:
            notif = create_notification(
                user_id=user.id if user else None,
                notif_type=NotificationType.REINDEX,
                db_session=db_session,
            )
            db_session.flush()
            db_session.commit()

            notifications.append(Notification.from_model(notif))
            return notifications

        if len(reindex_notifs) > 1:
            logger.error("User has multiple reindex notifications")

        reindex_notif = reindex_notifs[0]
        update_notification_last_shown(
            notification=reindex_notif, db_session=db_session
        )

        db_session.commit()
        notifications.append(Notification.from_model(reindex_notif))
        return notifications
    except SQLAlchemyError:
        logger.exception("Error while processing notifications")  # 处理通知时发生错误
        db_session.rollback()
        return notifications
