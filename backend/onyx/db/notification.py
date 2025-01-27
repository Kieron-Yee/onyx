"""
这个文件主要用于处理系统通知相关的数据库操作，包括创建、获取、更新和关闭通知等功能。
提供了一系列函数来管理通知的生命周期和状态。
"""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session
from sqlalchemy.sql import func

from onyx.auth.schemas import UserRole
from onyx.configs.constants import NotificationType
from onyx.db.models import Notification
from onyx.db.models import User


def create_notification(
    user_id: UUID | None,
    notif_type: NotificationType,
    db_session: Session,
    additional_data: dict | None = None,
) -> Notification:
    """
    创建一个新的通知，或更新现有的未关闭的相同类型通知。
    
    如果存在相同类型和数据的未关闭通知，则更新其最后显示时间；
    否则创建一个新的通知记录。
    """
    # Check if an undismissed notification of the same type and data exists
    # 检查是否存在相同类型和数据的未关闭通知
    existing_notification = (
        db_session.query(Notification)
        .filter_by(
            user_id=user_id,
            notif_type=notif_type,
            dismissed=False,
        )
        .filter(Notification.additional_data == additional_data)
        .first()
    )

    if existing_notification:
        # Update the last_shown timestamp
        existing_notification.last_shown = func.now()
        db_session.commit()
        return existing_notification

    # Create a new notification if none exists
    notification = Notification(
        user_id=user_id,
        notif_type=notif_type,
        dismissed=False,
        last_shown=func.now(),
        first_shown=func.now(),
        additional_data=additional_data,
    )
    db_session.add(notification)
    db_session.commit()
    return notification


def get_notification_by_id(
    notification_id: int, user: User | None, db_session: Session
) -> Notification:
    """
    根据通知ID获取特定的通知。
    
    包含权限验证：只有通知的所有者或管理员（对于全局通知）可以访问。
    如果通知不存在或用户无权访问，将抛出相应异常。
    """
    user_id = user.id if user else None
    notif = db_session.get(Notification, notification_id)
    if not notif:
        raise ValueError(f"No notification found with id {notification_id}")
    if notif.user_id != user_id and not (
        notif.user_id is None and user is not None and user.role == UserRole.ADMIN
    ):
        raise PermissionError(
            f"User {user_id} is not authorized to access notification {notification_id}"
        )
    return notif


def get_notifications(
    user: User | None,
    db_session: Session,
    notif_type: NotificationType | None = None,
    include_dismissed: bool = True,
) -> list[Notification]:
    """
    获取指定用户的所有通知。
    
    可以根据通知类型筛选，并可选择是否包含已关闭的通知。
    如果user为None，则获取全局通知。
    """
    query = select(Notification).where(
        Notification.user_id == user.id if user else Notification.user_id.is_(None)
    )
    if not include_dismissed:
        query = query.where(Notification.dismissed.is_(False))
    if notif_type:
        query = query.where(Notification.notif_type == notif_type)
    return list(db_session.execute(query).scalars().all())


def dismiss_all_notifications(
    notif_type: NotificationType,
    db_session: Session,
) -> None:
    """
    关闭所有指定类型的通知。
    
    批量更新操作，将指定类型的所有通知标记为已关闭。
    """
    db_session.query(Notification).filter(Notification.notif_type == notif_type).update(
        {"dismissed": True}
    )
    db_session.commit()


def dismiss_notification(notification: Notification, db_session: Session) -> None:
    """
    关闭单个通知。
    
    将指定的通知标记为已关闭状态。
    """
    notification.dismissed = True
    db_session.commit()


def update_notification_last_shown(
    notification: Notification, db_session: Session
) -> None:
    """
    更新通知的最后显示时间。
    
    将通知的last_shown时间戳更新为当前时间。
    """
    notification.last_shown = func.now()
    db_session.commit()
