"""
通知系统API模块

本模块提供了与通知系统相关的API端点，包括获取通知列表和处理通知的标记为已读功能。
主要功能包括：
- 获取用户的未读通知列表
- 将指定通知标记为已读
"""

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from sqlalchemy.orm import Session

from onyx.auth.users import current_user
from onyx.db.engine import get_session
from onyx.db.models import User
from onyx.db.notification import dismiss_notification
from onyx.db.notification import get_notification_by_id
from onyx.db.notification import get_notifications
from onyx.server.settings.models import Notification as NotificationModel
from onyx.utils.logger import setup_logger

logger = setup_logger()

router = APIRouter(prefix="/notifications")


@router.get("")
def get_notifications_api(
    user: User = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> list[NotificationModel]:
    """
    获取用户的未读通知列表
    
    Args:
        user (User): 当前登录用户，通过依赖注入获取
        db_session (Session): 数据库会话，通过依赖注入获取
    
    Returns:
        list[NotificationModel]: 返回未读通知列表
    """
    notifications = [
        NotificationModel.from_model(notif)
        for notif in get_notifications(user, db_session, include_dismissed=False)
    ]
    return notifications


@router.post("/{notification_id}/dismiss")
def dismiss_notification_endpoint(
    notification_id: int,
    user: User | None = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> None:
    """
    将指定通知标记为已读
    
    Args:
        notification_id (int): 需要标记为已读的通知ID
        user (User | None): 当前登录用户，通过依赖注入获取
        db_session (Session): 数据库会话，通过依赖注入获取
    
    Raises:
        HTTPException: 
            - 403: 用户没有权限处理该通知
            - 404: 通知不存在
    
    Returns:
        None
    """
    try:
        notification = get_notification_by_id(notification_id, user, db_session)
    except PermissionError:
        raise HTTPException(
            status_code=403, detail="Not authorized to dismiss this notification"  # 没有权限处理该通知
        )
    except ValueError:
        raise HTTPException(status_code=404, detail="Notification not found")  # 通知不存在

    dismiss_notification(notification, db_session)
