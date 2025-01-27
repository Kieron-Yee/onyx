"""
该文件包含系统设置相关的数据模型定义，主要用于处理用户设置、通知和页面类型等配置信息。
主要包括页面类型枚举、访问限制类型枚举、通知模型和设置模型等。
"""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel

from onyx.configs.constants import NotificationType
from onyx.db.models import Notification as NotificationDBModel


class PageType(str, Enum):
    """
    页面类型枚举类，用于定义系统中不同的页面类型
    """
    CHAT = "chat"     # 聊天页面
    SEARCH = "search" # 搜索页面


class GatingType(str, Enum):
    """
    访问限制类型枚举类，用于定义不同级别的功能访问限制
    """
    FULL = "full"     # Complete restriction of access to the product or service
                      # 完全限制访问产品或服务
    PARTIAL = "partial"  # Full access but warning (no credit card on file)
                        # 完全访问但有警告（未绑定信用卡）
    NONE = "none"     # No restrictions, full access to all features
                      # 无限制，可完全访问所有功能


class Notification(BaseModel):
    """
    通知模型类，用于处理系统通知相关的数据
    """
    id: int                          # 通知ID
    notif_type: NotificationType     # 通知类型
    dismissed: bool                  # 是否已关闭
    last_shown: datetime            # 最后显示时间
    first_shown: datetime           # 首次显示时间
    additional_data: dict | None = None  # 额外数据

    @classmethod
    def from_model(cls, notif: NotificationDBModel) -> "Notification":
        """
        从数据库模型创建通知对象的类方法

        参数:
            notif: NotificationDBModel - 数据库中的通知模型实例

        返回:
            Notification - 转换后的通知对象
        """
        return cls(
            id=notif.id,
            notif_type=notif.notif_type,
            dismissed=notif.dismissed,
            last_shown=notif.last_shown,
            first_shown=notif.first_shown,
            additional_data=notif.additional_data,
        )


class Settings(BaseModel):
    """
    General settings
    通用设置模型，用于管理系统的基本配置
    """
    maximum_chat_retention_days: int | None = None  # 聊天记录最大保留天数
    gpu_enabled: bool | None = None                # 是否启用GPU
    product_gating: GatingType = GatingType.NONE   # 产品访问限制类型
    anonymous_user_enabled: bool | None = None     # 是否启用匿名用户


class UserSettings(Settings):
    """
    用户设置模型，继承自Settings类，包含用户特定的设置信息
    """
    notifications: list[Notification]  # 用户通知列表
    needs_reindexing: bool            # 是否需要重新索引
