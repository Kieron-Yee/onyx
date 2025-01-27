"""
此模块主要用于处理无认证用户相关的功能，包括无认证用户的偏好设置管理和用户信息获取。
它提供了一套完整的接口来处理未登录用户的配置和权限管理。
"""

from collections.abc import Mapping
from typing import Any
from typing import cast

from onyx.auth.schemas import UserRole
from onyx.configs.constants import KV_NO_AUTH_USER_PREFERENCES_KEY
from onyx.configs.constants import NO_AUTH_USER_EMAIL
from onyx.configs.constants import NO_AUTH_USER_ID
from onyx.key_value_store.store import KeyValueStore
from onyx.key_value_store.store import KvKeyNotFoundError
from onyx.server.manage.models import UserInfo
from onyx.server.manage.models import UserPreferences


def set_no_auth_user_preferences(
    store: KeyValueStore, preferences: UserPreferences
) -> None:
    """
    设置无认证用户的偏好设置。

    Args:
        store (KeyValueStore): 键值存储对象，用于存储用户偏好
        preferences (UserPreferences): 用户偏好设置对象

    Returns:
        None
    """
    store.store(KV_NO_AUTH_USER_PREFERENCES_KEY, preferences.model_dump())


def load_no_auth_user_preferences(store: KeyValueStore) -> UserPreferences:
    """
    加载无认证用户的偏好设置。如果没有找到设置，则返回默认值。

    Args:
        store (KeyValueStore): 键值存储对象，用于获取用户偏好

    Returns:
        UserPreferences: 用户偏好设置对象，如果未找到则返回默认设置
    """
    try:
        preferences_data = cast(
            Mapping[str, Any], store.load(KV_NO_AUTH_USER_PREFERENCES_KEY)
        )
        return UserPreferences(**preferences_data)
    except KvKeyNotFoundError:
        return UserPreferences(
            chosen_assistants=None, default_model=None, auto_scroll=True
        )


def fetch_no_auth_user(
    store: KeyValueStore, *, anonymous_user_enabled: bool | None = None
) -> UserInfo:
    """
    获取无认证用户的用户信息。

    Args:
        store (KeyValueStore): 键值存储对象，用于获取用户偏好
        anonymous_user_enabled (bool | None): 是否启用匿名用户功能，默认为None
            - 如果为True，用户角色将被设置为BASIC
            - 如果为False，用户角色将被设置为ADMIN

    Returns:
        UserInfo: 包含无认证用户完整信息的用户信息对象，包括ID、邮箱、权限等
    """
    return UserInfo(
        id=NO_AUTH_USER_ID,
        email=NO_AUTH_USER_EMAIL,
        is_active=True,
        is_superuser=False,
        is_verified=True,
        role=UserRole.BASIC if anonymous_user_enabled else UserRole.ADMIN,
        preferences=load_no_auth_user_preferences(store),
        is_anonymous_user=anonymous_user_enabled,
    )
