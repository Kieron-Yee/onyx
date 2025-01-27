"""
该文件用于处理系统设置的加载和存储操作。
主要功能包括：
1. 从Redis加载系统设置
2. 将系统设置保存到Redis和KV存储中
"""

from onyx.configs.constants import KV_SETTINGS_KEY
from onyx.configs.constants import OnyxRedisLocks
from onyx.key_value_store.factory import get_kv_store
from onyx.redis.redis_pool import get_redis_client
from onyx.server.settings.models import Settings
from shared_configs.configs import MULTI_TENANT


def load_settings() -> Settings:
    """
    加载系统设置。
    
    主要功能：
    - 如果是多租户模式，匿名用户始终为禁用状态
    - 如果是单租户模式，从Redis中读取匿名用户状态
    - 如果Redis中没有相关设置，默认禁用匿名用户
    
    返回：
        Settings: 包含系统设置的对象
    """
    if MULTI_TENANT:
        # If multi-tenant, anonymous user is always false
        # 如果是多租户模式，匿名用户始终为禁用状态
        anonymous_user_enabled = False
    else:
        redis_client = get_redis_client(tenant_id=None)
        value = redis_client.get(OnyxRedisLocks.ANONYMOUS_USER_ENABLED)
        if value is not None:
            assert isinstance(value, bytes)
            anonymous_user_enabled = int(value.decode("utf-8")) == 1
        else:
            # Default to False
            # 默认设置为False
            anonymous_user_enabled = False
            # Optionally store the default back to Redis
            # 可选地将默认值存储回Redis
            redis_client.set(OnyxRedisLocks.ANONYMOUS_USER_ENABLED, "0")

    settings = Settings(anonymous_user_enabled=anonymous_user_enabled)
    return settings


def store_settings(settings: Settings) -> None:
    """
    存储系统设置。
    
    主要功能：
    - 在单租户模式下，将匿名用户状态保存到Redis
    - 将完整的设置信息保存到KV存储中
    
    参数：
        settings (Settings): 需要存储的设置对象
    """
    if not MULTI_TENANT and settings.anonymous_user_enabled is not None:
        # Only non-multi-tenant scenario can set the anonymous user enabled flag
        # 只有在非多租户场景下才能设置匿名用户启用标志
        redis_client = get_redis_client(tenant_id=None)
        redis_client.set(
            OnyxRedisLocks.ANONYMOUS_USER_ENABLED,
            "1" if settings.anonymous_user_enabled else "0",
        )

    get_kv_store().store(KV_SETTINGS_KEY, settings.model_dump())
