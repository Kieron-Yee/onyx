"""
该文件用于管理被邀请用户的相关操作，包括获取和写入被邀请用户列表。
主要提供了对存储在键值存储中的邀请用户邮箱列表的读写功能。
"""

from typing import cast

from onyx.configs.constants import KV_USER_STORE_KEY
from onyx.key_value_store.factory import get_kv_store
from onyx.key_value_store.interface import KvKeyNotFoundError
from onyx.utils.special_types import JSON_ro


def get_invited_users() -> list[str]:
    """
    获取所有被邀请用户的邮箱列表。
    
    返回值：
        list[str]: 返回包含所有被邀请用户邮箱的列表
        
    说明：
        如果键值存储中没有找到相关数据，则返回空列表
    """
    try:
        store = get_kv_store()
        return cast(list, store.load(KV_USER_STORE_KEY))
    except KvKeyNotFoundError:
        return list()


def write_invited_users(emails: list[str]) -> int:
    """
    将邀请用户的邮箱列表写入存储。
    
    参数：
        emails: list[str] - 需要存储的邮箱地址列表
        
    返回值：
        int: 返回写入的邮箱数量
        
    说明：
        该函数会覆盖原有的邀请用户列表
    """
    store = get_kv_store()
    store.store(KV_USER_STORE_KEY, cast(JSON_ro, emails))
    return len(emails)
