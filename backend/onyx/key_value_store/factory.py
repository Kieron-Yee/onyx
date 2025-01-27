"""
该模块提供了键值存储(Key-Value Store)的工厂函数。
主要用于创建和获取键值存储实例，支持多租户场景下的数据隔离。
"""

from onyx.key_value_store.interface import KeyValueStore
from onyx.key_value_store.store import PgRedisKVStore


def get_kv_store() -> KeyValueStore:
    """
    获取键值存储实例的工厂函数。
    
    # In the Multi Tenant case, the tenant context is picked up automatically, it does not need to be passed in
    # It's read from the global thread level variable
    # 在多租户场景下，租户上下文会自动获取，不需要手动传入
    # 租户信息从全局线程级变量中读取
    
    Returns:
        KeyValueStore: 返回一个键值存储实例，具体实现为PgRedisKVStore
    """
    return PgRedisKVStore()
