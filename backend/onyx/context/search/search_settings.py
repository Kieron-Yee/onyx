"""
本模块用于管理搜索设置的存储和检索功能。
主要负责从键值存储中获取用户配置的搜索设置，这些设置会影响搜索管道的行为。
"""

from typing import cast

from onyx.configs.constants import KV_SEARCH_SETTINGS
from onyx.context.search.models import SavedSearchSettings
from onyx.key_value_store.factory import get_kv_store
from onyx.key_value_store.interface import KvKeyNotFoundError
from onyx.utils.logger import setup_logger

logger = setup_logger()


def get_kv_search_settings() -> SavedSearchSettings | None:
    """获取所有影响搜索管道的用户配置的搜索设置
    
    Note: KV store is used in this case since there is no need to rollback the value or any need to audit past values
    注释：在这种情况下使用KV存储，因为不需要回滚值或审计过去的值
    
    Note: for now we can't cache this value because if the API server is scaled, the cache could be out of sync
    if the value is updated by another process/instance of the API server. If this reads from an in memory cache like
    reddis then it will be ok. Until then this has some performance implications (though minor)
    注释：目前我们不能缓存这个值，因为如果API服务器进行了扩展，当值被另一个进程/API服务器实例更新时，缓存可能会不同步。
    如果从类似Redis这样的内存缓存中读取则没有问题。在此之前，这会有一些性能影响（虽然影响较小）

    返回值:
        SavedSearchSettings | None: 返回保存的搜索设置对象，如果没有找到或出现错误则返回None
    """
    kv_store = get_kv_store()
    try:
        return SavedSearchSettings(**cast(dict, kv_store.load(KV_SEARCH_SETTINGS)))
    except KvKeyNotFoundError:
        return None
    except Exception as e:
        logger.error(f"Error loading search settings: {e}")
        # Wiping it so that next server startup, it can load the defaults
        # or the user can set it via the API/UI
        # 删除设置，这样在下次服务器启动时可以加载默认值
        # 或者用户可以通过API/UI设置它
        kv_store.delete(KV_SEARCH_SETTINGS)
        return None
