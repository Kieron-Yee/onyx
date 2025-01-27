"""
本模块提供了Redis连接器的核心实现。
主要功能：
1. 管理与Redis的连接
2. 处理索引、权限同步等后台任务
3. 提供连接器相关的辅助功能
"""

import time

import redis

from onyx.db.models import SearchSettings
from onyx.redis.redis_connector_delete import RedisConnectorDelete
from onyx.redis.redis_connector_doc_perm_sync import RedisConnectorPermissionSync
from onyx.redis.redis_connector_ext_group_sync import RedisConnectorExternalGroupSync
from onyx.redis.redis_connector_index import RedisConnectorIndex
from onyx.redis.redis_connector_prune import RedisConnectorPrune
from onyx.redis.redis_connector_stop import RedisConnectorStop
from onyx.redis.redis_pool import get_redis_client


class RedisConnector:
    """Composes several classes to simplify interacting with a connector and its
    associated background tasks / redis interactions.
    
    组合多个类以简化连接器及其相关后台任务/Redis交互的操作。
    """

    def __init__(self, tenant_id: str | None, id: int) -> None:
        """
        初始化Redis连接器
        
        参数:
            tenant_id: 租户ID
            id: 连接器ID
        """
        self.tenant_id: str | None = tenant_id
        self.id: int = id
        self.redis: redis.Redis = get_redis_client(tenant_id=tenant_id)

        self.stop = RedisConnectorStop(tenant_id, id, self.redis)
        self.prune = RedisConnectorPrune(tenant_id, id, self.redis)
        self.delete = RedisConnectorDelete(tenant_id, id, self.redis)
        self.permissions = RedisConnectorPermissionSync(tenant_id, id, self.redis)
        self.external_group_sync = RedisConnectorExternalGroupSync(
            tenant_id, id, self.redis
        )

    def new_index(self, search_settings_id: int) -> RedisConnectorIndex:
        """
        创建新的索引实例
        
        参数:
            search_settings_id: 搜索设置ID
        返回:
            RedisConnectorIndex实例
        """
        return RedisConnectorIndex(
            self.tenant_id, self.id, search_settings_id, self.redis
        )

    def wait_for_indexing_termination(
        self,
        search_settings_list: list[SearchSettings],
        timeout: float = 15.0,
    ) -> bool:
        """Returns True if all indexing for the given redis connector is finished within the given timeout.
        Returns False if the timeout is exceeded

        This check does not guarantee that current indexings being terminated
        won't get restarted midflight
        
        等待索引终止操作完成
        
        如果给定Redis连接器的所有索引在指定超时时间内完成，则返回True
        如果超过超时时间，则返回False
        
        注意：此检查不能保证当前终止的索引不会在执行过程中重新启动
        
        参数:
            search_settings_list: 搜索设置列表
            timeout: 超时时间（秒）
        返回:
            bool: 是否在超时前完成
        """

        finished = False

        start = time.monotonic()

        while True:
            still_indexing = False
            for search_settings in search_settings_list:
                redis_connector_index = self.new_index(search_settings.id)
                if redis_connector_index.fenced:
                    still_indexing = True
                    break

            if not still_indexing:
                finished = True
                break

            now = time.monotonic()
            if now - start > timeout:
                break

            time.sleep(1)
            continue

        return finished

    @staticmethod
    def get_id_from_fence_key(key: str) -> str | None:
        """Extracts the object ID from a fence key in the format `PREFIX_fence_X`.
        
        从格式为`PREFIX_fence_X`的fence key中提取对象ID
        
        参数:
            key: fence key字符串
        返回:
            str | None: 提取的ID，如果key格式不正确则返回None
        """
        parts = key.split("_")
        if len(parts) != 3:
            return None

        object_id = parts[2]
        return object_id

    @staticmethod
    def get_id_from_task_id(task_id: str) -> str | None:
        """Extracts the object ID from a task ID string.

        This method assumes the task ID is formatted as `prefix_objectid_suffix`, where:
        - `prefix` is an arbitrary string (e.g., the name of the task or entity),
        - `objectid` is the ID you want to extract,
        - `suffix` is another arbitrary string (e.g., a UUID).
        
        从任务ID字符串中提取对象ID。
        
        此方法假设任务ID的格式为`prefix_objectid_suffix`，其中：
        - `prefix`是任意字符串（例如任务或实体的名称）
        - `objectid`是要提取的ID
        - `suffix`是另一个任意字符串（例如UUID）
        
        参数:
            task_id: 任务ID字符串
        返回:
            str | None: 提取的对象ID，如果任务ID格式不正确则返回None
        """
        # example: task_id=documentset_1_cbfdc96a-80ca-4312-a242-0bb68da3c1dc
        parts = task_id.split("_")
        if len(parts) != 3:
            return None

        object_id = parts[1]
        return object_id
