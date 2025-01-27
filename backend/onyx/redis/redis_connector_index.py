"""
此文件用于管理Redis中索引任务相关的操作。
主要功能包括:
1. 管理索引任务的状态和进度
2. 处理任务锁定和解锁
3. 提供任务终止和活动状态控制
4. 管理生成器任务的各种状态
"""

from datetime import datetime
from typing import cast
from uuid import uuid4

import redis
from pydantic import BaseModel


class RedisConnectorIndexPayload(BaseModel):
    """Redis连接器索引任务的数据载体类"""
    index_attempt_id: int | None  # 索引尝试ID
    started: datetime | None      # 开始时间
    submitted: datetime          # 提交时间
    celery_task_id: str | None   # Celery任务ID


class RedisConnectorIndex:
    """Manages interactions with redis for indexing tasks. Should only be accessed
    through RedisConnector.
    管理Redis中索引任务的交互。只能通过RedisConnector访问。
    """

    # Redis键前缀定义
    PREFIX = "connectorindexing"
    FENCE_PREFIX = f"{PREFIX}_fence"  
    GENERATOR_TASK_PREFIX = PREFIX + "+generator"  
    GENERATOR_PROGRESS_PREFIX = PREFIX + "_generator_progress"
    GENERATOR_COMPLETE_PREFIX = PREFIX + "_generator_complete"
    GENERATOR_LOCK_PREFIX = "da_lock:indexing"
    TERMINATE_PREFIX = PREFIX + "_terminate"
    ACTIVE_PREFIX = PREFIX + "_active"

    def __init__(
        self,
        tenant_id: str | None,
        id: int,
        search_settings_id: int,
        redis: redis.Redis,
    ) -> None:
        """
        初始化Redis连接器索引实例
        
        参数:
            tenant_id: 租户ID
            id: 连接器ID
            search_settings_id: 搜索设置ID
            redis: Redis客户端实例
        """
        self.tenant_id: str | None = tenant_id
        self.id = id
        self.search_settings_id = search_settings_id
        self.redis = redis

        # 初始化各种Redis键
        self.fence_key: str = f"{self.FENCE_PREFIX}_{id}/{search_settings_id}"
        self.generator_progress_key = (
            f"{self.GENERATOR_PROGRESS_PREFIX}_{id}/{search_settings_id}"
        )
        self.generator_complete_key = (
            f"{self.GENERATOR_COMPLETE_PREFIX}_{id}/{search_settings_id}"
        )
        self.generator_lock_key = (
            f"{self.GENERATOR_LOCK_PREFIX}_{id}/{search_settings_id}"
        )
        self.terminate_key = f"{self.TERMINATE_PREFIX}_{id}/{search_settings_id}"
        self.active_key = f"{self.ACTIVE_PREFIX}_{id}/{search_settings_id}"

    @classmethod
    def fence_key_with_ids(cls, cc_pair_id: int, search_settings_id: int) -> str:
        """
        生成fence键
        
        参数:
            cc_pair_id: 连接器对ID
            search_settings_id: 搜索设置ID
        返回:
            生成的fence键字符串
        """
        return f"{cls.FENCE_PREFIX}_{cc_pair_id}/{search_settings_id}"

    def generate_generator_task_id(self) -> str:
        """
        生成生成器任务ID
        
        返回:
            生成的任务ID字符串
        """
        return f"{self.GENERATOR_TASK_PREFIX}_{self.id}/{self.search_settings_id}_{uuid4()}"

    @property
    def fenced(self) -> bool:
        """
        检查是否存在fence
        
        返回:
            如果存在fence返回True，否则返回False
        """
        if self.redis.exists(self.fence_key):
            return True
        return False

    @property
    def payload(self) -> RedisConnectorIndexPayload | None:
        """
        获取fence的数据载体
        
        返回:
            如果存在则返回RedisConnectorIndexPayload对象，否则返回None
        """
        fence_bytes = cast(bytes, self.redis.get(self.fence_key))
        if fence_bytes is None:
            return None

        fence_str = fence_bytes.decode("utf-8")
        payload = RedisConnectorIndexPayload.model_validate_json(cast(str, fence_str))
        return payload

    def set_fence(
        self,
        payload: RedisConnectorIndexPayload | None,
    ) -> None:
        """
        设置fence的数据载体
        
        参数:
            payload: RedisConnectorIndexPayload对象或None
        """
        if not payload:
            self.redis.delete(self.fence_key)
            return

        self.redis.set(self.fence_key, payload.model_dump_json())

    def terminating(self, celery_task_id: str) -> bool:
        """
        检查是否存在终止信号
        
        参数:
            celery_task_id: Celery任务ID
        返回:
            如果存在终止信号返回True，否则返回False
        """
        if self.redis.exists(f"{self.terminate_key}_{celery_task_id}"):
            return True
        return False

    def set_terminate(self, celery_task_id: str) -> None:
        """
        设置终止信号
        
        参数:
            celery_task_id: Celery任务ID
        """
        self.redis.set(f"{self.terminate_key}_{celery_task_id}", 0, ex=600)

    def set_active(self) -> None:
        """
        设置活动信号，防止索引流程在过期时间内被清理
        """
        self.redis.set(self.active_key, 0, ex=3600)

    def active(self) -> bool:
        """
        检查是否存在活动信号
        
        返回:
            如果存在活动信号返回True，否则返回False
        """
        if self.redis.exists(self.active_key):
            return True
        return False

    def generator_locked(self) -> bool:
        """
        检查生成器是否被锁定
        
        返回:
            如果生成器被锁定返回True，否则返回False
        """
        if self.redis.exists(self.generator_lock_key):
            return True
        return False

    def set_generator_complete(self, payload: int | None) -> None:
        """
        设置生成器完成状态
        
        参数:
            payload: 完成状态的整数值或None
        """
        if not payload:
            self.redis.delete(self.generator_complete_key)
            return

        self.redis.set(self.generator_complete_key, payload)

    def generator_clear(self) -> None:
        """
        清除生成器的进度和完成状态
        """
        self.redis.delete(self.generator_progress_key)
        self.redis.delete(self.generator_complete_key)

    def get_progress(self) -> int | None:
        """
        获取生成器的进度
        
        返回:
            进度的整数值或None
        """
        bytes = self.redis.get(self.generator_progress_key)
        if bytes is None:
            return None

        progress = int(cast(int, bytes))
        return progress

    def get_completion(self) -> int | None:
        """
        获取生成器的完成状态
        
        返回:
            完成状态的整数值或None
        """
        bytes = self.redis.get(self.generator_complete_key)
        if bytes is None:
            return None

        status = int(cast(int, bytes))
        return status

    def reset(self) -> None:
        """
        重置所有相关的Redis键
        """
        self.redis.delete(self.active_key)
        self.redis.delete(self.generator_lock_key)
        self.redis.delete(self.generator_progress_key)
        self.redis.delete(self.generator_complete_key)
        self.redis.delete(self.fence_key)

    @staticmethod
    def reset_all(r: redis.Redis) -> None:
        """
        删除所有连接器的Redis值
        
        参数:
            r: Redis客户端实例
        """
        for key in r.scan_iter(RedisConnectorIndex.ACTIVE_PREFIX + "*"):
            r.delete(key)

        for key in r.scan_iter(RedisConnectorIndex.GENERATOR_LOCK_PREFIX + "*"):
            r.delete(key)

        for key in r.scan_iter(RedisConnectorIndex.GENERATOR_COMPLETE_PREFIX + "*"):
            r.delete(key)

        for key in r.scan_iter(RedisConnectorIndex.GENERATOR_PROGRESS_PREFIX + "*"):
            r.delete(key)

        for key in r.scan_iter(RedisConnectorIndex.FENCE_PREFIX + "*"):
            r.delete(key)
