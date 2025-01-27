"""
此文件用于管理外部组同步任务的Redis交互。
主要功能包括：
- 管理外部组同步任务的状态
- 处理任务生成和进度信息
- 维护任务集合和子任务
"""

from datetime import datetime
from typing import cast

import redis
from celery import Celery
from pydantic import BaseModel
from redis.lock import Lock as RedisLock
from sqlalchemy.orm import Session


class RedisConnectorExternalGroupSyncPayload(BaseModel):
    """
    外部组同步任务的载荷数据模型
    
    属性:
        started: 任务开始时间
        celery_task_id: Celery任务ID
    """
    started: datetime | None
    celery_task_id: str | None


class RedisConnectorExternalGroupSync:
    """Manages interactions with redis for external group syncing tasks. Should only be accessed
    through RedisConnector.
    通过RedisConnector管理与Redis的外部组同步任务交互。
    """

    PREFIX = "connectorexternalgroupsync"

    FENCE_PREFIX = f"{PREFIX}_fence"

    # phase 1 - geneartor task and progress signals
    # 阶段1 - 生成器任务和进度信号
    GENERATORTASK_PREFIX = f"{PREFIX}+generator"  # connectorexternalgroupsync+generator
    GENERATOR_PROGRESS_PREFIX = (
        PREFIX + "_generator_progress"
    )  # connectorexternalgroupsync_generator_progress
    GENERATOR_COMPLETE_PREFIX = (
        PREFIX + "_generator_complete"
    )  # connectorexternalgroupsync_generator_complete

    TASKSET_PREFIX = f"{PREFIX}_taskset"  # connectorexternalgroupsync_taskset
    SUBTASK_PREFIX = f"{PREFIX}+sub"  # connectorexternalgroupsync+sub

    def __init__(self, tenant_id: str | None, id: int, redis: redis.Redis) -> None:
        """
        初始化Redis连接器实例
        
        参数:
            tenant_id: 租户ID
            id: 连接器ID
            redis: Redis客户端实例
        """
        self.tenant_id: str | None = tenant_id
        self.id = id
        self.redis = redis

        self.fence_key: str = f"{self.FENCE_PREFIX}_{id}"
        self.generator_task_key = f"{self.GENERATORTASK_PREFIX}_{id}"
        self.generator_progress_key = f"{self.GENERATOR_PROGRESS_PREFIX}_{id}"
        self.generator_complete_key = f"{self.GENERATOR_COMPLETE_PREFIX}_{id}"

        self.taskset_key = f"{self.TASKSET_PREFIX}_{id}"

        self.subtask_prefix: str = f"{self.SUBTASK_PREFIX}_{id}"

    def taskset_clear(self) -> None:
        """
        清除任务集合
        """
        self.redis.delete(self.taskset_key)

    def generator_clear(self) -> None:
        """
        清除生成器相关的进度和完成状态
        """
        self.redis.delete(self.generator_progress_key)
        self.redis.delete(self.generator_complete_key)

    def get_remaining(self) -> int:
        """
        获取剩余待处理任务数量
        
        返回值:
            int: 剩余任务数量
        """
        # todo: move into fence
        # 待办：移至fence中
        remaining = cast(int, self.redis.scard(self.taskset_key))
        return remaining

    def get_active_task_count(self) -> int:
        """Count of active external group syncing tasks
        获取活跃的外部组同步任务数量
        
        返回值:
            int: 活跃任务数量
        """
        count = 0
        for _ in self.redis.scan_iter(
            RedisConnectorExternalGroupSync.FENCE_PREFIX + "*"
        ):
            count += 1
        return count

    @property
    def fenced(self) -> bool:
        """
        检查是否存在fence标记
        
        返回值:
            bool: 是否存在fence标记
        """
        if self.redis.exists(self.fence_key):
            return True

        return False

    @property
    def payload(self) -> RedisConnectorExternalGroupSyncPayload | None:
        """
        获取fence的载荷数据
        
        返回值:
            RedisConnectorExternalGroupSyncPayload | None: 载荷数据或None
        """
        # read related data and evaluate/print task progress
        # 读取相关数据并评估/打印任务进度
        fence_bytes = cast(bytes, self.redis.get(self.fence_key))
        if fence_bytes is None:
            return None

        fence_str = fence_bytes.decode("utf-8")
        payload = RedisConnectorExternalGroupSyncPayload.model_validate_json(
            cast(str, fence_str)
        )

        return payload

    def set_fence(
        self,
        payload: RedisConnectorExternalGroupSyncPayload | None,
    ) -> None:
        """
        设置fence标记及其载荷
        
        参数:
            payload: fence载荷数据
        """
        if not payload:
            self.redis.delete(self.fence_key)
            return

        self.redis.set(self.fence_key, payload.model_dump_json())

    @property
    def generator_complete(self) -> int | None:
        """the fence payload is an int representing the starting number of
        external group syncing tasks to be processed ... just after the generator completes.
        fence载荷是一个整数，表示生成器完成后要处理的外部组同步任务的初始数量。
        
        返回值:
            int | None: 任务数量或None
        """
        fence_bytes = self.redis.get(self.generator_complete_key)
        if fence_bytes is None:
            return None

        if fence_bytes == b"None":
            return None

        fence_int = int(cast(bytes, fence_bytes).decode())
        return fence_int

    @generator_complete.setter
    def generator_complete(self, payload: int | None) -> None:
        """Set the payload to an int to set the fence, otherwise if None it will
        be deleted
        设置fence载荷为整数，如果为None则删除fence
        
        参数:
            payload: 要设置的任务数量
        """
        if payload is None:
            self.redis.delete(self.generator_complete_key)
            return

        self.redis.set(self.generator_complete_key, payload)

    def generate_tasks(
        self,
        celery_app: Celery,
        db_session: Session,
        lock: RedisLock | None,
    ) -> int | None:
        """
        生成同步任务
        
        参数:
            celery_app: Celery应用实例
            db_session: 数据库会话
            lock: Redis锁对象
        
        返回值:
            int | None: 生成的任务数量或None
        """
        pass

    @staticmethod
    def remove_from_taskset(id: int, task_id: str, r: redis.Redis) -> None:
        """
        从任务集合中移除指定任务
        
        参数:
            id: 连接器ID
            task_id: 要移除的任务ID
            r: Redis客户端实例
        """
        taskset_key = f"{RedisConnectorExternalGroupSync.TASKSET_PREFIX}_{id}"
        r.srem(taskset_key, task_id)
        return

    @staticmethod
    def reset_all(r: redis.Redis) -> None:
        """Deletes all redis values for all connectors
        删除所有连接器的所有Redis值
        
        参数:
            r: Redis客户端实例
        """
        for key in r.scan_iter(RedisConnectorExternalGroupSync.TASKSET_PREFIX + "*"):
            r.delete(key)

        for key in r.scan_iter(
            RedisConnectorExternalGroupSync.GENERATOR_COMPLETE_PREFIX + "*"
        ):
            r.delete(key)

        for key in r.scan_iter(
            RedisConnectorExternalGroupSync.GENERATOR_PROGRESS_PREFIX + "*"
        ):
            r.delete(key)

        for key in r.scan_iter(RedisConnectorExternalGroupSync.FENCE_PREFIX + "*"):
            r.delete(key)
