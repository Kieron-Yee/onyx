"""
这个文件实现了文档权限同步任务与Redis之间的交互管理。
主要功能：
1. 管理文档权限同步的任务状态
2. 处理任务生成和进度跟踪
3. 提供任务集合的管理功能
"""

import time
from datetime import datetime
from typing import cast
from uuid import uuid4

import redis
from celery import Celery
from pydantic import BaseModel
from redis.lock import Lock as RedisLock

from onyx.access.models import DocExternalAccess
from onyx.configs.constants import CELERY_VESPA_SYNC_BEAT_LOCK_TIMEOUT
from onyx.configs.constants import OnyxCeleryPriority
from onyx.configs.constants import OnyxCeleryQueues
from onyx.configs.constants import OnyxCeleryTask


class RedisConnectorPermissionSyncPayload(BaseModel):
    """
    权限同步任务的负载数据模型
    
    属性：
        started: 任务开始时间
        celery_task_id: Celery任务ID
    """
    started: datetime | None
    celery_task_id: str | None


class RedisConnectorPermissionSync:
    """Manages interactions with redis for doc permission sync tasks. Should only be accessed
    through RedisConnector.
    通过RedisConnector管理文档权限同步任务与Redis之间的交互。
    """

    PREFIX = "connectordocpermissionsync"

    FENCE_PREFIX = f"{PREFIX}_fence"

    # phase 1 - geneartor task and progress signals
    GENERATORTASK_PREFIX = f"{PREFIX}+generator"  # connectorpermissions+generator
    GENERATOR_PROGRESS_PREFIX = (
        PREFIX + "_generator_progress"
    )  # connectorpermissions_generator_progress
    GENERATOR_COMPLETE_PREFIX = (
        PREFIX + "_generator_complete"
    )  # connectorpermissions_generator_complete

    TASKSET_PREFIX = f"{PREFIX}_taskset"  # connectorpermissions_taskset
    SUBTASK_PREFIX = f"{PREFIX}+sub"  # connectorpermissions+sub

    def __init__(self, tenant_id: str | None, id: int, redis: redis.Redis) -> None:
        """
        初始化Redis连接器权限同步实例
        
        参数：
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
        清除任务集合中的所有任务
        """
        self.redis.delete(self.taskset_key)

    def generator_clear(self) -> None:
        """
        清除生成器的进度和完成状态
        """
        self.redis.delete(self.generator_progress_key)
        self.redis.delete(self.generator_complete_key)

    def get_remaining(self) -> int:
        """
        获取剩余待处理任务数量
        
        返回值：
            int: 剩余任务数量
        """
        remaining = cast(int, self.redis.scard(self.taskset_key))
        return remaining

    def get_active_task_count(self) -> int:
        """Count of active permission sync tasks
        获取当前活动的权限同步任务数量
        
        返回值：
            int: 活动任务数量
        """
        count = 0
        for _ in self.redis.scan_iter(RedisConnectorPermissionSync.FENCE_PREFIX + "*"):
            count += 1
        return count

    @property
    def fenced(self) -> bool:
        """
        检查是否存在fence标记
        
        返回值：
            bool: 是否存在fence标记
        """
        if self.redis.exists(self.fence_key):
            return True

        return False

    @property
    def payload(self) -> RedisConnectorPermissionSyncPayload | None:
        """
        获取当前fence的负载数据
        
        返回值：
            RedisConnectorPermissionSyncPayload | None: 负载数据或None
        """
        # read related data and evaluate/print task progress
        fence_bytes = cast(bytes, self.redis.get(self.fence_key))
        if fence_bytes is None:
            return None

        fence_str = fence_bytes.decode("utf-8")
        payload = RedisConnectorPermissionSyncPayload.model_validate_json(
            cast(str, fence_str)
        )

        return payload

    def set_fence(
        self,
        payload: RedisConnectorPermissionSyncPayload | None,
    ) -> None:
        """
        设置fence的负载数据
        
        参数：
            payload: 要设置的负载数据
        """
        if not payload:
            self.redis.delete(self.fence_key)
            return

        self.redis.set(self.fence_key, payload.model_dump_json())

    @property
    def generator_complete(self) -> int | None:
        """the fence payload is an int representing the starting number of
        permission sync tasks to be processed ... just after the generator completes.
        fence负载是一个整数，表示生成器完成后要处理的权限同步任务的初始数量。
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
        设置fence负载为整数，如果为None则删除fence
        """
        if payload is None:
            self.redis.delete(self.generator_complete_key)
            return

        self.redis.set(self.generator_complete_key, payload)

    def generate_tasks(
        self,
        celery_app: Celery,
        lock: RedisLock | None,
        new_permissions: list[DocExternalAccess],
        source_string: str,
        connector_id: int,
        credential_id: int,
    ) -> int | None:
        """
        生成权限同步任务
        
        参数：
            celery_app: Celery应用实例
            lock: Redis锁对象
            new_permissions: 新的权限列表
            source_string: 来源标识
            connector_id: 连接器ID
            credential_id: 凭证ID
            
        返回值：
            int | None: 生成的任务数量
        """
        last_lock_time = time.monotonic()
        async_results = []

        # Create a task for each document permission sync
        for doc_perm in new_permissions:
            current_time = time.monotonic()
            if lock and current_time - last_lock_time >= (
                CELERY_VESPA_SYNC_BEAT_LOCK_TIMEOUT / 4
            ):
                lock.reacquire()
                last_lock_time = current_time
            # Add task for document permissions sync
            custom_task_id = f"{self.subtask_prefix}_{uuid4()}"
            self.redis.sadd(self.taskset_key, custom_task_id)

            result = celery_app.send_task(
                OnyxCeleryTask.UPDATE_EXTERNAL_DOCUMENT_PERMISSIONS_TASK,
                kwargs=dict(
                    tenant_id=self.tenant_id,
                    serialized_doc_external_access=doc_perm.to_dict(),
                    source_string=source_string,
                    connector_id=connector_id,
                    credential_id=credential_id,
                ),
                queue=OnyxCeleryQueues.DOC_PERMISSIONS_UPSERT,
                task_id=custom_task_id,
                priority=OnyxCeleryPriority.HIGH,
            )
            async_results.append(result)

        return len(async_results)

    def reset(self) -> None:
        """
        重置所有相关的Redis键值
        """
        self.redis.delete(self.generator_progress_key)
        self.redis.delete(self.generator_complete_key)
        self.redis.delete(self.taskset_key)
        self.redis.delete(self.fence_key)

    @staticmethod
    def remove_from_taskset(id: int, task_id: str, r: redis.Redis) -> None:
        """
        从任务集合中移除指定任务
        
        参数：
            id: 连接器ID
            task_id: 任务ID
            r: Redis客户端实例
        """
        taskset_key = f"{RedisConnectorPermissionSync.TASKSET_PREFIX}_{id}"
        r.srem(taskset_key, task_id)
        return

    @staticmethod
    def reset_all(r: redis.Redis) -> None:
        """Deletes all redis values for all connectors
        删除所有连接器的所有Redis值
        """
        for key in r.scan_iter(RedisConnectorPermissionSync.TASKSET_PREFIX + "*"):
            r.delete(key)

        for key in r.scan_iter(
            RedisConnectorPermissionSync.GENERATOR_COMPLETE_PREFIX + "*"
        ):
            r.delete(key)

        for key in r.scan_iter(
            RedisConnectorPermissionSync.GENERATOR_PROGRESS_PREFIX + "*"
        ):
            r.delete(key)

        for key in r.scan_iter(RedisConnectorPermissionSync.FENCE_PREFIX + "*"):
            r.delete(key)
