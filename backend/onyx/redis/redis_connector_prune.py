"""
该文件主要用于管理与Redis的连接器清理任务相关的交互操作。
主要功能包括：
1. 管理连接器清理任务的状态
2. 处理任务生成和进度跟踪
3. 维护任务集合和子任务
"""

import time
from typing import cast
from uuid import uuid4

import redis
from celery import Celery
from redis.lock import Lock as RedisLock
from sqlalchemy.orm import Session

from onyx.configs.constants import CELERY_VESPA_SYNC_BEAT_LOCK_TIMEOUT
from onyx.configs.constants import OnyxCeleryPriority
from onyx.configs.constants import OnyxCeleryQueues
from onyx.configs.constants import OnyxCeleryTask
from onyx.db.connector_credential_pair import get_connector_credential_pair_from_id


class RedisConnectorPrune:
    """Manages interactions with redis for pruning tasks. Should only be accessed
    through RedisConnector.
    管理与Redis清理任务相关的交互操作。只能通过RedisConnector访问。
    """

    PREFIX = "connectorpruning"

    FENCE_PREFIX = f"{PREFIX}_fence"

    # phase 1 - geneartor task and progress signals
    GENERATORTASK_PREFIX = f"{PREFIX}+generator"  # connectorpruning+generator
    GENERATOR_PROGRESS_PREFIX = (
        PREFIX + "_generator_progress"
    )  # connectorpruning_generator_progress
    GENERATOR_COMPLETE_PREFIX = (
        PREFIX + "_generator_complete"
    )  # connectorpruning_generator_complete

    TASKSET_PREFIX = f"{PREFIX}_taskset"  # connectorpruning_taskset
    SUBTASK_PREFIX = f"{PREFIX}+sub"  # connectorpruning+sub

    def __init__(self, tenant_id: str | None, id: int, redis: redis.Redis) -> None:
        """
        初始化RedisConnectorPrune实例
        
        Args:
            tenant_id: 租户ID
            id: 连接器ID
            redis: Redis连接实例
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
        清除生成器的进度和完成状态
        """
        self.redis.delete(self.generator_progress_key)
        self.redis.delete(self.generator_complete_key)

    def get_remaining(self) -> int:
        """
        获取剩余待处理任务数量
        
        Returns:
            int: 剩余任务数量
        """
        remaining = cast(int, self.redis.scard(self.taskset_key))
        return remaining

    def get_active_task_count(self) -> int:
        """Count of active pruning tasks
        获取活跃清理任务的数量
        
        Returns:
            int: 活跃任务数量
        """
        count = 0
        for key in self.redis.scan_iter(RedisConnectorPrune.FENCE_PREFIX + "*"):
            count += 1
        return count

    @property
    def fenced(self) -> bool:
        """
        检查是否有fence标记
        
        Returns:
            bool: 是否存在fence标记
        """
        if self.redis.exists(self.fence_key):
            return True

        return False

    def set_fence(self, value: bool) -> None:
        """
        设置fence标记
        
        Args:
            value: 是否设置fence标记
        """
        if not value:
            self.redis.delete(self.fence_key)
            return

        self.redis.set(self.fence_key, 0)

    @property
    def generator_complete(self) -> int | None:
        """the fence payload is an int representing the starting number of
        pruning tasks to be processed ... just after the generator completes.
        fence负载是一个整数，表示生成器完成后要处理的清理任务的起始数量。
        
        Returns:
            int | None: 任务数量或None
        """
        fence_bytes = self.redis.get(self.generator_complete_key)
        if fence_bytes is None:
            return None

        fence_int = cast(int, fence_bytes)
        return fence_int

    @generator_complete.setter
    def generator_complete(self, payload: int | None) -> None:
        """Set the payload to an int to set the fence, otherwise if None it will
        be deleted
        设置fence负载为整数，如果为None则删除fence
        
        Args:
            payload: 要设置的负载值
        """
        if payload is None:
            self.redis.delete(self.generator_complete_key)
            return

        self.redis.set(self.generator_complete_key, payload)

    def generate_tasks(
        self,
        documents_to_prune: set[str],
        celery_app: Celery,
        db_session: Session,
        lock: RedisLock | None,
    ) -> int | None:
        """
        生成清理任务
        
        Args:
            documents_to_prune: 需要清理的文档ID集合
            celery_app: Celery应用实例
            db_session: 数据库会话
            lock: Redis锁实例
            
        Returns:
            int | None: 生成的任务数量或None
        """
        last_lock_time = time.monotonic()

        async_results = []
        cc_pair = get_connector_credential_pair_from_id(int(self.id), db_session)
        if not cc_pair:
            return None

        for doc_id in documents_to_prune:
            current_time = time.monotonic()
            if lock and current_time - last_lock_time >= (
                CELERY_VESPA_SYNC_BEAT_LOCK_TIMEOUT / 4
            ):
                lock.reacquire()
                last_lock_time = current_time

            # celery's default task id format is "dd32ded3-00aa-4884-8b21-42f8332e7fac"
            # the actual redis key is "celery-task-meta-dd32ded3-00aa-4884-8b21-42f8332e7fac"
            # we prefix the task id so it's easier to keep track of who created the task
            # aka "documentset_1_6dd32ded3-00aa-4884-8b21-42f8332e7fac"
            custom_task_id = f"{self.subtask_prefix}_{uuid4()}"

            # add to the tracking taskset in redis BEFORE creating the celery task.
            self.redis.sadd(self.taskset_key, custom_task_id)

            # Priority on sync's triggered by new indexing should be medium
            result = celery_app.send_task(
                OnyxCeleryTask.DOCUMENT_BY_CC_PAIR_CLEANUP_TASK,
                kwargs=dict(
                    document_id=doc_id,
                    connector_id=cc_pair.connector_id,
                    credential_id=cc_pair.credential_id,
                    tenant_id=self.tenant_id,
                ),
                queue=OnyxCeleryQueues.CONNECTOR_DELETION,
                task_id=custom_task_id,
                priority=OnyxCeleryPriority.MEDIUM,
            )

            async_results.append(result)

        return len(async_results)

    def reset(self) -> None:
        """
        重置所有状态，清除所有相关的Redis键
        """
        self.redis.delete(self.generator_progress_key)
        self.redis.delete(self.generator_complete_key)
        self.redis.delete(self.taskset_key)
        self.redis.delete(self.fence_key)

    @staticmethod
    def remove_from_taskset(id: int, task_id: str, r: redis.Redis) -> None:
        """
        从任务集合中移除指定任务
        
        Args:
            id: 连接器ID
            task_id: 任务ID
            r: Redis连接实例
        """
        taskset_key = f"{RedisConnectorPrune.TASKSET_PREFIX}_{id}"
        r.srem(taskset_key, task_id)
        return

    @staticmethod
    def reset_all(r: redis.Redis) -> None:
        """Deletes all redis values for all connectors
        删除所有连接器的所有Redis值
        
        Args:
            r: Redis连接实例
        """
        for key in r.scan_iter(RedisConnectorPrune.TASKSET_PREFIX + "*"):
            r.delete(key)

        for key in r.scan_iter(RedisConnectorPrune.GENERATOR_COMPLETE_PREFIX + "*"):
            r.delete(key)

        for key in r.scan_iter(RedisConnectorPrune.GENERATOR_PROGRESS_PREFIX + "*"):
            r.delete(key)

        for key in r.scan_iter(RedisConnectorPrune.FENCE_PREFIX + "*"):
            r.delete(key)
