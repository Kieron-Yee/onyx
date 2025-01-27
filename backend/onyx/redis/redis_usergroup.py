"""
用户组 Redis 操作相关功能模块
本模块提供了对用户组在 Redis 中进行操作的相关功能，包括用户组状态管理、任务生成等功能
"""

import time
from typing import cast
from uuid import uuid4

import redis
from celery import Celery
from redis import Redis
from redis.lock import Lock as RedisLock
from sqlalchemy.orm import Session

from onyx.configs.constants import CELERY_VESPA_SYNC_BEAT_LOCK_TIMEOUT
from onyx.configs.constants import OnyxCeleryPriority
from onyx.configs.constants import OnyxCeleryQueues
from onyx.configs.constants import OnyxCeleryTask
from onyx.db.models import Document
from onyx.redis.redis_object_helper import RedisObjectHelper
from onyx.utils.variable_functionality import fetch_versioned_implementation
from onyx.utils.variable_functionality import global_version


class RedisUserGroup(RedisObjectHelper):
    """
    Redis用户组类
    提供用户组相关的Redis操作功能，包括围栏状态管理和任务集合操作
    """
    
    PREFIX = "usergroup"
    FENCE_PREFIX = PREFIX + "_fence"
    TASKSET_PREFIX = PREFIX + "_taskset"

    def __init__(self, tenant_id: str | None, id: int) -> None:
        """
        初始化Redis用户组实例
        
        参数:
            tenant_id: 租户ID
            id: 用户组ID
        """
        super().__init__(tenant_id, str(id))

    @property
    def fenced(self) -> bool:
        """
        检查用户组是否被围栏
        
        返回值:
            bool: 如果用户组被围栏返回True，否则返回False
        """
        if self.redis.exists(self.fence_key):
            return True

        return False

    def set_fence(self, payload: int | None) -> None:
        """
        设置用户组的围栏状态
        
        参数:
            payload: 围栏状态值，如果为None则删除围栏
        """
        if payload is None:
            self.redis.delete(self.fence_key)
            return

        self.redis.set(self.fence_key, payload)

    @property
    def payload(self) -> int | None:
        """
        获取围栏状态值
        
        返回值:
            int|None: 围栏状态值，如果不存在则返回None
        """
        bytes = self.redis.get(self.fence_key)
        if bytes is None:
            return None

        progress = int(cast(int, bytes))
        return progress

    def generate_tasks(
        self,
        celery_app: Celery,
        db_session: Session,
        redis_client: Redis,
        lock: RedisLock,
        tenant_id: str | None,
    ) -> tuple[int, int] | None:
        """
        为用户组生成同步任务
        
        参数:
            celery_app: Celery应用实例
            db_session: 数据库会话
            redis_client: Redis客户端实例
            lock: Redis锁实例
            tenant_id: 租户ID

        返回值:
            tuple[int, int]|None: 返回创建的任务数量元组，如果不是企业版则返回(0,0)
        """
        last_lock_time = time.monotonic()

        async_results = []

        if not global_version.is_ee_version():
            return 0, 0

        try:
            construct_document_select_by_usergroup = fetch_versioned_implementation(
                "onyx.db.user_group",
                "construct_document_select_by_usergroup",
            )
        except ModuleNotFoundError:
            return 0, 0

        stmt = construct_document_select_by_usergroup(int(self._id))
        for doc in db_session.scalars(stmt).yield_per(1):
            doc = cast(Document, doc)
            current_time = time.monotonic()
            if current_time - last_lock_time >= (
                CELERY_VESPA_SYNC_BEAT_LOCK_TIMEOUT / 4
            ):
                lock.reacquire()
                last_lock_time = current_time

            # celery's default task id format is "dd32ded3-00aa-4884-8b21-42f8332e7fac"
            # the key for the result is "celery-task-meta-dd32ded3-00aa-4884-8b21-42f8332e7fac"
            # we prefix the task id so it's easier to keep track of who created the task
            # aka "documentset_1_6dd32ded3-00aa-4884-8b21-42f8332e7fac"
            custom_task_id = f"{self.task_id_prefix}_{uuid4()}"

            # add to the set BEFORE creating the task.
            redis_client.sadd(self.taskset_key, custom_task_id)

            result = celery_app.send_task(
                OnyxCeleryTask.VESPA_METADATA_SYNC_TASK,
                kwargs=dict(document_id=doc.id, tenant_id=tenant_id),
                queue=OnyxCeleryQueues.VESPA_METADATA_SYNC,
                task_id=custom_task_id,
                priority=OnyxCeleryPriority.LOW,
            )

            async_results.append(result)

        return len(async_results), len(async_results)

    def reset(self) -> None:
        """
        重置用户组状态
        清除任务集合和围栏状态
        """
        self.redis.delete(self.taskset_key)
        self.redis.delete(self.fence_key)

    @staticmethod
    def reset_all(r: redis.Redis) -> None:
        """
        重置所有用户组状态
        
        参数:
            r: Redis客户端实例
        """
        for key in r.scan_iter(RedisUserGroup.TASKSET_PREFIX + "*"):
            r.delete(key)

        for key in r.scan_iter(RedisUserGroup.FENCE_PREFIX + "*"):
            r.delete(key)
