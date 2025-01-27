"""
此文件实现了 RedisDocumentSet 类，用于管理文档集在 Redis 中的状态。
主要功能包括：
- 文档集的围栏(fence)管理
- 任务集(taskset)的生成和管理
- Redis 键值操作封装
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
from onyx.db.document_set import construct_document_select_by_docset
from onyx.db.models import Document
from onyx.redis.redis_object_helper import RedisObjectHelper


class RedisDocumentSet(RedisObjectHelper):
    """
    Redis文档集管理类
    用于处理文档集在Redis中的状态管理，包括围栏状态和任务集管理
    
    属性：
        PREFIX: 键前缀
        FENCE_PREFIX: 围栏键前缀
        TASKSET_PREFIX: 任务集键前缀
    """
    PREFIX = "documentset"
    FENCE_PREFIX = PREFIX + "_fence"
    TASKSET_PREFIX = PREFIX + "_taskset"

    def __init__(self, tenant_id: str | None, id: int) -> None:
        """
        初始化 RedisDocumentSet 实例
        
        参数：
            tenant_id: 租户ID
            id: 文档集ID
        """
        super().__init__(tenant_id, str(id))

    @property
    def fenced(self) -> bool:
        """
        检查当前文档集是否被围栏保护
        
        返回值：
            bool: 如果存在围栏则返回True，否则返回False
        """
        if self.redis.exists(self.fence_key):
            return True
        return False

    def set_fence(self, payload: int | None) -> None:
        """
        设置或清除文档集的围栏状态
        
        参数：
            payload: 围栏值，如果为None则清除围栏
        """
        if payload is None:
            self.redis.delete(self.fence_key)
            return
        self.redis.set(self.fence_key, payload)

    @property
    def payload(self) -> int | None:
        """
        获取围栏的载荷值
        
        返回值：
            int|None: 返回围栏值，如果不存在则返回None
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
        为文档集生成同步任务
        
        参数：
            celery_app: Celery应用实例
            db_session: 数据库会话
            redis_client: Redis客户端
            lock: Redis分布式锁
            tenant_id: 租户ID
        
        返回值：
            tuple[int, int]|None: 返回(已处理任务数, 总任务数)的元组
        """
        last_lock_time = time.monotonic()

        async_results = []
        stmt = construct_document_select_by_docset(int(self._id), current_only=False)
        for doc in db_session.scalars(stmt).yield_per(1):
            doc = cast(Document, doc)
            current_time = time.monotonic()
            if current_time - last_lock_time >= (
                CELERY_VESPA_SYNC_BEAT_LOCK_TIMEOUT / 4
            ):
                lock.reacquire()
                last_lock_time = current_time

            # celery默认任务ID格式为: "dd32ded3-00aa-4884-8b21-42f8332e7fac"
            # 结果的键为: "celery-task-meta-dd32ded3-00aa-4884-8b21-42f8332e7fac"
            # 我们添加前缀以便更容易追踪任务创建者
            # 例如: "documentset_1_6dd32ded3-00aa-4884-8b21-42f8332e7fac"
            custom_task_id = f"{self.task_id_prefix}_{uuid4()}"

            # 在创建任务前将任务ID添加到集合中
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
        重置文档集状态
        清除任务集和围栏的所有数据
        """
        self.redis.delete(self.taskset_key)
        self.redis.delete(self.fence_key)

    @staticmethod
    def reset_all(r: redis.Redis) -> None:
        """
        重置所有文档集的状态
        
        参数：
            r: Redis客户端实例
        """
        for key in r.scan_iter(RedisDocumentSet.TASKSET_PREFIX + "*"):
            r.delete(key)

        for key in r.scan_iter(RedisDocumentSet.FENCE_PREFIX + "*"):
            r.delete(key)
