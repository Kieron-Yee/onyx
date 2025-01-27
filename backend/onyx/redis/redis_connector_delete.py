"""
此文件用于管理连接器删除任务与Redis的交互。
主要功能包括：
1. 管理删除任务的状态追踪
2. 生成和维护删除任务队列
3. 处理任务删除的fence机制
4. 提供任务进度查询和重置功能
"""

import time
from datetime import datetime
from typing import cast
from uuid import uuid4

import redis
from celery import Celery
from pydantic import BaseModel
from redis.lock import Lock as RedisLock
from sqlalchemy.orm import Session

from onyx.configs.constants import CELERY_VESPA_SYNC_BEAT_LOCK_TIMEOUT
from onyx.configs.constants import OnyxCeleryPriority
from onyx.configs.constants import OnyxCeleryQueues
from onyx.configs.constants import OnyxCeleryTask
from onyx.db.connector_credential_pair import get_connector_credential_pair_from_id
from onyx.db.document import construct_document_select_for_connector_credential_pair
from onyx.db.models import Document as DbDocument


class RedisConnectorDeletePayload(BaseModel):
    """
    删除任务的负载数据模型
    
    属性:
        num_tasks: 任务总数，可为空
        submitted: 任务提交时间
    """
    num_tasks: int | None
    submitted: datetime


class RedisConnectorDelete:
    """Manages interactions with redis for deletion tasks. Should only be accessed
    through RedisConnector.
    管理删除任务与Redis的交互。只能通过RedisConnector访问。
    """

    PREFIX = "connectordeletion"  # 删除任务前缀
    FENCE_PREFIX = f"{PREFIX}_fence"  # 栅栏前缀
    TASKSET_PREFIX = f"{PREFIX}_taskset"  # 任务集前缀

    def __init__(self, tenant_id: str | None, id: int, redis: redis.Redis) -> None:
        """
        初始化Redis连接器删除实例
        
        参数:
            tenant_id: 租户ID
            id: 连接器ID
            redis: Redis客户端实例
        """
        self.tenant_id: str | None = tenant_id
        self.id = id
        self.redis = redis

        self.fence_key: str = f"{self.FENCE_PREFIX}_{id}"
        self.taskset_key = f"{self.TASKSET_PREFIX}_{id}"

    def taskset_clear(self) -> None:
        """清除任务集合"""
        self.redis.delete(self.taskset_key)

    def get_remaining(self) -> int:
        """
        获取剩余任务数量
        
        返回:
            剩余任务的数量
        """
        remaining = cast(int, self.redis.scard(self.taskset_key))
        return remaining

    @property
    def fenced(self) -> bool:
        """
        检查是否存在栅栏
        
        返回:
            如果存在栅栏返回True，否则返回False
        """
        if self.redis.exists(self.fence_key):
            return True
        return False

    @property
    def payload(self) -> RedisConnectorDeletePayload | None:
        """
        获取任务负载数据
        
        返回:
            任务负载数据对象，如果不存在则返回None
        """
        fence_bytes = cast(bytes, self.redis.get(self.fence_key))
        if fence_bytes is None:
            return None

        fence_str = fence_bytes.decode("utf-8")
        payload = RedisConnectorDeletePayload.model_validate_json(cast(str, fence_str))
        return payload

    def set_fence(self, payload: RedisConnectorDeletePayload | None) -> None:
        if not payload:
            self.redis.delete(self.fence_key)
            return

        self.redis.set(self.fence_key, payload.model_dump_json())

    def _generate_task_id(self) -> str:
        # celery's default task id format is "dd32ded3-00aa-4884-8b21-42f8332e7fac"
        # we prefix the task id so it's easier to keep track of who created the task
        # aka "connectordeletion_1_6dd32ded3-00aa-4884-8b21-42f8332e7fac"

        return f"{self.PREFIX}_{self.id}_{uuid4()}"

    def generate_tasks(
        self,
        celery_app: Celery,
        db_session: Session,
        lock: RedisLock,
    ) -> int | None:
        """
        生成删除任务
        
        参数:
            celery_app: Celery应用实例
            db_session: 数据库会话
            lock: Redis锁实例
        
        返回:
            如果连接器凭证对不存在返回None，否则返回生成的任务数量
        """
        last_lock_time = time.monotonic()

        async_results = []
        cc_pair = get_connector_credential_pair_from_id(int(self.id), db_session)
        if not cc_pair:
            return None

        stmt = construct_document_select_for_connector_credential_pair(
            cc_pair.connector_id, cc_pair.credential_id
        )
        for doc_temp in db_session.scalars(stmt).yield_per(1):
            doc: DbDocument = doc_temp
            current_time = time.monotonic()
            if current_time - last_lock_time >= (
                CELERY_VESPA_SYNC_BEAT_LOCK_TIMEOUT / 4
            ):
                lock.reacquire()
                last_lock_time = current_time

            custom_task_id = self._generate_task_id()

            # add to the tracking taskset in redis BEFORE creating the celery task.
            # note that for the moment we are using a single taskset key, not differentiated by cc_pair id
            self.redis.sadd(self.taskset_key, custom_task_id)

            # Priority on sync's triggered by new indexing should be medium
            result = celery_app.send_task(
                OnyxCeleryTask.DOCUMENT_BY_CC_PAIR_CLEANUP_TASK,
                kwargs=dict(
                    document_id=doc.id,
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
        """重置所有相关的Redis数据"""
        self.redis.delete(self.taskset_key)
        self.redis.delete(self.fence_key)

    @staticmethod
    def remove_from_taskset(id: int, task_id: str, r: redis.Redis) -> None:
        """
        从任务集合中移除指定任务
        
        参数:
            id: 连接器ID
            task_id: 任务ID
            r: Redis客户端实例
        """
        taskset_key = f"{RedisConnectorDelete.TASKSET_PREFIX}_{id}"
        r.srem(taskset_key, task_id)
        return

    @staticmethod
    def reset_all(r: redis.Redis) -> None:
        """
        删除所有连接器的Redis数据
        
        参数:
            r: Redis客户端实例
        """
        for key in r.scan_iter(RedisConnectorDelete.TASKSET_PREFIX + "*"):
            r.delete(key)

        for key in r.scan_iter(RedisConnectorDelete.FENCE_PREFIX + "*"):
            r.delete(key)
