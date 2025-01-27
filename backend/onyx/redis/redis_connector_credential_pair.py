"""
此文件用于管理连接器凭证对的 Redis 相关操作。
主要功能包括：
1. 扫描数据库中需要同步的文档
2. 将文档收集到统一的集合中进行同步
3. 管理文档同步状态和任务调度
"""

import time
from typing import cast
from uuid import uuid4

from celery import Celery
from redis import Redis
from redis.lock import Lock as RedisLock
from sqlalchemy.orm import Session

from onyx.configs.constants import CELERY_VESPA_SYNC_BEAT_LOCK_TIMEOUT
from onyx.configs.constants import OnyxCeleryPriority
from onyx.configs.constants import OnyxCeleryQueues
from onyx.configs.constants import OnyxCeleryTask
from onyx.db.connector_credential_pair import get_connector_credential_pair_from_id
from onyx.db.document import (
    construct_document_select_for_connector_credential_pair_by_needs_sync,
)
from onyx.db.models import Document
from onyx.redis.redis_object_helper import RedisObjectHelper


class RedisConnectorCredentialPair(RedisObjectHelper):
    """This class is used to scan documents by cc_pair in the db and collect them into
    a unified set for syncing.

    It differs from the other redis helpers in that the taskset used spans
    all connectors and is not per connector."""
    """
    此类用于通过 cc_pair 扫描数据库中的文档，并将它们收集到统一的集合中进行同步。
    
    与其他 redis 辅助类的不同之处在于，使用的任务集跨越所有连接器，而不是每个连接器单独使用。
    """

    # Redis 键前缀定义
    PREFIX = "connectorsync"
    FENCE_PREFIX = PREFIX + "_fence"
    TASKSET_PREFIX = PREFIX + "_taskset"
    SYNCING_PREFIX = PREFIX + ":vespa_syncing"

    def __init__(self, tenant_id: str | None, id: int) -> None:
        """
        初始化方法
        
        参数:
            tenant_id: 租户ID
            id: 连接器凭证对ID
        """
        super().__init__(tenant_id, str(id))
        # 需要跳过的文档集合
        self.skip_docs: set[str] = set()

    @classmethod
    def get_fence_key(cls) -> str:
        """
        获取fence键名
        
        返回值:
            str: fence键名
        """
        return RedisConnectorCredentialPair.FENCE_PREFIX

    @classmethod
    def get_taskset_key(cls) -> str:
        """
        获取任务集键名
        
        返回值:
            str: 任务集键名
        """
        return RedisConnectorCredentialPair.TASKSET_PREFIX

    @property
    def taskset_key(self) -> str:
        """Notice that this is intentionally reusing the same taskset for all
        connector syncs"""
        """
        获取任务集键名
        注意：这里故意为所有连接器同步重用相同的任务集
        
        返回值:
            str: 任务集键名
        """
        return f"{self.TASKSET_PREFIX}"

    def set_skip_docs(self, skip_docs: set[str]) -> None:
        """
        设置需要跳过的文档列表
        
        参数:
            skip_docs: 需要跳过的文档ID集合
        """
        self.skip_docs = skip_docs

    @staticmethod
    def make_redis_syncing_key(doc_id: str) -> str:
        """
        生成文档同步状态的Redis键名
        
        参数:
            doc_id: 文档ID
            
        返回值:
            str: Redis键名
        """
        return f"{RedisConnectorCredentialPair.SYNCING_PREFIX}:{doc_id}"

    def generate_tasks(
        self,
        celery_app: Celery,
        db_session: Session,
        redis_client: Redis,
        lock: RedisLock,
        tenant_id: str | None,
    ) -> tuple[int, int] | None:
        """
        生成同步任务
        
        参数:
            celery_app: Celery应用实例
            db_session: 数据库会话
            redis_client: Redis客户端
            lock: Redis锁
            tenant_id: 租户ID
            
        返回值:
            tuple[int, int] | None: 返回(已创建的任务数, 总文档数)的元组，如果失败返回None
        """
        # 设置同步过期时间（24小时），防止同一文档重复同步
        SYNC_EXPIRATION = 24 * 60 * 60

        last_lock_time = time.monotonic()

        async_results = []
        cc_pair = get_connector_credential_pair_from_id(int(self._id), db_session)
        if not cc_pair:
            return None

        # 构建需要同步的文档查询语句
        stmt = construct_document_select_for_connector_credential_pair_by_needs_sync(
            cc_pair.connector_id, cc_pair.credential_id
        )

        num_docs = 0

        # 遍历需要同步的文档
        for doc in db_session.scalars(stmt).yield_per(1):
            doc = cast(Document, doc)
            current_time = time.monotonic()
            
            # 定期重新获取锁，防止锁超时
            if current_time - last_lock_time >= (
                CELERY_VESPA_SYNC_BEAT_LOCK_TIMEOUT / 4
            ):
                lock.reacquire()
                last_lock_time = current_time

            num_docs += 1

            # 检查是否需要跳过该文档
            if doc.id in self.skip_docs:
                continue

            redis_syncing_key = self.make_redis_syncing_key(doc.id)
            if redis_client.exists(redis_syncing_key):
                continue

            # 生成自定义任务ID
            custom_task_id = f"{self.task_id_prefix}_{uuid4()}"

            # 将任务添加到Redis任务集
            redis_client.sadd(
                RedisConnectorCredentialPair.get_taskset_key(), custom_task_id
            )

            # 在Redis中记录文档同步状态
            redis_client.set(redis_syncing_key, custom_task_id, ex=SYNC_EXPIRATION)

            # 创建Celery同步任务
            result = celery_app.send_task(
                OnyxCeleryTask.VESPA_METADATA_SYNC_TASK,
                kwargs=dict(document_id=doc.id, tenant_id=tenant_id),
                queue=OnyxCeleryQueues.VESPA_METADATA_SYNC,
                task_id=custom_task_id,
                priority=OnyxCeleryPriority.MEDIUM,
            )

            async_results.append(result)
            self.skip_docs.add(doc.id)

        return len(async_results), num_docs
