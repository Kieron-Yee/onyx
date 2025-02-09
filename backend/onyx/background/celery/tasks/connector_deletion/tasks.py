"""
此文件实现了连接器删除相关的后台任务处理功能。
主要包括检查连接器删除状态、生成删除任务等功能。
该模块是系统异步任务处理机制的重要组成部分。
"""

from datetime import datetime
from datetime import timezone

from celery import Celery
from celery import shared_task
from celery import Task
from celery.exceptions import SoftTimeLimitExceeded
from redis.lock import Lock as RedisLock
from sqlalchemy.orm import Session

from onyx.background.celery.apps.app_base import task_logger
from onyx.configs.app_configs import JOB_TIMEOUT
from onyx.configs.constants import CELERY_VESPA_SYNC_BEAT_LOCK_TIMEOUT
from onyx.configs.constants import OnyxCeleryTask
from onyx.configs.constants import OnyxRedisLocks
from onyx.db.connector_credential_pair import get_connector_credential_pair_from_id
from onyx.db.connector_credential_pair import get_connector_credential_pairs
from onyx.db.engine import get_session_with_tenant
from onyx.db.enums import ConnectorCredentialPairStatus
from onyx.db.search_settings import get_all_search_settings
from onyx.redis.redis_connector import RedisConnector
from onyx.redis.redis_connector_delete import RedisConnectorDeletePayload
from onyx.redis.redis_pool import get_redis_client


class TaskDependencyError(RuntimeError):
    """Raised to the caller to indicate dependent tasks are running that would interfere
    with connector deletion.
    在连接器删除过程中，如果存在正在运行的依赖任务会引发此异常。
    """


@shared_task(
    name=OnyxCeleryTask.CHECK_FOR_CONNECTOR_DELETION,
    soft_time_limit=JOB_TIMEOUT,
    trail=False,
    bind=True,
)
def check_for_connector_deletion_task(
    self: Task, *, tenant_id: str | None
) -> bool | None:
    """
    检查连接器删除状态的定时任务。
    
    Args:
        self: Celery任务实例
        tenant_id: 租户ID
        
    Returns:
        bool | None: 任务执行成功返回True，如果任务已被锁定返回None
    """
    r = get_redis_client(tenant_id=tenant_id)

    lock_beat: RedisLock = r.lock(
        OnyxRedisLocks.CHECK_CONNECTOR_DELETION_BEAT_LOCK,
        timeout=CELERY_VESPA_SYNC_BEAT_LOCK_TIMEOUT,
    )

    try:
        # these tasks should never overlap
        # 这些任务不应该重叠执行
        if not lock_beat.acquire(blocking=False):
            return None

        # collect cc_pair_ids
        # 收集连接器凭证对ID列表
        cc_pair_ids: list[int] = []
        with get_session_with_tenant(tenant_id) as db_session:
            cc_pairs = get_connector_credential_pairs(db_session)
            for cc_pair in cc_pairs:
                cc_pair_ids.append(cc_pair.id)

        # try running cleanup on the cc_pair_ids
        # 尝试对连接器凭证对ID执行清理操作
        for cc_pair_id in cc_pair_ids:
            with get_session_with_tenant(tenant_id) as db_session:
                redis_connector = RedisConnector(tenant_id, cc_pair_id)
                try:
                    try_generate_document_cc_pair_cleanup_tasks(
                        self.app, cc_pair_id, db_session, lock_beat, tenant_id
                    )
                except TaskDependencyError as e:
                    # this means we wanted to start deleting but dependent tasks were running
                    # Leave a stop signal to clear indexing and pruning tasks more quickly
                    task_logger.info(str(e))
                    redis_connector.stop.set_fence(True)
                else:
                    # clear the stop signal if it exists ... no longer needed
                    # 如果存在停止信号则清除它，因为不再需要了
                    redis_connector.stop.set_fence(False)

    except SoftTimeLimitExceeded:
        task_logger.info(
            "Soft time limit exceeded, task is being terminated gracefully."
            # 已超过软时间限制，任务正在优雅地终止
        )
    except Exception:
        task_logger.exception("Unexpected exception during connector deletion check")
    finally:
        if lock_beat.owned():
            lock_beat.release()

    return True


def try_generate_document_cc_pair_cleanup_tasks(
    app: Celery,
    cc_pair_id: int,
    db_session: Session,
    lock_beat: RedisLock,
    tenant_id: str | None,
) -> int | None:
    """
    尝试为连接器凭证对生成清理任务。
    
    Args:
        app: Celery应用实例
        cc_pair_id: 连接器凭证对ID
        db_session: 数据库会话
        lock_beat: Redis锁实例
        tenant_id: 租户ID
    
    Returns:
        int | None: 如果需要同步，返回生成的同步任务数量；如果不需要同步，返回None
    
    Raises:
        TaskDependencyError: 当存在正在运行的依赖任务(如索引或清理任务)时抛出
    """

    lock_beat.reacquire()

    redis_connector = RedisConnector(tenant_id, cc_pair_id)

    # 不要在任务仍在等待时生成新的同步任务
    if redis_connector.delete.fenced:
        return None

    # 需要在围栏内加载对象状态，以避免和数据库提交/围栏删除产生竞态条件
    # we need to load the state of the object inside the fence
    # to avoid a race condition with db.commit/fence deletion
    # at the end of this taskset
    cc_pair = get_connector_credential_pair_from_id(cc_pair_id, db_session)
    if not cc_pair:
        return None

    if cc_pair.status != ConnectorCredentialPairStatus.DELETING:
        return None

    # set a basic fence to start
    # 设置基本的围栏开始
    fence_payload = RedisConnectorDeletePayload(
        num_tasks=None,
        submitted=datetime.now(timezone.utc),
    )

    redis_connector.delete.set_fence(fence_payload)

    try:
        # 如果连接器索引或连接器清理正在运行，则不继续执行
        # do not proceed if connector indexing or connector pruning are running
        search_settings_list = get_all_search_settings(db_session)
        for search_settings in search_settings_list:
            redis_connector_index = redis_connector.new_index(search_settings.id)
            if redis_connector_index.fenced:
                raise TaskDependencyError(
                    "Connector deletion - Delayed (indexing in progress): "
                    f"cc_pair={cc_pair_id} "
                    f"search_settings={search_settings.id}"
                )

        if redis_connector.prune.fenced:
            raise TaskDependencyError(
                "Connector deletion - Delayed (pruning in progress): "
                f"cc_pair={cc_pair_id}"
            )

        if redis_connector.permissions.fenced:
            raise TaskDependencyError(
                f"Connector deletion - Delayed (permissions in progress): "
                f"cc_pair={cc_pair_id}"
            )

        # add tasks to celery and build up the task set to monitor in redis
        # 将任务添加到celery并在redis中构建要监控的任务集
        redis_connector.delete.taskset_clear()

        # 将所有需要更新的文档添加到队列中
        # Add all documents that need to be updated into the queue
        task_logger.info(
            f"RedisConnectorDeletion.generate_tasks starting. cc_pair={cc_pair_id}"
        )
        tasks_generated = redis_connector.delete.generate_tasks(
            app, db_session, lock_beat
        )
        if tasks_generated is None:
            raise ValueError("RedisConnectorDeletion.generate_tasks returned None")
    except TaskDependencyError:
        redis_connector.delete.set_fence(None)
        raise
    except Exception:
        task_logger.exception("Unexpected exception")
        redis_connector.delete.set_fence(None)
        return None
    else:
        # Currently we are allowing the sync to proceed with 0 tasks.
        # It's possible for sets/groups to be generated initially with no entries
        # and they still need to be marked as up to date.
        # 当前我们允许同步在0个任务的情况下继续进行。
        # 这是因为集合/组最初可能没有条目生成，
        # 但它们仍然需要被标记为已更新。
        # if tasks_generated == 0:
        #     return 0

        task_logger.info(
            "RedisConnectorDeletion.generate_tasks finished. "
            f"cc_pair={cc_pair_id} tasks_generated={tasks_generated}"
        )

        # set this only after all tasks have been added
        # 仅在添加所有任务后设置此项
        fence_payload.num_tasks = tasks_generated
        redis_connector.delete.set_fence(fence_payload)

    return tasks_generated
