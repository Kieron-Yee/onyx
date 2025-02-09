"""
这个文件包含了与Vespa同步相关的Celery任务和辅助函数。
主要功能:
1. 监控和管理文档元数据的同步任务
2. 处理连接器和文档集的同步状态
3. 管理索引和权限的同步
4. 提供各种监控和清理功能
"""

import time
import traceback
from datetime import datetime
from datetime import timezone
from http import HTTPStatus
from typing import cast

import httpx
from celery import Celery
from celery import shared_task
from celery import Task
from celery.exceptions import SoftTimeLimitExceeded
from celery.result import AsyncResult
from celery.states import READY_STATES
from redis import Redis
from redis.lock import Lock as RedisLock
from sqlalchemy.orm import Session
from tenacity import RetryError

from onyx.access.access import get_access_for_document
from onyx.background.celery.apps.app_base import task_logger
from onyx.background.celery.celery_redis import celery_get_queue_length
from onyx.background.celery.celery_redis import celery_get_unacked_task_ids
from onyx.background.celery.tasks.shared.RetryDocumentIndex import RetryDocumentIndex
from onyx.background.celery.tasks.shared.tasks import LIGHT_SOFT_TIME_LIMIT
from onyx.background.celery.tasks.shared.tasks import LIGHT_TIME_LIMIT
from onyx.configs.app_configs import JOB_TIMEOUT
from onyx.configs.constants import CELERY_VESPA_SYNC_BEAT_LOCK_TIMEOUT
from onyx.configs.constants import OnyxCeleryQueues
from onyx.configs.constants import OnyxCeleryTask
from onyx.configs.constants import OnyxRedisLocks
from onyx.db.connector import fetch_connector_by_id
from onyx.db.connector import mark_cc_pair_as_permissions_synced
from onyx.db.connector import mark_ccpair_as_pruned
from onyx.db.connector_credential_pair import add_deletion_failure_message
from onyx.db.connector_credential_pair import (
    delete_connector_credential_pair__no_commit,
)
from onyx.db.connector_credential_pair import get_connector_credential_pair_from_id
from onyx.db.connector_credential_pair import get_connector_credential_pairs
from onyx.db.document import count_documents_by_needs_sync
from onyx.db.document import get_document
from onyx.db.document import get_document_ids_for_connector_credential_pair
from onyx.db.document import mark_document_as_synced
from onyx.db.document_set import delete_document_set
from onyx.db.document_set import delete_document_set_cc_pair_relationship__no_commit
from onyx.db.document_set import fetch_document_sets
from onyx.db.document_set import fetch_document_sets_for_document
from onyx.db.document_set import get_document_set_by_id
from onyx.db.document_set import mark_document_set_as_synced
from onyx.db.engine import get_session_with_tenant
from onyx.db.enums import IndexingStatus
from onyx.db.index_attempt import delete_index_attempts
from onyx.db.index_attempt import get_index_attempt
from onyx.db.index_attempt import mark_attempt_failed
from onyx.db.models import DocumentSet
from onyx.document_index.document_index_utils import get_both_index_names
from onyx.document_index.factory import get_default_document_index
from onyx.document_index.interfaces import VespaDocumentFields
from onyx.redis.redis_connector import RedisConnector
from onyx.redis.redis_connector_credential_pair import RedisConnectorCredentialPair
from onyx.redis.redis_connector_delete import RedisConnectorDelete
from onyx.redis.redis_connector_doc_perm_sync import RedisConnectorPermissionSync
from onyx.redis.redis_connector_doc_perm_sync import (
    RedisConnectorPermissionSyncPayload,
)
from onyx.redis.redis_connector_index import RedisConnectorIndex
from onyx.redis.redis_connector_prune import RedisConnectorPrune
from onyx.redis.redis_document_set import RedisDocumentSet
from onyx.redis.redis_pool import get_redis_client
from onyx.redis.redis_pool import redis_lock_dump
from onyx.redis.redis_usergroup import RedisUserGroup
from onyx.utils.logger import setup_logger
from onyx.utils.variable_functionality import fetch_versioned_implementation
from onyx.utils.variable_functionality import (
    fetch_versioned_implementation_with_fallback,
)
from onyx.utils.variable_functionality import global_version
from onyx.utils.variable_functionality import noop_fallback

logger = setup_logger()


# celery auto associates tasks created inside another task,
# which bloats the result metadata considerably. trail=False prevents this.
# celery会自动关联在另一个任务内创建的任务,这会大大增加结果元数据的体积。trail=False可以防止这种情况。
@shared_task(
    name=OnyxCeleryTask.CHECK_FOR_VESPA_SYNC_TASK,
    soft_time_limit=JOB_TIMEOUT,
    trail=False,
    bind=True,
)
def check_for_vespa_sync_task(self: Task, *, tenant_id: str | None) -> bool | None:
    """Runs periodically to check if any document needs syncing.
    Generates sets of tasks for Celery if syncing is needed.
    定期运行以检查是否有文档需要同步。如果需要同步,则为Celery生成任务集。
    
    参数:
        self: Task - Celery任务实例
        tenant_id: str | None - 租户ID
        
    返回:
        bool | None - 如果成功生成任务返回True,如果任务重叠返回None
    """
    time_start = time.monotonic()

    r = get_redis_client(tenant_id=tenant_id)

    lock_beat: RedisLock = r.lock(
        OnyxRedisLocks.CHECK_VESPA_SYNC_BEAT_LOCK,
        timeout=CELERY_VESPA_SYNC_BEAT_LOCK_TIMEOUT,
    )

    try:
        # these tasks should never overlap
        # 这些任务永远不应该重叠
        if not lock_beat.acquire(blocking=False):
            return None

        with get_session_with_tenant(tenant_id) as db_session:
            try_generate_stale_document_sync_tasks(
                self.app, db_session, r, lock_beat, tenant_id
            )

        # region document set scan
        lock_beat.reacquire()
        document_set_ids: list[int] = []
        with get_session_with_tenant(tenant_id) as db_session:
            # check if any document sets are not synced
            # 检查是否有文档集未同步
            document_set_info = fetch_document_sets(
                user_id=None, db_session=db_session, include_outdated=True
            )

            for document_set, _ in document_set_info:
                document_set_ids.append(document_set.id)

        for document_set_id in document_set_ids:
            lock_beat.reacquire()
            with get_session_with_tenant(tenant_id) as db_session:
                try_generate_document_set_sync_tasks(
                    self.app, document_set_id, db_session, r, lock_beat, tenant_id
                )
        # endregion

        # check if any user groups are not synced
        if global_version.is_ee_version():
            lock_beat.reacquire()

            try:
                fetch_user_groups = fetch_versioned_implementation(
                    "onyx.db.user_group", "fetch_user_groups"
                )
            except ModuleNotFoundError:
                # Always exceptions on the MIT version, which is expected
                # We shouldn't actually get here if the ee version check works
                # 在MIT版本上总是会抛出异常,这是预期的
                # 如果ee版本检查正常工作,我们实际上不应该到达这里
                pass
            else:
                usergroup_ids: list[int] = []
                with get_session_with_tenant(tenant_id) as db_session:
                    user_groups = fetch_user_groups(
                        db_session=db_session, only_up_to_date=False
                    )

                    for usergroup in user_groups:
                        usergroup_ids.append(usergroup.id)

                for usergroup_id in usergroup_ids:
                    lock_beat.reacquire()
                    with get_session_with_tenant(tenant_id) as db_session:
                        try_generate_user_group_sync_tasks(
                            self.app, usergroup_id, db_session, r, lock_beat, tenant_id
                        )

    except SoftTimeLimitExceeded:
        task_logger.info(
            "Soft time limit exceeded, task is being terminated gracefully."
        )
    except Exception:
        task_logger.exception("Unexpected exception during vespa metadata sync")
    finally:
        if lock_beat.owned():
            lock_beat.release()
        else:
            task_logger.error(
                "check_for_vespa_sync_task - Lock not owned on completion: "
                f"tenant={tenant_id}"
            )
            redis_lock_dump(lock_beat, r)

    time_elapsed = time.monotonic() - time_start
    task_logger.debug(f"check_for_vespa_sync_task finished: elapsed={time_elapsed:.2f}")
    return True


def try_generate_stale_document_sync_tasks(
    celery_app: Celery,
    db_session: Session,
    r: Redis,
    lock_beat: RedisLock,
    tenant_id: str | None,
) -> int | None:
    """
    尝试为过期的文档生成同步任务
    
    参数:
        celery_app: Celery - Celery应用实例
        db_session: Session - 数据库会话
        r: Redis - Redis客户端
        lock_beat: RedisLock - Redis锁
        tenant_id: str | None - 租户ID
        
    返回:
        int | None - 生成的任务总数,如果围栏已存在则返回None
    """
    # the fence is up, do nothing
    # 围栏已存在,不做任何操作
    if r.exists(RedisConnectorCredentialPair.get_fence_key()):
        return None

    r.delete(RedisConnectorCredentialPair.get_taskset_key())  # delete the taskset

    # add tasks to celery and build up the task set to monitor in redis
    # 将任务添加到celery并在redis中构建要监控的任务集
    stale_doc_count = count_documents_by_needs_sync(db_session)
    if stale_doc_count == 0:
        return None

    task_logger.info(
        f"Stale documents found (at least {stale_doc_count}). Generating sync tasks by cc pair."
    )

    task_logger.info(
        "RedisConnector.generate_tasks starting by cc_pair. "
        "Documents spanning multiple cc_pairs will only be synced once."
    )

    docs_to_skip: set[str] = set()

    # rkuo: we could technically sync all stale docs in one big pass.
    # but I feel it's more understandable to group the docs by cc_pair
    # rkuo: 从技术上讲,我们可以在一次大的传递中同步所有过期文档。
    # 但我觉得按cc_pair分组文档更容易理解
    total_tasks_generated = 0
    cc_pairs = get_connector_credential_pairs(db_session)
    for cc_pair in cc_pairs:
        rc = RedisConnectorCredentialPair(tenant_id, cc_pair.id)
        rc.set_skip_docs(docs_to_skip)
        result = rc.generate_tasks(celery_app, db_session, r, lock_beat, tenant_id)

        if result is None:
            continue

        if result[1] == 0:
            continue

        task_logger.info(
            f"RedisConnector.generate_tasks finished for single cc_pair. "
            f"cc_pair={cc_pair.id} tasks_generated={result[0]} tasks_possible={result[1]}"
        )

        total_tasks_generated += result[0]

    task_logger.info(
        f"RedisConnector.generate_tasks finished for all cc_pairs. total_tasks_generated={total_tasks_generated}"
    )

    r.set(RedisConnectorCredentialPair.get_fence_key(), total_tasks_generated)
    return total_tasks_generated


def try_generate_document_set_sync_tasks(
    celery_app: Celery,
    document_set_id: int,
    db_session: Session,
    r: Redis,
    lock_beat: RedisLock,
    tenant_id: str | None,
) -> int | None:
    """
    尝试为文档集生成同步任务
    
    参数:
        celery_app: Celery - Celery应用实例
        document_set_id: int - 文档集ID
        db_session: Session - 数据库会话 
        r: Redis - Redis客户端
        lock_beat: RedisLock - Redis锁
        tenant_id: str | None - 租户ID
        
    返回:
        int | None - 生成的任务数,如果任务已挂起或文档集已同步则返回None
    """
    lock_beat.reacquire()

    rds = RedisDocumentSet(tenant_id, document_set_id)

    # don't generate document set sync tasks if tasks are still pending
    # 如果任务仍在待处理,则不要生成文档集同步任务
    if rds.fenced:
        return None

    # don't generate sync tasks if we're up to date
    # race condition with the monitor/cleanup function if we use a cached result!
    # 如果我们是最新的,则不要生成同步任务
    # 如果使用缓存结果,则与监视器/清理功能存在竞争条件!
    document_set = get_document_set_by_id(db_session, document_set_id)
    if not document_set:
        return None

    if document_set.is_up_to_date:
        return None

    # add tasks to celery and build up the task set to monitor in redis
    r.delete(rds.taskset_key)

    task_logger.info(
        f"RedisDocumentSet.generate_tasks starting. document_set_id={document_set.id}"
    )

    # Add all documents that need to be updated into the queue
    result = rds.generate_tasks(celery_app, db_session, r, lock_beat, tenant_id)
    if result is None:
        return None

    tasks_generated = result[0]
    # Currently we are allowing the sync to proceed with 0 tasks.
    # It's possible for sets/groups to be generated initially with no entries
    # and they still need to be marked as up to date.
    # 目前我们允许同步以0个任务继续进行。
    # 集合/组最初可能没有条目生成,
    # 但它们仍然需要被标记为最新。
    # if tasks_generated == 0:
    #     return 0

    task_logger.info(
        f"RedisDocumentSet.generate_tasks finished. "
        f"document_set={document_set.id} tasks_generated={tasks_generated}"
    )

    # set this only after all tasks have been added
    # 仅在添加所有任务后才设置此项
    rds.set_fence(tasks_generated)
    return tasks_generated


def try_generate_user_group_sync_tasks(
    celery_app: Celery,
    usergroup_id: int,
    db_session: Session,
    r: Redis,
    lock_beat: RedisLock,
    tenant_id: str | None,
) -> int | None:
    """
    尝试为用户组生成同步任务
    
    参数:
        celery_app: Celery - Celery应用实例
        usergroup_id: int - 用户组ID 
        db_session: Session - 数据库会话
        r: Redis - Redis客户端
        lock_beat: RedisLock - Redis锁
        tenant_id: str | None - 租户ID
        
    返回:
        int | None - 生成的任务数,如果任务已挂起或用户组已同步则返回None
    """
    lock_beat.reacquire()

    rug = RedisUserGroup(tenant_id, usergroup_id)
    if rug.fenced:
        # don't generate sync tasks if tasks are still pending
        return None

    # race condition with the monitor/cleanup function if we use a cached result!
    fetch_user_group = fetch_versioned_implementation(
        "onyx.db.user_group", "fetch_user_group"
    )

    usergroup = fetch_user_group(db_session, usergroup_id)
    if not usergroup:
        return None

    if usergroup.is_up_to_date:
        return None

    # add tasks to celery and build up the task set to monitor in redis
    r.delete(rug.taskset_key)

    # Add all documents that need to be updated into the queue
    task_logger.info(
        f"RedisUserGroup.generate_tasks starting. usergroup_id={usergroup.id}"
    )
    result = rug.generate_tasks(celery_app, db_session, r, lock_beat, tenant_id)
    if result is None:
        return None

    tasks_generated = result[0]
    # Currently we are allowing the sync to proceed with 0 tasks.
    # It's possible for sets/groups to be generated initially with no entries
    # and they still need to be marked as up to date.
    # 目前我们允许同步以0个任务继续进行。
    # 集合/组最初可能没有条目生成,
    # 但它们仍然需要被标记为最新。
    # if tasks_generated == 0:
    #     return 0

    task_logger.info(
        f"RedisUserGroup.generate_tasks finished. "
        f"usergroup={usergroup.id} tasks_generated={tasks_generated}"
    )

    # set this only after all tasks have been added
    # 仅在添加所有任务后才设置此项
    rug.set_fence(tasks_generated)
    return tasks_generated


def monitor_connector_taskset(r: Redis) -> None:
    """
    监控连接器的任务集合
    
    参数:
        r: Redis - Redis客户端
    """
    fence_value = r.get(RedisConnectorCredentialPair.get_fence_key())
    if fence_value is None:
        return

    try:
        initial_count = int(cast(int, fence_value))
    except ValueError:
        task_logger.error("The value is not an integer.")
        return

    count = r.scard(RedisConnectorCredentialPair.get_taskset_key())
    task_logger.info(
        f"Stale document sync progress: remaining={count} initial={initial_count}"
    )
    if count == 0:
        r.delete(RedisConnectorCredentialPair.get_taskset_key())
        r.delete(RedisConnectorCredentialPair.get_fence_key())
        task_logger.info(f"Successfully synced stale documents. count={initial_count}")


def monitor_document_set_taskset(
    tenant_id: str | None, key_bytes: bytes, r: Redis, db_session: Session
) -> None:
    """
    监控文档集的任务集合
    
    参数:
        tenant_id: str | None - 租户ID
        key_bytes: bytes - Redis键 
        r: Redis - Redis客户端
        db_session: Session - 数据库会话
    """
    fence_key = key_bytes.decode("utf-8")
    document_set_id_str = RedisDocumentSet.get_id_from_fence_key(fence_key)
    if document_set_id_str is None:
        task_logger.warning(f"could not parse document set id from {fence_key}")
        return

    document_set_id = int(document_set_id_str)

    rds = RedisDocumentSet(tenant_id, document_set_id)
    if not rds.fenced:
        return

    initial_count = rds.payload
    if initial_count is None:
        return

    count = cast(int, r.scard(rds.taskset_key))
    task_logger.info(
        f"Document set sync progress: document_set={document_set_id} "
        f"remaining={count} initial={initial_count}"
    )
    if count > 0:
        return

    document_set = cast(
        DocumentSet,
        get_document_set_by_id(db_session=db_session, document_set_id=document_set_id),
    )  # casting since we "know" a document set with this ID exists
    # 由于我们"知道"存在具有此ID的文档集,所以进行类型转换
    if document_set:
        if not document_set.connector_credential_pairs:
            # if there are no connectors, then delete the document set.
            # 如果没有连接器,则删除文档集。
            delete_document_set(document_set_row=document_set, db_session=db_session)
            task_logger.info(
                f"Successfully deleted document set: document_set={document_set_id}"
            )
        else:
            mark_document_set_as_synced(document_set_id, db_session)
            task_logger.info(
                f"Successfully synced document set: document_set={document_set_id}"
            )

    rds.reset()


def monitor_connector_deletion_taskset(
    tenant_id: str | None, key_bytes: bytes, r: Redis
) -> None:
    """
    监控连接器删除任务集合的进度和完成状态
    
    参数:
        tenant_id: str | None - 租户ID
        key_bytes: bytes - Redis键
        r: Redis - Redis客户端
    """
    fence_key = key_bytes.decode("utf-8")
    cc_pair_id_str = RedisConnector.get_id_from_fence_key(fence_key)
    if cc_pair_id_str is None:
        task_logger.warning(f"could not parse cc_pair_id from {fence_key}")
        return

    cc_pair_id = int(cc_pair_id_str)

    redis_connector = RedisConnector(tenant_id, cc_pair_id)

    fence_data = redis_connector.delete.payload
    if not fence_data:
        task_logger.warning(
            f"Connector deletion - fence payload invalid: cc_pair={cc_pair_id}"
        )
        return

    if fence_data.num_tasks is None:
        # the fence is setting up but isn't ready yet
        # 围栏正在设置但尚未准备就绪
        return

    remaining = redis_connector.delete.get_remaining()
    task_logger.info(
        f"Connector deletion progress: cc_pair={cc_pair_id} remaining={remaining} initial={fence_data.num_tasks}"
    )
    if remaining > 0:
        return

    with get_session_with_tenant(tenant_id) as db_session:
        cc_pair = get_connector_credential_pair_from_id(cc_pair_id, db_session)
        if not cc_pair:
            task_logger.warning(
                f"Connector deletion - cc_pair not found: cc_pair={cc_pair_id}"
            )
            return

        try:
            doc_ids = get_document_ids_for_connector_credential_pair(
                db_session, cc_pair.connector_id, cc_pair.credential_id
            )
            if len(doc_ids) > 0:
                # NOTE(rkuo): if this happens, documents somehow got added while
                # deletion was in progress. Likely a bug gating off pruning and indexing
                # work before deletion starts.
                # 注意(rkuo): 如果发生这种情况,说明在删除过程中somehow添加了文档。
                # 可能是在删除开始前关闭清理和索引工作的bug。
                task_logger.warning(
                    "Connector deletion - documents still found after taskset completion. "
                    "Clearing the current deletion attempt and allowing deletion to restart: "
                    f"cc_pair={cc_pair_id} "
                    f"docs_deleted={fence_data.num_tasks} "
                    f"docs_remaining={len(doc_ids)}"
                )

                # We don't want to waive off why we get into this state, but resetting
                # our attempt and letting the deletion restart is a good way to recover
                # 我们不想放弃为什么会进入这种状态,但重置尝试并让删除重新启动是一个很好的恢复方式
                redis_connector.delete.reset()
                raise RuntimeError(
                    "Connector deletion - documents still found after taskset completion"
                )

            # clean up the rest of the related Postgres entities
            # 清理其余相关的Postgres实体
            # index attempts
            delete_index_attempts(
                db_session=db_session,
                cc_pair_id=cc_pair_id,
            )

            # document sets
            delete_document_set_cc_pair_relationship__no_commit(
                db_session=db_session,
                connector_id=cc_pair.connector_id,
                credential_id=cc_pair.credential_id,
            )

            # user groups
            cleanup_user_groups = fetch_versioned_implementation_with_fallback(
                "onyx.db.user_group",
                "delete_user_group_cc_pair_relationship__no_commit",
                noop_fallback,
            )
            cleanup_user_groups(
                cc_pair_id=cc_pair_id,
                db_session=db_session,
            )

            # finally, delete the cc-pair
            delete_connector_credential_pair__no_commit(
                db_session=db_session,
                connector_id=cc_pair.connector_id,
                credential_id=cc_pair.credential_id,
            )
            # if there are no credentials left, delete the connector
            # 如果没有剩余的凭证,则删除连接器
            connector = fetch_connector_by_id(
                db_session=db_session,
                connector_id=cc_pair.connector_id,
            )
            if not connector or not len(connector.credentials):
                task_logger.info(
                    "Connector deletion - Found no credentials left for connector, deleting connector"
                )
                db_session.delete(connector)
            db_session.commit()
        except Exception as e:
            db_session.rollback()
            stack_trace = traceback.format_exc()
            error_message = f"Error: {str(e)}\n\nStack Trace:\n{stack_trace}"
            add_deletion_failure_message(db_session, cc_pair_id, error_message)
            task_logger.exception(
                f"Connector deletion exceptioned: "
                f"cc_pair={cc_pair_id} connector={cc_pair.connector_id} credential={cc_pair.credential_id}"
            )
            raise e

    task_logger.info(
        f"Connector deletion succeeded: "
        f"cc_pair={cc_pair_id} "
        f"connector={cc_pair.connector_id} "
        f"credential={cc_pair.credential_id} "
        f"docs_deleted={fence_data.num_tasks}"
    )

    redis_connector.delete.reset()


def monitor_ccpair_pruning_taskset(
    tenant_id: str | None, key_bytes: bytes, r: Redis, db_session: Session
) -> None:
    """
    监控连接器凭证对的清理任务集合
    
    参数:
        tenant_id: str | None - 租户ID 
        key_bytes: bytes - Redis键
        r: Redis - Redis客户端
        db_session: Session - 数据库会话
    """
    fence_key = key_bytes.decode("utf-8")
    cc_pair_id_str = RedisConnector.get_id_from_fence_key(fence_key)
    if cc_pair_id_str is None:
        task_logger.warning(
            f"monitor_ccpair_pruning_taskset: could not parse cc_pair_id from {fence_key}"
        )
        return

    cc_pair_id = int(cc_pair_id_str)

    redis_connector = RedisConnector(tenant_id, cc_pair_id)
    if not redis_connector.prune.fenced:
        return

    initial = redis_connector.prune.generator_complete
    if initial is None:
        return

    remaining = redis_connector.prune.get_remaining()
    task_logger.info(
        f"Connector pruning progress: cc_pair={cc_pair_id} remaining={remaining} initial={initial}"
    )
    if remaining > 0:
        return

    mark_ccpair_as_pruned(int(cc_pair_id), db_session)
    task_logger.info(
        f"Successfully pruned connector credential pair. cc_pair={cc_pair_id}"
    )

    redis_connector.prune.taskset_clear()
    redis_connector.prune.generator_clear()
    redis_connector.prune.set_fence(False)


def monitor_ccpair_permissions_taskset(
    tenant_id: str | None, key_bytes: bytes, r: Redis, db_session: Session
) -> None:
    """
    监控连接器凭证对的权限同步任务集合
    
    参数:
        tenant_id: str | None - 租户ID
        key_bytes: bytes - Redis键
        r: Redis - Redis客户端 
        db_session: Session - 数据库会话
    """
    fence_key = key_bytes.decode("utf-8")
    cc_pair_id_str = RedisConnector.get_id_from_fence_key(fence_key)
    if cc_pair_id_str is None:
        task_logger.warning(
            f"monitor_ccpair_permissions_taskset: could not parse cc_pair_id from {fence_key}"
        )
        return

    cc_pair_id = int(cc_pair_id_str)

    redis_connector = RedisConnector(tenant_id, cc_pair_id)
    if not redis_connector.permissions.fenced:
        return

    initial = redis_connector.permissions.generator_complete
    if initial is None:
        return

    remaining = redis_connector.permissions.get_remaining()
    task_logger.info(
        f"Permissions sync progress: cc_pair={cc_pair_id} remaining={remaining} initial={initial}"
    )
    if remaining > 0:
        return

    payload: RedisConnectorPermissionSyncPayload | None = (
        redis_connector.permissions.payload
    )
    start_time: datetime | None = payload.started if payload else None

    mark_cc_pair_as_permissions_synced(db_session, int(cc_pair_id), start_time)
    task_logger.info(f"Successfully synced permissions for cc_pair={cc_pair_id}")

    redis_connector.permissions.reset()


def monitor_ccpair_indexing_taskset(
    tenant_id: str | None, key_bytes: bytes, r: Redis, db_session: Session
) -> None:
    """
    监控连接器凭证对的索引任务集合
    
    参数:
        tenant_id: str | None - 租户ID
        key_bytes: bytes - Redis键
        r: Redis - Redis客户端
        db_session: Session - 数据库会话
    """
    # if the fence doesn't exist, there's nothing to do
    # 如果围栏不存在则无需处理
    fence_key = key_bytes.decode("utf-8")
    composite_id = RedisConnector.get_id_from_fence_key(fence_key)
    if composite_id is None:
        task_logger.warning(
            f"monitor_ccpair_indexing_taskset: could not parse composite_id from {fence_key}"
        )
        return

    # parse out metadata and initialize the helper class with it
    # 解析元数据并用它初始化辅助类
    parts = composite_id.split("/")
    if len(parts) != 2:
        return

    cc_pair_id = int(parts[0])
    search_settings_id = int(parts[1])

    redis_connector = RedisConnector(tenant_id, cc_pair_id)
    redis_connector_index = redis_connector.new_index(search_settings_id)
    if not redis_connector_index.fenced:
        return

    payload = redis_connector_index.payload
    if not payload:
        return

    elapsed_started_str = None
    if payload.started:
        elapsed_started = datetime.now(timezone.utc) - payload.started
        elapsed_started_str = f"{elapsed_started.total_seconds():.2f}"

    elapsed_submitted = datetime.now(timezone.utc) - payload.submitted

    progress = redis_connector_index.get_progress()
    if progress is not None:
        task_logger.info(
            f"Connector indexing progress: "
            f"attempt={payload.index_attempt_id} "
            f"cc_pair={cc_pair_id} "
            f"search_settings={search_settings_id} "
            f"progress={progress} "
            f"elapsed_submitted={elapsed_submitted.total_seconds():.2f} "
            f"elapsed_started={elapsed_started_str}"
        )

    if payload.index_attempt_id is None or payload.celery_task_id is None:
        # the task is still setting up
        # 任务仍在设置中
        return

    # never use any blocking methods on the result from inside a task!
    # 永远不要从任务内部对结果使用任何阻塞方法!
    result: AsyncResult = AsyncResult(payload.celery_task_id)

    # inner/outer/inner double check pattern to avoid race conditions when checking for
    # bad state

    # inner = get_completion / generator_complete not signaled
    # outer = result.state in READY state
    # 内/外/内双重检查模式,用于在检查错误状态时避免竞争条件
    # inner = get_completion / generator_complete 未发出信号
    # outer = result.state 处于 READY 状态
    status_int = redis_connector_index.get_completion()
    if status_int is None:  # inner signal not set ... possible error
        task_state = result.state
        if (
            task_state in READY_STATES
        ):  # outer signal in terminal state ... possible error
            # Now double check!
            # 现在双重检查!
            if redis_connector_index.get_completion() is None:
                # inner signal still not set (and cannot change when outer result_state is READY)
                # Task is finished but generator complete isn't set.
                # We have a problem! Worker may have crashed.
                # 内部信号仍未设置(当外部result_state为READY时无法更改)
                # 任务已完成但generator complete未设置。
                # 我们有问题了!Worker可能已崩溃。
                task_result = str(result.result)
                task_traceback = str(result.traceback)

                msg = (
                    f"Connector indexing aborted or exceptioned: "
                    f"attempt={payload.index_attempt_id} "
                    f"celery_task={payload.celery_task_id} "
                    f"cc_pair={cc_pair_id} "
                    f"search_settings={search_settings_id} "
                    f"elapsed_submitted={elapsed_submitted.total_seconds():.2f} "
                    f"result.state={task_state} "
                    f"result.result={task_result} "
                    f"result.traceback={task_traceback}"
                )
                task_logger.warning(msg)

                try:
                    index_attempt = get_index_attempt(
                        db_session, payload.index_attempt_id
                    )
                    if index_attempt:
                        if (
                            index_attempt.status != IndexingStatus.CANCELED
                            and index_attempt.status != IndexingStatus.FAILED
                        ):
                            mark_attempt_failed(
                                index_attempt_id=payload.index_attempt_id,
                                db_session=db_session,
                                failure_reason=msg,
                            )
                except Exception:
                    task_logger.exception(
                        "monitor_ccpair_indexing_taskset - transient exception marking index attempt as failed: "
                        f"attempt={payload.index_attempt_id} "
                        f"tenant={tenant_id} "
                        f"cc_pair={cc_pair_id} "
                        f"search_settings={search_settings_id}"
                    )

                redis_connector_index.reset()
        return

    status_enum = HTTPStatus(status_int)

    task_logger.info(
        f"Connector indexing finished: "
        f"attempt={payload.index_attempt_id} "
        f"cc_pair={cc_pair_id} "
        f"search_settings={search_settings_id} "
        f"progress={progress} "
        f"status={status_enum.name} "
        f"elapsed_submitted={elapsed_submitted.total_seconds():.2f} "
        f"elapsed_started={elapsed_started_str}"
    )

    redis_connector_index.reset()


@shared_task(name=OnyxCeleryTask.MONITOR_VESPA_SYNC, soft_time_limit=300, bind=True)
def monitor_vespa_sync(self: Task, tenant_id: str | None) -> bool:
    """
    Celery beat任务,用于监控和完成元数据同步任务集。
    扫描围栏值并获取任务集计数。如果计数为0,表示所有任务完成,需要清理。
    
    This is a celery beat task that monitors and finalizes metadata sync tasksets.
    It scans for fence values and then gets the counts of any associated tasksets.
    If the count is 0, that means all tasks finished and we should clean up.

    任务锁超时时间为CELERY_METADATA_SYNC_BEAT_LOCK_TIMEOUT秒,
    因此这个函数中不要做太耗时的操作。
    
    参数:
        self: Task - Celery任务实例
        tenant_id: str | None - 租户ID
        
    返回:
        bool - 如果任务确实完成了工作返回True,如果提前退出以避免重叠返回False
    """
    time_start = time.monotonic()
    r = get_redis_client(tenant_id=tenant_id)

    lock_beat: RedisLock = r.lock(
        OnyxRedisLocks.MONITOR_VESPA_SYNC_BEAT_LOCK,
        timeout=CELERY_VESPA_SYNC_BEAT_LOCK_TIMEOUT,
    )

    try:
        # prevent overlapping tasks
        # 防止任务重叠
        if not lock_beat.acquire(blocking=False):
            return False

        # print current queue lengths
        # 打印当前队列长度
        r_celery = self.app.broker_connection().channel().client  # type: ignore
        n_celery = celery_get_queue_length("celery", r_celery)
        n_indexing = celery_get_queue_length(
            OnyxCeleryQueues.CONNECTOR_INDEXING, r_celery
        )
        n_sync = celery_get_queue_length(OnyxCeleryQueues.VESPA_METADATA_SYNC, r_celery)
        n_deletion = celery_get_queue_length(
            OnyxCeleryQueues.CONNECTOR_DELETION, r_celery
        )
        n_pruning = celery_get_queue_length(
            OnyxCeleryQueues.CONNECTOR_PRUNING, r_celery
        )
        n_permissions_sync = celery_get_queue_length(
            OnyxCeleryQueues.CONNECTOR_DOC_PERMISSIONS_SYNC, r_celery
        )
        n_external_group_sync = celery_get_queue_length(
            OnyxCeleryQueues.CONNECTOR_EXTERNAL_GROUP_SYNC, r_celery
        )
        n_permissions_upsert = celery_get_queue_length(
            OnyxCeleryQueues.DOC_PERMISSIONS_UPSERT, r_celery
        )

        prefetched = celery_get_unacked_task_ids(
            OnyxCeleryQueues.CONNECTOR_INDEXING, r_celery
        )

        task_logger.info(
            f"Queue lengths: celery={n_celery} "
            f"indexing={n_indexing} "
            f"indexing_prefetched={len(prefetched)} "
            f"sync={n_sync} "
            f"deletion={n_deletion} "
            f"pruning={n_pruning} "
            f"permissions_sync={n_permissions_sync} "
            f"external_group_sync={n_external_group_sync} "
            f"permissions_upsert={n_permissions_upsert} "
        )

        # scan and monitor activity to completion
        # 扫描并监控活动直至完成
        lock_beat.reacquire()
        if r.exists(RedisConnectorCredentialPair.get_fence_key()):
            monitor_connector_taskset(r)

        for key_bytes in r.scan_iter(RedisConnectorDelete.FENCE_PREFIX + "*"):
            lock_beat.reacquire()
            monitor_connector_deletion_taskset(tenant_id, key_bytes, r)

        for key_bytes in r.scan_iter(RedisDocumentSet.FENCE_PREFIX + "*"):
            lock_beat.reacquire()
            with get_session_with_tenant(tenant_id) as db_session:
                monitor_document_set_taskset(tenant_id, key_bytes, r, db_session)

        for key_bytes in r.scan_iter(RedisUserGroup.FENCE_PREFIX + "*"):
            lock_beat.reacquire()
            monitor_usergroup_taskset = fetch_versioned_implementation_with_fallback(
                "onyx.background.celery.tasks.vespa.tasks",
                "monitor_usergroup_taskset",
                noop_fallback,
            )
            with get_session_with_tenant(tenant_id) as db_session:
                monitor_usergroup_taskset(tenant_id, key_bytes, r, db_session)

        for key_bytes in r.scan_iter(RedisConnectorPrune.FENCE_PREFIX + "*"):
            lock_beat.reacquire()
            with get_session_with_tenant(tenant_id) as db_session:
                monitor_ccpair_pruning_taskset(tenant_id, key_bytes, r, db_session)

        for key_bytes in r.scan_iter(RedisConnectorIndex.FENCE_PREFIX + "*"):
            lock_beat.reacquire()
            with get_session_with_tenant(tenant_id) as db_session:
                monitor_ccpair_indexing_taskset(tenant_id, key_bytes, r, db_session)

        for key_bytes in r.scan_iter(RedisConnectorPermissionSync.FENCE_PREFIX + "*"):
            lock_beat.reacquire()
            with get_session_with_tenant(tenant_id) as db_session:
                monitor_ccpair_permissions_taskset(tenant_id, key_bytes, r, db_session)

    except SoftTimeLimitExceeded:
        task_logger.info(
            "Soft time limit exceeded, task is being terminated gracefully."
        )
    finally:
        if lock_beat.owned():
            lock_beat.release()

    time_elapsed = time.monotonic() - time_start
    task_logger.debug(f"monitor_vespa_sync finished: elapsed={time_elapsed:.2f}")
    return True


@shared_task(
    name=OnyxCeleryTask.VESPA_METADATA_SYNC_TASK,
    bind=True, 
    soft_time_limit=LIGHT_SOFT_TIME_LIMIT,
    time_limit=LIGHT_TIME_LIMIT,
    max_retries=3,
)
def vespa_metadata_sync_task(
    self: Task, document_id: str, tenant_id: str | None
) -> bool:
    """
    同步单个文档的元数据到Vespa
    
    参数:
        self: Task - Celery任务实例
        document_id: str - 文档ID
        tenant_id: str | None - 租户ID
        
    返回:
        bool - 成功返回True,失败返回False。
        如果遇到临时错误会重试,永久错误会返回False。
    """
    try:
        with get_session_with_tenant(tenant_id) as db_session:
            curr_ind_name, sec_ind_name = get_both_index_names(db_session)
            doc_index = get_default_document_index(
                primary_index_name=curr_ind_name, secondary_index_name=sec_ind_name
            )

            retry_index = RetryDocumentIndex(doc_index)

            doc = get_document(document_id, db_session)
            if not doc:
                return False

            # document set sync
            doc_sets = fetch_document_sets_for_document(document_id, db_session)
            update_doc_sets: set[str] = set(doc_sets)

            # User group sync
            doc_access = get_access_for_document(
                document_id=document_id, db_session=db_session
            )

            fields = VespaDocumentFields(
                document_sets=update_doc_sets,
                access=doc_access,
                boost=doc.boost,
                hidden=doc.hidden,
            )

            # update Vespa. OK if doc doesn't exist. Raises exception otherwise.
            # 更新Vespa。如果文档不存在则没问题。否则抛出异常。
            chunks_affected = retry_index.update_single(document_id, fields)

            # update db last. Worst case = we crash right before this and
            # the sync might repeat again later
            # 最后更新数据库。最坏的情况 = 我们在此之前崩溃,
            # 同步稍后可能会再次重复
            mark_document_as_synced(document_id, db_session)

            redis_syncing_key = RedisConnectorCredentialPair.make_redis_syncing_key(
                document_id
            )
            r = get_redis_client(tenant_id=tenant_id)
            r.delete(redis_syncing_key)
            # r.hdel(RedisConnectorCredentialPair.SYNCING_HASH, document_id)

            task_logger.info(f"doc={document_id} action=sync chunks={chunks_affected}")
    except SoftTimeLimitExceeded:
        task_logger.info(f"SoftTimeLimitExceeded exception. doc={document_id}")
    except Exception as ex:
        if isinstance(ex, RetryError):
            task_logger.warning(
                f"Tenacity retry failed: num_attempts={ex.last_attempt.attempt_number}"
            )

            # only set the inner exception if it is of type Exception
            # 仅当内部异常是Exception类型时才设置
            e_temp = ex.last_attempt.exception()
            if isinstance(e_temp, Exception):
                e = e_temp
        else:
            e = ex

        if isinstance(e, httpx.HTTPStatusError):
            if e.response.status_code == HTTPStatus.BAD_REQUEST:
                task_logger.exception(
                    f"Non-retryable HTTPStatusError: "
                    f"doc={document_id} "
                    f"status={e.response.status_code}"
                )
            return False

        task_logger.exception(
            f"Unexpected exception during vespa metadata sync: doc={document_id}"
        )

        # Exponential backoff from 2^4 to 2^6 ... i.e. 16, 32, 64
        # 指数退避从2^4到2^6 ... 即16、32、64
        countdown = 2 ** (self.request.retries + 4)
        self.retry(exc=e, countdown=countdown)

    return True
