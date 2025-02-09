"""
文件功能说明:
该文件主要负责文档权限同步相关的Celery任务。
包含了检查权限同步、生成同步任务、更新外部文档权限等核心功能。
实现了文档权限的自动同步和定期更新机制。

主要功能模块：
1. 权限同步检查 - 定期检查是否需要进行权限同步
2. 权限同步任务生成 - 为需要同步的文档创建同步任务
3. 权限更新执行 - 执行实际的权限更新操作
4. 异常处理和重试机制 - 确保同步过程的可靠性

关键组件：
- Redis锁：用于防止任务重复执行
- 任务围栏机制：确保任务按正确顺序执行
- 数据库会话管理：处理数据库操作和事务
"""

import time
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from time import sleep
from uuid import uuid4

from celery import Celery
from celery import shared_task
from celery import Task
from celery.exceptions import SoftTimeLimitExceeded
from redis import Redis
from redis.lock import Lock as RedisLock

from ee.onyx.db.connector_credential_pair import get_all_auto_sync_cc_pairs
from ee.onyx.db.document import upsert_document_external_perms
from ee.onyx.external_permissions.sync_params import DOC_PERMISSION_SYNC_PERIODS
from ee.onyx.external_permissions.sync_params import DOC_PERMISSIONS_FUNC_MAP
from onyx.access.models import DocExternalAccess
from onyx.background.celery.apps.app_base import task_logger
from onyx.configs.app_configs import JOB_TIMEOUT
from onyx.configs.constants import CELERY_PERMISSIONS_SYNC_LOCK_TIMEOUT
from onyx.configs.constants import CELERY_TASK_WAIT_FOR_FENCE_TIMEOUT
from onyx.configs.constants import CELERY_VESPA_SYNC_BEAT_LOCK_TIMEOUT
from onyx.configs.constants import DANSWER_REDIS_FUNCTION_LOCK_PREFIX
from onyx.configs.constants import DocumentSource
from onyx.configs.constants import OnyxCeleryPriority
from onyx.configs.constants import OnyxCeleryQueues
from onyx.configs.constants import OnyxCeleryTask
from onyx.configs.constants import OnyxRedisLocks
from onyx.db.connector_credential_pair import get_connector_credential_pair_from_id
from onyx.db.document import upsert_document_by_connector_credential_pair
from onyx.db.engine import get_session_with_tenant
from onyx.db.enums import AccessType
from onyx.db.enums import ConnectorCredentialPairStatus
from onyx.db.models import ConnectorCredentialPair
from onyx.db.users import batch_add_ext_perm_user_if_not_exists
from onyx.redis.redis_connector import RedisConnector
from onyx.redis.redis_connector_doc_perm_sync import (
    RedisConnectorPermissionSyncPayload,
)
from onyx.redis.redis_pool import get_redis_client
from onyx.utils.logger import doc_permission_sync_ctx
from onyx.utils.logger import setup_logger

logger = setup_logger()


# 权限更新最大重试次数
DOCUMENT_PERMISSIONS_UPDATE_MAX_RETRIES = 3

# 任务超时设置
# 相比RetryDocumentIndex的STOP_AFTER+MAX_WAIT多5秒
LIGHT_SOFT_TIME_LIMIT = 105  # 软超时限制，达到此时间会发出警告
LIGHT_TIME_LIMIT = LIGHT_SOFT_TIME_LIMIT + 15  # 硬超时限制，达到此时间任务会被强制终止


def _is_external_doc_permissions_sync_due(cc_pair: ConnectorCredentialPair) -> bool:
    """Returns boolean indicating if external doc permissions sync is due.
    返回布尔值表示是否需要进行外部文档权限同步。
    
    参数:
        cc_pair: ConnectorCredentialPair对象，包含连接器和凭证的配对信息
        
    返回:
        bool: 如果需要同步返回True，否则返回False
    """

    # 检查访问类型是否为SYNC
    if cc_pair.access_type != AccessType.SYNC:
        return False

    # skip doc permissions sync if not active
    # 如果不活跃则跳过文档权限同步
    if cc_pair.status != ConnectorCredentialPairStatus.ACTIVE:
        return False

    if cc_pair.status == ConnectorCredentialPairStatus.DELETING:
        return False

    # If the last sync is None, it has never been run so we run the sync
    # 如果上次同步为空，说明从未运行过，所以我们运行同步
    last_perm_sync = cc_pair.last_time_perm_sync
    if last_perm_sync is None:
        return True

    source_sync_period = DOC_PERMISSION_SYNC_PERIODS.get(cc_pair.connector.source)

    # If RESTRICTED_FETCH_PERIOD[source] is None, we always run the sync.
    # 如果 RESTRICTED_FETCH_PERIOD[source] 为空，我们总是运行同步
    if not source_sync_period:
        return True

    # If the last sync is greater than the full fetch period, we run the sync
    # 如果距离上次同步的时间超过了完整获取周期，我们运行同步
    next_sync = last_perm_sync + timedelta(seconds=source_sync_period)
    if datetime.now(timezone.utc) >= next_sync:
        return True

    return False


@shared_task(
    name=OnyxCeleryTask.CHECK_FOR_DOC_PERMISSIONS_SYNC,
    soft_time_limit=JOB_TIMEOUT,
    bind=True,
)
def check_for_doc_permissions_sync(self: Task, *, tenant_id: str | None) -> bool | None:
    """
    检查需要进行文档权限同步的连接器凭证对。
    
    工作流程：
    1. 获取Redis锁防止任务重复执行
    2. 查询所有需要自动同步的连接器凭证对
    3. 检查每个凭证对是否需要进行同步
    4. 为需要同步的凭证对创建同步任务
    
    参数:
        self: Celery任务实例
        tenant_id: 租户ID
        
    返回:
        bool | None: 任务执行成功返回True，如果已有任务在运行则返回None
    """
    r = get_redis_client(tenant_id=tenant_id)

    lock_beat: RedisLock = r.lock(
        OnyxRedisLocks.CHECK_CONNECTOR_DOC_PERMISSIONS_SYNC_BEAT_LOCK,
        timeout=CELERY_VESPA_SYNC_BEAT_LOCK_TIMEOUT,
    )

    try:
        # these tasks should never overlap
        # 这些任务永远不应该重叠
        if not lock_beat.acquire(blocking=False):
            return None

        # get all cc pairs that need to be synced
        # 获取所有需要同步的连接器凭证对
        cc_pair_ids_to_sync: list[int] = []
        with get_session_with_tenant(tenant_id) as db_session:
            cc_pairs = get_all_auto_sync_cc_pairs(db_session)

            for cc_pair in cc_pairs:
                if _is_external_doc_permissions_sync_due(cc_pair):
                    cc_pair_ids_to_sync.append(cc_pair.id)

        for cc_pair_id in cc_pair_ids_to_sync:
            tasks_created = try_creating_permissions_sync_task(
                self.app, cc_pair_id, r, tenant_id
            )
            if not tasks_created:
                continue

            task_logger.info(f"Doc permissions sync queued: cc_pair={cc_pair_id}")
    except SoftTimeLimitExceeded:
        task_logger.info(
            "Soft time limit exceeded, task is being terminated gracefully."
        )
    except Exception:
        task_logger.exception(f"Unexpected exception: tenant={tenant_id}")
    finally:
        if lock_beat.owned():
            lock_beat.release()

    return True


def try_creating_permissions_sync_task(
    app: Celery,
    cc_pair_id: int,
    r: Redis,
    tenant_id: str | None,
) -> int | None:
    """
    尝试创建权限同步任务。
    
    工作流程：
    1. 获取Redis锁以确保任务创建的原子性
    2. 检查是否存在任务围栏
    3. 清理之前的任务状态
    4. 创建新的同步任务
    5. 设置任务围栏
    
    安全机制：
    - 使用Redis锁防止并发创建
    - 检查多重围栏状态
    - 异常时自动清理任务状态
    
    参数和返回值说明同上
    """
    redis_connector = RedisConnector(tenant_id, cc_pair_id)

    LOCK_TIMEOUT = 30

    lock: RedisLock = r.lock(
        DANSWER_REDIS_FUNCTION_LOCK_PREFIX + "try_generate_permissions_sync_tasks",
        timeout=LOCK_TIMEOUT,
    )

    acquired = lock.acquire(blocking_timeout=LOCK_TIMEOUT / 2)
    if not acquired:
        return None

    try:
        if redis_connector.permissions.fenced:
            return None

        if redis_connector.delete.fenced:
            return None

        if redis_connector.prune.fenced:
            return None

        redis_connector.permissions.generator_clear()
        redis_connector.permissions.taskset_clear()

        custom_task_id = f"{redis_connector.permissions.generator_task_key}_{uuid4()}"

        result = app.send_task(
            OnyxCeleryTask.CONNECTOR_PERMISSION_SYNC_GENERATOR_TASK,
            kwargs=dict(
                cc_pair_id=cc_pair_id,
                tenant_id=tenant_id,
            ),
            queue=OnyxCeleryQueues.CONNECTOR_DOC_PERMISSIONS_SYNC,
            task_id=custom_task_id,
            priority=OnyxCeleryPriority.HIGH,
        )

        # set a basic fence to start
        payload = RedisConnectorPermissionSyncPayload(
            started=None, celery_task_id=result.id
        )

        redis_connector.permissions.set_fence(payload)
    except Exception:
        task_logger.exception(f"Unexpected exception: cc_pair={cc_pair_id}")
        return None
    finally:
        if lock.owned():
            lock.release()

    return 1


@shared_task(
    name=OnyxCeleryTask.CONNECTOR_PERMISSION_SYNC_GENERATOR_TASK,
    acks_late=False,
    soft_time_limit=JOB_TIMEOUT,
    track_started=True,
    trail=False,
    bind=True,
)
def connector_permission_sync_generator_task(
    self: Task,
    cc_pair_id: int,
    tenant_id: str | None,
) -> None:
    """
    权限同步任务生成器。
    
    主要职责：
    1. 等待任务围栏就绪
    2. 获取需要同步的文档权限信息
    3. 生成具体的权限同步任务
    4. 更新任务状态和围栏信息
    
    异常处理：
    - 超时检测
    - 围栏状态验证
    - 任务状态清理
    """
    doc_permission_sync_ctx_dict = doc_permission_sync_ctx.get()
    doc_permission_sync_ctx_dict["cc_pair_id"] = cc_pair_id
    doc_permission_sync_ctx_dict["request_id"] = self.request.id
    doc_permission_sync_ctx.set(doc_permission_sync_ctx_dict)

    redis_connector = RedisConnector(tenant_id, cc_pair_id)

    r = get_redis_client(tenant_id=tenant_id)

    # this wait is needed to avoid a race condition where
    # the primary worker sends the task and it is immediately executed
    # before the primary worker can finalize the fence
    # 需要这个等待来避免竞态条件，即主工作进程发送任务后，
    # 在其完成围栏设置之前任务就立即执行了
    start = time.monotonic()
    while True:
        if time.monotonic() - start > CELERY_TASK_WAIT_FOR_FENCE_TIMEOUT:
            raise ValueError(
                f"connector_permission_sync_generator_task - timed out waiting for fence to be ready: "
                f"fence={redis_connector.permissions.fence_key}"
            )

        # The fence must exist
        # 围栏必须存在
        if not redis_connector.permissions.fenced:
            raise ValueError(
                f"connector_permission_sync_generator_task - fence not found: "
                f"fence={redis_connector.permissions.fence_key}"
            )

        # The payload must exist
        # 负载必须存在
        payload = redis_connector.permissions.payload  # The payload must exist
        if not payload:
            raise ValueError(
                "connector_permission_sync_generator_task: payload invalid or not found"
            )

        if payload.celery_task_id is None:
            logger.info(
                f"connector_permission_sync_generator_task - Waiting for fence: "
                f"fence={redis_connector.permissions.fence_key}"
            )
            sleep(1)
            continue

        logger.info(
            f"connector_permission_sync_generator_task - Fence found, continuing...: "
            f"fence={redis_connector.permissions.fence_key}"
        )
        break

    lock: RedisLock = r.lock(
        OnyxRedisLocks.CONNECTOR_DOC_PERMISSIONS_SYNC_LOCK_PREFIX
        + f"_{redis_connector.id}",
        timeout=CELERY_PERMISSIONS_SYNC_LOCK_TIMEOUT,
    )

    acquired = lock.acquire(blocking=False)
    if not acquired:
        task_logger.warning(
            f"Permission sync task already running, exiting...: cc_pair={cc_pair_id}"
        )
        return None

    try:
        with get_session_with_tenant(tenant_id) as db_session:
            cc_pair = get_connector_credential_pair_from_id(cc_pair_id, db_session)
            if cc_pair is None:
                raise ValueError(
                    f"No connector credential pair found for id: {cc_pair_id}"
                )

            source_type = cc_pair.connector.source

            doc_sync_func = DOC_PERMISSIONS_FUNC_MAP.get(source_type)
            if doc_sync_func is None:
                raise ValueError(
                    f"No doc sync func found for {source_type} with cc_pair={cc_pair_id}"
                )

            logger.info(f"Syncing docs for {source_type} with cc_pair={cc_pair_id}")

            payload = redis_connector.permissions.payload
            if not payload:
                raise ValueError(f"No fence payload found: cc_pair={cc_pair_id}")

            new_payload = RedisConnectorPermissionSyncPayload(
                started=datetime.now(timezone.utc),
                celery_task_id=payload.celery_task_id,
            )
            redis_connector.permissions.set_fence(new_payload)

            document_external_accesses: list[DocExternalAccess] = doc_sync_func(cc_pair)

            task_logger.info(
                f"RedisConnector.permissions.generate_tasks starting. cc_pair={cc_pair_id}"
            )
            tasks_generated = redis_connector.permissions.generate_tasks(
                celery_app=self.app,
                lock=lock,
                new_permissions=document_external_accesses,
                source_string=source_type,
                connector_id=cc_pair.connector.id,
                credential_id=cc_pair.credential.id,
            )
            if tasks_generated is None:
                return None

            task_logger.info(
                f"RedisConnector.permissions.generate_tasks finished. "
                f"cc_pair={cc_pair_id} tasks_generated={tasks_generated}"
            )

            redis_connector.permissions.generator_complete = tasks_generated

    except Exception as e:
        task_logger.exception(f"Failed to run permission sync: cc_pair={cc_pair_id}")

        redis_connector.permissions.generator_clear()
        redis_connector.permissions.taskset_clear()
        redis_connector.permissions.set_fence(None)
        raise e
    finally:
        if lock.owned():
            lock.release()


@shared_task(
    name=OnyxCeleryTask.UPDATE_EXTERNAL_DOCUMENT_PERMISSIONS_TASK,
    soft_time_limit=LIGHT_SOFT_TIME_LIMIT,
    time_limit=LIGHT_TIME_LIMIT,
    max_retries=DOCUMENT_PERMISSIONS_UPDATE_MAX_RETRIES,
    bind=True,
)
def update_external_document_permissions_task(
    self: Task,
    tenant_id: str | None,
    serialized_doc_external_access: dict,
    source_string: str,
    connector_id: int,
    credential_id: int,
) -> bool:
    """
    更新外部文档权限的具体执行任务。
    
    执行流程：
    1. 反序列化文档访问权限信息
    2. 确保所有用户存在于数据库中
    3. 更新文档的外部权限
    4. 如果是新文档，建立与连接器凭证对的关联
    
    数据一致性：
    - 使用数据库事务确保操作原子性
    - 异常时自动回滚
    """
    document_external_access = DocExternalAccess.from_dict(
        serialized_doc_external_access
    )
    doc_id = document_external_access.doc_id
    external_access = document_external_access.external_access
    try:
        with get_session_with_tenant(tenant_id) as db_session:
            # Add the users to the DB if they don't exist
            # 如果用户不存在，则将其添加到数据库中
            batch_add_ext_perm_user_if_not_exists(
                db_session=db_session,
                emails=list(external_access.external_user_emails),
            )
            # Then we upsert the document's external permissions in postgres
            # 然后我们在postgres中更新文档的外部权限
            created_new_doc = upsert_document_external_perms(
                db_session=db_session,
                doc_id=doc_id,
                external_access=external_access,
                source_type=DocumentSource(source_string),
            )

            # If a new document was created, we associate it with the cc_pair
            # 如果创建了新文档，我们将其与连接器凭证对关联
            if created_new_doc:
                upsert_document_by_connector_credential_pair(
                    db_session=db_session,
                    connector_id=connector_id,
                    credential_id=credential_id,
                    document_ids=[doc_id],
                )

            logger.debug(
                f"Successfully synced postgres document permissions for {doc_id}"
            )
        return True
    except Exception:
        logger.exception("Error Syncing Document Permissions")
        return False
