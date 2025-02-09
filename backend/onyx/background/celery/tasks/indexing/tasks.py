import os
import sys
import time
from datetime import datetime
from datetime import timezone
from http import HTTPStatus
from time import sleep

import redis
import sentry_sdk
from celery import Celery
from celery import shared_task
from celery import Task
from celery.exceptions import SoftTimeLimitExceeded
from redis import Redis
from redis.exceptions import LockError
from redis.lock import Lock as RedisLock
from sqlalchemy.orm import Session

from onyx.background.celery.apps.app_base import task_logger
from onyx.background.celery.celery_redis import celery_find_task
from onyx.background.celery.celery_redis import celery_get_unacked_task_ids
from onyx.background.indexing.job_client import SimpleJobClient
from onyx.background.indexing.run_indexing import run_indexing_entrypoint
from onyx.configs.app_configs import DISABLE_INDEX_UPDATE_ON_SWAP
from onyx.configs.constants import CELERY_INDEXING_LOCK_TIMEOUT
from onyx.configs.constants import CELERY_TASK_WAIT_FOR_FENCE_TIMEOUT
from onyx.configs.constants import CELERY_VESPA_SYNC_BEAT_LOCK_TIMEOUT
from onyx.configs.constants import DANSWER_REDIS_FUNCTION_LOCK_PREFIX
from onyx.configs.constants import DocumentSource
from onyx.configs.constants import OnyxCeleryPriority
from onyx.configs.constants import OnyxCeleryQueues
from onyx.configs.constants import OnyxCeleryTask
from onyx.configs.constants import OnyxRedisLocks
from onyx.configs.constants import OnyxRedisSignals
from onyx.db.connector import mark_ccpair_with_indexing_trigger
from onyx.db.connector_credential_pair import fetch_connector_credential_pairs
from onyx.db.connector_credential_pair import get_connector_credential_pair_from_id
from onyx.db.engine import get_db_current_time
from onyx.db.engine import get_session_with_tenant
from onyx.db.enums import ConnectorCredentialPairStatus
from onyx.db.enums import IndexingMode
from onyx.db.enums import IndexingStatus
from onyx.db.enums import IndexModelStatus
from onyx.db.index_attempt import create_index_attempt
from onyx.db.index_attempt import delete_index_attempt
from onyx.db.index_attempt import get_all_index_attempts_by_status
from onyx.db.index_attempt import get_index_attempt
from onyx.db.index_attempt import get_last_attempt_for_cc_pair
from onyx.db.index_attempt import mark_attempt_canceled
from onyx.db.index_attempt import mark_attempt_failed
from onyx.db.models import ConnectorCredentialPair
from onyx.db.models import IndexAttempt
from onyx.db.models import SearchSettings
from onyx.db.search_settings import get_active_search_settings
from onyx.db.search_settings import get_current_search_settings
from onyx.db.swap_index import check_index_swap
from onyx.indexing.indexing_heartbeat import IndexingHeartbeatInterface
from onyx.natural_language_processing.search_nlp_models import EmbeddingModel
from onyx.natural_language_processing.search_nlp_models import warm_up_bi_encoder
from onyx.redis.redis_connector import RedisConnector
from onyx.redis.redis_connector_index import RedisConnectorIndex
from onyx.redis.redis_connector_index import RedisConnectorIndexPayload
from onyx.redis.redis_pool import get_redis_client
from onyx.redis.redis_pool import redis_lock_dump
from onyx.utils.logger import setup_logger
from onyx.utils.variable_functionality import global_version
from shared_configs.configs import INDEXING_MODEL_SERVER_HOST
from shared_configs.configs import INDEXING_MODEL_SERVER_PORT
from shared_configs.configs import MULTI_TENANT
from shared_configs.configs import SENTRY_DSN

logger = setup_logger()

"""
这个文件实现了后台索引任务的主要功能。主要包括:
1. 检查和启动索引任务
2. 管理索引任务的生命周期
3. 处理索引任务的执行状态
4. 实现索引任务的回调机制
5. 验证和清理索引栅栏(fence)
"""

class IndexingCallback(IndexingHeartbeatInterface):
    """索引回调接口的实现类,用于监控索引进度和状态
    
    属性:
        PARENT_CHECK_INTERVAL: 检查父进程的时间间隔(秒)
        parent_pid: 父进程ID 
        redis_lock: Redis锁对象
        stop_key: 停止信号的key
        generator_progress_key: 进度记录的key
        redis_client: Redis客户端
        started: 开始时间
        last_tag: 最后一个标签
        last_lock_reacquire: 最后一次重新获取锁的时间
        last_lock_monotonic: 最后一次锁定的单调时间
        last_parent_check: 最后一次检查父进程的时间
    """

    def __init__(
        self,
        parent_pid: int,
        stop_key: str,
        generator_progress_key: str,
        redis_lock: RedisLock,
        redis_client: Redis,
    ):
        super().__init__()
        self.parent_pid = parent_pid
        self.redis_lock: RedisLock = redis_lock
        self.stop_key: str = stop_key
        self.generator_progress_key: str = generator_progress_key
        self.redis_client = redis_client
        self.started: datetime = datetime.now(timezone.utc)
        self.redis_lock.reacquire()

        self.last_tag: str = "IndexingCallback.__init__"
        self.last_lock_reacquire: datetime = datetime.now(timezone.utc)
        self.last_lock_monotonic = time.monotonic()

        self.last_parent_check = time.monotonic()

    def should_stop(self) -> bool:
        """
        检查是否应该停止索引任务
        
        Returns:
            bool: 如果应该停止返回True,否则返回False
        """
        if self.redis_client.exists(self.stop_key):
            return True

        return False

    def progress(self, tag: str, amount: int) -> None:
        """
        更新索引进度并重新获取锁
        
        Args:
            tag: 进度标签
            amount: 进度增量
        """
        # rkuo: this shouldn't be necessary yet because we spawn the process this runs inside
        # with daemon = True. It seems likely some indexing tasks will need to spawn other processes eventually
        # so leave this code in until we're ready to test it.

        # if self.parent_pid:
        #     # check if the parent pid is alive so we aren't running as a zombie
        #     now = time.monotonic()
        #     if now - self.last_parent_check > IndexingCallback.PARENT_CHECK_INTERVAL:
        #         try:
        #             # this is unintuitive, but it checks if the parent pid is still running
        #             os.kill(self.parent_pid, 0)
        #         except Exception:
        #             logger.exception("IndexingCallback - parent pid check exceptioned")
        #             raise
        #         self.last_parent_check = now

        try:
            current_time = time.monotonic()
            if current_time - self.last_lock_monotonic >= (
                CELERY_VESPA_SYNC_BEAT_LOCK_TIMEOUT / 4
            ):
                self.redis_lock.reacquire()
                self.last_lock_reacquire = datetime.now(timezone.utc)
                self.last_lock_monotonic = time.monotonic()

            self.last_tag = tag
        except LockError:
            logger.exception(
                f"IndexingCallback - lock.reacquire exceptioned: "
                f"lock_timeout={self.redis_lock.timeout} "
                f"start={self.started} "
                f"last_tag={self.last_tag} "
                f"last_reacquired={self.last_lock_reacquire} "
                f"now={datetime.now(timezone.utc)}"
            )

            redis_lock_dump(self.redis_lock, self.redis_client)
            raise

        self.redis_client.incrby(self.generator_progress_key, amount)


def get_unfenced_index_attempt_ids(db_session: Session, r: redis.Redis) -> list[int]:
    """
    获取没有栅栏保护的索引任务ID列表
    
    Args:
        db_session: 数据库会话
        r: Redis客户端
        
    Returns:
        list[int]: 无栅栏保护的索引任务ID列表
    
    Note:
        Unfenced = attempt not in terminal state and fence does not exist.
        无栅栏 = 索引任务不在终止状态且没有栅栏存在
    """
    unfenced_attempts: list[int] = []

    # inner/outer/inner double check pattern to avoid race conditions when checking for
    # bad state
    # inner = index_attempt in non terminal state
    # outer = r.fence_key down
    # 内/外/内双重检查模式以避免检查错误状态时的竞态条件
    # 内部 = index_attempt不在终止状态
    # 外部 = r.fence_key关闭

    # check the db for index attempts in a non terminal state
    attempts: list[IndexAttempt] = []
    attempts.extend(
        get_all_index_attempts_by_status(IndexingStatus.NOT_STARTED, db_session)
    )
    attempts.extend(
        get_all_index_attempts_by_status(IndexingStatus.IN_PROGRESS, db_session)
    )

    for attempt in attempts:
        fence_key = RedisConnectorIndex.fence_key_with_ids(
            attempt.connector_credential_pair_id, attempt.search_settings_id
        )

        # if the fence is down / doesn't exist, possible error but not confirmed
        if r.exists(fence_key):
            continue

        # Between the time the attempts are first looked up and the time we see the fence down,
        # the attempt may have completed and taken down the fence normally.

        # We need to double check that the index attempt is still in a non terminal state
        # and matches the original state, which confirms we are really in a bad state.
        attempt_2 = get_index_attempt(db_session, attempt.id)
        if not attempt_2:
            continue

        if attempt.status != attempt_2.status:
            continue

        unfenced_attempts.append(attempt.id)

    return unfenced_attempts


@shared_task(
    name=OnyxCeleryTask.CHECK_FOR_INDEXING,
    soft_time_limit=300,
    bind=True,
)
def check_for_indexing(self: Task, *, tenant_id: str | None) -> int | None:
    """一个轻量级任务，用于启动索引任务。偶尔也会验证现有状态以清理错误条件。
    
    Args:
        self: Celery Task 实例
        tenant_id: 租户ID
        
    Returns:
        int | None: 成功创建的任务数量，如果未创建任务则返回None
    """
    time_start = time.monotonic()

    tasks_created = 0
    locked = False
    redis_client = get_redis_client(tenant_id=tenant_id)

    # we need to use celery's redis client to access its redis data
    # (which lives on a different db number)
    # 我们需要使用celery的redis客户端来访问其redis数据(存储在不同的db编号中)
    redis_client_celery: Redis = self.app.broker_connection().channel().client  # type: ignore

    lock_beat: RedisLock = redis_client.lock(
        OnyxRedisLocks.CHECK_INDEXING_BEAT_LOCK,
        timeout=CELERY_VESPA_SYNC_BEAT_LOCK_TIMEOUT,
    )

    try:
        # these tasks should never overlap
        # 这些任务不应该重叠
        if not lock_beat.acquire(blocking=False):
            return None

        locked = True

        # check for search settings swap
        with get_session_with_tenant(tenant_id=tenant_id) as db_session:
            old_search_settings = check_index_swap(db_session=db_session)
            current_search_settings = get_current_search_settings(db_session)
            # So that the first time users aren't surprised by really slow speed of first
            # batch of documents indexed
            # 这样首次用户就不会对第一批文档的索引速度太慢感到惊讶
            if current_search_settings.provider_type is None and not MULTI_TENANT:
                if old_search_settings:
                    embedding_model = EmbeddingModel.from_db_model(
                        search_settings=current_search_settings,
                        server_host=INDEXING_MODEL_SERVER_HOST,
                        server_port=INDEXING_MODEL_SERVER_PORT,
                    )

                    # only warm up if search settings were changed
                    # 仅在搜索设置改变时预热
                    warm_up_bi_encoder(
                        embedding_model=embedding_model,
                    )

        # gather cc_pair_ids
        # 收集cc_pair_ids
        cc_pair_ids: list[int] = []
        with get_session_with_tenant(tenant_id) as db_session:
            lock_beat.reacquire()
            cc_pairs = fetch_connector_credential_pairs(db_session)
            for cc_pair_entry in cc_pairs:
                cc_pair_ids.append(cc_pair_entry.id)

        # kick off index attempts
        # 启动索引尝试
        for cc_pair_id in cc_pair_ids:
            lock_beat.reacquire()

            redis_connector = RedisConnector(tenant_id, cc_pair_id)
            with get_session_with_tenant(tenant_id) as db_session:
                search_settings_list: list[SearchSettings] = get_active_search_settings(
                    db_session
                )
                for search_settings_instance in search_settings_list:
                    redis_connector_index = redis_connector.new_index(
                        search_settings_instance.id
                    )
                    if redis_connector_index.fenced:
                        continue

                    cc_pair = get_connector_credential_pair_from_id(
                        cc_pair_id, db_session
                    )
                    if not cc_pair:
                        continue

                    last_attempt = get_last_attempt_for_cc_pair(
                        cc_pair.id, search_settings_instance.id, db_session
                    )

                    search_settings_primary = False
                    if search_settings_instance.id == search_settings_list[0].id:
                        search_settings_primary = True

                    if not _should_index(
                        cc_pair=cc_pair,
                        last_index=last_attempt,
                        search_settings_instance=search_settings_instance,
                        search_settings_primary=search_settings_primary,
                        secondary_index_building=len(search_settings_list) > 1,
                        db_session=db_session,
                    ):
                        continue

                    reindex = False
                    if search_settings_instance.id == search_settings_list[0].id:
                        # the indexing trigger is only checked and cleared with the primary search settings
                        # 索引触发器仅与主要搜索设置一起检查和清除
                        if cc_pair.indexing_trigger is not None:
                            if cc_pair.indexing_trigger == IndexingMode.REINDEX:
                                reindex = True

                            task_logger.info(
                                f"Connector indexing manual trigger detected: "
                                f"cc_pair={cc_pair.id} "
                                f"search_settings={search_settings_instance.id} "
                                f"indexing_mode={cc_pair.indexing_trigger}"
                            )

                            mark_ccpair_with_indexing_trigger(
                                cc_pair.id, None, db_session
                            )

                    # using a task queue and only allowing one task per cc_pair/search_setting
                    # prevents us from starving out certain attempts
                    # 使用任务队列并且每个cc_pair/search_setting只允许一个任务
                    # 防止某些尝试被饿死
                    attempt_id = try_creating_indexing_task(
                        self.app,
                        cc_pair,
                        search_settings_instance,
                        reindex,
                        db_session,
                        redis_client,
                        tenant_id,
                    )
                    if attempt_id:
                        task_logger.info(
                            f"Connector indexing queued: "
                            f"index_attempt={attempt_id} "
                            f"cc_pair={cc_pair.id} "
                            f"search_settings={search_settings_instance.id}"
                        )
                        tasks_created += 1

        # Fail any index attempts in the DB that don't have fences
        # This shouldn't ever happen!
        # 将数据库中没有栅栏的索引尝试标记为失败
        # 这种情况不应该发生!
        with get_session_with_tenant(tenant_id) as db_session:
            lock_beat.reacquire()
            unfenced_attempt_ids = get_unfenced_index_attempt_ids(
                db_session, redis_client
            )
            for attempt_id in unfenced_attempt_ids:
                lock_beat.reacquire()

                attempt = get_index_attempt(db_session, attempt_id)
                if not attempt:
                    continue

                failure_reason = (
                    f"Unfenced index attempt found in DB: "
                    f"index_attempt={attempt.id} "
                    f"cc_pair={attempt.connector_credential_pair_id} "
                    f"search_settings={attempt.search_settings_id}"
                )
                task_logger.error(failure_reason)
                mark_attempt_failed(
                    attempt.id, db_session, failure_reason=failure_reason
                )

        # we want to run this less frequently than the overall task
        # 我们希望这个检查的运行频率低于整体任务
        if not redis_client.exists(OnyxRedisSignals.VALIDATE_INDEXING_FENCES):
            lock_beat.reacquire()
            # clear any indexing fences that don't have associated celery tasks in progress
            # tasks can be in the queue in redis, in reserved tasks (prefetched by the worker),
            # or be currently executing
            # 清除没有关联的正在进行的celery任务的索引栅栏
            # 任务可能在redis队列中、在保留任务中(被worker预取)或正在执行
            try:
                validate_indexing_fences(
                    tenant_id, self.app, redis_client, redis_client_celery, lock_beat
                )
            except Exception:
                task_logger.exception("Exception while validating indexing fences")

            redis_client.set(OnyxRedisSignals.VALIDATE_INDEXING_FENCES, 1, ex=60)

    except SoftTimeLimitExceeded:
        task_logger.info(
            "Soft time limit exceeded, task is being terminated gracefully."
        )
    except Exception:
        task_logger.exception("Unexpected exception during indexing check")
    finally:
        if locked:
            if lock_beat.owned():
                lock_beat.release()
            else:
                task_logger.error(
                    "check_for_indexing - Lock not owned on completion: "
                    f"tenant={tenant_id}"
                )
                redis_lock_dump(lock_beat, redis_client)

    time_elapsed = time.monotonic() - time_start
    task_logger.debug(f"check_for_indexing finished: elapsed={time_elapsed:.2f}")
    return tasks_created


def validate_indexing_fences(
    tenant_id: str | None,
    celery_app: Celery,
    r: Redis, 
    r_celery: Redis,
    lock_beat: RedisLock,
) -> None:
    """验证所有现有的索引栅栏
    
    Args:
        tenant_id: 租户ID
        celery_app: Celery应用实例 
        r: Redis客户端
        r_celery: Celery使用的Redis客户端
        lock_beat: Redis锁对象
    """
    reserved_indexing_tasks = celery_get_unacked_task_ids(
        OnyxCeleryQueues.CONNECTOR_INDEXING, r_celery
    )

    # validate all existing indexing jobs
    # 验证所有现有的索引任务
    for key_bytes in r.scan_iter(RedisConnectorIndex.FENCE_PREFIX + "*"):
        lock_beat.reacquire()
        with get_session_with_tenant(tenant_id) as db_session:
            validate_indexing_fence(
                tenant_id,
                key_bytes,
                reserved_indexing_tasks,
                r_celery,
                db_session,
            )
    return


def validate_indexing_fence(
    tenant_id: str | None,
    key_bytes: bytes,
    reserved_tasks: set[str],
    r_celery: Redis,
    db_session: Session,
) -> None:
    """检查索引栅栏设置但相关celery任务不存在的错误情况。
    当索引工作进程崩溃或被终止时可能发生这种情况。
    处于这种错误状态意味着如果没有帮助，栅栏将永远不会清除，因此该函数提供帮助。

    工作原理:
    1. 此函数在以下条件下使用5分钟TTL更新活动信号:
       1.2. 当在redis队列中看到任务时
       1.3. 当在保留/预取列表中看到任务
    
    2. 在外部,活动信号在以下情况下更新:
       2.1. 创建栅栏时
       2.2. 索引看门狗检查生成的任务时
    
    3. TTL允许我们在栅栏启动和任务开始执行时通过过渡期
    
    更多TTL说明:似乎不可能准确查询Celery任务是否在队列中或正在执行。
    1. 未知任务ID总是返回PENDING状态
    2. 可以在Redis中检查任务ID,但在工作进程接收任务和实际开始执行之间,任务ID会消失
    
    Args:
        tenant_id: 租户ID
        key_bytes: 栅栏键的字节数据 
        reserved_tasks: 保留任务ID集合
        r_celery: Celery Redis客户端
        db_session: 数据库会话
    """
    # if the fence doesn't exist, there's nothing to do
    # 如果栅栏不存在,就没什么可做的
    fence_key = key_bytes.decode("utf-8")
    composite_id = RedisConnector.get_id_from_fence_key(fence_key)
    if composite_id is None:
        task_logger.warning(
            f"validate_indexing_fence - could not parse composite_id from {fence_key}"
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

    # check to see if the fence/payload exists
    # 检查栅栏/负载是否存在
    if not redis_connector_index.fenced:
        return

    payload = redis_connector_index.payload
    if not payload:
        return

    # OK, there's actually something for us to validate
    # 好的,确实有东西需要我们验证

    if payload.celery_task_id is None:
        # the fence is just barely set up.
        # 栅栏刚刚建立
        if redis_connector_index.active():
            return

        # it would be odd to get here as there isn't that much that can go wrong during
        # initial fence setup, but it's still worth making sure we can recover
        logger.info(
            f"validate_indexing_fence - Resetting fence in basic state without any activity: fence={fence_key}"
        )
        redis_connector_index.reset()
        return

    found = celery_find_task(
        payload.celery_task_id, OnyxCeleryQueues.CONNECTOR_INDEXING, r_celery
    )
    if found:
        # the celery task exists in the redis queue
        # celery任务存在于redis队列中
        redis_connector_index.set_active()
        return

    if payload.celery_task_id in reserved_tasks:
        # the celery task was prefetched and is reserved within the indexing worker
        # celery任务已被预取并在索引worker中保留
        redis_connector_index.set_active()
        return

    # we may want to enable this check if using the active task list somehow isn't good enough
    # 如果使用活动任务列表somehow不够好,我们可能想启用这个检查

    # if redis_connector_index.generator_locked():
    #     logger.info(f"{payload.celery_task_id} is currently executing.")

    # if we get here, we didn't find any direct indication that the associated celery tasks exist,
    # but they still might be there due to gaps in our ability to check states during transitions
    # Checking the active signal safeguards us against these transition periods
    # (which has a duration that allows us to bridge those gaps)
    # 如果到达这里,我们没有找到相关celery任务存在的直接迹象,
    # 但由于我们在转换期间检查状态的能力存在空隙,它们可能仍然存在
    # 检查活动信号可以在这些过渡期间保护我们
    # (这个持续时间允许我们桥接那些空隙)
    if redis_connector_index.active():
        return

    # celery tasks don't exist and the active signal has expired, possibly due to a crash. Clean it up.
    logger.warning(
        f"validate_indexing_fence - Resetting fence because no associated celery tasks were found: "
        f"index_attempt={payload.index_attempt_id} "
        f"cc_pair={cc_pair_id} "
        f"search_settings={search_settings_id} "
        f"fence={fence_key}"
    )
    if payload.index_attempt_id:
        try:
            mark_attempt_failed(
                payload.index_attempt_id,
                db_session,
                "validate_indexing_fence - Canceling index attempt due to missing celery tasks",
            )
        except Exception:
            logger.exception(
                "validate_indexing_fence - Exception while marking index attempt as failed."
            )

    redis_connector_index.reset()
    return


def _should_index(
    cc_pair: ConnectorCredentialPair,
    last_index: IndexAttempt | None,
    search_settings_instance: SearchSettings,
    search_settings_primary: bool,
    secondary_index_building: bool,
    db_session: Session,
) -> bool:
    """检查各种全局设置和过去的索引尝试,以确定是否应该尝试为cc pair/search setting组合启动索引
    
    注意:不在此处处理防止与当前运行的任务重叠等战术性检查
    
    Args:
        cc_pair: 连接器凭证对
        last_index: 上次索引尝试,如果没有则为None  
        search_settings_instance: 搜索设置实例
        search_settings_primary: 是否为主搜索设置
        secondary_index_building: 是否在构建次要索引
        db_session: 数据库会话
        
    Returns:
        bool: 如果应该尝试索引返回True,否则返回False
    """
    connector = cc_pair.connector

    # uncomment for debugging
    # task_logger.info(f"_should_index: "
    #                  f"cc_pair={cc_pair.id} "
    #                  f"connector={cc_pair.connector_id} "
    #                  f"refresh_freq={connector.refresh_freq}")

    # don't kick off indexing for `NOT_APPLICABLE` sources
    if connector.source == DocumentSource.NOT_APPLICABLE:
        return False

    # User can still manually create single indexing attempts via the UI for the
    # currently in use index
    if DISABLE_INDEX_UPDATE_ON_SWAP:
        if (
            search_settings_instance.status == IndexModelStatus.PRESENT
            and secondary_index_building
        ):
            return False

    # When switching over models, always index at least once
    if search_settings_instance.status == IndexModelStatus.FUTURE:
        if last_index:
            # No new index if the last index attempt succeeded
            # Once is enough. The model will never be able to swap otherwise.
            if last_index.status == IndexingStatus.SUCCESS:
                return False

            # No new index if the last index attempt is waiting to start
            if last_index.status == IndexingStatus.NOT_STARTED:
                return False

            # No new index if the last index attempt is running
            if last_index.status == IndexingStatus.IN_PROGRESS:
                return False
        else:
            if (
                connector.id == 0 or connector.source == DocumentSource.INGESTION_API
            ):  # Ingestion API
                return False
        return True

    # If the connector is paused or is the ingestion API, don't index
    # NOTE: during an embedding model switch over, the following logic
    # is bypassed by the above check for a future model
    if (
        not cc_pair.status.is_active()
        or connector.id == 0
        or connector.source == DocumentSource.INGESTION_API
    ):
        return False

    if search_settings_primary:
        if cc_pair.indexing_trigger is not None:
            # if a manual indexing trigger is on the cc pair, honor it for primary search settings
            return True

    # if no attempt has ever occurred, we should index regardless of refresh_freq
    # 如果从未进行过尝试,我们应该不考虑refresh_freq直接进行索引
    if not last_index:
        return True

    if connector.refresh_freq is None:
        return False

    current_db_time = get_db_current_time(db_session)
    time_since_index = current_db_time - last_index.time_updated
    if time_since_index.total_seconds() < connector.refresh_freq:
        return False

    return True


def try_creating_indexing_task(
    celery_app: Celery,
    cc_pair: ConnectorCredentialPair,
    search_settings: SearchSettings,
    reindex: bool,
    db_session: Session,
    r: Redis,
    tenant_id: str | None,
) -> int | None:
    """检查是否存在应该阻止创建索引任务的条件,然后创建任务
    
    不检查与调度相关的条件,因为此函数用于立即触发索引
    
    Args:
        celery_app: Celery应用实例
        cc_pair: 连接器凭证对
        search_settings: 搜索设置 
        reindex: 是否重新索引
        db_session: 数据库会话
        r: Redis客户端
        tenant_id: 租户ID
        
    Returns:
        int | None: 创建的索引尝试ID,如果创建失败则返回None
    """
    LOCK_TIMEOUT = 30
    index_attempt_id: int | None = None

    # we need to serialize any attempt to trigger indexing since it can be triggered
    # either via celery beat or manually (API call)
    lock: RedisLock = r.lock(
        DANSWER_REDIS_FUNCTION_LOCK_PREFIX + "try_creating_indexing_task",
        timeout=LOCK_TIMEOUT,
    )

    acquired = lock.acquire(blocking_timeout=LOCK_TIMEOUT / 2)
    if not acquired:
        return None

    try:
        redis_connector = RedisConnector(tenant_id, cc_pair.id)
        redis_connector_index = redis_connector.new_index(search_settings.id)

        # skip if already indexing
        if redis_connector_index.fenced:
            return None

        # skip indexing if the cc_pair is deleting
        if redis_connector.delete.fenced:
            return None

        db_session.refresh(cc_pair)
        if cc_pair.status == ConnectorCredentialPairStatus.DELETING:
            return None

        # add a long running generator task to the queue
        redis_connector_index.generator_clear()

        # set a basic fence to start
        # 设置基本栅栏开始
        payload = RedisConnectorIndexPayload(
            index_attempt_id=None,
            started=None,
            submitted=datetime.now(timezone.utc),
            celery_task_id=None,
        )

        redis_connector_index.set_active()
        redis_connector_index.set_fence(payload)

        # create the index attempt for tracking purposes
        # code elsewhere checks for index attempts without an associated redis key
        # and cleans them up
        # therefore we must create the attempt and the task after the fence goes up
        # 创建索引尝试用于跟踪目的
        # 其他地方的代码会检查没有关联redis key的索引尝试并清理它们
        # 因此我们必须在栅栏建立后创建尝试和任务
        index_attempt_id = create_index_attempt(
            cc_pair.id,
            search_settings.id,
            from_beginning=reindex,
            db_session=db_session,
        )

        custom_task_id = redis_connector_index.generate_generator_task_id()

        # when the task is sent, we have yet to finish setting up the fence
        # therefore, the task must contain code that blocks until the fence is ready
        # 当任务被发送时,我们还没有完成栅栏的设置
        # 因此,任务必须包含阻塞代码直到栅栏准备就绪
        result = celery_app.send_task(
            OnyxCeleryTask.CONNECTOR_INDEXING_PROXY_TASK,
            kwargs=dict(
                index_attempt_id=index_attempt_id,
                cc_pair_id=cc_pair.id,
                search_settings_id=search_settings.id,
                tenant_id=tenant_id,
            ),
            queue=OnyxCeleryQueues.CONNECTOR_INDEXING,
            task_id=custom_task_id,
            priority=OnyxCeleryPriority.MEDIUM,
        )
        if not result:
            raise RuntimeError("send_task for connector_indexing_proxy_task failed.")

        # now fill out the fence with the rest of the data
        # 现在用剩余数据填充栅栏
        redis_connector_index.set_active()

        payload.index_attempt_id = index_attempt_id
        payload.celery_task_id = result.id
        redis_connector_index.set_fence(payload)
    except Exception:
        task_logger.exception(
            f"try_creating_indexing_task - Unexpected exception: "
            f"cc_pair={cc_pair.id} "
            f"search_settings={search_settings.id}"
        )

        if index_attempt_id is not None:
            delete_index_attempt(db_session, index_attempt_id)
        redis_connector_index.set_fence(None)
        return None
    finally:
        if lock.owned():
            lock.release()

    return index_attempt_id


@shared_task(
    name=OnyxCeleryTask.CONNECTOR_INDEXING_PROXY_TASK,
    bind=True,
    acks_late=False,
    track_started=True,
)
def connector_indexing_proxy_task(
    self: Task,
    index_attempt_id: int,
    cc_pair_id: int,
    search_settings_id: int,
    tenant_id: str | None,
) -> None:
    """索引任务的代理函数。由于celery任务使用fork但fork可能不稳定，所以通过此函数代理到spawned任务。
    
    Args:
        self: Celery Task对象 
        index_attempt_id: 索引尝试ID
        cc_pair_id: 连接器凭证对ID
        search_settings_id: 搜索设置ID 
        tenant_id: str | None
    """
    task_logger.info(
        f"Indexing watchdog - starting: attempt={index_attempt_id} "
        f"cc_pair={cc_pair_id} "
        f"search_settings={search_settings_id}"
    )

    if not self.request.id:
        task_logger.error("self.request.id is None!")

    client = SimpleJobClient()

    job = client.submit(
        connector_indexing_task_wrapper,
        index_attempt_id,
        cc_pair_id,
        search_settings_id,
        tenant_id,
        global_version.is_ee_version(),
        pure=False,
    )

    if not job:
        task_logger.info(
            f"Indexing watchdog - spawn failed: attempt={index_attempt_id} "
            f"cc_pair={cc_pair_id} "
            f"search_settings={search_settings_id}"
        )
        return

    task_logger.info(
        f"Indexing watchdog - spawn succeeded: attempt={index_attempt_id} "
        f"cc_pair={cc_pair_id} "
        f"search_settings={search_settings_id}"
    )

    redis_connector = RedisConnector(tenant_id, cc_pair_id)
    redis_connector_index = redis_connector.new_index(search_settings_id)

    while True:
        sleep(5)

        # renew active signal
        redis_connector_index.set_active()

        # if the job is done, clean up and break
        if job.done():
            try:
                if job.status == "error":
                    ignore_exitcode = False

                    exit_code: int | None = None
                    if job.process:
                        exit_code = job.process.exitcode

                    # seeing odd behavior where spawned tasks usually return exit code 1 in the cloud,
                    # even though logging clearly indicates successful completion
                    # to work around this, we ignore the job error state if the completion signal is OK
                    status_int = redis_connector_index.get_completion()
                    if status_int:
                        status_enum = HTTPStatus(status_int)
                        if status_enum == HTTPStatus.OK:
                            ignore_exitcode = True

                    if not ignore_exitcode:
                        raise RuntimeError("Spawned task exceptioned.")

                    task_logger.warning(
                        "Indexing watchdog - spawned task has non-zero exit code "
                        "but completion signal is OK. Continuing...: "
                        f"attempt={index_attempt_id} "
                        f"tenant={tenant_id} "
                        f"cc_pair={cc_pair_id} "
                        f"search_settings={search_settings_id} "
                        f"exit_code={exit_code}"
                    )
            except Exception:
                task_logger.error(
                    "Indexing watchdog - spawned task exceptioned: "
                    f"attempt={index_attempt_id} "
                    f"tenant={tenant_id} "
                    f"cc_pair={cc_pair_id} "
                    f"search_settings={search_settings_id} "
                    f"exit_code={exit_code} "
                    f"error={job.exception()}"
                )

                raise
            finally:
                job.release()

            break

        # if a termination signal is detected, clean up and break
        if self.request.id and redis_connector_index.terminating(self.request.id):
            task_logger.warning(
                "Indexing watchdog - termination signal detected: "
                f"attempt={index_attempt_id} "
                f"cc_pair={cc_pair_id} "
                f"search_settings={search_settings_id}"
            )

            try:
                with get_session_with_tenant(tenant_id) as db_session:
                    mark_attempt_canceled(
                        index_attempt_id,
                        db_session,
                        "Connector termination signal detected",
                    )
            except Exception:
                # if the DB exceptions, we'll just get an unfriendly failure message
                # in the UI instead of the cancellation message
                # 如果数据库抛出异常,我们在UI中只会看到不友好的失败消息而不是取消消息
                logger.exception(
                    "Indexing watchdog - transient exception marking index attempt as canceled: "
                    f"attempt={index_attempt_id} "
                    f"tenant={tenant_id} "
                    f"cc_pair={cc_pair_id} "
                    f"search_settings={search_settings_id}"
                )

            job.cancel()
            break

        # if the spawned task is still running, restart the check once again
        # if the index attempt is not in a finished status
        # 如果生成的任务仍在运行,且索引尝试未处于完成状态
        # 则重新开始检查
        try:
            with get_session_with_tenant(tenant_id) as db_session:
                index_attempt = get_index_attempt(
                    db_session=db_session, index_attempt_id=index_attempt_id
                )

                if not index_attempt:
                    continue

                if not index_attempt.is_finished():
                    continue
        except Exception:
            # if the DB exceptioned, just restart the check.
            # polling the index attempt status doesn't need to be strongly consistent
            logger.exception(
                "Indexing watchdog - transient exception looking up index attempt: "
                f"attempt={index_attempt_id} "
                f"tenant={tenant_id} "
                f"cc_pair={cc_pair_id} "
                f"search_settings={search_settings_id}"
            )
            continue

    task_logger.info(
        f"Indexing watchdog - finished: attempt={index_attempt_id} "
        f"cc_pair={cc_pair_id} "
        f"search_settings={search_settings_id}"
    )
    return


def connector_indexing_task_wrapper(
    index_attempt_id: int,
    cc_pair_id: int,
    search_settings_id: int,
    tenant_id: str | None,
    is_ee: bool,
) -> int | None:
    """包装connector_indexing_task以在重新抛出异常前记录日志
    
    Args:
        index_attempt_id: 索引尝试ID
        cc_pair_id: 连接器凭证对ID
        search_settings_id: 搜索设置ID
        tenant_id: 租户ID
        is_ee: 是否为企业版
        
    Returns:
        int | None: 索引文档数量或None（如果任务未运行）
        
    Notes:
        由于云环境中的bug,spawn的任务返回退出码1。
        不幸的是异常也返回退出码1,所以仅抛出异常无法提供有效信息。
        使用退出码255可以区分正常退出和异常。
    """
    result: int | None = None

    try:
        result = connector_indexing_task(
            index_attempt_id,
            cc_pair_id,
            search_settings_id,
            tenant_id,
            is_ee,
        )
    except Exception:
        logger.exception(
            f"connector_indexing_task exceptioned: "
            f"tenant={tenant_id} "
            f"index_attempt={index_attempt_id} "
            f"cc_pair={cc_pair_id} "
            f"search_settings={search_settings_id}"
        )

        # There is a cloud related bug outside of our code
        # where spawned tasks return with an exit code of 1.
        # Unfortunately, exceptions also return with an exit code of 1,
        # so just raising an exception isn't informative
        # Exiting with 255 makes it possible to distinguish between normal exits
        # and exceptions.
        sys.exit(255)

    return result


def connector_indexing_task(
    index_attempt_id: int,
    cc_pair_id: int,
    search_settings_id: int,
    tenant_id: str | None,
    is_ee: bool,
) -> int | None:
    """索引任务的主体函数。对于一个cc pair,此任务从源拉取所有文档ID,
    将这些ID与本地存储的文档进行比较,删除最近拉取的文档ID列表中缺失的所有本地存储ID。
    
    Args:
        index_attempt_id: 索引尝试ID
        cc_pair_id: 连接器凭证对ID 
        search_settings_id: 搜索设置ID
        tenant_id: 租户ID
        is_ee: 是否为企业版
        
    Returns:
        int | None: 如果任务未运行（可能由于冲突）返回None,
                   否则返回一个大于等于0的整数代表已索引的文档数量
        
    Notes:
        acks_late必须设置为False。否则,celery的可见性超时会导致
        任何运行时间超过超时的任务被broker重新分发。
        这似乎没有好的解决方法,所以我们需要手动处理重新分发。
        
        如果此任务抛出异常,主worker将检测到任务转换为"READY"状态
        但generator_complete_key不存在。这将导致主worker中止索引尝试并清理。
    """
    # Since connector_indexing_proxy_task spawns a new process using this function as
    # the entrypoint, we init Sentry here.
    if SENTRY_DSN:
        sentry_sdk.init(
            dsn=SENTRY_DSN,
            traces_sample_rate=0.1,
        )
        logger.info("Sentry initialized")
    else:
        logger.debug("Sentry DSN not provided, skipping Sentry initialization")

    logger.info(
        f"Indexing spawned task starting: "
        f"attempt={index_attempt_id} "
        f"tenant={tenant_id} "
        f"cc_pair={cc_pair_id} "
        f"search_settings={search_settings_id}"
    )

    attempt_found = False
    n_final_progress: int | None = None

    redis_connector = RedisConnector(tenant_id, cc_pair_id)
    redis_connector_index = redis_connector.new_index(search_settings_id)

    r = get_redis_client(tenant_id=tenant_id)

    if redis_connector.delete.fenced:
        raise RuntimeError(
            f"Indexing will not start because connector deletion is in progress: "
            f"attempt={index_attempt_id} "
            f"cc_pair={cc_pair_id} "
            f"fence={redis_connector.delete.fence_key}"
        )

    if redis_connector.stop.fenced:
        raise RuntimeError(
            f"Indexing will not start because a connector stop signal was detected: "
            f"attempt={index_attempt_id} "
            f"cc_pair={cc_pair_id} "
            f"fence={redis_connector.stop.fence_key}"
        )

    # this wait is needed to avoid a race condition where
    # the primary worker sends the task and it is immediately executed
    # before the primary worker can finalize the fence
    start = time.monotonic()
    while True:
        if time.monotonic() - start > CELERY_TASK_WAIT_FOR_FENCE_TIMEOUT:
            raise ValueError(
                f"connector_indexing_task - timed out waiting for fence to be ready: "
                f"fence={redis_connector.permissions.fence_key}"
            )

        if not redis_connector_index.fenced:  # The fence must exist
            raise ValueError(
                f"connector_indexing_task - fence not found: fence={redis_connector_index.fence_key}"
            )

        payload = redis_connector_index.payload  # The payload must exist
        if not payload:
            raise ValueError("connector_indexing_task: payload invalid or not found")

        if payload.index_attempt_id is None or payload.celery_task_id is None:
            logger.info(
                f"connector_indexing_task - Waiting for fence: fence={redis_connector_index.fence_key}"
            )
            sleep(1)
            continue

        if payload.index_attempt_id != index_attempt_id:
            raise ValueError(
                f"connector_indexing_task - id mismatch. Task may be left over from previous run.: "
                f"task_index_attempt={index_attempt_id} "
                f"payload_index_attempt={payload.index_attempt_id}"
            )

        logger.info(
            f"connector_indexing_task - Fence found, continuing...: fence={redis_connector_index.fence_key}"
        )
        break

    # set thread_local=False since we don't control what thread the indexing/pruning
    # might run our callback with
    # 设置thread_local=False因为我们无法控制索引/清理可能在哪个线程中运行我们的回调
    lock: RedisLock = r.lock(
        redis_connector_index.generator_lock_key,
        timeout=CELERY_INDEXING_LOCK_TIMEOUT,
        thread_local=False,
    )

    acquired = lock.acquire(blocking=False)
    if not acquired:
        logger.warning(
            f"Indexing task already running, exiting...: "
            f"index_attempt={index_attempt_id} "
            f"cc_pair={cc_pair_id} "
            f"search_settings={search_settings_id}"
        )
        return None

    payload.started = datetime.now(timezone.utc)
    redis_connector_index.set_fence(payload)

    try:
        with get_session_with_tenant(tenant_id) as db_session:
            attempt = get_index_attempt(db_session, index_attempt_id)
            if not attempt:
                raise ValueError(
                    f"Index attempt not found: index_attempt={index_attempt_id}"
                )
            attempt_found = True

            cc_pair = get_connector_credential_pair_from_id(
                cc_pair_id=cc_pair_id,
                db_session=db_session,
            )

            if not cc_pair:
                raise ValueError(f"cc_pair not found: cc_pair={cc_pair_id}")

            if not cc_pair.connector:
                raise ValueError(
                    f"Connector not found: cc_pair={cc_pair_id} connector={cc_pair.connector_id}"
                )

            if not cc_pair.credential:
                raise ValueError(
                    f"Credential not found: cc_pair={cc_pair_id} credential={cc_pair.credential_id}"
                )

        # define a callback class
        # 定义回调类
        callback = IndexingCallback(
            os.getppid(),
            redis_connector.stop.fence_key,
            redis_connector_index.generator_progress_key,
            lock,
            r,
        )

        logger.info(
            f"Indexing spawned task running entrypoint: attempt={index_attempt_id} "
            f"tenant={tenant_id} "
            f"cc_pair={cc_pair_id} "
            f"search_settings={search_settings_id}"
        )

        # This is where the heavy/real work happens
        # 这里是进行实际重要工作的地方
        # This is where the heavy/real work happens
        run_indexing_entrypoint(
            index_attempt_id,
            tenant_id,
            cc_pair_id,
            is_ee,
            callback=callback,
        )

        # get back the total number of indexed docs and return it
        # 获取已索引文档的总数并返回
        n_final_progress = redis_connector_index.get_progress()
        redis_connector_index.set_generator_complete(HTTPStatus.OK.value)
    except Exception as e:
        logger.exception(
            f"Indexing spawned task failed: attempt={index_attempt_id} "
            f"tenant={tenant_id} "
            f"cc_pair={cc_pair_id} "
            f"search_settings={search_settings_id}"
        )
        if attempt_found:
            try:
                with get_session_with_tenant(tenant_id) as db_session:
                    mark_attempt_failed(
                        index_attempt_id, db_session, failure_reason=str(e)
                    )
            except Exception:
                logger.exception(
                    "Indexing watchdog - transient exception looking up index attempt: "
                    f"attempt={index_attempt_id} "
                    f"tenant={tenant_id} "
                    f"cc_pair={cc_pair_id} "
                    f"search_settings={search_settings_id}"
                )

        raise e
    finally:
        if lock.owned():
            lock.release()

    logger.info(
        f"Indexing spawned task finished: attempt={index_attempt_id} "
        f"cc_pair={cc_pair_id} "
        f"search_settings={search_settings_id}"
    )
    return n_final_progress
