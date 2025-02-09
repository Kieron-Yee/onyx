"""
此文件是Celery主要工作进程的配置和初始化文件。
主要功能包括：
1. 配置和初始化Celery应用
2. 设置各种信号处理器
3. 管理Redis锁和清理操作
4. 处理工作进程的启动和关闭
5. 配置定期任务
"""

import logging
import multiprocessing
from typing import Any
from typing import cast

from celery import bootsteps  # type: ignore
from celery import Celery
from celery import signals
from celery import Task
from celery.exceptions import WorkerShutdown
from celery.signals import celeryd_init
from celery.signals import worker_init
from celery.signals import worker_ready
from celery.signals import worker_shutdown
from redis.lock import Lock as RedisLock

import onyx.background.celery.apps.app_base as app_base
from onyx.background.celery.apps.app_base import task_logger
from onyx.background.celery.celery_utils import celery_is_worker_primary
from onyx.background.celery.tasks.indexing.tasks import (
    get_unfenced_index_attempt_ids,
)
from onyx.configs.constants import CELERY_PRIMARY_WORKER_LOCK_TIMEOUT
from onyx.configs.constants import OnyxRedisLocks
from onyx.configs.constants import POSTGRES_CELERY_WORKER_PRIMARY_APP_NAME
from onyx.db.engine import get_session_with_default_tenant
from onyx.db.engine import SqlEngine
from onyx.db.index_attempt import get_index_attempt
from onyx.db.index_attempt import mark_attempt_canceled
from onyx.redis.redis_connector_credential_pair import RedisConnectorCredentialPair
from onyx.redis.redis_connector_delete import RedisConnectorDelete
from onyx.redis.redis_connector_doc_perm_sync import RedisConnectorPermissionSync
from onyx.redis.redis_connector_ext_group_sync import RedisConnectorExternalGroupSync
from onyx.redis.redis_connector_index import RedisConnectorIndex
from onyx.redis.redis_connector_prune import RedisConnectorPrune
from onyx.redis.redis_connector_stop import RedisConnectorStop
from onyx.redis.redis_document_set import RedisDocumentSet
from onyx.redis.redis_pool import get_redis_client
from onyx.redis.redis_usergroup import RedisUserGroup
from onyx.utils.logger import setup_logger
from shared_configs.configs import MULTI_TENANT

logger = setup_logger()

celery_app = Celery(__name__)
celery_app.config_from_object("onyx.background.celery.configs.primary")


@signals.task_prerun.connect
def on_task_prerun(
    sender: Any | None = None,
    task_id: str | None = None,
    task: Task | None = None,
    args: tuple | None = None,
    kwargs: dict | None = None,
    **kwds: Any,
) -> None:
    """
    任务执行前的信号处理函数
    
    参数:
        sender: 信号发送者
        task_id: 任务ID
        task: 任务对象
        args: 位置参数
        kwargs: 关键字参数
        kwds: 额外关键字参数
    """
    app_base.on_task_prerun(sender, task_id, task, args, kwargs, **kwds)


@signals.task_postrun.connect
def on_task_postrun(
    sender: Any | None = None,
    task_id: str | None = None,
    task: Task | None = None,
    args: tuple | None = None,
    kwargs: dict | None = None,
    retval: Any | None = None,
    state: str | None = None,
    **kwds: Any,
) -> None:
    """
    任务执行后的信号处理函数
    
    参数:
        sender: 信号发送者
        task_id: 任务ID
        task: 任务对象
        args: 位置参数
        kwargs: 关键字参数
        retval: 返回值
        state: 任务状态
        kwds: 额外关键字参数
    """
    app_base.on_task_postrun(sender, task_id, task, args, kwargs, retval, state, **kwds)


@celeryd_init.connect
def on_celeryd_init(sender: Any = None, conf: Any = None, **kwargs: Any) -> None:
    app_base.on_celeryd_init(sender, conf, **kwargs)


@worker_init.connect
def on_worker_init(sender: Any, **kwargs: Any) -> None:
    logger.info("worker_init signal received.")
    logger.info(f"Multiprocessing start method: {multiprocessing.get_start_method()}")

    SqlEngine.set_app_name(POSTGRES_CELERY_WORKER_PRIMARY_APP_NAME)
    SqlEngine.init_engine(pool_size=8, max_overflow=0)

    app_base.wait_for_redis(sender, **kwargs)
    app_base.wait_for_db(sender, **kwargs)
    app_base.wait_for_vespa(sender, **kwargs)

    # Less startup checks in multi-tenant case
    # 多租户情况下减少启动检查
    if MULTI_TENANT:
        return

    logger.info("Running as the primary celery worker.")

    # This is singleton work that should be done on startup exactly once
    # by the primary worker. This is unnecessary in the multi tenant scenario
    # 这是在启动时由主工作进程执行的单例工作。在多租户场景下不需要
    r = get_redis_client(tenant_id=None)

    # Log the role and slave count - being connected to a slave or slave count > 0 could be problematic
    # 记录角色和从节点数量 - 连接到从节点或从节点数量大于0可能会有问题
    info: dict[str, Any] = cast(dict, r.info("replication"))
    role: str = cast(str, info.get("role"))
    connected_slaves: int = info.get("connected_slaves", 0)

    logger.info(
        f"Redis INFO REPLICATION: role={role} connected_slaves={connected_slaves}"
    )

    # For the moment, we're assuming that we are the only primary worker
    # that should be running.
    # TODO: maybe check for or clean up another zombie primary worker if we detect it
    # 目前，我们假设只有一个主工作进程在运行
    # TODO: 如果检测到其他僵尸主工作进程，可能需要检查或清理
    r.delete(OnyxRedisLocks.PRIMARY_WORKER)

    # this process wide lock is taken to help other workers start up in order.
    # it is planned to use this lock to enforce singleton behavior on the primary
    # worker, since the primary worker does redis cleanup on startup, but this isn't
    # implemented yet.
    # 这个进程范围的锁用于帮助其他工作进程按顺序启动
    # 计划使用此锁来强制主工作进程的单例行为，因为主工作进程在启动时执行redis清理，但这尚未实现

    # set thread_local=False since we don't control what thread the periodic task might
    # reacquire the lock with
    # 设置thread_local=False，因为我们无法控制周期任务可能在哪个线程中重新获取锁
    lock: RedisLock = r.lock(
        OnyxRedisLocks.PRIMARY_WORKER,
        timeout=CELERY_PRIMARY_WORKER_LOCK_TIMEOUT,
        thread_local=False,
    )

    logger.info("Primary worker lock: Acquire starting.")
    acquired = lock.acquire(blocking_timeout=CELERY_PRIMARY_WORKER_LOCK_TIMEOUT / 2)
    if acquired:
        logger.info("Primary worker lock: Acquire succeeded.")
    else:
        logger.error("Primary worker lock: Acquire failed!")
        raise WorkerShutdown("Primary worker lock could not be acquired!")

    # tacking on our own user data to the sender
    # 将我们自己的用户数据附加到发送者
    sender.primary_worker_lock = lock

    # As currently designed, when this worker starts as "primary", we reinitialize redis
    # to a clean state (for our purposes, anyway)
    # 按照当前设计，当这个工作进程作为"主"启动时，我们将redis重新初始化为干净状态（至少对我们的目的而言）
    r.delete(OnyxRedisLocks.CHECK_VESPA_SYNC_BEAT_LOCK)
    r.delete(OnyxRedisLocks.MONITOR_VESPA_SYNC_BEAT_LOCK)

    r.delete(RedisConnectorCredentialPair.get_taskset_key())
    r.delete(RedisConnectorCredentialPair.get_fence_key())

    RedisDocumentSet.reset_all(r)

    RedisUserGroup.reset_all(r)

    RedisConnectorDelete.reset_all(r)

    RedisConnectorPrune.reset_all(r)

    RedisConnectorIndex.reset_all(r)

    RedisConnectorStop.reset_all(r)

    RedisConnectorPermissionSync.reset_all(r)

    RedisConnectorExternalGroupSync.reset_all(r)

    # mark orphaned index attempts as failed
    # 将孤立的索引尝试标记为失败
    with get_session_with_default_tenant() as db_session:
        unfenced_attempt_ids = get_unfenced_index_attempt_ids(db_session, r)
        for attempt_id in unfenced_attempt_ids:
            attempt = get_index_attempt(db_session, attempt_id)
            if not attempt:
                continue

            failure_reason = (
                f"Canceling leftover index attempt found on startup: "
                f"index_attempt={attempt.id} "
                f"cc_pair={attempt.connector_credential_pair_id} "
                f"search_settings={attempt.search_settings_id}"
            )
            logger.warning(failure_reason)
            mark_attempt_canceled(attempt.id, db_session, failure_reason)


@worker_ready.connect
def on_worker_ready(sender: Any, **kwargs: Any) -> None:
    app_base.on_worker_ready(sender, **kwargs)


@worker_shutdown.connect
def on_worker_shutdown(sender: Any, **kwargs: Any) -> None:
    app_base.on_worker_shutdown(sender, **kwargs)


@signals.setup_logging.connect
def on_setup_logging(
    loglevel: Any, logfile: Any, format: Any, colorize: Any, **kwargs: Any
) -> None:
    app_base.on_setup_logging(loglevel, logfile, format, colorize, **kwargs)

    # this can be spammy, so just enable it in the cloud for now
    # 这可能会产生大量日志，所以目前只在云环境中启用
    if MULTI_TENANT:
        app_base.set_task_finished_log_level(logging.INFO)


class HubPeriodicTask(bootsteps.StartStopStep):
    """
    Regularly reacquires the primary worker lock outside of the task queue.
    Use the task_logger in this class to avoid double logging.
    
    定期在任务队列外重新获取主工作进程锁。
    在这个类中使用task_logger以避免重复日志记录。

    This cannot be done inside a regular beat task because it must run on schedule and
    a queue of existing work would starve the task from running.
    
    这不能在常规的beat任务中完成，因为它必须按计划运行，
    而现有工作队列会导致任务无法执行。
    """

    requires = {"celery.worker.components:Hub"}

    def __init__(self, worker: Any, **kwargs: Any) -> None:
        """
        初始化周期任务
        
        参数:
            worker: 工作进程对象
            kwargs: 额外参数
        """
        self.interval = CELERY_PRIMARY_WORKER_LOCK_TIMEOUT / 8  # 间隔时间（秒）
        self.task_tref = None

    def start(self, worker: Any) -> None:
        """
        启动周期任务
        
        参数:
            worker: 工作进程对象
        """
        if not celery_is_worker_primary(worker):
            return

        hub = getattr(worker.consumer.controller, 'hub', None)  # 获取 hub
        if hub is None:
            task_logger.warning("Hub is not available, skipping periodic task scheduling.")
            return

        # 调度周期任务
        self.task_tref = hub.call_repeatedly(self.interval, self.run_periodic_task, worker)
        task_logger.info("计划的周期任务已启动。")

    def run_periodic_task(self, worker: Any) -> None:
        """
        执行周期任务
        
        参数:
            worker: 工作进程对象
        """
        try:
            if not celery_is_worker_primary(worker):
                return

            if not hasattr(worker, "primary_worker_lock"):
                return

            lock: RedisLock = worker.primary_worker_lock

            r = get_redis_client(tenant_id=None)

            if lock.owned():
                task_logger.debug("Reacquiring primary worker lock.")
                lock.reacquire()
            else:
                task_logger.warning(
                    "Full acquisition of primary worker lock. "
                    "Reasons could be worker restart or lock expiration."
                )
                lock = r.lock(
                    OnyxRedisLocks.PRIMARY_WORKER,
                    timeout=CELERY_PRIMARY_WORKER_LOCK_TIMEOUT,
                )

                task_logger.info("Primary worker lock: Acquire starting.")
                acquired = lock.acquire(
                    blocking_timeout=CELERY_PRIMARY_WORKER_LOCK_TIMEOUT / 2
                )
                if acquired:
                    task_logger.info("Primary worker lock: Acquire succeeded.")
                    worker.primary_worker_lock = lock
                else:
                    task_logger.error("Primary worker lock: Acquire failed!")
                    raise TimeoutError("Primary worker lock could not be acquired!")

        except Exception:
            task_logger.exception("周期任务失败。")

    def stop(self, worker: Any) -> None:
        """
        停止周期任务
        
        参数:
            worker: 工作进程对象
        """
        if self.task_tref:
            self.task_tref.cancel()
            task_logger.info("已取消计划的周期任务。")


celery_app.steps["worker"].add(HubPeriodicTask)

celery_app.autodiscover_tasks(
    [
        "onyx.background.celery.tasks.connector_deletion",
        "onyx.background.celery.tasks.indexing",
        "onyx.background.celery.tasks.periodic",
        "onyx.background.celery.tasks.doc_permission_syncing",
        "onyx.background.celery.tasks.external_group_syncing",
        "onyx.background.celery.tasks.pruning",
        "onyx.background.celery.tasks.shared",
        "onyx.background.celery.tasks.vespa",
        "onyx.background.celery.tasks.llm_model_update",
    ]
)
