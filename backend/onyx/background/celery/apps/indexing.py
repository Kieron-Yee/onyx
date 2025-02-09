"""
此文件为Celery索引应用程序的主要配置文件。
主要功能：
1. 配置Celery应用实例
2. 设置各种Celery信号处理器
3. 处理工作进程的初始化和清理
4. 管理数据库连接和其他依赖服务
"""

import multiprocessing
from typing import Any

from celery import Celery
from celery import signals
from celery import Task
from celery.signals import celeryd_init
from celery.signals import worker_init
from celery.signals import worker_process_init
from celery.signals import worker_ready
from celery.signals import worker_shutdown

import onyx.background.celery.apps.app_base as app_base
from onyx.configs.constants import POSTGRES_CELERY_WORKER_INDEXING_APP_NAME
from onyx.db.engine import SqlEngine
from onyx.utils.logger import setup_logger
from shared_configs.configs import MULTI_TENANT


logger = setup_logger()

celery_app = Celery(__name__)
celery_app.config_from_object("onyx.background.celery.configs.indexing")


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
    任务执行前的信号处理器
    
    参数：
        sender: 信号发送者
        task_id: 任务ID
        task: 任务实例
        args: 位置参数
        kwargs: 关键字参数
        kwds: 额外的关键字参数
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
    任务执行后的信号处理器
    
    参数：
        sender: 信号发送者
        task_id: 任务ID
        task: 任务实例
        args: 位置参数
        kwargs: 关键字参数
        retval: 任务返回值
        state: 任务状态
        kwds: 额外的关键字参数
    """
    app_base.on_task_postrun(sender, task_id, task, args, kwargs, retval, state, **kwds)


@celeryd_init.connect
def on_celeryd_init(sender: Any = None, conf: Any = None, **kwargs: Any) -> None:
    """
    Celery守护进程初始化信号处理器
    
    参数：
        sender: 信号发送者
        conf: Celery配置对象
        kwargs: 额外的关键字参数
    """
    app_base.on_celeryd_init(sender, conf, **kwargs)


@worker_init.connect
def on_worker_init(sender: Any, **kwargs: Any) -> None:
    """
    工作进程初始化信号处理器
    
    参数：
        sender: 信号发送者（工作进程）
        kwargs: 额外的关键字参数
    """
    logger.info("worker_init signal received.")  # 工作进程初始化信号已接收
    logger.info(f"Multiprocessing start method: {multiprocessing.get_start_method()}")  # 多进程启动方式

    SqlEngine.set_app_name(POSTGRES_CELERY_WORKER_INDEXING_APP_NAME)

    # rkuo: been seeing transient connection exceptions here, so upping the connection count
    # from just concurrency/concurrency to concurrency/concurrency*2
    # rkuo: 在这里遇到了临时连接异常，所以将连接数从 concurrency/concurrency 提升到 concurrency/concurrency*2
    SqlEngine.init_engine(
        pool_size=sender.concurrency, max_overflow=sender.concurrency * 2
    )

    app_base.wait_for_redis(sender, **kwargs)
    app_base.wait_for_db(sender, **kwargs)
    app_base.wait_for_vespa(sender, **kwargs)

    # Less startup checks in multi-tenant case
    # 多租户情况下减少启动检查
    if MULTI_TENANT:
        return

    app_base.on_secondary_worker_init(sender, **kwargs)


@worker_ready.connect
def on_worker_ready(sender: Any, **kwargs: Any) -> None:
    """
    工作进程就绪信号处理器
    
    参数：
        sender: 信号发送者
        kwargs: 额外的关键字参数
    """
    app_base.on_worker_ready(sender, **kwargs)


@worker_shutdown.connect
def on_worker_shutdown(sender: Any, **kwargs: Any) -> None:
    """
    工作进程关闭信号处理器
    
    参数：
        sender: 信号发送者
        kwargs: 额外的关键字参数
    """
    app_base.on_worker_shutdown(sender, **kwargs)


@worker_process_init.connect
def init_worker(**kwargs: Any) -> None:
    """
    工作进程初始化时重置数据库引擎
    
    参数：
        kwargs: 额外的关键字参数
    """
    SqlEngine.reset_engine()


@signals.setup_logging.connect
def on_setup_logging(
    loglevel: Any, logfile: Any, format: Any, colorize: Any, **kwargs: Any
) -> None:
    """
    日志设置信号处理器
    
    参数：
        loglevel: 日志级别
        logfile: 日志文件
        format: 日志格式
        colorize: 是否着色
        kwargs: 额外的关键字参数
    """
    app_base.on_setup_logging(loglevel, logfile, format, colorize, **kwargs)


celery_app.autodiscover_tasks(
    [
        "onyx.background.celery.tasks.indexing",
    ]
)
