"""
这个文件实现了一个轻量级的 Celery 应用程序。
主要功能：
1. 配置和初始化 Celery 应用
2. 设置各种 Celery 信号处理器
3. 管理工作进程的生命周期
4. 处理数据库连接和其他依赖项
"""

import multiprocessing
from typing import Any

from celery import Celery
from celery import signals
from celery import Task
from celery.signals import celeryd_init
from celery.signals import worker_init
from celery.signals import worker_ready
from celery.signals import worker_shutdown

import onyx.background.celery.apps.app_base as app_base
from onyx.configs.constants import POSTGRES_CELERY_WORKER_LIGHT_APP_NAME
from onyx.db.engine import SqlEngine
from onyx.utils.logger import setup_logger
from shared_configs.configs import MULTI_TENANT


logger = setup_logger()

celery_app = Celery(__name__)
celery_app.config_from_object("onyx.background.celery.configs.light")


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
    任务执行前的处理函数
    参数:
        sender: 信号发送者
        task_id: 任务ID
        task: 任务对象
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
    任务执行后的处理函数
    参数:
        sender: 信号发送者
        task_id: 任务ID
        task: 任务对象
        args: 位置参数
        kwargs: 关键字参数
        retval: 返回值
        state: 任务状态
        kwds: 额外的关键字参数
    """
    app_base.on_task_postrun(sender, task_id, task, args, kwargs, retval, state, **kwds)


@celeryd_init.connect
def on_celeryd_init(sender: Any = None, conf: Any = None, **kwargs: Any) -> None:
    """
    Celery 守护进程初始化时的处理函数
    参数:
        sender: 信号发送者
        conf: 配置对象
        kwargs: 额外的关键字参数
    """
    app_base.on_celeryd_init(sender, conf, **kwargs)


@worker_init.connect
def on_worker_init(sender: Any, **kwargs: Any) -> None:
    """
    工作进程初始化时的处理函数
    主要完成:
    1. 设置数据库连接
    2. 等待 Redis 连接就绪
    3. 等待数据库连接就绪
    4. 等待 Vespa 服务就绪
    
    参数:
        sender: 信号发送者
        kwargs: 额外的关键字参数
    """
    logger.info("工作进程初始化信号已接收")
    logger.info(f"多进程启动方式: {multiprocessing.get_start_method()}")

    SqlEngine.set_app_name(POSTGRES_CELERY_WORKER_LIGHT_APP_NAME)
    SqlEngine.init_engine(pool_size=sender.concurrency, max_overflow=8)

    app_base.wait_for_redis(sender, **kwargs)
    app_base.wait_for_db(sender, **kwargs)
    app_base.wait_for_vespa(sender, **kwargs)

    # 多租户情况下减少启动检查项
    if MULTI_TENANT:
        return

    app_base.on_secondary_worker_init(sender, **kwargs)


@worker_ready.connect
def on_worker_ready(sender: Any, **kwargs: Any) -> None:
    """
    工作进程就绪时的处理函数
    参数:
        sender: 信号发送者
        kwargs: 额外的关键字参数
    """
    app_base.on_worker_ready(sender, **kwargs)


@worker_shutdown.connect
def on_worker_shutdown(sender: Any, **kwargs: Any) -> None:
    """
    工作进程关闭时的处理函数
    参数:
        sender: 信号发送者
        kwargs: 额外的关键字参数
    """
    app_base.on_worker_shutdown(sender, **kwargs)


@signals.setup_logging.connect
def on_setup_logging(
    loglevel: Any, logfile: Any, format: Any, colorize: Any, **kwargs: Any
) -> None:
    """
    设置日志系统的处理函数
    参数:
        loglevel: 日志级别
        logfile: 日志文件
        format: 日志格式
        colorize: 是否着色
        kwargs: 额外的关键字参数
    """
    app_base.on_setup_logging(loglevel, logfile, format, colorize, **kwargs)


# 自动发现并注册任务
celery_app.autodiscover_tasks(
    [
        "onyx.background.celery.tasks.shared",
        "onyx.background.celery.tasks.vespa",
        "onyx.background.celery.tasks.connector_deletion",
        "onyx.background.celery.tasks.doc_permission_syncing",
    ]
)
