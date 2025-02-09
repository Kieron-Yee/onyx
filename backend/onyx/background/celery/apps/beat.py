"""
该文件实现了一个动态租户调度器，用于管理Celery定时任务。
主要功能：
1. 动态管理多租户的定时任务
2. 定期更新任务调度
3. 处理任务配置和同步
"""

from datetime import timedelta
from typing import Any

from celery import Celery
from celery import signals
from celery.beat import PersistentScheduler  # type: ignore
from celery.signals import beat_init

import onyx.background.celery.apps.app_base as app_base
from onyx.configs.constants import POSTGRES_CELERY_BEAT_APP_NAME
from onyx.db.engine import get_all_tenant_ids
from onyx.db.engine import SqlEngine
from onyx.utils.logger import setup_logger
from onyx.utils.variable_functionality import fetch_versioned_implementation
from shared_configs.configs import IGNORED_SYNCING_TENANT_LIST

logger = setup_logger(__name__)

celery_app = Celery(__name__)
celery_app.config_from_object("onyx.background.celery.configs.beat")


class DynamicTenantScheduler(PersistentScheduler):
    """
    动态租户调度器类，继承自PersistentScheduler
    用于管理多租户环境下的定时任务调度
    """
    
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        初始化动态租户调度器
        参数:
            args: 可变位置参数
            kwargs: 可变关键字参数
        """
        logger.info("Initializing DynamicTenantScheduler")  # 正在初始化动态租户调度器
        super().__init__(*args, **kwargs)
        self._reload_interval = timedelta(minutes=2)  # 设置重载间隔为2分钟
        self._last_reload = self.app.now() - self._reload_interval
        self.setup_schedule()
        self._update_tenant_tasks()
        logger.info(f"Set reload interval to {self._reload_interval}")  # 设置重载间隔完成

    def setup_schedule(self) -> None:
        """
        初始化调度计划
        """
        logger.info("Setting up initial schedule")  # 正在设置初始调度计划
        super().setup_schedule()
        logger.info("Initial schedule setup complete")  # 初始调度计划设置完成

    def tick(self) -> float:
        """
        执行调度器的定时检查
        返回值：
            float: 下次检查的间隔时间
        """
        retval = super().tick()
        now = self.app.now()
        if (
            self._last_reload is None
            or (now - self._last_reload) > self._reload_interval
        ):
            logger.info("Reload interval reached, initiating task update")  # 达到重载间隔，开始更新任务
            self._update_tenant_tasks()
            self._last_reload = now
            logger.info("Task update completed, reset reload timer")  # 任务更新完成，重置计时器
        return retval

    def _update_tenant_tasks(self) -> None:
        """
        更新租户任务
        负责获取所有租户ID并更新对应的任务调度
        """
        logger.info("Starting task update process")  # 开始任务更新进程
        try:
            logger.info("Fetching all IDs")
            tenant_ids = get_all_tenant_ids()
            logger.info(f"Found {len(tenant_ids)} IDs")

            logger.info("Fetching tasks to schedule")
            tasks_to_schedule = fetch_versioned_implementation(
                "onyx.background.celery.tasks.beat_schedule", "get_tasks_to_schedule"
            )

            new_beat_schedule: dict[str, dict[str, Any]] = {}

            current_schedule = self.schedule.items()

            existing_tenants = set()
            for task_name, _ in current_schedule:
                if "-" in task_name:
                    existing_tenants.add(task_name.split("-")[-1])
            logger.info(f"Found {len(existing_tenants)} existing items in schedule")

            for tenant_id in tenant_ids:
                if (
                    IGNORED_SYNCING_TENANT_LIST
                    and tenant_id in IGNORED_SYNCING_TENANT_LIST
                ):
                    logger.info(
                        f"Skipping tenant {tenant_id} as it is in the ignored syncing list"
                    )
                    continue

                if tenant_id not in existing_tenants:
                    logger.info(f"Processing new item: {tenant_id}")

                for task in tasks_to_schedule():
                    task_name = f"{task['name']}-{tenant_id}"
                    logger.debug(f"Creating task configuration for {task_name}")
                    new_task = {
                        "task": task["task"],
                        "schedule": task["schedule"],
                        "kwargs": {"tenant_id": tenant_id},
                    }
                    if options := task.get("options"):
                        logger.debug(f"Adding options to task {task_name}: {options}")
                        new_task["options"] = options
                    new_beat_schedule[task_name] = new_task

            if self._should_update_schedule(current_schedule, new_beat_schedule):
                logger.info(
                    "Schedule update required",
                    extra={
                        "new_tasks": len(new_beat_schedule),
                        "current_tasks": len(current_schedule),
                    },
                )

                # Create schedule entries
                entries = {}
                for name, entry in new_beat_schedule.items():
                    entries[name] = self.Entry(
                        name=name,
                        app=self.app,
                        task=entry["task"],
                        schedule=entry["schedule"],
                        options=entry.get("options", {}),
                        kwargs=entry.get("kwargs", {}),
                    )

                # Update the schedule using the scheduler's methods
                self.schedule.clear()
                self.schedule.update(entries)

                # Ensure changes are persisted
                self.sync()

                logger.info("Schedule update completed successfully")
            else:
                logger.info("Schedule is up to date, no changes needed")
        except (AttributeError, KeyError) as e:
            logger.exception(f"Failed to process task configuration: {str(e)}")
        except Exception as e:
            logger.exception(f"Unexpected error updating tasks: {str(e)}")

    def _should_update_schedule(
        self, current_schedule: dict, new_schedule: dict
    ) -> bool:
        """
        比较当前调度计划和新调度计划，判断是否需要更新
        参数:
            current_schedule: 当前的调度计划
            new_schedule: 新的调度计划
        返回值:
            bool: 是否需要更新调度计划
        """
        logger.debug("Comparing current and new schedules")  # 正在比较当前和新的调度计划
        current_tasks = set(name for name, _ in current_schedule)
        new_tasks = set(new_schedule.keys())
        needs_update = current_tasks != new_tasks
        logger.debug(f"Schedule update needed: {needs_update}")  # 是否需要更新调度计划
        return needs_update


@beat_init.connect
def on_beat_init(sender: Any, **kwargs: Any) -> None:
    """
    Beat初始化信号处理函数
    参数:
        sender: 信号发送者
        kwargs: 附加参数
    """
    logger.info("beat_init signal received.")  # 收到beat_init信号

    # Celery beat shouldn't touch the db at all. But just setting a low minimum here.
    # Celery beat不应该直接操作数据库，这里只设置最小连接数
    SqlEngine.set_app_name(POSTGRES_CELERY_BEAT_APP_NAME)
    SqlEngine.init_engine(pool_size=2, max_overflow=0)

    app_base.wait_for_redis(sender, **kwargs)


@signals.setup_logging.connect
def on_setup_logging(
    loglevel: Any, logfile: Any, format: Any, colorize: Any, **kwargs: Any
) -> None:
    """
    日志设置信号处理函数
    参数:
        loglevel: 日志级别
        logfile: 日志文件
        format: 日志格式
        colorize: 是否着色
        kwargs: 附加参数
    """
    app_base.on_setup_logging(loglevel, logfile, format, colorize, **kwargs)


# 设置Celery应用的调度器为DynamicTenantScheduler
celery_app.conf.beat_scheduler = DynamicTenantScheduler
