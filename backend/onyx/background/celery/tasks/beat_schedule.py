"""
此模块用于定义 Celery 定时任务的调度配置。
包含了各种周期性任务的调度设置，如同步检查、清理和监控任务等。
"""

from datetime import timedelta
from typing import Any

from onyx.configs.app_configs import LLM_MODEL_UPDATE_API_URL
from onyx.configs.constants import OnyxCeleryPriority
from onyx.configs.constants import OnyxCeleryTask

# choosing 15 minutes because it roughly gives us enough time to process many tasks
# we might be able to reduce this greatly if we can run a unified
# loop across all tenants rather than tasks per tenant
# 选择15分钟是因为这大致给我们足够的时间来处理许多任务
# 如果我们能够在所有租户之间运行统一的循环而不是每个租户的任务，我们可能可以大大减少这个时间

BEAT_EXPIRES_DEFAULT = 15 * 60  # 15 minutes (in seconds) # 15分钟（以秒为单位）

# we set expires because it isn't necessary to queue up these tasks
# it's only important that they run relatively regularly
# 我们设置过期时间是因为没有必要将这些任务排队
# 重要的是它们能相对定期地运行

tasks_to_schedule = [
    {
        "name": "check-for-vespa-sync",
        "task": OnyxCeleryTask.CHECK_FOR_VESPA_SYNC_TASK,
        "schedule": timedelta(seconds=20),
        "options": {
            "priority": OnyxCeleryPriority.HIGH,
            "expires": BEAT_EXPIRES_DEFAULT,
        },
    },
    {
        "name": "check-for-connector-deletion",
        "task": OnyxCeleryTask.CHECK_FOR_CONNECTOR_DELETION,
        "schedule": timedelta(seconds=20),
        "options": {
            "priority": OnyxCeleryPriority.HIGH,
            "expires": BEAT_EXPIRES_DEFAULT,
        },
    },
    {
        "name": "check-for-indexing",
        "task": OnyxCeleryTask.CHECK_FOR_INDEXING,
        "schedule": timedelta(seconds=15),
        "options": {
            "priority": OnyxCeleryPriority.HIGH,
            "expires": BEAT_EXPIRES_DEFAULT,
        },
    },
    {
        "name": "check-for-prune",
        "task": OnyxCeleryTask.CHECK_FOR_PRUNING,
        "schedule": timedelta(seconds=15),
        "options": {
            "priority": OnyxCeleryPriority.HIGH,
            "expires": BEAT_EXPIRES_DEFAULT,
        },
    },
    {
        "name": "kombu-message-cleanup",
        "task": OnyxCeleryTask.KOMBU_MESSAGE_CLEANUP_TASK,
        "schedule": timedelta(seconds=3600),
        "options": {
            "priority": OnyxCeleryPriority.LOWEST,
            "expires": BEAT_EXPIRES_DEFAULT,
        },
    },
    {
        "name": "monitor-vespa-sync",
        "task": OnyxCeleryTask.MONITOR_VESPA_SYNC,
        "schedule": timedelta(seconds=5),
        "options": {
            "priority": OnyxCeleryPriority.HIGH,
            "expires": BEAT_EXPIRES_DEFAULT,
        },
    },
    {
        "name": "check-for-doc-permissions-sync",
        "task": OnyxCeleryTask.CHECK_FOR_DOC_PERMISSIONS_SYNC,
        "schedule": timedelta(seconds=30),
        "options": {
            "priority": OnyxCeleryPriority.HIGH,
            "expires": BEAT_EXPIRES_DEFAULT,
        },
    },
    {
        "name": "check-for-external-group-sync",
        "task": OnyxCeleryTask.CHECK_FOR_EXTERNAL_GROUP_SYNC,
        "schedule": timedelta(seconds=20),
        "options": {
            "priority": OnyxCeleryPriority.HIGH,
            "expires": BEAT_EXPIRES_DEFAULT,
        },
    },
]

# Only add the LLM model update task if the API URL is configured
# 仅在配置了API URL的情况下添加LLM模型更新任务
if LLM_MODEL_UPDATE_API_URL:
    tasks_to_schedule.append(
        {
            "name": "check-for-llm-model-update",
            "task": OnyxCeleryTask.CHECK_FOR_LLM_MODEL_UPDATE,
            "schedule": timedelta(hours=1),  # Check every hour # 每小时检查一次
            "options": {
                "priority": OnyxCeleryPriority.LOW,
                "expires": BEAT_EXPIRES_DEFAULT,
            },
        }
    )


def get_tasks_to_schedule() -> list[dict[str, Any]]:
    """
    获取需要调度的任务列表。
    
    返回值:
        list[dict[str, Any]]: 包含所有需要调度的任务配置的列表，
                             每个任务配置包含名称、任务类型、调度间隔和选项等信息。
    """
    return tasks_to_schedule
