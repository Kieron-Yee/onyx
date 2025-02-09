"""
本模块包含定期执行的Celery任务。
主要功能是清理kombu_message表中的过期消息，以维护数据库性能。
"""

#####
# Periodic Tasks
# 定期任务
#####
import json
from typing import Any

from celery import shared_task
from celery.contrib.abortable import AbortableTask  # type: ignore
from celery.exceptions import TaskRevokedError
from sqlalchemy import inspect
from sqlalchemy import text
from sqlalchemy.orm import Session

from onyx.background.celery.apps.app_base import task_logger
from onyx.configs.app_configs import JOB_TIMEOUT
from onyx.configs.constants import OnyxCeleryTask
from onyx.configs.constants import PostgresAdvisoryLocks
from onyx.db.engine import get_session_with_tenant


@shared_task(
    name=OnyxCeleryTask.KOMBU_MESSAGE_CLEANUP_TASK,
    soft_time_limit=JOB_TIMEOUT,
    bind=True,
    base=AbortableTask,
)
def kombu_message_cleanup_task(self: Any, tenant_id: str | None) -> int:
    """Runs periodically to clean up the kombu_message table
    定期运行以清理kombu_message表

    Args:
        self: Celery任务实例
        tenant_id: 租户ID，可以为None

    Returns:
        int: 已删除的消息数量
    """

    # we will select messages older than this amount to clean up
    # 我们将清理早于这个时间的消息
    KOMBU_MESSAGE_CLEANUP_AGE = 7  # days 天数
    KOMBU_MESSAGE_CLEANUP_PAGE_LIMIT = 1000  # 每页处理的最大消息数

    ctx = {}
    ctx["last_processed_id"] = 0
    ctx["deleted"] = 0
    ctx["cleanup_age"] = KOMBU_MESSAGE_CLEANUP_AGE
    ctx["page_limit"] = KOMBU_MESSAGE_CLEANUP_PAGE_LIMIT
    with get_session_with_tenant(tenant_id) as db_session:
        # Exit the task if we can't take the advisory lock
        # 如果无法获取咨询锁，则退出任务
        result = db_session.execute(
            text("SELECT pg_try_advisory_lock(:id)"),
            {"id": PostgresAdvisoryLocks.KOMBU_MESSAGE_CLEANUP_LOCK_ID.value},
        ).scalar()
        if not result:
            return 0

        while True:
            if self.is_aborted():
                raise TaskRevokedError("kombu_message_cleanup_task was aborted.")

            b = kombu_message_cleanup_task_helper(ctx, db_session)
            if not b:
                break

            db_session.commit()

    if ctx["deleted"] > 0:
        task_logger.info(
            f"Deleted {ctx['deleted']} orphaned messages from kombu_message."
        )

    return ctx["deleted"]


def kombu_message_cleanup_task_helper(ctx: dict, db_session: Session) -> bool:
    """Helper function to clean up old messages from the `kombu_message` table that are no longer relevant.
    帮助函数，用于清理`kombu_message`表中不再相关的旧消息。

    This function retrieves messages from the `kombu_message` table that are no longer visible and
    older than a specified interval. It checks if the corresponding task_id exists in the
    `celery_taskmeta` table. If the task_id does not exist, the message is deleted.
    该函数从`kombu_message`表中检索不再可见且早于指定时间间隔的消息。
    它检查相应的task_id是否存在于`celery_taskmeta`表中。如果task_id不存在，则删除该消息。

    Args:
        ctx (dict): A context dictionary containing configuration parameters such as:
                   包含配置参数的上下文字典，包括：
            - 'cleanup_age' (int): The age in days after which messages are considered old.
                                 消息被视为过期的天数
            - 'page_limit' (int): The maximum number of messages to process in one batch.
                                 一批处理的最大消息数
            - 'last_processed_id' (int): The ID of the last processed message to handle pagination.
                                       用于处理分页的最后处理的消息ID
            - 'deleted' (int): A counter to track the number of deleted messages.
                              已删除消息的计数器
        db_session (Session): The SQLAlchemy database session for executing queries.
                            用于执行查询的SQLAlchemy数据库会话

    Returns:
        bool: Returns True if there are more rows to process, False if not.
              如果还有更多行需要处理则返回True，否则返回False
    """

    inspector = inspect(db_session.bind)
    if not inspector:
        return False

    # With the move to redis as celery's broker and backend, kombu tables may not even exist.
    # We can fail silently.
    # 随着将redis作为celery的代理和后端，kombu表可能不存在。
    # 我们可以静默失败。

    # With the move to redis as celery's broker and backend, kombu tables may not even exist.
    # We can fail silently.
    if not inspector.has_table("kombu_message"):
        return False

    query = text(
        """
    SELECT id, timestamp, payload
    FROM kombu_message WHERE visible = 'false'
    AND timestamp < CURRENT_TIMESTAMP - INTERVAL :interval_days
    AND id > :last_processed_id
    ORDER BY id
    LIMIT :page_limit
"""
    )
    kombu_messages = db_session.execute(
        query,
        {
            "interval_days": f"{ctx['cleanup_age']} days",
            "page_limit": ctx["page_limit"],
            "last_processed_id": ctx["last_processed_id"],
        },
    ).fetchall()

    if len(kombu_messages) == 0:
        return False

    for msg in kombu_messages:
        payload = json.loads(msg[2])
        task_id = payload["headers"]["id"]

        # Check if task_id exists in celery_taskmeta
        # 检查task_id是否存在于celery_taskmeta表中
        task_exists = db_session.execute(
            text("SELECT 1 FROM celery_taskmeta WHERE task_id = :task_id"),
            {"task_id": task_id},
        ).fetchone()

        # If task_id does not exist, delete the message
        # 如果task_id不存在，则删除该消息
        if not task_exists:
            result = db_session.execute(
                text("DELETE FROM kombu_message WHERE id = :message_id"),
                {"message_id": msg[0]},
            )
            if result.rowcount > 0:  # type: ignore
                ctx["deleted"] += 1

        ctx["last_processed_id"] = msg[0]

    return True
