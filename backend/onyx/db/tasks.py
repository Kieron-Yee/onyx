"""
此模块用于管理任务队列的状态，提供了一系列函数来处理任务的注册、更新和状态检查。
主要功能包括获取最新任务、注册新任务、标记任务开始和结束等数据库操作。
"""

from sqlalchemy import desc
from sqlalchemy import func
from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.configs.app_configs import JOB_TIMEOUT
from onyx.db.engine import get_db_current_time
from onyx.db.models import TaskQueueState
from onyx.db.models import TaskStatus


def get_latest_task(
    task_name: str,
    db_session: Session,
) -> TaskQueueState | None:
    """
    获取指定任务名称的最新任务记录
    
    Args:
        task_name: 任务名称
        db_session: 数据库会话对象
    
    Returns:
        返回最新的任务状态记录，如果不存在则返回None
    """
    stmt = (
        select(TaskQueueState)
        .where(TaskQueueState.task_name == task_name)
        .order_by(desc(TaskQueueState.id))
        .limit(1)
    )

    result = db_session.execute(stmt)
    latest_task = result.scalars().first()

    return latest_task


def get_latest_task_by_type(
    task_name: str,
    db_session: Session,
) -> TaskQueueState | None:
    """
    根据任务类型模糊查询最新的任务记录
    
    Args:
        task_name: 任务类型名称（支持模糊匹配）
        db_session: 数据库会话对象
    
    Returns:
        返回匹配到的最新任务状态记录，如果不存在则返回None
    """
    stmt = (
        select(TaskQueueState)
        .where(TaskQueueState.task_name.like(f"%{task_name}%"))
        .order_by(desc(TaskQueueState.id))
        .limit(1)
    )

    result = db_session.execute(stmt)
    latest_task = result.scalars().first()

    return latest_task


def register_task(
    task_name: str,
    db_session: Session,
) -> TaskQueueState:
    """
    注册新任务到任务队列
    
    Args:
        task_name: 任务名称
        db_session: 数据库会话对象
    
    Returns:
        返回新创建的任务状态对象
    """
    new_task = TaskQueueState(
        task_id="", task_name=task_name, status=TaskStatus.PENDING
    )

    db_session.add(new_task)
    db_session.commit()

    return new_task


def mark_task_start(
    task_name: str,
    db_session: Session,
) -> None:
    """
    标记任务开始执行
    
    Args:
        task_name: 任务名称
        db_session: 数据库会话对象
    
    Raises:
        ValueError: 当找不到指定名称的任务时抛出异常
    """
    task = get_latest_task(task_name, db_session)
    if not task:
        raise ValueError(f"No task found with name {task_name}")

    task.start_time = func.now()  # type: ignore
    db_session.commit()


def mark_task_finished(
    task_name: str,
    db_session: Session,
    success: bool = True,
) -> None:
    """
    标记任务执行完成
    
    Args:
        task_name: 任务名称
        db_session: 数据库会话对象
        success: 任务是否成功完成，默认为True
    
    Raises:
        ValueError: 当找不到指定名称的任务时抛出异常
    """
    latest_task = get_latest_task(task_name, db_session)
    if latest_task is None:
        raise ValueError(f"tasks for {task_name} do not exist")

    latest_task.status = TaskStatus.SUCCESS if success else TaskStatus.FAILURE
    db_session.commit()


def check_task_is_live_and_not_timed_out(
    task: TaskQueueState,
    db_session: Session,
    timeout: int = JOB_TIMEOUT,
) -> bool:
    """
    检查任务是否处于活动状态且未超时
    
    Args:
        task: 任务状态对象
        db_session: 数据库会话对象
        timeout: 超时时间（秒），默认使用JOB_TIMEOUT配置值
    
    Returns:
        如果任务仍在活动且未超时返回True，否则返回False
    """
    # 我们只关心活动的任务，以避免创建新的周期性任务 / We only care for live tasks to not create new periodic tasks
    if task.status in [TaskStatus.SUCCESS, TaskStatus.FAILURE]:
        return False

    current_db_time = get_db_current_time(db_session=db_session)

    last_update_time = task.register_time
    if task.start_time:
        last_update_time = max(task.register_time, task.start_time)

    time_elapsed = current_db_time - last_update_time
    return time_elapsed.total_seconds() < timeout
