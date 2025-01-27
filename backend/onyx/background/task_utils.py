"""
Celery Task Utilities Module
(Celery任务工具模块)

This module provides utility functions and decorators for wrapping Celery tasks to 
automatically track and manage task states.
(本模块提供了一系列工具函数和装饰器，用于包装Celery任务，实现任务状态的自动追踪和管理)

Main features include:
(主要功能包括:)
- Automatic task execution state recording
  (任务执行状态的自动记录)
- Database state updates on task creation and completion
  (任务创建和完成时的数据库状态更新)
- Exception handling and task result recording
  (异常处理和任务执行结果记录)
"""

from collections.abc import Callable
from functools import wraps
from typing import Any
from typing import cast
from typing import TypeVar

from celery import Task
from celery.result import AsyncResult
from sqlalchemy.orm import Session

from onyx.db.engine import get_sqlalchemy_engine
from onyx.db.tasks import mark_task_finished
from onyx.db.tasks import mark_task_start
from onyx.db.tasks import register_task


T = TypeVar("T", bound=Callable)


def build_run_wrapper(build_name_fn: Callable[..., str]) -> Callable[[T], T]:
    """Utility meant to wrap the celery task `run` function in order to
    automatically update our custom `task_queue_jobs` table appropriately
    (用于包装celery任务的run函数，以自动更新自定义task_queue_jobs表的工具函数)

    参数:
        build_name_fn: 用于构建任务名称的回调函数

    返回:
        包装任务执行函数的装饰器
    """

    def wrap_task_fn(task_fn: T) -> T:
        """
        包装任务执行函数的内部函数
        
        参数:
            task_fn: 需要被包装的任务函数
            
        返回:
            包装后的任务函数
        """
        @wraps(task_fn)
        def wrapped_task_fn(*args: list, **kwargs: dict) -> Any:
            engine = get_sqlalchemy_engine()

            task_name = build_name_fn(*args, **kwargs)
            with Session(engine) as db_session:
                # mark the task as started
                mark_task_start(task_name=task_name, db_session=db_session)

            result = None
            exception = None
            try:
                result = task_fn(*args, **kwargs)
            except Exception as e:
                exception = e

            with Session(engine) as db_session:
                mark_task_finished(
                    task_name=task_name,
                    db_session=db_session,
                    success=exception is None,
                )

            if not exception:
                return result
            else:
                raise exception

        return cast(T, wrapped_task_fn)

    return wrap_task_fn


# rough type signature for `apply_async`
AA = TypeVar("AA", bound=Callable[..., AsyncResult])


def build_apply_async_wrapper(build_name_fn: Callable[..., str]) -> Callable[[AA], AA]:
    """Utility meant to wrap celery `apply_async` function in order to automatically
    update create an entry in our `task_queue_jobs` table
    (用于包装celery的apply_async函数，以在task_queue_jobs表中自动创建条目的工具函数)

    参数:
        build_name_fn: 用于构建任务名称的回调函数

    返回:
        包装异步任务函数的装饰器
    """

    def wrapper(fn: AA) -> AA:
        """
        包装异步任务函数的内部函数
        
        参数:
            fn: 需要被包装的异步任务函数
            
        返回:
            包装后的异步任务函数
        """
        @wraps(fn)
        def wrapped_fn(
            args: tuple | None = None,
            kwargs: dict[str, Any] | None = None,
            *other_args: list,
            **other_kwargs: dict[str, Any],
        ) -> Any:
            # `apply_async` takes in args / kwargs directly as arguments
            args_for_build_name = args or tuple()
            kwargs_for_build_name = kwargs or {}
            task_name = build_name_fn(*args_for_build_name, **kwargs_for_build_name)
            with Session(get_sqlalchemy_engine()) as db_session:
                # register_task must come before fn = apply_async or else the task
                # might run mark_task_start (and crash) before the task row exists
                db_task = register_task(task_name, db_session)

                task = fn(args, kwargs, *other_args, **other_kwargs)

                # we update the celery task id for diagnostic purposes
                # but it isn't currently used by any code
                db_task.task_id = task.id
                db_session.commit()

            return task

        return cast(AA, wrapped_fn)

    return wrapper


def build_celery_task_wrapper(build_name_fn: Callable[..., str]) -> Callable[[Task], Task]:
    """Utility meant to wrap celery task functions in order to automatically
    update our custom `task_queue_jobs` table appropriately.
    (用于包装celery任务函数以自动更新自定义task_queue_jobs表的工具函数)

    On task creation (e.g. `apply_async`), a row is inserted into the table with 
    status `PENDING`.
    (在任务创建时(如apply_async)，会在表中插入一条PENDING状态的记录)

    On task start, the latest row is updated to have status `STARTED`.
    (任务开始时，最新记录的状态会更新为STARTED)

    On task success, the latest row is updated to have status `SUCCESS`.
    (任务成功时，最新记录的状态会更新为SUCCESS)
    
    On the task raising an unhandled exception, the latest row is updated to have
    status `FAILURE`.
    (任务抛出未处理异常时，最新记录的状态会更新为FAILURE)

    参数:
        build_name_fn: 用于构建任务名称的回调函数

    返回:
        包装Celery任务的装饰器
    """

    def wrap_task(task: Task) -> Task:
        """
        包装Celery任务的内部函数
        
        参数:
            task: 需要被包装的Celery任务
            
        返回:
            包装后的Celery任务
        """
        task.run = build_run_wrapper(build_name_fn)(task.run)  # type: ignore
        task.apply_async = build_apply_async_wrapper(build_name_fn)(task.apply_async)  # type: ignore
        return task

    return wrap_task
