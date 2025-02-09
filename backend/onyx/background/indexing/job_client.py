"""Custom client that works similarly to Dask, but simpler and more lightweight.
Dask jobs behaved very strangely - they would die all the time, retries would
not follow the expected behavior, etc.

NOTE: cannot use Celery directly due to
https://github.com/celery/celery/issues/7007#issuecomment-1740139367

[中文说明]
这是一个类似于Dask但更简单轻量的自定义客户端。
之前使用Dask任务表现异常 - 经常会死掉，重试也没有按照预期行为执行等。

注意：由于Celery的Issue #7007，无法直接使用Celery
"""
from collections.abc import Callable
from dataclasses import dataclass
from multiprocessing import Process
from typing import Any
from typing import Literal
from typing import Optional

from onyx.configs.constants import POSTGRES_CELERY_WORKER_INDEXING_CHILD_APP_NAME
from onyx.db.engine import SqlEngine
from onyx.utils.logger import setup_logger

logger = setup_logger()

JobStatusType = (
    Literal["error"]
    | Literal["finished"]
    | Literal["pending"]
    | Literal["running"]
    | Literal["cancelled"]
)


def _initializer(
    func: Callable, args: list | tuple, kwargs: dict[str, Any] | None = None
) -> Any:
    """Initialize the child process with a fresh SQLAlchemy Engine.

    Based on SQLAlchemy's recommendations to handle multiprocessing:
    https://docs.sqlalchemy.org/en/20/core/pooling.html#using-connection-pools-with-multiprocessing-or-os-fork
    
    [中文说明]
    使用新的SQLAlchemy引擎初始化子进程。
    基于SQLAlchemy关于处理多进程的建议实现。
    
    参数:
        func: 要执行的目标函数
        args: 位置参数列表或元组
        kwargs: 关键字参数字典，默认为None
        
    返回:
        Any: 目标函数的执行结果
    """
    if kwargs is None:
        kwargs = {}

    logger.info("Initializing spawned worker child process.")

    # Reset the engine in the child process
    SqlEngine.reset_engine()

    # Optionally set a custom app name for database logging purposes
    SqlEngine.set_app_name(POSTGRES_CELERY_WORKER_INDEXING_CHILD_APP_NAME)

    # Initialize a new engine with desired parameters
    SqlEngine.init_engine(pool_size=4, max_overflow=12, pool_recycle=60)

    # Proceed with executing the target function
    return func(*args, **kwargs)


def _run_in_process(
    func: Callable, args: list | tuple, kwargs: dict[str, Any] | None = None
) -> None:
    """在进程中运行初始化函数
    
    参数:
        func: 要执行的目标函数
        args: 位置参数列表或元组
        kwargs: 关键字参数字典，默认为None
    """
    _initializer(func, args, kwargs)


@dataclass
class SimpleJob:
    """Drop in replacement for `dask.distributed.Future`
    
    [中文说明]
    用于替代`dask.distributed.Future`的简单作业类
    
    属性:
        id: 作业ID
        process: 进程对象，可选
    """

    id: int
    process: Optional["Process"] = None

    def cancel(self) -> bool:
        """取消当前作业
        
        返回:
            bool: 是否成功取消作业
        """
        return self.release()

    def release(self) -> bool:
        """释放作业资源
        
        返回:
            bool: 是否成功释放资源
        """
        if self.process is not None and self.process.is_alive():
            self.process.terminate()
            return True
        return False

    @property
    def status(self) -> JobStatusType:
        """获取作业当前状态
        
        返回:
            JobStatusType: 作业状态（pending/running/cancelled/error/finished）
        """
        if not self.process:
            return "pending"
        elif self.process.is_alive():
            return "running"
        elif self.process.exitcode is None:
            return "cancelled"
        elif self.process.exitcode != 0:
            return "error"
        else:
            return "finished"

    def done(self) -> bool:
        """检查作业是否已完成
        
        返回:
            bool: 作业是否已完成（包括完成、取消或出错状态）
        """
        return (
            self.status == "finished"
            or self.status == "cancelled"
            or self.status == "error"
        )

    def exception(self) -> str:
        """Needed to match the Dask API, but not implemented since we don't currently
        have a way to get back the exception information from the child process.
        
        [中文说明]
        为了匹配Dask API而需要实现的方法，但由于目前无法从子进程获取异常信息，所以未完全实现。
        
        返回:
            str: 通用错误消息
        """
        return (
            f"Job with ID '{self.id}' was killed or encountered an unhandled exception."
        )


class SimpleJobClient:
    """Drop in replacement for `dask.distributed.Client`
    
    [中文说明]
    用于替代`dask.distributed.Client`的简单作业客户端
    
    参数:
        n_workers: 工作进程数量，默认为1
    """

    def __init__(self, n_workers: int = 1) -> None:
        self.n_workers = n_workers
        self.job_id_counter = 0
        self.jobs: dict[int, SimpleJob] = {}

    def _cleanup_completed_jobs(self) -> None:
        """清理已完成的作业"""
        current_job_ids = list(self.jobs.keys())
        for job_id in current_job_ids:
            job = self.jobs.get(job_id)
            if job and job.done():
                logger.debug(f"Cleaning up job with id: '{job.id}'")
                del self.jobs[job.id]

    def submit(self, func: Callable, *args: Any, pure: bool = True) -> SimpleJob | None:
        """NOTE: `pure` arg is needed so this can be a drop in replacement for Dask
        
        [中文说明]
        注意：需要`pure`参数以便可以直接替代Dask
        
        参数:
            func: 要执行的函数
            args: 传递给函数的位置参数
            pure: 是否为纯函数（用于兼容Dask接口）
            
        返回:
            SimpleJob | None: 作业对象，如果没有可用工作进程则返回None
        """
        self._cleanup_completed_jobs()
        if len(self.jobs) >= self.n_workers:
            logger.debug(
                f"No available workers to run job. "
                f"Currently running '{len(self.jobs)}' jobs, with a limit of '{self.n_workers}'."
            )
            return None

        job_id = self.job_id_counter
        self.job_id_counter += 1

        process = Process(target=_run_in_process, args=(func, args), daemon=True)
        job = SimpleJob(id=job_id, process=process)
        process.start()

        self.jobs[job_id] = job

        return job
