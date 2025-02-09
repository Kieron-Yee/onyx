"""
此模块提供了Dask分布式计算相关的工具类和函数。
主要包含用于监控Dask worker资源使用情况的插件。
"""

import asyncio

import psutil
from dask.distributed import WorkerPlugin
from distributed import Worker

from onyx.utils.logger import setup_logger

logger = setup_logger()


class ResourceLogger(WorkerPlugin):
    """
    Dask worker资源监控插件。
    
    用于定期记录worker节点的CPU和内存使用情况，帮助监控系统资源状态。
    
    参数:
        log_interval (int): 日志记录间隔时间（秒），默认为300秒（5分钟）
    """
    
    def __init__(self, log_interval: int = 60 * 5):
        """
        初始化资源记录器。
        
        参数:
            log_interval (int): 日志记录间隔时间（秒）
        """
        self.log_interval = log_interval

    def setup(self, worker: Worker) -> None:
        """
        This method will be called when the plugin is attached to a worker.
        当插件被附加到worker时会调用此方法。
        
        参数:
            worker (Worker): Dask worker实例
        """
        self.worker = worker
        worker.loop.add_callback(self.log_resources)

    async def log_resources(self) -> None:
        """
        Periodically log CPU and memory usage.
        定期记录CPU和内存使用情况。

        NOTE: must be async or else will clog up the worker indefinitely due to the fact that
        Dask uses Tornado under the hood (which is async)
        注意：必须是异步的，否则会因为Dask底层使用Tornado（异步框架）而导致worker永久阻塞
        
        周期性地记录并输出当前worker节点的以下信息：
        - CPU使用率百分比
        - 可用内存量（GB）
        """
        while True:
            cpu_percent = psutil.cpu_percent(interval=None)
            memory_available_gb = psutil.virtual_memory().available / (1024.0**3)
            # You can now log these values or send them to a monitoring service
            # 现在可以记录这些值或将它们发送到监控服务
            logger.debug(
                f"Worker {self.worker.address}: CPU usage {cpu_percent}%, Memory available {memory_available_gb}GB"
            )
            await asyncio.sleep(self.log_interval)
