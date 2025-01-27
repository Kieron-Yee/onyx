"""
此模块定义了索引心跳接口，用于监控和控制索引过程。
该接口提供了停止索引和报告进度的基本功能。
"""

from abc import ABC
from abc import abstractmethod


class IndexingHeartbeatInterface(ABC):
    """Defines a callback interface to be passed to
    to run_indexing_entrypoint.
    
    定义了一个回调接口，用于传递给索引入口点执行函数。
    此接口类用于监控索引过程的执行状态和进度。
    """

    @abstractmethod
    def should_stop(self) -> bool:
        """Signal to stop the looping function in flight.
        
        判断是否应该停止正在执行的循环函数。
        
        返回值：
            bool: True表示应该停止，False表示继续执行
        """

    @abstractmethod
    def progress(self, tag: str, amount: int) -> None:
        """Send progress updates to the caller.
        
        向调用者发送进度更新信息。
        
        参数：
            tag (str): 进度标识符，用于标识当前进度所属的处理阶段
            amount (int): 进度数值，表示当前阶段的完成数量
        """
