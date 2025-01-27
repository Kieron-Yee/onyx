"""
此文件用于定义回调处理相关的工具类和函数。
主要包含度量指标处理器类，用于记录和管理各种度量指标数据。
"""

from typing import Generic
from typing import TypeVar

# 定义泛型类型变量T
T = TypeVar("T")


class MetricsHander(Generic[T]):
    """
    度量指标处理器类
    用于存储和管理各种类型的度量指标数据
    
    泛型参数:
        T: 度量指标数据的类型
    """
    
    def __init__(self) -> None:
        """
        初始化度量指标处理器
        
        初始化时将度量指标设置为空
        """
        self.metrics: T | None = None

    def record_metric(self, metrics: T) -> None:
        """
        记录度量指标数据
        
        参数:
            metrics: T类型的度量指标数据
            
        返回:
            None
        """
        self.metrics = metrics
