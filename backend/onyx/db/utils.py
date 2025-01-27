"""
数据库工具模块
该模块提供了一些数据库相关的辅助函数，用于数据库模型和字典之间的转换等操作。
"""

# 导入类型注解支持
from typing import Any

from sqlalchemy import inspect

from onyx.db.models import Base


def model_to_dict(model: Base) -> dict[str, Any]:
    """
    将SQLAlchemy模型对象转换为字典
    
    该函数接收一个SQLAlchemy模型实例，返回包含该模型所有列属性的字典。
    字典的键为列名，值为对应的列值。
    
    Args:
        model: SQLAlchemy模型实例
        
    Returns:
        包含模型属性的字典
    """
    return {c.key: getattr(model, c.key) for c in inspect(model).mapper.column_attrs}  # type: ignore
