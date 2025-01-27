"""
此文件提供了一个用于在SQLAlchemy中处理Pydantic模型的自定义类型。
主要用于将Pydantic模型实例与PostgreSQL的JSONB类型之间进行转换。
"""

import json
from typing import Any
from typing import Optional
from typing import Type

from pydantic import BaseModel
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.types import TypeDecorator


class PydanticType(TypeDecorator):
    """
    自定义SQLAlchemy类型，用于处理Pydantic模型与数据库JSONB类型之间的转换。
    继承自SQLAlchemy的TypeDecorator，实现了Pydantic模型实例与JSON数据之间的序列化和反序列化。
    """
    impl = JSONB

    def __init__(
        self, pydantic_model: Type[BaseModel], *args: Any, **kwargs: Any
    ) -> None:
        """
        初始化PydanticType实例。
        
        Args:
            pydantic_model: 要转换的Pydantic模型类
            *args: 传递给父类的位置参数
            **kwargs: 传递给父类的关键字参数
        """
        super().__init__(*args, **kwargs)
        self.pydantic_model = pydantic_model

    def process_bind_param(
        self, value: Optional[BaseModel], dialect: Any
    ) -> Optional[dict]:
        """
        处理写入数据库前的参数转换。
        将Pydantic模型实例转换为可存储在数据库中的JSON格式。
        
        Args:
            value: 需要转换的Pydantic模型实例
            dialect: 数据库方言对象
            
        Returns:
            转换后的字典数据或None
        """
        if value is not None:
            return json.loads(value.json())
        return None

    def process_result_value(
        self, value: Optional[dict], dialect: Any
    ) -> Optional[BaseModel]:
        """
        处理从数据库读取的数据转换。
        将JSON数据转换回Pydantic模型实例。
        
        Args:
            value: 从数据库读取的JSON数据
            dialect: 数据库方言对象
            
        Returns:
            转换后的Pydantic模型实例或None
        """
        if value is not None:
            return self.pydantic_model.parse_obj(value)
        return None
