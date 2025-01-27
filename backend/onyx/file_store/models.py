"""
该模块用于定义文件存储相关的数据模型。
主要包含聊天文件类型的枚举、文件描述符的类型定义以及内存中聊天文件的数据模型。
"""

import base64
from enum import Enum
from typing import NotRequired
from typing_extensions import TypedDict  # noreorder
from pydantic import BaseModel


class ChatFileType(str, Enum):
    """
    聊天文件类型枚举类
    """
    # 图片类型仅包含二进制数据
    IMAGE = "image"
    # 文档类型同时保存二进制数据和解析后的文本
    DOC = "document"
    # 纯文本仅包含文本内容
    PLAIN_TEXT = "plain_text"
    # CSV文件类型
    CSV = "csv"


class FileDescriptor(TypedDict):
    """
    NOTE: is a `TypedDict` so it can be used as a type hint for a JSONB column
    in Postgres
    注意：这是一个`TypedDict`类型，用于在Postgres中作为JSONB列的类型提示

    文件描述符类型定义，用于描述文件的基本信息
    """
    id: str  # 文件唯一标识符
    type: ChatFileType  # 文件类型
    name: NotRequired[str | None]  # 文件名称（可选）


class InMemoryChatFile(BaseModel):
    """
    内存中的聊天文件模型类，用于处理文件数据
    
    属性：
        file_id: 文件唯一标识符
        content: 文件二进制内容
        file_type: 文件类型
        filename: 文件名称（可选）
    """
    file_id: str
    content: bytes
    file_type: ChatFileType
    filename: str | None = None

    def to_base64(self) -> str:
        """
        将文件内容转换为base64编码的字符串
        
        仅支持图片类型文件的转换
        
        返回：
            str: base64编码的字符串
            
        异常：
            RuntimeError: 当尝试转换非图片类型文件时抛出
        """
        if self.file_type == ChatFileType.IMAGE:
            return base64.b64encode(self.content).decode()
        else:
            raise RuntimeError(
                "Should not be trying to convert a non-image file to base64"
                "不应该尝试将非图片文件转换为base64格式"
            )

    def to_file_descriptor(self) -> FileDescriptor:
        """
        将文件对象转换为文件描述符
        
        返回：
            FileDescriptor: 包含文件基本信息的描述符
        """
        return {
            "id": str(self.file_id),
            "type": self.file_type,
            "name": self.filename,
        }
