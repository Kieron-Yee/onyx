"""
此文件定义了与文件夹功能相关的数据模型类。
主要包含文件夹的响应模型、请求模型等基础数据结构。
这些模型用于处理文件夹的创建、更新、删除以及与聊天会话相关的操作。
"""

from uuid import UUID

from pydantic import BaseModel

from onyx.server.query_and_chat.models import ChatSessionDetails


class FolderResponse(BaseModel):
    """
    文件夹响应模型类，用于返回文件夹相关信息
    
    属性:
        folder_id (int): 文件夹唯一标识ID
        folder_name (str | None): 文件夹名称，可以为空
        display_priority (int): 显示优先级
        chat_sessions (list[ChatSessionDetails]): 该文件夹下的聊天会话列表
    """
    folder_id: int
    folder_name: str | None
    display_priority: int
    chat_sessions: list[ChatSessionDetails]


class GetUserFoldersResponse(BaseModel):
    """
    获取用户文件夹列表的响应模型类
    
    属性:
        folders (list[FolderResponse]): 文件夹列表
    """
    folders: list[FolderResponse]


class FolderCreationRequest(BaseModel):
    """
    创建文件夹的请求模型类
    
    属性:
        folder_name (str | None): 文件夹名称，可选参数，默认为None
    """
    folder_name: str | None = None


class FolderUpdateRequest(BaseModel):
    """
    更新文件夹的请求模型类
    
    属性:
        folder_name (str | None): 新的文件夹名称，可选参数，默认为None
    """
    folder_name: str | None = None


class FolderChatSessionRequest(BaseModel):
    """
    文件夹聊天会话请求模型类
    
    属性:
        chat_session_id (UUID): 聊天会话的唯一标识ID
    """
    chat_session_id: UUID


class DeleteFolderOptions(BaseModel):
    """
    删除文件夹选项模型类
    
    属性:
        including_chats (bool): 是否同时删除文件夹中的聊天会话，默认为False
    """
    including_chats: bool = False
