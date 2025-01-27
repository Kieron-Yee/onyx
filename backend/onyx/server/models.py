"""
此模块定义了服务器端使用的各种数据模型和响应结构。
包含用户信息快照、API密钥模型、状态响应等基础数据结构。
"""

from typing import Generic
from typing import Optional
from typing import TypeVar
from uuid import UUID

from pydantic import BaseModel

from onyx.auth.schemas import UserRole
from onyx.db.models import User


DataT = TypeVar("DataT")


class StatusResponse(BaseModel, Generic[DataT]):
    """
    通用状态响应模型，用于封装API响应数据
    
    属性:
        success (bool): 操作是否成功
        message (str, 可选): 响应消息
        data (DataT, 可选): 返回的数据内容
    """
    success: bool
    message: Optional[str] = None
    data: Optional[DataT] = None


class ApiKey(BaseModel):
    """
    API密钥模型
    
    属性:
        api_key (str): API密钥字符串
    """
    api_key: str


class IdReturn(BaseModel):
    """
    ID返回模型
    
    属性:
        id (int): 返回的ID值
    """
    id: int


class MinimalUserSnapshot(BaseModel):
    """
    最小用户信息快照，包含基本用户信息
    
    属性:
        id (UUID): 用户唯一标识符
        email (str): 用户邮箱
    """
    id: UUID
    email: str


class FullUserSnapshot(BaseModel):
    """
    完整用户信息快照，包含用户的所有相关信息
    
    属性:
        id (UUID): 用户唯一标识符
        email (str): 用户邮箱
        role (UserRole): 用户角色
        is_active (bool): 用户是否处于活动状态
    """
    id: UUID
    email: str
    role: UserRole
    is_active: bool

    @classmethod
    def from_user_model(cls, user: User) -> "FullUserSnapshot":
        """
        从User模型创建FullUserSnapshot实例
        
        参数:
            user (User): 用户模型实例
        
        返回:
            FullUserSnapshot: 包含用户完整信息的快照对象
        """
        return cls(
            id=user.id,
            email=user.email,
            role=user.role,
            is_active=user.is_active,
        )


class InvitedUserSnapshot(BaseModel):
    """
    被邀请用户的快照信息
    
    属性:
        email (str): 被邀请用户的邮箱
    """
    email: str


class DisplayPriorityRequest(BaseModel):
    """
    显示优先级请求模型
    
    属性:
        display_priority_map (dict[int, int]): 显示优先级映射字典，
            键为项目ID，值为对应的显示优先级
    """
    display_priority_map: dict[int, int]
