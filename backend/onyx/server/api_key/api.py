"""
API密钥管理模块
本模块提供了API密钥的CRUD（创建、读取、更新、删除）操作接口
包括：列表展示、创建新密钥、重新生成密钥、更新密钥信息以及删除密钥等功能
所有操作都需要管理员权限
"""

from fastapi import APIRouter
from fastapi import Depends
from sqlalchemy.orm import Session

from onyx.auth.users import current_admin_user
from onyx.db.api_key import ApiKeyDescriptor
from onyx.db.api_key import fetch_api_keys
from onyx.db.api_key import insert_api_key
from onyx.db.api_key import regenerate_api_key
from onyx.db.api_key import remove_api_key
from onyx.db.api_key import update_api_key
from onyx.db.engine import get_session
from onyx.db.models import User
from onyx.server.api_key.models import APIKeyArgs


# API路由前缀设置为/admin/api-key
router = APIRouter(prefix="/admin/api-key")


@router.get("")
def list_api_keys(
    _: User | None = Depends(current_admin_user),
    db_session: Session = Depends(get_session),
) -> list[ApiKeyDescriptor]:
    """
    获取所有API密钥列表
    
    参数:
        _: 当前管理员用户，通过依赖注入获取
        db_session: 数据库会话对象
    
    返回:
        list[ApiKeyDescriptor]: API密钥描述符列表
    """
    return fetch_api_keys(db_session)


@router.post("")
def create_api_key(
    api_key_args: APIKeyArgs,
    user: User | None = Depends(current_admin_user),
    db_session: Session = Depends(get_session),
) -> ApiKeyDescriptor:
    """
    创建新的API密钥
    
    参数:
        api_key_args: API密钥创建参数
        user: 当前管理员用户
        db_session: 数据库会话对象
    
    返回:
        ApiKeyDescriptor: 新创建的API密钥描述符
    """
    return insert_api_key(db_session, api_key_args, user.id if user else None)


@router.post("/{api_key_id}/regenerate")
def regenerate_existing_api_key(
    api_key_id: int,
    _: User | None = Depends(current_admin_user),
    db_session: Session = Depends(get_session),
) -> ApiKeyDescriptor:
    """
    重新生成指定API密钥
    
    参数:
        api_key_id: 需要重新生成的API密钥ID
        _: 当前管理员用户
        db_session: 数据库会话对象
    
    返回:
        ApiKeyDescriptor: 重新生成后的API密钥描述符
    """
    return regenerate_api_key(db_session, api_key_id)


@router.patch("/{api_key_id}")
def update_existing_api_key(
    api_key_id: int,
    api_key_args: APIKeyArgs,
    _: User | None = Depends(current_admin_user),
    db_session: Session = Depends(get_session),
) -> ApiKeyDescriptor:
    """
    更新现有API密钥的信息
    
    参数:
        api_key_id: 需要更新的API密钥ID
        api_key_args: API密钥更新参数
        _: 当前管理员用户
        db_session: 数据库会话对象
    
    返回:
        ApiKeyDescriptor: 更新后的API密钥描述符
    """
    return update_api_key(db_session, api_key_id, api_key_args)


@router.delete("/{api_key_id}")
def delete_api_key(
    api_key_id: int,
    _: User | None = Depends(current_admin_user),
    db_session: Session = Depends(get_session),
) -> None:
    """
    删除指定的API密钥
    
    参数:
        api_key_id: 需要删除的API密钥ID
        _: 当前管理员用户
        db_session: 数据库会话对象
    
    返回:
        None
    """
    remove_api_key(db_session, api_key_id)
