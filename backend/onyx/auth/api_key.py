"""
此模块用于处理API密钥的生成、验证和管理。
主要功能包括：
- API密钥的生成和哈希
- API密钥格式验证
- 请求中API密钥的提取
- API密钥相关数据结构的定义
"""

import hashlib
import secrets
import uuid
from urllib.parse import quote
from urllib.parse import unquote

from fastapi import Request
from passlib.hash import sha256_crypt
from pydantic import BaseModel

from onyx.auth.schemas import UserRole
from onyx.configs.app_configs import API_KEY_HASH_ROUNDS


_API_KEY_HEADER_NAME = "Authorization"
# 注意：在HTTP头部上下文中，"X-"通常表示非标准、实验性或自定义头部。
# 它表明该头部不是由互联网工程任务组(IETF)等组织定义的官方标准的一部分。
# NOTE for others who are curious: In the context of a header, "X-" often refers
# to non-standard, experimental, or custom headers in HTTP or other protocols. It
# indicates that the header is not part of the official standards defined by
# organizations like the Internet Engineering Task Force (IETF).
_API_KEY_HEADER_ALTERNATIVE_NAME = "X-Onyx-Authorization"
_BEARER_PREFIX = "Bearer "
_API_KEY_PREFIX = "on_"
_DEPRECATED_API_KEY_PREFIX = "dn_"
_API_KEY_LEN = 192


class ApiKeyDescriptor(BaseModel):
    """
    API密钥描述符类，用于存储API密钥的相关信息
    
    属性：
        api_key_id (int): API密钥的唯一标识符
        api_key_display (str): 用于显示的API密钥格式
        api_key (str | None): 完整的API密钥，仅在初始创建时存在
        api_key_name (str | None): API密钥的名称
        api_key_role (UserRole): API密钥关联的用户角色
        user_id (uuid.UUID): 拥有该API密钥的用户ID
    """
    api_key_id: int
    api_key_display: str
    api_key: str | None = None
    api_key_name: str | None = None
    api_key_role: UserRole
    user_id: uuid.UUID


def generate_api_key(tenant_id: str | None = None) -> str:
    """
    生成新的API密钥
    
    参数:
        tenant_id (str | None): 租户ID，用于生成包含租户信息的API密钥
        
    返回:
        str: 生成的API密钥
    """
    # 为了向后兼容，如果没有tenant_id，则生成旧式密钥
    if not tenant_id:
        return _API_KEY_PREFIX + secrets.token_urlsafe(_API_KEY_LEN)

    encoded_tenant = quote(tenant_id)  # URL编码租户ID
    return f"{_API_KEY_PREFIX}{encoded_tenant}.{secrets.token_urlsafe(_API_KEY_LEN)}"


def extract_tenant_from_api_key_header(request: Request) -> str | None:
    """
    从请求头中提取租户ID
    
    参数:
        request (Request): FastAPI请求对象
        
    返回:
        str | None: 提取的租户ID，如果认证被禁用或格式无效则返回None
    """
    raw_api_key_header = request.headers.get(
        _API_KEY_HEADER_ALTERNATIVE_NAME
    ) or request.headers.get(_API_KEY_HEADER_NAME)

    if not raw_api_key_header or not raw_api_key_header.startswith(_BEARER_PREFIX):
        return None

    api_key = raw_api_key_header[len(_BEARER_PREFIX) :].strip()

    if not api_key.startswith(_API_KEY_PREFIX) and not api_key.startswith(
        _DEPRECATED_API_KEY_PREFIX
    ):
        return None

    parts = api_key[len(_API_KEY_PREFIX) :].split(".", 1)
    if len(parts) != 2:
        return None

    tenant_id = parts[0]
    return unquote(tenant_id) if tenant_id else None


def _deprecated_hash_api_key(api_key: str) -> str:
    """
    使用已废弃的方式对API密钥进行哈希处理
    
    参数:
        api_key (str): 需要哈希的API密钥
        
    返回:
        str: 哈希后的API密钥
    """
    return sha256_crypt.hash(api_key, salt="", rounds=API_KEY_HASH_ROUNDS)


def hash_api_key(api_key: str) -> str:
    """
    对API密钥进行哈希处理
    
    参数:
        api_key (str): 需要哈希的API密钥
        
    返回:
        str: 哈希后的API密钥
        
    异常:
        ValueError: 如果API密钥前缀无效
    """
    # 注意：不需要加盐，因为API密钥是随机生成的，不可能发生重叠
    # NOTE: no salt is needed, as the API key is randomly generated
    # and overlaps are impossible
    if api_key.startswith(_API_KEY_PREFIX):
        return hashlib.sha256(api_key.encode("utf-8")).hexdigest()
    elif api_key.startswith(_DEPRECATED_API_KEY_PREFIX):
        return _deprecated_hash_api_key(api_key)
    else:
        raise ValueError(f"Invalid API key prefix: {api_key[:3]}")


def build_displayable_api_key(api_key: str) -> str:
    """
    构建用于显示的API密钥格式
    
    参数:
        api_key (str): 原始API密钥
        
    返回:
        str: 处理后的可显示API密钥，中间部分用星号替代
    """
    if api_key.startswith(_API_KEY_PREFIX):
        api_key = api_key[len(_API_KEY_PREFIX) :]

    return _API_KEY_PREFIX + api_key[:4] + "********" + api_key[-4:]


def get_hashed_api_key_from_request(request: Request) -> str | None:
    """
    从请求中获取并哈希处理API密钥
    
    参数:
        request (Request): FastAPI请求对象
        
    返回:
        str | None: 哈希后的API密钥，如果请求中没有API密钥则返回None
    """
    raw_api_key_header = request.headers.get(
        _API_KEY_HEADER_ALTERNATIVE_NAME
    ) or request.headers.get(_API_KEY_HEADER_NAME)
    if raw_api_key_header is None:
        return None

    if raw_api_key_header.startswith(_BEARER_PREFIX):
        raw_api_key_header = raw_api_key_header[len(_BEARER_PREFIX) :].strip()

    return hash_api_key(raw_api_key_header)
