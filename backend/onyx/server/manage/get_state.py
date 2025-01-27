"""
此模块提供系统状态管理相关的API接口，包括健康检查、认证类型获取和版本信息获取等功能。
主要用于系统监控和状态查询。
"""

from fastapi import APIRouter

from onyx import __version__
from onyx.auth.users import anonymous_user_enabled
from onyx.auth.users import user_needs_to_be_verified
from onyx.configs.app_configs import AUTH_TYPE
from onyx.server.manage.models import AuthTypeResponse
from onyx.server.manage.models import VersionResponse
from onyx.server.models import StatusResponse

router = APIRouter()


@router.get("/health")
def healthcheck() -> StatusResponse:
    """
    健康检查接口
    
    用于检查系统是否正常运行
    
    Returns:
        StatusResponse: 包含系统状态的响应对象
            - success (bool): 表示系统是否正常
            - message (str): 状态描述信息
    """
    return StatusResponse(success=True, message="ok")


@router.get("/auth/type")
def get_auth_type() -> AuthTypeResponse:
    """
    获取系统认证类型的接口
    
    返回系统当前的认证配置信息，包括认证类型、是否需要验证和是否支持匿名用户
    
    Returns:
        AuthTypeResponse: 包含认证相关配置的响应对象
            - auth_type (str): 认证类型
            - requires_verification (bool): 是否需要验证
            - anonymous_user_enabled (bool): 是否启用匿名用户
    """
    return AuthTypeResponse(
        auth_type=AUTH_TYPE,
        requires_verification=user_needs_to_be_verified(),
        anonymous_user_enabled=anonymous_user_enabled(),
    )


@router.get("/version")
def get_version() -> VersionResponse:
    """
    获取系统版本信息的接口
    
    返回后端系统的版本号
    
    Returns:
        VersionResponse: 包含版本信息的响应对象
            - backend_version (str): 后端系统版本号
    """
    return VersionResponse(backend_version=__version__)
