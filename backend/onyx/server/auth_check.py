"""
该文件用于实现 Onyx 服务器的认证检查功能。
主要包含以下功能：
1. 定义公共端点列表
2. 检查路由是否在公共端点列表中
3. 检查路由器的认证配置
"""

from typing import cast

from fastapi import FastAPI
from fastapi.dependencies.models import Dependant
from starlette.routing import BaseRoute

from onyx.auth.users import current_admin_user
from onyx.auth.users import current_chat_accesssible_user
from onyx.auth.users import current_curator_or_admin_user
from onyx.auth.users import current_limited_user
from onyx.auth.users import current_user
from onyx.auth.users import current_user_with_expired_token
from onyx.configs.app_configs import APP_API_PREFIX
from onyx.server.onyx_api.ingestion import api_key_dep
from onyx.utils.variable_functionality import fetch_ee_implementation_or_noop

# 公共端点规范列表
# Public endpoint specifications list
# 定义了不需要认证就可以访问的API端点列表，包括路径和允许的HTTP方法
PUBLIC_ENDPOINT_SPECS = [
    # built-in documentation functions / 内置文档功能
    ("/openapi.json", {"GET", "HEAD"}),
    ("/docs", {"GET", "HEAD"}),
    ("/docs/oauth2-redirect", {"GET", "HEAD"}),
    ("/redoc", {"GET", "HEAD"}),
    
    # should always be callable, will just return 401 if not authenticated
    # 始终可调用，如未认证则返回401
    ("/me", {"GET"}),
    
    # just returns 200 to validate that the server is up
    # 仅返回200以验证服务器是否正常运行
    ("/health", {"GET"}),
    
    # just returns auth type, needs to be accessible before the user is logged
    # in to determine what flow to give the user
    # 仅返回认证类型，需要在用户登录前可访问以确定提供给用户的流程
    ("/auth/type", {"GET"}),
    
    # just gets the version of Onyx (e.g. 0.3.11)
    # 获取Onyx版本号（如0.3.11）
    ("/version", {"GET"}),
    
    # stuff related to basic auth / 基本认证相关的端点
    ("/auth/register", {"POST"}),
    ("/auth/login", {"POST"}),
    ("/auth/logout", {"POST"}),
    ("/auth/forgot-password", {"POST"}),
    ("/auth/reset-password", {"POST"}),
    ("/auth/request-verify-token", {"POST"}),
    ("/auth/verify", {"POST"}),
    ("/users/me", {"GET"}),
    ("/users/me", {"PATCH"}),
    ("/users/{id}", {"GET"}),
    ("/users/{id}", {"PATCH"}),
    ("/users/{id}", {"DELETE"}),
    
    # oauth 相关端点
    ("/auth/oauth/authorize", {"GET"}),
    ("/auth/oauth/callback", {"GET"}),
]

def is_route_in_spec_list(
    route: BaseRoute, public_endpoint_specs: list[tuple[str, set[str]]]
) -> bool:
    """
    检查给定的路由是否在公共端点规范列表中
    
    参数:
        route: 需要检查的路由对象
        public_endpoint_specs: 公共端点规范列表
        
    返回:
        bool: 如果路由在公共端点列表中返回True，否则返回False
    """
    if not hasattr(route, "path") or not hasattr(route, "methods"):
        return False

    # try adding the prefix AND not adding the prefix, since some endpoints
    # are not prefixed (e.g. /openapi.json)
    if (route.path, route.methods) in public_endpoint_specs:
        return True

    processed_global_prefix = f"/{APP_API_PREFIX.strip('/')}" if APP_API_PREFIX else ""
    if not processed_global_prefix:
        return False

    for endpoint_spec in public_endpoint_specs:
        base_path, methods = endpoint_spec
        prefixed_path = f"{processed_global_prefix}/{base_path.strip('/')}"

        if prefixed_path == route.path and route.methods == methods:
            return True

    return False

def check_router_auth(
    application: FastAPI,
    public_endpoint_specs: list[tuple[str, set[str]]] = PUBLIC_ENDPOINT_SPECS,
) -> None:
    """
    确保所有端点要么启用了认证，要么被明确标记为公共端点
    
    这个函数会检查应用程序中的所有路由，确保它们：
    1. 有认证依赖
    2. 或者在公共端点列表中明确列出
    
    参数:
        application: FastAPI应用实例
        public_endpoint_specs: 公共端点规范列表，默认使用PUBLIC_ENDPOINT_SPECS
        
    抛出:
        RuntimeError: 如果发现未受保护的私有路由
    """
    control_plane_dep = fetch_ee_implementation_or_noop(
        "onyx.server.tenants.access", "control_plane_dep"
    )
    current_cloud_superuser = fetch_ee_implementation_or_noop(
        "onyx.auth.users", "current_cloud_superuser"
    )

    for route in application.routes:
        # explicitly marked as public
        if is_route_in_spec_list(route, public_endpoint_specs):
            continue

        # check for auth
        found_auth = False
        route_dependant_obj = cast(
            Dependant | None, route.dependant if hasattr(route, "dependant") else None
        )
        if route_dependant_obj:
            for dependency in route_dependant_obj.dependencies:
                depends_fn = dependency.cache_key[0]
                if (
                    depends_fn == current_limited_user
                    or depends_fn == current_user
                    or depends_fn == current_admin_user
                    or depends_fn == current_curator_or_admin_user
                    or depends_fn == api_key_dep
                    or depends_fn == current_user_with_expired_token
                    or depends_fn == current_chat_accesssible_user
                    or depends_fn == control_plane_dep
                    or depends_fn == current_cloud_superuser
                ):
                    found_auth = True
                    break

        if not found_auth:
            # uncomment to print out all route(s) that are missing auth
            # print(f"(\"{route.path}\", {set(route.methods)}),")

            raise RuntimeError(
                f"Did not find user dependency in private route - {route}"
            )
