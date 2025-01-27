from collections.abc import Callable
from typing import List

from fastapi import Depends
from fastapi import Request
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter

from onyx.configs.app_configs import RATE_LIMIT_MAX_REQUESTS
from onyx.configs.app_configs import RATE_LIMIT_WINDOW_SECONDS
from onyx.redis.redis_pool import get_async_redis_connection


async def setup_limiter() -> None:
    """初始化速率限制器
    
    使用Redis作为存储后端来追踪和实施速率限制。
    在应用启动时需要调用此函数来设置限制器。
    """
    redis = await get_async_redis_connection()
    await FastAPILimiter.init(redis)


async def close_limiter() -> None:
    """关闭速率限制器
    
    在应用关闭时清理资源，关闭与Redis的连接。
    """
    await FastAPILimiter.close()


async def rate_limit_key(request: Request) -> str:
    """生成用于速率限制的唯一键
    
    Args:
        request: FastAPI请求对象
        
    Returns:
        str: 由客户端IP和User-Agent组合而成的唯一标识符
        
    说明:
        - 使用IP和User-Agent的组合来区分不同的客户端
        - 即使在NAT网络后的用户也能较好地区分
        - 对于缺失的信息会使用默认值避免键值错误
    """
    ip_part = request.client.host if request.client else "unknown"
    ua_part = request.headers.get("user-agent", "none").replace(" ", "_")
    return f"{ip_part}-{ua_part}"


def get_auth_rate_limiters() -> List[Callable]:
    """创建速率限制器依赖项列表
    
    Returns:
        List[Callable]: 包含速率限制器依赖项的列表
        
    说明:
        - 从配置中读取最大请求次数和时间窗口
        - 如果未配置限制参数，返回空列表（不启用限制）
        - 使用自定义的rate_limit_key函数来区分不同用户
        - 可以直接用于FastAPI的dependencies参数
    """
    if not (RATE_LIMIT_MAX_REQUESTS and RATE_LIMIT_WINDOW_SECONDS):
        return []

    return [
        Depends(
            RateLimiter(
                times=RATE_LIMIT_MAX_REQUESTS,
                seconds=RATE_LIMIT_WINDOW_SECONDS,
                identifier=rate_limit_key,
            )
        )
    ]
