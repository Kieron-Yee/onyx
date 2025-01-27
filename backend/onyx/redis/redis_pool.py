"""
Redis连接池和客户端管理模块

本模块提供了Redis连接池的实现，支持同步和异步连接，以及多租户隔离。
主要功能包括：
1. Redis连接池的创建和管理
2. 多租户Redis客户端的实现
3. 异步Redis连接的管理
4. 认证token在Redis中的存储和获取
"""

import asyncio
import functools
import json
import threading
from collections.abc import Callable
from typing import Any
from typing import cast
from typing import Optional

import redis
from fastapi import Request
from redis import asyncio as aioredis
from redis.client import Redis
from redis.lock import Lock as RedisLock

from onyx.configs.app_configs import REDIS_AUTH_KEY_PREFIX
from onyx.configs.app_configs import REDIS_DB_NUMBER
from onyx.configs.app_configs import REDIS_HEALTH_CHECK_INTERVAL
from onyx.configs.app_configs import REDIS_HOST
from onyx.configs.app_configs import REDIS_PASSWORD
from onyx.configs.app_configs import REDIS_POOL_MAX_CONNECTIONS
from onyx.configs.app_configs import REDIS_PORT
from onyx.configs.app_configs import REDIS_SSL
from onyx.configs.app_configs import REDIS_SSL_CA_CERTS
from onyx.configs.app_configs import REDIS_SSL_CERT_REQS
from onyx.configs.constants import REDIS_SOCKET_KEEPALIVE_OPTIONS
from onyx.utils.logger import setup_logger

logger = setup_logger()


class TenantRedis(redis.Redis):
    """
    多租户Redis客户端类
    
    通过为每个租户添加独特的前缀来实现多租户数据隔离
    继承自redis.Redis，重写了键操作相关的方法，自动为所有键添加租户前缀
    """

    def __init__(self, tenant_id: str, *args: Any, **kwargs: Any) -> None:
        """
        初始化多租户Redis客户端
        
        Args:
            tenant_id: 租户ID，用于生成键前缀
            *args: 传递给Redis基类的位置参数
            **kwargs: 传递给Redis基类的关键字参数
        """
        super().__init__(*args, **kwargs)
        self.tenant_id: str = tenant_id

    def _prefixed(self, key: str | bytes | memoryview) -> str | bytes | memoryview:
        """
        为键添加租户前缀
        
        Args:
            key: 原始键值，支持字符串、字节和内存视图类型
            
        Returns:
            添加了租户前缀的键值
            
        Raises:
            TypeError: 当键的类型不支持时抛出
        """
        prefix: str = f"{self.tenant_id}:"
        if isinstance(key, str):
            if key.startswith(prefix):
                return key
            else:
                return prefix + key
        elif isinstance(key, bytes):
            prefix_bytes = prefix.encode()
            if key.startswith(prefix_bytes):
                return key
            else:
                return prefix_bytes + key
        elif isinstance(key, memoryview):
            key_bytes = key.tobytes()
            prefix_bytes = prefix.encode()
            if key_bytes.startswith(prefix_bytes):
                return key
            else:
                return memoryview(prefix_bytes + key_bytes)
        else:
            raise TypeError(f"Unsupported key type: {type(key)}")

    def _prefix_method(self, method: Callable) -> Callable:
        """
        装饰器函数，用于为Redis方法添加键前缀处理
        
        Args:
            method: 需要处理的Redis方法
            
        Returns:
            包装后的方法，会自动为键添加租户前缀
        """
        @functools.wraps(method)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if "name" in kwargs:
                kwargs["name"] = self._prefixed(kwargs["name"])
            elif len(args) > 0:
                args = (self._prefixed(args[0]),) + args[1:]
            return method(*args, **kwargs)

        return wrapper

    def _prefix_scan_iter(self, method: Callable) -> Callable:
        """
        专门处理scan_iter方法的装饰器
        
        Args:
            method: scan_iter方法
            
        Returns:
            包装后的方法，会自动处理键的前缀，并在返回结果时去除前缀
        """
        @functools.wraps(method)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Prefix the match pattern if provided
            if "match" in kwargs:
                kwargs["match"] = self._prefixed(kwargs["match"])
            elif len(args) > 0:
                args = (self._prefixed(args[0]),) + args[1:]

            # Get the iterator
            iterator = method(*args, **kwargs)

            # Remove prefix from returned keys
            prefix = f"{self.tenant_id}:".encode()
            prefix_len = len(prefix)

            for key in iterator:
                if isinstance(key, bytes) and key.startswith(prefix):
                    yield key[prefix_len:]
                else:
                    yield key

        return wrapper

    def __getattribute__(self, item: str) -> Any:
        original_attr = super().__getattribute__(item)
        methods_to_wrap = [
            "lock",
            "unlock",
            "get",
            "set",
            "delete",
            "exists",
            "incrby",
            "hset",
            "hget",
            "getset",
            "owned",
            "reacquire",
            "create_lock",
            "startswith",
            "sadd",
            "srem",
            "scard",
            "hexists",
            "hset",
            "hdel",
        ]  # Regular methods that need simple prefixing

        if item == "scan_iter":
            return self._prefix_scan_iter(original_attr)
        elif item in methods_to_wrap and callable(original_attr):
            return self._prefix_method(original_attr)
        return original_attr


class RedisPool:
    """
    Redis连接池单例类
    
    管理Redis连接池，确保应用中只创建一个连接池实例
    提供获取Redis客户端的方法
    """
    _instance: Optional["RedisPool"] = None
    _lock: threading.Lock = threading.Lock()
    _pool: redis.BlockingConnectionPool

    def __new__(cls) -> "RedisPool":
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(RedisPool, cls).__new__(cls)
                    cls._instance._init_pool()
        return cls._instance

    def _init_pool(self) -> None:
        self._pool = RedisPool.create_pool(ssl=REDIS_SSL)

    def get_client(self, tenant_id: str | None) -> Redis:
        if tenant_id is None:
            tenant_id = "public"
        return TenantRedis(tenant_id, connection_pool=self._pool)

    @staticmethod
    def create_pool(
        host: str = REDIS_HOST,
        port: int = REDIS_PORT,
        db: int = REDIS_DB_NUMBER,
        password: str = REDIS_PASSWORD,
        max_connections: int = REDIS_POOL_MAX_CONNECTIONS,
        ssl_ca_certs: str | None = REDIS_SSL_CA_CERTS,
        ssl_cert_reqs: str = REDIS_SSL_CERT_REQS,
        ssl: bool = False,
    ) -> redis.BlockingConnectionPool:
        """
        创建Redis阻塞连接池
        
        我们使用BlockingConnectionPool是因为当达到最大连接数时，它会阻塞等待而不是报错。
        这种行为更加确定性，且符合我们使用Redis的方式。
        
        Args:
            host: Redis服务器地址
            port: Redis服务器端口
            db: 数据库编号
            password: Redis密码
            max_connections: 最大连接数
            ssl_ca_certs: SSL CA证书路径
            ssl_cert_reqs: SSL证书要求
            ssl: 是否启用SSL
            
        Returns:
            Redis阻塞连接池实例
        """
        # Using ConnectionPool is not well documented.
        # Useful examples: https://github.com/redis/redis-py/issues/780
        if ssl:
            return redis.BlockingConnectionPool(
                host=host,
                port=port,
                db=db,
                password=password,
                max_connections=max_connections,
                timeout=None,
                health_check_interval=REDIS_HEALTH_CHECK_INTERVAL,
                socket_keepalive=True,
                socket_keepalive_options=REDIS_SOCKET_KEEPALIVE_OPTIONS,
                connection_class=redis.SSLConnection,
                ssl_ca_certs=ssl_ca_certs,
                ssl_cert_reqs=ssl_cert_reqs,
            )

        return redis.BlockingConnectionPool(
            host=host,
            port=port,
            db=db,
            password=password,
            max_connections=max_connections,
            timeout=None,
            health_check_interval=REDIS_HEALTH_CHECK_INTERVAL,
            socket_keepalive=True,
            socket_keepalive_options=REDIS_SOCKET_KEEPALIVE_OPTIONS,
        )


redis_pool = RedisPool()


def get_redis_client(*, tenant_id: str | None) -> Redis:
    return redis_pool.get_client(tenant_id)


# # Usage example
# redis_pool = RedisPool()
# redis_client = redis_pool.get_client()

# # Example of setting and getting a value
# redis_client.set('key', 'value')
# value = redis_client.get('key')
# print(value.decode())  # Output: 'value'

_async_redis_connection: aioredis.Redis | None = None
_async_lock = asyncio.Lock()


async def get_async_redis_connection() -> aioredis.Redis:
    """
    获取共享的异步Redis连接
    
    使用相同的配置（主机、端口、SSL等）提供共享的异步Redis连接。
    确保连接只创建一次（懒加载）并在后续调用中重用。
    
    Returns:
        异步Redis连接实例
    """
    global _async_redis_connection

    # If we haven't yet created an async Redis connection, we need to create one
    if _async_redis_connection is None:
        # Acquire the lock to ensure that only one coroutine attempts to create the connection
        async with _async_lock:
            # Double-check inside the lock to avoid race conditions
            if _async_redis_connection is None:
                scheme = "rediss" if REDIS_SSL else "redis"
                url = f"{scheme}://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB_NUMBER}"

                # Create a new Redis connection (or connection pool) from the URL
                _async_redis_connection = aioredis.from_url(
                    url,
                    password=REDIS_PASSWORD,
                    max_connections=REDIS_POOL_MAX_CONNECTIONS,
                )

    # Return the established connection (or pool) for all future operations
    return _async_redis_connection


async def retrieve_auth_token_data_from_redis(request: Request) -> dict | None:
    """
    从Redis中获取认证令牌数据
    
    Args:
        request: FastAPI请求对象
        
    Returns:
        令牌数据字典，如果未找到或发生错误则返回None
        
    Raises:
        ValueError: 当发生意外错误时抛出
    """
    token = request.cookies.get("fastapiusersauth")
    if not token:
        logger.debug("No auth token cookie found")
        return None

    try:
        redis = await get_async_redis_connection()
        redis_key = REDIS_AUTH_KEY_PREFIX + token
        token_data_str = await redis.get(redis_key)

        if not token_data_str:
            logger.debug(f"Token key {redis_key} not found or expired in Redis")
            return None

        return json.loads(token_data_str)
    except json.JSONDecodeError:
        logger.error("Error decoding token data from Redis")
        return None
    except Exception as e:
        logger.error(
            f"Unexpected error in retrieve_auth_token_data_from_redis: {str(e)}"
        )
        raise ValueError(
            f"Unexpected error in retrieve_auth_token_data_from_redis: {str(e)}"
        )


def redis_lock_dump(lock: RedisLock, r: Redis) -> None:
    """
    输出Redis锁的诊断信息
    
    Args:
        lock: Redis锁对象
        r: Redis客户端实例
    """
    # diagnostic logging for lock errors
    name = lock.name
    ttl = r.ttl(name)
    locked = lock.locked()
    owned = lock.owned()
    local_token: str | None = lock.local.token  # type: ignore

    remote_token_raw = r.get(lock.name)
    if remote_token_raw:
        remote_token_bytes = cast(bytes, remote_token_raw)
        remote_token = remote_token_bytes.decode("utf-8")
    else:
        remote_token = None

    logger.warning(
        f"RedisLock diagnostic logging: "
        f"name={name} "
        f"locked={locked} "
        f"owned={owned} "
        f"local_token={local_token} "
        f"remote_token={remote_token} "
        f"ttl={ttl}"
    )
