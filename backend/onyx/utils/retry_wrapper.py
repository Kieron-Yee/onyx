"""
此模块提供了重试机制的包装器，用于处理可能由于速率限制、不稳定性或其他原因而失败的外部API调用。
主要功能包括：
1. 通用重试装饰器构建器
2. 带重试机制的HTTP请求包装器
"""

from collections.abc import Callable
from logging import Logger
from typing import Any
from typing import cast
from typing import TypeVar

import requests
from retry import retry

from onyx.configs.app_configs import REQUEST_TIMEOUT_SECONDS
from onyx.utils.logger import setup_logger

logger = setup_logger()


F = TypeVar("F", bound=Callable[..., Any])


def retry_builder(
    tries: int = 20,
    delay: float = 0.1,
    max_delay: float | None = 60,
    backoff: float = 2,
    jitter: tuple[float, float] | float = 1,
    exceptions: type[Exception] | tuple[type[Exception], ...] = (Exception,),
) -> Callable[[F], F]:
    """Builds a generic wrapper/decorator for calls to external APIs that
    may fail due to rate limiting, flakes, or other reasons. Applies exponential
    backoff with jitter to retry the call.
    
    为可能由于速率限制、不稳定性或其他原因而失败的外部API调用构建通用包装器/装饰器。
    应用带抖动的指数退避算法进行重试。

    参数:
        tries: 最大重试次数
        delay: 初始延迟时间（秒）
        max_delay: 最大延迟时间（秒）
        backoff: 退避指数
        jitter: 抖动范围
        exceptions: 需要重试的异常类型

    返回:
        装饰器函数
    """

    def retry_with_default(func: F) -> F:
        """
        内部包装函数，应用重试逻辑到目标函数
        
        参数:
            func: 需要被包装的函数
        
        返回:
            包装后的函数
        """
        @retry(
            tries=tries,
            delay=delay,
            max_delay=max_delay,
            backoff=backoff,
            jitter=jitter,
            logger=cast(Logger, logger),
            exceptions=exceptions,
        )
        def wrapped_func(*args: list, **kwargs: dict[str, Any]) -> Any:
            return func(*args, **kwargs)

        return cast(F, wrapped_func)

    return retry_with_default


def request_with_retries(
    method: str,
    url: str,
    *,
    data: dict[str, Any] | None = None,
    headers: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    timeout: int = REQUEST_TIMEOUT_SECONDS,
    stream: bool = False,
    tries: int = 8,
    delay: float = 1,
    backoff: float = 2,
) -> requests.Response:
    """
    带重试机制的HTTP请求函数

    参数:
        method: HTTP请求方法
        url: 请求URL
        data: 请求体数据
        headers: 请求头
        params: URL参数
        timeout: 超时时间（秒）
        stream: 是否使用流式传输
        tries: 最大重试次数
        delay: 初始延迟时间（秒）
        backoff: 退避指数

    返回:
        requests.Response对象
    """

    @retry(tries=tries, delay=delay, backoff=backoff, logger=cast(Logger, logger))
    def _make_request() -> requests.Response:
        """
        内部请求执行函数，包含重试逻辑
        
        返回:
            requests.Response对象
            
        异常:
            requests.exceptions.HTTPError: 当HTTP请求失败时抛出
        """
        response = requests.request(
            method=method,
            url=url,
            data=data,
            headers=headers,
            params=params,
            timeout=timeout,
            stream=stream,
        )
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError:
            logger.exception(
                "Request failed:\n%s",
                {
                    "method": method,
                    "url": url,
                    "data": data,
                    "headers": headers,
                    "params": params,
                    "timeout": timeout,
                    "stream": stream,
                },
            )
            raise
        return response

    return _make_request()
