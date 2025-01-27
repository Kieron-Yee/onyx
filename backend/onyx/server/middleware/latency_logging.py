"""
这个模块实现了一个延迟日志记录中间件。
主要功能是记录和监控HTTP请求的处理时间，用于性能分析和监控。
"""

import logging
import time
from collections.abc import Awaitable
from collections.abc import Callable

from fastapi import FastAPI
from fastapi import Request
from fastapi import Response


def add_latency_logging_middleware(app: FastAPI, logger: logging.LoggerAdapter) -> None:
    """
    为FastAPI应用添加延迟日志记录中间件。
    
    参数:
        app (FastAPI): FastAPI应用实例
        logger (logging.LoggerAdapter): 用于记录日志的logger适配器
    
    返回:
        None
    """
    @app.middleware("http")
    async def log_latency(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """
        记录HTTP请求处理延迟的中间件函数。
        
        参数:
            request (Request): FastAPI的请求对象
            call_next (Callable): 处理请求的下一个中间件或路由处理函数
        
        返回:
            Response: FastAPI的响应对象
            
        功能说明:
            1. 记录请求开始时间
            2. 执行请求处理
            3. 计算处理时间
            4. 记录请求路径、方法、状态码和处理时间
        """
        start_time = time.monotonic()
        response = await call_next(request)
        process_time = time.monotonic() - start_time
        logger.debug(
            f"Path: {request.url.path} - Method: {request.method} - "
            f"Status Code: {response.status_code} - Time: {process_time:.4f} secs"
        )
        return response
