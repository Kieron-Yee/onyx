"""
此模块提供了用于函数执行时间记录和性能监控的工具。
主要功能包括：
1. 函数执行时间的装饰器
2. 生成器函数执行时间的装饰器
3. 支持可选的遥测数据收集
"""

import time
from collections.abc import Callable
from collections.abc import Generator
from collections.abc import Iterator
from functools import wraps
from typing import Any
from typing import cast
from typing import TypeVar

from onyx.utils.logger import setup_logger
from onyx.utils.telemetry import optional_telemetry
from onyx.utils.telemetry import RecordType

logger = setup_logger()

F = TypeVar("F", bound=Callable)
FG = TypeVar("FG", bound=Callable[..., Generator | Iterator])


def log_function_time(
    func_name: str | None = None,
    print_only: bool = False,
    debug_only: bool = False,
    include_args: bool = False,
) -> Callable[[F], F]:
    """
    装饰器函数，用于记录被装饰函数的执行时间。

    参数:
        func_name: str | None - 自定义的函数名称，如果为None则使用原函数名
        print_only: bool - 是否只打印日志而不发送遥测数据
        debug_only: bool - 是否只在debug级别记录日志
        include_args: bool - 是否在日志中包含函数参数

    返回值:
        Callable - 装饰器函数
    """
    def decorator(func: F) -> F:
        """
        内部装饰器函数

        参数:
            func: F - 被装饰的函数

        返回值:
            F - 包装后的函数
        """
        @wraps(func)
        def wrapped_func(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            user = kwargs.get("user")
            result = func(*args, **kwargs)
            elapsed_time = time.time() - start_time
            elapsed_time_str = f"{elapsed_time:.3f}"
            log_name = func_name or func.__name__
            args_str = f" args={args} kwargs={kwargs}" if include_args else ""
            final_log = f"{log_name}{args_str} took {elapsed_time_str} seconds"
            if debug_only:
                logger.debug(final_log)
            else:
                # These are generally more important logs so the level is a bit higher
                logger.notice(final_log)

            if not print_only:
                optional_telemetry(
                    record_type=RecordType.LATENCY,
                    data={"function": log_name, "latency": str(elapsed_time_str)},
                    user_id=str(user.id) if user else "Unknown",
                )

            return result

        return cast(F, wrapped_func)

    return decorator


def log_generator_function_time(
    func_name: str | None = None, 
    print_only: bool = False
) -> Callable[[FG], FG]:
    """
    生成器函数的执行时间记录装饰器。

    参数:
        func_name: str | None - 自定义的函数名称，如果为None则使用原函数名
        print_only: bool - 是否只打印日志而不发送遥测数据

    返回值:
        Callable - 装饰器函数
    """
    def decorator(func: FG) -> FG:
        """
        内部装饰器函数

        参数:
            func: FG - 被装饰的生成器函数

        返回值:
            FG - 包装后的生成器函数
        """
        @wraps(func)
        def wrapped_func(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            user = kwargs.get("user")
            gen = func(*args, **kwargs)
            try:
                value = next(gen)
                while True:
                    yield value
                    value = next(gen)
            except StopIteration:
                pass
            finally:
                elapsed_time_str = str(time.time() - start_time)
                log_name = func_name or func.__name__
                logger.info(f"{log_name} took {elapsed_time_str} seconds")
                if not print_only:
                    optional_telemetry(
                        record_type=RecordType.LATENCY,
                        data={"function": log_name, "latency": str(elapsed_time_str)},
                        user_id=str(user.id) if user else "Unknown",
                    )

        return cast(FG, wrapped_func)

    return decorator
