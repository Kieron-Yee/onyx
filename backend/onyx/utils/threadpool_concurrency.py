"""
这个模块提供了并行执行函数的工具。
主要功能包括：
1. 使用线程池并行执行多个函数
2. 支持带参数的函数调用
3. 提供异常处理机制
"""

import uuid
from collections.abc import Callable
from concurrent.futures import as_completed
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from typing import Generic
from typing import TypeVar

from onyx.utils.logger import setup_logger

logger = setup_logger()

R = TypeVar("R")


def run_functions_tuples_in_parallel(
    functions_with_args: list[tuple[Callable, tuple]],
    allow_failures: bool = False,
    max_workers: int | None = None,
) -> list[Any]:
    """
    并行执行多个函数并返回每个函数的结果列表。

    参数:
        functions_with_args: 函数和参数的元组列表，每个元组包含可调用的函数和参数元组。
        allow_failures: 如果设置为True，则函数执行失败时结果将为None
        max_workers: 最大工作线程数

    返回:
        字典：将函数名映射到其结果或错误消息的字典。
    """
    workers = (
        min(max_workers, len(functions_with_args))
        if max_workers is not None
        else len(functions_with_args)
    )

    if workers <= 0:
        return []

    results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_index = {
            executor.submit(func, *args): i
            for i, (func, args) in enumerate(functions_with_args)
        }

        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                results.append((index, future.result()))
            except Exception as e:
                logger.exception(f"Function at index {index} failed due to {e}")
                results.append((index, None))

                if not allow_failures:
                    raise

    results.sort(key=lambda x: x[0])
    return [result for index, result in results]


class FunctionCall(Generic[R]):
    """
    用于run_functions_in_parallel的容器，通过FunctionCall.result_id从run_functions_in_parallel的输出中获取结果。
    """

    def __init__(
        self, func: Callable[..., R], args: tuple = (), kwargs: dict | None = None
    ):
        """
        初始化FunctionCall实例
        
        参数:
            func: 要执行的函数
            args: 位置参数
            kwargs: 关键字参数
        """
        self.func = func
        self.args = args
        self.kwargs = kwargs if kwargs is not None else {}
        self.result_id = str(uuid.uuid4())

    def execute(self) -> R:
        """
        执行函数调用
        
        返回:
            函数执行的结果
        """
        return self.func(*self.args, **self.kwargs)


def run_functions_in_parallel(
    function_calls: list[FunctionCall],
    allow_failures: bool = False,
) -> dict[str, Any]:
    """
    并行执行FunctionCall列表，并将结果存储在字典中，其中键是FunctionCall的result_id，值是调用的结果。

    参数:
        function_calls: FunctionCall对象列表
        allow_failures: 是否允许执行失败，如果为True，失败的调用结果将为None

    返回:
        包含执行结果的字典，键为result_id，值为对应的执行结果
    """
    results = {}

    with ThreadPoolExecutor(max_workers=len(function_calls)) as executor:
        future_to_id = {
            executor.submit(func_call.execute): func_call.result_id
            for func_call in function_calls
        }

        for future in as_completed(future_to_id):
            result_id = future_to_id[future]
            try:
                results[result_id] = future.result()
            except Exception as e:
                logger.exception(f"Function with ID {result_id} failed due to {e}")
                results[result_id] = None

                if not allow_failures:
                    raise

    return results
