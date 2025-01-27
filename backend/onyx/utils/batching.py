"""
该模块提供批处理相关的工具函数。
主要功能是将可迭代对象分批处理，提高数据处理效率。
"""

from collections.abc import Callable
from collections.abc import Generator
from collections.abc import Iterable
from itertools import islice
from typing import TypeVar

# 定义泛型类型变量T，用于表示批处理数据的类型
T = TypeVar("T")


def batch_generator(
    items: Iterable[T],
    batch_size: int,
    pre_batch_yield: Callable[[list[T]], None] | None = None,
) -> Generator[list[T], None, None]:
    """
    将可迭代对象分批处理的生成器函数。

    参数:
        items: 需要分批处理的可迭代对象
        batch_size: 每批数据的大小
        pre_batch_yield: 在生成每批数据之前要执行的回调函数，可选
                       该函数接收当前批次的数据列表作为参数

    返回:
        Generator[list[T], None, None]: 返回一个生成器，每次生成一个批次的数据列表

    用法示例:
        >>> items = range(10)
        >>> for batch in batch_generator(items, 3):
        ...     print(batch)
        [0, 1, 2]
        [3, 4, 5]
        [6, 7, 8]
        [9]
    """
    iterable = iter(items)
    while True:
        batch = list(islice(iterable, batch_size))
        if not batch:
            return

        if pre_batch_yield:
            pre_batch_yield(batch)
        yield batch
