from typing import TypeVar


T = TypeVar("T")


# 这个函数可以用于需要将大列表分割成小块进行处理的场景，例如批量处理数据或分页显示数据。
def batch_list(
    lst: list[T],
    batch_size: int,
) -> list[list[T]]:
    return [lst[i : i + batch_size] for i in range(0, len(lst), batch_size)]
