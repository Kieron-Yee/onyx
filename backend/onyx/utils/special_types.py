"""
This module defines special types used across the onyx backend.
该模块定义了在onyx后端中使用的特殊类型。
"""

from collections.abc import Mapping
from collections.abc import Sequence
from typing import TypeAlias

# JSON_ro defines a read-only JSON-like type that can be:
# - A mapping (dict) from strings to JSON_ro
# - A sequence (list/tuple) of JSON_ro
# - Basic types: str, int, float, bool, None
# JSON_ro 定义了一个只读的类JSON类型，可以是：
# - 从字符串到JSON_ro的映射（字典）
# - JSON_ro的序列（列表/元组）
# - 基本类型：str, int, float, bool, None
JSON_ro: TypeAlias = (
    Mapping[str, "JSON_ro"] | Sequence["JSON_ro"] | str | int | float | bool | None
)
