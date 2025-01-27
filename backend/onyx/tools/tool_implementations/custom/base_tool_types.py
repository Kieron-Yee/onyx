"""
此文件定义了工具（Tool）的基本类型。
主要包含了工具执行结果的类型定义，用于规范工具返回值的数据类型。
"""

# should really be `JSON_ro`, but this causes issues with pydantic
# 实际上应该是 `JSON_ro`，但这会导致 pydantic 出现问题
ToolResultType = dict | list | str | int | float | bool  # 工具结果类型：可以是字典、列表、字符串、整数、浮点数或布尔值
