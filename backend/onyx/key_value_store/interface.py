"""
该文件定义了键值存储的接口和相关异常类。
提供了一个抽象基类 KeyValueStore，用于实现键值存储的基本操作。
"""

import abc

from onyx.utils.special_types import JSON_ro


class KvKeyNotFoundError(Exception):
    """
    键值存储中未找到指定键时抛出的异常。
    当尝试访问不存在的键时会引发此异常。
    """
    pass


class KeyValueStore:
    """
    键值存储的抽象基类，定义了键值存储的基本接口。
    所有具体的键值存储实现都应该继承此类并实现其抽象方法。
    """

    # In the Multi Tenant case, the tenant context is picked up automatically, it does not need to be passed in
    # It's read from the global thread level variable
    # 在多租户情况下，租户上下文会自动获取，不需要传入
    # 它从全局线程级变量中读取
    @abc.abstractmethod
    def store(self, key: str, val: JSON_ro, encrypt: bool = False) -> None:
        """
        存储键值对的抽象方法。

        参数:
            key: str - 存储的键
            val: JSON_ro - 要存储的值（只读JSON类型）
            encrypt: bool - 是否需要加密存储，默认为False

        返回:
            None
        """
        raise NotImplementedError

    @abc.abstractmethod
    def load(self, key: str) -> JSON_ro:
        """
        加载指定键的值的抽象方法。

        参数:
            key: str - 要查询的键

        返回:
            JSON_ro - 键对应的值（只读JSON类型）

        异常:
            KvKeyNotFoundError - 当键不存在时抛出
        """
        raise NotImplementedError

    @abc.abstractmethod
    def delete(self, key: str) -> None:
        """
        删除指定键值对的抽象方法。

        参数:
            key: str - 要删除的键

        返回:
            None

        异常:
            KvKeyNotFoundError - 当键不存在时抛出
        """
        raise NotImplementedError
