"""
此文件提供了用于处理Python类继承关系的工具函数。
主要功能包括：
1. 从指定目录导入所有模块
2. 查找类的所有子类
3. 在指定目录中查找特定类的所有子类
"""

from __future__ import annotations

import importlib
import os
import pkgutil
import sys
from types import ModuleType
from typing import List
from typing import Type
from typing import TypeVar

T = TypeVar("T")


def import_all_modules_from_dir(dir_path: str) -> List[ModuleType]:
    """
    Imports all modules found in the given directory and its subdirectories,
    returning a list of imported module objects.
    导入指定目录及其子目录中的所有模块，返回已导入模块对象的列表。

    参数:
        dir_path: 要搜索模块的目录路径

    返回值:
        List[ModuleType]: 已导入模块的列表
    """
    dir_path = os.path.abspath(dir_path)

    if dir_path not in sys.path:
        sys.path.insert(0, dir_path)

    imported_modules: List[ModuleType] = []

    for _, package_name, _ in pkgutil.walk_packages([dir_path]):
        try:
            module = importlib.import_module(package_name)
            imported_modules.append(module)
        except Exception as e:
            # Handle or log exceptions as needed
            # 根据需要处理或记录异常
            print(f"Could not import {package_name}: {e}")

    return imported_modules


def all_subclasses(cls: Type[T]) -> List[Type[T]]:
    """
    Recursively find all subclasses of the given class.
    递归查找给定类的所有子类。

    参数:
        cls: 要查找子类的父类

    返回值:
        List[Type[T]]: 所有子类的列表
    """
    direct_subs = cls.__subclasses__()
    result: List[Type[T]] = []
    for subclass in direct_subs:
        result.append(subclass)
        # Extend the result by recursively calling all_subclasses
        # 通过递归调用all_subclasses扩展结果
        result.extend(all_subclasses(subclass))
    return result


def find_all_subclasses_in_dir(parent_class: Type[T], directory: str) -> List[Type[T]]:
    """
    Imports all modules from the given directory (and subdirectories),
    then returns all classes that are subclasses of parent_class.
    从给定目录（及其子目录）导入所有模块，然后返回parent_class的所有子类。

    参数:
        parent_class: 要查找子类的父类
        directory: 要搜索子类的目录

    返回值:
        List[Type[T]]: 在目录中找到的所有子类列表
    """
    # First import all modules to ensure classes are loaded into memory
    # 首先导入所有模块以确保类被加载到内存中
    import_all_modules_from_dir(directory)

    # Gather all subclasses of the given parent class
    # 收集给定父类的所有子类
    subclasses = all_subclasses(parent_class)
    return subclasses


# Example usage:
# 使用示例：
if __name__ == "__main__":

    class Animal:
        """
        示例父类
        """
        pass

    # Suppose "mymodules" contains files that define classes inheriting from Animal
    # 假设"mymodules"目录包含定义了继承自Animal的类的文件
    found_subclasses = find_all_subclasses_in_dir(Animal, "mymodules")
    for sc in found_subclasses:
        print("Found subclass:", sc.__name__)
