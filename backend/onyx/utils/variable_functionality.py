import functools
import importlib
import inspect
from typing import Any
from typing import TypeVar

from onyx.configs.app_configs import ENTERPRISE_EDITION_ENABLED
from onyx.utils.logger import setup_logger

logger = setup_logger()


class OnyxVersion:
    def __init__(self) -> None:
        self._is_ee = False

    def set_ee(self) -> None:
        self._is_ee = True

    def is_ee_version(self) -> bool:
        return self._is_ee


global_version = OnyxVersion()


def set_is_ee_based_on_env_variable() -> None:
    if ENTERPRISE_EDITION_ENABLED and not global_version.is_ee_version():
        logger.notice("Enterprise Edition enabled")
        global_version.set_ee()


@functools.lru_cache(maxsize=128)
def fetch_versioned_implementation(module: str, attribute: str) -> Any:
    """
    Fetches a versioned implementation of a specified attribute from a given module.
    This function first checks if the application is running in an Enterprise Edition (EE)
    context. If so, it attempts to import the attribute from the EE-specific module.
    If the module or attribute is not found, it falls back to the default module or
    raises the appropriate exception depending on the context.
    
    获取指定模块中特定属性的版本化实现。
    此函数首先检查应用程序是否在企业版（EE）环境中运行。如果是，则尝试从EE特定模块中导入属性。
    如果未找到模块或属性，则回退到默认模块或根据上下文抛出适当的异常。

    Args:
        module (str): The name of the module from which to fetch the attribute.
        attribute (str): The name of the attribute to fetch from the module.
        
        module (str): 要获取属性的模块名称
        attribute (str): 要从模块中获取的属性名称

    Returns:
        Any: The fetched implementation of the attribute.
        Any: 获取到的属性实现

    Raises:
        ModuleNotFoundError: If the module cannot be found and the error is not related to
                             the Enterprise Edition fallback logic.
        ModuleNotFoundError: 如果无法找到模块且错误与企业版回退逻辑无关时抛出

    Logs:
        Logs debug information about the fetching process and warnings if the versioned
        implementation cannot be found or loaded.
        
        记录获取过程的调试信息，如果无法找到或加载版本化实现则记录警告。
    """
    logger.debug("Fetching versioned implementation for %s.%s", module, attribute)
    is_ee = global_version.is_ee_version()

    module_full = f"ee.{module}" if is_ee else module
    try:
        return getattr(importlib.import_module(module_full), attribute)
    except ModuleNotFoundError as e:
        logger.warning(
            "Failed to fetch versioned implementation for %s.%s: %s",
            module_full,
            attribute,
            e,
        )

        if is_ee:
            if "ee.onyx" not in str(e):
                # If it's a non Onyx related import failure, this is likely because
                # a dependent library has not been installed. Should raise this failure
                # instead of letting the server start up
                raise e

            # Use the MIT version as a fallback, this allows us to develop MIT
            # versions independently and later add additional EE functionality
            # similar to feature flagging
            return getattr(importlib.import_module(module), attribute)

        raise


T = TypeVar("T")


def fetch_versioned_implementation_with_fallback(
    module: str, attribute: str, fallback: T
) -> T:
    """
    Attempts to fetch a versioned implementation of a specified attribute from a given module.
    If the attempt fails (e.g., due to an import error or missing attribute), the function logs
    a warning and returns the provided fallback implementation.
    
    尝试从给定模块中获取指定属性的版本化实现。
    如果尝试失败（例如，由于导入错误或缺少属性），函数会记录警告并返回提供的回退实现。

    Args:
        module (str): The name of the module from which to fetch the attribute.
        attribute (str): The name of the attribute to fetch from the module.
        fallback (T): The fallback implementation to return if fetching the attribute fails.
        
        module (str): 要获取属性的模块名称
        attribute (str): 要从模块中获取的属性名称
        fallback (T): 如果获取属性失败时要返回的回退实现

    Returns:
        T: The fetched implementation if successful, otherwise the provided fallback.
        T: 如果成功则返回获取的实现，否则返回提供的回退实现
    """
    try:
        return fetch_versioned_implementation(module, attribute)
    except Exception:
        return fallback


def noop_fallback(*args: Any, **kwargs: Any) -> None:
    """
    A no-op (no operation) fallback function that accepts any arguments but does nothing.
    This is often used as a default or placeholder callback function.
    
    一个无操作回退函数，接受任何参数但不执行任何操作。
    这通常用作默认或占位符回调函数。

    Args:
        *args (Any): Positional arguments, which are ignored.
        **kwargs (Any): Keyword arguments, which are ignored.
        
        *args (Any): 位置参数，被忽略
        **kwargs (Any): 关键字参数，被忽略

    Returns:
        None
    """


def fetch_ee_implementation_or_noop(
    module: str, attribute: str, noop_return_value: Any = None
) -> Any:
    """
    Fetches an EE implementation if EE is enabled, otherwise returns a no-op function.
    Raises an exception if EE is enabled but the fetch fails.
    
    如果启用了EE，则获取EE实现，否则返回一个无操作函数。
    如果启用了EE但获取失败则抛出异常。

    Args:
        module (str): The name of the module from which to fetch the attribute.
        attribute (str): The name of the attribute to fetch from the module.
        
        module (str): 要获取属性的模块名称
        attribute (str): 要从模块中获取的属性名称

    Returns:
        Any: The fetched EE implementation if successful and EE is enabled, otherwise a no-op function.
        Any: 如果成功且启用了EE则返回获取的EE实现，否则返回无操作函数

    Raises:
        Exception: If EE is enabled but the fetch fails.
        Exception: 如果启用了EE但获取失败时抛出
    """
    if not global_version.is_ee_version():
        if inspect.iscoroutinefunction(noop_return_value):

            async def async_noop(*args: Any, **kwargs: Any) -> Any:
                return await noop_return_value(*args, **kwargs)

            return async_noop

        else:

            def sync_noop(*args: Any, **kwargs: Any) -> Any:
                return noop_return_value

            return sync_noop
    try:
        return fetch_versioned_implementation(module, attribute)
    except Exception as e:
        logger.error(f"Failed to fetch implementation for {module}.{attribute}: {e}")
        raise
