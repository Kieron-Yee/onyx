"""
此模块提供了与 Unstructured API 交互的功能，用于文档解析和文本提取。
主要包含 API 密钥管理和文档处理功能。
"""

from typing import Any
from typing import cast
from typing import IO

from unstructured.staging.base import dict_to_elements
from unstructured_client import UnstructuredClient  # type: ignore
from unstructured_client.models import operations  # type: ignore
from unstructured_client.models import shared

from onyx.configs.constants import KV_UNSTRUCTURED_API_KEY
from onyx.key_value_store.factory import get_kv_store
from onyx.key_value_store.interface import KvKeyNotFoundError
from onyx.utils.logger import setup_logger


logger = setup_logger()


def get_unstructured_api_key() -> str | None:
    """
    获取 Unstructured API 密钥。
    
    返回:
        str | None: 返回存储的 API 密钥，如果未找到则返回 None
    """
    kv_store = get_kv_store()
    try:
        return cast(str, kv_store.load(KV_UNSTRUCTURED_API_KEY))
    except KvKeyNotFoundError:
        return None


def update_unstructured_api_key(api_key: str) -> None:
    """
    更新 Unstructured API 密钥。
    
    参数:
        api_key (str): 要存储的新 API 密钥
    """
    kv_store = get_kv_store()
    kv_store.store(KV_UNSTRUCTURED_API_KEY, api_key)


def delete_unstructured_api_key() -> None:
    """
    删除存储的 Unstructured API 密钥。
    """
    kv_store = get_kv_store()
    kv_store.delete(KV_UNSTRUCTURED_API_KEY)


def _sdk_partition_request(
    file: IO[Any], file_name: str, **kwargs: Any
) -> operations.PartitionRequest:
    """
    创建用于文档分区的 SDK 请求对象。
    
    参数:
        file (IO[Any]): 要处理的文件对象
        file_name (str): 文件名
        **kwargs: 额外的请求参数
    
    返回:
        operations.PartitionRequest: 分区请求对象
        
    异常:
        Exception: 创建请求时发生错误
    """
    try:
        request = operations.PartitionRequest(
            partition_parameters=shared.PartitionParameters(
                files=shared.Files(content=file.read(), file_name=file_name),
                **kwargs,
            ),
        )
        return request
    except Exception as e:
        logger.error(f"Error creating partition request for file {file_name}: {str(e)}")
        # 创建文件 {file_name} 的分区请求时发生错误：{str(e)}
        raise


def unstructured_to_text(file: IO[Any], file_name: str) -> str:
    """
    将文档转换为纯文本格式。
    
    参数:
        file (IO[Any]): 要处理的文件对象
        file_name (str): 文件名
    
    返回:
        str: 提取的文本内容
        
    异常:
        ValueError: API 返回非 200 状态码时抛出
    """
    logger.debug(f"Starting to read file: {file_name}")
    # 开始读取文件：{file_name}
    
    req = _sdk_partition_request(file, file_name, strategy="auto")

    unstructured_client = UnstructuredClient(api_key_auth=get_unstructured_api_key())

    response = unstructured_client.general.partition(req)  # type: ignore
    elements = dict_to_elements(response.elements)

    if response.status_code != 200:
        err = f"Received unexpected status code {response.status_code} from Unstructured API."
        # 从 Unstructured API 收到意外的状态码 {response.status_code}
        logger.error(err)
        raise ValueError(err)

    return "\n\n".join(str(el) for el in elements)
