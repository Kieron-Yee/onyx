"""
此文件实现了从 Vespa 搜索引擎中删除文档块的功能。
主要提供了删除单个文档块和批量删除文档块的功能，支持并发操作和失败重试机制。
"""

import concurrent.futures
from uuid import UUID

import httpx
from retry import retry

from onyx.document_index.vespa_constants import DOCUMENT_ID_ENDPOINT
from onyx.document_index.vespa_constants import NUM_THREADS
from onyx.utils.logger import setup_logger

logger = setup_logger()


CONTENT_SUMMARY = "content_summary"


@retry(tries=10, delay=1, backoff=2)
def _retryable_http_delete(http_client: httpx.Client, url: str) -> None:
    """
    执行可重试的 HTTP DELETE 请求
    
    Args:
        http_client (httpx.Client): HTTP 客户端实例
        url (str): 要删除的资源 URL
        
    Raises:
        HTTPStatusError: 当 HTTP 请求失败时抛出
    """
    res = http_client.delete(url)
    res.raise_for_status()


def _delete_vespa_chunk(
    doc_chunk_id: UUID, index_name: str, http_client: httpx.Client
) -> None:
    """
    删除单个 Vespa 文档块
    
    Args:
        doc_chunk_id (UUID): 文档块的唯一标识符
        index_name (str): 索引名称
        http_client (httpx.Client): HTTP 客户端实例
        
    Raises:
        httpx.HTTPStatusError: 删除操作失败时抛出
    """
    try:
        _retryable_http_delete(
            http_client,
            f"{DOCUMENT_ID_ENDPOINT.format(index_name=index_name)}/{doc_chunk_id}",
        )
    except httpx.HTTPStatusError as e:
        logger.error(f"Failed to delete chunk, details: {e.response.text}")
        raise


def delete_vespa_chunks(
    doc_chunk_ids: list[UUID],
    index_name: str,
    http_client: httpx.Client,
    executor: concurrent.futures.ThreadPoolExecutor | None = None,
) -> None:
    """
    并发删除多个 Vespa 文档块
    
    Args:
        doc_chunk_ids (list[UUID]): 要删除的文档块 ID 列表
        index_name (str): 索引名称
        http_client (httpx.Client): HTTP 客户端实例
        executor (ThreadPoolExecutor | None): 线程池执行器，如果为 None 则创建新的
    
    Note:
        - 如果没有提供 executor，函数会创建一个新的线程池
        - 所有删除操作完成后，如果是内部创建的线程池会自动关闭
    """
    external_executor = True

    if not executor:
        external_executor = False
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=NUM_THREADS)

    try:
        chunk_deletion_future = {
            executor.submit(
                _delete_vespa_chunk, doc_chunk_id, index_name, http_client
            ): doc_chunk_id
            for doc_chunk_id in doc_chunk_ids
        }
        for future in concurrent.futures.as_completed(chunk_deletion_future):
            # 检查删除操作是否有异常发生 / Will raise exception if the deletion raised an exception
            future.result()

    finally:
        if not external_executor:
            executor.shutdown(wait=True)
