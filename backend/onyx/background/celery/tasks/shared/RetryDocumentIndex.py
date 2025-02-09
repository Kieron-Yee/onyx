"""
此文件实现了一个文档索引重试包装器，用于处理与Vespa交互时的读取超时情况。
主要提供了对文档的删除和更新操作的重试机制。
"""

import httpx
from tenacity import retry
from tenacity import retry_if_exception_type
from tenacity import stop_after_delay
from tenacity import wait_random_exponential

from onyx.document_index.interfaces import DocumentIndex
from onyx.document_index.interfaces import VespaDocumentFields


class RetryDocumentIndex:
    """A wrapper class to help with specific retries against Vespa involving
    read timeouts.

    wait_random_exponential implements full jitter as per this article:
    https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/
    
    一个包装类，用于处理针对Vespa的特定重试操作，主要处理读取超时的情况。
    
    wait_random_exponential实现了完全抖动策略，详见文章：
    https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/
    """

    # 最大等待时间（秒）
    MAX_WAIT = 30

    # STOP_AFTER + MAX_WAIT应该比celery的soft_time_limit稍小（约5秒）
    # 停止重试的时间限制（秒）
    STOP_AFTER = 70

    def __init__(self, index: DocumentIndex):
        """
        初始化RetryDocumentIndex实例
        
        参数:
            index (DocumentIndex): 文档索引接口实例
        """
        self.index: DocumentIndex = index

    @retry(
        retry=retry_if_exception_type(httpx.ReadTimeout),
        wait=wait_random_exponential(multiplier=1, max=MAX_WAIT),
        stop=stop_after_delay(STOP_AFTER),
    )
    def delete_single(self, doc_id: str) -> int:
        """
        删除单个文档，带有重试机制
        
        参数:
            doc_id (str): 要删除的文档ID
            
        返回:
            int: 操作状态码
        """
        return self.index.delete_single(doc_id)

    @retry(
        retry=retry_if_exception_type(httpx.ReadTimeout),
        wait=wait_random_exponential(multiplier=1, max=MAX_WAIT),
        stop=stop_after_delay(STOP_AFTER),
    )
    def update_single(self, doc_id: str, fields: VespaDocumentFields) -> int:
        """
        更新单个文档，带有重试机制
        
        参数:
            doc_id (str): 要更新的文档ID
            fields (VespaDocumentFields): 要更新的字段
            
        返回:
            int: 操作状态码
        """
        return self.index.update_single(doc_id, fields)
