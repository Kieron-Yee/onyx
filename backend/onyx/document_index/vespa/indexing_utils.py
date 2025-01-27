"""
此文件包含与Vespa文档索引相关的工具函数。
主要功能包括：
- 文档块的存在性检查
- 文档的索引操作
- 批量索引处理
- 文档ID清理和验证
"""

import concurrent.futures
import json
import uuid
from datetime import datetime
from datetime import timezone
from http import HTTPStatus

import httpx
from retry import retry

from onyx.connectors.cross_connector_utils.miscellaneous_utils import (
    get_experts_stores_representations,
)
from onyx.document_index.document_index_utils import get_uuid_from_chunk
from onyx.document_index.document_index_utils import get_uuid_from_chunk_info_old
from onyx.document_index.interfaces import MinimalDocumentIndexingInfo
from onyx.document_index.vespa.shared_utils.utils import remove_invalid_unicode_chars
from onyx.document_index.vespa.shared_utils.utils import (
    replace_invalid_doc_id_characters,
)
from onyx.document_index.vespa_constants import ACCESS_CONTROL_LIST
from onyx.document_index.vespa_constants import BLURB
from onyx.document_index.vespa_constants import BOOST
from onyx.document_index.vespa_constants import CHUNK_ID
from onyx.document_index.vespa_constants import CONTENT
from onyx.document_index.vespa_constants import CONTENT_SUMMARY
from onyx.document_index.vespa_constants import DOC_UPDATED_AT
from onyx.document_index.vespa_constants import DOCUMENT_ID
from onyx.document_index.vespa_constants import DOCUMENT_ID_ENDPOINT
from onyx.document_index.vespa_constants import DOCUMENT_SETS
from onyx.document_index.vespa_constants import EMBEDDINGS
from onyx.document_index.vespa_constants import LARGE_CHUNK_REFERENCE_IDS
from onyx.document_index.vespa_constants import METADATA
from onyx.document_index.vespa_constants import METADATA_LIST
from onyx.document_index.vespa_constants import METADATA_SUFFIX
from onyx.document_index.vespa_constants import NUM_THREADS
from onyx.document_index.vespa_constants import PRIMARY_OWNERS
from onyx.document_index.vespa_constants import SECONDARY_OWNERS
from onyx.document_index.vespa_constants import SECTION_CONTINUATION
from onyx.document_index.vespa_constants import SEMANTIC_IDENTIFIER
from onyx.document_index.vespa_constants import SKIP_TITLE_EMBEDDING
from onyx.document_index.vespa_constants import SOURCE_LINKS
from onyx.document_index.vespa_constants import SOURCE_TYPE
from onyx.document_index.vespa_constants import TENANT_ID
from onyx.document_index.vespa_constants import TITLE
from onyx.document_index.vespa_constants import TITLE_EMBEDDING
from onyx.indexing.models import DocMetadataAwareIndexChunk
from onyx.utils.logger import setup_logger

logger = setup_logger()


@retry(tries=3, delay=1, backoff=2)
def _does_doc_chunk_exist(
    doc_chunk_id: uuid.UUID, index_name: str, http_client: httpx.Client
) -> bool:
    """
    检查指定的文档块是否存在于Vespa索引中。

    参数:
        doc_chunk_id: 文档块的唯一标识符
        index_name: 索引名称
        http_client: HTTP客户端实例

    返回:
        bool: 文档块是否存在
    """
    doc_url = f"{DOCUMENT_ID_ENDPOINT.format(index_name=index_name)}/{doc_chunk_id}"
    doc_fetch_response = http_client.get(doc_url)
    if doc_fetch_response.status_code == 404:
        return False

    if doc_fetch_response.status_code != 200:
        logger.debug(f"Failed to check for document with URL {doc_url}")  # 无法检查URL为{doc_url}的文档
        raise RuntimeError(
            f"Unexpected fetch document by ID value from Vespa: "  # 从Vespa获取文档时发生意外错误：
            f"error={doc_fetch_response.status_code} "  # 错误码
            f"index={index_name} "  # 索引
            f"doc_chunk_id={doc_chunk_id}"  # 文档块ID
        )
    return True


def _vespa_get_updated_at_attribute(t: datetime | None) -> int | None:
    """
    将datetime对象转换为Unix时间戳。

    参数:
        t: 日期时间对象，可以为None

    返回:
        int | None: Unix时间戳或None
    """
    if not t:
        return None

    if t.tzinfo != timezone.utc:
        raise ValueError("Connectors must provide document update time in UTC")  # 连接器必须提供UTC时间格式的文档更新时间

    return int(t.timestamp())


def get_existing_documents_from_chunks(
    chunks: list[DocMetadataAwareIndexChunk],
    index_name: str,
    http_client: httpx.Client,
    executor: concurrent.futures.ThreadPoolExecutor | None = None,
) -> set[str]:
    """
    从文档块列表中获取已存在的文档ID集合。

    参数:
        chunks: 文档块列表
        index_name: 索引名称
        http_client: HTTP客户端实例
        executor: 线程池执行器实例，可以为None

    返回:
        set[str]: 已存在的文档ID集合
    """
    external_executor = True

    if not executor:
        external_executor = False
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=NUM_THREADS)

    document_ids: set[str] = set()
    try:
        chunk_existence_future = {
            executor.submit(
                _does_doc_chunk_exist,
                get_uuid_from_chunk(chunk),
                index_name,
                http_client,
            ): chunk
            for chunk in chunks
        }
        for future in concurrent.futures.as_completed(chunk_existence_future):
            chunk = chunk_existence_future[future]
            chunk_already_existed = future.result()
            if chunk_already_existed:
                document_ids.add(chunk.source_document.id)

    finally:
        if not external_executor:
            executor.shutdown(wait=True)

    return document_ids


@retry(tries=5, delay=1, backoff=2)
def _index_vespa_chunk(
    chunk: DocMetadataAwareIndexChunk,
    index_name: str,
    http_client: httpx.Client,
    multitenant: bool,
) -> None:
    """
    将单个文档块索引到Vespa中。

    参数:
        chunk: 文档块实例
        index_name: 索引名称
        http_client: HTTP客户端实例
        multitenant: 是否为多租户模式
    """
    json_header = {
        "Content-Type": "application/json",
    }
    document = chunk.source_document

    # No minichunk documents in vespa, minichunk vectors are stored in the chunk itself
    # Vespa中没有minichunk文档，minichunk向量存储在chunk本身中
    vespa_chunk_id = str(get_uuid_from_chunk(chunk))
    embeddings = chunk.embeddings

    embeddings_name_vector_map = {"full_chunk": embeddings.full_embedding}

    if embeddings.mini_chunk_embeddings:
        for ind, m_c_embed in enumerate(embeddings.mini_chunk_embeddings):
            embeddings_name_vector_map[f"mini_chunk_{ind}"] = m_c_embed

    title = document.get_title_for_document_index()

    vespa_document_fields = {
        DOCUMENT_ID: document.id,
        CHUNK_ID: chunk.chunk_id,
        BLURB: remove_invalid_unicode_chars(chunk.blurb),
        TITLE: remove_invalid_unicode_chars(title) if title else None,
        SKIP_TITLE_EMBEDDING: not title,
        # For the BM25 index, the keyword suffix is used, the vector is already generated with the more
        # natural language representation of the metadata section
        # 对于BM25索引，使用关键词后缀，向量已经使用元数据部分的更自然语言表示生成
        CONTENT: remove_invalid_unicode_chars(
            f"{chunk.title_prefix}{chunk.content}{chunk.metadata_suffix_keyword}"
        ),
        # This duplication of `content` is needed for keyword highlighting
        # Note that it's not exactly the same as the actual content
        # which contains the title prefix and metadata suffix
        # 需要复制`content`用于关键词高亮
        # 注意这与实际内容并不完全相同，实际内容包含标题前缀和元数据后缀
        CONTENT_SUMMARY: remove_invalid_unicode_chars(chunk.content),
        SOURCE_TYPE: str(document.source.value),
        SOURCE_LINKS: json.dumps(chunk.source_links),
        SEMANTIC_IDENTIFIER: remove_invalid_unicode_chars(document.semantic_identifier),
        SECTION_CONTINUATION: chunk.section_continuation,
        LARGE_CHUNK_REFERENCE_IDS: chunk.large_chunk_reference_ids,
        METADATA: json.dumps(document.metadata),
        # Save as a list for efficient extraction as an Attribute
        # 保存为列表以便高效地作为属性提取
        METADATA_LIST: chunk.source_document.get_metadata_str_attributes(),
        METADATA_SUFFIX: chunk.metadata_suffix_keyword,
        EMBEDDINGS: embeddings_name_vector_map,
        TITLE_EMBEDDING: chunk.title_embedding,
        DOC_UPDATED_AT: _vespa_get_updated_at_attribute(document.doc_updated_at),
        PRIMARY_OWNERS: get_experts_stores_representations(document.primary_owners),
        SECONDARY_OWNERS: get_experts_stores_representations(document.secondary_owners),
        # the only `set` vespa has is `weightedset`, so we have to give each
        # element an arbitrary weight
        # Vespa中唯一的`set`类型是`weightedset`，所以我们必须给每个元素一个任意的权重
        # rkuo: acl, docset and boost metadata are also updated through the metadata sync queue
        # which only calls VespaIndex.update
        # rkuo注: acl、docset和boost元数据也通过metadata同步队列更新，该队列只调用VespaIndex.update
        ACCESS_CONTROL_LIST: {acl_entry: 1 for acl_entry in chunk.access.to_acl()},
        DOCUMENT_SETS: {document_set: 1 for document_set in chunk.document_sets},
        BOOST: chunk.boost,
    }

    if multitenant:
        if chunk.tenant_id:
            vespa_document_fields[TENANT_ID] = chunk.tenant_id

    vespa_url = f"{DOCUMENT_ID_ENDPOINT.format(index_name=index_name)}/{vespa_chunk_id}"
    logger.debug(f'Indexing to URL "{vespa_url}"')
    res = http_client.post(
        vespa_url, headers=json_header, json={"fields": vespa_document_fields}
    )
    try:
        res.raise_for_status()
    except Exception as e:
        logger.exception(
            f"Failed to index document: '{document.id}'. Got response: '{res.text}'"
        )
        if isinstance(e, httpx.HTTPStatusError):
            if e.response.status_code == HTTPStatus.INSUFFICIENT_STORAGE:
                logger.error(
                    "NOTE: HTTP Status 507 Insufficient Storage usually means "
                    "you need to allocate more memory or disk space to the "
                    "Vespa/index container."
                )

        raise e


def batch_index_vespa_chunks(
    chunks: list[DocMetadataAwareIndexChunk],
    index_name: str,
    http_client: httpx.Client,
    multitenant: bool,
    executor: concurrent.futures.ThreadPoolExecutor | None = None,
) -> None:
    """
    批量处理文档块的索引操作。

    参数:
        chunks: 文档块列表
        index_name: 索引名称
        http_client: HTTP客户端实例
        multitenant: 是否为多租户模式
        executor: 线程池执行器实例，可以为None
    """
    external_executor = True

    if not executor:
        external_executor = False
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=NUM_THREADS)

    try:
        chunk_index_future = {
            executor.submit(
                _index_vespa_chunk, chunk, index_name, http_client, multitenant
            ): chunk
            for chunk in chunks
        }
        for future in concurrent.futures.as_completed(chunk_index_future):
            # Will raise exception if any indexing raised an exception
            # 如果任何索引操作引发异常，将抛出异常
            future.result()

    finally:
        if not external_executor:
            executor.shutdown(wait=True)


def clean_chunk_id_copy(
    chunk: DocMetadataAwareIndexChunk,
) -> DocMetadataAwareIndexChunk:
    """
    清理和复制文档块ID。

    参数:
        chunk: 文档块实例

    返回:
        DocMetadataAwareIndexChunk: 清理后的文档块实例
    """
    clean_chunk = chunk.copy(
        update={
            "source_document": chunk.source_document.copy(
                update={
                    "id": replace_invalid_doc_id_characters(chunk.source_document.id)
                }
            )
        }
    )
    return clean_chunk


def check_for_final_chunk_existence(
    minimal_doc_info: MinimalDocumentIndexingInfo,
    start_index: int,
    index_name: str,
    http_client: httpx.Client,
) -> int:
    """
    检查最终文档块是否存在。

    参数:
        minimal_doc_info: 最小文档索引信息
        start_index: 起始索引
        index_name: 索引名称
        http_client: HTTP客户端实例

    返回:
        int: 最终文档块的索引
    """
    index = start_index
    while True:
        doc_chunk_id = get_uuid_from_chunk_info_old(
            document_id=minimal_doc_info.doc_id,
            chunk_id=index,
            large_chunk_reference_ids=[],
        )
        if not _does_doc_chunk_exist(doc_chunk_id, index_name, http_client):
            return index

        index += 1
