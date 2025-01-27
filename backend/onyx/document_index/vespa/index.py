"""
这是Onyx系统的Vespa索引实现模块。它提供了与Vespa搜索引擎的集成功能，包括文档的索引、更新、删除和检索等核心操作。
该模块实现了DocumentIndex接口，为系统提供了一个基于Vespa的文档存储和搜索解决方案。

主要功能包括:
- 文档的索引和批量处理
- 文档内容的更新
- 基于ID和混合方式的文档检索
- 多租户支持
- 文档访问控制
"""

import concurrent.futures
import io
import logging
import os
import random
import re
import time
import urllib
import zipfile
from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
from typing import BinaryIO
from typing import cast
from typing import List

import httpx  # type: ignore
import requests  # type: ignore

from onyx.configs.app_configs import DOCUMENT_INDEX_NAME
from onyx.configs.chat_configs import DOC_TIME_DECAY
from onyx.configs.chat_configs import NUM_RETURNED_HITS
from onyx.configs.chat_configs import TITLE_CONTENT_RATIO
from onyx.configs.chat_configs import VESPA_SEARCHER_THREADS
from onyx.configs.constants import KV_REINDEX_KEY
from onyx.context.search.models import IndexFilters
from onyx.context.search.models import InferenceChunkUncleaned
from onyx.document_index.document_index_utils import assemble_document_chunk_info
from onyx.document_index.interfaces import DocumentIndex
from onyx.document_index.interfaces import DocumentInsertionRecord
from onyx.document_index.interfaces import EnrichedDocumentIndexingInfo
from onyx.document_index.interfaces import IndexBatchParams
from onyx.document_index.interfaces import MinimalDocumentIndexingInfo
from onyx.document_index.interfaces import UpdateRequest
from onyx.document_index.interfaces import VespaChunkRequest
from onyx.document_index.interfaces import VespaDocumentFields
from onyx.document_index.vespa.chunk_retrieval import batch_search_api_retrieval
from onyx.document_index.vespa.chunk_retrieval import (
    get_all_vespa_ids_for_document_id,
)
from onyx.document_index.vespa.chunk_retrieval import (
    parallel_visit_api_retrieval,
)
from onyx.document_index.vespa.chunk_retrieval import query_vespa
from onyx.document_index.vespa.deletion import delete_vespa_chunks
from onyx.document_index.vespa.indexing_utils import batch_index_vespa_chunks
from onyx.document_index.vespa.indexing_utils import check_for_final_chunk_existence
from onyx.document_index.vespa.indexing_utils import clean_chunk_id_copy
from onyx.document_index.vespa.shared_utils.utils import get_vespa_http_client
from onyx.document_index.vespa.shared_utils.utils import (
    replace_invalid_doc_id_characters,
)
from onyx.document_index.vespa.shared_utils.vespa_request_builders import (
    build_vespa_filters,
)
from onyx.document_index.vespa_constants import ACCESS_CONTROL_LIST
from onyx.document_index.vespa_constants import BATCH_SIZE
from onyx.document_index.vespa_constants import BOOST
from onyx.document_index.vespa_constants import CONTENT_SUMMARY
from onyx.document_index.vespa_constants import DANSWER_CHUNK_REPLACEMENT_PAT
from onyx.document_index.vespa_constants import DATE_REPLACEMENT
from onyx.document_index.vespa_constants import DOCUMENT_ID_ENDPOINT
from onyx.document_index.vespa_constants import DOCUMENT_REPLACEMENT_PAT
from onyx.document_index.vespa_constants import DOCUMENT_SETS
from onyx.document_index.vespa_constants import HIDDEN
from onyx.document_index.vespa_constants import NUM_THREADS
from onyx.document_index.vespa_constants import SEARCH_THREAD_NUMBER_PAT
from onyx.document_index.vespa_constants import TENANT_ID_PAT
from onyx.document_index.vespa_constants import TENANT_ID_REPLACEMENT
from onyx.document_index.vespa_constants import VESPA_APPLICATION_ENDPOINT
from onyx.document_index.vespa_constants import VESPA_DIM_REPLACEMENT_PAT
from onyx.document_index.vespa_constants import VESPA_TIMEOUT
from onyx.document_index.vespa_constants import YQL_BASE
from onyx.indexing.models import DocMetadataAwareIndexChunk
from onyx.key_value_store.factory import get_kv_store
from onyx.utils.batching import batch_generator
from onyx.utils.logger import setup_logger
from shared_configs.configs import MULTI_TENANT
from shared_configs.model_server_models import Embedding


logger = setup_logger()

# Set the logging level to WARNING to ignore INFO and DEBUG logs
# 将日志级别设置为WARNING，忽略INFO和DEBUG日志
httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)


@dataclass
class _VespaUpdateRequest:
    """
    Vespa更新请求的数据类。
    
    属性:
        document_id: 文档ID
        url: 更新请求的URL
        update_request: 更新请求的具体内容字典
    """
    document_id: str
    url: str  
    update_request: dict[str, dict]


def in_memory_zip_from_file_bytes(file_contents: dict[str, bytes]) -> BinaryIO:
    """
    将文件内容压缩成内存中的ZIP文件。
    
    参数:
        file_contents: 包含文件名和对应内容的字典
        
    返回:
        一个包含ZIP文件内容的二进制IO对象
    """
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        for filename, content in file_contents.items():
            zipf.writestr(filename, content)
    zip_buffer.seek(0)
    return zip_buffer


def _create_document_xml_lines(doc_names: list[str | None] | list[str]) -> str:
    """
    生成Vespa文档XML配置行。
    
    参数:
        doc_names: 文档名称列表
        
    返回:
        包含所有文档配置的XML字符串
    """
    doc_lines = [
        f'<document type="{doc_name}" mode="index" />'
        for doc_name in doc_names
        if doc_name
    ]
    return "\n".join(doc_lines)


def add_ngrams_to_schema(schema_content: str) -> str:
    """
    向Vespa schema中的title和content字段添加n-gram配置。
    
    参数:
        schema_content: 原始schema内容
        
    返回:
        添加了n-gram配置后的schema内容
    """
    # Add the match blocks containing gram and gram-size to title and content fields
    # 向title和content字段添加包含gram和gram-size的匹配块
    schema_content = re.sub(
        r"(field title type string \{[^}]*indexing: summary \| index \| attribute)",
        r"\1\n            match {\n                gram\n                gram-size: 3\n            }",
        schema_content,
    )
    schema_content = re.sub(
        r"(field content type string \{[^}]*indexing: summary \| index)",
        r"\1\n            match {\n                gram\n                gram-size: 3\n            }",
        schema_content,
    )
    return schema_content


class VespaIndex(DocumentIndex):
    """
    Vespa索引实现类，提供文档的索引、更新、删除和检索功能。

    属性:
        index_name: 主索引名称
        secondary_index_name: 次要索引名称
        multitenant: 是否为多租户模式
        http_client: Vespa HTTP客户端
    """

    def __init__(
        self,
        index_name: str,
        secondary_index_name: str | None,
        multitenant: bool = False,
    ) -> None:
        """
        初始化VespaIndex实例。

        参数:
            index_name: 主索引名称
            secondary_index_name: 次要索引名称(可选)
            multitenant: 是否启用多租户模式
        """
        self.index_name = index_name
        self.secondary_index_name = secondary_index_name
        self.multitenant = multitenant
        self.http_client = get_vespa_http_client()

    def ensure_indices_exist(
        self,
        index_embedding_dim: int,
        secondary_index_embedding_dim: int | None,
    ) -> None:
        """
        确保Vespa索引存在，如果不存在则创建。
        
        参数:
            index_embedding_dim: 主索引的embedding维度
            secondary_index_embedding_dim: 次要索引的embedding维度(如果有的话)
        """
        if MULTI_TENANT:
            logger.info(
                "Skipping Vespa index seup for multitenant (would wipe all indices)"
                # 跳过多租户的Vespa索引设置(会清除所有索引)
            )
            return None

        deploy_url = f"{VESPA_APPLICATION_ENDPOINT}/tenant/default/prepareandactivate"
        logger.notice(f"Deploying Vespa application package to {deploy_url}")

        vespa_schema_path = os.path.join(
            os.getcwd(), "onyx", "document_index", "vespa", "app_config"
        )
        schema_file = os.path.join(vespa_schema_path, "schemas", "danswer_chunk.sd")
        services_file = os.path.join(vespa_schema_path, "services.xml")
        overrides_file = os.path.join(vespa_schema_path, "validation-overrides.xml")

        with open(services_file, "r") as services_f:
            services_template = services_f.read()

        schema_names = [self.index_name, self.secondary_index_name]

        doc_lines = _create_document_xml_lines(schema_names)
        services = services_template.replace(DOCUMENT_REPLACEMENT_PAT, doc_lines)
        services = services.replace(
            SEARCH_THREAD_NUMBER_PAT, str(VESPA_SEARCHER_THREADS)
        )

        kv_store = get_kv_store()

        needs_reindexing = False
        try:
            needs_reindexing = cast(bool, kv_store.load(KV_REINDEX_KEY))
        except Exception:
            logger.debug("Could not load the reindexing flag. Using ngrams")

        with open(overrides_file, "r") as overrides_f:
            overrides_template = overrides_f.read()

        # Vespa requires an override to erase data including the indices we're no longer using
        # It also has a 30 day cap from current so we set it to 7 dynamically
        # Vespa需要覆盖以擦除包括我们不再使用的索引在内的数据
        # 它有30天的当前时间上限，所以我们动态设置为7天
        now = datetime.now()
        date_in_7_days = now + timedelta(days=7)
        formatted_date = date_in_7_days.strftime("%Y-%m-%d")

        overrides = overrides_template.replace(DATE_REPLACEMENT, formatted_date)

        zip_dict = {
            "services.xml": services.encode("utf-8"),
            "validation-overrides.xml": overrides.encode("utf-8"),
        }

        with open(schema_file, "r") as schema_f:
            schema_template = schema_f.read()
        schema_template = schema_template.replace(TENANT_ID_PAT, "")

        schema = schema_template.replace(
            DANSWER_CHUNK_REPLACEMENT_PAT, self.index_name
        ).replace(VESPA_DIM_REPLACEMENT_PAT, str(index_embedding_dim))

        schema = add_ngrams_to_schema(schema) if needs_reindexing else schema
        schema = schema.replace(TENANT_ID_PAT, "")
        zip_dict[f"schemas/{schema_names[0]}.sd"] = schema.encode("utf-8")

        if self.secondary_index_name:
            upcoming_schema = schema_template.replace(
                DANSWER_CHUNK_REPLACEMENT_PAT, self.secondary_index_name
            ).replace(VESPA_DIM_REPLACEMENT_PAT, str(secondary_index_embedding_dim))
            zip_dict[f"schemas/{schema_names[1]}.sd"] = upcoming_schema.encode("utf-8")

        zip_file = in_memory_zip_from_file_bytes(zip_dict)

        headers = {"Content-Type": "application/zip"}
        response = requests.post(deploy_url, headers=headers, data=zip_file)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to prepare Vespa Onyx Index. Response: {response.text}"
            )

    @staticmethod
    def register_multitenant_indices(
        indices: list[str],
        embedding_dims: list[int],
    ) -> None:
        """
        注册多租户索引。仅在启用多租户模式时可用。

        参数:
            indices: 要注册的索引列表
            embedding_dims: 每个索引对应的嵌入维度列表

        异常:
            ValueError: 当多租户模式未启用时抛出
        """
        if not MULTI_TENANT:
            raise ValueError("Multi-tenant is not enabled")
            # 多租户未启用

        deploy_url = f"{VESPA_APPLICATION_ENDPOINT}/tenant/default/prepareandactivate"
        logger.info(f"Deploying Vespa application package to {deploy_url}")

        vespa_schema_path = os.path.join(
            os.getcwd(), "onyx", "document_index", "vespa", "app_config"
        )
        schema_file = os.path.join(vespa_schema_path, "schemas", "danswer_chunk.sd")
        services_file = os.path.join(vespa_schema_path, "services.xml")
        overrides_file = os.path.join(vespa_schema_path, "validation-overrides.xml")

        with open(services_file, "r") as services_f:
            services_template = services_f.read()

        # Generate schema names from index settings
        # 从索引设置生成schema名称
        schema_names = [index_name for index_name in indices]

        full_schemas = schema_names

        doc_lines = _create_document_xml_lines(full_schemas)

        services = services_template.replace(DOCUMENT_REPLACEMENT_PAT, doc_lines)
        services = services.replace(
            SEARCH_THREAD_NUMBER_PAT, str(VESPA_SEARCHER_THREADS)
        )

        kv_store = get_kv_store()

        needs_reindexing = False
        try:
            needs_reindexing = cast(bool, kv_store.load(KV_REINDEX_KEY))
        except Exception:
            logger.debug("Could not load the reindexing flag. Using ngrams")

        with open(overrides_file, "r") as overrides_f:
            overrides_template = overrides_f.read()

        # Vespa requires an override to erase data including the indices we're no longer using
        # It also has a 30 day cap from current so we set it to 7 dynamically
        # Vespa需要覆盖以擦除包括我们不再使用的索引在内的数据
        # 它有30天的当前时间上限，所以我们动态设置为7天
        now = datetime.now()
        date_in_7_days = now + timedelta(days=7)
        formatted_date = date_in_7_days.strftime("%Y-%m-%d")

        overrides = overrides_template.replace(DATE_REPLACEMENT, formatted_date)

        zip_dict = {
            "services.xml": services.encode("utf-8"),
            "validation-overrides.xml": overrides.encode("utf-8"),
        }

        with open(schema_file, "r") as schema_f:
            schema_template = schema_f.read()

        for i, index_name in enumerate(indices):
            embedding_dim = embedding_dims[i]
            logger.info(
                f"Creating index: {index_name} with embedding dimension: {embedding_dim}"
            )

            schema = schema_template.replace(
                DANSWER_CHUNK_REPLACEMENT_PAT, index_name
            ).replace(VESPA_DIM_REPLACEMENT_PAT, str(embedding_dim))
            schema = schema.replace(
                TENANT_ID_PAT, TENANT_ID_REPLACEMENT if MULTI_TENANT else ""
            )
            schema = add_ngrams_to_schema(schema) if needs_reindexing else schema
            zip_dict[f"schemas/{index_name}.sd"] = schema.encode("utf-8")

        zip_file = in_memory_zip_from_file_bytes(zip_dict)

        headers = {"Content-Type": "application/zip"}
        response = requests.post(deploy_url, headers=headers, data=zip_file)

        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to prepare Vespa Onyx Indexes. Response: {response.text}"
            )

    def index(
        self,
        chunks: list[DocMetadataAwareIndexChunk],
        index_batch_params: IndexBatchParams,
    ) -> set[DocumentInsertionRecord]:
        """
        将文档块批量索引到Vespa中并更新相关权限。
        假设一个文档不会被分割成多个块批次多次调用此函数，否则只会保留最后一组块。

        参数:
            chunks: 要索引的文档块列表
            index_batch_params: 索引批处理参数

        返回:
            文档插入记录集合
        """
        doc_id_to_previous_chunk_cnt = index_batch_params.doc_id_to_previous_chunk_cnt
        doc_id_to_new_chunk_cnt = index_batch_params.doc_id_to_new_chunk_cnt
        tenant_id = index_batch_params.tenant_id
        large_chunks_enabled = index_batch_params.large_chunks_enabled

        # IMPORTANT: This must be done one index at a time, do not use secondary index here
        # 重要：这必须一次处理一个索引，不要在这里使用次要索引
        cleaned_chunks = [clean_chunk_id_copy(chunk) for chunk in chunks]

        existing_docs: set[str] = set()

        # NOTE: using `httpx` here since `requests` doesn't support HTTP2. This is beneficial for
        # indexing / updates / deletes since we have to make a large volume of requests.
        # 注意：这里使用`httpx`是因为`requests`不支持HTTP2。这对于索引/更新/删除操作很有益，
        # 因为我们需要发送大量请求。
        with (
            concurrent.futures.ThreadPoolExecutor(max_workers=NUM_THREADS) as executor,
            get_vespa_http_client() as http_client,
        ):
            # We require the start and end index for each document in order to
            # know precisely which chunks to delete. This information exists for
            # documents that have `chunk_count` in the database, but not for
            # `old_version` documents.
            # 我们需要每个文档的起始和结束索引以准确知道要删除哪些块。
            # 这些信息存在于数据库中有`chunk_count`的文档中，但在`old_version`文档中不存在。

            enriched_doc_infos: list[EnrichedDocumentIndexingInfo] = []
            for document_id, _ in doc_id_to_previous_chunk_cnt.items():
                last_indexed_chunk = doc_id_to_previous_chunk_cnt.get(document_id, None)
                # If the document has no `chunk_count` in the database, we know that it
                # has the old chunk ID system and we must check for the final chunk index
                is_old_version = False
                if last_indexed_chunk is None:
                    is_old_version = True
                    minimal_doc_info = MinimalDocumentIndexingInfo(
                        doc_id=document_id,
                        chunk_start_index=doc_id_to_new_chunk_cnt.get(document_id, 0),
                    )
                    last_indexed_chunk = check_for_final_chunk_existence(
                        minimal_doc_info=minimal_doc_info,
                        start_index=doc_id_to_new_chunk_cnt[document_id],
                        index_name=self.index_name,
                        http_client=http_client,
                    )

                enriched_doc_info = EnrichedDocumentIndexingInfo(
                    doc_id=document_id,
                    chunk_start_index=doc_id_to_new_chunk_cnt.get(document_id, 0),
                    chunk_end_index=last_indexed_chunk,
                    old_version=is_old_version,
                )
                enriched_doc_infos.append(enriched_doc_info)

            # Now, for each doc, we know exactly where to start and end our deletion
            # So let's generate the chunk IDs for each chunk to delete
            # 现在，对于每个文档，我们确切知道从哪里开始和结束删除
            # 因此让我们为每个要删除的块生成块ID

            chunks_to_delete = assemble_document_chunk_info(
                enriched_document_info_list=enriched_doc_infos,
                tenant_id=tenant_id,
                large_chunks_enabled=large_chunks_enabled,
            )

            # Delete old Vespa documents
            for doc_chunk_ids_batch in batch_generator(chunks_to_delete, BATCH_SIZE):
                delete_vespa_chunks(
                    doc_chunk_ids=doc_chunk_ids_batch,
                    index_name=self.index_name,
                    http_client=http_client,
                    executor=executor,
                )

            for chunk_batch in batch_generator(cleaned_chunks, BATCH_SIZE):
                batch_index_vespa_chunks(
                    chunks=chunk_batch,
                    index_name=self.index_name,
                    http_client=http_client,
                    multitenant=self.multitenant,
                    executor=executor,
                )

        all_doc_ids = {chunk.source_document.id for chunk in cleaned_chunks}

        return {
            DocumentInsertionRecord(
                document_id=doc_id,
                already_existed=doc_id in existing_docs,
            )
            for doc_id in all_doc_ids
        }

    @staticmethod
    def _apply_updates_batched(
        updates: list[_VespaUpdateRequest],
        batch_size: int = BATCH_SIZE,
    ) -> None:
        """
        通过ThreadPoolExecutor并行执行批量更新请求。

        参数:
            updates: Vespa更新请求列表
            batch_size: 批处理大小，默认为BATCH_SIZE

        异常:
            requests.HTTPError: 当更新请求失败时抛出
        """
        def _update_chunk(
            update: _VespaUpdateRequest, http_client: httpx.Client
        ) -> httpx.Response:
            logger.debug(
                f"Updating with request to {update.url} with body {update.update_request}"
            )
            return http_client.put(
                update.url,
                headers={"Content-Type": "application/json"},
                json=update.update_request,
            )

        # NOTE: using `httpx` here since `requests` doesn't support HTTP2. This is beneficient for
        # indexing / updates / deletes since we have to make a large volume of requests.

        with (
            concurrent.futures.ThreadPoolExecutor(max_workers=NUM_THREADS) as executor,
            get_vespa_http_client() as http_client,
        ):
            for update_batch in batch_generator(updates, batch_size):
                future_to_document_id = {
                    executor.submit(
                        _update_chunk,
                        update,
                        http_client,
                    ): update.document_id
                    for update in update_batch
                }
                for future in concurrent.futures.as_completed(future_to_document_id):
                    res = future.result()
                    try:
                        res.raise_for_status()
                    except requests.HTTPError as e:
                        failure_msg = f"Failed to update document: {future_to_document_id[future]}"
                        raise requests.HTTPError(failure_msg) from e

    def update(self, update_requests: list[UpdateRequest]) -> None:
        """
        更新多个文档的字段。

        参数:
            update_requests: 更新请求列表，每个请求包含文档ID和要更新的字段
        """
        logger.debug(f"Updating {len(update_requests)} documents in Vespa")

        # Handle Vespa character limitations
        # Mutating update_requests but it's not used later anyway
        for update_request in update_requests:
            update_request.document_ids = [
                replace_invalid_doc_id_characters(doc_id)
                for doc_id in update_request.document_ids
            ]

        update_start = time.monotonic()

        processed_updates_requests: list[_VespaUpdateRequest] = []
        all_doc_chunk_ids: dict[str, list[str]] = {}

        # Fetch all chunks for each document ahead of time
        index_names = [self.index_name]
        if self.secondary_index_name:
            index_names.append(self.secondary_index_name)

        chunk_id_start_time = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
            future_to_doc_chunk_ids = {
                executor.submit(
                    get_all_vespa_ids_for_document_id,
                    document_id=document_id,
                    index_name=index_name,
                    filters=None,
                    get_large_chunks=True,
                ): (document_id, index_name)
                for index_name in index_names
                for update_request in update_requests
                for document_id in update_request.document_ids
            }
            for future in concurrent.futures.as_completed(future_to_doc_chunk_ids):
                document_id, index_name = future_to_doc_chunk_ids[future]
                try:
                    doc_chunk_ids = future.result()
                    if document_id not in all_doc_chunk_ids:
                        all_doc_chunk_ids[document_id] = []
                    all_doc_chunk_ids[document_id].extend(doc_chunk_ids)
                except Exception as e:
                    logger.error(
                        f"Error retrieving chunk IDs for document {document_id} in index {index_name}: {e}"
                    )
        logger.debug(
            f"Took {time.monotonic() - chunk_id_start_time:.2f} seconds to fetch all Vespa chunk IDs"
        )

        # Build the _VespaUpdateRequest objects
        for update_request in update_requests:
            update_dict: dict[str, dict] = {"fields": {}}
            if update_request.boost is not None:
                update_dict["fields"][BOOST] = {"assign": update_request.boost}
            if update_request.document_sets is not None:
                update_dict["fields"][DOCUMENT_SETS] = {
                    "assign": {
                        document_set: 1 for document_set in update_request.document_sets
                    }
                }
            if update_request.access is not None:
                update_dict["fields"][ACCESS_CONTROL_LIST] = {
                    "assign": {
                        acl_entry: 1 for acl_entry in update_request.access.to_acl()
                    }
                }
            if update_request.hidden is not None:
                update_dict["fields"][HIDDEN] = {"assign": update_request.hidden}

            if not update_dict["fields"]:
                logger.error("Update request received but nothing to update")
                continue

            for document_id in update_request.document_ids:
                for doc_chunk_id in all_doc_chunk_ids[document_id]:
                    processed_updates_requests.append(
                        _VespaUpdateRequest(
                            document_id=document_id,
                            url=f"{DOCUMENT_ID_ENDPOINT.format(index_name=self.index_name)}/{doc_chunk_id}",
                            update_request=update_dict,
                        )
                    )

        self._apply_updates_batched(processed_updates_requests)
        logger.debug(
            "Finished updating Vespa documents in %.2f seconds",
            time.monotonic() - update_start,
        )

    def update_single(self, doc_id: str, fields: VespaDocumentFields) -> int:
        """
        更新单个文档的字段。如果文档ID不存在，更新操作将是空操作且不会抛出异常。
        如果需要重试机制请自行处理其他异常。

        参数:
            doc_id: 要更新的文档ID
            fields: 要更新的字段值

        返回:
            更新的块数量
        """
        total_chunks_updated = 0

        # Handle Vespa character limitations
        # Mutating update_request but it's not used later anyway
        normalized_doc_id = replace_invalid_doc_id_characters(doc_id)

        # Build the _VespaUpdateRequest objects
        update_dict: dict[str, dict] = {"fields": {}}
        if fields.boost is not None:
            update_dict["fields"][BOOST] = {"assign": fields.boost}
        if fields.document_sets is not None:
            update_dict["fields"][DOCUMENT_SETS] = {
                "assign": {document_set: 1 for document_set in fields.document_sets}
            }
        if fields.access is not None:
            update_dict["fields"][ACCESS_CONTROL_LIST] = {
                "assign": {acl_entry: 1 for acl_entry in fields.access.to_acl()}
            }
        if fields.hidden is not None:
            update_dict["fields"][HIDDEN] = {"assign": fields.hidden}

        if not update_dict["fields"]:
            logger.error("Update request received but nothing to update")
            return 0

        index_names = [self.index_name]
        if self.secondary_index_name:
            index_names.append(self.secondary_index_name)

        with get_vespa_http_client(http2=False) as http_client:
            for index_name in index_names:
                params = httpx.QueryParams(
                    {
                        "selection": f"{index_name}.document_id=='{normalized_doc_id}'",
                        "cluster": DOCUMENT_INDEX_NAME,
                    }
                )

                while True:
                    try:
                        vespa_url = (
                            f"{DOCUMENT_ID_ENDPOINT.format(index_name=self.index_name)}"
                        )
                        logger.debug(f'update_single PUT on URL "{vespa_url}"')
                        resp = http_client.put(
                            vespa_url,
                            params=params,
                            headers={"Content-Type": "application/json"},
                            json=update_dict,
                        )

                        resp.raise_for_status()
                    except httpx.HTTPStatusError as e:
                        logger.error(
                            f"Failed to update chunks, details: {e.response.text}"
                        )
                        raise

                    resp_data = resp.json()

                    if "documentCount" in resp_data:
                        chunks_updated = resp_data["documentCount"]
                        total_chunks_updated += chunks_updated

                    # Check for continuation token to handle pagination
                    if "continuation" not in resp_data:
                        break  # Exit loop if no continuation token

                    if not resp_data["continuation"]:
                        break  # Exit loop if continuation token is empty

                    params = params.set("continuation", resp_data["continuation"])

                logger.debug(
                    f"VespaIndex.update_single: "
                    f"index={index_name} "
                    f"doc={normalized_doc_id} "
                    f"chunks_updated={total_chunks_updated}"
                )

        return total_chunks_updated

    def delete_single(self, doc_id: str) -> int:
        """
        删除单个文档。由于使用单个删除调用和选择查询，可能比批量删除方法更快。

        参数:
            doc_id: 要删除的文档ID

        返回:
            删除的块数量
        """
        total_chunks_deleted = 0

        # Vespa deletion is poorly documented ... luckily we found this
        # https://docs.vespa.ai/en/operations/batch-delete.html#example

        doc_id = replace_invalid_doc_id_characters(doc_id)

        # NOTE: using `httpx` here since `requests` doesn't support HTTP2. This is beneficial for
        # indexing / updates / deletes since we have to make a large volume of requests.
        index_names = [self.index_name]
        if self.secondary_index_name:
            index_names.append(self.secondary_index_name)

        with get_vespa_http_client(http2=False) as http_client:
            for index_name in index_names:
                params = httpx.QueryParams(
                    {
                        "selection": f"{index_name}.document_id=='{doc_id}'",
                        "cluster": DOCUMENT_INDEX_NAME,
                    }
                )

                while True:
                    try:
                        vespa_url = (
                            f"{DOCUMENT_ID_ENDPOINT.format(index_name=index_name)}"
                        )
                        logger.debug(f'delete_single DELETE on URL "{vespa_url}"')
                        resp = http_client.delete(
                            vespa_url,
                            params=params,
                        )
                        resp.raise_for_status()
                    except httpx.HTTPStatusError as e:
                        logger.error(
                            f"Failed to delete chunk, details: {e.response.text}"
                        )
                        raise

                    resp_data = resp.json()

                    if "documentCount" in resp_data:
                        chunks_deleted = resp_data["documentCount"]
                        total_chunks_deleted += chunks_deleted

                    # Check for continuation token to handle pagination
                    if "continuation" not in resp_data:
                        break  # Exit loop if no continuation token

                    if not resp_data["continuation"]:
                        break  # Exit loop if continuation token is empty

                    params = params.set("continuation", resp_data["continuation"])

                logger.debug(
                    f"VespaIndex.delete_single: "
                    f"index={index_name} "
                    f"doc={doc_id} "
                    f"chunks_deleted={total_chunks_deleted}"
                )

        return total_chunks_deleted

    def id_based_retrieval(
        self,
        chunk_requests: list[VespaChunkRequest],
        filters: IndexFilters,
        batch_retrieval: bool = False,
        get_large_chunks: bool = False,
    ) -> list[InferenceChunkUncleaned]:
        """
        基于ID的文档检索。

        参数:
            chunk_requests: 块请求列表
            filters: 索引过滤条件
            batch_retrieval: 是否使用批量检索
            get_large_chunks: 是否获取大块文档

        返回:
            检索到的文档块列表
        """
        if batch_retrieval:
            return batch_search_api_retrieval(
                index_name=self.index_name,
                chunk_requests=chunk_requests,
                filters=filters,
                get_large_chunks=get_large_chunks,
            )
        return parallel_visit_api_retrieval(
            index_name=self.index_name,
            chunk_requests=chunk_requests,
            filters=filters,
            get_large_chunks=get_large_chunks,
        )

    def hybrid_retrieval(
        self,
        query: str,
        query_embedding: Embedding,
        final_keywords: list[str] | None,
        filters: IndexFilters,
        hybrid_alpha: float,
        time_decay_multiplier: float,
        num_to_retrieve: int,
        offset: int = 0,
        title_content_ratio: float | None = TITLE_CONTENT_RATIO,
    ) -> list[InferenceChunkUncleaned]:
        """
        执行混合检索,结合向量搜索和关键词搜索。

        参数:
            query: 搜索查询
            query_embedding: 查询文本的embedding向量
            final_keywords: 最终的关键词列表
            filters: 索引过滤条件
            hybrid_alpha: 混合搜索权重系数
            time_decay_multiplier: 时间衰减乘数
            num_to_retrieve: 需要检索的文档数量
            offset: 分页偏移量
            title_content_ratio: 标题和内容的权重比例

        返回:
            符合条件的文档块列表
        """
        vespa_where_clauses = build_vespa_filters(filters)
        # Needs to be at least as much as the value set in Vespa schema config
        # 需要至少和Vespa schema配置中设置的值一样大
        target_hits = max(10 * num_to_retrieve, 1000)
        yql = (
            YQL_BASE.format(index_name=self.index_name)
            + vespa_where_clauses
            + f"(({{targetHits: {target_hits}}}nearestNeighbor(embeddings, query_embedding)) "
            + f"or ({{targetHits: {target_hits}}}nearestNeighbor(title_embedding, query_embedding)) "
            + 'or ({grammar: "weakAnd"}userInput(@query)) '
            + f'or ({{defaultIndex: "{CONTENT_SUMMARY}"}}userInput(@query)))'
        )

        final_query = " ".join(final_keywords) if final_keywords else query

        logger.debug(f"Query YQL: {yql}")

        params: dict[str, str | int | float] = {
            "yql": yql,
            "query": final_query,
            "input.query(query_embedding)": str(query_embedding),
            "input.query(decay_factor)": str(DOC_TIME_DECAY * time_decay_multiplier),
            "input.query(alpha)": hybrid_alpha,
            "input.query(title_content_ratio)": title_content_ratio
            if title_content_ratio is not None
            else TITLE_CONTENT_RATIO,
            "hits": num_to_retrieve,
            "offset": offset,
            "ranking.profile": f"hybrid_search{len(query_embedding)}",
            "timeout": VESPA_TIMEOUT,
        }

        return query_vespa(params)

    def admin_retrieval(
        self,
        query: str,
        filters: IndexFilters,
        num_to_retrieve: int = NUM_RETURNED_HITS,
        offset: int = 0,
    ) -> list[InferenceChunkUncleaned]:
        """
        执行管理员检索,可以检索到包括隐藏内容在内的所有文档。

        参数:
            query: 搜索查询
            filters: 索引过滤条件
            num_to_retrieve: 需要检索的文档数量
            offset: 分页偏移量
            
        返回:
            符合条件的文档块列表
        """
        vespa_where_clauses = build_vespa_filters(filters, include_hidden=True)
        yql = (
            YQL_BASE.format(index_name=self.index_name)
            + vespa_where_clauses
            + '({grammar: "weakAnd"}userInput(@query) '
            # `({defaultIndex: "content_summary"}userInput(@query))` section is
            # needed for highlighting while the N-gram highlighting is broken / 
            # not working as desired
            # 在N-gram高亮功能出现问题/无法按预期工作时,需要使用content_summary索引来实现高亮
            + f'or ({{defaultIndex: "{CONTENT_SUMMARY}"}}userInput(@query)))'
        )

        params: dict[str, str | int] = {
            "yql": yql,
            "query": query,
            "hits": num_to_retrieve,
            "offset": 0,
            "ranking.profile": "admin_search",
            "timeout": VESPA_TIMEOUT,
        }

        return query_vespa(params)

    @classmethod
    def delete_entries_by_tenant_id(cls, tenant_id: str, index_name: str) -> None:
        """
        Deletes all entries in the specified index with the given tenant_id.

        Parameters:
            tenant_id (str): The tenant ID whose documents are to be deleted.
            index_name (str): The name of the index from which to delete documents.
        """
        logger.info(
            f"Deleting entries with tenant_id: {tenant_id} from index: {index_name}"
        )

        # Step 1: Retrieve all document IDs with the given tenant_id
        document_ids = cls._get_all_document_ids_by_tenant_id(tenant_id, index_name)

        if not document_ids:
            logger.info(
                f"No documents found with tenant_id: {tenant_id} in index: {index_name}"
            )
            return

        # Step 2: Delete documents in batches
        delete_requests = [
            _VespaDeleteRequest(document_id=doc_id, index_name=index_name)
            for doc_id in document_ids
        ]

        cls._apply_deletes_batched(delete_requests)

    @classmethod
    def _get_all_document_ids_by_tenant_id(
        cls, tenant_id: str, index_name: str
    ) -> List[str]:
        """
        获取指定tenant_id的所有文档ID，处理分页。

        参数:
            tenant_id (str): 要搜索的租户ID
            index_name (str): 要搜索的索引名称

        返回:
            List[str]: 匹配tenant_id的文档ID列表
        """
        offset = 0
        limit = 1000  # Vespa's maximum hits per query
        document_ids = []

        logger.debug(
            f"Starting document ID retrieval for tenant_id: {tenant_id} in index: {index_name}"
        )

        while True:
            # Construct the query to fetch document IDs
            # 构造查询以获取文档ID
            query_params = {
                "yql": f'select id from sources * where tenant_id contains "{tenant_id}";',
                "offset": str(offset),
                "hits": str(limit),
                "timeout": "10s",
                "format": "json",
                "summary": "id",
            }

            url = f"{VESPA_APPLICATION_ENDPOINT}/search/"

            logger.debug(
                f"Querying for document IDs with tenant_id: {tenant_id}, offset: {offset}"
            )

            with get_vespa_http_client(no_timeout=True) as http_client:
                response = http_client.get(url, params=query_params)
                response.raise_for_status()

                search_result = response.json()
                hits = search_result.get("root", {}).get("children", [])

                if not hits:
                    break

                for hit in hits:
                    doc_id = hit.get("id")
                    if doc_id:
                        document_ids.append(doc_id)

                offset += limit  # Move to the next page

        logger.debug(
            f"Retrieved {len(document_ids)} document IDs for tenant_id: {tenant_id}"
        )
        return document_ids

    @classmethod
    def _apply_deletes_batched(
        cls,
        delete_requests: List["_VespaDeleteRequest"],
        batch_size: int = BATCH_SIZE,
    ) -> None:
        """
        使用多线程批量删除文档。

        参数:
            delete_requests: 删除请求列表，每个请求包含一个文档ID和对应的URL
            batch_size: 批处理大小，默认为BATCH_SIZE
        """

        def _delete_document(
            delete_request: "_VespaDeleteRequest", http_client: httpx.Client
        ) -> None:
            logger.debug(f"Deleting document with ID {delete_request.document_id}")
            response = http_client.delete(
                delete_request.url,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()

        logger.debug(f"Starting batch deletion for {len(delete_requests)} documents")

        with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
            with get_vespa_http_client(no_timeout=True) as http_client:
                for batch_start in range(0, len(delete_requests), batch_size):
                    batch = delete_requests[batch_start : batch_start + batch_size]

                    future_to_document_id = {
                        executor.submit(
                            _delete_document,
                            delete_request,
                            http_client,
                        ): delete_request.document_id
                        for delete_request in batch
                    }

                    for future in concurrent.futures.as_completed(
                        future_to_document_id
                    ):
                        doc_id = future_to_document_id[future]
                        try:
                            future.result()
                            logger.debug(f"Successfully deleted document: {doc_id}")
                        except httpx.HTTPError as e:
                            logger.error(f"Failed to delete document {doc_id}: {e}")
                            # Optionally, implement retry logic or error handling here

        logger.info("Batch deletion completed")

    def random_retrieval(
        self,
        filters: IndexFilters,
        num_to_retrieve: int = 10,
    ) -> list[InferenceChunkUncleaned]:
        """
        使用Vespa的随机排序检索匹配过滤条件的随机文档块。

        此方法当前用于在助手启动消息创建的上下文中进行随机块检索(作为样本上下文供助手使用)。

        参数:
            filters: 索引过滤条件
            num_to_retrieve: 需要检索的文档数量,默认为10

        返回:
            随机选择的符合条件的文档块列表
        """
        vespa_where_clauses = build_vespa_filters(filters, remove_trailing_and=True)

        yql = YQL_BASE.format(index_name=self.index_name) + vespa_where_clauses

        random_seed = random.randint(0, 1000000)

        params: dict[str, str | int | float] = {
            "yql": yql,
            "hits": num_to_retrieve,
            "timeout": VESPA_TIMEOUT,
            "ranking.profile": "random_",
            "ranking.properties.random.seed": random_seed,
        }

        return query_vespa(params)


class _VespaDeleteRequest:
    """
    Vespa删除请求的数据类。
    
    属性:
        document_id: 要删除的文档ID
        url: 删除请求的URL,会对document_id进行URL编码以确保安全
    """
    def __init__(self, document_id: str, index_name: str) -> None:
        self.document_id = document_id
        # Encode the document ID to ensure it's safe for use in the URL
        # 对文档ID进行URL编码以确保在URL中使用时的安全性
        encoded_doc_id = urllib.parse.quote_plus(self.document_id)
        self.url = (
            f"{VESPA_APPLICATION_ENDPOINT}/document/v1/"
            f"{index_name}/{index_name}/docid/{encoded_doc_id}"
        )
