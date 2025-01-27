"""
文件功能说明：
这个模块实现了文档索引流水线的核心功能。主要负责：
1. 文档的批量处理和索引
2. 文档的分块和嵌入向量生成
3. 文档访问权限的管理
4. 数据库操作的协调
"""

import traceback
from collections.abc import Callable
from functools import partial
from http import HTTPStatus
from typing import Protocol

import httpx
from pydantic import BaseModel
from pydantic import ConfigDict
from sqlalchemy.orm import Session

from onyx.access.access import get_access_for_documents
from onyx.access.models import DocumentAccess
from onyx.configs.app_configs import ENABLE_MULTIPASS_INDEXING
from onyx.configs.app_configs import INDEXING_EXCEPTION_LIMIT
from onyx.configs.app_configs import MAX_DOCUMENT_CHARS
from onyx.configs.constants import DEFAULT_BOOST
from onyx.connectors.cross_connector_utils.miscellaneous_utils import (
    get_experts_stores_representations,
)
from onyx.connectors.models import Document
from onyx.connectors.models import IndexAttemptMetadata
from onyx.db.document import fetch_chunk_counts_for_documents
from onyx.db.document import get_documents_by_ids
from onyx.db.document import prepare_to_modify_documents
from onyx.db.document import update_docs_chunk_count__no_commit
from onyx.db.document import update_docs_last_modified__no_commit
from onyx.db.document import update_docs_updated_at__no_commit
from onyx.db.document import upsert_document_by_connector_credential_pair
from onyx.db.document import upsert_documents
from onyx.db.document_set import fetch_document_sets_for_documents
from onyx.db.index_attempt import create_index_attempt_error
from onyx.db.models import Document as DBDocument
from onyx.db.search_settings import get_current_search_settings
from onyx.db.tag import create_or_add_document_tag
from onyx.db.tag import create_or_add_document_tag_list
from onyx.document_index.interfaces import DocumentIndex
from onyx.document_index.interfaces import DocumentMetadata
from onyx.document_index.interfaces import IndexBatchParams
from onyx.indexing.chunker import Chunker
from onyx.indexing.embedder import IndexingEmbedder
from onyx.indexing.indexing_heartbeat import IndexingHeartbeatInterface
from onyx.indexing.models import DocAwareChunk
from onyx.indexing.models import DocMetadataAwareIndexChunk
from onyx.utils.logger import setup_logger
from onyx.utils.timing import log_function_time
from shared_configs.enums import EmbeddingProvider

logger = setup_logger()


class DocumentBatchPrepareContext(BaseModel):
    """
    文档批处理上下文类，用于存储待更新的文档列表和数据库文档映射
    
    属性：
        updatable_docs: 需要更新的文档列表
        id_to_db_doc_map: 文档ID到数据库文档对象的映射字典
    """
    updatable_docs: list[Document]
    id_to_db_doc_map: dict[str, DBDocument]
    model_config = ConfigDict(arbitrary_types_allowed=True)


class IndexingPipelineProtocol(Protocol):
    """
    索引流水线协议类，定义了文档批量索引的接口
    """
    def __call__(
        self,
        document_batch: list[Document],
        index_attempt_metadata: IndexAttemptMetadata,
    ) -> tuple[int, int]:
        """
        处理文档批次的调用接口
        
        参数：
            document_batch: 待处理的文档列表
            index_attempt_metadata: 索引尝试的元数据
            
        返回：
            tuple[int, int]: 返回(新文档数量, 块数量)的元组
        """
        ...


def _upsert_documents_in_db(
    documents: list[Document],
    index_attempt_metadata: IndexAttemptMetadata,
    db_session: Session,
) -> None:
    """
    将文档信息更新或插入到数据库中
    
    参数：
        documents: 需要处理的文档列表
        index_attempt_metadata: 索引尝试的元数据
        db_session: 数据库会话对象
    """
    # Metadata here refers to basic document info, not metadata about the actual content
    # 这里的元数据指基本文档信息，而不是实际内容的元数据
    document_metadata_list: list[DocumentMetadata] = []
    for doc in documents:
        first_link = next(
            (section.link for section in doc.sections if section.link), ""
        )
        db_doc_metadata = DocumentMetadata(
            connector_id=index_attempt_metadata.connector_id,
            credential_id=index_attempt_metadata.credential_id,
            document_id=doc.id,
            semantic_identifier=doc.semantic_identifier,
            first_link=first_link,
            primary_owners=get_experts_stores_representations(doc.primary_owners),
            secondary_owners=get_experts_stores_representations(doc.secondary_owners),
            from_ingestion_api=doc.from_ingestion_api,
        )
        document_metadata_list.append(db_doc_metadata)

    upsert_documents(db_session, document_metadata_list)

    # Insert document content metadata
    for doc in documents:
        for k, v in doc.metadata.items():
            if isinstance(v, list):
                create_or_add_document_tag_list(
                    tag_key=k,
                    tag_values=v,
                    source=doc.source,
                    document_id=doc.id,
                    db_session=db_session,
                )
                continue

            create_or_add_document_tag(
                tag_key=k,
                tag_value=v,
                source=doc.source,
                document_id=doc.id,
                db_session=db_session,
            )


def get_doc_ids_to_update(
    documents: list[Document], db_docs: list[DBDocument]
) -> list[Document]:
    """
    确定需要更新的文档列表
    
    参数：
        documents: 待检查的文档列表
        db_docs: 数据库中已存在的文档列表
        
    返回：
        需要更新的文档列表
    
    说明：
        如果文档已存在且updated_at时间戳未发生变化，则不需要更新
    """
    id_update_time_map = {
        doc.id: doc.doc_updated_at for doc in db_docs if doc.doc_updated_at
    }

    updatable_docs: list[Document] = []
    for doc in documents:
        if (
            doc.id in id_update_time_map
            and doc.doc_updated_at
            and doc.doc_updated_at <= id_update_time_map[doc.id]
        ):
            continue
        updatable_docs.append(doc)

    return updatable_docs


def index_doc_batch_with_handler(
    *,
    chunker: Chunker,
    embedder: IndexingEmbedder,
    document_index: DocumentIndex,
    document_batch: list[Document],
    index_attempt_metadata: IndexAttemptMetadata,
    attempt_id: int | None,
    db_session: Session,
    ignore_time_skip: bool = False,
    tenant_id: str | None = None,
) -> tuple[int, int]:
    """
    文档批次索引处理函数，包含异常处理机制
    
    参数：
        chunker: 文档分块器
        embedder: 嵌入向量生成器
        document_index: 文档索引对象
        document_batch: 待处理的文档批次
        index_attempt_metadata: 索引尝试的元数据
        attempt_id: 尝试ID
        db_session: 数据库会话对象
        ignore_time_skip: 是否忽略时间跳过检查
        tenant_id: 租户ID
        
    返回：
        tuple[int, int]: 返回(新文档数量, 块数量)的元组
    """
    r = (0, 0)
    try:
        r = index_doc_batch(
            chunker=chunker,
            embedder=embedder,
            document_index=document_index,
            document_batch=document_batch,
            index_attempt_metadata=index_attempt_metadata,
            db_session=db_session,
            ignore_time_skip=ignore_time_skip,
            tenant_id=tenant_id,
        )
    except Exception as e:
        if isinstance(e, httpx.HTTPStatusError):
            if e.response.status_code == HTTPStatus.INSUFFICIENT_STORAGE:
                logger.error(
                    "NOTE: HTTP Status 507 Insufficient Storage indicates "
                    "you need to allocate more memory or disk space to the "
                    "Vespa/index container."
                )

        if INDEXING_EXCEPTION_LIMIT == 0:
            raise

        trace = traceback.format_exc()
        create_index_attempt_error(
            attempt_id,
            batch=index_attempt_metadata.batch_num,
            docs=document_batch,
            exception_msg=str(e),
            exception_traceback=trace,
            db_session=db_session,
        )
        logger.exception(
            f"Indexing batch {index_attempt_metadata.batch_num} failed. msg='{e}' trace='{trace}'"
        )

        index_attempt_metadata.num_exceptions += 1
        if index_attempt_metadata.num_exceptions == INDEXING_EXCEPTION_LIMIT:
            logger.warning(
                f"Maximum number of exceptions for this index attempt "
                f"({INDEXING_EXCEPTION_LIMIT}) has been reached. "
                f"The next exception will abort the indexing attempt."
            )
        elif index_attempt_metadata.num_exceptions > INDEXING_EXCEPTION_LIMIT:
            logger.warning(
                f"Maximum number of exceptions for this index attempt "
                f"({INDEXING_EXCEPTION_LIMIT}) has been exceeded."
            )
            raise RuntimeError(
                f"Maximum exception limit of {INDEXING_EXCEPTION_LIMIT} exceeded."
            )
        else:
            pass

    return r


def index_doc_batch_prepare(
    documents: list[Document],
    index_attempt_metadata: IndexAttemptMetadata,
    db_session: Session,
    ignore_time_skip: bool = False,
) -> DocumentBatchPrepareContext | None:
    """
    准备文档批次的索引工作
    
    参数：
        documents: 待处理的文档列表
        index_attempt_metadata: 索引尝试的元数据
        db_session: 数据库会话对象
        ignore_time_skip: 是否忽略时间跳过检查
        
    返回：
        DocumentBatchPrepareContext | None: 返回文档批处理上下文对象或None
    """
    # Create a trimmed list of docs that don't have a newer updated at
    # Shortcuts the time-consuming flow on connector index retries
    document_ids: list[str] = [document.id for document in documents]
    db_docs: list[DBDocument] = get_documents_by_ids(
        db_session=db_session,
        document_ids=document_ids,
    )

    updatable_docs = (
        get_doc_ids_to_update(documents=documents, db_docs=db_docs)
        if not ignore_time_skip
        else documents
    )

    # for all updatable docs, upsert into the DB
    # Does not include doc_updated_at which is also used to indicate a successful update
    if updatable_docs:
        _upsert_documents_in_db(
            documents=updatable_docs,
            index_attempt_metadata=index_attempt_metadata,
            db_session=db_session,
        )

    logger.info(
        f"Upserted {len(updatable_docs)} changed docs out of "
        f"{len(documents)} total docs into the DB"
    )

    # for all docs, upsert the document to cc pair relationship
    upsert_document_by_connector_credential_pair(
        db_session,
        index_attempt_metadata.connector_id,
        index_attempt_metadata.credential_id,
        document_ids,
    )

    # No docs to process because the batch is empty or every doc was already indexed
    if not updatable_docs:
        return None

    id_to_db_doc_map = {doc.id: doc for doc in db_docs}
    return DocumentBatchPrepareContext(
        updatable_docs=updatable_docs, id_to_db_doc_map=id_to_db_doc_map
    )


def filter_documents(document_batch: list[Document]) -> list[Document]:
    """
    过滤文档批次，移除无效或不符合条件的文档
    
    参数：
        document_batch: 待过滤的文档列表
        
    返回：
        过滤后的文档列表
    """
    documents: list[Document] = []
    for document in document_batch:
        # Remove any NUL characters from title/semantic_id
        # This is a known issue with the Zendesk connector
        # Postgres cannot handle NUL characters in text fields
        if document.title:
            document.title = document.title.replace("\x00", "")
        if document.semantic_identifier:
            document.semantic_identifier = document.semantic_identifier.replace(
                "\x00", ""
            )

        # Remove NUL characters from all sections
        for section in document.sections:
            if section.text is not None:
                section.text = section.text.replace("\x00", "")

        empty_contents = not any(section.text.strip() for section in document.sections)
        if (
            (not document.title or not document.title.strip())
            and not document.semantic_identifier.strip()
            and empty_contents
        ):
            # Skip documents that have neither title nor content
            # If the document doesn't have either, then there is no useful information in it
            # This is again verified later in the pipeline after chunking but at that point there should
            # already be no documents that are empty.
            logger.warning(
                f"Skipping document with ID {document.id} as it has neither title nor content."
            )
            continue

        if document.title is not None and not document.title.strip() and empty_contents:
            # The title is explicitly empty ("" and not None) and the document is empty
            # so when building the chunk text representation, it will be empty and unuseable
            logger.warning(
                f"Skipping document with ID {document.id} as the chunks will be empty."
            )
            continue

        section_chars = sum(len(section.text) for section in document.sections)
        if (
            MAX_DOCUMENT_CHARS
            and len(document.title or document.semantic_identifier) + section_chars
            > MAX_DOCUMENT_CHARS
        ):
            # Skip documents that are too long, later on there are more memory intensive steps done on the text
            # and the container will run out of memory and crash. Several other checks are included upstream but
            # those are at the connector level so a catchall is still needed.
            # Assumption here is that files that are that long, are generated files and not the type users
            # generally care for.
            logger.warning(
                f"Skipping document with ID {document.id} as it is too long."
            )
            continue

        documents.append(document)
    return documents


@log_function_time(debug_only=True)
def index_doc_batch(
    *,
    document_batch: list[Document],
    chunker: Chunker,
    embedder: IndexingEmbedder,
    document_index: DocumentIndex,
    index_attempt_metadata: IndexAttemptMetadata,
    db_session: Session,
    ignore_time_skip: bool = False,
    tenant_id: str | None = None,
    filter_fnc: Callable[[list[Document]], list[Document]] = filter_documents,
) -> tuple[int, int]:
    """
    处理文档批次并进行索引
    
    参数：
        document_batch: 待处理的文档列表
        chunker: 文档分块器
        embedder: 嵌入向量生成器
        document_index: 文档索引对象
        index_attempt_metadata: 索引尝试的元数据
        db_session: 数据库会话对象
        ignore_time_skip: 是否忽略时间跳过检查
        tenant_id: 租户ID
        filter_fnc: 文档过滤函数
        
    返回：
        tuple[int, int]: 返回(新文档数量, 块数量)的元组
    """
    no_access = DocumentAccess.build(
        user_emails=[],
        user_groups=[],
        external_user_emails=[],
        external_user_group_ids=[],
        is_public=False,
    )

    logger.debug("Filtering Documents")
    filtered_documents = filter_fnc(document_batch)

    ctx = index_doc_batch_prepare(
        documents=filtered_documents,
        index_attempt_metadata=index_attempt_metadata,
        ignore_time_skip=ignore_time_skip,
        db_session=db_session,
    )
    if not ctx:
        return 0, 0

    logger.debug("Starting chunking")
    chunks: list[DocAwareChunk] = chunker.chunk(ctx.updatable_docs)

    logger.debug("Starting embedding")
    chunks_with_embeddings = embedder.embed_chunks(chunks) if chunks else []

    updatable_ids = [doc.id for doc in ctx.updatable_docs]

    # Acquires a lock on the documents so that no other process can modify them
    # 获取文档锁，确保没有其他进程可以修改它们
    # NOTE: don't need to acquire till here, since this is when the actual race condition
    # with Vespa can occur.
    # 注意：直到这里才需要获取锁，因为这是与Vespa可能发生竞争条件的时刻
    with prepare_to_modify_documents(db_session=db_session, document_ids=updatable_ids):
        doc_id_to_access_info = get_access_for_documents(
            document_ids=updatable_ids, db_session=db_session
        )
        doc_id_to_document_set = {
            document_id: document_sets
            for document_id, document_sets in fetch_document_sets_for_documents(
                document_ids=updatable_ids, db_session=db_session
            )
        }

        doc_id_to_previous_chunk_cnt: dict[str, int | None] = {
            document_id: chunk_count
            for document_id, chunk_count in fetch_chunk_counts_for_documents(
                document_ids=updatable_ids,
                db_session=db_session,
            )
        }

        doc_id_to_new_chunk_cnt: dict[str, int] = {
            document_id: len(
                [
                    chunk
                    for chunk in chunks_with_embeddings
                    if chunk.source_document.id == document_id
                ]
            )
            for document_id in updatable_ids
        }

        # we're concerned about race conditions where multiple simultaneous indexings might result
        # in one set of metadata overwriting another one in vespa.
        # 我们担心多个同时索引可能导致一组元数据在vespa中覆盖另一组元数据的竞争条件。
        # we still write data here for the immediate and most likely correct sync, but
        # to resolve this, an update of the last modified field at the end of this loop
        # always triggers a final metadata sync via the celery queue
        # 我们仍然在这里写入数据以进行即时且可能正确的同步，但为了解决这个问题，在循环结束时更新最后修改字段总是通过celery队列触发最终的元数据同步
        access_aware_chunks = [
            DocMetadataAwareIndexChunk.from_index_chunk(
                index_chunk=chunk,
                access=doc_id_to_access_info.get(chunk.source_document.id, no_access),
                document_sets=set(
                    doc_id_to_document_set.get(chunk.source_document.id, [])
                ),
                boost=(
                    ctx.id_to_db_doc_map[chunk.source_document.id].boost
                    if chunk.source_document.id in ctx.id_to_db_doc_map
                    else DEFAULT_BOOST
                ),
                tenant_id=tenant_id,
            )
            for chunk in chunks_with_embeddings
        ]

        logger.debug(
            f"Indexing the following chunks: {[chunk.to_short_descriptor() for chunk in access_aware_chunks]}"
        )
        # A document will not be spread across different batches, so all the
        # documents with chunks in this set, are fully represented by the chunks
        # in this set
        # 一个文档不会分散在不同的批次中，因此该集合中具有块的所有文档都由该集合中的块完全表示
        insertion_records = document_index.index(
            chunks=access_aware_chunks,
            index_batch_params=IndexBatchParams(
                doc_id_to_previous_chunk_cnt=doc_id_to_previous_chunk_cnt,
                doc_id_to_new_chunk_cnt=doc_id_to_new_chunk_cnt,
                tenant_id=tenant_id,
                large_chunks_enabled=chunker.enable_large_chunks,
            ),
        )

        successful_doc_ids = [record.document_id for record in insertion_records]
        successful_docs = [
            doc for doc in ctx.updatable_docs if doc.id in successful_doc_ids
        ]

        last_modified_ids = []
        ids_to_new_updated_at = {}
        for doc in successful_docs:
            last_modified_ids.append(doc.id)
            # doc_updated_at is the source's idea (on the other end of the connector)
            # of when the doc was last modified
            # doc_updated_at 是源端（连接器的另一端）认为文档最后修改的时间
            if doc.doc_updated_at is None:
                continue
            ids_to_new_updated_at[doc.id] = doc.doc_updated_at

        update_docs_updated_at__no_commit(
            ids_to_new_updated_at=ids_to_new_updated_at, db_session=db_session
        )

        update_docs_last_modified__no_commit(
            document_ids=last_modified_ids, db_session=db_session
        )

        update_docs_chunk_count__no_commit(
            document_ids=updatable_ids,
            doc_id_to_chunk_count=doc_id_to_new_chunk_cnt,
            db_session=db_session,
        )

        db_session.commit()

    result = (
        len([r for r in insertion_records if r.already_existed is False]),
        len(access_aware_chunks),
    )

    return result


def check_enable_large_chunks_and_multipass(
    embedder: IndexingEmbedder, db_session: Session
) -> tuple[bool, bool]:
    """
    检查是否启用大块和多遍索引
    
    参数：
        embedder: 嵌入向量生成器
        db_session: 数据库会话对象
        
    返回：
        tuple[bool, bool]: 返回(是否启用多遍索引, 是否启用大块)的元组
    """
    search_settings = get_current_search_settings(db_session)
    multipass = (
        search_settings.multipass_indexing
        if search_settings
        else ENABLE_MULTIPASS_INDEXING
    )

    enable_large_chunks = (
        multipass
        and
        # Only local models that supports larger context are from Nomic
        (embedder.model_name.startswith("nomic-ai"))
        and
        # Cohere does not support larger context they recommend not going above 512 tokens
        embedder.provider_type != EmbeddingProvider.COHERE
    )
    return multipass, enable_large_chunks


def build_indexing_pipeline(
    *,
    embedder: IndexingEmbedder,
    document_index: DocumentIndex,
    db_session: Session,
    chunker: Chunker | None = None,
    ignore_time_skip: bool = False,
    attempt_id: int | None = None,
    tenant_id: str | None = None,
    callback: IndexingHeartbeatInterface | None = None,
) -> IndexingPipelineProtocol:
    """
    构建索引流水线，处理文档批次并进行索引
    
    参数：
        embedder: 嵌入向量生成器
        document_index: 文档索引对象
        db_session: 数据库会话对象
        chunker: 文档分块器
        ignore_time_skip: 是否忽略时间跳过检查
        attempt_id: 尝试ID
        tenant_id: 租户ID
        callback: 索引心跳接口
        
    返回：
        IndexingPipelineProtocol: 返回索引流水线协议对象
    """
    multipass, enable_large_chunks = check_enable_large_chunks_and_multipass(
        embedder, db_session
    )

    chunker = chunker or Chunker(
        tokenizer=embedder.embedding_model.tokenizer,
        enable_multipass=multipass,
        enable_large_chunks=enable_large_chunks,
        # after every doc, update status in case there are a bunch of really long docs
        # 在处理每个文档后更新状态，以防存在一批非常长的文档
        callback=callback,
    )

    return partial(
        index_doc_batch_with_handler,
        chunker=chunker,
        embedder=embedder,
        document_index=document_index,
        ignore_time_skip=ignore_time_skip,
        attempt_id=attempt_id,
        db_session=db_session,
        tenant_id=tenant_id,
    )
