"""
文件功能说明：
此文件包含了文档索引相关的工具函数，主要用于处理文档块的索引操作、UUID生成、
以及文档提升分数的计算等功能。主要服务于文档检索系统的索引管理部分。
"""

import math
import uuid
from uuid import UUID

from sqlalchemy.orm import Session

from onyx.db.search_settings import get_current_search_settings
from onyx.db.search_settings import get_secondary_search_settings
from onyx.document_index.interfaces import EnrichedDocumentIndexingInfo
from onyx.indexing.models import DocMetadataAwareIndexChunk


DEFAULT_BATCH_SIZE = 30
DEFAULT_INDEX_NAME = "danswer_chunk"


def get_both_index_names(db_session: Session) -> tuple[str, str | None]:
    """
    获取当前和次要搜索索引的名称。
    
    参数:
        db_session: 数据库会话对象
    
    返回:
        tuple: 包含主索引名称和次要索引名称（如果存在）的元组
    """
    search_settings = get_current_search_settings(db_session)

    search_settings_new = get_secondary_search_settings(db_session)
    if not search_settings_new:
        return search_settings.index_name, None

    return search_settings.index_name, search_settings_new.index_name


def translate_boost_count_to_multiplier(boost: int) -> float:
    """
    将boost整数值映射为乘数，使用S形曲线计算。分段处理，使得大量下投时得分为0.5倍，
    大量上投时得分为2倍。这与Vespa计算保持一致。

    参数:
        boost: 提升值（可正可负）
    
    返回:
        float: 计算得到的乘数值
    """
    if boost < 0:
        # 0.5 + sigmoid -> 范围为0.5到1
        return 0.5 + (1 / (1 + math.exp(-1 * boost / 3)))

    # 2 x sigmoid -> 范围为1到2
    return 2 / (1 + math.exp(-1 * boost / 3))


def assemble_document_chunk_info(
    enriched_document_info_list: list[EnrichedDocumentIndexingInfo],
    tenant_id: str | None,
    large_chunks_enabled: bool,
) -> list[UUID]:
    """
    组装文档块信息，生成对应的UUID列表。
    
    参数:
        enriched_document_info_list: 包含文档索引信息的列表
        tenant_id: 租户ID
        large_chunks_enabled: 是否启用大块处理
    
    返回:
        list[UUID]: 文档块UUID列表
    """
    doc_chunk_ids = []

    for enriched_document_info in enriched_document_info_list:
        for chunk_index in range(
            enriched_document_info.chunk_start_index,
            enriched_document_info.chunk_end_index,
        ):
            if not enriched_document_info.old_version:
                doc_chunk_ids.append(
                    get_uuid_from_chunk_info(
                        document_id=enriched_document_info.doc_id,
                        chunk_id=chunk_index,
                        tenant_id=tenant_id,
                    )
                )
            else:
                doc_chunk_ids.append(
                    get_uuid_from_chunk_info_old(
                        document_id=enriched_document_info.doc_id,
                        chunk_id=chunk_index,
                    )
                )

            if large_chunks_enabled and chunk_index % 4 == 0:
                large_chunk_id = int(chunk_index / 4)
                large_chunk_reference_ids = [
                    large_chunk_id + i
                    for i in range(4)
                    if large_chunk_id + i < enriched_document_info.chunk_end_index
                ]
                if enriched_document_info.old_version:
                    doc_chunk_ids.append(
                        get_uuid_from_chunk_info_old(
                            document_id=enriched_document_info.doc_id,
                            chunk_id=large_chunk_id,
                            large_chunk_reference_ids=large_chunk_reference_ids,
                        )
                    )
                else:
                    doc_chunk_ids.append(
                        get_uuid_from_chunk_info(
                            document_id=enriched_document_info.doc_id,
                            chunk_id=large_chunk_id,
                            tenant_id=tenant_id,
                            large_chunk_id=large_chunk_id,
                        )
                    )

    return doc_chunk_ids


def get_uuid_from_chunk_info(
    *,
    document_id: str,
    chunk_id: int,
    tenant_id: str | None,
    large_chunk_id: int | None = None,
) -> UUID:
    """
    根据文档块信息生成UUID。
    
    参数:
        document_id: 文档ID
        chunk_id: 块ID
        tenant_id: 租户ID
        large_chunk_id: 大块ID（可选）
    
    返回:
        UUID: 生成的UUID
    """
    doc_str = document_id

    # Web parsing URL duplicate catching
    if doc_str and doc_str[-1] == "/":
        doc_str = doc_str[:-1]

    chunk_index = (
        "large_" + str(large_chunk_id) if large_chunk_id is not None else str(chunk_id)
    )
    unique_identifier_string = "_".join([doc_str, chunk_index])
    if tenant_id:
        unique_identifier_string += "_" + tenant_id

    return uuid.uuid5(uuid.NAMESPACE_X500, unique_identifier_string)


def get_uuid_from_chunk_info_old(
    *, document_id: str, chunk_id: int, large_chunk_reference_ids: list[int] = []
) -> UUID:
    """
    使用旧方式根据文档块信息生成UUID。
    
    参数:
        document_id: 文档ID
        chunk_id: 块ID
        large_chunk_reference_ids: 大块引用ID列表
    
    返回:
        UUID: 生成的UUID
    """
    doc_str = document_id

    # Web parsing URL duplicate catching
    if doc_str and doc_str[-1] == "/":
        doc_str = doc_str[:-1]
    unique_identifier_string = "_".join([doc_str, str(chunk_id), "0"])
    if large_chunk_reference_ids:
        unique_identifier_string += "_large" + "_".join(
            [
                str(referenced_chunk_id)
                for referenced_chunk_id in large_chunk_reference_ids
            ]
        )
    return uuid.uuid5(uuid.NAMESPACE_X500, unique_identifier_string)


def get_uuid_from_chunk(chunk: DocMetadataAwareIndexChunk) -> uuid.UUID:
    """
    从文档块对象生成UUID。
    
    参数:
        chunk: 文档块对象
    
    返回:
        UUID: 生成的UUID
    """
    return get_uuid_from_chunk_info(
        document_id=chunk.source_document.id,
        chunk_id=chunk.chunk_id,
        tenant_id=chunk.tenant_id,
        large_chunk_id=chunk.large_chunk_id,
    )


def get_uuid_from_chunk_old(
    chunk: DocMetadataAwareIndexChunk, large_chunk_reference_ids: list[int] = []
) -> UUID:
    """
    使用旧方式从文档块对象生成UUID。
    
    参数:
        chunk: 文档块对象
        large_chunk_reference_ids: 大块引用ID列表
    
    返回:
        UUID: 生成的UUID
    """
    return get_uuid_from_chunk_info_old(
        document_id=chunk.source_document.id,
        chunk_id=chunk.chunk_id,
        large_chunk_reference_ids=large_chunk_reference_ids,
    )
