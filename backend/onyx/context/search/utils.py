"""
此文件包含了搜索上下文相关的工具函数。
主要功能：
1. 文档去重
2. 相关性段落处理
3. 文档转换工具
4. 搜索结果处理
"""

from collections.abc import Sequence
from typing import TypeVar

from onyx.chat.models import SectionRelevancePiece
from onyx.context.search.models import InferenceChunk
from onyx.context.search.models import InferenceSection
from onyx.context.search.models import SavedSearchDoc
from onyx.context.search.models import SavedSearchDocWithContent
from onyx.context.search.models import SearchDoc
from onyx.db.models import SearchDoc as DBSearchDoc


# 定义用于不同搜索文档类型的泛型类型变量
T = TypeVar(
    "T",
    InferenceSection,
    InferenceChunk,
    SearchDoc,
    SavedSearchDoc,
    SavedSearchDocWithContent,
)

TSection = TypeVar(
    "TSection",
    InferenceSection,
    SearchDoc,
    SavedSearchDoc,
    SavedSearchDocWithContent,
)


def dedupe_documents(items: list[T]) -> tuple[list[T], list[int]]:
    """
    对文档列表进行去重处理。
    
    参数:
        items: 需要去重的文档列表
        
    返回:
        tuple: 包含去重后的文档列表和被删除项的索引列表
    """
    seen_ids = set()
    deduped_items = []
    dropped_indices = []
    for index, item in enumerate(items):
        if isinstance(item, InferenceSection):
            document_id = item.center_chunk.document_id
        else:
            document_id = item.document_id

        if document_id not in seen_ids:
            seen_ids.add(document_id)
            deduped_items.append(item)
        else:
            dropped_indices.append(index)
    return deduped_items, dropped_indices


def relevant_sections_to_indices(
    relevance_sections: list[SectionRelevancePiece] | None, items: list[TSection]
) -> list[int]:
    """
    将相关段落转换为索引列表。
    
    参数:
        relevance_sections: 相关性段落列表
        items: 文档段落列表
        
    返回:
        list[int]: 相关段落在原列表中的索引列表
    """
    if not relevance_sections:
        return []

    relevant_set = {
        (chunk.document_id, chunk.chunk_id)
        for chunk in relevance_sections
        if chunk.relevant
    }

    return [
        index
        for index, item in enumerate(items)
        if (
            (
                isinstance(item, InferenceSection)
                and (item.center_chunk.document_id, item.center_chunk.chunk_id)
                in relevant_set
            )
            or (
                not isinstance(item, (InferenceSection))
                and (item.document_id, item.chunk_ind) in relevant_set
            )
        )
    ]


def drop_llm_indices(
    llm_indices: list[int],
    search_docs: Sequence[DBSearchDoc | SavedSearchDoc],
    dropped_indices: list[int],
) -> list[int]:
    """
    根据删除的索引更新LLM索引列表。
    
    参数:
        llm_indices: LLM模型生成的索引列表
        search_docs: 搜索文档列表
        dropped_indices: 需要删除的索引列表
        
    返回:
        list[int]: 更新后的LLM索引列表
    """
    llm_bools = [True if i in llm_indices else False for i in range(len(search_docs))]
    if dropped_indices:
        llm_bools = [
            val for ind, val in enumerate(llm_bools) if ind not in dropped_indices
        ]
    return [i for i, val in enumerate(llm_bools) if val]


def inference_section_from_chunks(
    center_chunk: InferenceChunk,
    chunks: list[InferenceChunk],
) -> InferenceSection | None:
    """
    将多个文档块组合成一个推理段落。
    
    参数:
        center_chunk: 中心文档块
        chunks: 相关文档块列表
        
    返回:
        InferenceSection | None: 组合后的推理段落，如果chunks为空则返回None
    """
    if not chunks:
        return None

    combined_content = "\n".join([chunk.content for chunk in chunks])

    return InferenceSection(
        center_chunk=center_chunk,
        chunks=chunks,
        combined_content=combined_content,
    )


def chunks_or_sections_to_search_docs(
    items: Sequence[InferenceChunk | InferenceSection] | None,
) -> list[SearchDoc]:
    """
    将文档块或段落转换为搜索文档格式。
    
    参数:
        items: 文档块或段落列表
        
    返回:
        list[SearchDoc]: 转换后的搜索文档列表
    """
    if not items:
        return []

    search_docs = [
        SearchDoc(
            document_id=(
                chunk := item.center_chunk
                if isinstance(item, InferenceSection)
                else item
            ).document_id,
            chunk_ind=chunk.chunk_id,
            semantic_identifier=chunk.semantic_identifier or "Unknown",
            link=chunk.source_links[0] if chunk.source_links else None,
            blurb=chunk.blurb,
            source_type=chunk.source_type,
            boost=chunk.boost,
            hidden=chunk.hidden,
            metadata=chunk.metadata,
            score=chunk.score,
            match_highlights=chunk.match_highlights,
            updated_at=chunk.updated_at,
            primary_owners=chunk.primary_owners,
            secondary_owners=chunk.secondary_owners,
            is_internet=False,
        )
        for item in items
    ]

    return search_docs
