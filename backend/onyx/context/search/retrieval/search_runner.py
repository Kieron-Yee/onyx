"""
这个文件实现了文档搜索的核心功能，包括:
1. 文档检索的主要逻辑实现
2. 查询处理和优化
3. 搜索结果的后处理
4. NLTK相关文本处理功能
"""

import string
from collections.abc import Callable

import nltk  # type:ignore
from nltk.corpus import stopwords  # type:ignore
from nltk.tokenize import word_tokenize  # type:ignore
from sqlalchemy.orm import Session

from onyx.context.search.models import ChunkMetric
from onyx.context.search.models import IndexFilters
from onyx.context.search.models import InferenceChunk
from onyx.context.search.models import InferenceChunkUncleaned
from onyx.context.search.models import InferenceSection
from onyx.context.search.models import MAX_METRICS_CONTENT
from onyx.context.search.models import RetrievalMetricsContainer
from onyx.context.search.models import SearchQuery
from onyx.context.search.postprocessing.postprocessing import cleanup_chunks
from onyx.context.search.utils import inference_section_from_chunks
from onyx.db.search_settings import get_current_search_settings
from onyx.db.search_settings import get_multilingual_expansion
from onyx.document_index.interfaces import DocumentIndex
from onyx.document_index.interfaces import VespaChunkRequest
from onyx.document_index.vespa.shared_utils.utils import (
    replace_invalid_doc_id_characters,
)
from onyx.natural_language_processing.search_nlp_models import EmbeddingModel
from onyx.secondary_llm_flows.query_expansion import multilingual_query_expansion
from onyx.utils.logger import setup_logger
from onyx.utils.threadpool_concurrency import run_functions_tuples_in_parallel
from onyx.utils.timing import log_function_time
from shared_configs.configs import MODEL_SERVER_HOST
from shared_configs.configs import MODEL_SERVER_PORT
from shared_configs.enums import EmbedTextType


logger = setup_logger()


def download_nltk_data() -> None:
    """
    下载和检查NLTK所需的数据资源
    包括停用词、分词器等必要组件
    如果资源已存在则跳过下载
    """
    resources = {
        "stopwords": "corpora/stopwords",
        # "wordnet": "corpora/wordnet",  # Not in use
        "punkt": "tokenizers/punkt",
    }

    for resource_name, resource_path in resources.items():
        try:
            nltk.data.find(resource_path)
            logger.info(f"{resource_name} is already downloaded.")
        except LookupError:
            try:
                logger.info(f"Downloading {resource_name}...")
                nltk.download(resource_name, quiet=True)
                logger.info(f"{resource_name} downloaded successfully.")
            except Exception as e:
                logger.error(f"Failed to download {resource_name}. Error: {e}")


def lemmatize_text(keywords: list[str]) -> list[str]:
    """
    对关键词列表进行词形还原处理
    注：当前未启用此功能
    
    Args:
        keywords: 需要处理的关键词列表
    Returns:
        处理后的关键词列表
    """
    raise NotImplementedError("Lemmatization should not be used currently")
    # try:
    #     query = " ".join(keywords)
    #     lemmatizer = WordNetLemmatizer()
    #     word_tokens = word_tokenize(query)
    #     lemmatized_words = [lemmatizer.lemmatize(word) for word in word_tokens]
    #     combined_keywords = list(set(keywords + lemmatized_words))
    #     return combined_keywords
    # except Exception:
    #     return keywords


def remove_stop_words_and_punctuation(keywords: list[str]) -> list[str]:
    """
    移除停用词和标点符号
    
    Args:
        keywords: 输入的关键词列表
    Returns:
        处理后的关键词列表，如果处理失败则返回原列表
    """
    try:
        # Re-tokenize using the NLTK tokenizer for better matching
        # 使用NLTK分词器重新分词以获得更好的匹配效果
        query = " ".join(keywords)
        stop_words = set(stopwords.words("english"))
        word_tokens = word_tokenize(query)
        text_trimmed = [
            word
            for word in word_tokens
            if (word.casefold() not in stop_words and word not in string.punctuation)
        ]
        return text_trimmed or word_tokens
    except Exception:
        return keywords


def combine_retrieval_results(
    chunk_sets: list[list[InferenceChunk]],
) -> list[InferenceChunk]:
    """
    合并多个检索结果集
    
    Args:
        chunk_sets: 多个检索结果集的列表
    Returns:
        合并并按分数排序后的去重结果列表
    """
    all_chunks = [chunk for chunk_set in chunk_sets for chunk in chunk_set]

    unique_chunks: dict[tuple[str, int], InferenceChunk] = {}
    for chunk in all_chunks:
        key = (chunk.document_id, chunk.chunk_id)
        if key not in unique_chunks:
            unique_chunks[key] = chunk
            continue

        stored_chunk_score = unique_chunks[key].score or 0
        this_chunk_score = chunk.score or 0
        if stored_chunk_score < this_chunk_score:
            unique_chunks[key] = chunk

    sorted_chunks = sorted(
        unique_chunks.values(), key=lambda x: x.score or 0, reverse=True
    )

    return sorted_chunks


@log_function_time(print_only=True)
def doc_index_retrieval(
    query: SearchQuery,
    document_index: DocumentIndex,
    db_session: Session,
) -> list[InferenceChunk]:
    """
    执行文档检索的核心函数
    
    功能包括:
    1. 执行搜索获取文档块
    2. 从大块文本中提取小块
    3. 保存分数信息
    4. 去重和清理处理
    
    Args:
        query: 搜索查询对象
        document_index: 文档索引接口
        db_session: 数据库会话
    Returns:
        处理后的检索结果列表
    """
    search_settings = get_current_search_settings(db_session)

    model = EmbeddingModel.from_db_model(
        search_settings=search_settings,
        # The below are globally set, this flow always uses the indexing one
        server_host=MODEL_SERVER_HOST,
        server_port=MODEL_SERVER_PORT,
    )

    query_embedding = model.encode([query.query], text_type=EmbedTextType.QUERY)[0]

    top_chunks = document_index.hybrid_retrieval(
        query=query.query,
        query_embedding=query_embedding,
        final_keywords=query.processed_keywords,
        filters=query.filters,
        hybrid_alpha=query.hybrid_alpha,
        time_decay_multiplier=query.recency_bias_multiplier,
        num_to_retrieve=query.num_hits,
        offset=query.offset,
    )

    retrieval_requests: list[VespaChunkRequest] = []
    normal_chunks: list[InferenceChunkUncleaned] = []
    referenced_chunk_scores: dict[tuple[str, int], float] = {}
    for chunk in top_chunks:
        if chunk.large_chunk_reference_ids:
            retrieval_requests.append(
                VespaChunkRequest(
                    document_id=replace_invalid_doc_id_characters(chunk.document_id),
                    min_chunk_ind=chunk.large_chunk_reference_ids[0],
                    max_chunk_ind=chunk.large_chunk_reference_ids[-1],
                )
            )
            # for each referenced chunk, persist the
            # highest score to the referenced chunk
            for chunk_id in chunk.large_chunk_reference_ids:
                key = (chunk.document_id, chunk_id)
                referenced_chunk_scores[key] = max(
                    referenced_chunk_scores.get(key, 0), chunk.score or 0
                )
        else:
            normal_chunks.append(chunk)

    # If there are no large chunks, just return the normal chunks
    # 如果没有大文本块，直接返回普通文本块
    if not retrieval_requests:
        return cleanup_chunks(normal_chunks)

    # Retrieve and return the referenced normal chunks from the large chunks
    # 从大文本块中检索并返回引用的普通文本块
    retrieved_inference_chunks = document_index.id_based_retrieval(
        chunk_requests=retrieval_requests,
        filters=query.filters,
        batch_retrieval=True,
    )

    # Apply the scores from the large chunks to the chunks referenced
    # by each large chunk
    # 将大文本块的分数应用到每个大文本块引用的小文本块上
    for chunk in retrieved_inference_chunks:
        if (chunk.document_id, chunk.chunk_id) in referenced_chunk_scores:
            chunk.score = referenced_chunk_scores[(chunk.document_id, chunk.chunk_id)]
            referenced_chunk_scores.pop((chunk.document_id, chunk.chunk_id))
        else:
            logger.error(
                f"Chunk {chunk.document_id} {chunk.chunk_id} not found in referenced chunk scores"
            )

    # Log any chunks that were not found in the retrieved chunks
    # 记录在检索结果中未找到的文本块
    for reference in referenced_chunk_scores.keys():
        logger.error(f"Chunk {reference} not found in retrieved chunks")

    unique_chunks: dict[tuple[str, int], InferenceChunkUncleaned] = {
        (chunk.document_id, chunk.chunk_id): chunk for chunk in normal_chunks
    }

    # persist the highest score of each deduped chunk
    # 保存每个去重文本块的最高分数
    for chunk in retrieved_inference_chunks:
        key = (chunk.document_id, chunk.chunk_id)
        # For duplicates, keep the highest score
        # 对于重复项，保留最高分数
        if key not in unique_chunks or (chunk.score or 0) > (
            unique_chunks[key].score or 0
        ):
            unique_chunks[key] = chunk

    # Deduplicate the chunks
    # 对文本块进行去重
    deduped_chunks = list(unique_chunks.values())
    deduped_chunks.sort(key=lambda chunk: chunk.score or 0, reverse=True)
    return cleanup_chunks(deduped_chunks)


def _simplify_text(text: str) -> str:
    """
    简化文本处理：移除标点符号和空白字符，并转换为小写
    
    Args:
        text: 输入文本
    Returns:
        处理后的文本
    """
    return "".join(
        char for char in text if char not in string.punctuation and not char.isspace()
    ).lower()


def retrieve_chunks(
    query: SearchQuery,
    document_index: DocumentIndex,
    db_session: Session,
    retrieval_metrics_callback: Callable[[RetrievalMetricsContainer], None]
    | None = None,
) -> list[InferenceChunk]:
    """
    执行文档块检索并返回最佳匹配结果
    
    支持多语言查询扩展，提供混合搜索功能
    
    Args:
        query: 搜索查询对象
        document_index: 文档索引接口
        db_session: 数据库会话
        retrieval_metrics_callback: 可选的检索指标回调函数
    Returns:
        检索到的最佳匹配文档块列表
    """
    multilingual_expansion = get_multilingual_expansion(db_session)
    # Don't do query expansion on complex queries, rephrasings likely would not work well
    # 对于复杂查询不进行查询扩展，因为重新措辞可能效果不好
    if not multilingual_expansion or "\n" in query.query or "\r" in query.query:
        top_chunks = doc_index_retrieval(
            query=query, document_index=document_index, db_session=db_session
        )
    else:
        simplified_queries = set()
        run_queries: list[tuple[Callable, tuple]] = []

        # Currently only uses query expansion on multilingual use cases
        # 目前仅在多语言场景下使用查询扩展
        query_rephrases = multilingual_query_expansion(
            query.query, multilingual_expansion
        )
        # Just to be extra sure, add the original query.
        # 为了以防万一，添加原始查询
        query_rephrases.append(query.query)
        for rephrase in set(query_rephrases):
            # Sometimes the model rephrases the query in the same language with minor changes
            # Avoid doing an extra search with the minor changes as this biases the results
            # 有时模型会用相同语言做细微的重述
            # 避免对这些细微变化进行额外搜索，因为这会导致结果偏差
            simplified_rephrase = _simplify_text(rephrase)
            if simplified_rephrase in simplified_queries:
                continue
            simplified_queries.add(simplified_rephrase)

            q_copy = query.copy(update={"query": rephrase}, deep=True)
            run_queries.append(
                (
                    doc_index_retrieval,
                    (q_copy, document_index, db_session),
                )
            )
        parallel_search_results = run_functions_tuples_in_parallel(run_queries)
        top_chunks = combine_retrieval_results(parallel_search_results)

    if not top_chunks:
        logger.warning(
            f"Hybrid ({query.search_type.value.capitalize()}) search returned no results "
            f"with filters: {query.filters}"
        )
        return []

    if retrieval_metrics_callback is not None:
        chunk_metrics = [
            ChunkMetric(
                document_id=chunk.document_id,
                chunk_content_start=chunk.content[:MAX_METRICS_CONTENT],
                first_link=chunk.source_links[0] if chunk.source_links else None,
                score=chunk.score if chunk.score is not None else 0,
            )
            for chunk in top_chunks
        ]
        retrieval_metrics_callback(
            RetrievalMetricsContainer(
                search_type=query.search_type, metrics=chunk_metrics
            )
        )

    return top_chunks


def inference_sections_from_ids(
    doc_identifiers: list[tuple[str, int]],
    document_index: DocumentIndex,
) -> list[InferenceSection]:
    """
    根据文档ID列表获取推理章节
    
    Args:
        doc_identifiers: 文档ID和块ID的元组列表
        document_index: 文档索引接口
    Returns:
        推理章节列表
    """
    # Currently only fetches whole docs
    # 当前仅获取完整文档
    doc_ids_set = set(doc_id for doc_id, _ in doc_identifiers)

    chunk_requests: list[VespaChunkRequest] = [
        VespaChunkRequest(document_id=doc_id) for doc_id in doc_ids_set
    ]

    # No need for ACL here because the doc ids were validated beforehand
    # 这里不需要ACL因为文档ID已经预先验证过了
    filters = IndexFilters(access_control_list=None)

    retrieved_chunks = document_index.id_based_retrieval(
        chunk_requests=chunk_requests,
        filters=filters,
    )

    cleaned_chunks = cleanup_chunks(retrieved_chunks)
    if not cleaned_chunks:
        return []

    # Group chunks by document ID
    # 按文档ID对文本块分组
    chunks_by_doc_id: dict[str, list[InferenceChunk]] = {}
    for chunk in cleaned_chunks:
        chunks_by_doc_id.setdefault(chunk.document_id, []).append(chunk)

    inference_sections = [
        section
        for chunks in chunks_by_doc_id.values()
        if chunks
        and (
            section := inference_section_from_chunks(
                # The scores will always be 0 because the fetching by id gives back
                # no search scores. This is not needed though if the user is explicitly
                # selecting a document.
                # 分数将始终为0，因为通过ID获取不会返回搜索分数。
                # 不过如果用户明确选择了文档，这个分数就不需要了。
                center_chunk=chunks[0],
                chunks=chunks,
            )
        )
    ]

    return inference_sections
