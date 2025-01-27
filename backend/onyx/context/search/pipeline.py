"""
搜索管道模块

本模块实现了一个完整的搜索流水线，用于处理文档检索、重排序和相关性评估等任务。
主要功能包括:
1. 搜索请求的预处理
2. 文档块的检索
3. 文档片段的扩展和合并
4. 使用LLM进行重排序和相关性评估
5. 结果的后处理和整合
"""

from collections import defaultdict
from collections.abc import Callable
from collections.abc import Iterator
from typing import cast

from sqlalchemy.orm import Session

from onyx.chat.models import PromptConfig
from onyx.chat.models import SectionRelevancePiece
from onyx.chat.prune_and_merge import _merge_sections
from onyx.chat.prune_and_merge import ChunkRange
from onyx.chat.prune_and_merge import merge_chunk_intervals
from onyx.configs.chat_configs import DISABLE_LLM_DOC_RELEVANCE
from onyx.context.search.enums import LLMEvaluationType
from onyx.context.search.enums import QueryFlow
from onyx.context.search.enums import SearchType
from onyx.context.search.models import IndexFilters
from onyx.context.search.models import InferenceChunk
from onyx.context.search.models import InferenceSection
from onyx.context.search.models import RerankMetricsContainer
from onyx.context.search.models import RetrievalMetricsContainer
from onyx.context.search.models import SearchQuery
from onyx.context.search.models import SearchRequest
from onyx.context.search.postprocessing.postprocessing import cleanup_chunks
from onyx.context.search.postprocessing.postprocessing import search_postprocessing
from onyx.context.search.preprocessing.preprocessing import retrieval_preprocessing
from onyx.context.search.retrieval.search_runner import retrieve_chunks
from onyx.context.search.utils import inference_section_from_chunks
from onyx.context.search.utils import relevant_sections_to_indices
from onyx.db.models import User
from onyx.db.search_settings import get_current_search_settings
from onyx.document_index.factory import get_default_document_index
from onyx.document_index.interfaces import VespaChunkRequest
from onyx.llm.interfaces import LLM
from onyx.secondary_llm_flows.agentic_evaluation import evaluate_inference_section
from onyx.utils.logger import setup_logger
from onyx.utils.threadpool_concurrency import FunctionCall
from onyx.utils.threadpool_concurrency import run_functions_in_parallel
from onyx.utils.timing import log_function_time

logger = setup_logger()


class SearchPipeline:
    """
    搜索管道类
    
    实现了完整的搜索流程，包括预处理、检索、重排序和后处理等步骤。
    
    属性:
        search_request (SearchRequest): 搜索请求对象
        user (User | None): 用户对象
        llm (LLM): 主要的大语言模型实例
        fast_llm (LLM): 快速处理用的大语言模型实例
        db_session (Session): 数据库会话
        bypass_acl (bool): 是否绕过访问控制检查
        retrieval_metrics_callback (Callable): 检索指标回调函数
        rerank_metrics_callback (Callable): 重排序指标回调函数
        prompt_config (PromptConfig): 提示配置
    """
    def __init__(
        self,
        search_request: SearchRequest,
        user: User | None,
        llm: LLM,
        fast_llm: LLM,
        db_session: Session,
        bypass_acl: bool = False,  # NOTE: VERY DANGEROUS, USE WITH CAUTION
        # 注意：非常危险，请谨慎使用
        retrieval_metrics_callback: (
            Callable[[RetrievalMetricsContainer], None] | None
        ) = None,
        rerank_metrics_callback: Callable[[RerankMetricsContainer], None] | None = None,
        prompt_config: PromptConfig | None = None,
    ):
        """
        初始化搜索管道
        
        参数:
            search_request: 搜索请求对象
            user: 用户对象
            llm: 主LLM模型
            fast_llm: 快速LLM模型
            db_session: 数据库会话
            bypass_acl: 是否绕过访问控制
            retrieval_metrics_callback: 检索指标回调
            rerank_metrics_callback: 重排序回调
            prompt_config: 提示词配置
        """
        self.search_request = search_request
        self.user = user
        self.llm = llm
        self.fast_llm = fast_llm
        self.db_session = db_session
        self.bypass_acl = bypass_acl
        self.retrieval_metrics_callback = retrieval_metrics_callback
        self.rerank_metrics_callback = rerank_metrics_callback

        self.search_settings = get_current_search_settings(db_session)
        self.document_index = get_default_document_index(
            primary_index_name=self.search_settings.index_name,
            secondary_index_name=None,
        )
        self.prompt_config: PromptConfig | None = prompt_config

        # Preprocessing steps generate this
        # 预处理步骤生成这些数据
        self._search_query: SearchQuery | None = None
        self._predicted_search_type: SearchType | None = None

        # Initial document index retrieval chunks
        # 初始文档索引检索的文档块
        self._retrieved_chunks: list[InferenceChunk] | None = None
        # Another call made to the document index to get surrounding sections
        # 另一次调用文档索引以获取周围的片段
        self._retrieved_sections: list[InferenceSection] | None = None
        # Reranking and LLM section selection can be run together
        # If only LLM selection is on, the reranked chunks are yielded immediatly
        # 重排序和LLM片段选择可以一起运行
        # 如果只启用了LLM选择，重排序的文档块会立即生成
        self._reranked_sections: list[InferenceSection] | None = None
        self._final_context_sections: list[InferenceSection] | None = None

        self._section_relevance: list[SectionRelevancePiece] | None = None

        # Generates reranked chunks and LLM selections
        # 生成重排序的文档块和LLM选择结果
        self._postprocessing_generator: (
            Iterator[list[InferenceSection] | list[SectionRelevancePiece]] | None
        ) = None

        # No longer computed but keeping around in case it's reintroduced later
        # 不再计算但保留以备后续可能重新引入
        self._predicted_flow: QueryFlow | None = QueryFlow.QUESTION_ANSWER

    """Pre-processing"""

    def _run_preprocessing(self) -> None:
        """
        运行预处理步骤
        
        处理搜索请求，设置搜索类型和查询信息
        """
        final_search_query = retrieval_preprocessing(
            search_request=self.search_request,
            user=self.user,
            llm=self.llm,
            db_session=self.db_session,
            bypass_acl=self.bypass_acl,
        )
        self._search_query = final_search_query
        self._predicted_search_type = final_search_query.search_type

    @property
    def search_query(self) -> SearchQuery:
        """
        获取搜索查询
        
        如果尚未处理，则运行预处理步骤
        
        返回:
            SearchQuery: 处理后的搜索查询对象
        """
        if self._search_query is not None:
            return self._search_query

        self._run_preprocessing()

        return cast(SearchQuery, self._search_query)

    @property 
    def predicted_search_type(self) -> SearchType:
        """
        获取预测的搜索类型
        
        如果尚未处理，则运行预处理步骤
        
        返回:
            SearchType: 预测的搜索类型
        """
        if self._predicted_search_type is not None:
            return self._predicted_search_type

        self._run_preprocessing()
        return cast(SearchType, self._predicted_search_type)

    @property
    def predicted_flow(self) -> QueryFlow:
        """
        获取预测的查询流程类型
        
        如果尚未处理，则运行预处理步骤
        
        返回:
            QueryFlow: 预测的查询流程类型
        """
        if self._predicted_flow is not None:
            return self._predicted_flow

        self._run_preprocessing()
        return cast(QueryFlow, self._predicted_flow)

    """Retrieval and Postprocessing"""

    def _get_chunks(self) -> list[InferenceChunk]:
        """
        获取检索的文档块
        
        执行实际的检索操作，获取相关的文档块
        
        返回:
            list[InferenceChunk]: 检索到的文档块列表
        """
        if self._retrieved_chunks is not None:
            return self._retrieved_chunks

        # These chunks do not include large chunks and have been deduped
        # 这些文档块不包含大型块且已经去重
        self._retrieved_chunks = retrieve_chunks(
            query=self.search_query,
            document_index=self.document_index,
            db_session=self.db_session,
            retrieval_metrics_callback=self.retrieval_metrics_callback,
        )

        return cast(list[InferenceChunk], self._retrieved_chunks)

    @log_function_time(print_only=True)
    def _get_sections(self) -> list[InferenceSection]:
        """
        获取扩展的文档片段
        
        Returns an expanded section from each of the chunks.
        If whole docs (instead of above/below context) is specified then it will give back all of the whole docs
        that have a corresponding chunk.
        
        从每个块获取扩展的片段。
        如果指定了完整文档（而不是上下文），则会返回所有具有对应块的完整文档。
        
        This step should be fast for any document index implementation.
        对任何文档索引实现来说，这一步都应该很快。
        
        返回:
            list[InferenceSection]: 扩展后的文档片段列表
        """
        if self._retrieved_sections is not None:
            return self._retrieved_sections

        # These chunks are ordered, deduped, and contain no large chunks
        # 这些文档块是有序的、已去重的，且不包含大型块
        retrieved_chunks = self._get_chunks()

        above = self.search_query.chunks_above
        below = self.search_query.chunks_below

        expanded_inference_sections = []
        inference_chunks: list[InferenceChunk] = []
        chunk_requests: list[VespaChunkRequest] = []

        # Full doc setting takes priority
        # 完整文档设置具有优先权
        if self.search_query.full_doc:
            seen_document_ids = set()

            # This preserves the ordering since the chunks are retrieved in score order
            # 这保持了顺序，因为文档块是按分数顺序检索的
            for chunk in retrieved_chunks:
                if chunk.document_id not in seen_document_ids:
                    seen_document_ids.add(chunk.document_id)
                    chunk_requests.append(
                        VespaChunkRequest(
                            document_id=chunk.document_id,
                        )
                    )

            inference_chunks.extend(
                cleanup_chunks(
                    self.document_index.id_based_retrieval(
                        chunk_requests=chunk_requests,
                        filters=IndexFilters(access_control_list=None),
                    )
                )
            )

            # Create a dictionary to group chunks by document_id
            # 创建字典以按文档ID对文档块进行分组
            grouped_inference_chunks: dict[str, list[InferenceChunk]] = {}
            for chunk in inference_chunks:
                if chunk.document_id not in grouped_inference_chunks:
                    grouped_inference_chunks[chunk.document_id] = []
                grouped_inference_chunks[chunk.document_id].append(chunk)

            for chunk_group in grouped_inference_chunks.values():
                inference_section = inference_section_from_chunks(
                    center_chunk=chunk_group[0],
                    chunks=chunk_group,
                )

                if inference_section is not None:
                    expanded_inference_sections.append(inference_section)
                else:
                    logger.warning(
                        "Skipped creation of section for full docs, no chunks found"
                        # 跳过创建完整文档的片段，未找到文档块
                    )

            self._retrieved_sections = expanded_inference_sections
            return expanded_inference_sections

        # General flow:
        # - Combine chunks into lists by document_id
        # - For each document, run merge-intervals to get combined ranges
        #   - This allows for less queries to the document index
        # - Fetch all of the new chunks with contents for the combined ranges
        # - Reiterate the chunks again and map to the results above based on the chunk.
        #   This maintains the original chunks ordering. Note, we cannot simply sort by score here
        #   as reranking flow may wipe the scores for a lot of the chunks.
        doc_chunk_ranges_map = defaultdict(list)
        for chunk in retrieved_chunks:
            # The list of ranges for each document is ordered by score
            doc_chunk_ranges_map[chunk.document_id].append(
                ChunkRange(
                    chunks=[chunk],
                    start=max(0, chunk.chunk_id - above),
                    # No max known ahead of time, filter will handle this anyway
                    end=chunk.chunk_id + below,
                )
            )

        # List of ranges, outside list represents documents, inner list represents ranges
        merged_ranges = [
            merge_chunk_intervals(ranges) for ranges in doc_chunk_ranges_map.values()
        ]

        flat_ranges: list[ChunkRange] = [r for ranges in merged_ranges for r in ranges]

        for chunk_range in flat_ranges:
            # Don't need to fetch chunks within range for merging if chunk_above / below are 0.
            if above == below == 0:
                inference_chunks.extend(chunk_range.chunks)

            else:
                chunk_requests.append(
                    VespaChunkRequest(
                        document_id=chunk_range.chunks[0].document_id,
                        min_chunk_ind=chunk_range.start,
                        max_chunk_ind=chunk_range.end,
                    )
                )

        if chunk_requests:
            inference_chunks.extend(
                cleanup_chunks(
                    self.document_index.id_based_retrieval(
                        chunk_requests=chunk_requests,
                        filters=IndexFilters(access_control_list=None),
                        batch_retrieval=True,
                    )
                )
            )

        doc_chunk_ind_to_chunk = {
            (chunk.document_id, chunk.chunk_id): chunk for chunk in inference_chunks
        }

        # In case of failed parallel calls to Vespa, at least we should have the initial retrieved chunks
        # 如果对Vespa的并行调用失败，至少我们还有最初检索到的文档块
        doc_chunk_ind_to_chunk.update(
            {(chunk.document_id, chunk.chunk_id): chunk for chunk in retrieved_chunks}
        )

        # Build the surroundings for all of the initial retrieved chunks
        # 为所有初始检索的文档块构建周围环境
        for chunk in retrieved_chunks:
            start_ind = max(0, chunk.chunk_id - above)
            end_ind = chunk.chunk_id + below

            # Since the index of the max_chunk is unknown, just allow it to be None and filter after
            # 由于最大块的索引未知，暂时允许为None并在之后进行过滤
            surrounding_chunks_or_none = [
                doc_chunk_ind_to_chunk.get((chunk.document_id, chunk_ind))
                for chunk_ind in range(start_ind, end_ind + 1)  # end_ind is inclusive
            ]
            # The None will apply to the would be "chunks" that are larger than the index of the last chunk
            # of the document
            # None将应用于那些大于文档最后一个块索引的"潜在块"
            surrounding_chunks = [
                chunk for chunk in surrounding_chunks_or_none if chunk is not None
            ]

            inference_section = inference_section_from_chunks(
                center_chunk=chunk,
                chunks=surrounding_chunks,
            )
            if inference_section is not None:
                expanded_inference_sections.append(inference_section)
            else:
                logger.warning(
                    "Skipped creation of section, no chunks found"
                    # 跳过创建片段，未找到文档块
                )

        self._retrieved_sections = expanded_inference_sections
        return expanded_inference_sections

    @property
    def reranked_sections(self) -> list[InferenceSection]:
        """
        获取重排序后的文档片段
        
        Reranking is always done at the chunk level since section merging could create arbitrarily
        long sections which could be:
        1. Longer than the maximum context limit of even large rerankers
        2. Slow to calculate due to the quadratic scaling laws of Transformers
        
        重排序总是在块级别进行，因为片段合并可能会创建任意长的片段，这可能会：
        1. 超过即使是大型重排序器的最大上下文限制
        2. 由于Transformer的二次方缩放法则而计算缓慢
        
        返回:
            list[InferenceSection]: 重排序后的文档片段列表
        """
        if self._reranked_sections is not None:
            return self._reranked_sections

        self._postprocessing_generator = search_postprocessing(
            search_query=self.search_query,
            retrieved_sections=self._get_sections(),
            llm=self.fast_llm,
            rerank_metrics_callback=self.rerank_metrics_callback,
        )

        self._reranked_sections = cast(
            list[InferenceSection], next(self._postprocessing_generator)
        )

        return self._reranked_sections

    @property
    def final_context_sections(self) -> list[InferenceSection]:
        """
        获取最终的上下文片段列表
        
        将重排序后的片段进行合并处理
        
        返回:
            list[InferenceSection]: 合并后的最终上下文片段列表
        """
        if self._final_context_sections is not None:
            return self._final_context_sections

        self._final_context_sections = _merge_sections(sections=self.reranked_sections)
        return self._final_context_sections

    @property
    def section_relevance(self) -> list[SectionRelevancePiece] | None:
        """
        获取片段相关性评估结果
        
        根据评估类型(SKIP, BASIC, AGENTIC)执行不同的相关性评估策略:
        - SKIP: 跳过评估，返回None
        - BASIC: 使用基础评估方法
        - AGENTIC: 使用代理评估方法，并行处理每个片段
        
        返回:
            list[SectionRelevancePiece] | None: 片段相关性评估结果列表，或None（如果跳过评估）
        
        异常:
            ValueError: 当评估类型未指定或配置错误时抛出
        """
        if self._section_relevance is not None:
            return self._section_relevance

        if (
            self.search_query.evaluation_type == LLMEvaluationType.SKIP
            or DISABLE_LLM_DOC_RELEVANCE
        ):
            return None

        if self.search_query.evaluation_type == LLMEvaluationType.UNSPECIFIED:
            raise ValueError(
                "Attempted to access section relevance scores on search query with evaluation type `UNSPECIFIED`."
                + "The search query evaluation type should have been specified."
            )

        if self.search_query.evaluation_type == LLMEvaluationType.AGENTIC:
            sections = self.final_context_sections
            functions = [
                FunctionCall(
                    evaluate_inference_section,
                    (section, self.search_query.query, self.llm),
                )
                for section in sections
            ]
            try:
                results = run_functions_in_parallel(function_calls=functions)
                self._section_relevance = list(results.values())
            except Exception as e:
                raise ValueError(
                    "An issue occured during the agentic evaluation process."
                    # 代理评估过程中发生错误
                ) from e

        elif self.search_query.evaluation_type == LLMEvaluationType.BASIC:
            if DISABLE_LLM_DOC_RELEVANCE:
                raise ValueError(
                    "Basic search evaluation operation called while DISABLE_LLM_DOC_RELEVANCE is enabled."
                    # 在启用DISABLE_LLM_DOC_RELEVANCE时调用了基础搜索评估操作
                )
            self._section_relevance = next(
                cast(
                    Iterator[list[SectionRelevancePiece]],
                    self._postprocessing_generator,
                )
            )

        else:
            # All other cases should have been handled above
            # 所有其他情况应该已在上面处理过
            raise ValueError(
                f"Unexpected evaluation type: {self.search_query.evaluation_type}"
            )

        return self._section_relevance

    @property
    def section_relevance_list(self) -> list[bool]:
        """
        获取片段相关性的布尔值列表
        
        将相关性评估结果转换为布尔值列表，表示每个片段是否相关
        
        返回:
            list[bool]: 每个片段相关性的布尔值列表
        """
        llm_indices = relevant_sections_to_indices(
            relevance_sections=self.section_relevance,
            items=self.final_context_sections,
        )
        return [ind in llm_indices for ind in range(len(self.final_context_sections))]
