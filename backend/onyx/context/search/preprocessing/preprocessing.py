"""
该模块主要用于搜索查询的预处理操作。
主要功能包括：
1. 查询分析
2. 检索预处理
3. 过滤条件构建
4. 搜索参数配置
"""

from sqlalchemy.orm import Session

from onyx.configs.chat_configs import BASE_RECENCY_DECAY
from onyx.configs.chat_configs import CONTEXT_CHUNKS_ABOVE
from onyx.configs.chat_configs import CONTEXT_CHUNKS_BELOW
from onyx.configs.chat_configs import DISABLE_LLM_DOC_RELEVANCE
from onyx.configs.chat_configs import FAVOR_RECENT_DECAY_MULTIPLIER
from onyx.configs.chat_configs import HYBRID_ALPHA
from onyx.configs.chat_configs import HYBRID_ALPHA_KEYWORD
from onyx.configs.chat_configs import NUM_POSTPROCESSED_RESULTS
from onyx.configs.chat_configs import NUM_RETURNED_HITS
from onyx.context.search.enums import LLMEvaluationType
from onyx.context.search.enums import RecencyBiasSetting
from onyx.context.search.enums import SearchType
from onyx.context.search.models import BaseFilters
from onyx.context.search.models import IndexFilters
from onyx.context.search.models import RerankingDetails
from onyx.context.search.models import SearchQuery
from onyx.context.search.models import SearchRequest
from onyx.context.search.preprocessing.access_filters import (
    build_access_filters_for_user,
)
from onyx.context.search.retrieval.search_runner import (
    remove_stop_words_and_punctuation,
)
from onyx.db.engine import CURRENT_TENANT_ID_CONTEXTVAR
from onyx.db.models import User
from onyx.db.search_settings import get_current_search_settings
from onyx.llm.interfaces import LLM
from onyx.natural_language_processing.search_nlp_models import QueryAnalysisModel
from onyx.secondary_llm_flows.source_filter import extract_source_filter
from onyx.secondary_llm_flows.time_filter import extract_time_filter
from onyx.utils.logger import setup_logger
from onyx.utils.threadpool_concurrency import FunctionCall
from onyx.utils.threadpool_concurrency import run_functions_in_parallel
from onyx.utils.timing import log_function_time
from shared_configs.configs import MULTI_TENANT


logger = setup_logger()


def query_analysis(query: str) -> tuple[bool, list[str]]:
    """
    对搜索查询进行分析。
    
    Args:
        query (str): 需要分析的查询字符串
    
    Returns:
        tuple[bool, list[str]]: 返回一个元组，包含:
            - 是否为关键词查询的布尔值
            - 提取出的关键词列表
    """
    analysis_model = QueryAnalysisModel()
    return analysis_model.predict(query)


@log_function_time(print_only=True)
def retrieval_preprocessing(
    search_request: SearchRequest,
    user: User | None,
    llm: LLM,
    db_session: Session,
    bypass_acl: bool = False,
    skip_query_analysis: bool = False,
    base_recency_decay: float = BASE_RECENCY_DECAY,
    favor_recent_decay_multiplier: float = FAVOR_RECENT_DECAY_MULTIPLIER,
) -> SearchQuery:
    """
    Logic is as follows:
    Any global disables apply first
    Then any filters or settings as part of the query are used
    Then defaults to Persona settings if not specified by the query
    
    处理逻辑如下：
    首先应用全局禁用设置
    然后使用查询中的过滤器或设置
    如果查询中未指定，则使用Persona设置作为默认值
    
    Args:
        search_request (SearchRequest): 搜索请求对象
        user (User | None): 用户对象或None
        llm (LLM): 语言模型接口
        db_session (Session): 数据库会话
        bypass_acl (bool): 是否绕过访问控制列表
        skip_query_analysis (bool): 是否跳过查询分析
        base_recency_decay (float): 基础时间衰减系数
        favor_recent_decay_multiplier (float): 偏好最近文档的衰减乘数
    
    Returns:
        SearchQuery: 处理后的搜索查询对象
    """
    query = search_request.query
    limit = search_request.limit
    offset = search_request.offset
    persona = search_request.persona

    preset_filters = search_request.human_selected_filters or BaseFilters()
    if persona and persona.document_sets and preset_filters.document_set is None:
        preset_filters.document_set = [
            document_set.name for document_set in persona.document_sets
        ]

    time_filter = preset_filters.time_cutoff
    if time_filter is None and persona:
        time_filter = persona.search_start_date

    source_filter = preset_filters.source_type

    auto_detect_time_filter = True
    auto_detect_source_filter = True
    if not search_request.enable_auto_detect_filters:
        logger.debug("Retrieval details disables auto detect filters")
        auto_detect_time_filter = False
        auto_detect_source_filter = False
    elif persona and persona.llm_filter_extraction is False:
        logger.debug("Persona disables auto detect filters")
        auto_detect_time_filter = False
        auto_detect_source_filter = False
    else:
        logger.debug("Auto detect filters enabled")

    if (
        time_filter is not None
        and persona
        and persona.recency_bias != RecencyBiasSetting.AUTO
    ):
        auto_detect_time_filter = False
        logger.debug("Not extract time filter - already provided")
    if source_filter is not None:
        logger.debug("Not extract source filter - already provided")
        auto_detect_source_filter = False

    # Based on the query figure out if we should apply any hard time filters /
    # if we should bias more recent docs even more strongly
    # 基于查询确定是否应该应用硬时间过滤器/是否应该更强烈地偏向最近的文档
    run_time_filters = (
        FunctionCall(extract_time_filter, (query, llm), {})
        if auto_detect_time_filter
        else None
    )

    # Based on the query, figure out if we should apply any source filters
    # 基于查询，确定是否应该应用任何来源过滤器
    run_source_filters = (
        FunctionCall(extract_source_filter, (query, llm, db_session), {})
        if auto_detect_source_filter
        else None
    )

    run_query_analysis = (
        None if skip_query_analysis else FunctionCall(query_analysis, (query,), {})
    )

    functions_to_run = [
        filter_fn
        for filter_fn in [
            run_time_filters,
            run_source_filters,
            run_query_analysis,
        ]
        if filter_fn
    ]
    parallel_results = run_functions_in_parallel(functions_to_run)

    predicted_time_cutoff, predicted_favor_recent = (
        parallel_results[run_time_filters.result_id]
        if run_time_filters
        else (None, None)
    )
    predicted_source_filters = (
        parallel_results[run_source_filters.result_id] if run_source_filters else None
    )

    # The extracted keywords right now are not very reliable, not using for now
    # Can maybe use for highlighting
    # 目前提取的关键词还不够可靠，暂时不使用
    # 可能可以用于高亮显示
    is_keyword, extracted_keywords = (
        parallel_results[run_query_analysis.result_id]
        if run_query_analysis
        else (None, None)
    )

    all_query_terms = query.split()
    processed_keywords = (
        remove_stop_words_and_punctuation(all_query_terms)
        # If the user is using a different language, don't edit the query or remove english stopwords
        # 如果用户使用的是其他语言，不要编辑查询或删除英语停用词
        if not search_request.multilingual_expansion
        else all_query_terms
    )

    user_acl_filters = (
        None if bypass_acl else build_access_filters_for_user(user, db_session)
    )
    final_filters = IndexFilters(
        source_type=preset_filters.source_type or predicted_source_filters,
        document_set=preset_filters.document_set,
        time_cutoff=time_filter or predicted_time_cutoff,
        tags=preset_filters.tags,  # Tags are never auto-extracted
        # 标签永远不会自动提取
        access_control_list=user_acl_filters,
        tenant_id=CURRENT_TENANT_ID_CONTEXTVAR.get() if MULTI_TENANT else None,
    )

    llm_evaluation_type = LLMEvaluationType.BASIC
    if search_request.evaluation_type is not LLMEvaluationType.UNSPECIFIED:
        llm_evaluation_type = search_request.evaluation_type

    elif persona:
        llm_evaluation_type = (
            LLMEvaluationType.BASIC
            if persona.llm_relevance_filter
            else LLMEvaluationType.SKIP
        )

    if DISABLE_LLM_DOC_RELEVANCE:
        if llm_evaluation_type:
            logger.info(
                "LLM chunk filtering would have run but has been globally disabled"
            )
        llm_evaluation_type = LLMEvaluationType.SKIP

    rerank_settings = search_request.rerank_settings
    # If not explicitly specified by the query, use the current settings
    # 如果查询没有明确指定，使用当前设置
    if rerank_settings is None:
        search_settings = get_current_search_settings(db_session)

        # For non-streaming flows, the rerank settings are applied at the search_request level
        # 对于非流式处理，重排序设置在搜索请求级别应用
        if not search_settings.disable_rerank_for_streaming:
            rerank_settings = RerankingDetails.from_db_model(search_settings)

    # Decays at 1 / (1 + (multiplier * num years))
    # 按照 1 / (1 + (乘数 * 年数)) 衰减
    if persona and persona.recency_bias == RecencyBiasSetting.NO_DECAY:
        recency_bias_multiplier = 0.0
    elif persona and persona.recency_bias == RecencyBiasSetting.BASE_DECAY:
        recency_bias_multiplier = base_recency_decay
    elif persona and persona.recency_bias == RecencyBiasSetting.FAVOR_RECENT:
        recency_bias_multiplier = base_recency_decay * favor_recent_decay_multiplier
    else:
        if predicted_favor_recent:
            recency_bias_multiplier = base_recency_decay * favor_recent_decay_multiplier
        else:
            recency_bias_multiplier = base_recency_decay

    hybrid_alpha = HYBRID_ALPHA_KEYWORD if is_keyword else HYBRID_ALPHA
    if search_request.hybrid_alpha:
        hybrid_alpha = search_request.hybrid_alpha

    # Search request overrides anything else as it's explicitly set by the request
    # If not explicitly specified, use the persona settings if they exist
    # Otherwise, use the global defaults
    # 搜索请求优先级最高，因为它是由请求明确设置的
    # 如果未明确指定，则使用persona设置（如果存在）
    # 否则，使用全局默认值
    chunks_above = (
        search_request.chunks_above
        if search_request.chunks_above is not None
        else (persona.chunks_above if persona else CONTEXT_CHUNKS_ABOVE)
    )
    chunks_below = (
        search_request.chunks_below
        if search_request.chunks_below is not None
        else (persona.chunks_below if persona else CONTEXT_CHUNKS_BELOW)
    )

    return SearchQuery(
        query=query,
        processed_keywords=processed_keywords,
        search_type=SearchType.KEYWORD if is_keyword else SearchType.SEMANTIC,
        evaluation_type=llm_evaluation_type,
        filters=final_filters,
        hybrid_alpha=hybrid_alpha,
        recency_bias_multiplier=recency_bias_multiplier,
        num_hits=limit if limit is not None else NUM_RETURNED_HITS,
        offset=offset or 0,
        rerank_settings=rerank_settings,
        # Should match the LLM filtering to the same as the reranked, it's understood as this is the number of results
        # the user wants to do heavier processing on, so do the same for the LLM if reranking is on
        # if no reranking settings are set, then use the global default
        # LLM过滤应该与重排序匹配，这被理解为用户想要进行更重处理的结果数量
        # 如果启用了重排序，对LLM也做同样的处理
        # 如果没有设置重排序设置，则使用全局默认值
        max_llm_filter_sections=rerank_settings.num_rerank
        if rerank_settings
        else NUM_POSTPROCESSED_RESULTS,
        chunks_above=chunks_above,
        chunks_below=chunks_below,
        full_doc=search_request.full_doc,
    )
