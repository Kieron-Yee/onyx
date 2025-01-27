"""
此文件定义了搜索系统相关的数据模型和数据结构。
主要包含以下功能：
1. 搜索设置和配置相关的模型
2. 搜索请求和响应的数据结构
3. 搜索结果的表示模型
4. 文档检索和排序的相关模型
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator

from onyx.configs.chat_configs import NUM_RETURNED_HITS
from onyx.configs.constants import DocumentSource
from onyx.context.search.enums import LLMEvaluationType
from onyx.context.search.enums import OptionalSearchSetting
from onyx.context.search.enums import SearchType
from onyx.db.models import Persona
from onyx.db.models import SearchSettings
from onyx.indexing.models import BaseChunk
from onyx.indexing.models import IndexingSetting
from shared_configs.enums import RerankerProvider


# 仅需要足够的字符来识别文档中块的位置
MAX_METRICS_CONTENT = 200


class RerankingDetails(BaseModel):
    """
    重排序配置模型，用于控制搜索结果的重排序行为
    
    属性：
        rerank_model_name: 重排序模型名称
        rerank_api_url: 重排序API地址
        rerank_provider_type: 重排序提供者类型
        rerank_api_key: 重排序API密钥
        num_rerank: 需要重排序的结果数量
        disable_rerank_for_streaming: 是否在流式处理时禁用重排序
    """
    # If model is None (or num_rerank is 0), then reranking is turned off
    rerank_model_name: str | None
    rerank_api_url: str | None
    rerank_provider_type: RerankerProvider | None
    rerank_api_key: str | None = None

    num_rerank: int

    # For faster flows where the results should start immediately
    # this more time intensive step can be skipped
    # 对于需要立即开始结果的快速流程，可以跳过这个耗时的步骤
    disable_rerank_for_streaming: bool = False

    @classmethod
    def from_db_model(cls, search_settings: SearchSettings) -> "RerankingDetails":
        """
        从数据库模型创建重排序配置实例
        
        参数:
            search_settings: 搜索设置数据库模型
        返回:
            RerankingDetails实例
        """
        return cls(
            rerank_model_name=search_settings.rerank_model_name,
            rerank_provider_type=search_settings.rerank_provider_type,
            rerank_api_key=search_settings.rerank_api_key,
            num_rerank=search_settings.num_rerank,
            rerank_api_url=search_settings.rerank_api_url,
        )


class InferenceSettings(RerankingDetails):
    """
    推理设置模型，继承自RerankingDetails
    增加了多语言扩展支持
    """
    # Empty for no additional expansion
    multilingual_expansion: list[str]


class SearchSettingsCreationRequest(InferenceSettings, IndexingSetting):
    @classmethod
    def from_db_model(
        cls, search_settings: SearchSettings
    ) -> "SearchSettingsCreationRequest":
        inference_settings = InferenceSettings.from_db_model(search_settings)
        indexing_setting = IndexingSetting.from_db_model(search_settings)

        return cls(**inference_settings.dict(), **indexing_setting.dict())


class SavedSearchSettings(InferenceSettings, IndexingSetting):
    @classmethod
    def from_db_model(cls, search_settings: SearchSettings) -> "SavedSearchSettings":
        return cls(
            # Indexing Setting
            model_name=search_settings.model_name,
            model_dim=search_settings.model_dim,
            normalize=search_settings.normalize,
            query_prefix=search_settings.query_prefix,
            passage_prefix=search_settings.passage_prefix,
            provider_type=search_settings.provider_type,
            index_name=search_settings.index_name,
            multipass_indexing=search_settings.multipass_indexing,
            # Reranking Details
            rerank_model_name=search_settings.rerank_model_name,
            rerank_provider_type=search_settings.rerank_provider_type,
            rerank_api_key=search_settings.rerank_api_key,
            num_rerank=search_settings.num_rerank,
            # Multilingual Expansion
            multilingual_expansion=search_settings.multilingual_expansion,
            rerank_api_url=search_settings.rerank_api_url,
            disable_rerank_for_streaming=search_settings.disable_rerank_for_streaming,
        )


class Tag(BaseModel):
    """
    标签模型类
    
    属性:
        tag_key: 标签键
        tag_value: 标签值
    """
    tag_key: str
    tag_value: str


class BaseFilters(BaseModel):
    """
    基础过滤器模型
    
    属性:
        source_type: 来源类型列表
        document_set: 文档集合列表
        time_cutoff: 时间截止点
        tags: 标签列表
    """
    source_type: list[DocumentSource] | None = None
    document_set: list[str] | None = None
    time_cutoff: datetime | None = None
    tags: list[Tag] | None = None


class IndexFilters(BaseFilters):
    """
    索引过滤器模型，继承自BaseFilters
    
    属性:
        access_control_list: 访问控制列表
        tenant_id: 租户ID
    """
    access_control_list: list[str] | None
    tenant_id: str | None = None


class ChunkMetric(BaseModel):
    """
    数据块度量模型
    
    属性:
        document_id: 文档ID
        chunk_content_start: 数据块内容起始部分
        first_link: 第一个链接
        score: 评分
    """
    document_id: str
    chunk_content_start: str
    first_link: str | None
    score: float


class ChunkContext(BaseModel):
    """
    数据块上下文模型
    
    属性:
        chunks_above: 上文数据块数量
        chunks_below: 下文数据块数量
        full_doc: 是否返回完整文档
    """
    # If not specified (None), picked up from Persona settings if there is space
    # if specified (even if 0), it always uses the specified number of chunks above and below
    # 如果未指定(None)，则从Persona设置中获取（如果有空间）
    # 如果指定了（即使是0），则始终使用指定数量的上下文数据块
    chunks_above: int | None = None
    chunks_below: int | None = None
    full_doc: bool = False

    @field_validator("chunks_above", "chunks_below")
    @classmethod
    def check_non_negative(cls, value: int, field: Any) -> int:
        if value is not None and value < 0:
            raise ValueError(f"{field.name} must be non-negative")
        return value


class SearchRequest(ChunkContext):
    """
    搜索请求模型，包含搜索所需的所有参数
    
    属性:
        query: 搜索查询字符串
        search_type: 搜索类型
        human_selected_filters: 用户选择的过滤条件
        enable_auto_detect_filters: 是否启用自动检测过滤
        persona: 用户画像信息
        offset: 分页偏移量
        limit: 返回结果数量限制
    """
    query: str

    search_type: SearchType = SearchType.SEMANTIC

    human_selected_filters: BaseFilters | None = None
    enable_auto_detect_filters: bool | None = None
    persona: Persona | None = None

    # if None, no offset / limit
    offset: int | None = None
    limit: int | None = None

    multilingual_expansion: list[str] | None = None
    recency_bias_multiplier: float = 1.0
    hybrid_alpha: float | None = None
    rerank_settings: RerankingDetails | None = None
    evaluation_type: LLMEvaluationType = LLMEvaluationType.UNSPECIFIED
    model_config = ConfigDict(arbitrary_types_allowed=True)


class SearchQuery(ChunkContext):
    """
    经过处理的搜索请求，直接传递给搜索管道
    
    属性:
        query: 查询字符串
        processed_keywords: 处理后的关键词列表
        search_type: 搜索类型
        evaluation_type: LLM评估类型
        filters: 索引过滤条件
        chunks_above: 上文块数量
        chunks_below: 下文块数量
        rerank_settings: 重排序设置
        hybrid_alpha: 混合搜索权重
        recency_bias_multiplier: 时间偏差乘数
        max_llm_filter_sections: LLM过滤部分的最大数量
        num_hits: 返回结果数量
        offset: 结果偏移量
    """
    query: str
    processed_keywords: list[str]
    search_type: SearchType
    evaluation_type: LLMEvaluationType
    filters: IndexFilters

    # by this point, the chunks_above and chunks_below must be set
    chunks_above: int
    chunks_below: int

    rerank_settings: RerankingDetails | None
    hybrid_alpha: float
    recency_bias_multiplier: float

    # Only used if LLM evaluation type is not skip, None to use default settings
    max_llm_filter_sections: int

    num_hits: int = NUM_RETURNED_HITS
    offset: int = 0
    model_config = ConfigDict(frozen=True)


class RetrievalDetails(ChunkContext):
    """
    检索详情配置类
    
    属性:
        run_search: 是否执行搜索的设置选项
        real_time: 是否为实时/流式调用
        filters: 基础过滤条件
        enable_auto_detect_filters: 是否启用自动检测过滤
        offset: 分页偏移量
        limit: 返回结果数量限制
        dedupe_docs: 是否只返回最匹配的文档块
    """
    # Use LLM to determine whether to do a retrieval or only rely on existing history
    # If the Persona is configured to not run search (0 chunks), this is bypassed
    # If no Prompt is configured, the only search results are shown, this is bypassed
    # 使用LLM来决定是否执行检索或仅依赖现有历史
    # 如果Persona配置为不运行搜索（0个块），则会跳过
    # 如果没有配置Prompt，则只显示搜索结果，会跳过
    run_search: OptionalSearchSetting = OptionalSearchSetting.ALWAYS
    
    # Is this a real-time/streaming call or a question where Onyx can take more time?
    # Used to determine reranking flow
    # 这是实时/流式调用还是Onyx可以花更多时间处理的问题？
    # 用于确定重排序流程
    real_time: bool = True
    # The following have defaults in the Persona settings which can be overridden via
    # the query, if None, then use Persona settings
    filters: BaseFilters | None = None
    enable_auto_detect_filters: bool | None = None
    # if None, no offset / limit
    offset: int | None = None
    limit: int | None = None

    # If this is set, only the highest matching chunk (or merged chunks) is returned
    dedupe_docs: bool = False


class InferenceChunk(BaseChunk):
    """
    推理数据块模型，表示搜索结果中的一个文档片段
    
    属性:
        document_id: 文档ID
        source_type: 来源类型
        semantic_identifier: 语义标识符
        title: 标题
        score: 相关性评分
        hidden: 是否隐藏
        metadata: 元数据信息
    """
    document_id: str
    source_type: DocumentSource
    semantic_identifier: str
    title: str | None  # Separate from Semantic Identifier though often same
    boost: int
    recency_bias: float
    score: float | None
    hidden: bool
    is_relevant: bool | None = None
    relevance_explanation: str | None = None
    metadata: dict[str, str | list[str]]
    # Matched sections in the chunk. Uses Vespa syntax e.g. <hi>TEXT</hi>
    # to specify that a set of words should be highlighted. For example:
    # ["<hi>the</hi> <hi>answer</hi> is 42", "he couldn't find an <hi>answer</hi>"]
    match_highlights: list[str]

    # when the doc was last updated
    updated_at: datetime | None
    primary_owners: list[str] | None = None
    secondary_owners: list[str] | None = None
    large_chunk_reference_ids: list[int] = Field(default_factory=list)

    @property
    def unique_id(self) -> str:
        """
        获取数据块的唯一标识符
        
        返回:
            由document_id和chunk_id组合而成的唯一标识符
        """
        return f"{self.document_id}__{self.chunk_id}"

    def __repr__(self) -> str:
        blurb_words = self.blurb.split()
        short_blurb = ""
        for word in blurb_words:
            if not short_blurb:
                short_blurb = word
                continue
            if len(short_blurb) > 25:
                break
            short_blurb += " " + word
        return f"Inference Chunk: {self.document_id} - {short_blurb}..."

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, InferenceChunk):
            return False
        return (self.document_id, self.chunk_id) == (other.document_id, other.chunk_id)

    def __hash__(self) -> int:
        return hash((self.document_id, self.chunk_id))

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, InferenceChunk):
            return NotImplemented
        if self.score is None:
            if other.score is None:
                return self.chunk_id > other.chunk_id
            return True
        if other.score is None:
            return False
        if self.score == other.score:
            return self.chunk_id > other.chunk_id
        return self.score < other.score

    def __gt__(self, other: Any) -> bool:
        if not isinstance(other, InferenceChunk):
            return NotImplemented
        if self.score is None:
            return False
        if other.score is None:
            return True
        if self.score == other.score:
            return self.chunk_id < other.chunk_id
        return self.score > other.score


class InferenceChunkUncleaned(InferenceChunk):
    metadata_suffix: str | None

    def to_inference_chunk(self) -> InferenceChunk:
        # Create a dict of all fields except 'metadata_suffix'
        # Assumes the cleaning has already been applied and just needs to translate to the right type
        inference_chunk_data = {
            k: v
            for k, v in self.model_dump().items()
            if k
            not in ["metadata_suffix"]  # May be other fields to throw out in the future
        }
        return InferenceChunk(**inference_chunk_data)


class InferenceSection(BaseModel):
    """
    具有组合内容的区块列表。一个区块可以是单个块、同一文档的多个块或整个文档。
    
    属性:
        center_chunk: 中心数据块
        chunks: 数据块列表
        combined_content: 组合后的内容
    """
    center_chunk: InferenceChunk
    chunks: list[InferenceChunk]
    combined_content: str


class SearchDoc(BaseModel):
    """
    搜索文档模型
    
    属性:
        document_id: 文档ID
        chunk_ind: 数据块索引
        semantic_identifier: 语义标识符
        link: 文档链接
        blurb: 文档摘要
        source_type: 来源类型
        boost: 提升因子
        hidden: 是否隐藏
        metadata: 元数据
        score: 评分
        is_relevant: 是否相关
        relevance_explanation: 相关性解释
        match_highlights: 匹配高亮部分
        updated_at: 更新时间
        primary_owners: 主要所有者
        secondary_owners: 次要所有者
        is_internet: 是否为互联网文档
    """
    document_id: str
    chunk_ind: int
    semantic_identifier: str
    link: str | None = None
    blurb: str
    source_type: DocumentSource
    boost: int
    # Whether the document is hidden when doing a standard search
    # since a standard search will never find a hidden doc, this can only ever
    # be `True` when doing an admin search
    # 当进行标准搜索时文档是否隐藏
    # 由于标准搜索永远不会找到隐藏的文档，这个值只有在进行管理员搜索时才可能为`True`
    hidden: bool
    metadata: dict[str, str | list[str]]
    score: float | None = None
    is_relevant: bool | None = None
    relevance_explanation: str | None = None
    # Matched sections in the doc. Uses Vespa syntax e.g. <hi>TEXT</hi>
    # to specify that a set of words should be highlighted. For example:
    # ["<hi>the</hi> <hi>answer</hi> is 42", "the answer is <hi>42</hi>""]
    match_highlights: list[str]
    # when the doc was last updated
    updated_at: datetime | None = None
    primary_owners: list[str] | None = None
    secondary_owners: list[str] | None = None
    is_internet: bool = False

    def model_dump(self, *args: list, **kwargs: dict[str, Any]) -> dict[str, Any]:  # type: ignore
        initial_dict = super().model_dump(*args, **kwargs)  # type: ignore
        initial_dict["updated_at"] = (
            self.updated_at.isoformat() if self.updated_at else None
        )
        return initial_dict


class SavedSearchDoc(SearchDoc):
    """
    警告：使用此方法时如果不提供db_doc_id需要小心。如果未提供db_doc_id，
    将无法实际获取保存的文档和信息。因此，仅在SavedSearchDoc将来不会被使用时才跳过提供此参数。
    """
    db_doc_id: int
    score: float = 0.0

    @classmethod
    def from_search_doc(
        cls, search_doc: SearchDoc, db_doc_id: int = 0
    ) -> "SavedSearchDoc":
        """IMPORTANT: careful using this and not providing a db_doc_id If db_doc_id is not
        provided, it won't be able to actually fetch the saved doc and info later on. So only skip
        providing this if the SavedSearchDoc will not be used in the future"""
        search_doc_data = search_doc.model_dump()
        search_doc_data["score"] = search_doc_data.get("score") or 0.0
        return cls(**search_doc_data, db_doc_id=db_doc_id)

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, SavedSearchDoc):
            return NotImplemented
        return self.score < other.score


class SavedSearchDocWithContent(SavedSearchDoc):
    """
    用于需要返回检索部分实际内容及匹配高亮的端点。
    
    属性:
        content: 文档内容
    """
    content: str


class RetrievalDocs(BaseModel):
    """
    检索文档集合模型
    
    属性:
        top_documents: 排名靠前的文档列表
    """
    top_documents: list[SavedSearchDoc]


class SearchResponse(RetrievalDocs):
    """
    搜索响应模型
    
    属性:
        llm_indices: LLM索引列表
    """
    llm_indices: list[int]


class RetrievalMetricsContainer(BaseModel):
    """
    检索度量容器模型
    
    属性:
        search_type: 搜索类型
        metrics: 评估指标列表，包含检索的分数
    """
    search_type: SearchType
    metrics: list[ChunkMetric]  # This contains the scores for retrieval as well


class RerankMetricsContainer(BaseModel):
    """
    此处保存的分数是集成交叉编码器的未经提升的平均分数
    
    属性:
        metrics: 评估指标列表
        raw_similarity_scores: 原始相似度分数列表
    """
    metrics: list[ChunkMetric]
    raw_similarity_scores: list[float]
