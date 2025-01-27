"""
此文件定义了聊天系统中使用的各种数据模型。
包含了对话消息、文档处理、LLM响应、工具调用等相关的数据结构定义。
主要用于处理对话系统中的数据交互和状态管理。
"""

from collections.abc import Callable
from collections.abc import Iterator
from datetime import datetime
from enum import Enum
from typing import Any
from typing import TYPE_CHECKING

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import model_validator

from onyx.configs.constants import DocumentSource
from onyx.configs.constants import MessageType
from onyx.context.search.enums import QueryFlow
from onyx.context.search.enums import RecencyBiasSetting
from onyx.context.search.enums import SearchType
from onyx.context.search.models import RetrievalDocs
from onyx.llm.override_models import PromptOverride
from onyx.tools.models import ToolCallFinalResult
from onyx.tools.models import ToolCallKickoff
from onyx.tools.models import ToolResponse
from onyx.tools.tool_implementations.custom.base_tool_types import ToolResultType

if TYPE_CHECKING:
    from onyx.db.models import Prompt


class LlmDoc(BaseModel):
    """This contains the minimal set information for the LLM portion including citations
    这包含了LLM部分所需的最小信息集合，包括引用信息
    
    属性说明：
        document_id: 文档唯一标识
        content: 文档内容
        blurb: 文档摘要
        semantic_identifier: 语义标识符
        source_type: 文档来源类型
        metadata: 元数据信息
        updated_at: 更新时间
        link: 文档链接
        source_links: 来源链接映射
        match_highlights: 匹配高亮片段
    """

    document_id: str
    content: str
    blurb: str
    semantic_identifier: str
    source_type: DocumentSource
    metadata: dict[str, str | list[str]]
    updated_at: datetime | None
    link: str | None
    source_links: dict[int, str] | None
    match_highlights: list[str] | None


# First chunk of info for streaming QA
class QADocsResponse(RetrievalDocs):
    """
    问答文档响应类，包含问答系统的文档检索结果
    
    属性说明：
        rephrased_query: 重新措辞的查询
        predicted_flow: 预测的查询流程
        predicted_search: 预测的搜索类型
        applied_source_filters: 应用的源过滤器
        applied_time_cutoff: 应用的时间截止点
        recency_bias_multiplier: 时效性偏差乘数
    """
    rephrased_query: str | None = None
    predicted_flow: QueryFlow | None
    predicted_search: SearchType | None
    applied_source_filters: list[DocumentSource] | None
    applied_time_cutoff: datetime | None
    recency_bias_multiplier: float

    def model_dump(self, *args: list, **kwargs: dict[str, Any]) -> dict[str, Any]:  # type: ignore
        """
        将模型转换为字典格式
        
        返回：
            dict: 包含模型数据的字典
        """
        initial_dict = super().model_dump(mode="json", *args, **kwargs)  # type: ignore
        initial_dict["applied_time_cutoff"] = (
            self.applied_time_cutoff.isoformat() if self.applied_time_cutoff else None
        )

        return initial_dict


class StreamStopReason(Enum):
    """
    流式响应停止原因枚举
    
    枚举值：
        CONTEXT_LENGTH: 上下文长度限制
        CANCELLED: 用户取消
    """
    CONTEXT_LENGTH = "context_length"
    CANCELLED = "cancelled"


class StreamStopInfo(BaseModel):
    """
    流式响应停止信息
    
    属性说明：
        stop_reason: 停止原因
    """
    stop_reason: StreamStopReason

    def model_dump(self, *args: list, **kwargs: dict[str, Any]) -> dict[str, Any]:  # type: ignore
        data = super().model_dump(mode="json", *args, **kwargs)  # type: ignore
        data["stop_reason"] = self.stop_reason.name
        return data


class LLMRelevanceFilterResponse(BaseModel):
    """
    LLM相关性过滤响应
    
    属性说明：
        llm_selected_doc_indices: LLM选择的文档索引列表
    """
    llm_selected_doc_indices: list[int]


class FinalUsedContextDocsResponse(BaseModel):
    """
    最终使用的上下文文档响应
    
    属性说明：
        final_context_docs: 最终上下文文档列表
    """
    final_context_docs: list[LlmDoc]


class RelevanceAnalysis(BaseModel):
    """
    相关性分析
    
    属性说明：
        relevant: 是否相关
        content: 内容
    """
    relevant: bool
    content: str | None = None


class SectionRelevancePiece(RelevanceAnalysis):
    """LLM analysis mapped to an Inference Section
    LLM分析映射到推理部分
    
    属性说明：
        document_id: 文档唯一标识
        chunk_id: 块ID
    """
    document_id: str
    chunk_id: int  # ID of the center chunk for a given inference section


class DocumentRelevance(BaseModel):
    """Contains all relevance information for a given search
    包含给定搜索的所有相关性信息
    
    属性说明：
        relevance_summaries: 相关性摘要
    """
    relevance_summaries: dict[str, RelevanceAnalysis]


class OnyxAnswerPiece(BaseModel):
    # A small piece of a complete answer. Used for streaming back answers.
    # 完整答案的一小部分。用于流式返回答案。
    answer_piece: str | None  # if None, specifies the end of an Answer


# An intermediate representation of citations, later translated into
# a mapping of the citation [n] number to SearchDoc
# 引用的中间表示，稍后转换为引用[n]编号到SearchDoc的映射
class CitationInfo(BaseModel):
    """
    引用信息
    
    属性说明：
        citation_num: 引用编号
        document_id: 文档唯一标识
    """
    citation_num: int
    document_id: str


class AllCitations(BaseModel):
    """
    所有引用信息
    
    属性说明：
        citations: 引用信息列表
    """
    citations: list[CitationInfo]


# This is a mapping of the citation number to the document index within
# the result search doc set
# 这是引用编号到结果搜索文档集内文档索引的映射
class MessageSpecificCitations(BaseModel):
    """
    特定消息引用信息
    
    属性说明：
        citation_map: 引用映射
    """
    citation_map: dict[int, int]


class MessageResponseIDInfo(BaseModel):
    """
    消息响应ID信息
    
    属性说明：
        user_message_id: 用户消息ID
        reserved_assistant_message_id: 预留助手消息ID
    """
    user_message_id: int | None
    reserved_assistant_message_id: int


class StreamingError(BaseModel):
    """
    流式错误信息
    
    属性说明：
        error: 错误信息
        stack_trace: 堆栈跟踪
    """
    error: str
    stack_trace: str | None = None


class OnyxContext(BaseModel):
    """
    Onyx上下文信息
    
    属性说明：
        content: 上下文内容
        document_id: 文档唯一标识
        semantic_identifier: 语义标识符
        blurb: 文档摘要
    """
    content: str
    document_id: str
    semantic_identifier: str
    blurb: str


class OnyxContexts(BaseModel):
    """
    Onyx上下文信息列表
    
    属性说明：
        contexts: 上下文信息列表
    """
    contexts: list[OnyxContext]


class OnyxAnswer(BaseModel):
    """
    Onyx答案信息
    
    属性说明：
        answer: 答案内容
    """
    answer: str | None


class ThreadMessage(BaseModel):
    """
    线程消息
    
    属性说明：
        message: 消息内容
        sender: 发送者
        role: 消息类型
    """
    message: str
    sender: str | None = None
    role: MessageType = MessageType.USER


class ChatOnyxBotResponse(BaseModel):
    """
    聊天Onyx机器人响应
    
    属性说明：
        answer: 答案内容
        citations: 引用信息列表
        docs: 问答文档响应
        llm_selected_doc_indices: LLM选择的文档索引列表
        error_msg: 错误信息
        chat_message_id: 聊天消息ID
        answer_valid: 答案是否有效
    """
    answer: str | None = None
    citations: list[CitationInfo] | None = None
    docs: QADocsResponse | None = None
    llm_selected_doc_indices: list[int] | None = None
    error_msg: str | None = None
    chat_message_id: int | None = None
    answer_valid: bool = True  # Reflexion result, default True if Reflexion not run


class FileChatDisplay(BaseModel):
    """
    文件聊天显示
    
    属性说明：
        file_ids: 文件ID列表
    """
    file_ids: list[str]


class CustomToolResponse(BaseModel):
    """
    自定义工具响应
    
    属性说明：
        response: 工具结果类型
        tool_name: 工具名称
    """
    response: ToolResultType
    tool_name: str


class ToolConfig(BaseModel):
    """
    工具配置
    
    属性说明：
        id: 工具ID
    """
    id: int


class PromptOverrideConfig(BaseModel):
    """
    提示覆盖配置
    
    属性说明：
        name: 配置名称
        description: 配置描述
        system_prompt: 系统提示
        task_prompt: 任务提示
        include_citations: 是否包含引用
        datetime_aware: 是否感知日期时间
    """
    name: str
    description: str = ""
    system_prompt: str
    task_prompt: str = ""
    include_citations: bool = True
    datetime_aware: bool = True


class PersonaOverrideConfig(BaseModel):
    """
    人物覆盖配置
    
    属性说明：
        name: 配置名称
        description: 配置描述
        search_type: 搜索类型
        num_chunks: 块数量
        llm_relevance_filter: 是否启用LLM相关性过滤
        llm_filter_extraction: 是否启用LLM过滤提取
        recency_bias: 时效性偏差设置
        llm_model_provider_override: LLM模型提供者覆盖
        llm_model_version_override: LLM模型版本覆盖
        prompts: 提示覆盖配置列表
        prompt_ids: 提示ID列表
        document_set_ids: 文档集ID列表
        tools: 工具配置列表
        tool_ids: 工具ID列表
        custom_tools_openapi: 自定义工具OpenAPI配置列表
    """
    name: str
    description: str
    search_type: SearchType = SearchType.SEMANTIC
    num_chunks: float | None = None
    llm_relevance_filter: bool = False
    llm_filter_extraction: bool = False
    recency_bias: RecencyBiasSetting = RecencyBiasSetting.AUTO
    llm_model_provider_override: str | None = None
    llm_model_version_override: str | None = None

    prompts: list[PromptOverrideConfig] = Field(default_factory=list)
    prompt_ids: list[int] = Field(default_factory=list)

    document_set_ids: list[int] = Field(default_factory=list)
    tools: list[ToolConfig] = Field(default_factory=list)
    tool_ids: list[int] = Field(default_factory=list)
    custom_tools_openapi: list[dict[str, Any]] = Field(default_factory=list)


AnswerQuestionPossibleReturn = (
    OnyxAnswerPiece
    | CitationInfo
    | OnyxContexts
    | FileChatDisplay
    | CustomToolResponse
    | StreamingError
    | StreamStopInfo
)


AnswerQuestionStreamReturn = Iterator[AnswerQuestionPossibleReturn]


class LLMMetricsContainer(BaseModel):
    """
    LLM指标容器
    
    属性说明：
        prompt_tokens: 提示词令牌数
        response_tokens: 响应令牌数
    """
    prompt_tokens: int
    response_tokens: int


StreamProcessor = Callable[[Iterator[str]], AnswerQuestionStreamReturn]


class DocumentPruningConfig(BaseModel):
    """
    文档裁剪配置
    
    属性说明：
        max_chunks: 最大块数
        max_window_percentage: 最大窗口百分比
        max_tokens: 最大令牌数
        is_manually_selected_docs: 是否手动选择文档
        use_sections: 是否使用部分
        tool_num_tokens: 工具令牌数
        using_tool_message: 是否使用工具消息
    """
    max_chunks: int | None = None
    max_window_percentage: float | None = None
    max_tokens: int | None = None
    is_manually_selected_docs: bool = False
    use_sections: bool = True
    tool_num_tokens: int = 0
    using_tool_message: bool = False


class ContextualPruningConfig(DocumentPruningConfig):
    """
    上下文裁剪配置
    
    属性说明：
        num_chunk_multiple: 块数量倍数
    """
    num_chunk_multiple: int

    @classmethod
    def from_doc_pruning_config(
        cls, num_chunk_multiple: int, doc_pruning_config: DocumentPruningConfig
    ) -> "ContextualPruningConfig":
        return cls(num_chunk_multiple=num_chunk_multiple, **doc_pruning_config.dict())


class CitationConfig(BaseModel):
    """
    引用配置
    
    属性说明：
        all_docs_useful: 所有文档是否有用
    """
    all_docs_useful: bool = False


class QuotesConfig(BaseModel):
    """
    引用文本配置
    """
    pass


class AnswerStyleConfig(BaseModel):
    """
    答案样式配置类
    
    属性说明：
        citation_config: 引用配置
        quotes_config: 引用文本配置
        document_pruning_config: 文档裁剪配置
        structured_response_format: 结构化响应格式
    """
    citation_config: CitationConfig | None = None
    quotes_config: QuotesConfig | None = None
    document_pruning_config: DocumentPruningConfig = Field(
        default_factory=DocumentPruningConfig
    )
    structured_response_format: dict | None = None

    @model_validator(mode="after")
    def check_quotes_and_citation(self) -> "AnswerStyleConfig":
        """
        检查引用和引文配置的有效性
        
        校验规则：
        1. citation_config 和 quotes_config 必须至少提供一个
        2. 不能同时提供这两个配置
        
        返回：
            AnswerStyleConfig: 验证后的配置对象
        
        异常：
            ValueError: 当配置无效时抛出
        """
        if self.citation_config is None and self.quotes_config is None:
            raise ValueError(
                "One of `citation_config` or `quotes_config` must be provided"
            )

        if self.citation_config is not None and self.quotes_config is not None:
            raise ValueError(
                "Only one of `citation_config` or `quotes_config` must be provided"
            )

        return self


class PromptConfig(BaseModel):
    """Final representation of the Prompt configuration passed
    into the `Answer` object.
    传递给`Answer`对象的提示配置的最终表示
    
    属性说明：
        system_prompt: 系统提示
        task_prompt: 任务提示
        datetime_aware: 是否感知日期时间
        include_citations: 是否包含引用
    """
    system_prompt: str
    task_prompt: str
    datetime_aware: bool
    include_citations: bool

    @classmethod
    def from_model(
        cls, model: "Prompt", prompt_override: PromptOverride | None = None
    ) -> "PromptConfig":
        override_system_prompt = (
            prompt_override.system_prompt if prompt_override else None
        )
        override_task_prompt = prompt_override.task_prompt if prompt_override else None

        return cls(
            system_prompt=override_system_prompt or model.system_prompt,
            task_prompt=override_task_prompt or model.task_prompt,
            datetime_aware=model.datetime_aware,
            include_citations=model.include_citations,
        )

    model_config = ConfigDict(frozen=True)


ResponsePart = (
    OnyxAnswerPiece
    | CitationInfo
    | ToolCallKickoff
    | ToolResponse
    | ToolCallFinalResult
    | StreamStopInfo
)
