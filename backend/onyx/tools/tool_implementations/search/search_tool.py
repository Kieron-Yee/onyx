"""
该文件实现了搜索工具的核心功能，用于在用户的知识库中执行语义搜索。
主要包含SearchTool类，该类提供了搜索功能的实现，包括查询处理、文档检索和结果处理等功能。
"""

import json
from collections.abc import Generator
from typing import Any
from typing import cast

from pydantic import BaseModel
from sqlalchemy.orm import Session

from onyx.chat.chat_utils import llm_doc_from_inference_section
from onyx.chat.llm_response_handler import LLMCall
from onyx.chat.models import AnswerStyleConfig
from onyx.chat.models import ContextualPruningConfig
from onyx.chat.models import DocumentPruningConfig
from onyx.chat.models import LlmDoc
from onyx.chat.models import OnyxContext
from onyx.chat.models import OnyxContexts
from onyx.chat.models import PromptConfig
from onyx.chat.models import SectionRelevancePiece
from onyx.chat.prompt_builder.build import AnswerPromptBuilder
from onyx.chat.prompt_builder.citations_prompt import compute_max_llm_input_tokens
from onyx.chat.prune_and_merge import prune_and_merge_sections
from onyx.chat.prune_and_merge import prune_sections
from onyx.configs.chat_configs import CONTEXT_CHUNKS_ABOVE
from onyx.configs.chat_configs import CONTEXT_CHUNKS_BELOW
from onyx.configs.model_configs import GEN_AI_MODEL_FALLBACK_MAX_TOKENS
from onyx.context.search.enums import LLMEvaluationType
from onyx.context.search.enums import QueryFlow
from onyx.context.search.enums import SearchType
from onyx.context.search.models import IndexFilters
from onyx.context.search.models import InferenceSection
from onyx.context.search.models import RerankingDetails
from onyx.context.search.models import RetrievalDetails
from onyx.context.search.models import SearchRequest
from onyx.context.search.pipeline import SearchPipeline
from onyx.db.models import Persona
from onyx.db.models import User
from onyx.llm.interfaces import LLM
from onyx.llm.models import PreviousMessage
from onyx.secondary_llm_flows.choose_search import check_if_need_search
from onyx.secondary_llm_flows.query_expansion import history_based_query_rephrase
from onyx.tools.message import ToolCallSummary
from onyx.tools.models import ToolResponse
from onyx.tools.tool import Tool
from onyx.tools.tool_implementations.search.search_utils import llm_doc_to_dict
from onyx.tools.tool_implementations.search_like_tool_utils import (
    build_next_prompt_for_search_like_tool,
)
from onyx.tools.tool_implementations.search_like_tool_utils import (
    FINAL_CONTEXT_DOCUMENTS_ID,
)
from onyx.tools.tool_implementations.search_like_tool_utils import (
    ORIGINAL_CONTEXT_DOCUMENTS_ID,
)
from onyx.utils.logger import setup_logger
from onyx.utils.special_types import JSON_ro

logger = setup_logger()

# 搜索响应摘要ID常量
SEARCH_RESPONSE_SUMMARY_ID = "search_response_summary"
# 搜索文档内容ID常量
SEARCH_DOC_CONTENT_ID = "search_doc_content"
# 章节相关性列表ID常量
SECTION_RELEVANCE_LIST_ID = "section_relevance_list"
# 搜索评估ID常量
SEARCH_EVALUATION_ID = "llm_doc_eval"


class SearchResponseSummary(BaseModel):
    """
    搜索响应摘要模型类
    
    属性:
        top_sections: 顶部段落列表
        rephrased_query: 重写后的查询语句
        predicted_flow: 预测的查询流程
        predicted_search: 预测的搜索类型
        final_filters: 最终的索引过滤器
        recency_bias_multiplier: 时效性偏差乘数
    """
    top_sections: list[InferenceSection]
    rephrased_query: str | None = None
    predicted_flow: QueryFlow | None
    predicted_search: SearchType | None
    final_filters: IndexFilters
    recency_bias_multiplier: float


# 在用户的知识库中运行语义搜索。这是默认的行为。只有在以下情况下才不应使用此工具：

# - 聊天历史中已经有足够的信息可以完整且准确地回答查询，且额外的信息或细节价值很小或没有价值。
# - 查询是某种不需要额外信息就能处理的请求。

# 提示：如果你不熟悉用户输入或认为用户输入可能是错误的，请使用此工具。
SEARCH_TOOL_DESCRIPTION = """
Runs a semantic search over the user's knowledge base. The default behavior is to use this tool. \
The only scenario where you should not use this tool is if:

- There is sufficient information in chat history to FULLY and ACCURATELY answer the query AND \
additional information or details would provide little or no value.
- The query is some form of request that does not require additional information to handle.

HINT: if you are unfamiliar with the user input OR think the user input is a typo, use this tool.
"""


class SearchTool(Tool):
    """
    搜索工具类，继承自Tool类
    
    提供了在用户知识库中执行语义搜索的功能，包括查询处理、文档检索和结果处理。
    """

    _NAME = "run_search"
    _DISPLAY_NAME = "Search Tool"
    _DESCRIPTION = SEARCH_TOOL_DESCRIPTION

    def __init__(
        self,
        db_session: Session,
        user: User | None,
        persona: Persona,
        retrieval_options: RetrievalDetails | None,
        prompt_config: PromptConfig,
        llm: LLM,
        fast_llm: LLM,
        pruning_config: DocumentPruningConfig,
        answer_style_config: AnswerStyleConfig,
        evaluation_type: LLMEvaluationType,
        # if specified, will not actually run a search and will instead return these
        # sections. Used when the user selects specific docs to talk to
        selected_sections: list[InferenceSection] | None = None,
        chunks_above: int | None = None,
        chunks_below: int | None = None,
        full_doc: bool = False,
        bypass_acl: bool = False,
        rerank_settings: RerankingDetails | None = None,
    ) -> None:
        """
        初始化SearchTool实例
        
        参数:
            db_session: 数据库会话
            user: 用户对象
            persona: 角色对象
            retrieval_options: 检索选项
            prompt_config: 提示配置
            llm: 语言模型实例
            fast_llm: 快速语言模型实例
            pruning_config: 文档剪枝配置
            answer_style_config: 答案样式配置
            evaluation_type: LLM评估类型
            selected_sections: 已选择的章节列表
            chunks_above: 上文chunk数量
            chunks_below: 下文chunk数量
            full_doc: 是否返回完整文档
            bypass_acl: 是否绕过访问控制
            rerank_settings: 重排序设置
        """
        self.user = user
        self.persona = persona
        self.retrieval_options = retrieval_options
        self.prompt_config = prompt_config
        self.llm = llm
        self.fast_llm = fast_llm
        self.evaluation_type = evaluation_type

        self.selected_sections = selected_sections

        self.full_doc = full_doc
        self.bypass_acl = bypass_acl
        self.db_session = db_session

        # Only used via API
        self.rerank_settings = rerank_settings

        self.chunks_above = (
            chunks_above
            if chunks_above is not None
            else (
                persona.chunks_above
                if persona.chunks_above is not None
                else CONTEXT_CHUNKS_ABOVE
            )
        )
        self.chunks_below = (
            chunks_below
            if chunks_below is not None
            else (
                persona.chunks_below
                if persona.chunks_below is not None
                else CONTEXT_CHUNKS_BELOW
            )
        )

        # For small context models, don't include additional surrounding context
        # The 3 here for at least minimum 1 above, 1 below and 1 for the middle chunk
        max_llm_tokens = compute_max_llm_input_tokens(self.llm.config)
        if max_llm_tokens < 3 * GEN_AI_MODEL_FALLBACK_MAX_TOKENS:
            self.chunks_above = 0
            self.chunks_below = 0

        num_chunk_multiple = self.chunks_above + self.chunks_below + 1

        self.answer_style_config = answer_style_config
        self.contextual_pruning_config = (
            ContextualPruningConfig.from_doc_pruning_config(
                num_chunk_multiple=num_chunk_multiple, doc_pruning_config=pruning_config
            )
        )

    @property
    def name(self) -> str:
        return self._NAME

    @property
    def description(self) -> str:
        return self._DESCRIPTION

    @property
    def display_name(self) -> str:
        return self._DISPLAY_NAME

    """For explicit tool calling"""

    def tool_definition(self) -> dict:
        """
        定义工具的接口规范
        
        返回值:
            包含工具名称、描述和参数定义的字典
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "What to search for",
                        },
                    },
                    "required": ["query"],
                },
            },
        }

    def build_tool_message_content(
        self, *args: ToolResponse
    ) -> str | list[str | dict[str, Any]]:
        """
        构建工具的响应消息内容
        
        参数:
            args: 工具响应对象列表
            
        返回值:
            格式化的响应消息内容
        """
        final_context_docs_response = next(
            response for response in args if response.id == FINAL_CONTEXT_DOCUMENTS_ID
        )
        final_context_docs = cast(list[LlmDoc], final_context_docs_response.response)

        return json.dumps(
            {
                "search_results": [
                    llm_doc_to_dict(doc, ind)
                    for ind, doc in enumerate(final_context_docs)
                ]
            }
        )

    """For LLMs that don't support tool calling"""

    def get_args_for_non_tool_calling_llm(
        self,
        query: str,
        history: list[PreviousMessage],
        llm: LLM,
        force_run: bool = False,
    ) -> dict[str, Any] | None:
        """
        为不支持工具调用的LLM获取参数
        
        参数:
            query: 查询字符串
            history: 历史消息列表
            llm: 语言模型实例
            force_run: 是否强制运行
            
        返回值:
            参数字典或None
        """
        if not force_run and not check_if_need_search(
            query=query, history=history, llm=llm
        ):
            return None

        rephrased_query = history_based_query_rephrase(
            query=query, history=history, llm=llm
        )
        return {"query": rephrased_query}

    """Actual tool execution"""

    def _build_response_for_specified_sections(
        self, query: str
    ) -> Generator[ToolResponse, None, None]:
        """
        为指定章节构建响应
        
        参数:
            query: 查询字符串
            
        返回值:
            生成器，产生工具响应对象
        """
        if self.selected_sections is None:
            raise ValueError("Sections must be specified")

        yield ToolResponse(
            id=SEARCH_RESPONSE_SUMMARY_ID,
            response=SearchResponseSummary(
                rephrased_query=None,
                top_sections=[],
                predicted_flow=None,
                predicted_search=None,
                final_filters=IndexFilters(access_control_list=None),  # dummy filters
                recency_bias_multiplier=1.0,
            ),
        )

        # Build selected sections for specified documents
        # 为指定文档构建选定的章节
        selected_sections = [
            SectionRelevancePiece(
                relevant=True,
                document_id=section.center_chunk.document_id,
                chunk_id=section.center_chunk.chunk_id,
            )
            for section in self.selected_sections
        ]

        yield ToolResponse(
            id=SECTION_RELEVANCE_LIST_ID,
            response=selected_sections,
        )

        final_context_sections = prune_and_merge_sections(
            sections=self.selected_sections,
            section_relevance_list=None,
            prompt_config=self.prompt_config,
            llm_config=self.llm.config,
            question=query,
            contextual_pruning_config=self.contextual_pruning_config,
        )

        llm_docs = [
            llm_doc_from_inference_section(section)
            for section in final_context_sections
        ]

        yield ToolResponse(id=FINAL_CONTEXT_DOCUMENTS_ID, response=llm_docs)

    def run(self, **kwargs: str) -> Generator[ToolResponse, None, None]:
        """
        执行搜索工具的主要逻辑
        
        参数:
            kwargs: 包含查询字符串的关键字参数
            
        返回值:
            生成器，产生工具响应对象
        """
        query = cast(str, kwargs["query"])

        if self.selected_sections:
            yield from self._build_response_for_specified_sections(query)
            return

        search_pipeline = SearchPipeline(
            search_request=SearchRequest(
                query=query,
                evaluation_type=self.evaluation_type,
                human_selected_filters=(
                    self.retrieval_options.filters if self.retrieval_options else None
                ),
                persona=self.persona,
                offset=(
                    self.retrieval_options.offset if self.retrieval_options else None
                ),
                limit=self.retrieval_options.limit if self.retrieval_options else None,
                rerank_settings=self.rerank_settings,
                chunks_above=self.chunks_above,
                chunks_below=self.chunks_below,
                full_doc=self.full_doc,
                enable_auto_detect_filters=(
                    self.retrieval_options.enable_auto_detect_filters
                    if self.retrieval_options
                    else None
                ),
            ),
            user=self.user,
            llm=self.llm,
            fast_llm=self.fast_llm,
            bypass_acl=self.bypass_acl,
            db_session=self.db_session,
            prompt_config=self.prompt_config,
        )

        yield ToolResponse(
            id=SEARCH_RESPONSE_SUMMARY_ID,
            response=SearchResponseSummary(
                rephrased_query=query,
                top_sections=search_pipeline.final_context_sections,
                predicted_flow=search_pipeline.predicted_flow,
                predicted_search=search_pipeline.predicted_search_type,
                final_filters=search_pipeline.search_query.filters,
                recency_bias_multiplier=search_pipeline.search_query.recency_bias_multiplier,
            ),
        )

        yield ToolResponse(
            id=SEARCH_DOC_CONTENT_ID,
            response=OnyxContexts(
                contexts=[
                    OnyxContext(
                        content=section.combined_content,
                        document_id=section.center_chunk.document_id,
                        semantic_identifier=section.center_chunk.semantic_identifier,
                        blurb=section.center_chunk.blurb,
                    )
                    for section in search_pipeline.reranked_sections
                ]
            ),
        )

        yield ToolResponse(
            id=SECTION_RELEVANCE_LIST_ID,
            response=search_pipeline.section_relevance,
        )

        pruned_sections = prune_sections(
            sections=search_pipeline.final_context_sections,
            section_relevance_list=search_pipeline.section_relevance_list,
            prompt_config=self.prompt_config,
            llm_config=self.llm.config,
            question=query,
            contextual_pruning_config=self.contextual_pruning_config,
        )

        llm_docs = [
            llm_doc_from_inference_section(section) for section in pruned_sections
        ]

        yield ToolResponse(id=FINAL_CONTEXT_DOCUMENTS_ID, response=llm_docs)

    def final_result(self, *args: ToolResponse) -> JSON_ro:
        """
        处理最终结果
        
        参数:
            args: 工具响应对象列表
            
        返回值:
            JSON格式的最终结果
            
        注释:
            需要使用json.loads(doc.json())是因为有些子字段默认不可序列化（如datetime）
            这样可以强制pydantic为我们生成JSON可序列化的格式
        """
        final_docs = cast(
            list[LlmDoc],
            next(arg.response for arg in args if arg.id == FINAL_CONTEXT_DOCUMENTS_ID),
        )
        # NOTE: need to do this json.loads(doc.json()) stuff because there are some
        # subfields that are not serializable by default (datetime)
        # this forces pydantic to make them JSON serializable for us
        return [json.loads(doc.model_dump_json()) for doc in final_docs]

    def build_next_prompt(
        self,
        prompt_builder: AnswerPromptBuilder,
        tool_call_summary: ToolCallSummary,
        tool_responses: list[ToolResponse],
        using_tool_calling_llm: bool,
    ) -> AnswerPromptBuilder:
        """
        构建下一个提示
        
        参数:
            prompt_builder: 答案提示构建器
            tool_call_summary: 工具调用摘要
            tool_responses: 工具响应列表
            using_tool_calling_llm: 是否使用支持工具调用的LLM
            
        返回值:
            更新后的答案提示构建器
        """
        return build_next_prompt_for_search_like_tool(
            prompt_builder=prompt_builder,
            tool_call_summary=tool_call_summary,
            tool_responses=tool_responses,
            using_tool_calling_llm=using_tool_calling_llm,
            answer_style_config=self.answer_style_config,
            prompt_config=self.prompt_config,
        )

    """Other utility functions"""

    @classmethod
    def get_search_result(
        cls, llm_call: LLMCall
    ) -> tuple[list[LlmDoc], list[LlmDoc]] | None:
        """
        Returns the final search results and a map of docs to their original search rank (which is what is displayed to user)
        """
        if not llm_call.tool_call_info:
            return None

        final_search_results = []
        initial_search_results = []

        for yield_item in llm_call.tool_call_info:
            if (
                isinstance(yield_item, ToolResponse)
                and yield_item.id == FINAL_CONTEXT_DOCUMENTS_ID
            ):
                final_search_results = cast(list[LlmDoc], yield_item.response)
            elif (
                isinstance(yield_item, ToolResponse)
                and yield_item.id == ORIGINAL_CONTEXT_DOCUMENTS_ID
            ):
                search_contexts = yield_item.response.contexts
                # original_doc_search_rank = 1
                for doc in search_contexts:
                    if doc.document_id not in initial_search_results:
                        initial_search_results.append(doc)

                initial_search_results = cast(list[LlmDoc], initial_search_results)

        return final_search_results, initial_search_results
