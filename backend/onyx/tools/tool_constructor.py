"""
这个文件负责构建和管理工具集。
主要功能包括：
1. 构建各种工具实例（搜索工具、图像生成工具、互联网搜索工具等）
2. 管理工具的配置信息
3. 处理工具的依赖关系和初始化
"""

from typing import cast
from uuid import UUID

from pydantic import BaseModel
from pydantic import Field
from sqlalchemy.orm import Session

from onyx.chat.models import AnswerStyleConfig
from onyx.chat.models import CitationConfig
from onyx.chat.models import DocumentPruningConfig
from onyx.chat.models import PromptConfig
from onyx.configs.app_configs import AZURE_DALLE_API_BASE
from onyx.configs.app_configs import AZURE_DALLE_API_KEY
from onyx.configs.app_configs import AZURE_DALLE_API_VERSION
from onyx.configs.app_configs import AZURE_DALLE_DEPLOYMENT_NAME
from onyx.configs.chat_configs import BING_API_KEY
from onyx.configs.model_configs import GEN_AI_TEMPERATURE
from onyx.context.search.enums import LLMEvaluationType
from onyx.context.search.models import InferenceSection
from onyx.context.search.models import RerankingDetails
from onyx.context.search.models import RetrievalDetails
from onyx.db.llm import fetch_existing_llm_providers
from onyx.db.models import Persona
from onyx.db.models import User
from onyx.file_store.models import InMemoryChatFile
from onyx.llm.interfaces import LLM
from onyx.llm.interfaces import LLMConfig
from onyx.natural_language_processing.utils import get_tokenizer
from onyx.tools.built_in_tools import get_built_in_tool_by_id
from onyx.tools.models import DynamicSchemaInfo
from onyx.tools.tool import Tool
from onyx.tools.tool_implementations.custom.custom_tool import (
    build_custom_tools_from_openapi_schema_and_headers,
)
from onyx.tools.tool_implementations.images.image_generation_tool import (
    ImageGenerationTool,
)
from onyx.tools.tool_implementations.internet_search.internet_search_tool import (
    InternetSearchTool,
)
from onyx.tools.tool_implementations.search.search_tool import SearchTool
from onyx.tools.utils import compute_all_tool_tokens
from onyx.tools.utils import explicit_tool_calling_supported
from onyx.utils.headers import header_dict_to_header_list
from onyx.utils.logger import setup_logger

logger = setup_logger()


def _get_image_generation_config(llm: LLM, db_session: Session) -> LLMConfig:
    """Helper function to get image generation LLM config based on available providers
    基于可用提供商获取图像生成LLM配置的辅助函数
    
    参数:
        llm: LLM实例，包含基本配置信息
        db_session: 数据库会话实例
        
    返回:
        LLMConfig: 图像生成的LLM配置对象
        
    异常:
        ValueError: 当找不到有效的OpenAI API密钥时抛出
    """
    if llm and llm.config.api_key and llm.config.model_provider == "openai":
        return LLMConfig(
            model_provider=llm.config.model_provider,
            model_name="dall-e-3",
            temperature=GEN_AI_TEMPERATURE,
            api_key=llm.config.api_key,
            api_base=llm.config.api_base,
            api_version=llm.config.api_version,
        )

    if llm.config.model_provider == "azure" and AZURE_DALLE_API_KEY is not None:
        return LLMConfig(
            model_provider="azure",
            model_name=f"azure/{AZURE_DALLE_DEPLOYMENT_NAME}",
            temperature=GEN_AI_TEMPERATURE,
            api_key=AZURE_DALLE_API_KEY,
            api_base=AZURE_DALLE_API_BASE,
            api_version=AZURE_DALLE_API_VERSION,
        )

    # Fallback to checking for OpenAI provider in database
    # 回退到数据库中检查OpenAI提供商
    llm_providers = fetch_existing_llm_providers(db_session)
    openai_provider = next(
        iter(
            [
                llm_provider
                for llm_provider in llm_providers
                if llm_provider.provider == "openai"
            ]
        ),
        None,
    )

    if not openai_provider or not openai_provider.api_key:
        raise ValueError("Image generation tool requires an OpenAI API key")

    return LLMConfig(
        model_provider=openai_provider.provider,
        model_name="dall-e-3",
        temperature=GEN_AI_TEMPERATURE,
        api_key=openai_provider.api_key,
        api_base=openai_provider.api_base,
        api_version=openai_provider.api_version,
    )


class SearchToolConfig(BaseModel):
    """
    搜索工具配置类
    
    属性:
        answer_style_config: 回答样式配置
        document_pruning_config: 文档剪枝配置
        retrieval_options: 检索选项详情
        rerank_settings: 重排序设置
        selected_sections: 选定的推理部分
        chunks_above: 上文块数
        chunks_below: 下文块数
        full_doc: 是否返回完整文档
        latest_query_files: 最近查询的文件列表
        bypass_acl: 是否绕过访问控制列表
    """
    answer_style_config: AnswerStyleConfig = Field(
        default_factory=lambda: AnswerStyleConfig(citation_config=CitationConfig())
    )
    document_pruning_config: DocumentPruningConfig = Field(
        default_factory=DocumentPruningConfig
    )
    retrieval_options: RetrievalDetails = Field(default_factory=RetrievalDetails)
    rerank_settings: RerankingDetails | None = None
    selected_sections: list[InferenceSection] | None = None
    chunks_above: int = 0
    chunks_below: int = 0
    full_doc: bool = False
    latest_query_files: list[InMemoryChatFile] | None = None
    # Use with care, should only be used for OnyxBot in channels with multiple users
    # 请谨慎使用，仅应在多用户频道中的OnyxBot中使用
    bypass_acl: bool = False


class InternetSearchToolConfig(BaseModel):
    """
    互联网搜索工具配置类
    
    属性:
        answer_style_config: 回答样式配置，默认设置所有文档都有用
    """
    answer_style_config: AnswerStyleConfig = Field(
        default_factory=lambda: AnswerStyleConfig(
            citation_config=CitationConfig(all_docs_useful=True)
        )
    )


class ImageGenerationToolConfig(BaseModel):
    """
    图像生成工具配置类
    
    属性:
        additional_headers: 额外的HTTP头信息
    """
    additional_headers: dict[str, str] | None = None


class CustomToolConfig(BaseModel):
    """
    自定义工具配置类
    
    属性:
        chat_session_id: 聊天会话ID
        message_id: 消息ID
        additional_headers: 额外的HTTP头信息
    """
    chat_session_id: UUID | None = None
    message_id: int | None = None
    additional_headers: dict[str, str] | None = None


def construct_tools(
    persona: Persona,
    prompt_config: PromptConfig,
    db_session: Session,
    user: User | None,
    llm: LLM,
    fast_llm: LLM,
    search_tool_config: SearchToolConfig | None = None,
    internet_search_tool_config: InternetSearchToolConfig | None = None,
    image_generation_tool_config: ImageGenerationToolConfig | None = None,
    custom_tool_config: CustomToolConfig | None = None,
) -> dict[int, list[Tool]]:
    """Constructs tools based on persona configuration and available APIs
    基于角色配置和可用API构建工具
    
    参数:
        persona: 角色实例
        prompt_config: 提示配置
        db_session: 数据库会话
        user: 用户实例
        llm: 语言模型实例
        fast_llm: 快速语言模型实例
        search_tool_config: 搜索工具配置
        internet_search_tool_config: 互联网搜索工具配置
        image_generation_tool_config: 图像生成工具配置
        custom_tool_config: 自定义工具配置
        
    返回:
        dict[int, list[Tool]]: 工具ID到工具列表的映射字典
        
    异常:
        ValueError: 当缺少必要的API密钥时抛出
    """
    tool_dict: dict[int, list[Tool]] = {}

    for db_tool_model in persona.tools:
        if db_tool_model.in_code_tool_id:
            tool_cls = get_built_in_tool_by_id(db_tool_model.id, db_session)

            # Handle Search Tool
            # 处理搜索工具
            if tool_cls.__name__ == SearchTool.__name__:
                if not search_tool_config:
                    search_tool_config = SearchToolConfig()

                search_tool = SearchTool(
                    db_session=db_session,
                    user=user,
                    persona=persona,
                    retrieval_options=search_tool_config.retrieval_options,
                    prompt_config=prompt_config,
                    llm=llm,
                    fast_llm=fast_llm,
                    pruning_config=search_tool_config.document_pruning_config,
                    answer_style_config=search_tool_config.answer_style_config,
                    selected_sections=search_tool_config.selected_sections,
                    chunks_above=search_tool_config.chunks_above,
                    chunks_below=search_tool_config.chunks_below,
                    full_doc=search_tool_config.full_doc,
                    evaluation_type=(
                        LLMEvaluationType.BASIC
                        if persona.llm_relevance_filter
                        else LLMEvaluationType.SKIP
                    ),
                    rerank_settings=search_tool_config.rerank_settings,
                    bypass_acl=search_tool_config.bypass_acl,
                )
                tool_dict[db_tool_model.id] = [search_tool]

            # Handle Image Generation Tool
            # 处理图像生成工具
            elif tool_cls.__name__ == ImageGenerationTool.__name__:
                if not image_generation_tool_config:
                    image_generation_tool_config = ImageGenerationToolConfig()

                img_generation_llm_config = _get_image_generation_config(
                    llm, db_session
                )

                tool_dict[db_tool_model.id] = [
                    ImageGenerationTool(
                        api_key=cast(str, img_generation_llm_config.api_key),
                        api_base=img_generation_llm_config.api_base,
                        api_version=img_generation_llm_config.api_version,
                        additional_headers=image_generation_tool_config.additional_headers,
                        model=img_generation_llm_config.model_name,
                    )
                ]

            # Handle Internet Search Tool
            # 处理互联网搜索工具
            elif tool_cls.__name__ == InternetSearchTool.__name__:
                if not internet_search_tool_config:
                    internet_search_tool_config = InternetSearchToolConfig()

                if not BING_API_KEY:
                    raise ValueError(
                        "Internet search tool requires a Bing API key, please contact your Onyx admin to get it added!"
                    )
                tool_dict[db_tool_model.id] = [
                    InternetSearchTool(
                        api_key=BING_API_KEY,
                        answer_style_config=internet_search_tool_config.answer_style_config,
                        prompt_config=prompt_config,
                    )
                ]

        # Handle custom tools
        # 处理自定义工具
        elif db_tool_model.openapi_schema:
            if not custom_tool_config:
                custom_tool_config = CustomToolConfig()

            tool_dict[db_tool_model.id] = cast(
                list[Tool],
                build_custom_tools_from_openapi_schema_and_headers(
                    db_tool_model.openapi_schema,
                    dynamic_schema_info=DynamicSchemaInfo(
                        chat_session_id=custom_tool_config.chat_session_id,
                        message_id=custom_tool_config.message_id,
                    ),
                    custom_headers=(db_tool_model.custom_headers or [])
                    + (
                        header_dict_to_header_list(
                            custom_tool_config.additional_headers or {}
                        )
                    ),
                ),
            )

    tools: list[Tool] = []
    for tool_list in tool_dict.values():
        tools.extend(tool_list)

    # factor in tool definition size when pruning
    # 在剪枝时考虑工具定义的大小
    if search_tool_config:
        search_tool_config.document_pruning_config.tool_num_tokens = (
            compute_all_tool_tokens(
                tools,
                get_tokenizer(
                    model_name=llm.config.model_name,
                    provider_type=llm.config.model_provider,
                ),
            )
        )
        search_tool_config.document_pruning_config.using_tool_message = (
            explicit_tool_calling_supported(
                llm.config.model_provider, llm.config.model_name
            )
        )

    return tool_dict
