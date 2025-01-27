"""
此模块用于构建带有引用功能的提示信息。
主要功能包括：
1. 计算和管理提示词的token数量
2. 构建系统消息和用户消息
3. 处理文档上下文和引用
"""

from langchain.schema.messages import HumanMessage
from langchain.schema.messages import SystemMessage

from onyx.chat.models import LlmDoc
from onyx.chat.models import PromptConfig
from onyx.configs.model_configs import GEN_AI_SINGLE_USER_MESSAGE_EXPECTED_MAX_TOKENS
from onyx.context.search.models import InferenceChunk
from onyx.db.models import Persona
from onyx.db.persona import get_default_prompt__read_only
from onyx.db.search_settings import get_multilingual_expansion
from onyx.llm.factory import get_llms_for_persona
from onyx.llm.factory import get_main_llm_from_tuple
from onyx.llm.interfaces import LLMConfig
from onyx.llm.utils import build_content_with_imgs
from onyx.llm.utils import check_number_of_tokens
from onyx.llm.utils import get_max_input_tokens
from onyx.llm.utils import message_to_prompt_and_imgs
from onyx.prompts.chat_prompts import REQUIRE_CITATION_STATEMENT
from onyx.prompts.constants import DEFAULT_IGNORE_STATEMENT
from onyx.prompts.direct_qa_prompts import CITATIONS_PROMPT
from onyx.prompts.direct_qa_prompts import CITATIONS_PROMPT_FOR_TOOL_CALLING
from onyx.prompts.direct_qa_prompts import HISTORY_BLOCK
from onyx.prompts.prompt_utils import add_date_time_to_prompt
from onyx.prompts.prompt_utils import build_complete_context_str
from onyx.prompts.prompt_utils import build_task_prompt_reminders
from onyx.prompts.token_counts import ADDITIONAL_INFO_TOKEN_CNT
from onyx.prompts.token_counts import (
    CHAT_USER_PROMPT_WITH_CONTEXT_OVERHEAD_TOKEN_CNT,
)
from onyx.prompts.token_counts import CITATION_REMINDER_TOKEN_CNT
from onyx.prompts.token_counts import CITATION_STATEMENT_TOKEN_CNT
from onyx.prompts.token_counts import LANGUAGE_HINT_TOKEN_CNT
from onyx.utils.logger import setup_logger

logger = setup_logger()


def get_prompt_tokens(prompt_config: PromptConfig) -> int:
    """
    计算给定提示配置的总token数。
    
    参数:
        prompt_config: 提示词配置对象
    
    返回:
        int: 计算得到的总token数
    
    注: 目前自定义提示词不支持日期时间感知功能，仅默认提示词支持
    Note: currently custom prompts do not allow datetime aware, only default prompts
    """
    return (
        check_number_of_tokens(prompt_config.system_prompt)
        + check_number_of_tokens(prompt_config.task_prompt)
        + CHAT_USER_PROMPT_WITH_CONTEXT_OVERHEAD_TOKEN_CNT
        + CITATION_STATEMENT_TOKEN_CNT
        + CITATION_REMINDER_TOKEN_CNT
        + (LANGUAGE_HINT_TOKEN_CNT if get_multilingual_expansion() else 0)
        + (ADDITIONAL_INFO_TOKEN_CNT if prompt_config.datetime_aware else 0)
    )


# buffer just to be safe so that we don't overflow the token limit due to
# a small miscalculation
# 为了安全起见设置缓冲区，避免由于小的计算误差导致超出token限制
_MISC_BUFFER = 40


def compute_max_document_tokens(
    prompt_config: PromptConfig,
    llm_config: LLMConfig,
    actual_user_input: str | None = None,
    tool_token_count: int = 0,
    max_llm_token_override: int | None = None,
) -> int:
    """
    计算上下文文档可用的最大token数量。

    计算公式大致为：
    (模型上下文窗口 - 预留输出tokens - 提示词tokens - (实际用户输入 OR 预留用户消息tokens) - 安全缓冲区)

    参数:
        prompt_config: 提示词配置
        llm_config: LLM配置
        actual_user_input: 实际的用户输入文本
        tool_token_count: 工具使用的token数量
        max_llm_token_override: LLM最大token数的覆盖值

    返回:
        int: 文档可用的最大token数
    """
    # if we can't find a number of tokens, just assume some common default
    max_input_tokens = (
        max_llm_token_override
        if max_llm_token_override
        else get_max_input_tokens(
            model_name=llm_config.model_name, model_provider=llm_config.model_provider
        )
    )
    prompt_tokens = get_prompt_tokens(prompt_config)

    user_input_tokens = (
        check_number_of_tokens(actual_user_input)
        if actual_user_input is not None
        else GEN_AI_SINGLE_USER_MESSAGE_EXPECTED_MAX_TOKENS
    )

    return (
        max_input_tokens
        - prompt_tokens
        - user_input_tokens
        - tool_token_count
        - _MISC_BUFFER
    )


def compute_max_document_tokens_for_persona(
    persona: Persona,
    actual_user_input: str | None = None,
    max_llm_token_override: int | None = None,
) -> int:
    """
    计算特定角色可用的最大文档token数量。

    参数:
        persona: 角色对象
        actual_user_input: 实际的用户输入
        max_llm_token_override: LLM最大token数的覆盖值

    返回:
        int: 计算得到的最大文档token数
    """
    prompt = persona.prompts[0] if persona.prompts else get_default_prompt__read_only()
    return compute_max_document_tokens(
        prompt_config=PromptConfig.from_model(prompt),
        llm_config=get_main_llm_from_tuple(get_llms_for_persona(persona)).config,
        actual_user_input=actual_user_input,
        max_llm_token_override=max_llm_token_override,
    )


def compute_max_llm_input_tokens(llm_config: LLMConfig) -> int:
    """
    计算LLM输入允许的最大token数量。

    参数:
        llm_config: LLM配置对象

    返回:
        int: LLM输入的最大token数
    """
    input_tokens = get_max_input_tokens(
        model_name=llm_config.model_name, model_provider=llm_config.model_provider
    )
    return input_tokens - _MISC_BUFFER


def build_citations_system_message(
    prompt_config: PromptConfig,
) -> SystemMessage:
    """
    构建包含引用功能的系统消息。

    参数:
        prompt_config: 提示词配置对象

    返回:
        SystemMessage: 构建的系统消息对象
    """
    system_prompt = prompt_config.system_prompt.strip()
    if prompt_config.include_citations:
        system_prompt += REQUIRE_CITATION_STATEMENT
    if prompt_config.datetime_aware:
        system_prompt = add_date_time_to_prompt(prompt_str=system_prompt)

    return SystemMessage(content=system_prompt)


def build_citations_user_message(
    message: HumanMessage,
    prompt_config: PromptConfig,
    context_docs: list[LlmDoc] | list[InferenceChunk],
    all_doc_useful: bool,
    history_message: str = "",
) -> HumanMessage:
    """
    构建包含引用功能的用户消息。

    参数:
        message: 用户消息对象
        prompt_config: 提示词配置
        context_docs: 上下文文档列表
        all_doc_useful: 是否所有文档都有用
        history_message: 历史消息内容

    返回:
        HumanMessage: 构建的用户消息对象
    """
    multilingual_expansion = get_multilingual_expansion()
    task_prompt_with_reminder = build_task_prompt_reminders(
        prompt=prompt_config, use_language_hint=bool(multilingual_expansion)
    )

    history_block = (
        HISTORY_BLOCK.format(history_str=history_message) + "\n"
        if history_message
        else ""
    )
    query, img_urls = message_to_prompt_and_imgs(message)

    if context_docs:
        context_docs_str = build_complete_context_str(context_docs)
        optional_ignore = "" if all_doc_useful else DEFAULT_IGNORE_STATEMENT

        user_prompt = CITATIONS_PROMPT.format(
            optional_ignore_statement=optional_ignore,
            context_docs_str=context_docs_str,
            task_prompt=task_prompt_with_reminder,
            user_query=query,
            history_block=history_block,
        )
    else:
        # if no context docs provided, assume we're in the tool calling flow
        user_prompt = CITATIONS_PROMPT_FOR_TOOL_CALLING.format(
            task_prompt=task_prompt_with_reminder,
            user_query=query,
            history_block=history_block,
        )

    user_prompt = user_prompt.strip()
    user_msg = HumanMessage(
        content=build_content_with_imgs(user_prompt, img_urls=img_urls)
        if img_urls
        else user_prompt
    )

    return user_msg
