"""
此模块用于构建引用相关的提示信息。
主要功能包括：
1. 构建带有上下文和历史记录的LLM提示信息
2. 处理用户消息并生成适当的提示格式
"""

from langchain.schema.messages import HumanMessage

from onyx.chat.models import LlmDoc
from onyx.chat.models import PromptConfig
from onyx.configs.chat_configs import LANGUAGE_HINT
from onyx.context.search.models import InferenceChunk
from onyx.db.search_settings import get_multilingual_expansion
from onyx.llm.utils import message_to_prompt_and_imgs
from onyx.prompts.direct_qa_prompts import CONTEXT_BLOCK
from onyx.prompts.direct_qa_prompts import HISTORY_BLOCK
from onyx.prompts.direct_qa_prompts import JSON_PROMPT
from onyx.prompts.prompt_utils import add_date_time_to_prompt
from onyx.prompts.prompt_utils import build_complete_context_str


def _build_strong_llm_quotes_prompt(
    question: str,
    context_docs: list[LlmDoc] | list[InferenceChunk],
    history_str: str,
    prompt: PromptConfig,
) -> HumanMessage:
    """
    构建带有强引用的LLM提示信息。

    参数:
        question: 用户的问题
        context_docs: 上下文文档列表，可以是LlmDoc或InferenceChunk类型
        history_str: 对话历史记录字符串
        prompt: 提示配置对象

    返回:
        HumanMessage: 构建好的人类消息对象
    """
    use_language_hint = bool(get_multilingual_expansion())

    context_block = ""
    if context_docs:
        context_docs_str = build_complete_context_str(context_docs)
        context_block = CONTEXT_BLOCK.format(context_docs_str=context_docs_str)

    history_block = ""
    if history_str:
        history_block = HISTORY_BLOCK.format(history_str=history_str)

    full_prompt = JSON_PROMPT.format(
        system_prompt=prompt.system_prompt,
        context_block=context_block,
        history_block=history_block,
        task_prompt=prompt.task_prompt,
        user_query=question,
        language_hint_or_none=LANGUAGE_HINT.strip() if use_language_hint else "",
    ).strip()

    if prompt.datetime_aware:
        full_prompt = add_date_time_to_prompt(prompt_str=full_prompt)

    return HumanMessage(content=full_prompt)


def build_quotes_user_message(
    message: HumanMessage,
    context_docs: list[LlmDoc] | list[InferenceChunk],
    history_str: str,
    prompt: PromptConfig,
) -> HumanMessage:
    """
    构建带有引用的用户消息。

    参数:
        message: 用户的原始消息
        context_docs: 上下文文档列表，可以是LlmDoc或InferenceChunk类型
        history_str: 对话历史记录字符串
        prompt: 提示配置对象

    返回:
        HumanMessage: 处理后的用户消息对象，包含完整的上下文和引用信息
    """
    query, _ = message_to_prompt_and_imgs(message)

    return _build_strong_llm_quotes_prompt(
        question=query,
        context_docs=context_docs,
        history_str=history_str,
        prompt=prompt,
    )
