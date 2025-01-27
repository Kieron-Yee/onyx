"""
此文件实现了查询扩展相关的功能，包括多语言查询扩展和基于历史对话的查询重构。
主要功能包括：
1. 多语言查询扩展：将查询翻译成多种语言
2. 基于历史对话的查询重构：根据聊天历史优化当前查询
3. 基于线程的查询重构：在线程上下文中优化查询
"""

from collections.abc import Callable

from onyx.chat.chat_utils import combine_message_chain
from onyx.configs.chat_configs import DISABLE_LLM_QUERY_REPHRASE
from onyx.configs.model_configs import GEN_AI_HISTORY_CUTOFF
from onyx.db.models import ChatMessage
from onyx.llm.exceptions import GenAIDisabledException
from onyx.llm.factory import get_default_llms
from onyx.llm.interfaces import LLM
from onyx.llm.models import PreviousMessage
from onyx.llm.utils import dict_based_prompt_to_langchain_prompt
from onyx.llm.utils import message_to_string
from onyx.prompts.chat_prompts import HISTORY_QUERY_REPHRASE
from onyx.prompts.miscellaneous_prompts import LANGUAGE_REPHRASE_PROMPT
from onyx.utils.logger import setup_logger
from onyx.utils.text_processing import count_punctuation
from onyx.utils.threadpool_concurrency import run_functions_tuples_in_parallel

logger = setup_logger()


def llm_multilingual_query_expansion(query: str, language: str) -> str:
    """
    使用LLM进行多语言查询扩展，将输入查询翻译成目标语言。
    
    参数:
        query: 原始查询字符串
        language: 目标语言
        
    返回:
        str: 翻译后的查询字符串
    """
    
    def _get_rephrase_messages() -> list[dict[str, str]]:
        """
        生成用于查询重构的消息列表。
        
        返回:
            list[dict[str, str]]: 包含重构提示的消息列表
        """
        messages = [
            {
                "role": "user",
                "content": LANGUAGE_REPHRASE_PROMPT.format(
                    query=query, target_language=language
                ),
            },
        ]

        return messages

    try:
        _, fast_llm = get_default_llms(timeout=5)
    except GenAIDisabledException:
        logger.warning(
            "Unable to perform multilingual query expansion, Gen AI disabled"
        )
        return query

    messages = _get_rephrase_messages()
    filled_llm_prompt = dict_based_prompt_to_langchain_prompt(messages)
    model_output = message_to_string(fast_llm.invoke(filled_llm_prompt))
    logger.debug(model_output)

    return model_output


def multilingual_query_expansion(
    query: str,
    expansion_languages: list[str],
    use_threads: bool = True,
) -> list[str]:
    """
    将查询扩展到多种语言。
    
    参数:
        query: 原始查询字符串
        expansion_languages: 需要扩展到的目标语言列表
        use_threads: 是否使用多线程处理，默认为True
        
    返回:
        list[str]: 各种语言版本的查询列表
    """
    languages = [language.strip() for language in expansion_languages]
    if use_threads:
        functions_with_args: list[tuple[Callable, tuple]] = [
            (llm_multilingual_query_expansion, (query, language))
            for language in languages
        ]

        query_rephrases = run_functions_tuples_in_parallel(functions_with_args)
        return query_rephrases

    else:
        query_rephrases = [
            llm_multilingual_query_expansion(query, language) for language in languages
        ]
        return query_rephrases


def get_contextual_rephrase_messages(
    question: str,
    history_str: str,
    prompt_template: str = HISTORY_QUERY_REPHRASE,
) -> list[dict[str, str]]:
    """
    生成考虑上下文的查询重构消息。
    
    参数:
        question: 当前问题
        history_str: 聊天历史字符串
        prompt_template: 提示模板，默认使用HISTORY_QUERY_REPHRASE
        
    返回:
        list[dict[str, str]]: 包含重构提示的消息列表
    """
    messages = [
        {
            "role": "user",
            "content": prompt_template.format(
                question=question, chat_history=history_str
            ),
        },
    ]

    return messages


def history_based_query_rephrase(
    query: str,
    history: list[ChatMessage] | list[PreviousMessage],
    llm: LLM,
    size_heuristic: int = 200,
    punctuation_heuristic: int = 10,
    skip_first_rephrase: bool = True,
    prompt_template: str = HISTORY_QUERY_REPHRASE,
) -> str:
    """
    基于历史对话进行查询重构。
    
    参数:
        query: 原始查询字符串
        history: 聊天历史记录
        llm: LLM模型实例
        size_heuristic: 查询长度阈值，默认200
        punctuation_heuristic: 标点符号数量阈值，默认10
        skip_first_rephrase: 是否跳过第一次重构，默认True
        prompt_template: 提示模板
        
    返回:
        str: 重构后的查询字符串
    """
    # 全局禁用时，直接使用原始查询 / Globally disabled, just use the exact user query
    if DISABLE_LLM_QUERY_REPHRASE:
        return query

    # For some use cases, the first query should be untouched. Later queries must be rephrased
    # due to needing context but the first query has no context.
    # 对于某些用例，第一次查询应保持不变。后续查询必须重构，因为需要上下文，但第一次查询没有上下文。
    if skip_first_rephrase and not history:
        return query

    # If it's a very large query, assume it's a copy paste which we may want to find exactly
    # or at least very closely, so don't rephrase it
    # 如果是一个非常大的查询，假设它是一个复制粘贴，我们可能希望精确或至少非常接近地找到它，所以不要重构它
    if len(query) >= size_heuristic:
        return query

    # If there is an unusually high number of punctuations, it's probably not natural language
    # so don't rephrase it
    # 如果标点符号数量异常多，它可能不是自然语言，所以不要重构它
    if count_punctuation(query) >= punctuation_heuristic:
        return query

    history_str = combine_message_chain(
        messages=history, token_limit=GEN_AI_HISTORY_CUTOFF
    )

    prompt_msgs = get_contextual_rephrase_messages(
        question=query, history_str=history_str, prompt_template=prompt_template
    )

    filled_llm_prompt = dict_based_prompt_to_langchain_prompt(prompt_msgs)
    rephrased_query = message_to_string(llm.invoke(filled_llm_prompt))

    logger.debug(f"Rephrased combined query: {rephrased_query}")

    return rephrased_query


def thread_based_query_rephrase(
    user_query: str,
    history_str: str,
    llm: LLM | None = None,
    size_heuristic: int = 200,
    punctuation_heuristic: int = 10,
) -> str:
    """
    基于线程的查询重构。
    
    参数:
        user_query: 用户原始查询
        history_str: 聊天历史字符串
        llm: LLM模型实例，可选
        size_heuristic: 查询长度阈值，默认200
        punctuation_heuristic: 标点符号数量阈值，默认10
        
    返回:
        str: 重构后的查询字符串
    """
    if not history_str:
        return user_query

    if len(user_query) >= size_heuristic:
        return user_query

    if count_punctuation(user_query) >= punctuation_heuristic:
        return user_query

    if llm is None:
        try:
            llm, _ = get_default_llms()
        except GenAIDisabledException:
            # If Generative AI is turned off, just return the original query
            # 如果生成式AI被关闭，只返回原始查询
            return user_query

    prompt_msgs = get_contextual_rephrase_messages(
        question=user_query, history_str=history_str
    )

    filled_llm_prompt = dict_based_prompt_to_langchain_prompt(prompt_msgs)
    rephrased_query = message_to_string(llm.invoke(filled_llm_prompt))

    logger.debug(f"Rephrased combined query: {rephrased_query}")

    return rephrased_query
