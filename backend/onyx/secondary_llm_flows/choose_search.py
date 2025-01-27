"""
本模块用于决定是否需要执行搜索操作。
主要包含两个核心函数，用于根据用户查询和历史记录判断是否需要执行搜索。
"""

from langchain.schema import BaseMessage
from langchain.schema import HumanMessage
from langchain.schema import SystemMessage

from onyx.chat.chat_utils import combine_message_chain
from onyx.chat.prompt_builder.utils import translate_onyx_msg_to_langchain
from onyx.configs.chat_configs import DISABLE_LLM_CHOOSE_SEARCH
from onyx.configs.model_configs import GEN_AI_HISTORY_CUTOFF
from onyx.db.models import ChatMessage
from onyx.llm.interfaces import LLM
from onyx.llm.models import PreviousMessage
from onyx.llm.utils import dict_based_prompt_to_langchain_prompt
from onyx.llm.utils import message_to_string
from onyx.prompts.chat_prompts import AGGRESSIVE_SEARCH_TEMPLATE
from onyx.prompts.chat_prompts import NO_SEARCH
from onyx.prompts.chat_prompts import REQUIRE_SEARCH_HINT
from onyx.prompts.chat_prompts import REQUIRE_SEARCH_SYSTEM_MSG
from onyx.prompts.chat_prompts import SKIP_SEARCH
from onyx.utils.logger import setup_logger


logger = setup_logger()


def check_if_need_search_multi_message(
    query_message: ChatMessage,
    history: list[ChatMessage],
    llm: LLM,
) -> bool:
    """
    根据多条消息历史记录判断是否需要执行搜索。
    
    参数:
        query_message: 当前的查询消息
        history: 聊天历史记录列表
        llm: 语言模型接口
        
    返回:
        bool: True表示需要搜索，False表示不需要搜索
    """
    # Retrieve on start or when choosing is globally disabled
    if not history or DISABLE_LLM_CHOOSE_SEARCH:
        return True

    prompt_msgs: list[BaseMessage] = [SystemMessage(content=REQUIRE_SEARCH_SYSTEM_MSG)]
    prompt_msgs.extend([translate_onyx_msg_to_langchain(msg) for msg in history])

    last_query = query_message.message

    prompt_msgs.append(HumanMessage(content=f"{last_query}\n\n{REQUIRE_SEARCH_HINT}"))

    model_out = message_to_string(llm.invoke(prompt_msgs))

    if (NO_SEARCH.split()[0] + " ").lower() in model_out.lower():
        return False

    return True


def check_if_need_search(
    query: str,
    history: list[PreviousMessage],
    llm: LLM,
) -> bool:
    """
    根据单条查询和历史记录判断是否需要执行搜索。
    
    参数:
        query: 用户查询字符串
        history: 历史消息列表
        llm: 语言模型接口
        
    返回:
        bool: True表示需要搜索，False表示不需要搜索
    """
    
    def _get_search_messages(
        question: str,
        history_str: str,
    ) -> list[dict[str, str]]:
        """
        构建用于判断是否需要搜索的消息列表。
        
        参数:
            question: 用户问题
            history_str: 历史记录字符串
            
        返回:
            list[dict[str, str]]: 格式化的消息列表
        """
        messages = [
            {
                "role": "user",
                "content": AGGRESSIVE_SEARCH_TEMPLATE.format(
                    final_query=question, chat_history=history_str
                ).strip(),
            },
        ]

        return messages

    # Choosing is globally disabled, use search
    # 全局禁用选择功能时，执行搜索
    if DISABLE_LLM_CHOOSE_SEARCH:
        return True

    history_str = combine_message_chain(
        messages=history, token_limit=GEN_AI_HISTORY_CUTOFF
    )

    prompt_msgs = _get_search_messages(question=query, history_str=history_str)

    filled_llm_prompt = dict_based_prompt_to_langchain_prompt(prompt_msgs)
    require_search_output = message_to_string(llm.invoke(filled_llm_prompt))

    logger.debug(f"Run search prediction: {require_search_output}")  # 运行搜索预测

    if (SKIP_SEARCH.split()[0]).lower() in require_search_output.lower():
        return False

    return True
