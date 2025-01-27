"""
聊天会话命名模块
本模块主要负责处理聊天会话的自动命名功能，通过分析聊天历史记录来生成合适的会话名称。
"""

from onyx.chat.chat_utils import combine_message_chain
from onyx.configs.chat_configs import LANGUAGE_CHAT_NAMING_HINT
from onyx.configs.model_configs import GEN_AI_HISTORY_CUTOFF
from onyx.db.models import ChatMessage
from onyx.db.search_settings import get_multilingual_expansion
from onyx.llm.interfaces import LLM
from onyx.llm.utils import dict_based_prompt_to_langchain_prompt
from onyx.llm.utils import message_to_string
from onyx.prompts.chat_prompts import CHAT_NAMING
from onyx.utils.logger import setup_logger

logger = setup_logger()


def get_renamed_conversation_name(
    full_history: list[ChatMessage],
    llm: LLM,
) -> str:
    """
    为聊天会话生成新的名称
    
    参数:
        full_history (list[ChatMessage]): 完整的聊天历史记录列表
        llm (LLM): 语言模型实例
        
    返回:
        str: 生成的会话名称
    """
    # 将聊天历史记录合并成字符串，并限制token数量
    history_str = combine_message_chain(
        messages=full_history, token_limit=GEN_AI_HISTORY_CUTOFF
    )

    # 根据是否启用多语言扩展来添加语言提示
    language_hint = (
        f"\n{LANGUAGE_CHAT_NAMING_HINT.strip()}"
        if bool(get_multilingual_expansion())
        else ""
    )

    # 构建提示消息
    prompt_msgs = [
        {
            "role": "user",
            "content": CHAT_NAMING.format(
                language_hint_or_empty=language_hint, chat_history=history_str
            ),
        },
    ]

    # 将提示消息转换为LLM可处理的格式并获取响应
    filled_llm_prompt = dict_based_prompt_to_langchain_prompt(prompt_msgs)
    new_name_raw = message_to_string(llm.invoke(filled_llm_prompt))

    # 清理生成的名称（去除多余的空格和引号）
    new_name = new_name_raw.strip().strip(' "')

    # 记录调试信息
    logger.debug(f"新的会话名称: {new_name}")

    return new_name
