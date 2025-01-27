"""
此模块提供了聊天提示和消息处理的实用工具函数。
主要功能包括：
1. 构建模型提示
2. 消息格式转换
3. 聊天历史记录处理
"""

from langchain.schema.messages import AIMessage
from langchain.schema.messages import BaseMessage
from langchain.schema.messages import HumanMessage

from onyx.configs.constants import MessageType
from onyx.db.models import ChatMessage
from onyx.file_store.models import InMemoryChatFile
from onyx.llm.models import PreviousMessage
from onyx.llm.utils import build_content_with_imgs
from onyx.prompts.direct_qa_prompts import PARAMATERIZED_PROMPT
from onyx.prompts.direct_qa_prompts import PARAMATERIZED_PROMPT_WITHOUT_CONTEXT


def build_dummy_prompt(
    system_prompt: str, task_prompt: str, retrieval_disabled: bool
) -> str:
    """
    构建模型提示的示例格式。

    参数:
        system_prompt (str): 系统提示文本
        task_prompt (str): 任务提示文本
        retrieval_disabled (bool): 是否禁用检索功能

    返回:
        str: 格式化后的提示文本
    """
    if retrieval_disabled:
        return PARAMATERIZED_PROMPT_WITHOUT_CONTEXT.format(
            user_query="<USER_QUERY>",
            system_prompt=system_prompt,
            task_prompt=task_prompt,
        ).strip()

    return PARAMATERIZED_PROMPT.format(
        context_docs_str="<CONTEXT_DOCS>",
        user_query="<USER_QUERY>",
        system_prompt=system_prompt,
        task_prompt=task_prompt,
    ).strip()


def translate_onyx_msg_to_langchain(
    msg: ChatMessage | PreviousMessage,
) -> BaseMessage:
    """
    将Onyx消息格式转换为Langchain消息格式。

    参数:
        msg (ChatMessage | PreviousMessage): 需要转换的消息

    返回:
        BaseMessage: 转换后的Langchain消息对象

    异常:
        ValueError: 当消息类型不支持时抛出
    """
    files: list[InMemoryChatFile] = []

    # If the message is a `ChatMessage`, it doesn't have the downloaded files
    # attached. Just ignore them for now.
    # 如果消息是ChatMessage类型，它没有附带下载的文件，暂时忽略这些文件
    if not isinstance(msg, ChatMessage):
        files = msg.files
    content = build_content_with_imgs(msg.message, files, message_type=msg.message_type)

    if msg.message_type == MessageType.SYSTEM:
        raise ValueError("System messages are not currently part of history")
    if msg.message_type == MessageType.ASSISTANT:
        return AIMessage(content=content)
    if msg.message_type == MessageType.USER:
        return HumanMessage(content=content)

    raise ValueError(f"New message type {msg.message_type} not handled")


def translate_history_to_basemessages(
    history: list[ChatMessage] | list["PreviousMessage"],
) -> tuple[list[BaseMessage], list[int]]:
    """
    将聊天历史记录转换为Langchain基础消息格式。

    参数:
        history (list[ChatMessage] | list[PreviousMessage]): 聊天历史记录列表

    返回:
        tuple[list[BaseMessage], list[int]]: 
            - 转换后的Langchain消息列表
            - 对应的token计数列表
    """
    history_basemessages = [
        translate_onyx_msg_to_langchain(msg) for msg in history if msg.token_count != 0
    ]
    history_token_counts = [msg.token_count for msg in history if msg.token_count != 0]
    return history_basemessages, history_token_counts
