"""
This module contains the data models for Slack message processing
本模块包含 Slack 消息处理的数据模型
"""

from pydantic import BaseModel
from onyx.chat.models import ThreadMessage


class SlackMessageInfo(BaseModel):
    """
    Represents information about a Slack message and its context
    表示 Slack 消息及其上下文的信息
    """
    
    thread_messages: list[ThreadMessage]  # 线程消息列表
    channel_to_respond: str  # 需要回复的频道
    msg_to_respond: str | None  # 需要回复的消息，可为空
    thread_to_respond: str | None  # 需要回复的线程，可为空
    sender: str | None  # 消息发送者，可为空
    email: str | None  # 发送者邮箱，可为空
    bypass_filters: bool  # User has tagged @OnyxBot / 用户是否@了OnyxBot
    is_bot_msg: bool  # User is using /OnyxBot / 用户是否使用了/OnyxBot命令
    is_bot_dm: bool  # User is direct messaging to OnyxBot / 用户是否在和OnyxBot私聊
