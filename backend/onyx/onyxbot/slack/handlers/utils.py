"""
此模块提供了处理Slack消息线程的工具函数。
主要功能包括：
1. 格式化消息线程内容
2. 发送团队成员通知消息
"""

from slack_sdk import WebClient

from onyx.chat.models import ThreadMessage
from onyx.configs.constants import MessageType
from onyx.onyxbot.slack.utils import respond_in_thread


def slackify_message_thread(messages: list[ThreadMessage]) -> str:
    """
    将消息线程列表转换为格式化的字符串。

    # Note: this does not handle extremely long threads, every message will be included
    # with weaker LLMs, this could cause issues with exceeeding the token limit
    # 注意：这个函数不处理极长的消息线程，所有消息都会被包含
    # 对于较弱的LLM模型，这可能会导致超出token限制的问题

    参数:
        messages: ThreadMessage对象列表，包含需要格式化的消息

    返回:
        str: 格式化后的消息字符串，每条消息之间用两个换行符分隔
    """
    if not messages:
        return ""

    message_strs: list[str] = []
    for message in messages:
        if message.role == MessageType.USER:
            message_text = (
                f"{message.sender or 'Unknown User'} said in Slack:\n{message.message}"
            )
        elif message.role == MessageType.ASSISTANT:
            message_text = f"AI said in Slack:\n{message.message}"
        else:
            message_text = (
                f"{message.role.value.upper()} said in Slack:\n{message.message}"
            )
        message_strs.append(message_text)

    return "\n\n".join(message_strs)


def send_team_member_message(
    client: WebClient,
    channel: str,
    thread_ts: str,
) -> None:
    """
    在Slack线程中发送团队成员通知消息。

    参数:
        client: Slack WebClient实例，用于发送消息
        channel: 目标频道ID或名称
        thread_ts: 消息线程的时间戳，用于在正确的线程中回复

    返回:
        None
    """
    respond_in_thread(
        client=client,
        channel=channel,
        text=(
            "👋 Hi, we've just gathered and forwarded the relevant "
            + "information to the team. They'll get back to you shortly!"
        ),
        thread_ts=thread_ts,
    )
