"""
Slack Bot 工具类库

此模块包含了与 Slack Bot 交互相关的工具函数和类。主要功能包括:
- Slack 消息处理和回复
- 反馈和评分功能
- 用户和群组信息获取
- 速率限制控制
- 线程消息处理
"""

import logging
import random
import re
import string
import time
import uuid
from typing import Any
from typing import cast

from retry import retry
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from slack_sdk.models.blocks import Block
from slack_sdk.models.blocks import SectionBlock
from slack_sdk.models.metadata import Metadata
from slack_sdk.socket_mode import SocketModeClient

from onyx.configs.app_configs import DISABLE_TELEMETRY
from onyx.configs.constants import ID_SEPARATOR
from onyx.configs.constants import MessageType
from onyx.configs.onyxbot_configs import DANSWER_BOT_FEEDBACK_VISIBILITY
from onyx.configs.onyxbot_configs import DANSWER_BOT_MAX_QPM
from onyx.configs.onyxbot_configs import DANSWER_BOT_MAX_WAIT_TIME
from onyx.configs.onyxbot_configs import DANSWER_BOT_NUM_RETRIES
from onyx.configs.onyxbot_configs import (
    DANSWER_BOT_RESPONSE_LIMIT_PER_TIME_PERIOD,
)
from onyx.configs.onyxbot_configs import (
    DANSWER_BOT_RESPONSE_LIMIT_TIME_PERIOD_SECONDS,
)
from onyx.connectors.slack.utils import make_slack_api_rate_limited
from onyx.connectors.slack.utils import SlackTextCleaner
from onyx.db.engine import get_session_with_tenant
from onyx.db.users import get_user_by_email
from onyx.llm.exceptions import GenAIDisabledException
from onyx.llm.factory import get_default_llms
from onyx.llm.utils import dict_based_prompt_to_langchain_prompt
from onyx.llm.utils import message_to_string
from onyx.onyxbot.slack.constants import FeedbackVisibility
from onyx.onyxbot.slack.models import ThreadMessage
from onyx.prompts.miscellaneous_prompts import SLACK_LANGUAGE_REPHRASE_PROMPT
from onyx.utils.logger import setup_logger
from onyx.utils.telemetry import optional_telemetry
from onyx.utils.telemetry import RecordType
from onyx.utils.text_processing import replace_whitespaces_w_space

logger = setup_logger()


_DANSWER_BOT_SLACK_BOT_ID: str | None = None
_DANSWER_BOT_MESSAGE_COUNT: int = 0
_DANSWER_BOT_COUNT_START_TIME: float = time.time()


def get_onyx_bot_slack_bot_id(web_client: WebClient) -> Any:
    """
    获取 Onyx Bot 的 Slack Bot ID
    
    Args:
        web_client: Slack Web 客户端实例
        
    Returns:
        Any: Slack Bot 的唯一标识符
    """
    global _DANSWER_BOT_SLACK_BOT_ID
    if _DANSWER_BOT_SLACK_BOT_ID is None:
        _DANSWER_BOT_SLACK_BOT_ID = web_client.auth_test().get("user_id")
    return _DANSWER_BOT_SLACK_BOT_ID


def check_message_limit() -> bool:
    """
    检查消息数量是否超出限制
    
    This isnt a perfect solution.
    High traffic at the end of one period and start of another could cause
    the limit to be exceeded.
    这不是一个完美的解决方案。
    在一个时期结束和另一个时期开始时的高流量可能会导致超出限制。
    
    Returns:
        bool: 如果未超出限制返回 True，否则返回 False
    """
    if DANSWER_BOT_RESPONSE_LIMIT_PER_TIME_PERIOD == 0:
        return True
    global _DANSWER_BOT_MESSAGE_COUNT
    global _DANSWER_BOT_COUNT_START_TIME
    time_since_start = time.time() - _DANSWER_BOT_COUNT_START_TIME
    if time_since_start > DANSWER_BOT_RESPONSE_LIMIT_TIME_PERIOD_SECONDS:
        _DANSWER_BOT_MESSAGE_COUNT = 0
        _DANSWER_BOT_COUNT_START_TIME = time.time()
    if (_DANSWER_BOT_MESSAGE_COUNT + 1) > DANSWER_BOT_RESPONSE_LIMIT_PER_TIME_PERIOD:
        logger.error(
            f"OnyxBot has reached the message limit {DANSWER_BOT_RESPONSE_LIMIT_PER_TIME_PERIOD}"
            f" for the time period {DANSWER_BOT_RESPONSE_LIMIT_TIME_PERIOD_SECONDS} seconds."
            " These limits are configurable in backend/onyx/configs/onyxbot_configs.py"
        )
        return False
    _DANSWER_BOT_MESSAGE_COUNT += 1
    return True


def rephrase_slack_message(msg: str) -> str:
    """
    使用 LLM 重新表述 Slack 消息
    
    Args:
        msg: 原始消息文本
        
    Returns:
        str: 重新表述后的消息文本
    """
    def _get_rephrase_message() -> list[dict[str, str]]:
        messages = [
            {
                "role": "user",
                "content": SLACK_LANGUAGE_REPHRASE_PROMPT.format(query=msg),
            },
        ]

        return messages

    try:
        llm, _ = get_default_llms(timeout=5)
    except GenAIDisabledException:
        logger.warning("Unable to rephrase Slack user message, Gen AI disabled")
        return msg
    messages = _get_rephrase_message()
    filled_llm_prompt = dict_based_prompt_to_langchain_prompt(messages)
    model_output = message_to_string(llm.invoke(filled_llm_prompt))
    logger.debug(model_output)

    return model_output


def update_emote_react(
    emoji: str,
    channel: str,
    message_ts: str | None,
    remove: bool,
    client: WebClient,
) -> None:
    """
    更新 Slack 消息的表情反应
    
    Args:
        emoji: 表情符号
        channel: 频道 ID
        message_ts: 消息时间戳
        remove: 是否移除表情反应
        client: Slack Web 客户端实例
    """
    try:
        if not message_ts:
            logger.error(
                f"Tried to remove a react in {channel} but no message specified"
            )
            return

        func = client.reactions_remove if remove else client.reactions_add
        slack_call = make_slack_api_rate_limited(func)  # type: ignore
        slack_call(
            name=emoji,
            channel=channel,
            timestamp=message_ts,
        )
    except SlackApiError as e:
        if remove:
            logger.error(f"Failed to remove Reaction due to: {e}")
        else:
            logger.error(f"Was not able to react to user message due to: {e}")


def remove_onyx_bot_tag(message_str: str, client: WebClient) -> str:
    """
    移除消息中的 Onyx Bot 标记
    
    Args:
        message_str: 原始消息文本
        client: Slack Web 客户端实例
        
    Returns:
        str: 移除了 bot 标记的消息文本
    """
    bot_tag_id = get_onyx_bot_slack_bot_id(web_client=client)
    return re.sub(rf"<@{bot_tag_id}>\s", "", message_str)


def _check_for_url_in_block(block: Block) -> bool:
    """
    Check if the block has a key that contains "url" in it
    检查消息块中是否包含带有 "url" 的键
    """
    block_dict = block.to_dict()

    def check_dict_for_url(d: dict) -> bool:
        for key, value in d.items():
            if "url" in key.lower():
                return True
            if isinstance(value, dict):
                if check_dict_for_url(value):
                    return True
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict) and check_dict_for_url(item):
                        return True
        return False

    return check_dict_for_url(block_dict)


def _build_error_block(error_message: str) -> Block:
    """
    Build an error block to display in slack so that the user can see
    the error without completely breaking
    构建一个错误消息块在 Slack 中显示，让用户能够看到错误信息而不会完全中断
    """
    display_text = (
        "There was an error displaying all of the Onyx answers."
        f" Please let an admin or an onyx developer know. Error: {error_message}"
    )
    return SectionBlock(text=display_text)


@retry(
    tries=DANSWER_BOT_NUM_RETRIES,
    delay=0.25,
    backoff=2,
    logger=cast(logging.Logger, logger),
)
def respond_in_thread(
    client: WebClient,
    channel: str,
    thread_ts: str | None,
    text: str | None = None,
    blocks: list[Block] | None = None,
    receiver_ids: list[str] | None = None,
    metadata: Metadata | None = None,
    unfurl: bool = True,
) -> list[str]:
    """
    在线程中回复消息
    
    Args:
        client: Slack Web 客户端实例
        channel: 频道 ID
        thread_ts: 线程时间戳
        text: 文本内容（可选）
        blocks: 消息块列表（可选）
        receiver_ids: 接收者 ID 列表（可选）
        metadata: 元数据（可选）
        unfurl: 是否展开链接
        
    Returns:
        list[str]: 消息 ID 列表
        
    Raises:
        ValueError: 当 text 和 blocks 都为空时抛出
    """
    if not text and not blocks:
        raise ValueError("One of `text` or `blocks` must be provided")
        # 必须提供 text 或 blocks 中的一个

    message_ids: list[str] = []
    if not receiver_ids:
        slack_call = make_slack_api_rate_limited(client.chat_postMessage)
        try:
            response = slack_call(
                channel=channel,
                text=text,
                blocks=blocks,
                thread_ts=thread_ts,
                metadata=metadata,
                unfurl_links=unfurl,
                unfurl_media=unfurl,
            )
        except Exception as e:
            logger.warning(f"Failed to post message: {e} \n blocks: {blocks}")
            logger.warning("Trying again without blocks that have urls")

            if not blocks:
                raise e

            blocks_without_urls = [
                block for block in blocks if not _check_for_url_in_block(block)
            ]
            blocks_without_urls.append(_build_error_block(str(e)))

            # Try again wtihout blocks containing url
            # 重新尝试发送不包含 URL 的消息块
            response = slack_call(
                channel=channel,
                text=text,
                blocks=blocks_without_urls,
                thread_ts=thread_ts,
                metadata=metadata,
                unfurl_links=unfurl,
                unfurl_media=unfurl,
            )

        message_ids.append(response["message_ts"])
    else:
        slack_call = make_slack_api_rate_limited(client.chat_postEphemeral)
        for receiver in receiver_ids:
            try:
                response = slack_call(
                    channel=channel,
                    user=receiver,
                    text=text,
                    blocks=blocks,
                    thread_ts=thread_ts,
                    metadata=metadata,
                    unfurl_links=unfurl,
                    unfurl_media=unfurl,
                )
            except Exception as e:
                logger.warning(f"Failed to post message: {e} \n blocks: {blocks}")
                logger.warning("Trying again without blocks that have urls")

                if not blocks:
                    raise e

                blocks_without_urls = [
                    block for block in blocks if not _check_for_url_in_block(block)
                ]
                blocks_without_urls.append(_build_error_block(str(e)))

                # Try again wtihout blocks containing url
                # 重新尝试发送不包含 URL 的消息块
                response = slack_call(
                    channel=channel,
                    user=receiver,
                    text=text,
                    blocks=blocks_without_urls,
                    thread_ts=thread_ts,
                    metadata=metadata,
                    unfurl_links=unfurl,
                    unfurl_media=unfurl,
                )

            message_ids.append(response["message_ts"])

    return message_ids


def build_feedback_id(
    message_id: int,
    document_id: str | None = None,
    document_rank: int | None = None,
) -> str:
    """
    构建反馈 ID
    
    Args:
        message_id: 消息 ID
        document_id: 文档 ID（可选）
        document_rank: 文档排名（可选）
        
    Returns:
        str: 生成的反馈 ID
    """
    unique_prefix = "".join(random.choice(string.ascii_letters) for _ in range(10))
    if document_id is not None:
        if not document_id or document_rank is None:
            raise ValueError("Invalid document, missing information")
        if ID_SEPARATOR in document_id:
            raise ValueError(
                "Separator pattern should not already exist in document id"
            )
        feedback_id = ID_SEPARATOR.join(
            [str(message_id), document_id, str(document_rank)]
        )
    else:
        feedback_id = str(message_id)

    return unique_prefix + ID_SEPARATOR + feedback_id


def build_continue_in_web_ui_id(
    message_id: int,
) -> str:
    """
    构建网页界面中的继续操作 ID
    
    Args:
        message_id: 消息 ID
        
    Returns:
        str: 生成的继续操作 ID
    """
    unique_prefix = str(uuid.uuid4())[:10]
    return unique_prefix + ID_SEPARATOR + str(message_id)


def decompose_action_id(feedback_id: str) -> tuple[int, str | None, int | None]:
    """
    Decompose into query_id, document_id, document_rank, see above function
    分解为查询 ID、文档 ID 和文档排名，见上面的函数
    
    Args:
        feedback_id: 反馈 ID 字符串
        
    Returns:
        tuple: (查询 ID, 文档 ID, 文档排名)
        
    Raises:
        ValueError: 当反馈 ID 格式无效时抛出
    """
    try:
        components = feedback_id.split(ID_SEPARATOR)
        if len(components) != 2 and len(components) != 4:
            raise ValueError("Feedback ID does not contain right number of elements")

        if len(components) == 2:
            return int(components[-1]), None, None

        return int(components[1]), components[2], int(components[3])

    except Exception as e:
        logger.error(e)
        raise ValueError("Received invalid Feedback Identifier")


def get_view_values(state_values: dict[str, Any]) -> dict[str, str]:
    """
    Extract view values
    提取视图值
    
    Args:
        state_values (dict): The Slack view-submission values
        state_values (dict): Slack 视图提交的值

    Returns:
        dict: keys/values of the view state content
        dict: 视图状态内容的键值对
    """
    view_values = {}
    for _, view_data in state_values.items():
        for k, v in view_data.items():
            if (
                "selected_option" in v
                and isinstance(v["selected_option"], dict)
                and "value" in v["selected_option"]
            ):
                view_values[k] = v["selected_option"]["value"]
            elif "selected_options" in v and isinstance(v["selected_options"], list):
                view_values[k] = [
                    x["value"] for x in v["selected_options"] if "value" in x
                ]
            elif "selected_date" in v:
                view_values[k] = v["selected_date"]
            elif "value" in v:
                view_values[k] = v["value"]
    return view_values


def translate_vespa_highlight_to_slack(match_strs: list[str], used_chars: int) -> str:
    """
    将 Vespa 高亮文本转换为 Slack 格式
    
    Args:
        match_strs: 匹配的文本列表
        used_chars: 已使用的字符数
        
    Returns:
        str: Slack 格式的高亮文本
    """
    def _replace_highlight(s: str) -> str:
        s = re.sub(r"(?<=[^\s])<hi>(.*?)</hi>", r"\1", s)
        s = s.replace("</hi>", "*").replace("<hi>", "*")
        return s

    final_matches = [
        replace_whitespaces_w_space(_replace_highlight(match_str)).strip()
        for match_str in match_strs
        if match_str
    ]
    combined = "... ".join(final_matches)

    # Slack introduces "Show More" after 300 on desktop which is ugly
    # But don't trim the message if there is still a highlight after 300 chars
    # Slack 在桌面端 300 字符后会引入"显示更多"，这很难看
    # 但如果 300 字符后仍有高亮，则不要修剪消息
    remaining = 300 - used_chars
    if len(combined) > remaining and "*" not in combined[remaining:]:
        combined = combined[: remaining - 3] + "..."

    return combined


def remove_slack_text_interactions(slack_str: str) -> str:
    """
    移除 Slack 文本中的交互元素
    
    Args:
        slack_str: 原始 Slack 文本
        
    Returns:
        str: 清理后的文本
    """
    slack_str = SlackTextCleaner.replace_tags_basic(slack_str)
    slack_str = SlackTextCleaner.replace_channels_basic(slack_str)
    slack_str = SlackTextCleaner.replace_special_mentions(slack_str)
    slack_str = SlackTextCleaner.replace_special_catchall(slack_str)
    slack_str = SlackTextCleaner.add_zero_width_whitespace_after_tag(slack_str)
    return slack_str


def get_channel_from_id(client: WebClient, channel_id: str) -> dict[str, Any]:
    """
    通过频道 ID 获取频道信息
    
    Args:
        client: Slack Web 客户端实例
        channel_id: 频道 ID
        
    Returns:
        dict: 频道信息字典
    """
    response = client.conversations_info(channel=channel_id)
    response.validate()
    return response["channel"]


def get_channel_name_from_id(
    client: WebClient, channel_id: str
) -> tuple[str | None, bool]:
    """
    通过频道 ID 获取频道名称和类型
    
    Args:
        client: Slack Web 客户端实例
        channel_id: 频道 ID
        
    Returns:
        tuple: (频道名称, 是否为私聊)
        
    Raises:
        SlackApiError: Slack API 调用失败时抛出
    """
    try:
        channel_info = get_channel_from_id(client, channel_id)
        name = channel_info.get("name")
        is_dm = any([channel_info.get("is_im"), channel_info.get("is_mpim")])
        return name, is_dm
    except SlackApiError as e:
        logger.exception(f"Couldn't fetch channel name from id: {channel_id}")
        raise e


def fetch_slack_user_ids_from_emails(
    user_emails: list[str], client: WebClient
) -> tuple[list[str], list[str]]:
    """
    从邮箱列表获取 Slack 用户 ID
    
    Args:
        user_emails: 用户邮箱列表
        client: Slack Web 客户端实例
        
    Returns:
        tuple: (成功获取的用户 ID 列表, 未找到的邮箱列表)
    """
    user_ids: list[str] = []
    failed_to_find: list[str] = []
    for email in user_emails:
        try:
            user = client.users_lookupByEmail(email=email)
            user_ids.append(user.data["user"]["id"])  # type: ignore
        except Exception:
            logger.error(f"Was not able to find slack user by email: {email}")
            failed_to_find.append(email)

    return user_ids, failed_to_find


def fetch_user_ids_from_groups(
    given_names: list[str], client: WebClient
) -> tuple[list[str], list[str]]:
    """
    从用户组名称获取用户 ID 列表
    
    Args:
        given_names: 用户组名称列表
        client: Slack Web 客户端实例
        
    Returns:
        tuple: (成功获取的用户 ID 列表, 未找到的用户组名称列表)
    """
    user_ids: list[str] = []
    failed_to_find: list[str] = []
    try:
        response = client.usergroups_list()
        if not isinstance(response.data, dict):
            logger.error("Error fetching user groups")
            return user_ids, given_names

        all_group_data = response.data.get("usergroups", [])
        name_id_map = {d["name"]: d["id"] for d in all_group_data}
        handle_id_map = {d["handle"]: d["id"] for d in all_group_data}
        for given_name in given_names:
            group_id = name_id_map.get(given_name) or handle_id_map.get(
                given_name.lstrip("@")
            )
            if not group_id:
                failed_to_find.append(given_name)
                continue
            try:
                response = client.usergroups_users_list(usergroup=group_id)
                if isinstance(response.data, dict):
                    user_ids.extend(response.data.get("users", []))
                else:
                    failed_to_find.append(given_name)
            except Exception as e:
                logger.error(f"Error fetching user group ids: {str(e)}")
                failed_to_find.append(given_name)
    except Exception as e:
        logger.error(f"Error fetching user groups: {str(e)}")
        failed_to_find = given_names

    return user_ids, failed_to_find


def fetch_group_ids_from_names(
    given_names: list[str], client: WebClient
) -> tuple[list[str], list[str]]:
    """
    从用户组名称获取组 ID
    
    Args:
        given_names: 用户组名称列表
        client: Slack Web 客户端实例
        
    Returns:
        tuple: (成功获取的组 ID 列表, 未找到的组名称列表)
    """
    group_data: list[str] = []
    failed_to_find: list[str] = []

    try:
        response = client.usergroups_list()
        if not isinstance(response.data, dict):
            logger.error("Error fetching user groups")
            return group_data, given_names

        all_group_data = response.data.get("usergroups", [])

        name_id_map = {d["name"]: d["id"] for d in all_group_data}
        handle_id_map = {d["handle"]: d["id"] for d in all_group_data}

        for given_name in given_names:
            id = handle_id_map.get(given_name.lstrip("@"))
            id = id or name_id_map.get(given_name)
            if id:
                group_data.append(id)
            else:
                failed_to_find.append(given_name)
    except Exception as e:
        failed_to_find = given_names
        logger.error(f"Error fetching user groups: {str(e)}")

    return group_data, failed_to_find


def fetch_user_semantic_id_from_id(
    user_id: str | None, client: WebClient
) -> str | None:
    """
    从用户 ID 获取语义化 ID
    
    Args:
        user_id: Slack 用户 ID
        client: Slack Web 客户端实例
        
    Returns:
        str | None: 用户的语义化 ID，如果未找到则返回 None
    """
    if not user_id:
        return None

    response = make_slack_api_rate_limited(client.users_info)(user=user_id)
    if not response["ok"]:
        return None

    user: dict = cast(dict[Any, dict], response.data).get("user", {})

    return (
        user.get("real_name")
        or user.get("name")
        or user.get("profile", {}).get("email")
    )


def read_slack_thread(
    channel: str, thread: str, client: WebClient
) -> list[ThreadMessage]:
    """
    读取 Slack 线程消息
    
    Args:
        channel: 频道 ID
        thread: 线程 ID
        client: Slack Web 客户端实例
        
    Returns:
        list[ThreadMessage]: 线程消息列表
    """
    thread_messages: list[ThreadMessage] = []
    response = client.conversations_replies(channel=channel, ts=thread)
    replies = cast(dict, response.data).get("messages", [])
    for reply in replies:
        if "user" in reply and "bot_id" not in reply:
            message = reply["text"]
            user_sem_id = (
                fetch_user_semantic_id_from_id(reply.get("user"), client)
                or "Unknown User"
            )
            message_type = MessageType.USER
        else:
            self_slack_bot_id = get_onyx_bot_slack_bot_id(client)

            if reply.get("user") == self_slack_bot_id:
                # OnyxBot response
                # OnyxBot 响应
                message_type = MessageType.ASSISTANT
                user_sem_id = "Assistant"

                # OnyxBot responses have both text and blocks
                # The useful content is in the blocks, specifically the first block unless there are
                # auto-detected filters
                # OnyxBot 响应同时包含文本和消息块
                # 有用的内容在消息块中，特别是第一个块，除非有自动检测的过滤器
                blocks = reply.get("blocks")
                if not blocks:
                    logger.warning(f"OnyxBot response has no blocks: {reply}")
                    continue

                message = blocks[0].get("text", {}).get("text")

                # If auto-detected filters are on, use the second block for the actual answer
                # The first block is the auto-detected filters
                # 如果启用了自动检测过滤器，使用第二个块作为实际回答
                # 第一个块是自动检测的过滤器
                if message.startswith("_Filters"):
                    if len(blocks) < 2:
                        logger.warning(f"Only filter blocks found: {reply}")
                        continue
                    # This is the OnyxBot answer format, if there is a change to how we respond,
                    # this will need to be updated to get the correct "answer" portion
                    # 这是 OnyxBot 的回答格式，如果我们改变响应方式，
                    # 这部分需要更新以获取正确的"回答"部分
                    message = reply["blocks"][1].get("text", {}).get("text")
            else:
                # Other bots are not counted as the LLM response which only comes from Onyx
                # 其他机器人不算作 LLM 响应，LLM 响应只来自 Onyx
                message_type = MessageType.USER
                bot_user_name = fetch_user_semantic_id_from_id(
                    reply.get("user"), client
                )
                user_sem_id = bot_user_name or "Unknown" + " Bot"

                # For other bots, just use the text as we have no way of knowing that the
                # useful portion is
                # 对于其他机器人，直接使用文本，因为我们无法知道有用的部分在哪里
                message = reply.get("text")
                if not message:
                    message = blocks[0].get("text", {}).get("text")

            if not message:
                logger.warning("Skipping Slack thread message, no text found")
                continue

        message = remove_onyx_bot_tag(message, client=client)
        thread_messages.append(
            ThreadMessage(message=message, sender=user_sem_id, role=message_type)
        )

    return thread_messages


def slack_usage_report(
    action: str, sender_id: str | None, client: WebClient, tenant_id: str | None
) -> None:
    """
    记录 Slack 使用情况报告
    
    Args:
        action: 操作类型
        sender_id: 发送者 ID
        client: Slack Web 客户端实例
        tenant_id: 租户 ID
    """
    if DISABLE_TELEMETRY:
        return

    onyx_user = None
    sender_email = None
    try:
        sender_email = client.users_info(user=sender_id).data["user"]["profile"]["email"]  # type: ignore
    except Exception:
        logger.warning("Unable to find sender email")

    if sender_email is not None:
        with get_session_with_tenant(tenant_id) as db_session:
            onyx_user = get_user_by_email(email=sender_email, db_session=db_session)

    optional_telemetry(
        record_type=RecordType.USAGE,
        data={"action": action},
        user_id=str(onyx_user.id) if onyx_user else "Non-Onyx-Or-No-Auth-User",
    )


class SlackRateLimiter:
    """
    Slack 消息速率限制器
    
    用于控制消息发送频率，防止超出 Slack API 限制。
    包含队列管理和等待通知功能。
    """
    
    def __init__(self) -> None:
        """初始化速率限制器"""
        self.max_qpm: int | None = DANSWER_BOT_MAX_QPM
        self.max_wait_time = DANSWER_BOT_MAX_WAIT_TIME
        self.active_question = 0
        self.last_reset_time = time.time()
        self.waiting_questions: list[int] = []

    def refill(self) -> None:
        """重置活动问题计数"""
        # If elapsed time is greater than the period, reset the active question count
        if (time.time() - self.last_reset_time) > 60:
            self.active_question = 0
            self.last_reset_time = time.time()

    def notify(
        self, client: WebClient, channel: str, position: int, thread_ts: str | None
    ) -> None:
        """
        通知用户当前在队列中的位置
        
        Args:
            client: Slack Web 客户端实例
            channel: 频道 ID
            position: 队列位置
            thread_ts: 线程时间戳
        """
        respond_in_thread(
            client=client,
            channel=channel,
            receiver_ids=None,
            text=f"Your question has been queued. You are in position {position}.\n"
            f"Please wait a moment :hourglass_flowing_sand:",
            thread_ts=thread_ts,
        )

    def is_available(self) -> bool:
        """
        检查是否有可用的请求槽位
        
        Returns:
            bool: 如果有可用槽位返回 True，否则返回 False
        """
        if self.max_qpm is None:
            return True

        self.refill()
        return self.active_question < self.max_qpm

    def acquire_slot(self) -> None:
        """获取一个请求槽位"""
        self.active_question += 1

    def init_waiter(self) -> tuple[int, int]:
        """
        初始化等待者
        
        Returns:
            tuple[int, int]: (随机 ID, 队列位置)
        """
        func_randid = random.getrandbits(128)
        self.waiting_questions.append(func_randid)
        position = self.waiting_questions.index(func_randid) + 1

        return func_randid, position

    def waiter(self, func_randid: int) -> None:
        """
        等待轮到自己的回合
        
        Args:
            func_randid: 随机生成的等待者 ID
            
        Raises:
            TimeoutError: 如果等待时间超过最大等待时间
        """
        if self.max_qpm is None:
            return

        wait_time = 0
        while (
            self.active_question >= self.max_qpm
            or self.waiting_questions[0] != func_randid
        ):
            if wait_time > self.max_wait_time:
                raise TimeoutError
            time.sleep(2)
            wait_time += 2
            self.refill()

        del self.waiting_questions[0]


def get_feedback_visibility() -> FeedbackVisibility:
    """
    获取反馈可见性设置
    
    Returns:
        FeedbackVisibility: 反馈可见性枚举值
    """
    try:
        return FeedbackVisibility(DANSWER_BOT_FEEDBACK_VISIBILITY.lower())
    except ValueError:
        return FeedbackVisibility.PRIVATE


class TenantSocketModeClient(SocketModeClient):
    """
    租户 Socket 模式客户端
    
    继承自 SocketModeClient，添加了租户 ID 和 Slack Bot ID 支持
    """
    
    def __init__(
        self, tenant_id: str | None, slack_bot_id: int, *args: Any, **kwargs: Any
    ):
        super().__init__(*args, **kwargs)
        self.tenant_id = tenant_id
        self.slack_bot_id = slack_bot_id
