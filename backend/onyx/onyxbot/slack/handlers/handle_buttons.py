"""
此文件用于处理Slack机器人中各种按钮事件的处理逻辑。
主要包括文档反馈、答案生成、用户反馈、后续跟进等按钮的处理函数。
"""

from typing import Any
from typing import cast

from slack_sdk import WebClient
from slack_sdk.models.blocks import SectionBlock
from slack_sdk.models.views import View
from slack_sdk.socket_mode.request import SocketModeRequest

from onyx.configs.constants import MessageType
from onyx.configs.constants import SearchFeedbackType
from onyx.configs.onyxbot_configs import DANSWER_FOLLOWUP_EMOJI
from onyx.connectors.slack.utils import expert_info_from_slack_id
from onyx.connectors.slack.utils import make_slack_api_rate_limited
from onyx.db.engine import get_session_with_tenant
from onyx.db.feedback import create_chat_message_feedback
from onyx.db.feedback import create_doc_retrieval_feedback
from onyx.document_index.document_index_utils import get_both_index_names
from onyx.document_index.factory import get_default_document_index
from onyx.onyxbot.slack.blocks import build_follow_up_resolved_blocks
from onyx.onyxbot.slack.blocks import get_document_feedback_blocks
from onyx.onyxbot.slack.config import get_slack_channel_config_for_bot_and_channel
from onyx.onyxbot.slack.constants import DISLIKE_BLOCK_ACTION_ID
from onyx.onyxbot.slack.constants import FeedbackVisibility
from onyx.onyxbot.slack.constants import LIKE_BLOCK_ACTION_ID
from onyx.onyxbot.slack.constants import VIEW_DOC_FEEDBACK_ID
from onyx.onyxbot.slack.handlers.handle_message import (
    remove_scheduled_feedback_reminder,
)
from onyx.onyxbot.slack.handlers.handle_regular_answer import (
    handle_regular_answer,
)
from onyx.onyxbot.slack.models import SlackMessageInfo
from onyx.onyxbot.slack.utils import build_feedback_id
from onyx.onyxbot.slack.utils import decompose_action_id
from onyx.onyxbot.slack.utils import fetch_group_ids_from_names
from onyx.onyxbot.slack.utils import fetch_slack_user_ids_from_emails
from onyx.onyxbot.slack.utils import get_channel_name_from_id
from onyx.onyxbot.slack.utils import get_feedback_visibility
from onyx.onyxbot.slack.utils import read_slack_thread
from onyx.onyxbot.slack.utils import respond_in_thread
from onyx.onyxbot.slack.utils import TenantSocketModeClient
from onyx.onyxbot.slack.utils import update_emote_react
from onyx.utils.logger import setup_logger


logger = setup_logger()


def handle_doc_feedback_button(
    req: SocketModeRequest,
    client: TenantSocketModeClient,
) -> None:
    """
    处理文档反馈按钮的点击事件
    
    参数:
        req: Slack的Socket模式请求对象
        client: 租户Socket模式客户端对象
    
    功能:
    - 从请求中提取反馈ID相关信息
    - 创建反馈视图并显示给用户
    """
    if not (actions := req.payload.get("actions")):
        logger.error("Missing actions. Unable to build the source feedback view")
        return

    # Extracts the feedback_id coming from the 'source feedback' button
    # and generates a new one for the View, to keep track of the doc info
    query_event_id, doc_id, doc_rank = decompose_action_id(actions[0].get("value"))
    external_id = build_feedback_id(query_event_id, doc_id, doc_rank)

    channel_id = req.payload["container"]["channel_id"]
    thread_ts = req.payload["container"]["thread_ts"]

    data = View(
        type="modal",
        callback_id=VIEW_DOC_FEEDBACK_ID,
        external_id=external_id,
        # We use the private metadata to keep track of the channel id and thread ts
        private_metadata=f"{channel_id}_{thread_ts}",
        title="Give Feedback",
        blocks=[get_document_feedback_blocks()],
        submit="send",
        close="cancel",
    )

    client.web_client.views_open(
        trigger_id=req.payload["trigger_id"], view=data.to_dict()
    )


def handle_generate_answer_button(
    req: SocketModeRequest,
    client: TenantSocketModeClient,
) -> None:
    """
    处理生成答案按钮的点击事件
    
    参数:
        req: Slack的Socket模式请求对象
        client: 租户Socket模式客户端对象
    
    功能:
    - 获取对话线程的上下文信息
    - 基于最后一个用户问题生成新的回答
    - 发送临时消息通知用户正在生成答案
    """
    channel_id = req.payload["channel"]["id"]
    channel_name = req.payload["channel"]["name"]
    message_ts = req.payload["message"]["ts"]
    thread_ts = req.payload["container"]["thread_ts"]
    user_id = req.payload["user"]["id"]
    expert_info = expert_info_from_slack_id(user_id, client.web_client, user_cache={})
    email = expert_info.email if expert_info else None

    if not thread_ts:
        raise ValueError("Missing thread_ts in the payload")

    thread_messages = read_slack_thread(
        channel=channel_id, thread=thread_ts, client=client.web_client
    )
    # remove all assistant messages till we get to the last user message
    # we want the new answer to be generated off of the last "question" in
    # the thread
    for i in range(len(thread_messages) - 1, -1, -1):
        if thread_messages[i].role == MessageType.USER:
            break
        if thread_messages[i].role == MessageType.ASSISTANT:
            thread_messages.pop(i)

    # tell the user that we're working on it
    # Send an ephemeral message to the user that we're generating the answer
    respond_in_thread(
        client=client.web_client,
        channel=channel_id,
        receiver_ids=[user_id],
        text="I'm working on generating a full answer for you. This may take a moment...",
        thread_ts=thread_ts,
    )

    with get_session_with_tenant(client.tenant_id) as db_session:
        slack_channel_config = get_slack_channel_config_for_bot_and_channel(
            db_session=db_session,
            slack_bot_id=client.slack_bot_id,
            channel_name=channel_name,
        )

        handle_regular_answer(
            message_info=SlackMessageInfo(
                thread_messages=thread_messages,
                channel_to_respond=channel_id,
                msg_to_respond=cast(str, message_ts or thread_ts),
                thread_to_respond=cast(str, thread_ts or message_ts),
                sender=user_id or None,
                email=email or None,
                bypass_filters=True,
                is_bot_msg=False,
                is_bot_dm=False,
            ),
            slack_channel_config=slack_channel_config,
            receiver_ids=None,
            client=client.web_client,
            tenant_id=client.tenant_id,
            channel=channel_id,
            logger=logger,
            feedback_reminder_id=None,
        )


def handle_slack_feedback(
    feedback_id: str,
    feedback_type: str,
    feedback_msg_reminder: str,
    client: WebClient,
    user_id_to_post_confirmation: str,
    channel_id_to_post_confirmation: str,
    thread_ts_to_post_confirmation: str,
    tenant_id: str | None,
) -> None:
    """
    处理Slack反馈事件
    
    参数:
        feedback_id: 反馈标识符
        feedback_type: 反馈类型
        feedback_msg_reminder: 反馈提醒消息
        client: Slack Web客户端
        user_id_to_post_confirmation: 发送确认消息的用户ID
        channel_id_to_post_confirmation: 发送确认消息的频道ID
        thread_ts_to_post_confirmation: 发送确认消息的会话时间戳
        tenant_id: 租户ID
    
    功能:
    - 处理不同类型的反馈（点赞、踩、认可、拒绝、隐藏等）
    - 记录反馈到数据库
    - 发送反馈确认消息
    """
    message_id, doc_id, doc_rank = decompose_action_id(feedback_id)

    with get_session_with_tenant(tenant_id) as db_session:
        if feedback_type in [LIKE_BLOCK_ACTION_ID, DISLIKE_BLOCK_ACTION_ID]:
            create_chat_message_feedback(
                is_positive=feedback_type == LIKE_BLOCK_ACTION_ID,
                feedback_text="",
                chat_message_id=message_id,
                user_id=None,  # no "user" for Slack bot for now
                db_session=db_session,
            )
            remove_scheduled_feedback_reminder(
                client=client,
                channel=user_id_to_post_confirmation,
                msg_id=feedback_msg_reminder,
            )
        elif feedback_type in [
            SearchFeedbackType.ENDORSE.value,
            SearchFeedbackType.REJECT.value,
            SearchFeedbackType.HIDE.value,
        ]:
            if doc_id is None or doc_rank is None:
                raise ValueError("Missing information for Document Feedback")

            if feedback_type == SearchFeedbackType.ENDORSE.value:
                feedback = SearchFeedbackType.ENDORSE
            elif feedback_type == SearchFeedbackType.REJECT.value:
                feedback = SearchFeedbackType.REJECT
            else:
                feedback = SearchFeedbackType.HIDE

            curr_ind_name, sec_ind_name = get_both_index_names(db_session)
            document_index = get_default_document_index(
                primary_index_name=curr_ind_name, secondary_index_name=sec_ind_name
            )

            create_doc_retrieval_feedback(
                message_id=message_id,
                document_id=doc_id,
                document_rank=doc_rank,
                document_index=document_index,
                db_session=db_session,
                clicked=False,  # Not tracking this for Slack
                feedback=feedback,
            )
        else:
            logger.error(f"Feedback type '{feedback_type}' not supported")

    if get_feedback_visibility() == FeedbackVisibility.PRIVATE or feedback_type not in [
        LIKE_BLOCK_ACTION_ID,
        DISLIKE_BLOCK_ACTION_ID,
    ]:
        client.chat_postEphemeral(
            channel=channel_id_to_post_confirmation,
            user=user_id_to_post_confirmation,
            thread_ts=thread_ts_to_post_confirmation,
            text="Thanks for your feedback!",
        )
    else:
        feedback_response_txt = (
            "liked" if feedback_type == LIKE_BLOCK_ACTION_ID else "disliked"
        )

        if get_feedback_visibility() == FeedbackVisibility.ANONYMOUS:
            msg = f"A user has {feedback_response_txt} the AI Answer"
        else:
            msg = f"<@{user_id_to_post_confirmation}> has {feedback_response_txt} the AI Answer"

        respond_in_thread(
            client=client,
            channel=channel_id_to_post_confirmation,
            text=msg,
            thread_ts=thread_ts_to_post_confirmation,
            unfurl=False,
        )


def handle_followup_button(
    req: SocketModeRequest,
    client: TenantSocketModeClient,
) -> None:
    """
    处理后续跟进按钮的点击事件
    
    参数:
        req: Slack的Socket模式请求对象
        client: 租户Socket模式客户端对象
    
    功能:
    - 添加跟进表情反应
    - 获取相关标签和用户组
    - 发送跟进确认消息
    """
    action_id = None
    if actions := req.payload.get("actions"):
        action = cast(dict[str, Any], actions[0])
        action_id = cast(str, action.get("block_id"))

    channel_id = req.payload["container"]["channel_id"]
    thread_ts = req.payload["container"]["thread_ts"]

    update_emote_react(
        emoji=DANSWER_FOLLOWUP_EMOJI,
        channel=channel_id,
        message_ts=thread_ts,
        remove=False,
        client=client.web_client,
    )

    tag_ids: list[str] = []
    group_ids: list[str] = []
    with get_session_with_tenant(client.tenant_id) as db_session:
        channel_name, is_dm = get_channel_name_from_id(
            client=client.web_client, channel_id=channel_id
        )
        slack_channel_config = get_slack_channel_config_for_bot_and_channel(
            db_session=db_session,
            slack_bot_id=client.slack_bot_id,
            channel_name=channel_name,
        )
        if slack_channel_config:
            tag_names = slack_channel_config.channel_config.get("follow_up_tags")
            remaining = None
            if tag_names:
                tag_ids, remaining = fetch_slack_user_ids_from_emails(
                    tag_names, client.web_client
                )
            if remaining:
                group_ids, _ = fetch_group_ids_from_names(remaining, client.web_client)

    blocks = build_follow_up_resolved_blocks(tag_ids=tag_ids, group_ids=group_ids)

    respond_in_thread(
        client=client.web_client,
        channel=channel_id,
        text="Received your request for more help",
        blocks=blocks,
        thread_ts=thread_ts,
        unfurl=False,
    )

    if action_id is not None:
        message_id, _, _ = decompose_action_id(action_id)

        create_chat_message_feedback(
            is_positive=None,
            feedback_text="",
            chat_message_id=message_id,
            user_id=None,  # no "user" for Slack bot for now
            db_session=db_session,
            required_followup=True,
        )


def get_clicker_name(
    req: SocketModeRequest,
    client: TenantSocketModeClient,
) -> str:
    """
    获取点击按钮的用户名称
    
    参数:
        req: Slack的Socket模式请求对象
        client: 租户Socket模式客户端对象
    
    返回:
        str: 用户的真实姓名或用户名
    """
    clicker_name = req.payload.get("user", {}).get("name", "Someone")
    clicker_real_name = None
    try:
        clicker = client.web_client.users_info(user=req.payload["user"]["id"])
        clicker_real_name = (
            cast(dict, clicker.data).get("user", {}).get("profile", {}).get("real_name")
        )
    except Exception:
        # Likely a scope issue
        pass

    if clicker_real_name:
        clicker_name = clicker_real_name

    return clicker_name


def handle_followup_resolved_button(
    req: SocketModeRequest,
    client: TenantSocketModeClient,
    immediate: bool = False,
) -> None:
    """
    处理标记问题已解决按钮的点击事件
    
    参数:
        req: Slack的Socket模式请求对象
        client: 租户Socket模式客户端对象
        immediate: 是否立即标记为已解决
    
    功能:
    - 移除跟进表情反应
    - 删除原有的解决选项消息
    - 发送问题已解决的确认消息
    """
    channel_id = req.payload["container"]["channel_id"]
    message_ts = req.payload["container"]["message_ts"]
    thread_ts = req.payload["container"]["thread_ts"]

    clicker_name = get_clicker_name(req, client)

    update_emote_react(
        emoji=DANSWER_FOLLOWUP_EMOJI,
        channel=channel_id,
        message_ts=thread_ts,
        remove=True,
        client=client.web_client,
    )

    # Delete the message with the option to mark resolved
    if not immediate:
        slack_call = make_slack_api_rate_limited(client.web_client.chat_delete)
        response = slack_call(
            channel=channel_id,
            ts=message_ts,
        )

        if not response.get("ok"):
            logger.error("Unable to delete message for resolved")

    if immediate:
        msg_text = f"{clicker_name} has marked this question as resolved!"
    else:
        msg_text = (
            f"{clicker_name} has marked this question as resolved! "
            f'\n\n You can always click the "I need more help button" to let the team '
            f"know that your problem still needs attention."
        )

    resolved_block = SectionBlock(text=msg_text)

    respond_in_thread(
        client=client.web_client,
        channel=channel_id,
        text="Your request for help as been addressed!",
        blocks=[resolved_block],
        thread_ts=thread_ts,
        unfurl=False,
    )
