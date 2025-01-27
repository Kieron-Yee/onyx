"""
此文件实现了Slack机器人的监听器功能,是Slack机器人系统的核心组件。

主要功能:
1. 管理多租户的Slack WebSocket连接
2. 处理来自Slack的各种事件和消息
3. 实现租户的自动获取和释放
4. 维护租户连接的心跳机制
5. 处理用户反馈和交互
6. 实现优雅的启动和关闭机制

技术特点:
- 使用WebSocket保持与Slack的实时通信
- 支持多租户架构
- 实现了完整的错误处理和日志记录
- 集成了Prometheus监控指标
- 使用Redis实现分布式锁和心跳机制
"""

import asyncio
import os
import signal
import sys
import threading
import time
from collections.abc import Callable
from threading import Event
from types import FrameType
from typing import Any
from typing import cast
from typing import Dict
from typing import Set

from prometheus_client import Gauge
from prometheus_client import start_http_server
from slack_sdk import WebClient
from slack_sdk.socket_mode.request import SocketModeRequest
from slack_sdk.socket_mode.response import SocketModeResponse
from sqlalchemy.orm import Session

from onyx.chat.models import ThreadMessage
from onyx.configs.app_configs import DEV_MODE
from onyx.configs.app_configs import POD_NAME
from onyx.configs.app_configs import POD_NAMESPACE
from onyx.configs.constants import MessageType
from onyx.configs.constants import OnyxRedisLocks
from onyx.configs.onyxbot_configs import DANSWER_BOT_REPHRASE_MESSAGE
from onyx.configs.onyxbot_configs import DANSWER_BOT_RESPOND_EVERY_CHANNEL
from onyx.configs.onyxbot_configs import NOTIFY_SLACKBOT_NO_ANSWER
from onyx.connectors.slack.utils import expert_info_from_slack_id
from onyx.context.search.retrieval.search_runner import download_nltk_data
from onyx.db.engine import get_all_tenant_ids
from onyx.db.engine import get_session_with_tenant
from onyx.db.models import SlackBot
from onyx.db.search_settings import get_current_search_settings
from onyx.db.slack_bot import fetch_slack_bots
from onyx.key_value_store.interface import KvKeyNotFoundError
from onyx.natural_language_processing.search_nlp_models import EmbeddingModel
from onyx.natural_language_processing.search_nlp_models import warm_up_bi_encoder
from onyx.onyxbot.slack.config import get_slack_channel_config_for_bot_and_channel
from onyx.onyxbot.slack.config import MAX_TENANTS_PER_POD
from onyx.onyxbot.slack.config import TENANT_ACQUISITION_INTERVAL
from onyx.onyxbot.slack.config import TENANT_HEARTBEAT_EXPIRATION
from onyx.onyxbot.slack.config import TENANT_HEARTBEAT_INTERVAL
from onyx.onyxbot.slack.config import TENANT_LOCK_EXPIRATION
from onyx.onyxbot.slack.constants import DISLIKE_BLOCK_ACTION_ID
from onyx.onyxbot.slack.constants import FEEDBACK_DOC_BUTTON_BLOCK_ACTION_ID
from onyx.onyxbot.slack.constants import FOLLOWUP_BUTTON_ACTION_ID
from onyx.onyxbot.slack.constants import FOLLOWUP_BUTTON_RESOLVED_ACTION_ID
from onyx.onyxbot.slack.constants import GENERATE_ANSWER_BUTTON_ACTION_ID
from onyx.onyxbot.slack.constants import IMMEDIATE_RESOLVED_BUTTON_ACTION_ID
from onyx.onyxbot.slack.constants import LIKE_BLOCK_ACTION_ID
from onyx.onyxbot.slack.constants import VIEW_DOC_FEEDBACK_ID
from onyx.onyxbot.slack.handlers.handle_buttons import handle_doc_feedback_button
from onyx.onyxbot.slack.handlers.handle_buttons import handle_followup_button
from onyx.onyxbot.slack.handlers.handle_buttons import (
    handle_followup_resolved_button,
)
from onyx.onyxbot.slack.handlers.handle_buttons import (
    handle_generate_answer_button,
)
from onyx.onyxbot.slack.handlers.handle_buttons import handle_slack_feedback
from onyx.onyxbot.slack.handlers.handle_message import handle_message
from onyx.onyxbot.slack.handlers.handle_message import (
    remove_scheduled_feedback_reminder,
)
from onyx.onyxbot.slack.handlers.handle_message import schedule_feedback_reminder
from onyx.onyxbot.slack.models import SlackMessageInfo
from onyx.onyxbot.slack.utils import check_message_limit
from onyx.onyxbot.slack.utils import decompose_action_id
from onyx.onyxbot.slack.utils import get_channel_name_from_id
from onyx.onyxbot.slack.utils import get_onyx_bot_slack_bot_id
from onyx.onyxbot.slack.utils import read_slack_thread
from onyx.onyxbot.slack.utils import remove_onyx_bot_tag
from onyx.onyxbot.slack.utils import rephrase_slack_message
from onyx.onyxbot.slack.utils import respond_in_thread
from onyx.onyxbot.slack.utils import TenantSocketModeClient
from onyx.redis.redis_pool import get_redis_client
from onyx.server.manage.models import SlackBotTokens
from onyx.utils.logger import setup_logger
from onyx.utils.variable_functionality import set_is_ee_based_on_env_variable
from shared_configs.configs import DISALLOWED_SLACK_BOT_TENANT_LIST
from shared_configs.configs import MODEL_SERVER_HOST
from shared_configs.configs import MODEL_SERVER_PORT
from shared_configs.configs import POSTGRES_DEFAULT_SCHEMA
from shared_configs.configs import SLACK_CHANNEL_ID
from shared_configs.contextvars import CURRENT_TENANT_ID_CONTEXTVAR


logger = setup_logger()

# Prometheus metric for HPA
active_tenants_gauge = Gauge(
    "active_tenants",
    "Number of active tenants handled by this pod",
    ["namespace", "pod"],
)

# In rare cases, some users have been experiencing a massive amount of trivial messages coming through
# to the Slack Bot with trivial messages. Adding this to avoid exploding LLM costs while we track down
# the cause.
# 在极少数情况下,一些用户会收到大量琐碎的消息发送到Slack机器人。添加此项以避免在追踪原因时导致LLM成本激增。
_SLACK_GREETINGS_TO_IGNORE = {
    "Welcome back!",
    "It's going to be a great day.",
    "Salutations!",
    "Greetings!",
    "Feeling great!",
    "Hi there",
    ":wave:",
}

# This is always (currently) the user id of Slack's official slackbot
# 这是Slack官方机器人的用户ID(当前)
_OFFICIAL_SLACKBOT_USER_ID = "USLACKBOT"


class SlackbotHandler:
    """
    Slack机器人处理器类,负责管理多租户的Slack连接。
    主要功能:
    - 初始化和维护与Slack的WebSocket连接
    - 管理租户的获取和心跳
    - 处理优雅关闭
    """

    def __init__(self) -> None:
        """
        初始化SlackbotHandler实例。
        设置基本属性、启动Prometheus指标服务器、注册信号处理器并启动后台线程。
        """
        logger.info("Initializing SlackbotHandler")
        self.tenant_ids: Set[str | None] = set()
        # The keys for these dictionaries are tuples of (tenant_id, slack_bot_id)
        # 这些字典的键是(tenant_id, slack_bot_id)元组
        self.socket_clients: Dict[tuple[str | None, int], TenantSocketModeClient] = {}
        self.slack_bot_tokens: Dict[tuple[str | None, int], SlackBotTokens] = {}

        self.running = True
        self.pod_id = self.get_pod_id()
        self._shutdown_event = Event()
        logger.info(f"Pod ID: {self.pod_id}")

        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self.shutdown)
        signal.signal(signal.SIGINT, self.shutdown)
        logger.info("Signal handlers registered")

        # Start the Prometheus metrics server
        logger.info("Starting Prometheus metrics server")
        start_http_server(8000)
        logger.info("Prometheus metrics server started")

        # Start background threads
        logger.info("Starting background threads")
        self.acquire_thread = threading.Thread(
            target=self.acquire_tenants_loop, daemon=True
        )
        self.heartbeat_thread = threading.Thread(
            target=self.heartbeat_loop, daemon=True
        )

        self.acquire_thread.start()
        self.heartbeat_thread.start()
        logger.info("Background threads started")

    def get_pod_id(self) -> str:
        """
        获取Pod的ID。
        
        返回:
            str: Pod的ID,如果未设置则返回'unknown_pod'
        """
        pod_id = os.environ.get("HOSTNAME", "unknown_pod")
        logger.info(f"Retrieved pod ID: {pod_id}")
        return pod_id

    def acquire_tenants_loop(self) -> None:
        """
        循环获取租户的后台线程。
        定期尝试获取新租户并更新活跃租户数量指标。
        """
        while not self._shutdown_event.is_set():
            try:
                self.acquire_tenants()
                active_tenants_gauge.labels(namespace=POD_NAMESPACE, pod=POD_NAME).set(
                    len(self.tenant_ids)
                )
                logger.debug(f"Current active tenants: {len(self.tenant_ids)}")
            except Exception as e:
                logger.exception(f"Error in Slack acquisition: {e}")
            self._shutdown_event.wait(timeout=TENANT_ACQUISITION_INTERVAL)

    def heartbeat_loop(self) -> None:
        """
        心跳循环的后台线程。
        定期为所有活跃租户发送心跳信号以保持连接。
        """
        while not self._shutdown_event.is_set():
            try:
                self.send_heartbeats()
                logger.debug(f"Sent heartbeats for {len(self.tenant_ids)} tenants")
            except Exception as e:
                logger.exception(f"Error in heartbeat loop: {e}")
            self._shutdown_event.wait(timeout=TENANT_HEARTBEAT_INTERVAL)

    def _manage_clients_per_tenant(
        self, db_session: Session, tenant_id: str | None, bot: SlackBot
    ) -> None:
        """
        管理每个租户的客户端连接。
        
        参数:
            db_session: 数据库会话
            tenant_id: 租户ID 
            bot: Slack机器人实例
        """
        slack_bot_tokens = SlackBotTokens(
            bot_token=bot.bot_token,
            app_token=bot.app_token,
        )
        tenant_bot_pair = (tenant_id, bot.id)

        # If the tokens are not set, we need to close the socket client and delete the tokens
        # for the tenant and app
        if not slack_bot_tokens:
            logger.debug(
                f"No Slack bot token found for tenant {tenant_id}, bot {bot.id}"
            )
            if tenant_bot_pair in self.socket_clients:
                asyncio.run(self.socket_clients[tenant_bot_pair].close())
                del self.socket_clients[tenant_bot_pair]
                del self.slack_bot_tokens[tenant_bot_pair]
            return

        tokens_exist = tenant_bot_pair in self.slack_bot_tokens
        tokens_changed = (
            tokens_exist and slack_bot_tokens != self.slack_bot_tokens[tenant_bot_pair]
        )
        if not tokens_exist or tokens_changed:
            if tokens_exist:
                logger.info(
                    f"Slack Bot tokens have changed for tenant {tenant_id}, bot {bot.id} - reconnecting"
                )
            else:
                search_settings = get_current_search_settings(db_session)
                embedding_model = EmbeddingModel.from_db_model(
                    search_settings=search_settings,
                    server_host=MODEL_SERVER_HOST,
                    server_port=MODEL_SERVER_PORT,
                )
                warm_up_bi_encoder(embedding_model=embedding_model)

            self.slack_bot_tokens[tenant_bot_pair] = slack_bot_tokens

            if tenant_bot_pair in self.socket_clients:
                asyncio.run(self.socket_clients[tenant_bot_pair].close())

            self.start_socket_client(bot.id, tenant_id, slack_bot_tokens)

    def acquire_tenants(self) -> None:
        """
        尝试获取新的租户并建立连接。
        处理租户的锁定机制并为每个获取的租户创建socket客户端。
        """
        tenant_ids = get_all_tenant_ids()

        for tenant_id in tenant_ids:
            if (
                DISALLOWED_SLACK_BOT_TENANT_LIST is not None
                and tenant_id in DISALLOWED_SLACK_BOT_TENANT_LIST
            ):
                logger.debug(f"Tenant {tenant_id} is in the disallowed list, skipping")
                continue

            if tenant_id in self.tenant_ids:
                logger.debug(f"Tenant {tenant_id} already in self.tenant_ids")
                continue

            if len(self.tenant_ids) >= MAX_TENANTS_PER_POD:
                logger.info(
                    f"Max tenants per pod reached ({MAX_TENANTS_PER_POD}) Not acquiring any more tenants"
                )
                break

            redis_client = get_redis_client(tenant_id=tenant_id)
            pod_id = self.pod_id
            acquired = redis_client.set(
                OnyxRedisLocks.SLACK_BOT_LOCK,
                pod_id,
                nx=True,
                ex=TENANT_LOCK_EXPIRATION,
            )
            if not acquired and not DEV_MODE:
                logger.debug(f"Another pod holds the lock for tenant {tenant_id}")
                continue

            logger.debug(f"Acquired lock for tenant {tenant_id}")

            self.tenant_ids.add(tenant_id)

        for tenant_id in self.tenant_ids:
            token = CURRENT_TENANT_ID_CONTEXTVAR.set(
                tenant_id or POSTGRES_DEFAULT_SCHEMA
            )
            try:
                with get_session_with_tenant(tenant_id) as db_session:
                    try:
                        bots = fetch_slack_bots(db_session=db_session)
                        for bot in bots:
                            self._manage_clients_per_tenant(
                                db_session=db_session,
                                tenant_id=tenant_id,
                                bot=bot,
                            )

                    except KvKeyNotFoundError:
                        logger.debug(f"Missing Slack Bot tokens for tenant {tenant_id}")
                        if (tenant_id, bot.id) in self.socket_clients:
                            asyncio.run(self.socket_clients[tenant_id, bot.id].close())
                            del self.socket_clients[tenant_id, bot.id]
                            del self.slack_bot_tokens[tenant_id, bot.id]
                    except Exception as e:
                        logger.exception(f"Error handling tenant {tenant_id}: {e}")
            finally:
                CURRENT_TENANT_ID_CONTEXTVAR.reset(token)

    def send_heartbeats(self) -> None:
        """
        为所有活跃租户发送心跳信号。
        更新Redis中的心跳时间戳。
        """
        current_time = int(time.time())
        logger.debug(f"Sending heartbeats for {len(self.tenant_ids)} tenants")
        for tenant_id in self.tenant_ids:
            redis_client = get_redis_client(tenant_id=tenant_id)
            heartbeat_key = f"{OnyxRedisLocks.SLACK_BOT_HEARTBEAT_PREFIX}:{self.pod_id}"
            redis_client.set(
                heartbeat_key, current_time, ex=TENANT_HEARTBEAT_EXPIRATION
            )

    def start_socket_client(
        self, slack_bot_id: int, tenant_id: str | None, slack_bot_tokens: SlackBotTokens
    ) -> None:
        """
        为指定的租户启动socket客户端。
        
        参数:
            slack_bot_id: Slack机器人ID
            tenant_id: 租户ID
            slack_bot_tokens: Slack机器人的token信息
        """
        logger.info(
            f"Starting socket client for tenant: {tenant_id}, app: {slack_bot_id}"
        )
        socket_client: TenantSocketModeClient = _get_socket_client(
            slack_bot_tokens, tenant_id, slack_bot_id
        )

        # Append the event handler
        process_slack_event = create_process_slack_event()
        socket_client.socket_mode_request_listeners.append(process_slack_event)  # type: ignore

        # Establish a WebSocket connection to the Socket Mode servers
        logger.info(
            f"Connecting socket client for tenant: {tenant_id}, app: {slack_bot_id}"
        )
        socket_client.connect()
        self.socket_clients[tenant_id, slack_bot_id] = socket_client
        self.tenant_ids.add(tenant_id)
        logger.info(
            f"Started SocketModeClient for tenant: {tenant_id}, app: {slack_bot_id}"
        )

    def stop_socket_clients(self) -> None:
        """
        停止所有活跃的socket客户端连接。
        在关闭时调用以清理资源。
        """
        logger.info(f"Stopping {len(self.socket_clients)} socket clients")
        for (tenant_id, slack_bot_id), client in self.socket_clients.items():
            asyncio.run(client.close())
            logger.info(
                f"Stopped SocketModeClient for tenant: {tenant_id}, app: {slack_bot_id}"
            )

    def shutdown(self, signum: int | None, frame: FrameType | None) -> None:
        """
        优雅关闭处理器。
        清理所有连接、释放锁并停止后台线程。
        
        参数:
            signum: 信号编号
            frame: 当前执行帧
        """
        if not self.running:
            return

        logger.info("Shutting down gracefully")
        self.running = False
        self._shutdown_event.set()

        # Stop all socket clients
        logger.info(f"Stopping {len(self.socket_clients)} socket clients")
        self.stop_socket_clients()

        # Release locks for all tenants
        logger.info(f"Releasing locks for {len(self.tenant_ids)} tenants")
        for tenant_id in self.tenant_ids:
            try:
                redis_client = get_redis_client(tenant_id=tenant_id)
                redis_client.delete(OnyxRedisLocks.SLACK_BOT_LOCK)
                logger.info(f"Released lock for tenant {tenant_id}")
            except Exception as e:
                logger.error(f"Error releasing lock for tenant {tenant_id}: {e}")

        # Wait for background threads to finish (with timeout)
        logger.info("Waiting for background threads to finish...")
        self.acquire_thread.join(timeout=5)
        self.heartbeat_thread.join(timeout=5)

        logger.info("Shutdown complete")
        sys.exit(0)


def prefilter_requests(req: SocketModeRequest, client: TenantSocketModeClient) -> bool:
    """
    预过滤Slack请求,决定是否需要处理该请求。
    
    参数:
        req: Slack socket模式请求
        client: 租户socket客户端
    
    返回:
        bool: True表示继续处理,False表示忽略该请求
    """
    if req.type == "events_api":
        # Verify channel is valid
        # 验证频道是否有效
        event = cast(dict[str, Any], req.payload.get("event", {}))
        msg = cast(str | None, event.get("text"))
        channel = cast(str | None, event.get("channel"))
        channel_specific_logger = setup_logger(extra={SLACK_CHANNEL_ID: channel})

        # This should never happen, but we can't continue without a channel since
        # we can't send a response without it
        # 这种情况永远不应该发生,但如果没有频道我们就无法继续,因为无法发送响应
        if not channel:
            channel_specific_logger.warning("Found message without channel - skipping")
            return False

        if not msg:
            channel_specific_logger.warning(
                "Cannot respond to empty message - skipping"
            )
            return False

        if (
            req.payload.setdefault("event", {}).get("user", "")
            == _OFFICIAL_SLACKBOT_USER_ID
        ):
            channel_specific_logger.info(
                "Ignoring messages from Slack's official Slackbot"
            )
            return False

        if (
            msg in _SLACK_GREETINGS_TO_IGNORE
            or remove_onyx_bot_tag(msg, client=client.web_client)
            in _SLACK_GREETINGS_TO_IGNORE
        ):
            channel_specific_logger.error(
                f"Ignoring weird Slack greeting message: '{msg}'"
            )
            channel_specific_logger.error(
                f"Weird Slack greeting message payload: '{req.payload}'"
            )
            return False

        # Ensure that the message is a new message of expected type
        # 确保消息是预期类型的新消息
        event_type = event.get("type")
        if event_type not in ["app_mention", "message"]:
            channel_specific_logger.info(
                f"Ignoring non-message event of type '{event_type}' for channel '{channel}'"
            )
            return False

        bot_tag_id = get_onyx_bot_slack_bot_id(client.web_client)
        if event_type == "message":
            is_dm = event.get("channel_type") == "im"
            is_tagged = bot_tag_id and bot_tag_id in msg
            is_onyx_bot_msg = bot_tag_id and bot_tag_id in event.get("user", "")

            # OnyxBot should never respond to itself
            # OnyxBot永远不应该回复自己的消息
            if is_onyx_bot_msg:
                logger.info("Ignoring message from OnyxBot")
                return False

            # DMs with the bot don't pick up the @OnyxBot so we have to keep the caught events_api
            # 与机器人的私信不会捕获@OnyxBot标记,所以我们必须保留捕获的events_api
            if is_tagged and not is_dm:
                # Let the tag flow handle this case, don't reply twice
                # 让标记流程处理这种情况,不要重复回复
                return False

        if event.get("bot_profile"):
            channel_name, _ = get_channel_name_from_id(
                client=client.web_client, channel_id=channel
            )

            with get_session_with_tenant(client.tenant_id) as db_session:
                slack_channel_config = get_slack_channel_config_for_bot_and_channel(
                    db_session=db_session,
                    slack_bot_id=client.slack_bot_id,
                    channel_name=channel_name,
                )
            # If OnyxBot is not specifically tagged and the channel is not set to respond to bots, ignore the message
            # 如果OnyxBot没有被特别标记且频道未设置回复机器人,则忽略该消息
            if (not bot_tag_id or bot_tag_id not in msg) and (
                not slack_channel_config
                or not slack_channel_config.channel_config.get("respond_to_bots")
            ):
                channel_specific_logger.info("Ignoring message from bot")
                return False

        # Ignore things like channel_join, channel_leave, etc.
        # NOTE: "file_share" is just a message with a file attachment, so we
        # should not ignore it
        # 忽略channel_join、channel_leave等消息
        # 注意："file_share"只是带有文件附件的消息,所以不应该忽略
        message_subtype = event.get("subtype")
        if message_subtype not in [None, "file_share"]:
            channel_specific_logger.info(
                f"Ignoring message with subtype '{message_subtype}' since it is a special message type"
            )
            return False

        message_ts = event.get("ts")
        thread_ts = event.get("thread_ts")
        # Pick the root of the thread (if a thread exists)
        # Can respond in thread if it's an "im" directly to Onyx or @OnyxBot is tagged
        # 选择线程的根消息(如果线程存在)
        # 如果是直接发给Onyx的私信或@OnyxBot被标记,可以在线程中回复
        if (
            thread_ts
            and message_ts != thread_ts
            and event_type != "app_mention"
            and event.get("channel_type") != "im"
        ):
            channel_specific_logger.debug(
                "Skipping message since it is not the root of a thread"
            )
            return False

        msg = cast(str, event.get("text", ""))
        if not msg:
            channel_specific_logger.error("Unable to process empty message")
            return False

    if req.type == "slash_commands":
        # Verify that there's an associated channel
        # 验证是否有相关联的频道
        channel = req.payload.get("channel_id")
        channel_specific_logger = setup_logger(extra={SLACK_CHANNEL_ID: channel})

        if not channel:
            channel_specific_logger.error(
                "Received OnyxBot command without channel - skipping"
            )
            return False

        sender = req.payload.get("user_id")
        if not sender:
            channel_specific_logger.error(
                "Cannot respond to OnyxBot command without sender to respond to."
            )
            return False

    if not check_message_limit():
        return False

    logger.debug(f"Handling Slack request with Payload: '{req.payload}'")
    return True


def process_feedback(req: SocketModeRequest, client: TenantSocketModeClient) -> None:
    """
    处理Slack反馈请求。
    
    参数:
        req: Slack socket模式请求
        client: 租户socket客户端
    """
    if actions := req.payload.get("actions"):
        action = cast(dict[str, Any], actions[0])
        feedback_type = cast(str, action.get("action_id"))
        feedback_msg_reminder = cast(str, action.get("value"))
        feedback_id = cast(str, action.get("block_id"))
        channel_id = cast(str, req.payload["container"]["channel_id"])
        thread_ts = cast(str, req.payload["container"]["thread_ts"])
    else:
        logger.error("Unable to process feedback. Action not found")
        return

    user_id = cast(str, req.payload["user"]["id"])

    handle_slack_feedback(
        feedback_id=feedback_id,
        feedback_type=feedback_type,
        feedback_msg_reminder=feedback_msg_reminder,
        client=client.web_client,
        user_id_to_post_confirmation=user_id,
        channel_id_to_post_confirmation=channel_id,
        thread_ts_to_post_confirmation=thread_ts,
        tenant_id=client.tenant_id,
    )

    query_event_id, _, _ = decompose_action_id(feedback_id)
    logger.info(f"Successfully handled QA feedback for event: {query_event_id}")


def build_request_details(
    req: SocketModeRequest, client: TenantSocketModeClient
) -> SlackMessageInfo:
    """
    构建Slack请求的详细信息。
    
    参数:
        req: Slack socket模式请求
        client: 租户socket客户端
    
    返回:
        SlackMessageInfo: 包含消息详细信息的对象
        
    异常:
        RuntimeError: 当请求类型无效时抛出
    """
    if req.type == "events_api":
        event = cast(dict[str, Any], req.payload["event"])
        msg = cast(str, event["text"])
        channel = cast(str, event["channel"])
        tagged = event.get("type") == "app_mention"
        message_ts = event.get("ts")
        thread_ts = event.get("thread_ts")
        sender = event.get("user") or None
        expert_info = expert_info_from_slack_id(
            sender, client.web_client, user_cache={}
        )
        email = expert_info.email if expert_info else None

        msg = remove_onyx_bot_tag(msg, client=client.web_client)

        if DANSWER_BOT_REPHRASE_MESSAGE:
            logger.info(f"Rephrasing Slack message. Original message: {msg}")
            try:
                msg = rephrase_slack_message(msg)
                logger.info(f"Rephrased message: {msg}")
            except Exception as e:
                logger.error(f"Error while trying to rephrase the Slack message: {e}")
        else:
            logger.info(f"Received Slack message: {msg}")

        if tagged:
            logger.debug("User tagged OnyxBot")

        if thread_ts != message_ts and thread_ts is not None:
            thread_messages = read_slack_thread(
                channel=channel, thread=thread_ts, client=client.web_client
            )
        else:
            thread_messages = [
                ThreadMessage(message=msg, sender=None, role=MessageType.USER)
            ]

        return SlackMessageInfo(
            thread_messages=thread_messages,
            channel_to_respond=channel,
            msg_to_respond=cast(str, message_ts or thread_ts),
            thread_to_respond=cast(str, thread_ts or message_ts),
            sender=sender,
            email=email,
            bypass_filters=tagged,
            is_bot_msg=False,
            is_bot_dm=event.get("channel_type") == "im",
        )

    elif req.type == "slash_commands":
        channel = req.payload["channel_id"]
        msg = req.payload["text"]
        sender = req.payload["user_id"]
        expert_info = expert_info_from_slack_id(
            sender, client.web_client, user_cache={}
        )
        email = expert_info.email if expert_info else None

        single_msg = ThreadMessage(message=msg, sender=None, role=MessageType.USER)

        return SlackMessageInfo(
            thread_messages=[single_msg],
            channel_to_respond=channel,
            msg_to_respond=None,
            thread_to_respond=None,
            sender=sender,
            email=email,
            bypass_filters=True,
            is_bot_msg=True,
            is_bot_dm=False,
        )

    raise RuntimeError("Programming fault, this should never happen.")


def apologize_for_fail(
    details: SlackMessageInfo,
    client: TenantSocketModeClient,
) -> None:
    """
    当无法找到相关回答时发送道歉消息。
    
    参数:
        details: 消息详细信息
        client: 租户socket客户端
    """
    respond_in_thread(
        client=client.web_client,
        channel=details.channel_to_respond,
        thread_ts=details.msg_to_respond,
        text="Sorry, we weren't able to find anything relevant :cold_sweat:",
    )


def process_message(
    req: SocketModeRequest,
    client: TenantSocketModeClient,
    respond_every_channel: bool = DANSWER_BOT_RESPOND_EVERY_CHANNEL,
    notify_no_answer: bool = NOTIFY_SLACKBOT_NO_ANSWER,
) -> None:
    """
    处理Slack消息请求。
    
    参数:
        req: Slack socket模式请求
        client: 租户socket客户端
        respond_every_channel: 是否响应所有频道
        notify_no_answer: 无答案时是否通知
    """
    logger.debug(
        f"Received Slack request of type: '{req.type}' for tenant, {client.tenant_id}"
    )

    # Throw out requests that can't or shouldn't be handled
    if not prefilter_requests(req, client):
        return

    details = build_request_details(req, client)
    channel = details.channel_to_respond
    channel_name, is_dm = get_channel_name_from_id(
        client=client.web_client, channel_id=channel
    )

    # Set the current tenant ID at the beginning for all DB calls within this thread
    if client.tenant_id:
        logger.info(f"Setting tenant ID to {client.tenant_id}")
        token = CURRENT_TENANT_ID_CONTEXTVAR.set(client.tenant_id)
    try:
        with get_session_with_tenant(client.tenant_id) as db_session:
            slack_channel_config = get_slack_channel_config_for_bot_and_channel(
                db_session=db_session,
                slack_bot_id=client.slack_bot_id,
                channel_name=channel_name,
            )

            # Be careful about this default, don't want to accidentally spam every channel
            # 请谨慎对待这个默认设置,不要意外地向所有频道发送垃圾消息
            # 不过用户应该能够在他们的私人频道中与slack机器人进行私信
            if (
                slack_channel_config is None
                and not respond_every_channel
                # Can't have configs for DMs so don't toss them out
                # 私信无法设置配置,所以不要排除它们
                and not is_dm
                # If /OnyxBot (is_bot_msg) or @OnyxBot (bypass_filters)
                # always respond with the default configs
                # 如果是/OnyxBot(is_bot_msg)或@OnyxBot(bypass_filters)
                # 总是使用默认配置进行响应
                and not (details.is_bot_msg or details.bypass_filters)
            ):
                return

            follow_up = bool(
                slack_channel_config
                and slack_channel_config.channel_config
                and slack_channel_config.channel_config.get("follow_up_tags")
                is not None
            )
            feedback_reminder_id = schedule_feedback_reminder(
                details=details, client=client.web_client, include_followup=follow_up
            )

            failed = handle_message(
                message_info=details,
                slack_channel_config=slack_channel_config,
                client=client.web_client,
                feedback_reminder_id=feedback_reminder_id,
                tenant_id=client.tenant_id,
            )

            if failed:
                if feedback_reminder_id:
                    remove_scheduled_feedback_reminder(
                        client=client.web_client,
                        channel=details.sender,
                        msg_id=feedback_reminder_id,
                    )
                # Skipping answering due to pre-filtering is not considered a failure
                if notify_no_answer:
                    apologize_for_fail(details, client)
    finally:
        if client.tenant_id:
            CURRENT_TENANT_ID_CONTEXTVAR.reset(token)


def acknowledge_message(req: SocketModeRequest, client: TenantSocketModeClient) -> None:
    """
    确认收到Slack消息。
    
    参数:
        req: Slack socket模式请求
        client: 租户socket客户端
    """
    response = SocketModeResponse(envelope_id=req.envelope_id)
    client.send_socket_mode_response(response)


def action_routing(req: SocketModeRequest, client: TenantSocketModeClient) -> None:
    """
    路由处理Slack交互动作。
    
    参数:
        req: Slack socket模式请求
        client: 租户socket客户端
    """
    if actions := req.payload.get("actions"):
        action = cast(dict[str, Any], actions[0])

        if action["action_id"] in [DISLIKE_BLOCK_ACTION_ID, LIKE_BLOCK_ACTION_ID]:
            # AI Answer feedback
            # AI回答反馈
            return process_feedback(req, client)
        elif action["action_id"] == FEEDBACK_DOC_BUTTON_BLOCK_ACTION_ID:
            # Activation of the "source feedback" button
            # 激活"源反馈"按钮
            return handle_doc_feedback_button(req, client)
        elif action["action_id"] == FOLLOWUP_BUTTON_ACTION_ID:
            return handle_followup_button(req, client)
        elif action["action_id"] == IMMEDIATE_RESOLVED_BUTTON_ACTION_ID:
            return handle_followup_resolved_button(req, client, immediate=True)
        elif action["action_id"] == FOLLOWUP_BUTTON_RESOLVED_ACTION_ID:
            return handle_followup_resolved_button(req, client, immediate=False)
        elif action["action_id"] == GENERATE_ANSWER_BUTTON_ACTION_ID:
            return handle_generate_answer_button(req, client)


def view_routing(req: SocketModeRequest, client: TenantSocketModeClient) -> None:
    """
    路由处理Slack视图提交。
    
    参数:
        req: Slack socket模式请求
        client: 租户socket客户端
    """
    if view := req.payload.get("view"):
        if view["callback_id"] == VIEW_DOC_FEEDBACK_ID:
            return process_feedback(req, client)


def create_process_slack_event() -> (
    Callable[[TenantSocketModeClient, SocketModeRequest], None]
):
    """
    创建处理Slack事件的函数。
    
    返回:
        Callable: 处理Slack事件的函数
    """
    
    def process_slack_event(
        client: TenantSocketModeClient, req: SocketModeRequest
    ) -> None:
        """
        处理Slack事件的内部函数。
        
        参数:
            client: 租户socket客户端
            req: Slack socket模式请求
        """
        # Always respond right away, if Slack doesn't receive these frequently enough
        # it will assume the Bot is DEAD!!! :(
        # 始终立即响应,如果Slack没有足够频繁地收到响应
        # 它会认为机器人已经死了!!! :(
        acknowledge_message(req, client)

        try:
            if req.type == "interactive":
                if req.payload.get("type") == "block_actions":
                    return action_routing(req, client)
                elif req.payload.get("type") == "view_submission":
                    return view_routing(req, client)
            elif req.type == "events_api" or req.type == "slash_commands":
                return process_message(req, client)
        except Exception:
            logger.exception("Failed to process slack event")

    return process_slack_event


def _get_socket_client(
    slack_bot_tokens: SlackBotTokens, tenant_id: str | None, slack_bot_id: int
) -> TenantSocketModeClient:
    """
    获取socket客户端实例。
    
    参数:
        slack_bot_tokens: Slack机器人的token信息
        tenant_id: 租户ID
        slack_bot_id: Slack机器人ID
    
    返回:
        TenantSocketModeClient: socket客户端实例
    """
    # For more info on how to set this up, checkout the docs:
    # 关于如何设置的更多信息,请查看文档:
    # https://docs.onyx.app/slack_bot_setup
    return TenantSocketModeClient(
        # This app-level token will be used only for establishing a connection
        # 这个应用级别的token只用于建立连接
        app_token=slack_bot_tokens.app_token,
        web_client=WebClient(token=slack_bot_tokens.bot_token),
        tenant_id=tenant_id,
        slack_bot_id=slack_bot_id,
    )


if __name__ == "__main__":
    """
    主程序入口:
    1. 初始化SlackbotHandler
    2. 设置环境变量
    3. 下载NLTK数据
    4. 保持主线程运行
    """
    # Initialize the tenant handler which will manage tenant connections
    # 初始化租户处理器,用于管理租户连接
    logger.info("Starting SlackbotHandler")
    tenant_handler = SlackbotHandler()

    set_is_ee_based_on_env_variable()

    logger.info("Verifying query preprocessing (NLTK) data is downloaded")
    download_nltk_data()

    try:
        # Keep the main thread alive
        # 保持主线程运行
        while tenant_handler.running:
            time.sleep(1)

    except Exception:
        logger.exception("Fatal error in main thread")
        tenant_handler.shutdown(None, None)
