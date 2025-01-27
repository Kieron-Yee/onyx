"""
此文件用于管理Slack频道的配置信息，包括:
- Slack频道配置的获取和验证
- 多租户处理的配置参数
- Slack过滤器的定义
"""

import os

from sqlalchemy.orm import Session

from onyx.db.models import SlackChannelConfig
from onyx.db.slack_channel_config import fetch_slack_channel_configs


# Slack有效过滤器列表
VALID_SLACK_FILTERS = [
    "answerable_prefilter",      # 可回答预过滤器
    "well_answered_postfilter",  # 良好回答后过滤器
    "questionmark_prefilter",    # 问号预过滤器
]


def get_slack_channel_config_for_bot_and_channel(
    db_session: Session,
    slack_bot_id: int,
    channel_name: str | None,
) -> SlackChannelConfig | None:
    """
    获取指定机器人和频道的Slack配置信息

    参数:
        db_session: 数据库会话
        slack_bot_id: Slack机器人ID
        channel_name: 频道名称

    返回:
        SlackChannelConfig对象或None
    """
    if not channel_name:
        return None

    slack_bot_configs = fetch_slack_channel_configs(
        db_session=db_session, slack_bot_id=slack_bot_id
    )
    for config in slack_bot_configs:
        if channel_name in config.channel_config["channel_name"]:
            return config

    return None


def validate_channel_name(
    db_session: Session,
    current_slack_bot_id: int,
    channel_name: str,
    current_slack_channel_config_id: int | None,
) -> str:
    """
    确保此频道名称在其他Slack频道配置中不存在。
    返回清理后的频道名称（例如：删除前缀'#'）

    参数:
        db_session: 数据库会话
        current_slack_bot_id: 当前Slack机器人ID
        channel_name: 频道名称
        current_slack_channel_config_id: 当前Slack频道配置ID

    返回:
        清理后的频道名称

    异常:
        ValueError: 当频道名称已存在时抛出
    """
    slack_bot_configs = fetch_slack_channel_configs(
        db_session=db_session,
        slack_bot_id=current_slack_bot_id,
    )
    cleaned_channel_name = channel_name.lstrip("#").lower()
    for slack_channel_config in slack_bot_configs:
        if slack_channel_config.id == current_slack_channel_config_id:
            continue

        if cleaned_channel_name == slack_channel_config.channel_config["channel_name"]:
            raise ValueError(
                f"Channel name '{channel_name}' already exists in "
                "another Slack channel config with in Slack Bot with name: "
                f"{slack_channel_config.slack_bot.name}"
            )

    return cleaned_channel_name


# Scaling configurations for multi-tenant Slack channel handling
# 多租户Slack频道处理的扩展配置
TENANT_LOCK_EXPIRATION = 1800  # Pod可以独占访问租户的时长（秒）

TENANT_HEARTBEAT_INTERVAL = 15  # Pod发送心跳以表明其仍在处理租户的频率（秒）

TENANT_HEARTBEAT_EXPIRATION = 30  # 租户心跳过期时间，过期后允许其他Pod接管（秒）

TENANT_ACQUISITION_INTERVAL = 60  # Pod尝试获取未处理的租户并检查新令牌的频率（秒）

# 每个Pod最大租户数量
MAX_TENANTS_PER_POD = int(os.getenv("MAX_TENANTS_PER_POD", 50))
