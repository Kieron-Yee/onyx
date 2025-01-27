"""
此模块提供了与Slack机器人相关的数据库操作功能，包括创建、更新、查询和删除Slack机器人实例。
该模块主要处理SlackBot模型的数据库交互操作。
"""

from collections.abc import Sequence
from sqlalchemy import select
from sqlalchemy.orm import Session
from onyx.db.models import SlackBot


def insert_slack_bot(
    db_session: Session,
    name: str,
    enabled: bool,
    bot_token: str,
    app_token: str,
) -> SlackBot:
    """
    创建并插入新的Slack机器人实例到数据库中。
    
    Args:
        db_session: 数据库会话对象
        name: Slack机器人名称
        enabled: 是否启用机器人
        bot_token: Slack机器人令牌
        app_token: Slack应用令牌
    
    Returns:
        新创建的SlackBot实例
    """
    slack_bot = SlackBot(
        name=name,
        enabled=enabled,
        bot_token=bot_token,
        app_token=app_token,
    )
    db_session.add(slack_bot)
    db_session.commit()

    return slack_bot


def update_slack_bot(
    db_session: Session,
    slack_bot_id: int,
    name: str,
    enabled: bool,
    bot_token: str,
    app_token: str,
) -> SlackBot:
    """
    更新现有Slack机器人的信息。
    
    Args:
        db_session: 数据库会话对象
        slack_bot_id: Slack机器人ID
        name: 新的机器人名称
        enabled: 新的启用状态
        bot_token: 新的机器人令牌
        app_token: 新的应用令牌
    
    Returns:
        更新后的SlackBot实例
    
    Raises:
        ValueError: 当指定ID的机器人不存在时抛出
    """
    slack_bot = db_session.scalar(select(SlackBot).where(SlackBot.id == slack_bot_id))
    if slack_bot is None:
        raise ValueError(f"无法找到ID为 {slack_bot_id} 的Slack机器人") # Unable to find Slack Bot with ID {slack_bot_id}

    # 更新应用信息 # update the app
    slack_bot.name = name
    slack_bot.enabled = enabled
    slack_bot.bot_token = bot_token
    slack_bot.app_token = app_token

    db_session.commit()

    return slack_bot


def fetch_slack_bot(
    db_session: Session,
    slack_bot_id: int,
) -> SlackBot:
    """
    通过ID获取特定的Slack机器人实例。
    
    Args:
        db_session: 数据库会话对象
        slack_bot_id: Slack机器人ID
    
    Returns:
        查找到的SlackBot实例
    
    Raises:
        ValueError: 当指定ID的机器人不存在时抛出
    """
    slack_bot = db_session.scalar(select(SlackBot).where(SlackBot.id == slack_bot_id))
    if slack_bot is None:
        raise ValueError(f"无法找到ID为 {slack_bot_id} 的Slack机器人") # Unable to find Slack Bot with ID {slack_bot_id}

    return slack_bot


def remove_slack_bot(
    db_session: Session,
    slack_bot_id: int,
) -> None:
    """
    从数据库中删除指定的Slack机器人实例。
    
    Args:
        db_session: 数据库会话对象
        slack_bot_id: 要删除的Slack机器人ID
    
    Raises:
        ValueError: 当指定ID的机器人不存在时抛出
    """
    slack_bot = fetch_slack_bot(
        db_session=db_session,
        slack_bot_id=slack_bot_id,
    )

    db_session.delete(slack_bot)
    db_session.commit()


def fetch_slack_bots(db_session: Session) -> Sequence[SlackBot]:
    """
    获取所有Slack机器人实例列表。
    
    Args:
        db_session: 数据库会话对象
    
    Returns:
        SlackBot实例序列
    """
    return db_session.scalars(select(SlackBot)).all()
