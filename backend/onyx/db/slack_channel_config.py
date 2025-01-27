"""
此文件主要用于管理Slack频道的配置信息，包括persona设置、标准答案类别、自动过滤等功能的管理。
提供了创建、更新、删除和查询Slack频道配置的相关功能。
"""

from collections.abc import Sequence
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.configs.chat_configs import MAX_CHUNKS_FED_TO_CHAT
from onyx.context.search.enums import RecencyBiasSetting
from onyx.db.constants import SLACK_BOT_PERSONA_PREFIX
from onyx.db.models import ChannelConfig
from onyx.db.models import Persona
from onyx.db.models import Persona__DocumentSet
from onyx.db.models import SlackChannelConfig
from onyx.db.models import User
from onyx.db.persona import get_default_prompt
from onyx.db.persona import mark_persona_as_deleted
from onyx.db.persona import upsert_persona
from onyx.utils.errors import EERequiredError
from onyx.utils.variable_functionality import (
    fetch_versioned_implementation_with_fallback,
)


def _build_persona_name(channel_name: str) -> str:
    """
    根据频道名称构建persona名称
    Args:
        channel_name: Slack频道名称
    Returns:
        构建的persona名称
    """
    return f"{SLACK_BOT_PERSONA_PREFIX}{channel_name}"


def _cleanup_relationships(db_session: Session, persona_id: int) -> None:
    """
    清理persona与document_set之间的关联关系
    
    NOTE: does not commit changes
    注意：不提交更改
    
    Args:
        db_session: 数据库会话
        persona_id: 需要清理关系的persona ID
    """
    # delete existing persona-document_set relationships
    # 删除已存在的persona-document_set关联关系
    existing_relationships = db_session.scalars(
        select(Persona__DocumentSet).where(
            Persona__DocumentSet.persona_id == persona_id
        )
    )
    for rel in existing_relationships:
        db_session.delete(rel)


def create_slack_channel_persona(
    db_session: Session,
    channel_name: str,
    document_set_ids: list[int],
    existing_persona_id: int | None = None,
    num_chunks: float = MAX_CHUNKS_FED_TO_CHAT,
    enable_auto_filters: bool = False,
) -> Persona:
    """
    创建或更新Slack频道关联的persona
    
    NOTE: does not commit changes
    注意：不提交更改
    
    Args:
        db_session: 数据库会话
        channel_name: 频道名称
        document_set_ids: 文档集ID列表
        existing_persona_id: 现有的persona ID（如果是更新操作）
        num_chunks: 提供给聊天的最大块数
        enable_auto_filters: 是否启用自动过滤
    Returns:
        创建或更新的Persona对象
    """
    # 创建/更新与Slack频道关联的persona
    persona_name = _build_persona_name(channel_name)
    default_prompt = get_default_prompt(db_session)
    persona = upsert_persona(
        user=None,  # Slack频道的Personas不与用户关联
        persona_id=existing_persona_id,
        name=persona_name,
        description="",
        num_chunks=num_chunks,
        llm_relevance_filter=True,
        llm_filter_extraction=enable_auto_filters,
        recency_bias=RecencyBiasSetting.AUTO,
        prompt_ids=[default_prompt.id],
        document_set_ids=document_set_ids,
        llm_model_provider_override=None,
        llm_model_version_override=None,
        starter_messages=None,
        is_public=True,
        is_default_persona=False,
        db_session=db_session,
        commit=False,
    )

    return persona


def _no_ee_standard_answer_categories(*args: Any, **kwargs: Any) -> list:
    """
    非企业版的标准答案类别处理函数，始终返回空列表
    """
    return []


def insert_slack_channel_config(
    db_session: Session,
    slack_bot_id: int,
    persona_id: int | None,
    channel_config: ChannelConfig,
    standard_answer_category_ids: list[int],
    enable_auto_filters: bool,
) -> SlackChannelConfig:
    """
    插入新的Slack频道配置
    
    Args:
        db_session: 数据库会话
        slack_bot_id: Slack机器人ID
        persona_id: persona ID
        channel_config: 频道配置
        standard_answer_category_ids: 标准答案类别ID列表
        enable_auto_filters: 是否启用自动过滤
    Returns:
        创建的SlackChannelConfig对象
    """
    versioned_fetch_standard_answer_categories_by_ids = (
        fetch_versioned_implementation_with_fallback(
            "onyx.db.standard_answer",
            "fetch_standard_answer_categories_by_ids",
            _no_ee_standard_answer_categories,
        )
    )
    existing_standard_answer_categories = (
        versioned_fetch_standard_answer_categories_by_ids(
            standard_answer_category_ids=standard_answer_category_ids,
            db_session=db_session,
        )
    )

    if len(existing_standard_answer_categories) != len(standard_answer_category_ids):
        if len(existing_standard_answer_categories) == 0:
            raise EERequiredError(
                "Standard answers are a paid Enterprise Edition feature - enable EE or remove standard answer categories"
            )
        else:
            raise ValueError(
                f"Some or all categories with ids {standard_answer_category_ids} do not exist"
            )

    slack_channel_config = SlackChannelConfig(
        slack_bot_id=slack_bot_id,
        persona_id=persona_id,
        channel_config=channel_config,
        standard_answer_categories=existing_standard_answer_categories,
        enable_auto_filters=enable_auto_filters,
    )
    db_session.add(slack_channel_config)
    db_session.commit()

    return slack_channel_config


def update_slack_channel_config(
    db_session: Session,
    slack_channel_config_id: int,
    persona_id: int | None,
    channel_config: ChannelConfig,
    standard_answer_category_ids: list[int],
    enable_auto_filters: bool,
) -> SlackChannelConfig:
    """
    更新现有的Slack频道配置
    
    Args:
        db_session: 数据库会话
        slack_channel_config_id: 需要更新的配置ID
        persona_id: 新的persona ID
        channel_config: 新的频道配置
        standard_answer_category_ids: 新的标准答案类别ID列表
        enable_auto_filters: 是否启用自动过滤
    Returns:
        更新后的SlackChannelConfig对象
    """
    slack_channel_config = db_session.scalar(
        select(SlackChannelConfig).where(
            SlackChannelConfig.id == slack_channel_config_id
        )
    )
    if slack_channel_config is None:
        raise ValueError(
            f"Unable to find Slack channel config with ID {slack_channel_config_id}"
        )

    versioned_fetch_standard_answer_categories_by_ids = (
        fetch_versioned_implementation_with_fallback(
            "onyx.db.standard_answer",
            "fetch_standard_answer_categories_by_ids",
            _no_ee_standard_answer_categories,
        )
    )
    existing_standard_answer_categories = (
        versioned_fetch_standard_answer_categories_by_ids(
            standard_answer_category_ids=standard_answer_category_ids,
            db_session=db_session,
        )
    )
    if len(existing_standard_answer_categories) != len(standard_answer_category_ids):
        raise ValueError(
            f"Some or all categories with ids {standard_answer_category_ids} do not exist"
        )

    # 在更新对象之前获取现有的persona id
    existing_persona_id = slack_channel_config.persona_id

    # 更新配置
    # 注意：需要在清理旧persona之前执行此操作，否则会遇到外键约束错误
    slack_channel_config.persona_id = persona_id
    slack_channel_config.channel_config = channel_config
    slack_channel_config.standard_answer_categories = list(
        existing_standard_answer_categories
    )
    slack_channel_config.enable_auto_filters = enable_auto_filters

    # 如果persona发生了变化，则清理旧的persona
    if persona_id != existing_persona_id and existing_persona_id:
        existing_persona = db_session.scalar(
            select(Persona).where(Persona.id == existing_persona_id)
        )
        # 如果现有的persona是专门为这个Slack频道创建的，则清理它
        if existing_persona and existing_persona.name.startswith(
            SLACK_BOT_PERSONA_PREFIX
        ):
            _cleanup_relationships(
                db_session=db_session, persona_id=existing_persona_id
            )

    db_session.commit()

    return slack_channel_config


def remove_slack_channel_config(
    db_session: Session,
    slack_channel_config_id: int,
    user: User | None,
) -> None:
    """
    删除指定的Slack频道配置
    
    Args:
        db_session: 数据库会话
        slack_channel_config_id: 需要删除的配置ID
        user: 执行删除操作的用户
    """
    slack_channel_config = db_session.scalar(
        select(SlackChannelConfig).where(
            SlackChannelConfig.id == slack_channel_config_id
        )
    )
    if slack_channel_config is None:
        raise ValueError(
            f"Unable to find Slack channel config with ID {slack_channel_config_id}"
        )

    existing_persona_id = slack_channel_config.persona_id
    if existing_persona_id:
        existing_persona = db_session.scalar(
            select(Persona).where(Persona.id == existing_persona_id)
        )
        # if the existing persona was one created just for use with this Slack channel,
        # then clean it up
        # 如果现有的persona是专门为这个Slack频道创建的，则清理它
        if existing_persona and existing_persona.name.startswith(
            SLACK_BOT_PERSONA_PREFIX
        ):
            _cleanup_relationships(
                db_session=db_session, persona_id=existing_persona_id
            )
            mark_persona_as_deleted(
                persona_id=existing_persona_id, user=user, db_session=db_session
            )

    db_session.delete(slack_channel_config)
    db_session.commit()


def fetch_slack_channel_configs(
    db_session: Session, slack_bot_id: int | None = None
) -> Sequence[SlackChannelConfig]:
    """
    获取Slack频道配置列表
    
    Args:
        db_session: 数据库会话
        slack_bot_id: Slack机器人ID（可选）
    Returns:
        SlackChannelConfig对象列表
    """
    if not slack_bot_id:
        return db_session.scalars(select(SlackChannelConfig)).all()

    return db_session.scalars(
        select(SlackChannelConfig).where(
            SlackChannelConfig.slack_bot_id == slack_bot_id
        )
    ).all()


def fetch_slack_channel_config(
    db_session: Session, slack_channel_config_id: int
) -> SlackChannelConfig | None:
    """
    获取指定ID的Slack频道配置
    
    Args:
        db_session: 数据库会话
        slack_channel_config_id: 配置ID
    Returns:
        SlackChannelConfig对象，如果不存在则返回None
    """
    return db_session.scalar(
        select(SlackChannelConfig).where(
            SlackChannelConfig.id == slack_channel_config_id
        )
    )
