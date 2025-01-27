"""
此文件定义了后端服务器管理相关的数据模型。
主要包含用户信息、权限管理、Slack机器人配置等相关的数据模型类。
这些模型类主要用于API请求和响应的数据验证与转换。
"""

from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator

from ee.onyx.server.manage.models import StandardAnswerCategory
from onyx.auth.schemas import UserRole
from onyx.configs.app_configs import TRACK_EXTERNAL_IDP_EXPIRY
from onyx.configs.constants import AuthType
from onyx.context.search.models import SavedSearchSettings
from onyx.db.models import AllowedAnswerFilters
from onyx.db.models import ChannelConfig
from onyx.db.models import SlackBot as SlackAppModel
from onyx.db.models import SlackChannelConfig as SlackChannelConfigModel
from onyx.db.models import User
from onyx.onyxbot.slack.config import VALID_SLACK_FILTERS
from onyx.server.features.persona.models import PersonaSnapshot
from onyx.server.models import FullUserSnapshot
from onyx.server.models import InvitedUserSnapshot


if TYPE_CHECKING:
    pass


class VersionResponse(BaseModel):
    """后端版本响应模型类"""
    backend_version: str  # 后端版本号


class AuthTypeResponse(BaseModel):
    """
    认证类型响应模型类
    """
    auth_type: AuthType  # 认证类型
    # specifies whether the current auth setup requires
    # users to have verified emails
    # 指定当前认证设置是否要求用户验证邮箱
    requires_verification: bool
    anonymous_user_enabled: bool | None = None  # 是否启用匿名用户


class UserPreferences(BaseModel):
    """
    用户偏好设置模型类
    """
    chosen_assistants: list[int] | None = None  # 已选择的助手ID列表
    hidden_assistants: list[int] = []  # 隐藏的助手ID列表
    visible_assistants: list[int] = []  # 可见的助手ID列表
    recent_assistants: list[int] | None = None  # 最近使用的助手ID列表
    default_model: str | None = None  # 默认模型
    auto_scroll: bool | None = None  # 是否自动滚动


class UserInfo(BaseModel):
    """
    用户信息模型类
    """
    id: str  # 用户ID
    email: str  # 用户邮箱
    is_active: bool  # 是否激活
    is_superuser: bool  # 是否超级用户
    is_verified: bool  # 是否已验证
    role: UserRole  # 用户角色
    preferences: UserPreferences  # 用户偏好设置
    oidc_expiry: datetime | None = None  # OIDC过期时间
    current_token_created_at: datetime | None = None  # 当前token创建时间
    current_token_expiry_length: int | None = None  # 当前token过期时长
    is_cloud_superuser: bool = False  # 是否云端超级用户
    organization_name: str | None = None  # 组织名称
    is_anonymous_user: bool | None = None  # 是否匿名用户

    @classmethod
    def from_model(
        cls,
        user: User,
        current_token_created_at: datetime | None = None,
        expiry_length: int | None = None,
        is_cloud_superuser: bool = False,
        organization_name: str | None = None,
        is_anonymous_user: bool | None = None,
    ) -> "UserInfo":
        """
        从User模型创建UserInfo对象
        Args:
            user: User模型对象
            current_token_created_at: 当前token创建时间
            expiry_length: token过期时长
            is_cloud_superuser: 是否云端超级用户
            organization_name: 组织名称
            is_anonymous_user: 是否匿名用户
        Returns:
            UserInfo对象
        """
        return cls(
            id=str(user.id),
            email=user.email,
            is_active=user.is_active,
            is_superuser=user.is_superuser,
            is_verified=user.is_verified,
            role=user.role,
            preferences=(
                UserPreferences(
                    auto_scroll=user.auto_scroll,
                    chosen_assistants=user.chosen_assistants,
                    default_model=user.default_model,
                    hidden_assistants=user.hidden_assistants,
                    visible_assistants=user.visible_assistants,
                )
            ),
            organization_name=organization_name,
            # set to None if TRACK_EXTERNAL_IDP_EXPIRY is False so that we avoid cases
            # where they previously had this set + used OIDC, and now they switched to
            # basic auth are now constantly getting redirected back to the login page
            # since their "oidc_expiry is old"
            # 如果TRACK_EXTERNAL_IDP_EXPIRY为False，则设置为None，以避免他们之前设置了这个+使用了OIDC，
            # 现在他们切换到基本认证，现在不断被重定向回登录页面，因为他们的"oidc_expiry是旧的"
            oidc_expiry=user.oidc_expiry if TRACK_EXTERNAL_IDP_EXPIRY else None,
            current_token_created_at=current_token_created_at,
            current_token_expiry_length=expiry_length,
            is_cloud_superuser=is_cloud_superuser,
            is_anonymous_user=is_anonymous_user,
        )


class UserByEmail(BaseModel):
    """用户邮箱查询模型类"""
    user_email: str  # 用户邮箱


class UserRoleUpdateRequest(BaseModel):
    """用户角色更新请求模型类"""
    user_email: str  # 用户邮箱
    new_role: UserRole  # 新角色


class UserRoleResponse(BaseModel):
    """用户角色响应模型类"""
    role: str  # 角色名称


class BoostDoc(BaseModel):
    """文档提升权重模型类"""
    document_id: str  # 文档ID
    semantic_id: str  # 语义ID
    link: str  # 文档链接
    boost: int  # 提升权重值
    hidden: bool  # 是否隐藏


class BoostUpdateRequest(BaseModel):
    """文档权重更新请求模型类"""
    document_id: str  # 文档ID
    boost: int  # 新的权重值


class HiddenUpdateRequest(BaseModel):
    """文档隐藏状态更新请求模型类"""
    document_id: str  # 文档ID
    hidden: bool  # 是否隐藏


class AutoScrollRequest(BaseModel):
    """自动滚动设置请求模型类"""
    auto_scroll: bool | None  # 是否启用自动滚动


class SlackBotCreationRequest(BaseModel):
    """Slack机器人创建请求模型类"""
    name: str  # 机器人名称
    enabled: bool  # 是否启用
    bot_token: str  # 机器人令牌
    app_token: str  # 应用令牌


class SlackBotTokens(BaseModel):
    """Slack机器人令牌模型类"""
    bot_token: str  # 机器人令牌
    app_token: str  # 应用令牌
    model_config = ConfigDict(frozen=True)


# TODO No longer in use, remove later
# TODO 不再使用，稍后移除
class SlackBotResponseType(str, Enum):
    """
    Slack机器人响应类型枚举
    """
    QUOTES = "quotes"  # 引用
    CITATIONS = "citations"  # 引证


class SlackChannelConfigCreationRequest(BaseModel):
    """Slack频道配置创建请求模型类"""
    slack_bot_id: int  # Slack机器人ID
    # currently, a persona is created for each Slack channel config
    # in the future, `document_sets` will probably be replaced
    # by an optional `PersonaSnapshot` object. Keeping it like this
    # for now for simplicity / speed of development
    # 目前，每个Slack频道配置都会创建一个persona
    # 将来，`document_sets`可能会被一个可选的`PersonaSnapshot`对象取代。
    # 目前保持这样是为了简化/加快开发速度
    document_sets: list[int] | None = None  # 文档集ID列表

    # NOTE: only one of `document_sets` / `persona_id` should be set
    # 注意：`document_sets` / `persona_id`中只能设置一个
    persona_id: int | None = None  # Persona ID

    channel_name: str  # 频道名称
    respond_tag_only: bool = False  # 是否仅响应标签
    respond_to_bots: bool = False  # 是否响应机器人
    show_continue_in_web_ui: bool = False  # 是否在Web UI中显示继续按钮
    enable_auto_filters: bool = False  # 是否启用自动过滤器
    # If no team members, assume respond in the channel to everyone
    # 如果没有团队成员，假设在频道中对所有人响应
    respond_member_group_list: list[str] = Field(default_factory=list)  # 响应成员组列表
    answer_filters: list[AllowedAnswerFilters] = Field(default_factory=list)  # 答案过滤器列表
    # list of user emails
    # 用户邮箱列表
    follow_up_tags: list[str] | None = None  # 跟进标签列表
    response_type: SlackBotResponseType  # 响应类型
    # XXX this is going away soon
    # XXX 这将很快消失
    standard_answer_categories: list[int] = Field(default_factory=list)  # 标准答案类别列表

    @field_validator("answer_filters", mode="before")
    @classmethod
    def validate_filters(cls, value: list[str]) -> list[str]:
        if any(test not in VALID_SLACK_FILTERS for test in value):
            raise ValueError(
                f"Slack Answer filters must be one of {VALID_SLACK_FILTERS}"
            )
        return value

    @model_validator(mode="after")
    def validate_document_sets_and_persona_id(
        self,
    ) -> "SlackChannelConfigCreationRequest":
        if self.document_sets and self.persona_id:
            raise ValueError("Only one of `document_sets` / `persona_id` should be set")

        return self


class SlackChannelConfig(BaseModel):
    """Slack频道配置模型类"""
    slack_bot_id: int  # Slack机器人ID
    id: int  # 配置ID
    persona: PersonaSnapshot | None  # Persona快照
    channel_config: ChannelConfig  # 频道配置
    # XXX this is going away soon
    # XXX 这将很快消失
    standard_answer_categories: list[StandardAnswerCategory]  # 标准答案类别列表
    enable_auto_filters: bool  # 是否启用自动过滤器

    @classmethod
    def from_model(
        cls, slack_channel_config_model: SlackChannelConfigModel
    ) -> "SlackChannelConfig":
        return cls(
            id=slack_channel_config_model.id,
            slack_bot_id=slack_channel_config_model.slack_bot_id,
            persona=(
                PersonaSnapshot.from_model(
                    slack_channel_config_model.persona, allow_deleted=True
                )
                if slack_channel_config_model.persona
                else None
            ),
            channel_config=slack_channel_config_model.channel_config,
            # XXX this is going away soon
            # XXX 这将很快消失
            standard_answer_categories=[
                StandardAnswerCategory.from_model(standard_answer_category_model)
                for standard_answer_category_model in slack_channel_config_model.standard_answer_categories
            ],
            enable_auto_filters=slack_channel_config_model.enable_auto_filters,
        )


class SlackBot(BaseModel):
    """
    这个模型与SlackAppModel相同，但它包含一个`configs_count`字段，
    以便更容易获取与SlackBot关联的SlackChannelConfigs数量。
    """
    id: int  # 机器人ID
    name: str  # 机器人名称
    enabled: bool  # 是否启用
    configs_count: int  # 配置数量

    bot_token: str  # 机器人令牌
    app_token: str  # 应用令牌

    @classmethod
    def from_model(cls, slack_bot_model: SlackAppModel) -> "SlackBot":
        return cls(
            id=slack_bot_model.id,
            name=slack_bot_model.name,
            enabled=slack_bot_model.enabled,
            bot_token=slack_bot_model.bot_token,
            app_token=slack_bot_model.app_token,
            configs_count=len(slack_bot_model.slack_channel_configs),
        )


class FullModelVersionResponse(BaseModel):
    """完整模型版本响应类"""
    current_settings: SavedSearchSettings  # 当前搜索设置
    secondary_settings: SavedSearchSettings | None  # 次要搜索设置


class AllUsersResponse(BaseModel):
    """所有用户响应模型类"""
    accepted: list[FullUserSnapshot]  # 已接受的用户列表
    invited: list[InvitedUserSnapshot]  # 已邀请的用户列表
    slack_users: list[FullUserSnapshot]  # Slack用户列表
    accepted_pages: int  # 已接受用户的总页数
    invited_pages: int  # 已邀请用户的总页数
    slack_users_pages: int  # Slack用户的总页数
