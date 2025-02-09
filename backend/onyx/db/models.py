"""
这个文件包含了Onyx系统的所有数据库模型定义。
主要包括以下几个部分：
1. 基础模型和类型定义
2. 认证/授权相关模型
3. 文档/索引相关模型
4. 消息/对话相关模型
5. 反馈/日志/指标相关模型
6. 企业版特有功能相关模型
"""

import datetime
import json
from typing import Any
from typing import Literal
from typing import NotRequired
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel
from typing_extensions import TypedDict  # noreorder
from uuid import UUID

from sqlalchemy.dialects.postgresql import UUID as PGUUID

from fastapi_users_db_sqlalchemy import SQLAlchemyBaseOAuthAccountTableUUID
from fastapi_users_db_sqlalchemy import SQLAlchemyBaseUserTableUUID
from fastapi_users_db_sqlalchemy.access_token import SQLAlchemyBaseAccessTokenTableUUID
from fastapi_users_db_sqlalchemy.generics import TIMESTAMPAware
from sqlalchemy import Boolean
from sqlalchemy import DateTime
from sqlalchemy import Enum
from sqlalchemy import Float
from sqlalchemy import ForeignKey
from sqlalchemy import func
from sqlalchemy import Index
from sqlalchemy import Integer
from sqlalchemy import Sequence
from sqlalchemy import String
from sqlalchemy import Text
from sqlalchemy import UniqueConstraint
from sqlalchemy.dialects import postgresql
from sqlalchemy.engine.interfaces import Dialect
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship
from sqlalchemy.types import LargeBinary
from sqlalchemy.types import TypeDecorator

from onyx.auth.schemas import UserRole
from onyx.configs.chat_configs import NUM_POSTPROCESSED_RESULTS
from onyx.configs.constants import DEFAULT_BOOST, MilestoneRecordType
from onyx.configs.constants import DocumentSource
from onyx.configs.constants import FileOrigin
from onyx.configs.constants import MessageType
from onyx.db.enums import AccessType, IndexingMode
from onyx.configs.constants import NotificationType
from onyx.configs.constants import SearchFeedbackType
from onyx.configs.constants import TokenRateLimitScope
from onyx.connectors.models import InputType
from onyx.db.enums import ChatSessionSharedStatus
from onyx.db.enums import ConnectorCredentialPairStatus
from onyx.db.enums import IndexingStatus
from onyx.db.enums import IndexModelStatus
from onyx.db.enums import TaskStatus
from onyx.db.pydantic_type import PydanticType
from onyx.utils.logger import setup_logger
from onyx.utils.special_types import JSON_ro
from onyx.file_store.models import FileDescriptor
from onyx.llm.override_models import LLMOverride
from onyx.llm.override_models import PromptOverride
from onyx.context.search.enums import RecencyBiasSetting
from onyx.utils.encryption import decrypt_bytes_to_string
from onyx.utils.encryption import encrypt_string_to_bytes
from onyx.utils.headers import HeaderItemDict
from shared_configs.enums import EmbeddingProvider
from shared_configs.enums import RerankerProvider

logger = setup_logger()


class Base(DeclarativeBase):
    """基础数据库模型类
    所有其他模型类都继承自这个基类"""
    __abstract__ = True


class EncryptedString(TypeDecorator):
    """加密字符串类型
    用于在数据库中安全存储敏感字符串数据,自动处理加密和解密"""
    impl = LargeBinary
    cache_ok = True

    def process_bind_param(self, value: str | None, dialect: Dialect) -> bytes | None:
        """加密字符串值以便存储
        Args:
            value: 要加密的字符串
            dialect: 数据库方言
        Returns:
            加密后的字节数据"""
        if value is not None:
            return encrypt_string_to_bytes(value)
        return value

    def process_result_value(self, value: bytes | None, dialect: Dialect) -> str | None:
        """解密存储的加密数据
        Args:
            value: 加密的字节数据
            dialect: 数据库方言
        Returns:
            解密后的字符串"""
        if value is not None:
            return decrypt_bytes_to_string(value)
        return value


class EncryptedJson(TypeDecorator):
    """加密JSON类型
    用于在数据库中安全存储JSON数据,自动处理加密和解密"""
    impl = LargeBinary
    cache_ok = True

    def process_bind_param(self, value: dict | None, dialect: Dialect) -> bytes | None:
        """加密JSON数据以便存储
        Args:
            value: 要加密的JSON数据
            dialect: 数据库方言
        Returns:
            加密后的字节数据"""
        if value is not None:
            json_str = json.dumps(value)
            return encrypt_string_to_bytes(json_str)
        return value

    def process_result_value(
        self, value: bytes | None, dialect: Dialect
    ) -> dict | None:
        """解密存储的加密JSON数据
        Args:
            value: 加密的字节数据 
            dialect: 数据库方言
        Returns:
            解密后的JSON数据"""
        if value is not None:
            json_str = decrypt_bytes_to_string(value)
            return json.loads(json_str)
        return value


class NullFilteredString(TypeDecorator):
    """过滤空字符的字符串类型
    用于去除字符串中的null字符"""
    impl = String
    cache_ok = True

    def process_bind_param(self, value: str | None, dialect: Dialect) -> str | None:
        """处理输入字符串,移除null字符
        Args:
            value: 输入字符串
            dialect: 数据库方言
        Returns:
            处理后的字符串"""
        if value is not None and "\x00" in value:
            logger.warning(f"NUL characters found in value: {value}")  # 在值中发现NUL字符
            return value.replace("\x00", "")
        return value

    def process_result_value(self, value: str | None, dialect: Dialect) -> str | None:
        """返回存储的字符串值
        Args: 
            value: 存储的字符串
            dialect: 数据库方言
        Returns:
            字符串值"""
        return value


"""
认证/授权相关类
"""

class OAuthAccount(SQLAlchemyBaseOAuthAccountTableUUID, Base):
    """OAuth账户模型类
    用于存储OAuth认证相关的账户信息"""
    # even an almost empty token from keycloak will not fit the default 1024 bytes
    access_token: Mapped[str] = mapped_column(Text, nullable=False)  # type: ignore


class User(SQLAlchemyBaseUserTableUUID, Base):
    """用户模型类
    存储用户的基本信息、权限和首选项设置"""
    oauth_accounts: Mapped[list[OAuthAccount]] = relationship(
        "OAuthAccount", lazy="joined", cascade="all, delete-orphan"
    )
    role: Mapped[UserRole] = mapped_column(
        Enum(UserRole, native_enum=False, default=UserRole.BASIC)
    )

    """
    首选项在未来可能会放在单独的表中，但目前为了简单起见放在这里
    """

    # if specified, controls the assistants that are shown to the user + their order
    # if not specified, all assistants are shown
    auto_scroll: Mapped[bool] = mapped_column(Boolean, default=True)
    chosen_assistants: Mapped[list[int] | None] = mapped_column(
        postgresql.JSONB(), nullable=True, default=None
    )
    visible_assistants: Mapped[list[int]] = mapped_column(
        postgresql.JSONB(), nullable=False, default=[]
    )
    hidden_assistants: Mapped[list[int]] = mapped_column(
        postgresql.JSONB(), nullable=False, default=[]
    )
    recent_assistants: Mapped[list[dict]] = mapped_column(
        postgresql.JSONB(), nullable=False, default=list, server_default="[]"
    )

    oidc_expiry: Mapped[datetime.datetime] = mapped_column(
        TIMESTAMPAware(timezone=True), nullable=True
    )

    default_model: Mapped[str] = mapped_column(Text, nullable=True)
    # organized in typical structured fashion
    # formatted as `displayName__provider__modelName`

    # relationships
    credentials: Mapped[list["Credential"]] = relationship(
        "Credential", back_populates="user", lazy="joined"
    )
    chat_sessions: Mapped[list["ChatSession"]] = relationship(
        "ChatSession", back_populates="user"
    )
    chat_folders: Mapped[list["ChatFolder"]] = relationship(
        "ChatFolder", back_populates="user"
    )

    prompts: Mapped[list["Prompt"]] = relationship("Prompt", back_populates="user")

    # Personas owned by this user
    personas: Mapped[list["Persona"]] = relationship("Persona", back_populates="user")
    # Custom tools created by this user
    custom_tools: Mapped[list["Tool"]] = relationship("Tool", back_populates="user")
    # Notifications for the UI
    notifications: Mapped[list["Notification"]] = relationship(
        "Notification", back_populates="user"
    )
    cc_pairs: Mapped[list["ConnectorCredentialPair"]] = relationship(
        "ConnectorCredentialPair",
        back_populates="creator",
        primaryjoin="User.id == foreign(ConnectorCredentialPair.creator_id)",
    )


class AccessToken(SQLAlchemyBaseAccessTokenTableUUID, Base):
    """访问令牌模型类
    用于存储用户的访问令牌信息"""
    pass


class ApiKey(Base):
    """API密钥模型类
    用于存储和管理API访问密钥"""
    __tablename__ = "api_key"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str | None] = mapped_column(String, nullable=True)
    hashed_api_key: Mapped[str] = mapped_column(String, unique=True)
    api_key_display: Mapped[str] = mapped_column(String, unique=True)
    # the ID of the "user" who represents the access credentials for the API key
    user_id: Mapped[UUID] = mapped_column(ForeignKey("user.id"), nullable=False)
    # the ID of the user who owns the key
    owner_id: Mapped[UUID | None] = mapped_column(ForeignKey("user.id"), nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Add this relationship to access the User object via user_id
    user: Mapped["User"] = relationship("User", foreign_keys=[user_id])


class Notification(Base):
    """通知模型类
    用于存储和管理系统向用户发送的各类通知"""
    __tablename__ = "notification"

    id: Mapped[int] = mapped_column(primary_key=True)
    notif_type: Mapped[NotificationType] = mapped_column(
        Enum(NotificationType, native_enum=False)
    )
    user_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("user.id", ondelete="CASCADE"), nullable=True
    )
    dismissed: Mapped[bool] = mapped_column(Boolean, default=False)
    last_shown: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True))
    first_shown: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True))

    user: Mapped[User] = relationship("User", back_populates="notifications")
    additional_data: Mapped[dict | None] = mapped_column(
        postgresql.JSONB(), nullable=True
    )


"""
关联表类
NOTE: must be at the top since they are referenced by other tables
"""


class Persona__DocumentSet(Base):
    """人格与文档集关联表
    用于建立人格和文档集之间的多对多关系"""
    __tablename__ = "persona__document_set"

    persona_id: Mapped[int] = mapped_column(ForeignKey("persona.id"), primary_key=True)
    document_set_id: Mapped[int] = mapped_column(
        ForeignKey("document_set.id"), primary_key=True
    )


class Persona__Prompt(Base):
    """人格与提示词关联表
    用于建立人格和提示词之间的多对多关系"""
    __tablename__ = "persona__prompt"

    persona_id: Mapped[int] = mapped_column(ForeignKey("persona.id"), primary_key=True)
    prompt_id: Mapped[int] = mapped_column(ForeignKey("prompt.id"), primary_key=True)


class Persona__User(Base):
    """人格与用户关联表
    用于建立人格和用户之间的多对多关系""" 
    __tablename__ = "persona__user"

    persona_id: Mapped[int] = mapped_column(ForeignKey("persona.id"), primary_key=True)
    user_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("user.id", ondelete="CASCADE"), primary_key=True, nullable=True
    )


class DocumentSet__User(Base):
    """文档集与用户关联表
    用于建立文档集和用户之间的多对多关系"""
    __tablename__ = "document_set__user"

    document_set_id: Mapped[int] = mapped_column(
        ForeignKey("document_set.id"), primary_key=True
    )
    user_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("user.id", ondelete="CASCADE"), primary_key=True, nullable=True
    )


class DocumentSet__ConnectorCredentialPair(Base):
    """文档集与连接器凭据对关联表
    用于管理文档集和连接器凭据对之间的关系"""
    __tablename__ = "document_set__connector_credential_pair"

    document_set_id: Mapped[int] = mapped_column(
        ForeignKey("document_set.id"), primary_key=True
    )
    connector_credential_pair_id: Mapped[int] = mapped_column(
        ForeignKey("connector_credential_pair.id"), primary_key=True
    )
    # 如果为True，则是文档集当前状态的一部分 
    # 如果为False，则是文档集之前状态的一部分
    # 当文档集更新后，is_current=False的行应该被删除
    # 当DocumentSet.is_up_to_date为True时，不应该存在这样的行
    is_current: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        primary_key=True,
    )

    document_set: Mapped["DocumentSet"] = relationship("DocumentSet")


class ChatMessage__SearchDoc(Base):
    """聊天消息与搜索文档关联表
    用于关联聊天消息和相关的搜索文档"""
    __tablename__ = "chat_message__search_doc"

    chat_message_id: Mapped[int] = mapped_column(
        ForeignKey("chat_message.id"), primary_key=True
    )
    search_doc_id: Mapped[int] = mapped_column(
        ForeignKey("search_doc.id"), primary_key=True
    )


class Document__Tag(Base):
    """文档与标签关联表
    用于管理文档和标签之间的多对多关系"""
    __tablename__ = "document__tag"

    document_id: Mapped[str] = mapped_column(
        ForeignKey("document.id"), primary_key=True
    )
    tag_id: Mapped[int] = mapped_column(ForeignKey("tag.id"), primary_key=True)


class Persona__Tool(Base):
    """人格与工具关联表
    用于管理人格和工具之间的多对多关系"""
    __tablename__ = "persona__tool"

    persona_id: Mapped[int] = mapped_column(ForeignKey("persona.id"), primary_key=True)
    tool_id: Mapped[int] = mapped_column(ForeignKey("tool.id"), primary_key=True)


class StandardAnswer__StandardAnswerCategory(Base):
    """标准答案与分类关联表
    用于管理标准答案和分类之间的多对多关系"""
    __tablename__ = "standard_answer__standard_answer_category"

    standard_answer_id: Mapped[int] = mapped_column(
        ForeignKey("standard_answer.id"), primary_key=True
    )
    standard_answer_category_id: Mapped[int] = mapped_column(
        ForeignKey("standard_answer_category.id"), primary_key=True
    )


class SlackChannelConfig__StandardAnswerCategory(Base):
    __tablename__ = "slack_channel_config__standard_answer_category"

    slack_channel_config_id: Mapped[int] = mapped_column(
        ForeignKey("slack_channel_config.id"), primary_key=True
    )
    standard_answer_category_id: Mapped[int] = mapped_column(
        ForeignKey("standard_answer_category.id"), primary_key=True
    )


class ChatMessage__StandardAnswer(Base):
    """聊天消息与标准答案关联表
    用于关联聊天消息和使用的标准答案"""
    __tablename__ = "chat_message__standard_answer"

    chat_message_id: Mapped[int] = mapped_column(
        ForeignKey("chat_message.id"), primary_key=True
    )
    standard_answer_id: Mapped[int] = mapped_column(
        ForeignKey("standard_answer.id"), primary_key=True
    )


"""
文档和索引相关类
"""

class ConnectorCredentialPair(Base):
    """连接器凭据对模型类
    用于管理连接器和凭据之间的关系，支持文档索引和同步"""
    __tablename__ = "connector_credential_pair"
    # NOTE: this `id` column has to use `Sequence` instead of `autoincrement=True`
    # due to some SQLAlchemy quirks + this not being a primary key column
    id: Mapped[int] = mapped_column(
        Integer,
        Sequence("connector_credential_pair_id_seq"),
        unique=True,
        nullable=False,
    )
    name: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[ConnectorCredentialPairStatus] = mapped_column(
        Enum(ConnectorCredentialPairStatus, native_enum=False), nullable=False
    )
    connector_id: Mapped[int] = mapped_column(
        ForeignKey("connector.id"), primary_key=True
    )

    deletion_failure_message: Mapped[str | None] = mapped_column(String, nullable=True)

    credential_id: Mapped[int] = mapped_column(
        ForeignKey("credential.id"), primary_key=True
    )
    # controls whether the documents indexed by this CC pair are visible to all
    # or if they are only visible to those with that are given explicit access
    # (e.g. via owning the credential or being a part of a group that is given access)
    access_type: Mapped[AccessType] = mapped_column(
        Enum(AccessType, native_enum=False), nullable=False
    )

    # special info needed for the auto-sync feature. The exact structure depends on the

    # source type (defined in the connector's `source` field)
    # E.g. for google_drive perm sync:
    # {"customer_id": "123567", "company_domain": "@onyx.app"}
    auto_sync_options: Mapped[dict[str, Any] | None] = mapped_column(
        postgresql.JSONB(), nullable=True
    )
    last_time_perm_sync: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    last_time_external_group_sync: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # Time finished, not used for calculating backend jobs which uses time started (created)
    last_successful_index_time: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), default=None
    )

    # last successful prune
    last_pruned: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True
    )

    total_docs_indexed: Mapped[int] = mapped_column(Integer, default=0)

    indexing_trigger: Mapped[IndexingMode | None] = mapped_column(
        Enum(IndexingMode, native_enum=False), nullable=True
    )

    connector: Mapped["Connector"] = relationship(
        "Connector", back_populates="credentials"
    )
    credential: Mapped["Credential"] = relationship(
        "Credential", back_populates="connectors"
    )
    document_sets: Mapped[list["DocumentSet"]] = relationship(
        "DocumentSet",
        secondary=DocumentSet__ConnectorCredentialPair.__table__,
        primaryjoin=(
            (DocumentSet__ConnectorCredentialPair.connector_credential_pair_id == id)
            & (DocumentSet__ConnectorCredentialPair.is_current.is_(True))
        ),
        back_populates="connector_credential_pairs",
        overlaps="document_set",
    )
    index_attempts: Mapped[list["IndexAttempt"]] = relationship(
        "IndexAttempt", back_populates="connector_credential_pair"
    )

    # the user id of the user that created this cc pair
    creator_id: Mapped[UUID | None] = mapped_column(nullable=True)
    creator: Mapped["User"] = relationship(
        "User",
        back_populates="cc_pairs",
        primaryjoin="foreign(ConnectorCredentialPair.creator_id) == remote(User.id)",
    )


class Document(Base):
    """文档模型类 
    存储所有索引文档的元数据和访问控制信息"""
    __tablename__ = "document"
    # NOTE: if more sensitive data is added here for display, make sure to add user/group permission

    # this should correspond to the ID of the document
    # (as is passed around in Onyx)
    id: Mapped[str] = mapped_column(NullFilteredString, primary_key=True)
    from_ingestion_api: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=True
    )
    # 0 for neutral, positive for mostly endorse, negative for mostly reject
    boost: Mapped[int] = mapped_column(Integer, default=DEFAULT_BOOST)
    hidden: Mapped[bool] = mapped_column(Boolean, default=False)
    semantic_id: Mapped[str] = mapped_column(NullFilteredString)
    # First Section's link
    link: Mapped[str | None] = mapped_column(NullFilteredString, nullable=True)

    # The updated time is also used as a measure of the last successful state of the doc
    # pulled from the source (to help skip reindexing already updated docs in case of
    # connector retries)
    # TODO: rename this column because it conflates the time of the source doc
    # with the local last modified time of the doc and any associated metadata
    # it should just be the server timestamp of the source doc
    doc_updated_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Number of chunks in the document (in Vespa)
    # Only null for documents indexed prior to this change
    chunk_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # last time any vespa relevant row metadata or the doc changed.
    # does not include last_synced
    last_modified: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True, default=func.now()
    )

    # last successful sync to vespa
    last_synced: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True
    )
    # The following are not attached to User because the account/email may not be known
    # within Onyx
    # Something like the document creator
    primary_owners: Mapped[list[str] | None] = mapped_column(
        postgresql.ARRAY(String), nullable=True
    )
    secondary_owners: Mapped[list[str] | None] = mapped_column(
        postgresql.ARRAY(String), nullable=True
    )
    # Permission sync columns
    # Email addresses are saved at the document level for externally synced permissions
    # This is becuase the normal flow of assigning permissions is through the cc_pair
    # doesn't apply here
    external_user_emails: Mapped[list[str] | None] = mapped_column(
        postgresql.ARRAY(String), nullable=True
    )
    # These group ids have been prefixed by the source type
    external_user_group_ids: Mapped[list[str] | None] = mapped_column(
        postgresql.ARRAY(String), nullable=True
    )
    is_public: Mapped[bool] = mapped_column(Boolean, default=False)

    retrieval_feedbacks: Mapped[list["DocumentRetrievalFeedback"]] = relationship(
        "DocumentRetrievalFeedback",
        back_populates="document",
        primaryjoin="Document.id == DocumentRetrievalFeedback.document_id"
    )
    tags = relationship(
        "Tag",
        secondary=Document__Tag.__table__,
        back_populates="documents",
    )


class Tag(Base):
    """标签模型类
    用于对文档进行分类和标记"""
    __tablename__ = "tag"

    id: Mapped[int] = mapped_column(primary_key=True)
    tag_key: Mapped[str] = mapped_column(String)
    tag_value: Mapped[str] = mapped_column(String)
    source: Mapped[DocumentSource] = mapped_column(
        Enum(DocumentSource, native_enum=False)
    )

    documents = relationship(
        "Document",
        secondary=Document__Tag.__table__,
        back_populates="tags",
    )

    __table_args__ = (
        UniqueConstraint(
            "tag_key", "tag_value", "source", name="_tag_key_value_source_uc"
        ),
    )


class Connector(Base):
    """连接器模型类
    定义了不同类型的数据源连接器，用于从各种来源获取文档"""
    __tablename__ = "connector"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String)
    source: Mapped[DocumentSource] = mapped_column(
        Enum(DocumentSource, native_enum=False)
    )
    input_type = mapped_column(Enum(InputType, native_enum=False))
    connector_specific_config: Mapped[dict[str, Any]] = mapped_column(
        postgresql.JSONB()
    )
    indexing_start: Mapped[datetime.datetime | None] = mapped_column(
        DateTime, nullable=True
    )
    refresh_freq: Mapped[int | None] = mapped_column(Integer, nullable=True)
    prune_freq: Mapped[int | None] = mapped_column(Integer, nullable=True)
    time_created: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    time_updated: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    credentials: Mapped[list["ConnectorCredentialPair"]] = relationship(
        "ConnectorCredentialPair",
        back_populates="connector",
        cascade="all, delete-orphan",
    )
    documents_by_connector: Mapped[
        list["DocumentByConnectorCredentialPair"]
    ] = relationship("DocumentByConnectorCredentialPair", back_populates="connector")

    # synchronize this validation logic with RefreshFrequencySchema etc on front end
    # until we have a centralized validation schema

    # TODO(rkuo): experiment with SQLAlchemy validators rather than manual checks
    # https://docs.sqlalchemy.org/en/20/orm/mapped_attributes.html
    def validate_refresh_freq(self) -> None:
        """验证刷新频率
        确保刷新频率不小于60秒"""
        if self.refresh_freq is not None:
            if self.refresh_freq < 60:
                raise ValueError(
                    "刷新频率必须大于或等于60秒"
                )

    def validate_prune_freq(self) -> None:
        """验证清理频率
        确保清理频率不小于86400秒(1天)"""
        if self.prune_freq is not None:
            if self.prune_freq < 86400:
                raise ValueError(
                    "清理频率必须大于或等于86400秒"
                )


class Credential(Base):
    """凭据模型类
    存储访问各种数据源所需的认证信息"""
    __tablename__ = "credential"

    name: Mapped[str] = mapped_column(String, nullable=True)

    source: Mapped[DocumentSource] = mapped_column(
        Enum(DocumentSource, native_enum=False)
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    credential_json: Mapped[dict[str, Any]] = mapped_column(EncryptedJson())
    user_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("user.id", ondelete="CASCADE"), nullable=True
    )
    # if `true`, then all Admins will have access to the credential
    admin_public: Mapped[bool] = mapped_column(Boolean, default=True)
    time_created: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    time_updated: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    curator_public: Mapped[bool] = mapped_column(Boolean, default=False)

    connectors: Mapped[list["ConnectorCredentialPair"]] = relationship(
        "ConnectorCredentialPair",
        back_populates="credential",
        cascade="all, delete-orphan",
    )
    documents_by_credential: Mapped[
        list["DocumentByConnectorCredentialPair"]
    ] = relationship("DocumentByConnectorCredentialPair", back_populates="credential")

    user: Mapped[User | None] = relationship("User", back_populates="credentials")


class SearchSettings(Base):
    """搜索设置模型类
    管理文档检索和嵌入模型的配置"""
    __tablename__ = "search_settings"

    id: Mapped[int] = mapped_column(primary_key=True)
    model_name: Mapped[str] = mapped_column(String)
    model_dim: Mapped[int] = mapped_column(Integer)
    normalize: Mapped[bool] = mapped_column(Boolean)
    query_prefix: Mapped[str | None] = mapped_column(String, nullable=True)
    passage_prefix: Mapped[str | None] = mapped_column(String, nullable=True)

    status: Mapped[IndexModelStatus] = mapped_column(
        Enum(IndexModelStatus, native_enum=False)
    )
    index_name: Mapped[str] = mapped_column(String)
    provider_type: Mapped[EmbeddingProvider | None] = mapped_column(
        ForeignKey("embedding_provider.provider_type"), nullable=True
    )

    # Mini and Large Chunks (large chunk also checks for model max context)
    multipass_indexing: Mapped[bool] = mapped_column(Boolean, default=True)

    multilingual_expansion: Mapped[list[str]] = mapped_column(
        postgresql.ARRAY(String), default=[]
    )

    # Reranking settings
    disable_rerank_for_streaming: Mapped[bool] = mapped_column(Boolean, default=False)
    rerank_model_name: Mapped[str | None] = mapped_column(String, nullable=True)
    rerank_provider_type: Mapped[RerankerProvider | None] = mapped_column(
        Enum(RerankerProvider, native_enum=False), nullable=True
    )
    rerank_api_key: Mapped[str | None] = mapped_column(String, nullable=True)
    rerank_api_url: Mapped[str | None] = mapped_column(String, nullable=True)

    num_rerank: Mapped[int] = mapped_column(Integer, default=NUM_POSTPROCESSED_RESULTS)

    cloud_provider: Mapped["CloudEmbeddingProvider"] = relationship(
        "CloudEmbeddingProvider",
        back_populates="search_settings",
        foreign_keys=[provider_type],
    )

    index_attempts: Mapped[list["IndexAttempt"]] = relationship(
        "IndexAttempt", back_populates="search_settings"
    )

    __table_args__ = (
        Index(
            "ix_embedding_model_present_unique",
            "status",
            unique=True,
            postgresql_where=(status == IndexModelStatus.PRESENT),
        ),
        Index(
            "ix_embedding_model_future_unique",
            "status",
            unique=True,
            postgresql_where=(status == IndexModelStatus.FUTURE),
        ),
    )

    def __repr__(self) -> str:
        return f"<EmbeddingModel(model_name='{self.model_name}', status='{self.status}',\
          cloud_provider='{self.cloud_provider.provider_type if self.cloud_provider else 'None'}')>"

    @property
    def api_version(self) -> str | None:
        """获取API版本号
        如果存在cloud_provider则返回其api_version，否则返回None"""
        return (
            self.cloud_provider.api_version if self.cloud_provider is not None else None
        )

    @property 
    def deployment_name(self) -> str | None:
        """获取部署名称
        如果存在cloud_provider则返回其deployment_name，否则返回None"""
        return (
            self.cloud_provider.deployment_name
            if self.cloud_provider is not None
            else None
        )

    @property
    def api_url(self) -> str | None:
        """获取API URL
        如果存在cloud_provider则返回其api_url，否则返回None"""
        return self.cloud_provider.api_url if self.cloud_provider is not None else None

    @property
    def api_key(self) -> str | None:
        """获取API密钥
        如果存在cloud_provider则返回其api_key，否则返回None"""
        return self.cloud_provider.api_key if self.cloud_provider is not None else None


class IndexAttempt(Base):
    """
    索引尝试模型类
    表示对一个或多个文档进行索引的尝试。
    例如，从Google Drive进行一次拉取，从slack事件API接收一个事件，
    或者对网站进行一次爬取。
    """

    __tablename__ = "index_attempt"

    id: Mapped[int] = mapped_column(primary_key=True)

    connector_credential_pair_id: Mapped[int] = mapped_column(
        ForeignKey("connector_credential_pair.id"),
        nullable=False,
    )

    # 从头开始运行的一些索引尝试仍然会将其设置为False
    # 这仅用于通过运行一次API显式标记为从头开始的尝试
    from_beginning: Mapped[bool] = mapped_column(Boolean)
    status: Mapped[IndexingStatus] = mapped_column(
        Enum(IndexingStatus, native_enum=False)
    )
    # 如果切换了嵌入模型，下面两个字段可能会略有不同步
    new_docs_indexed: Mapped[int | None] = mapped_column(Integer, default=0)
    total_docs_indexed: Mapped[int | None] = mapped_column(Integer, default=0)
    docs_removed_from_index: Mapped[int | None] = mapped_column(Integer, default=0)
    # 仅在status = "failed"时填充
    error_msg: Mapped[str | None] = mapped_column(Text, default=None)
    # 仅在status = "failed"且出现未处理异常时填充
    full_exception_trace: Mapped[str | None] = mapped_column(Text, default=None)
    # 可为空是因为过去我们不允许动态切换嵌入模型
    search_settings_id: Mapped[int] = mapped_column(
        ForeignKey("search_settings.id", ondelete="SET NULL"),
        nullable=True,
    )

    time_created: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    # 当实际索引运行开始时
    # 注意：将使用api_server时钟而不是DB服务器时钟
    time_started: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), default=None
    )
    time_updated: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )

    connector_credential_pair: Mapped[ConnectorCredentialPair] = relationship(
        "ConnectorCredentialPair", back_populates="index_attempts"
    )

    search_settings: Mapped[SearchSettings | None] = relationship(
        "SearchSettings", back_populates="index_attempts"
    )

    error_rows = relationship(
        "IndexAttemptError",
        back_populates="index_attempt",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index(
            "ix_index_attempt_latest_for_connector_credential_pair",
            "connector_credential_pair_id",
            "time_created",
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<IndexAttempt(id={self.id!r}, "
            f"status={self.status!r}, "
            f"error_msg={self.error_msg!r})>"
            f"time_created={self.time_created!r}, "
            f"time_updated={self.time_updated!r}, "
        )

    def is_finished(self) -> bool:
        """检查索引任务是否完成
        通过检查状态是否为终态来判断"""
        return self.status.is_terminal()


class IndexAttemptError(Base):
    """
    索引尝试错误模型类
    记录索引尝试过程中遇到的错误
    """

    __tablename__ = "index_attempt_errors"

    id: Mapped[int] = mapped_column(primary_key=True)

    index_attempt_id: Mapped[int] = mapped_column(
        ForeignKey("index_attempt.id"),
        nullable=True,
    )

    # 批处理中发生错误的索引(如果正在遍历批次)
    # 仅供参考
    batch: Mapped[int | None] = mapped_column(Integer, default=None)
    doc_summaries: Mapped[list[Any]] = mapped_column(postgresql.JSONB())
    error_msg: Mapped[str | None] = mapped_column(Text, default=None)
    traceback: Mapped[str | None] = mapped_column(Text, default=None)
    time_created: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    # 这是关系的反向引用
    index_attempt = relationship("IndexAttempt", back_populates="error_rows")

    __table_args__ = (
        Index(
            "index_attempt_id",
            "time_created",
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<IndexAttempt(id={self.id!r}, "
            f"index_attempt_id={self.index_attempt_id!r}, "
            f"error_msg={self.error_msg!r})>"
            f"time_created={self.time_created!r}, "
        )


class DocumentByConnectorCredentialPair(Base):
    """Represents an indexing of a document by a specific connector / credential pair"""

    __tablename__ = "document_by_connector_credential_pair"

    id: Mapped[str] = mapped_column(ForeignKey("document.id"), primary_key=True)
    # TODO: transition this to use the ConnectorCredentialPair id directly
    connector_id: Mapped[int] = mapped_column(
        ForeignKey("connector.id"), primary_key=True
    )
    credential_id: Mapped[int] = mapped_column(
        ForeignKey("credential.id"), primary_key=True
    )

    connector: Mapped[Connector] = relationship(
        "Connector", back_populates="documents_by_connector"
    )
    credential: Mapped[Credential] = relationship(
        "Credential", back_populates="documents_by_credential"
    )

    __table_args__ = (
        Index(
            "idx_document_cc_pair_connector_credential",
            "connector_id",
            "credential_id",
            unique=False,
        ),
    )


"""
消息相关类
"""

class SearchDoc(Base):
    """搜索文档模型类
    存储检索到的文档片段状态，支持对话重放"""
    __tablename__ = "search_doc"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[str] = mapped_column(String)
    chunk_ind: Mapped[int] = mapped_column(Integer)
    semantic_id: Mapped[str] = mapped_column(String)
    link: Mapped[str | None] = mapped_column(String, nullable=True)
    blurb: Mapped[str] = mapped_column(String)
    boost: Mapped[int] = mapped_column(Integer)
    source_type: Mapped[DocumentSource] = mapped_column(
        Enum(DocumentSource, native_enum=False)
    )
    hidden: Mapped[bool] = mapped_column(Boolean)
    doc_metadata: Mapped[dict[str, str | list[str]]] = mapped_column(postgresql.JSONB())
    score: Mapped[float] = mapped_column(Float)
    match_highlights: Mapped[list[str]] = mapped_column(postgresql.ARRAY(String))
    # This is for the document, not this row in the table
    updated_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    primary_owners: Mapped[list[str] | None] = mapped_column(
        postgresql.ARRAY(String), nullable=True
    )
    secondary_owners: Mapped[list[str] | None] = mapped_column(
        postgresql.ARRAY(String), nullable=True
    )
    is_internet: Mapped[bool] = mapped_column(Boolean, default=False, nullable=True)

    is_relevant: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    relevance_explanation: Mapped[str | None] = mapped_column(String, nullable=True)

    chat_messages = relationship(
        "ChatMessage",
        secondary=ChatMessage__SearchDoc.__table__,
        back_populates="search_docs",
    )


class ToolCall(Base):
    """工具调用模型类
    记录单次工具调用的信息，包括参数和结果"""
    __tablename__ = "tool_call"

    id: Mapped[int] = mapped_column(primary_key=True)
    # 不使用外键因为我们希望可以删除工具但保留调用记录
    tool_id: Mapped[int] = mapped_column(Integer())
    tool_name: Mapped[str] = mapped_column(String())
    tool_arguments: Mapped[dict[str, JSON_ro]] = mapped_column(postgresql.JSONB())
    tool_result: Mapped[JSON_ro] = mapped_column(postgresql.JSONB())

    message_id: Mapped[int | None] = mapped_column(
        ForeignKey("chat_message.id"), nullable=False
    )

    # Update the relationship
    message: Mapped["ChatMessage"] = relationship(
        "ChatMessage",
        back_populates="tool_call",
        uselist=False,
    )


class ChatSession(Base):
    """聊天会话模型类
    管理用户的聊天对话，包含会话设置和消息历史"""
    __tablename__ = "chat_session"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid4
    )
    user_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("user.id", ondelete="CASCADE"), nullable=True
    )
    persona_id: Mapped[int | None] = mapped_column(
        ForeignKey("persona.id"), nullable=True
    )
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    # This chat created by OnyxBot
    onyxbot_flow: Mapped[bool] = mapped_column(Boolean, default=False)
    # Only ever set to True if system is set to not hard-delete chats
    deleted: Mapped[bool] = mapped_column(Boolean, default=False)
    # controls whether or not this conversation is viewable by others
    shared_status: Mapped[ChatSessionSharedStatus] = mapped_column(
        Enum(ChatSessionSharedStatus, native_enum=False),
        default=ChatSessionSharedStatus.PRIVATE,
    )
    folder_id: Mapped[int | None] = mapped_column(
        ForeignKey("chat_folder.id"), nullable=True
    )

    current_alternate_model: Mapped[str | None] = mapped_column(String, default=None)

    slack_thread_id: Mapped[str | None] = mapped_column(
        String, nullable=True, default=None
    )

    # the latest "overrides" specified by the user. These take precedence over
    # the attached persona. However, overrides specified directly in the
    # `send-message` call will take precedence over these.
    # NOTE: currently only used by the chat seeding flow, will be used in the
    # future once we allow users to override default values via the Chat UI
    # itself
    llm_override: Mapped[LLMOverride | None] = mapped_column(
        PydanticType(LLMOverride), nullable=True
    )
    prompt_override: Mapped[PromptOverride | None] = mapped_column(
        PydanticType(PromptOverride), nullable=True
    )
    time_updated: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )
    time_created: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    user: Mapped[User] = relationship("User", back_populates="chat_sessions")
    folder: Mapped["ChatFolder"] = relationship(
        "ChatFolder", back_populates="chat_sessions"
    )
    messages: Mapped[list["ChatMessage"]] = relationship(
        "ChatMessage", back_populates="chat_session", cascade="all, delete-orphan"
    )
    persona: Mapped["Persona"] = relationship("Persona")


class ChatMessage(Base):
    """聊天消息模型类
    存储单条聊天消息，包含用户输入和AI响应"""
    __tablename__ = "chat_message"

    id: Mapped[int] = mapped_column(primary_key=True)
    chat_session_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("chat_session.id")
    )

    alternate_assistant_id = mapped_column(
        Integer, ForeignKey("persona.id"), nullable=True
    )

    overridden_model: Mapped[str | None] = mapped_column(String, nullable=True)
    parent_message: Mapped[int | None] = mapped_column(Integer, nullable=True)
    latest_child_message: Mapped[int | None] = mapped_column(Integer, nullable=True)
    message: Mapped[str] = mapped_column(Text)
    rephrased_query: Mapped[str] = mapped_column(Text, nullable=True)
    # 如果为None，则不生成答案，这是仅向用户显示检索文档的特殊情况
    prompt_id: Mapped[int | None] = mapped_column(ForeignKey("prompt.id"))
    # 如果prompt为None，则token_count为0，因为该消息不会传入LLM的上下文中(不包含在消息历史中)
    token_count: Mapped[int] = mapped_column(Integer)
    message_type: Mapped[MessageType] = mapped_column(
        Enum(MessageType, native_enum=False)
    )
    # 将引用编号映射到SearchDoc id
    citations: Mapped[dict[int, int]] = mapped_column(postgresql.JSONB(), nullable=True)
    # 与此消息关联的文件(例如用户上传的用于提问的图片)
    files: Mapped[list[FileDescriptor] | None] = mapped_column(
        postgresql.JSONB(), nullable=True
    )
    # 仅适用于LLM
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    time_sent: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    chat_session: Mapped[ChatSession] = relationship("ChatSession")
    prompt: Mapped[Optional["Prompt"]] = relationship("Prompt")

    chat_message_feedbacks: Mapped[list["ChatMessageFeedback"]] = relationship(
        "ChatMessageFeedback",
        back_populates="chat_message",
    )

    document_feedbacks: Mapped[list["DocumentRetrievalFeedback"]] = relationship(
        "DocumentRetrievalFeedback",
        back_populates="chat_message",
    )
    search_docs: Mapped[list["SearchDoc"]] = relationship(
        "SearchDoc",
        secondary=ChatMessage__SearchDoc.__table__,
        back_populates="chat_messages",
        cascade="all, delete-orphan",
        single_parent=True,
    )

    tool_call: Mapped["ToolCall"] = relationship(
        "ToolCall",
        back_populates="message",
        uselist=False,
    )

    standard_answers: Mapped[list["StandardAnswer"]] = relationship(
        "StandardAnswer",
        secondary=ChatMessage__StandardAnswer.__table__,
        back_populates="chat_messages",
    )


class ChatFolder(Base):
    """聊天文件夹模型类
    用于组织和管理聊天会话"""
    __tablename__ = "chat_folder"

    id: Mapped[int] = mapped_column(primary_key=True)
    # 只有在关闭认证时才为空
    user_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("user.id", ondelete="CASCADE"), nullable=True
    )
    name: Mapped[str | None] = mapped_column(String, nullable=True)
    display_priority: Mapped[int] = mapped_column(Integer, nullable=True, default=0)

    user: Mapped[User] = relationship("User", back_populates="chat_folders")
    chat_sessions: Mapped[list["ChatSession"]] = relationship(
        "ChatSession", back_populates="folder"
    )

    def __lt__(self, other: Any) -> bool:
        """比较两个聊天文件夹的排序优先级
        优先按display_priority排序，相同则按id倒序排序"""
        if not isinstance(other, ChatFolder):
            return NotImplemented
        if self.display_priority == other.display_priority:
            # ID越大(越晚创建)显示越靠前
            return self.id > other.id
        return self.display_priority < other.display_priority


"""
反馈和日志相关类
"""

class DocumentRetrievalFeedback(Base):
    """文档检索反馈模型类
    记录用户对检索文档的反馈信息"""
    __tablename__ = "document_retrieval_feedback"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("document.id"))
    chat_message_id: Mapped[int | None] = mapped_column(
        ForeignKey("chat_message.id", ondelete="SET NULL"), nullable=True
    )
    document_rank: Mapped[int] = mapped_column(Integer)
    clicked: Mapped[bool] = mapped_column(Boolean, default=False)
    feedback: Mapped[SearchFeedbackType | None] = mapped_column(
        Enum(SearchFeedbackType, native_enum=False), nullable=True
    )

    chat_message: Mapped[ChatMessage] = relationship(
        "ChatMessage",
        back_populates="document_feedbacks",
        foreign_keys=[chat_message_id],
    )
    document: Mapped[Document] = relationship(
        "Document", back_populates="retrieval_feedbacks"
    )


class ChatMessageFeedback(Base):
    """聊天消息反馈模型类
    存储用户对聊天消息的反馈"""
    __tablename__ = "chat_feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    chat_message_id: Mapped[int | None] = mapped_column(
        ForeignKey("chat_message.id", ondelete="SET NULL"), nullable=True
    )
    is_positive: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    required_followup: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    feedback_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    predefined_feedback: Mapped[str | None] = mapped_column(String, nullable=True)

    chat_message: Mapped[ChatMessage] = relationship(
        "ChatMessage",
        back_populates="chat_message_feedbacks",
        foreign_keys=[chat_message_id],
    )


class LLMProvider(Base):
    """语言模型提供者模型类
    管理不同LLM服务提供者的配置和认证信息"""
    __tablename__ = "llm_provider"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, unique=True)
    provider: Mapped[str] = mapped_column(String)
    api_key: Mapped[str | None] = mapped_column(EncryptedString(), nullable=True)
    api_base: Mapped[str | None] = mapped_column(String, nullable=True)
    api_version: Mapped[str | None] = mapped_column(String, nullable=True)
    # 需要在推理时传递给LLM提供者的自定义配置
    # (例如 AWS Bedrock 需要的 AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY 等)
    custom_config: Mapped[dict[str, str] | None] = mapped_column(
        postgresql.JSONB(), nullable=True
    )
    default_model_name: Mapped[str] = mapped_column(String)
    fast_default_model_name: Mapped[str | None] = mapped_column(String, nullable=True)

    # 实际展示给用户的模型
    # 如果为空，则在应用逻辑中默认展示所有模型
    display_model_names: Mapped[list[str] | None] = mapped_column(
        postgresql.ARRAY(String), nullable=True
    )
    # 该提供者可用的LLM列表。只在非默认提供者时需要。
    # 如果是默认提供者，则从options.py文件获取LLM选项。
    # 如果需要，可以在未来拆分为独立的表。
    model_names: Mapped[list[str] | None] = mapped_column(
        postgresql.ARRAY(String), nullable=True
    )

    deployment_name: Mapped[str | None] = mapped_column(String, nullable=True)

    # should only be set for a single provider
    is_default_provider: Mapped[bool | None] = mapped_column(Boolean, unique=True)
    # EE only
    is_public: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    groups: Mapped[list["UserGroup"]] = relationship(
        "UserGroup",
        secondary="llm_provider__user_group",
        viewonly=True,
    )


class CloudEmbeddingProvider(Base):
    """云端嵌入模型提供者模型类
    管理文本嵌入服务的配置信息"""
    __tablename__ = "embedding_provider"

    provider_type: Mapped[EmbeddingProvider] = mapped_column(
        Enum(EmbeddingProvider), primary_key=True
    )
    api_url: Mapped[str | None] = mapped_column(String, nullable=True)
    api_key: Mapped[str | None] = mapped_column(EncryptedString())
    api_version: Mapped[str | None] = mapped_column(String, nullable=True)
    deployment_name: Mapped[str | None] = mapped_column(String, nullable=True)

    search_settings: Mapped[list["SearchSettings"]] = relationship(
        "SearchSettings",
        back_populates="cloud_provider",
        foreign_keys="SearchSettings.provider_type",
    )

    def __repr__(self) -> str:
        return f"<EmbeddingProvider(type='{self.provider_type}')>"


class DocumentSet(Base):
    __tablename__ = "document_set"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, unique=True)
    description: Mapped[str | None] = mapped_column(String)
    user_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("user.id", ondelete="CASCADE"), nullable=True
    )
    # Whether changes to the document set have been propagated
    is_up_to_date: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    # If `False`, then the document set is not visible to users who are not explicitly
    # given access to it either via the `users` or `groups` relationships
    is_public: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    connector_credential_pairs: Mapped[list[ConnectorCredentialPair]] = relationship(
        "ConnectorCredentialPair",
        secondary=DocumentSet__ConnectorCredentialPair.__table__,
        primaryjoin=(
            (DocumentSet__ConnectorCredentialPair.document_set_id == id)
            & (DocumentSet__ConnectorCredentialPair.is_current.is_(True))
        ),
        secondaryjoin=(
            DocumentSet__ConnectorCredentialPair.connector_credential_pair_id
            == ConnectorCredentialPair.id
        ),
        back_populates="document_sets",
        overlaps="document_set",
    )
    personas: Mapped[list["Persona"]] = relationship(
        "Persona",
        secondary=Persona__DocumentSet.__table__,
        back_populates="document_sets",
    )
    # Other users with access
    users: Mapped[list[User]] = relationship(
        "User",
        secondary=DocumentSet__User.__table__,
        viewonly=True,
    )
    # EE only
    groups: Mapped[list["UserGroup"]] = relationship(
        "UserGroup",
        secondary="document_set__user_group",
        viewonly=True,
    )


class Prompt(Base):
    __tablename__ = "prompt"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("user.id", ondelete="CASCADE"), nullable=True
    )
    name: Mapped[str] = mapped_column(String)
    description: Mapped[str] = mapped_column(String)
    system_prompt: Mapped[str] = mapped_column(Text)
    task_prompt: Mapped[str] = mapped_column(Text)
    include_citations: Mapped[bool] = mapped_column(Boolean, default=True)
    datetime_aware: Mapped[bool] = mapped_column(Boolean, default=True)
    # Default prompts are configured via backend during deployment
    # Treated specially (不能被用户编辑等)
    default_prompt: Mapped[bool] = mapped_column(Boolean, default=False)
    deleted: Mapped[bool] = mapped_column(Boolean, default=False)

    user: Mapped[User] = relationship("User", back_populates="prompts")
    personas: Mapped[list["Persona"]] = relationship(
        "Persona",
        secondary=Persona__Prompt.__table__,
        back_populates="prompts",
    )


class Tool(Base):
    """工具模型类
    定义系统可用的各种工具和功能"""
    __tablename__ = "tool"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=True)
    # ID of the tool in the codebase, only applies for in-code tools.
    # tools defined via the UI will have this as None
    in_code_tool_id: Mapped[str | None] = mapped_column(String, nullable=True)
    display_name: Mapped[str] = mapped_column(String, nullable=True)

    # OpenAPI scheme for the tool. Only applies to tools defined via the UI.
    openapi_schema: Mapped[dict[str, Any] | None] = mapped_column(
        postgresql.JSONB(), nullable=True
    )
    custom_headers: Mapped[list[HeaderItemDict] | None] = mapped_column(
        postgresql.JSONB(), nullable=True
    )
    # user who created / owns the tool. Will be None for built-in tools.
    user_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("user.id", ondelete="CASCADE"), nullable=True
    )

    user: Mapped[User | None] = relationship("User", back_populates="custom_tools")
    # Relationship to Persona through the association table
    personas: Mapped[list["Persona"]] = relationship(
        "Persona",
        secondary=Persona__Tool.__table__,
        back_populates="tools",
    )


class StarterMessage(TypedDict):
    """NOTE: is a `TypedDict` so it can be used as a type hint for a JSONB column
    in Postgres"""

    name: str
    message: str


class StarterMessageModel(BaseModel):
    name: str
    message: str


class Persona(Base):
    __tablename__ = "persona"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("user.id", ondelete="CASCADE"), nullable=True
    )
    name: Mapped[str] = mapped_column(String)
    description: Mapped[str] = mapped_column(String)
    # Number of chunks to pass to the LLM for generation.
    num_chunks: Mapped[float | None] = mapped_column(Float, nullable=True)
    chunks_above: Mapped[int] = mapped_column(Integer)
    chunks_below: Mapped[int] = mapped_column(Integer)
    # Pass every chunk through LLM for evaluation, fairly expensive
    # Can be turned off globally by admin, in which case, this setting is ignored
    llm_relevance_filter: Mapped[bool] = mapped_column(Boolean)
    # Enables using LLM to extract time and source type filters
    # Can also be admin disabled globally
    llm_filter_extraction: Mapped[bool] = mapped_column(Boolean)
    recency_bias: Mapped[RecencyBiasSetting] = mapped_column(
        Enum(RecencyBiasSetting, native_enum=False)
    )
    category_id: Mapped[int | None] = mapped_column(
        ForeignKey("persona_category.id"), nullable=True
    )
    # Allows the Persona to specify a different LLM version than is controlled
    # globablly via env variables. For flexibility, validity is not currently enforced
    # NOTE: only is applied on the actual response generation - is not used for things like
    # auto-detected time filters, relevance filters, etc.
    llm_model_provider_override: Mapped[str | None] = mapped_column(
        String, nullable=True
    )
    llm_model_version_override: Mapped[str | None] = mapped_column(
        String, nullable=True
    )
    starter_messages: Mapped[list[StarterMessage] | None] = mapped_column(
        postgresql.JSONB(), nullable=True
    )
    search_start_date: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), default=None
    )
    # Built-in personas are configured via backend during deployment
    # Treated specially (不能被用户编辑等)
    builtin_persona: Mapped[bool] = mapped_column(Boolean, default=False)

    # Default personas are personas created by admins and are automatically added
    # to all users' assistants list.
    is_default_persona: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )
    # controls whether the persona is available to be selected by users
    is_visible: Mapped[bool] = mapped_column(Boolean, default=True)
    # controls the ordering of personas in the UI
    # higher priority personas are displayed first, ties are resolved by the ID,
    # where lower value IDs (e.g. created earlier) are displayed first
    display_priority: Mapped[int | None] = mapped_column(
        Integer, nullable=True, default=None
    )
    deleted: Mapped[bool] = mapped_column(Boolean, default=False)

    uploaded_image_id: Mapped[str | None] = mapped_column(String, nullable=True)
    icon_color: Mapped[str | None] = mapped_column(String, nullable=True)
    icon_shape: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # These are only defaults, users can select from all if desired
    prompts: Mapped[list[Prompt]] = relationship(
        "Prompt",
        secondary=Persona__Prompt.__table__,
        back_populates="personas",
    )
    # These are only defaults, users can select from all if desired
    document_sets: Mapped[list[DocumentSet]] = relationship(
        "DocumentSet",
        secondary=Persona__DocumentSet.__table__,
        back_populates="personas",
    )
    tools: Mapped[list[Tool]] = relationship(
        "Tool",
        secondary=Persona__Tool.__table__,
        back_populates="personas",
    )
    # Owner
    user: Mapped[User | None] = relationship("User", back_populates="personas")
    # Other users with access
    users: Mapped[list[User]] = relationship(
        "User",
        secondary=Persona__User.__table__,
        viewonly=True,
    )
    # EE only
    is_public: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    groups: Mapped[list["UserGroup"]] = relationship(
        "UserGroup",
        secondary="persona__user_group",
        viewonly=True,
    )
    category: Mapped["PersonaCategory"] = relationship(
        "PersonaCategory", back_populates="personas"
    )

    # Default personas loaded via yaml cannot have the same name
    __table_args__ = (
        Index(
            "_builtin_persona_name_idx",
            "name",
            unique=True,
            postgresql_where=(builtin_persona == True),  # noqa: E712
        ),
    )


class PersonaCategory(Base):
    """人格类别模型类
    用于对不同的人格助手进行分类"""
    __tablename__ = "persona_category"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String, unique=True)
    description: Mapped[str | None] = mapped_column(String, nullable=True)
    personas: Mapped[list["Persona"]] = relationship(
        "Persona", back_populates="category"
    )


AllowedAnswerFilters = (
    Literal["well_answered_postfilter"] | Literal["questionmark_prefilter"]
)


class ChannelConfig(TypedDict):
    """NOTE: is a `TypedDict` so it can be used as a type hint for a JSONB column
    in Postgres"""

    channel_name: str
    respond_tag_only: NotRequired[bool]  # defaults to False
    respond_to_bots: NotRequired[bool]  # defaults to False
    respond_member_group_list: NotRequired[list[str]]
    answer_filters: NotRequired[list[AllowedAnswerFilters]]
    # 如果为None则不进行后续跟进
    # 如果为空列表，则不带标签进行后续跟进
    follow_up_tags: NotRequired[list[str]]
    show_continue_in_web_ui: NotRequired[bool]  # defaults to False


class SlackChannelConfig(Base):
    """Slack频道配置模型类
    管理Slack集成的频道配置信息"""
    __tablename__ = "slack_channel_config"

    id: Mapped[int] = mapped_column(primary_key=True)
    slack_bot_id: Mapped[int] = mapped_column(
        ForeignKey("slack_bot.id"), nullable=False
    )
    persona_id: Mapped[int | None] = mapped_column(
        ForeignKey("persona.id"), nullable=True
    )
    # JSON for flexibility. Contains things like: channel name, team members, etc.
    channel_config: Mapped[ChannelConfig] = mapped_column(
        postgresql.JSONB(), nullable=False
    )

    enable_auto_filters: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )

    persona: Mapped[Persona | None] = relationship("Persona")
    slack_bot: Mapped["SlackBot"] = relationship(
        "SlackBot",
        back_populates="slack_channel_configs",
    )
    standard_answer_categories: Mapped[list["StandardAnswerCategory"]] = relationship(
        "StandardAnswerCategory",
        secondary=SlackChannelConfig__StandardAnswerCategory.__table__,
        back_populates="slack_channel_configs",
    )


class SlackBot(Base):
    """Slack机器人模型类
    管理Slack机器人的配置和认证信息"""
    __tablename__ = "slack_bot"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)

    bot_token: Mapped[str] = mapped_column(EncryptedString(), unique=True)
    app_token: Mapped[str] = mapped_column(EncryptedString(), unique=True)

    slack_channel_configs: Mapped[list[SlackChannelConfig]] = relationship(
        "SlackChannelConfig",
        back_populates="slack_bot",
        cascade="all, delete-orphan",
    )


class Milestone(Base):
    """里程碑模型类
    跟踪部署过程中的重要事件和进展"""
    # This table is used to track significant events for a deployment towards finding value
    # The table is currently not used for features but it may be used in the future to inform
    # users about the product features and encourage usage/exploration.
    __tablename__ = "milestone"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid4
    )
    user_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("user.id", ondelete="CASCADE"), nullable=True
    )
    event_type: Mapped[MilestoneRecordType] = mapped_column(String)
    # Need to track counts and specific ids of certain events to know if the Milestone has been reached
    event_tracker: Mapped[dict | None] = mapped_column(
        postgresql.JSONB(), nullable=True
    )
    time_created: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    user: Mapped[User | None] = relationship("User")

    __table_args__ = (UniqueConstraint("event_type", name="uq_milestone_event_type"),)


class TaskQueueState(Base):
    """任务队列状态模型类
    管理后台任务的执行状态"""
    # Currently refers to Celery Tasks
    __tablename__ = "task_queue_jobs"

    id: Mapped[int] = mapped_column(primary_key=True)
    # Celery task id. currently only for readability/diagnostics
    task_id: Mapped[str] = mapped_column(String)
    # For any job type, this would be the same
    task_name: Mapped[str] = mapped_column(String)
    # Note that if the task dies, this won't necessarily be marked FAILED correctly
    status: Mapped[TaskStatus] = mapped_column(Enum(TaskStatus, native_enum=False))
    start_time: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
    register_time: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class KVStore(Base):
    """键值存储模型类
    通用键值对存储，支持加密存储"""
    __tablename__ = "key_value_store"

    key: Mapped[str] = mapped_column(String, primary_key=True)
    value: Mapped[JSON_ro] = mapped_column(postgresql.JSONB(), nullable=True)
    encrypted_value: Mapped[JSON_ro] = mapped_column(EncryptedJson(), nullable=True)


class PGFileStore(Base):
    """PostgreSQL文件存储模型类
    用于在数据库中存储文件数据"""
    __tablename__ = "file_store"

    file_name: Mapped[str] = mapped_column(String, primary_key=True)
    display_name: Mapped[str] = mapped_column(String, nullable=True)
    file_origin: Mapped[FileOrigin] = mapped_column(Enum(FileOrigin, native_enum=False))
    file_type: Mapped[str] = mapped_column(String, default="text/plain")
    file_metadata: Mapped[JSON_ro] = mapped_column(postgresql.JSONB(), nullable=True)
    lobj_oid: Mapped[int] = mapped_column(Integer, nullable=False)


"""
************************************************************************
Enterprise Edition Models
************************************************************************

These models are only used in Enterprise Edition only features in Onyx.
They are kept here to simplify the codebase and avoid having different assumptions
on the shape of data being passed around between the MIT and EE versions of Onyx.

In the MIT version of Onyx, assume these tables are always empty.
"""

"""
企业版特有类
"""

class SamlAccount(Base):
    """SAML账户模型类
    用于企业版SAML认证集成"""
    __tablename__ = "saml"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("user.id", ondelete="CASCADE"), unique=True
    )
    encrypted_cookie: Mapped[str] = mapped_column(Text, unique=True)
    expires_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    user: Mapped[User] = relationship("User")


class User__UserGroup(Base):
    """用户与用户组关联表
    用于建立用户和用户组之间的多对多关系"""
    __tablename__ = "user__user_group"

    is_curator: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    user_group_id: Mapped[int] = mapped_column(
        ForeignKey("user_group.id"), primary_key=True
    )
    user_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("user.id", ondelete="CASCADE"), primary_key=True, nullable=True
    )


class UserGroup__ConnectorCredentialPair(Base):
    """用户组与连接器凭据对关联表
    用于管理用户组和连接器凭据对之间的关系"""
    __tablename__ = "user_group__connector_credential_pair"

    user_group_id: Mapped[int] = mapped_column(
        ForeignKey("user_group.id"), primary_key=True
    )
    cc_pair_id: Mapped[int] = mapped_column(
        ForeignKey("connector_credential_pair.id"), primary_key=True
    )
    # 如果为True，则是用户组当前状态的一部分
    # 如果为False，则是用户组之前状态的一部分
    # 当用户组更新时，is_current=False的行应该被删除
    # 当UserGroup.is_up_to_date为True时，不应该存在这样的行
    is_current: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        primary_key=True,
    )

    cc_pair: Mapped[ConnectorCredentialPair] = relationship(
        "ConnectorCredentialPair",
    )


class Persona__UserGroup(Base):
    __tablename__ = "persona__user_group"

    persona_id: Mapped[int] = mapped_column(ForeignKey("persona.id"), primary_key=True)
    user_group_id: Mapped[int] = mapped_column(
        ForeignKey("user_group.id"), primary_key=True
    )


class LLMProvider__UserGroup(Base):
    """LLM提供者与用户组关联表
    用于管理LLM提供者和用户组之间的访问权限"""
    __tablename__ = "llm_provider__user_group"

    llm_provider_id: Mapped[int] = mapped_column(
        ForeignKey("llm_provider.id"), primary_key=True
    )
    user_group_id: Mapped[int] = mapped_column(
        ForeignKey("user_group.id"), primary_key=True
    )


class DocumentSet__UserGroup(Base):
    """文档集与用户组关联表
    用于管理文档集和用户组之间的访问权限"""
    __tablename__ = "document_set__user_group"

    document_set_id: Mapped[int] = mapped_column(
        ForeignKey("document_set.id"), primary_key=True
    )
    user_group_id: Mapped[int] = mapped_column(
        ForeignKey("user_group.id"), primary_key=True
    )


class Credential__UserGroup(Base):
    """凭据与用户组关联表
    用于管理凭据和用户组之间的访问权限"""
    __tablename__ = "credential__user_group"

    credential_id: Mapped[int] = mapped_column(
        ForeignKey("credential.id"), primary_key=True
    )
    user_group_id: Mapped[int] = mapped_column(
        ForeignKey("user_group.id"), primary_key=True
    )


class UserGroup(Base):
    """用户组模型类
    企业版用户组管理功能"""
    __tablename__ = "user_group"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String, unique=True)
    # whether or not changes to the UserGroup have been propagated to Vespa
    is_up_to_date: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    # tell the sync job to clean up the group
    is_up_for_deletion: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )

    users: Mapped[list[User]] = relationship(
        "User",
        secondary=User__UserGroup.__table__,
    )
    user_group_relationships: Mapped[list[User__UserGroup]] = relationship(
        "User__UserGroup",
        viewonly=True,
    )
    cc_pairs: Mapped[list[ConnectorCredentialPair]] = relationship(
        "ConnectorCredentialPair",
        secondary=UserGroup__ConnectorCredentialPair.__table__,
        viewonly=True,
    )
    cc_pair_relationships: Mapped[
        list[UserGroup__ConnectorCredentialPair]
    ] = relationship(
        "UserGroup__ConnectorCredentialPair",
        viewonly=True,
    )
    personas: Mapped[list[Persona]] = relationship(
        "Persona",
        secondary=Persona__UserGroup.__table__,
        viewonly=True,
    )
    document_sets: Mapped[list[DocumentSet]] = relationship(
        "DocumentSet",
        secondary=DocumentSet__UserGroup.__table__,
        viewonly=True,
    )
    credentials: Mapped[list[Credential]] = relationship(
        "Credential",
        secondary=Credential__UserGroup.__table__,
    )


"""Tables related to Token Rate Limiting
NOTE: `TokenRateLimit` is partially an MIT feature (global rate limit)
"""

class TokenRateLimit(Base):
    """令牌速率限制模型类
    管理API调用的速率限制策略"""
    __tablename__ = "token_rate_limit"

    id: Mapped[int] = mapped_column(primary_key=True)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    token_budget: Mapped[int] = mapped_column(Integer, nullable=False)
    period_hours: Mapped[int] = mapped_column(Integer, nullable=False)
    scope: Mapped[TokenRateLimitScope] = mapped_column(
        Enum(TokenRateLimitScope, native_enum=False)
    )
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class TokenRateLimit__UserGroup(Base):
    """令牌限制与用户组关联表
    用于管理令牌使用限制和用户组之间的关系"""
    __tablename__ = "token_rate_limit__user_group"

    rate_limit_id: Mapped[int] = mapped_column(
        ForeignKey("token_rate_limit.id"), primary_key=True
    )
    user_group_id: Mapped[int] = mapped_column(
        ForeignKey("user_group.id"), primary_key=True
    )


class StandardAnswerCategory(Base):
    """标准答案类别模型类
    用于对标准答案进行分类管理"""
    __tablename__ = "standard_answer_category"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String, unique=True)
    standard_answers: Mapped[list["StandardAnswer"]] = relationship(
        "StandardAnswer",
        secondary=StandardAnswer__StandardAnswerCategory.__table__,
        back_populates="categories",
    )
    slack_channel_configs: Mapped[list["SlackChannelConfig"]] = relationship(
        "SlackChannelConfig",
        secondary=SlackChannelConfig__StandardAnswerCategory.__table__,
        back_populates="standard_answer_categories",
    )


class StandardAnswer(Base):
    """标准答案模型类
    存储预定义的问答对，用于快速响应常见问题"""
    __tablename__ = "standard_answer"

    id: Mapped[int] = mapped_column(primary_key=True)
    keyword: Mapped[str] = mapped_column(String)
    answer: Mapped[str] = mapped_column(String)
    active: Mapped[bool] = mapped_column(Boolean)
    match_regex: Mapped[bool] = mapped_column(Boolean)
    match_any_keywords: Mapped[bool] = mapped_column(Boolean)

    __table_args__ = (
        Index(
            "unique_keyword_active",
            keyword,
            active,
            unique=True,
            postgresql_where=(active == True),  # noqa: E712
        ),
    )

    categories: Mapped[list[StandardAnswerCategory]] = relationship(
        "StandardAnswerCategory",
        secondary=StandardAnswer__StandardAnswerCategory.__table__,
        back_populates="standard_answers",
    )
    chat_messages: Mapped[list[ChatMessage]] = relationship(
        "ChatMessage",
        secondary=ChatMessage__StandardAnswer.__table__,
        back_populates="standard_answers",
    )


"""Tables related to Permission Sync"""


class User__ExternalUserGroupId(Base):
    """用户与外部用户组ID映射表
    将用户映射到其所有外部组，以便在查询时进行ACL列表匹配
    用户级别的权限可以通过直接将Onyx用户添加到文档ACL列表来处理"""
    __tablename__ = "user__external_user_group_id"

    user_id: Mapped[UUID] = mapped_column(ForeignKey("user.id"), primary_key=True)
    # These group ids have been prefixed by the source type
    external_user_group_id: Mapped[str] = mapped_column(String, primary_key=True)
    cc_pair_id: Mapped[int] = mapped_column(
        ForeignKey("connector_credential_pair.id"), primary_key=True
    )


class UsageReport(Base):
    """This stores metadata about usage reports generated by admin including user who generated
    them as well las the period they cover. The actual zip file of the report is stored as a lo
    using the PGFileStore
    """

    __tablename__ = "usage_reports"

    id: Mapped[int] = mapped_column(primary_key=True)
    report_name: Mapped[str] = mapped_column(ForeignKey("file_store.file_name"))

    # 如果为None，则表示自动生成的报告
    requestor_user_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("user.id", ondelete="CASCADE"), nullable=True
    )
    time_created: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    period_from: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
    period_to: Mapped[datetime.datetime | None] = mapped_column(DateTime(timezone=True))

    requestor = relationship("User")
    file = relationship("PGFileStore")


"""
Multi-tenancy related tables
"""


class PublicBase(DeclarativeBase):
    __abstract__ = True


class UserTenantMapping(Base):
    """用户租户映射模型类
    管理多租户环境下用户与租户的对应关系"""
    __tablename__ = "user_tenant_mapping"
    __table_args__ = (
        UniqueConstraint("email", "tenant_id", name="uq_user_tenant"),
        {"schema": "public"},
    )

    email: Mapped[str] = mapped_column(String, nullable=False, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String, nullable=False)
