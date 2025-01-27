"""
这个文件定义了文档系统相关的数据模型类。
包含了文档同步状态、连接器配置、凭证管理、索引任务等相关的数据模型。
主要用于处理文档的同步、索引和管理功能。
"""

from datetime import datetime
from typing import Any
from typing import Generic
from typing import TypeVar
from uuid import UUID

from pydantic import BaseModel
from pydantic import Field

from onyx.configs.app_configs import MASK_CREDENTIAL_PREFIX
from onyx.configs.constants import DocumentSource
from onyx.connectors.models import DocumentErrorSummary
from onyx.connectors.models import InputType
from onyx.db.enums import AccessType
from onyx.db.enums import ConnectorCredentialPairStatus
from onyx.db.models import Connector
from onyx.db.models import ConnectorCredentialPair
from onyx.db.models import Credential
from onyx.db.models import Document as DbDocument
from onyx.db.models import IndexAttempt
from onyx.db.models import IndexAttemptError as DbIndexAttemptError
from onyx.db.models import IndexingStatus
from onyx.db.models import TaskStatus
from onyx.server.models import FullUserSnapshot
from onyx.server.models import InvitedUserSnapshot
from onyx.server.utils import mask_credential_dict


class DocumentSyncStatus(BaseModel):
    """文档同步状态模型，用于跟踪文档的同步状态"""
    doc_id: str  # 文档ID
    last_synced: datetime | None  # 最后同步时间
    last_modified: datetime | None  # 最后修改时间

    @classmethod
    def from_model(cls, doc: DbDocument) -> "DocumentSyncStatus":
        """
        从数据库文档模型创建同步状态对象
        Args:
            doc: 数据库文档模型
        Returns:
            DocumentSyncStatus: 文档同步状态对象
        """
        return DocumentSyncStatus(
            doc_id=doc.id,
            last_synced=doc.last_synced,
            last_modified=doc.last_modified,
        )


class DocumentInfo(BaseModel):
    """文档信息模型，包含文档的基本统计信息"""
    num_chunks: int  # 文档分块数量
    num_tokens: int  # 文档令牌数量


class ChunkInfo(BaseModel):
    """文档块信息模型"""
    content: str  # 块内容
    num_tokens: int  # 块中的令牌数量


class DeletionAttemptSnapshot(BaseModel):
    """删除尝试快照模型,用于记录删除操作的状态"""
    connector_id: int  # 连接器ID
    credential_id: int  # 凭证ID 
    status: TaskStatus  # 任务状态


class ConnectorBase(BaseModel):
    """连接器基础模型"""
    name: str  # 连接器名称
    source: DocumentSource  # 文档来源
    input_type: InputType  # 输入类型
    connector_specific_config: dict[str, Any]  # 连接器特定配置
    # In seconds, None for one time index with no refresh
    # 刷新频率(秒)，None表示一次性索引无需刷新
    refresh_freq: int | None = None
    prune_freq: int | None = None  # 清理频率
    indexing_start: datetime | None = None  # 索引开始时间


class ConnectorUpdateRequest(ConnectorBase):
    """连接器更新请求模型"""
    access_type: AccessType  # 访问类型
    groups: list[int] = Field(default_factory=list)  # 组ID列表

    def to_connector_base(self) -> ConnectorBase:
        """
        将更新请求转换为基础连接器模型
        Returns:
            ConnectorBase: 基础连接器模型
        """
        return ConnectorBase(**self.model_dump(exclude={"access_type", "groups"}))


class ConnectorSnapshot(ConnectorBase):
    """连接器快照模型,包含连接器的完整信息"""
    id: int  # 连接器ID
    credential_ids: list[int]  # 关联的凭证ID列表
    time_created: datetime  # 创建时间
    time_updated: datetime  # 更新时间
    source: DocumentSource  # 数据来源

    @classmethod
    def from_connector_db_model(cls, connector: Connector) -> "ConnectorSnapshot":
        return ConnectorSnapshot(
            id=connector.id,
            name=connector.name,
            source=connector.source,
            input_type=connector.input_type,
            connector_specific_config=connector.connector_specific_config,
            refresh_freq=connector.refresh_freq,
            prune_freq=connector.prune_freq,
            credential_ids=[
                association.credential.id for association in connector.credentials
            ],
            indexing_start=connector.indexing_start,
            time_created=connector.time_created,
            time_updated=connector.time_updated,
        )


class CredentialSwapRequest(BaseModel):
    """凭证交换请求模型,用于更换连接器使用的凭证"""
    new_credential_id: int  # 新凭证ID
    connector_id: int  # 连接器ID


class CredentialDataUpdateRequest(BaseModel):
    """凭证数据更新请求模型"""
    name: str  # 凭证名称
    credential_json: dict[str, Any]  # 凭证JSON数据


class CredentialBase(BaseModel):
    """凭证基础模型"""
    credential_json: dict[str, Any]  # 凭证JSON数据
    # if `true`, then all Admins will have access to the credential
    # 如果为true,则所有管理员都可以访问该凭证
    admin_public: bool  # 是否对管理员公开
    source: DocumentSource  # 数据来源
    name: str | None = None  # 凭证名称
    curator_public: bool = False  # 是否对策展人公开
    groups: list[int] = Field(default_factory=list)  # 组ID列表


class CredentialSnapshot(CredentialBase):
    """凭证快照模型,包含凭证的完整信息"""
    id: int  # 凭证ID
    user_id: UUID | None  # 用户ID
    time_created: datetime  # 创建时间
    time_updated: datetime  # 更新时间

    @classmethod
    def from_credential_db_model(cls, credential: Credential) -> "CredentialSnapshot":
        """
        从数据库凭证模型创建快照对象
        Args:
            credential: 数据库中的凭证模型
        Returns:
            CredentialSnapshot: 凭证快照对象
        """
        return CredentialSnapshot(
            id=credential.id,
            credential_json=(
                mask_credential_dict(credential.credential_json)
                if MASK_CREDENTIAL_PREFIX and credential.credential_json
                else credential.credential_json
            ),
            user_id=credential.user_id,
            admin_public=credential.admin_public,
            time_created=credential.time_created,
            time_updated=credential.time_updated,
            source=credential.source or DocumentSource.NOT_APPLICABLE,
            name=credential.name,
            curator_public=credential.curator_public,
        )


class IndexAttemptSnapshot(BaseModel):
    """索引尝试快照模型,记录索引操作的状态和结果"""
    id: int  # 索引尝试ID
    status: IndexingStatus | None  # 索引状态
    new_docs_indexed: int  # 仅包含完全新的文档数量 / only includes completely new docs
    total_docs_indexed: int  # 包含更新的文档总数 / includes docs that are updated
    docs_removed_from_index: int  # 从索引中移除的文档数
    error_msg: str | None  # 错误信息
    error_count: int  # 错误数量
    full_exception_trace: str | None  # 完整异常堆栈
    time_started: str | None  # 开始时间
    time_updated: str  # 更新时间

    @classmethod
    def from_index_attempt_db_model(
        cls, index_attempt: IndexAttempt
    ) -> "IndexAttemptSnapshot":
        """
        从数据库索引尝试模型创建快照对象
        Args:
            index_attempt: 数据库中的索引尝试模型
        Returns:
            IndexAttemptSnapshot: 索引尝试快照对象
        """
        return IndexAttemptSnapshot(
            id=index_attempt.id,
            status=index_attempt.status,
            new_docs_indexed=index_attempt.new_docs_indexed or 0,
            total_docs_indexed=index_attempt.total_docs_indexed or 0,
            docs_removed_from_index=index_attempt.docs_removed_from_index or 0,
            error_msg=index_attempt.error_msg,
            error_count=len(index_attempt.error_rows),
            full_exception_trace=index_attempt.full_exception_trace,
            time_started=(
                index_attempt.time_started.isoformat()
                if index_attempt.time_started
                else None
            ),
            time_updated=index_attempt.time_updated.isoformat(),
        )


class IndexAttemptError(BaseModel):
    """索引尝试错误模型,记录索引过程中发生的错误"""
    id: int  # 错误ID
    index_attempt_id: int | None  # 关联的索引尝试ID
    batch_number: int | None  # 批次号
    doc_summaries: list[DocumentErrorSummary]  # 文档错误摘要列表
    error_msg: str | None  # 错误信息
    traceback: str | None  # 错误堆栈
    time_created: str  # 创建时间

    @classmethod
    def from_db_model(cls, error: DbIndexAttemptError) -> "IndexAttemptError":
        """
        从数据库错误模型创建错误对象
        Args:
            error: 数据库中的错误模型
        Returns:
            IndexAttemptError: 索引尝试错误对象
        """
        doc_summaries = [
            DocumentErrorSummary.from_dict(summary) for summary in error.doc_summaries
        ]
        return IndexAttemptError(
            id=error.id,
            index_attempt_id=error.index_attempt_id,
            batch_number=error.batch,
            doc_summaries=doc_summaries,
            error_msg=error.error_msg,
            traceback=error.traceback,
            time_created=error.time_created.isoformat(),
        )


# These are the types currently supported by the pagination hook
# More api endpoints can be refactored and be added here for use with the pagination hook
PaginatedType = TypeVar(
    "PaginatedType",
    IndexAttemptSnapshot,
    FullUserSnapshot,
    InvitedUserSnapshot,
)


class PaginatedReturn(BaseModel, Generic[PaginatedType]):
    """分页返回模型"""
    items: list[PaginatedType]  # 分页项目列表
    total_items: int  # 总项目数


class CCPairFullInfo(BaseModel):
    """
    连接器-凭证对完整信息模型
    用于展示连接器和凭证配对的详细信息
    """
    id: int  # 配对ID
    name: str  # 配对名称
    status: ConnectorCredentialPairStatus  # 配对状态
    num_docs_indexed: int
    connector: ConnectorSnapshot
    credential: CredentialSnapshot
    number_of_index_attempts: int
    last_index_attempt_status: IndexingStatus | None
    latest_deletion_attempt: DeletionAttemptSnapshot | None
    access_type: AccessType
    is_editable_for_current_user: bool
    deletion_failure_message: str | None
    indexing: bool
    creator: UUID | None
    creator_email: str | None

    @classmethod
    def from_models(
        cls,
        cc_pair_model: ConnectorCredentialPair,
        latest_deletion_attempt: DeletionAttemptSnapshot | None,
        number_of_index_attempts: int,
        last_index_attempt: IndexAttempt | None,
        num_docs_indexed: int,  # not ideal, but this must be computed separately
        is_editable_for_current_user: bool,
        indexing: bool,
    ) -> "CCPairFullInfo":
        """
        从数据库模型创建完整信息对象
        Args:
            cc_pair_model: 连接器-凭证对模型
            latest_deletion_attempt: 最新删除尝试
            number_of_index_attempts: 索引尝试次数
            last_index_attempt: 最后索引尝试
            num_docs_indexed: 已索引文档数
            is_editable为当前用户: 当前用户是否可编辑
            indexing: 是否正在索引
        Returns:
            CCPairFullInfo: 完整信息对象
        """
        # figure out if we need to artificially deflate the number of docs indexed.
        # This is required since the total number of docs indexed by a CC Pair is
        # updated before the new docs for an indexing attempt. If we don't do this,
        # there is a mismatch between these two numbers which may confuse users.
        # 判断是否需要人为减少已索引文档数。
        # 这是必要的，因为CC对的总索引文档数在新文档索引尝试之前就更新了。
        # 如果不这样做，这两个数字之间可能会不匹配，可能会使用户感到困惑。
        last_indexing_status = last_index_attempt.status if last_index_attempt else None
        if (
            last_indexing_status == IndexingStatus.SUCCESS
            and number_of_index_attempts == 1
            and last_index_attempt
            and last_index_attempt.new_docs_indexed
        ):
            num_docs_indexed = (
                last_index_attempt.new_docs_indexed if last_index_attempt else 0
            )

        return cls(
            id=cc_pair_model.id,
            name=cc_pair_model.name,
            status=cc_pair_model.status,
            num_docs_indexed=num_docs_indexed,
            connector=ConnectorSnapshot.from_connector_db_model(
                cc_pair_model.connector
            ),
            credential=CredentialSnapshot.from_credential_db_model(
                cc_pair_model.credential
            ),
            number_of_index_attempts=number_of_index_attempts,
            last_index_attempt_status=last_indexing_status,
            latest_deletion_attempt=latest_deletion_attempt,
            access_type=cc_pair_model.access_type,
            is_editable_for_current_user=is_editable_for_current_user,
            deletion_failure_message=cc_pair_model.deletion_failure_message,
            indexing=indexing,
            creator=cc_pair_model.creator_id,
            creator_email=cc_pair_model.creator.email
            if cc_pair_model.creator
            else None,
        )


class CeleryTaskStatus(BaseModel):
    """Celery任务状态模型"""
    id: str  # 任务ID
    name: str  # 任务名称
    status: TaskStatus  # 任务状态
    start_time: datetime | None  # 开始时间
    register_time: datetime | None  # 注册时间


class FailedConnectorIndexingStatus(BaseModel):
    """
    失败的连接器索引状态模型
    简化版的ConnectorIndexingStatus,用于记录失败的索引尝试
    Simplified version of ConnectorIndexingStatus for failed indexing attempts
    """
    cc_pair_id: int  # 连接器-凭证对ID
    name: str | None  # 名称
    error_msg: str | None  # 错误信息
    is_deletable: bool  # 是否可删除
    connector_id: int  # 连接器ID
    credential_id: int  # 凭证ID


class ConnectorIndexingStatus(BaseModel):
    """
    连接器索引状态模型
    表示连接器的最新索引状态
    Represents the latest indexing status of a connector
    """
    cc_pair_id: int  # 连接器-凭证对ID
    name: str | None  # 名称
    cc_pair_status: ConnectorCredentialPairStatus  # 配对状态
    connector: ConnectorSnapshot  # 连接器快照
    credential: CredentialSnapshot  # 凭证快照
    owner: str  # 所有者
    groups: list[int]  # 组列表
    access_type: AccessType  # 访问类型
    last_finished_status: IndexingStatus | None  # 最后完成状态
    last_status: IndexingStatus | None  # 最后状态
    last_success: datetime | None  # 最后成功时间
    docs_indexed: int  # 已索引文档数
    error_msg: str | None  # 错误信息
    latest_index_attempt: IndexAttemptSnapshot | None  # 最新索引尝试
    deletion_attempt: DeletionAttemptSnapshot | None  # 删除尝试
    is_deletable: bool  # 是否可删除
    
    # index attempt in db can be marked successful while celery/redis
    # is stil running/cleaning up
    # 数据库中的索引尝试可能被标记为成功,而Celery/Redis仍在运行/清理中
    in_progress: bool  # 是否正在进行中


class ConnectorCredentialPairIdentifier(BaseModel):
    """连接器-凭证对标识符模型"""
    connector_id: int  # 连接器ID
    credential_id: int  # 凭证ID


class ConnectorCredentialPairMetadata(BaseModel):
    """连接器-凭证对元数据模型"""
    name: str | None = None  # 名称
    access_type: AccessType  # 访问类型
    auto_sync_options: dict[str, Any] | None = None  # 自动同步选项
    groups: list[int] = Field(default_factory=list)  # 组列表


class CCStatusUpdateRequest(BaseModel):
    """连接器-凭证对状态更新请求"""
    status: ConnectorCredentialPairStatus  # 更新后的状态


class ConnectorCredentialPairDescriptor(BaseModel):
    """连接器-凭证对描述符模型"""
    id: int  # 配对ID
    name: str | None = None  # 配对名称
    connector: ConnectorSnapshot  # 连接器快照
    credential: CredentialSnapshot  # 凭证快照


class RunConnectorRequest(BaseModel):
    """运行连接器请求模型"""
    connector_id: int  # 连接器ID
    credential_ids: list[int] | None = None  # 凭证ID列表
    from_beginning: bool = False  # 是否从头开始


class CCPropertyUpdateRequest(BaseModel):
    """连接器-凭证对属性更新请求"""
    name: str  # 属性名
    value: str  # 属性值


"""Connectors Models"""


class GoogleAppWebCredentials(BaseModel):
    """Google应用Web凭证模型"""
    client_id: str  # 客户端ID
    project_id: str  # 项目ID
    auth_uri: str  # 认证URI
    token_uri: str  # 令牌URI
    auth_provider_x509_cert_url: str  # 认证提供者证书URL
    client_secret: str  # 客户端密钥
    redirect_uris: list[str]  # 重定向URI列表
    javascript_origins: list[str]  # JavaScript源列表


class GoogleAppCredentials(BaseModel):
    """Google应用凭证模型"""
    web: GoogleAppWebCredentials  # Web凭证信息


class GoogleServiceAccountKey(BaseModel):
    """Google服务账号密钥模型"""
    type: str  # 类型
    project_id: str  # 项目ID
    private_key_id: str  # 私钥ID
    private_key: str  # 私钥
    client_email: str  # 客户端邮箱
    client_id: str  # 客户端ID
    auth_uri: str  # 认证URI
    token_uri: str  # 令牌URI
    auth_provider_x509_cert_url: str  # 认证提供者证书URL
    client_x509_cert_url: str  # 客户端证书URL
    universe_domain: str  # 域名


class GoogleServiceAccountCredentialRequest(BaseModel):
    google_primary_admin: str | None = None  # email of user to impersonate


class FileUploadResponse(BaseModel):
    """文件上传响应模型"""
    file_paths: list[str]  # 上传文件路径列表


class ObjectCreationIdResponse(BaseModel):
    """对象创建ID响应模型"""
    id: int | str  # 创建的对象ID
    credential: CredentialSnapshot | None = None  # 关联的凭证快照


class AuthStatus(BaseModel):
    """认证状态模型"""
    authenticated: bool  # 是否已认证


class AuthUrl(BaseModel):
    """认证URL模型"""
    auth_url: str  # 认证URL


class GmailCallback(BaseModel):
    """Gmail回调模型"""
    state: str  # 状态
    code: str  # 授权码


class GDriveCallback(BaseModel):
    """Google Drive回调模型"""
    state: str  # 状态
    code: str  # 授权码
