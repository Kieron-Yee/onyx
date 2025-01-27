"""
This file contains all the constants used throughout the Onyx application.
It defines configuration values, system settings, and enumerated types that are used across different components.
此文件包含 Onyx 应用程序中使用的所有常量。
它定义了在不同组件中使用的配置值、系统设置和枚举类型。
"""

import platform
import socket
from enum import auto
from enum import Enum

SOURCE_TYPE = "source_type"  # 数据源类型

# stored in the `metadata` of a chunk. Used to signify that this chunk should
# not be used for QA. For example, Google Drive file types which can't be parsed
# are still useful as a search result but not for QA.
# 存储在块的 `metadata` 中。用于表示该块不应用于问答。
# 例如，无法解析的 Google Drive 文件类型作为搜索结果仍然有用，但不适用于问答。
IGNORE_FOR_QA = "ignore_for_qa"

# NOTE: deprecated, only used for porting key from old system
# 注意：已弃用，仅用于从旧系统移植密钥
GEN_AI_API_KEY_STORAGE_KEY = "genai_api_key"

PUBLIC_DOC_PAT = "PUBLIC"  # 公共文档模式
ID_SEPARATOR = ":;:"      # ID分隔符
DEFAULT_BOOST = 0         # 默认提升值
SESSION_KEY = "session"   # 会话密钥

NO_AUTH_USER_ID = "__no_auth_user__"          # 未认证用户ID
NO_AUTH_USER_EMAIL = "anonymous@onyx.app"     # 未认证用户邮箱

# For chunking/processing chunks
# 用于分块/处理文本块
RETURN_SEPARATOR = "\n\r\n"     # 返回分隔符
SECTION_SEPARATOR = "\n\n"      # 段落分隔符
# For combining attributes, doesn't have to be unique/perfect to work
# 用于组合属性，不需要唯一/完美也能工作
INDEX_SEPARATOR = "==="         # 索引分隔符

# For File Connector Metadata override file
# 用于文件连接器元数据覆盖文件
DANSWER_METADATA_FILENAME = ".onyx_metadata.json"

# Messages
# 消息
DISABLED_GEN_AI_MSG = (
    "Your System Admin has disabled the Generative AI functionalities of Onyx.\n"
    "Please contact them if you wish to have this enabled.\n"
    "You can still use Onyx as a search engine."
    # 您的系统管理员已禁用 Onyx 的生成式 AI 功能。
    # 如果您希望启用此功能，请联系他们。
    # 您仍然可以将 Onyx 用作搜索引擎。
)

DEFAULT_PERSONA_ID = 0  # 默认角色ID

DEFAULT_CC_PAIR_ID = 1  # 默认CC对ID

# Postgres connection constants for application_name
# Postgres 连接常量，用于 application_name
POSTGRES_WEB_APP_NAME = "web"
POSTGRES_INDEXER_APP_NAME = "indexer"
POSTGRES_CELERY_APP_NAME = "celery"
POSTGRES_CELERY_BEAT_APP_NAME = "celery_beat"
POSTGRES_CELERY_WORKER_PRIMARY_APP_NAME = "celery_worker_primary"
POSTGRES_CELERY_WORKER_LIGHT_APP_NAME = "celery_worker_light"
POSTGRES_CELERY_WORKER_HEAVY_APP_NAME = "celery_worker_heavy"
POSTGRES_CELERY_WORKER_INDEXING_APP_NAME = "celery_worker_indexing"
POSTGRES_CELERY_WORKER_INDEXING_CHILD_APP_NAME = "celery_worker_indexing_child"
POSTGRES_PERMISSIONS_APP_NAME = "permissions"
POSTGRES_UNKNOWN_APP_NAME = "unknown"

SSL_CERT_FILE = "bundle.pem"  # SSL证书文件
# API Keys
# API 密钥
DANSWER_API_KEY_PREFIX = "API_KEY__"
DANSWER_API_KEY_DUMMY_EMAIL_DOMAIN = "onyxapikey.ai"
UNNAMED_KEY_PLACEHOLDER = "Unnamed"  # 未命名密钥占位符

# Key-Value store keys
# 键值存储键
KV_REINDEX_KEY = "needs_reindexing"
KV_SEARCH_SETTINGS = "search_settings"
KV_UNSTRUCTURED_API_KEY = "unstructured_api_key"
KV_USER_STORE_KEY = "INVITED_USERS"
KV_NO_AUTH_USER_PREFERENCES_KEY = "no_auth_user_preferences"
KV_CRED_KEY = "credential_id_{}"
KV_GMAIL_CRED_KEY = "gmail_app_credential"
KV_GMAIL_SERVICE_ACCOUNT_KEY = "gmail_service_account_key"
KV_GOOGLE_DRIVE_CRED_KEY = "google_drive_app_credential"
KV_GOOGLE_DRIVE_SERVICE_ACCOUNT_KEY = "google_drive_service_account_key"
KV_GEN_AI_KEY_CHECK_TIME = "genai_api_key_last_check_time"
KV_SETTINGS_KEY = "onyx_settings"
KV_CUSTOMER_UUID_KEY = "customer_uuid"
KV_INSTANCE_DOMAIN_KEY = "instance_domain"
KV_ENTERPRISE_SETTINGS_KEY = "onyx_enterprise_settings"
KV_CUSTOM_ANALYTICS_SCRIPT_KEY = "__custom_analytics_script__"
KV_DOCUMENTS_SEEDED_KEY = "documents_seeded"

# NOTE: we use this timeout / 4 in various places to refresh a lock
# might be worth separating this timeout into separate timeouts for each situation
# 注意：我们在多个地方使用此超时/4来刷新锁
# 可能值得将此超时分为不同情况的单独超时
CELERY_VESPA_SYNC_BEAT_LOCK_TIMEOUT = 120

CELERY_PRIMARY_WORKER_LOCK_TIMEOUT = 120

# needs to be long enough to cover the maximum time it takes to download an object
# if we can get callbacks as object bytes download, we could lower this a lot.
# 需要足够长的时间来覆盖下载对象所需的最长时间
# 如果我们可以在对象字节下载时获得回调，我们可以大大降低这个时间。
CELERY_INDEXING_LOCK_TIMEOUT = 3 * 60 * 60  # 60 min

# how long a task should wait for associated fence to be ready
# 任务应等待关联栅栏准备就绪的时间
CELERY_TASK_WAIT_FOR_FENCE_TIMEOUT = 5 * 60  # 5 min

# needs to be long enough to cover the maximum time it takes to download an object
# if we can get callbacks as object bytes download, we could lower this a lot.
# 需要足够长的时间来覆盖下载对象所需的最长时间
# 如果我们可以在对象字节下载时获得回调，我们可以大大降低这个时间。
CELERY_PRUNING_LOCK_TIMEOUT = 300  # 5 min

CELERY_PERMISSIONS_SYNC_LOCK_TIMEOUT = 300  # 5 min

CELERY_EXTERNAL_GROUP_SYNC_LOCK_TIMEOUT = 300  # 5 min

DANSWER_REDIS_FUNCTION_LOCK_PREFIX = "da_function_lock:"


class DocumentSource(str, Enum):
    """
    文档来源枚举类
    定义了系统支持的所有文档源类型
    """
    # Special case, document passed in via Onyx APIs without specifying a source type
    # 特殊情况：通过Onyx API传入但未指定来源类型的文档
    INGESTION_API = "ingestion_api"
    # 各种文档源类型定义
    SLACK = "slack"                   # Slack消息和频道
    WEB = "web"                       # 网页内容
    GOOGLE_DRIVE = "google_drive"     # Google Drive文档
    GMAIL = "gmail"                   # Gmail邮件
    REQUESTTRACKER = "requesttracker"  # RequestTracker工单系统
    GITHUB = "github"                  # GitHub代码仓库
    GITLAB = "gitlab"                  # GitLab代码仓库
    GURU = "guru"                      # Guru知识库
    BOOKSTACK = "bookstack"            # BookStack文档系统
    CONFLUENCE = "confluence"          # Confluence协作平台
    SLAB = "slab"                      # Slab团队知识库
    JIRA = "jira"                      # Jira项目管理
    PRODUCTBOARD = "productboard"      # ProductBoard产品管理
    FILE = "file"                      # 本地文件
    NOTION = "notion"                  # Notion协作平台
    ZULIP = "zulip"                   # Zulip聊天平台
    LINEAR = "linear"                  # Linear项目管理
    HUBSPOT = "hubspot"               # HubSpot客户关系
    DOCUMENT360 = "document360"        # Document360文档系统
    GONG = "gong"                      # Gong销售分析
    GOOGLE_SITES = "google_sites"      # Google Sites网站
    ZENDESK = "zendesk"               # Zendesk客服系统
    LOOPIO = "loopio"                 # Loopio RFP解决方案
    DROPBOX = "dropbox"               # Dropbox文件存储
    SHAREPOINT = "sharepoint"         # SharePoint协作平台
    TEAMS = "teams"                   # Microsoft Teams
    SALESFORCE = "salesforce"         # Salesforce CRM系统
    DISCOURSE = "discourse"           # Discourse论坛
    AXERO = "axero"                   # Axero内部网
    CLICKUP = "clickup"               # ClickUp项目管理
    MEDIAWIKI = "mediawiki"           # MediaWiki平台
    WIKIPEDIA = "wikipedia"           # Wikipedia百科
    ASANA = "asana"                   # Asana项目管理
    S3 = "s3"
    R2 = "r2"
    GOOGLE_CLOUD_STORAGE = "google_cloud_storage"
    OCI_STORAGE = "oci_storage"
    XENFORO = "xenforo"
    NOT_APPLICABLE = "not_applicable"
    FRESHDESK = "freshdesk"
    FIREFLIES = "fireflies"
    EGNYTE = "egnyte"
    AIRTABLE = "airtable"


DocumentSourceRequiringTenantContext: list[DocumentSource] = [DocumentSource.FILE]


class NotificationType(str, Enum):
    """
    通知类型枚举类
    定义系统中的各种通知类型
    """
    REINDEX = "reindex"              # 重新索引通知
    PERSONA_SHARED = "persona_shared" # 角色共享通知
    TRIAL_ENDS_TWO_DAYS = "two_day_trial_ending"  # 试用期剩余两天通知


class BlobType(str, Enum):
    """
    二进制大对象存储类型枚举类
    定义了系统支持的存储服务类型
    """
    R2 = "r2"                        # Cloudflare R2存储
    S3 = "s3"                        # AWS S3存储
    GOOGLE_CLOUD_STORAGE = "google_cloud_storage"  # Google云存储
    OCI_STORAGE = "oci_storage"      # Oracle云存储
    NOT_APPLICABLE = "not_applicable" # 不适用存储类型（用于互联网搜索）


class DocumentIndexType(str, Enum):
    """
    文档索引类型枚举类
    定义了系统支持的索引存储方式
    """
    COMBINED = "combined"  # Vespa综合存储
    SPLIT = "split"       # Typesense和Qdrant分离存储


class AuthType(str, Enum):
    """
    认证类型枚举类
    定义了系统支持的所有认证方式
    """
    DISABLED = "disabled"           # 禁用认证
    BASIC = "basic"                # 基础认证
    GOOGLE_OAUTH = "google_oauth"   # Google OAuth认证
    OIDC = "oidc"                  # OpenID Connect认证
    SAML = "saml"                  # SAML认证
    CLOUD = "cloud"                # 云服务认证（包含Google认证和基础认证）


# Special characters for password validation
# 密码验证的特殊字符
PASSWORD_SPECIAL_CHARS = "!@#$%^&*()_+-=[]{}|;:,.<>?"


class SessionType(str, Enum):
    """
    会话类型枚举类
    定义了系统中的会话交互类型
    """
    CHAT = "Chat"      # 聊天会话
    SEARCH = "Search"  # 搜索会话
    SLACK = "Slack"    # Slack集成会话


class QAFeedbackType(str, Enum):
    LIKE = "like"  # User likes the answer, used for metrics
    # 用户喜欢这个答案，用于指标
    DISLIKE = "dislike"  # User dislikes the answer, used for metrics
    # 用户不喜欢这个答案，用于指标


class SearchFeedbackType(str, Enum):
    ENDORSE = "endorse"  # boost this document for all future queries
    # 提升此文档在所有未来查询中的排名
    REJECT = "reject"  # down-boost this document for all future queries
    # 降低此文档在所有未来查询中的排名
    HIDE = "hide"  # mark this document as untrusted, hide from LLM
    # 将此文档标记为不可信，从LLM中隐藏
    UNHIDE = "unhide"  # 取消隐藏


class MessageType(str, Enum):
    # Using OpenAI standards, Langchain equivalent shown in comment
    # 使用OpenAI标准，Langchain等效项在注释中显示
    # System message is always constructed on the fly, not saved
    # 系统消息总是动态构建的，不会保存
    SYSTEM = "system"  # SystemMessage
    USER = "user"  # HumanMessage
    ASSISTANT = "assistant"  # AIMessage


class TokenRateLimitScope(str, Enum):
    """
    令牌限流范围枚举类
    定义了令牌限流的应用范围
    """
    USER = "user"              # 用户级别限流
    USER_GROUP = "user_group"  # 用户组级别限流
    GLOBAL = "global"          # 全局级别限流


class FileOrigin(str, Enum):
    """
    文件来源枚举类
    定义了系统中文件的不同来源类型
    """
    CHAT_UPLOAD = "chat_upload"          # 聊天上传
    CHAT_IMAGE_GEN = "chat_image_gen"    # 聊天图片生成
    CONNECTOR = "connector"               # 连接器导入
    GENERATED_REPORT = "generated_report" # 生成的报告
    OTHER = "other"                       # 其他来源


class MilestoneRecordType(str, Enum):
    """
    里程碑记录类型枚举类
    定义了系统中的重要事件记录类型
    """
    TENANT_CREATED = "tenant_created"           # 租户创建
    USER_SIGNED_UP = "user_signed_up"          # 用户注册
    MULTIPLE_USERS = "multiple_users"          # 多用户达成
    VISITED_ADMIN_PAGE = "visited_admin_page"  # 访问管理页面
    CREATED_CONNECTOR = "created_connector"     # 创建连接器
    CONNECTOR_SUCCEEDED = "connector_succeeded" # 连接器成功
    RAN_QUERY = "ran_query"                    # 执行查询
    MULTIPLE_ASSISTANTS = "multiple_assistants" # 多助手达成
    CREATED_ASSISTANT = "created_assistant"     # 创建助手
    CREATED_ONYX_BOT = "created_onyx_bot"      # 创建Onyx机器人


class PostgresAdvisoryLocks(Enum):
    """
    PostgreSQL建议锁枚举类
    定义了系统使用的PostgreSQL建议锁ID
    """
    KOMBU_MESSAGE_CLEANUP_LOCK_ID = auto()


class OnyxCeleryQueues:
    """
    Celery队列配置类
    定义了系统中不同类型的任务队列
    """
    # Light queue - 轻量级队列，用于快速处理的任务
    VESPA_METADATA_SYNC = "vespa_metadata_sync"        # Vespa元数据同步
    DOC_PERMISSIONS_UPSERT = "doc_permissions_upsert"  # 文档权限更新
    CONNECTOR_DELETION = "connector_deletion"          # 连接器删除
    LLM_MODEL_UPDATE = "llm_model_update"             # LLM模型更新

    # Heavy queue - 重量级队列，用于耗时任务
    CONNECTOR_PRUNING = "connector_pruning"           # 连接器清理
    CONNECTOR_DOC_PERMISSIONS_SYNC = "connector_doc_permissions_sync"  # 文档权限同步
    CONNECTOR_EXTERNAL_GROUP_SYNC = "connector_external_group_sync"   # 外部组同步

    # Indexing queue - 索引队列，专门用于索引相关任务
    CONNECTOR_INDEXING = "connector_indexing"         # 连接器索引


class OnyxRedisLocks:
    """
    Redis锁配置类
    定义了系统中使用的各种分布式锁
    """
    PRIMARY_WORKER = "da_lock:primary_worker"  # 主工作进程锁
    CHECK_VESPA_SYNC_BEAT_LOCK = "da_lock:check_vespa_sync_beat"  # Vespa同步检查锁
    CHECK_CONNECTOR_DELETION_BEAT_LOCK = "da_lock:check_connector_deletion_beat"  # 连接器删除检查锁
    CHECK_PRUNE_BEAT_LOCK = "da_lock:check_prune_beat"                          # 清理检查锁
    CHECK_INDEXING_BEAT_LOCK = "da_lock:check_indexing_beat"                    # 索引检查锁
    CHECK_CONNECTOR_DOC_PERMISSIONS_SYNC_BEAT_LOCK = (                          # 文档权限同步检查锁
        "da_lock:check_connector_doc_permissions_sync_beat"
    )
    CHECK_CONNECTOR_EXTERNAL_GROUP_SYNC_BEAT_LOCK = (                           # 外部组同步检查锁
        "da_lock:check_connector_external_group_sync_beat"
    )
    MONITOR_VESPA_SYNC_BEAT_LOCK = "da_lock:monitor_vespa_sync_beat"           # Vespa同步监控锁

    CONNECTOR_DOC_PERMISSIONS_SYNC_LOCK_PREFIX = (
        "da_lock:connector_doc_permissions_sync"
    )
    CONNECTOR_EXTERNAL_GROUP_SYNC_LOCK_PREFIX = "da_lock:connector_external_group_sync"
    PRUNING_LOCK_PREFIX = "da_lock:pruning"
    INDEXING_METADATA_PREFIX = "da_metadata:indexing"

    SLACK_BOT_LOCK = "da_lock:slack_bot"
    SLACK_BOT_HEARTBEAT_PREFIX = "da_heartbeat:slack_bot"
    ANONYMOUS_USER_ENABLED = "anonymous_user_enabled"


class OnyxRedisSignals:
    """
    Redis信号类
    定义了系统中使用的Redis信号标识符
    """
    VALIDATE_INDEXING_FENCES = "signal:validate_indexing_fences"  # 验证索引栅栏信号


class OnyxCeleryPriority(int, Enum):
    """
    Celery任务优先级枚举类
    定义了任务的优先级等级，从最高到最低
    """
    HIGHEST = 0  # 最高优先级
    HIGH = auto()  # 高优先级
    MEDIUM = auto()  # 中等优先级
    LOW = auto()  # 低优先级
    LOWEST = auto()  # 最低优先级


class OnyxCeleryTask:
    """
    Celery任务配置类
    定义了系统中所有Celery任务的标识符
    """
    CHECK_FOR_CONNECTOR_DELETION = "check_for_connector_deletion_task"  # 检查连接器删除任务
    CHECK_FOR_VESPA_SYNC_TASK = "check_for_vespa_sync_task"          # 检查Vespa同步任务
    CHECK_FOR_INDEXING = "check_for_indexing"                         # 检查索引任务
    CHECK_FOR_PRUNING = "check_for_pruning"                          # 检查清理任务
    CHECK_FOR_DOC_PERMISSIONS_SYNC = "check_for_doc_permissions_sync"  # 检查文档权限同步
    CHECK_FOR_EXTERNAL_GROUP_SYNC = "check_for_external_group_sync"   # 检查外部组同步
    CHECK_FOR_LLM_MODEL_UPDATE = "check_for_llm_model_update"         # 检查LLM模型更新
    MONITOR_VESPA_SYNC = "monitor_vespa_sync"                         # 监控Vespa同步
    KOMBU_MESSAGE_CLEANUP_TASK = "kombu_message_cleanup_task"         # Kombu消息清理
    CONNECTOR_PERMISSION_SYNC_GENERATOR_TASK = (                       # 连接器权限同步生成器
        "connector_permission_sync_generator_task"
    )
    UPDATE_EXTERNAL_DOCUMENT_PERMISSIONS_TASK = (                      # 更新外部文档权限
        "update_external_document_permissions_task"
    )
    CONNECTOR_EXTERNAL_GROUP_SYNC_GENERATOR_TASK = (                   # 连接器外部组同步生成器
        "connector_external_group_sync_generator_task"
    )
    CONNECTOR_INDEXING_PROXY_TASK = "connector_indexing_proxy_task"   # 连接器索引代理
    CONNECTOR_PRUNING_GENERATOR_TASK = "connector_pruning_generator_task"  # 连接器清理生成器
    DOCUMENT_BY_CC_PAIR_CLEANUP_TASK = "document_by_cc_pair_cleanup_task"  # CC对文档清理
    VESPA_METADATA_SYNC_TASK = "vespa_metadata_sync_task"             # Vespa元数据同步
    CHECK_TTL_MANAGEMENT_TASK = "check_ttl_management_task"           # 检查TTL管理
    AUTOGENERATE_USAGE_REPORT_TASK = "autogenerate_usage_report_task" # 自动生成使用报告


REDIS_SOCKET_KEEPALIVE_OPTIONS = {}
REDIS_SOCKET_KEEPALIVE_OPTIONS[socket.TCP_KEEPINTVL] = 15
REDIS_SOCKET_KEEPALIVE_OPTIONS[socket.TCP_KEEPCNT] = 3

if platform.system() == "Darwin":
    REDIS_SOCKET_KEEPALIVE_OPTIONS[socket.TCP_KEEPALIVE] = 60  # type: ignore
else:
    REDIS_SOCKET_KEEPALIVE_OPTIONS[socket.TCP_KEEPIDLE] = 60  # type: ignore
