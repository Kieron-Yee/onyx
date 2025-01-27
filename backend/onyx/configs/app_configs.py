"""
This file contains all the configuration settings for the Onyx application.
It loads settings from environment variables and sets default values when needed.
这个文件包含了Onyx应用的所有配置设置。
它从环境变量加载设置,并在需要时设置默认值。
"""

import json
import os
import urllib.parse
from typing import cast

from onyx.configs.constants import AuthType
from onyx.configs.constants import DocumentIndexType
from onyx.file_processing.enums import HtmlBasedConnectorTransformLinksStrategy

#####
# App Configs 应用配置
#####
# Host for the application server
# 应用服务器主机地址
APP_HOST = "0.0.0.0"

# Port for the application server
# 应用服务器端口
APP_PORT = 8080

# API prefix used for reverse proxy configurations
# Used if proxy doesn't support stripping '/api' prefix
# API前缀,用于反向代理配置
# 当代理不支持去除'/api'前缀时使用
APP_API_PREFIX = os.environ.get("API_PREFIX", "")

#####
# User Facing Features Configs 用户功能配置
#####
# Number of encoder tokens included in chunk preview
# 块预览中包含的编码器令牌数量
BLURB_SIZE = 128

# Frequency to check generative model access (in seconds)
# 检查生成模型访问权限的频率(秒)
GENERATIVE_MODEL_ACCESS_CHECK_FREQ = int(
    os.environ.get("GENERATIVE_MODEL_ACCESS_CHECK_FREQ") or 86400
)  # 1 day

# Enable/disable generative AI features
# 启用/禁用生成式AI功能
DISABLE_GENERATIVE_AI = os.environ.get("DISABLE_GENERATIVE_AI", "").lower() == "true"

#####
# Web Configs 网页配置
#####
# WEB_DOMAIN is used to set the redirect_uri after login flows
# WEB_DOMAIN用于设置登录流程后的重定向URI
# NOTE: if you are having problems accessing the Onyx web UI locally (especially
# on Windows, try  setting this to `http://127.0.0.1:3000` instead and see if that
# fixes it)
# 注意:如果在本地访问Onyx网页界面遇到问题(特别是在Windows上),
# 可以尝试将此值设置为`http://127.0.0.1:3000`看是否能解决问题
WEB_DOMAIN = os.environ.get("WEB_DOMAIN") or "http://localhost:3000"

#####
# Auth Configs 认证配置
#####
# Authentication type setting - can be DISABLED, GOOGLE_OAUTH, etc
# 认证类型设置 - 可以是DISABLED(禁用)、GOOGLE_OAUTH(谷歌OAuth)等
AUTH_TYPE = AuthType((os.environ.get("AUTH_TYPE") or AuthType.DISABLED.value).lower())
DISABLE_AUTH = AUTH_TYPE == AuthType.DISABLED

# Encryption key for sensitive data
# 用于加密敏感数据的密钥
ENCRYPTION_KEY_SECRET = os.environ.get("ENCRYPTION_KEY_SECRET") or ""

# Mask credentials in admin view
# 在管理员视图中是否掩码显示凭证
MASK_CREDENTIAL_PREFIX = (
    os.environ.get("MASK_CREDENTIAL_PREFIX", "True").lower() != "false"
)

# Redis auth token expiration time
# Redis认证令牌过期时间
REDIS_AUTH_EXPIRE_TIME_SECONDS = int(
    os.environ.get("REDIS_AUTH_EXPIRE_TIME_SECONDS") or 3600
)

# Session expiration time (7 days default)
# 会话过期时间(默认7天)
SESSION_EXPIRE_TIME_SECONDS = int(
    os.environ.get("SESSION_EXPIRE_TIME_SECONDS") or 86400 * 7
)  # 7 days

# Default request timeout, mostly used by connectors
# 默认请求超时时间,主要用于连接器
REQUEST_TIMEOUT_SECONDS = int(os.environ.get("REQUEST_TIMEOUT_SECONDS") or 60)

# set `VALID_EMAIL_DOMAINS` to a comma seperated list of domains in order to
# restrict access to Onyx to only users with emails from those domains.
# E.g. `VALID_EMAIL_DOMAINS=example.com,example.org` will restrict Onyx
# signups to users with either an @example.com or an @example.org email.
# NOTE: maintaining `VALID_EMAIL_DOMAIN` to keep backwards compatibility
# 设置`VALID_EMAIL_DOMAINS`为逗号分隔的域名列表,以限制只有这些域名的用户才能访问Onyx。
# 例如,`VALID_EMAIL_DOMAINS=example.com,example.org`将限制Onyx注册为@example.com或@example.org的用户。
# 注意:保留`VALID_EMAIL_DOMAIN`以保持向后兼容性
_VALID_EMAIL_DOMAIN = os.environ.get("VALID_EMAIL_DOMAIN", "")
_VALID_EMAIL_DOMAINS_STR = (
    os.environ.get("VALID_EMAIL_DOMAINS", "") or _VALID_EMAIL_DOMAIN
)
VALID_EMAIL_DOMAINS = (
    [domain.strip() for domain in _VALID_EMAIL_DOMAINS_STR.split(",")]
    if _VALID_EMAIL_DOMAINS_STR
    else []
)
# OAuth Login Flow
# Used for both Google OAuth2 and OIDC flows
# OAuth登录流程
# 用于Google OAuth2和OIDC流程
OAUTH_CLIENT_ID = (
    os.environ.get("OAUTH_CLIENT_ID", os.environ.get("GOOGLE_OAUTH_CLIENT_ID")) or ""
)
OAUTH_CLIENT_SECRET = (
    os.environ.get("OAUTH_CLIENT_SECRET", os.environ.get("GOOGLE_OAUTH_CLIENT_SECRET"))
    or ""
)

USER_AUTH_SECRET = os.environ.get("USER_AUTH_SECRET", "")

# for basic auth
# 用于基本认证
REQUIRE_EMAIL_VERIFICATION = (
    os.environ.get("REQUIRE_EMAIL_VERIFICATION", "").lower() == "true"
)
SMTP_SERVER = os.environ.get("SMTP_SERVER") or "smtp.gmail.com"
SMTP_PORT = int(os.environ.get("SMTP_PORT") or "587")
SMTP_USER = os.environ.get("SMTP_USER", "your-email@gmail.com")
SMTP_PASS = os.environ.get("SMTP_PASS", "your-gmail-password")
EMAIL_CONFIGURED = all([SMTP_SERVER, SMTP_USER, SMTP_PASS])
EMAIL_FROM = os.environ.get("EMAIL_FROM") or SMTP_USER

# If set, Onyx will listen to the `expires_at` returned by the identity
# provider (e.g. Okta, Google, etc.) and force the user to re-authenticate
# after this time has elapsed. Disabled since by default many auth providers
# have very short expiry times (e.g. 1 hour) which provide a poor user experience
# 如果设置,Onyx将监听身份提供者(如Okta,Google等)返回的`expires_at`,并在此时间过后强制用户重新认证。
# 默认禁用,因为许多认证提供者的过期时间非常短(如1小时),这会带来不好的用户体验
TRACK_EXTERNAL_IDP_EXPIRY = (
    os.environ.get("TRACK_EXTERNAL_IDP_EXPIRY", "").lower() == "true"
)

#####
# DB Configs 数据库配置
#####
DOCUMENT_INDEX_NAME = "danswer_index"
# Vespa is now the default document index store for both keyword and vector
# Vespa现在是关键字和向量的默认文档索引存储
DOCUMENT_INDEX_TYPE = os.environ.get(
    "DOCUMENT_INDEX_TYPE", DocumentIndexType.COMBINED.value
)
VESPA_HOST = os.environ.get("VESPA_HOST") or "localhost"
# NOTE: this is used if and only if the vespa config server is accessible via a
# different host than the main vespa application
# 注意:仅当vespa配置服务器通过与主vespa应用不同的主机访问时使用
VESPA_CONFIG_SERVER_HOST = os.environ.get("VESPA_CONFIG_SERVER_HOST") or VESPA_HOST
VESPA_PORT = os.environ.get("VESPA_PORT") or "8081"
VESPA_TENANT_PORT = os.environ.get("VESPA_TENANT_PORT") or "19071"
# the number of times to try and connect to vespa on startup before giving up
# 启动时尝试连接vespa的次数,在放弃之前
VESPA_NUM_ATTEMPTS_ON_STARTUP = int(os.environ.get("NUM_RETRIES_ON_STARTUP") or 10)

VESPA_CLOUD_URL = os.environ.get("VESPA_CLOUD_URL", "")

# The default below is for dockerized deployment
# 下面的默认值用于docker化部署
VESPA_DEPLOYMENT_ZIP = (
    os.environ.get("VESPA_DEPLOYMENT_ZIP") or "/app/onyx/vespa-app.zip"
)
VESPA_CLOUD_CERT_PATH = os.environ.get("VESPA_CLOUD_CERT_PATH")
VESPA_CLOUD_KEY_PATH = os.environ.get("VESPA_CLOUD_KEY_PATH")

# Number of documents in a batch during indexing (further batching done by chunks before passing to bi-encoder)
# 索引期间每批文档的数量(在传递给双编码器之前由块进一步批处理)
try:
    INDEX_BATCH_SIZE = int(os.environ.get("INDEX_BATCH_SIZE", 16))
except ValueError:
    INDEX_BATCH_SIZE = 16

# Below are intended to match the env variables names used by the official postgres docker image
# https://hub.docker.com/_/postgres
# 下面的内容旨在匹配官方postgres docker镜像使用的环境变量名称
POSTGRES_USER = os.environ.get("POSTGRES_USER") or "postgres"
# URL-encode the password for asyncpg to avoid issues with special characters on some machines.
# 对asyncpg的密码进行URL编码,以避免在某些机器上出现特殊字符问题。
POSTGRES_PASSWORD = urllib.parse.quote_plus(
    os.environ.get("POSTGRES_PASSWORD") or "password"
)
POSTGRES_HOST = os.environ.get("POSTGRES_HOST") or "localhost"
POSTGRES_PORT = os.environ.get("POSTGRES_PORT") or "5432"
POSTGRES_DB = os.environ.get("POSTGRES_DB") or "postgres"
AWS_REGION_NAME = os.environ.get("AWS_REGION_NAME") or "us-east-2"

POSTGRES_API_SERVER_POOL_SIZE = int(
    os.environ.get("POSTGRES_API_SERVER_POOL_SIZE") or 40
)
POSTGRES_API_SERVER_POOL_OVERFLOW = int(
    os.environ.get("POSTGRES_API_SERVER_POOL_OVERFLOW") or 10
)
# defaults to False
# 默认为False
POSTGRES_POOL_PRE_PING = os.environ.get("POSTGRES_POOL_PRE_PING", "").lower() == "true"

# recycle timeout in seconds
# 回收超时时间(秒)
POSTGRES_POOL_RECYCLE_DEFAULT = 60 * 20  # 20 minutes
try:
    POSTGRES_POOL_RECYCLE = int(
        os.environ.get("POSTGRES_POOL_RECYCLE", POSTGRES_POOL_RECYCLE_DEFAULT)
    )
except ValueError:
    POSTGRES_POOL_RECYCLE = POSTGRES_POOL_RECYCLE_DEFAULT

# Experimental setting to control idle transactions
# 控制空闲事务的实验性设置
POSTGRES_IDLE_SESSIONS_TIMEOUT_DEFAULT = 0  # milliseconds
try:
    POSTGRES_IDLE_SESSIONS_TIMEOUT = int(
        os.environ.get(
            "POSTGRES_IDLE_SESSIONS_TIMEOUT", POSTGRES_IDLE_SESSIONS_TIMEOUT_DEFAULT
        )
    )
except ValueError:
    POSTGRES_IDLE_SESSIONS_TIMEOUT = POSTGRES_IDLE_SESSIONS_TIMEOUT_DEFAULT

USE_IAM_AUTH = os.getenv("USE_IAM_AUTH", "False").lower() == "true"

REDIS_SSL = os.getenv("REDIS_SSL", "").lower() == "true"
REDIS_HOST = os.environ.get("REDIS_HOST") or "localhost"
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD") or ""

REDIS_AUTH_KEY_PREFIX = "fastapi_users_token:"

# Rate limiting for auth endpoints
# 认证端点的速率限制
RATE_LIMIT_WINDOW_SECONDS: int | None = None
_rate_limit_window_seconds_str = os.environ.get("RATE_LIMIT_WINDOW_SECONDS")
if _rate_limit_window_seconds_str is not None:
    try:
        RATE_LIMIT_WINDOW_SECONDS = int(_rate_limit_window_seconds_str)
    except ValueError:
        pass

RATE_LIMIT_MAX_REQUESTS: int | None = None
_rate_limit_max_requests_str = os.environ.get("RATE_LIMIT_MAX_REQUESTS")
if _rate_limit_max_requests_str is not None:
    try:
        RATE_LIMIT_MAX_REQUESTS = int(_rate_limit_max_requests_str)
    except ValueError:
        pass

# Used for general redis things
# 用于一般的redis操作
REDIS_DB_NUMBER = int(os.environ.get("REDIS_DB_NUMBER", 0))

# Used by celery as broker and backend
# 由celery用作代理和后端
REDIS_DB_NUMBER_CELERY_RESULT_BACKEND = int(
    os.environ.get("REDIS_DB_NUMBER_CELERY_RESULT_BACKEND", 14)
)
REDIS_DB_NUMBER_CELERY = int(os.environ.get("REDIS_DB_NUMBER_CELERY", 15))  # broker

# will propagate to both our redis client as well as celery's redis client
# 将传播到我们的redis客户端和celery的redis客户端
REDIS_HEALTH_CHECK_INTERVAL = int(os.environ.get("REDIS_HEALTH_CHECK_INTERVAL", 60))

# our redis client only, not celery's
# 仅我们的redis客户端,不包括celery的
REDIS_POOL_MAX_CONNECTIONS = int(os.environ.get("REDIS_POOL_MAX_CONNECTIONS", 128))

# https://docs.celeryq.dev/en/stable/userguide/configuration.html#redis-backend-settings
# should be one of "required", "optional", or "none"
# 应该是"required"、"optional"或"none"之一
REDIS_SSL_CERT_REQS = os.getenv("REDIS_SSL_CERT_REQS", "none")
REDIS_SSL_CA_CERTS = os.getenv("REDIS_SSL_CA_CERTS", None)

CELERY_RESULT_EXPIRES = int(os.environ.get("CELERY_RESULT_EXPIRES", 86400))  # seconds

# https://docs.celeryq.dev/en/stable/userguide/configuration.html#broker-pool-limit
# Setting to None may help when there is a proxy in the way closing idle connections
# 设置为None可能有助于在有代理关闭空闲连接时
CELERY_BROKER_POOL_LIMIT_DEFAULT = 10
try:
    CELERY_BROKER_POOL_LIMIT = int(
        os.environ.get("CELERY_BROKER_POOL_LIMIT", CELERY_BROKER_POOL_LIMIT_DEFAULT)
    )
except ValueError:
    CELERY_BROKER_POOL_LIMIT = CELERY_BROKER_POOL_LIMIT_DEFAULT

CELERY_WORKER_LIGHT_CONCURRENCY_DEFAULT = 24
try:
    CELERY_WORKER_LIGHT_CONCURRENCY = int(
        os.environ.get(
            "CELERY_WORKER_LIGHT_CONCURRENCY", CELERY_WORKER_LIGHT_CONCURRENCY_DEFAULT
        )
    )
except ValueError:
    CELERY_WORKER_LIGHT_CONCURRENCY = CELERY_WORKER_LIGHT_CONCURRENCY_DEFAULT

CELERY_WORKER_LIGHT_PREFETCH_MULTIPLIER_DEFAULT = 8
try:
    CELERY_WORKER_LIGHT_PREFETCH_MULTIPLIER = int(
        os.environ.get(
            "CELERY_WORKER_LIGHT_PREFETCH_MULTIPLIER",
            CELERY_WORKER_LIGHT_PREFETCH_MULTIPLIER_DEFAULT,
        )
    )
except ValueError:
        CELERY_WORKER_LIGHT_PREFETCH_MULTIPLIER = (
        CELERY_WORKER_LIGHT_PREFETCH_MULTIPLIER_DEFAULT
    )

CELERY_WORKER_INDEXING_CONCURRENCY_DEFAULT = 3
try:
    env_value = os.environ.get("CELERY_WORKER_INDEXING_CONCURRENCY")
    if not env_value:
        env_value = os.environ.get("NUM_INDEXING_WORKERS")

    if not env_value:
        env_value = str(CELERY_WORKER_INDEXING_CONCURRENCY_DEFAULT)
    CELERY_WORKER_INDEXING_CONCURRENCY = int(env_value)
except ValueError:
    CELERY_WORKER_INDEXING_CONCURRENCY = CELERY_WORKER_INDEXING_CONCURRENCY_DEFAULT

#####
# Connector Configs 连接器配置
#####
POLL_CONNECTOR_OFFSET = 30  # Minutes overlap between poll windows
# 轮询窗口之间的分钟重叠

# View the list here:
# https://github.com/onyx-dot-app/onyx/blob/main/backend/onyx/connectors/factory.py
# If this is empty, all connectors are enabled, this is an option for security heavy orgs where
# only very select connectors are enabled and admins cannot add other connector types
# 如果为空,则启用所有连接器,这是一个选项,适用于安全性较高的组织,其中仅启用非常选择的连接器,管理员无法添加其他连接器类型
ENABLED_CONNECTOR_TYPES = os.environ.get("ENABLED_CONNECTOR_TYPES") or ""

# Some calls to get information on expert users are quite costly especially with rate limiting
# Since experts are not used in the actual user experience, currently it is turned off
# for some connectors
# 获取专家用户信息的一些调用非常昂贵,特别是在速率限制的情况下
# 由于专家不用于实际的用户体验,目前对某些连接器关闭
ENABLE_EXPENSIVE_EXPERT_CALLS = False

# TODO these should be available for frontend configuration, via advanced options expandable
# 这些应该可以通过高级选项扩展用于前端配置
WEB_CONNECTOR_IGNORED_CLASSES = os.environ.get(
    "WEB_CONNECTOR_IGNORED_CLASSES", "sidebar,footer"
).split(",")
WEB_CONNECTOR_IGNORED_ELEMENTS = os.environ.get(
    "WEB_CONNECTOR_IGNORED_ELEMENTS", "nav,footer,meta,script,style,symbol,aside"
).split(",")
WEB_CONNECTOR_OAUTH_CLIENT_ID = os.environ.get("WEB_CONNECTOR_OAUTH_CLIENT_ID")
WEB_CONNECTOR_OAUTH_CLIENT_SECRET = os.environ.get("WEB_CONNECTOR_OAUTH_CLIENT_SECRET")
WEB_CONNECTOR_OAUTH_TOKEN_URL = os.environ.get("WEB_CONNECTOR_OAUTH_TOKEN_URL")
WEB_CONNECTOR_VALIDATE_URLS = os.environ.get("WEB_CONNECTOR_VALIDATE_URLS")

HTML_BASED_CONNECTOR_TRANSFORM_LINKS_STRATEGY = os.environ.get(
    "HTML_BASED_CONNECTOR_TRANSFORM_LINKS_STRATEGY",
    HtmlBasedConnectorTransformLinksStrategy.STRIP,
)

NOTION_CONNECTOR_ENABLE_RECURSIVE_PAGE_LOOKUP = (
    os.environ.get("NOTION_CONNECTOR_ENABLE_RECURSIVE_PAGE_LOOKUP", "").lower()
    == "true"
)

CONFLUENCE_CONNECTOR_LABELS_TO_SKIP = [
    ignored_tag
    for ignored_tag in os.environ.get("CONFLUENCE_CONNECTOR_LABELS_TO_SKIP", "").split(
        ","
    )
    if ignored_tag
]

# Avoid to get archived pages
# 避免获取已归档页面
CONFLUENCE_CONNECTOR_INDEX_ARCHIVED_PAGES = (
    os.environ.get("CONFLUENCE_CONNECTOR_INDEX_ARCHIVED_PAGES", "").lower() == "true"
)

# Attachments exceeding this size will not be retrieved (in bytes)
# 超过此大小的附件将不会被检索(以字节为单位)
CONFLUENCE_CONNECTOR_ATTACHMENT_SIZE_THRESHOLD = int(
    os.environ.get("CONFLUENCE_CONNECTOR_ATTACHMENT_SIZE_THRESHOLD", 10 * 1024 * 1024)
)
# Attachments with more chars than this will not be indexed. This is to prevent extremely
# large files from freezing indexing. 200,000 is ~100 google doc pages.
# 超过此字符数的附件将不会被索引。这是为了防止极大的文件冻结索引。200,000约为100个Google文档页面。
CONFLUENCE_CONNECTOR_ATTACHMENT_CHAR_COUNT_THRESHOLD = int(
    os.environ.get("CONFLUENCE_CONNECTOR_ATTACHMENT_CHAR_COUNT_THRESHOLD", 200_000)
)

# Due to breakages in the confluence API, the timezone offset must be specified client side
# to match the user's specified timezone.

# The current state of affairs:
# CQL queries are parsed in the user's timezone and cannot be specified in UTC
# no API retrieves the user's timezone
# All data is returned in UTC, so we can't derive the user's timezone from that

# https://community.developer.atlassian.com/t/confluence-cloud-time-zone-get-via-rest-api/35954/16
# https://jira.atlassian.com/browse/CONFCLOUD-69670

# enter as a floating point offset from UTC in hours (-24 < val < 24)
# this will be applied globally, so it probably makes sense to transition this to per
# connector as some point.
# 由于confluence API的中断,必须在客户端指定时区偏移量以匹配用户指定的时区。
# 当前情况:
# CQL查询在用户的时区中解析,不能在UTC中指定
# 没有API检索用户的时区
# 所有数据都以UTC返回,因此我们无法从中推导出用户的时区
# 以浮点数形式输入从UTC的偏移量(小时)(-24 < val < 24)
# 这将全局应用,因此在某些时候将其转换为每个连接器可能是有意义的。
CONFLUENCE_TIMEZONE_OFFSET = float(os.environ.get("CONFLUENCE_TIMEZONE_OFFSET", 0.0))

JIRA_CONNECTOR_LABELS_TO_SKIP = [
    ignored_tag
    for ignored_tag in os.environ.get("JIRA_CONNECTOR_LABELS_TO_SKIP", "").split(",")
    if ignored_tag
]
# Maximum size for Jira tickets in bytes (default: 100KB)
# Jira票据的最大大小(字节)(默认:100KB)
JIRA_CONNECTOR_MAX_TICKET_SIZE = int(
    os.environ.get("JIRA_CONNECTOR_MAX_TICKET_SIZE", 100 * 1024)
)

GONG_CONNECTOR_START_TIME = os.environ.get("GONG_CONNECTOR_START_TIME")

GITHUB_CONNECTOR_BASE_URL = os.environ.get("GITHUB_CONNECTOR_BASE_URL") or None

GITLAB_CONNECTOR_INCLUDE_CODE_FILES = (
    os.environ.get("GITLAB_CONNECTOR_INCLUDE_CODE_FILES", "").lower() == "true"
)

# Typically set to http://localhost:3000 for OAuth connector development
# 通常设置为http://localhost:3000用于OAuth连接器开发
CONNECTOR_LOCALHOST_OVERRIDE = os.getenv("CONNECTOR_LOCALHOST_OVERRIDE")

# Egnyte specific configs
# Egnyte特定配置
EGNYTE_CLIENT_ID = os.getenv("EGNYTE_CLIENT_ID")
EGNYTE_CLIENT_SECRET = os.getenv("EGNYTE_CLIENT_SECRET")

# Linear specific configs
# Linear特定配置
LINEAR_CLIENT_ID = os.getenv("LINEAR_CLIENT_ID")
LINEAR_CLIENT_SECRET = os.getenv("LINEAR_CLIENT_SECRET")

DASK_JOB_CLIENT_ENABLED = (
    os.environ.get("DASK_JOB_CLIENT_ENABLED", "").lower() == "true"
)
EXPERIMENTAL_CHECKPOINTING_ENABLED = (
    os.environ.get("EXPERIMENTAL_CHECKPOINTING_ENABLED", "").lower() == "true"
)

PRUNING_DISABLED = -1
DEFAULT_PRUNING_FREQ = 60 * 60 * 24  # Once a day
# 一天一次

ALLOW_SIMULTANEOUS_PRUNING = (
    os.environ.get("ALLOW_SIMULTANEOUS_PRUNING", "").lower() == "true"
)

# This is the maximum rate at which documents are queried for a pruning job. 0 disables the limitation.
# 这是修剪作业查询文档的最大速率。0禁用限制。
MAX_PRUNING_DOCUMENT_RETRIEVAL_PER_MINUTE = int(
    os.environ.get("MAX_PRUNING_DOCUMENT_RETRIEVAL_PER_MINUTE", 0)
)

# comma delimited list of zendesk article labels to skip indexing for
# 逗号分隔的zendesk文章标签列表,跳过索引
ZENDESK_CONNECTOR_SKIP_ARTICLE_LABELS = os.environ.get(
    "ZENDESK_CONNECTOR_SKIP_ARTICLE_LABELS", ""
).split(",")

#####
# Indexing Configs 索引配置
#####
# NOTE: Currently only supported in the Confluence and Google Drive connectors +
# only handles some failures (Confluence = handles API call failures, Google
# Drive = handles failures pulling files / parsing them)
# 目前仅在Confluence和Google Drive连接器中支持+仅处理一些失败(Confluence=处理API调用失败,Google Drive=处理拉取文件/解析文件失败)
CONTINUE_ON_CONNECTOR_FAILURE = os.environ.get(
    "CONTINUE_ON_CONNECTOR_FAILURE", ""
).lower() not in ["false", ""]
# When swapping to a new embedding model, a secondary index is created in the background, to conserve
# resources, we pause updates on the primary index by default while the secondary index is created
# 当切换到新的嵌入模型时,在后台创建一个辅助索引,为了节省资源,在创建辅助索引时,默认暂停主索引的更新
DISABLE_INDEX_UPDATE_ON_SWAP = (
    os.environ.get("DISABLE_INDEX_UPDATE_ON_SWAP", "").lower() == "true"
)
# Controls how many worker processes we spin up to index documents in the
# background. This is useful for speeding up indexing, but does require a
# fairly large amount of memory in order to increase substantially, since
# each worker loads the embedding models into memory.
# 控制我们在后台启动多少个工作进程来索引文档。这对于加速索引很有用,但确实需要相当大量的内存才能显著增加,因为每个工作进程都将嵌入模型加载到内存中。
NUM_INDEXING_WORKERS = int(os.environ.get("NUM_INDEXING_WORKERS") or 1)
NUM_SECONDARY_INDEXING_WORKERS = int(
    os.environ.get("NUM_SECONDARY_INDEXING_WORKERS") or NUM_INDEXING_WORKERS
)
# More accurate results at the expense of indexing speed and index size (stores additional 4 MINI_CHUNK vectors)
# 以索引速度和索引大小为代价获得更准确的结果(存储额外的4个MINI_CHUNK向量)
ENABLE_MULTIPASS_INDEXING = (
    os.environ.get("ENABLE_MULTIPASS_INDEXING", "").lower() == "true"
)
# Finer grained chunking for more detail retention
# 更细粒度的分块以保留更多细节
# Slightly larger since the sentence aware split is a max cutoff so most minichunks will be under MINI_CHUNK_SIZE
# tokens. But we need it to be at least as big as 1/4th chunk size to avoid having a tiny mini-chunk at the end
# 稍大一些,因为句子感知分割是最大截止,所以大多数小块将小于MINI_CHUNK_SIZE标记。但我们需要它至少与1/4块大小一样大,以避免在末尾有一个小小的迷你块
MINI_CHUNK_SIZE = 150

# This is the number of regular chunks per large chunk
# 这是每个大块的常规块数
LARGE_CHUNK_RATIO = 4

# Include the document level metadata in each chunk. If the metadata is too long, then it is thrown out
# We don't want the metadata to overwhelm the actual contents of the chunk
# 在每个块中包含文档级元数据。如果元数据太长,则将其丢弃
# 我们不希望元数据淹没块的实际内容
SKIP_METADATA_IN_CHUNK = os.environ.get("SKIP_METADATA_IN_CHUNK", "").lower() == "true"
# Timeout to wait for job's last update before killing it, in hours
# 等待作业最后更新的超时时间,以小时为单位
CLEANUP_INDEXING_JOBS_TIMEOUT = int(
    os.environ.get("CLEANUP_INDEXING_JOBS_TIMEOUT") or 3
)

# The indexer will warn in the logs whenver a document exceeds this threshold (in bytes)
# 当文档超过此阈值(以字节为单位)时,索引器将在日志中发出警告
INDEXING_SIZE_WARNING_THRESHOLD = int(
    os.environ.get("INDEXING_SIZE_WARNING_THRESHOLD") or 100 * 1024 * 1024
)

# during indexing, will log verbose memory diff stats every x batches and at the end.
# 0 disables this behavior and is the default.
# 在索引期间,将每x批次和结束时记录详细的内存差异统计信息。0禁用此行为,并且是默认值。
INDEXING_TRACER_INTERVAL = int(os.environ.get("INDEXING_TRACER_INTERVAL") or 0)

# During an indexing attempt, specifies the number of batches which are allowed to
# exception without aborting the attempt.
# 在索引尝试期间,指定允许异常而不中止尝试的批次数量。
INDEXING_EXCEPTION_LIMIT = int(os.environ.get("INDEXING_EXCEPTION_LIMIT") or 0)

# Maximum file size in a document to be indexed
# 要索引的文档中的最大文件大小
MAX_DOCUMENT_CHARS = int(os.environ.get("MAX_DOCUMENT_CHARS") or 5_000_000)
MAX_FILE_SIZE_BYTES = int(
    os.environ.get("MAX_FILE_SIZE_BYTES") or 2 * 1024 * 1024 * 1024
)  # 2GB in bytes

#####
# Miscellaneous 杂项
#####
JOB_TIMEOUT = 60 * 60 * 6  # 6 hours default
# used to allow the background indexing jobs to use a different embedding
# model server than the API server
# 用于允许后台索引作业使用与API服务器不同的嵌入模型服务器
CURRENT_PROCESS_IS_AN_INDEXING_JOB = (
    os.environ.get("CURRENT_PROCESS_IS_AN_INDEXING_JOB", "").lower() == "true"
)
# Sets LiteLLM to verbose logging
# 将LiteLLM设置为详细日志记录
LOG_ALL_MODEL_INTERACTIONS = (
    os.environ.get("LOG_ALL_MODEL_INTERACTIONS", "").lower() == "true"
)
# Logs Onyx only model interactions like prompts, responses, messages etc.
# 仅记录Onyx模型交互,如提示、响应、消息等。
LOG_DANSWER_MODEL_INTERACTIONS = (
    os.environ.get("LOG_DANSWER_MODEL_INTERACTIONS", "").lower() == "true"
)
LOG_INDIVIDUAL_MODEL_TOKENS = (
    os.environ.get("LOG_INDIVIDUAL_MODEL_TOKENS", "").lower() == "true"
)
# If set to `true` will enable additional logs about Vespa query performance
# (time spent on finding the right docs + time spent fetching summaries from disk)
# 如果设置为`true`,将启用有关Vespa查询性能的其他日志
# (查找正确文档的时间+从磁盘获取摘要的时间)
LOG_VESPA_TIMING_INFORMATION = (
    os.environ.get("LOG_VESPA_TIMING_INFORMATION", "").lower() == "true"
)
LOG_ENDPOINT_LATENCY = os.environ.get("LOG_ENDPOINT_LATENCY", "").lower() == "true"
LOG_POSTGRES_LATENCY = os.environ.get("LOG_POSTGRES_LATENCY", "").lower() == "true"
LOG_POSTGRES_CONN_COUNTS = (
    os.environ.get("LOG_POSTGRES_CONN_COUNTS", "").lower() == "true"
)
# Anonymous usage telemetry
# 匿名使用遥测
DISABLE_TELEMETRY = os.environ.get("DISABLE_TELEMETRY", "").lower() == "true"

TOKEN_BUDGET_GLOBALLY_ENABLED = (
    os.environ.get("TOKEN_BUDGET_GLOBALLY_ENABLED", "").lower() == "true"
)

# Defined custom query/answer conditions to validate the query and the LLM answer.
# Format: list of strings
# 定义自定义查询/答案条件以验证查询和LLM答案。
# 格式:字符串列表
CUSTOM_ANSWER_VALIDITY_CONDITIONS = json.loads(
    os.environ.get("CUSTOM_ANSWER_VALIDITY_CONDITIONS", "[]")
)

VESPA_REQUEST_TIMEOUT = int(os.environ.get("VESPA_REQUEST_TIMEOUT") or "15")

SYSTEM_RECURSION_LIMIT = int(os.environ.get("SYSTEM_RECURSION_LIMIT") or "1000")

PARSE_WITH_TRAFILATURA = os.environ.get("PARSE_WITH_TRAFILATURA", "").lower() == "true"

# allow for custom error messages for different errors returned by litellm
# for example, can specify: {"Violated content safety policy": "EVIL REQUEST!!!"}
# to make it so that if an LLM call returns an error containing "Violated content safety policy"
# the end user will see "EVIL REQUEST!!!" instead of the default error message.
# 允许为litellm返回的不同错误自定义错误消息
# 例如,可以指定:{"Violated content safety policy":"EVIL REQUEST!!!"}
# 这样,如果LLM调用返回包含"Violated content safety policy"的错误
# 最终用户将看到"EVIL REQUEST!!!"而不是默认错误消息。
_LITELLM_CUSTOM_ERROR_MESSAGE_MAPPINGS = os.environ.get(
    "LITELLM_CUSTOM_ERROR_MESSAGE_MAPPINGS", ""
)
LITELLM_CUSTOM_ERROR_MESSAGE_MAPPINGS: dict[str, str] | None = None
try:
    LITELLM_CUSTOM_ERROR_MESSAGE_MAPPINGS = cast(
        dict[str, str], json.loads(_LITELLM_CUSTOM_ERROR_MESSAGE_MAPPINGS)
    )
except json.JSONDecodeError:
    pass

# LLM Model Update API endpoint
# LLM模型更新API端点
LLM_MODEL_UPDATE_API_URL = os.environ.get("LLM_MODEL_UPDATE_API_URL")

#####
# Enterprise Edition Configs 企业版配置
#####
# NOTE: this should only be enabled if you have purchased an enterprise license.
# if you're interested in an enterprise license, please reach out to us at
# founders@onyx.app OR message Chris Weaver or Yuhong Sun in the Onyx
# Slack community (https://join.slack.com/t/danswer/shared_invite/zt-1w76msxmd-HJHLe3KNFIAIzk_0dSOKaQ)
# 注意:仅当您购买了企业许可证时才应启用此功能。
# 如果您对企业许可证感兴趣,请通过founders@onyx.app与我们联系,或在Onyx Slack社区中给Chris Weaver或Yuhong Sun发送消息
# (https://join.slack.com/t/danswer/shared_invite/zt-1w76msxmd-HJHLe3KNFIAIzk_0dSOKaQ)
ENTERPRISE_EDITION_ENABLED = (
    os.environ.get("ENABLE_PAID_ENTERPRISE_EDITION_FEATURES", "").lower() == "true"
)

# Azure DALL-E Configurations
# Azure DALL-E配置
AZURE_DALLE_API_VERSION = os.environ.get("AZURE_DALLE_API_VERSION")
AZURE_DALLE_API_KEY = os.environ.get("AZURE_DALLE_API_KEY")
AZURE_DALLE_API_BASE = os.environ.get("AZURE_DALLE_API_BASE")
AZURE_DALLE_DEPLOYMENT_NAME = os.environ.get("AZURE_DALLE_DEPLOYMENT_NAME")

# Use managed Vespa (Vespa Cloud). If set, must also set VESPA_CLOUD_URL, VESPA_CLOUD_CERT_PATH and VESPA_CLOUD_KEY_PATH
# 使用托管Vespa(Vespa Cloud)。如果设置,还必须设置VESPA_CLOUD_URL、VESPA_CLOUD_CERT_PATH和VESPA_CLOUD_KEY_PATH
MANAGED_VESPA = os.environ.get("MANAGED_VESPA", "").lower() == "true"

ENABLE_EMAIL_INVITES = os.environ.get("ENABLE_EMAIL_INVITES", "").lower() == "true"

# Security and authentication
# 安全和认证
DATA_PLANE_SECRET = os.environ.get(
    "DATA_PLANE_SECRET", ""
)  # Used for secure communication between the control and data plane
# 用于控制平面和数据平面之间的安全通信
EXPECTED_API_KEY = os.environ.get(
    "EXPECTED_API_KEY", ""
)  # Additional security check for the control plane API
# 控制平面API的额外安全检查

# API configuration
# API配置
CONTROL_PLANE_API_BASE_URL = os.environ.get(
    "CONTROL_PLANE_API_BASE_URL", "http://localhost:8082"
)

# JWT configuration
# JWT配置
JWT_ALGORITHM = "HS256"

#####
# API Key Configs API密钥配置
#####
# refers to the rounds described here: https://passlib.readthedocs.io/en/stable/lib/passlib.hash.sha256_crypt.html
# 参考此处描述的轮次:https://passlib.readthedocs.io/en/stable/lib/passlib.hash.sha256_crypt.html
_API_KEY_HASH_ROUNDS_RAW = os.environ.get("API_KEY_HASH_ROUNDS")
API_KEY_HASH_ROUNDS = (
    int(_API_KEY_HASH_ROUNDS_RAW) if _API_KEY_HASH_ROUNDS_RAW else None
)

POD_NAME = os.environ.get("POD_NAME")
POD_NAMESPACE = os.environ.get("POD_NAMESPACE")

DEV_MODE = os.environ.get("DEV_MODE", "").lower() == "true"

TEST_ENV = os.environ.get("TEST_ENV", "").lower() == "true"
