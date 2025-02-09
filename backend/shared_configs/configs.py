import os
from typing import Any
from typing import List
from urllib.parse import urlparse

from shared_configs.model_server_models import SupportedEmbeddingModel

# Used for logging
# 用于日志记录
SLACK_CHANNEL_ID = "channel_id"  # Slack通知频道ID

# 模型服务器配置
MODEL_SERVER_HOST = os.environ.get("MODEL_SERVER_HOST") or "localhost"  # 模型服务器主机地址
MODEL_SERVER_ALLOWED_HOST = os.environ.get("MODEL_SERVER_HOST") or "0.0.0.0"  # 允许访问的主机地址
MODEL_SERVER_PORT = int(os.environ.get("MODEL_SERVER_PORT") or "9000")  # 模型服务器端口

# Model server for indexing should use a separate one to not allow indexing to introduce delay for inference
# 索引用的模型服务器应该使用单独的实例，以避免索引过程影响推理性能
INDEXING_MODEL_SERVER_HOST = (
    os.environ.get("INDEXING_MODEL_SERVER_HOST") or MODEL_SERVER_HOST
)  # 索引服务器主机
INDEXING_MODEL_SERVER_PORT = int(
    os.environ.get("INDEXING_MODEL_SERVER_PORT") or MODEL_SERVER_PORT
)  # 索引服务器端口

# Onyx custom Deep Learning Models
# Onyx自定义深度学习模型
CONNECTOR_CLASSIFIER_MODEL_REPO = "Danswer/filter-extraction-model"  # 连接器分类模型仓库
CONNECTOR_CLASSIFIER_MODEL_TAG = "1.0.0"  # 连接器分类模型版本
INTENT_MODEL_VERSION = "danswer/hybrid-intent-token-classifier"  # 意图分类模型版本
INTENT_MODEL_TAG = "v1.0.3"  # 意图分类模型标签

# Bi-Encoder, other details
# 双向编码器及其他详细配置
DOC_EMBEDDING_CONTEXT_SIZE = 512  # 文档嵌入上下文大小，控制每次处理的文本长度

# Used to distinguish alternative indices
# 用于区分替代索引
ALT_INDEX_SUFFIX = "__danswer_alt_index"  # 替代索引的后缀名

# Used for loading defaults for automatic deployments and dev flows
# 用于加载自动部署和开发流程的默认配置
# For local, use: mixedbread-ai/mxbai-rerank-xsmall-v1
DEFAULT_CROSS_ENCODER_MODEL_NAME = (
    os.environ.get("DEFAULT_CROSS_ENCODER_MODEL_NAME") or None
)  # 默认交叉编码器模型名称
DEFAULT_CROSS_ENCODER_API_KEY = os.environ.get("DEFAULT_CROSS_ENCODER_API_KEY") or None  # 默认交叉编码器API密钥
DEFAULT_CROSS_ENCODER_PROVIDER_TYPE = (
    os.environ.get("DEFAULT_CROSS_ENCODER_PROVIDER_TYPE") or None
)  # 默认交叉编码器提供商类型
DISABLE_RERANK_FOR_STREAMING = (
    os.environ.get("DISABLE_RERANK_FOR_STREAMING", "").lower() == "true"
)  # 是否禁用流式重排序

# This controls the minimum number of pytorch "threads" to allocate to the embedding
# model. If torch finds more threads on its own, this value is not used.
# 控制分配给嵌入模型的最小PyTorch线程数
MIN_THREADS_ML_MODELS = int(os.environ.get("MIN_THREADS_ML_MODELS") or 1)  # 机器学习模型最小线程数

# Model server that has indexing only set will throw exception if used for reranking
# or intent classification
# 仅用于索引的模型服务器如果用于重排序或意图分类时将抛出异常
INDEXING_ONLY = os.environ.get("INDEXING_ONLY", "").lower() == "true"  # 是否仅用于索引

# The process needs to have this for the log file to write to
# otherwise, it will not create additional log files
# 进程需要此配置才能写入日志文件，否则不会创建额外的日志文件
LOG_FILE_NAME = os.environ.get("LOG_FILE_NAME") or "onyx"  # 日志文件名称

# Enable generating persistent log files for local dev environments
# 启用本地开发环境的持久化日志文件生成
DEV_LOGGING_ENABLED = os.environ.get("DEV_LOGGING_ENABLED", "").lower() == "true"  # 是否启用开发环境日志
# notset, debug, info, notice, warning, error, or critical
# 日志级别：notset, debug, info, notice, warning, error, 或 critical
LOG_LEVEL = os.environ.get("LOG_LEVEL", "debug")  # 日志记录级别

# Timeout for API-based embedding models
# API嵌入模型的超时时间
# NOTE: does not apply for Google VertexAI, since the python client doesn't
# allow us to specify a custom timeout
# 注意：不适用于Google VertexAI，因为Python客户端不允许我们指定自定义超时时间
API_BASED_EMBEDDING_TIMEOUT = int(os.environ.get("API_BASED_EMBEDDING_TIMEOUT", "600"))  # API嵌入模型超时时间

# Only used for OpenAI
# 仅用于OpenAI
OPENAI_EMBEDDING_TIMEOUT = int(
    os.environ.get("OPENAI_EMBEDDING_TIMEOUT", API_BASED_EMBEDDING_TIMEOUT)
)  # OpenAI嵌入模型超时时间

# Whether or not to strictly enforce token limit for chunking.
# 是否严格执行分块的令牌限制
STRICT_CHUNK_TOKEN_LIMIT = (
    os.environ.get("STRICT_CHUNK_TOKEN_LIMIT", "").lower() == "true"
)  # 是否严格执行令牌限制

# Set up Sentry integration (for error logging)
# 设置Sentry集成（用于错误日志记录）
SENTRY_DSN = os.environ.get("SENTRY_DSN")  # Sentry数据源名称

# Fields which should only be set on new search setting
# 仅在新搜索设置中设置的字段
PRESERVED_SEARCH_FIELDS = [
    "id",
    "provider_type",
    "api_key",
    "model_name",
    "api_url",
    "index_name",
    "multipass_indexing",
    "model_dim",
    "normalize",
    "passage_prefix",
    "query_prefix",
]  # 保留的搜索字段

def validate_cors_origin(origin: str) -> None:
    parsed = urlparse(origin)
    if parsed.scheme not in ["http", "https"] or not parsed.netloc:
        raise ValueError(f"Invalid CORS origin: '{origin}'")  # 验证CORS来源

# Examples of valid values for the environment variable:
# 环境变量的有效值示例：
# - "" (allow all origins)
# - "http://example.com" (single origin)
# - "http://example.com,https://example.org" (multiple origins)
# - "*" (allow all origins)
# - ""（允许所有来源）
# - "http://example.com"（单一来源）
# - "http://example.com,https://example.org"（多个来源）
# - "*"（允许所有来源）
CORS_ALLOWED_ORIGIN_ENV = os.environ.get("CORS_ALLOWED_ORIGIN", "")  # CORS允许的来源环境变量

# Explicitly declare the type of CORS_ALLOWED_ORIGIN
# 显式声明CORS_ALLOWED_ORIGIN的类型
CORS_ALLOWED_ORIGIN: List[str]

if CORS_ALLOWED_ORIGIN_ENV:
    # Split the environment variable into a list of origins
    # 将环境变量拆分为来源列表
    CORS_ALLOWED_ORIGIN = [
        origin.strip()
        for origin in CORS_ALLOWED_ORIGIN_ENV.split(",")
        if origin.strip()
    ]
    # Validate each origin in the list
    # 验证列表中的每个来源
    for origin in CORS_ALLOWED_ORIGIN:
        validate_cors_origin(origin)
else:
    # If the environment variable is empty, allow all origins
    # 如果环境变量为空，则允许所有来源
    CORS_ALLOWED_ORIGIN = ["*"]

# Multi-tenancy configuration
# 多租户配置
MULTI_TENANT = os.environ.get("MULTI_TENANT", "").lower() == "true"  # 是否启用多租户

POSTGRES_DEFAULT_SCHEMA = os.environ.get("POSTGRES_DEFAULT_SCHEMA") or "public"  # PostgreSQL默认schema

async def async_return_default_schema(*args: Any, **kwargs: Any) -> str:
    return POSTGRES_DEFAULT_SCHEMA  # 返回默认schema

# Prefix used for all tenant ids
# 用于所有租户ID的前缀
TENANT_ID_PREFIX = "tenant_"  # 租户ID前缀

DISALLOWED_SLACK_BOT_TENANT_IDS = os.environ.get("DISALLOWED_SLACK_BOT_TENANT_IDS")
DISALLOWED_SLACK_BOT_TENANT_LIST = (
    [tenant.strip() for tenant in DISALLOWED_SLACK_BOT_TENANT_IDS.split(",")]
    if DISALLOWED_SLACK_BOT_TENANT_IDS
    else None
)  # 不允许的Slack机器人租户ID列表

IGNORED_SYNCING_TENANT_IDS = os.environ.get("IGNORED_SYNCING_TENANT_IDS")
IGNORED_SYNCING_TENANT_LIST = (
    [tenant.strip() for tenant in IGNORED_SYNCING_TENANT_IDS.split(",")]
    if IGNORED_SYNCING_TENANT_IDS
    else None
)  # 忽略同步的租户ID列表

# 支持的嵌入模型配置
SUPPORTED_EMBEDDING_MODELS = [
    # Cloud-based models
    # 云端模型
    SupportedEmbeddingModel(
        name="cohere/embed-english-v3.0",
        dim=1024,
        index_name="danswer_chunk_cohere_embed_english_v3_0",
    ),
    SupportedEmbeddingModel(
        name="cohere/embed-english-v3.0",
        dim=1024,
        index_name="danswer_chunk_embed_english_v3_0",
    ),
    SupportedEmbeddingModel(
        name="cohere/embed-english-light-v3.0",
        dim=384,
        index_name="danswer_chunk_cohere_embed_english_light_v3_0",
    ),
    SupportedEmbeddingModel(
        name="cohere/embed-english-light-v3.0",
        dim=384,
        index_name="danswer_chunk_embed_english_light_v3_0",
    ),
    SupportedEmbeddingModel(
        name="openai/text-embedding-3-large",
        dim=3072,
        index_name="danswer_chunk_openai_text_embedding_3_large",
    ),
    SupportedEmbeddingModel(
        name="openai/text-embedding-3-large",
        dim=3072,
        index_name="danswer_chunk_text_embedding_3_large",
    ),
    SupportedEmbeddingModel(
        name="openai/text-embedding-3-small",
        dim=1536,
        index_name="danswer_chunk_openai_text_embedding_3_small",
    ),
    SupportedEmbeddingModel(
        name="openai/text-embedding-3-small",
        dim=1536,
        index_name="danswer_chunk_text_embedding_3_small",
    ),
    SupportedEmbeddingModel(
        name="google/text-embedding-004",
        dim=768,
        index_name="danswer_chunk_google_text_embedding_004",
    ),
    SupportedEmbeddingModel(
        name="google/text-embedding-004",
        dim=768,
        index_name="danswer_chunk_text_embedding_004",
    ),
    SupportedEmbeddingModel(
        name="google/textembedding-gecko@003",
        dim=768,
        index_name="danswer_chunk_google_textembedding_gecko_003",
    ),
    SupportedEmbeddingModel(
        name="google/textembedding-gecko@003",
        dim=768,
        index_name="danswer_chunk_textembedding_gecko_003",
    ),
    SupportedEmbeddingModel(
        name="voyage/voyage-large-2-instruct",
        dim=1024,
        index_name="danswer_chunk_voyage_large_2_instruct",
    ),
    SupportedEmbeddingModel(
        name="voyage/voyage-large-2-instruct",
        dim=1024,
        index_name="danswer_chunk_large_2_instruct",
    ),
    SupportedEmbeddingModel(
        name="voyage/voyage-light-2-instruct",
        dim=384,
        index_name="danswer_chunk_voyage_light_2_instruct",
    ),
    SupportedEmbeddingModel(
        name="voyage/voyage-light-2-instruct",
        dim=384,
        index_name="danswer_chunk_light_2_instruct",
    ),
    # Self-hosted models
    # 自托管模型
    SupportedEmbeddingModel(
        name="nomic-ai/nomic-embed-text-v1",
        dim=768,
        index_name="danswer_chunk_nomic_ai_nomic_embed_text_v1",
    ),
    SupportedEmbeddingModel(
        name="nomic-ai/nomic-embed-text-v1",
        dim=768,
        index_name="danswer_chunk_nomic_embed_text_v1",
    ),
    SupportedEmbeddingModel(
        name="intfloat/e5-base-v2",
        dim=768,
        index_name="danswer_chunk_intfloat_e5_base_v2",
    ),
    SupportedEmbeddingModel(
        name="intfloat/e5-small-v2",
        dim=384,
        index_name="danswer_chunk_intfloat_e5_small_v2",
    ),
    SupportedEmbeddingModel(
        name="intfloat/multilingual-e5-base",
        dim=768,
        index_name="danswer_chunk_intfloat_multilingual_e5_base",
    ),
    SupportedEmbeddingModel(
        name="intfloat/multilingual-e5-small",
        dim=384,
        index_name="danswer_chunk_intfloat_multilingual_e5_small",
    ),
]
