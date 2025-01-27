"""
This configuration file defines all the model-related settings for the system,
including embedding models, reranking models and generative AI models.
本配置文件定义了系统所有与模型相关的设置，包括嵌入模型、重排序模型和生成式AI模型。
"""

import json
import os

#####
# Embedding/Reranking Model Configs
# 嵌入/重排序模型配置
#####
# Important considerations when choosing models
# Max tokens count needs to be high considering use case (at least 512)
# Models used must be MIT or Apache license
# Inference/Indexing speed
"""
选择模型时的重要考虑因素：
- 最大令牌数需要考虑使用场景（至少512）
- 使用的模型必须是MIT或Apache许可证
- 推理/索引速度
"""

# https://huggingface.co/DOCUMENT_ENCODER_MODEL
# The useable models configured as below must be SentenceTransformer compatible
# NOTE: DO NOT CHANGE SET THESE UNLESS YOU KNOW WHAT YOU ARE DOING
# IDEALLY, YOU SHOULD CHANGE EMBEDDING MODELS VIA THE UI
"""
以下配置的可用模型必须与SentenceTransformer兼容
注意：除非你知道自己在做什么，否则不要更改这些设置
理想情况下，你应该通过UI来更改嵌入模型
"""

DEFAULT_DOCUMENT_ENCODER_MODEL = "nomic-ai/nomic-embed-text-v1"  # 默认文档编码器模型
DOCUMENT_ENCODER_MODEL = (
    os.environ.get("DOCUMENT_ENCODER_MODEL") or DEFAULT_DOCUMENT_ENCODER_MODEL
)
# If the below is changed, Vespa deployment must also be changed
# 如果更改以下设置，Vespa部署也必须相应更改
DOC_EMBEDDING_DIM = int(os.environ.get("DOC_EMBEDDING_DIM") or 768)  # 文档嵌入维度
# Model should be chosen with 512 context size, ideally don't change this
# If multipass_indexing is enabled, the max context size would be set to
# DOC_EMBEDDING_CONTEXT_SIZE * LARGE_CHUNK_RATIO
"""
模型应选择512上下文大小，理想情况下不要更改此值
如果启用了multipass_indexing，最大上下文大小将设置为
DOC_EMBEDDING_CONTEXT_SIZE * LARGE_CHUNK_RATIO
"""
DOC_EMBEDDING_CONTEXT_SIZE = 512  # 文档嵌入上下文大小

# 是否规范化嵌入向量
NORMALIZE_EMBEDDINGS = (
    os.environ.get("NORMALIZE_EMBEDDINGS") or "true"
).lower() == "true"

# Old default model settings, which are needed for an automatic easy upgrade
# 旧的默认模型设置，用于自动轻松升级
OLD_DEFAULT_DOCUMENT_ENCODER_MODEL = "thenlper/gte-small"  # 旧的默认文档编码器模型
OLD_DEFAULT_MODEL_DOC_EMBEDDING_DIM = 384  # 旧的默认文档嵌入维度
OLD_DEFAULT_MODEL_NORMALIZE_EMBEDDINGS = False  # 旧的默认是否规范化嵌入

# These are only used if reranking is turned off, to normalize the direct retrieval scores for display
# Currently unused
# 这些仅在关闭重排序时使用，用于规范化直接检索分数以供显示
# 当前未使用
SIM_SCORE_RANGE_LOW = float(os.environ.get("SIM_SCORE_RANGE_LOW") or 0.0)  # 相似度分数范围下限
SIM_SCORE_RANGE_HIGH = float(os.environ.get("SIM_SCORE_RANGE_HIGH") or 1.0)  # 相似度分数范围上限
# Certain models like e5, BGE, etc use a prefix for asymmetric retrievals
# 某些模型（如e5、BGE等）在非对称检索时使用前缀（查询通常比文档短）
ASYM_QUERY_PREFIX = os.environ.get("ASYM_QUERY_PREFIX", "search_query: ")  # 查询前缀
ASYM_PASSAGE_PREFIX = os.environ.get("ASYM_PASSAGE_PREFIX", "search_document: ")  # 文档前缀
# Purely an optimization, memory limitation consideration

# User's set embedding batch size overrides the default encoding batch sizes
# 用户设置的嵌入批量大小会覆盖默认的编码批量大小
EMBEDDING_BATCH_SIZE = int(os.environ.get("EMBEDDING_BATCH_SIZE") or 0) or None  # 嵌入批量大小

BATCH_SIZE_ENCODE_CHUNKS = EMBEDDING_BATCH_SIZE or 8  # 块编码批量大小
# don't send over too many chunks at once, as sending too many could cause timeouts
# 不要一次发送太多块，因为发送太多可能会导致超时
BATCH_SIZE_ENCODE_CHUNKS_FOR_API_EMBEDDING_SERVICES = EMBEDDING_BATCH_SIZE or 512  # API嵌入服务的块编码批量大小

# For score display purposes, only way is to know the expected ranges
# 用于分数显示目的，唯一的方法是知道预期范围
CROSS_ENCODER_RANGE_MAX = 1  # 交叉编码器范围最大值
CROSS_ENCODER_RANGE_MIN = 0  # 交叉编码器范围最小值


#####
# Generative AI Model Configs
# 生成式AI模型配置
#####

# NOTE: the 3 below should only be used for dev.
# 注意：以下3项仅用于开发环境
GEN_AI_API_KEY = os.environ.get("GEN_AI_API_KEY")  # 生成式AI API密钥
GEN_AI_MODEL_VERSION = os.environ.get("GEN_AI_MODEL_VERSION")  # 生成式AI模型版本
FAST_GEN_AI_MODEL_VERSION = os.environ.get("FAST_GEN_AI_MODEL_VERSION")  # 快速生成式AI模型版本

# Override the auto-detection of LLM max context length
# 覆盖LLM最大上下文长度的自动检测
GEN_AI_MAX_TOKENS = int(os.environ.get("GEN_AI_MAX_TOKENS") or 0) or None  # 生成AI最大令牌数

# Set this to be enough for an answer + quotes. Also used for Chat
# This is the minimum token context we will leave for the LLM to generate an answer
"""
设置这个值要足够生成答案和引用。也用于聊天
这是我们为LLM生成答案保留的最小令牌上下文
"""
GEN_AI_NUM_RESERVED_OUTPUT_TOKENS = int(
    os.environ.get("GEN_AI_NUM_RESERVED_OUTPUT_TOKENS") or 1024
)  # 保留用于输出的令牌数

# Typically, GenAI models nowadays are at least 4K tokens
# 通常，现代GenAI模型至少有4K令牌
GEN_AI_MODEL_FALLBACK_MAX_TOKENS = int(
    os.environ.get("GEN_AI_MODEL_FALLBACK_MAX_TOKENS") or 4096
)  # 生成AI模型回退最大令牌数

# Number of tokens from chat history to include at maximum
# 最大包含的聊天历史记录令牌数
# 3000 should be enough context regardless of use
# 3000个令牌对于任何用途都应该足够
GEN_AI_HISTORY_CUTOFF = 3000
# This is used when computing how much context space is available for documents
# ahead of time in order to let the user know if they can "select" more documents
# It represents a maximum "expected" number of input tokens from the latest user
# message. At query time, we don't actually enforce this - we will only throw an
# error if the total # of tokens exceeds the max input tokens.
"""
这用于提前计算文档可用的上下文空间，以便让用户知道是否可以"选择"更多文档
它表示来自最新用户消息的最大"预期"输入令牌数
在查询时，我们实际上不强制执行此限制 - 仅当令牌总数超过最大输入令牌时才会抛出错误
"""
GEN_AI_SINGLE_USER_MESSAGE_EXPECTED_MAX_TOKENS = 512  # 单个用户消息预期最大令牌数
GEN_AI_TEMPERATURE = float(os.environ.get("GEN_AI_TEMPERATURE") or 0)  # 生成AI温度参数

# should be used if you are using a custom LLM inference provider that doesn't support
# streaming format AND you are still using the langchain/litellm LLM class
"""
如果你使用不支持流式格式的自定义LLM推理提供程序，
并且仍在使用langchain/litellm LLM类，则应使用此选项
"""
DISABLE_LITELLM_STREAMING = (
    os.environ.get("DISABLE_LITELLM_STREAMING") or "false"
).lower() == "true"  # 禁用LiteLLM流式传输

# extra headers to pass to LiteLLM
# 传递给LiteLLM的额外头部信息
LITELLM_EXTRA_HEADERS: dict[str, str] | None = None
_LITELLM_EXTRA_HEADERS_RAW = os.environ.get("LITELLM_EXTRA_HEADERS")
if _LITELLM_EXTRA_HEADERS_RAW:
    try:
        LITELLM_EXTRA_HEADERS = json.loads(_LITELLM_EXTRA_HEADERS_RAW)
    except Exception:
        # need to import here to avoid circular imports
        # 需要在这里导入以避免循环导入
        from onyx.utils.logger import setup_logger

        logger = setup_logger()
        logger.error(
            "Failed to parse LITELLM_EXTRA_HEADERS, must be a valid JSON object"
            # "解析LITELLM_EXTRA_HEADERS失败，必须是有效的JSON对象"
        )

# if specified, will pass through request headers to the call to the LLM
# 如果指定，将请求头传递给LLM调用
LITELLM_PASS_THROUGH_HEADERS: list[str] | None = None
_LITELLM_PASS_THROUGH_HEADERS_RAW = os.environ.get("LITELLM_PASS_THROUGH_HEADERS")
if _LITELLM_PASS_THROUGH_HEADERS_RAW:
    try:
        LITELLM_PASS_THROUGH_HEADERS = json.loads(_LITELLM_PASS_THROUGH_HEADERS_RAW)
    except Exception:
        # need to import here to avoid circular imports
        # 需要在这里导入以避免循环导入
        from onyx.utils.logger import setup_logger

        logger = setup_logger()
        logger.error(
            "Failed to parse LITELLM_PASS_THROUGH_HEADERS, must be a valid JSON object"
            # "解析LITELLM_PASS_THROUGH_HEADERS失败，必须是有效的JSON对象"
        )

# if specified, will merge the specified JSON with the existing body of the
# request before sending it to the LLM
"""
如果指定，在发送给LLM之前，
将指定的JSON与请求的现有主体合并
"""
LITELLM_EXTRA_BODY: dict | None = None
_LITELLM_EXTRA_BODY_RAW = os.environ.get("LITELLM_EXTRA_BODY")
if _LITELLM_EXTRA_BODY_RAW:
    try:
        LITELLM_EXTRA_BODY = json.loads(_LITELLM_EXTRA_BODY_RAW)
    except Exception:
        pass  # 忽略解析错误
