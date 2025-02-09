# This file contains all the configuration settings for the chat system
# 此文件包含聊天系统的所有配置设置
import os

# Path to YAML files containing prompts and personas configurations
# 包含提示词和角色配置的YAML文件路径
PROMPTS_YAML = r"D:\codes\onyx\backend/onyx/seeding/prompts.yaml"
PERSONAS_YAML = r"D:\codes\onyx\backend/onyx/seeding/personas.yaml"
# PROMPTS_YAML = "./onyx/seeding/prompts.yaml"
# PERSONAS_YAML = "./onyx/seeding/personas.yaml"

# Number of search results to return initially
# 初始返回的搜索结果数量
NUM_RETURNED_HITS = 50

# Number of results after LLM filtering and reranking
# Used to control first page results while considering cost and latency
# LLM过滤和重排序后的结果数量
# 用于控制首页显示的结果,同时考虑成本和延迟因素
NUM_POSTPROCESSED_RESULTS = 20

# Maximum number of document chunks that can be fed to the chat model
# May vary depending on the specific model being used
# 可以输入到聊天模型的最大文档块数量
# 具体数量可能因模型而异
MAX_CHUNKS_FED_TO_CHAT = float(os.environ.get("MAX_CHUNKS_FED_TO_CHAT") or 10.0)

# Chat input space allocation:
# ~3k total input, split between docs and chat history + prompts
# 聊天输入空间分配:
# 总共约3k输入,在文档和聊天历史+提示词之间分配
CHAT_TARGET_CHUNK_PERCENTAGE = 512 * 3 / 3072

# Document time decay factor for search relevance
# Formula: 1 / (1 + DOC_TIME_DECAY * doc-age-in-years)
# Capped at 0.5 in Vespa
# 文档时间衰减因子,用于搜索相关性计算
# 计算公式: 1 / (1 + DOC_TIME_DECAY * 文档年龄)
# 在Vespa中上限为0.5
DOC_TIME_DECAY = float(
    os.environ.get("DOC_TIME_DECAY") or 0.5  # Hits limit at 2 years by default
)

# Base time decay parameters
# 基础时间衰减参数
BASE_RECENCY_DECAY = 0.5
FAVOR_RECENT_DECAY_MULTIPLIER = 2.0

# For the highest matching base size chunk, how many chunks above and below do we pull in by default
# Note this is not in any of the deployment configs yet
# Currently only applies to search flow not chat
# 搜索结果上下文窗口设置
# 仅适用于搜索流程,不适用于聊天
CONTEXT_CHUNKS_ABOVE = int(os.environ.get("CONTEXT_CHUNKS_ABOVE") or 1)
CONTEXT_CHUNKS_BELOW = int(os.environ.get("CONTEXT_CHUNKS_BELOW") or 1)

# LLM feature toggles
# LLM功能开关
DISABLE_LLM_CHOOSE_SEARCH = (
    os.environ.get("DISABLE_LLM_CHOOSE_SEARCH", "").lower() == "true"
)
DISABLE_LLM_QUERY_REPHRASE = (
    os.environ.get("DISABLE_LLM_QUERY_REPHRASE", "").lower() == "true"
)

# 1 edit per 20 characters, currently unused due to fuzzy match being too slow
# 每20个字符允许1次编辑,由于模糊匹配太慢目前未使用
QUOTE_ALLOWED_ERROR_PERCENT = 0.05
# Maximum timeout for QA operations (60 seconds)
# QA操作的最大超时时间(60秒)
QA_TIMEOUT = int(os.environ.get("QA_TIMEOUT") or "60")  # 60 seconds

# Weighting factor between Vector and Keyword Search, 1 for completely vector search
# 向量搜索和关键词搜索的权重(0-1)
# 1 = 纯向量搜索
HYBRID_ALPHA = max(0, min(1, float(os.environ.get("HYBRID_ALPHA") or 0.5)))
HYBRID_ALPHA_KEYWORD = max(
    0, min(1, float(os.environ.get("HYBRID_ALPHA_KEYWORD") or 0.4))
)

# Weighting factor between Title and Content of documents during search
# Title based value of 1 means completely title-based search
# Default heavily favors Content because Title is included at top of Content
# This avoids cases where Content is relevant but title separation is unclear
# Title acts more as a boost than a separate field
# 搜索时标题和内容之间的权重因子
# 值为1表示完全基于标题搜索
# 默认heavily偏向内容,因为标题已包含在内容顶部
# 这避免了内容相关但标题分离不清的情况
# 标题更像是一个提升因子而不是独立字段
TITLE_CONTENT_RATIO = max(
    0, min(1, float(os.environ.get("TITLE_CONTENT_RATIO") or 0.10))
)

# A list of languages passed to the LLM to rephase the query
# For example "English,French,Spanish", be sure to use the "," separator
# 多语言支持设置
MULTILINGUAL_QUERY_EXPANSION = os.environ.get("MULTILINGUAL_QUERY_EXPANSION") or None
LANGUAGE_HINT = "\n" + (
    os.environ.get("LANGUAGE_HINT")
    or "IMPORTANT: Respond in the same language as my query!"
)
LANGUAGE_CHAT_NAMING_HINT = (
    os.environ.get("LANGUAGE_CHAT_NAMING_HINT")
    or "The name of the conversation must be in the same language as the user query."
)

# Number of prompts each persona should have
# 每个角色应有的提示数量
NUM_PERSONA_PROMPTS = 4
NUM_PERSONA_PROMPT_GENERATION_CHUNKS = 5

# Agentic search takes significantly more tokens and therefore has much higher cost.
# This configuration allows users to get a search-only experience with instant results
# and no involvement from the LLM.
# Additionally, some LLM providers have strict rate limits which may prohibit
# sending many API requests at once (as is done in agentic search).
# Whether the LLM should evaluate all of the document chunks passed in for usefulness
# in relation to the user query
# 基于代理的搜索使用更多token且成本更高
# 此配置允许仅搜索体验,无需LLM即可获得即时结果
# 某些LLM提供商有严格的速率限制,无法同时发送多个API请求
# 控制LLM是否评估文档块与查询的相关性
DISABLE_LLM_DOC_RELEVANCE = (
    os.environ.get("DISABLE_LLM_DOC_RELEVANCE", "").lower() == "true"
)

# Stops streaming answers back to the UI if this pattern is seen
# 当检测到此模式时停止向UI流式传输答案
STOP_STREAM_PAT = os.environ.get("STOP_STREAM_PAT") or None

# Set this to "true" to hard delete chats
# Makes chats unviewable by admins after user deletion
# Differs from soft delete which only hides from non-admin users
# 设置为"true"启用聊天记录硬删除
# 用户删除后管理员也无法查看聊天记录
# 区别于软删除(仅对非管理员用户隐藏)
HARD_DELETE_CHATS = os.environ.get("HARD_DELETE_CHATS", "").lower() == "true"

# Internet Search Configuration
# 互联网搜索配置
BING_API_KEY = os.environ.get("BING_API_KEY") or None

# Enable in-house model for detecting connector-based filtering in queries
# 启用内部模型来检测查询中的连接器过滤
ENABLE_CONNECTOR_CLASSIFIER = os.environ.get("ENABLE_CONNECTOR_CLASSIFIER", False)

# Search configuration
# 搜索配置
VESPA_SEARCHER_THREADS = int(os.environ.get("VESPA_SEARCHER_THREADS") or 2)
