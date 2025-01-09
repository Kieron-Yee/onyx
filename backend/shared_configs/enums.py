from enum import Enum

# 嵌入模型
class EmbeddingProvider(str, Enum):
    OPENAI = "openai"
    COHERE = "cohere"
    VOYAGE = "voyage"
    GOOGLE = "google"
    LITELLM = "litellm"
    AZURE = "azure"

# 重排序模型
class RerankerProvider(str, Enum):
    COHERE = "cohere"
    LITELLM = "litellm"

# 文本嵌入类型
class EmbedTextType(str, Enum):
    QUERY = "query"
    PASSAGE = "passage"
