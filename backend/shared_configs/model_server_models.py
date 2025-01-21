from pydantic import BaseModel

from shared_configs.enums import EmbeddingProvider
from shared_configs.enums import EmbedTextType
from shared_configs.enums import RerankerProvider

Embedding = list[float]

# ConnectorClassificationRequest类用于表示连接器分类请求
class ConnectorClassificationRequest(BaseModel):
    available_connectors: list[str]
    query: str

# ConnectorClassificationResponse类用于表示连接器分类响应
class ConnectorClassificationResponse(BaseModel):
    connectors: list[str]

# EmbedRequest类用于表示嵌入请求
class EmbedRequest(BaseModel):
    texts: list[str] # 需要进行嵌入的文本列表
    # Can be none for cloud embedding model requests, error handling logic exists for other cases
    # 对于云嵌入模型请求可以为None，其他情况下存在错误处理逻辑
    model_name: str | None = None            # 嵌入模型的名称(可选)
    deployment_name: str | None = None       # Azure部署的模型名称(可选)
    max_context_length: int                  # 文本最大上下文长度限制
    normalize_embeddings: bool               # 是否对嵌入向量进行归一化
    api_key: str | None = None              # API密钥(用于云服务)
    provider_type: EmbeddingProvider | None  # 提供商类型(如OpenAI/Cohere等)
    text_type: EmbedTextType                # 文本类型(查询或段落)
    manual_query_prefix: str | None = None   # 查询文本的手动前缀
    manual_passage_prefix: str | None = None # 段落文本的手动前缀
    api_url: str | None = None              # API端点URL
    api_version: str | None = None          # API版本
    # This disables the "model_" protected namespace for pydantic
    # 这将禁用pydantic的"model_"保护命名空间
    model_config = {"protected_namespaces": ()}

# EmbedResponse类用于表示嵌入响应
class EmbedResponse(BaseModel):
    embeddings: list[Embedding]

# RerankRequest类用于表示重新排序请求
class RerankRequest(BaseModel):
    query: str                              # 查询文本
    documents: list[str]                    # 需要重新排序的文档列表
    model_name: str                         # 重排序模型名称
    provider_type: RerankerProvider | None  # 重排序服务提供商
    api_key: str | None = None             # API密钥
    api_url: str | None = None             # API端点URL

    # This disables the "model_" protected namespace for pydantic
    # 这将禁用pydantic的"model_"保护命名空间
    model_config = {"protected_namespaces": ()}

# RerankResponse类用于表示重新排序响应
class RerankResponse(BaseModel):
    scores: list[float]

# IntentRequest类用于表示意图请求
class IntentRequest(BaseModel):
    query: str
    # Sequence classification threshold
    # 序列分类阈值
    semantic_percent_threshold: float
    # Token classification threshold
    # 令牌分类阈值
    keyword_percent_threshold: float

# IntentResponse类用于表示意图响应
class IntentResponse(BaseModel):
    is_keyword: bool
    keywords: list[str]

# SupportedEmbeddingModel类用于表示支持的嵌入模型
class SupportedEmbeddingModel(BaseModel):
    name: str
    dim: int
    index_name: str
