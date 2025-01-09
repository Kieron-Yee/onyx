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
    texts: list[str]
    # Can be none for cloud embedding model requests, error handling logic exists for other cases
    # 对于云嵌入模型请求可以为None，其他情况下存在错误处理逻辑
    model_name: str | None = None
    deployment_name: str | None = None
    max_context_length: int
    normalize_embeddings: bool
    api_key: str | None = None
    provider_type: EmbeddingProvider | None = None
    text_type: EmbedTextType
    manual_query_prefix: str | None = None
    manual_passage_prefix: str | None = None
    api_url: str | None = None
    api_version: str | None = None
    # This disables the "model_" protected namespace for pydantic
    # 这将禁用pydantic的"model_"保护命名空间
    model_config = {"protected_namespaces": ()}

# EmbedResponse类用于表示嵌入响应
class EmbedResponse(BaseModel):
    embeddings: list[Embedding]

# RerankRequest类用于表示重新排序请求
class RerankRequest(BaseModel):
    query: str
    documents: list[str]
    model_name: str
    provider_type: RerankerProvider | None = None
    api_key: str | None = None
    api_url: str | None = None

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
