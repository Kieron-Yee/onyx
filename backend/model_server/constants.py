from shared_configs.enums import EmbeddingProvider
from shared_configs.enums import EmbedTextType


MODEL_WARM_UP_STRING = "hi " * 512  # 模型预热字符串
DEFAULT_OPENAI_MODEL = "text-embedding-3-small"  # OpenAI默认模型
DEFAULT_COHERE_MODEL = "embed-english-light-v3.0"  # Cohere默认模型
DEFAULT_VOYAGE_MODEL = "voyage-large-2-instruct"  # Voyage默认模型
DEFAULT_VERTEX_MODEL = "text-embedding-004"  # Google Vertex默认模型


class EmbeddingModelTextType:
    PROVIDER_TEXT_TYPE_MAP = {
        EmbeddingProvider.COHERE: {
            EmbedTextType.QUERY: "search_query",
            EmbedTextType.PASSAGE: "search_document",
        },
        EmbeddingProvider.VOYAGE: {
            EmbedTextType.QUERY: "query",
            EmbedTextType.PASSAGE: "document",
        },
        EmbeddingProvider.GOOGLE: {
            EmbedTextType.QUERY: "RETRIEVAL_QUERY",
            EmbedTextType.PASSAGE: "RETRIEVAL_DOCUMENT",
        },
    }

    @staticmethod
    def get_type(provider: EmbeddingProvider, text_type: EmbedTextType) -> str:
        return EmbeddingModelTextType.PROVIDER_TEXT_TYPE_MAP[provider][text_type]
