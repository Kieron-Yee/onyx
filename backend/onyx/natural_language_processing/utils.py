"""
自然语言处理工具模块
本模块提供了文本分词、编码解码等自然语言处理相关的基础功能。
主要包含以下功能：
1. 基础分词器抽象类及其实现
2. 分词器缓存管理
3. 文本长度控制工具
"""

import os
from abc import ABC
from abc import abstractmethod
from copy import copy

from transformers import logging as transformer_logging  # type:ignore

from onyx.configs.model_configs import DOC_EMBEDDING_CONTEXT_SIZE
from onyx.configs.model_configs import DOCUMENT_ENCODER_MODEL
from onyx.context.search.models import InferenceChunk
from onyx.utils.logger import setup_logger
from shared_configs.enums import EmbeddingProvider

logger = setup_logger()
transformer_logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"


class BaseTokenizer(ABC):
    """
    分词器的基础抽象类
    定义了分词器必须实现的基本接口
    """
    
    @abstractmethod
    def encode(self, string: str) -> list[int]:
        """
        将文本编码为token ID列表
        
        参数:
            string: 需要编码的文本
        返回:
            编码后的token ID列表
        """
        pass

    @abstractmethod
    def tokenize(self, string: str) -> list[str]:
        """
        将文本分割为token列表
        
        参数:
            string: 需要分词的文本
        返回:
            分词后的token字符串列表
        """
        pass

    @abstractmethod
    def decode(self, tokens: list[int]) -> str:
        """
        将token ID列表解码为文本
        
        参数:
            tokens: 需要解码的token ID列表
        返回:
            解码后的文本
        """
        pass


class TiktokenTokenizer(BaseTokenizer):
    """
    OpenAI Tiktoken分词器实现
    使用单例模式确保每个模型只创建一个分词器实例
    """
    
    _instances: dict[str, "TiktokenTokenizer"] = {}

    def __new__(cls, model_name: str) -> "TiktokenTokenizer":
        """
        创建或获取分词器实例
        
        参数:
            model_name: 模型名称
        返回:
            TiktokenTokenizer实例
        """
        if model_name not in cls._instances:
            cls._instances[model_name] = super(TiktokenTokenizer, cls).__new__(cls)
        return cls._instances[model_name]

    def __init__(self, model_name: str):
        """
        初始化Tiktoken分词器
        
        参数:
            model_name: 模型名称
        """
        if not hasattr(self, "encoder"):
            import tiktoken

            self.encoder = tiktoken.encoding_for_model(model_name)

    def encode(self, string: str) -> list[int]:
        """
        将文本编码为token ID列表，忽略特殊token
        # this ignores special tokens that the model is trained on, see encode_ordinary for details
        # 这将忽略模型训练时使用的特殊token，详见encode_ordinary
        
        参数:
            string: 需要编码的文本
        返回:
            编码后的token ID列表
        """
        return self.encoder.encode_ordinary(string)

    def tokenize(self, string: str) -> list[str]:
        encoded = self.encode(string)
        decoded = [self.encoder.decode([token]) for token in encoded]

        if len(decoded) != len(encoded):
            logger.warning(
                f"OpenAI tokenized length {len(decoded)} does not match encoded length {len(encoded)} for string: {string}"
                # OpenAI分词后的长度与编码长度不匹配，对应字符串为：{string}
            )

        return decoded

    def decode(self, tokens: list[int]) -> str:
        return self.encoder.decode(tokens)


class HuggingFaceTokenizer(BaseTokenizer):
    """
    HuggingFace分词器实现
    """
    
    def __init__(self, model_name: str):
        """
        初始化HuggingFace分词器
        
        参数:
            model_name: 模型名称
        """
        from tokenizers import Tokenizer  # type: ignore

        self.encoder = Tokenizer.from_pretrained(model_name)

    def encode(self, string: str) -> list[int]:
        # this returns no special tokens
        # 这里不返回特殊token
        return self.encoder.encode(string, add_special_tokens=False).ids

    def tokenize(self, string: str) -> list[str]:
        return self.encoder.encode(string, add_special_tokens=False).tokens

    def decode(self, tokens: list[int]) -> str:
        return self.encoder.decode(tokens)


_TOKENIZER_CACHE: dict[tuple[EmbeddingProvider | None, str | None], BaseTokenizer] = {}


def _check_tokenizer_cache(
    model_provider: EmbeddingProvider | None, model_name: str | None
) -> BaseTokenizer:
    """
    检查并获取分词器缓存
    
    参数:
        model_provider: 模型提供者
        model_name: 模型名称
    返回:
        对应的分词器实例
    """
    global _TOKENIZER_CACHE
    id_tuple = (model_provider, model_name)

    if id_tuple not in _TOKENIZER_CACHE:
        tokenizer = None

        if model_name:
            tokenizer = _try_initialize_tokenizer(model_name, model_provider)

        if not tokenizer:
            logger.info(
                f"Falling back to default embedding model: {DOCUMENT_ENCODER_MODEL}"
            )
            tokenizer = HuggingFaceTokenizer(DOCUMENT_ENCODER_MODEL)

        _TOKENIZER_CACHE[id_tuple] = tokenizer

    return _TOKENIZER_CACHE[id_tuple]


def _try_initialize_tokenizer(
    model_name: str, model_provider: EmbeddingProvider | None
) -> BaseTokenizer | None:
    """
    尝试初始化分词器
    
    参数:
        model_name: 模型名称
        model_provider: 模型提供者
    返回:
        初始化成功返回分词器实例，失败返回None
    """
    tokenizer: BaseTokenizer | None = None

    if model_provider is not None:
        # Try using TiktokenTokenizer first if model_provider exists
        # 如果存在model_provider，首先尝试使用TiktokenTokenizer
        try:
            tokenizer = TiktokenTokenizer(model_name)
            logger.info(f"Initialized TiktokenTokenizer for: {model_name}")
            return tokenizer
        except Exception as tiktoken_error:
            logger.debug(
                f"TiktokenTokenizer not available for model {model_name}: {tiktoken_error}"
            )
    else:
        # If no provider specified, try HuggingFaceTokenizer
        # 如果未指定提供者，尝试使用HuggingFaceTokenizer
        try:
            tokenizer = HuggingFaceTokenizer(model_name)
            logger.info(f"Initialized HuggingFaceTokenizer for: {model_name}")
            return tokenizer
        except Exception as hf_error:
            logger.warning(
                f"Failed to initialize HuggingFaceTokenizer for {model_name}: {hf_error}"
            )

    # If both initializations fail, return None
    # 如果两种初始化都失败，返回None
    return None


_DEFAULT_TOKENIZER: BaseTokenizer = HuggingFaceTokenizer(DOCUMENT_ENCODER_MODEL)


def get_tokenizer(
    model_name: str | None, provider_type: EmbeddingProvider | str | None
) -> BaseTokenizer:
    """
    获取分词器实例
    
    参数:
        model_name: 模型名称
        provider_type: 提供者类型
    返回:
        分词器实例
    """
    if isinstance(provider_type, str):
        try:
            provider_type = EmbeddingProvider(provider_type)
        except ValueError:
            logger.debug(
                f"Invalid provider_type '{provider_type}'. Falling back to default tokenizer."
            )
            return _DEFAULT_TOKENIZER
    return _check_tokenizer_cache(provider_type, model_name)


def tokenizer_trim_content(
    content: str, desired_length: int, tokenizer: BaseTokenizer
) -> str:
    """
    将文本内容截断到指定token长度
    
    参数:
        content: 需要截断的文本内容
        desired_length: 期望的token长度
        tokenizer: 使用的分词器
    返回:
        截断后的文本
    """
    tokens = tokenizer.encode(content)
    if len(tokens) > desired_length:
        content = tokenizer.decode(tokens[:desired_length])
    return content


def tokenizer_trim_chunks(
    chunks: list[InferenceChunk],
    tokenizer: BaseTokenizer,
    max_chunk_toks: int = DOC_EMBEDDING_CONTEXT_SIZE,
) -> list[InferenceChunk]:
    """
    将文本块列表截断到指定token长度
    
    参数:
        chunks: 文本块列表
        tokenizer: 使用的分词器
        max_chunk_toks: 最大token长度
    返回:
        处理后的文本块列表
    """
    new_chunks = copy(chunks)
    for ind, chunk in enumerate(new_chunks):
        new_content = tokenizer_trim_content(chunk.content, max_chunk_toks, tokenizer)
        if len(new_content) != len(chunk.content):
            new_chunk = copy(chunk)
            new_chunk.content = new_content
            new_chunks[ind] = new_chunk
    return new_chunks
