"""
此文件包含了搜索相关的自然语言处理模型实现。
主要功能包括:
1. 文本嵌入(Embedding)模型 - 用于将文本转换为向量表示
2. 重排序(Reranking)模型 - 用于对搜索结果进行重新排序
3. 查询分析模型 - 用于分析用户查询的意图
4. 连接器分类模型 - 用于将查询分配给适当的连接器
"""

import threading
import time
from collections.abc import Callable
from functools import wraps
from typing import Any

import requests
from httpx import HTTPError
from requests import JSONDecodeError
from requests import RequestException
from requests import Response
from retry import retry

from onyx.configs.app_configs import LARGE_CHUNK_RATIO
from onyx.configs.model_configs import BATCH_SIZE_ENCODE_CHUNKS
from onyx.configs.model_configs import (
    BATCH_SIZE_ENCODE_CHUNKS_FOR_API_EMBEDDING_SERVICES,
)
from onyx.configs.model_configs import DOC_EMBEDDING_CONTEXT_SIZE
from onyx.db.models import SearchSettings
from onyx.indexing.indexing_heartbeat import IndexingHeartbeatInterface
from onyx.natural_language_processing.exceptions import (
    ModelServerRateLimitError,
)
from onyx.natural_language_processing.utils import get_tokenizer
from onyx.natural_language_processing.utils import tokenizer_trim_content
from onyx.utils.logger import setup_logger
from shared_configs.configs import MODEL_SERVER_HOST
from shared_configs.configs import MODEL_SERVER_PORT
from shared_configs.enums import EmbeddingProvider
from shared_configs.enums import EmbedTextType
from shared_configs.enums import RerankerProvider
from shared_configs.model_server_models import ConnectorClassificationRequest
from shared_configs.model_server_models import ConnectorClassificationResponse
from shared_configs.model_server_models import Embedding
from shared_configs.model_server_models import EmbedRequest
from shared_configs.model_server_models import EmbedResponse
from shared_configs.model_server_models import IntentRequest
from shared_configs.model_server_models import IntentResponse
from shared_configs.model_server_models import RerankRequest
from shared_configs.model_server_models import RerankResponse
from shared_configs.utils import batch_list

logger = setup_logger()


WARM_UP_STRINGS = [
    "Onyx is amazing!",
    "Check out our easy deployment guide at",
    "https://docs.onyx.app/quickstart",
]


def clean_model_name(model_str: str) -> str:
    """
    清理模型名称,将特殊字符替换为下划线
    
    Args:
        model_str: 原始模型名称
    
    Returns:
        清理后的模型名称
    """
    return model_str.replace("/", "_").replace("-", "_").replace(".", "_")


def build_model_server_url(
    model_server_host: str,
    model_server_port: int,
) -> str:
    """
    构建模型服务器的URL
    
    Args:
        model_server_host: 服务器主机地址
        model_server_port: 服务器端口号
    
    Returns:
        完整的服务器URL
    """
    model_server_url = f"{model_server_host}:{model_server_port}"

    # use protocol if provided
    # 如果提供了协议则直接使用
    if "http" in model_server_url:
        return model_server_url

    # otherwise default to http
    # 否则默认使用http协议
    return f"http://{model_server_url}"


class EmbeddingModel:
    """
    文本嵌入模型类
    
    用于将文本转换为向量表示的模型实现。支持本地模型和API服务两种方式。
    """
    
    def __init__(
        self,
        server_host: str,  # Changes depending on indexing or inference
                          # 根据索引或推理过程而变化
        server_port: int,
        model_name: str | None,
        normalize: bool,
        query_prefix: str | None,
        passage_prefix: str | None,
        api_key: str | None,
        api_url: str | None,
        provider_type: EmbeddingProvider | None,
        retrim_content: bool = False,
        callback: IndexingHeartbeatInterface | None = None,
        api_version: str | None = None,
        deployment_name: str | None = None,
    ) -> None:
        """
        初始化嵌入模型
        
        Args:
            server_host: 服务器主机地址
            server_port: 服务器端口
            model_name: 模型名称
            normalize: 是否对向量进行归一化
            query_prefix: 查询前缀
            passage_prefix: 文档段落前缀
            api_key: API密钥
            api_url: API URL地址
            provider_type: 提供商类型
            retrim_content: 是否重新裁剪内容
            callback: 索引进度回调接口
            api_version: API版本
            deployment_name: 部署名称
        """
        self.api_key = api_key
        self.provider_type = provider_type
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.normalize = normalize
        self.model_name = model_name
        self.retrim_content = retrim_content
        self.api_url = api_url
        self.api_version = api_version
        self.deployment_name = deployment_name
        self.tokenizer = get_tokenizer(
            model_name=model_name, provider_type=provider_type
        )
        self.callback = callback

        model_server_url = build_model_server_url(server_host, server_port)
        self.embed_server_endpoint = f"{model_server_url}/encoder/bi-encoder-embed"

    def _make_model_server_request(self, embed_request: EmbedRequest) -> EmbedResponse:
        """
        向模型服务器发送嵌入请求
        
        Args:
            embed_request: 嵌入请求对象
            
        Returns:
            嵌入响应对象
            
        Raises:
            HTTPError: 当HTTP请求失败时
            ModelServerRateLimitError: 当触发服务器速率限制时
        """
        def _make_request() -> Response:
            response = requests.post(
                self.embed_server_endpoint, json=embed_request.model_dump()
            )
            # signify that this is a rate limit error
            # 表明这是一个速率限制错误
            if response.status_code == 429:
                raise ModelServerRateLimitError(response.text)

            response.raise_for_status()
            return response

        final_make_request_func = _make_request

        # if the text type is a passage, add some default
        # retries + handling for rate limiting
        # 如果文本类型是段落，添加默认的重试次数和速率限制处理
        if embed_request.text_type == EmbedTextType.PASSAGE:
            final_make_request_func = retry(
                tries=3,
                delay=5,
                exceptions=(RequestException, ValueError, JSONDecodeError),
            )(final_make_request_func)
            # use 10 second delay as per Azure suggestion
            # 按照Azure建议使用10秒延迟
            final_make_request_func = retry(
                tries=10, delay=10, exceptions=ModelServerRateLimitError
            )(final_make_request_func)

        response: Response | None = None

        try:
            response = final_make_request_func()
            return EmbedResponse(**response.json())
        except requests.HTTPError as e:
            if not response:
                raise HTTPError("HTTP error occurred - response is None.") from e

            try:
                error_detail = response.json().get("detail", str(e))
            except Exception:
                error_detail = response.text
            raise HTTPError(f"HTTP error occurred: {error_detail}") from e
        except requests.RequestException as e:
            raise HTTPError(f"Request failed: {str(e)}") from e

    def _batch_encode_texts(
        self,
        texts: list[str],
        text_type: EmbedTextType,
        batch_size: int,
        max_seq_length: int,
    ) -> list[Embedding]:
        """
        批量编码文本
        
        Args:
            texts: 待编码的文本列表
            text_type: 文本类型(查询或文档)
            batch_size: 批处理大小
            max_seq_length: 最大序列长度
            
        Returns:
            编码后的嵌入向量列表
            
        Raises:
            RuntimeError: 当检测到停止信号时
        """
        text_batches = batch_list(texts, batch_size)

        logger.debug(
            f"Encoding {len(texts)} texts in {len(text_batches)} batches for local model"
        )

        embeddings: list[Embedding] = []
        for idx, text_batch in enumerate(text_batches, start=1):
            if self.callback:
                if self.callback.should_stop():
                    raise RuntimeError("_batch_encode_texts detected stop signal")

            logger.debug(f"Encoding batch {idx} of {len(text_batches)}")
            embed_request = EmbedRequest(
                model_name=self.model_name,
                texts=text_batch,
                api_version=self.api_version,
                deployment_name=self.deployment_name,
                max_context_length=max_seq_length,
                normalize_embeddings=self.normalize,
                api_key=self.api_key,
                provider_type=self.provider_type,
                text_type=text_type,
                manual_query_prefix=self.query_prefix,
                manual_passage_prefix=self.passage_prefix,
                api_url=self.api_url,
            )

            response = self._make_model_server_request(embed_request)
            embeddings.extend(response.embeddings)

            if self.callback:
                self.callback.progress("_batch_encode_texts", 1)
        return embeddings

    def encode(
        self,
        texts: list[str],
        text_type: EmbedTextType,
        large_chunks_present: bool = False,
        local_embedding_batch_size: int = BATCH_SIZE_ENCODE_CHUNKS,
        api_embedding_batch_size: int = BATCH_SIZE_ENCODE_CHUNKS_FOR_API_EMBEDDING_SERVICES,
        max_seq_length: int = DOC_EMBEDDING_CONTEXT_SIZE,
    ) -> list[Embedding]:
        """
        编码文本为向量表示
        
        Args:
            texts: 待编码的文本列表
            text_type: 文本类型(查询或文档)
            large_chunks_present: 是否存在大块文本
            local_embedding_batch_size: 本地模型的批处理大小
            api_embedding_batch_size: API服务的批处理大小
            max_seq_length: 最大序列长度
            
        Returns:
            编码后的嵌入向量列表
            
        Raises:
            ValueError: 当输入文本为空或无效时
        """
        if not texts or not all(texts):
            raise ValueError(f"Empty or missing text for embedding: {texts}")

        if large_chunks_present:
            max_seq_length *= LARGE_CHUNK_RATIO

        if self.retrim_content:
            # This is applied during indexing as a catchall for overly long titles (or other uncapped fields)
            # 这在索引过程中用作处理过长标题(或其他未限制长度的字段)的通用方法
            # Note that this uses just the default tokenizer which may also lead to very minor miscountings
            # 注意这里使用默认分词器可能会导致轻微的计数误差
            # However this slight miscounting is very unlikely to have any material impact.
            # 但这种轻微的计数误差不太可能产生实质性影响
            texts = [
                tokenizer_trim_content(
                    content=text,
                    desired_length=max_seq_length,
                    tokenizer=self.tokenizer,
                )
                for text in texts
            ]

        batch_size = (
            api_embedding_batch_size
            if self.provider_type
            else local_embedding_batch_size
        )

        return self._batch_encode_texts(
            texts=texts,
            text_type=text_type,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
        )

    @classmethod
    def from_db_model(
        cls,
        search_settings: SearchSettings,
        server_host: str,  # Changes depending on indexing or inference
                          # 根据索引或推理过程而变化
        server_port: int,
        retrim_content: bool = False,
    ) -> "EmbeddingModel":
        """
        从数据库模型创建嵌入模型实例
        
        Args:
            search_settings: 搜索设置对象
            server_host: 服务器主机地址
            server_port: 服务器端口
            retrim_content: 是否重新裁剪内容
            
        Returns:
            嵌入模型实例
        """
        return cls(
            server_host=server_host,
            server_port=server_port,
            model_name=search_settings.model_name,
            normalize=search_settings.normalize,
            query_prefix=search_settings.query_prefix,
            passage_prefix=search_settings.passage_prefix,
            api_key=search_settings.api_key,
            provider_type=search_settings.provider_type,
            api_url=search_settings.api_url,
            retrim_content=retrim_content,
            api_version=search_settings.api_version,
            deployment_name=search_settings.deployment_name,
        )


class RerankingModel:
    """
    重排序模型类
    
    用于对搜索结果进行重新排序的模型实现。
    """
    
    def __init__(
        self,
        model_name: str,
        provider_type: RerankerProvider | None,
        api_key: str | None,
        api_url: str | None,
        model_server_host: str = MODEL_SERVER_HOST,
        model_server_port: int = MODEL_SERVER_PORT,
    ) -> None:
        """
        初始化重排序模型
        
        Args:
            model_name: 模型名称
            provider_type: 提供商类型
            api_key: API密钥
            api_url: API URL地址
            model_server_host: 服务器主机地址
            model_server_port: 服务器端口
        """
        model_server_url = build_model_server_url(model_server_host, model_server_port)
        self.rerank_server_endpoint = model_server_url + "/encoder/cross-encoder-scores"
        self.model_name = model_name
        self.provider_type = provider_type
        self.api_key = api_key
        self.api_url = api_url

    def predict(self, query: str, passages: list[str]) -> list[float]:
        """
        预测查询和文档段落的相关性得分
        
        Args:
            query: 查询字符串
            passages: 文档段落列表
        
        Returns:
            相关性得分列表
        """
        rerank_request = RerankRequest(
            query=query,
            documents=passages,
            model_name=self.model_name,
            provider_type=self.provider_type,
            api_key=self.api_key,
            api_url=self.api_url,
        )

        response = requests.post(
            self.rerank_server_endpoint, json=rerank_request.model_dump()
        )
        response.raise_for_status()

        return RerankResponse(**response.json()).scores


class QueryAnalysisModel:
    """
    查询分析模型类
    
    用于分析用户查询意图的模型实现。
    """
    
    def __init__(
        self,
        model_server_host: str = MODEL_SERVER_HOST,
        model_server_port: int = MODEL_SERVER_PORT,
        # Lean heavily towards not throwing out keywords
        # 倾向于保留关键词
        keyword_percent_threshold: float = 0.1,
        # Lean towards semantic which is the default
        # 倾向于语义分析(这是默认行为)
        semantic_percent_threshold: float = 0.4,
    ) -> None:
        """
        初始化查询分析模型
        
        Args:
            model_server_host: 服务器主机地址
            model_server_port: 服务器端口
            keyword_percent_threshold: 关键词百分比阈值
            semantic_percent_threshold: 语义百分比阈值
        """
        model_server_url = build_model_server_url(model_server_host, model_server_port)
        self.intent_server_endpoint = model_server_url + "/custom/query-analysis"
        self.keyword_percent_threshold = keyword_percent_threshold
        self.semantic_percent_threshold = semantic_percent_threshold

    def predict(
        self,
        query: str,
    ) -> tuple[bool, list[str]]:
        """
        预测查询的意图和关键词
        
        Args:
            query: 查询字符串
        
        Returns:
            是否为关键词查询, 关键词列表
        """
        intent_request = IntentRequest(
            query=query,
            keyword_percent_threshold=self.keyword_percent_threshold,
            semantic_percent_threshold=self.semantic_percent_threshold,
        )

        response = requests.post(
            self.intent_server_endpoint, json=intent_request.model_dump()
        )
        response.raise_for_status()

        response_model = IntentResponse(**response.json())

        return response_model.is_keyword, response_model.keywords


class ConnectorClassificationModel:
    """
    连接器分类模型类
    
    用于将查询分配给适当连接器的模型实现。
    """
    
    def __init__(
        self,
        model_server_host: str = MODEL_SERVER_HOST,
        model_server_port: int = MODEL_SERVER_PORT,
    ):
        """
        初始化连接器分类模型
        
        Args:
            model_server_host: 服务器主机地址
            model_server_port: 服务器端口
        """
        model_server_url = build_model_server_url(model_server_host, model_server_port)
        self.connector_classification_endpoint = (
            model_server_url + "/custom/connector-classification"
        )

    def predict(
        self,
        query: str,
        available_connectors: list[str],
    ) -> list[str]:
        """
        预测查询适用的连接器
        
        Args:
            query: 查询字符串
            available_connectors: 可用连接器列表
        
        Returns:
            适用的连接器列表
        """
        connector_classification_request = ConnectorClassificationRequest(
            available_connectors=available_connectors,
            query=query,
        )
        response = requests.post(
            self.connector_classification_endpoint,
            json=connector_classification_request.dict(),
        )
        response.raise_for_status()

        response_model = ConnectorClassificationResponse(**response.json())

        return response_model.connectors


def warm_up_retry(
    func: Callable[..., Any],
    tries: int = 20,
    delay: int = 5,
    *args: Any,
    **kwargs: Any,
) -> Callable[..., Any]:
    """
    尝试多次执行函数, 直到成功或达到最大重试次数
    
    Args:
        func: 需要执行的函数
        tries: 最大重试次数
        delay: 每次重试的延迟时间(秒)
    
    Returns:
        包装后的函数
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        exceptions = []
        for attempt in range(tries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                exceptions.append(e)
                logger.info(
                    f"Attempt {attempt + 1}/{tries} failed; retrying in {delay} seconds..."
                    # 第{attempt + 1}/{tries}次尝试失败; 将在{delay}秒后重试...
                )
                time.sleep(delay)
        raise Exception(f"All retries failed: {exceptions}")

    return wrapper


def warm_up_bi_encoder(
    embedding_model: EmbeddingModel,
    non_blocking: bool = False,
) -> None:
    """
    预热双编码器模型
    
    Args:
        embedding_model: 嵌入模型对象
        non_blocking: 是否非阻塞执行
    """
    warm_up_str = " ".join(WARM_UP_STRINGS)

    logger.debug(f"Warming up encoder model: {embedding_model.model_name}")
    get_tokenizer(
        model_name=embedding_model.model_name,
        provider_type=embedding_model.provider_type,
    ).encode(warm_up_str)

    def _warm_up() -> None:
        try:
            embedding_model.encode(texts=[warm_up_str], text_type=EmbedTextType.QUERY)
            logger.debug(
                f"Warm-up complete for encoder model: {embedding_model.model_name}"
                # 编码器模型预热完成: {embedding_model.model_name}
            )
        except Exception as e:
            logger.warning(
                f"Warm-up request failed for encoder model {embedding_model.model_name}: {e}"
                # 编码器模型预热请求失败 {embedding_model.model_name}: {e}
            )

    if non_blocking:
        threading.Thread(target=_warm_up, daemon=True).start()
        logger.debug(
            f"Started non-blocking warm-up for encoder model: {embedding_model.model_name}"
            # 已启动编码器模型的非阻塞预热: {embedding_model.model_name}
        )
    else:
        retry_encode = warm_up_retry(embedding_model.encode)
        retry_encode(texts=[warm_up_str], text_type=EmbedTextType.QUERY)


def warm_up_cross_encoder(
    rerank_model_name: str,
    non_blocking: bool = False,
) -> None:
    """
    预热交叉编码器模型
    
    Args:
        rerank_model_name: 重排序模型名称
        non_blocking: 是否非阻塞执行
    """
    logger.debug(f"Warming up reranking model: {rerank_model_name}")

    reranking_model = RerankingModel(
        model_name=rerank_model_name,
        provider_type=None,
        api_url=None,
        api_key=None,
    )

    def _warm_up() -> None:
        """
        预热交叉编码器模型的内部实现
        
        尝试使用预热字符串进行模型预测,并记录执行结果
        """
        try:
            reranking_model.predict(WARM_UP_STRINGS[0], WARM_UP_STRINGS[1:])
            logger.debug(f"Warm-up complete for reranking model: {rerank_model_name}")
        except Exception as e:
            logger.warning(
                f"Warm-up request failed for reranking model {rerank_model_name}: {e}"
            )

    if non_blocking:
        threading.Thread(target=_warm_up, daemon=True).start()
        logger.debug(
            f"Started non-blocking warm-up for reranking model: {rerank_model_name}"
        )
    else:
        retry_rerank = warm_up_retry(reranking_model.predict)
        retry_rerank(WARM_UP_STRINGS[0], WARM_UP_STRINGS[1:])
